"""Aggregate per-image rows into per-category metrics and write reports.

With several backends, the report starts with a side-by-side comparison per
category, followed by a detailed section per backend.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backends.base import FAILURE_KINDS, NO_FACE, UNREADABLE
from .evaluate import BenchResult

CLEAN = "clean"
CLEAN_LABEL = "clean (held-out)"
TWIN_LABEL = "twin (look-alike, accept = error)"
BACKEND_LABELS = {"arcface": "ArcFace", "qwen_vl": "Qwen-VL"}
SCORE_LABELS = {"arcface": "cosine", "qwen_vl": "mean P(same)"}
SUMMARY_COLUMNS = [
    "category",
    "probes",
    "recognized",
    "recognition_rate",
    "ci_low",
    "ci_high",
    "mean_score",
    "retention",
    "mean_face_score",
    "vote_rate",
    "failures",
    "detection_fails",
    *FAILURE_KINDS,
]


def label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (robust for small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _nanmean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def category_stats(group: pd.DataFrame, clean_mean: float) -> dict:
    n = int(len(group))
    rec = int(group["recognized"].sum())
    lo, hi = wilson_interval(rec, n)
    mean_score = _nanmean(group["score"])
    votes = group["vote_same"].dropna()
    errors = group["error"].fillna("")
    kinds = {k: int((errors == k).sum()) for k in FAILURE_KINDS}
    return {
        "probes": n,
        "recognized": rec,
        "recognition_rate": rec / n * 100 if n else float("nan"),
        "ci_low": lo * 100,
        "ci_high": hi * 100,
        "mean_score": mean_score,
        "retention": mean_score / clean_mean if clean_mean and not math.isnan(clean_mean) else float("nan"),
        "mean_face_score": _nanmean(group["face_score"]),
        "vote_rate": float(votes.astype(bool).mean() * 100) if len(votes) else float("nan"),
        "failures": int((~group["valid"].astype(bool)).sum()),
        "detection_fails": kinds[NO_FACE] + kinds[UNREADABLE],
        **kinds,
    }


def summarize(rows: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """One row per category (clean first), pooled over persons, for one backend."""
    clean = rows[rows["role"] == CLEAN]
    clean_mean = _nanmean(clean["score"])
    out = [{"category": CLEAN_LABEL, **category_stats(clean, clean_mean)}]
    for cat in categories:
        group = rows[(rows["role"] == "probe") & (rows["category"] == cat)]
        if len(group):
            out.append({"category": cat, **category_stats(group, clean_mean)})
    twins = rows[rows["role"] == "twin"]
    if len(twins):
        out.append({"category": TWIN_LABEL, **category_stats(twins, clean_mean)})
    return pd.DataFrame(out, columns=SUMMARY_COLUMNS)


def per_person(rows: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Recognition rate (%) per person x category."""
    records = []
    for person, prow in rows.groupby("person", sort=True):
        rec: dict = {"person": person}
        for name, mask in [(CLEAN, prow["role"] == CLEAN)] + [
            (c, (prow["role"] == "probe") & (prow["category"] == c)) for c in categories
        ]:
            g = prow[mask]
            rec[name] = f"{g['recognized'].mean() * 100:.0f}% ({len(g)})" if len(g) else "-"
        records.append(rec)
    return pd.DataFrame(records)


def _fmt(v, spec: str) -> str:
    try:
        if v is None or math.isnan(float(v)):
            return "-"
    except (TypeError, ValueError):
        return "-"
    return format(v, spec)


def _fails_cell(r) -> str:
    total = int(r["failures"])
    parts = [f"{k} {int(r[k])}" for k in FAILURE_KINDS if int(r[k])]
    return f"{total} ({', '.join(parts)})" if parts else str(total)


def markdown_table(summary: pd.DataFrame, backend: str = "arcface") -> str:
    vote = backend == "qwen_vl"
    header = (
        f"| Category | Probes | Recognition % (95% CI) | Mean {SCORE_LABELS.get(backend, 'score')} | Retention "
        f"| Face score |{' Vote same % |' if vote else ''} Failures |"
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for _, r in summary.iterrows():
        rate = f"{_fmt(r.recognition_rate, '.1f')} ({_fmt(r.ci_low, '.0f')}-{_fmt(r.ci_high, '.0f')})"
        lines.append(
            f"| {r.category} | {r.probes} | {rate} | {_fmt(r.mean_score, '.3f')} | "
            f"{_fmt(r.retention, '.2f')} | {_fmt(r.mean_face_score, '.1f')} |"
            + (f" {_fmt(r.vote_rate, '.1f')} |" if vote else "")
            + f" {_fails_cell(r)} |"
        )
    return "\n".join(lines)


def comparison_table(results: list[BenchResult]) -> str:
    """Side-by-side per category: recognition %, face score and failures per backend."""
    summaries = {r.backend: summarize(r.rows, r.categories).set_index("category") for r in results}
    cats = [CLEAN_LABEL]
    for r in results:
        cats += [c for c in r.categories if c not in cats]
    if any(TWIN_LABEL in s.index for s in summaries.values()):
        cats.append(TWIN_LABEL)
    names = [label(r.backend) for r in results]
    header = (
        "| Category | Probes | "
        + " | ".join(f"{n} rec %" for n in names) + " | "
        + " | ".join(f"{n} face score" for n in names) + " | "
        + " | ".join(f"{n} failures" for n in names) + " |"
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for cat in cats:
        cells = {b: (s.loc[cat] if cat in s.index else None) for b, s in summaries.items()}
        probes = sorted({int(c["probes"]) for c in cells.values() if c is not None})
        rec = [_fmt(c["recognition_rate"], ".1f") if c is not None else "-" for c in cells.values()]
        fs = [_fmt(c["mean_face_score"], ".1f") if c is not None else "-" for c in cells.values()]
        fails = [_fails_cell(c) if c is not None else "-" for c in cells.values()]
        lines.append(
            f"| {cat} | {' / '.join(map(str, probes)) or '-'} | "
            + " | ".join(rec) + " | " + " | ".join(fs) + " | " + " | ".join(fails) + " |"
        )
    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def observations(result: BenchResult, summary: pd.DataFrame, max_files: int = 15) -> list[str]:
    obs: list[str] = []
    probes = summary[~summary["category"].isin([CLEAN_LABEL, TWIN_LABEL])]
    if len(probes):
        ranked = probes.sort_values(["recognition_rate", "mean_score"])
        worst, best = ranked.iloc[0], ranked.iloc[-1]
        obs.append(
            f"Hardest category: **{worst['category']}** "
            f"({worst['recognition_rate']:.1f}% recognized, retention {_fmt(worst['retention'], '.2f')})."
        )
        if best["category"] != worst["category"]:
            obs.append(
                f"Least affected category: **{best['category']}** ({best['recognition_rate']:.1f}% recognized)."
            )
    else:
        obs.append("No probe images were found; only the clean baseline was evaluated.")

    rows = result.rows
    failed = rows[rows["role"].isin(["probe", CLEAN]) & ~rows["recognized"]]
    for cat, g in failed.groupby("category", sort=False):
        names = [
            f"`{f}`" + (f" ({e})" if not v else f" (score {s:.2f})")
            for f, v, e, s in zip(g["file"], g["valid"], g["error"], g["score"])
        ]
        extra = f" … and {len(names) - max_files} more" if len(names) > max_files else ""
        obs.append(f"Not recognized in *{cat}*: " + ", ".join(names[:max_files]) + extra)
    return obs


def _threshold_line(result: BenchResult) -> str:
    t = result.threshold
    far = f"{t.far_effective:.2%}" + (" (fallback)" if t.warning else "")
    return (
        f"Model: {result.model_name} | FAR: {far} | tau: {t.tau:.4f}{' (manual)' if t.manual else ''} | "
        f"persons: {len(result.persons)}\n\n"
        f"_Genuine scores: {t.n_genuine}, impostor scores: {t.n_impostor}, "
        f"achieved FAR: {_fmt(t.far_achieved * 100, '.2f')}%, clean FRR: {t.frr_clean:.1%}, "
        f"genuine mean: {t.genuine_mean:.3f}, impostor mean: {_fmt(t.impostor_mean, '.3f')}._"
    )


def backend_section(result: BenchResult, level: str = "##") -> list[str]:
    summary = summarize(result.rows, result.categories)
    lines = [
        f"{level} {label(result.backend)}",
        "",
        _threshold_line(result),
        "",
        markdown_table(summary, result.backend),
        "",
        f"{level}# Per person (recognition %, probe count)",
        "",
        _df_to_markdown(per_person(result.rows, result.categories)),
        "",
        f"{level}# Observations",
        "",
        *[f"- {o}" for o in observations(result, summary)],
        "",
    ]
    if result.stats:
        lines += [f"_Run stats: {', '.join(f'{k}={v}' for k, v in result.stats.items())}._", ""]
    return lines


NOTES = [
    "Recognition % uses each backend's own tau, calibrated at the same FAR on the same "
    "clean-genuine vs impostor pairs.",
    "Mean score and face score are averaged over valid rows only. Failures (no_face, "
    "unreadable, refusal, invalid_json, backend_error) count as not recognized.",
    "Qwen-VL score per probe = mean over 3 gallery references of P(same) = confidence/100 "
    "if the answer is 'same', else 1 - confidence/100. 'Vote same %' is the majority vote.",
]


def results_markdown(results: list[BenchResult]) -> str:
    multi = len(results) > 1
    lines = ["# Results", "", f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}._", ""]
    if multi:
        lines += ["## Comparison", "", comparison_table(results), ""]
        lines += [f"- {n}" for n in NOTES] + [""]
        for r in results:
            lines += backend_section(r, "##")
    else:
        r = results[0]
        section = backend_section(r, "#")[2:]  # drop the backend heading
        lines += section + [f"- {n}" for n in NOTES[:2]] + [""]
    warnings = list(dict.fromkeys(w for r in results for w in r.warnings))
    if warnings:
        lines += ["## Warnings", "", *[f"- {w}" for w in warnings], ""]
    lines += [
        "## Limitations",
        "",
        "- Few persons, small sample sizes: results are indicative only (see 95% CIs).",
        "- tau quality depends on the number of impostor scores.",
        "- Results apply to the tested models only: " + ", ".join(r.model_name for r in results) + ".",
        "",
    ]
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _jsonable(obj.item())
    return obj


def write_reports(results: BenchResult | list[BenchResult], results_md: Path | None = None) -> dict[str, Path]:
    """Write per_image.csv, summary.json, summary.md and (optionally) docs/RESULTS.md."""
    results = [results] if isinstance(results, BenchResult) else results
    out = results[0].config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    md = results_markdown(results)

    paths = {
        "per_image": out / "per_image.csv",
        "summary_json": out / "summary.json",
        "summary_md": out / "summary.md",
    }
    pd.concat([r.rows for r in results], ignore_index=True).to_csv(
        paths["per_image"], index=False, float_format="%.6f"
    )
    payload = {
        "backends": {
            r.backend: {
                "model": r.model_name,
                "threshold": r.threshold.to_dict(),
                "persons": r.persons,
                "skipped_persons": r.skipped,
                "categories": summarize(r.rows, r.categories).to_dict(orient="records"),
                "stats": r.stats,
            }
            for r in results
        },
        "warnings": list(dict.fromkeys(w for r in results for w in r.warnings)),
        "config": results[0].config.to_dict(),
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2))
    paths["summary_md"].write_text(md)
    if results_md is not None:
        results_md.parent.mkdir(parents=True, exist_ok=True)
        results_md.write_text(md)
        paths["results_md"] = results_md
    return paths
