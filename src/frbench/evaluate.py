"""Run the benchmark for one or more recognition backends.

Data layout (see README):

    data/gallery/<person_id>/*.jpg                 clean photos
    data/probes/<person_id>/<category>/*.jpg       filter | makeup | effect | botox
    data/impostors/<other_id>/*.jpg                other people
    data/twins/<person_id>/*.jpg                   look-alikes of the person (should NOT match)

The dataset is prepared once (deterministic split, probe discovery, leak
checks) and then every backend scores exactly the same pairs, so results are
directly comparable. Every image becomes one row, including failures
(``valid=False``, ``recognized=False``), so nothing is dropped silently.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import CATEGORIES
from .backends.base import Recognizer, ScoreOutcome
from .gallery import (
    DEFAULT_GALLERY_FRACTION,
    DEFAULT_SEED,
    MIN_CLEAN_PHOTOS,
    InsufficientDataError,
    cached_file_hash,
    dedupe_by_hash,
    list_images,
    split_clean,
)
from .scoring import face_score
from .threshold import DEFAULT_FAR, ThresholdResult, calibrate

log = logging.getLogger(__name__)

ROW_COLUMNS = [
    "backend",
    "person",
    "role",  # gallery | clean | probe | impostor | twin
    "category",  # gallery | clean | <probe category> | impostor | twin
    "file",
    "sha256",
    "valid",
    "error",  # failure kind, see backends.base.FAILURE_KINDS
    "message",
    "score",  # cosine (arcface) or mean p_same (qwen_vl)
    "face_score",
    "recognized",
    "impostor_id",
    "vote_same",  # qwen_vl majority vote
    "n_same",  # qwen_vl answers saying "same"
    "n_valid",
    "mean_confidence",
    "n_faces",  # arcface face audit: faces detected in the image
    "best_face_score",  # arcface face audit: best cosine over all faces
]


@dataclass
class BenchConfig:
    data_dir: Path
    out_dir: Path = Path("results")
    far: float = DEFAULT_FAR
    seed: int = DEFAULT_SEED
    gallery_fraction: float = DEFAULT_GALLERY_FRACTION
    min_clean: int = MIN_CLEAN_PHOTOS
    manual_tau: float | None = None
    cross_impostors: bool = True  # also use other enrolled persons as impostors
    use_cache: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"], d["out_dir"] = str(self.data_dir), str(self.out_dir)
        return d


@dataclass
class PersonData:
    person_id: str
    gallery: tuple[Path, ...]
    heldout: tuple[Path, ...]
    clean_hashes: set[str]
    probes: dict[str, list[Path]] = field(default_factory=dict)
    twins: list[Path] = field(default_factory=list)  # look-alikes; never used for calibration


@dataclass
class PreparedData:
    persons: list[PersonData]
    impostors: list[tuple[str, Path]]  # (impostor_id, path) from data/impostors
    skipped: dict[str, str]
    warnings: list[str]


@dataclass
class BenchResult:
    backend: str
    rows: pd.DataFrame
    threshold: ThresholdResult
    persons: list[str]
    skipped: dict[str, str]
    warnings: list[str]
    config: BenchConfig
    model_name: str
    categories: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dataset preparation (backend independent)
# --------------------------------------------------------------------------- #


def subdirs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))


def discover_probe_categories(person_dir: Path, warnings: list[str]) -> dict[str, list[Path]]:
    """Map category -> sorted image paths. Unknown categories are kept with a warning."""
    out: dict[str, list[Path]] = {}
    loose = list_images(person_dir)
    if loose:
        warnings.append(
            f"{len(loose)} probe image(s) directly in {person_dir} have no category folder; ignored."
        )
    for cat_dir in subdirs(person_dir):
        cat = cat_dir.name.lower()
        if cat not in CATEGORIES:
            warnings.append(f"Unknown probe category '{cat_dir.name}' in {person_dir}; included as-is.")
        images = list_images(cat_dir)
        if not images:
            warnings.append(f"Empty category folder {cat_dir}; skipped.")
            continue
        out.setdefault(cat, []).extend(images)
    return out


def prepare_data(cfg: BenchConfig) -> PreparedData:
    data_dir = cfg.data_dir
    gallery_root = data_dir / "gallery"
    if not gallery_root.is_dir():
        raise FileNotFoundError(f"Missing {gallery_root}; see README for the data layout.")
    warnings: list[str] = []
    skipped: dict[str, str] = {}
    persons: list[PersonData] = []

    for person_dir in subdirs(gallery_root):
        try:
            split = split_clean(list_images(person_dir), cfg.gallery_fraction, cfg.seed, cfg.min_clean)
        except InsufficientDataError as exc:
            msg = f"Person '{person_dir.name}' skipped: {exc}"
            log.warning(msg)
            warnings.append(msg)
            skipped[person_dir.name] = str(exc)
            continue
        hashes = {cached_file_hash(p) for p in (*split.gallery, *split.heldout)}
        persons.append(PersonData(person_dir.name, split.gallery, split.heldout, hashes))
    if not persons:
        raise InsufficientDataError("No person could be enrolled; check data/gallery/.")

    for p in persons:
        probe_dir = data_dir / "probes" / p.person_id
        if not probe_dir.is_dir():
            warnings.append(f"No probes for person '{p.person_id}'.")
            continue
        for cat, paths in discover_probe_categories(probe_dir, warnings).items():
            kept = []
            for path in paths:
                if cached_file_hash(path) in p.clean_hashes:
                    warnings.append(f"Probe {path} is identical to a clean photo; excluded (leak).")
                else:
                    kept.append(path)
            if kept:
                p.probes[cat] = kept

    for p in persons:
        for path in dedupe_by_hash(list_images(data_dir / "twins" / p.person_id)):
            if cached_file_hash(path) in p.clean_hashes:
                warnings.append(f"Twin photo {path} is identical to a clean photo; excluded.")
            else:
                p.twins.append(path)

    enrolled = {p.person_id for p in persons}
    for twin_dir in subdirs(data_dir / "twins"):
        if twin_dir.name not in enrolled:
            warnings.append(f"Twin photos for '{twin_dir.name}' ignored: person not enrolled.")
    for probe_person in subdirs(data_dir / "probes"):
        if probe_person.name not in enrolled:
            warnings.append(f"Probes for '{probe_person.name}' ignored: person not enrolled.")

    impostors: list[tuple[str, Path]] = []
    for imp_dir in subdirs(data_dir / "impostors"):
        if imp_dir.name in enrolled:
            warnings.append(
                f"Impostor folder '{imp_dir.name}' has the same id as an enrolled person; "
                "excluded from that person's impostor set."
            )
        impostors += [(imp_dir.name, p) for p in dedupe_by_hash(list_images(imp_dir))]
    impostors += [(p.stem, p) for p in list_images(data_dir / "impostors")]  # flat layout
    if not impostors and not cfg.cross_impostors:
        warnings.append("No impostor photos found under data/impostors/.")
    return PreparedData(persons, impostors, skipped, warnings)


def impostor_pairs(
    person: PersonData, data: PreparedData, cfg: BenchConfig, cap: int | None
) -> list[tuple[str, Path]]:
    """Impostor images to score against ``person``'s template."""
    pairs = [(iid, p) for iid, p in data.impostors if iid != person.person_id]
    if cfg.cross_impostors:
        for other in data.persons:
            if other.person_id != person.person_id:
                pairs += [(other.person_id, p) for p in (*other.gallery, *other.heldout)]
    pairs = [(iid, p) for iid, p in pairs if cached_file_hash(p) not in person.clean_hashes]
    if cap is not None and len(pairs) > cap:
        pairs = sorted(random.Random(f"{cfg.seed}:{person.person_id}").sample(pairs, cap))
    return pairs


# --------------------------------------------------------------------------- #
# Scoring one backend
# --------------------------------------------------------------------------- #


def make_rows(
    backend: str,
    person: str,
    role: str,
    category: str,
    outcomes: list[ScoreOutcome],
    data_dir: Path,
    impostor_ids: list[str] | None = None,
) -> list[dict]:
    rows = []
    for i, o in enumerate(outcomes):
        try:
            rel = str(o.path.relative_to(data_dir))
        except ValueError:
            rel = str(o.path)
        rows.append(
            {
                "backend": backend,
                "person": person,
                "role": role,
                "category": category,
                "file": rel,
                "sha256": cached_file_hash(o.path)[:12] if o.path.exists() else "",
                "valid": o.valid,
                "error": o.error or "",
                "message": o.message,
                "score": o.score if o.score is not None else np.nan,
                "impostor_id": impostor_ids[i] if impostor_ids else "",
                "vote_same": o.extra.get("vote_same", pd.NA),
                "n_same": o.extra.get("n_same", pd.NA),
                "n_valid": o.extra.get("n_valid", pd.NA),
                "mean_confidence": o.extra.get("mean_confidence", np.nan),
                "n_faces": o.extra.get("n_faces", pd.NA),
                "best_face_score": o.extra.get("best_face_score", np.nan),
            }
        )
    return rows


def run_backend(cfg: BenchConfig, data: PreparedData, recognizer: Recognizer) -> BenchResult:
    name = recognizer.name
    warnings = list(data.warnings)
    skipped = dict(data.skipped)
    rows: list[dict] = []
    genuine: list[float] = []
    impostor: list[float] = []

    templates = {}
    for p in data.persons:
        log.info("[%s] enrolling %s", name, p.person_id)
        try:
            templates[p.person_id] = recognizer.enroll(p.person_id, p.gallery)
        except InsufficientDataError as exc:
            msg = f"[{name}] person '{p.person_id}' skipped: {exc}"
            log.warning(msg)
            warnings.append(msg)
            skipped[p.person_id] = str(exc)
    if not templates:
        raise InsufficientDataError(f"[{name}] no person could be enrolled.")
    persons = [p for p in data.persons if p.person_id in templates]

    for p in persons:
        t = templates[p.person_id]
        rows += make_rows(name, p.person_id, "gallery", "gallery", t.gallery_outcomes, cfg.data_dir)
        clean = recognizer.score(t, p.heldout)
        rows += make_rows(name, p.person_id, "clean", "clean", clean, cfg.data_dir)
        genuine += [o.score for o in clean if o.score is not None]

        pairs = impostor_pairs(p, data, cfg, recognizer.max_impostors_per_person)
        log.info("[%s] %s: %d impostor images", name, p.person_id, len(pairs))
        imp = recognizer.score(t, [path for _, path in pairs])
        rows += make_rows(name, p.person_id, "impostor", "impostor", imp, cfg.data_dir, [i for i, _ in pairs])
        impostor += [o.score for o in imp if o.score is not None]

    threshold = calibrate(np.array(genuine), np.array(impostor), cfg.far, cfg.manual_tau)
    if threshold.warning:
        warnings.append(f"[{name}] {threshold.warning}")
    log.info("[%s] tau=%.4f (FAR %.2f%%)", name, threshold.tau, threshold.far_effective * 100)

    audit = getattr(recognizer, "audit", None)  # optional per-image face audit (ArcFace)

    def score_audited(template, paths):
        outcomes = recognizer.score(template, paths)
        if callable(audit):
            for o, info in zip(outcomes, audit(template, paths)):
                o.extra.update(info)
        return outcomes

    categories_seen: set[str] = set()
    for p in persons:
        t = templates[p.person_id]
        for cat, paths in p.probes.items():
            log.info("[%s] scoring %s / %s (%d images)", name, p.person_id, cat, len(paths))
            categories_seen.add(cat)
            rows += make_rows(name, p.person_id, "probe", cat, score_audited(t, paths), cfg.data_dir)
        if p.twins:
            log.info("[%s] scoring %s / twins (%d images)", name, p.person_id, len(p.twins))
            rows += make_rows(name, p.person_id, "twin", "twin", score_audited(t, p.twins), cfg.data_dir)

    stats = recognizer.finish()
    df = finalize_rows(pd.DataFrame(rows, columns=ROW_COLUMNS), threshold)
    ordered = [c for c in CATEGORIES if c in categories_seen] + sorted(categories_seen - set(CATEGORIES))
    return BenchResult(
        backend=name,
        rows=df,
        threshold=threshold,
        persons=[p.person_id for p in persons],
        skipped=skipped,
        warnings=warnings,
        config=cfg,
        model_name=recognizer.model_name,
        categories=ordered,
        stats=stats,
    )


def run_benchmark(cfg: BenchConfig, recognizer: Recognizer) -> BenchResult:
    """Prepare the data and run a single backend."""
    return run_backend(cfg, prepare_data(cfg), recognizer)


def run_all(cfg: BenchConfig, recognizers: list[Recognizer]) -> list[BenchResult]:
    """Run several backends on the same prepared data (same split and pairs)."""
    for r in recognizers:  # e.g. Ollama reachable, before hours of other work
        preflight = getattr(r, "preflight", None)
        if callable(preflight):
            preflight()
    data = prepare_data(cfg)
    return [run_backend(cfg, data, r) for r in recognizers]


def finalize_rows(df: pd.DataFrame, threshold: ThresholdResult) -> pd.DataFrame:
    """Add face_score and recognized (failures -> not recognized)."""
    df = df.copy()
    score = df["score"].to_numpy(dtype=np.float64)
    if np.isnan(threshold.impostor_mean) or threshold.genuine_mean <= threshold.impostor_mean:
        df["face_score"] = np.nan
    else:
        df["face_score"] = face_score(score, threshold.impostor_mean, threshold.genuine_mean)
    df["valid"] = df["valid"].astype(bool)
    df["recognized"] = df["valid"] & (np.nan_to_num(score, nan=-np.inf) >= threshold.tau)
    return df


def save_raw(results: BenchResult | list[BenchResult]) -> list[Path]:
    """Write results/raw/<person>.csv (all backends, ``backend`` column)."""
    results = [results] if isinstance(results, BenchResult) else results
    rows = pd.concat([r.rows for r in results], ignore_index=True)
    raw_dir = results[0].config.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for person, group in rows.groupby("person", sort=True):
        path = raw_dir / f"{person}.csv"
        group.to_csv(path, index=False, float_format="%.6f")
        written.append(path)
    return written
