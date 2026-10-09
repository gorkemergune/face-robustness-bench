"""Per-photo and per-effect report (Turkish) from benchmark outputs.

Inputs (no model needed, so it can be re-run after labels are filled in):

    results/per_image.csv     rows from ``frbench.cli run``
    results/summary.json      thresholds and per-category metrics
    docs/<category>_labels.csv  no,file,efekt,aciklama   (filled by the user)

Sections: one per probe category (e.g. ``dirty``, ``bad``: the person, should
be recognized) and one for look-alike ``twin`` photos (not the person, should
be rejected).

    python -m frbench.cli effects
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from .report import label, wilson_interval

UNLABELED = "(etiketsiz)"
WEAK_FACE_SCORE = 50.0  # recognized, but closer to a stranger than to a clean photo
WRONG_FACE_MARGIN = 0.10  # another face in the photo beats the largest face by this much
TWIN = "twin"
DEFAULT_SECTIONS = ("dirty", "bad", TWIN)
TITLES = {
    "dirty": "Değişiklikli fotoğraflar (efekt / makyaj / filtre)",
    "bad": "Kötü fotoğraflar (düşük kalite / yarım yüz / yanında başkası)",
    TWIN: "İkiz (benzer yüz) fotoğrafları",
}
ERROR_NAMES = {
    "no_face": "yüz bulunamadı", "refusal": "cevap reddedildi", "invalid_json": "geçersiz cevap",
    "unreadable": "okunamadı", "backend_error": "bağlantı hatası",
}


def load_labels(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["no", "file", "efekt", "aciklama"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("efekt", "aciklama"):
        if col not in df:
            df[col] = ""
    df["efekt"] = df["efekt"].str.strip().str.lower()
    df["aciklama"] = df["aciklama"].str.strip()
    return df


def _is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower() == "true"


def _num(value) -> float:
    return pd.to_numeric(value, errors="coerce")


def photo_table(rows: pd.DataFrame, labels: pd.DataFrame, category: str) -> pd.DataFrame:
    """One row per photo of ``category`` (probe category or ``twin``), one column group per backend."""
    mask = rows["role"] == TWIN if category == TWIN else (rows["role"] == "probe") & (rows["category"] == category)
    sel = rows[mask].copy()
    sel["fname"] = sel["file"].map(lambda f: Path(f).name)
    wanted = ["valid", "error", "score", "face_score", "recognized", "n_same", "n_valid", "n_faces", "best_face_score"]
    out = None
    for backend, g in sel.groupby("backend", sort=False):
        g = g.reindex(columns=[*g.columns, *(c for c in wanted if c not in g)])
        cols = g[["person", "fname", *wanted]].rename(
            columns=lambda c: c if c in ("person", "fname") else f"{backend}:{c}"
        )
        out = cols if out is None else out.merge(cols, on=["person", "fname"], how="outer")
    if out is None:
        return pd.DataFrame()
    out = out.merge(labels.rename(columns={"file": "fname"}), on="fname", how="left")
    out["efekt"] = out["efekt"].fillna("").replace("", UNLABELED)
    out["aciklama"] = out["aciklama"].fillna("")
    out["no"] = pd.to_numeric(out["no"], errors="coerce")
    return out.sort_values(["person", "no", "fname"]).reset_index(drop=True)


def backends_in(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(c.split(":")[0] for c in table.columns if ":" in c))


def _score_text(row: pd.Series, backend: str) -> str:
    text = f"{float(row[f'{backend}:score']):.3f}"
    n_same = _num(row.get(f"{backend}:n_same"))
    if backend == "qwen_vl" and not pd.isna(n_same):
        text += f" ({int(n_same)}/{int(_num(row[f'{backend}:n_valid']))} aynı)"
    return text


def _weak(row: pd.Series, backend: str) -> bool:
    """Close call: face score below 50, or (Qwen) majority vote disagrees with the mean score."""
    fs = _num(row.get(f"{backend}:face_score"))
    n_same, n_valid = _num(row.get(f"{backend}:n_same")), _num(row.get(f"{backend}:n_valid"))
    recognized = str(row.get(f"{backend}:recognized")).lower() == "true"
    vote_same = None if pd.isna(n_same) or pd.isna(n_valid) else 2 * n_same > n_valid
    if recognized:
        return (not pd.isna(fs) and fs < WEAK_FACE_SCORE) or vote_same is False
    return (not pd.isna(fs) and fs >= WEAK_FACE_SCORE) or vote_same is True


def _cell(row: pd.Series, backend: str, twin: bool = False) -> tuple[str, str]:
    """(score text, verdict text) for one backend. For twins, rejecting is the correct outcome."""
    if str(row.get(f"{backend}:valid")).lower() != "true":
        err = row.get(f"{backend}:error") or "hata"
        return "-", f"⚠️ {ERROR_NAMES.get(err, err)}"
    text = _score_text(row, backend)
    recognized = str(row.get(f"{backend}:recognized")).lower() == "true"
    weak = _weak(row, backend)
    if twin:
        if recognized:
            return text, "❌ kabul edildi (karıştırdı)"
        return text, "✅ reddedildi (yakın)" if weak else "✅ reddedildi"
    if not recognized:
        return text, "❌ tanınmadı"
    return text, "✅ tanındı (zayıf)" if weak else "✅ tanındı"


def _rate(series: pd.Series) -> str:
    rec, n = int(_is_true(series).sum()), int(len(series))
    if n == 0:
        return "-"
    lo, hi = wilson_interval(rec, n)
    return f"%{rec / n * 100:.1f} ({rec}/{n}; GA %{lo * 100:.0f}-{hi * 100:.0f})"


def effect_summary(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for effect, g in table.groupby("efekt", sort=True):
        rec = {"Efekt": effect, "Foto": len(g)}
        for b in backends_in(table):
            rec[f"{label(b)} doğruluk"] = _rate(g[f"{b}:recognized"])
        records.append(rec)
    return pd.DataFrame(records)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _fmt(v, spec: str) -> str:
    try:
        return "-" if v is None or math.isnan(float(v)) else format(v, spec)
    except (TypeError, ValueError):
        return "-"


def _no(r: pd.Series) -> str:
    return str(int(r["no"])) if not pd.isna(r["no"]) else r["fname"]


def _has_audit(table: pd.DataFrame) -> bool:
    col = "arcface:n_faces"
    return col in table and _num(table[col]).notna().any()


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def summary_section(summary: dict, rows: pd.DataFrame, categories: list[str]) -> list[str]:
    probe_cats = [c for c in categories if c != TWIN]
    header = ["Model", "Eşik (tau)", "Gerçekleşen FAR", "Temiz test tanıma"]
    header += [f"{c} tanıma" for c in probe_cats]
    if TWIN in categories:
        header.append("İkiz kabul (düşük = iyi)")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for backend, info in summary["backends"].items():
        t = info["threshold"]
        b = rows[rows["backend"] == backend]
        cells = [f"{label(backend)} ({info['model']})", f"{t['tau']:.3f}",
                 f"%{_fmt((t['far_achieved'] or 0) * 100, '.2f')} ({t['n_impostor']} yabancı skoru)",
                 _rate(b[b["role"] == "clean"]["recognized"])]
        cells += [f"**{_rate(b[(b['role'] == 'probe') & (b['category'] == c)]['recognized'])}**" for c in probe_cats]
        if TWIN in categories:
            cells.append(f"**{_rate(b[b['role'] == TWIN]['recognized'])}**")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def probe_section(num: int, category: str, table: pd.DataFrame, labels_path: Path | None) -> list[str]:
    backends = backends_in(table)
    labeled = int((table["efekt"] != UNLABELED).sum())
    lines = [f"## {num}. {TITLES.get(category, f'Kategori: {category}')}", "",
             f"_{len(table)} fotoğraf. Kişinin kendisi; doğru sonuç: **tanınmalı**._", "",
             f"### {num}.1 Efekt / durum bazında doğruluk", ""]
    if labeled == 0:
        lines += [f"> Efekt sütunu henüz doldurulmadı (`{labels_path}`). Her fotoğraf numarası için `efekt` ve "
                  "isteğe bağlı `aciklama` yazıp `python -m frbench.cli effects` çalıştırın.", ""]
    elif labeled < len(table):
        lines += [f"> {len(table) - labeled} fotoğraf henüz etiketlenmedi ({UNLABELED} satırı).", ""]
    lines += [_md(effect_summary(table)), "", f"### {num}.2 Öne çıkanlar", ""]

    for b in backends:
        scored = table[_is_true(table[f"{b}:valid"])].copy()
        scored["fs"] = _num(scored[f"{b}:face_score"])
        worst = ", ".join(f"{_no(r)} (yüz skoru {r['fs']:.0f})" for _, r in scored.sort_values("fs").head(3).iterrows())
        failed = table[~_is_true(table[f"{b}:recognized"])]
        no_face = table[~_is_true(table[f"{b}:valid"])]
        weak = [_no(r) for _, r in table.iterrows() if _cell(r, b)[1] == "✅ tanındı (zayıf)"]
        lines.append(f"- **{label(b)}:** tanınmayan: {', '.join(_no(r) for _, r in failed.iterrows()) or 'yok'}"
                     + (f" (bunların {len(no_face)} tanesinde {('yüz bulunamadı' if b == 'arcface' else 'cevap alınamadı')})"
                        if len(no_face) else "")
                     + f"; zayıf eşleşme: {', '.join(weak) or 'yok'}; en düşük skorlar: {worst or '-'}.")
    if len(backends) > 1:
        rec = {b: _is_true(table[f"{b}:recognized"]) for b in backends}
        only = {b: table[rec[b] & ~rec[o]] for b, o in (backends[:2], backends[1::-1])}
        for b, g in only.items():
            if len(g):
                lines.append(f"- **Sadece {label(b)} tanıdı:** {', '.join(_no(r) for _, r in g.iterrows())}.")
        if not any(len(g) for g in only.values()):
            lines.append("- İki model her fotoğrafta aynı kararı verdi.")
    if _has_audit(table):
        multi = table[_num(table["arcface:n_faces"]) > 1]
        wrong = multi[(_num(multi["arcface:best_face_score"]) - _num(multi["arcface:score"])) > WRONG_FACE_MARGIN]
        if len(multi):
            lines.append(f"- **Birden fazla yüz:** {len(multi)} fotoğrafta (" + ", ".join(_no(r) for _, r in multi.iterrows())
                         + "). ArcFace en büyük yüzü kullanır.")
        if len(wrong):
            lines.append("- **Yanlış yüz seçilmiş olabilir (ArcFace):** " + ", ".join(
                f"{_no(r)} (seçilen yüz {float(r['arcface:score']):.2f}, fotoğraftaki en benzer yüz "
                f"{float(r['arcface:best_face_score']):.2f})" for _, r in wrong.iterrows())
                + ". Kişi fotoğrafta var ama en büyük yüz o değil.")

    lines += ["", f"### {num}.3 Fotoğraf bazında sonuçlar", ""]
    audit = _has_audit(table)
    header = ["No", "Efekt", "Açıklama"]
    for b in backends:
        header += [f"{label(b)} skor", f"{label(b)} sonuç"]
        if b == "arcface" and audit:
            header += ["Yüz sayısı", "En benzer yüz"]
    body = []
    for _, r in table.iterrows():
        cells = [_no(r), "" if r["efekt"] == UNLABELED else r["efekt"], r["aciklama"]]
        for b in backends:
            cells += list(_cell(r, b))
            if b == "arcface" and audit:
                n = _num(r.get("arcface:n_faces"))
                cells += ["-" if pd.isna(n) else str(int(n)), _fmt(_num(r.get("arcface:best_face_score")), ".3f")]
        body.append(cells)
    return lines + [_md(pd.DataFrame(body, columns=header)), ""]


def twin_section(num: int, table: pd.DataFrame, summary: dict, rows: pd.DataFrame) -> list[str]:
    backends = backends_in(table)
    lines = [f"## {num}. {TITLES[TWIN]}", "",
             f"_{len(table)} fotoğraf. Kişinin kendisi **değil**, ona benzetilmiş yüzler; doğru sonuç: **reddedilmeli**. "
             "Kabul edilmesi, sistemin kişiyi \"ikiziyle\" karıştırdığı anlamına gelir._", "",
             f"### {num}.1 İkiz kabul oranı", "",
             "| Model | İkiz kabul (düşük = iyi) | Ort. ikiz skoru | Ort. temiz skor | Ort. yabancı (LFW) skor | Ort. ikiz yüz skoru |",
             "|---|---|---|---|---|---|"]
    for b in backends:
        t = summary["backends"][b]["threshold"]
        clean = rows[(rows["backend"] == b) & (rows["role"] == "clean")]
        lines.append(
            f"| {label(b)} | **{_rate(table[f'{b}:recognized'])}** | {_fmt(_num(table[f'{b}:score']).mean(), '.3f')} | "
            f"{_fmt(_num(clean['score']).mean(), '.3f')} | {_fmt(t['impostor_mean'], '.3f')} | "
            f"{_fmt(_num(table[f'{b}:face_score']).mean(), '.0f')} |")
    lines += ["", "- Ort. ikiz yüz skoru: 0 = ortalama bir yabancı kadar benzer, 100 = kişinin kendi temiz fotoğrafı kadar benzer.",
              "- \"reddedildi (yakın)\": doğru karar ama skor yabancılardan çok kişinin kendisine yakın (yüz skoru ≥ 50 "
              "veya Qwen'de çoğunluk \"aynı\" dedi).", "", f"### {num}.2 Fotoğraf bazında sonuçlar", ""]
    header = ["No", "Açıklama"]
    for b in backends:
        header += [f"{label(b)} skor", f"{label(b)} sonuç"]
    body = []
    for _, r in table.iterrows():
        cells = [_no(r), r["aciklama"]]
        for b in backends:
            cells += list(_cell(r, b, twin=True))
        body.append(cells)
    lines += [_md(pd.DataFrame(body, columns=header)), ""]
    if "arcface" in backends:
        lines += threshold_tradeoff(num, rows, summary["backends"]["arcface"]["threshold"]["tau"])
    return lines


TRADEOFF_TAUS = (0.30, 0.40, 0.50, 0.60)


def threshold_tradeoff(num: int, rows: pd.DataFrame, tau: float) -> list[str]:
    """ArcFace: what a stricter threshold would do to twins vs the person's own photos."""
    a = rows[rows["backend"] == "arcface"]
    order = {c: i for i, c in enumerate(DEFAULT_SECTIONS)}
    cats = sorted(set(a.loc[a["role"] == "probe", "category"]), key=lambda c: (order.get(c, len(order)), c))
    groups = [("Temiz test", a["role"] == "clean")]
    groups += [(f"{c} tanıma", (a["role"] == "probe") & (a["category"] == c)) for c in cats]
    groups += [("İkiz kabul ↓", a["role"] == TWIN), ("LFW yabancı kabul ↓", a["role"] == "impostor")]
    header = ["Eşik"] + [g for g, _ in groups]
    body = []
    for name, t in [(f"{tau:.3f} (mevcut, FAR %1)", tau)] + [(f"{t:.2f}", t) for t in TRADEOFF_TAUS]:
        cells = [name]
        for _, m in groups:
            sc = _num(a.loc[m, "score"])
            cells.append(f"%{(sc.fillna(-9) >= t).mean() * 100:.1f}" if len(sc) else "-")
        body.append(cells)
    return [f"### {num}.3 Eşik yükseltilirse ne olur? (ArcFace)", "",
            "İkizleri reddetmek için eşik yükseltilirse kişinin kendi fotoğraflarının tanınması da düşer:", "",
            _md(pd.DataFrame(body, columns=header)), "",
            "- Hiçbir eşik ikizleri kişinin kendi fotoğraflarından temiz şekilde ayıramıyorsa, sistem bu ikizleri "
            "kişiden ayırt edemiyor demektir.", ""]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _resolve_sections(rows: pd.DataFrame, labels, category: str | None, docs: Path) -> dict[str, Path | None]:
    if category is not None:
        return {category: labels if isinstance(labels, Path) else docs / f"{category}_labels.csv"}
    if isinstance(labels, dict):
        return labels
    present = set(rows.loc[rows["role"] == "probe", "category"]) | ({TWIN} if (rows["role"] == TWIN).any() else set())
    ordered = [c for c in DEFAULT_SECTIONS if c in present] + sorted(present - set(DEFAULT_SECTIONS))
    return {c: docs / f"{c}_labels.csv" for c in ordered}


def build_report(
    results_dir: Path,
    labels: Path | dict[str, Path] | None = None,
    category: str | None = None,
    docs: Path = Path("docs"),
) -> str:
    rows = pd.read_csv(results_dir / "per_image.csv", dtype={"error": str}, keep_default_na=False,
                       na_values={c: [""] for c in ("score", "face_score", "mean_confidence", "best_face_score")})
    summary = json.loads((results_dir / "summary.json").read_text())
    sections = _resolve_sections(rows, labels, category, docs)
    tables = {c: photo_table(rows, load_labels(p), c) for c, p in sections.items()}
    tables = {c: t for c, t in tables.items() if len(t)}
    persons = sorted(rows["person"].unique())
    first_backend = rows["backend"].iloc[0]
    count = lambda role: int(((rows["role"] == role) & (rows["backend"] == first_backend)).sum())  # noqa: E731
    backends = list(summary["backends"])

    lines = [
        f"# Yüz Tanıma Dayanıklılık Raporu: {', '.join(persons)}",
        "",
        f"_Oluşturulma: {datetime.now():%d.%m.%Y %H:%M} · Modeller: "
        + ", ".join(f"{label(b)} ({i['model']})" for b, i in summary["backends"].items())
        + f" · Hedef FAR: %{summary['config']['far'] * 100:.0f}_",
        "",
        "## 1. Özet",
        "",
        *summary_section(summary, rows, list(tables)),
        "",
        "- **Tanıma:** kişinin kendi fotoğraflarının tanınma oranı (yüksek = iyi). GA = %95 güven aralığı.",
        *(["- **İkiz kabul:** benzer yüzlerin kişi sanılma oranı (düşük = iyi)."] if TWIN in tables else []),
        f"- Veri: {count('gallery')} galeri + {count('clean')} temiz test fotoğrafı, "
        + ", ".join(f"{len(t)} {c}" for c, t in tables.items())
        + f" fotoğraf; eşik için {summary['backends'][first_backend]['threshold']['n_impostor'] // max(1, len(persons))} "
        "yabancı yüz (LFW).",
        "",
    ]
    num = 2
    for cat, table in tables.items():
        lines += twin_section(num, table, summary, rows) if cat == TWIN else probe_section(num, cat, table, sections[cat])
        num += 1

    lines += [
        "**Notlar:** Yüz skoru (0-100): 0 = ortalama bir yabancı kadar benzer, 100 = ortalama bir temiz foto kadar benzer. "
        "ArcFace skoru: kişinin şablonuna kosinüs benzerliği. Qwen-VL skoru: 3 galeri fotoğrafıyla sorulan "
        "\"aynı kişi mi?\" cevaplarından ortalama P(aynı); parantez içinde kaç cevabın \"aynı\" dediği. "
        "\"zayıf\": tanındı ama yüz skoru 50'nin altında veya Qwen'de çoğunluk oyu \"farklı\". "
        f"Her model kendi eşiğini (tau) kullanır; eşik, temiz test fotoğrafları ile LFW yabancıları arasında "
        f"%{summary['config']['far'] * 100:.0f} yanlış kabul oranına göre ayarlandı. İkiz fotoğraflar eşik ayarına katılmadı.",
        "",
    ]

    excluded = [w for w in summary.get("warnings", []) if "is identical to a clean photo" in w]
    if excluded:
        names = [w.split(" is identical")[0].split()[-1] for w in excluded]
        lines += ["**Hariç tutulan fotoğraflar:** "
                  + ", ".join(f"`{Path(n).parent.name}/{Path(n).name}`" for n in names)
                  + " bir temiz fotoğrafın birebir kopyası olduğu için değerlendirmeye alınmadı "
                  "(aynı fotoğraf hem şablonda hem testte olursa sonuç yapay olarak iyi çıkar).", ""]

    clean_fail = rows[(rows["role"] == "clean") & ~_is_true(rows["recognized"])]
    lines += [f"## {num}. Temiz test fotoğrafları (referans)", ""]
    for b in backends:
        g = clean_fail[clean_fail["backend"] == b]
        names = ", ".join(f"`{Path(f).name}`" + (f" ({e})" if e else "") for f, e in zip(g["file"], g["error"]))
        lines.append(f"- {label(b)}: tanınmayan temiz foto: {names or 'yok'}")
    lines += [
        "",
        f"## {num + 1}. Yöntem",
        "",
        "1. Fotoğraflar HEIC/PNG/JPG'den JPEG'e çevrildi, yönleri düzeltildi, konum (EXIF/GPS) bilgisi silindi; temiz "
        "fotoğraflar `clean_01…`, diğerleri her klasörde `01…` olarak yeniden adlandırıldı.",
        "2. Temiz fotoğraflar sabit seed (42) ile %60 galeri / %40 test olarak bölündü. Galeri kişinin şablonunu oluşturur.",
        "3. **ArcFace:** her fotoğrafta en büyük yüz bulunur, hizalanır, 512 boyutlu vektöre çevrilir; "
        "skor = galeri ortalamasına kosinüs benzerliği. Ayrıca fotoğraftaki tüm yüzler sayılır ve kişiye en çok "
        "benzeyen yüzün skoru kaydedilir (yanlış yüz seçimini görmek için).",
        "4. **Qwen2.5-VL (Ollama, yerel):** her fotoğraf 3 farklı galeri fotoğrafıyla ikili olarak gösterilip "
        "\"aynı kişi mi? makyaj/filtre/efekti yok say\" diye soruldu (temperature 0, sabit seed). "
        "Reddetme veya geçersiz cevap hata olarak sayıldı.",
        "5. Eşik: LFW'den rastgele seçilen yabancı yüzlerle ("
        + ", ".join(f"{label(b)} {i['threshold']['n_impostor']} skor" for b, i in summary["backends"].items())
        + f") %{summary['config']['far'] * 100:.0f} FAR'a göre ayarlandı.",
        "",
        f"## {num + 2}. Kısıtlar",
        "",
        f"- {len(persons)} kişi; kategori başına birkaç düzine fotoğraf: sonuçlar yol göstericidir, güven aralıkları geniştir.",
        "- İkiz fotoğraflar gerçek bir ikiz değil, yapay zekâ (Gemini) ile kişiye benzetilmiş yüzlerdir."
        if TWIN in tables else
        "- Yabancılar ünlü fotoğraflarından (LFW) oluşur; kişiye benzeyen biri (ör. ikiz) ayrıca test edilmedi.",
    ]
    if "qwen_vl" in backends:
        lines.append("- Qwen-VL daha az yabancı skoruyla kalibre edilir (maliyet nedeniyle örneklem); eşiği ArcFace'inkinden "
                     "daha kabadır. Qwen'in güven değerleri kaba olduğundan (çoğunlukla 50 veya 95) skoru da birkaç "
                     "değerde toplanır.")
    lines.append("")
    return "\n".join(lines)


def write_effects_report(
    results_dir: Path,
    labels: Path | dict[str, Path] | None,
    out: Path,
    category: str | None = None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(results_dir, labels, category))
    return out
