"""Per-photo / per-effect Turkish report built from run outputs."""

from __future__ import annotations

import csv

from frbench import cli
from frbench.backends import ArcFaceRecognizer
from frbench.effects import build_report, load_labels

from conftest import FakeEmbedder, write_coded_image


def run(dataset, out):
    return cli.main(
        ["run", "--data", str(dataset), "--out", str(out), "--results-md", "", "--no-cache"],
        factories={"arcface": lambda args, cache: ArcFaceRecognizer(FakeEmbedder())},
    )


def write_labels(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "file", "efekt", "aciklama"])
        w.writerows(rows)


def test_report_without_labels_says_fill_in(dataset, tmp_path):
    out = tmp_path / "results"
    assert run(dataset, out) == 0
    labels = tmp_path / "labels.csv"
    write_labels(labels, [[i + 1, f"e_{i}.png", "", ""] for i in range(4)])
    md = build_report(out, labels, category="effect")
    assert "Efekt sütunu henüz doldurulmadı" in md
    assert "(etiketsiz)" in md
    assert "⚠️ yüz bulunamadı" in md  # e_0 / e_1 have no face
    assert "### 2.3 Fotoğraf bazında sonuçlar" in md and "### 2.2 Öne çıkanlar" in md
    assert "Qwen-VL daha az" not in md  # Qwen caveat only when Qwen ran


def test_report_groups_by_effect(dataset, tmp_path):
    out = tmp_path / "results"
    run(dataset, out)
    labels = tmp_path / "labels.csv"
    write_labels(labels, [[1, "e_0.png", "Lens", "mavi"], [2, "e_1.png", "lens", ""],
                          [3, "e_2.png", "makeup", "ağır"], [4, "e_3.png", "", ""]])
    md = build_report(out, labels, category="effect")
    lens = next(line for line in md.splitlines() if line.startswith("| lens |"))
    assert "%0.0" in lens  # both lens photos have no face -> not recognized
    assert "| makeup |" in md
    assert "3 fotoğraf henüz etiketlenmedi" in md  # e_3 of each of the 3 persons
    assert "| 1 | lens | mavi |" in md


def test_effects_cli(dataset, tmp_path, capsys):
    out = tmp_path / "results"
    run(dataset, out)
    labels = tmp_path / "labels.csv"
    write_labels(labels, [[1, "f_0.png", "filter", ""]])
    report = tmp_path / "RAPOR.md"
    code = cli.main(["effects", "--results", str(out), "--labels", str(labels),
                     "--category", "filter", "--out", str(report)])
    assert code == 0 and report.exists()
    assert "| filter |" in report.read_text()


def test_load_labels_missing_file(tmp_path):
    assert load_labels(tmp_path / "nope.csv").empty


def test_weak_match_flagged(dataset, tmp_path):
    out = tmp_path / "results"
    run(dataset, out)
    import pandas as pd

    rows = pd.read_csv(out / "per_image.csv")
    target = rows[(rows.role == "probe") & rows.file.str.endswith("alice/filter/f_0.png")].index[0]
    rows.loc[target, "face_score"] = 12.0
    rows.to_csv(out / "per_image.csv", index=False)
    labels = tmp_path / "labels.csv"
    write_labels(labels, [[1, "f_0.png", "", ""]])
    md = build_report(out, labels, category="filter")
    assert "✅ tanındı (zayıf)" in md and "zayıf eşleşme: 1;" in md


def test_default_report_covers_all_categories(dataset, tmp_path):
    out = tmp_path / "results"
    run(dataset, out)
    md = build_report(out, docs=tmp_path)  # no label sheets -> all unlabeled
    for cat in ("filter", "makeup", "effect"):
        assert f"Kategori: {cat}" in md
    assert "filter tanıma" in md and "İkiz" not in md.split("## 2.")[0]


def test_twin_section_inverts_verdicts(dataset, tmp_path):
    write_coded_image(dataset / "twins" / "alice" / "t_0.png", 90, 50_000, 0.3)  # different identity
    write_coded_image(dataset / "twins" / "alice" / "t_1.png", 1, 50_001, 0.3)  # indistinguishable
    out = tmp_path / "results"
    run(dataset, out)
    md = build_report(out, docs=tmp_path)
    assert "İkiz (benzer yüz) fotoğrafları" in md and "İkiz kabul (düşük = iyi)" in md
    assert "✅ reddedildi" in md and "❌ kabul edildi (karıştırdı)" in md
    assert "%50.0 (1/2" in md


def test_wrong_face_flagged_in_report(dataset, tmp_path):
    import pandas as pd

    out = tmp_path / "results"
    run(dataset, out)
    rows = pd.read_csv(out / "per_image.csv")
    i = rows[(rows.role == "probe") & rows.file.str.endswith("alice/filter/f_0.png")].index[0]
    rows.loc[i, ["n_faces", "best_face_score", "score"]] = [3, 0.9, 0.2]
    rows.to_csv(out / "per_image.csv", index=False)
    md = build_report(out, category="filter", labels=tmp_path / "none.csv")
    assert "Birden fazla yüz:** 1 fotoğrafta" in md
    assert "Yanlış yüz seçilmiş olabilir" in md and "| Yüz sayısı | En benzer yüz |" in md
