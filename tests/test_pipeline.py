"""T12-T17: pipeline integration on a synthetic dataset with a fake embedder."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from frbench import cli
from frbench.backends import ArcFaceRecognizer
from frbench.evaluate import BenchConfig, run_benchmark, save_raw
from frbench.report import summarize, write_reports

from conftest import FakeEmbedder, write_coded_image


@pytest.fixture
def result(dataset, tmp_path, fake_embedder):
    cfg = BenchConfig(data_dir=dataset, out_dir=tmp_path / "results", use_cache=False)
    return run_benchmark(cfg, ArcFaceRecognizer(fake_embedder))


def probe_rows(result, person, category):
    df = result.rows
    return df[(df.person == person) & (df.role == "probe") & (df.category == category)]


def test_t12_same_person_clean_recognized(result):
    clean = result.rows[result.rows.role == "clean"]
    assert len(clean) > 0 and clean.recognized.all()


def test_t13_different_person_not_recognized(result):
    imp = result.rows[result.rows.role == "impostor"]
    assert len(imp) > 0
    assert imp.recognized.mean() <= result.threshold.far_effective + 1e-9
    assert (imp.impostor_id != imp.person).all()


def test_t14_detection_failure_counted(result):
    eff = probe_rows(result, "alice", "effect")
    fails = eff[~eff.valid]
    assert len(fails) == 2
    assert not fails.recognized.any()
    assert (fails.error == "no_face").all()
    summ = summarize(result.rows, result.categories).set_index("category")
    assert summ.loc["effect", "detection_fails"] == 2 * len(result.persons)
    assert summ.loc["effect", "probes"] == 4 * len(result.persons)  # failures not dropped (B3)


def test_t15_empty_category_skipped(result):
    assert "botox" not in result.categories
    assert any("Empty category folder" in w for w in result.warnings)


def test_t16_person_with_few_photos_warned(result):
    assert "tiny" in result.skipped and "tiny" not in result.persons
    assert any("tiny" in w and "skipped" in w for w in result.warnings)
    assert any("'tiny' ignored" in w for w in result.warnings)


def test_metrics_ordering(result):
    s = summarize(result.rows, result.categories).set_index("category")
    clean = s.loc["clean (held-out)"]
    assert clean.retention == pytest.approx(1.0)
    assert s.loc["makeup", "mean_score"] < s.loc["filter", "mean_score"] < clean.mean_score
    assert 0 <= s.loc["makeup", "mean_face_score"] <= 100


def test_b6_probe_identical_to_clean_excluded(dataset, tmp_path, fake_embedder):
    src = dataset / "gallery" / "alice" / "clean_00.png"
    leak = dataset / "probes" / "alice" / "filter" / "leak.png"
    leak.write_bytes(src.read_bytes())
    res = run_benchmark(BenchConfig(dataset, tmp_path / "r", use_cache=False), ArcFaceRecognizer(fake_embedder))
    assert not res.rows.file.str.contains("leak.png").any()
    assert any("leak" in w for w in res.warnings)


def test_b7_deterministic(dataset, tmp_path):
    a = run_benchmark(BenchConfig(dataset, tmp_path / "a", use_cache=False), ArcFaceRecognizer(FakeEmbedder()))
    b = run_benchmark(BenchConfig(dataset, tmp_path / "b", use_cache=False), ArcFaceRecognizer(FakeEmbedder()))
    assert a.threshold.tau == b.threshold.tau
    pd.testing.assert_frame_equal(a.rows, b.rows)


def test_cache_reuses_embeddings(dataset, tmp_path):
    cfg = BenchConfig(dataset, tmp_path / "r")
    first, second = FakeEmbedder(), FakeEmbedder()
    r1 = run_benchmark(cfg, ArcFaceRecognizer(first, tmp_path / "cache"))
    r2 = run_benchmark(cfg, ArcFaceRecognizer(second, tmp_path / "cache"))
    assert first.calls > 0 and second.calls == 0
    assert r1.threshold.tau == r2.threshold.tau


def test_no_cross_impostors_uses_only_impostor_folder(dataset, tmp_path, fake_embedder):
    cfg = BenchConfig(dataset, tmp_path / "r", use_cache=False, cross_impostors=False)
    res = run_benchmark(cfg, ArcFaceRecognizer(fake_embedder))
    imp = res.rows[res.rows.role == "impostor"]
    assert set(imp.impostor_id) == {f"other_{i}" for i in range(6)}


def test_raw_csv_and_reports(result, tmp_path):
    written = save_raw(result)
    assert sorted(p.stem for p in written) == ["alice", "bob", "carol"]
    paths = write_reports(result, tmp_path / "docs" / "RESULTS.md")
    md = paths["results_md"].read_text()
    assert "| clean (held-out) |" in md and "| makeup |" in md
    payload = json.loads(paths["summary_json"].read_text())
    assert payload["backends"]["arcface"]["skipped_persons"].keys() == {"tiny"}


def test_t17_full_cli_run(dataset, tmp_path, capsys):
    out, md = tmp_path / "results", tmp_path / "docs" / "RESULTS.md"
    code = cli.main(
        ["run", "--data", str(dataset), "--out", str(out), "--results-md", str(md), "--no-cache"],
        factories={"arcface": lambda args, cache: ArcFaceRecognizer(FakeEmbedder(), cache)},
    )
    assert code == 0
    for name in ("per_image.csv", "summary.json", "summary.md", "raw/alice.csv"):
        assert (out / name).exists(), name
    assert md.exists()
    assert "tau=" in capsys.readouterr().out


def test_cli_check(dataset, capsys):
    code = cli.main(["check", "--data", str(dataset)])
    out = capsys.readouterr().out
    assert code == 1  # 'tiny' has too few clean photos
    assert "alice" in out and "needs >= 10" in out


def test_unreadable_probe_is_failure_not_crash(dataset, tmp_path, fake_embedder):
    (dataset / "probes" / "bob" / "filter" / "broken.jpg").write_bytes(b"garbage")
    res = run_benchmark(BenchConfig(dataset, tmp_path / "r", use_cache=False), ArcFaceRecognizer(fake_embedder))
    row = res.rows[res.rows.file.str.endswith("broken.jpg")].iloc[0]
    assert not row.valid and not row.recognized and row.error == "unreadable"
    assert np.isnan(row.score)


def test_gallery_with_mostly_failed_detections_skipped(tmp_path, fake_embedder):
    root = tmp_path / "data"
    for i in range(12):
        write_coded_image(root / "gallery" / "ghost" / f"{i}.png", 9, 100 + i, face=i < 2)
    for i in range(12):
        write_coded_image(root / "gallery" / "ok" / f"{i}.png", 8, 200 + i)
    for i in range(5):
        write_coded_image(root / "impostors" / "x" / f"{i}.png", 70, 300 + i)
    res = run_benchmark(BenchConfig(root, tmp_path / "r", use_cache=False), ArcFaceRecognizer(fake_embedder))
    assert "ghost" in res.skipped and res.persons == ["ok"]


# --------------------------------------------------------------------------- #
# Twins (look-alikes) and the ArcFace face audit
# --------------------------------------------------------------------------- #


def add_twins(root):
    """alice gets 3 look-alikes of other identities and 1 that the model can't tell apart."""
    for i in range(3):
        write_coded_image(root / "twins" / "alice" / f"t_{i}.png", 90 + i, 50_000 + i, 0.3)
    write_coded_image(root / "twins" / "alice" / "t_3.png", 1, 50_100, 0.3)  # same identity code as alice


def test_twins_scored_but_not_used_for_calibration(dataset, tmp_path):
    base = run_benchmark(BenchConfig(dataset, tmp_path / "a", use_cache=False), ArcFaceRecognizer(FakeEmbedder()))
    add_twins(dataset)
    res = run_benchmark(BenchConfig(dataset, tmp_path / "b", use_cache=False), ArcFaceRecognizer(FakeEmbedder()))
    assert res.threshold.tau == base.threshold.tau
    assert res.threshold.n_impostor == base.threshold.n_impostor
    twins = res.rows[res.rows.role == "twin"]
    assert len(twins) == 4 and set(twins.person) == {"alice"}
    assert twins.recognized.sum() == 1  # only the indistinguishable one is (wrongly) accepted
    summ = summarize(res.rows, res.categories).set_index("category")
    assert summ.loc["twin (look-alike, accept = error)", "recognition_rate"] == pytest.approx(25.0)


class MultiFaceEmbedder(FakeEmbedder):
    """Adds a second, bigger face of identity 77 to every image whose name contains 'crowd'."""

    def embed_all(self, path):
        own = self.embed(path)
        faces = [] if own is None else [(100.0, own)]
        if "crowd" in str(path):
            other = np.random.default_rng(77).normal(size=own.size if own is not None else 512)
            faces.append((400.0, (other / np.linalg.norm(other)).astype(np.float32)))
        return faces


def test_face_audit_records_faces_and_best_match(dataset, tmp_path):
    write_coded_image(dataset / "probes" / "bob" / "filter" / "crowd_0.png", 2, 51_000, 0.3)
    res = run_benchmark(BenchConfig(dataset, tmp_path / "r", use_cache=False), ArcFaceRecognizer(MultiFaceEmbedder()))
    probes = res.rows[res.rows.role == "probe"]
    crowd = probes[probes.file.str.endswith("crowd_0.png")].iloc[0]
    assert crowd.n_faces == 2 and crowd.best_face_score > 0.5
    assert (probes[~probes.file.str.contains("crowd")].n_faces.dropna() <= 1).all()
    assert res.rows[res.rows.role == "impostor"].n_faces.isna().all()  # audit skipped for impostors
