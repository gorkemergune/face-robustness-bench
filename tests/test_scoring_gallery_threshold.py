"""T4-T11: scoring, split, centroid, face score, threshold."""

from __future__ import annotations

import numpy as np
import pytest

from frbench.gallery import InsufficientDataError, build_centroid, dedupe_by_hash, split_clean
from frbench.scoring import cosine, cosine_many, face_score
from frbench.threshold import calibrate, tau_at_far


def test_t4_cosine_identical():
    v = np.random.default_rng(0).normal(size=512)
    assert cosine(v, v) == pytest.approx(1.0)


def test_t5_cosine_orthogonal():
    assert cosine(np.array([1.0, 0, 0]), np.array([0, 2.0, 0])) == pytest.approx(0.0)


def test_cosine_many_matches_cosine():
    rng = np.random.default_rng(1)
    m, r = rng.normal(size=(5, 16)), rng.normal(size=16)
    assert np.allclose(cosine_many(m, r), [cosine(row, r) for row in m])


def make_files(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i:02d}.jpg"
        p.write_bytes(f"photo-{i}".encode())
        paths.append(p)
    return paths


def test_t6_split_deterministic(tmp_path):
    paths = make_files(tmp_path, 22)
    a = split_clean(paths, seed=42)
    b = split_clean(list(reversed(paths)), seed=42)
    assert a == b


@pytest.mark.parametrize("n,expected_gallery", [(20, 12), (25, 15)])
def test_t7_split_sizes_and_disjoint(tmp_path, n, expected_gallery):
    s = split_clean(make_files(tmp_path, n))
    assert len(s.gallery) == expected_gallery
    assert len(s.heldout) == n - expected_gallery
    assert not set(s.gallery) & set(s.heldout)


def test_b6_duplicates_removed_before_split(tmp_path):
    paths = make_files(tmp_path, 12)
    dup = tmp_path / "copy_of_0.jpg"
    dup.write_bytes(paths[0].read_bytes())
    assert len(dedupe_by_hash([*paths, dup])) == 12
    s = split_clean([*paths, dup])
    assert len(s.gallery) + len(s.heldout) == 12


def test_t16_too_few_photos(tmp_path):
    with pytest.raises(InsufficientDataError, match="at least 10"):
        split_clean(make_files(tmp_path, 9))


def test_t8_centroid_normalized():
    rng = np.random.default_rng(2)
    c = build_centroid([rng.normal(size=512) for _ in range(12)])
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6)


def test_b4_centroid_not_dominated_by_large_norm():
    c = build_centroid([np.array([100.0, 0]), np.array([0, 1.0])])
    assert c == pytest.approx([np.sqrt(0.5), np.sqrt(0.5)])


def test_t9_face_score_endpoints():
    assert face_score(0.1, impostor_mean=0.1, genuine_mean=0.7) == pytest.approx(0.0)
    assert face_score(0.7, impostor_mean=0.1, genuine_mean=0.7) == pytest.approx(100.0)
    assert face_score(0.4, impostor_mean=0.1, genuine_mean=0.7) == pytest.approx(50.0)


def test_t10_face_score_clipped():
    assert face_score(-0.5, 0.1, 0.7) == 0.0
    assert face_score(0.99, 0.1, 0.7) == 100.0
    out = face_score(np.array([-1.0, np.nan, 2.0]), 0.1, 0.7)
    assert out[0] == 0 and np.isnan(out[1]) and out[2] == 100


def test_face_score_invalid_span():
    with pytest.raises(ValueError):
        face_score(0.5, 0.6, 0.5)


def test_t11_threshold_far_1_percent():
    rng = np.random.default_rng(42)
    impostor = rng.normal(0.05, 0.08, 10_000)
    genuine = rng.normal(0.7, 0.1, 500)
    res = calibrate(genuine, impostor, far=0.01)
    assert res.far_achieved == pytest.approx(0.01, abs=0.001)
    assert res.far_achieved <= 0.01
    assert res.warning is None
    assert res.frr_clean < 0.01


def test_tau_at_far_exact_count():
    scores = np.arange(200) / 200.0
    tau = tau_at_far(scores, 0.01)  # 2 of 200 may pass
    assert int(np.sum(scores >= tau)) == 2


def test_b5_too_few_impostors_falls_back():
    impostor = np.linspace(0.0, 0.3, 20)
    res = calibrate(np.array([0.6, 0.7]), impostor, far=0.01)
    assert res.warning and "Falling back" in res.warning
    assert res.far_effective == pytest.approx(0.05)
    assert res.far_achieved == 0.0  # tau above the max impostor
    assert res.tau > impostor.max()


def test_calibrate_requires_impostors_unless_manual():
    with pytest.raises(ValueError):
        calibrate(np.array([0.7]), np.array([]))
    res = calibrate(np.array([0.7, 0.2]), np.array([]), manual_tau=0.4)
    assert res.tau == 0.4 and res.manual and res.frr_clean == 0.5
