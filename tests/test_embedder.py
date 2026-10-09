"""T1-T3 plus loader regressions (B1 EXIF, B2 BGR) and safe_embed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from frbench.embedder import (
    InsightFaceEmbedder,
    l2_normalize,
    load_image_bgr,
    safe_embed,
    select_largest_face,
)


def face(x1, y1, x2, y2, emb):
    return SimpleNamespace(bbox=np.array([x1, y1, x2, y2], float), embedding=np.asarray(emb, float))


class FakeApp:
    def __init__(self, faces_per_call):
        self.faces_per_call = list(faces_per_call)
        self.shapes = []

    def get(self, img):
        self.shapes.append(img.shape)
        return self.faces_per_call.pop(0) if self.faces_per_call else []


def embedder_with(app) -> InsightFaceEmbedder:
    e = InsightFaceEmbedder()
    e._app = app
    return e


# --- unit: no model needed -------------------------------------------------- #


def test_select_largest_face_picks_biggest_box():
    small, big = face(0, 0, 10, 10, [1]), face(0, 0, 50, 40, [2])
    assert select_largest_face([small, big]) is big
    assert select_largest_face([]) is None


def test_t1_embed_returns_unit_512_vector(tmp_path):
    vec = np.arange(1, 513, dtype=float)
    e = embedder_with(FakeApp([[face(0, 0, 10, 10, vec)]]))
    path = tmp_path / "x.png"
    Image.new("RGB", (32, 32)).save(path)
    out = e.embed(path)
    assert out.shape == (512,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_t2_no_face_returns_none_after_pad_retry(tmp_path):
    app = FakeApp([[], []])
    e = embedder_with(app)
    path = tmp_path / "blank.png"
    Image.new("RGB", (40, 20)).save(path)
    assert e.embed(path) is None
    assert len(app.shapes) == 2 and app.shapes[1][0] > app.shapes[0][0]  # retried padded


def test_t3_two_faces_uses_largest(tmp_path):
    small = face(0, 0, 10, 10, np.r_[1.0, np.zeros(511)])
    big = face(0, 0, 80, 80, np.r_[0.0, 1.0, np.zeros(510)])
    e = embedder_with(FakeApp([[small, big]]))
    path = tmp_path / "two.png"
    Image.new("RGB", (100, 100)).save(path)
    assert np.argmax(e.embed(path)) == 1


def test_b1_exif_rotation_is_applied(tmp_path):
    img = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    path = tmp_path / "rot.jpg"
    img.save(path, exif=exif)
    assert load_image_bgr(path).shape[:2] == (40, 20)


def test_b2_loader_returns_bgr(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(path)
    arr = load_image_bgr(path)
    assert tuple(arr[0, 0]) == (0, 0, 255)
    assert np.array_equal(arr, cv2.imread(str(path)))


def test_safe_embed_unreadable_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    out = safe_embed(InsightFaceEmbedder(), bad)
    assert out.embedding is None and out.error.startswith("unreadable")


def test_l2_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(3))


# --- real model ------------------------------------------------------------- #


def sample(name: str) -> np.ndarray:
    from insightface.data import get_image

    return get_image(name)


@pytest.mark.model
def test_t1_real_model_on_face(real_embedder):
    vec = real_embedder.embed_image(sample("t1"))
    assert vec.shape == (512,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.model
def test_t2_real_model_no_face(real_embedder):
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert real_embedder.embed_image(noise) is None
    assert real_embedder.embed_image(np.zeros((480, 640, 3), np.uint8)) is None


@pytest.mark.model
def test_t3_real_model_group_photo_uses_largest(real_embedder):
    img = sample("t1")
    faces = real_embedder.detect(img)
    assert len(faces) >= 2
    largest = select_largest_face(faces)
    assert np.allclose(real_embedder.embed_image(img), l2_normalize(largest.embedding))


@pytest.mark.model
def test_pad_retry_detects_tight_crop(real_embedder):
    """A tightly cropped face (fills the frame) is found thanks to padding."""
    img = sample("t1")
    x1, y1, x2, y2 = select_largest_face(real_embedder.detect(img)).bbox.astype(int)
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    assert real_embedder.embed_image(crop) is not None
