"""T12/T13/T18 with the real buffalo_l model on InsightFace's bundled photos.

The bundled group photo (t1.jpg) has six different people. We build a
template for one person from photometric variants of their face and check
that a mirrored / re-lit version still matches while other people do not.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from frbench.embedder import select_largest_face
from frbench.gallery import build_centroid
from frbench.scoring import cosine

pytestmark = pytest.mark.model

TAU_BUFFALO = 0.30  # conservative literature value for w600k_r50 at low FAR


@pytest.fixture(scope="module")
def faces(real_embedder):
    from insightface.data import get_image

    img = get_image("t1")
    detected = sorted(real_embedder.detect(img), key=lambda f: f.bbox[0])
    assert len(detected) == 6
    crops = []
    for f in detected:
        x1, y1, x2, y2 = f.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        x1, y1 = max(0, x1 - w), max(0, y1 - h)
        crops.append(np.ascontiguousarray(img[y1 : y2 + h, x1 : x2 + w]))
    return crops


def variants(img):
    yield img
    yield cv2.convertScaleAbs(img, alpha=1.15, beta=10)
    yield cv2.convertScaleAbs(img, alpha=0.85, beta=-10)
    yield cv2.GaussianBlur(img, (3, 3), 0)


@pytest.fixture(scope="module")
def template(real_embedder, faces):
    return build_centroid([real_embedder.embed_image(v) for v in variants(faces[0])])


def test_t18_mirrored_photo_high_similarity(real_embedder, faces, template):
    mirrored = real_embedder.embed_image(np.ascontiguousarray(faces[0][:, ::-1]))
    assert cosine(mirrored, template) > 0.6


def test_t12_same_person_recognized(real_embedder, faces, template):
    scaled = cv2.resize(faces[0], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    assert cosine(real_embedder.embed_image(scaled), template) >= TAU_BUFFALO


def test_t13_different_people_not_recognized(real_embedder, faces, template):
    for other in faces[1:]:
        assert cosine(real_embedder.embed_image(other), template) < TAU_BUFFALO


def test_crop_keeps_single_largest_face(real_embedder, faces):
    assert select_largest_face(real_embedder.detect(faces[0])) is not None
