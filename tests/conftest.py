"""Shared fixtures: a fake embedder driven by pixel-encoded metadata and a
tiny synthetic dataset, so the pipeline can be tested without the model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from frbench.embedder import EMBEDDING_DIM, load_image_bgr

MODEL_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"
NO_FACE = 255


def write_coded_image(path: Path, identity: int, seed: int, noise: float = 0.3, face: bool = True) -> Path:
    """8x8 PNG whose first pixel encodes (identity, seed, noise or NO_FACE)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[0, 0] = (identity, seed % 256, NO_FACE if not face else int(round(noise * 100)))
    arr[0, 1] = (seed // 256, 0, 0)
    Image.fromarray(arr, "RGB").save(path)
    return path


class FakeEmbedder:
    """Decodes the pixel metadata into identity base vector + seeded noise."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, path):
        self.calls += 1
        bgr = load_image_bgr(path)
        b, g, r = (int(v) for v in bgr[0, 0])
        if b == NO_FACE:
            return None
        identity, seed, noise = r, g + 256 * int(bgr[0, 1, 2]), b / 100.0
        base = np.random.default_rng(10_000 + identity).normal(size=EMBEDDING_DIM)
        jitter = np.random.default_rng(seed).normal(size=EMBEDDING_DIM)
        vec = base + noise * jitter
        return (vec / np.linalg.norm(vec)).astype(np.float32)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def build_dataset(root: Path) -> Path:
    """Small dataset exercising every branch of the pipeline.

    persons: alice (20 clean), bob (25 clean), carol (12 clean), tiny (5 clean -> skipped)
    probes:  filter (mild), makeup (stronger), effect (incl. 2 no-face), botox (empty folder)
    impostors: 6 ids x 5 photos
    """
    seed = iter(range(1, 60_000))
    persons = {"alice": (1, 20), "bob": (2, 25), "carol": (3, 12), "tiny": (4, 5)}
    for name, (ident, n_clean) in persons.items():
        for i in range(n_clean):
            write_coded_image(root / "gallery" / name / f"clean_{i:02d}.png", ident, next(seed), 0.3)
        probe = root / "probes" / name
        for i in range(6):
            write_coded_image(probe / "filter" / f"f_{i}.png", ident, next(seed), 0.5)
            write_coded_image(probe / "makeup" / f"m_{i}.png", ident, next(seed), 0.9)
        for i in range(4):
            write_coded_image(probe / "effect" / f"e_{i}.png", ident, next(seed), 0.6, face=i >= 2)
        (probe / "botox").mkdir(parents=True, exist_ok=True)
    for imp in range(6):
        for i in range(5):
            write_coded_image(root / "impostors" / f"other_{imp}" / f"{i}.png", 50 + imp, next(seed), 0.3)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return build_dataset(tmp_path / "data")


@pytest.fixture(scope="session")
def real_embedder():
    if not (MODEL_DIR / "w600k_r50.onnx").exists():
        pytest.skip("buffalo_l weights not found in ~/.insightface/models")
    from frbench.embedder import InsightFaceEmbedder

    return InsightFaceEmbedder()
