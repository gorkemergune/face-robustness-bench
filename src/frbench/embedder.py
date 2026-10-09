"""Face detection, alignment and embedding.

Pipeline: load image (EXIF-corrected, BGR) -> detect faces -> pick largest
-> 5-point alignment (done inside InsightFace) -> ArcFace 512-d embedding
-> L2 normalize.

The model sits behind the ``Embedder`` protocol so other models (or test
fakes) can be swapped in.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# insightface's alignment calls a scikit-image API that warns on every face.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.")

EMBEDDING_DIM = 512

try:  # Optional HEIC/HEIF support for iPhone photos.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pass


class ImageLoadError(RuntimeError):
    """Raised when an image file cannot be decoded."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns an image file into an L2-normalized embedding."""

    name: str

    def embed(self, path: str | Path) -> np.ndarray | None:
        """Return an L2-normalized vector, or None if no face is detected."""
        ...


@dataclass(frozen=True)
class EmbedOutcome:
    """Result of ``safe_embed``: never raises, always explains failures."""

    path: Path
    embedding: np.ndarray | None
    error: str | None = None

    @property
    def detected(self) -> bool:
        return self.embedding is not None


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (float32)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("Cannot normalize a zero vector.")
    return vec / norm


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image as an OpenCV-style BGR uint8 array.

    EXIF orientation is applied first (phone photos are often stored rotated)
    and the result is converted RGB -> BGR, which is what InsightFace expects.
    """
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            rgb = np.asarray(img.convert("RGB"))
    except Exception as exc:  # PIL raises many different types
        raise ImageLoadError(f"Cannot read image {path}: {exc}") from exc
    return np.ascontiguousarray(rgb[:, :, ::-1])


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_largest_face(faces: Sequence[Any]) -> Any | None:
    """Pick the face with the largest bounding box (ties: first one)."""
    if not faces:
        return None
    return max(faces, key=lambda f: bbox_area(f.bbox))


def pad_image(img: np.ndarray, ratio: float = 0.3) -> np.ndarray:
    """Add a black border so tightly-cropped faces become detectable."""
    h, w = img.shape[:2]
    py, px = int(h * ratio), int(w * ratio)
    return np.pad(img, ((py, py), (px, px), (0, 0)), mode="constant")


class InsightFaceEmbedder:
    """InsightFace ``buffalo_l`` (SCRFD detector + ArcFace R100) on CPU."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        pad_retry: bool = True,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.name = f"insightface:{model_name}"
        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.pad_retry = pad_retry
        self.providers = list(providers)
        self._app: Any | None = None

    @property
    def app(self) -> Any:
        if self._app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=self.model_name,
                providers=self.providers,
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=-1, det_thresh=self.det_thresh, det_size=self.det_size)
            self._app = app
        return self._app

    def detect(self, img_bgr: np.ndarray) -> list[Any]:
        faces = list(self.app.get(img_bgr))
        if not faces and self.pad_retry:
            faces = list(self.app.get(pad_image(img_bgr)))
        return faces

    def embed_image(self, img_bgr: np.ndarray) -> np.ndarray | None:
        face = select_largest_face(self.detect(img_bgr))
        if face is None:
            return None
        return l2_normalize(face.embedding)

    def embed(self, path: str | Path) -> np.ndarray | None:
        return self.embed_image(load_image_bgr(path))

    def embed_all(self, path: str | Path) -> list[tuple[float, np.ndarray]]:
        """(bbox area, L2-normalized embedding) for every detected face."""
        faces = self.detect(load_image_bgr(path))
        return [(bbox_area(f.bbox), l2_normalize(f.embedding)) for f in faces]


def safe_embed(embedder: Embedder, path: str | Path) -> EmbedOutcome:
    """Embed ``path`` without raising.

    Detection failures and unreadable files are returned as an outcome with
    ``embedding=None`` and an error message, so callers can count them
    instead of silently dropping rows.
    """
    path = Path(path)
    try:
        emb = embedder.embed(path)
    except ImageLoadError as exc:
        log.warning("%s", exc)
        return EmbedOutcome(path, None, f"unreadable: {exc}")
    except Exception as exc:  # model/runtime errors must not stop the run
        log.exception("Embedding failed for %s", path)
        return EmbedOutcome(path, None, f"error: {type(exc).__name__}: {exc}")
    if emb is None:
        return EmbedOutcome(path, None, "no_face")
    emb = np.asarray(emb, dtype=np.float32).ravel()
    return EmbedOutcome(path, l2_normalize(emb))
