"""Clean-photo handling: listing, de-duplication, split and centroid template."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import IMAGE_EXTENSIONS
from .embedder import l2_normalize

log = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_GALLERY_FRACTION = 0.6
MIN_CLEAN_PHOTOS = 10


class InsufficientDataError(ValueError):
    """Raised when a person does not have enough usable clean photos."""


@dataclass(frozen=True)
class Split:
    gallery: tuple[Path, ...]
    heldout: tuple[Path, ...]


def list_images(directory: Path) -> list[Path]:
    """Image files directly inside ``directory``, sorted for determinism."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def _cached_hash(path: str, size: int, mtime_ns: int) -> str:
    return file_hash(Path(path))


def cached_file_hash(path: Path) -> str:
    """``file_hash`` memoized on (path, size, mtime) so each file is read once."""
    st = path.stat()
    return _cached_hash(str(path), st.st_size, st.st_mtime_ns)


def dedupe_by_hash(paths: Iterable[Path]) -> list[Path]:
    """Drop byte-identical duplicates, keeping the first (sorted) occurrence."""
    seen: dict[str, Path] = {}
    for p in sorted(paths):
        digest = cached_file_hash(p)
        if digest in seen:
            log.warning("Duplicate photo %s is identical to %s; ignoring it.", p, seen[digest])
            continue
        seen[digest] = p
    return list(seen.values())


def split_clean(
    paths: Sequence[Path],
    gallery_fraction: float = DEFAULT_GALLERY_FRACTION,
    seed: int = DEFAULT_SEED,
    min_photos: int = MIN_CLEAN_PHOTOS,
) -> Split:
    """Deterministic gallery / held-out split of one person's clean photos."""
    if not 0 < gallery_fraction < 1:
        raise ValueError("gallery_fraction must be in (0, 1).")
    unique = dedupe_by_hash(paths)
    if len(unique) < min_photos:
        raise InsufficientDataError(
            f"Need at least {min_photos} distinct clean photos, found {len(unique)}."
        )
    shuffled = sorted(unique)
    random.Random(seed).shuffle(shuffled)
    n_gallery = min(max(1, round(len(shuffled) * gallery_fraction)), len(shuffled) - 1)
    return Split(tuple(shuffled[:n_gallery]), tuple(shuffled[n_gallery:]))


def build_centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalized embeddings, re-normalized to unit length."""
    if len(embeddings) == 0:
        raise InsufficientDataError("Cannot build a template from zero embeddings.")
    stacked = np.stack([l2_normalize(e) for e in embeddings])
    return l2_normalize(stacked.mean(axis=0))
