"""Cosine similarity and the 0-100 face score."""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (normalizes defensively)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def cosine_many(matrix: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of ``matrix`` against ``ref``."""
    m = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    if m.size == 0:
        return np.empty(0)
    r = np.asarray(ref, dtype=np.float64).ravel()
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    r = r / np.linalg.norm(r)
    return np.clip(m @ r, -1.0, 1.0)


def face_score(
    cos: float | np.ndarray, impostor_mean: float, genuine_mean: float
) -> float | np.ndarray:
    """Map cosine to 0-100: impostor_mean -> 0, genuine_mean -> 100, clipped.

    NaN inputs (detection failures) stay NaN.
    """
    span = genuine_mean - impostor_mean
    if span <= 0:
        raise ValueError(
            f"genuine_mean ({genuine_mean:.4f}) must exceed impostor_mean "
            f"({impostor_mean:.4f}); the model cannot separate identities."
        )
    score = np.clip((np.asarray(cos, dtype=np.float64) - impostor_mean) / span, 0.0, 1.0) * 100.0
    return float(score) if np.ndim(score) == 0 else score
