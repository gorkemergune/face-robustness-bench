"""Calibrate the decision threshold tau from genuine vs impostor scores."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FAR = 0.01


@dataclass(frozen=True)
class ThresholdResult:
    tau: float
    far_requested: float
    far_effective: float  # coarsest FAR the impostor count can resolve
    far_achieved: float  # fraction of impostor scores >= tau
    frr_clean: float  # fraction of genuine scores < tau
    n_genuine: int
    n_impostor: int
    genuine_mean: float
    impostor_mean: float
    manual: bool = False
    warning: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def tau_at_far(impostor: np.ndarray, far: float) -> float:
    """Smallest threshold such that at most ``floor(far * n)`` impostors pass.

    Accept rule is ``score >= tau``. The returned tau sits just above the
    (k+1)-th highest impostor score, so exactly k impostors pass (fewer on ties).
    """
    scores = np.sort(np.asarray(impostor, dtype=np.float64))[::-1]
    k = int(math.floor(far * len(scores) + 1e-9))
    k = min(k, len(scores) - 1)
    return float(np.nextafter(scores[k], np.inf))


def calibrate(
    genuine: np.ndarray,
    impostor: np.ndarray,
    far: float = DEFAULT_FAR,
    manual_tau: float | None = None,
) -> ThresholdResult:
    """Pick tau at ``far``; warn and fall back to a coarser FAR if needed."""
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if not 0 < far < 1:
        raise ValueError("far must be in (0, 1).")
    if genuine.size == 0:
        raise ValueError("No genuine scores: cannot calibrate a threshold.")
    if impostor.size == 0 and manual_tau is None:
        raise ValueError(
            "No impostor scores: add photos under data/impostors/ or more persons, "
            "or pass a fixed --tau."
        )

    warning = None
    far_effective = far
    if impostor.size and impostor.size * far < 1:
        far_effective = 1.0 / impostor.size
        warning = (
            f"Only {impostor.size} impostor scores; FAR={far:.2%} needs at least "
            f"{math.ceil(1 / far)}. Falling back to FAR={far_effective:.2%} "
            "(tau above the highest impostor score)."
        )
        log.warning(warning)

    if manual_tau is not None:
        tau = float(manual_tau)
    else:
        # With too few impostors, floor(far * n) == 0, so this puts tau just
        # above the highest impostor score; far_effective records the resolution.
        tau = tau_at_far(impostor, far)

    return ThresholdResult(
        tau=tau,
        far_requested=far,
        far_effective=far_effective,
        far_achieved=float(np.mean(impostor >= tau)) if impostor.size else float("nan"),
        frr_clean=float(np.mean(genuine < tau)),
        n_genuine=int(genuine.size),
        n_impostor=int(impostor.size),
        genuine_mean=float(genuine.mean()),
        impostor_mean=float(impostor.mean()) if impostor.size else float("nan"),
        manual=manual_tau is not None,
        warning=warning,
    )
