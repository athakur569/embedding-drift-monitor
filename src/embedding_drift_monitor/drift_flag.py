"""Combined drift flag from Mahalanobis + energy score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mahalanobis import MahalanobisMonitor
from .energy_score import energy_score, energy_threshold_from_reference


@dataclass
class DriftResult:
    mahalanobis_mean: float
    mahalanobis_p95: float
    energy_mean: float
    flagged_maha: bool
    flagged_energy: bool
    flagged: bool  # OR of both signals

    def __str__(self) -> str:
        status = "DRIFTED" if self.flagged else "OK"
        return (
            f"[{status}] maha={self.mahalanobis_mean:.3f} "
            f"energy={self.energy_mean:.3f}"
        )


class EmbeddingMonitor:
    """Combined Mahalanobis + energy monitor.

    Args:
        maha_threshold:    Mean Mahalanobis above this = flagged.
        energy_threshold:  Mean energy below this = flagged (more OOD).
                           If None, estimated from reference logits.
    """

    def __init__(
        self,
        maha_threshold: float = 3.0,
        energy_threshold: Optional[float] = None,
    ) -> None:
        self.maha_threshold = maha_threshold
        self.energy_threshold = energy_threshold
        self._maha = MahalanobisMonitor()
        self._fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        ref_logits: Optional[np.ndarray] = None,
    ) -> "EmbeddingMonitor":
        self._maha.fit(embeddings)
        if self.energy_threshold is None and ref_logits is not None:
            self.energy_threshold = energy_threshold_from_reference(ref_logits)
        self._fitted = True
        return self

    def score(
        self,
        embeddings: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> DriftResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")

        maha_stats = self._maha.score(embeddings)
        flagged_maha = maha_stats["mean"] > self.maha_threshold

        energy_mean = 0.0
        flagged_energy = False
        if logits is not None and self.energy_threshold is not None:
            energies = energy_score(logits)
            energy_mean = float(energies.mean())
            flagged_energy = energy_mean < self.energy_threshold

        return DriftResult(
            mahalanobis_mean=maha_stats["mean"],
            mahalanobis_p95=maha_stats["p95"],
            energy_mean=energy_mean,
            flagged_maha=flagged_maha,
            flagged_energy=flagged_energy,
            flagged=flagged_maha or flagged_energy,
        )
