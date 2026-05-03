"""Energy score for out-of-distribution detection (Liu et al., 2020)."""

from __future__ import annotations

import numpy as np


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute energy score per sample from classifier logits.

    E(x; f) = -T * log Σ_y exp(f_y(x) / T)

    More negative energy = more out-of-distribution.
    In-distribution samples cluster around a higher (less negative) energy.

    Args:
        logits:      (n, n_classes) logit array.
        temperature: Scaling temperature T (default 1.0).

    Returns:
        (n,) float array of energy scores.

    Reference:
        Liu, W. et al. "Energy-based out-of-distribution detection." NeurIPS 2020.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[:, None]
    log_sum_exp = temperature * np.log(
        np.exp(logits / temperature).sum(axis=1)
    )
    return -log_sum_exp


def energy_threshold_from_reference(
    ref_logits: np.ndarray,
    percentile: float = 5.0,
    temperature: float = 1.0,
) -> float:
    """Estimate a threshold below which a sample is flagged as OOD.

    Sets the threshold at the p-th percentile of reference energy scores.
    Any deployment score below this threshold is considered OOD.
    """
    ref_energies = energy_score(ref_logits, temperature)
    return float(np.percentile(ref_energies, percentile))
