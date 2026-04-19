"""Mahalanobis distance for embedding-space drift detection."""

from __future__ import annotations

import numpy as np


class MahalanobisMonitor:
    """Fit a Gaussian on reference embeddings; score new batches by distance.

    The Mahalanobis distance accounts for feature correlations, unlike
    Euclidean distance. High mean Mahalanobis in a deployment batch indicates
    the embeddings have shifted away from the training distribution.
    """

    def __init__(self, regularization: float = 1e-4) -> None:
        self.regularization = regularization
        self._mu: np.ndarray | None = None
        self._precision: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "MahalanobisMonitor":
        Z = np.asarray(embeddings, dtype=np.float64)
        self._mu = Z.mean(axis=0)
        cov = np.cov(Z.T) + np.eye(Z.shape[1]) * self.regularization
        self._precision = np.linalg.inv(cov)
        return self

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Return per-sample Mahalanobis distances."""
        if self._mu is None:
            raise RuntimeError("Call fit() first")
        Z = np.asarray(embeddings, dtype=np.float64)
        diff = Z - self._mu
        return np.sqrt(np.maximum((diff @ self._precision * diff).sum(axis=1), 0))

    def score(self, embeddings: np.ndarray) -> dict:
        d = self.distances(embeddings)
        return {
            "mean": float(d.mean()),
            "max": float(d.max()),
            "p95": float(np.percentile(d, 95)),
            "n": len(d),
        }
