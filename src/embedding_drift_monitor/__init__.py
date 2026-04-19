"""embedding-drift-monitor: Mahalanobis and energy-score drift detection."""

from .mahalanobis import MahalanobisMonitor
from .energy_score import energy_score
from .drift_flag import EmbeddingMonitor, DriftResult

__version__ = "0.1.0"
__all__ = ["MahalanobisMonitor", "energy_score", "EmbeddingMonitor", "DriftResult"]
