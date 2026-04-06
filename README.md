# embedding-drift-monitor

Detect model-response drift from frozen encoder embeddings: Mahalanobis distance, energy score OOD detection, and drift flagging for ML monitoring pipelines.

## Use case
After deployment, periodically extract embeddings from new data batches using a frozen encoder. Compare against the reference distribution to detect silent model degradation before it becomes visible in downstream outputs.

## Metrics
- **Mahalanobis distance** — how far new embeddings are from the training distribution (in ellipsoidal distance)
- **Energy score** — −T log Σexp(logits/T); more negative = more out-of-distribution
- **Drift flag** — raised when mean Mahalanobis or energy exceeds learned thresholds

## Installation
```bash
pip install embedding-drift-monitor
```

## Quick start
```python
from embedding_drift_monitor import EmbeddingMonitor

monitor = EmbeddingMonitor()
monitor.fit(reference_embeddings)          # (n, d) numpy array

result = monitor.score(deployment_embeddings)
print(result.mahalanobis_mean, result.flagged)
```

## License
MIT
