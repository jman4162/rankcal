# rankcal

Calibration and uncertainty quantification for ranking systems. PyTorch-first.

## Why rankcal?

Existing calibration libraries treat calibration as a classification problem. But ranking decisions happen at the **top-k**, and miscalibration there is what actually breaks business outcomes.

rankcal provides:
- **Ranking-aware calibration metrics** - ECE@k, top-k reliability diagrams
- **Monotonic calibrators** - Temperature scaling, isotonic regression, splines, neural networks
- **Decision analysis** - Risk-coverage curves, utility optimization

## Installation

```bash
pip install rankcal
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from rankcal import TemperatureScaling, ece_at_k, reliability_diagram

# Your ranking scores and binary relevance labels
scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
relevance = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0, 0])

# Fit a calibrator
calibrator = TemperatureScaling()
calibrator.fit(scores, relevance)

# Calibrate scores
calibrated = calibrator(scores)

# Evaluate calibration at top-k
ece = ece_at_k(calibrated, relevance, k=5)
print(f"ECE@5: {ece:.4f}")

# Visualize calibration
fig = reliability_diagram(calibrated, relevance, k=5)
fig.savefig("reliability.png")
```

## Calibrators

| Calibrator | Differentiable | Parametric | Description |
|------------|----------------|------------|-------------|
| `TemperatureScaling` | ✓ | ✓ | Single learned temperature parameter |
| `IsotonicCalibrator` | ✗ | ✗ | Non-parametric, piecewise constant |
| `SplineCalibrator` | ✓ | ✓ | Smooth monotonic spline |
| `MonotonicNNCalibrator` | ✓ | ✓ | Neural network with monotonicity constraints |

## Metrics

- `ece(scores, labels)` - Expected Calibration Error
- `ece_at_k(scores, labels, k)` - ECE computed only on top-k items
- `reliability_diagram(scores, labels, k=None)` - Visualization of calibration

## License

MIT
