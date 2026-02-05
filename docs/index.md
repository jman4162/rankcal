# rankcal

Calibration and uncertainty quantification for ranking systems. PyTorch-first.

## Why rankcal?

Existing calibration libraries treat calibration as a classification problem. But ranking decisions happen at the **top-k**, and miscalibration there is what actually breaks business outcomes.

rankcal provides:

- **Ranking-aware calibration metrics** - ECE@k, top-k reliability diagrams
- **Monotonic calibrators** - Temperature scaling, isotonic regression, splines, neural networks
- **Decision analysis** - Risk-coverage curves, utility optimization

## Calibrators

| Calibrator | Differentiable | Parametric | Description |
|------------|----------------|------------|-------------|
| `TemperatureScaling` | Yes | Yes | Single learned temperature parameter |
| `IsotonicCalibrator` | No | No | Non-parametric, piecewise constant |
| `PiecewiseLinearCalibrator` | Yes | Yes | Monotonic piecewise linear interpolation |
| `MonotonicNNCalibrator` | Yes | Yes | Neural network with monotonicity constraints |

## GPU Support

All calibrators are PyTorch `nn.Module` subclasses and support GPU acceleration:

```python
import torch
from rankcal import TemperatureScaling

# Move calibrator to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calibrator = TemperatureScaling().to(device)

# Fit with data on GPU
scores = scores.to(device)
labels = labels.to(device)
calibrator.fit(scores, labels)

# Inference on GPU
test_scores = test_scores.to(device)
calibrated = calibrator(test_scores)
```

## Metrics

- `ece(scores, labels)` - Expected Calibration Error
- `ece_at_k(scores, labels, k)` - ECE computed only on top-k items
- `reliability_diagram(scores, labels, k=None)` - Visualization of calibration

## License

MIT
