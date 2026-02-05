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
| `PiecewiseLinearCalibrator` | ✓ | ✓ | Monotonic piecewise linear interpolation |
| `MonotonicNNCalibrator` | ✓ | ✓ | Neural network with monotonicity constraints |

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

Run GPU tests with:
```bash
pytest tests/test_gpu.py --device cuda  # or --device mps on Mac
```

## Metrics

- `ece(scores, labels)` - Expected Calibration Error
- `ece_at_k(scores, labels, k)` - ECE computed only on top-k items
- `reliability_diagram(scores, labels, k=None)` - Visualization of calibration

## Citation

If you use rankcal in academic work, please cite:

```bibtex
@software{hodge2025rankcal,
  author = {Hodge, John},
  title = {rankcal: Calibration for Ranking Systems},
  year = {2025},
  url = {https://github.com/jman4162/rankcal},
  version = {0.2.0}
}
```

## License

MIT
