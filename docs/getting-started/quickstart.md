# Quick Start

This guide shows you how to calibrate ranking scores in just a few lines of code.

## Basic Example

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

## What Just Happened?

1. **Input data**: We have ranking scores (model confidence) and binary relevance labels (ground truth)
2. **Fit calibrator**: The `TemperatureScaling` calibrator learns to adjust scores so they reflect true probabilities
3. **Transform scores**: Calling `calibrator(scores)` applies the learned transformation
4. **Evaluate**: `ece_at_k` measures how well calibrated the top-k scores are
5. **Visualize**: The reliability diagram shows calibration quality graphically

## Choosing a Calibrator

Different calibrators have different tradeoffs:

```python
from rankcal import (
    TemperatureScaling,      # Simple, 1 parameter
    IsotonicCalibrator,      # Non-parametric, robust
    PiecewiseLinearCalibrator,  # Flexible, differentiable
    MonotonicNNCalibrator,   # Most flexible, for complex patterns
)

# For most cases, start with IsotonicCalibrator
calibrator = IsotonicCalibrator()
calibrator.fit(scores, labels)
calibrated = calibrator(scores)
```

See the [User Guide](../guide/choosing-calibrator.md) for detailed guidance on choosing a calibrator.

## Next Steps

- [Choosing a Calibrator](../guide/choosing-calibrator.md) - Decision tree for selecting the right calibrator
- [Hyperparameters](../guide/hyperparameters.md) - Tuning calibrator parameters
- [Evaluation](../guide/evaluation.md) - Understanding calibration metrics
- [API Reference](../api/calibrators.md) - Full API documentation
