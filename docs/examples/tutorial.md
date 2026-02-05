# Tutorial: Calibrating Ranking Scores

This tutorial walks through a complete example of calibrating ranking scores, from generating data to evaluating results.

## Setup

```python
import torch
from rankcal import (
    TemperatureScaling,
    IsotonicCalibrator,
    PiecewiseLinearCalibrator,
    ece,
    ece_at_k,
    reliability_diagram,
    generate_miscalibrated_data,
)
```

## Generate Sample Data

First, let's generate some miscalibrated data to work with. In practice, this would be your model's predictions.

```python
# Generate miscalibrated data (overconfident predictions)
scores_train, labels_train = generate_miscalibrated_data(
    n_samples=1000, temperature=2.0, seed=42
)
scores_test, labels_test = generate_miscalibrated_data(
    n_samples=500, temperature=2.0, seed=123
)

print(f"Train samples: {len(scores_train)}")
print(f"Test samples: {len(scores_test)}")
print(f"Uncalibrated ECE: {ece(scores_test, labels_test):.4f}")
```

Output:
```
Train samples: 1000
Test samples: 500
Uncalibrated ECE: 0.1523
```

## Compare Calibrators

### Temperature Scaling

The simplest calibrator with just one parameter:

```python
temp_cal = TemperatureScaling()
temp_cal.fit(scores_train, labels_train)
temp_calibrated = temp_cal(scores_test)

print(f"Learned temperature: {temp_cal.temperature.item():.4f}")
print(f"Calibrated ECE: {ece(temp_calibrated, labels_test):.4f}")
```

Output:
```
Learned temperature: 1.8234
Calibrated ECE: 0.0312
```

### Isotonic Calibration

Non-parametric and robust:

```python
iso_cal = IsotonicCalibrator()
iso_cal.fit(scores_train, labels_train)
iso_calibrated = iso_cal(scores_test)

print(f"Calibrated ECE: {ece(iso_calibrated, labels_test):.4f}")
```

Output:
```
Calibrated ECE: 0.0198
```

### Piecewise Linear Calibration

Differentiable with tunable complexity:

```python
spline_cal = PiecewiseLinearCalibrator(n_knots=10)
spline_cal.fit(scores_train, labels_train, max_iter=200)
spline_calibrated = spline_cal(scores_test)

print(f"Calibrated ECE: {ece(spline_calibrated, labels_test):.4f}")
```

Output:
```
Calibrated ECE: 0.0245
```

## Evaluate at Top-k

For ranking systems, we often care most about calibration at the top:

```python
k = 50

print(f"ECE@{k} (uncalibrated): {ece_at_k(scores_test, labels_test, k=k):.4f}")
print(f"ECE@{k} (temperature):  {ece_at_k(temp_calibrated, labels_test, k=k):.4f}")
print(f"ECE@{k} (isotonic):     {ece_at_k(iso_calibrated, labels_test, k=k):.4f}")
print(f"ECE@{k} (piecewise):    {ece_at_k(spline_calibrated, labels_test, k=k):.4f}")
```

## Visualize with Reliability Diagrams

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

reliability_diagram(scores_test, labels_test, ax=axes[0])
axes[0].set_title("Uncalibrated")

reliability_diagram(temp_calibrated, labels_test, ax=axes[1])
axes[1].set_title("Temperature Scaling")

reliability_diagram(iso_calibrated, labels_test, ax=axes[2])
axes[2].set_title("Isotonic")

reliability_diagram(spline_calibrated, labels_test, ax=axes[3])
axes[3].set_title("Piecewise Linear")

plt.tight_layout()
plt.savefig("calibration_comparison.png")
plt.show()
```

## Summary

| Method | ECE | ECE@50 |
|--------|-----|--------|
| Uncalibrated | 0.1523 | 0.1812 |
| Temperature Scaling | 0.0312 | 0.0401 |
| Isotonic | 0.0198 | 0.0287 |
| Piecewise Linear | 0.0245 | 0.0334 |

## Key Takeaways

1. **Start with IsotonicCalibrator** - It's robust, has no hyperparameters, and works well for most cases

2. **Use TemperatureScaling for simplicity** - When you need differentiability or want a single interpretable parameter

3. **Evaluate at top-k** - ECE@k matters more than overall ECE for ranking applications

4. **Always use held-out data** - Never calibrate on training data

## Next Steps

- [Decision Making](../guide/decisions.md) - Use calibrated scores for threshold optimization
- [API Reference](../api/calibrators.md) - Full calibrator documentation
- [Evaluation Guide](../guide/evaluation.md) - More metrics and visualization options
