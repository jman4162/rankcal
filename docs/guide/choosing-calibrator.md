# Choosing a Calibrator

Use this decision tree to select the appropriate calibrator for your use case.

## Decision Tree

```
Need differentiable calibration?
├── No → IsotonicCalibrator
│        (simplest, most robust, no hyperparameters)
│
└── Yes → Continue below
          │
          ├── Simple miscalibration (just needs scaling)?
          │   └── Yes → TemperatureScaling
          │             (1 parameter, fast, handles overconfidence)
          │
          └── Complex miscalibration pattern?
              ├── Moderate → PiecewiseLinearCalibrator
              │              (10-20 parameters, interpretable)
              │
              └── Severe → MonotonicNNCalibrator
                           (most flexible, handles arbitrary patterns)
```

## Quick Reference

| Calibrator | Differentiable | Parameters | Best For |
|------------|----------------|------------|----------|
| `IsotonicCalibrator` | No | 0 | Post-hoc calibration, production |
| `TemperatureScaling` | Yes | 1 | Simple overconfidence/underconfidence |
| `PiecewiseLinearCalibrator` | Yes | 10-20 | Moderate miscalibration |
| `MonotonicNNCalibrator` | Yes | ~100-500 | Complex patterns, end-to-end training |

## When to Use Each

### IsotonicCalibrator

Best default choice for most use cases:

- Post-hoc calibration of a trained model
- Production systems where simplicity matters
- When you don't need gradients through calibration
- Limited calibration data (works well with small samples)

```python
from rankcal import IsotonicCalibrator

calibrator = IsotonicCalibrator()
calibrator.fit(scores, labels)
calibrated = calibrator(scores)
```

### TemperatureScaling

Use when miscalibration is simple and uniform:

- Model is consistently overconfident or underconfident
- You need differentiable calibration
- Fast training is important
- You want a single interpretable parameter

```python
from rankcal import TemperatureScaling

calibrator = TemperatureScaling()
calibrator.fit(scores, labels)
# Access learned temperature: calibrator.temperature
```

### PiecewiseLinearCalibrator

Use for moderate complexity:

- Miscalibration varies across the score range
- You need differentiable calibration
- You want interpretable knot points
- Balance between flexibility and simplicity

```python
from rankcal import PiecewiseLinearCalibrator

calibrator = PiecewiseLinearCalibrator(n_knots=10)
calibrator.fit(scores, labels)
```

### MonotonicNNCalibrator

Use for complex patterns or end-to-end training:

- Severe, non-linear miscalibration
- Training calibration jointly with your model
- Large calibration datasets available
- Maximum flexibility needed

```python
from rankcal import MonotonicNNCalibrator

calibrator = MonotonicNNCalibrator(hidden_dims=(16, 16))
calibrator.fit(scores, labels)
```

## Train/Calibration/Test Splits

!!! warning "Critical Rule"
    Never calibrate on training data. Calibration must happen on held-out data.

### Recommended Split

```
Total Data
├── Training Set (70%)    → Train your ranking model
├── Calibration Set (15%) → Fit the calibrator
└── Test Set (15%)        → Final evaluation
```

### Example Code

```python
import torch
from sklearn.model_selection import train_test_split
from rankcal import IsotonicCalibrator, ece

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Train your model
model.fit(X_train, y_train)

# Get uncalibrated scores on calibration set
uncalibrated_scores = torch.tensor(model.predict_proba(X_cal)[:, 1])
cal_labels = torch.tensor(y_cal, dtype=torch.float32)

# Fit calibrator
calibrator = IsotonicCalibrator()
calibrator.fit(uncalibrated_scores, cal_labels)

# Evaluate on test set
test_scores = torch.tensor(model.predict_proba(X_test)[:, 1])
calibrated_scores = calibrator(test_scores)
test_labels = torch.tensor(y_test, dtype=torch.float32)

print(f"ECE before: {ece(test_scores, test_labels):.4f}")
print(f"ECE after:  {ece(calibrated_scores, test_labels):.4f}")
```

### Cross-Validation Pattern

For smaller datasets, use cross-validation to generate calibration data:

```python
from sklearn.model_selection import cross_val_predict

# Get out-of-fold predictions
oof_scores = cross_val_predict(
    model, X_train, y_train, cv=5, method='predict_proba'
)[:, 1]

# Now you can use all training data for calibration
calibrator.fit(
    torch.tensor(oof_scores),
    torch.tensor(y_train, dtype=torch.float32)
)
```
