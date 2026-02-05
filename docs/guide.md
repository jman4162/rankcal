# rankcal Conceptual Guide

This guide helps you choose the right calibration approach for your ranking system and understand the key concepts behind calibration.

## Which Calibrator Should I Use?

Use this decision tree to select the appropriate calibrator:

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

### Quick Reference

| Calibrator | Differentiable | Parameters | Best For |
|------------|----------------|------------|----------|
| `IsotonicCalibrator` | No | 0 | Post-hoc calibration, production |
| `TemperatureScaling` | Yes | 1 | Simple overconfidence/underconfidence |
| `PiecewiseLinearCalibrator` | Yes | 10-20 | Moderate miscalibration |
| `MonotonicNNCalibrator` | Yes | ~100-500 | Complex patterns, end-to-end training |

## Hyperparameter Guidance

### TemperatureScaling

```python
from rankcal import TemperatureScaling

calibrator = TemperatureScaling(init_temperature=1.0)
```

- `init_temperature=1.0`: Starting value for optimization
  - Values > 1 soften predictions (fix overconfidence)
  - Values < 1 sharpen predictions (fix underconfidence)
  - The optimizer will find the best value; default is usually fine

### PiecewiseLinearCalibrator

```python
from rankcal import PiecewiseLinearCalibrator

calibrator = PiecewiseLinearCalibrator(n_knots=10)
```

- `n_knots=10`: Number of knot points (default)
  - Increase to 15-20 for complex miscalibration patterns
  - Decrease to 5-7 if you have limited calibration data (<500 samples)
  - More knots = more flexibility but higher risk of overfitting

### MonotonicNNCalibrator

```python
from rankcal import MonotonicNNCalibrator

calibrator = MonotonicNNCalibrator(hidden_dims=(16, 16))
```

- `hidden_dims=(16, 16)`: Network architecture (default)
  - Use `(8,)` for simpler patterns or small datasets
  - Use `(32, 32)` for very complex patterns with large datasets
  - More layers/units = more flexibility but needs more data

## Train/Calibration/Test Splits

**Critical Rule**: Never calibrate on training data. Calibration must happen on held-out data.

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

## When Calibration Matters

Calibration is essential when you need to **interpret scores as probabilities** or **make decisions based on score thresholds**.

### 1. Decision-Making with Thresholds

If you're selecting items above a threshold, miscalibration leads to wrong decisions:

```python
from rankcal import optimal_threshold

# Find threshold that maximizes utility
# benefit = value of showing a relevant item
# cost = cost of showing an irrelevant item
threshold, utility = optimal_threshold(
    calibrated_scores, labels, benefit=1.0, cost=0.5
)

# Select items above threshold
selected = scores >= threshold
```

### 2. Ranking with Budget Constraints

When you can only show k items, calibration helps predict expected outcomes:

```python
from rankcal import expected_precision_at_k, precision_at_k

# With calibrated scores, expected precision ≈ actual precision
expected = expected_precision_at_k(calibrated_scores, k=10)
actual = precision_at_k(calibrated_scores, labels, k=10)

# The gap between these indicates calibration quality
```

### 3. Combining Scores from Multiple Models

When merging rankings from different models, scores must be calibrated to be comparable:

```python
# Without calibration, Model A's 0.7 might mean something
# different than Model B's 0.7

# Calibrate both to the same scale
scores_a_cal = calibrator_a(scores_a)
scores_b_cal = calibrator_b(scores_b)

# Now they can be meaningfully combined
combined = 0.5 * scores_a_cal + 0.5 * scores_b_cal
```

## Integration with Ranking Pipelines

### Post-hoc Calibration (Recommended for Most Cases)

Use this when your ranking model is already trained:

```python
from rankcal import IsotonicCalibrator, ece, ece_at_k

# 1. Get model predictions on calibration set
raw_scores = model.predict(X_cal)

# 2. Fit calibrator
calibrator = IsotonicCalibrator()
calibrator.fit(raw_scores, cal_labels)

# 3. Use in production
def predict_with_calibration(X):
    raw = model.predict(X)
    return calibrator(torch.tensor(raw))

# 4. Monitor calibration
print(f"Overall ECE: {ece(calibrated_scores, labels):.4f}")
print(f"ECE@10: {ece_at_k(calibrated_scores, labels, k=10):.4f}")
```

### End-to-End Differentiable Training

Use `MonotonicNNCalibrator` when you want to train calibration jointly with your model:

```python
import torch.nn as nn
from rankcal import MonotonicNNCalibrator

class CalibratedRanker(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.calibrator = MonotonicNNCalibrator(hidden_dims=(16, 16))

    def forward(self, x):
        raw_scores = self.base_model(x)
        return self.calibrator(raw_scores)

# Train with combined loss
model = CalibratedRanker(base_model)
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    x, labels = batch
    calibrated_scores = model(x)

    # Your ranking loss + calibration regularization
    loss = ranking_loss(calibrated_scores, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Evaluating Calibration

### Core Metrics

```python
from rankcal import ece, ece_at_k, adaptive_ece, mce

# Expected Calibration Error (lower is better)
print(f"ECE: {ece(scores, labels):.4f}")

# ECE focused on top-k (where decisions happen)
print(f"ECE@10: {ece_at_k(scores, labels, k=10):.4f}")

# Adaptive ECE (better for skewed score distributions)
print(f"Adaptive ECE: {adaptive_ece(scores, labels):.4f}")

# Maximum Calibration Error (worst-case bin)
print(f"MCE: {mce(scores, labels):.4f}")
```

### Visualizing Calibration

```python
from rankcal import reliability_diagram
import matplotlib.pyplot as plt

# Reliability diagram shows calibration visually
fig = reliability_diagram(scores, labels, n_bins=10)
plt.show()

# For top-k focused view
fig = reliability_diagram(scores, labels, k=50, n_bins=10)
plt.show()
```

### Interpreting Results

| ECE Value | Interpretation |
|-----------|----------------|
| < 0.02 | Excellent calibration |
| 0.02 - 0.05 | Good calibration |
| 0.05 - 0.10 | Moderate miscalibration |
| > 0.10 | Poor calibration, consider recalibrating |

Note: ECE@k at small k may be higher due to fewer samples. Focus on trends rather than absolute values for small k.
