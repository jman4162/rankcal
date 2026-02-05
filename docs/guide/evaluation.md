# Evaluating Calibration

This guide explains how to measure and interpret calibration quality.

## Core Metrics

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

## Understanding Each Metric

### ECE (Expected Calibration Error)

The most common calibration metric. It measures the average absolute difference between predicted probabilities and observed frequencies across bins.

```python
from rankcal import ece

error = ece(scores, labels, n_bins=10)
```

- Lower is better (0 = perfectly calibrated)
- Uses equal-width bins by default
- Weighted by the number of samples in each bin

### ECE@k (ECE at Top-k)

Measures calibration only for the top-k ranked items. This is crucial for ranking systems where decisions are made at the top.

```python
from rankcal import ece_at_k

# Evaluate calibration for top 10 items
error = ece_at_k(scores, labels, k=10)

# Evaluate calibration for top 100 items
error = ece_at_k(scores, labels, k=100)
```

- Use this when your application shows only top-k results
- More relevant than overall ECE for ranking systems

### Adaptive ECE

Uses equal-mass (quantile) bins instead of equal-width bins. Better for skewed score distributions.

```python
from rankcal import adaptive_ece

error = adaptive_ece(scores, labels, n_bins=10)
```

- Ensures each bin has roughly the same number of samples
- More robust when scores are not uniformly distributed

### MCE (Maximum Calibration Error)

The worst calibration error across all bins. Useful for understanding worst-case behavior.

```python
from rankcal import mce

error = mce(scores, labels, n_bins=10)
```

- Shows the maximum miscalibration in any bin
- Important when worst-case errors matter

## Interpreting Results

| ECE Value | Interpretation |
|-----------|----------------|
| < 0.02 | Excellent calibration |
| 0.02 - 0.05 | Good calibration |
| 0.05 - 0.10 | Moderate miscalibration |
| > 0.10 | Poor calibration, consider recalibrating |

!!! note
    ECE@k at small k may be higher due to fewer samples. Focus on trends rather than absolute values for small k.

## Visualizing Calibration

### Reliability Diagram

The reliability diagram is the standard way to visualize calibration quality.

```python
from rankcal import reliability_diagram
import matplotlib.pyplot as plt

# Full reliability diagram
fig = reliability_diagram(scores, labels, n_bins=10)
plt.show()

# Top-k focused view
fig = reliability_diagram(scores, labels, k=50, n_bins=10)
plt.show()
```

**Reading the diagram:**

- The diagonal line represents perfect calibration
- Points above the diagonal: underconfident (actual > predicted)
- Points below the diagonal: overconfident (predicted > actual)
- Bar heights show the number of samples in each bin

## Comparing Before and After Calibration

```python
from rankcal import IsotonicCalibrator, ece, ece_at_k, reliability_diagram
import matplotlib.pyplot as plt

# Fit calibrator
calibrator = IsotonicCalibrator()
calibrator.fit(scores, labels)
calibrated = calibrator(scores)

# Compare metrics
print("Before calibration:")
print(f"  ECE: {ece(scores, labels):.4f}")
print(f"  ECE@10: {ece_at_k(scores, labels, k=10):.4f}")

print("\nAfter calibration:")
print(f"  ECE: {ece(calibrated, labels):.4f}")
print(f"  ECE@10: {ece_at_k(calibrated, labels, k=10):.4f}")

# Compare visually
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
reliability_diagram(scores, labels, ax=axes[0])
axes[0].set_title("Before Calibration")
reliability_diagram(calibrated, labels, ax=axes[1])
axes[1].set_title("After Calibration")
plt.tight_layout()
plt.show()
```

## Ranking-Specific Metrics

### Precision at k

```python
from rankcal import precision_at_k, expected_precision_at_k

# Actual precision in top-k
actual = precision_at_k(scores, labels, k=10)

# Expected precision based on calibrated scores
expected = expected_precision_at_k(calibrated_scores, k=10)

# The gap indicates calibration quality
print(f"Actual P@10: {actual:.4f}")
print(f"Expected P@10: {expected:.4f}")
print(f"Gap: {abs(actual - expected):.4f}")
```

### Calibration Gap at k

Directly measures the gap between expected and actual precision at k.

```python
from rankcal import calibration_gap_at_k

gap = calibration_gap_at_k(scores, labels, k=10)
print(f"Calibration gap @10: {gap:.4f}")
```
