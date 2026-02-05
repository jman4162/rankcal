# Decision Making with Calibrated Scores

This guide explains when calibration matters and how to use calibrated scores for decision-making.

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

## Decision Tools

### Optimal Threshold

Find the threshold that maximizes expected utility:

```python
from rankcal import optimal_threshold

# Define your utility function via benefit and cost
threshold, expected_utility = optimal_threshold(
    scores, labels,
    benefit=1.0,  # Value of a true positive
    cost=0.5      # Cost of a false positive
)

print(f"Optimal threshold: {threshold:.4f}")
print(f"Expected utility: {expected_utility:.4f}")
```

### Utility Curves

Visualize how utility changes with threshold:

```python
from rankcal import utility_curve, plot_utility_curve

# Get curve data
thresholds, utilities = utility_curve(scores, labels, benefit=1.0, cost=0.5)

# Or plot directly
fig = plot_utility_curve(scores, labels, benefit=1.0, cost=0.5)
fig.savefig("utility_curve.png")
```

### Risk-Coverage Curves

Understand the tradeoff between coverage (how many items you select) and risk (error rate):

```python
from rankcal import risk_coverage_curve, plot_risk_coverage

# Get curve data
coverages, risks = risk_coverage_curve(scores, labels)

# Or plot directly
fig = plot_risk_coverage(scores, labels)
fig.savefig("risk_coverage.png")
```

### Budget-Constrained Selection

Select the best k items given a budget constraint:

```python
from rankcal import budget_constrained_selection, expected_utility_at_budget

# Select top-k items
selected_indices = budget_constrained_selection(scores, k=10)

# Estimate utility at a given budget
utility = expected_utility_at_budget(
    scores, labels, k=10, benefit=1.0, cost=0.5
)
```

### Threshold for Coverage

Find the threshold that achieves a target coverage:

```python
from rankcal import threshold_for_coverage

# Find threshold to select approximately 20% of items
threshold = threshold_for_coverage(scores, coverage=0.2)
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
