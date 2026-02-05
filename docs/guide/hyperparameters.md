# Hyperparameter Guidance

This guide helps you tune hyperparameters for each calibrator.

## TemperatureScaling

```python
from rankcal import TemperatureScaling

calibrator = TemperatureScaling(init_temperature=1.0)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_temperature` | 1.0 | Starting value for optimization |

### Guidance

- **Values > 1** soften predictions (fix overconfidence)
- **Values < 1** sharpen predictions (fix underconfidence)
- The optimizer will find the best value; default is usually fine
- After fitting, access the learned temperature via `calibrator.temperature`

### Example

```python
from rankcal import TemperatureScaling

calibrator = TemperatureScaling(init_temperature=1.0)
calibrator.fit(scores, labels)

# Check the learned temperature
print(f"Learned temperature: {calibrator.temperature.item():.4f}")
```

## PiecewiseLinearCalibrator

```python
from rankcal import PiecewiseLinearCalibrator

calibrator = PiecewiseLinearCalibrator(n_knots=10)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_knots` | 10 | Number of knot points |

### Guidance

- **Increase to 15-20** for complex miscalibration patterns
- **Decrease to 5-7** if you have limited calibration data (<500 samples)
- More knots = more flexibility but higher risk of overfitting

### Choosing n_knots

| Calibration Data Size | Recommended n_knots |
|----------------------|---------------------|
| < 500 samples | 5-7 |
| 500-2000 samples | 10 (default) |
| 2000-10000 samples | 15-20 |
| > 10000 samples | 20-30 |

### Example

```python
from rankcal import PiecewiseLinearCalibrator

# For a dataset with 1000 calibration samples
calibrator = PiecewiseLinearCalibrator(n_knots=10)
calibrator.fit(scores, labels)

# For a dataset with 5000 calibration samples
calibrator = PiecewiseLinearCalibrator(n_knots=15)
calibrator.fit(scores, labels)
```

## MonotonicNNCalibrator

```python
from rankcal import MonotonicNNCalibrator

calibrator = MonotonicNNCalibrator(hidden_dims=(16, 16))
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | (16, 16) | Network architecture |

### Guidance

- **Use `(8,)`** for simpler patterns or small datasets
- **Use `(32, 32)`** for very complex patterns with large datasets
- More layers/units = more flexibility but needs more data

### Choosing Architecture

| Calibration Data Size | Recommended hidden_dims |
|----------------------|-------------------------|
| < 500 samples | (8,) |
| 500-2000 samples | (16,) or (16, 16) |
| 2000-10000 samples | (16, 16) (default) |
| > 10000 samples | (32, 32) or (32, 32, 32) |

### Example

```python
from rankcal import MonotonicNNCalibrator

# For a small dataset
calibrator = MonotonicNNCalibrator(hidden_dims=(8,))
calibrator.fit(scores, labels)

# For a large dataset with complex miscalibration
calibrator = MonotonicNNCalibrator(hidden_dims=(32, 32))
calibrator.fit(scores, labels)
```

## IsotonicCalibrator

```python
from rankcal import IsotonicCalibrator

calibrator = IsotonicCalibrator()
```

### Parameters

The `IsotonicCalibrator` has no hyperparameters to tune. It automatically adapts to the data.

### Why No Hyperparameters?

Isotonic regression is a non-parametric method that:

- Finds the optimal piecewise-constant monotonic function
- Automatically determines the number and location of steps
- Works well across different data sizes

This makes it an excellent default choice when you don't want to tune hyperparameters.
