# Calibrators

All calibrators inherit from `BaseCalibrator` and provide a consistent API for fitting and applying calibration transformations.

## BaseCalibrator

::: rankcal.BaseCalibrator
    options:
      show_source: false
      members:
        - fit
        - forward

## TemperatureScaling

Single-parameter calibration that scales logits by a learned temperature.

::: rankcal.TemperatureScaling
    options:
      show_source: false

## IsotonicCalibrator

Non-parametric calibration using isotonic regression.

::: rankcal.IsotonicCalibrator
    options:
      show_source: false

## PiecewiseLinearCalibrator

Differentiable piecewise linear calibration with monotonicity constraints.

::: rankcal.PiecewiseLinearCalibrator
    options:
      show_source: false

## MonotonicNNCalibrator

Neural network calibrator with monotonicity constraints for maximum flexibility.

::: rankcal.MonotonicNNCalibrator
    options:
      show_source: false
