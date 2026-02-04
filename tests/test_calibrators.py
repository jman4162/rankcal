"""Tests for calibrators."""

import pytest
import torch

from rankcal import (
    IsotonicCalibrator,
    MonotonicNNCalibrator,
    SplineCalibrator,
    TemperatureScaling,
)
from rankcal.utils import generate_calibrated_data, generate_miscalibrated_data


class TestTemperatureScaling:
    def test_init(self):
        cal = TemperatureScaling()
        assert cal.temperature.item() == 1.0
        assert not cal.fitted

    def test_init_custom_temperature(self):
        cal = TemperatureScaling(init_temperature=2.0)
        assert cal.temperature.item() == 2.0

    def test_fit_returns_self(self):
        cal = TemperatureScaling()
        scores, labels = generate_calibrated_data(100, seed=42)
        result = cal.fit(scores, labels)
        assert result is cal
        assert cal.fitted

    def test_forward_requires_fit(self):
        cal = TemperatureScaling()
        scores = torch.rand(10)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal(scores)

    def test_forward_preserves_shape(self):
        cal = TemperatureScaling()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels)

        test_scores = torch.rand(50)
        calibrated = cal(test_scores)
        assert calibrated.shape == test_scores.shape

    def test_calibration_improves_miscalibrated(self):
        """Temperature scaling should reduce miscalibration."""
        scores, labels = generate_miscalibrated_data(500, temperature=2.0, seed=42)

        cal = TemperatureScaling()
        cal.fit(scores, labels)

        # After fitting, temperature should be > 1 to counter the miscalibration
        assert cal.temperature.item() > 1.0

    def test_output_in_valid_range(self):
        cal = TemperatureScaling()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels)

        calibrated = cal(scores)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()


class TestIsotonicCalibrator:
    def test_init(self):
        cal = IsotonicCalibrator()
        assert not cal.fitted

    def test_fit_returns_self(self):
        cal = IsotonicCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        result = cal.fit(scores, labels)
        assert result is cal
        assert cal.fitted

    def test_forward_requires_fit(self):
        cal = IsotonicCalibrator()
        scores = torch.rand(10)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal(scores)

    def test_forward_preserves_shape(self):
        cal = IsotonicCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels)

        test_scores = torch.rand(50)
        calibrated = cal(test_scores)
        assert calibrated.shape == test_scores.shape

    def test_output_is_monotonic(self):
        """Isotonic calibrator should produce monotonically increasing outputs."""
        cal = IsotonicCalibrator()
        scores, labels = generate_miscalibrated_data(200, seed=42)
        cal.fit(scores, labels)

        # Test on sorted inputs
        test_scores = torch.linspace(0.01, 0.99, 100)
        calibrated = cal(test_scores)

        # Check monotonicity (allowing for floating point tolerance)
        diffs = calibrated[1:] - calibrated[:-1]
        assert (diffs >= -1e-6).all()

    def test_output_in_valid_range(self):
        cal = IsotonicCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels)

        calibrated = cal(scores)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()


class TestSplineCalibrator:
    def test_init(self):
        cal = SplineCalibrator()
        assert cal.n_knots == 10
        assert not cal.fitted

    def test_init_custom_knots(self):
        cal = SplineCalibrator(n_knots=5)
        assert cal.n_knots == 5

    def test_fit_returns_self(self):
        cal = SplineCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        result = cal.fit(scores, labels, max_iter=50)
        assert result is cal
        assert cal.fitted

    def test_forward_requires_fit(self):
        cal = SplineCalibrator()
        scores = torch.rand(10)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal(scores)

    def test_forward_preserves_shape(self):
        cal = SplineCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels, max_iter=50)

        test_scores = torch.rand(50)
        calibrated = cal(test_scores)
        assert calibrated.shape == test_scores.shape

    def test_output_is_monotonic(self):
        """Spline calibrator should produce monotonically increasing outputs."""
        cal = SplineCalibrator()
        scores, labels = generate_calibrated_data(200, seed=42)
        cal.fit(scores, labels, max_iter=100)

        test_scores = torch.linspace(0.01, 0.99, 100)
        calibrated = cal(test_scores)

        diffs = calibrated[1:] - calibrated[:-1]
        assert (diffs >= -1e-5).all()

    def test_output_in_valid_range(self):
        cal = SplineCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels, max_iter=50)

        calibrated = cal(scores)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()


class TestMonotonicNNCalibrator:
    def test_init(self):
        cal = MonotonicNNCalibrator()
        assert cal.hidden_dims == (16, 16)
        assert not cal.fitted

    def test_init_custom_dims(self):
        cal = MonotonicNNCalibrator(hidden_dims=(8, 8, 8))
        assert cal.hidden_dims == (8, 8, 8)

    def test_fit_returns_self(self):
        cal = MonotonicNNCalibrator(hidden_dims=(8,))
        scores, labels = generate_calibrated_data(100, seed=42)
        result = cal.fit(scores, labels, max_iter=50)
        assert result is cal
        assert cal.fitted

    def test_forward_requires_fit(self):
        cal = MonotonicNNCalibrator()
        scores = torch.rand(10)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal(scores)

    def test_forward_preserves_shape(self):
        cal = MonotonicNNCalibrator(hidden_dims=(8,))
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels, max_iter=50)

        test_scores = torch.rand(50)
        calibrated = cal(test_scores)
        assert calibrated.shape == test_scores.shape

    def test_output_is_monotonic(self):
        """Monotonic NN should produce monotonically increasing outputs."""
        cal = MonotonicNNCalibrator(hidden_dims=(16,))
        scores, labels = generate_calibrated_data(200, seed=42)
        cal.fit(scores, labels, max_iter=200)

        test_scores = torch.linspace(0.01, 0.99, 100)
        calibrated = cal(test_scores)

        diffs = calibrated[1:] - calibrated[:-1]
        assert (diffs >= -1e-5).all()

    def test_output_in_valid_range(self):
        cal = MonotonicNNCalibrator(hidden_dims=(8,))
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels, max_iter=50)

        calibrated = cal(scores)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()


class TestInputValidation:
    """Test input validation across calibrators."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, SplineCalibrator],
    )
    def test_shape_mismatch_raises(self, calibrator_cls):
        cal = calibrator_cls()
        scores = torch.rand(10)
        labels = torch.rand(5)
        with pytest.raises(ValueError, match="same shape"):
            cal.fit(scores, labels)

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, SplineCalibrator],
    )
    def test_accepts_numpy_arrays(self, calibrator_cls):
        import numpy as np

        cal = calibrator_cls()
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100).astype(float)

        if calibrator_cls == SplineCalibrator:
            cal.fit(scores, labels, max_iter=10)
        else:
            cal.fit(scores, labels)

        assert cal.fitted
