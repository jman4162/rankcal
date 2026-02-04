"""Edge case tests for calibrators and metrics.

Tests for handling of unusual inputs that could cause errors in production:
- Empty inputs
- NaN and inf values
- Scores outside [0,1]
- Batch processing
- Gradient flow
- Numerical stability
"""

import pytest
import torch
import warnings

from rankcal import (
    IsotonicCalibrator,
    MonotonicNNCalibrator,
    PiecewiseLinearCalibrator,
    TemperatureScaling,
    ece,
    ece_at_k,
)
from rankcal.utils import generate_calibrated_data


class TestEmptyInputs:
    """Test handling of empty inputs."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_empty_fit_handles_gracefully(self, calibrator_cls):
        """Fitting with empty tensors should either raise or succeed gracefully."""
        cal = calibrator_cls()
        empty_scores = torch.tensor([])
        empty_labels = torch.tensor([])

        # Either should raise or succeed without crashing
        try:
            if calibrator_cls == PiecewiseLinearCalibrator:
                cal.fit(empty_scores, empty_labels, max_iter=10)
            else:
                cal.fit(empty_scores, empty_labels)
            # If it succeeded, that's fine - the calibrator is fitted
            # (even if the fit is degenerate)
        except (ValueError, RuntimeError, IndexError):
            pass  # Raising is also acceptable

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_empty_forward_after_fit(self, calibrator_cls):
        """Forward with empty tensor after fitting should return empty tensor."""
        cal = calibrator_cls()
        scores, labels = generate_calibrated_data(100, seed=42)

        if calibrator_cls == PiecewiseLinearCalibrator:
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        empty_scores = torch.tensor([])
        result = cal(empty_scores)

        assert result.shape == empty_scores.shape
        assert len(result) == 0


class TestNaNAndInf:
    """Test handling of NaN and inf values."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_nan_in_scores_fit(self, calibrator_cls):
        """NaN in scores during fit should not silently corrupt results."""
        cal = calibrator_cls()
        scores = torch.tensor([0.1, 0.5, float("nan"), 0.9])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        # Either should raise or result in nan (not silently wrong values)
        try:
            if calibrator_cls == PiecewiseLinearCalibrator:
                cal.fit(scores, labels, max_iter=10)
            else:
                cal.fit(scores, labels)

            # If it didn't raise, result should contain nan or handle it
            test_scores = torch.tensor([0.5])
            result = cal(test_scores)
            # We allow NaN propagation or handling
        except (ValueError, RuntimeError):
            pass  # Raising is acceptable

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_inf_in_scores_forward(self, calibrator_cls):
        """Inf in scores during forward should be handled gracefully."""
        cal = calibrator_cls()
        scores, labels = generate_calibrated_data(100, seed=42)

        if calibrator_cls == PiecewiseLinearCalibrator:
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Test with infinity
        test_scores = torch.tensor([0.5, float("inf"), float("-inf")])
        result = cal(test_scores)

        # At minimum, the result should be a tensor of same shape
        assert result.shape == test_scores.shape
        # Non-inf input should produce valid output
        assert 0 <= result[0] <= 1


class TestScoresOutsideRange:
    """Test handling of scores outside [0, 1]."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_scores_above_one(self, calibrator_cls):
        """Scores > 1 should be handled (clamped or produce valid output)."""
        cal = calibrator_cls()
        scores, labels = generate_calibrated_data(100, seed=42)

        if calibrator_cls == PiecewiseLinearCalibrator:
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Scores above 1
        test_scores = torch.tensor([0.5, 1.5, 2.0])
        result = cal(test_scores)

        assert result.shape == test_scores.shape
        # Output should still be in valid range (clamped)
        assert (result >= 0).all()
        assert (result <= 1).all()

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_scores_below_zero(self, calibrator_cls):
        """Scores < 0 should be handled (clamped or produce valid output)."""
        cal = calibrator_cls()
        scores, labels = generate_calibrated_data(100, seed=42)

        if calibrator_cls == PiecewiseLinearCalibrator:
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Scores below 0
        test_scores = torch.tensor([-0.5, -1.0, 0.5])
        result = cal(test_scores)

        assert result.shape == test_scores.shape
        # Output should still be in valid range
        assert (result >= 0).all()
        assert (result <= 1).all()


class TestBatchProcessing:
    """Test batch processing with 2D+ tensors."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, PiecewiseLinearCalibrator, MonotonicNNCalibrator],
    )
    def test_2d_input(self, calibrator_cls):
        """2D input should work (batch of samples)."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        # Fit on 1D
        scores, labels = generate_calibrated_data(100, seed=42)
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Forward on 2D
        batch_scores = torch.rand(5, 10)
        result = cal(batch_scores)

        assert result.shape == batch_scores.shape
        assert (result >= 0).all()
        assert (result <= 1).all()

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, PiecewiseLinearCalibrator, MonotonicNNCalibrator],
    )
    def test_3d_input(self, calibrator_cls):
        """3D input should work."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        scores, labels = generate_calibrated_data(100, seed=42)
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Forward on 3D
        batch_scores = torch.rand(2, 3, 10)
        result = cal(batch_scores)

        assert result.shape == batch_scores.shape


class TestGradientFlow:
    """Test gradient flow through calibrators."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, PiecewiseLinearCalibrator, MonotonicNNCalibrator],
    )
    def test_gradients_flow_through_forward(self, calibrator_cls):
        """Gradients should flow through differentiable calibrators."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        scores, labels = generate_calibrated_data(100, seed=42)
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Create input that requires grad
        test_scores = torch.rand(10, requires_grad=True)
        result = cal(test_scores)

        # Compute a loss and backprop
        loss = result.sum()
        loss.backward()

        # Gradient should exist and be non-zero for at least some inputs
        assert test_scores.grad is not None
        assert (test_scores.grad != 0).any()

    def test_isotonic_no_gradient(self):
        """IsotonicCalibrator is not differentiable - output doesn't require grad."""
        cal = IsotonicCalibrator()
        scores, labels = generate_calibrated_data(100, seed=42)
        cal.fit(scores, labels)

        test_scores = torch.rand(10, requires_grad=True)
        result = cal(test_scores)

        # The output uses indexing which breaks the computation graph
        # Result will not require grad
        assert not result.requires_grad


class TestNumericalStability:
    """Test numerical stability near boundaries."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, PiecewiseLinearCalibrator, MonotonicNNCalibrator],
    )
    def test_scores_near_zero(self, calibrator_cls):
        """Scores very close to 0 should be handled stably."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        scores, labels = generate_calibrated_data(100, seed=42)
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Test with very small scores
        test_scores = torch.tensor([1e-10, 1e-8, 1e-6, 0.0])
        result = cal(test_scores)

        # Should produce finite, valid results
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert (result >= 0).all()
        assert (result <= 1).all()

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, PiecewiseLinearCalibrator, MonotonicNNCalibrator],
    )
    def test_scores_near_one(self, calibrator_cls):
        """Scores very close to 1 should be handled stably."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        scores, labels = generate_calibrated_data(100, seed=42)
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Test with scores very close to 1
        test_scores = torch.tensor([1.0, 1 - 1e-10, 1 - 1e-8, 1 - 1e-6])
        result = cal(test_scores)

        # Should produce finite, valid results
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert (result >= 0).all()
        assert (result <= 1).all()


class TestMetricsEdgeCases:
    """Test edge cases in metrics."""

    def test_ece_all_same_score(self):
        """ECE with all identical scores."""
        scores = torch.full((100,), 0.5)
        labels = torch.randint(0, 2, (100,)).float()

        result = ece(scores, labels)

        assert not torch.isnan(result)
        assert result >= 0

    def test_ece_at_k_k_equals_n(self):
        """ECE@k when k equals n should match full ECE."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()

        ece_full = ece(scores, labels)
        ece_k = ece_at_k(scores, labels, k=100)

        torch.testing.assert_close(ece_full, ece_k, rtol=1e-5, atol=1e-5)

    def test_ece_at_k_k_greater_than_n(self):
        """ECE@k when k > n should handle gracefully."""
        scores = torch.rand(50)
        labels = torch.randint(0, 2, (50,)).float()

        # Should not raise and should compute on all samples
        result = ece_at_k(scores, labels, k=100)

        assert not torch.isnan(result)
        assert result >= 0

    def test_ece_perfect_calibration(self):
        """ECE with perfectly calibrated scores should be near zero."""
        # Create perfectly calibrated data
        n = 1000
        scores = torch.rand(n)
        labels = (torch.rand(n) < scores).float()

        result = ece(scores, labels)

        # Should be very small (allowing for random variation)
        assert result < 0.1


class TestSingleSample:
    """Test handling of single samples."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [TemperatureScaling, IsotonicCalibrator, PiecewiseLinearCalibrator],
    )
    def test_single_sample_forward(self, calibrator_cls):
        """Forward with a single sample should work."""
        cal = calibrator_cls()
        scores, labels = generate_calibrated_data(100, seed=42)

        if calibrator_cls == PiecewiseLinearCalibrator:
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        single_score = torch.tensor([0.5])
        result = cal(single_score)

        assert result.shape == single_score.shape
        assert 0 <= result.item() <= 1

    def test_ece_single_sample(self):
        """ECE with a single sample should not raise."""
        scores = torch.tensor([0.5])
        labels = torch.tensor([1.0])

        # Should not raise
        result = ece(scores, labels)

        assert not torch.isnan(result)
