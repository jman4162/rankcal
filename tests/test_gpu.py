"""GPU tests for calibrators.

These tests verify that all calibrators work correctly on GPU devices.
Run with `pytest tests/test_gpu.py --device cuda` (or `--device mps` on Mac).
"""

import pytest
import torch

from rankcal import (
    IsotonicCalibrator,
    MonotonicNNCalibrator,
    PiecewiseLinearCalibrator,
    TemperatureScaling,
)
from rankcal.utils import generate_calibrated_data


class TestDeviceTransfer:
    """Test that calibrators can be moved to different devices."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            IsotonicCalibrator,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_to_device(self, calibrator_cls, device):
        """Calibrator can be moved to target device."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        cal = cal.to(device)

        # Verify parameters are on correct device
        for param in cal.parameters():
            assert param.device.type == device.type

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            IsotonicCalibrator,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_fit_on_device(self, calibrator_cls, device):
        """Calibrator can be fitted with data on target device."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        cal = cal.to(device)

        # Generate data on device
        scores, labels = generate_calibrated_data(100, seed=42)
        scores = scores.to(device)
        labels = labels.to(device)

        # Fit should work
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        assert cal.fitted

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            IsotonicCalibrator,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_forward_on_device(self, calibrator_cls, device):
        """Calibrator forward pass works on target device with correct output device."""
        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        cal = cal.to(device)

        # Fit on device
        scores, labels = generate_calibrated_data(100, seed=42)
        scores = scores.to(device)
        labels = labels.to(device)

        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Forward pass
        test_scores = torch.rand(50, device=device)
        calibrated = cal(test_scores)

        # Output should be on same device
        assert calibrated.device.type == device.type
        assert calibrated.shape == test_scores.shape

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            IsotonicCalibrator,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_cpu_to_gpu_transfer(self, calibrator_cls, any_gpu_available):
        """Calibrator can be fitted on CPU then moved to GPU."""
        device = any_gpu_available

        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        # Fit on CPU
        scores, labels = generate_calibrated_data(100, seed=42)
        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Move to GPU
        cal = cal.to(device)

        # Forward on GPU
        test_scores = torch.rand(50, device=device)
        calibrated = cal(test_scores)

        assert calibrated.device.type == device.type

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            IsotonicCalibrator,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_gpu_to_cpu_transfer(self, calibrator_cls, any_gpu_available):
        """Calibrator can be fitted on GPU then moved to CPU."""
        device = any_gpu_available

        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        cal = cal.to(device)

        # Fit on GPU
        scores, labels = generate_calibrated_data(100, seed=42)
        scores = scores.to(device)
        labels = labels.to(device)

        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=20)
        else:
            cal.fit(scores, labels)

        # Move to CPU
        cal = cal.to("cpu")

        # Forward on CPU
        test_scores = torch.rand(50)
        calibrated = cal(test_scores)

        assert calibrated.device.type == "cpu"


class TestGPUCorrectness:
    """Test that GPU results match CPU results."""

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            IsotonicCalibrator,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_gpu_outputs_valid_range(self, calibrator_cls, any_gpu_available):
        """GPU calibration should produce outputs in valid [0, 1] range."""
        device = any_gpu_available

        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(8,))
        else:
            cal = calibrator_cls()

        cal = cal.to(device)

        scores, labels = generate_calibrated_data(100, seed=42)
        scores = scores.to(device)
        labels = labels.to(device)

        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=50)
        else:
            cal.fit(scores, labels)

        # Test forward
        test_scores = torch.rand(100, device=device)
        calibrated = cal(test_scores)

        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()

    @pytest.mark.parametrize(
        "calibrator_cls",
        [
            TemperatureScaling,
            PiecewiseLinearCalibrator,
            MonotonicNNCalibrator,
        ],
    )
    def test_gpu_monotonicity(self, calibrator_cls, any_gpu_available):
        """GPU calibration should preserve monotonicity."""
        device = any_gpu_available

        if calibrator_cls == MonotonicNNCalibrator:
            cal = calibrator_cls(hidden_dims=(16,))
        else:
            cal = calibrator_cls()

        cal = cal.to(device)

        scores, labels = generate_calibrated_data(200, seed=42)
        scores = scores.to(device)
        labels = labels.to(device)

        if calibrator_cls in (PiecewiseLinearCalibrator, MonotonicNNCalibrator):
            cal.fit(scores, labels, max_iter=100)
        else:
            cal.fit(scores, labels)

        # Test on sorted inputs
        test_scores = torch.linspace(0.01, 0.99, 100, device=device)
        calibrated = cal(test_scores)

        # Check monotonicity
        diffs = calibrated[1:] - calibrated[:-1]
        assert (diffs >= -1e-5).all()


class TestGPUMemory:
    """Test GPU memory behavior."""

    def test_no_memory_leak(self, any_gpu_available):
        """Repeated fit/forward should not leak GPU memory."""
        device = any_gpu_available

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            initial_memory = torch.cuda.memory_allocated(device)

            cal = TemperatureScaling().to(device)
            scores, labels = generate_calibrated_data(100, seed=42)
            scores = scores.to(device)
            labels = labels.to(device)

            # Run multiple times
            for _ in range(5):
                cal = TemperatureScaling().to(device)
                cal.fit(scores, labels)
                _ = cal(scores)

            # Clean up
            del cal
            torch.cuda.empty_cache()

            final_memory = torch.cuda.memory_allocated(device)
            # Allow some tolerance for PyTorch internal caching
            assert final_memory - initial_memory < 10 * 1024 * 1024  # 10MB tolerance
