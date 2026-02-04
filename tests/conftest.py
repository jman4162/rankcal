"""Pytest configuration and fixtures."""

import pytest
import torch


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on: cpu, cuda, or mps",
    )


@pytest.fixture
def device(request):
    """Get the device to use for tests.

    Returns the device specified by --device option, or skips if unavailable.
    """
    device_name = request.config.getoption("--device")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return torch.device("cuda")
    elif device_name == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS is not available")
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def cuda_available():
    """Check if CUDA is available, skip test if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return True


@pytest.fixture
def any_gpu_available():
    """Check if any GPU (CUDA or MPS) is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        pytest.skip("No GPU available (neither CUDA nor MPS)")
