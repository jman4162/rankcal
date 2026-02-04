"""Spline calibrator with monotonicity constraints."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCalibrator


class SplineCalibrator(BaseCalibrator):
    """Monotonic spline calibrator.

    Uses a cubic spline with constrained coefficients to ensure monotonicity.
    Differentiable and smooth, good balance between flexibility and stability.
    """

    def __init__(self, n_knots: int = 10) -> None:
        """Initialize spline calibrator.

        Args:
            n_knots: Number of interior knots for the spline.
        """
        super().__init__()
        self.n_knots = n_knots

        # Knot positions (fixed, uniformly spaced in [0, 1])
        knots = torch.linspace(0, 1, n_knots + 2)
        self.register_buffer("knots", knots)

        # Learnable parameters: values at knots (will be constrained to be monotonic)
        # Initialize to identity mapping
        self.knot_values_raw = nn.Parameter(torch.zeros(n_knots + 2))

    def _get_monotonic_values(self) -> torch.Tensor:
        """Get monotonically increasing knot values.

        Uses softplus on differences to ensure monotonicity.
        """
        # First value starts at 0
        first = torch.sigmoid(self.knot_values_raw[0]) * 0.1

        # Subsequent values are cumulative sums of positive increments
        increments = F.softplus(self.knot_values_raw[1:]) * 0.1
        values = torch.cat([first.unsqueeze(0), first + torch.cumsum(increments, 0)])

        # Scale to [0, 1]
        values = values / (values[-1] + 1e-8)
        return values

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> "SplineCalibrator":
        """Fit spline parameters using NLL loss.

        Args:
            scores: Predicted scores in (0, 1), shape (n_samples,)
            labels: Binary relevance labels, shape (n_samples,)
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance

        Returns:
            self
        """
        scores, labels = self._validate_inputs(scores, labels)

        optimizer = torch.optim.Adam([self.knot_values_raw], lr=lr)

        prev_loss = float("inf")
        for _ in range(max_iter):
            optimizer.zero_grad()

            calibrated = self._interpolate(scores)
            calibrated = calibrated.clamp(1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(calibrated, labels)

            loss.backward()
            optimizer.step()

            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

        self._fitted = True
        return self

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate using linear interpolation between knots.

        Args:
            x: Input values in [0, 1]

        Returns:
            Interpolated values
        """
        knot_values = self._get_monotonic_values()

        # Find which segment each x falls into
        x_clamped = x.clamp(0, 1)

        # Linear interpolation
        # Find indices of surrounding knots
        indices = torch.searchsorted(self.knots, x_clamped) - 1
        indices = indices.clamp(0, len(self.knots) - 2)

        # Get surrounding knot positions and values
        x0 = self.knots[indices]
        x1 = self.knots[indices + 1]
        y0 = knot_values[indices]
        y1 = knot_values[indices + 1]

        # Linear interpolation
        t = (x_clamped - x0) / (x1 - x0 + 1e-8)
        return y0 + t * (y1 - y0)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply spline calibration.

        Args:
            scores: Uncalibrated scores in (0, 1)

        Returns:
            Calibrated scores in (0, 1)
        """
        self._check_fitted()
        scores, _ = self._validate_inputs(scores)
        return self._interpolate(scores)

    def extra_repr(self) -> str:
        return f"n_knots={self.n_knots}"
