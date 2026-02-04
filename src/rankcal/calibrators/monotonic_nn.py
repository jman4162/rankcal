"""Monotonic neural network calibrator."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCalibrator


class MonotonicLinear(nn.Module):
    """Linear layer with non-negative weights for monotonicity."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use softplus to ensure non-negative weights
        weight = F.softplus(self.weight_raw)
        return F.linear(x, weight, self.bias)


class MonotonicNNCalibrator(BaseCalibrator):
    """Monotonic neural network calibrator.

    Uses a neural network with constrained architecture to ensure monotonicity:
    - All weights are non-negative (via softplus)
    - All activations are monotonic (ELU)

    This is the most flexible calibrator, suitable for complex calibration patterns.
    Fully differentiable and trainable end-to-end.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (16, 16),
    ) -> None:
        """Initialize monotonic neural network.

        Args:
            hidden_dims: Tuple of hidden layer dimensions.
        """
        super().__init__()
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        in_dim = 1
        for hidden_dim in hidden_dims:
            layers.append(MonotonicLinear(in_dim, hidden_dim))
            layers.append(nn.ELU())  # ELU is monotonic
            in_dim = hidden_dim
        layers.append(MonotonicLinear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        batch_size: Optional[int] = None,
    ) -> MonotonicNNCalibrator:
        """Fit neural network using NLL loss.

        Args:
            scores: Predicted scores in (0, 1), shape (n_samples,)
            labels: Binary relevance labels, shape (n_samples,)
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            batch_size: Batch size for training. None for full batch.

        Returns:
            self
        """
        scores, labels = self._validate_inputs(scores, labels)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_samples = len(scores)
        if batch_size is None:
            batch_size = n_samples

        prev_loss = float("inf")
        for epoch in range(max_iter):
            # Shuffle data
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i : i + batch_size]
                batch_scores = scores[batch_idx]
                batch_labels = labels[batch_idx]

                optimizer.zero_grad()

                calibrated = self._forward_unchecked(batch_scores)
                calibrated = calibrated.clamp(1e-7, 1 - 1e-7)
                loss = F.binary_cross_entropy(calibrated, batch_labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_idx)

            epoch_loss /= n_samples

            if abs(prev_loss - epoch_loss) < tol:
                break
            prev_loss = epoch_loss

        self._fitted = True
        return self

    def _forward_unchecked(self, scores: torch.Tensor) -> torch.Tensor:
        """Forward pass without fitted check (for training)."""
        scores, _ = self._validate_inputs(scores)

        # Transform to logit space for better numerical properties
        scores_clamped = scores.clamp(1e-7, 1 - 1e-7)
        logits = torch.logit(scores_clamped)

        # Normalize to roughly [-3, 3] range for the network
        x = logits.unsqueeze(-1) / 3.0

        # Forward through monotonic network
        out = self.network(x).squeeze(-1)

        # Map back to [0, 1]
        return torch.sigmoid(out)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply neural network calibration.

        Args:
            scores: Uncalibrated scores in (0, 1)

        Returns:
            Calibrated scores in (0, 1)
        """
        self._check_fitted()
        return self._forward_unchecked(scores)

    def extra_repr(self) -> str:
        return f"hidden_dims={self.hidden_dims}"
