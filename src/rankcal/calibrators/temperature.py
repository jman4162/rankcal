"""Temperature scaling calibrator."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCalibrator


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling calibrator.

    Applies a learned temperature parameter to scale logits:
        calibrated = sigmoid(logit(scores) / temperature)

    This is the simplest parametric calibrator and serves as a strong baseline.
    It's differentiable and can be trained end-to-end.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        """Initialize temperature scaling.

        Args:
            init_temperature: Initial temperature value. Default 1.0 (no scaling).
        """
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs: Any,
    ) -> "TemperatureScaling":
        """Fit temperature parameter using NLL loss.

        Args:
            scores: Predicted scores in (0, 1), shape (n_samples,)
            labels: Binary relevance labels, shape (n_samples,)
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            **kwargs: Unused, for API compatibility

        Returns:
            self
        """
        scores, validated_labels = self._validate_inputs(scores, labels)
        assert validated_labels is not None
        labels = validated_labels

        # Clamp scores to avoid inf logits
        scores = scores.clamp(1e-7, 1 - 1e-7)
        logits = torch.logit(scores)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss

        prev_loss = float("inf")
        for _ in range(max_iter):
            loss = optimizer.step(closure)
            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

        self._fitted = True
        return self

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to scores.

        Args:
            scores: Uncalibrated scores in (0, 1)

        Returns:
            Calibrated scores in (0, 1)
        """
        self._check_fitted()
        scores, _ = self._validate_inputs(scores)

        # Clamp to avoid inf logits
        scores = scores.clamp(1e-7, 1 - 1e-7)
        logits = torch.logit(scores)
        scaled_logits = logits / self.temperature
        return torch.sigmoid(scaled_logits)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature.item():.4f}"
