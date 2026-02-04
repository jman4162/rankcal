"""Base calibrator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseCalibrator(nn.Module, ABC):
    """Abstract base class for all calibrators.

    All calibrators are nn.Module subclasses for PyTorch compatibility.
    They implement a monotonic transformation of scores.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> "BaseCalibrator":
        """Fit the calibrator to validation data.

        Args:
            scores: Predicted scores, shape (n_samples,)
            labels: Binary relevance labels, shape (n_samples,)
            **kwargs: Additional calibrator-specific parameters

        Returns:
            self
        """
        pass

    @abstractmethod
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply calibration to scores.

        Args:
            scores: Uncalibrated scores, shape (n_samples,) or (batch, n_samples)

        Returns:
            Calibrated scores, same shape as input
        """
        pass

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling forward(). "
                "Call .fit(scores, labels) first."
            )

    def _validate_inputs(
        self, scores: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Validate and convert inputs to appropriate format.

        Args:
            scores: Input scores
            labels: Optional labels

        Returns:
            Validated (scores, labels) tuple
        """
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, dtype=torch.float32)
        else:
            scores = scores.float()

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, dtype=torch.float32)
            else:
                labels = labels.float()

            if scores.shape != labels.shape:
                raise ValueError(
                    f"scores and labels must have same shape, "
                    f"got {scores.shape} and {labels.shape}"
                )

        return scores, labels
