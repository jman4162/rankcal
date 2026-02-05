"""Base calibrator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import rankcal


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

    def save(self, path: str) -> None:
        """Save calibrator to disk.

        Args:
            path: File path to save to (e.g., 'calibrator.pt')

        Raises:
            RuntimeError: If calibrator has not been fitted

        Example:
            >>> calibrator = TemperatureScaling()
            >>> calibrator.fit(scores, labels)
            >>> calibrator.save('my_calibrator.pt')
        """
        self._check_fitted()

        metadata = {
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "rankcal_version": rankcal.__version__,
            "state_dict": self.state_dict(),
            "fitted": self._fitted,
        }
        torch.save(metadata, path)

    @classmethod
    def load(cls, path: str) -> "BaseCalibrator":
        """Load calibrator from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded calibrator instance

        Example:
            >>> calibrator = TemperatureScaling.load('my_calibrator.pt')
            >>> calibrated = calibrator(new_scores)
        """
        metadata = torch.load(path, weights_only=False)

        # Verify class matches
        if cls.__name__ != "BaseCalibrator" and cls.__name__ != metadata["class_name"]:
            raise ValueError(
                f"Saved calibrator is {metadata['class_name']}, "
                f"but loading with {cls.__name__}"
            )

        # Get the actual class
        if cls.__name__ == "BaseCalibrator":
            # Called via BaseCalibrator.load() - need to find the right class
            calibrator_cls = getattr(rankcal, metadata["class_name"])
        else:
            calibrator_cls = cls

        # Create instance and load state
        instance = calibrator_cls()

        # Manually load buffers and parameters to handle dynamic sizes
        # (e.g., IsotonicCalibrator has variable-length buffers)
        state_dict = metadata["state_dict"]
        for name, param in state_dict.items():
            parts = name.split(".")
            module = instance
            for part in parts[:-1]:
                module = getattr(module, part)
            attr_name = parts[-1]

            # Check if it's a registered buffer or parameter
            if attr_name in dict(module.named_buffers(recurse=False)):
                module.register_buffer(attr_name, param)
            elif attr_name in dict(module.named_parameters(recurse=False)):
                with torch.no_grad():
                    getattr(module, attr_name).copy_(param)
            else:
                setattr(module, attr_name, param)

        instance._fitted = metadata["fitted"]

        return instance
