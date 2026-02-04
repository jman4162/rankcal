"""Isotonic regression calibrator."""

import torch

from .base import BaseCalibrator


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibrator.

    Fits a piecewise constant monotonic function using pool adjacent violators
    algorithm (PAVA). Non-parametric and guaranteed monotonic.

    Note: Not differentiable, but fast and reliable baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        # Store the mapping as buffers (not parameters)
        self.register_buffer("_x_thresholds", torch.tensor([]))
        self.register_buffer("_y_values", torch.tensor([]))

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression using PAVA algorithm.

        Args:
            scores: Predicted scores, shape (n_samples,)
            labels: Binary relevance labels, shape (n_samples,)

        Returns:
            self
        """
        scores, labels = self._validate_inputs(scores, labels)

        # Sort by scores
        sorted_indices = torch.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # PAVA algorithm
        y_values, weights = self._pava(sorted_labels)

        # Build lookup table: store unique score thresholds and corresponding y values
        # We use the sorted scores as thresholds
        self._x_thresholds = sorted_scores.clone()
        self._y_values = y_values

        self._fitted = True
        return self

    def _pava(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool Adjacent Violators Algorithm.

        Args:
            y: Sorted target values

        Returns:
            Isotonic y values and weights
        """
        n = len(y)
        y = y.clone()
        weights = torch.ones_like(y)

        # Indices of blocks
        blocks = list(range(n))

        while True:
            # Find violators (adjacent pairs where y[i] > y[i+1])
            violator_found = False
            i = 0
            while i < len(blocks) - 1:
                curr_idx = blocks[i]
                next_idx = blocks[i + 1]

                if y[curr_idx] > y[next_idx]:
                    # Pool the blocks
                    total_weight = weights[curr_idx] + weights[next_idx]
                    pooled_value = (
                        weights[curr_idx] * y[curr_idx]
                        + weights[next_idx] * y[next_idx]
                    ) / total_weight

                    y[curr_idx] = pooled_value
                    weights[curr_idx] = total_weight

                    # Remove the next block
                    blocks.pop(i + 1)
                    violator_found = True
                else:
                    i += 1

            if not violator_found:
                break

        # Expand block values to all original positions
        result = torch.zeros_like(y)
        block_ptr = 0
        for i in range(n):
            if block_ptr < len(blocks) - 1 and i >= blocks[block_ptr + 1]:
                block_ptr += 1
            result[i] = y[blocks[block_ptr]]

        return result, weights

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply isotonic calibration via interpolation.

        Args:
            scores: Uncalibrated scores

        Returns:
            Calibrated scores
        """
        self._check_fitted()
        scores, _ = self._validate_inputs(scores)

        original_shape = scores.shape
        scores_flat = scores.flatten()

        # Use searchsorted for efficient lookup
        indices = torch.searchsorted(self._x_thresholds, scores_flat)

        # Clamp indices to valid range
        indices = indices.clamp(0, len(self._y_values) - 1)

        # Look up calibrated values
        calibrated = self._y_values[indices]

        return calibrated.reshape(original_shape)

    def extra_repr(self) -> str:
        if self._fitted:
            return f"n_thresholds={len(self._x_thresholds)}"
        return "not fitted"
