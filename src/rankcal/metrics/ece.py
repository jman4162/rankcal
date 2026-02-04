"""Expected Calibration Error metrics."""

from __future__ import annotations

from typing import Tuple

import torch


def ece(
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """Compute Expected Calibration Error.

    ECE measures the average absolute difference between predicted confidence
    and actual accuracy across bins.

    Args:
        scores: Predicted scores in (0, 1), shape (n_samples,)
        labels: Binary labels, shape (n_samples,)
        n_bins: Number of bins for bucketing scores

    Returns:
        ECE value as a scalar tensor
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    scores = scores.flatten()
    labels = labels.flatten()

    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_indices = torch.bucketize(scores, bin_boundaries[1:-1])

    ece_value = torch.tensor(0.0)
    total_samples = len(scores)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_size = mask.sum()

        if bin_size > 0:
            bin_scores = scores[mask]
            bin_labels = labels[mask]

            avg_confidence = bin_scores.mean()
            avg_accuracy = bin_labels.mean()

            ece_value = ece_value + (bin_size / total_samples) * torch.abs(
                avg_confidence - avg_accuracy
            )

    return ece_value


def ece_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    n_bins: int = 10,
) -> torch.Tensor:
    """Compute Expected Calibration Error at top-k.

    Only considers the top-k items by score when computing ECE.
    This measures calibration where ranking decisions actually happen.

    Args:
        scores: Predicted scores in (0, 1), shape (n_samples,)
        labels: Binary labels, shape (n_samples,)
        k: Number of top items to consider
        n_bins: Number of bins for bucketing scores

    Returns:
        ECE@k value as a scalar tensor
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    scores = scores.flatten()
    labels = labels.flatten()

    # Get top-k indices
    k = min(k, len(scores))
    _, top_k_indices = torch.topk(scores, k)

    top_k_scores = scores[top_k_indices]
    top_k_labels = labels[top_k_indices]

    return ece(top_k_scores, top_k_labels, n_bins=n_bins)


def calibration_error_per_bin(
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-bin calibration statistics.

    Useful for building reliability diagrams.

    Args:
        scores: Predicted scores in (0, 1), shape (n_samples,)
        labels: Binary labels, shape (n_samples,)
        n_bins: Number of bins

    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_confidences, bin_counts)
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    scores = scores.flatten()
    labels = labels.flatten()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_indices = torch.bucketize(scores, bin_boundaries[1:-1])

    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracies = torch.zeros(n_bins)
    bin_confidences = torch.zeros(n_bins)
    bin_counts = torch.zeros(n_bins)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_size = mask.sum()
        bin_counts[bin_idx] = bin_size

        if bin_size > 0:
            bin_accuracies[bin_idx] = labels[mask].mean()
            bin_confidences[bin_idx] = scores[mask].mean()

    return bin_centers, bin_accuracies, bin_confidences, bin_counts
