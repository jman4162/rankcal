"""Ranking-specific calibration metrics."""

from __future__ import annotations

from typing import Optional

import torch


def precision_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute precision at k.

    Args:
        scores: Predicted scores, shape (n_samples,)
        labels: Binary relevance labels, shape (n_samples,)
        k: Number of top items to consider

    Returns:
        Precision@k as a scalar tensor
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    k = min(k, len(scores))
    _, top_k_indices = torch.topk(scores, k)
    return labels[top_k_indices].mean()


def expected_precision_at_k(
    scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute expected precision at k from calibrated scores.

    If scores are well-calibrated, this should match actual precision@k.

    Args:
        scores: Calibrated scores (probabilities), shape (n_samples,)
        k: Number of top items to consider

    Returns:
        Expected precision@k as a scalar tensor
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)

    k = min(k, len(scores))
    top_k_scores, _ = torch.topk(scores, k)
    return top_k_scores.mean()


def calibration_gap_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute the calibration gap at top-k.

    This is the difference between expected precision (from scores) and
    actual precision (from labels) at top-k.

    Args:
        scores: Calibrated scores, shape (n_samples,)
        labels: Binary relevance labels, shape (n_samples,)
        k: Number of top items to consider

    Returns:
        Calibration gap as a scalar tensor (positive = overconfident)
    """
    expected = expected_precision_at_k(scores, k)
    actual = precision_at_k(scores, labels, k)
    return expected - actual


def mean_predicted_relevance(
    scores: torch.Tensor,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Compute mean predicted relevance.

    Args:
        scores: Predicted scores, shape (n_samples,)
        k: If provided, only consider top-k items

    Returns:
        Mean predicted relevance
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)

    if k is not None:
        k = min(k, len(scores))
        top_k_scores, _ = torch.topk(scores, k)
        return top_k_scores.mean()
    return scores.mean()


def mean_actual_relevance(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Compute mean actual relevance at top-k by score.

    Args:
        scores: Predicted scores, shape (n_samples,)
        labels: Binary relevance labels, shape (n_samples,)
        k: If provided, only consider top-k items by score

    Returns:
        Mean actual relevance
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    if k is not None:
        k = min(k, len(scores))
        _, top_k_indices = torch.topk(scores, k)
        return labels[top_k_indices].mean()
    return labels.mean()
