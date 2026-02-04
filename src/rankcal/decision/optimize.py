"""Threshold and budget optimization for decision making."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def optimal_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    benefit: float = 1.0,
    cost: float = 1.0,
    n_thresholds: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the threshold that maximizes utility.

    Utility = benefit * true_positives - cost * false_positives

    Args:
        scores: Calibrated scores, shape (n_samples,)
        labels: Binary relevance labels, shape (n_samples,)
        benefit: Benefit of a true positive
        cost: Cost of a false positive
        n_thresholds: Number of thresholds to evaluate

    Returns:
        Tuple of (optimal_threshold, maximum_utility)
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    thresholds = torch.linspace(0, 1, n_thresholds)
    best_threshold = torch.tensor(0.5)
    best_utility = torch.tensor(float("-inf"))

    for thresh in thresholds:
        predictions = (scores >= thresh).float()
        true_positives = (predictions * labels).sum()
        false_positives = (predictions * (1 - labels)).sum()
        utility = benefit * true_positives - cost * false_positives

        if utility > best_utility:
            best_utility = utility
            best_threshold = thresh

    return best_threshold, best_utility


def budget_constrained_selection(
    scores: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Select top-k items given a budget constraint.

    Args:
        scores: Calibrated scores, shape (n_samples,)
        budget: Maximum number of items to select

    Returns:
        Boolean mask indicating selected items, shape (n_samples,)
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)

    n_samples = len(scores)
    budget = min(budget, n_samples)

    _, top_indices = torch.topk(scores, budget)

    mask = torch.zeros(n_samples, dtype=torch.bool)
    mask[top_indices] = True

    return mask


def expected_utility_at_budget(
    scores: torch.Tensor,
    labels: torch.Tensor,
    budget: int,
    benefit: float = 1.0,
    cost: float = 1.0,
) -> torch.Tensor:
    """Compute expected utility given a budget constraint.

    Args:
        scores: Calibrated scores
        labels: Binary relevance labels
        budget: Maximum number of items to select
        benefit: Benefit per true positive
        cost: Cost per false positive

    Returns:
        Utility value
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    mask = budget_constrained_selection(scores, budget)
    selected_labels = labels[mask]

    true_positives = selected_labels.sum()
    false_positives = (1 - selected_labels).sum()

    utility: torch.Tensor = benefit * true_positives - cost * false_positives
    return utility


def utility_budget_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
    max_budget: Optional[int] = None,
    benefit: float = 1.0,
    cost: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute utility as a function of budget.

    Args:
        scores: Calibrated scores
        labels: Binary relevance labels
        max_budget: Maximum budget to evaluate (default: all samples)
        benefit: Benefit per true positive
        cost: Cost per false positive

    Returns:
        Tuple of (budgets, utilities) tensors
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    n_samples = len(scores)
    if max_budget is None:
        max_budget = n_samples

    max_budget = min(max_budget, n_samples)

    budgets = torch.arange(1, max_budget + 1)
    utilities = torch.zeros(max_budget)

    for i, budget in enumerate(budgets):
        utilities[i] = expected_utility_at_budget(
            scores, labels, budget.item(), benefit, cost
        )

    return budgets, utilities


def threshold_for_coverage(
    scores: torch.Tensor,
    target_coverage: float,
) -> torch.Tensor:
    """Find the threshold that achieves a target coverage.

    Args:
        scores: Calibrated scores
        target_coverage: Desired fraction of items to include (0, 1]

    Returns:
        Threshold value
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)

    target_coverage = max(0.0, min(1.0, target_coverage))

    n_samples = len(scores)
    k = max(1, int(n_samples * target_coverage))

    sorted_scores, _ = torch.sort(scores, descending=True)

    if k >= n_samples:
        return torch.tensor(0.0)

    # Threshold is between the k-th and (k+1)-th highest scores
    return sorted_scores[k - 1]
