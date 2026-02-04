"""Risk-coverage and utility curves for decision analysis."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def risk_coverage_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute risk-coverage curve.

    Shows the trade-off between coverage (fraction of items above threshold)
    and risk (error rate on items above threshold).

    Args:
        scores: Calibrated scores, shape (n_samples,)
        labels: Binary relevance labels, shape (n_samples,)
        n_thresholds: Number of threshold points

    Returns:
        Tuple of (coverage, risk) tensors, each shape (n_thresholds,)
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    thresholds = torch.linspace(0, 1, n_thresholds)
    coverage = torch.zeros(n_thresholds)
    risk = torch.zeros(n_thresholds)

    n_samples = len(scores)

    for i, thresh in enumerate(thresholds):
        mask = scores >= thresh
        n_above = mask.sum()

        coverage[i] = n_above / n_samples

        if n_above > 0:
            # Risk = 1 - accuracy = error rate
            risk[i] = 1 - labels[mask].mean()
        else:
            risk[i] = 0  # No samples, no risk

    return coverage, risk


def utility_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
    benefit: float = 1.0,
    cost: float = 1.0,
    n_thresholds: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute utility curve across thresholds.

    Utility = benefit * true_positives - cost * false_positives

    Args:
        scores: Calibrated scores, shape (n_samples,)
        labels: Binary relevance labels, shape (n_samples,)
        benefit: Benefit of a true positive
        cost: Cost of a false positive
        n_thresholds: Number of threshold points

    Returns:
        Tuple of (thresholds, utility) tensors
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    thresholds = torch.linspace(0, 1, n_thresholds)
    utility = torch.zeros(n_thresholds)

    for i, thresh in enumerate(thresholds):
        predictions = (scores >= thresh).float()
        true_positives = (predictions * labels).sum()
        false_positives = (predictions * (1 - labels)).sum()
        utility[i] = benefit * true_positives - cost * false_positives

    return thresholds, utility


def plot_risk_coverage(
    scores: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 100,
    title: str = "Risk-Coverage Curve",
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Plot risk-coverage curve.

    Args:
        scores: Calibrated scores
        labels: Binary relevance labels
        n_thresholds: Number of threshold points
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    coverage, risk = risk_coverage_curve(scores, labels, n_thresholds)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(coverage.numpy(), risk.numpy(), "b-", linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add AUC annotation
    auc = torch.trapezoid(risk, coverage).abs()
    ax.annotate(f"AUC: {auc:.3f}", xy=(0.7, 0.9), fontsize=12)

    return fig


def plot_utility_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
    benefit: float = 1.0,
    cost: float = 1.0,
    n_thresholds: int = 100,
    title: str = "Utility Curve",
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Plot utility curve.

    Args:
        scores: Calibrated scores
        labels: Binary relevance labels
        benefit: Benefit of a true positive
        cost: Cost of a false positive
        n_thresholds: Number of threshold points
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    thresholds, utility = utility_curve(scores, labels, benefit, cost, n_thresholds)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds.numpy(), utility.numpy(), "b-", linewidth=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Utility")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Mark optimal threshold
    opt_idx = utility.argmax()
    opt_thresh = thresholds[opt_idx]
    opt_utility = utility[opt_idx]
    ax.axvline(x=opt_thresh.item(), color="r", linestyle="--", alpha=0.7)
    ax.plot(opt_thresh.item(), opt_utility.item(), "ro", markersize=10)
    ax.annotate(
        f"Optimal: {opt_thresh:.3f}\nUtility: {opt_utility:.1f}",
        xy=(opt_thresh.item(), opt_utility.item()),
        xytext=(opt_thresh.item() + 0.1, opt_utility.item()),
        fontsize=10,
    )

    return fig
