"""Reliability diagram visualization."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from .ece import calibration_error_per_bin


def reliability_diagram(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: Optional[int] = None,
    n_bins: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Create a reliability diagram.

    Shows predicted confidence vs actual accuracy per bin, with a perfect
    calibration line for reference.

    Args:
        scores: Predicted scores in (0, 1), shape (n_samples,)
        labels: Binary labels, shape (n_samples,)
        k: If provided, only use top-k items by score
        n_bins: Number of bins for bucketing
        title: Optional plot title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    scores = scores.flatten()
    labels = labels.flatten()

    # Filter to top-k if specified
    if k is not None:
        k = min(k, len(scores))
        _, top_k_indices = torch.topk(scores, k)
        scores = scores[top_k_indices]
        labels = labels[top_k_indices]

    # Get per-bin statistics
    bin_centers, bin_accuracies, bin_confidences, bin_counts = calibration_error_per_bin(
        scores, labels, n_bins
    )

    # Convert to numpy for plotting
    bin_centers_np = bin_centers.numpy()
    bin_accuracies_np = bin_accuracies.numpy()
    bin_confidences_np = bin_confidences.numpy()
    bin_counts_np = bin_counts.numpy()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Main reliability diagram
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)

    # Bar chart showing accuracy per bin
    bar_width = 1.0 / n_bins * 0.8
    mask = bin_counts_np > 0
    ax1.bar(
        bin_confidences_np[mask],
        bin_accuracies_np[mask],
        width=bar_width,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Model",
    )

    # Gap visualization (calibration error)
    for i in range(n_bins):
        if bin_counts_np[i] > 0:
            conf = bin_confidences_np[i]
            acc = bin_accuracies_np[i]
            if acc > conf:
                ax1.bar(
                    conf,
                    acc - conf,
                    bottom=conf,
                    width=bar_width,
                    alpha=0.3,
                    color="green",
                )
            else:
                ax1.bar(
                    conf,
                    conf - acc,
                    bottom=acc,
                    width=bar_width,
                    alpha=0.3,
                    color="red",
                )

    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")

    if title is None:
        if k is not None:
            title = f"Reliability Diagram (Top-{k})"
        else:
            title = "Reliability Diagram"
    ax1.set_title(title)

    # Histogram of predictions
    ax2.bar(
        bin_centers_np,
        bin_counts_np / bin_counts_np.sum(),
        width=bar_width,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Samples")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    return fig
