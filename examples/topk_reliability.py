"""Top-k reliability diagram example.

Demonstrates the core insight of rankcal: calibration at top-k matters more
than overall calibration for ranking decisions.
"""

import torch
import matplotlib.pyplot as plt

from rankcal import (
    IsotonicCalibrator,
    ece_at_k,
    generate_miscalibrated_data,
)
from rankcal.metrics.ece import calibration_error_per_bin


def plot_reliability(ax, scores, labels, k=None, title=""):
    """Plot a reliability diagram on the given axes.

    Args:
        ax: Matplotlib axes to plot on
        scores: Calibrated scores
        labels: Binary relevance labels
        k: If provided, only use top-k scores
        title: Plot title
    """
    if k is not None:
        k = min(k, len(scores))
        _, top_k_indices = torch.topk(scores, k)
        scores = scores[top_k_indices]
        labels = labels[top_k_indices]

    centers, accs, confs, counts = calibration_error_per_bin(scores, labels)
    mask = counts > 0

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")
    ax.bar(
        confs[mask].numpy(),
        accs[mask].numpy(),
        width=0.08,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def main():
    # Generate miscalibrated data
    print("Generating data...")
    scores, labels = generate_miscalibrated_data(n_samples=2000, temperature=2.5, seed=42)

    # Split into train/test
    n_train = 1000
    scores_train, labels_train = scores[:n_train], labels[:n_train]
    scores_test, labels_test = scores[n_train:], labels[n_train:]

    # Calibrate
    print("Calibrating...")
    calibrator = IsotonicCalibrator()
    calibrator.fit(scores_train, labels_train)
    calibrated_scores = calibrator(scores_test)

    # Compute ECE at different k values
    print("\nECE at different k values:")
    print("-" * 40)
    print(f"{'k':<10} {'Uncalibrated':>15} {'Calibrated':>15}")
    print("-" * 40)

    k_values = [10, 50, 100, 500, len(scores_test)]
    for k in k_values:
        k_label = k if k < len(scores_test) else "all"
        uncal_ece = ece_at_k(scores_test, labels_test, k=k)
        cal_ece = ece_at_k(calibrated_scores, labels_test, k=k)
        print(f"{k_label:<10} {uncal_ece:>15.4f} {cal_ece:>15.4f}")

    # Create reliability diagrams
    print("\nGenerating reliability diagrams...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_reliability(axes[0, 0], scores_test, labels_test, title="Uncalibrated (All)")
    plot_reliability(axes[0, 1], scores_test, labels_test, k=100, title="Uncalibrated (Top-100)")
    plot_reliability(axes[1, 0], calibrated_scores, labels_test, title="Calibrated (All)")
    plot_reliability(axes[1, 1], calibrated_scores, labels_test, k=100, title="Calibrated (Top-100)")

    plt.tight_layout()
    plt.savefig("topk_reliability.png", dpi=150)
    print("Saved: topk_reliability.png")
    plt.close()


if __name__ == "__main__":
    main()
