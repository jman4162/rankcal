"""Decision curves example.

Shows how to use risk-coverage and utility curves for threshold selection.
"""

import matplotlib.pyplot as plt

from rankcal import (
    IsotonicCalibrator,
    generate_miscalibrated_data,
    optimal_threshold,
    risk_coverage_curve,
    threshold_for_coverage,
    utility_budget_curve,
    utility_curve,
)


def main():
    # Generate data
    print("Generating data...")
    scores, labels = generate_miscalibrated_data(n_samples=2000, temperature=2.0, seed=42)

    # Split and calibrate
    n_train = 1000
    scores_train, labels_train = scores[:n_train], labels[:n_train]
    scores_test, labels_test = scores[n_train:], labels[n_train:]

    calibrator = IsotonicCalibrator()
    calibrator.fit(scores_train, labels_train)
    calibrated = calibrator(scores_test)

    # Find optimal threshold for different cost ratios
    print("\nOptimal thresholds for different benefit/cost ratios:")
    print("-" * 60)
    print(f"{'Benefit':>10} {'Cost':>10} {'Threshold':>12} {'Utility':>12}")
    print("-" * 60)

    cost_ratios = [(1, 1), (1, 2), (1, 5), (2, 1), (5, 1)]
    for benefit, cost in cost_ratios:
        thresh, util = optimal_threshold(calibrated, labels_test, benefit=benefit, cost=cost)
        print(f"{benefit:>10} {cost:>10} {thresh:>12.3f} {util:>12.1f}")

    # Coverage-based threshold selection
    print("\nThreshold for target coverage:")
    print("-" * 40)
    print(f"{'Target Coverage':>15} {'Threshold':>12}")
    print("-" * 40)

    for target in [0.1, 0.2, 0.5, 0.8]:
        thresh = threshold_for_coverage(calibrated, target)
        actual_coverage = (calibrated >= thresh).float().mean()
        print(f"{target:>15.0%} {thresh:>12.3f} (actual: {actual_coverage:.1%})")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Risk-coverage curve
    ax = axes[0, 0]
    coverage, risk = risk_coverage_curve(calibrated, labels_test)
    ax.plot(coverage.numpy(), risk.numpy(), "b-", linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title("Risk-Coverage Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Utility curves for different costs using the utility_curve function
    ax = axes[0, 1]
    colors = {"1:1": "blue", "1:2": "orange", "1:5": "red"}
    for (benefit, cost), color in zip([(1, 1), (1, 2), (1, 5)], colors.values()):
        thresholds, utilities = utility_curve(
            calibrated, labels_test, benefit=benefit, cost=cost
        )
        ax.plot(
            thresholds.numpy(),
            utilities.numpy(),
            color=color,
            label=f"b={benefit}, c={cost}",
            linewidth=2,
        )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Utility")
    ax.set_title("Utility Curves (Different Cost Ratios)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Utility vs budget
    ax = axes[1, 0]
    budgets, utilities = utility_budget_curve(calibrated, labels_test, max_budget=500)
    ax.plot(budgets.numpy(), utilities.numpy(), "b-", linewidth=2)
    ax.set_xlabel("Budget (# items reviewed)")
    ax.set_ylabel("Utility")
    ax.set_title("Utility vs Budget")
    ax.grid(True, alpha=0.3)

    # Score distributions
    ax = axes[1, 1]
    pos_scores = calibrated[labels_test == 1]
    neg_scores = calibrated[labels_test == 0]
    ax.hist(
        neg_scores.numpy(),
        bins=30,
        alpha=0.5,
        label="Negative",
        color="red",
        density=True,
    )
    ax.hist(
        pos_scores.numpy(),
        bins=30,
        alpha=0.5,
        label="Positive",
        color="green",
        density=True,
    )
    ax.set_xlabel("Calibrated Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions by Class")
    ax.legend()

    plt.tight_layout()
    plt.savefig("decision_curves.png", dpi=150)
    print("\nSaved: decision_curves.png")
    plt.close()


if __name__ == "__main__":
    main()
