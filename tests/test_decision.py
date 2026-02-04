"""Tests for decision analysis tools."""

import matplotlib
import torch

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from rankcal import (
    budget_constrained_selection,
    expected_utility_at_budget,
    optimal_threshold,
    plot_risk_coverage,
    plot_utility_curve,
    risk_coverage_curve,
    threshold_for_coverage,
    utility_budget_curve,
    utility_curve,
)


class TestRiskCoverageCurve:
    def test_returns_correct_shapes(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        coverage, risk = risk_coverage_curve(scores, labels, n_thresholds=50)
        assert coverage.shape == (50,)
        assert risk.shape == (50,)

    def test_coverage_decreases_with_threshold(self):
        """Higher thresholds should give lower coverage."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        coverage, _ = risk_coverage_curve(scores, labels)
        # Coverage should generally decrease (not strictly due to binning)
        assert coverage[0] >= coverage[-1]

    def test_coverage_range(self):
        """Coverage should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        coverage, _ = risk_coverage_curve(scores, labels)
        assert (coverage >= 0).all()
        assert (coverage <= 1).all()

    def test_risk_range(self):
        """Risk should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        _, risk = risk_coverage_curve(scores, labels)
        assert (risk >= 0).all()
        assert (risk <= 1).all()


class TestUtilityCurve:
    def test_returns_correct_shapes(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        thresholds, utility = utility_curve(scores, labels, n_thresholds=50)
        assert thresholds.shape == (50,)
        assert utility.shape == (50,)

    def test_utility_with_different_costs(self):
        """Higher cost should decrease utility."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        _, utility_low_cost = utility_curve(scores, labels, benefit=1, cost=0.1)
        _, utility_high_cost = utility_curve(scores, labels, benefit=1, cost=10)
        # Max utility should be lower with higher cost
        assert utility_low_cost.max() >= utility_high_cost.max()


class TestPlotFunctions:
    def test_plot_risk_coverage_returns_figure(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        fig = plot_risk_coverage(scores, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_utility_curve_returns_figure(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        fig = plot_utility_curve(scores, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestOptimalThreshold:
    def test_returns_threshold_and_utility(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        threshold, utility = optimal_threshold(scores, labels)
        assert 0 <= threshold <= 1
        assert isinstance(utility, torch.Tensor)

    def test_optimal_threshold_maximizes_utility(self):
        """The returned threshold should achieve maximum utility."""
        torch.manual_seed(42)
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()

        opt_thresh, opt_utility = optimal_threshold(scores, labels)

        # Check utility at a few other thresholds
        for other_thresh in [0.3, 0.5, 0.7]:
            predictions = (scores >= other_thresh).float()
            tp = (predictions * labels).sum()
            fp = (predictions * (1 - labels)).sum()
            other_utility = tp - fp
            assert opt_utility >= other_utility - 1e-6  # Allow small tolerance


class TestBudgetConstrainedSelection:
    def test_selects_correct_number(self):
        scores = torch.rand(100)
        mask = budget_constrained_selection(scores, budget=20)
        assert mask.sum() == 20

    def test_selects_top_scores(self):
        scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7])
        mask = budget_constrained_selection(scores, budget=3)
        # Should select indices 3, 4, 1 (scores 0.9, 0.7, 0.5)
        assert mask[3]  # 0.9
        assert mask[4]  # 0.7
        assert mask[1]  # 0.5
        assert not mask[0]  # 0.1
        assert not mask[2]  # 0.3

    def test_handles_budget_larger_than_n(self):
        scores = torch.rand(10)
        mask = budget_constrained_selection(scores, budget=100)
        assert mask.sum() == 10


class TestExpectedUtilityAtBudget:
    def test_utility_calculation(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])

        utility = expected_utility_at_budget(scores, labels, budget=3, benefit=1, cost=1)
        # Top 3: indices 0, 1, 2 -> labels 1, 1, 0
        # TP = 2, FP = 1 -> utility = 2 - 1 = 1
        assert utility == 1.0

    def test_utility_increases_with_benefit(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()

        util_1 = expected_utility_at_budget(scores, labels, budget=20, benefit=1, cost=1)
        util_2 = expected_utility_at_budget(scores, labels, budget=20, benefit=2, cost=1)

        assert util_2 >= util_1


class TestUtilityBudgetCurve:
    def test_returns_correct_shapes(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        budgets, utilities = utility_budget_curve(scores, labels, max_budget=50)
        assert len(budgets) == 50
        assert len(utilities) == 50

    def test_budgets_are_sequential(self):
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        budgets, _ = utility_budget_curve(scores, labels, max_budget=20)
        expected = torch.arange(1, 21)
        assert (budgets == expected).all()


class TestThresholdForCoverage:
    def test_achieves_target_coverage(self):
        torch.manual_seed(42)
        scores = torch.rand(1000)
        target = 0.3
        threshold = threshold_for_coverage(scores, target)

        actual_coverage = (scores >= threshold).float().mean()
        # Should be approximately equal (within 1/n tolerance)
        assert abs(actual_coverage - target) < 0.01

    def test_full_coverage_gives_zero_threshold(self):
        scores = torch.rand(100)
        threshold = threshold_for_coverage(scores, target_coverage=1.0)
        assert threshold == 0.0

    def test_handles_edge_cases(self):
        scores = torch.rand(100)
        # Very small coverage
        thresh_small = threshold_for_coverage(scores, target_coverage=0.01)
        assert 0 <= thresh_small <= 1

        # Coverage > 1 (clamped)
        thresh_over = threshold_for_coverage(scores, target_coverage=1.5)
        assert thresh_over == 0.0
