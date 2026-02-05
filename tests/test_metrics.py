"""Tests for metrics."""

import matplotlib
import torch

matplotlib.use("Agg")  # Use non-interactive backend

from rankcal import (
    adaptive_ece,
    adaptive_ece_at_k,
    calibration_error_per_bin,
    calibration_gap_at_k,
    ece,
    ece_at_k,
    expected_precision_at_k,
    mce,
    mce_at_k,
    mean_actual_relevance,
    mean_predicted_relevance,
    precision_at_k,
    reliability_diagram,
)
from rankcal.utils import generate_calibrated_data


class TestECE:
    def test_perfect_calibration_has_low_ece(self):
        """ECE should be near zero for perfectly calibrated data."""
        torch.manual_seed(42)
        # Large sample for stable estimate
        scores, labels = generate_calibrated_data(10000, seed=42)
        ece_val = ece(scores, labels)
        # With large sample, ECE should be small (< 0.05)
        assert ece_val < 0.05

    def test_ece_range(self):
        """ECE should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        ece_val = ece(scores, labels)
        assert 0 <= ece_val <= 1

    def test_overconfident_has_positive_ece(self):
        """Overconfident predictions should have positive ECE."""
        # All predictions are 0.9 but only 50% are relevant
        scores = torch.full((100,), 0.9)
        labels = torch.cat([torch.ones(50), torch.zeros(50)])
        ece_val = ece(scores, labels)
        assert ece_val > 0.3  # Should be around 0.4

    def test_ece_with_different_n_bins(self):
        """ECE should work with different bin counts."""
        scores, labels = generate_calibrated_data(100, seed=42)
        ece_5 = ece(scores, labels, n_bins=5)
        ece_20 = ece(scores, labels, n_bins=20)
        # Both should be valid
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_20 <= 1


class TestECEAtK:
    def test_ece_at_k_with_small_k(self):
        """ECE@k should work with small k."""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ece_val = ece_at_k(scores, labels, k=3)
        assert 0 <= ece_val <= 1

    def test_ece_at_k_equals_ece_when_k_equals_n(self):
        """ECE@k should equal ECE when k equals number of samples."""
        scores, labels = generate_calibrated_data(100, seed=42)
        ece_full = ece(scores, labels)
        ece_k = ece_at_k(scores, labels, k=100)
        assert torch.isclose(ece_full, ece_k, atol=1e-6)

    def test_ece_at_k_handles_k_larger_than_n(self):
        """Should handle k > n gracefully."""
        scores = torch.rand(10)
        labels = torch.randint(0, 2, (10,)).float()
        ece_val = ece_at_k(scores, labels, k=100)
        assert 0 <= ece_val <= 1


class TestCalibrationErrorPerBin:
    def test_returns_correct_shapes(self):
        scores, labels = generate_calibrated_data(100, seed=42)
        centers, accs, confs, counts = calibration_error_per_bin(scores, labels, n_bins=10)
        assert centers.shape == (10,)
        assert accs.shape == (10,)
        assert confs.shape == (10,)
        assert counts.shape == (10,)

    def test_counts_sum_to_n(self):
        scores, labels = generate_calibrated_data(100, seed=42)
        _, _, _, counts = calibration_error_per_bin(scores, labels, n_bins=10)
        assert counts.sum() == 100


class TestReliabilityDiagram:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        scores, labels = generate_calibrated_data(100, seed=42)
        fig = reliability_diagram(scores, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_k_parameter(self):
        import matplotlib.pyplot as plt

        scores, labels = generate_calibrated_data(100, seed=42)
        fig = reliability_diagram(scores, labels, k=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAdaptiveECE:
    def test_adaptive_ece_range(self):
        """Adaptive ECE should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        ece_val = adaptive_ece(scores, labels)
        assert 0 <= ece_val <= 1

    def test_adaptive_ece_with_calibrated_data(self):
        """Adaptive ECE should be small for well-calibrated data."""
        torch.manual_seed(42)
        scores, labels = generate_calibrated_data(10000, seed=42)
        ece_val = adaptive_ece(scores, labels)
        assert ece_val < 0.1

    def test_adaptive_ece_handles_skewed_distribution(self):
        """Adaptive ECE should handle non-uniform score distributions."""
        # Skewed distribution: most scores near 0.9
        scores = torch.clamp(torch.randn(100) * 0.1 + 0.9, 0, 1)
        labels = torch.randint(0, 2, (100,)).float()
        ece_val = adaptive_ece(scores, labels)
        assert 0 <= ece_val <= 1

    def test_adaptive_ece_identical_scores(self):
        """Adaptive ECE should handle all identical scores."""
        scores = torch.full((100,), 0.7)
        labels = torch.randint(0, 2, (100,)).float()
        ece_val = adaptive_ece(scores, labels)
        assert 0 <= ece_val <= 1

    def test_adaptive_ece_with_different_n_bins(self):
        """Adaptive ECE should work with different bin counts."""
        scores, labels = generate_calibrated_data(100, seed=42)
        ece_5 = adaptive_ece(scores, labels, n_bins=5)
        ece_20 = adaptive_ece(scores, labels, n_bins=20)
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_20 <= 1


class TestAdaptiveECEAtK:
    def test_adaptive_ece_at_k_range(self):
        """Adaptive ECE@k should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        ece_val = adaptive_ece_at_k(scores, labels, k=20)
        assert 0 <= ece_val <= 1

    def test_adaptive_ece_at_k_equals_adaptive_ece_when_k_equals_n(self):
        """Adaptive ECE@k should equal adaptive ECE when k equals n."""
        scores, labels = generate_calibrated_data(100, seed=42)
        ece_full = adaptive_ece(scores, labels)
        ece_k = adaptive_ece_at_k(scores, labels, k=100)
        assert torch.isclose(ece_full, ece_k, atol=1e-6)

    def test_adaptive_ece_at_k_handles_k_larger_than_n(self):
        """Should handle k > n gracefully."""
        scores = torch.rand(10)
        labels = torch.randint(0, 2, (10,)).float()
        ece_val = adaptive_ece_at_k(scores, labels, k=100)
        assert 0 <= ece_val <= 1


class TestMCE:
    def test_mce_range(self):
        """MCE should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        mce_val = mce(scores, labels)
        assert 0 <= mce_val <= 1

    def test_mce_greater_than_or_equal_to_ece(self):
        """MCE should be >= ECE (max >= weighted mean)."""
        torch.manual_seed(42)
        for _ in range(10):
            scores = torch.rand(100)
            labels = torch.randint(0, 2, (100,)).float()
            ece_val = ece(scores, labels)
            mce_val = mce(scores, labels)
            assert mce_val >= ece_val - 1e-6  # Small tolerance for floating point

    def test_mce_overconfident(self):
        """MCE should detect worst-case miscalibration."""
        # All predictions are 0.9 but only 50% are relevant
        scores = torch.full((100,), 0.9)
        labels = torch.cat([torch.ones(50), torch.zeros(50)])
        mce_val = mce(scores, labels)
        assert mce_val > 0.3  # Should be around 0.4

    def test_mce_with_different_n_bins(self):
        """MCE should work with different bin counts."""
        scores, labels = generate_calibrated_data(100, seed=42)
        mce_5 = mce(scores, labels, n_bins=5)
        mce_20 = mce(scores, labels, n_bins=20)
        assert 0 <= mce_5 <= 1
        assert 0 <= mce_20 <= 1

    def test_mce_perfect_calibration(self):
        """MCE should be small for well-calibrated data."""
        torch.manual_seed(42)
        scores, labels = generate_calibrated_data(10000, seed=42)
        mce_val = mce(scores, labels)
        # MCE can be higher than ECE even for calibrated data
        assert mce_val < 0.15


class TestMCEAtK:
    def test_mce_at_k_range(self):
        """MCE@k should be in [0, 1]."""
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        mce_val = mce_at_k(scores, labels, k=20)
        assert 0 <= mce_val <= 1

    def test_mce_at_k_equals_mce_when_k_equals_n(self):
        """MCE@k should equal MCE when k equals n."""
        scores, labels = generate_calibrated_data(100, seed=42)
        mce_full = mce(scores, labels)
        mce_k = mce_at_k(scores, labels, k=100)
        assert torch.isclose(mce_full, mce_k, atol=1e-6)

    def test_mce_at_k_handles_k_larger_than_n(self):
        """Should handle k > n gracefully."""
        scores = torch.rand(10)
        labels = torch.randint(0, 2, (10,)).float()
        mce_val = mce_at_k(scores, labels, k=100)
        assert 0 <= mce_val <= 1

    def test_mce_at_k_greater_than_or_equal_to_ece_at_k(self):
        """MCE@k should be >= ECE@k."""
        torch.manual_seed(42)
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,)).float()
        ece_val = ece_at_k(scores, labels, k=30)
        mce_val = mce_at_k(scores, labels, k=30)
        assert mce_val >= ece_val - 1e-6


class TestRankingMetrics:
    def test_precision_at_k(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        p_at_3 = precision_at_k(scores, labels, k=3)
        # Top 3 by score: indices 0, 1, 2 -> labels 1, 0, 1 -> precision = 2/3
        assert torch.isclose(p_at_3, torch.tensor(2 / 3))

    def test_expected_precision_at_k(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        ep_at_3 = expected_precision_at_k(scores, k=3)
        # Mean of top 3 scores: (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert torch.isclose(ep_at_3, torch.tensor(0.8))

    def test_calibration_gap_at_k(self):
        # Overconfident: high scores but low relevance
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        gap = calibration_gap_at_k(scores, labels, k=3)
        # Expected = 0.8, actual = 0 -> gap = 0.8
        assert gap > 0.7

    def test_mean_predicted_relevance(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        mpr = mean_predicted_relevance(scores)
        assert torch.isclose(mpr, scores.mean())

    def test_mean_predicted_relevance_at_k(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        mpr = mean_predicted_relevance(scores, k=2)
        assert torch.isclose(mpr, torch.tensor(0.85))

    def test_mean_actual_relevance(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        mar = mean_actual_relevance(scores, labels)
        assert torch.isclose(mar, torch.tensor(0.4))

    def test_mean_actual_relevance_at_k(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        mar = mean_actual_relevance(scores, labels, k=2)
        # Top 2 by score have labels [1, 1]
        assert torch.isclose(mar, torch.tensor(1.0))
