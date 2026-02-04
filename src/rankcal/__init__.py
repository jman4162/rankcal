"""rankcal: Calibration and uncertainty quantification for ranking systems.

A PyTorch-first library for calibrating ranking scores and evaluating
calibration at top-k positions where decisions actually happen.
"""

__version__ = "0.1.0"

# Calibrators
from .calibrators import (
    BaseCalibrator,
    IsotonicCalibrator,
    MonotonicNNCalibrator,
    SplineCalibrator,
    TemperatureScaling,
)

# Decision tools
from .decision import (
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

# Metrics
from .metrics import (
    calibration_error_per_bin,
    calibration_gap_at_k,
    ece,
    ece_at_k,
    expected_precision_at_k,
    mean_actual_relevance,
    mean_predicted_relevance,
    precision_at_k,
    reliability_diagram,
)

# Utils
from .utils import generate_calibrated_data, generate_miscalibrated_data

__all__ = [
    # Version
    "__version__",
    # Calibrators
    "BaseCalibrator",
    "TemperatureScaling",
    "IsotonicCalibrator",
    "SplineCalibrator",
    "MonotonicNNCalibrator",
    # Metrics
    "ece",
    "ece_at_k",
    "calibration_error_per_bin",
    "reliability_diagram",
    "precision_at_k",
    "expected_precision_at_k",
    "calibration_gap_at_k",
    "mean_predicted_relevance",
    "mean_actual_relevance",
    # Decision
    "risk_coverage_curve",
    "utility_curve",
    "plot_risk_coverage",
    "plot_utility_curve",
    "optimal_threshold",
    "budget_constrained_selection",
    "expected_utility_at_budget",
    "utility_budget_curve",
    "threshold_for_coverage",
    # Utils
    "generate_calibrated_data",
    "generate_miscalibrated_data",
]
