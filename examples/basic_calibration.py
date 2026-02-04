"""Basic calibration example.

Shows how to calibrate miscalibrated ranking scores using different calibrators.
"""

import torch

from rankcal import (
    IsotonicCalibrator,
    SplineCalibrator,
    TemperatureScaling,
    ece,
    generate_miscalibrated_data,
)


def main():
    # Generate miscalibrated data (overconfident predictions)
    print("Generating miscalibrated data...")
    scores_train, labels_train = generate_miscalibrated_data(
        n_samples=1000, temperature=2.0, seed=42
    )
    scores_test, labels_test = generate_miscalibrated_data(
        n_samples=500, temperature=2.0, seed=123
    )

    print(f"Train samples: {len(scores_train)}")
    print(f"Test samples: {len(scores_test)}")
    print(f"Uncalibrated ECE: {ece(scores_test, labels_test):.4f}")
    print()

    # Temperature Scaling
    print("=" * 50)
    print("Temperature Scaling")
    print("=" * 50)
    temp_cal = TemperatureScaling()
    temp_cal.fit(scores_train, labels_train)
    temp_calibrated = temp_cal(scores_test)
    print(f"Learned temperature: {temp_cal.temperature.item():.4f}")
    print(f"Calibrated ECE: {ece(temp_calibrated, labels_test):.4f}")
    print()

    # Isotonic Calibration
    print("=" * 50)
    print("Isotonic Calibration")
    print("=" * 50)
    iso_cal = IsotonicCalibrator()
    iso_cal.fit(scores_train, labels_train)
    iso_calibrated = iso_cal(scores_test)
    print(f"Calibrated ECE: {ece(iso_calibrated, labels_test):.4f}")
    print()

    # Spline Calibration
    print("=" * 50)
    print("Spline Calibration")
    print("=" * 50)
    spline_cal = SplineCalibrator(n_knots=10)
    spline_cal.fit(scores_train, labels_train, max_iter=200)
    spline_calibrated = spline_cal(scores_test)
    print(f"Calibrated ECE: {ece(spline_calibrated, labels_test):.4f}")
    print()

    # Summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"{'Method':<20} {'ECE':>10}")
    print("-" * 30)
    print(f"{'Uncalibrated':<20} {ece(scores_test, labels_test):>10.4f}")
    print(f"{'Temperature':<20} {ece(temp_calibrated, labels_test):>10.4f}")
    print(f"{'Isotonic':<20} {ece(iso_calibrated, labels_test):>10.4f}")
    print(f"{'Spline':<20} {ece(spline_calibrated, labels_test):>10.4f}")


if __name__ == "__main__":
    main()
