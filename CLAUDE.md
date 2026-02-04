# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**rankcal** is a PyTorch-first library for calibration and uncertainty quantification in ranking systems. It addresses a gap in existing calibration libraries which focus on classification—rankcal calibrates ranking scores and evaluates calibration at top-k positions where decisions actually happen.

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run single test
pytest tests/test_calibrators.py::TestTemperatureScaling -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

## Project Structure

```
src/rankcal/
├── calibrators/      # Monotonic score transformers
│   ├── base.py       # BaseCalibrator ABC
│   ├── temperature.py
│   ├── isotonic.py
│   ├── spline.py
│   └── monotonic_nn.py
├── metrics/          # Calibration and ranking metrics
│   ├── ece.py        # ECE, ECE@k
│   ├── ranking.py    # precision@k, calibration gap
│   └── reliability.py
├── decision/         # Decision analysis tools
│   ├── curves.py     # risk-coverage, utility curves
│   └── optimize.py   # threshold/budget optimization
└── utils/
    └── data.py       # Synthetic data generation
```

## Key Design Decisions

- **PyTorch-first**: All calibrators are `nn.Module` subclasses for GPU support and differentiability
- **Python 3.9+**: Uses `from __future__ import annotations` and `typing.Optional`/`typing.Tuple` for backwards compatibility
- **Monotonicity**: All calibrators preserve score ordering (monotonic transformations)

## Testing

71 tests covering calibrators, metrics, and decision tools. Tests use `matplotlib.use("Agg")` for headless environments.
