# Installation

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher

## Install from PyPI

```bash
pip install rankcal
```

## Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/jman4162/rankcal.git
cd rankcal
pip install -e ".[dev]"
```

## Optional Dependencies

### Development

Install development dependencies for testing and linting:

```bash
pip install -e ".[dev]"
```

This includes:

- pytest for testing
- ruff for linting
- mypy for type checking

### Documentation

To build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Verify Installation

```python
import rankcal
print(rankcal.__version__)
```

## GPU Support

rankcal automatically uses GPU when available. No additional installation is needed beyond having a CUDA-compatible PyTorch installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")  # Apple Silicon
```
