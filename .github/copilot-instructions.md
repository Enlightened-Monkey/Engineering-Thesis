# Copilot Instructions

## Project Overview

This repository contains the implementation for an engineering thesis on reinforcement learning algorithms with quasi-hyperbolic discounting in Markov Decision Processes (MDPs). The project focuses on time-inconsistent decision makers (precommitted agents) and includes practical applications to inventory management models.

## Repository Structure

- `src/`: Source code
  - `algorithms/`: Core QH algorithms (Q-Learning, Policy Evaluation)
  - `models/`: MDP environment models (Inventory, GridWorld)
  - `experiments/`: Experimental framework
  - `utils/`: Utility functions for analysis and visualization
  - `tests/`: Unit tests (pytest)
- `docs/`: Documentation and thesis (LaTeX)
- `notebooks/`: Jupyter notebooks for interactive analysis
- `data/`: Datasets, results, and generated plots
- `references/`: Research papers and reference materials

## Technology Stack

- **Language**: Python 3.8+
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Machine Learning**: scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx, LaTeX
- **Large Files**: Git LFS (for PDFs)

## Code Style and Conventions

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable names that reflect mathematical notation where appropriate (e.g., `sigma` for σ, `gamma` for γ)
- Include docstrings for all public functions and classes using NumPy-style documentation
- Use type hints for function parameters and return values
- Keep mathematical formulas documented in comments where they inform the implementation

## Development Guidelines

### Running Tests

```bash
# Run all tests
python -m pytest src/tests/

# Run with coverage
python -m pytest src/tests/ --cov=src
```

### Running Experiments

```bash
# Quick start comparison
cd src/experiments
python comparison_standard_vs_qh.py

# Use main.py for full experiments
python main.py --experiment comparison --env inventory --runs 5
```

### Key Algorithms

1. **QH Q-Learning** (`src/algorithms/qh_qlearning.py`): Q-Learning with quasi-hyperbolic discounting
2. **QH Policy Evaluation** (`src/algorithms/qh_policy_evaluation.py`): Two-timescale stochastic approximation for policy evaluation

### Mathematical Notation

- `σ (sigma)`: Present-bias parameter, range [0, 1]
- `γ (gamma)`: Discount factor, range [0, 1)
- `α (alpha)`: Learning rate
- `ε (epsilon)`: Exploration rate for ε-greedy policies

## Important Considerations

- When modifying algorithms, ensure convergence properties are maintained
- Use appropriate default values: σ=0.8, γ=0.95, α=0.1, ε=0.1
- Maintain backward compatibility with existing experiment configurations
- Large binary files (PDFs, datasets) should use Git LFS

## Building Documentation

```bash
# Build thesis PDF
cd docs/thesis/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
