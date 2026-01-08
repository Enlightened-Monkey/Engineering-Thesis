# Examples Directory

This directory contains standalone example scripts that demonstrate the usage of the QH algorithms and provide comparison tests.

## Contents

### Test Scripts

- **`test_comparison.py`** - Comparison test demonstrating global vs local count-based step sizes
  - Shows the improvement achieved by using local (per state-action) visit counters
  - Standalone demo script, not a unit test
  - Usage: `python examples/test_comparison.py`

- **`test_convergence.py`** - Convergence test for QH Q-Learning
  - Verifies that local count-based step sizes improve convergence
  - Tests rarely-visited state-action pairs in the Inventory MDP
  - Usage: `python examples/test_convergence.py`

### Experiment Scripts

- **`run_experiments_cuda.py`** - CUDA-enabled experiments with two-state MDP
  - Implements Q-learning and policy evaluation using PyTorch backend
  - Requires CUDA-capable GPU for acceleration
  - Usage: `python examples/run_experiments_cuda.py`

## Usage

These scripts are intended to be run from the repository root:

```bash
# Run comparison test
python examples/test_comparison.py

# Run convergence test
python examples/test_convergence.py

# Run CUDA experiments (requires GPU)
python examples/run_experiments_cuda.py
```

## Note

These are long-running demonstration scripts, not unit tests. They are excluded from pytest collection to keep test runs fast. For unit tests, see `src/tests/`.
