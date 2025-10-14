# Experiments Module

This module contains experimental frameworks for comparing and analyzing reinforcement learning algorithms with different discounting schemes.

## Available Experiments

### 1. Standard vs Quasi-Hyperbolic Comparison

**File:** `comparison_standard_vs_qh.py`

Complete framework for comparing standard exponential discounting with quasi-hyperbolic discounting.

**Features:**
- Parallel training of both algorithms
- Policy and value function comparison
- Time-consistency analysis
- Comprehensive visualization
- Automated report generation

**Quick Start:**
```bash
python comparison_standard_vs_qh.py
```

### 2. Quick Test

**File:** `test_comparison.py`

Fast verification test to ensure the comparison framework works correctly.

**Usage:**
```bash
python test_comparison.py
```

### 3. Demo Scripts

**File:** `demo_comparison.py`

Interactive demonstrations showing key differences between discounting approaches.

**Available Demos:**
```bash
# Basic comparison
python demo_comparison.py --demo basic

# Sensitivity analysis for σ parameter
python demo_comparison.py --demo sigma

# Create visualizations
python demo_comparison.py --demo viz

# Run all demos
python demo_comparison.py --demo all
```

### 4. Pole Balancing Training

**File:** `pole_balancing_training.py`

Train a quasi-hyperbolic Q-learning agent on the physics-based pole balancing environment described in the thesis.

**Usage (recommended):**
```bash
python -m src.experiments.pole_balancing_training --episodes 100000 --eval-episodes 25 --seed 123
```

The script now saves a compressed agent snapshot (`.npz`) alongside a JSON summary (both timestamped). Key parameters:

**Key flags:**

- `--sigma` – present-bias parameter σ (default 0.7)
- `--gamma` – exponential discount γ (default 0.97)
- `--alpha` – learning rate
- `--epsilon` / `--min-epsilon` / `--epsilon-decay` – exploration schedule
- `--seed` – seed for reproducibility
- `--results-dir` – directory for JSON summaries (default `data/results`)
- `--model-dir` – directory for agent weight snapshots (default `data/models`)
- `--no-save` – skip JSON summary storage
- `--no-save-model` – skip saving the trained agent snapshot

JSON logs contain per-episode rewards (sizeable for 100k runs); adjust `--episodes` if disk space is a concern.

### 5. Pole Balancing Simulation (GIF)

**File:** `pole_balancing_simulation.py`

Load a trained agent snapshot and produce a Matplotlib GIF (defaults to a ~10s clip at environment fidelity):

```bash
python -m src.experiments.pole_balancing_simulation \
    --model data/models/pole_balancing_100k/pole_balancing_qh_<timestamp>.npz \
    --output data/plots/pole_balancing_100k.gif \
    --duration 10
```

Additional options:

- `--fps` – override frame rate (defaults to environment step frequency)
- `--seed` – reproducible reset for the rollout
- `--env-json` – optional path to a training summary JSON if metadata is unavailable
- `--title` – customise the animation title

## Jupyter Notebooks

For interactive exploration, see:
- `../notebooks/standard_vs_qh_comparison.ipynb`

## Documentation

For detailed documentation, see:
- `../docs/COMPARISON_GUIDE.md`
- Pole balancing environment details are available via inline docstrings in `models.mdp_environments.PoleBalancingMDP`.

## Example Usage

```python
from comparison_standard_vs_qh import MDPComparison
from models.mdp_environments import InventoryMDP, PoleBalancingMDP

# Create environment
env = InventoryMDP(max_inventory=15, max_order=8)
# For pole balancing instead, use:
# env = PoleBalancingMDP()

# Setup comparison
comparison = MDPComparison(
    env=env,
    sigma=0.7,   # Present-bias parameter
    gamma=0.95,  # Discount factor
    alpha=0.1,   # Learning rate
    epsilon=0.1  # Exploration rate
)

# Train both algorithms
comparison.train(n_episodes=5000, record_interval=100)

# Analyze results
print(comparison.generate_report())
comparison.plot_comparison(save_path='results.png')

# Check policies
policy_comp = comparison.compare_policies()
print(f"Agreement: {policy_comp['agreement_percentage']:.1f}%")

# Check time-consistency
consistency = comparison.analyze_time_consistency(
    initial_state=5, 
    horizon=10
)
print(f"Time-consistent: {consistency['is_time_consistent']}")
```

## Key Results to Analyze

1. **Policy Differences:** Where do the policies differ and why?
2. **Value Functions:** How do value estimates compare?
3. **Time-Consistency:** Is the QH policy time-consistent?
4. **Present-Bias Impact:** How does σ affect behavior?

## Output

Results are saved to:
- `../../data/results/` - Numerical results
- `../../data/plots/` - Visualizations

## Dependencies

```python
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
```
