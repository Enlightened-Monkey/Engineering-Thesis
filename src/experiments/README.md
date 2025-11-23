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

### 4. Discounting Comparison Plots

**File:** `plot_discounting_comparison.py`

Generate side-by-side visualisations that contrast standard exponential discounting with quasi-hyperbolic preferences for multiple σ values. The plots highlight how present-bias increases the weight of early rewards.

**Usage:**
```bash
python -m src.experiments.plot_discounting_comparison --no-show --output data/plots/discounting_comparison.png
```

**Key flags:**

- `--sigma-values` – comma-separated σ values to compare (default `0.6,0.8,0.95`)
- `--gamma` – common exponential discount factor γ (default `0.95`)
- `--horizon` – number of time steps displayed (default `30`)
- `--no-show` – skip opening a GUI window (useful on remote servers)

The script saves the figure if `--output` is provided; directories are created automatically.

### 5. Pole Balancing Training

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

### 6. Pole Balancing Simulation (GIF)

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

### 7. Inventory Control Benchmark

**File:** `inventory_control_experiment.py`

Train a quasi-hyperbolic Q-learning agent on the finite-horizon inventory control problem described in the thesis (capacity ``M = 2`` with demand probabilities ``0.2, 0.3, 0.5``).

**Usage:**
```bash
python -m src.experiments.inventory_control_experiment --episodes 5000 --episode-length 30
```

**Key flags:**

- `--sigma` / `--gamma` – quasi-hyperbolic discount parameters (defaults ``0.3`` and ``0.9``).
- `--max-inventory`, `--procurement-cost`, `--holding-cost`, `--selling-price` – environment economics.
- `--demand-support`, `--demand-prob` – demand distribution (comma-separated lists).
- `--results-dir` – directory for JSON summaries (default `data/results`).
- `--no-save` – skip persisting the summary to disk.

The script reports average reward, order quantity, sales and ending inventory over an evaluation rollout.

### 8. Inventory Policy Evaluation Convergence

**File:** `policy_evaluation_convergence.py`

Recreates the convergence study from the thesis (Figure 1) for the inventory-control benchmark, comparing three policy pairs under the quasi-hyperbolic policy-evaluation algorithm.

**Usage:**
```bash
python -m src.experiments.policy_evaluation_convergence --iterations 200000 --eta 0.3 --theta 0.03
```

**Outputs:**

- ``data/plots/policy_evaluation_inventory_convergence.png`` – single-axis figure plotting $\lVert W_k - V^\beta_{\phi_s} \rVert_2$ on a logarithmic iteration scale for the three policy pairs $\big(\mu^*, \phi_s^*\big)$, $\big(\mu^*, \phi_s^u\big)$, and $\big(\mu^u, \phi_s^*\big)$.

Optional flags mirror the notation used in the thesis:
- `--sigma` / `--beta` – quasi-hyperbolic parameters ($\sigma = 0.3$, $\beta = 0.9$ by default),
- `--eta` / `--theta` – step-sizes for the fast and slow timescales,
- `--iterations` – total number of updates (200k recommended for smooth curves),
- `--seed` – RNG seed for reproducibility.

The script saves the figure automatically and reports its location on completion.

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
