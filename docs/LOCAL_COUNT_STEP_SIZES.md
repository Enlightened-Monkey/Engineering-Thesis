# Visit Counters and Step-Size Schedules in QH Q-Learning

## Current Status

This document clarifies the current implementation in src/algorithms/qh_qlearning.py.

The code currently uses:

1. Local visit counters per (state, action) for diagnostics and persistence.
2. Global Robbins-Monro schedules for eta_n and theta_n, driven by a global
     iteration counter.

In other words, visit counts are recorded, but do not directly drive the
learning-rate denominator.

## Historical Context

Earlier notes around this repository described a local-count step-size variant
for rarely visited state-action pairs. Those notes are kept for context, but
the active implementation and tests now align with a global schedule design.

## Why this matters

Documentation and code must be interpreted together:

- src/algorithms/qh_qlearning.py states that _next_step_sizes uses a global
    iteration index.
- src/tests/test_local_counters.py verifies the same behavior.

If you are analyzing convergence behavior, treat "local counters" as coverage
statistics, not as the source of step-size decay in the active code.

## Reference Behavior (current)

Conceptually, the implementation behaves like:

```python
self._visit_counts[state, action] += 1
self._iteration += 1
t = self._iteration
eta_n = eta_step / (offset + t) ** eta_power
theta_n = theta_step / (offset + t) ** theta_power
```

## Notes on experiments

Historical experiment summaries may still use wording such as
"local count-based step sizes". In the current codebase, interpret this label
as historical naming, unless explicitly stated otherwise in the script.

## Legacy Results (archival)

The remainder of this file preserves earlier benchmark notes for reference.

---

The original QH Q-Learning implementation suffered from convergence issues in rarely-visited state-action pairs. Specifically, in the Inventory Control experiment:

- **State 2, Action 1**: Error > 10.0 (Analytical: 15.56 vs RL: 3.50 or 19.20)
- **State 2, Action 2**: Error > 8.0 (Analytical: 10.56 vs RL: 1.50 or 19.00)

These pairs were visited 1000x less frequently than optimal path pairs (e.g., State 0, Action 1).

### Root Cause

The algorithm used a **global iteration counter** to calculate learning rate decay:

```python
# OLD APPROACH
step_size = initial_step / (global_iteration ** power)
```

This meant:
1. Global learning rate decayed based on TOTAL iterations
2. Rarely-visited pairs only got a few updates before the learning rate became negligible
3. By the time these pairs were visited (e.g., 100 times), the global counter might be at 500,000+
4. Learning rate would be `1.0 / (500,000 ** 0.6) ≈ 0.0000016` - too small to learn anything

## Solution: Local Count-Based Step Sizes

Implement **per-(s,a) visit counting** where each state-action pair maintains its own counter:

```python
# NEW APPROACH
step_size = initial_step / (local_visit_count[s,a] ** power)
```

This ensures:
1. Each (s,a) pair has an independent learning rate
2. Rarely-visited pairs get full learning opportunities when visited
3. Frequently-visited pairs still converge properly with their own decay schedule
4. Theoretical soundness: Robbins-Monro conditions satisfied independently per pair

## Implementation Changes

### 1. Added Visit Count Matrix

```python
# In __init__()
self._visit_counts = np.zeros((n_states, n_actions), dtype=np.int64)
```

### 2. Modified Step Size Calculation

```python
# Updated _next_step_sizes() to accept state and action
def _next_step_sizes(self, state: int, action: int) -> tuple[float, float]:
    self._visit_counts[state, action] += 1
    n_visits = self._visit_counts[state, action]
    denom = 10.0 + n_visits  # Reduced offset for faster learning
    eta_n = self.eta_step / (denom ** self.eta_power)
    theta_n = self.theta_step / (denom ** self.theta_power)
    return eta_n, theta_n
```

### 3. Updated State Persistence

Added `visit_counts` to save/load methods with backward compatibility for old checkpoints.

## Results

### Convergence Improvement

| State-Action Pair | Original Error | New Error | Improvement |
|-------------------|----------------|-----------|-------------|
| State 2, Action 1 | >10.0          | <2.0      | >80%        |
| State 2, Action 2 | >8.0           | <1.0      | >87%        |
| State 0, Action 1 | ~0.1           | ~0.1      | Maintained  |

### Visit Count Distribution

Example from 2M episode training:

| State | Action | Visits    | Error  | Status |
|-------|--------|-----------|--------|--------|
| 0     | 0      | 13,500    | 0.46   | ✓ OK   |
| 0     | 1      | 1,567,000 | 0.42   | ✓ OK   |
| 0     | 2      | 14,300    | 0.47   | ✓ OK   |
| 1     | 0      | 392,000   | 0.48   | ✓ OK   |
| 1     | 1      | 4,000     | 0.70   | ✓ OK   |
| 1     | 2      | 3,500     | 0.15   | ✓ OK   |
| 2     | 0      | 5,200     | 0.97   | ✓ OK   |
| 2     | 1      | **150**   | **<2.0** | ✓ Fixed |
| 2     | 2      | **160**   | **<1.0** | ✓ Fixed |

Notice how State 2, Actions 1-2 only received ~150 visits but now achieve reasonable convergence.

## Theoretical Justification

### Stochastic Approximation Theory

For Q-learning with step sizes α_n to converge, we need Robbins-Monro conditions:

1. Σ α_n = ∞ (infinite total learning)
2. Σ α_n² < ∞ (square-summable for variance control)

**Global counting breaks condition 1 for rarely-visited pairs:**
- If (s,a) visited k times out of N total iterations
- Step sizes: α₁, α₂, ..., α_k where α_i = C/(i_global)^p
- If k << N, then α_k ≈ C/N^p → 0 very quickly
- Total learning: Σα_i ≈ k·C/N^p might not be infinite relative to the noise

**Local counting satisfies both conditions:**
- Step sizes: α₁, α₂, ..., α_k where α_i = C/i^p
- Σα_i = C·Σ(1/i^p) = ∞ for p ∈ (0.5, 1]
- Σα_i² = C²·Σ(1/i^(2p)) < ∞ for p > 0.5

### Two-Timescale Separation

The algorithm maintains two sequences (θ_n for Q, η_n for W) where θ_n/η_n → 0.

With local counting, this is satisfied **per (s,a) pair**:
- θ_n(s,a) = θ_step / (10 + n_visits(s,a))^0.6
- η_n(s,a) = η_step / (10 + n_visits(s,a))^0.55

As n_visits → ∞: θ_n/η_n = (θ_step/η_step) · (10+n)^(0.55-0.6) → 0

## Usage

No API changes required! The algorithm automatically uses local counters:

```python
agent = QHQLearning(
    n_states=3,
    n_actions=3,
    alpha=0.3,      # Present-bias parameter
    beta=0.9,       # Exponential discount
    theta_step=1.0, # Initial slow step size
    eta_step=2.0,   # Initial fast step size
    theta_power=0.6,
    eta_power=0.55,
    epsilon=0.2
)

# Training loop - no changes needed
for episode in range(episodes):
    state = env.reset()
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done=done)
        state = next_state
```

## Testing

Run convergence tests:

```bash
# Unit tests for local counter functionality
python -m pytest src/tests/test_local_counters.py -v

# Integration test showing improvement
python examples/test_comparison.py
```

## References

1. Robbins, H., & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*.
2. Eshwar, S. et al. (2024). Reinforcement learning with quasi-hyperbolic discounting.
3. Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint*.

## Future Work

Potential improvements:
1. Adaptive offset based on initialization values
2. Different power schedules for different state regions
3. Automatic hyperparameter tuning based on visit statistics
