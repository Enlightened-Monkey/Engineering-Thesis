# Quick Start Guide: Standard vs QH Discounting Comparison

## 5-Minute Quick Start

### Option 1: Run the Demo
```bash
cd src/experiments
python demo_comparison.py --demo basic
```

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook notebooks/standard_vs_qh_comparison.ipynb
```

### Option 3: Python Script
```python
from src.experiments.comparison_standard_vs_qh import MDPComparison
from src.models.mdp_environments import InventoryMDP

# Create and run comparison
env = InventoryMDP(max_inventory=10, max_order=5)
comparison = MDPComparison(env=env, sigma=0.7, gamma=0.95)
comparison.train(n_episodes=2000)
print(comparison.generate_report())
```

## What to Expect

### Standard Q-Learning
- Uses exponential discounting: V(s) = E[Σ γ^t r_t]
- Time-consistent preferences
- Optimal for long-term planning

### QH Q-Learning  
- Uses quasi-hyperbolic discounting: V(s) = E[r_0 + σ Σ γ^t r_t]
- Present-bias when σ < 1
- May be time-inconsistent
- Better models human behavior

## Key Parameters

- **σ (sigma)**: Present-bias parameter
  - σ = 1.0: No present-bias (same as standard)
  - σ = 0.7: Moderate present-bias
  - σ = 0.5: Strong present-bias
  
- **γ (gamma)**: Discount factor (typically 0.95)

- **n_episodes**: Training episodes (2000-5000 recommended)

## Understanding the Output

### Policy Agreement
- **100%**: Policies are identical
- **80-99%**: Minor differences
- **<80%**: Significant divergence

### Value Differences
- Shows how differently the approaches value each state
- Larger differences indicate stronger impact of present-bias

### Time-Consistency
- **Consistent**: Agent doesn't want to deviate from plan
- **Inconsistent**: Agent would prefer different actions later
- Common when σ < 1

## Example Output

```
======================================================================
COMPARISON: Standard vs Quasi-Hyperbolic Discounting
======================================================================

Environment: InventoryMDP
States: 11, Actions: 6
Gamma (discount): 0.95
Sigma (present-bias): 0.7
Episodes trained: 2000

POLICY COMPARISON
----------------------------------------------------------------------
Agreement: 72.7%
Different states: 3
States where policies differ: [3 5 8]

VALUE FUNCTION COMPARISON
----------------------------------------------------------------------
Mean absolute difference: 1.2345
Max absolute difference: 3.4567

PERFORMANCE COMPARISON
----------------------------------------------------------------------
Final episode reward (Standard): 45.23
Final episode reward (QH): 43.67
Difference: 1.56

======================================================================
```

## Next Steps

1. **Experiment with different σ values**: See how present-bias affects behavior
2. **Try different environments**: Test on various MDP structures  
3. **Analyze time-consistency**: Study when policies become inconsistent
4. **Read full documentation**: See `docs/COMPARISON_GUIDE.md`

## Troubleshooting

**Issue:** Import errors
**Solution:** Make sure you're in the correct directory and have installed dependencies

**Issue:** No convergence
**Solution:** Try more episodes or adjust learning rate (alpha)

**Issue:** All policies identical
**Solution:** Try lower σ value (stronger present-bias)

## Getting Help

- Full documentation: `docs/COMPARISON_GUIDE.md`
- Example notebook: `notebooks/standard_vs_qh_comparison.ipynb`
- Test script: `src/experiments/test_comparison.py`
