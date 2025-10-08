# Engineering Thesis Repository

## Algorytmy oparte o wzmocnione nauczanie maszynowe w dyskontowanych modelach markowskich

**Title (English):** Algorithms based on reinforcement learning in discounted Markovian models

**Author:** [Author Name]  
**Institution:** [Institution Name]  
**Year:** 2024

### Abstract

This repository contains the complete implementation and documentation for an engineering thesis on reinforcement learning algorithms with quasi-hyperbolic discounting in Markov Decision Processes (MDPs). The research focuses on time-inconsistent decision makers (precommitted agents) and includes a practical application to inventory management models.

**Keywords:** Reinforcement Learning, Quasi-Hyperbolic Discounting, Markov Decision Processes, Time Inconsistency, Policy Evaluation, Inventory Control

**AMS Classification:** 90C39, 90C40, 90B05, 93E03, 93E35

---

## Repository Structure

```
├── docs/                          # Documentation and thesis
│   ├── thesis/                    # Main thesis document
│   │   ├── chapters/              # Thesis chapters (LaTeX)
│   │   ├── figures/               # Thesis figures
│   │   ├── bibliography/          # BibTeX references
│   │   └── main.tex               # Main LaTeX document
│   ├── literature/                # Literature reviews and summaries
│   └── presentation/              # Defense presentation materials
├── src/                           # Source code
│   ├── algorithms/                # Core QH algorithms
│   │   ├── qh_qlearning.py       # QH Q-Learning implementation
│   │   └── qh_policy_evaluation.py # Policy evaluation algorithm
│   ├── models/                    # MDP environment models
│   │   └── mdp_environments.py    # Inventory and GridWorld MDPs
│   ├── experiments/               # Experimental framework
│   │   ├── experiment_runner.py   # Main experiment runner
│   │   └── comparison_standard_vs_qh.py # Standard vs QH comparison
│   ├── utils/                     # Utility functions
│   │   └── analysis_tools.py      # Analysis and visualization tools
│   └── tests/                     # Unit tests
├── notebooks/                     # Jupyter notebooks
│   └── standard_vs_qh_comparison.ipynb # Interactive comparison analysis
├── data/                          # Data directory
│   ├── datasets/                  # Input datasets
│   ├── results/                   # Experimental results
│   └── plots/                     # Generated figures
├── references/                    # Reference materials
│   ├── papers/                    # Research papers (PDFs)
│   └── books/                     # Books and monographs
└── README.md                      # This file
```

## Research Overview

### Problem Statement

Traditional reinforcement learning assumes exponential discounting and time-consistent preferences. However, real-world decision-making often involves:

- **Time inconsistency**: Preferences between future rewards change over time
- **Present bias**: Overweighting immediate rewards relative to future ones
- **Bounded rationality**: Systematic deviations from perfectly rational behavior

This thesis investigates **quasi-hyperbolic (QH) discounting** as an alternative to exponential discounting, providing more realistic models of decision-making behavior.

### Key Contributions

1. **Theoretical Framework**: Extension of MDP theory to quasi-hyperbolic discounting
2. **Algorithm Development**: 
   - Model-free policy evaluation using two-timescale stochastic approximation
   - QH Q-Learning algorithm with convergence guarantees
3. **Practical Application**: Inventory management model demonstrating time-inconsistent optimal policies
4. **Empirical Validation**: Comprehensive experimental evaluation and comparison with traditional methods

### Quasi-Hyperbolic Discounting Model

The cumulative discounted return follows:
```
G = r₀ + σ∑(γᵗrₜ) for t=1 to ∞
```

Where:
- `σ ∈ [0,1]`: Present-bias parameter
- `γ ∈ [0,1)`: Standard exponential discount factor
- `r₀`: Immediate reward
- `rₜ`: Reward at time t

When `σ = 1`, this reduces to standard exponential discounting. When `σ < 1`, the agent exhibits present bias.

## Getting Started

### Prerequisites

```bash
# Python 3.8+ required
python -m pip install numpy scipy matplotlib pandas seaborn
```

### Running Experiments

#### Quick Start: Standard vs QH Comparison

```bash
# Run the comparison script
cd src/experiments
python comparison_standard_vs_qh.py
```

Or use the interactive Jupyter notebook:
```bash
jupyter notebook notebooks/standard_vs_qh_comparison.ipynb
```

#### Programmatic Usage

```python
from src.experiments.comparison_standard_vs_qh import MDPComparison
from src.models.mdp_environments import InventoryMDP

# Create environment
env = InventoryMDP(max_inventory=15, max_order=8)

# Initialize comparison
comparison = MDPComparison(
    env=env,
    sigma=0.7,   # Present-bias parameter
    gamma=0.95,  # Discount factor
    alpha=0.1,   # Learning rate
    epsilon=0.1  # Exploration rate
)

# Train both standard and QH algorithms
comparison.train(n_episodes=5000, record_interval=100)

# Generate report
print(comparison.generate_report())

# Analyze policies and values
policy_comp = comparison.compare_policies()
value_comp = comparison.compare_values()

# Check time-consistency
consistency = comparison.analyze_time_consistency(initial_state=5, horizon=10)

# Visualize results
comparison.plot_comparison(save_path='results.png')
```

### Building the Thesis

```bash
cd docs/thesis/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Research Background

### Literature

The research builds upon key works:

- **Bertsekas (2019)**: "Reinforcement learning and optimal control" - Theoretical foundations
- **Jaśkiewicz & Nowak (2021)**: "Markov decision processes with quasi-hyperbolic discounting" - Core theoretical framework  
- **Eshwar et al. (2024)**: "Reinforcement learning with quasi-hyperbolic discounting" - Algorithmic innovations

### Applications

The thesis demonstrates practical applications in:
- **Inventory Management**: Optimal ordering policies under time-inconsistent preferences
- **Behavioral Economics**: Modeling bounded rationality in sequential decision-making
- **Financial Planning**: Long-term investment strategies with present bias

## Key Results

### Theoretical Results

1. **Policy Structure**: Optimal policies for precommitted agents have a specific two-component structure
2. **Convergence**: Model-free algorithms converge to optimal policies under standard conditions
3. **Time Inconsistency**: QH discounting leads to time-inconsistent optimal policies

### Experimental Findings

- QH algorithms successfully learn optimal policies in various environments
- Present-bias parameter significantly affects optimal behavior
- Traditional exponential discounting is a special case (σ = 1)

### Comparison: Standard vs Quasi-Hyperbolic Discounting

The framework provides comprehensive comparison between:

**Standard Q-Learning** (Exponential Discounting):
- Value function: $V(s) = E[\sum_{t=0}^{\infty} \gamma^t r_t]$
- Time-consistent preferences
- Optimal from long-term perspective

**QH Q-Learning** (Quasi-Hyperbolic Discounting):
- Value function: $V(s) = E[r_0 + \sigma \sum_{t=1}^{\infty} \gamma^t r_t]$
- Present-bias when σ < 1
- Potentially time-inconsistent
- Better models human behavior

**Key Observations:**
- Policy agreement decreases as σ decreases (stronger present-bias)
- QH discounting leads to more myopic (short-term focused) decisions
- Value function differences are significant in states requiring long-term planning
- Time-inconsistency manifests when precommitted and myopic choices differ

## Available Documents

- **[Literature Summary](./docs/literature/1285_MS_Summary.md)** - Comprehensive summary (in English and Polish) of the research paper "Teaching Precommitted Agents: Model-Free Policy Evaluation and Control in Quasi-Hyperbolic Discounted MDPs" by S.R. Eshwar
- **[Reference Paper](./references/papers/1285_MS.pdf)** - Original research paper on quasi-hyperbolic discounting in reinforcement learning

## Future Work

- Extension to partially observable MDPs (POMDPs)
- Multi-agent systems with time-inconsistent preferences
- Continuous state/action spaces
- Empirical validation with human subjects
- Real-world applications in robotics and finance

## Contributing

This repository supports academic research. For questions or collaborations:

1. Check the documentation in `docs/`
2. Review existing experiments in `src/experiments/`
3. Run unit tests: `python -m pytest src/tests/`

## License

Academic use only. Please cite this work if you use any part of the code or methodology.

## Citation

```bibtex
@mastersthesis{author2024qh,
  title={Algorytmy oparte o wzmocnione nauczanie maszynowe w dyskontowanych modelach markowskich},
  author={[Author Name]},
  school={[Institution Name]},
  year={2024},
  type={Engineering Thesis},
  keywords={Reinforcement Learning, Quasi-Hyperbolic Discounting, Markov Decision Processes}
}
```

## Contact

**Author:** [Author Name]  
**Email:** [author@institution.edu]  
**Institution:** [Institution Name]

---

*This repository contains all materials for the engineering thesis on quasi-hyperbolic discounting in reinforcement learning, including theoretical analysis, algorithmic implementations, and experimental validation.*