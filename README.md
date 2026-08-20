# Reinforcement Learning with Quasi-Hyperbolic Discounting in Markov Decision Processes

**Engineering thesis — Wrocław University of Science and Technology, Faculty of Mathematics (2024)**

Full thesis: [`thesis.pdf`](thesis.pdf)

## About

This repository contains the code and experiments for my engineering thesis on
reinforcement learning algorithms with **quasi-hyperbolic discounting** in Markov
Decision Processes (MDPs). Classic RL assumes exponential discounting and
time-consistent preferences; this work extends it to **present-biased,
time-inconsistent decision makers** (precommitted agents), where the discounted
return is

$$G = r_0 + \alpha \sum_{t=1}^{\infty} \beta^t r_t, \qquad \alpha \in [0,1],\ \beta \in [0,1).$$

When $\alpha = 1$ this reduces to standard exponential discounting; when
$\alpha < 1$ the agent exhibits present bias.

## Practical applications

- **Inventory management** — a stock-replenishment MDP showing how present bias
  systematically changes optimal ordering policies.
- **Cart-pole balancing** — a discretized, physics-based control task solved
  with the QH agents.

## Code layout

- `src/algorithms/` — QH Q-Learning and model-free QH policy evaluation
  (two-timescale stochastic approximation)
- `src/models/` — MDP environments (inventory, grid world, pole balancing)
- `src/experiments/` — experiment scripts
- `src/tests/` — unit tests (pytest)
- `examples/`, `scripts/`, `main.py` — runnable examples and entry points

## Quick start

```bash
pip install -r requirements.txt
python -m src.experiments.InventoryMDP   # inventory experiment
python -m pytest src/tests -q            # run unit tests
```
