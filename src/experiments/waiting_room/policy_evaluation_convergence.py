"""Inventory-control convergence study for QH policy evaluation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class InventoryParams:
    max_inventory: int = 2
    procurement_cost: float = 5.0
    holding_cost: float = 2.0
    selling_price: float = 9.0
    demand_support: Sequence[int] = (0, 1, 2)
    demand_prob: Sequence[float] = (0.2, 0.3, 0.5)


@dataclass
class ExperimentConfig:
    sigma: float = 0.3
    beta: float = 0.9
    eta: float = 0.3
    theta: float = 0.03
    iterations: int = 200_000
    seed: int = 42
    inventory: InventoryParams = field(default_factory=InventoryParams)
    results_dir: Path = Path("data/plots")
    filename: str = "policy_evaluation_inventory_convergence.png"


class InventoryDynamics:
    """Utility routines operating on the finite inventory-control MDP."""

    def __init__(self, params: InventoryParams):
        self.params = params
        self.max_inventory = params.max_inventory
        self.n_states = self.max_inventory + 1
        self.n_actions = self.max_inventory + 1
        self.support = np.asarray(params.demand_support, dtype=int)
        self.prob = np.asarray(params.demand_prob, dtype=float)
        self.prob = self.prob / self.prob.sum()

    def sample_transition(self, state: int, action: int, rng: np.random.Generator) -> Tuple[int, float]:
        inventory_post = min(state + action, self.max_inventory)
        demand = int(rng.choice(self.support, p=self.prob))
        sales = min(inventory_post, demand)
        next_state = inventory_post - sales
        reward = (
            self.params.selling_price * sales
            - self.params.procurement_cost * action
            - self.params.holding_cost * next_state
        )
        return next_state, float(reward)

    def transition_probabilities(self, state: int, action: int) -> np.ndarray:
        probs = np.zeros(self.n_states)
        inventory_post = min(state + action, self.max_inventory)
        for demand, p_d in zip(self.support, self.prob):
            sales = min(inventory_post, demand)
            next_state = inventory_post - sales
            probs[next_state] += p_d
        return probs

    def expected_reward(self, state: int, action: int) -> float:
        total = 0.0
        inventory_post = min(state + action, self.max_inventory)
        for demand, p_d in zip(self.support, self.prob):
            sales = min(inventory_post, demand)
            next_state = inventory_post - sales
            reward = (
                self.params.selling_price * sales
                - self.params.procurement_cost * action
                - self.params.holding_cost * next_state
            )
            total += p_d * reward
        return float(total)

    def true_exponential_value(self, policy: np.ndarray, beta: float) -> np.ndarray:
        transition = np.zeros((self.n_states, self.n_states))
        rewards = np.zeros(self.n_states)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_a = policy[s, a]
                if prob_a == 0.0:
                    continue
                transition[s] += prob_a * self.transition_probabilities(s, a)
            rewards[s] = sum(
                policy[s, a] * self.expected_reward(s, a)
                for a in range(self.n_actions)
            )

        identity = np.eye(self.n_states)
        return np.linalg.solve(identity - beta * transition, rewards)


def deterministic_policy(actions: Sequence[int], n_states: int, n_actions: int) -> np.ndarray:
    policy = np.zeros((n_states, n_actions))
    for s, a in enumerate(actions):
        policy[s, a] = 1.0
    return policy


def uniform_policy(n_states: int, n_actions: int) -> np.ndarray:
    policy = np.full((n_states, n_actions), 1.0 / n_actions)
    return policy


def ensure_support(policy: np.ndarray, base: np.ndarray) -> np.ndarray:
    mask = policy > 0
    adjusted = np.where(mask, policy, base)
    # Renormalise rows to keep valid distributions.
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def run_qh_policy_evaluation(
    dynamics: InventoryDynamics,
    cfg: ExperimentConfig,
    mu: np.ndarray,
    phi: np.ndarray,
    nu: np.ndarray,
    v_beta_true: np.ndarray,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n_states = dynamics.n_states
    n_actions = dynamics.n_actions

    W = np.zeros(n_states)
    V = np.zeros(n_states)

    diff_history = np.zeros(cfg.iterations)

    state = 0  # Start from empty inventory as in the paper.
    for t in range(cfg.iterations):
        action = int(rng.choice(n_actions, p=nu[state]))
        next_state, reward = dynamics.sample_transition(state, action, rng)

        follow_action = int(rng.choice(n_actions, p=phi[next_state]))
        _, reward_follow = dynamics.sample_transition(next_state, follow_action, rng)

        r_target = reward - (1.0 - cfg.sigma) * cfg.beta * reward_follow + cfg.beta * W[next_state]

        weight_phi = phi[state, action] / max(nu[state, action], 1e-12)
        weight_mu = mu[state, action] / max(nu[state, action], 1e-12)

        W[state] += cfg.eta * (weight_phi * r_target - W[state])
        V[state] += cfg.theta * (weight_mu * r_target - V[state])

        diff_history[t] = np.linalg.norm(W - v_beta_true)

        state = next_state

    return {
        "diff": diff_history,
        "W": W,
        "V": V,
    }


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Reproduce the inventory-control convergence figure (Fig. 1)."
    )
    parser.add_argument("--sigma", type=float, default=ExperimentConfig.sigma, help="Present-bias parameter σ")
    parser.add_argument("--beta", type=float, default=ExperimentConfig.beta, help="Exponential discount factor β")
    parser.add_argument("--eta", type=float, default=ExperimentConfig.eta, help="Fast timescale stepsize η_n")
    parser.add_argument("--theta", type=float, default=ExperimentConfig.theta, help="Slow timescale stepsize θ_n")
    parser.add_argument("--iterations", type=int, default=ExperimentConfig.iterations, help="Total update iterations")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--results-dir", type=Path, default=ExperimentConfig.results_dir)
    parser.add_argument("--filename", type=str, default=ExperimentConfig.filename)

    args = parser.parse_args()

    return ExperimentConfig(
        sigma=args.sigma,
        beta=args.beta,
        eta=args.eta,
        theta=args.theta,
        iterations=args.iterations,
        seed=args.seed,
        results_dir=args.results_dir,
        filename=args.filename,
    )


def plot_convergence(
    cfg: ExperimentConfig,
    iteration_axis: np.ndarray,
    curves: Iterable[Tuple[str, np.ndarray]],
) -> Path:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for label, diff in curves:
        ax.plot(iteration_axis, diff, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Iterations (log-scale)")
    ax.set_ylabel(r"$\|W_k - V^\beta_{\phi_s}\|_2$")
    ax.set_title("Convergence of Policy Evaluation in Inventory Control")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.results_dir / cfg.filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main() -> Path:  # pragma: no cover
    cfg = parse_args()
    dynamics = InventoryDynamics(cfg.inventory)

    n_states, n_actions = dynamics.n_states, dynamics.n_actions

    mu_star = deterministic_policy([1, 0, 0], n_states, n_actions)
    phi_star = deterministic_policy([2, 1, 0], n_states, n_actions)
    mu_uniform = uniform_policy(n_states, n_actions)
    phi_uniform = uniform_policy(n_states, n_actions)

    nu_uniform = uniform_policy(n_states, n_actions)

    pairs = [
        ("μ = (μ*, φ_s^*)", mu_star, phi_star, nu_uniform),
        ("μ = (μ*, φ_s^u)", mu_star, phi_uniform, nu_uniform),
        ("μ = (μ^u, φ_s^*)", mu_uniform, phi_star, nu_uniform),
    ]

    curves = []
    iteration_axis = np.arange(1, cfg.iterations + 1, dtype=float)

    for label, mu_policy, phi_policy, nu_policy in pairs:
        v_beta = dynamics.true_exponential_value(phi_policy, cfg.beta)
        # Ensure sampling policy covers evaluation policy support.
        nu_adjusted = ensure_support(nu_policy, phi_policy)

        result = run_qh_policy_evaluation(dynamics, cfg, mu_policy, phi_policy, nu_adjusted, v_beta)
        curves.append((label, result["diff"]))

    output_path = plot_convergence(cfg, iteration_axis, curves)
    print(f"Saved convergence plot to {output_path}")
    return output_path


if __name__ == "__main__":  # pragma: no cover
    main()
