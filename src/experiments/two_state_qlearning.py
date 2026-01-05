"""Two-state counter-example using QHQLearning (Algorithm 2).

Trains a quasi-hyperbolic Q-learning agent on the Fig. 1b MDP and reports
estimated Q-values for comparison with Table 1c (alpha=0.5, beta=0.8).
"""

from __future__ import annotations

import numpy as np

from src.algorithms.qh_qlearning import QHQLearning, train_qh_qlearning_sweep


def make_sampler(rng: np.random.Generator):
    """Generative model for the two-state counterexample.

    Returns a callable sampler(state, action) -> (next_state, reward, done).
    """

    def sampler(state: int, action: int):
        state = int(state)
        action = int(action)

        # State 0: a0 -> (state 1, reward 0); a1 -> reward 2, 50/50 stay/go to 1
        if state == 0:
            if action == 0:
                next_state, reward = 1, 0.0
            else:
                next_state = 0 if rng.random() < 0.5 else 1
                reward = 2.0
        else:
            # State 1: both actions are equivalent here; return to state 0 with reward 17
            next_state, reward = 0, 17.0

        done = False
        return int(next_state), float(reward), bool(done)

    return sampler


def train_agent(
    steps: int = 20_000_000,
    *,
    n_sweeps: int | None = None,
    seed: int = 123,
) -> QHQLearning:
    alpha = 0.5
    beta = 0.8
    rng = np.random.default_rng(int(seed))

    agent = QHQLearning(
        n_states=2,
        n_actions=2,
        alpha=alpha,
        beta=beta,
        theta_step=0.2,
        eta_step=0.4,
        theta_power=0.8,
        eta_power=0.6,
    )

    # Sweep driver (Algorithm 2 pseudocode): one (eta_n, theta_n) per sweep iteration.
    # Keep `steps` for backward compatibility with the notebook: convert to sweeps.
    updates_per_sweep = agent.n_states * agent.n_actions
    inferred_sweeps = max(1, int(steps) // int(updates_per_sweep))
    sweeps = int(inferred_sweeps if n_sweeps is None else n_sweeps)

    sampler = make_sampler(rng)
    train_qh_qlearning_sweep(sampler=sampler, agent=agent, n_iterations=sweeps)

    return agent


def main() -> None:
    agent = train_agent()

    targets = {
        (0, 0): 18.88,
        (0, 1): 19.00,
        (1, 0): 32.11,
    }

    print("--- QH Q-learning on two-state MDP (alpha=0.5, beta=0.8) ---")

    print("Q-table (rows=state, cols=action):")
    for s in range(agent.n_states):
        print(f"s={s}: {agent.Q[s, 0]:8.3f} {agent.Q[s, 1]:8.3f}")

    print("\nW-table (rows=state, cols=action):")
    for s in range(agent.n_states):
        print(f"s={s}: {agent.W[s, 0]:8.3f} {agent.W[s, 1]:8.3f}")

    policy_q = agent.get_policy()
    policy_w = np.argmax(agent.W, axis=1)

    print("\nGreedy policy (first-step, argmax_a Q):")
    print(policy_q)

    print("\nGreedy continuation policy (argmax_a W):")
    print(policy_w)

    print("\nComparison to Table 1c targets (f/g/h share same numeric targets per (s,a)):")
    for (s, a), tgt in targets.items():
        est = agent.Q[s, a]
        print(f"(s={s}, a={a}) -> est={est:7.3f} | target≈{tgt}")

    return policy_q, policy_w


if __name__ == "__main__":
    main()
