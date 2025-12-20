"""Two-state counter-example using QHQLearning (Algorithm 2).

Trains a quasi-hyperbolic Q-learning agent on the Fig. 1b MDP and reports
estimated Q-values for comparison with Table 1c (alpha=0.5, beta=0.8).
"""

from __future__ import annotations

import numpy as np

from src.algorithms.qh_qlearning import QHQLearning


class TwoStateEnv:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.state = 0

    def reset(self) -> int:
        self.state = int(self.rng.integers(0, 2))
        return self.state

    def step(self, action: int):
        # State 0: a0 -> (state 1, reward 0); a1 -> reward 2, 50/50 stay/go to 1
        if self.state == 0:
            if action == 0:
                next_state, reward = 1, 0.0
            else:
                next_state = 0 if self.rng.random() < 0.5 else 1
                reward = 2.0
        else:
            # State 1: only meaningful action is a0; return to state 0 with reward 17
            next_state, reward = 0, 17.0

        self.state = next_state
        return next_state, reward, False, {}


def train_agent(steps: int = 300_000) -> QHQLearning:
    alpha = 0.5
    beta = 0.8
    rng = np.random.default_rng(123)
    env = TwoStateEnv(rng)

    agent = QHQLearning(
        n_states=2,
        n_actions=2,
        alpha=alpha,
        beta=beta,
        theta_step=0.2,
        eta_step=0.4,
        theta_power=0.8,
        eta_power=0.6,
        epsilon=0.2,
    )

    state = env.reset()
    for t in range(steps):
        action = agent.get_action(state, exploration=True)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done=done)
        state = next_state

        if t % 5000 == 0 and agent.epsilon > 0.01:
            agent.epsilon *= 0.98

    return agent


def main() -> None:
    agent = train_agent()

    targets = {
        (0, 0): 18.88,
        (0, 1): 19.00,
        (1, 0): 32.11,
    }

    print("--- QH Q-learning on two-state MDP (alpha=0.5, beta=0.8) ---")
    print(f"epsilon final: {agent.epsilon:.4f}")
    print("Q-table (rows=state, cols=action):")
    for s in range(agent.n_states):
        print(f"s={s}: {agent.Q[s, 0]:8.3f} {agent.Q[s, 1]:8.3f}")

    print("\nGreedy policy (argmax_a Q):")
    print(agent.get_policy())

    print("\nComparison to Table 1c targets (f/g/h share same numeric targets per (s,a)):")
    for (s, a), tgt in targets.items():
        est = agent.Q[s, a]
        print(f"(s={s}, a={a}) -> est={est:7.3f} | target≈{tgt}")


if __name__ == "__main__":
    main()
