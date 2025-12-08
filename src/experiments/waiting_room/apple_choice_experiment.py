#!/usr/bin/env python3
"""Apple choice thought experiment for discounting schemes.

This script reproduces the well-known behavioural economics question:
"Do you want one apple today or two tomorrow?" and compares it with the
"one apple in 50 days vs two apples in 51 days" variant.  The goal is to
illustrate how exponential discounting remains time-consistent, whereas a
quasi-hyperbolic (present-biased) agent may change its preference once both
rewards lie in the future.

Run directly:
    python -m src.experiments.apple_choice_experiment --sigma 0.45 --gamma 0.95
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.algorithms.qh_qlearning import QHQLearning, train_qh_qlearning
from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation

TransitionSampler = Callable[[int, int], Tuple[int, float]]


@dataclass(frozen=True)
class AppleScenario:
    name: str
    option_a_desc: str
    option_b_desc: str
    option_a_reward: float
    option_a_delay: int
    option_b_reward: float
    option_b_delay: int


class AppleChoiceEnv:
    ACTION_A = 0
    ACTION_B = 1

    def __init__(self, scenario: AppleScenario):
        self.scenario = scenario
        self.initial_state = 0
        self._state_roles: Dict[int, Tuple[str, int | None]] = {0: ("initial", None)}
        self._a_states: Dict[int, int] = {}
        self._b_states: Dict[int, int] = {}

        next_state = 1
        for remaining in range(scenario.option_a_delay, 0, -1):
            self._a_states[remaining] = next_state
            self._state_roles[next_state] = ("A", remaining)
            next_state += 1
        for remaining in range(scenario.option_b_delay, 0, -1):
            self._b_states[remaining] = next_state
            self._state_roles[next_state] = ("B", remaining)
            next_state += 1

        self.terminal_state = next_state
        self._state_roles[self.terminal_state] = ("terminal", None)
        self.n_states = self.terminal_state + 1
        self.n_actions = 2
        self.state = self.initial_state

    def reset(self) -> int:
        self.state = self.initial_state
        return self.state

    def transition(self, state: int, action: int) -> Tuple[int, float]:
        role, meta = self._state_roles[state]
        if role == "initial":
            if action == self.ACTION_A:
                return self._advance_branch("A", self.scenario.option_a_reward, self.scenario.option_a_delay)
            return self._advance_branch("B", self.scenario.option_b_reward, self.scenario.option_b_delay)

        if role == "A":
            return self._advance_wait("A", meta)
        if role == "B":
            return self._advance_wait("B", meta)
        return self.terminal_state, 0.0

    def _advance_branch(self, branch: str, reward: float, delay: int) -> Tuple[int, float]:
        if delay == 0:
            return self.terminal_state, reward
        target = self._a_states[delay] if branch == "A" else self._b_states[delay]
        return target, 0.0

    def _advance_wait(self, branch: str, remaining: int | None) -> Tuple[int, float]:
        assert remaining is not None
        reward = self.scenario.option_a_reward if branch == "A" else self.scenario.option_b_reward
        states = self._a_states if branch == "A" else self._b_states
        if remaining == 1:
            return self.terminal_state, reward
        return states[remaining - 1], 0.0

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, float]]:
        next_state, reward = self.transition(self.state, action)
        self.state = next_state
        done = self.state == self.terminal_state
        return next_state, reward, done, {}


def _uniform_policy(n_states: int, n_actions: int) -> np.ndarray:
    return np.full((n_states, n_actions), 1.0 / n_actions)


def _deterministic_policy(n_states: int, n_actions: int, preferred_action: int) -> np.ndarray:
    policy = np.zeros((n_states, n_actions))
    policy[:, preferred_action] = 1.0
    return policy


def _run_policy_evaluation(
    scenario: AppleScenario,
    alpha: float,
    beta: float,
    preferred_action: int,
    n_iterations: int,
) -> Dict[str, np.ndarray]:
    env = AppleChoiceEnv(scenario)
    evaluator = QHPolicyEvaluation(env.n_states, alpha=alpha, beta=beta)
    sampler: TransitionSampler = lambda state, action: env.transition(state, action)
    sampling_policy = _uniform_policy(env.n_states, env.n_actions)
    deterministic_policy = _deterministic_policy(env.n_states, env.n_actions, preferred_action)

    return evaluator.evaluate_policy(
        sampler=sampler,
        sampling_policy=sampling_policy,
        mu_policy=deterministic_policy,
        phi_policy=deterministic_policy,
        n_iterations=n_iterations,
        initial_state=env.initial_state,
        terminal_function=lambda s: s == env.terminal_state,
        reset_fn=lambda: env.initial_state,
    )


def _train_agent(
    scenario: AppleScenario,
    alpha: float,
    beta: float,
    n_episodes: int,
) -> Tuple[QHQLearning, Dict[str, object]]:
    env = AppleChoiceEnv(scenario)
    agent = QHQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        beta=beta,
        theta_step=0.4,
        eta_step=0.4,
        epsilon=0.1,
    )
    stats = train_qh_qlearning(env, agent, n_episodes=n_episodes)
    return agent, stats


def evaluate_scenario(
    scenario: AppleScenario,
    sigma: float,
    gamma: float,
    n_episodes: int,
    n_eval_iterations: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run QH vs exponential (alpha=1) using RL algorithms."""

    comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
    variants = {
        "quasi_hyperbolic": min(max(sigma, 0.0), 1.0),
        "exponential": 1.0,
    }

    for label, alpha in variants.items():
        agent, stats = _train_agent(scenario, alpha, gamma, n_episodes)
        policy = stats["final_policy"]
        env = AppleChoiceEnv(scenario)
        best_action = int(policy[env.initial_state])
        policy_eval = _run_policy_evaluation(scenario, alpha, gamma, best_action, n_eval_iterations)

        comparisons[label] = {
            "alpha": alpha,
            "decision": "option A" if best_action == env.ACTION_A else "option B",
            "J0": float(policy_eval["J"][env.initial_state]),
            "W0": float(policy_eval["W"][env.initial_state]),
            "Q_values": {
                "option_a": float(agent.Q[env.initial_state, env.ACTION_A]),
                "option_b": float(agent.Q[env.initial_state, env.ACTION_B]),
            },
            "episodes": len(stats["episode_rewards"]),
            "final_episode_reward": float(stats["episode_rewards"][-1]) if stats["episode_rewards"] else 0.0,
        }

    return {
        "scenario": scenario.name,
        "descriptions": {
            "option_a": scenario.option_a_desc,
            "option_b": scenario.option_b_desc,
        },
        "schemes": comparisons,
    }


def pretty_print(results: List[Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    """Display tabular output for the evaluated scenarios."""

    print("\n" + "=" * 72)
    print("APPLE CHOICE COMPARISON")
    print("=" * 72)

    for res in results:
        print(f"\nScenario: {res['scenario']}")
        print(f"  Option A: {res['descriptions']['option_a']}")
        print(f"  Option B: {res['descriptions']['option_b']}")

        for scheme_name in ("exponential", "quasi_hyperbolic"):
            scheme = res["schemes"][scheme_name]
            label = "Exponential" if scheme_name == "exponential" else "Quasi-hyperbolic"
            q_vals = scheme["Q_values"]
            print(
                f"    {label:<18} (alpha={scheme['alpha']:.2f})\n"
                f"        Q(A)={q_vals['option_a']:.4f}, Q(B)={q_vals['option_b']:.4f}\n"
                f"        Policy eval: J0={scheme['J0']:.4f}, W0={scheme['W0']:.4f} -> decision: {scheme['decision']}"
            )

    print("\nTime-consistency summary:")
    exp_decisions = [res["schemes"]["exponential"]["decision"] for res in results]
    qh_decisions = [res["schemes"]["quasi_hyperbolic"]["decision"] for res in results]

    def summarize(name: str, decisions: List[str]) -> None:
        consistent = len(set(decisions)) == 1
        status = "CONSISTENT" if consistent else "INCONSISTENT"
        decision_path = " → ".join(decisions)
        print(f"  - {name:<18}: {status:>11} (decisions: {decision_path})")

    summarize("Exponential agent", exp_decisions)
    summarize("QH agent", qh_decisions)
    print("\n")


def parse_args() -> Tuple[float, float, int, int]:
    parser = argparse.ArgumentParser(
        description="Compare exponential vs quasi-hyperbolic decisions in the apple experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sigma", type=float, default=0.45,
                        help="Present-bias parameter (σ)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Exponential discount factor (γ)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes for Q-learning")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Stochastic approximation updates for policy evaluation")
    args = parser.parse_args()

    if not 0.0 <= args.sigma <= 1.0:
        raise ValueError("sigma must be in [0, 1]")
    if not 0.0 <= args.gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.eval_steps <= 0:
        raise ValueError("eval-steps must be positive")

    return args.sigma, args.gamma, args.episodes, args.eval_steps


def main() -> None:
    sigma, gamma, episodes, eval_steps = parse_args()

    scenarios = [
        AppleScenario(
            name="Today vs Tomorrow",
            option_a_desc="1 apple today",
            option_b_desc="2 apples tomorrow",
            option_a_reward=1.0,
            option_a_delay=0,
            option_b_reward=2.0,
            option_b_delay=1,
        ),
        AppleScenario(
            name="In 50 days vs 51 days",
            option_a_desc="1 apple in 50 days",
            option_b_desc="2 apples in 51 days",
            option_a_reward=1.0,
            option_a_delay=50,
            option_b_reward=2.0,
            option_b_delay=51,
        ),
    ]

    results = [evaluate_scenario(s, sigma, gamma, episodes, eval_steps) for s in scenarios]
    pretty_print(results)


if __name__ == "__main__":
    main()
