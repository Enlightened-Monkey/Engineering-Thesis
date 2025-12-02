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

from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse


@dataclass(frozen=True)
class AppleScenario:
    """Two-option delayed-reward choice used in the apple experiment."""

    name: str
    option_a_desc: str
    option_b_desc: str
    option_a_reward: float
    option_a_delay: int
    option_b_reward: float
    option_b_delay: int


def exponential_value(reward: float, delay: int, gamma: float) -> float:
    """Return the exponentially discounted value of a reward."""

    if delay < 0:
        raise ValueError("Delay must be non-negative")
    return reward if delay == 0 else reward * (gamma ** delay)


def quasi_hyperbolic_value(reward: float, delay: int, sigma: float, gamma: float) -> float:
    """Return the quasi-hyperbolic value using the thesis convention."""

    if delay < 0:
        raise ValueError("Delay must be non-negative")
    if delay == 0:
        return reward
    return reward * sigma * (gamma ** delay)


def evaluate_scenario(scenario: AppleScenario, sigma: float, gamma: float) -> Dict[str, Dict[str, float]]:
    """Compute values and decisions for both discounting schemes."""

    exp_a = exponential_value(scenario.option_a_reward, scenario.option_a_delay, gamma)
    exp_b = exponential_value(scenario.option_b_reward, scenario.option_b_delay, gamma)
    qh_a = quasi_hyperbolic_value(scenario.option_a_reward, scenario.option_a_delay, sigma, gamma)
    qh_b = quasi_hyperbolic_value(scenario.option_b_reward, scenario.option_b_delay, sigma, gamma)

    def decide(value_a: float, value_b: float) -> str:
        if abs(value_a - value_b) < 1e-9:
            return "indifferent"
        return "option A" if value_a > value_b else "option B"

    return {
        "scenario": scenario.name,
        "exponential": {
            "option_a": exp_a,
            "option_b": exp_b,
            "decision": decide(exp_a, exp_b),
        },
        "quasi_hyperbolic": {
            "option_a": qh_a,
            "option_b": qh_b,
            "decision": decide(qh_a, qh_b),
        },
        "descriptions": {
            "option_a": scenario.option_a_desc,
            "option_b": scenario.option_b_desc,
        },
    }


def pretty_print(results: List[Dict[str, Dict[str, float]]]) -> None:
    """Display tabular output for the evaluated scenarios."""

    print("\n" + "=" * 72)
    print("APPLE CHOICE COMPARISON")
    print("=" * 72)

    for res in results:
        print(f"\nScenario: {res['scenario']}")
        print(f"  Option A: {res['descriptions']['option_a']}")
        print(f"  Option B: {res['descriptions']['option_b']}")

        for scheme_name in ("exponential", "quasi_hyperbolic"):
            scheme = res[scheme_name]
            label = "Exponential" if scheme_name == "exponential" else "Quasi-hyperbolic"
            print(
                f"    {label:<18} V(A)={scheme['option_a']:.4f}, "
                f"V(B)={scheme['option_b']:.4f}  ->  decision: {scheme['decision']}"
            )

    print("\nTime-consistency summary:")
    exp_decisions = [res["exponential"]["decision"] for res in results]
    qh_decisions = [res["quasi_hyperbolic"]["decision"] for res in results]

    def summarize(name: str, decisions: List[str]) -> None:
        consistent = "indifferent" in decisions or len(set(decisions)) == 1
        status = "CONSISTENT" if consistent else "INCONSISTENT"
        decision_path = " → ".join(decisions)
        print(f"  - {name:<18}: {status:>11} (decisions: {decision_path})")

    summarize("Exponential agent", exp_decisions)
    summarize("QH agent", qh_decisions)
    print("\n")


def parse_args() -> Tuple[float, float]:
    parser = argparse.ArgumentParser(
        description="Compare exponential vs quasi-hyperbolic decisions in the apple experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sigma", type=float, default=0.45,
                        help="Present-bias parameter (σ)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Exponential discount factor (γ)")
    args = parser.parse_args()

    if not 0.0 <= args.sigma <= 1.0:
        raise ValueError("sigma must be in [0, 1]")
    if not 0.0 <= args.gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")

    return args.sigma, args.gamma


def main() -> None:
    sigma, gamma = parse_args()

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

    results = [evaluate_scenario(s, sigma, gamma) for s in scenarios]
    pretty_print(results)


if __name__ == "__main__":
    main()
