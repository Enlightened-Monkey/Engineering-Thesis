"""Reproduction of the two-state MDP counter-example (Fig. 1b / Table 1c).

Uses the shared QHPolicyEvaluation implementation from src.algorithms.
"""

from __future__ import annotations

from typing import Callable, Tuple

import time
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running as a plain script from the repo root, e.g.
#   python ./src/experiments/two_state_counterexample.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation

TransitionOutcome = Tuple[int, float, bool]
TransitionSampler = Callable[[int, int], TransitionOutcome]


def _progress_bar(prefix: str, current: int, total: int, *, start_time: float) -> None:
    width = 32
    frac = 0.0 if total <= 0 else min(1.0, max(0.0, current / total))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(1e-9, time.time() - start_time)
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else float("inf")
    if not np.isfinite(remaining):
        eta_str = "ETA: ?"
    else:
        eta_str = f"ETA: {remaining:,.0f}s"

    msg = f"{prefix} [{bar}] {frac * 100:6.2f}% ({current:,}/{total:,}) | {eta_str}"
    print("\r" + msg, end="", flush=True)


def _two_state_dynamics():
    """Return (P, R) arrays for the two-state MDP.

    P[s, a, s'] and R[s, a] = E[r | s, a].
    Matches create_simple_mdp().
    """

    n_states = 2
    n_actions = 2
    P = np.zeros((n_states, n_actions, n_states), dtype=float)
    R = np.zeros((n_states, n_actions), dtype=float)

    # state 0
    # a0 -> (1, 0)
    P[0, 0, 1] = 1.0
    R[0, 0] = 0.0
    # a1 -> reward 2, next 0/1 with 0.5/0.5
    P[0, 1, 0] = 0.5
    P[0, 1, 1] = 0.5
    R[0, 1] = 2.0

    # state 1: environment ignores action and always returns to 0 with reward 17
    for a in range(n_actions):
        P[1, a, 0] = 1.0
        R[1, a] = 17.0

    return P, R


def _induced_PR(policy: np.ndarray, P: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return induced (P_pi, R_pi) under a (state,action) policy."""

    P_pi = (policy[:, :, None] * P).sum(axis=1)
    R_pi = (policy * R).sum(axis=1)
    return P_pi, R_pi


def _true_W_for_policy(phi_s_policy: np.ndarray, *, beta: float) -> np.ndarray:
    P, R = _two_state_dynamics()
    P_pi, R_pi = _induced_PR(phi_s_policy, P, R)
    I = np.eye(P_pi.shape[0])
    return np.linalg.solve(I - beta * P_pi, R_pi)


def _sample_indices(n: int, *, n_points: int = 500) -> np.ndarray:
    if n <= 1:
        return np.array([0], dtype=int)
    idx = np.unique(
        np.concatenate(
            [
                np.array([0], dtype=int),
                np.logspace(0, np.log10(n - 1), n_points).astype(int),
                np.array([n - 1], dtype=int),
            ]
        )
    )
    return idx


def estimate_W_with_convergence(
    *,
    phi_s_policy: np.ndarray,
    env: TransitionSampler,
    nu_policy: np.ndarray,
    alpha: float,
    beta: float,
    n_sweeps: int,
    seed: int,
    label: str,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate W and return (W_hat, sample_x, sample_err)."""

    W_true = _true_W_for_policy(phi_s_policy, beta=beta)
    sample_x = _sample_indices(n_sweeps)
    sample_err = np.full(sample_x.shape[0], np.nan, dtype=float)

    evaluator = QHPolicyEvaluation(
        n_states=2,
        alpha=alpha,
        beta=beta,
        # Slow learning (small steps) + long run.
        theta_step=1e-3,
        eta_step=5e-2,
    )

    rng = np.random.default_rng(seed)
    start = time.time()
    next_sample_pos = 0
    completed = 0

    while completed < n_sweeps:
        remaining = n_sweeps - completed
        this_chunk = min(chunk_size, remaining)

        res = evaluator.evaluate_policy(
            sampler=env,
            sampling_policy=nu_policy,
            mu_policy=phi_s_policy,
            phi_policy=phi_s_policy,
            n_iterations=this_chunk,
            rng=rng,
            reference_values=W_true,
            reference_kind="W",
            adjust_support=True,
        )

        # Map chunk-local reference_diff into global sampled points.
        diff = res.get("reference_diff")
        if diff is not None:
            # Global iteration indices covered by this chunk: [completed, completed+this_chunk-1]
            end = completed + this_chunk - 1
            while next_sample_pos < sample_x.shape[0] and sample_x[next_sample_pos] <= end:
                g = int(sample_x[next_sample_pos])
                local = g - completed
                sample_err[next_sample_pos] = float(diff[local])
                next_sample_pos += 1

        completed += this_chunk
        _progress_bar(f"{label}", completed, n_sweeps, start_time=start)

    print("")
    return evaluator.W.copy(), sample_x, sample_err


def create_simple_mdp(rng: np.random.Generator) -> TransitionSampler:
    """Two-state MDP from Fig. 1b with rewards as in Table 1c."""

    def sampler(state: int, action: int) -> TransitionOutcome:
        if state == 0:
            if action == 0:
                return 1, 0.0, False
            next_s = 0 if rng.random() < 0.5 else 1
            return next_s, 2.0, False

        return 0, 17.0, False

    return sampler


def main() -> None:
    alpha = 0.5
    beta = 0.8
    n_states = 2
    n_actions = 2

    # Sweeps = outer iterations (each sweep updates all states once).
    # 10_000_000 sweeps => 20_000_000 state-updates for this 2-state MDP.
    n_sweeps = 10_000_000

    rng = np.random.default_rng(42)
    env = create_simple_mdp(rng)

    policy_f = np.zeros((n_states, n_actions))
    policy_f[0, 0] = 1.0
    policy_f[1, 0] = 1.0

    policy_g = np.zeros((n_states, n_actions))
    policy_g[0, 1] = 1.0
    policy_g[1, 0] = 1.0

    policy_h = np.zeros((n_states, n_actions))
    policy_h[0, 0] = 0.5
    policy_h[0, 1] = 0.5
    policy_h[1, 0] = 1.0

    nu_policy = np.ones((n_states, n_actions)) * 0.5

    def estimate_W_for_policy(phi_s_policy: np.ndarray, *, seed: int, label: str):
        """Estimate W(s) for a fixed continuation policy phi_s=phi_s_policy.

        Adds:
        - a progress bar (chunked evaluation),
        - convergence trace of ||W_k - W*||_2 sampled on a log grid.
        """

        return estimate_W_with_convergence(
            phi_s_policy=phi_s_policy,
            env=env,
            nu_policy=nu_policy,
            alpha=alpha,
            beta=beta,
            n_sweeps=n_sweeps,
            seed=seed,
            label=label,
        )

    def expected_immediate_reward_state0(phi_s_policy: np.ndarray) -> float:
        # In state 0: a0 => reward 0, a1 => reward 2.
        return 2.0 * float(phi_s_policy[0, 1])

    def q_from_W(phi_s_policy: np.ndarray, W: np.ndarray, state_idx: int, action_idx: int) -> float:
        """Compute Q(s,a) for 'take a in s then follow pi' using the paper's target form."""

        if state_idx == 0 and action_idx == 0:
            # Deterministic to state 1 with reward 0.
            next_state = 1
            reward = 0.0
            follow_reward = 17.0  # in state 1 reward is always 17
            return reward - (1.0 - alpha) * beta * follow_reward + beta * float(W[next_state])

        if state_idx == 0 and action_idx == 1:
            # 50/50 to state 0 or 1 with reward 2.
            reward = 2.0
            # If next_state=0 => follow_reward = E[r(0,a'~pi)] = 2*pi(a1|0)
            follow0 = expected_immediate_reward_state0(phi_s_policy)
            # If next_state=1 => follow_reward = 17
            follow1 = 17.0
            term0 = reward - (1.0 - alpha) * beta * follow0 + beta * float(W[0])
            term1 = reward - (1.0 - alpha) * beta * follow1 + beta * float(W[1])
            return 0.5 * term0 + 0.5 * term1

        if state_idx == 1 and action_idx == 0:
            # Deterministic to state 0 with reward 17.
            next_state = 0
            reward = 17.0
            follow_reward = expected_immediate_reward_state0(phi_s_policy)
            return reward - (1.0 - alpha) * beta * follow_reward + beta * float(W[next_state])

        raise ValueError("Unsupported (state, action) pair for this 2-state MDP.")

    scenarios = [
        ("1, a1", 0, 0, 18.88, 16.83, 18.00),
        ("1, a2", 0, 1, 19.00, 16.77, 18.00),
        ("2, a1", 1, 0, 32.11, 29.56, 31.00),
    ]

    print(f"--- Reproducing Table 1c (alpha={alpha}, beta={beta}) ---")
    print(f"{'Pair':<10} | {'f (Target)':<12} | {'g (Target)':<12} | {'h (Target)':<12}")
    print("-" * 50)

    # Estimate W once per background policy (expensive step).
    W_f, xs_f, err_f = estimate_W_for_policy(policy_f, seed=100, label="Policy f")
    W_g, xs_g, err_g = estimate_W_for_policy(policy_g, seed=200, label="Policy g")
    W_h, xs_h, err_h = estimate_W_for_policy(policy_h, seed=300, label="Policy h")

    # Plot convergence similar to InventoryMDP: sampled errors on a log x-axis.
    plt.figure(figsize=(8, 5))
    plt.plot(xs_f + 1, err_f, label="f", linewidth=1)
    plt.plot(xs_g + 1, err_g, label="g", linewidth=1)
    plt.plot(xs_h + 1, err_h, label="h", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Sweeps (log-scale)")
    plt.ylabel(r"$||W_k - W^*||_2$")
    plt.title("Two-state: convergence of $W_k$ (sampled)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("two_state_convergence_Wk.png", dpi=150)

    for label, s, a, tgt_f, tgt_g, tgt_h in scenarios:
        val_f = q_from_W(policy_f, W_f, s, a)
        val_g = q_from_W(policy_g, W_g, s, a)
        val_h = q_from_W(policy_h, W_h, s, a)

        print(f"{label:<10} | {val_f:<7.2f} ({tgt_f}) | {val_g:<7.2f} ({tgt_g}) | {val_h:<7.2f} ({tgt_h})")

    print("\nLegenda: Wartość obliczona (Wartość z artykułu)")


if __name__ == "__main__":
    main()
