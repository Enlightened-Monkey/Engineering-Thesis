"""Inventory Control experiment (variant).

This file is a parameter variant of `InventoryMDP.py` with:
- M = 3
- c = 6, h = 3, p = 11
- demand values [0, 1, 2, 3] with probabilities [0.1, 0.2, 0.3, 0.4]
- alpha = 0.35, beta = 0.95

Kept as a separate script to avoid changing the original reproduction setup.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from pathlib import Path

# Allow running as a plain script from the repo root, e.g.
#   python ./src/experiments/InventoryMDP_M3.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation
from src.algorithms.qh_qlearning import QHQLearning


# --- 1. Environment Definition ---
class InventoryEnv:
    def __init__(self):
        self.M = 3  # Max capacity
        self.c = 6  # Purchase cost
        self.h = 3  # Holding cost
        self.p = 11  # Selling price

        self.demand_values = np.array([0, 1, 2, 3])
        self.demand_probs = np.array([0.1, 0.2, 0.3, 0.4])

        self.n_states = self.M + 1
        self.n_actions = self.M + 1

        self.rng = np.random.default_rng(42)
        self.state = 0

    def reset(self):
        self.state = self.rng.choice(self.n_states)
        return self.state

    def step(self, action):
        s_hat = min(self.state + action, self.M)
        d = self.rng.choice(self.demand_values, p=self.demand_probs)
        next_state = max(s_hat - d, 0)

        cost = (
            self.c * action
            + self.h * max(s_hat - d, 0)
            - self.p * min(s_hat, d)
        )

        reward = -cost
        self.state = next_state
        return next_state, reward, False, {}

    def get_dynamics_matrices(self):
        """Computes P(s'|s,a) and R(s,a) for analytical solution."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                expected_reward = 0.0
                for d_idx, d in enumerate(self.demand_values):
                    prob = float(self.demand_probs[d_idx])
                    s_hat = min(s + a, self.M)
                    s_next = max(s_hat - int(d), 0)

                    cost = (
                        self.c * a
                        + self.h * max(s_hat - int(d), 0)
                        - self.p * min(s_hat, int(d))
                    )

                    expected_reward += prob * (-float(cost))
                    P[s, a, s_next] += prob
                R[s, a] = expected_reward
        return P, R


# --- 2. Analytical Solvers ---

def calculate_qh_value_for_policy(env, mu_policy, phi_s_policy, alpha, beta):
    P, R = env.get_dynamics_matrices()
    S = env.n_states

    def induce(policy):
        P_induced = np.zeros((S, S))
        R_induced = np.zeros(S)
        for s in range(S):
            for a in range(env.n_actions):
                prob = policy[s, a]
                P_induced[s] += prob * P[s, a]
                R_induced[s] += prob * R[s, a]
        return P_induced, R_induced

    P_phi_s, R_phi_s = induce(phi_s_policy)
    P_mu, R_mu = induce(mu_policy)

    I = np.eye(S)
    W_phi_s = np.linalg.solve(I - beta * P_phi_s, R_phi_s)
    J_qh = R_mu + alpha * beta * (P_mu @ W_phi_s)
    return J_qh


def enumerate_deterministic_policies(n_states: int, n_actions: int):
    """Yield all deterministic policies as one-hot matrices."""
    choices = np.arange(n_actions)
    for actions in np.array(np.meshgrid(*[choices] * n_states)).T.reshape(-1, n_states):
        pol = np.zeros((n_states, n_actions))
        pol[np.arange(n_states), actions] = 1.0
        yield pol, actions


def find_optimal_policies(env, alpha=0.35, beta=0.95):
    """Grid-search over deterministic (mu, phi_s) to maximize quasi-hyperbolic value.

    Criterion: maximize expected start-state value at s=0 and, as tie-breaker,
    maximize the sum of J over states.

    Returns: (mu_star, phi_s_star, J_star, W_phi_s)
    """

    best = None
    P, R = env.get_dynamics_matrices()
    S, A = env.n_states, env.n_actions
    I = np.eye(S)

    for phi_s, _ in enumerate_deterministic_policies(S, A):
        P_phi_s = (phi_s[:, :, None] * P).sum(axis=1)
        R_phi_s = (phi_s * R).sum(axis=1)
        W_phi_s = np.linalg.solve(I - beta * P_phi_s, R_phi_s)

        for mu, _ in enumerate_deterministic_policies(S, A):
            P_mu = (mu[:, :, None] * P).sum(axis=1)
            R_mu = (mu * R).sum(axis=1)
            J = R_mu + alpha * beta * (P_mu @ W_phi_s)

            score = (J[0], J.sum())
            if best is None or score > best[0]:
                best = (score, mu, phi_s, J, W_phi_s)

    if best is None:
        raise RuntimeError("Failed to find an optimal policy (no candidates evaluated).")

    _, mu_star, phi_s_star, J_star, W_phi_s_star = best
    return mu_star, phi_s_star, J_star, W_phi_s_star


def solve_analytical_optimal_table(env, alpha, beta, tol=1e-8):
    P, R = env.get_dynamics_matrices()
    S, _ = env.n_states, env.n_actions

    V = np.zeros(S)
    while True:
        prev_V = V.copy()
        Q_temp = R + beta * np.sum(P * V.reshape(1, 1, S), axis=2)
        V = np.max(Q_temp, axis=1)
        if np.max(np.abs(V - prev_V)) < tol:
            break

    Q_beta = R + beta * np.sum(P * V.reshape(1, 1, S), axis=2)
    Q_qh = (1 - alpha) * R + alpha * Q_beta
    return Q_qh


def tune_hyperparameters(env, alpha, beta):
    print("\n--- Starting Hyperparameter Fine-Tuning (Local Counters) ---")

    theta_steps = [0.45, 0.5, 0.55]
    eta_steps = [1.15, 1.2, 1.25]
    theta_powers = [0.65, 0.7, 0.75]
    eta_powers = [0.6, 0.65, 0.7]

    best_max_diff = float("inf")
    best_params = None

    Q_analytical = solve_analytical_optimal_table(env, alpha, beta)

    combinations = list(itertools.product(theta_steps, eta_steps, theta_powers, eta_powers))
    print(f"Testing {len(combinations)} combinations (with filtering)...")

    count = 0
    for t_step, e_step, t_pow, e_pow in combinations:
        if e_step < t_step:
            continue
        if t_pow <= e_pow:
            continue

        count += 1
        agent = QHQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=alpha,
            beta=beta,
            theta_step=t_step,
            eta_step=e_step,
            theta_power=t_pow,
            eta_power=e_pow,
        )

        from src.algorithms.qh_qlearning import train_qh_qlearning_sweep

        def make_sampler(seed: int):
            sampler_rng = np.random.default_rng(seed)

            def sampler(state: int, action: int):
                s_hat = min(int(state) + int(action), env.M)
                d = sampler_rng.choice(env.demand_values, p=env.demand_probs)
                next_state = max(s_hat - int(d), 0)

                cost = (
                    env.c * int(action)
                    + env.h * max(s_hat - int(d), 0)
                    - env.p * min(s_hat, int(d))
                )
                reward = -float(cost)
                done = False
                return int(next_state), reward, bool(done)

            return sampler

        episodes = 500_000
        updates_per_sweep = agent.n_states * agent.n_actions
        n_sweeps = max(1, int(episodes) // int(updates_per_sweep))
        train_qh_qlearning_sweep(
            sampler=make_sampler(1000 + count),
            agent=agent,
            n_iterations=int(n_sweeps),
        )

        max_diff = float(np.max(np.abs(agent.Q - Q_analytical)))

        if max_diff < best_max_diff:
            best_max_diff = max_diff
            best_params = {
                "theta_step": t_step,
                "eta_step": e_step,
                "theta_power": t_pow,
                "eta_power": e_pow,
            }
            print(f"New best: MaxDiff={max_diff:.6f} | Params: {best_params}")

    print(f"\nBest Hyperparameters found: {best_params} with MaxDiff: {best_max_diff:.6f}")
    return best_params


# --- 3. Experiments ---

def run_optimal_control(best_params=None):
    print("\n--- Running Optimal Control (High Precision Mode) ---")
    env = InventoryEnv()
    alpha = 0.35
    beta = 0.95

    params = {
        "eta_step": 1.2,
        "eta_power": 0.65,
        "theta_step": 0.5,
        "theta_power": 0.7,
    }

    if best_params:
        params.update(best_params)
        print(f"Using tuned parameters: {params}")

    mu_star, phi_s_star, _, W_phi_s = find_optimal_policies(env, alpha=alpha, beta=beta)

    P, R = env.get_dynamics_matrices()
    Q_opt = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            Q_opt[s, a] = R[s, a] + alpha * beta * (P[s, a] @ W_phi_s)

    print("\nOptimal deterministic policies (found by exhaustive search):")
    print(f"mu* actions per state: {np.argmax(mu_star, axis=1).tolist()}")
    print(f"phi_s* actions per state: {np.argmax(phi_s_star, axis=1).tolist()}")

    print("\nQH Q-Value Function Q*_{mu*,phi_s*} (Profits):")
    header = f"{'s/a':<5} " + " ".join(f"{a:<8}" for a in range(env.n_actions))
    print(header)
    print("-" * len(header))
    for s in range(env.n_states):
        row_str = f"{s:<5} " + " ".join(f"{val:8.2f}" for val in Q_opt[s])
        print(row_str)

    class OptimalResult:
        def __init__(self, mu, phi_s, Q, W):
            self.mu = mu
            self.phi_s = phi_s
            self.Q = Q
            self.W = W

    return OptimalResult(mu_star, phi_s_star, Q_opt, W_phi_s)


def run_policy_evaluation_plot(optimal_agent):
    print("\n--- Running Policy Evaluation (Variant) ---")
    env = InventoryEnv()
    alpha = 0.35
    beta = 0.95

    def make_sampler(seed: int):
        sampler_rng = np.random.default_rng(seed)

        def sampler(state: int, action: int):
            s_hat = min(state + action, env.M)
            d = sampler_rng.choice(env.demand_values, p=env.demand_probs)
            next_state = max(s_hat - int(d), 0)

            cost = (
                env.c * action
                + env.h * max(s_hat - int(d), 0)
                - env.p * min(s_hat, int(d))
            )
            reward = -float(cost)
            return int(next_state), reward

        return sampler

    mu_star = optimal_agent.mu
    phi_s_star = optimal_agent.phi_s

    # Simple baseline sampling policy (uniform over actions)
    nu_equil = np.full((env.n_states, env.n_actions), 1.0 / float(env.n_actions))

    scenarios = [
        {"label": r"$\rho = (\mu^*, \varphi_s^*)$", "mu": mu_star, "phi": phi_s_star, "color": "blue"},
        {"label": r"$\rho = (\mu^*, \varphi_s^u)$", "mu": mu_star, "phi": nu_equil, "color": "red"},
        {"label": r"$\rho = (\mu^u, \varphi_s^*)$", "mu": nu_equil, "phi": phi_s_star, "color": "green"},
    ]

    plt.figure(figsize=(8, 5))

    n_iterations = 5_000_000
    n_seeds = 3
    indices = np.unique(np.logspace(0, np.log10(n_iterations - 1), 500).astype(int))

    for scen in scenarios:
        J_true = calculate_qh_value_for_policy(env, scen["mu"], scen["phi"], alpha, beta)
        print(f"Analytical J^{{alpha, beta}} for {scen['label']}: {J_true}")

        sampled_errors = np.zeros(indices.shape[0], dtype=float)
        final_J = np.zeros((n_seeds, env.n_states), dtype=float)

        for k in range(n_seeds):
            evaluator = QHPolicyEvaluation(
                n_states=env.n_states,
                alpha=alpha,
                beta=beta,
                theta_step=0.2,
                eta_step=0.2,
                theta_exponent=0.6,
                eta_exponent=0.51,
            )

            res = evaluator.evaluate_policy(
                sampler=make_sampler(1000 + 13 * k),
                sampling_policy=nu_equil,
                mu_policy=scen["mu"],
                phi_policy=scen["phi"],
                n_iterations=n_iterations,
                reference_values=J_true,
                reference_kind="J",
                adjust_support=True,
            )

            errors = res["reference_diff"]
            sampled_errors += errors[indices]
            final_J[k] = res["J"]

        sampled_errors /= float(n_seeds)
        plt.plot(indices, sampled_errors, label=scen["label"], color=scen["color"], linewidth=1)

        J_mean = final_J.mean(axis=0)
        J_std = final_J.std(axis=0)
        print(f"Final J estimate for {scen['label']} (mean±std over {n_seeds} seeds): {J_mean} ± {J_std}")

    plt.xscale("log")
    plt.xlabel("Iterations (log-scale)")
    plt.ylabel(r"$||J_k - J^{\alpha,\beta}_\rho||_2$")
    plt.title("Convergence of Policy Evaluation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reproduced_fig1_m3.png", dpi=150)
    print("Plot saved to reproduced_fig1_m3.png")


if __name__ == "__main__":
    best_params = None
    opt_agent = run_optimal_control(best_params)
    run_policy_evaluation_plot(opt_agent)
