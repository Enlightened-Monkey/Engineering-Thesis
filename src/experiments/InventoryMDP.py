"""
Reproduction script for 'Teaching Precommitted Agents' (Inventory Control Experiment).
References:
- Inventory Setup: Section V-A
- Algorithm 1 (Policy Eval): Section III, Implemented in qh_policy_evaluation.py
- Algorithm 2 (Q-Learning): Section IV-B, Implemented in qh_qlearning.py
- Analytical Solution: Value Iteration based on Section IV-A
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import itertools
from pathlib import Path

# Allow running as a plain script from the repo root, e.g.
#   python ./src/experiments/InventoryMDP.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation
from src.algorithms.qh_qlearning import QHQLearning

# --- 1. Environment Definition ---
class InventoryEnv:
    def __init__(self):
        self.M = 2  # Max capacity
        self.c = 5  # Purchase cost
        self.h = 2  # Holding cost
        self.p = 9  # Selling price
        
        self.demand_values = np.array([0, 1, 2])
        self.demand_probs = np.array([0.2, 0.3, 0.5])
        
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
        
        # Koszt operacyjny (może być ujemny, jeśli sprzedaż > wydatki)
        cost = (self.c * action + 
                self.h * max(s_hat - d, 0) - 
                self.p * min(s_hat, d))
        
        # Reward = -Cost. Ponieważ system jest zyskowny, Reward będzie dodatni.
        reward = -cost
        self.state = next_state
        return next_state, reward, False, {}

    def get_dynamics_matrices(self):
        """Computes P(s'|s,a) and R(s,a) for analytical solution."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                expected_reward = 0
                for d_idx, d in enumerate(self.demand_values):
                    prob = self.demand_probs[d_idx]
                    s_hat = min(s + a, self.M)
                    s_next = max(s_hat - d, 0)
                    
                    cost = (self.c * a + 
                            self.h * max(s_hat - d, 0) - 
                            self.p * min(s_hat, d))
                    
                    expected_reward += prob * (-cost)
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


# --- 2b. Exhaustive search for deterministic optimal policies ---
def enumerate_deterministic_policies(n_states: int, n_actions: int):
    """Yield all deterministic policies as one-hot matrices."""
    choices = np.arange(n_actions)
    for actions in np.array(np.meshgrid(*[choices]*n_states)).T.reshape(-1, n_states):
        pol = np.zeros((n_states, n_actions))
        pol[np.arange(n_states), actions] = 1.0
        yield pol, actions


def find_optimal_policies(env, alpha=0.3, beta=0.9):
    """Grid-search over deterministic (mu, phi_s) to maximize quasi-hyperbolic value.

    Criterion: maximize expected start-state value at s=0 (paper setup) and, as
    tie-breaker, maximize the sum of J over states.
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

            score = (J[0], J.sum())  # prefer higher start-state value, then total
            if best is None or score > best[0]:
                best = (score, mu, phi_s, J, W_phi_s)

    if best is None:
        raise RuntimeError("Failed to find an optimal policy (no candidates evaluated).")

    _, mu_star, phi_s_star, J_star, W_phi_s_star = best
    return mu_star, phi_s_star, J_star, W_phi_s_star

def solve_analytical_optimal_table(env, alpha, beta, tol=1e-8):
    P, R = env.get_dynamics_matrices()
    S, A = env.n_states, env.n_actions
    
    # 1. Value Iteration for Exponential Discounting
    V = np.zeros(S)
    while True:
        prev_V = V.copy()
        Q_temp = R + beta * np.sum(P * V.reshape(1, 1, S), axis=2)
        V = np.max(Q_temp, axis=1)
        if np.max(np.abs(V - prev_V)) < tol:
            break
            
    # 2. Optimal Exponential Q
    Q_beta = R + beta * np.sum(P * V.reshape(1, 1, S), axis=2)
    
    # 3. Quasi-Hyperbolic Q
    Q_qh = (1 - alpha) * R + alpha * Q_beta
    return Q_qh

def compute_policy_q_tables(env, actions, alpha, beta):
    """Compute V_pi, Q_gamma, and Q_qh for a deterministic policy.

    Args:
        env: InventoryEnv instance
        actions: list/array of length n_states with chosen action per state
        alpha: quasi-hyperbolic present-bias parameter
        beta: exponential discount factor

    Returns:
        W_phi_s: value vector following the policy (exponential / baseline)
        Q_beta: discounted Q for all (s,a) then following the policy
        Q_qh: quasi-hyperbolic Q for all (s,a)
    """
    P, R = env.get_dynamics_matrices()
    S, A = env.n_states, env.n_actions

    P_pi = np.zeros((S, S))
    R_pi = np.zeros(S)
    for s in range(S):
        a = actions[s]
        P_pi[s] = P[s, a]
        R_pi[s] = R[s, a]

    I = np.eye(S)
    W_phi_s = np.linalg.solve(I - beta * P_pi, R_pi)

    Q_beta = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            Q_beta[s, a] = R[s, a] + beta * (P[s, a] @ W_phi_s)

    Q_qh = (1 - alpha) * R + alpha * Q_beta
    return W_phi_s, Q_beta, Q_qh

def tune_hyperparameters(env, alpha, beta):
    print("\n--- Starting Hyperparameter Fine-Tuning (Local Counters) ---")
    
    # Define grid (Refined around best: theta_step=0.5, eta_step=1.2, theta_power=0.7, eta_power=0.65)
    theta_steps = [0.45, 0.5, 0.55]
    eta_steps = [1.15, 1.2, 1.25]
    theta_powers = [0.65, 0.7, 0.75]
    eta_powers = [0.6, 0.65, 0.7]
    
    best_max_diff = float('inf')
    best_params = None
    
    # Calculate analytical solution once
    Q_analytical = solve_analytical_optimal_table(env, alpha, beta)
    
    combinations = list(itertools.product(theta_steps, eta_steps, theta_powers, eta_powers))
    print(f"Testing {len(combinations)} combinations (with filtering)...")
    
    count = 0
    for t_step, e_step, t_pow, e_pow in combinations:
        # Constraints check
        if e_step < t_step: # eta_step must be >= theta_step
             continue
        if t_pow <= e_pow: # theta_power must be > eta_power
             continue
             
        count += 1
        # Run training
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

        # Sweep-based training with a generative model sampler(state, action).
        # Match the previous compute budget ("episodes" single-step updates) by
        # converting it to sweeps over S×A.
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

        episodes = 500000
        updates_per_sweep = agent.n_states * agent.n_actions
        n_sweeps = max(1, int(episodes) // int(updates_per_sweep))
        train_qh_qlearning_sweep(sampler=make_sampler(1000 + count), agent=agent, n_iterations=int(n_sweeps))
            
        # Calculate Max Absolute Difference (Worst Case Error)
        max_diff = np.max(np.abs(agent.Q - Q_analytical))
        
        if max_diff < best_max_diff:
            best_max_diff = max_diff
            best_params = {
                'theta_step': t_step,
                'eta_step': e_step,
                'theta_power': t_pow,
                'eta_power': e_pow
            }
            print(f"New best: MaxDiff={max_diff:.6f} | Params: {best_params}")
            
    print(f"\nBest Hyperparameters found: {best_params} with MaxDiff: {best_max_diff:.6f}")
    return best_params

# --- 3. Experiments ---

def run_optimal_control(best_params=None):
    print("\n--- Running Optimal Control (High Precision Mode) ---")
    env = InventoryEnv()
    alpha = 0.3
    beta = 0.9
    
    # Default params (Local Counters Baseline - Best Found)
    params = {
        'eta_step': 1.2,
        'eta_power': 0.65,
        'theta_step': 0.5,
        'theta_power': 0.7
    }
    
    if best_params:
        params.update(best_params)
        print(f"Using tuned parameters: {params}")
    
    # KLUCZOWA ZMIANA: Separacja skal czasowych (Timescale Separation)
    # W (eta) musi być "szybsze" niż Q (theta).
    # Warunek matematyczny: theta_n / eta_n -> 0 gdy n -> nieskończoność.
    
    # Deterministic exhaustive search for (mu*, phi_s*)
    mu_star, phi_s_star, J_star, W_phi_s = find_optimal_policies(env, alpha=alpha, beta=beta)

    # Compute Q^{*, phi_s*}_{alpha,beta}(s,a) = r(s,a) + alpha*beta * P(s,a) W_phi_s
    P, R = env.get_dynamics_matrices()
    Q_opt = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            Q_opt[s, a] = R[s, a] + alpha * beta * (P[s, a] @ W_phi_s)

    print("\nOptimal deterministic policies (found by exhaustive search):")
    print(f"mu* actions per state: {np.argmax(mu_star, axis=1).tolist()}")
    print(f"phi_s* actions per state: {np.argmax(phi_s_star, axis=1).tolist()}")

    print("\nQH Q-Value Function Q*_{mu*,phi_s*} (Profits):")
    print(f"{'s/a':<5} {'0':<8} {'1':<8} {'2':<8}")
    print("-" * 30)
    for s in range(env.n_states):
        row_str = f"{s:<5} " + " ".join(f"{val:8.2f}" for val in Q_opt[s])
        print(row_str)

    # Wrap into a lightweight object for downstream use
    class OptimalResult:
        def __init__(self, mu, phi_s, Q, W):
            self.mu = mu
            self.phi_s = phi_s
            self.Q = Q
            self.W = W

    return OptimalResult(mu_star, phi_s_star, Q_opt, W_phi_s)

def run_policy_evaluation_plot(optimal_agent):
    print("\n--- Running Policy Evaluation (Reproducing Fig 1) ---")
    env = InventoryEnv()
    alpha = 0.3
    beta = 0.9

    # Stateless sampler compatible with the synchronous-sweep implementation of
    # QHPolicyEvaluation: sampler(state, action) must simulate from the provided
    # state, not from an internal env.state.
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
        
    nu_equil = np.array([
        [1/3, 1/3, 1/3],  # state 0
        [0.5, 0.5, 0.0],  # state 1
        [1.0, 0.0, 0.0],  # state 2
    ])
    
    scenarios = [
        {"label": r"$\rho = (\mu^*, \varphi_s^*)$", "mu": mu_star, "phi": phi_s_star, "color": "blue"},
        {"label": r"$\rho = (\mu^*, \varphi_s^u)$", "mu": mu_star, "phi": nu_equil, "color": "red"},
        {"label": r"$\rho = (\mu^u, \varphi_s^*)$", "mu": nu_equil, "phi": phi_s_star, "color": "green"}
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
                n_states=3,
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

    plt.xscale('log')
    plt.xlabel("Iterations (log-scale)")
    plt.ylabel(r"$||J_k - J^{\alpha,\beta}_\rho||_2$")
    plt.title("Convergence of Policy Evaluation [Fig. 1]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reproduced_fig1.png", dpi=150)
    print("Plot saved to reproduced_fig1.png")

if __name__ == "__main__":
    # Grid search disabled; exhaustive search gives exact optimum for M=2
    best_params = None
    opt_agent = run_optimal_control(best_params)
    run_policy_evaluation_plot(opt_agent)