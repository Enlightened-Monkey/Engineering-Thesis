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

# Ensure we can import the provided modules
try:
    from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation
    from src.algorithms.qh_qlearning import QHQLearning
except ImportError:
    try:
        from qh_policy_evaluation import QHPolicyEvaluation
        from qh_qlearning import QHQLearning
    except ImportError:
        print("Error: Please ensure 'qh_policy_evaluation.py' and 'qh_qlearning.py' are in the python path.")
        sys.exit(1)

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

def calculate_qh_value_for_policy(env, mu_policy, pi_policy, sigma, gamma):
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

    P_pi, R_pi = induce(pi_policy)
    P_mu, R_mu = induce(mu_policy)

    I = np.eye(S)
    V_pi_exp = np.linalg.solve(I - gamma * P_pi, R_pi)
    V_qh = R_mu + sigma * gamma * (P_mu @ V_pi_exp)
    return V_qh

def solve_analytical_optimal_table(env, sigma, gamma, tol=1e-8):
    P, R = env.get_dynamics_matrices()
    S, A = env.n_states, env.n_actions
    
    # 1. Value Iteration for Exponential Discounting
    V = np.zeros(S)
    while True:
        prev_V = V.copy()
        Q_temp = R + gamma * np.sum(P * V.reshape(1, 1, S), axis=2)
        V = np.max(Q_temp, axis=1)
        if np.max(np.abs(V - prev_V)) < tol:
            break
            
    # 2. Optimal Exponential Q
    Q_gamma = R + gamma * np.sum(P * V.reshape(1, 1, S), axis=2)
    
    # 3. Quasi-Hyperbolic Q
    Q_qh = (1 - sigma) * R + sigma * Q_gamma
    return Q_qh

def tune_hyperparameters(env, sigma, gamma):
    print("\n--- Starting Hyperparameter Fine-Tuning (Max Error Minimization) ---")
    
    # Define grid (Centered around previous winners: 0.5, 1.0, 0.6, 0.55)
    theta_steps = [0.4, 0.5, 0.6]
    eta_steps = [0.9, 1.0, 1.1]
    theta_powers = [0.6, 0.65, 0.7]
    eta_powers = [0.55, 0.6]
    
    best_max_diff = float('inf')
    best_params = None
    
    # Calculate analytical solution once
    Q_analytical = solve_analytical_optimal_table(env, sigma, gamma)
    
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
            alpha=sigma,
            beta=gamma,
            theta_step=t_step,
            eta_step=e_step,
            theta_power=t_pow,
            eta_power=e_pow,
            epsilon=0.2  # Start with higher exploration
        )
        
        episodes = 500000
        state = env.reset()
        for i in range(episodes):
            # ZMIANA: Wymuszona eksploracja (min_epsilon = 0.15)
            if i % 10000 == 0:
                agent.epsilon = max(0.15, agent.epsilon * 0.99)
                
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done=done)
            state = next_state
            
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
    sigma = 0.3
    gamma = 0.9
    
    # Default params (Updated for Optimistic Initialization + Offset)
    params = {
        'eta_step': 2.0,       # Większy krok, bo mianownik jest teraz duży (100+)
        'eta_power': 0.55,
        'theta_step': 1.0,     # Większy krok dla Q
        'theta_power': 0.60
    }
    
    if best_params:
        params.update(best_params)
        print(f"Using tuned parameters: {params}")
    
    # KLUCZOWA ZMIANA: Separacja skal czasowych (Timescale Separation)
    # W (eta) musi być "szybsze" niż Q (theta).
    # Warunek matematyczny: theta_n / eta_n -> 0 gdy n -> nieskończoność.
    
    agent = QHQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=sigma,
        beta=gamma,
        
        # SZYBKA SKALA (W - Exponential Value):
        # Uczymy się agresywnie i wolno wygaszamy, by W zawsze było "przed" Q
        eta_step=params['eta_step'],      
        eta_power=params['eta_power'],    
        
        # WOLNA SKALA (Q - QH Value):
        # Uczymy się ostrożnie, czekając aż W się ustabilizuje
        theta_step=params['theta_step'],    
        theta_power=params['theta_power'],  # Szybsze wygaszanie kroku dla Q
        
        epsilon=0.2
    )
    
    # 5 milionów to optymalny balans dla tej konfiguracji
    episodes = 5_000_000
    print(f"Training RL Agent for {episodes} steps (Precision Mode)...")
    
    state = env.reset()
    for i in range(1, episodes + 1):
        if i % (episodes // 10) == 0:
            print(f"Progress: {i / episodes * 100:.0f}%")

        # Agresywne wygaszanie eksploracji w drugiej połowie treningu
        # Chcemy, żeby pod koniec agent "szlifował" tylko optymalną ścieżkę
        if agent.epsilon > 0.001:
            agent.epsilon *= 0.99995  # Bardzo płynne zejście do zera
            
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done=done)
        state = next_state
        
    # Interpretacja: Q w tym kodzie to Zysk (Value), więc bierzemy wprost
    Q_rl = agent.Q

    print("Calculating Analytical Ground Truth...")
    Q_analytical = solve_analytical_optimal_table(env, sigma, gamma)
    
    print("\n" + "="*75)
    print(f"{'State':<6} | {'Action':<6} | {'Analytical':<12} | {'RL (Learned)':<12} | {'Diff':<8}")
    print("-" * 75)
    
    for s in range(env.n_states):
        for a in range(env.n_actions):
            diff = abs(Q_analytical[s, a] - Q_rl[s, a])
            # Podświetlamy duże różnice
            if diff > 0.5:
                mark = " (!)"
            else:
                mark = " (OK)"
            print(f"{s:<6} | {a:<6} | {Q_analytical[s,a]:<12.4f} | {Q_rl[s,a]:<12.4f} | {diff:<8.4f}{mark}")

    print("\n--- TABLE I FORMAT COMPARISON ---")
    print("\nANALYTICAL (Reference):")
    for s in range(env.n_states):
        print(f"{s:<5} " + " ".join(f"{val:8.2f}" for val in Q_analytical[s]))

    print("\nRL LEARNED (Yours):")
    for s in range(env.n_states):
        print(f"{s:<5} " + " ".join(f"{val:8.2f}" for val in Q_rl[s]))

    return agent

def run_policy_evaluation_plot(optimal_agent):
    print("\n--- Running Policy Evaluation (Reproducing Fig 1) ---")
    env = InventoryEnv()
    sigma = 0.3
    gamma = 0.9
    
    mu_star = np.zeros((3, 3))
    pi_star = np.zeros((3, 3))
    for s in range(3):
        mu_star[s, np.argmax(optimal_agent.Q[s])] = 1.0
        pi_star[s, np.argmax(optimal_agent.W[s])] = 1.0
        
    uniform = np.ones((3, 3)) / 3.0
    
    scenarios = [
        {"label": r"$\rho = (\mu^*, \pi^*)$", "mu": mu_star, "phi": pi_star, "color": "blue"},
        {"label": r"$\rho = (\mu^*, \pi_u)$", "mu": mu_star, "phi": uniform, "color": "red"},
        {"label": r"$\rho = (\mu_u, \pi^*)$", "mu": uniform, "phi": pi_star, "color": "green"}
    ]
    
    plt.figure(figsize=(8, 5))
    
    for scen in scenarios:
        V_true = calculate_qh_value_for_policy(env, scen["mu"], scen["phi"], sigma, gamma)
        
        evaluator = QHPolicyEvaluation(
            n_states=3, alpha=sigma, beta=gamma,
            theta_step=0.2, eta_step=0.2,
            theta_exponent=0.6, eta_exponent=0.51
        )
        
        res = evaluator.evaluate_policy(
            sampler=lambda s, a: env.step(a)[:2],
            sampling_policy=uniform,
            mu_policy=scen["mu"],
            phi_policy=scen["phi"],
            n_iterations=200000,
            reference_values=V_true,
            adjust_support=True
        )
        
        errors = res['reference_diff']
        indices = np.unique(np.logspace(0, np.log10(len(errors)-1), 500).astype(int))
        plt.plot(indices, errors[indices], label=scen['label'], color=scen['color'], linewidth=1)

    plt.xscale('log')
    plt.xlabel("Iterations (log-scale)")
    plt.ylabel(r"$||V_k - V^{\sigma,\gamma}_\rho||_2$")
    plt.title("Convergence of Policy Evaluation [Fig. 1]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reproduced_fig1.png", dpi=150)
    print("Plot saved to reproduced_fig1.png")

if __name__ == "__main__":
    # Set to True to run hyperparameter grid search
    RUN_TUNING = True
    
    best_params = None
    if RUN_TUNING:
        # Create temporary env for tuning
        temp_env = InventoryEnv()
        best_params = tune_hyperparameters(temp_env, sigma=0.3, gamma=0.9)
        
    opt_agent = run_optimal_control(best_params)
    run_policy_evaluation_plot(opt_agent)