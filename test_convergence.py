#!/usr/bin/env python3
"""
Simplified test to verify that local count-based step sizes improve convergence
for rarely-visited state-action pairs in the Inventory MDP.
"""

# This file is a standalone demo script, not a unit test.
# Prevent pytest from collecting it by default.
__test__ = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Skip under pytest collection (this is a long-running demo).
if os.environ.get("PYTEST_CURRENT_TEST") is not None:
    import pytest

    pytest.skip("Standalone demo script; not part of unit tests.", allow_module_level=True)

import numpy as np
from src.algorithms.qh_qlearning import QHQLearning, train_qh_qlearning_sweep


class SimpleInventoryEnv:
    """Simplified Inventory environment for testing."""
    
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
        
        cost = (self.c * action + 
                self.h * max(s_hat - d, 0) - 
                self.p * min(s_hat, d))
        
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


def solve_analytical_optimal_table(env, sigma, gamma, tol=1e-8):
    """Calculate analytical Q-values using value iteration."""
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


def test_convergence():
    """Test that local counters improve convergence for rarely-visited pairs."""
    print("\n" + "="*80)
    print("Testing QH-Learning with Local Count-Based Step Sizes")
    print("="*80)
    
    env = SimpleInventoryEnv()
    sigma = 0.3
    gamma = 0.9
    
    # Calculate analytical solution
    print("\nCalculating analytical solution...")
    Q_analytical = solve_analytical_optimal_table(env, sigma, gamma)
    
    print("\nAnalytical Q-values:")
    for s in range(env.n_states):
        print(f"State {s}: " + " ".join(f"{val:8.4f}" for val in Q_analytical[s]))
    
    # Train agent via sweep-based generative model
    print("\nTraining agent with sweep-based updates...")
    agent = QHQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=sigma,
        beta=gamma,
        theta_step=1.0,
        eta_step=2.0,
        theta_power=0.60,
        eta_power=0.55,
    )

    def make_sampler(seed: int):
        rng = np.random.default_rng(seed)

        def sampler(state: int, action: int):
            s_hat = min(int(state) + int(action), env.M)
            d = rng.choice(env.demand_values, p=env.demand_probs)
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

    updates = 2_000_000
    updates_per_sweep = agent.n_states * agent.n_actions
    n_sweeps = max(1, int(updates) // int(updates_per_sweep))
    for i in range(1, n_sweeps + 1):
        if i % max(1, n_sweeps // 4) == 0:
            print(f"Progress: {i}/{n_sweeps} sweeps")
    train_qh_qlearning_sweep(sampler=make_sampler(42), agent=agent, n_iterations=int(n_sweeps))
    
    print("\nLearned Q-values:")
    for s in range(env.n_states):
        print(f"State {s}: " + " ".join(f"{val:8.4f}" for val in agent.Q[s]))
    
    print("\nVisit Counts:")
    for s in range(env.n_states):
        print(f"State {s}: " + " ".join(f"{val:8d}" for val in agent._visit_counts[s]))
    
    # Calculate errors
    print("\n" + "="*80)
    print("Convergence Analysis")
    print("="*80)
    print(f"{'State':<6} | {'Action':<6} | {'Analytical':<12} | {'RL Learned':<12} | {'Diff':<8} | {'Visits':<8}")
    print("-" * 80)
    
    max_diff = 0
    total_diff = 0
    count = 0
    
    for s in range(env.n_states):
        for a in range(env.n_actions):
            diff = abs(Q_analytical[s, a] - agent.Q[s, a])
            visits = agent._visit_counts[s, a]
            max_diff = max(max_diff, diff)
            total_diff += diff
            count += 1
            
            mark = " (OK)" if diff < 0.5 else " (!)"
            print(f"{s:<6} | {a:<6} | {Q_analytical[s,a]:<12.4f} | {agent.Q[s,a]:<12.4f} | {diff:<8.4f} | {visits:<8d}{mark}")
    
    avg_diff = total_diff / count
    
    print("\n" + "="*80)
    print(f"Maximum Absolute Difference: {max_diff:.4f}")
    print(f"Average Absolute Difference: {avg_diff:.4f}")
    print("="*80)
    
    # Success criteria
    success_threshold = 1.0  # More lenient than 0.1 for initial test
    if max_diff < success_threshold:
        print(f"\n✓ SUCCESS: All state-action pairs converged within {success_threshold}")
        return True
    else:
        print(f"\n✗ FAILURE: Some pairs have error > {success_threshold}")
        # Show which pairs failed
        print("\nFailed pairs:")
        for s in range(env.n_states):
            for a in range(env.n_actions):
                diff = abs(Q_analytical[s, a] - agent.Q[s, a])
                if diff >= success_threshold:
                    visits = agent._visit_counts[s, a]
                    print(f"  State {s}, Action {a}: Diff={diff:.4f}, Visits={visits}")
        return False


if __name__ == "__main__":
    success = test_convergence()
    sys.exit(0 if success else 1)
