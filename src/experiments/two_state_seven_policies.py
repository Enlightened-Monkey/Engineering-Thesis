"""
Two-state MDP with seven policies for quasi-hyperbolic discounting analysis.

This module implements a simple two-state MDP based on the diagram:
- State 1: Initial state
- State 2: Target state
- Actions: a1 (deterministic transition to state 2) and a2 (probabilistic transitions)

Seven policies are tested:
1. Always a1
2. Always a2
3. 50/50 mix in state 1, always a1 in state 2
4. 50/50 mix in state 1, always a2 in state 2
5. 50/50 mix in both states
6. 75/25 (a1/a2) in state 1, always a1 in state 2
7. 25/75 (a1/a2) in state 1, always a1 in state 2
"""

import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


class TwoStateSevenPoliciesMDP:
    """Two-state MDP with seven distinct stationary policies."""
    
    def __init__(self, sigma: float = 0.8, gamma: float = 0.95):
        """
        Initialize the two-state MDP.
        
        Parameters
        ----------
        sigma : float
            Present-bias parameter (quasi-hyperbolic discount), range [0, 1]
        gamma : float
            Standard discount factor, range [0, 1)
        """
        self.sigma = sigma  # α in mathematical notation
        self.gamma = gamma  # β in mathematical notation
        
        # State space: {1, 2}
        self.states = [1, 2]
        # Action space: {a1, a2}
        self.actions = ['a1', 'a2']
        
        # Transition probabilities: q(s' | s, a)
        # Format: self.transitions[(s, a)] = {s': probability}
        self.transitions = {
            (1, 'a1'): {2: 1.0},                    # From state 1, action a1 → state 2 (deterministic)
            (1, 'a2'): {1: 0.5, 2: 0.5},           # From state 1, action a2 → 50/50 mix
            (2, 'a1'): {2: 0.5, 1: 0.5},           # From state 2, action a1 → 50/50 mix
            (2, 'a2'): {1: 0.5, 2: 0.5},           # From state 2, action a2 → 50/50 mix (or absorbing)
        }
        
        # Reward function: r(s, a)
        self.rewards = {
            (1, 'a1'): 0,      # Cost/reward for action a1 in state 1
            (1, 'a2'): 2,      # Higher immediate reward for a2
            (2, 'a1'): 5,      # High reward in state 2 with a1
            (2, 'a2'): 20,     # Very high reward in state 2 with a2
        }
    
    def define_seven_policies(self) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Define seven stationary policies.
        
        Returns
        -------
        Dict[str, Dict[int, np.ndarray]]
            Dictionary mapping policy name to policy definition.
            Policy format: {state: [prob(a1), prob(a2)]}
        """
        policies = {
            'always_a1': {
                1: np.array([1.0, 0.0]),  # State 1: always a1
                2: np.array([1.0, 0.0]),  # State 2: always a1
            },
            'always_a2': {
                1: np.array([0.0, 1.0]),  # State 1: always a2
                2: np.array([0.0, 1.0]),  # State 2: always a2
            },
            'fifty_fifty_s1_always_a1_s2': {
                1: np.array([0.5, 0.5]),  # State 1: 50/50
                2: np.array([1.0, 0.0]),  # State 2: always a1
            },
            'fifty_fifty_s1_always_a2_s2': {
                1: np.array([0.5, 0.5]),  # State 1: 50/50
                2: np.array([0.0, 1.0]),  # State 2: always a2
            },
            'fifty_fifty_both': {
                1: np.array([0.5, 0.5]),  # State 1: 50/50
                2: np.array([0.5, 0.5]),  # State 2: 50/50
            },
            'seventy_five_a1_s1_always_a1_s2': {
                1: np.array([0.75, 0.25]),  # State 1: 75/25 (favor a1)
                2: np.array([1.0, 0.0]),    # State 2: always a1
            },
            'twenty_five_a1_s1_always_a1_s2': {
                1: np.array([0.25, 0.75]),  # State 1: 25/75 (favor a2)
                2: np.array([1.0, 0.0]),    # State 2: always a1
            },
        }
        return policies
    
    def evaluate_policy_qh(self, policy: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Evaluate a stationary policy under quasi-hyperbolic discounting.
        
        Uses the value function definition:
        V(s) = sum_a π(a|s) [r(s,a) + σ·γ·sum_{s'} q(s'|s,a)·V(s')]
        
        Parameters
        ----------
        policy : Dict[int, np.ndarray]
            Policy definition: {state: [prob(a1), prob(a2)]}
        
        Returns
        -------
        Dict[int, float]
            Value function for each state: {state: value}
        """
        sigma_gamma = self.sigma * self.gamma
        
        # Extract probabilities
        pi_1 = policy[1]  # [prob(a1), prob(a2)] in state 1
        pi_2 = policy[2]  # [prob(a1), prob(a2] in state 2
        
        # Expected immediate reward in each state
        r_1 = pi_1[0] * self.rewards[(1, 'a1')] + pi_1[1] * self.rewards[(1, 'a2')]
        r_2 = pi_2[0] * self.rewards[(2, 'a1')] + pi_2[1] * self.rewards[(2, 'a2')]
        
        # Build coefficient matrix: (I - σγ·P)·V = r
        # where P is the transition matrix under the policy
        
        # P[s, s'] = sum_a π(a|s) · q(s'|s,a)
        P = np.zeros((2, 2))
        
        # From state 1:
        # Action a1: goes to state 2 (prob 1.0)
        # Action a2: goes to state 1 (prob 0.5) or state 2 (prob 0.5)
        P[0, 0] = pi_1[1] * 0.5          # Prob of going to state 1 from state 1
        P[0, 1] = pi_1[0] * 1.0 + pi_1[1] * 0.5  # Prob of going to state 2 from state 1
        
        # From state 2:
        # Action a1: goes to state 2 (prob 0.5) or state 1 (prob 0.5)
        # Action a2: goes to state 1 (prob 0.5) or state 2 (prob 0.5)
        P[1, 0] = pi_2[0] * 0.5 + pi_2[1] * 0.5  # Prob of going to state 1 from state 2
        P[1, 1] = pi_2[0] * 0.5 + pi_2[1] * 0.5  # Prob of going to state 2 from state 2
        
        # Solve (I - σγ·P)·V = r
        A = np.eye(2) - sigma_gamma * P
        b = np.array([r_1, r_2])
        
        V = np.linalg.solve(A, b)
        
        return {1: V[0], 2: V[1]}
    
    def evaluate_all_policies(self) -> Dict[str, Dict[int, float]]:
        """
        Evaluate all seven policies and compute Q-values for each (state, action) pair.
        
        Returns
        -------
        Dict[str, Dict[tuple, float]]
            Q-values for each policy: {policy_name: {(s, a): q_value}}
        """
        policies = self.define_seven_policies()
        results = {}
        
        for policy_name, policy in policies.items():
            # First compute V for this policy
            V = self.evaluate_policy_qh(policy)
            
            # Then compute Q(s,a) for all (s,a) pairs
            Q_values = {}
            sigma_gamma = self.sigma * self.gamma
            
            for s in [1, 2]:
                for a in ['a1', 'a2']:
                    r_sa = self.rewards[(s, a)]
                    next_value = sum(
                        self.transitions[(s, a)].get(s_prime, 0) * V[s_prime]
                        for s_prime in [1, 2]
                    )
                    q_sa = r_sa + sigma_gamma * next_value
                    Q_values[(s, a)] = q_sa
            
            results[policy_name] = Q_values
        
        return results
    
    def compare_sigma_sensitivity(self, sigma_values: List[float] = None) -> Dict[float, Dict[str, Dict[int, float]]]:
        """
        Evaluate all policies for different sigma values.
        
        Parameters
        ----------
        sigma_values : List[float], optional
            Sigma values to test. Default: [0.5, 0.7, 0.9, 1.0]
        
        Returns
        -------
        Dict[float, Dict[str, Dict[int, float]]]
            Results structure: {sigma: {policy_name: {state: value}}}
        """
        if sigma_values is None:
            sigma_values = [0.5, 0.7, 0.9, 1.0]
        
        results = {}
        for sigma in sigma_values:
            self.sigma = sigma
            results[sigma] = self.evaluate_all_policies()
        
        return results
    
    def plot_policy_comparison(self):
        """Visualize value functions for all seven policies."""
        results = self.evaluate_all_policies()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        policy_names = list(results.keys())
        values_s1 = [results[p][1] for p in policy_names]
        values_s2 = [results[p][2] for p in policy_names]
        
        x = np.arange(len(policy_names))
        width = 0.35
        
        ax1.bar(x - width/2, values_s1, width, label='State 1', alpha=0.8)
        ax1.bar(x + width/2, values_s2, width, label='State 2', alpha=0.8)
        ax1.set_xlabel('Policy')
        ax1.set_ylabel('Value Function')
        ax1.set_title(f'Policy Comparison (σ={self.sigma}, γ={self.gamma})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(policy_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Sensitivity analysis
        sigma_results = self.compare_sigma_sensitivity()
        sigmas = sorted(sigma_results.keys())
        
        for policy_name in ['always_a1', 'always_a2', 'fifty_fifty_both']:
            values = [sigma_results[s][policy_name][1] for s in sigmas]
            ax2.plot(sigmas, values, marker='o', label=policy_name)
        
        ax2.set_xlabel('σ (present-bias parameter)')
        ax2.set_ylabel('Value in State 1')
        ax2.set_title('Sensitivity to Quasi-Hyperbolic Parameter')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig


if __name__ == '__main__':
    import pandas as pd
    from src.algorithms.qh_qlearning import QHQLearning
    
    # ==================== ANALYTICAL EVALUATION ====================
    print("="*80)
    print("ANALYTICAL EVALUATION: 7 POLICIES")
    print("="*80)
    
    mdp = TwoStateSevenPoliciesMDP(sigma=0.8, gamma=0.95)
    
    results = mdp.evaluate_all_policies()
    
    # Build table: rows = (state, action), columns = policies
    data = {}
    for policy_name, q_values in results.items():
        data[policy_name] = [
            q_values[(1, 'a1')],
            q_values[(1, 'a2')],
            q_values[(2, 'a1')],
            q_values[(2, 'a2')],
        ]
    
    df = pd.DataFrame(
        data,
        index=['(1, a1)', '(1, a2)', '(2, a1)', '(2, a2)']
    )
    
    print("\nQ-values under quasi-hyperbolic discounting (σ=0.8, γ=0.95):")
    print(df.to_string())
    
    # ==================== QH Q-LEARNING ====================
    print("\n" + "="*80)
    print("QH Q-LEARNING TRAINING ON TWO-STATE MDP")
    print("="*80)
    
    # Simple environment wrapper (convert between 1-indexed MDP and 0-indexed agent)
    class SimpleEnv:
        def __init__(self, mdp_instance):
            self.mdp = mdp_instance
            self.current_state = 0  # 0-indexed for agent
        
        def reset(self):
            self.current_state = 0  # Start in state 0 (corresponds to MDP state 1)
            return self.current_state
        
        def step(self, action):
            s_mdp = self.current_state + 1  # Convert to 1-indexed for MDP
            a = ['a1', 'a2'][action]
            r = self.mdp.rewards[(s_mdp, a)]
            transitions = self.mdp.transitions[(s_mdp, a)]
            s_prime_mdp = np.random.choice(list(transitions.keys()), p=list(transitions.values()))
            self.current_state = s_prime_mdp - 1  # Convert back to 0-indexed
            done = False
            return self.current_state, r, done, {}
    
    # Create environment and agent
    env = SimpleEnv(mdp)
    agent = QHQLearning(n_states=2, n_actions=2, alpha=0.8, beta=0.95,
                        theta_step=0.1, eta_step=0.1, epsilon=0.1)
    
    # Train
    n_episodes = 5000
    max_steps_per_episode = 200  # <- ważne, inaczej epizod może nie skończyć się nigdy

    for episode in range(n_episodes):
        state = env.reset()
        done = False

        for _t in range(max_steps_per_episode):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done=done)
            state = next_state
            if done:
                break

        # Decay exploration
        if (episode + 1) % 500 == 0:
            agent.epsilon *= 0.95
    
    # Extract learned policy
    learned_policy = agent.get_policy()
    
    print(f"\nAfter {n_episodes} episodes:")
    print(f"\nLearned Q-table:")
    for s in range(2):
        s_mdp = s + 1  # Convert to 1-indexed for display
        print(f"  State {s_mdp}: Q(a1)={agent.Q[s, 0]:9.4f}, Q(a2)={agent.Q[s, 1]:9.4f}")
    
    state_0_action = ['a1', 'a2'][learned_policy[0]]
    state_1_action = ['a1', 'a2'][learned_policy[1]]
    
    print(f"\nLearned Policy:")
    print(f"  State 1 → {state_0_action}")
    print(f"  State 2 → {state_1_action}")
    
    # Find best analytical policy based on Q-values
    print(f"\n" + "="*80)
    print("COMPARISON WITH ANALYTICAL SOLUTIONS")
    print("="*80)
    
    # For each state, find the optimal action from the table
    print("\nAnalytical Q-table by state-action pair:")
    for sa_pair in ['(1, a1)', '(1, a2)', '(2, a1)', '(2, a2)']:
        print(f"  {sa_pair}: {df[df.index == sa_pair].values[0]}")
    
    # Find optimal actions
    q_1_a1 = df.loc['(1, a1)'].mean()
    q_1_a2 = df.loc['(1, a2)'].mean()
    opt_a1 = 'a1' if q_1_a1 > q_1_a2 else 'a2'
    
    q_2_a1 = df.loc['(2, a1)'].mean()
    q_2_a2 = df.loc['(2, a2)'].mean()
    opt_a2 = 'a1' if q_2_a1 > q_2_a2 else 'a2'
    
    print(f"\nAnalytical Optimal Actions (greedy on Q-values):")
    print(f"  State 1 → {opt_a1}")
    print(f"  State 2 → {opt_a2}")
    
    match_s1 = '✓' if state_0_action == opt_a1 else '✗'
    match_s2 = '✓' if state_1_action == opt_a2 else '✗'
    print(f"\nMatch: State 1: {match_s1}, State 2: {match_s2}")