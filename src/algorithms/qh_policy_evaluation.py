"""
Policy Evaluation for Quasi-Hyperbolic Discounting

Model-free policy evaluation algorithm using two-timescale stochastic approximation
for policies under quasi-hyperbolic discounting.
"""

import numpy as np
from typing import Dict, Callable, Optional

class QHPolicyEvaluation:
    """
    Model-free policy evaluation for quasi-hyperbolic discounted MDPs.
    
    Uses two-timescale stochastic approximation to estimate value functions
    for policies under QH discounting without requiring environment model.
    """
    
    def __init__(self,
                 n_states: int,
                 sigma: float = 0.8,
                 gamma: float = 0.95,
                 alpha_slow: float = 0.01,
                 alpha_fast: float = 0.1):
        """
        Initialize QH policy evaluation.
        
        Args:
            n_states: Number of states
            sigma: Present-bias parameter
            gamma: Exponential discount factor
            alpha_slow: Learning rate for slow timescale
            alpha_fast: Learning rate for fast timescale
        """
        self.n_states = n_states
        self.sigma = sigma
        self.gamma = gamma
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        
        # Initialize value function estimates
        self.v_exp = np.zeros(n_states)  # Exponential value function
        self.v_qh = np.zeros(n_states)   # Quasi-hyperbolic value function
        
    def update(self, state: int, reward: float, next_state: int) -> None:
        """
        Update value function estimates using two-timescale SA.
        
        Args:
            state: Current state
            reward: Received reward
            next_state: Next state
        """
        # Fast timescale: update exponential value function
        td_error_exp = reward + self.gamma * self.v_exp[next_state] - self.v_exp[state]
        self.v_exp[state] += self.alpha_fast * td_error_exp
        
        # Slow timescale: update QH value function
        qh_target = reward + self.sigma * self.gamma * self.v_qh[next_state]
        td_error_qh = qh_target - self.v_qh[state]
        self.v_qh[state] += self.alpha_slow * td_error_qh
    
    def evaluate_policy(self, env, policy: Callable[[int], int], 
                       n_episodes: int = 1000) -> Dict:
        """
        Evaluate given policy using the environment.
        
        Args:
            env: Environment
            policy: Policy function (state -> action)
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        episode_returns = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_return = 0
            discount = 1.0
            done = False
            
            while not done:
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                
                # Update value functions
                self.update(state, reward, next_state)
                
                # Track episode return
                episode_return += discount * reward
                discount *= self.gamma
                
                state = next_state
            
            episode_returns.append(episode_return)
        
        return {
            'episode_returns': episode_returns,
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'v_exp': self.v_exp.copy(),
            'v_qh': self.v_qh.copy()
        }
    
    def get_convergence_metrics(self) -> Dict:
        """
        Get metrics to assess convergence of the algorithm.
        
        Returns:
            Convergence metrics
        """
        return {
            'v_exp_norm': np.linalg.norm(self.v_exp),
            'v_qh_norm': np.linalg.norm(self.v_qh),
            'difference_norm': np.linalg.norm(self.v_qh - self.v_exp)
        }