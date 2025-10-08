"""
Comparison of Standard vs Quasi-Hyperbolic Discounting in MDPs

This module compares the behavior and performance of:
1. Standard exponential discounting (traditional RL)
2. Quasi-hyperbolic discounting (precommitted agent)

The comparison includes:
- Policy differences
- Value function differences
- Reward accumulation over time
- Time-consistency analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('..')

from algorithms.qh_qlearning import QHQLearning
from models.mdp_environments import MDPEnvironment


class StandardQLearning:
    """
    Standard Q-Learning with exponential discounting.
    
    This serves as the baseline for comparison with QH Q-Learning.
    """
    
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        """
        Initialize standard Q-Learning.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Standard Q-Learning update."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """Get state-value function."""
        return np.max(self.q_table, axis=1)


class MDPComparison:
    """
    Framework for comparing standard and QH discounting approaches.
    """
    
    def __init__(self, 
                 env: MDPEnvironment,
                 sigma: float = 0.8,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        """
        Initialize comparison framework.
        
        Args:
            env: MDP environment to test on
            sigma: Present-bias parameter for QH discounting
            gamma: Discount factor (used by both algorithms)
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.env = env
        self.sigma = sigma
        self.gamma = gamma
        
        # Initialize both algorithms
        self.standard_qlearning = StandardQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon
        )
        
        self.qh_qlearning = QHQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            sigma=sigma,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon
        )
        
        # Storage for results
        self.standard_rewards: List[float] = []
        self.qh_rewards: List[float] = []
        self.standard_values: List[np.ndarray] = []
        self.qh_values: List[np.ndarray] = []
        
    def train_episode(self, max_steps: int = 100) -> Tuple[float, float]:
        """
        Train both algorithms for one episode.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (standard_total_reward, qh_total_reward)
        """
        # Train standard Q-learning
        state = self.env.reset()
        standard_total_reward = 0.0
        
        for _ in range(max_steps):
            action = self.standard_qlearning.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.standard_qlearning.update(state, action, reward, next_state, done)
            standard_total_reward += reward
            
            if done:
                break
            state = next_state
        
        # Train QH Q-learning
        state = self.env.reset()
        qh_total_reward = 0.0
        
        for _ in range(max_steps):
            action = self.qh_qlearning.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.qh_qlearning.update(state, action, reward, next_state, done)
            qh_total_reward += reward
            
            if done:
                break
            state = next_state
        
        return standard_total_reward, qh_total_reward
    
    def train(self, n_episodes: int = 1000, record_interval: int = 100) -> None:
        """
        Train both algorithms for multiple episodes.
        
        Args:
            n_episodes: Number of training episodes
            record_interval: Interval for recording metrics
        """
        print(f"Training both algorithms for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            standard_reward, qh_reward = self.train_episode()
            
            if episode % record_interval == 0:
                self.standard_rewards.append(standard_reward)
                self.qh_rewards.append(qh_reward)
                self.standard_values.append(self.standard_qlearning.get_value_function().copy())
                self.qh_values.append(self.qh_qlearning.get_value_function().copy())
                
                print(f"Episode {episode}: Standard={standard_reward:.2f}, QH={qh_reward:.2f}")
    
    def compare_policies(self) -> Dict[str, np.ndarray]:
        """
        Compare the learned policies.
        
        Returns:
            Dictionary containing both policies and their differences
        """
        standard_policy = self.standard_qlearning.get_policy()
        qh_policy = self.qh_qlearning.get_policy()
        
        # Find states where policies differ
        policy_diff = (standard_policy != qh_policy).astype(int)
        
        return {
            'standard_policy': standard_policy,
            'qh_policy': qh_policy,
            'different_states': np.where(policy_diff)[0],
            'agreement_percentage': 100 * (1 - policy_diff.mean())
        }
    
    def compare_values(self) -> Dict[str, np.ndarray]:
        """
        Compare the learned value functions.
        
        Returns:
            Dictionary containing value functions and their differences
        """
        standard_values = self.standard_qlearning.get_value_function()
        qh_values = self.qh_qlearning.get_value_function()
        
        value_diff = standard_values - qh_values
        
        return {
            'standard_values': standard_values,
            'qh_values': qh_values,
            'value_difference': value_diff,
            'mean_abs_difference': np.mean(np.abs(value_diff)),
            'max_abs_difference': np.max(np.abs(value_diff))
        }
    
    def analyze_time_consistency(self, initial_state: int, horizon: int = 10) -> Dict:
        """
        Analyze time-consistency by checking if the agent would want to 
        deviate from its precommitted policy at each time step.
        
        Args:
            initial_state: Starting state
            horizon: Planning horizon
            
        Returns:
            Dictionary with time-consistency analysis
        """
        # Get QH precommitted policy
        qh_policy = self.qh_qlearning.get_policy()
        
        # Simulate trajectory under precommitted policy
        state = initial_state
        trajectory = [state]
        actions = []
        
        for t in range(horizon):
            action = qh_policy[state]
            actions.append(action)
            
            # Get next state (deterministic for analysis)
            next_state, _, done, _ = self.env.step(action)
            trajectory.append(next_state)
            
            if done:
                break
            state = next_state
        
        # Check for time-inconsistency
        # At each step, would the agent prefer a different action?
        inconsistencies = []
        for t, state in enumerate(trajectory[:-1]):
            precommitted_action = qh_policy[state]
            
            # What would myopic agent choose? (sigma applied to immediate reward)
            myopic_values = self.qh_qlearning.q_qh[state]
            myopic_action = np.argmax(myopic_values)
            
            if precommitted_action != myopic_action:
                inconsistencies.append({
                    'time': t,
                    'state': state,
                    'precommitted_action': precommitted_action,
                    'myopic_action': myopic_action
                })
        
        return {
            'trajectory': trajectory,
            'actions': actions,
            'inconsistencies': inconsistencies,
            'is_time_consistent': len(inconsistencies) == 0
        }
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization comparing both approaches.
        
        Args:
            save_path: Path to save the plot (if None, displays instead)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Learning curves
        ax = axes[0, 0]
        episodes = np.arange(len(self.standard_rewards)) * 100
        ax.plot(episodes, self.standard_rewards, label='Standard Q-Learning', marker='o')
        ax.plot(episodes, self.qh_rewards, label='QH Q-Learning', marker='s')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True)
        
        # 2. Value function comparison
        ax = axes[0, 1]
        comparison = self.compare_values()
        states = np.arange(self.env.n_states)
        ax.plot(states, comparison['standard_values'], label='Standard', marker='o')
        ax.plot(states, comparison['qh_values'], label='QH', marker='s')
        ax.set_xlabel('State')
        ax.set_ylabel('Value')
        ax.set_title('Value Functions Comparison')
        ax.legend()
        ax.grid(True)
        
        # 3. Policy comparison
        ax = axes[1, 0]
        policy_comp = self.compare_policies()
        x = np.arange(self.env.n_states)
        width = 0.35
        ax.bar(x - width/2, policy_comp['standard_policy'], width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, policy_comp['qh_policy'], width, label='QH', alpha=0.8)
        ax.set_xlabel('State')
        ax.set_ylabel('Action')
        ax.set_title(f"Policy Comparison (Agreement: {policy_comp['agreement_percentage']:.1f}%)")
        ax.legend()
        ax.grid(True, axis='y')
        
        # 4. Value difference heatmap
        ax = axes[1, 1]
        value_diff = comparison['value_difference']
        ax.bar(states, value_diff, color=['red' if v < 0 else 'green' for v in value_diff])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('State')
        ax.set_ylabel('Value Difference (Standard - QH)')
        ax.set_title(f"Value Difference (Mean: {comparison['mean_abs_difference']:.2f})")
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report summarizing the comparison.
        
        Returns:
            Formatted report string
        """
        policy_comp = self.compare_policies()
        value_comp = self.compare_values()
        
        report = "=" * 70 + "\n"
        report += "COMPARISON: Standard vs Quasi-Hyperbolic Discounting\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Environment: {self.env.__class__.__name__}\n"
        report += f"States: {self.env.n_states}, Actions: {self.env.n_actions}\n"
        report += f"Gamma (discount): {self.gamma}\n"
        report += f"Sigma (present-bias): {self.sigma}\n"
        report += f"Episodes trained: {len(self.standard_rewards) * 100}\n\n"
        
        report += "POLICY COMPARISON\n"
        report += "-" * 70 + "\n"
        report += f"Agreement: {policy_comp['agreement_percentage']:.1f}%\n"
        report += f"Different states: {len(policy_comp['different_states'])}\n"
        if len(policy_comp['different_states']) > 0:
            report += f"States where policies differ: {policy_comp['different_states']}\n"
        report += "\n"
        
        report += "VALUE FUNCTION COMPARISON\n"
        report += "-" * 70 + "\n"
        report += f"Mean absolute difference: {value_comp['mean_abs_difference']:.4f}\n"
        report += f"Max absolute difference: {value_comp['max_abs_difference']:.4f}\n"
        report += "\n"
        
        report += "PERFORMANCE COMPARISON\n"
        report += "-" * 70 + "\n"
        if len(self.standard_rewards) > 0:
            final_standard = self.standard_rewards[-1]
            final_qh = self.qh_rewards[-1]
            report += f"Final episode reward (Standard): {final_standard:.2f}\n"
            report += f"Final episode reward (QH): {final_qh:.2f}\n"
            report += f"Difference: {final_standard - final_qh:.2f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report


def run_comparison_example():
    """Example usage of the comparison framework."""
    from models.mdp_environments import InventoryMDP
    
    # Create environment
    env = InventoryMDP(max_inventory=10, max_order=5)
    
    # Create comparison object
    comparison = MDPComparison(
        env=env,
        sigma=0.7,  # Present-bias parameter
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1
    )
    
    # Train both algorithms
    comparison.train(n_episodes=5000, record_interval=100)
    
    # Print report
    print(comparison.generate_report())
    
    # Analyze time-consistency
    consistency_analysis = comparison.analyze_time_consistency(initial_state=5, horizon=10)
    print("\nTIME-CONSISTENCY ANALYSIS")
    print("-" * 70)
    print(f"Is time-consistent: {consistency_analysis['is_time_consistent']}")
    print(f"Number of inconsistencies: {len(consistency_analysis['inconsistencies'])}")
    
    # Plot comparison
    comparison.plot_comparison(save_path='../../data/plots/standard_vs_qh_comparison.png')


if __name__ == "__main__":
    run_comparison_example()
