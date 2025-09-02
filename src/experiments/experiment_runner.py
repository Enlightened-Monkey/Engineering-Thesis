"""
Experimental Framework for QH Discounting Algorithms

This module provides tools for running comprehensive experiments
to validate and compare quasi-hyperbolic discounting algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

from ..algorithms.qh_qlearning import QHQLearning, train_qh_qlearning
from ..algorithms.qh_policy_evaluation import QHPolicyEvaluation
from ..models.mdp_environments import InventoryMDP, GridWorldMDP

class ExperimentRunner:
    """
    Main class for running and managing experiments.
    """
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_inventory_experiment(self, 
                                sigma_values: List[float],
                                n_runs: int = 5,
                                n_episodes: int = 2000) -> Dict:
        """
        Run inventory management experiment with different sigma values.
        
        Args:
            sigma_values: List of present-bias parameters to test
            n_runs: Number of independent runs per parameter
            n_episodes: Episodes per training run
            
        Returns:
            Experiment results
        """
        print("Running inventory management experiments...")
        
        results = {
            'sigma_values': sigma_values,
            'convergence_data': [],
            'policy_comparison': [],
            'performance_metrics': []
        }
        
        for sigma in sigma_values:
            print(f"Testing sigma = {sigma:.2f}")
            
            sigma_results = {
                'sigma': sigma,
                'runs': [],
                'mean_performance': 0,
                'std_performance': 0
            }
            
            run_performances = []
            
            for run in range(n_runs):
                # Create environment and agent
                env = InventoryMDP(max_inventory=15, max_order=8)
                agent = QHQLearning(
                    n_states=env.n_states,
                    n_actions=env.n_actions,
                    sigma=sigma,
                    gamma=0.95,
                    alpha=0.1,
                    epsilon=0.2
                )
                
                # Train agent
                start_time = time.time()
                training_results = train_qh_qlearning(env, agent, n_episodes)
                training_time = time.time() - start_time
                
                # Evaluate final policy
                final_performance = np.mean(training_results['episode_rewards'][-100:])
                run_performances.append(final_performance)
                
                sigma_results['runs'].append({
                    'run_id': run,
                    'training_time': training_time,
                    'final_performance': final_performance,
                    'episode_rewards': training_results['episode_rewards'],
                    'final_policy': training_results['final_policy']
                })
            
            sigma_results['mean_performance'] = np.mean(run_performances)
            sigma_results['std_performance'] = np.std(run_performances)
            
            results['performance_metrics'].append(sigma_results)
        
        return results
    
    def run_convergence_analysis(self, 
                                env_type: str = 'inventory',
                                sigma: float = 0.8) -> Dict:
        """
        Analyze convergence properties of QH algorithms.
        
        Args:
            env_type: Type of environment ('inventory' or 'gridworld')
            sigma: Present-bias parameter
            
        Returns:
            Convergence analysis results
        """
        print(f"Running convergence analysis for {env_type} environment...")
        
        # Create environment
        if env_type == 'inventory':
            env = InventoryMDP(max_inventory=10, max_order=5)
        else:
            env = GridWorldMDP(width=4, height=4)
        
        # Create agent
        agent = QHQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            sigma=sigma,
            gamma=0.95,
            alpha=0.05,
            epsilon=0.1
        )
        
        # Track convergence metrics during training
        convergence_metrics = {
            'episodes': [],
            'q_function_changes': [],
            'policy_changes': [],
            'value_function_estimates': []
        }
        
        n_episodes = 1500
        check_interval = 50
        prev_policy = None
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
            
            # Check convergence metrics periodically
            if episode % check_interval == 0:
                current_policy = agent.get_policy()
                current_values = agent.get_value_function()
                
                convergence_metrics['episodes'].append(episode)
                convergence_metrics['value_function_estimates'].append(current_values.copy())
                
                if prev_policy is not None:
                    policy_change = np.mean(current_policy != prev_policy)
                    convergence_metrics['policy_changes'].append(policy_change)
                else:
                    convergence_metrics['policy_changes'].append(1.0)
                
                prev_policy = current_policy.copy()
        
        return convergence_metrics
    
    def compare_traditional_vs_qh(self, 
                                 env_type: str = 'inventory',
                                 n_runs: int = 10) -> Dict:
        """
        Compare traditional exponential vs QH discounting performance.
        
        Args:
            env_type: Environment type
            n_runs: Number of comparison runs
            
        Returns:
            Comparison results
        """
        print(f"Comparing traditional vs QH discounting on {env_type} environment...")
        
        # Create environment
        if env_type == 'inventory':
            env = InventoryMDP()
        else:
            env = GridWorldMDP()
        
        traditional_results = []
        qh_results = []
        
        for run in range(n_runs):
            # Traditional Q-learning (sigma = 1.0)
            agent_traditional = QHQLearning(
                n_states=env.n_states,
                n_actions=env.n_actions,
                sigma=1.0,  # No present bias
                gamma=0.95
            )
            
            # QH Q-learning (sigma = 0.7)
            agent_qh = QHQLearning(
                n_states=env.n_states,
                n_actions=env.n_actions,
                sigma=0.7,  # Present bias
                gamma=0.95
            )
            
            # Train both agents
            traditional_training = train_qh_qlearning(env, agent_traditional, 1000)
            qh_training = train_qh_qlearning(env, agent_qh, 1000)
            
            # Store results
            traditional_results.append({
                'final_performance': np.mean(traditional_training['episode_rewards'][-100:]),
                'policy': traditional_training['final_policy'],
                'values': traditional_training['final_values']
            })
            
            qh_results.append({
                'final_performance': np.mean(qh_training['episode_rewards'][-100:]),
                'policy': qh_training['final_policy'],
                'values': qh_training['final_values']
            })
        
        return {
            'traditional': traditional_results,
            'qh': qh_results,
            'comparison': {
                'traditional_mean': np.mean([r['final_performance'] for r in traditional_results]),
                'qh_mean': np.mean([r['final_performance'] for r in qh_results]),
                'traditional_std': np.std([r['final_performance'] for r in traditional_results]),
                'qh_std': np.std([r['final_performance'] for r in qh_results])
            }
        }
    
    def save_results(self, results: Dict, filename: str) -> None:
        """Save experiment results to file."""
        import pickle
        
        filepath = self.results_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")
    
    def generate_plots(self, results: Dict, plot_type: str) -> None:
        """
        Generate visualization plots for experiment results.
        
        Args:
            results: Experiment results dictionary
            plot_type: Type of plot to generate
        """
        if plot_type == 'performance_vs_sigma':
            self._plot_performance_vs_sigma(results)
        elif plot_type == 'convergence':
            self._plot_convergence(results)
        elif plot_type == 'policy_comparison':
            self._plot_policy_comparison(results)
    
    def _plot_performance_vs_sigma(self, results: Dict) -> None:
        """Plot performance vs sigma parameter."""
        sigma_values = [r['sigma'] for r in results['performance_metrics']]
        mean_performance = [r['mean_performance'] for r in results['performance_metrics']]
        std_performance = [r['std_performance'] for r in results['performance_metrics']]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(sigma_values, mean_performance, yerr=std_performance, 
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('Present-bias parameter (σ)')
        plt.ylabel('Average performance')
        plt.title('Performance vs Present-bias Parameter')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_vs_sigma.png', dpi=300)
        plt.show()
    
    def _plot_convergence(self, results: Dict) -> None:
        """Plot convergence metrics."""
        episodes = results['episodes']
        policy_changes = results['policy_changes']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episodes, policy_changes, 'b-', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Policy change rate')
        plt.title('Policy Convergence')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        value_estimates = np.array(results['value_function_estimates'])
        for i in range(min(5, value_estimates.shape[1])):
            plt.plot(episodes, value_estimates[:, i], label=f'State {i}')
        plt.xlabel('Episodes')
        plt.ylabel('Value estimate')
        plt.title('Value Function Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'convergence_analysis.png', dpi=300)
        plt.show()
    
    def _plot_policy_comparison(self, results: Dict) -> None:
        """Plot comparison between traditional and QH policies."""
        # This would visualize policy differences
        # Implementation depends on specific environment structure
        pass