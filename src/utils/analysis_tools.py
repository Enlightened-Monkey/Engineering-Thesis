"""
Utility functions for quasi-hyperbolic discounting algorithms.

This module contains helper functions for data processing, visualization,
and mathematical computations related to QH discounting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

def calculate_qh_return(rewards: List[float], 
                       sigma: float, 
                       gamma: float,
                       t0: int = 0) -> float:
    """
    Calculate quasi-hyperbolic discounted return from a sequence of rewards.
    
    Args:
        rewards: List of rewards
        sigma: Present-bias parameter
        gamma: Exponential discount factor
        t0: Starting time index
        
    Returns:
        QH discounted return
    """
    if len(rewards) == 0:
        return 0.0
    
    # Immediate reward (no discounting)
    qh_return = rewards[0]
    
    # Future rewards (with QH discounting)
    for t, reward in enumerate(rewards[1:], start=1):
        qh_return += sigma * (gamma ** t) * reward
    
    return qh_return

def compare_discounting_schemes(rewards: List[float], 
                               sigma: float, 
                               gamma: float) -> Dict[str, float]:
    """
    Compare different discounting schemes on the same reward sequence.
    
    Args:
        rewards: Sequence of rewards
        sigma: Present-bias parameter
        gamma: Discount factor
        
    Returns:
        Dictionary with returns under different schemes
    """
    # Exponential discounting
    exp_return = sum(reward * (gamma ** t) for t, reward in enumerate(rewards))
    
    # Quasi-hyperbolic discounting
    qh_return = calculate_qh_return(rewards, sigma, gamma)
    
    # Hyperbolic discounting (for comparison)
    hyp_return = sum(reward / (1 + gamma * t) for t, reward in enumerate(rewards))
    
    return {
        'exponential': exp_return,
        'quasi_hyperbolic': qh_return,
        'hyperbolic': hyp_return
    }

def analyze_time_inconsistency(policy_sequence: List[np.ndarray],
                              state: int) -> Dict:
    """
    Analyze time inconsistency in a sequence of policies.
    
    Args:
        policy_sequence: List of policy arrays over time
        state: State to analyze
        
    Returns:
        Time inconsistency metrics
    """
    if len(policy_sequence) < 2:
        return {'inconsistency_rate': 0.0, 'changes': []}
    
    changes = []
    for t in range(1, len(policy_sequence)):
        prev_action = policy_sequence[t-1][state]
        curr_action = policy_sequence[t][state]
        if prev_action != curr_action:
            changes.append(t)
    
    inconsistency_rate = len(changes) / (len(policy_sequence) - 1)
    
    return {
        'inconsistency_rate': inconsistency_rate,
        'changes': changes,
        'total_periods': len(policy_sequence) - 1
    }

def compute_policy_similarity(policy1: np.ndarray, 
                             policy2: np.ndarray) -> float:
    """
    Compute similarity between two policies.
    
    Args:
        policy1: First policy
        policy2: Second policy
        
    Returns:
        Similarity score (0 = completely different, 1 = identical)
    """
    if len(policy1) != len(policy2):
        raise ValueError("Policies must have same length")
    
    return np.mean(policy1 == policy2)

def visualize_value_function(value_function: np.ndarray,
                           title: str = "Value Function",
                           save_path: Optional[str] = None) -> None:
    """
    Visualize value function as a bar chart.
    
    Args:
        value_function: Array of state values
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    states = range(len(value_function))
    plt.bar(states, value_function, alpha=0.7)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_policy(policy: np.ndarray,
                    n_actions: int,
                    title: str = "Policy",
                    save_path: Optional[str] = None) -> None:
    """
    Visualize policy as a heatmap or bar chart.
    
    Args:
        policy: Policy array (state -> action)
        n_actions: Number of possible actions
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 4))
    
    # Create policy matrix for visualization
    n_states = len(policy)
    policy_matrix = np.zeros((n_actions, n_states))
    
    for state, action in enumerate(policy):
        policy_matrix[int(action), state] = 1
    
    sns.heatmap(policy_matrix, 
                xticklabels=range(n_states),
                yticklabels=range(n_actions),
                cmap='Blues',
                cbar_kws={'label': 'Selected'})
    
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_learning_curves(results_dict: Dict[str, List[float]],
                           title: str = "Learning Curves",
                           save_path: Optional[str] = None) -> None:
    """
    Compare learning curves from multiple algorithms.
    
    Args:
        results_dict: Dictionary mapping algorithm names to reward lists
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    for name, rewards in results_dict.items():
        # Smooth the curve using moving average
        window_size = max(10, len(rewards) // 100)
        smoothed_rewards = pd.Series(rewards).rolling(window=window_size).mean()
        
        plt.plot(smoothed_rewards, label=name, linewidth=2)
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_experiment_summary(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from experiment results.
    
    Args:
        results: Experiment results dictionary
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    if 'performance_metrics' in results:
        for metric in results['performance_metrics']:
            summary_data.append({
                'sigma': metric['sigma'],
                'mean_performance': metric['mean_performance'],
                'std_performance': metric['std_performance'],
                'n_runs': len(metric['runs'])
            })
    
    return pd.DataFrame(summary_data)

def statistical_significance_test(group1: List[float], 
                                group2: List[float],
                                test_type: str = 'ttest') -> Dict:
    """
    Perform statistical significance test between two groups.
    
    Args:
        group1: First group of values
        group2: Second group of values
        test_type: Type of test ('ttest' or 'mannwhitney')
        
    Returns:
        Test results
    """
    from scipy import stats
    
    if test_type == 'ttest':
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Student's t-test"
    elif test_type == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'effect_size': (np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
    }

def export_results_to_latex(results_df: pd.DataFrame, 
                           filename: str = "results_table.tex") -> None:
    """
    Export results DataFrame to LaTeX table format.
    
    Args:
        results_df: Results DataFrame
        filename: Output filename
    """
    latex_table = results_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Experimental Results Summary",
        label="tab:results",
        position="h!"
    )
    
    with open(filename, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {filename}")

def validate_qh_parameters(sigma: float, gamma: float) -> bool:
    """
    Validate QH discounting parameters.
    
    Args:
        sigma: Present-bias parameter
        gamma: Exponential discount factor
        
    Returns:
        True if parameters are valid
    """
    if not (0 <= sigma <= 1):
        print(f"Warning: sigma = {sigma} is outside valid range [0, 1]")
        return False
    
    if not (0 <= gamma < 1):
        print(f"Warning: gamma = {gamma} is outside valid range [0, 1)")
        return False
    
    return True