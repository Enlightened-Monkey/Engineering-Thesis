"""
Quick test script for Standard vs QH comparison

Run this to verify the comparison framework works correctly.
"""

import sys
sys.path.append('..')

from experiments.comparison_standard_vs_qh import MDPComparison, StandardQLearning
from models.mdp_environments import InventoryMDP
from algorithms.qh_qlearning import QHQLearning

def quick_test():
    """Quick test with small MDP."""
    print("=" * 70)
    print("QUICK TEST: Standard vs QH Discounting Comparison")
    print("=" * 70)
    print()
    
    # Create small environment for fast testing
    env = InventoryMDP(max_inventory=5, max_order=3)
    print(f"Environment: {env.n_states} states, {env.n_actions} actions")
    
    # Test with strong present-bias
    sigma = 0.6
    print(f"Present-bias parameter: σ = {sigma}")
    print()
    
    # Create comparison object
    comparison = MDPComparison(
        env=env,
        sigma=sigma,
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1
    )
    
    # Quick training
    print("Training for 500 episodes...")
    comparison.train(n_episodes=500, record_interval=100)
    print()
    
    # Compare results
    print("COMPARISON RESULTS:")
    print("-" * 70)
    
    policy_comp = comparison.compare_policies()
    print(f"Policy agreement: {policy_comp['agreement_percentage']:.1f}%")
    print(f"States where policies differ: {policy_comp['different_states']}")
    print()
    
    value_comp = comparison.compare_values()
    print(f"Mean absolute value difference: {value_comp['mean_abs_difference']:.4f}")
    print(f"Max absolute value difference: {value_comp['max_abs_difference']:.4f}")
    print()
    
    # Time-consistency check
    consistency = comparison.analyze_time_consistency(initial_state=2, horizon=5)
    print(f"Time-consistent: {consistency['is_time_consistent']}")
    print(f"Number of inconsistencies: {len(consistency['inconsistencies'])}")
    print()
    
    # Display policies
    print("Standard Policy:", policy_comp['standard_policy'])
    print("QH Policy:      ", policy_comp['qh_policy'])
    print()
    
    print("=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("To run full experiments:")
    print("  - Execute: python comparison_standard_vs_qh.py")
    print("  - Or use: jupyter notebook ../../notebooks/standard_vs_qh_comparison.ipynb")


if __name__ == "__main__":
    quick_test()
