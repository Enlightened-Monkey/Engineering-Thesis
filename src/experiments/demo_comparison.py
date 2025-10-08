#!/usr/bin/env python3
"""
Demo: Standard vs Quasi-Hyperbolic Discounting

Quick demonstration showing the key differences between
standard exponential and quasi-hyperbolic discounting.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from experiments.comparison_standard_vs_qh import MDPComparison
from models.mdp_environments import InventoryMDP


def demo_basic_comparison():
    """Basic comparison demonstration."""
    print("\n" + "="*70)
    print("DEMO: Standard vs Quasi-Hyperbolic Discounting")
    print("="*70 + "\n")
    
    # Setup
    env = InventoryMDP(max_inventory=10, max_order=5)
    sigma = 0.7
    
    print(f"Environment: Inventory MDP")
    print(f"  States: {env.n_states} (inventory levels 0-{env.n_states-1})")
    print(f"  Actions: {env.n_actions} (order quantities 0-{env.n_actions-1})")
    print(f"\nParameters:")
    print(f"  σ (present-bias): {sigma}")
    print(f"  γ (discount): 0.95")
    print(f"  Training episodes: 2000\n")
    
    # Create comparison
    comparison = MDPComparison(
        env=env,
        sigma=sigma,
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1
    )
    
    # Train
    print("Training both algorithms...")
    comparison.train(n_episodes=2000, record_interval=500)
    
    # Results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70 + "\n")
    
    # Policies
    policy_comp = comparison.compare_policies()
    print(f"Policy Agreement: {policy_comp['agreement_percentage']:.1f}%")
    print(f"Standard Policy: {policy_comp['standard_policy']}")
    print(f"QH Policy:       {policy_comp['qh_policy']}")
    
    if len(policy_comp['different_states']) > 0:
        print(f"\nStates with different actions: {policy_comp['different_states']}")
        for state in policy_comp['different_states']:
            std_action = policy_comp['standard_policy'][state]
            qh_action = policy_comp['qh_policy'][state]
            print(f"  State {state}: Standard orders {std_action}, QH orders {qh_action}")
    
    # Values
    print()
    value_comp = comparison.compare_values()
    print(f"Mean value difference: {value_comp['mean_abs_difference']:.4f}")
    print(f"Max value difference:  {value_comp['max_abs_difference']:.4f}")
    
    # Time consistency
    print()
    initial_state = env.n_states // 2
    consistency = comparison.analyze_time_consistency(initial_state, horizon=10)
    print(f"Time-consistency analysis (starting from state {initial_state}):")
    print(f"  Time-consistent: {consistency['is_time_consistent']}")
    print(f"  Inconsistencies found: {len(consistency['inconsistencies'])}")
    
    if not consistency['is_time_consistent']:
        print("\n  Details of time-inconsistencies:")
        for inc in consistency['inconsistencies'][:3]:  # Show first 3
            print(f"    Step {inc['time']}, State {inc['state']}: "
                  f"Precommitted action={inc['precommitted_action']}, "
                  f"Myopic action={inc['myopic_action']}")
    
    print("\n" + "="*70 + "\n")
    
    return comparison


def demo_sigma_sensitivity():
    """Demonstrate effect of sigma parameter."""
    print("\n" + "="*70)
    print("DEMO: Effect of Present-Bias Parameter σ")
    print("="*70 + "\n")
    
    env = InventoryMDP(max_inventory=8, max_order=4)
    sigma_values = [0.5, 0.7, 0.9, 1.0]
    results = []
    
    print(f"Testing σ values: {sigma_values}")
    print("(Lower σ = stronger present-bias)\n")
    
    for sigma in sigma_values:
        print(f"Training with σ = {sigma}...")
        comp = MDPComparison(env=env, sigma=sigma, gamma=0.95, alpha=0.1, epsilon=0.1)
        comp.train(n_episodes=1000, record_interval=1000)
        
        policy_comp = comp.compare_policies()
        value_comp = comp.compare_values()
        
        results.append({
            'sigma': sigma,
            'agreement': policy_comp['agreement_percentage'],
            'mean_value_diff': value_comp['mean_abs_difference']
        })
    
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"{'σ':<10} {'Policy Agreement':<20} {'Mean Value Diff':<20}")
    print("-"*70)
    for r in results:
        print(f"{r['sigma']:<10.1f} {r['agreement']:<20.1f}% {r['mean_value_diff']:<20.4f}")
    
    print("\nObservations:")
    print("  - σ = 1.0: Standard exponential discounting (baseline)")
    print("  - Lower σ: More present-bias, policies diverge more")
    print("  - Agreement decreases as present-bias increases")
    
    print("\n" + "="*70 + "\n")


def demo_visualization():
    """Create and save visualizations."""
    print("\n" + "="*70)
    print("DEMO: Creating Visualizations")
    print("="*70 + "\n")
    
    env = InventoryMDP(max_inventory=12, max_order=6)
    comparison = MDPComparison(env=env, sigma=0.6, gamma=0.95)
    
    print("Training for visualization...")
    comparison.train(n_episodes=3000, record_interval=100)
    
    output_path = '../../data/plots/demo_comparison.png'
    comparison.plot_comparison(save_path=output_path)
    print(f"\nVisualization saved to: {output_path}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo of Standard vs QH Discounting')
    parser.add_argument('--demo', type=str, default='basic',
                       choices=['basic', 'sigma', 'viz', 'all'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    if args.demo == 'basic' or args.demo == 'all':
        demo_basic_comparison()
    
    if args.demo == 'sigma' or args.demo == 'all':
        demo_sigma_sensitivity()
    
    if args.demo == 'viz' or args.demo == 'all':
        demo_visualization()
    
    print("\n✓ Demo completed!")
    print("\nNext steps:")
    print("  - Run full comparison: python comparison_standard_vs_qh.py")
    print("  - Interactive analysis: jupyter notebook ../../notebooks/standard_vs_qh_comparison.ipynb")
    print("  - Read documentation: ../../docs/COMPARISON_GUIDE.md")
