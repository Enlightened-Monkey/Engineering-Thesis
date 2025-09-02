#!/usr/bin/env python3
"""
Main script for running quasi-hyperbolic discounting experiments.

Usage:
    python main.py --experiment inventory --sigma 0.8 --runs 5
    python main.py --experiment convergence --env gridworld
    python main.py --experiment comparison
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from experiments.experiment_runner import ExperimentRunner
from models.mdp_environments import InventoryMDP, GridWorldMDP

def main():
    parser = argparse.ArgumentParser(description='Run QH discounting experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['inventory', 'convergence', 'comparison'],
                       help='Type of experiment to run')
    parser.add_argument('--sigma', type=float, default=0.8,
                       help='Present-bias parameter (default: 0.8)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of experiment runs (default: 5)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per run (default: 1000)')
    parser.add_argument('--env', type=str, default='inventory',
                       choices=['inventory', 'gridworld'],
                       help='Environment type (default: inventory)')
    parser.add_argument('--output', type=str, default='data/results',
                       help='Output directory (default: data/results)')
    
    args = parser.parse_args()
    
    print(f"Running {args.experiment} experiment with σ={args.sigma}, γ={args.gamma}")
    print(f"Environment: {args.env}, Runs: {args.runs}, Episodes: {args.episodes}")
    print("-" * 60)
    
    # Create experiment runner
    runner = ExperimentRunner(results_dir=args.output)
    
    if args.experiment == 'inventory':
        # Run inventory management experiment
        sigma_values = [0.5, 0.7, 0.8, 0.9, 1.0]
        results = runner.run_inventory_experiment(
            sigma_values=sigma_values,
            n_runs=args.runs,
            n_episodes=args.episodes
        )
        
        # Save results
        runner.save_results(results, 'inventory_experiment')
        
        # Generate plots
        runner.generate_plots(results, 'performance_vs_sigma')
        
    elif args.experiment == 'convergence':
        # Run convergence analysis
        results = runner.run_convergence_analysis(
            env_type=args.env,
            sigma=args.sigma
        )
        
        # Save results
        runner.save_results(results, f'convergence_{args.env}')
        
        # Generate plots
        runner.generate_plots(results, 'convergence')
        
    elif args.experiment == 'comparison':
        # Run traditional vs QH comparison
        results = runner.compare_traditional_vs_qh(
            env_type=args.env,
            n_runs=args.runs
        )
        
        # Save results
        runner.save_results(results, f'comparison_{args.env}')
        
        # Print summary
        comparison = results['comparison']
        print("\nComparison Results:")
        print(f"Traditional (σ=1.0): {comparison['traditional_mean']:.3f} ± {comparison['traditional_std']:.3f}")
        print(f"QH (σ=0.7): {comparison['qh_mean']:.3f} ± {comparison['qh_std']:.3f}")
        
        # Statistical significance test
        from src.utils.analysis_tools import statistical_significance_test
        traditional_perfs = [r['final_performance'] for r in results['traditional']]
        qh_perfs = [r['final_performance'] for r in results['qh']]
        
        test_result = statistical_significance_test(traditional_perfs, qh_perfs)
        print(f"\nStatistical test: {test_result['test_name']}")
        print(f"p-value: {test_result['p_value']:.6f}")
        print(f"Significant: {test_result['significant']}")
    
    print(f"\nExperiment completed. Results saved to {args.output}/")

if __name__ == '__main__':
    main()