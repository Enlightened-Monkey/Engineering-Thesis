#!/usr/bin/env python3
"""
Główny skrypt do uruchamiania eksperymentów z dyskontowaniem quasi-hiperbolicznym.

Użycie:
    python main.py --experiment inventory --alpha 0.8 --runs 5
    python main.py --experiment convergence --env gridworld
    python main.py --experiment comparison
"""

import argparse
import sys
from pathlib import Path

# Dodanie src do ścieżki
sys.path.append(str(Path(__file__).parent / 'src'))

from experiments.experiment_runner import ExperimentRunner
from models.mdp_environments import InventoryMDP, GridWorldMDP

def main():
    parser = argparse.ArgumentParser(description='Uruchom eksperymenty dyskontowania QH')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['inventory', 'convergence', 'comparison'],
                       help='Typ eksperymentu do uruchomienia')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Parametr uprzedzenia teraźniejszości (domyślnie: 0.8)')
    parser.add_argument('--beta', type=float, default=0.95,
                       help='Współczynnik dyskontowania (domyślnie: 0.95)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Liczba uruchomień eksperymentu (domyślnie: 5)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Epizody na uruchomienie (domyślnie: 1000)')
    parser.add_argument('--env', type=str, default='inventory',
                       choices=['inventory', 'gridworld'],
                       help='Typ środowiska (domyślnie: inventory)')
    parser.add_argument('--output', type=str, default='data/results',
                       help='Katalog wyjściowy (domyślnie: data/results)')
    
    args = parser.parse_args()
    
    print(f"Uruchamianie eksperymentu {args.experiment} z α={args.alpha}, β={args.beta}")
    print(f"Środowisko: {args.env}, Uruchomienia: {args.runs}, Epizody: {args.episodes}")
    print("-" * 60)
    
    # Tworzenie runnera eksperymentów
    runner = ExperimentRunner(results_dir=args.output)
    
    if args.experiment == 'inventory':
        # Uruchomienie eksperymentu zarządzania zapasami
        alpha_values = [0.5, 0.7, 0.8, 0.9, 1.0]
        results = runner.run_inventory_experiment(
            alpha_values=alpha_values,
            n_runs=args.runs,
            n_episodes=args.episodes
        )
        
        # Zapisanie wyników
        runner.save_results(results, 'inventory_experiment')
        
        # Generowanie wykresów
        runner.generate_plots(results, 'performance_vs_alpha')
        
    elif args.experiment == 'convergence':
        # Uruchomienie analizy zbieżności
        results = runner.run_convergence_analysis(
            env_type=args.env,
            alpha=args.alpha
        )
        
        # Zapisanie wyników
        runner.save_results(results, f'convergence_{args.env}')
        
        # Generowanie wykresów
        runner.generate_plots(results, 'convergence')
        
    elif args.experiment == 'comparison':
        # Uruchomienie porównania tradycyjny vs QH
        results = runner.compare_traditional_vs_qh(
            env_type=args.env,
            n_runs=args.runs
        )
        
        # Zapisanie wyników
        runner.save_results(results, f'comparison_{args.env}')
        
        # Wyświetlenie podsumowania
        comparison = results['comparison']
        print("\nWyniki porównania:")
        print(f"Tradycyjny (α=1.0): {comparison['traditional_mean']:.3f} ± {comparison['traditional_std']:.3f}")
        print(f"QH (α=0.7): {comparison['qh_mean']:.3f} ± {comparison['qh_std']:.3f}")
        
        # Test istotności statystycznej
        from src.utils.analysis_tools import statistical_significance_test
        traditional_perfs = [r['final_performance'] for r in results['traditional']]
        qh_perfs = [r['final_performance'] for r in results['qh']]
        
        test_result = statistical_significance_test(traditional_perfs, qh_perfs)
        print(f"\nTest statystyczny: {test_result['test_name']}")
        print(f"p-wartość: {test_result['p_value']:.6f}")
        print(f"Istotny: {test_result['significant']}")
    
    print(f"\nEksperyment zakończony. Wyniki zapisane do {args.output}/")

if __name__ == '__main__':
    main()