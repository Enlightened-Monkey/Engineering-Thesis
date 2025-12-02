#!/usr/bin/env python3
"""
Demo: Standardowe vs Quasi-Hiperboliczne Dyskontowanie

Szybka demonstracja pokazująca kluczowe różnice między
standardowym dyskontowaniem wykładniczym a quasi-hiperbolicznym.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from experiments.comparison_standard_vs_qh import MDPComparison
from models.mdp_environments import InventoryMDP


def demo_basic_comparison():
    """Podstawowa demonstracja porównania."""
    print("\n" + "="*70)
    print("DEMO: Standardowe vs Quasi-Hiperboliczne Dyskontowanie")
    print("="*70 + "\n")
    
    # Konfiguracja
    env = InventoryMDP(max_inventory=10, max_order=5)
    alpha = 0.7
    
    print(f"Środowisko: Inventory MDP")
    print(f"  Stany: {env.n_states} (poziomy zapasów 0-{env.n_states-1})")
    print(f"  Akcje: {env.n_actions} (ilości zamówień 0-{env.n_actions-1})")
    print(f"\nParametry:")
    print(f"  α (uprzedzenie teraźniejszości): {alpha}")
    print(f"  β (dyskontowanie): 0.95")
    print(f"  Epizody treningowe: 2000\n")
    
    # Tworzenie porównania
    comparison = MDPComparison(
        env=env,
        alpha=alpha,
        beta=0.95,
        theta_step=0.1,
        epsilon=0.1
    )
    
    # Trenowanie
    print("Trenowanie obu algorytmów...")
    comparison.train(n_episodes=2000, record_interval=500)
    
    # Wyniki
    print("\n" + "-"*70)
    print("WYNIKI")
    print("-"*70 + "\n")
    
    # Polityki
    policy_comp = comparison.compare_policies()
    print(f"Zgodność polityk: {policy_comp['agreement_percentage']:.1f}%")
    print(f"Polityka standardowa: {policy_comp['standard_policy']}")
    print(f"Polityka QH:         {policy_comp['qh_policy']}")
    
    if len(policy_comp['different_states']) > 0:
        print(f"\nStany z różnymi akcjami: {policy_comp['different_states']}")
        for state in policy_comp['different_states']:
            std_action = policy_comp['standard_policy'][state]
            qh_action = policy_comp['qh_policy'][state]
            print(f"  Stan {state}: Standardowy zamawia {std_action}, QH zamawia {qh_action}")
    
    # Wartości
    print()
    value_comp = comparison.compare_values()
    print(f"Średnia różnica wartości: {value_comp['mean_abs_difference']:.4f}")
    print(f"Maksymalna różnica wartości: {value_comp['max_abs_difference']:.4f}")
    
    # Spójność czasowa
    print()
    initial_state = env.n_states // 2
    consistency = comparison.analyze_time_consistency(initial_state, horizon=10)
    print(f"Analiza spójności czasowej (zaczynając od stanu {initial_state}):")
    print(f"  Spójny czasowo: {consistency['is_time_consistent']}")
    print(f"  Znalezione niespójności: {len(consistency['inconsistencies'])}")
    
    if not consistency['is_time_consistent']:
        print("\n  Szczegóły niespójności czasowych:")
        for inc in consistency['inconsistencies'][:3]:  # Pokaż pierwsze 3
            print(f"    Krok {inc['time']}, Stan {inc['state']}: "
                  f"Akcja zobowiązana={inc['precommitted_action']}, "
                  f"Akcja krótkowzroczna={inc['myopic_action']}")
    
    print("\n" + "="*70 + "\n")
    
    return comparison


def demo_alpha_sensitivity():
    """Demonstracja wpływu parametru alpha."""
    print("\n" + "="*70)
    print("DEMO: Wpływ parametru uprzedzenia teraźniejszości α")
    print("="*70 + "\n")
    
    env = InventoryMDP(max_inventory=8, max_order=4)
    alpha_values = [0.5, 0.7, 0.9, 1.0]
    results = []
    
    print(f"Testowanie wartości α: {alpha_values}")
    print("(Niższe α = silniejsze uprzedzenie teraźniejszości)\n")
    
    for alpha in alpha_values:
        print(f"Trenowanie z α = {alpha}...")
        comp = MDPComparison(env=env, alpha=alpha, beta=0.95, theta_step=0.1, epsilon=0.1)
        comp.train(n_episodes=1000, record_interval=1000)
        
        policy_comp = comp.compare_policies()
        value_comp = comp.compare_values()
        
        results.append({
            'alpha': alpha,
            'agreement': policy_comp['agreement_percentage'],
            'mean_value_diff': value_comp['mean_abs_difference']
        })
    
    print("\n" + "-"*70)
    print("WYNIKI")
    print("-"*70)
    print(f"{'α':<10} {'Zgodność polityk':<20} {'Średnia różnica wartości':<20}")
    print("-"*70)
    for r in results:
        print(f"{r['alpha']:<10.1f} {r['agreement']:<20.1f}% {r['mean_value_diff']:<20.4f}")
    
    print("\nObserwacje:")
    print("  - α = 1.0: Standardowe dyskontowanie wykładnicze (punkt odniesienia)")
    print("  - Niższe α: Więcej uprzedzenia teraźniejszości, polityki bardziej się różnią")
    print("  - Zgodność maleje wraz ze wzrostem uprzedzenia teraźniejszości")
    
    print("\n" + "="*70 + "\n")


def demo_visualization():
    """Tworzenie i zapisywanie wizualizacji."""
    print("\n" + "="*70)
    print("DEMO: Tworzenie wizualizacji")
    print("="*70 + "\n")
    
    env = InventoryMDP(max_inventory=12, max_order=6)
    comparison = MDPComparison(env=env, alpha=0.6, beta=0.95)
    
    print("Trenowanie dla wizualizacji...")
    comparison.train(n_episodes=3000, record_interval=100)
    
    output_path = '../../data/plots/demo_comparison.png'
    comparison.plot_comparison(save_path=output_path)
    print(f"\nWizualizacja zapisana do: {output_path}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Standardowego vs QH Dyskontowania')
    parser.add_argument('--demo', type=str, default='basic',
                       choices=['basic', 'alpha', 'viz', 'all'],
                       help='Które demo uruchomić')
    
    args = parser.parse_args()
    
    if args.demo == 'basic' or args.demo == 'all':
        demo_basic_comparison()
    
    if args.demo == 'alpha' or args.demo == 'all':
        demo_alpha_sensitivity()
    
    if args.demo == 'viz' or args.demo == 'all':
        demo_visualization()
    
    print("\n✓ Demo zakończone!")
    print("\nNastępne kroki:")
    print("  - Pełne porównanie: python comparison_standard_vs_qh.py")
    print("  - Interaktywna analiza: jupyter notebook ../../notebooks/standard_vs_qh_comparison.ipynb")
    print("  - Dokumentacja: ../../docs/COMPARISON_GUIDE.md")
