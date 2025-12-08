"""
Szybki skrypt testowy dla porównania Standardowego vs QH

Uruchom aby zweryfikować poprawność działania frameworku porównawczego.
"""

import sys
sys.path.append('..')

from experiments.comparison_standard_vs_qh import MDPComparison, StandardQLearning
from models.mdp_environments import InventoryMDP
from algorithms.qh_qlearning import QHQLearning

def quick_test():
    """Szybki test z małym MDP."""
    print("=" * 70)
    print("SZYBKI TEST: Porównanie Standardowego vs QH Dyskontowania")
    print("=" * 70)
    print()
    
    # Tworzenie małego środowiska dla szybkiego testowania
    env = InventoryMDP(max_inventory=5, max_order=3)
    print(f"Środowisko: {env.n_states} stanów, {env.n_actions} akcji")
    
    # Test z silnym uprzedzeniem teraźniejszości
    alpha = 0.6
    print(f"Parametr uprzedzenia teraźniejszości: α = {alpha}")
    print()
    
    # Tworzenie obiektu porównawczego
    comparison = MDPComparison(
        env=env,
        alpha=alpha,
        beta=0.95,
        theta_step=0.1,
        epsilon=0.1
    )
    
    # Szybkie trenowanie
    print("Trenowanie przez 500 epizodów...")
    comparison.train(n_episodes=500, record_interval=100)
    print()
    
    # Porównanie wyników
    print("WYNIKI PORÓWNANIA:")
    print("-" * 70)
    
    policy_comp = comparison.compare_policies()
    print(f"Zgodność polityk: {policy_comp['agreement_percentage']:.1f}%")
    print(f"Stany, w których polityki się różnią: {policy_comp['different_states']}")
    print()
    
    value_comp = comparison.compare_values()
    print(f"Średnia różnica bezwzględna wartości: {value_comp['mean_abs_difference']:.4f}")
    print(f"Maksymalna różnica bezwzględna wartości: {value_comp['max_abs_difference']:.4f}")
    print()
    
    # Sprawdzenie spójności czasowej
    consistency = comparison.analyze_time_consistency(initial_state=2, horizon=5)
    print(f"Spójny czasowo: {consistency['is_time_consistent']}")
    print(f"Liczba niespójności: {len(consistency['inconsistencies'])}")
    print()
    
    # Wyświetlenie polityk
    print("Polityka standardowa:", policy_comp['standard_policy'])
    print("Polityka QH:        ", policy_comp['qh_policy'])
    print()
    
    print("=" * 70)
    print("TEST ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 70)
    print()
    print("Aby uruchomić pełne eksperymenty:")
    print("  - Wykonaj: python comparison_standard_vs_qh.py")
    print("  - Lub użyj: jupyter notebook ../../notebooks/standard_vs_qh_comparison.ipynb")


if __name__ == "__main__":
    quick_test()
