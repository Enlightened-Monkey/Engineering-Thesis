"""
Funkcje pomocnicze dla algorytmów dyskontowania quasi-hiperbolicznego.

Moduł zawiera funkcje pomocnicze do przetwarzania danych, wizualizacji
i obliczeń matematycznych związanych z dyskontowaniem QH.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

def calculate_qh_return(rewards: List[float], 
                       alpha: float, 
                       beta: float,
                       t0: int = 0) -> float:
    """
    Obliczenie zdyskontowanego zwrotu quasi-hiperbolicznego z sekwencji nagród.
    
    Args:
        rewards: Lista nagród
        alpha: Parametr uprzedzenia teraźniejszości
        beta: Współczynnik dyskontowania wykładniczego
        t0: Początkowy indeks czasowy
        
    Returns:
        Zdyskontowany zwrot QH
    """
    if len(rewards) == 0:
        return 0.0
    
    # Natychmiastowa nagroda (bez dyskontowania)
    qh_return = rewards[0]
    
    # Przyszłe nagrody (z dyskontowaniem QH)
    for t, reward in enumerate(rewards[1:], start=1):
        qh_return += alpha * (beta ** t) * reward
    
    return qh_return

def compare_discounting_schemes(rewards: List[float], 
                               alpha: float, 
                               beta: float) -> Dict[str, float]:
    """
    Porównanie różnych schematów dyskontowania na tej samej sekwencji nagród.
    
    Args:
        rewards: Sekwencja nagród
        alpha: Parametr uprzedzenia teraźniejszości
        beta: Współczynnik dyskontowania
        
    Returns:
        Słownik ze zwrotami dla różnych schematów
    """
    # Dyskontowanie wykładnicze
    exp_return = sum(reward * (beta ** t) for t, reward in enumerate(rewards))
    
    # Dyskontowanie quasi-hiperboliczne
    qh_return = calculate_qh_return(rewards, alpha, beta)
    
    # Dyskontowanie hiperboliczne (dla porównania)
    hyp_return = sum(reward / (1 + beta * t) for t, reward in enumerate(rewards))
    
    return {
        'exponential': exp_return,
        'quasi_hyperbolic': qh_return,
        'hyperbolic': hyp_return
    }

def analyze_time_inconsistency(policy_sequence: List[np.ndarray],
                              state: int) -> Dict:
    """
    Analiza niespójności czasowej w sekwencji polityk.
    
    Args:
        policy_sequence: Lista tablic polityk w czasie
        state: Stan do analizy
        
    Returns:
        Metryki niespójności czasowej
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
    Obliczenie podobieństwa między dwiema politykami.
    
    Args:
        policy1: Pierwsza polityka
        policy2: Druga polityka
        
    Returns:
        Współczynnik podobieństwa (0 = zupełnie różne, 1 = identyczne)
    """
    if len(policy1) != len(policy2):
        raise ValueError("Polityki muszą mieć tę samą długość")
    
    return np.mean(policy1 == policy2)

def visualize_value_function(value_function: np.ndarray,
                           title: str = "Funkcja wartości",
                           save_path: Optional[str] = None) -> None:
    """
    Wizualizacja funkcji wartości jako wykres słupkowy.
    
    Args:
        value_function: Tablica wartości stanów
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wykresu (opcjonalna)
    """
    plt.figure(figsize=(10, 6))
    states = range(len(value_function))
    plt.bar(states, value_function, alpha=0.7)
    plt.xlabel('Stan')
    plt.ylabel('Wartość')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_policy(policy: np.ndarray,
                    n_actions: int,
                    title: str = "Polityka",
                    save_path: Optional[str] = None) -> None:
    """
    Wizualizacja polityki jako mapa cieplna lub wykres słupkowy.
    
    Args:
        policy: Tablica polityki (stan -> akcja)
        n_actions: Liczba możliwych akcji
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 4))
    
    # Tworzenie macierzy polityki do wizualizacji
    n_states = len(policy)
    policy_matrix = np.zeros((n_actions, n_states))
    
    for state, action in enumerate(policy):
        policy_matrix[int(action), state] = 1
    
    sns.heatmap(policy_matrix, 
                xticklabels=range(n_states),
                yticklabels=range(n_actions),
                cmap='Blues',
                cbar_kws={'label': 'Wybrana'})
    
    plt.xlabel('Stan')
    plt.ylabel('Akcja')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_learning_curves(results_dict: Dict[str, List[float]],
                           title: str = "Krzywe uczenia",
                           save_path: Optional[str] = None) -> None:
    """
    Porównanie krzywych uczenia z wielu algorytmów.
    
    Args:
        results_dict: Słownik mapujący nazwy algorytmów na listy nagród
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu wykresu (opcjonalna)
    """
    plt.figure(figsize=(12, 8))
    
    for name, rewards in results_dict.items():
        # Wygładzenie krzywej średnią ruchomą
        window_size = max(10, len(rewards) // 100)
        smoothed_rewards = pd.Series(rewards).rolling(window=window_size).mean()
        
        plt.plot(smoothed_rewards, label=name, linewidth=2)
    
    plt.xlabel('Epizody')
    plt.ylabel('Średnia nagroda')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_experiment_summary(results: Dict) -> pd.DataFrame:
    """
    Tworzenie podsumowania DataFrame z wyników eksperymentu.
    
    Args:
        results: Słownik wyników eksperymentu
        
    Returns:
        Podsumowanie DataFrame
    """
    summary_data = []
    
    if 'performance_metrics' in results:
        for metric in results['performance_metrics']:
            summary_data.append({
                'alpha': metric['alpha'],
                'mean_performance': metric['mean_performance'],
                'std_performance': metric['std_performance'],
                'n_runs': len(metric['runs'])
            })
    
    return pd.DataFrame(summary_data)

def statistical_significance_test(group1: List[float], 
                                group2: List[float],
                                test_type: str = 'ttest') -> Dict:
    """
    Przeprowadzenie testu istotności statystycznej między dwoma grupami.
    
    Args:
        group1: Pierwsza grupa wartości
        group2: Druga grupa wartości
        test_type: Typ testu ('ttest' lub 'mannwhitney')
        
    Returns:
        Wyniki testu
    """
    from scipy import stats
    
    if test_type == 'ttest':
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Test t-Studenta"
    elif test_type == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Test Manna-Whitneya U"
    else:
        raise ValueError(f"Nieznany typ testu: {test_type}")
    
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
    Eksport wyników DataFrame do formatu tabeli LaTeX.
    
    Args:
        results_df: DataFrame z wynikami
        filename: Nazwa pliku wyjściowego
    """
    latex_table = results_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Podsumowanie wyników eksperymentalnych",
        label="tab:results",
        position="h!"
    )
    
    with open(filename, 'w') as f:
        f.write(latex_table)
    
    print(f"Tabela LaTeX zapisana do {filename}")

def validate_qh_parameters(alpha: float, beta: float) -> bool:
    """
    Walidacja parametrów dyskontowania QH.
    
    Args:
        alpha: Parametr uprzedzenia teraźniejszości
        beta: Współczynnik dyskontowania wykładniczego
        
    Returns:
        True jeśli parametry są prawidłowe
    """
    if not (0 <= alpha <= 1):
        print(f"Ostrzeżenie: alpha = {alpha} jest poza prawidłowym zakresem [0, 1]")
        return False
    
    if not (0 <= beta < 1):
        print(f"Ostrzeżenie: beta = {beta} jest poza prawidłowym zakresem [0, 1)")
        return False
    
    return True