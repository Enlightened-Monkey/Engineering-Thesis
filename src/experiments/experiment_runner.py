"""
Framework eksperymentalny dla algorytmów dyskontowania QH

Moduł dostarcza narzędzia do przeprowadzania kompleksowych eksperymentów
walidujących i porównujących algorytmy dyskontowania quasi-hiperbolicznego.
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
    Główna klasa do uruchamiania i zarządzania eksperymentami.
    """
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_inventory_experiment(self, 
                                alpha_values: List[float],
                                n_runs: int = 5,
                                n_episodes: int = 2000) -> Dict:
        """
        Uruchomienie eksperymentu zarządzania zapasami z różnymi wartościami alpha.
        
        Args:
            alpha_values: Lista parametrów uprzedzenia teraźniejszości do testowania
            n_runs: Liczba niezależnych uruchomień na parametr
            n_episodes: Epizody na jedno uruchomienie treningowe
            
        Returns:
            Wyniki eksperymentu
        """
        print("Uruchamianie eksperymentów zarządzania zapasami...")
        
        results = {
            'alpha_values': alpha_values,
            'convergence_data': [],
            'policy_comparison': [],
            'performance_metrics': []
        }
        
        for alpha in alpha_values:
            print(f"Testowanie alpha = {alpha:.2f}")
            
            alpha_results = {
                'alpha': alpha,
                'runs': [],
                'mean_performance': 0,
                'std_performance': 0
            }
            
            run_performances = []
            
            for run in range(n_runs):
                # Tworzenie środowiska i agenta
                env = InventoryMDP(max_inventory=15, max_order=8)
                agent = QHQLearning(
                    n_states=env.n_states,
                    n_actions=env.n_actions,
                    alpha=alpha,
                    beta=0.95,
                    theta_step=0.1,
                    epsilon=0.2
                )
                
                # Trenowanie agenta
                start_time = time.time()
                training_results = train_qh_qlearning(env, agent, n_episodes)
                training_time = time.time() - start_time
                
                # Ocena końcowej polityki
                final_performance = np.mean(training_results['episode_rewards'][-100:])
                run_performances.append(final_performance)
                
                alpha_results['runs'].append({
                    'run_id': run,
                    'training_time': training_time,
                    'final_performance': final_performance,
                    'episode_rewards': training_results['episode_rewards'],
                    'final_policy': training_results['final_policy']
                })
            
            alpha_results['mean_performance'] = np.mean(run_performances)
            alpha_results['std_performance'] = np.std(run_performances)
            
            results['performance_metrics'].append(alpha_results)
        
        return results
    
    def run_convergence_analysis(self, 
                                env_type: str = 'inventory',
                                alpha: float = 0.8) -> Dict:
        """
        Analiza właściwości zbieżności algorytmów QH.
        
        Args:
            env_type: Typ środowiska ('inventory' lub 'gridworld')
            alpha: Parametr uprzedzenia teraźniejszości
            
        Returns:
            Wyniki analizy zbieżności
        """
        print(f"Uruchamianie analizy zbieżności dla środowiska {env_type}...")
        
        # Tworzenie środowiska
        if env_type == 'inventory':
            env = InventoryMDP(max_inventory=10, max_order=5)
        else:
            env = GridWorldMDP(width=4, height=4)
        
        # Tworzenie agenta
        agent = QHQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=alpha,
            beta=0.95,
            theta_step=0.05,
            epsilon=0.1
        )
        
        # Śledzenie metryk zbieżności podczas treningu
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
            
            # Okresowe sprawdzanie metryk zbieżności
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
        Porównanie wydajności tradycyjnego dyskontowania wykładniczego vs QH.
        
        Args:
            env_type: Typ środowiska
            n_runs: Liczba uruchomień porównawczych
            
        Returns:
            Wyniki porównania
        """
        print(f"Porównanie tradycyjnego vs QH dyskontowania na środowisku {env_type}...")
        
        # Tworzenie środowiska
        if env_type == 'inventory':
            env = InventoryMDP()
        else:
            env = GridWorldMDP()
        
        traditional_results = []
        qh_results = []
        
        for run in range(n_runs):
            # Tradycyjny Q-learning (alpha = 1.0)
            agent_traditional = QHQLearning(
                n_states=env.n_states,
                n_actions=env.n_actions,
                alpha=1.0,  # Brak uprzedzenia teraźniejszości
                beta=0.95
            )
            
            # QH Q-learning (alpha = 0.7)
            agent_qh = QHQLearning(
                n_states=env.n_states,
                n_actions=env.n_actions,
                alpha=0.7,  # Uprzedzenie teraźniejszości
                beta=0.95
            )
            
            # Trenowanie obu agentów
            traditional_training = train_qh_qlearning(env, agent_traditional, 1000)
            qh_training = train_qh_qlearning(env, agent_qh, 1000)
            
            # Zapisanie wyników
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
        """Zapis wyników eksperymentu do pliku."""
        import pickle
        
        filepath = self.results_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Wyniki zapisane do {filepath}")
    
    def generate_plots(self, results: Dict, plot_type: str) -> None:
        """
        Generowanie wykresów wizualizacyjnych dla wyników eksperymentu.
        
        Args:
            results: Słownik wyników eksperymentu
            plot_type: Typ wykresu do wygenerowania
        """
        if plot_type == 'performance_vs_alpha':
            self._plot_performance_vs_alpha(results)
        elif plot_type == 'convergence':
            self._plot_convergence(results)
        elif plot_type == 'policy_comparison':
            self._plot_policy_comparison(results)
    
    def _plot_performance_vs_alpha(self, results: Dict) -> None:
        """Wykres wydajności vs parametr alpha."""
        alpha_values = [r['alpha'] for r in results['performance_metrics']]
        mean_performance = [r['mean_performance'] for r in results['performance_metrics']]
        std_performance = [r['std_performance'] for r in results['performance_metrics']]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(alpha_values, mean_performance, yerr=std_performance, 
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('Parametr uprzedzenia teraźniejszości (α)')
        plt.ylabel('Średnia wydajność')
        plt.title('Wydajność vs parametr uprzedzenia teraźniejszości')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_vs_alpha.png', dpi=300)
        plt.show()
    
    def _plot_convergence(self, results: Dict) -> None:
        """Wykres metryk zbieżności."""
        episodes = results['episodes']
        policy_changes = results['policy_changes']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episodes, policy_changes, 'b-', linewidth=2)
        plt.xlabel('Epizody')
        plt.ylabel('Współczynnik zmiany polityki')
        plt.title('Zbieżność polityki')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        value_estimates = np.array(results['value_function_estimates'])
        for i in range(min(5, value_estimates.shape[1])):
            plt.plot(episodes, value_estimates[:, i], label=f'Stan {i}')
        plt.xlabel('Epizody')
        plt.ylabel('Estymata wartości')
        plt.title('Zbieżność funkcji wartości')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'convergence_analysis.png', dpi=300)
        plt.show()
    
    def _plot_policy_comparison(self, results: Dict) -> None:
        """Wykres porównania tradycyjnych i QH polityk."""
        # Wizualizacja różnic w politykach
        # Implementacja zależy od konkretnej struktury środowiska
        pass