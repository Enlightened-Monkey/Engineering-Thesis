"""
Porównanie Standardowego i Quasi-Hiperbolicznego Dyskontowania w MDP

Moduł porównuje zachowanie i wydajność:
1. Standardowego dyskontowania wykładniczego (klasyczne RL)
2. Dyskontowania quasi-hiperbolicznego (precommitted agent)

Porównanie obejmuje:
- Różnice w politykach
- Różnice w funkcjach wartości
- Akumulację nagród w czasie
- Analizę spójności czasowej
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('..')

from algorithms.qh_qlearning import QHQLearning
from models.mdp_environments import MDPEnvironment


class StandardQLearning:
    """
    Standardowy Q-Learning z dyskontowaniem wykładniczym.
    
    Służy jako punkt odniesienia do porównania z QH Q-Learning.
    """
    
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 beta: float = 0.95,
                 theta_step: float = 0.1,
                 epsilon: float = 0.1):
        """
        Inicjalizacja standardowego Q-Learning.
        
        Args:
            n_states: Liczba stanów
            n_actions: Liczba akcji
            beta: Współczynnik dyskontowania
            theta_step: Krok uczenia
            epsilon: Współczynnik eksploracji
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.beta = beta
        self.theta_step = theta_step
        self.epsilon = epsilon
        
        # Inicjalizacja tablicy Q
        self.q_table = np.zeros((n_states, n_actions))
        
    def select_action(self, state: int) -> int:
        """Wybór akcji metodą ε-zachłanną."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Standardowa aktualizacja Q-Learning."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.beta * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.theta_step * td_error
    
    def get_policy(self) -> np.ndarray:
        """Ekstrakcja polityki zachłannej z tablicy Q."""
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """Obliczenie funkcji wartości stanów."""
        return np.max(self.q_table, axis=1)


class MDPComparison:
    """
    Framework do porównywania podejść standardowego i QH dyskontowania.
    """
    
    def __init__(self, 
                 env: MDPEnvironment,
                 alpha: float = 0.8,
                 beta: float = 0.95,
                 theta_step: float = 0.1,
                 epsilon: float = 0.1):
        """
        Inicjalizacja frameworku porównawczego.
        
        Args:
            env: Środowisko MDP do testowania
            alpha: Parametr uprzedzenia teraźniejszości dla QH dyskontowania
            beta: Współczynnik dyskontowania (używany przez oba algorytmy)
            theta_step: Krok uczenia
            epsilon: Współczynnik eksploracji
        """
        self.env = env
        self.alpha = alpha
        self.beta = beta
        
        # Inicjalizacja obu algorytmów
        self.standard_qlearning = StandardQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            beta=beta,
            theta_step=theta_step,
            epsilon=epsilon
        )
        
        self.qh_qlearning = QHQLearning(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=alpha,
            beta=beta,
            theta_step=theta_step,
            epsilon=epsilon
        )
        
        # Przechowywanie wyników
        self.standard_rewards: List[float] = []
        self.qh_rewards: List[float] = []
        self.standard_values: List[np.ndarray] = []
        self.qh_values: List[np.ndarray] = []
        
    def train_episode(self, max_steps: int = 100) -> Tuple[float, float]:
        """
        Trenowanie obu algorytmów przez jeden epizod.
        
        Args:
            max_steps: Maksymalna liczba kroków na epizod
            
        Returns:
            Krotka (nagroda_standardowa, nagroda_qh)
        """
        # Trenowanie standardowego Q-learning
        state = self.env.reset()
        standard_total_reward = 0.0
        
        for _ in range(max_steps):
            action = self.standard_qlearning.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.standard_qlearning.update(state, action, reward, next_state, done)
            standard_total_reward += reward
            
            if done:
                break
            state = next_state
        
        # Trenowanie QH Q-learning
        state = self.env.reset()
        qh_total_reward = 0.0
        
        for _ in range(max_steps):
            action = self.qh_qlearning.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.qh_qlearning.update(state, action, reward, next_state)
            qh_total_reward += reward
            
            if done:
                break
            state = next_state
        
        return standard_total_reward, qh_total_reward
    
    def train(self, n_episodes: int = 1000, record_interval: int = 100) -> None:
        """
        Trenowanie obu algorytmów przez wiele epizodów.
        
        Args:
            n_episodes: Liczba epizodów treningowych
            record_interval: Interwał zapisu metryk
        """
        print(f"Trenowanie obu algorytmów przez {n_episodes} epizodów...")
        
        for episode in range(n_episodes):
            standard_reward, qh_reward = self.train_episode()
            
            if episode % record_interval == 0:
                self.standard_rewards.append(standard_reward)
                self.qh_rewards.append(qh_reward)
                self.standard_values.append(self.standard_qlearning.get_value_function().copy())
                self.qh_values.append(self.qh_qlearning.get_value_function().copy())
                
                print(f"Epizod {episode}: Standardowy={standard_reward:.2f}, QH={qh_reward:.2f}")
    
    def compare_policies(self) -> Dict[str, np.ndarray]:
        """
        Porównanie nauczonych polityk.
        
        Returns:
            Słownik zawierający obie polityki i ich różnice
        """
        standard_policy = self.standard_qlearning.get_policy()
        qh_policy = self.qh_qlearning.get_policy()
        
        # Znalezienie stanów, w których polityki się różnią
        policy_diff = (standard_policy != qh_policy).astype(int)
        
        return {
            'standard_policy': standard_policy,
            'qh_policy': qh_policy,
            'different_states': np.where(policy_diff)[0],
            'agreement_percentage': 100 * (1 - policy_diff.mean())
        }
    
    def compare_values(self) -> Dict[str, np.ndarray]:
        """
        Porównanie nauczonych funkcji wartości.
        
        Returns:
            Słownik zawierający funkcje wartości i ich różnice
        """
        standard_values = self.standard_qlearning.get_value_function()
        qh_values = self.qh_qlearning.get_value_function()
        
        value_diff = standard_values - qh_values
        
        return {
            'standard_values': standard_values,
            'qh_values': qh_values,
            'value_difference': value_diff,
            'mean_abs_difference': np.mean(np.abs(value_diff)),
            'max_abs_difference': np.max(np.abs(value_diff))
        }
    
    def analyze_time_consistency(self, initial_state: int, horizon: int = 10) -> Dict:
        """
        Analiza spójności czasowej poprzez sprawdzenie, czy agent chciałby
        odstąpić od swojej zobowiązanej polityki w każdym kroku czasowym.
        
        Args:
            initial_state: Stan początkowy
            horizon: Horyzont planowania
            
        Returns:
            Słownik z analizą spójności czasowej
        """
        # Pobierz zobowiązaną politykę QH
        qh_policy = self.qh_qlearning.get_policy()
        
        # Symuluj trajektorię pod zobowiązaną polityką
        state = initial_state
        trajectory = [state]
        actions = []
        
        for t in range(horizon):
            action = qh_policy[state]
            actions.append(action)
            
            # Pobierz następny stan (deterministycznie dla analizy)
            next_state, _, done, _ = self.env.step(action)
            trajectory.append(next_state)
            
            if done:
                break
            state = next_state
        
        # Sprawdź niespójności czasowe
        # W każdym kroku, czy agent wolałby inną akcję?
        inconsistencies = []
        for t, state in enumerate(trajectory[:-1]):
            precommitted_action = qh_policy[state]
            
            # Jaką akcję wybrałby krótkowzroczny agent? (alpha zastosowane do natychmiastowej nagrody)
            myopic_values = self.qh_qlearning.Q[state]
            myopic_action = np.argmax(myopic_values)
            
            if precommitted_action != myopic_action:
                inconsistencies.append({
                    'time': t,
                    'state': state,
                    'precommitted_action': precommitted_action,
                    'myopic_action': myopic_action
                })
        
        return {
            'trajectory': trajectory,
            'actions': actions,
            'inconsistencies': inconsistencies,
            'is_time_consistent': len(inconsistencies) == 0
        }
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Tworzenie wizualizacji porównującej oba podejścia.
        
        Args:
            save_path: Ścieżka do zapisu wykresu (jeśli None, wyświetla na ekranie)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Krzywe uczenia
        ax = axes[0, 0]
        episodes = np.arange(len(self.standard_rewards)) * 100
        ax.plot(episodes, self.standard_rewards, label='Standardowy Q-Learning', marker='o')
        ax.plot(episodes, self.qh_rewards, label='QH Q-Learning', marker='s')
        ax.set_xlabel('Epizod')
        ax.set_ylabel('Całkowita nagroda')
        ax.set_title('Porównanie krzywych uczenia')
        ax.legend()
        ax.grid(True)
        
        # 2. Porównanie funkcji wartości
        ax = axes[0, 1]
        comparison = self.compare_values()
        states = np.arange(self.env.n_states)
        ax.plot(states, comparison['standard_values'], label='Standardowy', marker='o')
        ax.plot(states, comparison['qh_values'], label='QH', marker='s')
        ax.set_xlabel('Stan')
        ax.set_ylabel('Wartość')
        ax.set_title('Porównanie funkcji wartości')
        ax.legend()
        ax.grid(True)
        
        # 3. Porównanie polityk
        ax = axes[1, 0]
        policy_comp = self.compare_policies()
        x = np.arange(self.env.n_states)
        width = 0.35
        ax.bar(x - width/2, policy_comp['standard_policy'], width, label='Standardowy', alpha=0.8)
        ax.bar(x + width/2, policy_comp['qh_policy'], width, label='QH', alpha=0.8)
        ax.set_xlabel('Stan')
        ax.set_ylabel('Akcja')
        ax.set_title(f"Porównanie polityk (Zgodność: {policy_comp['agreement_percentage']:.1f}%)")
        ax.legend()
        ax.grid(True, axis='y')
        
        # 4. Mapa cieplna różnic wartości
        ax = axes[1, 1]
        value_diff = comparison['value_difference']
        ax.bar(states, value_diff, color=['red' if v < 0 else 'green' for v in value_diff])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Stan')
        ax.set_ylabel('Różnica wartości (Standardowy - QH)')
        ax.set_title(f"Różnica wartości (Średnia: {comparison['mean_abs_difference']:.2f})")
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wykres zapisano do {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """
        Generowanie raportu tekstowego podsumowującego porównanie.
        
        Returns:
            Sformatowany ciąg znaków raportu
        """
        policy_comp = self.compare_policies()
        value_comp = self.compare_values()
        
        report = "=" * 70 + "\n"
        report += "PORÓWNANIE: Standardowe vs Quasi-Hiperboliczne Dyskontowanie\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Środowisko: {self.env.__class__.__name__}\n"
        report += f"Stany: {self.env.n_states}, Akcje: {self.env.n_actions}\n"
        report += f"Beta (dyskontowanie): {self.beta}\n"
        report += f"Alpha (uprzedzenie teraźniejszości): {self.alpha}\n"
        report += f"Przeszkolone epizody: {len(self.standard_rewards) * 100}\n\n"
        
        report += "PORÓWNANIE POLITYK\n"
        report += "-" * 70 + "\n"
        report += f"Zgodność: {policy_comp['agreement_percentage']:.1f}%\n"
        report += f"Różne stany: {len(policy_comp['different_states'])}\n"
        if len(policy_comp['different_states']) > 0:
            report += f"Stany, w których polityki się różnią: {policy_comp['different_states']}\n"
        report += "\n"
        
        report += "PORÓWNANIE FUNKCJI WARTOŚCI\n"
        report += "-" * 70 + "\n"
        report += f"Średnia różnica bezwzględna: {value_comp['mean_abs_difference']:.4f}\n"
        report += f"Maksymalna różnica bezwzględna: {value_comp['max_abs_difference']:.4f}\n"
        report += "\n"
        
        report += "PORÓWNANIE WYDAJNOŚCI\n"
        report += "-" * 70 + "\n"
        if len(self.standard_rewards) > 0:
            final_standard = self.standard_rewards[-1]
            final_qh = self.qh_rewards[-1]
            report += f"Nagroda końcowego epizodu (Standardowy): {final_standard:.2f}\n"
            report += f"Nagroda końcowego epizodu (QH): {final_qh:.2f}\n"
            report += f"Różnica: {final_standard - final_qh:.2f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report


def run_comparison_example():
    """Przykład użycia frameworku porównawczego."""
    from models.mdp_environments import InventoryMDP
    
    # Tworzenie środowiska
    env = InventoryMDP(max_inventory=10, max_order=5)
    
    # Tworzenie obiektu porównawczego
    comparison = MDPComparison(
        env=env,
        alpha=0.7,  # Parametr uprzedzenia teraźniejszości
        beta=0.95,
        theta_step=0.1,
        epsilon=0.1
    )
    
    # Trenowanie obu algorytmów
    comparison.train(n_episodes=5000, record_interval=100)
    
    # Wyświetlenie raportu
    print(comparison.generate_report())
    
    # Analiza spójności czasowej
    consistency_analysis = comparison.analyze_time_consistency(initial_state=5, horizon=10)
    print("\nANALIZA SPÓJNOŚCI CZASOWEJ")
    print("-" * 70)
    print(f"Spójny czasowo: {consistency_analysis['is_time_consistent']}")
    print(f"Liczba niespójności: {len(consistency_analysis['inconsistencies'])}")
    
    # Wykres porównawczy
    comparison.plot_comparison(save_path='../../data/plots/standard_vs_qh_comparison.png')


if __name__ == "__main__":
    run_comparison_example()
