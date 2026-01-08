"""
Eksperymenty z dwustanowym MDP używając torch backend (CUDA support).
Implementuje Q-learning i policy evaluation dla eksperymentu 1 z thesis.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.torch_utils import get_device, TORCH_AVAILABLE
from src.algorithms.qh_qlearning_torch import QHQLearningTorch
from src.algorithms.qh_policy_evaluation_torch import QHPolicyEvaluationTorch


class TwoStateMDP:
    """
    Dwustanowy MDP z eksperymentu 1.
    
    Stany: S = {0, 1} (odpowiada {1, 2} z dokumentacji)
    Akcje: A = {0, 1} (odpowiada {a_1, a_2})
    """
    
    def __init__(self, alpha=0.5, beta=0.9):
        self.n_states = 2
        self.n_actions = 2
        self.alpha = alpha
        self.beta = beta
        
        # Macierz przejść P[s, a, s']
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        
        # Stan 0 (odpowiada s=1):
        # a=0 (a_1): deterministycznie do s=1
        self.P[0, 0, 1] = 1.0
        # a=1 (a_2): 50% do s=0, 50% do s=1
        self.P[0, 1, 0] = 0.5
        self.P[0, 1, 1] = 0.5
        
        # Stan 1 (odpowiada s=2):
        # a=0 (a_1): 50% do s=0, 50% do s=1
        self.P[1, 0, 0] = 0.5
        self.P[1, 0, 1] = 0.5
        # a=1 (a_2): 50% do s=0, 50% do s=1
        self.P[1, 1, 0] = 0.5
        self.P[1, 1, 1] = 0.5
        
        # Macierz nagród R[s, a, s']
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        
        # Stan 0:
        self.R[0, 0, :] = 0.0  # r(1, a_1) = 0
        self.R[0, 1, :] = 2.0  # r(1, a_2) = 2
        
        # Stan 1:
        self.R[1, 0, :] = 20.0  # r(2, a_1) = 20
        self.R[1, 1, 0] = 20.0  # r(2, a_2, 1) = 20
        self.R[1, 1, 1] = 5.0   # r(2, a_2, 2) = 5
    
    def get_expected_reward(self, s, a):
        """Oczekiwana nagroda dla pary (s, a)."""
        return np.sum(self.P[s, a, :] * self.R[s, a, :])
    
    def sample_transition(self, s, a):
        """Losuj następny stan i nagrodę."""
        s_next = np.random.choice(self.n_states, p=self.P[s, a, :])
        r = self.R[s, a, s_next]
        return s_next, r
    
    def get_analytic_values(self):
        """
        Wartości analityczne Q i W dla 5 polityk z tabeli w thesis.
        Indeksy: s=0,1 (odpowiada 1,2), a=0,1 (odpowiada a_1,a_2)
        """
        Q_analytic = {
            'phi1': np.array([[35.5263, 35.7500], [46.9737, 46.2500]], dtype=np.float32),
            'phi2': np.array([[34.9875, 34.6250], [45.2625, 45.1250]], dtype=np.float32),
            'phi3': np.array([[38.7931, 38.8534], [49.9138, 49.3534]], dtype=np.float32),
            'phi4': np.array([[31.1897, 31.1724], [42.1552, 41.6724]], dtype=np.float32),
            'phi5': np.array([[35.1278, 35.1610], [46.2166, 45.6983]], dtype=np.float32),
        }
        
        W_analytic = {
            'phi1': np.array([[71.0526, 69.5000], [78.9474, 80.0000]], dtype=np.float32),
            'phi2': np.array([[69.9750, 67.2500], [75.5250, 77.7500]], dtype=np.float32),
            'phi3': np.array([[77.5862, 75.7069], [84.8276, 86.2069]], dtype=np.float32),
            'phi4': np.array([[62.3793, 60.3448], [69.3103, 70.8448]], dtype=np.float32),
            'phi5': np.array([[70.3319, 68.3750], [77.4181, 78.8750]], dtype=np.float32),
        }
        
        return Q_analytic, W_analytic


def make_uniform_sampling_policy(env):
    """Polityka losowa 50/50 dla wszystkich akcji."""
    def policy(s):
        return np.random.choice(env.n_actions)
    return policy


def make_deterministic_policy(actions):
    """
    Polityka deterministyczna.
    actions: dict {state: action} lub list [action_for_s0, action_for_s1]
    """
    if isinstance(actions, dict):
        action_map = actions
    else:
        action_map = {i: actions[i] for i in range(len(actions))}
    
    def policy(s):
        return action_map[s]
    
    return policy


def make_stochastic_policy(probs):
    """
    Polityka stochastyczna.
    probs: dict {state: [prob_a0, prob_a1]} lub array[n_states, n_actions]
    """
    if isinstance(probs, dict):
        prob_map = probs
    else:
        prob_map = {s: probs[s, :] for s in range(len(probs))}
    
    def policy(s):
        return np.random.choice(len(prob_map[s]), p=prob_map[s])
    
    return policy


def get_five_policies(env):
    """5 polityk z eksperymentu."""
    policies = {
        'phi1': make_deterministic_policy([0, 0]),  # zawsze a_1
        'phi2': make_deterministic_policy([1, 1]),  # zawsze a_2
        'phi3': make_deterministic_policy([0, 1]),  # a_1 w s=0, a_2 w s=1
        'phi4': make_deterministic_policy([1, 0]),  # a_2 w s=0, a_1 w s=1
        'phi5': make_stochastic_policy(np.array([[0.5, 0.5], [0.5, 0.5]])),  # 50/50
    }
    return policies


def run_qlearning(env, n_iterations=25_000_000, seed=42):
    """Uruchom QH Q-learning."""
    print("\n" + "=" * 60)
    print("QH Q-LEARNING")
    print("=" * 60)
    
    device = get_device()
    print(f"Urządzenie: {device}")
    
    # Parametry kroków uczenia
    eta_0 = 0.2
    kappa = 0.6
    theta_0 = 0.05
    lam = 0.9
    c = 1.0
    
    print(f"\nParametry:")
    print(f"  α={env.alpha}, β={env.beta}")
    print(f"  η_n: η_0={eta_0}, κ={kappa}, c={c}")
    print(f"  θ_n: θ_0={theta_0}, λ={lam}, c={c}")
    print(f"  Iteracji: {n_iterations:,}")
    
    # Inicjalizacja
    learner = QHQLearningTorch(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=env.alpha,
        beta=env.beta,
        device=device
    )
    
    # Polityka próbkowania (uniform)
    nu = make_uniform_sampling_policy(env)
    
    np.random.seed(seed)
    s = np.random.choice(env.n_states)
    
    # Tracking
    check_every = n_iterations // 100
    Q_hist = []
    W_hist = []
    iterations = []
    
    print("\nTrening...")
    for n in tqdm(range(1, n_iterations + 1), desc="Q-learning"):
        # Kroki uczenia
        eta_n = eta_0 / ((c + n) ** kappa)
        theta_n = theta_0 / ((c + n) ** lam)
        
        # Wybierz akcję z polityki próbkowania
        a = nu(s)
        
        # Wykonaj krok
        s_next, r = env.sample_transition(s, a)
        
        # Aktualizuj Q i W
        learner.update(s, a, r, s_next, eta_n, theta_n)
        
        # Nowy stan
        s = s_next
        
        # Zapisz co jakiś czas
        if n % check_every == 0:
            Q_np = learner.Q.cpu().numpy()
            W_np = learner.W.cpu().numpy()
            Q_hist.append(Q_np.copy())
            W_hist.append(W_np.copy())
            iterations.append(n)
    
    # Finalne wartości
    Q_final = learner.Q.cpu().numpy()
    W_final = learner.W.cpu().numpy()
    
    print("\n✓ Trening zakończony")
    print("\nWyuczone wartości Q:")
    print(Q_final)
    print("\nWyuczone wartości W:")
    print(W_final)
    
    # Wyznacz polityki
    mu_hat = np.argmax(Q_final, axis=1)
    phi_hat = np.argmax(W_final, axis=1)
    
    print(f"\nPolityka μ̂ (z Q): {mu_hat}")
    print(f"Polityka φ̂_s (z W): {phi_hat}")
    
    # Porównaj z analitycznym phi3 (optymalna)
    Q_opt, W_opt = env.get_analytic_values()
    print("\nPorównanie z wartościami analitycznymi (φ₃ - optymalna):")
    print(f"  Q błąd L2: {np.linalg.norm(Q_final - Q_opt['phi3']):.4f}")
    print(f"  W błąd L2: {np.linalg.norm(W_final - W_opt['phi3']):.4f}")
    
    return {
        'Q': Q_final,
        'W': W_final,
        'mu': mu_hat,
        'phi': phi_hat,
        'Q_hist': Q_hist,
        'W_hist': W_hist,
        'iterations': iterations,
    }


def run_policy_evaluation(env, n_iterations=1_000_000, seed=42):
    """Uruchom policy evaluation dla 5 polityk."""
    print("\n" + "=" * 60)
    print("POLICY EVALUATION")
    print("=" * 60)
    
    device = get_device()
    print(f"Urządzenie: {device}")
    
    # Parametry kroków uczenia
    eta_0 = 0.2
    kappa = 0.6
    theta_0 = 0.05
    lam = 0.9
    c = 1.0
    
    print(f"\nParametry:")
    print(f"  α={env.alpha}, β={env.beta}")
    print(f"  η_n: η_0={eta_0}, κ={kappa}, c={c}")
    print(f"  θ_n: θ_0={theta_0}, λ={lam}, c={c}")
    print(f"  Iteracji: {n_iterations:,}")
    
    # Pobierz polityki
    policies = get_five_policies(env)
    Q_analytic, W_analytic = env.get_analytic_values()
    
    # Polityka próbkowania
    nu = make_uniform_sampling_policy(env)
    
    results = {}
    
    for policy_name, policy in policies.items():
        print(f"\n--- Ewaluacja {policy_name} ---")
        
        # Inicjalizacja
        evaluator = QHPolicyEvaluationTorch(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=env.alpha,
            beta=env.beta,
            device=device
        )
        
        np.random.seed(seed)
        s = np.random.choice(env.n_states)
        
        # Tracking błędów
        check_every = n_iterations // 100
        W_errors = []
        J_errors = []
        iterations = []
        
        Q_ref = Q_analytic[policy_name]
        W_ref = W_analytic[policy_name]
        
        for n in tqdm(range(1, n_iterations + 1), desc=f"PE {policy_name}", leave=False):
            # Kroki uczenia
            eta_n = eta_0 / ((c + n) ** kappa)
            theta_n = theta_0 / ((c + n) ** lam)
            
            # Wybierz akcję z polityki próbkowania
            a_sample = nu(s)
            
            # Akcja z ewaluowanej polityki
            a_policy = policy(s)
            
            # Wykonaj krok
            s_next, r = env.sample_transition(s, a_sample)
            
            # Aktualizuj
            evaluator.update(s, a_sample, a_policy, r, s_next, eta_n, theta_n)
            
            s = s_next
            
            # Zapisz błąd
            if n % check_every == 0:
                W_np = evaluator.W.cpu().numpy()
                J_np = evaluator.J.cpu().numpy()
                
                W_err = np.linalg.norm(W_np - W_ref)
                J_err = np.linalg.norm(J_np - Q_ref)
                
                W_errors.append(W_err)
                J_errors.append(J_err)
                iterations.append(n)
        
        # Finalne wartości
        W_final = evaluator.W.cpu().numpy()
        J_final = evaluator.J.cpu().numpy()
        
        W_err_final = np.linalg.norm(W_final - W_ref)
        J_err_final = np.linalg.norm(J_final - Q_ref)
        
        print(f"  W błąd L2: {W_err_final:.6f}")
        print(f"  J błąd L2: {J_err_final:.6f}")
        
        results[policy_name] = {
            'W': W_final,
            'J': J_final,
            'W_errors': W_errors,
            'J_errors': J_errors,
            'iterations': iterations,
            'W_ref': W_ref,
            'J_ref': Q_ref,
        }
    
    return results


def plot_qlearning_results(qlearning_results, env, save_path=None):
    """Wykres zbieżności Q-learning."""
    Q_opt, W_opt = env.get_analytic_values()
    Q_ref = Q_opt['phi3']  # optymalna
    W_ref = W_opt['phi3']
    
    Q_hist = qlearning_results['Q_hist']
    W_hist = qlearning_results['W_hist']
    iterations = qlearning_results['iterations']
    
    Q_errors = [np.linalg.norm(Q - Q_ref) for Q in Q_hist]
    W_errors = [np.linalg.norm(W - W_ref) for W in W_hist]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(iterations, Q_errors, 'b-', linewidth=2, label='Błąd Q')
    axes[0].set_xlabel('Iteracja')
    axes[0].set_ylabel('Błąd $\\ell_2$')
    axes[0].set_title('Zbieżność Q (quasi-hyperboliczne)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].plot(iterations, W_errors, 'r-', linewidth=2, label='Błąd W')
    axes[1].set_xlabel('Iteracja')
    axes[1].set_ylabel('Błąd $\\ell_2$')
    axes[1].set_title('Zbieżność W (wykładnicze)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Wykres zapisany: {save_path}")
    
    plt.show()


def plot_policy_evaluation_results(pe_results, save_path=None):
    """Wykres zbieżności policy evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for (policy_name, result), color in zip(pe_results.items(), colors):
        iterations = result['iterations']
        W_errors = result['W_errors']
        J_errors = result['J_errors']
        
        axes[0].plot(iterations, W_errors, color=color, linewidth=2, 
                    label=f'{policy_name}', alpha=0.7)
        axes[1].plot(iterations, J_errors, color=color, linewidth=2, 
                    label=f'{policy_name}', alpha=0.7)
    
    axes[0].set_xlabel('Iteracja')
    axes[0].set_ylabel('Błąd $\\ell_2$')
    axes[0].set_title('Zbieżność W (wykładnicze)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Iteracja')
    axes[1].set_ylabel('Błąd $\\ell_2$')
    axes[1].set_title('Zbieżność J (quasi-hyperboliczne)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Wykres zapisany: {save_path}")
    
    plt.show()


def main():
    """Główna funkcja eksperymentu."""
    print("=" * 60)
    print("EKSPERYMENT 1: DWUSTANOWY MDP")
    print("=" * 60)
    
    # Sprawdź torch/CUDA
    if not TORCH_AVAILABLE:
        print("⚠ Torch niedostępny, używam NumPy backend")
    else:
        device = get_device()
        print(f"✓ Torch dostępny, urządzenie: {device}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Inicjalizacja środowiska
    env = TwoStateMDP(alpha=0.5, beta=0.9)
    print(f"\n✓ Środowisko: {env.n_states} stany, {env.n_actions} akcje")
    print(f"  Parametry: α={env.alpha}, β={env.beta}")
    
    # Output directory
    output_dir = REPO_ROOT / "data" / "results" / "two_state_cuda"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Katalog wyników: {output_dir}")
    
    # 1. Q-learning
    qlearning_results = run_qlearning(env, n_iterations=25_000_000, seed=42)
    
    # Plot Q-learning
    plot_qlearning_results(
        qlearning_results, 
        env, 
        save_path=output_dir / "qlearning_convergence.png"
    )
    
    # Zapisz wyniki Q-learning
    np.savez(
        output_dir / "qlearning_results.npz",
        Q=qlearning_results['Q'],
        W=qlearning_results['W'],
        mu=qlearning_results['mu'],
        phi=qlearning_results['phi'],
    )
    
    # 2. Policy evaluation
    pe_results = run_policy_evaluation(env, n_iterations=1_000_000, seed=42)
    
    # Plot policy evaluation
    plot_policy_evaluation_results(
        pe_results,
        save_path=output_dir / "policy_evaluation_convergence.png"
    )
    
    # Zapisz wyniki PE
    pe_data = {}
    for policy_name, result in pe_results.items():
        pe_data[f'{policy_name}_W'] = result['W']
        pe_data[f'{policy_name}_J'] = result['J']
    np.savez(output_dir / "policy_evaluation_results.npz", **pe_data)
    
    print("\n" + "=" * 60)
    print("✓ EKSPERYMENT ZAKOŃCZONY")
    print("=" * 60)
    print(f"\nWyniki zapisane w: {output_dir}")
    print("  - qlearning_results.npz")
    print("  - qlearning_convergence.png")
    print("  - policy_evaluation_results.npz")
    print("  - policy_evaluation_convergence.png")


if __name__ == "__main__":
    main()
