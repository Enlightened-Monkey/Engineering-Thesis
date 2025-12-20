import numpy as np
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ŚWIATA ---
MAX_PRICE = 10000.0   
MAX_WEALTH = 100000.0 
MAX_AGE = 90.0        

# Wagi wpływu na Sigmę (Present Bias)
W_PRICE = 0.5
W_WEALTH = 0.3
W_AGE = 0.2

def sigmoid(x, k=10, x0=0.5):
    return (1 / (1 + np.exp(-k * (x - x0))))/2 + 0.5

def calculate_alpha(price, wealth, age):
    """Oblicza parametr alpha (0-1)."""
    p_norm = np.clip(price / MAX_PRICE, 0, 1)
    w_norm = np.clip(wealth / MAX_WEALTH, 0, 1)
    a_norm = np.clip(age / MAX_AGE, 0, 1)
    
    score = (W_PRICE * p_norm) - (W_WEALTH * w_norm) + (W_AGE * a_norm)
    
    # Mapowanie na alpha (Present Bias)
    alpha = sigmoid(score, k=6, x0=0.1)
    
    return np.clip(alpha, 0.05, 0.98)

def get_reward_curve_root_like(steps, price):
    """
    Generuje krzywą nagrody R(t) o kształcie pierwiastka/logarytmu (szybki wzrost, malejące zyski).
    Wzór: R(t) = 1 + Potential * (1 - exp(-A * t))
    """
    base_reward = 1.0
    
    # Potencjał zysku (max_upside) – Taki sam jak wcześniej, duży dla drogich produktów
    potential_gain = 0.10 + 0.60 * (price / MAX_PRICE)
    
    rewards = []
    
    # Parametr 'a' kontroluje, jak szybko nagroda rośnie (stromość)
    A_RATE = 0.5 
    
    for t in steps:
        if t == 0:
            rewards.append(base_reward)
        else:
            # Wzrost nagrody: 1 - e^(-A*t). Rośnie szybko do 1, zwalnia po t=4/5
            growth = potential_gain * (1 - np.exp(-A_RATE * t))
            current_reward = base_reward + growth
            rewards.append(current_reward)
            
    return np.array(rewards)

def simulate_decision(name, price, wealth, age, beta=0.99, max_steps=15):
    """Symuluje proces decyzyjny i zwraca wyniki."""
    steps = np.arange(max_steps)
    alpha = calculate_alpha(price, wealth, age)
    
    # Użycie nowej funkcji nagrody
    rewards_Rt = get_reward_curve_root_like(steps, price)
    
    utilities = []
    for t in steps:
        if t == 0:
            u = rewards_Rt[t]
        else:
            u = alpha * (beta ** t) * rewards_Rt[t]
        utilities.append(u)
    
    utilities = np.array(utilities)
    best_step = np.argmax(utilities)
    
    decision = "KUP TERAZ" if best_step == 0 else f"RESEARCH ({best_step} kr.)"
    
    return {
        "name": name,
        "price": price, 
        "wealth": wealth,
        "age": age,
        "alpha": alpha,
        "steps": steps,
        "rewards": rewards_Rt,
        "utilities": utilities,
        "decision": decision,
        "best_step": best_step
    }

def plot_and_save(result):
    """Tworzy wykres dla pojedynczego profilu i zapisuje go jako PNG."""
    
    safe_name = result['name'].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    filename = f"Decyzja_{safe_name}_Pierwiastek.png"
    
    plt.figure(figsize=(10, 6))

    # Linia nagrody obiektywnej
    plt.plot(result['steps'], result['rewards'], color='gray', linestyle='--', alpha=0.6, label='Obiektywna Wartość R(t) - Funkcja pierwiastkowa')
    
    # Linia użyteczności subiektywnej
    plt.plot(result['steps'], result['utilities'], 'o-', linewidth=2.5, color='#1f77b4', label='Postrzegana Użyteczność (z QH)')
    
    # Punkt decyzji
    best_t = result['best_step']
    best_u = result['utilities'][best_t]
    dot_color = 'red' if best_t == 0 else 'green'
    
    plt.scatter([best_t], [best_u], s=200, c=dot_color, zorder=10, edgecolors='black')
    
    # Etykieta decyzji
    plt.annotate(f"{result['decision']}\n(U={best_u:.2f})", 
                 (best_t, best_u), 
                 xytext=(best_t + 0.5, best_u + (0.1 if best_t < 5 else -0.2)), 
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Tytuł i opis
    title_text = (f"{result['name']} | Cena: {result['price']} PLN\n"
                  f"Alpha: {result['alpha']:.3f} (Present Bias) | Max Nagroda: {max(result['rewards']):.2f}")
    plt.title(title_text, fontsize=12, fontweight='bold', loc='left')
    
    plt.grid(True, alpha=0.3)
    plt.ylabel('Użyteczność')
    plt.xlabel('Czas (kroki researchu)')
    plt.legend(loc='lower right')
    
    plt.savefig(filename)
    plt.close() 
    print(f"Zapisano wykres do pliku: {filename}")


# --- SCENARIUSZE ---

profiles = [
    {"name": "Młody Bogacz (Tani Produkt)", "price": 200, "wealth": 90000, "age": 20},
    {"name": "Emeryt (Drogi Produkt)", "price": 8000, "wealth": 5000, "age": 75},
    {"name": "Student (Średni Produkt)", "price": 3000, "wealth": 15000, "age": 22}
]

# --- URUCHOMIENIE I ZAPIS ---

for profile in profiles:
    result = simulate_decision(**profile)
    plot_and_save(result)

print("\n--- Zakończono symulację i zapisano wszystkie wykresy z nagrodą pierwiastkową. ---")