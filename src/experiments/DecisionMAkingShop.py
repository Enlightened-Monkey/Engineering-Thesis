import numpy as np
import os
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    PLOTTING_AVAILABLE = False

# --- UPROSZCZONY MODEL KATEGORII (1..5) ---
LEVEL_MIN = 1
LEVEL_MAX = 5

# Mapy etykiet (PL)
PRICE_LABELS = {
    1: "B. tani",
    2: "Tani",
    3: "Normalna cena",
    4: "Drogi",
    5: "B. drogi",
}
WEALTH_LABELS = {
    1: "B. bogaty",
    2: "Bogaty",
    3: "Normalny",
    4: "Biedny",
    5: "B. biedny",
}
AGE_LABELS = {
    1: "B. młody",
    2: "Młody",
    3: "Średni wiek",
    4: "Stary",
    5: "B. stary",
}
  

def norm_level(level: int) -> float:
    """Normalizacja poziomu 1..5 do [0,1] z mapowaniem 1->0, 5->1."""
    return float(np.clip((level - LEVEL_MIN) / (LEVEL_MAX - LEVEL_MIN), 0.0, 1.0))

# Wagi wpływu na Sigmę (Present Bias)
W_PRICE = 1/3
W_WEALTH = 1/3
W_AGE = 1/3

def sigmoid(x, k=10, x0=0.5):
    return (1 / (1 + np.exp(-k * (x - x0))))/2 + 0.5

def calculate_alpha(price_level: int, wealth_level: int, age_level: int) -> float:
    """Oblicza parametr alpha (0-1) na bazie kategorii 1..5.
    
    Wyjaśnienie: starszy (level 1, a_norm=0) → brak odejmowania → wyższy score → wyższy alpha → bardziej cierpliwy.
    Młodszy (level 5, a_norm=1) → odejmowanie W_AGE → niższy score → niższy alpha → mniej cierpliwy (bardziej impulsywny).
    """
    p_norm = norm_level(price_level)
    w_norm = norm_level(wealth_level)
    a_norm = norm_level(age_level)

    score = (W_PRICE * p_norm) + (W_WEALTH * w_norm) + (W_AGE * a_norm)

    # Mapowanie na alpha (Present Bias)
    alpha = sigmoid(score, k=7, x0=0.5)

    return float(alpha)

def get_reward_curve_root_like(steps, price_level: int):
    """
    Generuje krzywą nagrody R(t) o kształcie pierwiastka/logarytmu (szybki wzrost, malejące zyski).
    Wzór: R(t) = 1 + Potential * (1 - exp(-A * t))
    Zależność od ceny: używa znormalizowanego poziomu ceny (1..5 -> [0,1]).
    """
    base_reward = 1.0

    # Potencjał zysku (max_upside) – większy dla droższych produktów
    price_norm = norm_level(price_level)
    potential_gain = 0.10 + 0.60 * price_norm

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

def simulate_decision(name: str, price_level: int, wealth_level: int, age_level: int, beta=0.99, max_steps=15):
    """Symuluje proces decyzyjny i zwraca wyniki dla poziomów 1..5."""
    steps = np.arange(max_steps)
    alpha = calculate_alpha(price_level, wealth_level, age_level)

    # Użycie funkcji nagrody zależnej od poziomu ceny
    rewards_Rt = get_reward_curve_root_like(steps, price_level)

    utilities = []
    for t in steps:
        if t == 0:
            u = rewards_Rt[t]
        else:
            u = alpha * (beta ** t) * rewards_Rt[t]
        utilities.append(u)

    utilities = np.array(utilities)
    best_step = int(np.argmax(utilities))

    decision = "KUP TERAZ" if best_step == 0 else f"RESEARCH ({best_step} kr.)"

    return {
        "name": name,
        "price_level": int(price_level),
        "wealth_level": int(wealth_level),
        "age_level": int(age_level),
        "price_label": PRICE_LABELS.get(int(price_level), str(price_level)),
        "wealth_label": WEALTH_LABELS.get(int(wealth_level), str(wealth_level)),
        "age_label": AGE_LABELS.get(int(age_level), str(age_level)),
        "alpha": alpha,
        "steps": steps,
        "rewards": rewards_Rt,
        "utilities": utilities,
        "decision": decision,
        "best_step": best_step,
    }

def plot_and_save(result):
    """Tworzy wykres dla pojedynczego profilu i zapisuje go jako PNG."""
    if not PLOTTING_AVAILABLE:
        print("[INFO] Pomijam rysowanie: matplotlib niedostępny.")
        return

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
    title_text = (
        f"{result['name']}\n"
        f"Cena: {result['price_label']} | Majątek: {result['wealth_label']} | Wiek: {result['age_label']}\n"
        f"Alpha: {result['alpha']:.3f} (Present Bias) | Max Nagroda: {max(result['rewards']):.2f}"
    )
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
    {"name": "Młody Bogacz (Tani Produkt)", "price_level": 2, "wealth_level": 1, "age_level": 2},
    {"name": "Emeryt (Drogi Produkt)", "price_level": 5, "wealth_level": 2, "age_level": 5},
    {"name": "Student (Normalna cena)", "price_level": 3, "wealth_level": 2, "age_level": 2},
]

def print_all_combinations_table():
    """Drukuje tabelę decyzji dla wszystkich kombinacji poziomów (1..5)."""
    print("price_level,price_label,wealth_level,wealth_label,age_level,age_label,alpha,decision,best_step")
    for p in range(1, 6):
        for w in range(1, 6):
            for a in range(1, 6):
                res = simulate_decision(
                    name=f"P{p}-W{w}-A{a}",
                    price_level=p,
                    wealth_level=w,
                    age_level=a,
                )
                print(
                    f"{p},{PRICE_LABELS[p]},{w},{WEALTH_LABELS[w]},{a},{AGE_LABELS[a]},{res['alpha']:.3f},{res['decision']},{res['best_step']}"
                )

def plot_sigmoid_alpha_curve(k=7, x0=0.5):
    """Rysuje samą funkcję sigmoidalną używaną do wyznaczania parametru alpha."""
    if not PLOTTING_AVAILABLE:
        print("[INFO] Pomijam rysowanie sigmoid: matplotlib niedostępny.")
        return

    scores = np.linspace(0, 1, 200)
    alphas = sigmoid(scores, k=k, x0=x0)

    plt.figure(figsize=(8, 5))
    plt.plot(scores, alphas, color='#1f77b4', linewidth=2.5, label=r"$\sigma(\text{score}; k, x_0)=\alpha$")
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    plt.axvline(x0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    plt.text(x0 + 0.02, 0.52, f"x0={x0}", fontsize=10, color='gray')
    plt.title("Funkcja sigmoidalna dla parametru $\\alpha$", fontsize=12, fontweight='bold')
    plt.xlabel("score")
    plt.ylabel(r"$\alpha$")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Sigmoid_alpha.png", dpi=150)
    plt.close()
    print("Zapisano wykres: Sigmoid_alpha.png")


def plot_reward_discount_combo(price_level=2, wealth_level=3, age_level=3, beta=0.99, max_steps=15):
    """Rysuje krzywą nagrody R(t), czynnik alpha*beta^t i ich iloczyn (u_t).

    Domyślnie używa price_level=2 oraz profilu środkowego (wealth=3, age=3).
    """
    if not PLOTTING_AVAILABLE:
        print("[INFO] Pomijam rysowanie nagród/discount: matplotlib niedostępny.")
        return

    steps = np.arange(max_steps)
    alpha = calculate_alpha(price_level, wealth_level, age_level)
    rewards = get_reward_curve_root_like(steps, price_level)
    # d(t) = alpha * beta^t z uskokiem: d(0)=1, d(t>=1)=alpha*beta^t
    discount = np.ones_like(steps, dtype=float)
    discount[1:] = alpha * (beta ** steps[1:])

    perceived = discount * rewards

    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, 'o--', label='R(t) (nagroda obiektywna)', color='gray')
    plt.plot(steps, discount, 's-', label=r"$d(t)=\alpha\,\beta^t$ z $d(0)=1$", color='#2ca02c')
    plt.plot(steps, perceived, 'd-', label=r"$d(t) \cdot R(t)$", color='#d62728')

    plt.title(
        f"Nagroda i dyskontowanie (price_level={price_level}, wealth={wealth_level}, age={age_level})\n"
        f"alpha={alpha:.3f}, beta={beta}",
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Czas (kroki)')
    plt.ylabel('Wartość')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Reward_Discount_combo.png", dpi=150)
    plt.close()
    print("Zapisano wykres: Reward_Discount_combo.png")


def plot_three_profiles_together(profiles_list, beta=0.99, max_steps=15):
    """Rysuje R(t), d(t) oraz d(t)*R(t) dla trzech profili na jednym obrazku (3 panele)."""
    if not PLOTTING_AVAILABLE:
        print("[INFO] Pomijam rysowanie wspólnego wykresu: matplotlib niedostępny.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for idx_ax, (ax, profile) in enumerate(zip(axes, profiles_list)):
        res = simulate_decision(**profile, beta=beta, max_steps=max_steps)
        steps = res["steps"]
        alpha = res["alpha"]
        rewards = res["rewards"]
        discount = np.ones_like(steps, dtype=float)
        discount[1:] = alpha * (beta ** steps[1:])
        perceived = discount * rewards

        # --- nowy fragment: detekcja momentu zakupu ---
        decision_step = res.get("decision_step")
        if decision_step is None and "decisions" in res:
            for s, dec in zip(res["steps"], res["decisions"]):
                if isinstance(dec, str) and ("kup" in dec.lower() or "buy" in dec.lower()):
                    decision_step = s
                    break
        purchase_label = "Moment zakupu" if idx_ax == 0 else None
        # --- koniec nowego fragmentu ---

        ax.plot(steps, rewards, 'o--', color='gray', label='R(t)')
        ax.plot(steps, discount, 's-', color='#2ca02c', label=r"$d(t)=\alpha\beta^t$")
        ax.plot(steps, perceived, 'd-', color='#d62728', label=r"$d(t)\cdot R(t)$")

        if decision_step is not None:
            # znajdź indeks kroku w tablicy steps, by pobrać wartość d(t)*R(t)
            try:
                j = int(np.where(steps == decision_step)[0][0])
                ax.axvline(decision_step, color="#1f77b4", linestyle="--", alpha=0.5, label=purchase_label)
                ax.scatter(decision_step, perceived[j], color="#1f77b4", marker="x", zorder=6)
            except Exception:
                pass

        ax.set_title(f"{profile['name']}\nα={alpha:.3f}, β={beta}", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Czas (kroki)")
        ax.set_xlim(0, max_steps-1)

    axes[0].set_ylabel("Wartość")
    axes[0].legend(loc="lower right")
    fig.suptitle("Nagroda, dyskontowanie i użyteczność — trzy profile", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig("Three_profiles_combo.png", dpi=150)
    plt.close()
    print("Zapisano wykres: Three_profiles_combo.png")


def main():
    # 1) Tabela wszystkich kombinacji
    print_all_combinations_table()

    # 2a) Dodatkowe wizualizacje
    plot_sigmoid_alpha_curve()
    plot_reward_discount_combo()

    # 2b) Wspólny wykres dla trzech profili
    plot_three_profiles_together(profiles, beta=0.99, max_steps=15)

    # 3) Przykładowe profile + pojedyncze wykresy (jeśli matplotlib dostępny)
    for profile in profiles:
        result = simulate_decision(**profile)
        plot_and_save(result)

    print("\n--- Zakończono: wydrukowano tabelę, analizę alfa i (opcjonalnie) zapisano wykresy. ---")


if __name__ == "__main__":
    main()