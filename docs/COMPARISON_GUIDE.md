# Porównanie Standardowego i Quasi-Hyperbolicznego Dyskontowania w MDPs

## Przegląd

Ten moduł dostarcza kompletny framework do porównywania dwóch podejść do dyskontowania w Markowskich Procesach Decyzyjnych:

1. **Standard Q-Learning** - klasyczne wykładnicze dyskontowanie
2. **Quasi-Hyperbolic Q-Learning** - dyskontowanie z present-bias

## Motywacja

Tradycyjne podejście zakłada wykładnicze dyskontowanie:
```
V(s) = E[∑_{t=0}^∞ γ^t r_t]
```

Jednak rzeczywiste zachowania ludzkie często wykazują **present-bias** - nadmierne preferowanie nagród natychmiastowych. Quasi-hyperbolic discounting modeluje to przez:
```
V(s) = E[r_0 + σ ∑_{t=1}^∞ γ^t r_t]
```

gdzie `σ ∈ [0,1]` to parametr present-bias.

## Struktura Modułu

### Główne Klasy

#### 1. `StandardQLearning`
Implementacja klasycznego Q-Learning z wykładniczym dyskontowaniem.

**Parametry:**
- `n_states`: liczba stanów
- `n_actions`: liczba akcji
- `gamma`: współczynnik dyskontowania (0 ≤ γ < 1)
- `alpha`: współczynnik uczenia
- `epsilon`: współczynnik eksploracji

**Kluczowe metody:**
- `select_action(state)`: wybór akcji według ε-greedy
- `update(state, action, reward, next_state, done)`: aktualizacja Q-table
- `get_policy()`: ekstrakcja polityki zachłannej
- `get_value_function()`: obliczenie funkcji wartości

#### 2. `QHQLearning`
Implementacja Q-Learning z quasi-hyperbolic discounting (zdefiniowana w `src/algorithms/qh_qlearning.py`).

**Dodatkowe parametry:**
- `sigma`: parametr present-bias (0 ≤ σ ≤ 1)

**Specyfika:**
- Utrzymuje dwie funkcje Q: wykładniczą i quasi-hyperboliczną
- Implementuje aktualizację dwu-czasową
- Może wykazywać niespójność czasową

#### 3. `MDPComparison`
Framework porównawczy łączący oba algorytmy.

**Funkcjonalność:**
- Równoległe trenowanie obu algorytmów
- Porównanie polityk i funkcji wartości
- Analiza spójności czasowej
- Wizualizacja wyników
- Generowanie raportów

### Kluczowe Metody

#### `train(n_episodes, record_interval)`
Trenuje oba algorytmy przez określoną liczbę epizodów.

```python
comparison.train(n_episodes=5000, record_interval=100)
```

#### `compare_policies()`
Porównuje nauczone polityki.

**Zwraca:**
```python
{
    'standard_policy': np.array,     # Polityka standardowa
    'qh_policy': np.array,           # Polityka QH
    'different_states': np.array,    # Stany gdzie różnią się
    'agreement_percentage': float    # % zgodności
}
```

#### `compare_values()`
Porównuje funkcje wartości.

**Zwraca:**
```python
{
    'standard_values': np.array,     # Wartości standardowe
    'qh_values': np.array,           # Wartości QH
    'value_difference': np.array,    # Różnice
    'mean_abs_difference': float,    # Średnia bezwzględna różnica
    'max_abs_difference': float      # Maksymalna różnica
}
```

#### `analyze_time_consistency(initial_state, horizon)`
Analizuje spójność czasową polityki QH.

**Sprawdza:** Czy agent chciałby odstąpić od precommitted policy w każdym kroku czasowym.

**Zwraca:**
```python
{
    'trajectory': list,              # Trajektoria stanów
    'actions': list,                 # Wykonane akcje
    'inconsistencies': list,         # Lista niespójności
    'is_time_consistent': bool       # Czy spójna
}
```

#### `plot_comparison(save_path)`
Generuje kompleksową wizualizację porównawczą:
- Krzywe uczenia
- Funkcje wartości
- Porównanie polityk
- Różnice w wartościach

#### `generate_report()`
Generuje tekstowy raport ze wszystkimi metrykami.

## Przykład Użycia

### Podstawowy Przykład

```python
from src.experiments.comparison_standard_vs_qh import MDPComparison
from src.models.mdp_environments import InventoryMDP

# Utworzenie środowiska
env = InventoryMDP(max_inventory=15, max_order=8)

# Inicjalizacja porównania
comparison = MDPComparison(
    env=env,
    sigma=0.7,   # Silny present-bias
    gamma=0.95,
    alpha=0.1,
    epsilon=0.1
)

# Trenowanie
comparison.train(n_episodes=5000, record_interval=100)

# Analiza
print(comparison.generate_report())
comparison.plot_comparison(save_path='results.png')
```

### Analiza Wrażliwości na σ

```python
sigma_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for sigma in sigma_values:
    comp = MDPComparison(env=env, sigma=sigma, gamma=0.95)
    comp.train(n_episodes=2000)
    
    policy_comp = comp.compare_policies()
    results.append({
        'sigma': sigma,
        'agreement': policy_comp['agreement_percentage']
    })

# Wizualizacja wpływu σ na zgodność polityk
import matplotlib.pyplot as plt
plt.plot([r['sigma'] for r in results], 
         [r['agreement'] for r in results])
plt.xlabel('σ (present-bias)')
plt.ylabel('Policy Agreement (%)')
plt.show()
```

### Analiza Spójności Czasowej

```python
# Sprawdź czy polityka jest time-consistent
consistency = comparison.analyze_time_consistency(
    initial_state=5,
    horizon=15
)

print(f"Time-consistent: {consistency['is_time_consistent']}")
print(f"Inconsistencies: {len(consistency['inconsistencies'])}")

for inc in consistency['inconsistencies']:
    print(f"Step {inc['time']}, State {inc['state']}: "
          f"Precommitted={inc['precommitted_action']}, "
          f"Myopic={inc['myopic_action']}")
```

## Interpretacja Wyników

### Zgodność Polityk

- **100%**: Polityki identyczne (zwykle gdy σ ≈ 1)
- **80-99%**: Niewielkie różnice w niektórych stanach
- **<80%**: Znaczące różnice, silny wpływ present-bias

### Różnice w Wartościach

- **Dodatnie różnice**: Standard wycenia wyżej (lepsze dla długoterminowego planowania)
- **Ujemne różnice**: QH wycenia wyżej (bardziej optymistyczne krótkoterminowo)
- **Duże różnice**: Stany wymagające kompromisu między teraz a później

### Spójność Czasowa

- **Time-consistent**: Agent nie chce zmieniać decyzji w czasie
- **Time-inconsistent**: Agent chciałby odstąpić od początkowego planu
  - Typowe dla σ < 1
  - Pokazuje konflikt między precommitment a myopic choices

## Zastosowania

### 1. Zarządzanie Zapasami
```python
env = InventoryMDP(max_inventory=20, max_order=10)
# Porównanie zachowania przy różnych poziomach zapasów
```

### 2. Planowanie Finansowe
```python
# Modelowanie decyzji inwestycyjnych z present-bias
# QH lepiej modeluje rzeczywiste zachowania ludzkie
```

### 3. Ekonomia Behawioralna
```python
# Badanie wpływu present-bias na optymalne decyzje
# Analiza time-inconsistency w sekwencyjnym podejmowaniu decyzji
```

## Metryki i Wskaźniki

### Metryki Wydajności
- **Total reward per episode**: Suma nagród w epizodzie
- **Convergence rate**: Szybkość zbieżności do optymalnej polityki
- **Final performance**: Wydajność po zakończeniu uczenia

### Metryki Porównawcze
- **Policy agreement %**: Procent stanów z identycznymi akcjami
- **Mean value difference**: Średnia różnica w wartościach stanów
- **Max value difference**: Maksymalna różnica w wartościach
- **Time-inconsistency count**: Liczba niespójności czasowych

## Wymagania

```python
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
```

## Pliki

- **`comparison_standard_vs_qh.py`**: Główny moduł
- **`examples/test_comparison.py`**: Szybki test weryfikacyjny
- **`../notebooks/standard_vs_qh_comparison.ipynb`**: Interaktywny notebook

## Dalszy Rozwój

Możliwe rozszerzenia:

1. **Więcej algorytmów**: SARSA, Actor-Critic
2. **Inne modele dyskontowania**: Hyperbolic, generalized hyperbolic
3. **Sophisticated agents**: Porównanie z precommitted agents
4. **Continuous spaces**: Rozszerzenie na ciągłe przestrzenie stanów/akcji
5. **Multi-agent**: Gry z graczami o różnych preferencjach czasowych

## Bibliografia

- Jaśkiewicz & Nowak (2021): "Markov decision processes with quasi-hyperbolic discounting"
- Eshwar et al. (2024): "Reinforcement learning with quasi-hyperbolic discounting"
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"

## Kontakt

Dla pytań i sugestii, proszę otworzyć issue w repozytorium.
