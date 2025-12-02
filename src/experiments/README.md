# Moduł eksperymentów

Moduł zawiera frameworki eksperymentalne do porównywania i analizowania algorytmów uczenia ze wzmocnieniem z różnymi schematami dyskontowania.

## Dostępne eksperymenty

### 1. Porównanie Standardowego vs Quasi-Hiperbolicznego

**Plik:** `comparison_standard_vs_qh.py`

Kompletny framework do porównywania standardowego dyskontowania wykładniczego z dyskontowaniem quasi-hiperbolicznym.

**Funkcjonalności:**
- Równoległe trenowanie obu algorytmów
- Porównanie polityk i funkcji wartości
- Analiza spójności czasowej
- Kompleksowa wizualizacja
- Automatyczne generowanie raportów

**Szybki start:**
```bash
python comparison_standard_vs_qh.py
```

### 2. Szybki test

**Plik:** `test_comparison.py`

Szybki test weryfikacyjny sprawdzający poprawność działania frameworku porównawczego.

**Użycie:**
```bash
python test_comparison.py
```

### 3. Skrypty demo

**Plik:** `demo_comparison.py`

Interaktywne demonstracje pokazujące kluczowe różnice między podejściami dyskontowania.

**Dostępne demo:**
```bash
# Podstawowe porównanie
python demo_comparison.py --demo basic

# Analiza wrażliwości na parametr α
python demo_comparison.py --demo alpha

# Tworzenie wizualizacji
python demo_comparison.py --demo viz

# Uruchom wszystkie demo
python demo_comparison.py --demo all
```

### 4. Wykresy porównania dyskontowania

**Plik:** `plot_discounting_comparison.py`

Generowanie wizualizacji porównujących standardowe dyskontowanie wykładnicze z preferencjami quasi-hiperbolicznymi dla wielu wartości α. Wykresy pokazują jak uprzedzenie teraźniejszości zwiększa wagę wcześniejszych nagród.

**Użycie:**
```bash
python -m src.experiments.plot_discounting_comparison --no-show --output data/plots/discounting_comparison.png
```

**Główne flagi:**

- `--alpha-values` – rozdzielone przecinkami wartości α do porównania (domyślnie `0.6,0.8,0.95`)
- `--beta` – wspólny wykładniczy współczynnik dyskontowania β (domyślnie `0.95`)
- `--horizon` – liczba wyświetlanych kroków czasowych (domyślnie `30`)
- `--no-show` – pomiń otwieranie okna GUI (przydatne na zdalnych serwerach)

Skrypt zapisuje wykres jeśli podano `--output`; katalogi są tworzone automatycznie.

### 5. Trenowanie balansowania kija

**Plik:** `pole_balancing_training.py`

Trenowanie agenta quasi-hiperbolicznego Q-learning na fizycznym środowisku balansowania kija opisanym w pracy.

**Użycie (zalecane):**
```bash
python -m src.experiments.pole_balancing_training --episodes 100000 --eval-episodes 25 --seed 123
```

Skrypt zapisuje skompresowany snapshot agenta (`.npz`) wraz z podsumowaniem JSON (oba ze znacznikiem czasu). Główne parametry:

**Główne flagi:**

- `--alpha` – parametr uprzedzenia teraźniejszości α (domyślnie 0.7)
- `--beta` – wykładniczy współczynnik dyskontowania β (domyślnie 0.97)
- `--theta-step` – krok uczenia
- `--epsilon` / `--min-epsilon` / `--epsilon-decay` – harmonogram eksploracji
- `--seed` – ziarno dla powtarzalności
- `--results-dir` – katalog dla podsumowań JSON (domyślnie `data/results`)
- `--model-dir` – katalog dla snapshotów wag agenta (domyślnie `data/models`)
- `--no-save` – pomiń zapisywanie podsumowania JSON
- `--no-save-model` – pomiń zapisywanie snapshotu wytrenowanego agenta

Logi JSON zawierają nagrody na epizod (duże dla 100k uruchomień); dostosuj `--episodes` jeśli miejsce na dysku jest ograniczone.

### 6. Symulacja balansowania kija (GIF)

**Plik:** `pole_balancing_simulation.py`

Wczytanie wytrenowanego snapshotu agenta i wyprodukowanie GIF-a Matplotlib (domyślnie ~10s klip z dokładnością środowiska):

```bash
python -m src.experiments.pole_balancing_simulation \
    --model data/models/pole_balancing_100k/pole_balancing_qh_<timestamp>.npz \
    --output data/plots/pole_balancing_100k.gif \
    --duration 10
```

Dodatkowe opcje:

- `--fps` – nadpisanie częstotliwości klatek (domyślnie częstotliwość kroku środowiska)
- `--seed` – powtarzalny reset dla rollout-u
- `--env-json` – opcjonalna ścieżka do JSON-a podsumowania treningu jeśli metadane są niedostępne
- `--title` – dostosowanie tytułu animacji

### 7. Benchmark zarządzania zapasami

**Plik:** `inventory_control_experiment.py`

Trenowanie agenta quasi-hiperbolicznego Q-learning na problemie zarządzania zapasami ze skończonym horyzontem opisanym w pracy (pojemność ``M = 2`` z prawdopodobieństwami popytu ``0.2, 0.3, 0.5``).

**Użycie:**
```bash
python -m src.experiments.inventory_control_experiment --episodes 5000 --episode-length 30
```

**Główne flagi:**

- `--alpha` / `--beta` – parametry dyskontowania quasi-hiperbolicznego (domyślnie ``0.3`` i ``0.9``).
- `--max-inventory`, `--procurement-cost`, `--holding-cost`, `--selling-price` – ekonomia środowiska.
- `--demand-support`, `--demand-prob` – rozkład popytu (listy rozdzielone przecinkami).
- `--results-dir` – katalog dla podsumowań JSON (domyślnie `data/results`).
- `--no-save` – pomiń zapisywanie podsumowania na dysk.

Skrypt raportuje średnią nagrodę, ilość zamówień, sprzedaż i końcowy poziom zapasów w ewaluacyjnym rollout-cie.

### 8. Zbieżność oceny polityki zapasów

**Plik:** `policy_evaluation_convergence.py`

Odtworzenie badania zbieżności z pracy (Rysunek 1) dla benchmarku zarządzania zapasami, porównując trzy pary polityk w algorytmie oceny polityki quasi-hiperbolicznej.

**Użycie:**
```bash
python -m src.experiments.policy_evaluation_convergence --iterations 200000 --eta 0.3 --theta 0.03
```

**Wyjścia:**

- ``data/plots/policy_evaluation_inventory_convergence.png`` – wykres jednoosowy przedstawiający $\lVert W_k - V^\beta_{\phi_s} \rVert_2$ na logarytmicznej skali iteracji dla trzech par polityk $\big(\mu^*, \phi_s^*\big)$, $\big(\mu^*, \phi_s^u\big)$, oraz $\big(\mu^u, \phi_s^*\big)$.

Opcjonalne flagi odpowiadają notacji używanej w pracy:
- `--alpha` / `--beta` – parametry quasi-hiperboliczne ($\alpha = 0.3$, $\beta = 0.9$ domyślnie),
- `--eta` / `--theta` – rozmiary kroków dla szybkiej i wolnej skali czasowej,
- `--iterations` – całkowita liczba aktualizacji (zalecane 200k dla gładkich krzywych),
- `--seed` – ziarno RNG dla powtarzalności.

Skrypt automatycznie zapisuje wykres i raportuje jego lokalizację po zakończeniu.

### 9. Eksperyment myślowy wyboru jabłka

**Plik:** `apple_choice_experiment.py`

Replikacja klasycznego pytania behawioralnego "jedno jabłko dziś czy dwa jutro?"
i porównanie z wariantem "jedno za 50 dni vs dwa za 51 dni". Skrypt
wypisuje zdyskontowane wartości i wynikające decyzje dla agentów wykładniczych i
quasi-hiperbolicznych, czyniąc niespójność czasową indukowaną przez uprzedzenie teraźniejszości
jawną.

**Użycie:**
```bash
python -m src.experiments.apple_choice_experiment --alpha 0.45 --beta 0.95
```

Dostosuj `--alpha` aby kontrolować siłę uprzedzenia teraźniejszości (niższe wartości czynią
agenta bardziej niecierpliwym). Wyjście podsumowuje którą opcję preferuje każdy schemat
dyskontowania w obu scenariuszach i czy sekwencja wyborów jest
spójna czasowo.

## Notatniki Jupyter

Dla interaktywnej eksploracji, zobacz:
- `../notebooks/standard_vs_qh_comparison.ipynb`

## Dokumentacja

Dla szczegółowej dokumentacji, zobacz:
- `../docs/COMPARISON_GUIDE.md`
- Szczegóły środowiska balansowania kija są dostępne przez docstringi w `models.mdp_environments.PoleBalancingMDP`.

## Przykład użycia

```python
from comparison_standard_vs_qh import MDPComparison
from models.mdp_environments import InventoryMDP, PoleBalancingMDP

# Tworzenie środowiska
env = InventoryMDP(max_inventory=15, max_order=8)
# Dla balansowania kija zamiast tego użyj:
# env = PoleBalancingMDP()

# Konfiguracja porównania
comparison = MDPComparison(
    env=env,
    alpha=0.7,   # Parametr uprzedzenia teraźniejszości
    beta=0.95,   # Współczynnik dyskontowania
    theta_step=0.1,   # Krok uczenia
    epsilon=0.1  # Współczynnik eksploracji
)

# Trenowanie obu algorytmów
comparison.train(n_episodes=5000, record_interval=100)

# Analiza wyników
print(comparison.generate_report())
comparison.plot_comparison(save_path='results.png')

# Sprawdzenie polityk
policy_comp = comparison.compare_policies()
print(f"Zgodność: {policy_comp['agreement_percentage']:.1f}%")

# Sprawdzenie spójności czasowej
consistency = comparison.analyze_time_consistency(
    initial_state=5, 
    horizon=10
)
print(f"Spójny czasowo: {consistency['is_time_consistent']}")
```

## Kluczowe wyniki do analizy

1. **Różnice w politykach:** Gdzie polityki się różnią i dlaczego?
2. **Funkcje wartości:** Jak porównują się estymaty wartości?
3. **Spójność czasowa:** Czy polityka QH jest spójna czasowo?
4. **Wpływ uprzedzenia teraźniejszości:** Jak α wpływa na zachowanie?

## Wyjście

Wyniki są zapisywane do:
- `../../data/results/` - Wyniki numeryczne
- `../../data/plots/` - Wizualizacje

## Zależności

```python
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
```
