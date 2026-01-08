# Przewodnik szybkiego startu: Porównanie Standardowego vs QH Dyskontowania

## 5-minutowy szybki start

### Opcja 1: Uruchom demo
```bash
cd src/experiments
python demo_comparison.py --demo basic
```

### Opcja 2: Użyj notatnika Jupyter
```bash
jupyter notebook notebooks/standard_vs_qh_comparison.ipynb
```

### Opcja 3: Skrypt Python
```python
from src.experiments.comparison_standard_vs_qh import MDPComparison
from src.models.mdp_environments import InventoryMDP

# Tworzenie i uruchamianie porównania
env = InventoryMDP(max_inventory=10, max_order=5)
comparison = MDPComparison(env=env, alpha=0.7, beta=0.95)
comparison.train(n_episodes=2000)
print(comparison.generate_report())
```

## Czego się spodziewać

### Standardowy Q-Learning
- Używa dyskontowania wykładniczego: V(s) = E[Σ β^t r_t]
- Spójne czasowo preferencje
- Optymalny dla długoterminowego planowania

### QH Q-Learning  
- Używa dyskontowania quasi-hiperbolicznego: V(s) = E[r_0 + α Σ β^t r_t]
- Uprzedzenie teraźniejszości gdy α < 1
- Może być niespójny czasowo
- Lepiej modeluje ludzkie zachowania

## Kluczowe parametry

- **α (alpha)**: Parametr uprzedzenia teraźniejszości
  - α = 1.0: Brak uprzedzenia teraźniejszości (tak samo jak standardowy)
  - α = 0.7: Umiarkowane uprzedzenie teraźniejszości
  - α = 0.5: Silne uprzedzenie teraźniejszości
  
- **β (beta)**: Współczynnik dyskontowania (typowo 0.95)

- **n_episodes**: Epizody treningowe (zalecane 2000-5000)

## Zrozumienie wyjścia

### Zgodność polityk
- **100%**: Polityki są identyczne
- **80-99%**: Niewielkie różnice
- **<80%**: Znaczące rozbieżności

### Różnice wartości
- Pokazuje jak różnie oba podejścia wyceniają każdy stan
- Większe różnice wskazują na silniejszy wpływ uprzedzenia teraźniejszości

### Spójność czasowa
- **Spójny**: Agent nie chce odstępować od planu
- **Niespójny**: Agent wolałby później inne akcje
- Częste gdy α < 1

## Przykładowe wyjście

```
======================================================================
PORÓWNANIE: Standardowe vs Quasi-Hiperboliczne Dyskontowanie
======================================================================

Środowisko: InventoryMDP
Stany: 11, Akcje: 6
Beta (dyskontowanie): 0.95
Alpha (uprzedzenie teraźniejszości): 0.7
Przeszkolone epizody: 2000

PORÓWNANIE POLITYK
----------------------------------------------------------------------
Zgodność: 72.7%
Różne stany: 3
Stany gdzie polityki się różnią: [3 5 8]

PORÓWNANIE FUNKCJI WARTOŚCI
----------------------------------------------------------------------
Średnia różnica bezwzględna: 1.2345
Maksymalna różnica bezwzględna: 3.4567

PORÓWNANIE WYDAJNOŚCI
----------------------------------------------------------------------
Nagroda końcowego epizodu (Standardowy): 45.23
Nagroda końcowego epizodu (QH): 43.67
Różnica: 1.56

======================================================================
```

## Następne kroki

1. **Eksperymentuj z różnymi wartościami α**: Zobacz jak uprzedzenie teraźniejszości wpływa na zachowanie
2. **Wypróbuj różne środowiska**: Testuj na różnych strukturach MDP  
3. **Analizuj spójność czasową**: Badaj kiedy polityki stają się niespójne
4. **Przeczytaj pełną dokumentację**: Zobacz `docs/COMPARISON_GUIDE.md`

## Rozwiązywanie problemów

**Problem:** Błędy importu
**Rozwiązanie:** Upewnij się, że jesteś w odpowiednim katalogu i masz zainstalowane zależności

**Problem:** Brak zbieżności
**Rozwiązanie:** Spróbuj więcej epizodów lub dostosuj krok uczenia (theta_step)

**Problem:** Wszystkie polityki identyczne
**Rozwiązanie:** Spróbuj niższej wartości α (silniejsze uprzedzenie teraźniejszości)

## Uzyskiwanie pomocy

- Pełna dokumentacja: `docs/COMPARISON_GUIDE.md`
- Przykładowy notatnik: `notebooks/standard_vs_qh_comparison.ipynb`
- Skrypt testowy: `examples/test_comparison.py`
