# Repozytorium Pracy Inżynierskiej

## Algorytmy oparte o wzmocnione nauczanie maszynowe w dyskontowanych modelach markowskich

**Tytuł (angielski):** Reinforcement learning algorithms in discounted Markov decision processes

**Autor:** Michał Wrona  
**Opiekun:** prof. dr hab. inż. Anna Jaśkiewicz  
**Instytucja:** Politechnika Wrocławska, Wydział Matematyki  
**Rok:** 2024

### Streszczenie

Repozytorium zawiera kompletną implementację i dokumentację pracy inżynierskiej dotyczącej algorytmów uczenia ze wzmocnieniem z dyskontowaniem quasi-hiperbolicznym w Markowskich Procesach Decyzyjnych (MDP). Badania koncentrują się na niespójnych czasowo decydentach (precommitted agents) i obejmują praktyczne zastosowanie do modeli gromadzenia zapasów.

**Słowa kluczowe:** Uczenie ze wzmocnieniem, Dyskontowanie quasi-hiperboliczne, Markowskie procesy decyzyjne, Niespójność czasowa, Ocena polityki, Zarządzanie zapasami

**Klasyfikacja AMS:** 90C39, 90C40, 90B05, 93E03, 93E35

---

## Struktura repozytorium

```
├── docs/                          # Dokumentacja i praca dyplomowa
│   ├── thesis/                    # Główny dokument pracy
│   │   ├── chapters/              # Rozdziały pracy (LaTeX)
│   │   ├── figures/               # Rysunki
│   │   ├── bibliography/          # Bibliografia BibTeX
│   │   └── main.tex               # Główny dokument LaTeX
│   ├── literature/                # Przeglądy i podsumowania literatury
│   └── presentation/              # Materiały do obrony
├── src/                           # Kod źródłowy
│   ├── algorithms/                # Podstawowe algorytmy QH
│   │   ├── qh_qlearning.py       # Implementacja QH Q-Learning
│   │   └── qh_policy_evaluation.py # Algorytm oceny polityki
│   ├── models/                    # Modele środowisk MDP
│   │   └── mdp_environments.py    # MDP zapasów i GridWorld
│   ├── experiments/               # Framework eksperymentalny
│   │   ├── experiment_runner.py   # Główny runner eksperymentów
│   │   └── comparison_standard_vs_qh.py # Porównanie Standard vs QH
│   ├── utils/                     # Funkcje pomocnicze
│   │   └── analysis_tools.py      # Narzędzia analizy i wizualizacji
│   └── tests/                     # Testy jednostkowe
├── notebooks/                     # Notatniki Jupyter
│   └── standard_vs_qh_comparison.ipynb # Interaktywna analiza porównawcza
├── data/                          # Katalog danych
│   ├── datasets/                  # Zestawy danych wejściowych
│   ├── results/                   # Wyniki eksperymentów
│   └── plots/                     # Wygenerowane wykresy
├── references/                    # Materiały referencyjne
│   ├── papers/                    # Artykuły naukowe (PDF)
│   └── books/                     # Książki i monografie
└── README.md                      # Ten plik
```

## Przegląd badań

### Sformułowanie problemu

Tradycyjne uczenie ze wzmocnieniem zakłada dyskontowanie wykładnicze i spójne czasowo preferencje. Jednak rzeczywiste podejmowanie decyzji często obejmuje:

- **Niespójność czasowa**: Preferencje między przyszłymi nagrodami zmieniają się w czasie
- **Uprzedzenie teraźniejszości**: Przeważanie nagród natychmiastowych nad przyszłymi
- **Ograniczona racjonalność**: Systematyczne odchylenia od idealnie racjonalnego zachowania

Praca bada **dyskontowanie quasi-hiperboliczne (QH)** jako alternatywę dla dyskontowania wykładniczego, dostarczając bardziej realistyczne modele zachowań decyzyjnych.

### Główne wkłady

1. **Ramy teoretyczne**: Rozszerzenie teorii MDP o dyskontowanie quasi-hiperboliczne
2. **Rozwój algorytmów**: 
   - Bezmodelowa ocena polityki z wykorzystaniem aproksymacji stochastycznej dwuskalowej
   - Algorytm QH Q-Learning z gwarancjami zbieżności
   - **Lokalne liczniki wizyt**: Poprawiona zbieżność dla rzadko odwiedzanych par stan-akcja (nowa implementacja)
3. **Zastosowanie praktyczne**: Model gromadzenia zapasów demonstrujący niespójne czasowo optymalne polityki
4. **Walidacja empiryczna**: Kompleksowa ewaluacja eksperymentalna i porównanie z tradycyjnymi metodami

> **Uwaga:** Implementacja QH Q-Learning używa lokalnych liczników wizyt per para (stan, akcja) zamiast globalnego licznika iteracji. To zapewnia poprawną zbieżność dla rzadko odwiedzanych par (błąd zmniejszony z >10.0 do <2.0). Zobacz `docs/LOCAL_COUNT_STEP_SIZES.md` po szczegóły.

### Model dyskontowania quasi-hiperbolicznego

Skumulowany zdyskontowany zwrot ma postać:
```
G = r₀ + α∑(β^t r_t) dla t=1 do ∞
```

Gdzie:
- `α ∈ [0,1]`: Parametr uprzedzenia teraźniejszości
- `β ∈ [0,1)`: Standardowy wykładniczy współczynnik dyskontowania
- `r₀`: Natychmiastowa nagroda
- `r_t`: Nagroda w czasie t

Gdy `α = 1`, redukuje się to do standardowego dyskontowania wykładniczego. Gdy `α < 1`, agent wykazuje uprzedzenie teraźniejszości.

## Rozpoczęcie pracy

### Wymagania wstępne

```bash
# Wymagany Python 3.8+
python -m pip install numpy scipy matplotlib pandas seaborn
```

### Uruchamianie eksperymentów

#### Szybki start: Porównanie Standard vs QH

```bash
# Uruchom skrypt porównawczy
cd src/experiments
python comparison_standard_vs_qh.py
```

Lub użyj interaktywnego notatnika Jupyter:
```bash
jupyter notebook notebooks/standard_vs_qh_comparison.ipynb
```

#### Użycie programistyczne

```python
from src.experiments.comparison_standard_vs_qh import MDPComparison
from src.models.mdp_environments import InventoryMDP

# Tworzenie środowiska
env = InventoryMDP(max_inventory=15, max_order=8)

# Inicjalizacja porównania
comparison = MDPComparison(
    env=env,
    alpha=0.7,   # Parametr uprzedzenia teraźniejszości
    beta=0.95,   # Współczynnik dyskontowania
    theta_step=0.1,   # Krok uczenia
    epsilon=0.1  # Współczynnik eksploracji
)

# Trenowanie obu algorytmów
comparison.train(n_episodes=5000, record_interval=100)

# Generowanie raportu
print(comparison.generate_report())

# Analiza polityk i wartości
policy_comp = comparison.compare_policies()
value_comp = comparison.compare_values()

# Sprawdzenie spójności czasowej
consistency = comparison.analyze_time_consistency(initial_state=5, horizon=10)

# Wizualizacja wyników
comparison.plot_comparison(save_path='results.png')
```

### Budowanie pracy dyplomowej

```bash
cd docs/thesis/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Zarządzanie dużymi plikami z Git LFS

Repozytorium wykorzystuje [Git Large File Storage](https://git-lfs.com/) dla plików binarnych jak PDF-y pracy i materiały referencyjne.

- Zainstaluj Git LFS raz na maszynie:
  ```bash
  git lfs install
  ```
- Śledź nowe wzorce binarne przed ich commitowaniem (przykłady):
  ```bash
  git lfs track "*.pdf"
  git lfs track "*.npz"
  git lfs track "data/results/*.json"
  ```
  To aktualizuje `.gitattributes`; commituj zmianę aby współpracownicy mieli te same reguły.
- Dodawaj, commituj i pushuj jak zwykle:
  ```bash
  git add path/to/artifact
  git commit -m "Dodaj nowe artefakty eksperymentów"
  git push origin main
  ```
- Jeśli musisz wymusić upload wszystkich obiektów LFS (np. na CI):
  ```bash
  git lfs push --all origin main
  ```

## Kontekst badawczy

### Literatura

Badania opierają się na kluczowych pracach:

- **Bertsekas (2019)**: "Reinforcement learning and optimal control" - Podstawy teoretyczne
- **Jaśkiewicz & Nowak (2021)**: "Markov decision processes with quasi-hyperbolic discounting" - Podstawowe ramy teoretyczne  
- **Eshwar et al. (2024)**: "Reinforcement learning with quasi-hyperbolic discounting" - Innowacje algorytmiczne

### Zastosowania

Praca demonstruje praktyczne zastosowania w:
- **Zarządzanie zapasami**: Optymalne polityki zamawiania przy niespójnych czasowo preferencjach
- **Ekonomia behawioralna**: Modelowanie ograniczonej racjonalności w sekwencyjnym podejmowaniu decyzji
- **Planowanie finansowe**: Długoterminowe strategie inwestycyjne z uprzedzeniem teraźniejszości

## Główne wyniki

### Wyniki teoretyczne

1. **Struktura polityki**: Optymalne polityki dla zobowiązanych agentów mają specyficzną strukturę dwuskładnikową
2. **Zbieżność**: Bezmodelowe algorytmy zbiegają do optymalnych polityk w standardowych warunkach
3. **Niespójność czasowa**: Dyskontowanie QH prowadzi do niespójnych czasowo optymalnych polityk

### Wyniki eksperymentalne

- Algorytmy QH skutecznie uczą się optymalnych polityk w różnych środowiskach
- Parametr uprzedzenia teraźniejszości znacząco wpływa na optymalne zachowanie
- Tradycyjne dyskontowanie wykładnicze jest przypadkiem szczególnym (α = 1)

### Porównanie: Standardowe vs Quasi-Hiperboliczne Dyskontowanie

Framework zapewnia kompleksowe porównanie między:

**Standardowy Q-Learning** (Dyskontowanie wykładnicze):
- Funkcja wartości: $V(s) = E[\sum_{t=0}^{\infty} \beta^t r_t]$
- Spójne czasowo preferencje
- Optymalny z perspektywy długoterminowej

**QH Q-Learning** (Dyskontowanie quasi-hiperboliczne):
- Funkcja wartości: $V(s) = E[r_0 + \alpha \sum_{t=1}^{\infty} \beta^t r_t]$
- Uprzedzenie teraźniejszości gdy α < 1
- Potencjalnie niespójny czasowo
- Lepiej modeluje ludzkie zachowania

**Kluczowe obserwacje:**
- Zgodność polityk maleje gdy α maleje (silniejsze uprzedzenie teraźniejszości)
- Dyskontowanie QH prowadzi do bardziej krótkowzrocznych decyzji
- Różnice w funkcjach wartości są znaczące w stanach wymagających długoterminowego planowania
- Niespójność czasowa przejawia się gdy wybory zobowiązane i krótkowzroczne się różnią

## Dostępne dokumenty

- **[Podsumowanie literatury](./docs/literature/1285_MS_Summary.md)** - Kompleksowe podsumowanie (po angielsku i polsku) artykułu badawczego "Teaching Precommitted Agents: Model-Free Policy Evaluation and Control in Quasi-Hyperbolic Discounted MDPs" autorstwa S.R. Eshwar
- **[Artykuł referencyjny](./references/papers/1285_MS.pdf)** - Oryginalny artykuł badawczy o dyskontowaniu quasi-hiperbolicznym w uczeniu ze wzmocnieniem

## Przyszłe prace

- Rozszerzenie na częściowo obserwowalne MDP (POMDP)
- Systemy wieloagentowe z niespójnymi czasowo preferencjami
- Ciągłe przestrzenie stanów/akcji
- Walidacja empiryczna z udziałem ludzi
- Zastosowania w robotyce i finansach

## Współpraca

Repozytorium wspiera badania naukowe. W przypadku pytań lub współpracy:

1. Sprawdź dokumentację w `docs/`
2. Przejrzyj istniejące eksperymenty w `src/experiments/`
3. Uruchom testy jednostkowe: `python -m pytest src/tests/`

## Licencja

Tylko do użytku akademickiego. Proszę cytować tę pracę przy wykorzystaniu jakiejkolwiek części kodu lub metodologii.

## Cytowanie

```bibtex
@mastersthesis{wrona2024qh,
  title={Algorytmy oparte o wzmocnione nauczanie maszynowe w dyskontowanych modelach markowskich},
  author={Michał Wrona},
  school={Politechnika Wrocławska, Wydział Matematyki},
  year={2024},
  type={Praca inżynierska},
  keywords={Uczenie ze wzmocnieniem, Dyskontowanie quasi-hiperboliczne, Markowskie procesy decyzyjne}
}
```

## Kontakt

**Autor:** Michał Wrona  
**Opiekun:** prof. dr hab. inż. Anna Jaśkiewicz  
**Instytucja:** Politechnika Wrocławska, Wydział Matematyki

---

*Repozytorium zawiera wszystkie materiały do pracy inżynierskiej o dyskontowaniu quasi-hiperbolicznym w uczeniu ze wzmocnieniem, włącznie z analizą teoretyczną, implementacjami algorytmów i walidacją eksperymentalną.*