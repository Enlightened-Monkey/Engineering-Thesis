# Przewodnik szybkiego startu

Ten plik opisuje aktualne, dzialajace entrypointy eksperymentow.

## 5-minutowy szybki start

### Opcja 1: Reprodukcja eksperymentu inventory (Fig. 1)
```bash
python -m src.experiments.InventoryMDP
```

### Opcja 2: Kontrprzyklad 2-stanowy (Table 1c)
```bash
python -m src.experiments.two_state_counterexample
```

### Opcja 3: Wersja inventory M=3 (wariant parametrow)
```bash
python -m src.experiments.InventoryMDP_M3
```

### Opcja 4: Notatnik Jupyter
```bash
jupyter notebook notebooks/run_experiments.ipynb
```

## Srodowisko i zaleznosci

```bash
python -m pip install -r requirements.txt
```

Minimalnie do uruchamiania testow i narzedzi analizy potrzebne sa m.in.:
- numpy
- scipy
- matplotlib
- pandas
- seaborn
- pytest

## Uruchamianie testow

```bash
python -m pytest src/tests -q
```

Jesli testy zatrzymuja sie na imporcie seaborn, doinstaluj zaleznosci z requirements.txt.

## Co jest aktualnie nieaktywne

W repozytorium nie ma obecnie modulow:
- src/experiments/comparison_standard_vs_qh.py
- src/experiments/demo_comparison.py
- src/experiments/experiment_runner.py

Z tego powodu stare instrukcje oparte o te pliki sa traktowane jako historyczne.

## Gdzie dalej

- Aktualne entrypointy: src/experiments/README.md
- Opis Algorytmu 1: docs/qh_policy_evaluation_algorithm_notes.md
- Uwagi o harmonogramach krokow: docs/LOCAL_COUNT_STEP_SIZES.md
