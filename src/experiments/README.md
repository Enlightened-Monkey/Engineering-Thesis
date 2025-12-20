# Moduł eksperymentów

Ten katalog zawiera skrypty do reprodukcji wyników (tabele/rysunki) i krótkich sanity-checków.

Zalecane uruchamianie:

```bash
python -m src.experiments.<nazwa_modułu>
```

## Gdzie trafiają wyniki

- Wykresy: zwykle do `plots/` (czasem do katalogu roboczego — patrz `print()` w skrypcie).
- Artefakty treningu (np. snapshoty modeli): `data/models/`.
- Wyniki/metryki (np. JSON): `data/results/`.
- Notebook do odpalania eksperymentów: `notebooks/run_experiments.ipynb` (zapisuje do `plots/notebook_runs/`).

## Aktualne entrypointy

### 1) Inventory Control (Fig. 1)

**Moduł:** `InventoryMDP`

Reprodukcja eksperymentu inventory + zbieżność policy evaluation. Skrypt zapisuje `reproduced_fig1.png`.

```bash
python -m src.experiments.InventoryMDP
```

### 2) 2‑state counterexample (Table 1c)

**Moduł:** `two_state_counterexample`

Reprodukcja kontrprzykładu 2‑stanowego i wartości z Table 1c.

```bash
python -m src.experiments.two_state_counterexample
```

## Notatniki Jupyter

- `../notebooks/run_experiments.ipynb`
- `../notebooks/standard_vs_qh_comparison.ipynb`

## Dokumentacja

- `../docs/COMPARISON_GUIDE.md`
- `../docs/LOCAL_COUNT_STEP_SIZES.md`
