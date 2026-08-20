# Przewodnik Porownan: Status Dokumentu

## Uwaga

Ten dokument opisuje w duzej czesci historyczny modul
src/experiments/comparison_standard_vs_qh.py, ktorego obecnie nie ma
w repozytorium.

Dokument zostaje zachowany jako notatka koncepcyjna, ale nie powinien byc
traktowany jako instrukcja uruchamiania aktualnego kodu.

## Aktualne sciezki uruchamiania

Do reprodukcji wynikow i pracy z obecnym kodem uzywaj:

1. python -m src.experiments.InventoryMDP
2. python -m src.experiments.two_state_counterexample
3. python -m src.experiments.InventoryMDP_M3

Aktualna lista wejsc: src/experiments/README.md

## Co nadal jest wartosciowe z tego dokumentu

- Opis roznicy miedzy dyskontowaniem wykladniczym i quasi-hiperbolicznym
- Intuicja present-bias i niespojnosci czasowej
- Kierunki rozszerzen badawczych

## Co wymaga ostroznosci

- Wszystkie importy i przyklady oparte o MDPComparison
- Opisy StandardQLearning jako gotowej klasy produkcyjnej w tym repo
- Odniesienia do notebooka standard_vs_qh_comparison.ipynb jako glownej sciezki

## Rekomendacja

Jesli potrzebny jest ponownie dedykowany framework porownawczy Standard vs QH,
najpierw nalezy odtworzyc brakujacy modul comparison_standard_vs_qh.py,
a dopiero potem reaktywowac ten dokument jako instrukcje wykonawcza.
