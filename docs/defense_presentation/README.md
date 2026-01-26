# Prezentacja obrony pracy inżynierskiej

Ten folder zawiera prezentację LaTeX do obrony pracy inżynierskiej pt. "Algorytmy oparte o wzmocnione nauczanie maszynowe w dyskontowanych modelach markowskich".

## Struktura

- `presentation.tex` - główny plik prezentacji
- `chapters/` - poszczególne rozdziały prezentacji:
  - `title.tex` - strona tytułowa
  - `introduction.tex` - wstęp (motywacja, oznaczenia, kluczowe wyniki z literatury)
  - `results.tex` - wyniki własne (eksperymenty)
  - `summary.tex` - podsumowanie (wnioski, dalsze prace, bibliografia)

## Kompilacja

### Metoda 1: pdflatex
```bash
pdflatex presentation.tex
pdflatex presentation.tex  # drugie uruchomienie dla poprawnych odnośników
```

### Metoda 2: Makefile
```bash
make          # kompilacja
make clean    # czyszczenie plików pomocniczych
```

### Metoda 3: latexmk (zalecane)
```bash
latexmk -pdf presentation.tex
```

## Wymagania

- LaTeX (dystrybucja TeX Live lub MiKTeX)
- Pakiety: beamer, babel, algorithm, algorithmic, booktabs, hyperref
- Temat: Madrid (beamer)
- Schemat kolorów: beaver

## Informacje o prezentacji

- **Czas trwania**: 10 minut
- **Format**: Beamer 16:9
- **Język**: Polski
- **Autor**: Michał Wrona
- **Promotor**: prof. dr hab. inż. Anna Jaśkiewicz
- **Uczelnia**: Politechnika Wrocławska, Wydział Matematyki

## Zawartość prezentacji

1. **Strona tytułowa** - tytuł, autor, promotor
2. **Wstęp**:
   - Plan prezentacji
   - Motywacja wyboru tematu
   - Wprowadzenie oznaczeń matematycznych
   - Kluczowe wyniki z literatury (twierdzenia bez dowodów, z odnośnikami)
3. **Wyniki własne**:
   - Środowiska testowe
   - MDP z dwoma stanami
   - Problem zarządzania zapasami
   - Implementacja i metodologia
4. **Podsumowanie**:
   - Wnioski
   - Znaczenie niespójności czasowej
   - Dalsze prace
   - Bibliografia
   - Pytania

## Uwagi

- **WAŻNE:** Prezentacja wykorzystuje wykresy i diagramy z folderu `../thesis/fig/`
  - Wymagany plik: `../thesis/fig/example1.png` 
  - Upewnij się, że ten plik istnieje przed kompilacją
- Bibliografia jest zawarta bezpośrednio w pliku `summary.tex` (środowisko `thebibliography`)
- Wszystkie wzory matematyczne są w notacji LaTeX zgodnej z pracą dyplomową

## Zależności struktury folderów

Prezentacja zakłada następującą strukturę katalogów:
```
docs/
├── defense_presentation/  (ten folder)
│   ├── presentation.tex
│   └── chapters/
└── thesis/
    └── fig/
        └── example1.png   (wymagany!)
```
