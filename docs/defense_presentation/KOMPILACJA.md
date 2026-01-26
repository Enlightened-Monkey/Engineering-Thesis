# Instrukcja kompilacji prezentacji obronowej

## Wymagania systemowe

Aby skompilować prezentację, potrzebujesz zainstalowanej dystrybucji LaTeX z następującymi pakietami:

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-lang-polish
```

### macOS
```bash
brew install --cask mactex
```
lub pobierz BasicTeX i doinstaluj potrzebne pakiety:
```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install beamer babel-polish
```

### Windows
Pobierz i zainstaluj [MiKTeX](https://miktex.org/download) lub [TeX Live](https://www.tug.org/texlive/)

## Kompilacja

### Metoda 1: Przez terminal (zalecana)
```bash
cd docs/defense_presentation
pdflatex presentation.tex
pdflatex presentation.tex  # druga kompilacja dla poprawnych odnośników
```

### Metoda 2: Używając Makefile
```bash
cd docs/defense_presentation
make           # kompiluje prezentację
make view      # otwiera skompilowany PDF (Linux)
make clean     # usuwa pliki pomocnicze
make cleanall  # usuwa wszystkie wygenerowane pliki
```

### Metoda 3: Używając latexmk (najwygodniejsza)
```bash
cd docs/defense_presentation
latexmk -pdf presentation.tex
latexmk -c  # czyszczenie plików pomocniczych
```

### Metoda 4: W edytorze LaTeX
Otwórz plik `presentation.tex` w swoim ulubionym edytorze LaTeX (TeXstudio, Overleaf, TeXworks, etc.) i użyj wbudowanej funkcji kompilacji.

## Rozwiązywanie problemów

### Problem: Brakujące pakiety
Jeśli podczas kompilacji pojawią się błędy o brakujących pakietach:

**TeX Live:**
```bash
sudo tlmgr install <nazwa-pakietu>
```

**MiKTeX:**
MiKTeX automatycznie pobierze brakujące pakiety podczas kompilacji (jeśli ta opcja jest włączona w ustawieniach).

### Problem: Polskie znaki się nie wyświetlają
Upewnij się, że:
1. Plik jest zapisany w kodowaniu UTF-8
2. Zainstalowany jest pakiet `babel-polish`
3. W preambule jest `\usepackage[polish]{babel}`

### Problem: Brak obrazków
Upewnij się, że folder `../thesis/fig/` zawiera plik `example1.png`. Ścieżka jest relatywna do lokalizacji pliku `presentation.tex`.

## Struktura plików wyjściowych

Po kompilacji zostaną utworzone następujące pliki:
- `presentation.pdf` - skompilowana prezentacja (główny plik)
- `presentation.aux` - plik pomocniczy (można usunąć)
- `presentation.log` - log kompilacji (można usunąć)
- `presentation.nav` - nawigacja beamera (można usunąć)
- `presentation.out` - bookmarki PDF (można usunąć)
- `presentation.snm` - fragmenty beamera (można usunąć)
- `presentation.toc` - spis treści (można usunąć)
- `presentation.vrb` - verbatim beamera (można usunąć)

Pliki pomocnicze (*.aux, *.log, etc.) można bezpiecznie usunąć za pomocą `make clean`.

## Szybki start dla początkujących

1. Zainstaluj TeX Live (Linux/Mac) lub MiKTeX (Windows)
2. Otwórz terminal i przejdź do folderu:
   ```bash
   cd docs/defense_presentation
   ```
3. Uruchom kompilację:
   ```bash
   pdflatex presentation.tex
   pdflatex presentation.tex
   ```
4. Otwórz plik `presentation.pdf`

## Edycja prezentacji

Główne pliki do edycji:
- `presentation.tex` - ustawienia globalne, struktura prezentacji
- `chapters/title.tex` - strona tytułowa
- `chapters/introduction.tex` - wstęp (motywacja, oznaczenia, teoria)
- `chapters/results.tex` - wyniki eksperymentów
- `chapters/summary.tex` - podsumowanie i bibliografia

Po każdej edycji należy ponownie skompilować prezentację (2x `pdflatex` lub `make`).

## Wsparcie

W razie problemów z kompilacją:
1. Sprawdź plik `presentation.log` w poszukiwaniu komunikatów o błędach
2. Upewnij się, że wszystkie wymagane pakiety są zainstalowane
3. Sprawdź, czy wszystkie pliki w folderze `chapters/` są obecne
4. Zweryfikuj, czy obrazki w `../thesis/fig/` istnieją
