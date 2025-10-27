# Presentation Materials

This directory contains materials for thesis defense and presentations.

## Contents

- `presentation.tex` – Main Beamer presentation (16:9) with modular chapter inputs
- `chapters/` – Section files (`title.tex`, `intro.tex`, `results.tex`, `summary.tex`)
- `presentation.pdf` – Compiled presentation (not committed)
- `figures/` – Presentation-specific figures and diagrams
- `handout/` – Handout materials for committee members
- `poster/` – Conference poster materials (if applicable)
- `../../data/plots/pole_balancing_100k_frames/` – Auto-generated animation frames and manifests

## Building the Presentation

# Frame extraction (one-time, regenerates animation assets)
```bash
cd /home/michal/Engineering-Thesis
python3 src/utils/gif_to_frames.py
```

This converts `data/plots/pole_balancing_100k.gif` into PNG frames and writes
`manifest.tex`, which defines macros used by `chapters/results.tex`.

```bash
cd docs/presentation/
make            # builds presentation.pdf
make clean      # removes build artefacts and the pdf
```

## Presentation Structure

1. **Wstęp** – Motywacja, cele i plan wystąpienia (`chapters/intro.tex`)
2. **Wyniki własne** – Najważniejsze wnioski z eksperymentów (`chapters/results.tex`)
3. **Podsumowanie** – Wnioski oraz dalsze kierunki (`chapters/summary.tex`)

Każda część prezentacji ma osobny plik w katalogu `chapters/` i jest dołączana do `presentation.tex` przez `\input{...}`.

## Notes

- Presentation duration: 20-25 minutes + questions
- Focus on key theoretical and practical contributions
- Animated pole-balancing demo uses the `animate` LaTeX package; ensure your
	TeX distribution includes it (TeX Live: `collection-latexextra`).
- Include live demonstration of algorithms (optional)
- Prepare backup slides for detailed questions