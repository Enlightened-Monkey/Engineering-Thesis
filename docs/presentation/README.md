# Presentation Materials

This directory contains materials for thesis defense and presentations.

## Contents

- `defense_slides.tex` - Main defense presentation (LaTeX Beamer)
- `defense_slides.pdf` - Compiled presentation
- `figures/` - Presentation-specific figures and diagrams
- `handout/` - Handout materials for committee members
- `poster/` - Conference poster materials (if applicable)

## Building the Presentation

```bash
cd docs/presentation/
pdflatex defense_slides.tex
pdflatex defense_slides.tex  # Run twice for references
```

## Presentation Structure

1. **Introduction** - Problem motivation and research goals
2. **Background** - Theoretical foundations and literature review
3. **Methodology** - QH discounting framework and algorithms
4. **Results** - Experimental validation and key findings
5. **Applications** - Inventory management case study
6. **Conclusions** - Contributions and future work

## Notes

- Presentation duration: 20-25 minutes + questions
- Focus on key theoretical and practical contributions
- Include live demonstration of algorithms (optional)
- Prepare backup slides for detailed questions