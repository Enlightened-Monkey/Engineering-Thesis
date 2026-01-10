# Thesis Compilation Guide

This directory contains the LaTeX source files for the engineering thesis on reinforcement learning algorithms with quasi-hyperbolic discounting.

## Prerequisites

To compile the thesis, you need the following packages installed:

- `texlive` - Basic LaTeX distribution
- `texlive-latex-extra` - Additional LaTeX packages
- `texlive-bibtex-extra` - BibTeX styles and tools
- `texlive-lang-polish` - Polish language support
- `texlive-science` - Scientific packages (including algorithm)
- `biber` - Bibliography processor (optional, if using biblatex)

### Installation on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y texlive texlive-latex-extra texlive-bibtex-extra \
                        texlive-lang-polish texlive-science biber
```

## Compilation

To compile the thesis, run the following commands from this directory:

```bash
# First pass: generate auxiliary files
pdflatex main.tex

# Process bibliography
bibtex main

# Second pass: resolve citations
pdflatex main.tex

# Third pass: resolve all cross-references
pdflatex main.tex
```

Or use the shorthand command:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The compiled PDF will be generated as `main.pdf`.

## Cleaning Build Artifacts

To clean up auxiliary files:

```bash
rm -f *.aux *.bbl *.blg *.log *.out *.toc chapters/*.aux
```

Note: The `.gitignore` file is already configured to exclude these auxiliary files from version control.

## File Structure

- `main.tex` - Main thesis file
- `chapters/` - Chapter files
  - `introduction.tex` - Introduction chapter
  - `model.tex` - Model description chapter
  - `main_part.tex` - Main theorems and algorithms
  - `experiments.tex` - Experiments chapter
  - `conclusion_simplified.tex` - Conclusions
- `bibliography/` - Bibliography files
  - `references.bib` - BibTeX references
- `fig/` - Figures directory
- `../plots/` - Generated plots (from parent docs/ directory)

## Known Issues and Solutions

### Issue 1: Missing Polish Language Support

**Error:** `Package babel Error: Unknown option 'polish'`

**Solution:** Install `texlive-lang-polish` package.

### Issue 2: Missing algorithm Package

**Error:** `LaTeX Error: File 'algorithm.sty' not found`

**Solution:** Install `texlive-science` package.

### Issue 3: Missing Images

If you encounter errors about missing image files, ensure that:
1. All image paths are relative (not absolute)
2. Images exist in the referenced locations:
   - `fig/` for diagrams
   - `../plots/` for generated plots
   - `../../data/plots/` for data-related plots

## Recent Fixes (January 2026)

The following issues were resolved:

1. **Absolute paths**: Replaced user-specific absolute paths with relative paths
2. **Case sensitivity**: Fixed filename case mismatches (e.g., `ex2plots.png` → `ex2Plots.png`)
3. **Missing files**: Commented out references to non-existent image files
4. **Path corrections**: Fixed incorrect relative paths to match actual file locations

All bibliography and reference issues have been resolved, and the document now compiles cleanly.
