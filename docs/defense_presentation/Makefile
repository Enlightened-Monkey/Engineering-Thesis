# Makefile for building the defense presentation

# Main file name (without .tex extension)
TARGET = presentation

# LaTeX compiler
LATEX = pdflatex

# Output directory
OUTDIR = .

.PHONY: all clean cleanall view

# Default target
all: $(TARGET).pdf

# Build PDF
$(TARGET).pdf: $(TARGET).tex chapters/*.tex
	$(LATEX) -output-directory=$(OUTDIR) $(TARGET).tex
	$(LATEX) -output-directory=$(OUTDIR) $(TARGET).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.fls *.fdb_latexmk

# Clean all generated files including PDF
cleanall: clean
	rm -f $(TARGET).pdf

# View PDF (Linux with evince)
view: $(TARGET).pdf
	@if command -v evince >/dev/null 2>&1; then \
		evince $(TARGET).pdf & \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(TARGET).pdf & \
	else \
		echo "No PDF viewer found. Please open $(TARGET).pdf manually."; \
	fi
