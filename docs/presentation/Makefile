LATEX=pdflatex
MAIN=presentation
TEX=$(MAIN).tex
PDF=$(MAIN).pdf

all: $(PDF)

$(PDF): $(TEX) $(wildcard chapters/*.tex)
	$(LATEX) -interaction=nonstopmode $(TEX)
	$(LATEX) -interaction=nonstopmode $(TEX)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).nav $(MAIN).out $(MAIN).snm $(MAIN).toc $(MAIN).vrb $(PDF)

.PHONY: all clean
