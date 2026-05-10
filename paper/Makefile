LATEXMK := latexmk
REVIEW_TEX := paper.tex
FINAL_TEX := paper_final.tex
PREPRINT_TEX := paper_preprint.tex
REVIEW_PDF := paper.pdf
FINAL_PDF := paper_final.pdf
PREPRINT_PDF := paper_preprint.pdf

.PHONY: all review final preprint clean cleanall open check check-review check-final check-preprint

all: review

review:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(REVIEW_TEX)

final:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(FINAL_TEX)

preprint:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(PREPRINT_TEX)

clean:
	$(LATEXMK) -c

cleanall:
	$(LATEXMK) -C
	rm -f $(REVIEW_PDF) $(FINAL_PDF) $(PREPRINT_PDF)

check:
	@echo "Use check-review, check-final, or check-preprint."

check-review:
	pdffonts $(REVIEW_PDF)
	pdfinfo $(REVIEW_PDF) | rg "Page size"

check-final:
	pdffonts $(FINAL_PDF)
	pdfinfo $(FINAL_PDF) | rg "Page size"

check-preprint:
	pdffonts $(PREPRINT_PDF)
	pdfinfo $(PREPRINT_PDF) | rg "Page size"

open:
	@echo "Open one of: $(REVIEW_PDF), $(FINAL_PDF), $(PREPRINT_PDF)."
