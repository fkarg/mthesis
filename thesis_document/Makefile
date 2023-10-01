# Latex Makefile using latexmk
# Modified by Dogukan Cagatay <dcagatay@gmail.com>
# Modified by Philipp Jund
# Modified by Felix Karg <fkarg10@gmail.com>
# Adapted from : http://tex.stackexchange.com/a/40759
# Additional Source:
# https://serverfault.com/questions/153875/how-to-let-cp-command-dont-fire-an-error-when-source-file-does-not-exist


PROJNAME=main
PDFNAMES=$(addsuffix .pdf, $(PROJNAME))
OUT_DIR=out

.PHONY: all spell clean dist-clean 

all: $(PDFNAMES)

diff: $(addsuffix -diff.pdf, $(PROJNAME))

spell: chapters/*.tex
	nix-build -E 'with import <nixpkgs> {}; aspellWithDicts (d: [d.de])'; \
	for file in $^; do \
	    echo ; \
	    echo ------------------------------; \
	    echo Correcting file: echo $${file}; \
	    bash result/bin/aspell -l de check $${file}; \
	done

watch:
	# watch --interval 5 --errexit make
	watch -n 5 -e make

clean:
	rm -rf $(OUT_DIR)

dist-clean: clean
	rm -rf $(PDFNAMES)
	rm -rf $(addsuffix -diff.pdf, $(PROJNAME))


%-diff.pdf: %.tex *.tex */*.tex
	# nix-shell -p git-latexdiff
	git latexdiff --latexmk --main main.tex --output main-diff.pdf HEAD --

%.pdf: %.tex *.tex */*.tex references.bib code/* img/*
	mkdir -p $(OUT_DIR)/chapters
	mkdir -p $(OUT_DIR)/include
	mkdir -p $(OUT_DIR)/figures
	# -interaction=nonstopmode
	latexmk -outdir=$(OUT_DIR) -pdf -use-make -file-line-error -shell-escape -g $<
	cp -v $(foreach PDF, $(PDFNAMES),$(OUT_DIR)/$(PDF)) . || :
