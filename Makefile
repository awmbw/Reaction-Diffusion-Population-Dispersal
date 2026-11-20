PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: test outputs report report-tectonic

test:
	$(PYTHON) -m pytest tests -q

outputs:
	$(PYTHON) run_all.py

# Requires TeX Live with latexmk, latex-extra, fonts-recommended and pictures.
report:
	cd report && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error report.tex
	cd report && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error code_listing.tex
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('submission'); p.mkdir(exist_ok=True); shutil.copy2('report/report.pdf', p/'Reaction_Diffusion_Project_Report.pdf'); shutil.copy2('report/code_listing.pdf', p/'Reaction_Diffusion_Complete_Code.pdf')"

# Portable alternative; Tectonic fetches its own TeX packages on first use.
report-tectonic:
	cd report && tectonic report.tex
	cd report && tectonic code_listing.tex
	$(PYTHON) -c "from pathlib import Path; import shutil; p=Path('submission'); p.mkdir(exist_ok=True); shutil.copy2('report/report.pdf', p/'Reaction_Diffusion_Project_Report.pdf'); shutil.copy2('report/code_listing.pdf', p/'Reaction_Diffusion_Complete_Code.pdf')"
