# Controls paper

`controls.tex` -> `controls.pdf` (10 pages). Documents the four round-1 controls,
the condition confound, and the round-2 matched / selection-corrected re-analysis.

Figures are pulled from `../reports/` at build time, so run the analysis scripts first:

```
cd ..
python3 controls_check.py
python3 controls_check2.py
cd paper
pdflatex controls.tex && pdflatex controls.tex   # twice, for refs
```
