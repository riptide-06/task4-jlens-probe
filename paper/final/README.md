# Final paper — build instructions

`main.pdf` is the **anonymized** build (submission). `main_named.pdf` is the
non-anonymized build with a placeholder author block. Neither contains any real
name, affiliation, repository URL, or handle — verified by grepping the PDF text.

## Full rebuild from scratch

Figures are pulled from `../../reports/` at build time, so the analysis must run
first. From the repository root:

```bash
# 1. Regenerate every number and figure (CPU only, a few minutes)
python3 controls_check.py          # -> reports/controls_output.txt  + 3 figures
python3 controls_check2.py         # -> reports/controls2_output.txt + 4 figures
python3 sim/existence_null_sim.py  # -> reports/existence_null_sim.pdf (Figure 1)

# 2. Build the anonymized PDF
cd paper/final
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`pdflatex` must run **twice after** `bibtex` or citations and cross-references
will be undefined.

## Switches (top of `main.tex`)

| Switch | Effect |
|---|---|
| `\anontrue` | anonymized build (default). Produces `main.pdf` |
| `\anonfalse` | named build with a placeholder author block |
| `\checklisttrue` | include the NeurIPS checklist (mandatory — keep on) |
| `\venuename` | workshop name, fed to `\workshoptitle` |

To produce both PDFs:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main   # anon
cp main.pdf main_anon.pdf
sed -i '' 's/\\anontrue/\\anonfalse/' main.tex
pdflatex main && bibtex main && pdflatex main && pdflatex main
cp main.pdf main_named.pdf
sed -i '' 's/\\anonfalse/\\anontrue/' main.tex   # restore the default
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Venue / track

`neurips_2026.sty` is the **official** style file (dated 2026-01-29), used
verbatim. The track option is set in the preamble:

```latex
\usepackage[dblblindworkshop]{neurips_2026}   % workshop, double-blind
```

Other options the style accepts: `main`, `position`, `eandd`, `creativeai`,
`education`, `sglblindworkshop`. Add `final` alongside the track for the
camera-ready. Note the style prints the generic "Submitted to …" notice for all
submission builds and only substitutes the track/workshop string in the `final`
build — that is expected, not a misconfiguration.

## Requirements

TeX Live / TinyTeX with `psnfss`, `courier`, `times`, `helvetic`, `natbib`,
`microtype`, `booktabs`, `environ`, `lineno`. Install missing ones with
`tlmgr install <pkg>`.

## Length

Main text ends on page 6; references start on page 7; appendices and the
checklist follow. The 8-page main-text limit is met with room to spare. Re-check
after any track change, since margins differ between tracks.
