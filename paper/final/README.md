# Final paper — build instructions

`main.pdf` is the **anonymized** build (submission). `main_named.pdf` is the
non-anonymized build with a placeholder author block. Neither contains any real
name, affiliation, repository URL, or handle — verified by grepping the PDF text.

## Full rebuild from scratch

Figures are pulled from `../../reports/` at build time, so the analysis must run
first. From the repository root:

```bash
# 1. Regenerate every number and figure (CPU only, a few minutes)
python3 controls_check.py          # round-1 controls, depth profiles, saturation
python3 controls_check2.py         # matched stratum, nulls, split-half, attn sweep
python3 sim/existence_null_sim.py  # Figure 1 + reports/sim_summary.json
python3 depth_profile_matched.py   # Figures 3 and 5 + per-depth CSVs
python3 paper_extras.py            # per-head corroboration, behavioural rates
python3 run_metadata.py            # compute facts from the per-run configs

# 2. Build BOTH PDFs and verify them
cd paper/final && ./build.sh
```

**`build.sh` is the only supported way to build the paper.** Do not run
`pdflatex` by hand and do not `cp main.pdf main_named.pdf` — that is exactly how
the two builds silently drifted apart once before.

### What build.sh does

1. Cleans aux files and builds the **anonymized** PDF (`main.pdf`), then the
   **named** PDF (`main_named.pdf`), both from the single `main.tex`.
2. Enforces on each build: 0 LaTeX errors, 0 undefined references or citations,
   0 overfull boxes above 10 pt.
3. Extracts text from both and **diffs them outside a whitelist**, failing if
   they differ anywhere else.
4. Checks the **main-text page budget**: the page on which References starts,
   minus one, must be at most 8. Total pages including references, appendices and
   the checklist is not the number that matters and is reported separately.
5. Asserts the **venue notice**: the style's default NeurIPS string must be
   absent from both PDFs and the current `\venuename` value must be present.

Exit codes: `0` ok, `1` LaTeX error, `2` text diff outside the whitelist,
`3` assertion failure (page budget or venue notice).

### The whitelist

The only differences permitted between the two builds:

1. **Author block** — anonymous placeholder vs the named placeholder and its
   `\thanks` footnote.
2. **Notice string** — the two builds carry different venue notices.
3. **Line-number gutter** — the anonymous build enables `lineno` for reviewers.
   Rather than strip the gutter out of extracted text (unsafe: it also removes
   figure axis tick labels), the diff uses a third throwaway anonymous build
   compiled with `\DIFFBUILD`, which suppresses the gutter at the source. The
   shipped `main.pdf` keeps its line numbers.

## Switches (top of `main.tex`)

| Switch | Effect |
|---|---|
| `\anontrue` | anonymized build (default). Produces `main.pdf` |
| `\anonfalse` | named build with a placeholder author block |
| `\checklisttrue` | include the NeurIPS checklist (mandatory — keep on) |
| `\venuename` | workshop name, fed to `\workshoptitle` |

`build.sh` sets and restores `\ifanon` itself; you should not edit it by hand.

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

## Supplementary archive

`supplementary_anon.zip` is a **build product** and is gitignored. Rebuild it
from the repository root by staging the scripts, the per-depth CSVs and configs,
and everything in `reports/`, then zipping. It must be anonymity-grepped before
release. Current size 7.8 MB, 62 files — well under a 50 MB cap, so the raw CSVs
are included.

## Length

`build.sh` reports and enforces this. Main text currently ends on page 7 with
References starting on page 8; appendices and the checklist follow and do not
count. Re-check after any track change, since margins differ between tracks.
