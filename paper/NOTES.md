# NOTES.md — working notes for the final paper

Branch `paper/final`. Deliverable: `paper/final/main.pdf` (anonymized) and
`main_named.pdf`. The earlier draft `paper/controls.tex` / `controls.pdf` is
**superseded** and retained only for diffing; it is not the deliverable.

---

## 0. Blocking issue found at the start

**`docs/TASK4_CONTEXT.md` does not exist in this repository.** The task prompt
cites it for the J-lens self-check disclosure (§9) and the patching spec (§10).
Neither the file nor any `docs/` directory is present, on any branch. The prompt
reproduces the substance of both sections, so the work was not blocked, but the
non-negotiable "every number comes from a script in this repo" meant those
claims had to be checked against the code rather than copied. See §4 below —
**two of the numbers attributed to §9 could not be reproduced and are not in the
paper.**

---

## 1. Outline diff

| Draft (`paper/controls.tex`, 10pp) | Final (`paper/final/main.tex`) | Change |
|---|---|---|
| Title: *Controls on an Internal-Representation Claim … Stale-State Tracking Failures in Qwen* | *"Present in Some Layer" Is Not Evidence: Controls for Layer-Existence Claims in Mechanistic Interpretability* | Reframed: general pattern first, case study second |
| §1 Introduction (about the stale-state claim) | §1 Introduction (about the inference pattern) | Rewritten. States the claim was **our own** prior analysis |
| *(absent)* | §2 Background and related work | **New.** 15 verified references |
| *(absent)* | §3 The statistic and its nulls (model-free) | **New.** Notation, $E$/$G$, Null A, Null B as max-$T$, split-half, excluded bands, Figure 1 from simulation |
| §2 Data | §4 Case study (setup) | Condensed; self-contained so it can be swapped out |
| §3 Round 1: Four Controls | §5.1 (one table) | Four subsections → one table |
| §3.3 Magnitude / reconciliation | §5.2 | Kept, condensed |
| §4 A Condition Confound | §5.3 | Kept |
| §5 Round 2 | §5.4 "What survives" | Merged into one table |
| §6 Limitations (4 items) | §6 Limitations (5 items) | **Added J-lens self-check disclosure**; kept "correlational" prominent |
| §7 Recommendations | §7 Recommendations | Generalized away from the case study |
| Appendix: reproduction | Appendices A–E | Added provenance, patching spec, J-lens diagnostic, LLM statement |
| *(absent)* | Official NeurIPS checklist | **New**, 16 items filled |

Sections 1–3 contain no case-study-specific numbers, so the case study
(§4, §5, Appendices A–C) can be swapped without touching them.

---

## 2. Number map

Rebuild with `python3 controls_check.py && python3 controls_check2.py &&
python3 sim/existence_null_sim.py`. Every figure below was copied from the named
output file; none was typed from memory.

### From `reports/controls_output.txt` (controls_check.py)
| Number | Where in paper | Locate in file |
|---|---|---|
| 137/137 = 100.0% correct wins ≥1 depth | §5.1 Table 2 | "CONTROL 1 … correct outranks stale" |
| 137/137 = 100.0% stale wins ≥1 depth | §5.1 Table 2, abstract | "symmetry control" |
| 488/488 = 100.0% on successes | §5.1 Table 2, abstract | "CONTROL 2 … SUCCESSES" |
| median 9 / 20 depths won | §3, §5.1 | "median # layers correct wins" |
| 47,343.5 and 2.3 | §5.2, abstract | "RECONCILIATION" block |
| depth ≥26 in 100% of failures | §5.2 | "best rank_correct is achieved" |
| 13.1% correct beats stale at that depth | §5.2 | "AT that best-rank layer" |
| median depth 9; ranks 24,183 vs 103,440 | §5.2 | "MAX-MARGIN layer" |
| V = 152,064 | §4 | "vocabulary size V" |
| 29 residual-stream positions, depths 0..28 | §4 | "residual-stream positions used for the max" |

### From `reports/controls2_output.txt` (controls_check2.py)
| Number | Where | Locate in file |
|---|---|---|
| 100% / 29.3% / 70.7% / 52.5% confound table | §5.3 | "CONFOUND CHECK" (also §1 of controls_check2 output) |
| 72 of 488; 682 of 4,853 matched | §5.3, §6 | "MATCHED STRATUM" |
| Null A: point mass at 1.0000, p = 1.00 | §5.1, abstract | "STATISTIC 1 (the claim)" |
| G = 10.93 vs null 14.48 [13.12, 15.82] | §5.1, abstract | "STATISTIC 2 (graded)" |
| J-lens G = 8.41 vs 14.34 | §5.1 | same block, jlens |
| attn depth 22: raw .994, matched .973 | §5.4 Table 3 | "VERDICT ON THE TWO SURVIVING" |
| jlens depth 5: raw .958, matched .888 | §5.4 Table 3 | same |
| logitlens depth 18: matched .703, 32% stable | §5.4 | same |
| held-out .970 / .882 / .669 | §5.4 Table 3 | "SPLIT-HALF" |
| depth stability 90% / 96% / 32% | §5.4 Table 3 | "selected-layer distribution" |
| Null B p < 1e-4 (all three) | §5.4 Table 3 | "selection-corrected p" |
| jlens depth 13: .952 → .700 | §5.3 | "RAW vs MATCHED" table |
| 5k replication .883 | §5.4 | "large-sample matched replication" |
| deviation AUC .871; cross-depth corr .268 vs .538 | §6 | "4b. J-LENS DECOMPOSITION" |
| 5,733 attn NaNs, all depth 28 | §5.4, §6 | "MISSING-DATA POLICY" |

### From `reports/existence_null_sim_output.txt` (sim)
| Number | Where | Locate |
|---|---|---|
| P(E=1) = 0.9740 at L=29 | §3 | "(a) P(E=1) for two EXCHANGEABLE" |
| ~92% with c genuinely worse | §3 | panel (a), offset −0.75 curve |
| E null sd 5e-4; G 11.86 vs 14.50 | §3 | "(b) Null A at L = 29" |
| E stays above 0.88 across offsets | §3 | "(c) Sensitivity" |

### From `outputs/*/config.json`
Model `Qwen/Qwen2.5-7B-Instruct`, bf16, `attn_impl: eager`, chat format,
`jlens_mode: autograd`, `jlens_baseline: zero`, transformers 5.14.1,
`self_check: **false**` in all three. Run sizes 5000 / 500 / 137.
Behavioural 97.1 / 2.7 / 0.2 % = 4853 / 137 / 10 of 5000.

### Resolved: numbers that were previously unmapped
The draft's J-lens decomposition figures (`0.911` deviation AUC, `0.425`/`0.578`
cross-depth correlations) came from an ad-hoc session and were not reproducible.
Folding the computation into `controls_check2.py` (section 4b) showed **why they
differed**: they had been computed on the UNMATCHED groups (137 vs 488). On the
matched groups the values are deviation AUC **0.871** at depth 5 (against matched
0.888) and cross-depth correlation **0.268** for J-lens against **0.538** for the
logit lens. The conclusion is unchanged and slightly stronger. The paper now
carries the matched, reproducible values.

**Every number in the final PDF is now mapped to a committed output file.**

---

## 3. Decisions

1. **Title.** Chose option 1 ("Present in Some Layer" Is Not Evidence). It names
   the pattern, is not case-study-specific, and survives swapping the case study.
2. **Self-correction framing.** Every mention of the 137/137 claim says it was
   ours ("our initial analysis reported", "our own prior result"). No other group
   is implicated.
3. **Terminology.** "29 residual-stream positions (depths 0..28)", never "29
   layers" — the model has 28 decoder layers. Read from `ResidualCapture` in
   `jlens_probe.py`: a pre-hook on each layer's input gives depths 0..27, a
   post-hook on the last gives depth 28.
4. **Depth 28 has no attention** because `AttentionCapture` hooks
   `layers[i].self_attn` for i in 0..27; attention is indexed by decoder layer, so
   no module corresponds to residual position 28. The draft's explanation ("the
   last block's attention output is not captured") was wrong and is fixed in code.
5. **Rank indexing / vocabulary.** `rank_of()` returns a **1-based** rank
   (1 = argmax). The draft said 0-indexed and computed V = max + 1 = 152,065.
   Corrected to **V = 152,064**, which matches the model config exactly. The
   chance baseline is (V+1)/(n+1), not (V−1)/(n+1). Committed in `c594a6a`.
6. **Venue.** Not decided, so parameterized. The official style emits the
   submission notice itself and only substitutes the track string at camera-ready,
   so the venue is parameterized at `\workshoptitle{\venuename}` rather than
   printed a second time by hand. Track option is `dblblindworkshop`
   (workshop, double-blind); change it in the preamble if the target changes.
7. **NeurIPS template.** Using the **official `neurips_2026.sty`** (dated
   2026-01-29), copied into `paper/final/` and used verbatim. An earlier version
   of this work used a hand-written stand-in because the file could not be
   downloaded (media.neurips.cc returns 404/403); that stand-in has been deleted.
   The shipped `checklist.tex` is used, with the instruction block removed as the
   template directs and all 16 answers filled.
8. **J-lens downgraded to provisional.** See §4.
9. **Figures.** Okabe-Ito colorblind-safe palette, vector PDF with Type-42
   embedded fonts, pulled from `reports/` at build time. The depth-profile plot
   now labels its inverted log axis explicitly.
10. **Old report not deleted.** A retraction banner was added at the top of
    `reports/Task 4 Online Verification Report.md` pointing at the paper; the body
    is unaltered.

---

## 4. J-lens self-check: what is and is not established

The task prompt (citing the missing `docs/TASK4_CONTEXT.md` §9) states that the
self-check reports **max|Δlogit| = 1.125** against a 0.5 threshold with
**correlation 0.999064**.

**Neither number appears anywhere in this repository** — not in the scripts, the
outputs, the notebook (which has no executed outputs), the README, or the git
history of any branch. They cannot be sourced, so **they are not in the paper.**

What *is* verifiable from the code, and is what the paper says:
- `self_check: false` in **all three** run configs — the self-check was never
  enabled for the data behind this paper. This is the honest disclosure.
- The code threshold is `0.5`; the docstring said `~1e-1`. Real mismatch, fixed.
- `nan > 0.5` is False, so a non-finite comparison printed "OK". Real bug, fixed.
- `README.md` line ~198 shows a *passing* example (`0.0043 / 0.999998`), not a
  failing one. Provenance of that number is also unknown; it is not cited.
- `jlens_diag.py` exists, already computes KL and a dtype ladder, and is
  runnable. It has not been run (no GPU).

Consequently every J-lens result in the paper carries a dagger and the phrase
"provisional pending replay validation". The surviving attention result does not
depend on J-lens.

---

## 4b. Uncommitted observations reported by the author

Recorded here for completeness. **These are unsourced: no committed script,
output file, notebook cell or git object in this repository contains them, and
none could be reproduced without a CUDA device. They are excluded from the
paper**, which states only that a separate diagnostic invocation exceeded the
check's threshold and cites no number.

- bf16 arm: the self-check reported `max|Δlogit|` **above** the 0.5 threshold,
  with correlation ≈0.999 and argmax agreement. (Argmax agreement plus high
  correlation alongside a large max-difference is the signature of a deep-tail
  token dominating the maximum, which is exactly why the repaired check now also
  reports KL.)
- fp16 arm: produced **non-finite logits**, and `out=other` on all records. This
  is the case the old code reported as "OK", because `nan > 0.5` is False. Fixed
  in commit `31d6d07`; a non-finite comparison is now an explicit FAIL.

Neither observation is usable as evidence in its current state. Fix 2.4 (run the
repaired check on ≥20 seeded records and commit
`reports/self_check_bf16.json`) **was skipped: `nvidia-smi` is absent on this
machine.** See "Skipped / partial".

## 4c. Compute facts (Fix 3, item 8)

`run_metadata.py` -> `reports/run_metadata.json`, read from the committed
`outputs/*/config.json`. **Wall-clock IS recorded; GPU model is NOT.**

| run | n | depths | elapsed (s) | s/record |
|---|---|---|---|---|
| 5k, three depths | 5000 | 3 | 2884.3 | 0.577 |
| 137 failures, full sweep | 137 | 29 | 482.3 | 3.520 |
| 500, full sweep | 500 | 29 | 1745.4 | 3.491 |
| **total** | | | **5112.0 (1.42 h)** | |

Section 4 states these; the paper claims **no GPU model**, because none is
logged. The author reports **A100-40GB** and roughly 3.5 s/record at 29 depths,
0.6 s/record at 3 depths — the timings match the configs, but the **hardware is
unsourced** and is therefore not in the paper. Checklist item 8 says exactly
what is recorded.

## 5. Unverified references

**None.** All 15 entries in `refs.bib` were verified on 2026-09-06 against arXiv
abstract pages, publisher pages, or DOI records:

nostalgebraist 2020 (LessWrong URL) · Belrose 2303.08112 · Yom Din 2303.09435
(LREC-COLING 2024) · Geva 2203.14680 (EMNLP 2022) · Orgad 2410.02707 ·
Hewitt & Liang 1909.03368 (EMNLP 2019) · Belinkov 2102.12452 (Computational
Linguistics) · Ravichander 2005.00719 (EACL 2021) · Westfall & Young 1993
(Wiley, ISBN 978-0471557616) · Nichols & Holmes 2002 (HBM 15(1):1–25,
10.1002/hbm.1058) · Kriegeskorte 2009 (Nat Neuro 12(5):535–540, 10.1038/nn.2303)
· Gelman & Loken 2014 (Am Sci 102(6):460) · Meng 2202.05262 (NeurIPS 2022) ·
Wang 2211.00593 · Zhang & Nanda 2309.16042 (ICLR 2024).

Two deliberate conservatisms: Belinkov is cited without volume/pages (venue
verified, pagination not), and Wang et al. is cited as arXiv rather than
ICLR 2023 (arXiv metadata verified, venue not).

---

## 5b. Build desync (Fix 1) — root cause

`main_named.pdf` carried the superseded, UNMATCHED J-lens decomposition values
(0.911 / 0.425 / 0.578) while `main.pdf` carried the corrected matched ones
(0.871 / 0.268 / 0.538), for several commits.

**Cause: a build invocation that did not rerun after the correction.** There has
only ever been one `main.tex`; there was no second source, no stale `.aux`, and
no hand-edited PDF. The named build was produced once by `cp main.pdf
main_named.pdf` after flipping `\anonfalse`, and when Section 6 was later
corrected only the anonymous build was regenerated. Nothing checked the two.

**Fix:** `paper/final/build.sh` is now the only supported build path. It compiles
both PDFs from the single source in one invocation and diffs their extracted
text, failing non-zero on any difference outside a documented whitelist (author
block, notice string, line-number gutter).

One implementation note worth recording: stripping the `lineno` gutter out of
`pdftotext` output *by pattern* is unsafe — a rule like "leading integer plus two
spaces" also removes figure axis tick labels (`400  observed G=11.86`), and a
running-counter version desynchronises on page 1 where the numbers extract
inline. `build.sh` instead compiles a third, throwaway anonymous build with
`\DIFFBUILD` set, which suppresses the gutter at the source, and diffs that. The
shipped `main.pdf` keeps its line numbers for reviewers.

## 6. Open TODOs

1. ~~Fold the J-lens decomposition into `controls_check2.py`~~ **DONE** (section 4b); the paper now uses matched, reproducible values.

2. **Run `patch_probe.py`.** Needs CUDA. Appendix B is a specification, not a
   result; if it runs and the identified arm beats all three controls with a low
   donor-copy rate, the paper gains a causal section and the abstract/title
   should change.
3. **Run `jlens_diag.py`.** Needs CUDA. Resolves the provisional J-lens label.
4. **Re-run attention at depths 19–26 for the 682 matched successes** in the 5k
   run so depth 22 rests on 682 controls rather than 72. Forward pass only, no
   JVP — the cheapest of the three GPU jobs and the highest value.
5. **Confirm author list and affiliation** in the non-anonymized build.
6. **Set the venue** — `\venuename` and the track option in `main.tex`.
7. **Re-check the page count** if the track option changes.

---

## 7. Skipped / partial

| Item | Status | Why |
|---|---|---|
| Fix 2.4 — run repaired self-check on ≥20 seeded records, commit `reports/self_check_bf16.json` | **Skipped** | `nvidia-smi` absent; no CUDA device on this machine. The one-sentence disclosure path was taken instead, citing no number. |
| Appendix C — fill with `jlens_diag.py` results | **Skipped** | Same. Appendix C now states it remains unrun and what running it would settle either way. |
