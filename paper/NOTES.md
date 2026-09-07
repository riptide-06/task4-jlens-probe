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
| 100% / 29.3% / 70.7% / 52.5% confound table | §5.3 | "CONDITION CONFOUND (example-level...)" |
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

### From `reports/jlens_depth28_degeneracy.csv` (controls_check2.py §4c)
| Number | Where | Locate |
|---|---|---|
| depth-28 mean \|score\| = 2.2e-06 | §6, App. D | `mean_abs_score`, depth 28 |
| depth-28 mean p_correct = 6.58e-06 (= uniform 1/152,064) | App. D | `mean_p_correct`, depth 28 |
| margin exactly 0 in 28.5%; both ranks tie at 1 in 28.5% | App. D | `margin_exactly_zero`, `both_ranks_1` |
| depths 26–27 scores O(1), no exact-zero margins | App. D | rows 26, 27 |

### From `reports/jlens_convergence.csv` (controls_check2.py §4c)
| Number | Where | Locate |
|---|---|---|
| J-lens median rank of the emitted token: 74,329 at depth 26 | §6 | `jlens_median`, depth 26 |
| ... 61,842 at depth 27 | §6 | `jlens_median`, depth 27 |
| J-lens recovers rank 1 at depth 28 in 71.5% of failures | §6 | `jlens_frac_rank1`, depth 28 |
| logit lens ranks it 1 in 100% at depth 28 | §6 | `logit_frac_rank1`, depth 28 |

### From `reports/depth_profile_matched_*.csv` + stdout (depth_profile_matched.py)
| Number | Where | Locate |
|---|---|---|
| matched internal-band gap: failures −16,569, successes +9,242 | §5.3 | "MATCHED failures / successes" |
| unmatched gap: failures −16,569, successes +5,612 | Appendix B only | "unmatched ... (appendix only)" |
| Figure 3 (matched depth profile, logit lens) | §5.3 | `depth_profile_matched_logitlens.pdf` |
| Figure 5 (J-lens version) | Appendix B | `depth_profile_matched_jlens.pdf` |

### From `reports/paper_extras.json` (paper_extras.py)
| Number | Where | Locate |
|---|---|---|
| top-20 heads: 15 in depths 19–22, 20 in 19–26, 9 at depth 22 | §5.4 | `heads.top20_in_19_22` etc. |
| highest-scoring head at depth 23 | §5.4 | `heads.top_head_depth` |
| 784 (depth, head) pairs ranked | §5.4 | `heads.n_ranked` |
| stale rate 0.00% / 0.00% / 8.22% by complexity | §4 | `stale_rate_by_complexity_level` |
| stale rate 5.47% agent, 0.00% object | §4 | `stale_rate_by_entity_type` |

### From `reports/run_metadata.json` (run_metadata.py)
| Number | Where | Locate |
|---|---|---|
| 5,112 s total (1.42 h); 3.52 / 3.49 / 0.577 s per record | §4, checklist item 8 | `total_elapsed_sec`, `sec_per_record` |

### From `reports/sim_summary.json` (sim/existence_null_sim.py)
| Number | Where | Locate (JSON key) |
|---|---|---|
| P(E=1) at L=29, exchangeable: 0.840–1.000 | §3 | `panel_a_exchangeable_range_at_L29` |
| P(E=1) = 0.9145 when c genuinely worse | §3 | `panel_a_P_E_at_L29["c_genuinely_worse_offset_-0.75"]` |
| E observed 0.998, null 0.999, sd 5.0e-4 | §3 | `panel_b_null_A.E_*` |
| G observed 11.86, null 14.50, sd 0.138 | §3 | `panel_b_null_A.G_*` |
| E ranges 0.888–0.995; G/L 0.318–0.685 | §3 | `panel_c_offsets` |

**Correction made here:** the draft said P(E=1) at L=29 was "0.97–1.00 depending
on noise". The measured range is **0.840–1.000** — the noise=0.5 curve reaches
only 0.8395, which the draft had not accounted for. Text now matches the JSON
exactly.

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

## 4b. J-lens demoted (why, and what replaced the provisional label)

The J-lens result is no longer "provisional pending replay validation". It is
**exploratory**, for a reason that does not depend on the replay at all and that
no GPU run would change.

**What the probe computes** is `jlens_L = f_L(0) + J_L(h_L)·h_L`: the remaining
network's output at zero input, plus the directional derivative of `f_L` at the
residual `h_L` along `h_L`. Verified from the code — `jacobian_lens` passes `h`
as the **primal** to `jvp_autograd` (which does
`h.requires_grad_(True); stack.forward_from(layer_idx, h, ...)`) and `v = h - b`
only as the **tangent**, with the anchor `f_L(b)` computed separately. So the
Jacobian is evaluated at the residual, *not* at the point the anchor is taken
from, and **this is not a first-order expansion about 0**.

That distinction is load-bearing. A true expansion about 0 would be
`f_L(0) + J_L(0)·h_L`; at the final depth `J_28(0)·h = W_U(γ⊙h)/√ε`, which is
large and, since ranks are scale-invariant, reproduces the model's own ordering
exactly. The quantity actually computed does the opposite: it vanishes.

Whatever it is, it is meant to stand in for `f_L(h_L)`, and it is pinned at one
end — as `L → 28` there is less and less remaining computation, and a quantity
tracking the remaining network would converge to the model's output.

**It does not.** `controls_check2.py` section 4c (new) tests this with no GPU. In
a stale-output failure the token the model emits is by construction its own
rank-1 token, so the logit lens — being the model's own norm and unembed — must
rank it 1 at depth 28, and does, in 100% of failures. J-lens puts that same token
at median rank **74,329 at depth 26** and **61,842 at depth 27**, and recovers
rank 1 at depth 28 in only **71.5%** of failures. Source:
`reports/jlens_convergence.csv`.

Consequence for the paper: the depth-5 J-lens margin is read as a fixed,
untrained linear functional `J(0)·h_5` that separates the matched groups
(AUC 0.888, held-out 0.882) — a real and reproducible separation, but not
evidence about what the remaining network would do with `h_5`. The abstract now
reports **one** surviving signal (attention at depth 22). Table 3 keeps the
J-lens row, with a Verdict column reading "exploratory".

**The dtype ladder is no longer open work for this result.** The self-check and
`jlens_diag.py` both validate the *replay* — that `f_L` is computed correctly.
The convergence failure is a property of the *surrogate*, and would survive an
exactly faithful `f_L` in double precision. Appendix D says so explicitly. The
first thing to try instead is a baseline nearer the data (`prev_layer`, which the
probe already supports).

### Uncommitted observations reported by the author

Retained for completeness, still **unsourced and excluded from the paper**: the
bf16 self-check reportedly exceeded its 0.5 threshold with correlation ~0.999 and
argmax agreement; the fp16 arm produced non-finite logits and `out=other` on all
records. Neither appears in any committed script, output, notebook or git object.
The fp16 case is the one the old code reported as "OK" (`nan > 0.5` is False);
fixed in `31d6d07`.

## 4b-bis. Depth 28 is annihilation, not precision — and what analyze_heads.py actually does

**Two claims in the review note were checked and one did not hold.**

*The depth-28 claim.* The note proposed saying the quantity at depth 28 is
"exact up to a positive scale factor", so the 71.5% argmax agreement there
"bounds the numerical precision of the bf16 JVP path". That is true of a genuine
first-order expansion about 0 — but the probe does not compute one (see above). Verified the architecture
first, from the installed `transformers` 5.14.1 source (the version all three
runs used): `Qwen2RMSNorm` has only a `weight` parameter and no bias, and
`lm_head` is `nn.Linear(hidden, vocab, bias=False)`. Both bias-free, as the note
assumed — but the consequence runs the other way:

- `f_28(0) = 0`, because there is no bias anywhere.
- RMSNorm divides out the scale, so `f_28` is positively homogeneous of **degree
  zero**. By Euler's theorem the directional derivative along `h` of a
  degree-zero homogeneous function is zero, so `J_28(h)·h = 0` — and `J(h)`, the
  Jacobian at the residual, is exactly what the probe contracts with `h`.
- Hence `jlens_28 = f_28(0) + J_28(h)·h = 0` **identically**, for any bias-free
  scale-invariant readout. The computed quantity annihilates the signal.

The data match that prediction exactly (`reports/jlens_depth28_degeneracy.csv`):
mean |logit| 2.2e-06 at depth 28 against O(1) at 26–27; mean `p_correct`
6.58e-06, the uniform value `1/152,064` to three significant figures; margin
**exactly** zero in 28.5% of failures and both tokens tied at rank 1 in the same
28.5%. So **the 71.5% is a tie artifact of the rank convention** (`rank_of`
counts strictly-greater scores, so an all-zero vector ranks everything 1) and
bounds nothing about bf16 precision.

The note's instinct that there are *two separable failures* is right, with
different content: **annihilation at depth 28** (structural, would survive any
precision) and **linearization error at depths 26–27** (scores O(1), no exact
zeros, genuine but wrong distribution). The paper says so.

*The head-ranking claim.* Read `analyze_heads.py`. It ranks all 784 (depth, head)
pairs by a difference in differences: Δ = stale-anchor mass − update-mention
mass, scored as `delta_fail - delta_succ`. The success set is chosen by
`--baseline`, whose **default is `matched`** — successes with a single stale and
a single correct mention, which is the regime all 137 failures occupy by
construction. The committed `head_ranking.csv` records `n_succ = 72`, confirming
it was produced with that default. **So the ranking does not share the
Section 5.3 confound**, and no recomputation was needed.

It is still not *independent corroboration* in the strongest sense: it uses the
same 72 controls as the depth-22 AUC. Section 5.4 now says precisely what it is —
corroboration by a different statistic computed from different data on disk
(per-head tensors rather than per-example aggregates), not from an independent
control group — and notes that the 72-record ceiling applies to both. The
`\paragraph` heading dropped the word "Independent".

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

## 4d. What Orgad et al. (2024) actually do (Fix 6)

Checked against the full text of arXiv:2410.02707v2 (fetched and extracted, not
from the abstract). Their method is **probing classifiers**, not a rank test:

- "a classifier trained on intermediate LLM representations within the exact
  answer tokens, referred to as 'probing classifiers'" (§1).
- The encoding-vs-behaviour result uses sampling plus probe-based selection:
  they "sample K = 30 responses at a temperature setting of T = 1" and then
  "use our probe, trained on error detection, to select an answer from a pool of
  30 generated responses" (§5-6).
- **The string "logit lens" occurs zero times in the paper.** There is no
  max-over-layers rank comparison anywhere in it.

The draft's Section 1 sentence read as though they relied on the
existence-over-depth statistic. Rewritten to cite them as an example of the
**conclusion type only**, explicitly naming their different route, and stating
that the existence statistic is one common route to such conclusions and the one
this paper examines. No reference added or changed in refs.bib.

## 4e. Number audit (Fix 10)

Method: extract every substantive numeric token from `main.pdf` (main text and
appendices; the checklist restates the same figures), then require each to
appear verbatim in a committed output under `reports/` or `outputs/*/config.json`.

**Result: 95 tokens checked, 0 unmapped.** Excluded by category:
- 6 bibliography identifiers (arXiv IDs, DOIs, an ISBN) — not statistics.
- Publication years.
- One derived value: **91.5%** in §3 is `sim_summary.json`'s
  `panel_a_P_E_at_L29["c_genuinely_worse_offset_-0.75"]` = `0.9145`, expressed as
  a percentage. The JSON holds the proportion; the paper states the percentage.

**The audit caught a real gap.** The Section 5.3 confound percentages
(100% / 29.3% / 70.7% / 52.5%) were *not* produced by any committed script — they
came from ad-hoc exploration during the first pass and had been carried into the
paper unsourced. `controls_check2.py` now computes and prints the full
example-level confound table, and all four figures reproduce exactly. This was
the last unmapped number in the paper.

The audit is repeatable; the extraction is a dozen lines and is described above.

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

## 5c. Reviewer pass (updated, Fix 10)

Each objection with the sentence that answers it. Verified present in the current
`main.pdf` by text search.

| # | Likely objection | Where it is answered |
|---|---|---|
| **1** | **"The depth-22 result rests on only 72 matched controls."** | §6 Statistical power: "The matched full-depth comparison rests on 72 successes." Mitigated but not removed — the 682-control re-sweep is the top open TODO, and §5.4's per-head corroboration is independent of the 72. |
| 2 | "This is all correlational." | Abstract ("Every surviving result is correlational"); §6: "A measure that predicts an outcome need not participate in producing it." |
| 3 | "Is J-lens even valid?" | §6 answers it directly and negatively: the lens fails a convergence check its own definition requires (median rank 74,329 / 61,842 at depths 26–27 for the model's own argmax token). The result is demoted to exploratory and the surviving attention result does not depend on it. |
| 4 | "Single model, single synthetic dataset." | §6 Scope. |
| 5 | "Who cares about one internal result of yours?" | §1 opens on the general pattern; §3 is model-free; the case study is confined to §4–5 and Appendices A–C. |
| 6 | "Isn't depth 22 itself selected post hoc?" | §3 Null B (max-$T$) and split-half; Table 3 reports both, and logit-lens margin fails the second. |
| 7 | "Doesn't the per-head result just restate the depth-22 AUC?" | §5.4: per-head, built from raw attention tensors, "a check on the depth-22 result rather than a restatement of it." |

Objection 1 is the strongest remaining and is now first in the list.

## 6. Open TODOs

1. ~~Fold the J-lens decomposition into `controls_check2.py`~~ **DONE** (section 4b); the paper now uses matched, reproducible values.

2. **Run `patch_probe.py`.** Needs CUDA. Appendix B is a specification, not a
   result; if it runs and the identified arm beats all three controls with a low
   donor-copy rate, the paper gains a causal section and the abstract/title
   should change.
3. ~~Run `jlens_diag.py` to resolve the J-lens label~~ **DROPPED as open work.**
   The ladder tests the replay, not the expansion; the convergence failure that
   demotes J-lens is a property of the surrogate and would survive a perfect
   replay. If J-lens is to be rehabilitated the experiment is a per-depth
   approximation-error measurement (KL between `f_L(h_L)` and `jlens_L`).
   Note `--jlens-baseline prev_layer` is **not** a fix on its own: it changes the
   anchor and tangent but not the primal, giving
   `f_L(h_{L-1}) + J_L(h_L)·(h_L − h_{L-1})`, still not a Taylor expansion. No run
   used it. Making the object an expansion requires evaluating the Jacobian at the
   baseline — a one-line change to the primal passed to `jvp_autograd`.
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
| Appendix D — fill with `jlens_diag.py` results | **Skipped** | Same. It now states it remains unrun and what running it would settle either way. |
| Determinism sentence (Fix 5d) | **Omitted** | No second-pass output is committed anywhere in `outputs/`, so there is nothing to cite. Recorded rather than asserted. |
