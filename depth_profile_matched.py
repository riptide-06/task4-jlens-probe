#!/usr/bin/env python3
"""
depth_profile_matched.py -- Figure: rank of the correct and stale tokens at every
residual-stream depth, failures vs successes, on CONDITION-MATCHED groups.

The draft's depth-profile figure was computed on UNMATCHED groups (137 failures
against all 488 successes), which is the confound documented in the paper. This
recomputes it on the matched stratum: 137 failures against the 72 matched
successes. The unmatched gap values are reported too, but only ever labelled
"unmatched", and never in the main text.

Writes, for each lens:
  reports/depth_profile_matched_<lens>.pdf / .png   (Figure)
  reports/depth_profile_matched_<lens>.csv          (per-depth means and medians)
and prints the mean gap over the internal band for each group.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
os.makedirs(REPORTS, exist_ok=True)

# Must stay in step with controls_check2.py.
STRATUM = dict(complexity_level=3, entity_type="agent", distractor_position="middle")
STRATUM_NDIST = [3, 4]
INTERNAL = list(range(2, 24))        # internal band; 0-1 near-embedding, 24-28 output-committed
N_DEPTHS = 29
LENSES = ["logitlens", "jlens"]

CB = {"fc": "#D55E00", "fs": "#E69F00", "sc": "#0072B2", "ss": "#666666"}


def match(d):
    q = d
    for c, v in STRATUM.items():
        q = q[q[c] == v]
    return q[q.n_distractors.isin(STRATUM_NDIST)]


fail = pd.read_csv(os.path.join(HERE, "outputs/colab_run_qwen_failures_fullsweep/layers.csv"))
mixed = pd.read_csv(os.path.join(HERE, "outputs/colab_run_qwen_fullsweep_500/layers.csv"))
succ_raw = mixed[mixed.outcome == "correct"]
succ, failm = match(succ_raw), match(fail)

print("=" * 78)
print("MATCHED DEPTH PROFILE")
print("=" * 78)
print(f"  failures (all in stratum): {failm.example_id.nunique()}")
print(f"  successes: {succ_raw.example_id.nunique()} raw -> {succ.example_id.nunique()} matched")
print(f"  internal band: depths {INTERNAL[0]}..{INTERNAL[-1]}\n")

for lens in LENSES:
    rc, rs = f"{lens}_rank_correct", f"{lens}_rank_stale"
    prof = pd.DataFrame({"depth": range(N_DEPTHS)})
    for nm, d in (("fail", failm), ("succ", succ)):
        for tok, col in (("correct", rc), ("stale", rs)):
            prof[f"{nm}_{tok}_mean"] = [d[d.layer == L][col].mean() for L in range(N_DEPTHS)]
            prof[f"{nm}_{tok}_med"] = [d[d.layer == L][col].median() for L in range(N_DEPTHS)]
    prof.to_csv(os.path.join(REPORTS, f"depth_profile_matched_{lens}.csv"), index=False)

    band = prof[prof.depth.isin(INTERNAL)]
    gap_f = (band.fail_stale_mean - band.fail_correct_mean).mean()
    gap_s = (band.succ_stale_mean - band.succ_correct_mean).mean()
    # Unmatched, for the appendix only -- never the main text.
    gap_f_un = np.mean([fail[fail.layer == L][rs].mean() - fail[fail.layer == L][rc].mean()
                        for L in INTERNAL])
    gap_s_un = np.mean([succ_raw[succ_raw.layer == L][rs].mean() - succ_raw[succ_raw.layer == L][rc].mean()
                        for L in INTERNAL])
    print(f"  [{lens}] mean gap (stale rank - correct rank) over depths {INTERNAL[0]}..{INTERNAL[-1]};")
    print(f"           positive favours the correct token")
    print(f"      MATCHED   failures {gap_f:>12,.0f}   successes {gap_s:>12,.0f}")
    print(f"      unmatched failures {gap_f_un:>12,.0f}   successes {gap_s_un:>12,.0f}   (appendix only)")
    # where do the groups separate?
    sep = [L for L in range(N_DEPTHS)
           if abs(prof.loc[L, "fail_correct_mean"] - prof.loc[L, "succ_correct_mean"])
           > 0.5 * max(prof.loc[L, "fail_correct_mean"], prof.loc[L, "succ_correct_mean"])]
    print(f"      depths where the two correct-token curves differ by >50%: {sep}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for ax, stat, lab in ((axes[0], "mean", "Mean"), (axes[1], "med", "Median")):
        ax.plot(prof.depth, prof[f"fail_correct_{stat}"], "-o", ms=3.2, color=CB["fc"],
                label="failure: correct token")
        ax.plot(prof.depth, prof[f"fail_stale_{stat}"], "--o", ms=3.2, color=CB["fs"],
                label="failure: stale token")
        ax.plot(prof.depth, prof[f"succ_correct_{stat}"], "-s", ms=3.2, color=CB["sc"],
                label="success: correct token")
        ax.plot(prof.depth, prof[f"succ_stale_{stat}"], "--s", ms=3.2, color=CB["ss"],
                label="success: stale token")
        ax.axvspan(23.5, 28.5, color="grey", alpha=.15)
        ax.axvspan(-.5, 1.5, color="grey", alpha=.15)
        ax.set_yscale("log"); ax.invert_yaxis()
        ax.set_xlabel("residual-stream depth")
        ax.set_ylabel(f"{lab} rank, log scale; rank 1 at top")
        ax.set_title(f"{lab} rank per depth", fontsize=10)
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Matched depth profile, {lens}: {failm.example_id.nunique()} failures vs "
                 f"{succ.example_id.nunique()} matched successes "
                 f"(grey = excluded bands)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(REPORTS, f"depth_profile_matched_{lens}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved reports/depth_profile_matched_{lens}.pdf (+ .png, .csv)\n")
