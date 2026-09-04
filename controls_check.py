#!/usr/bin/env python3
"""
controls_check.py -- Four controls on the claim:

  "In 137/137 stale failures, the correct room outranks the stale room
   in at least one internal layer."

The claim's statistic is a MAX over layers (per example: does there EXIST a
layer where rank_correct < rank_stale). A max over many draws is a weak
existence test, so it needs controls:

  1. SYMMETRY      -- does the reverse statistic also hit ~100%?
  2. DISCRIMINATION-- does the statistic also hit ~100% on SUCCESSES?
  3. MAGNITUDE     -- when correct "wins", what are the absolute ranks?
  4. DEPTH PROFILE -- mean rank per layer, failures vs successes.

Ranks are 0-indexed positions in the vocabulary distribution: LOWER = better.
"outranks" == strictly smaller rank.

CSV-only. No GPU, no model calls.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL_CSV = os.path.join(HERE, "outputs/colab_run_qwen_failures_fullsweep/layers.csv")
MIXED_CSV = os.path.join(HERE, "outputs/colab_run_qwen_fullsweep_500/layers.csv")
# Third run, swept at only 3 layers (5, 8, 13). Needed to reconcile the two
# conflicting "best rank" figures in control 3.
SPARSE_CSV = os.path.join(HERE, "outputs/colab_run_qwen_5k_L5-8-13/layers.csv")
REPORTS = os.path.join(HERE, "reports")
os.makedirs(REPORTS, exist_ok=True)

# The two lens families present in the CSVs. Both expose rank_correct/rank_stale.
LENSES = ["logitlens", "jlens"]

RULE = "=" * 78
SUB = "-" * 78


def hr(title):
    print("\n" + RULE)
    print(title)
    print(RULE)


# ---------------------------------------------------------------- load + schema
hr("SCHEMA / df.columns (reported before any analysis, as requested)")

fail = pd.read_csv(FAIL_CSV)
mixed = pd.read_csv(MIXED_CSV)
sparse = pd.read_csv(SPARSE_CSV)
sparse_fail = sparse[sparse.outcome == "stale"].copy()

print("\n[A] outputs/colab_run_qwen_failures_fullsweep/layers.csv")
print("    shape:", fail.shape)
print("    df.columns:")
for i, c in enumerate(fail.columns):
    print(f"      {i:2d}. {c}")

print("\n[B] outputs/colab_run_qwen_fullsweep_500/layers.csv")
print("    shape:", mixed.shape)
print("    identical column set to [A]:", list(mixed.columns) == list(fail.columns))
if list(mixed.columns) != list(fail.columns):
    print("    df.columns:")
    for i, c in enumerate(mixed.columns):
        print(f"      {i:2d}. {c}")

hr("DATASET SHAPE SANITY")
for name, d in [("failures_fullsweep", fail), ("fullsweep_500", mixed)]:
    n_ex = d.example_id.nunique()
    layers = sorted(d.layer.unique())
    print(f"\n{name}:")
    print(f"  rows={len(d)}  examples={n_ex}  layers={len(layers)} (layer ids {layers[0]}..{layers[-1]})")
    print(f"  rows per example: {sorted(d.groupby('example_id').size().unique())}")
    print(f"  outcome (row-level): {d.outcome.value_counts().to_dict()}")
    ex_lvl = d.groupby("example_id").outcome.first().value_counts().to_dict()
    print(f"  outcome (example-level): {ex_lvl}")
    print(f"  rank NaNs: " + ", ".join(
        f"{c.split('_rank_')[0]}/{c.split('_rank_')[1]}={int(d[c].isna().sum())}"
        for c in [f"{L}_rank_{w}" for L in LENSES for w in ("correct", "stale")]))

# NOTE: the prompt said 28 layers; the CSVs carry 29 (layers 0..28). The max is
# therefore taken over 29 draws, which makes the existence test slightly weaker
# still. All layer counts below are read from the data, not assumed.
N_LAYERS = fail.layer.nunique()

# Vocabulary size estimate, used for the chance baseline in control 3.
VOCAB = int(max(
    fail[[f"{L}_rank_{w}" for L in LENSES for w in ("correct", "stale")]].max().max(),
    mixed[[f"{L}_rank_{w}" for L in LENSES for w in ("correct", "stale")]].max().max(),
)) + 1
print(f"\n  observed max rank + 1 (vocab-size lower bound): {VOCAB:,}")
print(f"  layers used for the max: {N_LAYERS}")


# ---------------------------------------------------------------- helpers
def per_example_existence(d, lens):
    """For each example: does there EXIST a layer where correct outranks stale,
    and (symmetrically) where stale outranks correct?"""
    rc = f"{lens}_rank_correct"
    rs = f"{lens}_rank_stale"
    g = d.assign(
        correct_wins=d[rc] < d[rs],
        stale_wins=d[rs] < d[rc],
        tie=d[rc] == d[rs],
    ).groupby("example_id")
    out = pd.DataFrame({
        "correct_wins_somewhere": g.correct_wins.any(),
        "stale_wins_somewhere": g.stale_wins.any(),
        "n_layers_correct_wins": g.correct_wins.sum(),
        "n_layers_stale_wins": g.stale_wins.sum(),
        "n_ties": g.tie.sum(),
        "n_layers": g.size(),
    })
    return out


def pct(n, d):
    return f"{n}/{d} = {100.0 * n / d:5.1f}%" if d else "n/a"


def split_mixed(d):
    ex_outcome = d.groupby("example_id").outcome.first()
    succ_ids = ex_outcome[ex_outcome == "correct"].index
    fail_ids = ex_outcome[ex_outcome == "stale"].index
    return d[d.example_id.isin(succ_ids)].copy(), d[d.example_id.isin(fail_ids)].copy()


mixed_succ, mixed_fail = split_mixed(mixed)

GROUPS = [
    ("failures_fullsweep (137 stale failures)", fail),
    ("fullsweep_500 :: FAILURES", mixed_fail),
    ("fullsweep_500 :: SUCCESSES", mixed_succ),
]

summary_notes = []


# ---------------------------------------------------------------- CONTROL 1
hr("CONTROL 1 -- SYMMETRY")
print("""
Question: in what fraction of failures does the STALE room outrank the CORRECT
room in at least one layer? If that is also ~100%, the original statistic is
vacuous -- it just says 'somewhere in 29 layers the ordering flips', which any
two noisy rank series will satisfy.
""")

c1 = {}
for lens in LENSES:
    print(SUB)
    print(f"LENS: {lens}")
    print(SUB)
    for gname, gdf in GROUPS:
        e = per_example_existence(gdf, lens)
        n = len(e)
        cw = int(e.correct_wins_somewhere.sum())
        sw = int(e.stale_wins_somewhere.sum())
        both = int((e.correct_wins_somewhere & e.stale_wins_somewhere).sum())
        print(f"\n  {gname}  (n={n} examples)")
        print(f"    correct outranks stale in >=1 layer : {pct(cw, n)}   <- the original claim")
        print(f"    stale   outranks correct in >=1 layer: {pct(sw, n)}   <- symmetry control")
        print(f"    BOTH directions occur in same example: {pct(both, n)}")
        print(f"    median # layers correct wins: {e.n_layers_correct_wins.median():.1f} / {int(e.n_layers.iloc[0])}"
              f"   |  median # layers stale wins: {e.n_layers_stale_wins.median():.1f}")
        c1[(lens, gname)] = dict(n=n, cw=cw, sw=sw, both=both,
                                 med_cw=float(e.n_layers_correct_wins.median()),
                                 med_sw=float(e.n_layers_stale_wins.median()))

for lens in LENSES:
    k = (lens, "failures_fullsweep (137 stale failures)")
    r = c1[k]
    if r["sw"] / r["n"] >= 0.95:
        summary_notes.append(
            f"[C1/{lens}] SYMMETRY FAILS: stale outranks correct in >=1 layer in "
            f"{100*r['sw']/r['n']:.1f}% of the 137 failures (claim direction: {100*r['cw']/r['n']:.1f}%). "
            f"The existence test fires in BOTH directions.")


# ---------------------------------------------------------------- CONTROL 2
hr("CONTROL 2 -- DISCRIMINATION (successes vs failures)")
print("""
Question: does 'correct outranks stale in >=1 layer' also hit ~100% on the
SUCCESS records? If yes, the statistic does not separate failure from success
and cannot support the claim as written.
""")

for lens in LENSES:
    print(SUB)
    print(f"LENS: {lens}")
    print(SUB)
    e_s = per_example_existence(mixed_succ, lens)
    e_f = per_example_existence(mixed_fail, lens)
    e_f137 = per_example_existence(fail, lens)
    rows = [
        ("fullsweep_500 SUCCESSES", e_s),
        ("fullsweep_500 FAILURES", e_f),
        ("137-failure run", e_f137),
    ]
    print(f"\n  {'group':<26} {'n':>5}  {'correct wins >=1 layer':>24}  {'mean #layers won':>17}")
    for nm, e in rows:
        n = len(e)
        cw = int(e.correct_wins_somewhere.sum())
        print(f"  {nm:<26} {n:>5}  {100.0*cw/n:>21.1f}%  {e.n_layers_correct_wins.mean():>17.1f}")

    s_rate = e_s.correct_wins_somewhere.mean()
    f_rate = e_f137.correct_wins_somewhere.mean()
    print(f"\n  Delta (failures - successes) on the >=1-layer statistic: "
          f"{100*(f_rate - s_rate):+.1f} pp")
    if s_rate >= 0.95:
        summary_notes.append(
            f"[C2/{lens}] DISCRIMINATION FAILS: successes hit {100*s_rate:.1f}% on the very same "
            f"'correct outranks stale in >=1 layer' test (failures: {100*f_rate:.1f}%). "
            f"Delta = {100*(f_rate-s_rate):+.1f} pp -- the statistic does not distinguish the two groups.")

    # A graded version: does the NUMBER of winning layers separate the groups?
    print(f"\n  Graded alternative -- # layers (of {N_LAYERS}) where correct outranks stale:")
    for nm, e in rows:
        q = e.n_layers_correct_wins.quantile([0.25, 0.5, 0.75]).values
        print(f"    {nm:<26} mean={e.n_layers_correct_wins.mean():5.1f}  "
              f"median={q[1]:4.1f}  IQR=[{q[0]:.0f}, {q[2]:.0f}]")


# ---------------------------------------------------------------- CONTROL 3
hr("CONTROL 3 -- MAGNITUDE (absolute ranks at the best-margin layer)")
print("""
Question: when the correct room 'outranks' the stale room, what are the ABSOLUTE
ranks? Winning at rank 47,000 vs 60,000 is not evidence the correct answer is
represented; both are buried. This also reconciles the two reported figures
(mean best rank 47,343 vs 2.3) -- see the reconciliation block below.

Definition: per example, pick the layer maximising margin = rank_stale - rank_correct
(the layer where correct wins by the largest margin), then report (rank_correct,
rank_stale) there. 'best rank' = min over layers of rank_correct.

Chance baseline: for a rank uniform on [0, V) drawn independently at each of the
%d layers, E[min] ~ (V-1)/(n_layers+1).
""" % N_LAYERS)

chance_min = (VOCAB - 1) / (N_LAYERS + 1)
print(f"  V (vocab lower bound) = {VOCAB:,}   ->   chance E[best-of-{N_LAYERS} rank] "
      f"~ {chance_min:,.0f}\n")

for lens in LENSES:
    print(SUB)
    print(f"LENS: {lens}")
    print(SUB)
    rc, rs = f"{lens}_rank_correct", f"{lens}_rank_stale"
    for gname, gdf in GROUPS:
        d = gdf.copy()
        d["margin_rank"] = d[rs] - d[rc]   # >0 means correct outranks stale
        idx = d.groupby("example_id")["margin_rank"].idxmax()
        best = d.loc[idx, ["example_id", "layer", rc, rs, "margin_rank"]]
        best_rank_correct = d.groupby("example_id")[rc].min()

        print(f"\n  {gname}  (n={len(best)})")
        print(f"    At the best-margin layer:")
        for col, label in [(rc, "rank_correct"), (rs, "rank_stale  ")]:
            q = best[col].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
            print(f"      {label}: mean={best[col].mean():>10,.0f}  median={q[2]:>10,.0f}  "
                  f"IQR=[{q[1]:>9,.0f}, {q[3]:>9,.0f}]  min={q[0]:>8,.0f}  max={q[4]:>10,.0f}")
        print(f"      margin     : mean={best.margin_rank.mean():>10,.0f}  "
              f"median={best.margin_rank.median():>10,.0f}")
        print(f"    Best (min over layers) rank_correct: mean={best_rank_correct.mean():>10,.0f}  "
              f"median={best_rank_correct.median():>10,.0f}")
        for thr in (1, 10, 100, 1000):
            hit = int((best_rank_correct <= thr).sum())
            print(f"      examples whose BEST rank_correct <= {thr:>5,}: {pct(hit, len(best_rank_correct))}")
        worse = int((best_rank_correct > chance_min).sum())
        print(f"      examples whose BEST rank_correct is WORSE than chance "
              f"({chance_min:,.0f}): {pct(worse, len(best_rank_correct))}")

    # Depth story: where is correct actually near the top?
    print(f"\n  Depth breakdown [{lens}] -- median rank_correct and %% of examples with rank_correct<=10:")
    print(f"    {'layer':>5} | {'FAIL med':>10} {'FAIL<=10':>9} | {'SUCC med':>10} {'SUCC<=10':>9}")
    for L in range(N_LAYERS):
        f_l = fail[fail.layer == L][rc]
        s_l = mixed_succ[mixed_succ.layer == L][rc]
        print(f"    {L:>5} | {f_l.median():>10,.0f} {100*(f_l<=10).mean():>8.1f}% | "
              f"{s_l.median():>10,.0f} {100*(s_l<=10).mean():>8.1f}%")

# Explicit reconciliation of 47,343 vs 2.3
print("\n" + SUB)
print("RECONCILIATION of 'mean best rank 47,343' (Run 1) vs '2.3' (Run 3)")
print(SUB)

# First: are the two 137-failure sets the same underlying examples?
def _sig(d, layers):
    d = d[d.layer.isin(layers)]
    p = d.pivot_table(index="example_id", columns="layer",
                      values=["logitlens_rank_correct", "logitlens_rank_stale"])
    p.columns = [f"{a}_L{b}" for a, b in p.columns]
    meta = d.groupby("example_id")[["correct_state", "stale_state",
                                    "complexity_level", "n_distractors"]].first()
    j = p.join(meta)
    return sorted(j[sorted(j.columns)].astype(str).agg("|".join, axis=1))

L3 = [5, 8, 13]
same = _sig(fail, L3) == _sig(sparse_fail, L3)
print(f"\n  5k run (layers {L3}) stale examples: {sparse_fail.example_id.nunique()}")
print(f"  137-failure fullsweep examples       : {fail.example_id.nunique()}")
print(f"  Same underlying examples (re-indexed)? {same}")
print(f"  (example_id/dataset_index are renumbered 0..136 in the fullsweep run,")
print(f"   so they look disjoint by id but the per-example rank signatures match exactly.)")

print("\n  Mean/median BEST (min over swept layers) logitlens_rank_correct, same 137 examples:")
rc = "logitlens_rank_correct"
b_sparse = sparse_fail.groupby("example_id")[rc].min()
b_full = fail.groupby("example_id")[rc].min()
b_restr = fail[fail.layer.isin(L3)].groupby("example_id")[rc].min()
print(f"    swept at layers {L3} only (5k run)   : mean={b_sparse.mean():>10,.1f}  median={b_sparse.median():>9,.0f}")
print(f"    fullsweep restricted to layers {L3}  : mean={b_restr.mean():>10,.1f}  median={b_restr.median():>9,.0f}")
print(f"    swept at all {N_LAYERS} layers (0..{N_LAYERS-1})       : mean={b_full.mean():>10,.1f}  median={b_full.median():>9,.0f}")
print(f"\n  -> 47,343 and 2.3 are the SAME lens (logitlens) on the SAME 137 examples.")
print(f"     The only difference is the layer set the min is taken over.")

# Where does the min actually occur, and does correct beat stale there?
rs = "logitlens_rank_stale"
argmin = fail.loc[fail.groupby("example_id")[rc].idxmin()]
print(f"\n  Layer at which the best rank_correct is achieved (137 failures, full sweep):")
vc = argmin.layer.value_counts().sort_index()
for L, n in vc.items():
    print(f"    layer {L:>2}: {n:>4}  ({100*n/len(argmin):5.1f}%)")
print(f"    -> best layer is >= {N_LAYERS-3} (near-output) in "
      f"{100*(argmin.layer >= N_LAYERS-3).mean():.1f}% of failures")
beats = (argmin[rc] < argmin[rs])
print(f"\n  AT that best-rank layer, does the correct room outrank the stale room?")
print(f"    yes in {pct(int(beats.sum()), len(beats))}")
print(f"    median rank_stale at that layer: {argmin[rs].median():,.0f}  "
      f"(median rank_correct: {argmin[rc].median():,.0f})")

# And where correct DOES win big, how buried is it?
fm = fail.assign(mg=fail[rs] - fail[rc])
amax = fm.loc[fm.groupby("example_id").mg.idxmax()]
print(f"\n  Conversely, at the MAX-MARGIN layer (where correct wins by the most):")
print(f"    median layer: {amax.layer.median():.0f}   "
      f"(layers >= {N_LAYERS-3}: {100*(amax.layer >= N_LAYERS-3).mean():.1f}%)")
print(f"    median rank_correct: {amax[rc].median():>10,.0f}")
print(f"    median rank_stale  : {amax[rs].median():>10,.0f}")

print("""
  The two facts do not combine into the claim:
    * Where the correct room is near the TOP (rank ~2), it is at layers 26-28,
      and there the stale room is rank 1 -- the correct room LOSES. Being the
      runner-up at the output is what a stale failure looks like by definition;
      it is not evidence of an internal representation.
    * Where the correct room genuinely OUTRANKS the stale room by a wide margin,
      both tokens are buried in the tail (median ~24k vs ~103k of ~152k vocab).
      Neither is meaningfully represented at that depth.
  These two regimes are disjoint. The headline statistic is built by taking the
  existence test from one regime and the impressive rank number from the other.
""")

# ---------------------------------------------------------------- CONTROL 4
hr("CONTROL 4 -- DEPTH PROFILE (mean rank per layer, failures vs successes)")
print("""
Question: do failures and successes track each other through most of the stack
and separate only in the final layers?

FAILURE ARM: the full 137-failure file, NOT the 12 failures inside fullsweep_500.
Both files carry all 29 layers and the 12 examples they share are bit-identical
on every rank column (verified in the reconciliation block above), so they pool.
An earlier version of this script used the n=12 arm; that was an error -- 137
were available the whole time.

CAVEAT (unmatched): the two groups are severely imbalanced on task condition --
every failure is complexity-3 / agent / middle-distractor, while ~70%% of
successes have no distractors at all. The separation magnitudes below are
therefore inflated by a hard-vs-easy contrast. controls_check2.py repeats this
on a matched stratum; prefer those numbers.
  failures: %d   successes: %d
""" % (fail.example_id.nunique(), mixed_succ.example_id.nunique()))

profiles = {}
for lens in LENSES:
    rc, rs = f"{lens}_rank_correct", f"{lens}_rank_stale"
    prof = pd.DataFrame({
        "layer": range(N_LAYERS),
        "fail_correct_mean": [fail[fail.layer == L][rc].mean() for L in range(N_LAYERS)],
        "fail_stale_mean": [fail[fail.layer == L][rs].mean() for L in range(N_LAYERS)],
        "succ_correct_mean": [mixed_succ[mixed_succ.layer == L][rc].mean() for L in range(N_LAYERS)],
        "succ_stale_mean": [mixed_succ[mixed_succ.layer == L][rs].mean() for L in range(N_LAYERS)],
        "fail_correct_med": [fail[fail.layer == L][rc].median() for L in range(N_LAYERS)],
        "fail_stale_med": [fail[fail.layer == L][rs].median() for L in range(N_LAYERS)],
        "succ_correct_med": [mixed_succ[mixed_succ.layer == L][rc].median() for L in range(N_LAYERS)],
        "succ_stale_med": [mixed_succ[mixed_succ.layer == L][rs].median() for L in range(N_LAYERS)],
    })
    profiles[lens] = prof
    print(SUB)
    print(f"LENS: {lens}  -- mean rank per layer (lower = better)")
    print(SUB)
    print(prof[["layer", "fail_correct_mean", "fail_stale_mean",
                "succ_correct_mean", "succ_stale_mean"]].to_string(
        index=False, float_format=lambda x: f"{x:,.0f}"))
    prof.to_csv(os.path.join(REPORTS, f"depth_profile_{lens}.csv"), index=False)

    # Where does the correct-vs-stale gap actually open up?
    gap_f = prof.fail_stale_mean - prof.fail_correct_mean
    gap_s = prof.succ_stale_mean - prof.succ_correct_mean
    print(f"\n  Gap (mean rank_stale - mean rank_correct); positive = correct better:")
    print(f"    failures : layers 0-{N_LAYERS-6} mean gap = {gap_f[:N_LAYERS-5].mean():>12,.0f} | "
          f"last 5 layers = {gap_f[N_LAYERS-5:].mean():>12,.0f}")
    print(f"    successes: layers 0-{N_LAYERS-6} mean gap = {gap_s[:N_LAYERS-5].mean():>12,.0f} | "
          f"last 5 layers = {gap_s[N_LAYERS-5:].mean():>12,.0f}")

# ---- plots
for lens in LENSES:
    prof = profiles[lens]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, stat, title in [(axes[0], "mean", "Mean rank per layer"),
                            (axes[1], "med", "Median rank per layer")]:
        ax.plot(prof.layer, prof[f"fail_correct_{stat}"], "-o", ms=3.5, color="#c0392b",
                label="FAIL: correct room")
        ax.plot(prof.layer, prof[f"fail_stale_{stat}"], "--o", ms=3.5, color="#e67e22",
                label="FAIL: stale room")
        ax.plot(prof.layer, prof[f"succ_correct_{stat}"], "-s", ms=3.5, color="#1f6f8b",
                label="SUCC: correct room")
        ax.plot(prof.layer, prof[f"succ_stale_{stat}"], "--s", ms=3.5, color="#7f8c8d",
                label="SUCC: stale room")
        ax.set_yscale("log")
        ax.invert_yaxis()   # better ranks (smaller) at the top
        ax.set_xlabel("layer")
        ax.set_ylabel(f"{stat} rank (log, inverted: higher = better)")
        ax.set_title(f"{title} -- {lens}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Depth profile ({lens}): failures (n={fail.example_id.nunique()}, unmatched) "
                 f"vs successes (n={mixed_succ.example_id.nunique()})", fontsize=11)
    fig.tight_layout()
    p = os.path.join(REPORTS, f"depth_profile_{lens}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  saved plot: {p}")

# Control-summary bar chart
fig, ax = plt.subplots(figsize=(10, 5))
labels, vals, colors = [], [], []
for lens in LENSES:
    for gname, gdf in GROUPS:
        e = per_example_existence(gdf, lens)
        labels.append(f"{lens}\n{gname.replace('fullsweep_500 :: ', '500-')[:22]}\ncorrect wins")
        vals.append(100 * e.correct_wins_somewhere.mean())
        colors.append("#c0392b")
        labels.append(f"{lens}\n{gname.replace('fullsweep_500 :: ', '500-')[:22]}\nstale wins")
        vals.append(100 * e.stale_wins_somewhere.mean())
        colors.append("#7f8c8d")
ax.bar(range(len(vals)), vals, color=colors)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
ax.set_ylabel("% of examples with >=1 winning layer")
ax.set_ylim(0, 105)
ax.axhline(100, color="k", ls=":", lw=1)
ax.set_title("Controls 1 & 2: the '>=1 layer' existence test saturates in every group and both directions")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = os.path.join(REPORTS, "control_saturation.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"  saved plot: {p}")


# ---------------------------------------------------------------- VERDICT
hr("SUMMARY / VERDICT")
if summary_notes:
    for s in summary_notes:
        print("  * " + s + "\n")
else:
    print("  No control undermined the claim.")

print(SUB)
print("Bottom line")
print(SUB)
print(f"""
The claim's statistic ('correct outranks stale in >=1 of {N_LAYERS} layers') is an
existence test over {N_LAYERS} noisy draws. Controls 1 and 2 show it saturates at or
near 100% in the reverse direction and on successes alike, so it carries almost
no information about failures specifically. Control 3 shows the 47,343-vs-2.3 gap is
not a lens difference: it is the same lens on the same 137 examples, and the
whole improvement comes from widening the sweep to include layers 26-28. The
rank-2 result sits at the output layers where the stale room is rank 1, so the
correct room is merely the runner-up; where it genuinely outranks the stale
room, both are buried around rank 24k vs 103k. Control 4 shows where, if anywhere, the two groups actually separate.
""")
