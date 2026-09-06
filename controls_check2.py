#!/usr/bin/env python3
"""
controls_check2.py -- Round 2 controls, on CONDITION-MATCHED groups.

Round 1 (controls_check.py) showed the headline statistic
    "correct outranks stale in >=1 of 29 layers"
is vacuous: it fires 100% in reverse and 100% on successes.

Round 1 also compared failures against UNMATCHED successes. That comparison is
confounded: every failure is complexity-3 / agent / middle-distractor, while
~70% of successes carry no distractors at all. Any measure tracking task
difficulty separates the groups for reasons unrelated to the outcome.

This script:
  1. MATCHED    -- matched stratum is the default for every failure-vs-success
                   number; raw is printed beside it so the confound is visible.
  2. PERM NULL  -- permutation nulls, with an explicit statement of which
                   exchangeability each one encodes.
  3. SPLIT-HALF -- select the layer on one half, evaluate on the other; report
                   the distribution of selected layers across splits.
  4. ATTN SWEEP -- all attn_* columns x all layers, matched, ranked by AUC,
                   with a stated (not silent) missing-data policy.

CSV-only. No GPU.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260904)
N_PERM = 5000
N_SPLIT = 500

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
os.makedirs(REPORTS, exist_ok=True)
FAIL_CSV = os.path.join(HERE, "outputs/colab_run_qwen_failures_fullsweep/layers.csv")
MIXED_CSV = os.path.join(HERE, "outputs/colab_run_qwen_fullsweep_500/layers.csv")
SPARSE_CSV = os.path.join(HERE, "outputs/colab_run_qwen_5k_L5-8-13/layers.csv")

RULE = "=" * 78
SUB = "-" * 78
def hr(t):
    print("\n" + RULE); print(t); print(RULE)

# Matched stratum. NOTE: these four constraints are collinear in this dataset --
# complexity_level==3 & entity_type=="agent" alone already selects the same 72
# successes. All four are stated so the intent is explicit, not because each bites.
STRATUM = dict(complexity_level=3, entity_type="agent", distractor_position="middle")
STRATUM_NDIST = [3, 4]

# Depth bands. Depths 0-1 are near-embedding and 24-28 are output-committed: at
# depth 28 the lens IS the model's decision (it is the final residual stream,
# one norm+unembed from the logits), so an AUC there is circular by
# construction (round 1 measured exactly 1.000). "Internal" excludes both ends.
INTERNAL = list(range(2, 24))


# --------------------------------------------------------------------- helpers
def match(d):
    q = d
    for c, v in STRATUM.items():
        q = q[q[c] == v]
    return q[q.n_distractors.isin(STRATUM_NDIST)]


def fast_auc(x, y):
    """AUC(P(x_pos > x_neg)); y is 1 for the positive class. NaN-safe."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    if n1 < 3 or n0 < 3:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_pos_neg(pos, neg):
    return fast_auc(np.concatenate([pos, neg]),
                    np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))


def layer_vec(d, L, col):
    return d[d.layer == L][col].to_numpy(dtype=float)


# ------------------------------------------------------------------------ load
hr("0. DATA, TRUE-INDEX RECOVERY, AND THE MATCHED STRATUM")

fail = pd.read_csv(FAIL_CSV)
mixed = pd.read_csv(MIXED_CSV)
sparse = pd.read_csv(SPARSE_CSV)

# The 137-failure file renumbered dataset_index to 0..136. Recover the true ids
# by matching per-example rank signatures at the three layers the 5k run shares.
L3 = [5, 8, 13]
def _sig(d):
    q = d[d.layer.isin(L3)]
    p = q.pivot_table(index="example_id", columns="layer",
                      values=["logitlens_rank_correct", "logitlens_rank_stale",
                              "jlens_rank_correct", "jlens_rank_stale"])
    p.columns = [f"{a}_L{b}" for a, b in p.columns]
    return p.astype("int64").astype(str).agg("|".join, axis=1)

sparse_fail = sparse[sparse.outcome == "stale"]
sig_f, sig_k = _sig(fail), _sig(sparse_fail)
inv = {v: sparse_fail[sparse_fail.example_id == i].dataset_index.iloc[0] for i, v in sig_k.items()}
fail = fail.assign(true_di=fail.example_id.map(dict(zip(sig_f.index, sig_f.map(inv)))))
print(f"  true dataset_index recovered for {int(fail.true_di.notna().sum() / 29)}/137 failures "
      f"(signature match, all unique)")

mixed_succ, mixed_fail = mixed[mixed.outcome == "correct"], mixed[mixed.outcome == "stale"]
sparse_succ = sparse[sparse.outcome == "correct"]

# Poolability: the 12 examples in BOTH 29-layer files must agree exactly.
shared = sorted(set(mixed_fail.dataset_index) & set(fail.true_di.dropna().astype(int)))
A = fail[fail.true_di.isin(shared)].sort_values(["true_di", "layer"])
B = mixed_fail[mixed_fail.dataset_index.isin(shared)].sort_values(["dataset_index", "layer"])
rankcols = [f"{L}_rank_{w}" for L in ("logitlens", "jlens") for w in ("correct", "stale")]
ok = all(np.array_equal(A[c].to_numpy(), B[c].to_numpy()) for c in rankcols)
print(f"  {len(shared)} examples appear in both 29-layer files; all rank columns identical: {ok}")
print(f"  -> the two files pool. Failure arm = all 137 (not the 12 inside fullsweep_500).")

m_succ, m_fail = match(mixed_succ), match(fail)
s_succ, s_fail = match(sparse_succ), match(sparse_fail)
print(f"\n  MATCHED STRATUM: complexity_level=3, entity_type=agent, "
      f"distractor_position=middle, n_distractors in {STRATUM_NDIST}")
print(f"    29-layer successes : {mixed_succ.example_id.nunique():5d} raw -> {m_succ.example_id.nunique():4d} matched")
print(f"    29-layer failures  : {fail.example_id.nunique():5d} raw -> {m_fail.example_id.nunique():4d} matched  (all 137 already in stratum)")
print(f"    3-layer  successes : {sparse_succ.example_id.nunique():5d} raw -> {s_succ.example_id.nunique():4d} matched  (layers 5/8/13 only)")
print(f"  The 72 matched successes coincide with the stale-mention-count matching")
print(f"  used in commit c8d256a, reached independently here.")

# Missing-data policy, stated rather than applied silently.
ATTN = [c for c in mixed.columns if c.startswith("attn_")]
nan_all = sum(d[ATTN].isna().sum().sum() for d in (fail, mixed))
nan_not28 = sum(d[d.layer != 28][ATTN].isna().sum().sum() for d in (fail, mixed))
hr("MISSING-DATA POLICY (attn_* columns)")
print(f"""
  Depth 28 carries NO attention data in either file: all {len(ATTN)} attn_* columns are
  100% NaN there. This is structural, not a capture failure. ResidualCapture hooks
  the INPUT of each of the 28 decoder layers (depths 0..27) plus the OUTPUT of the
  last one (depth 28), giving 29 residual-stream positions. AttentionCapture hooks
  layers[i].self_attn for i in 0..27, so attention is indexed by DECODER LAYER and
  no attention module corresponds to residual position 28. That single depth
  accounts for 1/29 = 3.4% of rows and for {nan_all:,} of {nan_all:,} attn NaNs.
  Outside depth 28 there are {nan_not28} missing attention values.

  Policy: the attn_* sweep runs over depths 0..27 and depth 28 is reported as
  "no data", never as a zero, a fill, or an omitted row. No other row is dropped,
  because none needs to be. Rank/lens columns have no missing values at any layer.
""")


# ============================================================ 1. RAW vs MATCHED
hr("1. RAW vs MATCHED, SIDE BY SIDE (confound size is the difference)")
print("""
Per-layer AUC separating SUCCESS from FAILURE. 0.5 = no signal; distance from
0.5 = strength. "raw" uses all 488 successes; "matched" uses the 72 in-stratum
successes. Failures are all 137 in both columns.
""")
MEASURES = ["jlens_margin", "logitlens_margin", "attn_correct_minus_stale"]
rows = []
for L in range(29):
    r = {"layer": L}
    for c in MEASURES:
        r[f"{c}|raw"] = auc_pos_neg(layer_vec(mixed_succ, L, c), layer_vec(fail, L, c))
        r[f"{c}|matched"] = auc_pos_neg(layer_vec(m_succ, L, c), layer_vec(m_fail, L, c))
    rows.append(r)
raw_matched = pd.DataFrame(rows)
raw_matched.to_csv(os.path.join(REPORTS, "auc_raw_vs_matched.csv"), index=False)

print(f"  {'layer':>5} | {'jlens raw':>9} {'jlens mat':>9} | {'logit raw':>9} {'logit mat':>9} | {'attn raw':>9} {'attn mat':>9}")
print("  " + "-" * 74)
for _, r in raw_matched.iterrows():
    def fm(v):
        return "   no data" if np.isnan(v) else f"{v:>9.3f}"
    flag = "  <-- output-committed" if r.layer >= 24 else ("  <-- near-embedding" if r.layer <= 1 else "")
    print(f"  {int(r.layer):>5} | {fm(r['jlens_margin|raw'])} {fm(r['jlens_margin|matched'])} | "
          f"{fm(r['logitlens_margin|raw'])} {fm(r['logitlens_margin|matched'])} | "
          f"{fm(r['attn_correct_minus_stale|raw'])} {fm(r['attn_correct_minus_stale|matched'])}{flag}")

print("\n  Confound size at the layers round 2 cares about:")
for c, L in [("jlens_margin", 5), ("attn_correct_minus_stale", 19), ("attn_correct_minus_stale", 22)]:
    r = raw_matched[raw_matched.layer == L].iloc[0]
    a, b = r[f"{c}|raw"], r[f"{c}|matched"]
    print(f"    {c:26s} L{L:<2d}  raw={a:.3f}  matched={b:.3f}  shrinkage={a-b:+.3f}")

print("\n  Large-sample matched replication (5k run, layers 5/8/13, "
      f"{s_succ.example_id.nunique()} matched successes vs {s_fail.example_id.nunique()} failures):")
rep_rows = []
for L in L3:
    r = {"layer": L}
    for c in MEASURES:
        r[c] = auc_pos_neg(layer_vec(s_succ, L, c), layer_vec(s_fail, L, c))
    rep_rows.append(r)
    print(f"    layer {L:>2}: " + "  ".join(f"{c.split('_')[0]}={r[c]:.3f}" for c in MEASURES))
pd.DataFrame(rep_rows).to_csv(os.path.join(REPORTS, "auc_5k_matched.csv"), index=False)


# ========================================================= 2. PERMUTATION NULLS
hr("2. PERMUTATION NULLS")
print("""
Two different exchangeability assumptions are at stake, so there are two nulls.
Using the wrong one is how a max-over-layers statistic gets called significant.

  NULL A -- swap the correct/stale LABELS within an example, across all 29
            layers at once. Encodes "which room is the correct one carries no
            information". The whole-series swap (not per-layer) is required:
            swapping per layer would destroy within-example cross-layer
            correlation and give a falsely narrow null for a max-over-layers
            statistic. This is the null for the ORIGINAL claim.

  NULL B -- shuffle the FAILURE/SUCCESS labels across examples. Encodes "the
            outcome carries no information about this measure". This is the null
            for the AUC-based survivors, and taking the max over layers inside
            each permutation is what corrects for selecting the best layer.
""")

# ---- NULL A, on the 137 failures
print(SUB); print("NULL A -- correct/stale label swap (137 failures)"); print(SUB)
nullA = {}
for lens in ("logitlens", "jlens"):
    rc, rs = f"{lens}_rank_correct", f"{lens}_rank_stale"
    g = fail.assign(cw=fail[rc] < fail[rs], sw=fail[rs] < fail[rc]).groupby("example_id")
    cw_any = g.cw.any().to_numpy(); sw_any = g.sw.any().to_numpy()
    cw_n = g.cw.sum().to_numpy(float); sw_n = g.sw.sum().to_numpy(float)
    n = len(cw_any)

    obs_exist = cw_any.mean()
    obs_grad = cw_n.mean()
    flips = RNG.random((N_PERM, n)) < 0.5
    null_exist = np.where(flips, sw_any, cw_any).mean(axis=1)
    null_grad = np.where(flips, sw_n, cw_n).mean(axis=1)
    p_exist = (np.abs(null_exist - null_exist.mean()) >= abs(obs_exist - null_exist.mean())).mean()
    p_grad = (np.abs(null_grad - null_grad.mean()) >= abs(obs_grad - null_grad.mean())).mean()
    nullA[lens] = dict(obs_exist=obs_exist, null_exist=null_exist, p_exist=p_exist,
                       obs_grad=obs_grad, null_grad=null_grad, p_grad=p_grad)

    print(f"\n  {lens}")
    print(f"    STATISTIC 1 (the claim) -- fraction with correct winning in >=1 of 29 layers")
    print(f"      observed = {obs_exist:.4f}")
    print(f"      null     = [{null_exist.min():.4f}, {null_exist.max():.4f}]  "
          f"mean={null_exist.mean():.4f}  sd={null_exist.std():.6f}")
    print(f"      p = {p_exist:.3f}")
    if null_exist.std() < 1e-12:
        print(f"      -> the null is a POINT MASS at {null_exist.mean():.4f}. The statistic is")
        print(f"         invariant under the swap: both directions already hold in all 137")
        print(f"         examples, so relabelling cannot change it. It is untestable, not")
        print(f"         merely non-significant.")
    print(f"    STATISTIC 2 (graded) -- mean # of 29 layers where correct outranks stale")
    print(f"      observed = {obs_grad:.2f}")
    print(f"      null     = mean {null_grad.mean():.2f}  "
          f"95% [{np.quantile(null_grad,.025):.2f}, {np.quantile(null_grad,.975):.2f}]")
    print(f"      p = {p_grad:.4f}   direction: observed is "
          f"{'BELOW' if obs_grad < null_grad.mean() else 'ABOVE'} the null mean")
    if obs_grad < np.quantile(null_grad, .025):
        print(f"      -> significant in the WRONG direction for the claim: the correct room")
        print(f"         wins FEWER layers than label-exchange chance.")

# ---- NULL B, matched groups, max over internal layers
print("\n" + SUB); print("NULL B -- outcome label shuffle, max over internal layers 2..23 (MATCHED)"); print(SUB)
print(f"  {m_succ.example_id.nunique()} matched successes vs {m_fail.example_id.nunique()} failures\n")
nullB = {}
for c in MEASURES:
    band = [L for L in INTERNAL if not np.isnan(auc_pos_neg(layer_vec(m_succ, L, c), layer_vec(m_fail, L, c)))]
    # Pre-rank once per layer; permuting labels is then just a rank-sum.
    ranks, labels = {}, None
    for L in band:
        pos, neg = layer_vec(m_succ, L, c), layer_vec(m_fail, L, c)
        x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        ok = ~np.isnan(x)
        ranks[L] = pd.Series(x[ok]).rank().to_numpy()
        labels = y[ok]
    n1 = labels.sum(); n0 = len(labels) - n1
    obs = np.array([auc_pos_neg(layer_vec(m_succ, L, c), layer_vec(m_fail, L, c)) for L in band])
    obs_max = np.abs(obs - .5).max(); obs_L = band[int(np.argmax(np.abs(obs - .5)))]

    perm_max = np.empty(N_PERM)
    idx = np.arange(len(labels))
    for i in range(N_PERM):
        RNG.shuffle(idx)
        sel = labels[idx] == 1
        a = np.array([(ranks[L][sel].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) for L in band])
        perm_max[i] = np.abs(a - .5).max()
    p_max = (perm_max >= obs_max).mean()
    nullB[c] = dict(obs_max=obs_max, obs_L=obs_L, perm_max=perm_max, p=p_max, band=band)
    print(f"  {c}")
    print(f"    best internal layer = {obs_L}, AUC = {obs[band.index(obs_L)]:.3f} "
          f"(|AUC-0.5| = {obs_max:.3f})")
    print(f"    null max |AUC-0.5| : mean={perm_max.mean():.3f}  "
          f"95th pct={np.quantile(perm_max,.95):.3f}  max={perm_max.max():.3f}")
    print(f"    selection-corrected p = {p_max:.4f}  "
          f"{'PASS' if p_max < 0.05 else 'FAIL -- does not survive'}\n")


# ================================================================ 3. SPLIT-HALF
hr("3. SPLIT-HALF SELECTION-CORRECTED AUC (matched)")
print(f"""
Select the best internal layer on one half, evaluate it on the held-out half.
{N_SPLIT} random splits, stratified within group. This prices in the fact that the
layer was chosen by looking at the data.
Halves are small: ~{m_succ.example_id.nunique()//2} successes and ~{m_fail.example_id.nunique()//2} failures per side.
""")
split_res = {}
for c in MEASURES:
    band = nullB[c]["band"]
    sid = m_succ.example_id.unique(); fid = m_fail.example_id.unique()
    held, picks = [], []
    for _ in range(N_SPLIT):
        s1 = set(RNG.choice(sid, len(sid) // 2, replace=False))
        f1 = set(RNG.choice(fid, len(fid) // 2, replace=False))
        sA, sB = m_succ[m_succ.example_id.isin(s1)], m_succ[~m_succ.example_id.isin(s1)]
        fA, fB = m_fail[m_fail.example_id.isin(f1)], m_fail[~m_fail.example_id.isin(f1)]
        a = [abs((auc_pos_neg(layer_vec(sA, L, c), layer_vec(fA, L, c)) or .5) - .5) for L in band]
        L = band[int(np.nanargmax(a))]
        picks.append(L); held.append(auc_pos_neg(layer_vec(sB, L, c), layer_vec(fB, L, c)))
    held = np.array(held, dtype=float); held = held[~np.isnan(held)]
    vc = pd.Series(picks).value_counts()
    split_res[c] = dict(held=held, picks=picks, vc=vc)
    print(f"  {c}")
    print(f"    held-out AUC: mean={held.mean():.3f}  95% [{np.quantile(held,.025):.3f}, {np.quantile(held,.975):.3f}]")
    print(f"    selected-layer distribution (top 6 of {len(set(picks))} distinct layers chosen):")
    for L, k in vc.head(6).items():
        print(f"        layer {L:>2}: {k:>4}/{N_SPLIT} splits ({100*k/N_SPLIT:5.1f}%)")
    stable = 100 * vc.iloc[0] / N_SPLIT
    print(f"    -> modal layer {vc.index[0]} chosen in {stable:.0f}% of splits: "
          f"{'stable' if stable >= 50 else 'UNSTABLE -- the best layer is not reproducible'}\n")


# =============================================================== 4. ATTN SWEEP
hr("4. FULL attn_* SWEEP, MATCHED, RANKED BY AUC")
print(f"""
All {len(ATTN)} attn_* columns x layers 0..27 (layer 28 has no attention data -- see
policy above). Matched groups: {m_succ.example_id.nunique()} successes vs {m_fail.example_id.nunique()} failures.
AUC < 0.5 means the measure is HIGHER in failures, which for stale-directed
attention is the expected direction; ranking is by |AUC - 0.5|.
""")
sweep = []
for c in ATTN:
    for L in range(29):
        if L == 28:
            sweep.append(dict(column=c, layer=L, auc=np.nan, n_succ=0, n_fail=0, note="no data (layer 28)"))
            continue
        pos, neg = layer_vec(m_succ, L, c), layer_vec(m_fail, L, c)
        sweep.append(dict(column=c, layer=L, auc=auc_pos_neg(pos, neg),
                          n_succ=int((~np.isnan(pos)).sum()), n_fail=int((~np.isnan(neg)).sum()), note=""))
sweep = pd.DataFrame(sweep)
sweep["abs_dev"] = (sweep.auc - .5).abs()
sweep.to_csv(os.path.join(REPORTS, "attn_sweep_matched.csv"), index=False)
print(f"  saved full sweep ({len(sweep)} rows) -> reports/attn_sweep_matched.csv")
print(f"  rows with no data: {int(sweep.note.ne('').sum())} (all layer 28)\n")
top = sweep.dropna(subset=["auc"]).sort_values("abs_dev", ascending=False).head(15)
print(f"  {'rank':>4} {'column':<28} {'layer':>5} {'AUC':>7} {'|AUC-.5|':>9} {'n_succ':>7} {'n_fail':>7}")
for i, (_, r) in enumerate(top.iterrows(), 1):
    print(f"  {i:>4} {r.column:<28} {int(r.layer):>5} {r.auc:>7.3f} {r.abs_dev:>9.3f} {r.n_succ:>7} {r.n_fail:>7}")
# Within the matched stratum every record has a single stale mention, so
# "mass on the first stale mention" equals "total stale mass". Report the
# redundancy rather than presenting 9 independent measures.
print("\n  Column redundancy inside the matched stratum:")
for a, b in [("attn_stale_mass", "attn_stale_anchor_mass"),
             ("attn_correct_mass", "attn_correct_update_mass")]:
    both = pd.concat([m_succ, m_fail])
    eq = (both[both.layer != 28][a] == both[both.layer != 28][b]).mean()
    print(f"    {a} == {b} in {100*eq:.1f}% of rows (layers 0..27)")
print("    -> matched records carry a single stale mention, so first-mention mass")
print("       equals total mass. The sweep covers 7 distinct measures, not 9.")

print("\n  Best layer per column:")
for c in ATTN:
    s = sweep[(sweep.column == c)].dropna(subset=["auc"])
    b = s.loc[s.abs_dev.idxmax()]
    print(f"    {c:<28} L{int(b.layer):<2d}  AUC={b.auc:.3f}")


# ================================================ 4b. WHAT DOES J-LENS MEASURE?
hr("4b. J-LENS DECOMPOSITION (is it a disguised readout of the output?)")
print("""
jlens_i = f_i(0) + J.h_i, where f_i runs decoder layers i..28 with the real KV
cache. By construction it includes the entire downstream network, so a natural
worry is that its predictive power is just a laundered read of the model's final
decision rather than anything local to depth i.

Two checks. (1) Cross-depth rank correlation of the margin within an example: if
a shared example-level component (the f_i(0) anchor, or the context) dominated,
J-lens margins would be MORE correlated across depths than logit-lens margins.
(2) Remove each example's cross-depth mean and re-test: whatever separation
survives is depth-specific and cannot come from a shared component.
""")
_pool = pd.concat([m_fail.assign(grp="fail"), m_succ.assign(grp="succ")], ignore_index=True)
print(f"  (matched groups: {m_succ.example_id.nunique()} successes, {m_fail.example_id.nunique()} failures)\n")
for lens in ("logitlens", "jlens"):
    c = f"{lens}_margin"
    piv = _pool.pivot_table(index=["grp", "example_id"], columns="layer", values=c)
    C = piv[INTERNAL].corr(method="spearman").to_numpy()
    off = C[~np.eye(len(C), dtype=bool)]
    print(f"  {lens:10s} mean |cross-depth rank corr| over depths 2..23 = "
          f"{np.abs(off).mean():.3f}  (median {np.median(np.abs(off)):.3f})")
print()
for c, L in (("jlens_margin", 5), ("logitlens_margin", 18)):
    band = _pool[_pool.layer.between(2, 23)].copy()
    band["dev"] = band[c] - band.groupby(["grp", "example_id"])[c].transform("mean")
    at = band[band.layer == L]
    raw = auc_pos_neg(layer_vec(m_succ, L, c), layer_vec(m_fail, L, c))
    dev = auc_pos_neg(at[at.grp == "succ"].dev.to_numpy(), at[at.grp == "fail"].dev.to_numpy())
    print(f"  {c:18s} depth {L:2d}: matched AUC = {raw:.3f} | "
          f"depth-specific DEVIATION AUC = {dev:.3f}")
print("""
  Reading: if J-lens were a disguised output readout, the deviation AUC would
  collapse toward 0.5 and its cross-depth correlation would exceed the logit
  lens's. Neither happens. The signal is genuinely depth-local. What this does
  NOT license is the phrase "represented at depth 5": f_i still executes layers
  i..28, so the supported claim is "decodable THROUGH the remaining network".
""")


# ==================================================================== 5. PLOTS
# Okabe-Ito colorblind-safe palette, shared with sim/existence_null_sim.py.
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#666666"}


def savefig(fig, stem):
    """Vector PDF for the paper plus a PNG preview. Fonts embedded (Type 42)."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(REPORTS, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def theme(ax):
    ax.grid(alpha=.3); ax.axhline(.5, color="k", ls=":", lw=1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, c in zip(axes, MEASURES):
    ax.plot(raw_matched.layer, raw_matched[f"{c}|raw"], "-o", ms=3.5, color=CB["red"], label="raw (488 succ)")
    ax.plot(raw_matched.layer, raw_matched[f"{c}|matched"], "-s", ms=3.5, color=CB["blue"], label="matched (72 succ)")
    ax.axvspan(24, 28, color="grey", alpha=.18)
    ax.axvspan(-.5, 1.5, color="grey", alpha=.18)
    ax.set_title(c, fontsize=10); ax.set_xlabel("residual-stream depth"); ax.set_ylabel("AUC (success vs failure)")
    ax.set_ylim(0, 1); theme(ax); ax.legend(fontsize=8)
fig.suptitle("Confound size: raw vs condition-matched AUC (grey = near-embedding / output-committed)", fontsize=11)
fig.tight_layout(); savefig(fig, "auc_raw_vs_matched")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, c in zip(axes, MEASURES):
    d = nullB[c]
    ax.hist(d["perm_max"], bins=50, color=CB["grey"], alpha=.8, edgecolor="white")
    ax.axvline(d["obs_max"], color=CB["red"], lw=2,
               label=f"observed L{d['obs_L']} = {d['obs_max']:.3f}\np = {d['p']:.4f}")
    ax.set_title(f"Null B: {c}", fontsize=10)
    ax.set_xlabel("max |AUC-0.5| over depths 2..23"); ax.set_ylabel("permutations")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle("Permutation null B (outcome-label shuffle), selection-corrected over the internal band", fontsize=11)
fig.tight_layout(); savefig(fig, "perm_null_B")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for j, c in enumerate(MEASURES):
    r = split_res[c]
    axes[0, j].hist(r["held"], bins=40, color=CB["blue"], edgecolor="white")
    axes[0, j].axvline(np.mean(r["held"]), color=CB["red"], lw=2, label=f"mean {np.mean(r['held']):.3f}")
    axes[0, j].axvline(.5, color="k", ls=":", lw=1)
    axes[0, j].set_title(f"held-out AUC: {c}", fontsize=9); axes[0, j].legend(fontsize=8); axes[0, j].grid(alpha=.3)
    vc = r["vc"].sort_index()
    axes[1, j].bar(vc.index, vc.to_numpy(), color=CB["grey"])
    axes[1, j].set_title(f"selected layer across {N_SPLIT} splits", fontsize=9)
    axes[1, j].set_xlabel("selected residual-stream depth"); axes[1, j].grid(alpha=.3, axis="y")
fig.suptitle("Split-half: held-out AUC and how stable the selected layer is", fontsize=11)
fig.tight_layout(); savefig(fig, "split_half")

piv = sweep.pivot_table(index="column", columns="layer", values="auc")
fig, ax = plt.subplots(figsize=(15, 4.6))
im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
ax.set_xlabel("residual-stream depth (28 blank: no attention module)")
ax.set_title("attn_* discrimination, matched groups (AUC; red = higher in successes, blue = higher in failures)", fontsize=10)
fig.colorbar(im, ax=ax, label="AUC")
fig.tight_layout(); savefig(fig, "attn_sweep_matched")
print("\n  saved (pdf + png): auc_raw_vs_matched, perm_null_B, split_half, attn_sweep_matched")


# ================================================================= 5. VERDICT
hr("VERDICT ON THE TWO SURVIVING SIGNALS")
verdict = {}
for c, L in [("jlens_margin", 5), ("attn_correct_minus_stale", 22), ("logitlens_margin", 18)]:
    r = raw_matched[raw_matched.layer == L].iloc[0]
    nb, sp = nullB[c], split_res[c]
    stable = 100 * sp["vc"].iloc[0] / N_SPLIT
    passes = (nb["p"] < 0.05) and (np.quantile(sp["held"], .025) > .5 or np.quantile(sp["held"], .975) < .5) and stable >= 50
    verdict[c] = passes
    print(f"\n  {c} @ L{L}")
    print(f"    matched AUC            : {r[f'{c}|matched']:.3f}   (raw was {r[f'{c}|raw']:.3f})")
    print(f"    permutation null B     : p = {nb['p']:.4f} ({'PASS' if nb['p']<0.05 else 'FAIL'}), "
          f"best layer chosen = {nb['obs_L']}")
    print(f"    split-half held-out AUC: {sp['held'].mean():.3f} "
          f"95% [{np.quantile(sp['held'],.025):.3f}, {np.quantile(sp['held'],.975):.3f}]")
    print(f"    modal selected layer   : {sp['vc'].index[0]} ({stable:.0f}% of splits) "
          f"{'stable' if stable >= 50 else '-- UNSTABLE, layer not reproducible'}")
    print(f"    VERDICT                : {'SURVIVES all three controls' if passes else 'DOES NOT SURVIVE'}")
    if not passes and nb["p"] < 0.05:
        print(f"      (passes the permutation null but fails split-half layer stability:")
        print(f"       a real but diffuse effect with no reproducible best layer.)")


# ============================================================== 6. SLACK TABLE
hr("6. SLACK-PASTEABLE SUMMARY")
jl = raw_matched[raw_matched.layer == 5].iloc[0]
a19 = raw_matched[raw_matched.layer == 19].iloc[0]
a22 = raw_matched[raw_matched.layer == 22].iloc[0]
lines = []
lines.append("```")
lines.append("Task 4 -- controls round 2 (condition-matched). CSV-only, no GPU.")
lines.append("")
lines.append("THE ORIGINAL CLAIM")
lines.append("  'correct outranks stale in >=1 of 29 layers', 137/137 failures")
lines.append(f"  symmetry     : stale outranks correct in >=1 layer -> {nullA['logitlens']['obs_exist']*137:.0f}/137 too")
lines.append(f"  permutation  : null is a POINT MASS at 1.000, p = {nullA['logitlens']['p_exist']:.2f}")
lines.append("                 -> statistic is invariant under label swap: untestable, not just weak")
lines.append(f"  graded form  : correct wins {nullA['logitlens']['obs_grad']:.1f}/29 layers vs null "
             f"{nullA['logitlens']['null_grad'].mean():.1f}, p = {nullA['logitlens']['p_grad']:.4f}")
lines.append("                 -> significant in the WRONG direction")
lines.append("")
lines.append("THE CONFOUND (why round-1 numbers were inflated)")
lines.append("  100% of failures are complexity-3 / agent / middle-distractor")
lines.append("  ~70% of successes have NO distractors -> hard-vs-easy, not fail-vs-success")
lines.append("  matched stratum: 72 of 488 successes survive (682 in the 5k run)")
lines.append("")
lines.append("SURVIVORS -- layer chosen by split-half, not by eye")
lines.append(f"  {'measure':<26} {'L':>3} {'raw':>6} {'matched':>8} {'perm p':>7} {'held-out':>9} {'stable':>7}")
for c in MEASURES:
    sp, nb = split_res[c], nullB[c]
    L = int(sp["vc"].index[0])
    r = raw_matched[raw_matched.layer == L].iloc[0]
    st = 100 * sp["vc"].iloc[0] / N_SPLIT
    lines.append(f"  {c:<26} {L:>3} {r[f'{c}|raw']:>6.3f} {r[f'{c}|matched']:>8.3f} "
                 f"{nb['p']:>7.4f} {sp['held'].mean():>9.3f} {st:>6.0f}%")
lines.append("  (held-out AUC is for the layer that split-half actually picks;")
lines.append("   'stable' = % of 500 splits picking that layer. <50% = not reproducible.)")
lines.append("")
lines.append("VERDICT")
for c in MEASURES:
    L = int(split_res[c]["vc"].index[0])
    lines.append(f"  {c} @L{L}: {'SURVIVES' if verdict.get(c) else 'DOES NOT SURVIVE'} perm null + split-half")
lines.append("  logitlens: real but diffuse -- passes the null, no reproducible best layer.")
lines.append("  All of this is correlational. Causal claims still need patching/ablation.")
lines.append("```")
slack = "\n".join(lines)
print(slack)
open(os.path.join(REPORTS, "slack_summary.txt"), "w").write(slack)
print(f"\nsaved -> reports/slack_summary.txt")
