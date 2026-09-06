#!/usr/bin/env python3
"""
paper_extras.py -- two small sourced facts the paper cites.

(A) PER-HEAD CORROBORATION of the depth-22 attention result.
    Source: reports/head_ranking.csv, produced by analyze_heads.py (committed in
    6c1311e). Its `score` column is delta_fail - delta_succ, where each delta is
    (stale-anchor attention mass - update-mention mass) for that head. A high
    score therefore means: this head attends to the stale anchor over the update
    much more in failures than in successes. That is an independent statistic --
    per-head, and built from the raw attention tensors rather than the
    per-example aggregates in layers.csv -- so where its top heads sit is a
    genuine check on the depth-22 result rather than a restatement of it.

(B) BEHAVIOURAL RATES by condition, from the 5k run's examples.csv.

Writes reports/paper_extras.json and prints both.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
OUT = {}

# ---------------------------------------------------------------- (A) heads
hr = pd.read_csv(os.path.join(REPORTS, "head_ranking.csv")).sort_values("score", ascending=False)
top20 = hr.head(20)
in_1922 = int(top20.layer.between(19, 22).sum())
in_1926 = int(top20.layer.between(19, 26).sum())
at_22 = int((top20.layer == 22).sum())
best = hr.iloc[0]

print("=" * 78)
print("(A) PER-HEAD CORROBORATION  (reports/head_ranking.csv, from analyze_heads.py)")
print("=" * 78)
print(f"  heads ranked: {len(hr)}   score = delta_fail - delta_succ")
print(f"  top-20 heads falling in depths 19-22 : {in_1922}/20")
print(f"  top-20 heads falling in depths 19-26 : {in_1926}/20")
print(f"  top-20 heads at depth 22 alone       : {at_22}/20")
print(f"  highest-scoring head                 : depth {int(best["layer"])}, head {int(best["head"])} "
      f"(score {best["score"]:.3f})")
print(f"  depth distribution of the top 20     : "
      f"{dict(sorted(top20.layer.value_counts().to_dict().items()))}")
OUT["heads"] = dict(n_ranked=int(len(hr)), top20_in_19_22=in_1922, top20_in_19_26=in_1926,
                    top20_at_22=at_22, top_head_depth=int(best["layer"]),
                    top_head_index=int(best["head"]), top_head_score=float(best["score"]),
                    top20_depth_counts={int(k): int(v) for k, v in top20.layer.value_counts().items()})

# ------------------------------------------------------- (B) behavioural rates
ex = pd.read_csv(os.path.join(HERE, "outputs/colab_run_qwen_5k_L5-8-13/examples.csv"))
n = len(ex)
overall = ex.outcome.value_counts(normalize=True) * 100
print("\n" + "=" * 78)
print(f"(B) BEHAVIOURAL RATES  (5k run examples.csv, n={n})")
print("=" * 78)
print(f"  overall: " + ", ".join(f"{k} {v:.1f}%" for k, v in overall.items()))
OUT["overall_pct"] = {k: round(float(v), 2) for k, v in overall.items()}

for col in ("complexity_level", "entity_type"):
    print(f"\n  stale-output rate by {col}:")
    g = ex.groupby(col).outcome.apply(lambda s: 100 * (s == "stale").mean())
    cnt = ex.groupby(col).size()
    for k in g.index:
        print(f"    {col}={k}: {g[k]:5.2f}%  (n={cnt[k]})")
    OUT[f"stale_rate_by_{col}"] = {str(k): round(float(g[k]), 3) for k in g.index}
    OUT[f"n_by_{col}"] = {str(k): int(cnt[k]) for k in cnt.index}

json.dump(OUT, open(os.path.join(REPORTS, "paper_extras.json"), "w"), indent=2)
print(f"\nsaved -> reports/paper_extras.json")
