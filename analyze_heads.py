#!/usr/bin/env python3
"""
analyze_heads.py — per-(layer, head) attention analysis of stale-state failures.

    python3 analyze_heads.py                    # defaults below
    python3 analyze_heads.py --out-dir reports

Deliverables
    head_ranking.csv  every (layer, head), ranked by how much more the head
                      favours the STALE ANCHOR over the UPDATE MENTION in
                      stale-output failures relative to correct-output successes.
    heatmap.png       28x28 grid, cell = mean(stale anchor) - mean(update mention)
                      over the 137 stale-output failures.

--------------------------------------------------------------------------
COLUMN MAPPING — read this before trusting any number
--------------------------------------------------------------------------
jlens_probe.py:attention_stats() computes FOUR attention quantities per layer:

    attn_correct_mass        mass on ALL mentions of the updated room
    attn_stale_mass          mass on ALL mentions of the stale room
    attn_correct_update_mass mass on the LAST correct mention  == "update mention"
    attn_stale_anchor_mass   mass on the FIRST stale mention   == "stale anchor"

layers.csv carries all four. heads.csv carries only the first two — the
per-head arrays (_per_head_correct / _per_head_stale) are computed over the
BROAD spans only, so no anchor/update decomposition exists per head anywhere
on disk. Recovering it needs the raw attention rows, which were not saved.

This matters differently for the two sides of the contrast:

  FAILURES  All 137 records in the failure run have n_stale_mentions == 1 and
            n_correct_mentions == 1, so broad span == restricted span exactly.
            heads.csv attn_stale_mass IS the stale-anchor mass and
            attn_correct_mass IS the update-mention mass. Verified to 2e-8
            against layers.csv. The heatmap is exact.

  SUCCESSES 416 of the 500 mixed-run records have >= 2 stale mentions, so the
            broad span sweeps in post-update stale mentions the anchor measure
            excludes. Broad stale mass over the 488 successes is 0.0226; the
            paper's anchor-restricted figure is 0.0086. They are not the same
            measurement and heads.csv cannot produce the latter.

            --baseline matched (default) therefore restricts the success set to
            records with n_stale_mentions == 1 AND n_correct_mentions == 1, where
            broad == anchor/update by construction, matching the failure run's
            regime. That is span-clean but a smaller, different population: its
            anchor mass is 0.0153, not 0.0086.

            --baseline broad uses all 488 successes with the broad span. Larger
            n, but the span definition no longer matches the failure side.

NEITHER success baseline reproduces the 0.0086 headline, because that number is
not derivable from heads.csv. Closing the gap needs a GPU re-run that dumps
per-head anchor/update masses (add per_head=True variants of last_correct /
first_stale in attention_stats and widen the heads.csv writer).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

# heads.csv column -> what it actually measures, given single-mention records
STALE_ANCHOR = "attn_stale_mass"     # first (== only) stale mention
UPDATE_MENTION = "attn_correct_mass"  # last (== only) correct mention

# Headline figures this analysis must reproduce on the failure side.
EXPECTED_FAIL_ANCHOR = 0.030
FAIL_ANCHOR_TOL = 0.008


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one probe run's heads.csv + examples.csv."""
    heads = pd.read_csv(run_dir / "heads.csv")
    examples = pd.read_csv(run_dir / "examples.csv")
    missing = {STALE_ANCHOR, UPDATE_MENTION, "example_id", "layer", "head"} - set(heads.columns)
    if missing:
        sys.exit(f"[fatal] {run_dir}/heads.csv is missing columns: {sorted(missing)}")
    return heads, examples


def join(heads: pd.DataFrame, examples: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Join per-head rows to per-example metadata, checking the join is sound."""
    meta = examples[["example_id", "output_is_stale",
                     "n_stale_mentions", "n_correct_mentions"]]
    h_ids, e_ids = set(heads.example_id), set(examples.example_id)
    if h_ids != e_ids:
        sys.exit(f"[fatal] {run_dir}: example_id mismatch — "
                 f"{len(h_ids - e_ids)} only in heads.csv, {len(e_ids - h_ids)} only in examples.csv")

    merged = heads.merge(meta, on="example_id", how="inner", validate="many_to_one")
    if len(merged) != len(heads):
        sys.exit(f"[fatal] {run_dir}: join changed row count {len(heads)} -> {len(merged)}")

    # heads.csv denormalises `outcome`; it must agree with examples.csv.
    disagree = (merged["outcome"].eq("stale") != merged["output_is_stale"]).sum()
    if disagree:
        sys.exit(f"[fatal] {run_dir}: {disagree} rows where heads.outcome "
                 f"disagrees with examples.output_is_stale — join or column mapping is wrong")

    # One row per (example, layer, head), no duplicates.
    if merged.duplicated(["example_id", "layer", "head"]).any():
        sys.exit(f"[fatal] {run_dir}: duplicate (example_id, layer, head) rows")

    merged["single_mention"] = (merged.n_stale_mentions == 1) & (merged.n_correct_mentions == 1)
    return merged


def per_head_means(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Mean stale-anchor and update-mention mass for every (layer, head)."""
    g = (df.groupby(["layer", "head"])
           .agg(**{f"stale_anchor_{tag}": (STALE_ANCHOR, "mean"),
                   f"update_{tag}": (UPDATE_MENTION, "mean"),
                   f"n_{tag}": ("example_id", "nunique")})
           .reset_index())
    g[f"delta_{tag}"] = g[f"stale_anchor_{tag}"] - g[f"update_{tag}"]
    return g


def build_heatmap(fail_heads: pd.DataFrame, out_png: Path,
                  clip_pct: float | None = None) -> None:
    """28x28 diverging heatmap of mean(stale anchor) - mean(update) on failures.

    clip_pct saturates the colour scale at that percentile of |delta| so the
    mid-range stays visible; None (default) keeps the full linear range, which
    hides nothing but leaves ~95%% of cells near white because the effect really
    is carried by a couple of dozen heads.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = (fail_heads.assign(delta=fail_heads[STALE_ANCHOR] - fail_heads[UPDATE_MENTION])
                      .pivot_table(index="head", columns="layer", values="delta", aggfunc="mean")
                      .sort_index()
                      .reindex(columns=sorted(fail_heads.layer.unique())))
    n_ex = fail_heads.example_id.nunique()
    absvals = np.abs(grid.to_numpy())
    full = float(np.nanmax(absvals))
    lim = float(np.nanpercentile(absvals, clip_pct)) if clip_pct else full

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(grid.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="equal", origin="lower", interpolation="nearest")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Head", fontsize=11)
    ax.set_title("Attention at readout: stale anchor − update mention\n"
                 f"Qwen2.5-7B-Instruct, {n_ex} stale-output failures",
                 fontsize=11, pad=8)

    layers = list(grid.columns)
    heads = list(grid.index)
    ax.set_xticks(range(0, len(layers), 2), [str(layers[i]) for i in range(0, len(layers), 2)],
                  fontsize=9)
    ax.set_yticks(range(0, len(heads), 2), [str(heads[i]) for i in range(0, len(heads), 2)],
                  fontsize=9)
    # Faint cell separators keep the grid readable when the figure is scaled down.
    ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heads), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                        extend="both" if clip_pct else "neither")
    cbar.set_label("mean attention mass difference\n(+ favours stale anchor)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    if clip_pct:
        cbar.ax.set_title(f"clipped at\np{clip_pct:g} (max {full:.2f})", fontsize=7, pad=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--failures-dir", type=Path,
                   default=Path("outputs/colab_run_qwen_failures_fullsweep"),
                   help="run containing the stale-output failures")
    p.add_argument("--success-dir", type=Path,
                   default=Path("outputs/colab_run_qwen_fullsweep_500"),
                   help="mixed run supplying the correct-output baseline")
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--baseline", choices=("matched", "broad"), default="matched",
                   help="matched: successes with a single stale+correct mention, where the "
                        "broad span equals the anchor/update span. broad: all successes.")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--clip-pct", type=float, default=None,
                   help="saturate the heatmap colour scale at this percentile of |delta| "
                        "(e.g. 98) to bring out the mid-range; default keeps the full range")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fail_heads, fail_ex = load_run(args.failures_dir)
    succ_heads, succ_ex = load_run(args.success_dir)
    fail = join(fail_heads, fail_ex, args.failures_dir)
    succ = join(succ_heads, succ_ex, args.success_dir)

    # The two runs index different JSONL files, so example_ids collide across runs.
    # Never pool them into one id space: failures come from one run, successes the other.
    failures = fail[fail.output_is_stale]
    successes = succ[~succ.output_is_stale]
    if not failures.single_mention.all():
        sys.exit("[fatal] failure run contains multi-mention records — heads.csv "
                 "attn_stale_mass is no longer the stale-anchor mass there")
    matched = successes[successes.single_mention]

    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    fail_anchor = failures[STALE_ANCHOR].mean()
    fail_update = failures[UPDATE_MENTION].mean()
    print(f"  failures      n={failures.example_id.nunique():4d}  "
          f"stale anchor={fail_anchor:.4f}  update mention={fail_update:.4f}")
    print(f"  successes     n={successes.example_id.nunique():4d}  "
          f"stale anchor={successes[STALE_ANCHOR].mean():.4f}  "
          f"update mention={successes[UPDATE_MENTION].mean():.4f}   [broad span]")
    print(f"  successes     n={matched.example_id.nunique():4d}  "
          f"stale anchor={matched[STALE_ANCHOR].mean():.4f}  "
          f"update mention={matched[UPDATE_MENTION].mean():.4f}   [matched, span-clean]")

    if abs(fail_anchor - EXPECTED_FAIL_ANCHOR) > FAIL_ANCHOR_TOL:
        sys.exit(f"\n[fatal] failure-side stale-anchor mass {fail_anchor:.4f} is not within "
                 f"{FAIL_ANCHOR_TOL} of the known {EXPECTED_FAIL_ANCHOR:.3f}. "
                 f"The join or column mapping is wrong — stopping.")
    print(f"\n  OK  failure-side stale-anchor mass {fail_anchor:.4f} reproduces the "
          f"known ~{EXPECTED_FAIL_ANCHOR:.3f}.")
    print("  WARNING  the ~0.009 success figure is anchor-restricted and is NOT derivable")
    print("           from heads.csv, which stores broad-span masses only. See the module")
    print(f"           docstring. Using --baseline {args.baseline} for the ranking.")

    baseline = matched if args.baseline == "matched" else successes

    ranking = (per_head_means(failures, "fail")
               .merge(per_head_means(baseline, "succ"), on=["layer", "head"], how="outer")
               .merge(per_head_means(successes, "succ_broad")
                      .drop(columns=["n_succ_broad"]), on=["layer", "head"], how="outer"))
    # Difference in differences: how much more this head favours the stale anchor
    # over the update mention in failures than it does in successes.
    ranking["score"] = ranking["delta_fail"] - ranking["delta_succ"]
    ranking["score_broad"] = ranking["delta_fail"] - ranking["delta_succ_broad"]
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    out_csv = args.out_dir / "head_ranking.csv"
    ranking.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print(f"TOP {args.top} HEADS  (score = failure Δ − success Δ, "
          f"Δ = stale anchor − update mention)")
    print("=" * 78)
    cols = ["rank", "layer", "head", "stale_anchor_fail", "update_fail", "delta_fail",
            "stale_anchor_succ", "update_succ", "delta_succ", "score"]
    print(ranking.head(args.top)[cols].to_string(index=False,
          formatters={c: "{:.4f}".format for c in cols[3:]}))

    top_layers = ranking.head(args.top).layer.value_counts().sort_index()
    print(f"\n  layer distribution of the top {args.top}: "
          + ", ".join(f"L{l}×{n}" for l, n in top_layers.items()))

    out_png = args.out_dir / "heatmap.png"
    build_heatmap(failures, out_png, clip_pct=args.clip_pct)

    print(f"\n  wrote {out_csv}")
    print(f"  wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
