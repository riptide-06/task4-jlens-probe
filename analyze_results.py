#!/usr/bin/env python3
"""
analyze_results.py — offline analysis of a jlens_probe.py run.

    python analyze_results.py outputs/run1
    python analyze_results.py outputs/run1 --plot          # writes PNGs next to the CSVs

Produces the four tables Axis 2 actually needs:

  T1  Behaviour           — how often the model emits the STALE room, per condition.
  T2  ENCODE vs RETRIEVE  — of the stale-output failures, how many had the *correct* room
                            decodable in some internal layer? That fraction is the
                            retrieval-failure rate; 1 - it is the representation-failure rate.
  T3  Depth profile       — mean rank of correct vs stale room by layer, both lenses.
                            The shape of this curve is the headline figure.
  T4  Attention           — attention mass on the update mention vs the stale anchor,
                            split by whether the model got it right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def _rate(series) -> str:
    return f"{series.mean():6.1%}" if len(series) else "   n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory written by jlens_probe.py --out-dir")
    ap.add_argument("--plot", action="store_true", help="also write depth-profile PNGs")
    ap.add_argument("--group", nargs="+",
                    default=["compression", "complexity_level", "entity_type"],
                    help="metadata columns to break results down by")
    args = ap.parse_args()

    run = Path(args.run_dir)
    ex_path, ly_path = run / "examples.csv", run / "layers.csv"
    for p in (ex_path, ly_path):
        if not p.exists():
            print(f"missing {p} — point me at a --out-dir from jlens_probe.py")
            return 1

    ex = pd.read_csv(ex_path)
    ly = pd.read_csv(ly_path)
    n_layers = int(ly["layer"].max())
    print(f"loaded {len(ex)} examples x {ly['layer'].nunique()} depths "
          f"(model depth {n_layers}) from {run}")

    # ------------------------------------------------------------------ T1
    print("\n" + "=" * 78)
    print("T1  BEHAVIOUR — what the model actually emits")
    print("=" * 78)
    print(ex["outcome"].value_counts(dropna=False).to_frame("n")
          .assign(pct=lambda d: (d["n"] / len(ex)).map("{:.1%}".format)))
    for col in args.group:
        if col in ex.columns and ex[col].nunique() > 1:
            print(f"\n  stale-output rate by {col}:")
            t = (ex.groupby(col)["output_is_stale"]
                   .agg(n="size", stale_rate="mean")
                   .assign(stale_rate=lambda d: d["stale_rate"].map("{:.1%}".format)))
            print(t.to_string())

    # ------------------------------------------------------------------ T2
    print("\n" + "=" * 78)
    print("T2  ENCODE vs RETRIEVE — the core Axis 2 measurement")
    print("=" * 78)
    stale = ex[ex["output_is_stale"]].copy()
    if stale.empty:
        print("  no stale-output examples in this run — nothing to dissect.")
    else:
        print(f"  {len(stale)} examples where the OUTPUT was the stale room.\n")
        for lens in ("logitlens", "jlens"):
            cross = stale[f"{lens}_crossover_layer"] >= 0
            print(f"  [{lens}]")
            print(f"    correct room outranked stale in >=1 internal layer : "
                  f"{cross.sum():4d}/{len(stale)}  ({cross.mean():.1%})   <- RETRIEVAL failure")
            print(f"    correct room never won at any depth                : "
                  f"{(~cross).sum():4d}/{len(stale)}  ({(~cross).mean():.1%})   <- REPRESENTATION failure")
            print(f"    best internal rank of correct room  (median)       : "
                  f"{stale[f'{lens}_best_rank_correct'].median():.0f}")
            print(f"    depth of that best rank (fraction of stack, median): "
                  f"{stale[f'{lens}_best_rank_depth_frac'].median():.2f}")
            print(f"    layers where correct beat stale     (mean)         : "
                  f"{stale[f'{lens}_n_layers_correct_wins'].mean():.1f}\n")

        for col in args.group:
            if col in stale.columns and stale[col].nunique() > 1:
                t = stale.groupby(col).agg(
                    n=("example_id", "size"),
                    retrieval_failure_rate=("jlens_crossover_layer", lambda s: (s >= 0).mean()),
                    median_best_rank=("jlens_best_rank_correct", "median"),
                )
                t["retrieval_failure_rate"] = t["retrieval_failure_rate"].map("{:.1%}".format)
                print(f"  J-Lens breakdown by {col}:")
                print(t.to_string(), "\n")

    # ------------------------------------------------------------------ T3
    print("=" * 78)
    print("T3  DEPTH PROFILE — mean rank by layer (lower = more decodable)")
    print("=" * 78)
    cols = {}
    for lens in ("logitlens", "jlens"):
        for which in ("correct", "stale"):
            c = f"{lens}_rank_{which}"
            if c in ly.columns:
                cols[c] = "mean"
    prof = ly.groupby("layer").agg(cols).round(1)
    prof.index.name = "layer"
    # print a compact sample of depths so the console stays readable
    step = max(1, len(prof) // 14)
    print(prof.iloc[::step].to_string())

    if not stale.empty:
        ly_stale = ly[ly["example_id"].isin(stale["example_id"])]
        print("\n  restricted to STALE-OUTPUT failures:")
        prof_s = ly_stale.groupby("layer").agg(cols).round(1)
        print(prof_s.iloc[::step].to_string())
        best = prof_s["jlens_rank_correct"].idxmin()
        print(f"\n  -> J-Lens rank of the CORRECT room bottoms out at layer {best} "
              f"(rank {prof_s.loc[best, 'jlens_rank_correct']:.1f}) "
              f"then degrades to {prof_s['jlens_rank_correct'].iloc[-1]:.1f} at the output.")
        print("     A deep dip followed by a rise is the encoded-then-discarded signature.")

    # ------------------------------------------------------------------ T4
    print("\n" + "=" * 78)
    print("T4  ATTENTION at the readout position")
    print("=" * 78)
    acols = [c for c in ("attn_correct_update_mass", "attn_stale_anchor_mass",
                         "attn_correct_mass", "attn_stale_mass") if c in ly.columns]
    if not acols or ly[acols].isna().all().all():
        print("  no attention captured (run without --no-attention to populate these).")
    else:
        t = ly.groupby("output_is_stale")[acols].mean().round(4)
        t.index = t.index.map({False: "answered correctly", True: "answered STALE"})
        print(t.to_string())
        print("\n  by layer (top 10 layers by attention to the update mention):")
        by_layer = ly.groupby("layer")[acols].mean()
        print(by_layer.nlargest(10, "attn_correct_update_mass").round(4).to_string())

    # ------------------------------------------------------------------ plots
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[plot] matplotlib not installed — skipping figures")
            return 0

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, subset, title in (
            (axes[0], ly, f"all examples (n={len(ex)})"),
            (axes[1], ly[ly["output_is_stale"]] if stale is not None else ly,
             f"stale-output failures (n={len(stale)})"),
        ):
            g = subset.groupby("layer")
            for lens, style in (("logitlens", "--"), ("jlens", "-")):
                for which, colour in (("correct", "tab:green"), ("stale", "tab:red")):
                    c = f"{lens}_rank_{which}"
                    if c in subset.columns:
                        ax.plot(g[c].mean(), style, color=colour, alpha=0.9,
                                label=f"{lens} {which}")
            ax.set_yscale("log")
            ax.set_xlabel("residual depth")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("mean rank in vocabulary (log, lower = better)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Is the updated state decodable inside the model?")
        fig.tight_layout()
        out = run / "depth_profile.png"
        fig.savefig(out, dpi=140)
        print(f"\n[plot] wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
