#!/usr/bin/env python3
"""
existence_null_sim.py -- model-free simulation of the existence-over-depth statistic.

Shows, with no model and no data, why

    E = 1[ exists a depth L : rank_L(c) < rank_L(s) ]

is not evidence that c is represented. Two exchangeable noisy series cross almost
surely once the number of depths is moderate, so E saturates at 1; and under the
label-swap null (Null A) the distribution of E collapses to a point mass, leaving
no null to test against.

Panels
  (a) P(E = 1) as a function of the number of depths L, for several noise levels
      and a small true offset. Saturates well before the 29 depths of a real sweep.
  (b) Null A distribution of E at L = 29: a point mass at 1.0. Compared against the
      graded statistic G, whose null is non-degenerate and therefore testable.
  (c) G under Null A vs observed G, for a chosen offset. Shows that a genuine
      offset moves G but leaves E untouched.

Seeded and deterministic. Writes a vector PDF plus a PNG preview.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260906
RNG = np.random.default_rng(SEED)
N_EX = 2000          # simulated examples per configuration
N_PERM = 5000        # permutations for the Null A distributions
L_REAL = 29          # residual-stream positions in the case study
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe (Okabe-Ito).
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#666666", "yellow": "#F0E442"}


def simulate(n_ex, L, noise, offset, rng):
    """Two latent 'quality' series over L depths, read out as a comparison.

    c and s each get an example-level level plus depth-level noise. `offset` shifts
    c relative to s: 0.0 makes the two exchangeable. Returns per-example (E, G).
    Larger latent value = better (i.e. lower rank), so c wins where c > s.
    """
    base_c = rng.normal(0, 1, size=(n_ex, 1))
    base_s = rng.normal(0, 1, size=(n_ex, 1))
    c = base_c + offset + rng.normal(0, noise, size=(n_ex, L))
    s = base_s + rng.normal(0, noise, size=(n_ex, L))
    wins = c > s
    return wins.any(axis=1), wins.sum(axis=1)


def null_A(wins_c, wins_s, n_perm, rng):
    """Null A: swap the correct/stale labels for a whole example at once.

    The whole-series swap is the correct exchangeability. Swapping independently at
    each depth would destroy within-example cross-depth correlation and produce a
    falsely narrow null for a maximum statistic.
    """
    n = len(wins_c)
    flips = rng.random((n_perm, n)) < 0.5
    return np.where(flips, wins_s, wins_c)


def main():
    fig = plt.figure(figsize=(13.5, 4.0))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # ---------------------------------------------------------------- panel (a)
    ax = fig.add_subplot(gs[0, 0])
    Ls = np.arange(1, 49)
    for noise, col in [(0.5, CB["blue"]), (1.0, CB["orange"]), (2.0, CB["green"])]:
        pe = [simulate(N_EX, L, noise, 0.0, np.random.default_rng(SEED + L))[0].mean() for L in Ls]
        ax.plot(Ls, pe, "-", color=col, lw=1.8, label=f"exchangeable, noise={noise}")
    pe_off = [simulate(N_EX, L, 1.0, -0.75, np.random.default_rng(SEED + L))[0].mean() for L in Ls]
    ax.plot(Ls, pe_off, "--", color=CB["red"], lw=1.8, label="c genuinely worse (offset $-0.75$)")
    ax.axvline(L_REAL, color=CB["grey"], ls=":", lw=1.4)
    ax.text(L_REAL - 0.8, 0.28, f"L={L_REAL}", rotation=90, fontsize=8, color=CB["grey"], ha="right")
    ax.set_xlabel("number of depths $L$"); ax.set_ylabel(r"$P(E=1)$")
    ax.set_title("(a) The existence test saturates", fontsize=10)
    ax.set_ylim(0, 1.03); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower right")

    # ---------------------------------------------------------------- panel (b)
    # Give c a genuine DISADVANTAGE (offset < 0), mirroring the case study where the
    # correct state wins fewer depths than label-swap chance. E cannot see it; G can.
    ax = fig.add_subplot(gs[0, 1])
    NOISE_B, OFFSET_B = 2.0, -0.75
    rng = np.random.default_rng(SEED)
    base_c = rng.normal(0, 1, (N_EX, 1)); base_s = rng.normal(0, 1, (N_EX, 1))
    C = base_c + OFFSET_B + rng.normal(0, NOISE_B, (N_EX, L_REAL))
    S = base_s + rng.normal(0, NOISE_B, (N_EX, L_REAL))
    win_c, win_s = (C > S), (S > C)
    Ec, Es = win_c.any(1), win_s.any(1)
    Gc, Gs = win_c.sum(1), win_s.sum(1)
    nullE = null_A(Ec, Es, N_PERM, np.random.default_rng(SEED + 1)).mean(axis=1)
    nullG = null_A(Gc, Gs, N_PERM, np.random.default_rng(SEED + 1)).mean(axis=1)
    ax.hist(nullG, bins=40, color=CB["blue"], alpha=.85, edgecolor="white",
            label=f"$G$ null (sd={nullG.std():.2f})")
    ax.axvline(Gc.mean(), color=CB["red"], lw=2.2,
               label=f"observed $G$={Gc.mean():.2f}")
    lo, hi = min(Gc.mean(), nullG.min()), max(Gc.mean(), nullG.max())
    pad = 0.12 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(r"mean $G$ under Null A"); ax.set_ylabel("permutations")
    ax.set_title(r"(b) $G$ has a null; $E$ does not", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right")
    ax.text(.03, .55,
            f"$E$: observed {Ec.mean():.3f}, null mass at {nullE.mean():.3f}\n"
            f"(sd {nullE.std():.1e}) -- no separation.\n"
            f"$G$ separates at the same effect.",
            transform=ax.transAxes, va="center", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.35", fc=CB["yellow"], ec=CB["grey"], alpha=.9))

    # ---------------------------------------------------------------- panel (c)
    ax = fig.add_subplot(gs[0, 2])
    offsets = np.linspace(-1.0, 1.0, 21)
    obsE, obsG, loG, hiG = [], [], [], []
    for off in offsets:
        r = np.random.default_rng(SEED + int((off + 2) * 1000))
        e, g = simulate(N_EX, L_REAL, 1.0, off, r)
        obsE.append(e.mean()); obsG.append(g.mean())
    ax.plot(offsets, obsE, "-o", ms=3, color=CB["grey"], label=r"$E$ (existence)")
    ax.plot(offsets, np.array(obsG) / L_REAL, "-s", ms=3, color=CB["blue"], label=r"$G/L$ (graded)")
    ax.axhline(0.5, color=CB["red"], ls=":", lw=1.2, label="exchangeable value of $G/L$")
    ax.set_xlabel("true offset of $c$ relative to $s$")
    ax.set_ylabel("statistic value")
    ax.set_title("(c) $E$ is blind to the effect size", fontsize=10)
    ax.set_ylim(0, 1.03); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="center right")

    fig.suptitle("The existence-over-depth statistic has no usable null; the graded form does",
                 fontsize=11, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"existence_null_sim.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ numbers
    print("=" * 72)
    print("EXISTENCE-NULL SIMULATION  (seed %d, %d examples, %d permutations)"
          % (SEED, N_EX, N_PERM))
    print("=" * 72)
    print("\n(a) P(E=1) for two EXCHANGEABLE series (offset = 0, noise = 1.0):")
    for L in (4, 8, 12, 16, 24, L_REAL, 48):
        e, _ = simulate(N_EX, L, 1.0, 0.0, np.random.default_rng(SEED + L))
        print(f"      L = {L:2d} depths : P(E=1) = {e.mean():.4f}")
    print(f"\n    At L = {L_REAL}, the number of residual-stream positions in the case study,")
    print(f"    two exchangeable series already satisfy the existence test essentially always.")
    print("\n(b) Null A at L = %d, with c given a genuine disadvantage (offset %.2f):"
          % (L_REAL, OFFSET_B))
    print(f"      E : observed {Ec.mean():.4f} | null mean {nullE.mean():.4f} "
          f"| null sd {nullE.std():.3e}")
    print(f"          -> near-degenerate here; in the case-study data E = 1 in BOTH")
    print(f"             directions for all 137 examples, so the null is an EXACT")
    print(f"             point mass and no test is possible at all.")
    print(f"      G : observed {Gc.mean():.2f}  | null mean {nullG.mean():.2f} "
          f"| null sd {nullG.std():.3f}      -> testable")
    print("\n(c) Sensitivity to a genuine effect (offset sweep):")
    for off in (-1.0, -0.5, 0.0, 0.5, 1.0):
        r = np.random.default_rng(SEED + int((off + 2) * 1000))
        e, g = simulate(N_EX, L_REAL, 1.0, off, r)
        print(f"      offset {off:+.1f} : E = {e.mean():.4f}   G/L = {g.mean()/L_REAL:.3f}")
    print("\n    E is pinned near 1 across the whole range, including where c is")
    print("    genuinely WORSE than s. G tracks the offset monotonically.")
    print("\nsaved: reports/existence_null_sim.pdf (+ .png)")


if __name__ == "__main__":
    main()
