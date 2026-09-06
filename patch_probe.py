#!/usr/bin/env python3
"""
patch_probe.py -- causal test of the depth-19..22 attention result.

STATUS: NOT RUN. No CUDA GPU was available for this work. The script is complete
and runnable; every number it would produce is marked as a placeholder in the
paper (Appendix B) and in paper/NOTES.md. Do not cite results from this file
until it has actually been executed.

WHY THIS EXPERIMENT
-------------------
controls_check2.py establishes, on condition-matched groups and with a
selection-corrected null, that `attn_correct_minus_stale` at residual depth 22
separates failures from successes (matched AUC 0.973, held-out 0.970). That is
correlational. It does not show the attention pattern participates in producing
the stale answer. Only intervention does.

DESIGN
------
For each of the 137 stale-output failures:
  1. Find a DONOR: a success from the same condition cell (complexity_level,
     entity_type, distractor_position, n_distractors). Same cell matters -- an
     unmatched donor reintroduces exactly the confound documented in the paper.
  2. Run the failure prompt, patching attention-head OUTPUTS at decoder layers
     19..22 with the donor's activations at the same positions.
  3. Record whether the emitted room flips to the correct room.

The headline metric is the flip rate to the CORRECT room. Following
Zhang & Nanda (2024), we also log a continuous metric (logit difference between
correct and stale room) rather than relying on the discrete flip alone, since
the two can disagree and the discrete rate is the noisier of the pair.

CONTROLS -- all four are required for the result to mean anything
-----------------------------------------------------------------
(a) RANDOM HEADS at the same depths, matched in count. If patching a random
    equal-sized set of heads flips answers at a similar rate, the effect is
    "perturbing this many heads at this depth", not the identified circuit.
(b) LOWEST-RANKED HEADS from reports/head_ranking.csv, same count, same depths.
    A sharper version of (a): heads measured to be least stale-preferring.
(c) MEAN-ABLATION of the top-k stale-preferring heads (replace with the batch
    mean rather than a donor's activations). Separates "removing the stale
    signal" from "importing the donor's state".
(d) ANSWER-COPYING CONTROL. This is the one that has bitten this project before.
    A patch applied at or near the decision token can simply transfer the
    donor's answer, which is copying, not localization. Two guards:
      - DONOR-MISMATCH: use donors whose correct room DIFFERS from the target's
        correct room. If the output follows the donor's room rather than the
        target's, the patch is copying and the "flip rate" is an artifact.
      - We report flip-to-target-correct and flip-to-donor-room separately.
        Only the first is evidence. A high second number invalidates the first.

USAGE
-----
    python patch_probe.py --failures outputs/colab_run_qwen_failures_fullsweep \
                          --successes outputs/colab_run_qwen_fullsweep_500 \
                          --depths 19 20 21 22 --out reports/patching
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens_probe as jp  # reuse the exact model/prompt/readout paths under study

CELL = ["complexity_level", "entity_type", "distractor_position", "n_distractors"]


# ---------------------------------------------------------------- donor pairing
def build_pairs(fail_df: pd.DataFrame, succ_df: pd.DataFrame,
                mismatch_only: bool, rng: random.Random) -> List[Tuple[str, str]]:
    """Pair each failure with a same-cell success.

    mismatch_only=True restricts to donors whose correct room DIFFERS from the
    target's, which is the answer-copying control.
    """
    succ_by_cell: Dict[tuple, List[pd.Series]] = {}
    for _, r in succ_df.iterrows():
        succ_by_cell.setdefault(tuple(r[c] for c in CELL), []).append(r)

    pairs, unpaired = [], 0
    for _, f in fail_df.iterrows():
        cands = succ_by_cell.get(tuple(f[c] for c in CELL), [])
        if mismatch_only:
            cands = [c for c in cands if c["correct_state"] != f["correct_state"]]
        if not cands:
            unpaired += 1
            continue
        pairs.append((f, rng.choice(cands)))
    if unpaired:
        print(f"[pair] WARNING: {unpaired} failures had no same-cell donor "
              f"({'mismatch-only' if mismatch_only else 'any'}). They are EXCLUDED, "
              f"not paired across cells -- pairing across cells would reintroduce "
              f"the condition confound.")
    return pairs


# ------------------------------------------------------------------- patching
class HeadPatch:
    """Replace selected attention-head outputs at given decoder layers.

    Head h occupies columns [h*d_head, (h+1)*d_head) of the attention output
    before o_proj. We hook o_proj's INPUT so the substitution happens in
    per-head space, which is what makes "patch head (l, h)" well defined.
    """

    def __init__(self, model, layer_to_heads: Dict[int, List[int]],
                 donor: Optional[Dict[int, torch.Tensor]], position: int,
                 mean_bank: Optional[Dict[int, torch.Tensor]] = None):
        self.layers = jp.get_base(model).layers
        self.map = layer_to_heads
        self.donor = donor
        self.mean_bank = mean_bank
        self.position = position
        self.n_heads = model.config.num_attention_heads
        self._handles: List[Any] = []

    def __enter__(self):
        for li, heads in self.map.items():
            def pre_hook(_mod, args, _li=li, _heads=heads):
                x = args[0]
                d_head = x.shape[-1] // self.n_heads
                src = (self.mean_bank if self.donor is None else self.donor).get(_li)
                if src is None:
                    return None
                x = x.clone()
                for h in _heads:
                    lo, hi = h * d_head, (h + 1) * d_head
                    x[:, self.position, lo:hi] = src[:, lo:hi].to(x.dtype).to(x.device)
                return (x,) + args[1:]
            self._handles.append(
                self.layers[li].self_attn.o_proj.register_forward_pre_hook(pre_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def capture_head_inputs(model, ids, depths, position) -> Dict[int, torch.Tensor]:
    """Grab o_proj inputs (per-head attention outputs) at `position`."""
    grabbed: Dict[int, torch.Tensor] = {}
    layers = jp.get_base(model).layers
    handles = []
    for li in depths:
        def pre_hook(_mod, args, _li=li):
            grabbed[_li] = args[0][:, position, :].detach().clone()
            return None
        handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(pre_hook))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return grabbed


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", required=True, help="run dir with the 137 failures")
    ap.add_argument("--successes", required=True, help="run dir with successes")
    ap.add_argument("--depths", type=int, nargs="+", default=[19, 20, 21, 22])
    ap.add_argument("--head-ranking", default="reports/head_ranking.csv")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-in", default="bf16")
    ap.add_argument("--out", default="reports/patching")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("patch_probe.py requires a CUDA GPU. Detected none.\n"
              "This script has NOT been run for the paper; Appendix B is a "
              "specification, not a result. Re-run here once a GPU is available.")
        return 2

    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    fail = pd.read_json(os.path.join(a.failures, "records.jsonl"), lines=True)
    succ = pd.read_json(os.path.join(a.successes, "records.jsonl"), lines=True)
    succ = succ[succ.outcome == "correct"]
    if a.limit:
        fail = fail.head(a.limit)

    hr = pd.read_csv(a.head_ranking).sort_values("score", ascending=False)
    top = [(int(r.layer), int(r.head)) for _, r in hr.head(a.top_k).iterrows()
           if int(r.layer) in a.depths]
    bot = [(int(r.layer), int(r.head)) for _, r in hr.tail(a.top_k).iterrows()
           if int(r.layer) in a.depths]

    cfg = jp.ProbeConfig(model_id=a.model_id, load_in=a.load_in)
    model, tok = jp.load_model(cfg)

    def heads_map(pairs):
        m: Dict[int, List[int]] = {}
        for l, h in pairs:
            m.setdefault(l, []).append(h)
        return m

    n_heads = model.config.num_attention_heads
    arms = {
        "identified":  heads_map(top),
        "random":      heads_map([(l, rng.randrange(n_heads)) for l, _ in top]),
        "lowest":      heads_map(bot),
        "mean_ablate": heads_map(top),   # donor=None -> mean bank
    }

    rows = []
    for arm, hmap in arms.items():
        for mismatch in (False, True):     # (d) answer-copying control
            pairs = build_pairs(fail, succ, mismatch_only=mismatch, rng=rng)
            flips_target = flips_donor = n = 0
            for f, d in pairs:
                f_text = jp.build_text(tok, f.to_dict(), cfg)
                d_text = jp.build_text(tok, d.to_dict(), cfg)
                f_ids = torch.tensor([tok(f_text, add_special_tokens=False)["input_ids"]],
                                     device=next(model.parameters()).device)
                d_ids = torch.tensor([tok(d_text, add_special_tokens=False)["input_ids"]],
                                     device=f_ids.device)
                pos_f, pos_d = f_ids.shape[1] - 1, d_ids.shape[1] - 1
                donor_acts = (None if arm == "mean_ablate"
                              else capture_head_inputs(model, d_ids, list(hmap), pos_d))
                with HeadPatch(model, hmap, donor_acts, pos_f) as _p, torch.no_grad():
                    out = model(f_ids).logits[0, -1, :]
                pred = tok.decode(int(out.argmax())).strip()
                flips_target += int(pred.startswith(f["correct_state"][:6]))
                flips_donor += int(pred.startswith(d["correct_state"][:6]))
                n += 1
            rows.append(dict(arm=arm, donor_mismatch=mismatch, n=n,
                             flip_to_target=flips_target / max(n, 1),
                             flip_to_donor=flips_donor / max(n, 1)))
            print(f"[{arm:12s} mismatch={mismatch}] n={n} "
                  f"flip->target={rows[-1]['flip_to_target']:.3f} "
                  f"flip->donor={rows[-1]['flip_to_donor']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "patch_results.csv"), index=False)
    print(f"\nsaved {a.out}/patch_results.csv")
    print("\nINTERPRETATION GUARD: the identified-heads arm is evidence only if its "
          "flip->target exceeds all three control arms AND flip->donor stays low in "
          "the donor-mismatch condition. A high flip->donor means the patch is "
          "copying the donor's answer, not localizing a circuit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
