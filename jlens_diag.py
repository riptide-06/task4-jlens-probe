#!/usr/bin/env python3
"""
jlens_diag.py — READ-ONLY diagnostics for the --self-check max|Delta logit| discrepancy.

Changes nothing in jlens_probe.py. Imports it so every measurement goes through the
EXACT code paths under investigation (TruncatedStack, ResidualCapture, _legacy_kv...).

Usage (from the repo root, next to jlens_probe.py):

    # the failing case
    python jlens_diag.py --load-in bf16

    # the NaN case
    python jlens_diag.py --load-in fp16

    # the dtype ladder — decisive, and small enough for a T4 or even CPU
    python jlens_diag.py --model-id Qwen/Qwen3-0.6B --load-in fp32
    python jlens_diag.py --model-id Qwen/Qwen3-0.6B --load-in bf16
    python jlens_diag.py --model-id Qwen/Qwen3-0.6B --load-in fp16

Paste the whole stdout back. It is designed to be compact.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens_probe as jp  # noqa: E402


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def finite_report(name: str, x: torch.Tensor) -> bool:
    """Print a NaN/Inf summary for one tensor. Returns True if all-finite."""
    x = x.detach()
    n = x.numel()
    n_nan = int(torch.isnan(x).sum())
    n_inf = int(torch.isinf(x).sum())
    ok = (n_nan == 0 and n_inf == 0)
    if ok:
        amax = x.abs().max().item()
        print(f"    {name:<28} finite  n={n:<9} max|.|={amax:.4g}")
    else:
        finite_vals = x[torch.isfinite(x)]
        amax = finite_vals.abs().max().item() if finite_vals.numel() else float("nan")
        print(f"    {name:<28} BAD     n={n:<9} nan={n_nan} inf={n_inf} "
              f"max|finite|={amax:.4g}")
    return ok


def compare(real: torch.Tensor, replay: torch.Tensor, topk: int = 50):
    """Scale-aware comparison of two logit vectors. Both are 1-D float32."""
    d = (real - replay).abs()
    out = {}
    out["max"] = d.max().item()
    out["rms"] = d.pow(2).mean().sqrt().item()
    # the metric that actually matters: error on the tokens with real probability mass
    top_idx = torch.topk(real, k=topk).indices
    out["max_top"] = d[top_idx].max().item()
    out["argmax_agree"] = int(real.argmax()) == int(replay.argmax())
    try:
        out["corr"] = torch.corrcoef(torch.stack([real, replay]))[0, 1].item()
    except Exception:
        out["corr"] = float("nan")
    # decision-relevant divergence
    lp_r = torch.log_softmax(real, -1)
    lp_p = torch.log_softmax(replay, -1)
    out["kl"] = float((lp_r.exp() * (lp_r - lp_p)).sum())
    # scale-free: max error relative to the spread of the real logits
    out["rel"] = out["max"] / max(real.std().item(), 1e-9)
    return out


def hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# model loading (adds an fp32 path that jp.load_model does not expose)
# -----------------------------------------------------------------------------

def load(cfg, load_in: str):
    if load_in != "fp32":
        return jp.load_model(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float32,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_impl,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True
    return model, tok, "fp32", False


# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--load-in", default="bf16",
                   choices=["fp32", "bf16", "fp16", "8bit", "4bit", "auto"])
    p.add_argument("--jsonl", default="data/sample_probe_input.jsonl")
    p.add_argument("--record-index", type=int, default=0)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--skip-eager", action="store_true",
                   help="skip D3's eager pass (it materialises [1,H,T,T])")
    a = p.parse_args()

    cfg = jp.ProbeConfig(model_id=a.model_id, load_in=a.load_in,
                         jsonl=a.jsonl, device_map=a.device_map)

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    try:
        import transformers
        print(f"transformers={transformers.__version__}")
    except Exception:
        pass
    print(f"model={a.model_id}  load_in={a.load_in}")

    records = jp.load_records(cfg)
    record = records[a.record_index]

    model, tokenizer, precision, is_quant = load(cfg, a.load_in)
    stack = jp.TruncatedStack(model)
    n_layers = stack.n_layers
    param_dtype = next(model.parameters()).dtype
    print(f"resolved precision={precision}  param dtype={param_dtype}  layers={n_layers}")
    print(f"cache kwarg={stack._cache_kw}  "
          f"attn_impl={getattr(model.config, '_attn_implementation', '?')}")

    device = next(model.parameters()).device
    text = jp.build_text(tokenizer, record, cfg)
    ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]],
                       device=device)
    t = ids.shape[1] - 1
    print(f"prompt tokens={ids.shape[1]}  readout position t={t}")

    # ---------------- the real forward, exactly as run_self_check does ----------
    with jp.ResidualCapture(model, t) as cap, torch.no_grad():
        out = jp.forward_with_cache(model, ids)
    real = out.logits[0, -1, :].float()
    kv = jp._legacy_kv(out.past_key_values)
    kv_prefix = [(k[:, :, :t, :], v[:, :, :t, :]) for (k, v) in kv]
    resid = cap.resid
    del out

    # =========================================================================
    hdr("D5  NaN / overflow localisation  (which side is bad?)")
    # =========================================================================
    print("  real forward:")
    real_ok = finite_report("logits[0,-1,:]", real)
    kv_ok = True
    for i in (0, n_layers // 2, n_layers - 1):
        kv_ok &= finite_report(f"kv[{i}].keys", kv_prefix[i][0].float())
        kv_ok &= finite_report(f"kv[{i}].values", kv_prefix[i][1].float())

    print("  residual stream, max|activation| per depth "
          f"(fp16 overflows above 65504, bf16 above ~3.4e38):")
    first_bad = None
    for d in sorted(resid):
        x = resid[d].float()
        bad = bool(torch.isnan(x).any() or torch.isinf(x).any())
        if bad and first_bad is None:
            first_bad = d
        if d % max(1, n_layers // 12) == 0 or bad or d == n_layers:
            flag = "  <-- NON-FINITE" if bad else ""
            fin = x[torch.isfinite(x)]
            amax = fin.abs().max().item() if fin.numel() else float("nan")
            print(f"    resid[{d:>3}]  max|.|={amax:>12.4g}{flag}")
    if first_bad is not None:
        print(f"  ==> residual stream first goes non-finite at depth {first_bad} "
              f"IN THE REAL FORWARD -> TruncatedStack is not the cause.")

    with torch.no_grad():
        replay0 = stack.forward_from(0, resid[0].unsqueeze(1), kv_prefix, t)[0].float()
    print("  truncated-stack replay from layer 0:")
    replay_ok = finite_report("replay logits", replay0)

    print(f"\n  VERDICT: real finite={real_ok}  kv finite={kv_ok}  replay finite={replay_ok}")
    if not real_ok:
        print("  -> the NaN originates in the MODEL's own forward pass, before any replay.")
    elif not replay_ok:
        print("  -> real is clean but the replay is not: a genuine TruncatedStack bug.")

    if not (real_ok and replay_ok):
        print("\n  logit-space diagnostics below are meaningless with non-finite values; "
              "stopping here. Re-run this script in bf16 or fp32.")
        return 0

    # =========================================================================
    hdr("D3  noise floor: two computations that are BOTH correct")
    # =========================================================================
    with torch.no_grad():
        again = jp.forward_with_cache(model, ids).logits[0, -1, :].float()
    det = (real - again).abs().max().item()
    print(f"  same forward twice (determinism)      max|d| = {det:.6f}"
          f"   {'(bitwise identical)' if det == 0 else '(NON-DETERMINISTIC!)'}")

    with torch.no_grad():
        r2 = stack.forward_from(0, resid[0].unsqueeze(1), kv_prefix, t)[0].float()
    print(f"  same replay twice (determinism)       max|d| = "
          f"{(replay0 - r2).abs().max().item():.6f}")

    if not a.skip_eager and ids.shape[1] <= 2048:
        try:
            with jp.eager_attention(model), torch.no_grad():
                eager = jp.forward_with_cache(model, ids).logits[0, -1, :].float()
            c = compare(real, eager)
            print(f"  sdpa vs eager, SAME full forward      max|d| = {c['max']:.6f}  "
                  f"rms={c['rms']:.6f}  corr={c['corr']:.6f}  argmax_agree={c['argmax_agree']}")
            print("  ^^ this is the irreducible kernel/reduction-order disagreement between "
                  "two\n     provably-correct computations at this dtype. Any self-check "
                  "threshold\n     BELOW this number is guaranteed to produce false alarms.")
        except Exception as exc:
            print(f"  sdpa vs eager: skipped ({type(exc).__name__}: {exc})")
    else:
        print("  sdpa vs eager: skipped (--skip-eager or prompt too long)")

    # =========================================================================
    hdr("D1  residual-stream divergence per depth  (replay from layer 0)")
    # =========================================================================
    print("  A smooth ramp => accumulated precision error.")
    print("  A step change at one depth => a bug in THAT layer's replay.")
    print("  Large from the very first layer => cache slicing / RoPE / kwargs bug.\n")
    with jp.ResidualCapture(model, 0) as rcap, torch.no_grad():
        stack.forward_from(0, resid[0].unsqueeze(1), kv_prefix, t)

    print(f"    {'depth':>5} {'||real||':>12} {'max|d|':>12} {'rel_l2':>10} {'jump':>8}")
    prev_rel = None
    worst_jump = (0.0, -1)
    for d in sorted(rcap.resid):
        if d not in resid:
            continue
        rl = resid[d].float().flatten()
        rp = rcap.resid[d].float().flatten()
        if rl.shape != rp.shape:
            continue
        rel = ((rp - rl).norm() / rl.norm().clamp_min(1e-9)).item()
        jump = (rel / prev_rel) if (prev_rel and prev_rel > 0) else float("nan")
        if prev_rel is not None and jump == jump and jump > worst_jump[0]:
            worst_jump = (jump, d)
        print(f"    {d:>5} {rl.norm().item():>12.4g} {(rp - rl).abs().max().item():>12.4g} "
              f"{rel:>10.3e} {jump:>8.2f}x")
        prev_rel = rel
    print(f"\n  largest single-depth amplification: {worst_jump[0]:.2f}x entering depth "
          f"{worst_jump[1]}")
    print("  (steady 1.0-1.5x per layer = precision.  A lone >10x = the bug's layer.)")

    # =========================================================================
    hdr("D2  max|Delta logit| vs replay start layer L")
    # =========================================================================
    print("  L = n_layers  -> zero decoder layers, only norm+lm_head: the CONTROL, expect ~0")
    print("  L = n_layers-1-> ONE layer replayed: if this is already large, the bug is in")
    print("                   cache slicing / RoPE / kwargs, which every layer shares.\n")
    print(f"    {'L':>4} {'max|d|':>10} {'max|d|top50':>12} {'rms':>10} {'rel':>9} "
          f"{'corr':>9} {'KL':>10} {'argmax':>7}")
    rows = []
    for L in range(n_layers, -1, -1):
        if L not in resid:
            continue
        with torch.no_grad():
            rp = stack.forward_from(L, resid[L].unsqueeze(1), kv_prefix, t)[0].float()
        c = compare(real, rp)
        rows.append((L, c))
        print(f"    {L:>4} {c['max']:>10.4f} {c['max_top']:>12.4f} {c['rms']:>10.4f} "
              f"{c['rel']:>9.4f} {c['corr']:>9.6f} {c['kl']:>10.3e} "
              f"{str(c['argmax_agree']):>7}")

    # =========================================================================
    hdr("D4  where does max|Delta| actually live?  (replay from layer 0)")
    # =========================================================================
    d = (real - replay0).abs()
    j = int(d.argmax())
    rank = int((real > real[j]).sum()) + 1
    print(f"  argmax|d| token id = {j}  ({tokenizer.convert_ids_to_tokens([j])[0]!r})")
    print(f"    real   logit = {real[j].item():.4f}")
    print(f"    replay logit = {replay0[j].item():.4f}")
    print(f"    |delta|      = {d[j].item():.4f}")
    print(f"    rank of this token in the real distribution = {rank} / {real.numel()}")
    print(f"    softmax prob of this token = {torch.softmax(real, -1)[j].item():.3e}")
    if rank > 100:
        print("  ==> the reported max|Delta logit| is attained on a token nobody reads. "
              "As a\n      gate metric it is measuring noise in the tail, not fidelity.")
    print(f"\n  for comparison, on the real top-10 tokens:")
    for r, tid in enumerate(torch.topk(real, 10).indices.tolist(), 1):
        print(f"    rank {r:>2} id={tid:<8} {tokenizer.convert_ids_to_tokens([tid])[0]!r:<16} "
              f"real={real[tid].item():>9.4f} replay={replay0[tid].item():>9.4f} "
              f"d={d[tid].item():>7.4f}")

    hdr("SUMMARY (paste this back)")
    c0 = compare(real, replay0)
    print(f"  model={a.model_id} dtype={param_dtype} layers={n_layers} t={t}")
    print(f"  replay-from-0:  max={c0['max']:.4f}  max_top50={c0['max_top']:.4f}  "
          f"rms={c0['rms']:.4f}  rel={c0['rel']:.4f}  corr={c0['corr']:.6f}  "
          f"KL={c0['kl']:.3e}  argmax_agree={c0['argmax_agree']}")
    if rows:
        L1 = [c for (L, c) in rows if L == n_layers - 1]
        Lc = [c for (L, c) in rows if L == n_layers]
        if Lc:
            print(f"  control (L={n_layers}, zero layers):   max={Lc[0]['max']:.6f}")
        if L1:
            print(f"  one layer (L={n_layers - 1}):            max={L1[0]['max']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
