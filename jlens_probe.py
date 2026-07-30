#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jlens_probe.py — Mechanistic-interpretability probe for STALE-STATE tracking in Qwen3-8B.

Team tommy-hjra · Algoverse AI Research · Axis 2 (State Tracking & Belief Update)

--------------------------------------------------------------------------------
RESEARCH QUESTION
--------------------------------------------------------------------------------
When a fact is overwritten mid-context ("the key is in the Greenhouse" ... later ...
"you moved it to the Laboratory"), and the model *answers with the stale room*, did it
nonetheless compute the correct updated room somewhere inside?

    representation failure  ->  correct state never appears in any layer
    retrieval   failure     ->  correct state is decodable mid-stack, then loses at the top

This script measures exactly that, per layer, per example.

--------------------------------------------------------------------------------
WHAT THE TWO LENSES ARE
--------------------------------------------------------------------------------
Let  resid[i]  = residual stream entering decoder layer i at the readout position t
     (resid[0] = token embeddings, resid[L] = output of the last layer),
and  f_i(.)    = "run decoder layers i..L-1, then the final RMSNorm, then lm_head",
     i.e. the model's own remaining computation, evaluated at position t with the real
     (frozen) KV cache of positions < t.

1) LOGIT LENS (baseline, Nostalgebraist 2020)
       logitlens_i = lm_head(final_norm(resid[i]))
   Cheap, but it *skips every remaining layer* — it asks "what if layers i..L were the
   identity?" That systematically under-reads mid-stack states whose meaning is only
   realised by later attention/MLP reads.

2) J-LENS / JACOBIAN LENS (this file's main instrument)
   Decode resid[i] *through the model's own downstream map*, linearised at the real
   activation. With J = d f_i / d h evaluated at h = resid[i] and a baseline b:

       jlens_i = f_i(b) + J · (resid[i] - b)          # default b = 0

   This is a first-order Taylor readout of the actual remaining network, so it keeps the
   contribution of every later layer instead of throwing it away. Output is in *logit
   units*, directly comparable to the logit lens (and softmax-able).
   With b = 0 this is the familiar "gradient x input + baseline" attribution, evaluated
   in the network's true operating point.

   J is never materialised (it would be 151936 x 4096). We only ever need the
   Jacobian-VECTOR product J·v, obtained by either:
     * mode "autograd" — exact, via the double-backward (VJP-of-VJP) identity
           g(u) = J^T u  ->  d/du [v . g(u)] = J v
       one reverse pass builds g, a second differentiates it. Exact to machine precision.
     * mode "fd"       — central finite differences, [f(h+eps*v) - f(h-eps*v)] / (2 eps).
       Slightly noisier but works through bitsandbytes 4-bit/8-bit layers, whose custom
       autograd Functions do not always support double backward.
   Default is "auto": autograd when the model is unquantised, fd when it is quantised,
   with an automatic fallback to fd if the double backward raises.

NOTE ON NAMING: "J-Lens" has no canonical public implementation, so the projection is
implemented here from its definition above and is swappable — see `LENS_REGISTRY` at the
bottom if the team standardises on a different formulation.

--------------------------------------------------------------------------------
HOW THE HOOKS WORK (the part you asked to have explained)
--------------------------------------------------------------------------------
Three separate capture mechanisms, all torch hooks, all removed in a `finally`:

  A) RESIDUAL STREAM  — `register_forward_pre_hook` on every `model.model.layers[i]`.
     A *pre*-hook fires with the layer's INPUT, which is precisely resid[i]. That gives us
     the residual stream at every depth with no `output_hidden_states=True` (which would
     materialise L x T x 4096 floats for the whole sequence; we only want column t).
     The hook slices `hidden[:, t, :]` immediately and clones it to fp32 on CPU/GPU, so
     memory is O(L x 4096) instead of O(L x T x 4096). One extra *post*-hook on the last
     layer supplies resid[L].

  B) ATTENTION SCORES — `register_forward_hook` on every `model.model.layers[i].self_attn`.
     Qwen3's attention returns `(attn_output, attn_weights)`; the hook grabs element [1].
     Weights only exist under the *eager* attention kernel (sdpa/flash never build the
     matrix), so the attention pass runs inside `eager_attention(model)`, which flips
     `config._attn_implementation` for that pass only.
     Crucially the attention pass is a SINGLE-TOKEN forward (query length 1) replayed on
     top of the frozen prefix KV cache: the captured tensor is [1, H, 1, t+1], ~19 MB for
     an 8k context, instead of the [1, H, T, T] blow-up (hundreds of GB) that
     `output_attentions=True` over a full long prompt would cost.

  C) TRUNCATED STACK  — not a hook but the same idea in reverse: `TruncatedStack` calls
     `layers[i:]` by hand on a supplied hidden state, reusing the frozen prefix cache. That
     is what makes both J-Lens modes cheap: each probe is one token through <= L layers.
     `--self-check` validates the whole apparatus by replaying from layer 0 and asserting
     it reproduces the model's real logits.

--------------------------------------------------------------------------------
USAGE (Colab)
--------------------------------------------------------------------------------
    !pip -q install "transformers>=4.51.0" accelerate bitsandbytes pandas

    !python jlens_probe.py \
        --jsonl data/sample_probe_input.jsonl \
        --out-dir outputs/run1 \
        --limit 50 --load-in 4bit --self-check

Outputs (see README.md for the full column dictionary):
    outputs/run1/layers.csv      one row per (example, layer)  <- main analysis table
    outputs/run1/examples.csv    one row per example (behaviour + summary stats)
    outputs/run1/heads.csv       one row per (example, layer, head)   [--dump-heads]
    outputs/run1/records.jsonl   everything, incl. per-layer top-k vocabulary
    outputs/run1/config.json     exact configuration used
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# torch/transformers are imported lazily inside load_model() so that --dry-run works on a
# machine with only the tokenizer stack installed.
import torch


# =============================================================================
# 1. CONFIG
# =============================================================================

@dataclass
class ProbeConfig:
    # --- data -----------------------------------------------------------------
    jsonl: str = "data/sample_probe_input.jsonl"
    out_dir: str = "outputs/run"
    limit: Optional[int] = None
    shuffle: bool = False
    seed: int = 0
    filter_compression: Optional[List[str]] = None
    filter_entity_type: Optional[List[str]] = None
    filter_level: Optional[List[int]] = None

    # --- model ----------------------------------------------------------------
    model_id: str = "Qwen/Qwen3-8B"
    load_in: str = "auto"           # auto | 4bit | 8bit | bf16 | fp16
    device_map: str = "auto"
    attn_impl: str = "sdpa"         # kernel for the main pass (eager is forced for attn capture)
    max_prompt_tokens: int = 8192

    # --- prompt / readout position -------------------------------------------
    prompt_format: str = "chat"     # chat | raw
    enable_thinking: bool = False
    answer_template: str = "{subject} is currently in the"
    raw_answer_lead: str = "\n\nAnswer: "
    target_mode: str = "prefill"    # prefill | generate
    max_new_tokens: int = 24

    # --- lenses ---------------------------------------------------------------
    jlens_mode: str = "auto"        # auto | autograd | fd | off
    jlens_baseline: str = "zero"    # zero | prev_layer | none
    jlens_eps: float = 0.05         # relative step for the fd Jacobian-vector product
    layers: Optional[List[int]] = None   # None = all residual depths 0..L
    top_k: int = 10

    # --- attention ------------------------------------------------------------
    capture_attention: bool = True
    dump_heads: bool = False

    # --- misc -----------------------------------------------------------------
    self_check: bool = False
    dry_run: bool = False
    verbose: bool = True


# =============================================================================
# 2. DATA
# =============================================================================

# Fields written by stale-state-data-generator/staleness/schema.py::Record
REQUIRED_FIELDS = ("prompt", "correct_state", "stale_state")
META_FIELDS = (
    "complexity_level", "entity_type", "entity",
    "compression", "distractor_position", "distractor_role", "n_distractors",
)


def load_records(cfg: ProbeConfig) -> List[Dict[str, Any]]:
    """Read the generator's .jsonl, validate the invariants we depend on, apply filters."""
    path = Path(cfg.jsonl)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path.resolve()}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON — {exc}") from exc

            missing = [f for f in REQUIRED_FIELDS if not row.get(f)]
            if missing:
                raise ValueError(f"{path}:{lineno} missing required field(s): {missing}")

            # The generator guarantees these; re-assert because the whole probe rests on them.
            if row["correct_state"] == row["stale_state"]:
                raise ValueError(f"{path}:{lineno} correct_state == stale_state")
            for f in ("correct_state", "stale_state"):
                if row[f] not in row["prompt"]:
                    raise ValueError(f"{path}:{lineno} {f}={row[f]!r} not present verbatim in prompt")

            row.setdefault("_dataset_index", lineno - 1)
            for f in META_FIELDS:
                row.setdefault(f, None)
            rows.append(row)

    def keep(r: Dict[str, Any]) -> bool:
        if cfg.filter_compression and r.get("compression") not in cfg.filter_compression:
            return False
        if cfg.filter_entity_type and r.get("entity_type") not in cfg.filter_entity_type:
            return False
        if cfg.filter_level and r.get("complexity_level") not in cfg.filter_level:
            return False
        return True

    rows = [r for r in rows if keep(r)]
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(rows)
    if cfg.limit is not None:
        rows = rows[: cfg.limit]
    return rows


# =============================================================================
# 3. PROMPT CONSTRUCTION  +  CANDIDATE TOKENS
# =============================================================================

def subject_phrase(record: Dict[str, Any]) -> str:
    """'the rusty key' -> 'The rusty key';  'Drone R-7' -> 'Drone R-7'.

    The generator stores objects as bare common nouns (article added by the template) and
    agents as article-free proper nouns, so the answer prefill has to branch here.
    """
    entity = (record.get("entity") or "").strip()
    if not entity:
        return "It"
    if (record.get("entity_type") or "object") == "agent":
        return entity
    return "The " + entity


def build_text(tokenizer, record: Dict[str, Any], cfg: ProbeConfig) -> str:
    """Full string fed to the model, ending exactly where the room name must be predicted."""
    prefill = cfg.answer_template.format(subject=subject_phrase(record))

    if cfg.prompt_format == "raw":
        return record["prompt"] + cfg.raw_answer_lead + prefill

    messages = [{"role": "user", "content": record["prompt"]}]
    try:
        head = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
    except TypeError:
        # Older/newer templates without the Qwen3 `enable_thinking` switch.
        head = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return head + prefill


def first_continuation_token(tokenizer, context: str, continuation: str) -> Tuple[int, str, bool]:
    """Token id the model must emit *next* for `continuation` to begin.

    Tokenising " Laboratory" in isolation is not the same as tokenising it in context (BPE
    merges across the boundary), so we tokenise context and context+continuation and read
    the first divergent position. Returns (token_id, token_string, alignment_ok).
    """
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    aligned = len(full_ids) > len(ctx_ids) and full_ids[: len(ctx_ids)] == ctx_ids
    if aligned:
        tid = full_ids[len(ctx_ids)]
    else:
        # Re-tokenisation shifted an earlier token; fall back to the standalone encoding.
        tid = tokenizer(continuation, add_special_tokens=False)["input_ids"][0]
    return tid, tokenizer.convert_ids_to_tokens(tid), aligned


def char_spans(text: str, needle: str) -> List[Tuple[int, int]]:
    """All verbatim occurrences of `needle` in `text`, as (start, end) char offsets."""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def token_indices_for_spans(offsets: Sequence[Tuple[int, int]],
                            spans: Sequence[Tuple[int, int]],
                            upto: int) -> List[List[int]]:
    """Map char spans -> lists of token indices (one list per occurrence), clipped at `upto`."""
    out: List[List[int]] = []
    for (s, e) in spans:
        idx = [i for i, (a, b) in enumerate(offsets)
               if i <= upto and b > a and a < e and b > s]
        if idx:
            out.append(idx)
    return out


# =============================================================================
# 4. MODEL LOADING
# =============================================================================

def get_base(model):
    """The decoder trunk that owns `.layers` / `.norm` / `.rotary_emb`.

    Qwen3ForCausalLM -> model.model. Some transformers releases nest text decoders one
    level deeper for multimodal checkpoints, so probe for the attribute rather than
    hard-coding the path.
    """
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base
    inner = getattr(base, "language_model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def _gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def resolve_precision(cfg: ProbeConfig) -> str:
    """auto -> bf16 on a big Ampere+ card, else 4bit (Qwen3-8B is ~16 GB in bf16, ~5.5 GB in NF4)."""
    if cfg.load_in != "auto":
        return cfg.load_in
    total = _gpu_total_gb()
    if total >= 40 and _supports_bf16():
        return "bf16"          # A100 / H100
    if total >= 24 and _supports_bf16():
        return "8bit"          # L4 / A10
    return "4bit"              # T4 (no bf16), or anything small


def load_model(cfg: ProbeConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = resolve_precision(cfg)
    compute_dtype = torch.bfloat16 if _supports_bf16() else torch.float16

    kwargs: Dict[str, Any] = dict(
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_impl,
        low_cpu_mem_usage=True,
    )

    if precision in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if precision == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # transformers keeps lm_head out of the quantised set, so the unembedding used by
        # both lenses stays a plain fp16/bf16 Linear — logit lens remains exact.
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if precision == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "is_fast", False):
        print("[warn] slow tokenizer: no character offsets, so room-mention token spans "
              "and all attention_* columns will be NaN. Install `tokenizers` for the fast one.")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    model.config.use_cache = True

    is_quantized = precision in ("4bit", "8bit")
    if cfg.verbose:
        print(f"[model] {cfg.model_id} | precision={precision} | "
              f"loaded in {time.time() - t0:.1f}s | layers={model.config.num_hidden_layers} | "
              f"d_model={model.config.hidden_size} | vocab={model.config.vocab_size}")
        if torch.cuda.is_available():
            print(f"[model] VRAM allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GB "
                  f"of {_gpu_total_gb():.1f} GB")
    return model, tokenizer, precision, is_quantized


@contextlib.contextmanager
def eager_attention(model):
    """Temporarily switch every attention module to the eager kernel.

    sdpa / flash-attention fuse the softmax and never build the [.., q, k] score matrix, so
    they return `attn_weights = None` and there is nothing for a hook to grab. Qwen3's
    attention reads `self.config._attn_implementation` at *call* time, so flipping the
    shared config object is enough — no reloading, no re-instantiation.
    """
    touched = []
    seen = set()
    for module in [model] + list(model.modules()):
        conf = getattr(module, "config", None)
        if conf is None or id(conf) in seen:
            continue
        if hasattr(conf, "_attn_implementation"):
            seen.add(id(conf))
            touched.append((conf, conf._attn_implementation))
    try:
        for conf, _ in touched:
            try:
                conf._attn_implementation = "eager"
            except Exception:  # some versions guard the setter
                pass
        yield
    finally:
        for conf, old in touched:
            try:
                conf._attn_implementation = old
            except Exception:
                pass


# =============================================================================
# 5. HOOKS
# =============================================================================

class ResidualCapture:
    """(A) Residual stream at ONE position, every depth, via forward PRE-hooks.

    A pre-hook on `layers[i]` sees that layer's input == resid[i]; a post-hook on the last
    layer supplies resid[L]. We slice column `t` inside the hook, so nothing bigger than
    [1, d_model] is ever retained — this is what keeps long contexts affordable.
    """

    def __init__(self, model, position: int):
        self.layers = get_base(model).layers
        self.position = position
        self.resid: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def _grab(self, tensor: torch.Tensor, depth: int) -> None:
        if tensor.dim() != 3:
            return
        self.resid[depth] = tensor[:, self.position, :].detach().clone()

    def __enter__(self) -> "ResidualCapture":
        for i, layer in enumerate(self.layers):
            def pre_hook(_mod, args, kwargs, _i=i):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is not None:
                    self._grab(hidden, _i)
                return None
            self._handles.append(
                layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

        def post_hook(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._grab(hidden, len(self.layers))
            return None
        self._handles.append(self.layers[-1].register_forward_hook(post_hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class AttentionCapture:
    """(B) Attention probabilities for the readout query row, every layer.

    Hooks each `self_attn` module and keeps `output[1]` — Qwen3 attention returns
    `(attn_output, attn_weights)`. Only populated under the eager kernel, and only worth
    doing on a query-length-1 forward (shape [1, H, 1, kv_len]).
    """

    def __init__(self, model):
        self.layers = get_base(model).layers
        self.attn: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def __enter__(self) -> "AttentionCapture":
        for i, layer in enumerate(self.layers):
            def hook(_mod, _args, output, _i=i):
                if isinstance(output, (tuple, list)) and len(output) > 1:
                    w = output[1]
                    if isinstance(w, torch.Tensor) and w.dim() == 4:
                        # [batch, heads, q_len, kv_len] -> keep the last query row only
                        self.attn[_i] = w[0, :, -1, :].detach().float().cpu()
                return None
            self._handles.append(layer.self_attn.register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# =============================================================================
# 6. TRUNCATED STACK  (the machinery both J-Lens modes ride on)
# =============================================================================

def _legacy_kv(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pull (key, value) tensors out of whatever Cache class this transformers version uses."""
    if cache is None:
        raise RuntimeError("model returned no past_key_values; is use_cache disabled?")
    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            if legacy:
                return [(k, v) for (k, v) in legacy]
        except Exception:
            pass
    if hasattr(cache, "layers"):                      # transformers >= ~4.54
        try:
            return [(l.keys, l.values) for l in cache.layers]
        except Exception:
            pass
    if hasattr(cache, "key_cache"):                   # older DynamicCache
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):              # legacy tuple format
        return [(k, v) for (k, v) in cache]
    raise RuntimeError(f"unsupported cache type {type(cache)!r}; pin transformers>=4.51")


def _build_cache(kvs: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    """Fresh DynamicCache holding `kvs`. Inputs may be views — Cache.update() cats, never
    writes in place, so the originals are safe to reuse across probes."""
    from transformers.cache_utils import DynamicCache
    try:
        return DynamicCache.from_legacy_cache(tuple(kvs))
    except Exception:
        cache = DynamicCache()
        for i, (k, v) in enumerate(kvs):
            cache.update(k, v, i)
        return cache


class TruncatedStack:
    """Run `layers[i:] -> final_norm -> lm_head` on a supplied hidden state.

    Only ONE token is pushed through (the readout position t), re-using the frozen KV cache
    of positions < t. That is exact: with causal attention, perturbing position t cannot
    affect any position < t, so their keys/values are unchanged by construction.

    Query length is 1, so `attention_mask=None` is correct for every kernel (a single query
    at the end of the sequence legitimately attends to all cached keys) — no causal-mask
    plumbing needed.
    """

    def __init__(self, model):
        self.model = model
        self.base = get_base(model)
        self.layers = self.base.layers
        self.norm = self.base.norm
        self.lm_head = model.lm_head
        self.n_layers = len(self.layers)

        params = inspect.signature(self.layers[0].forward).parameters
        self._params = params
        self._has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        # transformers renamed this between releases; pick whichever the layer declares.
        self._cache_kw = "past_key_values" if "past_key_values" in params else (
            "past_key_value" if "past_key_value" in params else "past_key_values")

    def _accepts(self, name: str) -> bool:
        return name in self._params or self._has_var_kw

    @staticmethod
    def _dev(module) -> torch.device:
        """Device a submodule lives on — works with a sharded `device_map="auto"` layout."""
        for t in module.parameters():
            return t.device
        for t in module.buffers():          # parameter-free modules (e.g. rotary embeddings)
            return t.device
        return torch.device("cpu")

    def rotary(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        """(cos, sin) for the readout position — the same tensors Qwen3Model would build."""
        return self.base.rotary_emb(hidden, position_ids)

    def forward_from(self, layer_idx: int, hidden: torch.Tensor, kvs, position: int) -> torch.Tensor:
        """hidden: [1, 1, d_model] entering layer `layer_idx`. Returns logits [1, vocab]."""
        cache = _build_cache(kvs)
        dev0 = self._dev(self.layers[min(layer_idx, self.n_layers - 1)])
        position_ids = torch.tensor([[position]], device=dev0)
        cache_position = torch.tensor([position], device=dev0)
        cos, sin = self.rotary(hidden.to(dev0), position_ids)

        h = hidden
        for i in range(layer_idx, self.n_layers):
            layer = self.layers[i]
            dev = self._dev(layer)
            h = h.to(dev)
            kw: Dict[str, Any] = {}
            if self._accepts("attention_mask"):
                kw["attention_mask"] = None
            if self._accepts("position_ids"):
                kw["position_ids"] = position_ids.to(dev)
            if self._accepts("cache_position"):
                kw["cache_position"] = cache_position.to(dev)
            if self._accepts("position_embeddings"):
                kw["position_embeddings"] = (cos.to(dev), sin.to(dev))
            if self._accepts("use_cache"):
                kw["use_cache"] = True
            kw[self._cache_kw] = cache
            out = layer(h, **kw)
            h = out[0] if isinstance(out, (tuple, list)) else out

        h = self.norm(h.to(self._dev(self.norm)))
        return self.lm_head(h.to(self._dev(self.lm_head)))[:, -1, :]


# =============================================================================
# 7. THE LENSES
# =============================================================================

def forward_with_cache(model, ids: torch.Tensor):
    """One full forward that returns the KV cache but only the LAST position's logits.

    `logits_to_keep=1` (renamed from `num_logits_to_keep` in transformers 4.53) keeps the
    [1, T, 151936] logit tensor from being materialised — 2.5 GB at T=8k in fp16.

    The name is resolved by inspecting the signature rather than by try/except TypeError:
    many forwards accept **kwargs (so a wrong name is silently ignored), and catching
    TypeError would also swallow genuine errors raised from inside the model.
    """
    params = inspect.signature(model.forward).parameters
    extra = {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            extra[name] = 1
            break
    return model(ids, use_cache=True, **extra)


def logit_lens(stack: TruncatedStack, resid: torch.Tensor) -> torch.Tensor:
    """lm_head(final_norm(h)) — the classic lens, i.e. "skip all remaining layers"."""
    with torch.no_grad():
        h = resid.to(stack._dev(stack.norm))
        return stack.lm_head(stack.norm(h).to(stack._dev(stack.lm_head))).float()


def jvp_finite_difference(stack: TruncatedStack, layer_idx: int, h: torch.Tensor,
                          v: torch.Tensor, kvs, position: int, eps_rel: float) -> torch.Tensor:
    """J·v by central differences. Quantisation-safe (no autograd through bnb kernels)."""
    # Norms in fp32: late-layer residual norms are large enough that ||h||^2 overflows fp16.
    v_norm = v.float().norm().clamp_min(1e-6)
    eps = float(eps_rel * h.float().norm().clamp_min(1e-6) / v_norm)
    with torch.no_grad():
        plus = stack.forward_from(layer_idx, (h + eps * v), kvs, position).float()
        minus = stack.forward_from(layer_idx, (h - eps * v), kvs, position).float()
    return (plus - minus) / (2.0 * eps)


def jvp_autograd(stack: TruncatedStack, layer_idx: int, h: torch.Tensor,
                 v: torch.Tensor, kvs, position: int) -> torch.Tensor:
    """Exact J·v via the double-backward identity  d/du [ v . (J^T u) ] = J v.

    Reverse mode only, so it needs no forward-mode AD support — but it does need the
    backward graph to be twice differentiable, which bitsandbytes kernels may not be.
    """
    h = h.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = stack.forward_from(layer_idx, h, kvs, position)
        u = torch.zeros_like(logits, requires_grad=True)
        (g,) = torch.autograd.grad(logits, h, grad_outputs=u, create_graph=True)
        (jv,) = torch.autograd.grad(g, u, grad_outputs=v.to(g.dtype), retain_graph=False)
    return jv.detach().float()


def jacobian_lens(stack: TruncatedStack, layer_idx: int, resid: torch.Tensor, kvs,
                  position: int, mode: str, baseline: str, eps_rel: float,
                  prev_resid: Optional[torch.Tensor]) -> Tuple[torch.Tensor, str]:
    """jlens = f(b) + J·(h - b), with J evaluated at the real activation h.

    Returns (logit-scale scores over the vocabulary, mode actually used).
    """
    h = resid  # [1, 1, d_model]
    if baseline == "zero":
        b = torch.zeros_like(h)
    elif baseline == "prev_layer" and prev_resid is not None:
        b = prev_resid
    else:  # "none" -> pure directional derivative, anchored at 0 logits
        b = None

    v = h if b is None else (h - b)

    used = mode
    if mode == "autograd":
        try:
            jv = jvp_autograd(stack, layer_idx, h, v, kvs, position)
        except Exception as exc:                      # bnb double-backward, OOM, ...
            print(f"    [jlens] autograd JVP failed ({type(exc).__name__}: {exc}); "
                  f"falling back to finite differences for the rest of the run")
            jv = jvp_finite_difference(stack, layer_idx, h, v, kvs, position, eps_rel)
            used = "fd"
    else:
        jv = jvp_finite_difference(stack, layer_idx, h, v, kvs, position, eps_rel)

    if b is None:
        return jv, used
    with torch.no_grad():
        anchor = stack.forward_from(layer_idx, b, kvs, position).float()
    return anchor + jv, used


# =============================================================================
# 8. SCORING HELPERS
# =============================================================================

def rank_of(scores: torch.Tensor, token_id: int) -> int:
    """1-based rank of `token_id` (1 = argmax)."""
    return int((scores > scores[token_id]).sum().item()) + 1


def readout(scores: torch.Tensor, correct_id: int, stale_id: int, tokenizer,
            top_k: int) -> Dict[str, Any]:
    """Everything we log about one vocabulary-space vector."""
    probs = torch.softmax(scores, dim=-1)
    top = torch.topk(scores, k=top_k)
    return {
        "score_correct": float(scores[correct_id]),
        "score_stale": float(scores[stale_id]),
        "margin": float(scores[correct_id] - scores[stale_id]),
        "p_correct": float(probs[correct_id]),
        "p_stale": float(probs[stale_id]),
        "rank_correct": rank_of(scores, correct_id),
        "rank_stale": rank_of(scores, stale_id),
        "top1_token": tokenizer.convert_ids_to_tokens(int(top.indices[0])),
        "top1_score": float(top.values[0]),
        "topk": [
            {"token": tokenizer.convert_ids_to_tokens(int(i)), "id": int(i), "score": float(s)}
            for i, s in zip(top.indices.tolist(), top.values.tolist())
        ],
    }


def attention_stats(attn_row: torch.Tensor,
                    correct_tokens: List[List[int]],
                    stale_tokens: List[List[int]]) -> Dict[str, Any]:
    """Per-layer attention mass landing on the updated vs the stale room mentions.

    attn_row: [heads, kv_len] — the readout query's distribution over the context.
    """
    flat_correct = sorted({i for occ in correct_tokens for i in occ})
    flat_stale = sorted({i for occ in stale_tokens for i in occ})
    kv_len = attn_row.shape[-1]
    flat_correct = [i for i in flat_correct if i < kv_len]
    flat_stale = [i for i in flat_stale if i < kv_len]

    mean_over_heads = attn_row.mean(dim=0)

    def mass(idx: List[int], per_head: bool = False):
        if not idx:
            return float("nan") if not per_head else torch.full((attn_row.shape[0],), float("nan"))
        sel = attn_row[:, idx].sum(dim=-1)
        return sel if per_head else float(sel.mean())

    # last mention of the correct room == the update event; first mention of the stale room
    # == the original anchor. Those two are the interesting competitors.
    last_correct = correct_tokens[-1] if correct_tokens else []
    first_stale = stale_tokens[0] if stale_tokens else []
    last_correct = [i for i in last_correct if i < kv_len]
    first_stale = [i for i in first_stale if i < kv_len]

    probs = mean_over_heads.clamp_min(1e-12)
    entropy = float(-(probs * probs.log()).sum())

    correct_per_head = mass(flat_correct, per_head=True)
    stale_per_head = mass(flat_stale, per_head=True)

    return {
        "attn_correct_mass": mass(flat_correct),
        "attn_stale_mass": mass(flat_stale),
        "attn_correct_update_mass": mass(last_correct),
        "attn_stale_anchor_mass": mass(first_stale),
        "attn_correct_minus_stale": (mass(flat_correct) - mass(flat_stale)),
        "attn_max_head_correct": float(correct_per_head.max()) if flat_correct else float("nan"),
        "attn_max_head_stale": float(stale_per_head.max()) if flat_stale else float("nan"),
        "attn_argmax_head_correct": int(correct_per_head.argmax()) if flat_correct else -1,
        "attn_entropy": entropy,
        "_per_head_correct": correct_per_head.tolist() if flat_correct else [],
        "_per_head_stale": stale_per_head.tolist() if flat_stale else [],
    }


# =============================================================================
# 9. PER-EXAMPLE PROBE
# =============================================================================

def probe_example(model, tokenizer, stack: TruncatedStack, record: Dict[str, Any],
                  cfg: ProbeConfig, jlens_mode: str) -> Tuple[Dict[str, Any], str]:
    """Run the full pipeline on one dataset row. Returns (record dict, jlens mode used)."""
    device = next(model.parameters()).device
    text = build_text(tokenizer, record, cfg)

    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except (TypeError, NotImplementedError):
        # slow (non-Rust) tokenizer: no offsets -> attention/room spans unavailable
        enc = tokenizer(text, add_special_tokens=False)
        offsets = []
    input_ids = enc["input_ids"]
    if len(input_ids) > cfg.max_prompt_tokens:
        raise ValueError(f"prompt is {len(input_ids)} tokens, over --max-prompt-tokens "
                         f"{cfg.max_prompt_tokens}")

    correct_id, correct_tok, ok_c = first_continuation_token(tokenizer, text, " " + record["correct_state"])
    stale_id, stale_tok, ok_s = first_continuation_token(tokenizer, text, " " + record["stale_state"])
    collision = correct_id == stale_id

    ids = torch.tensor([input_ids], device=device)
    generated_text = None

    # ---- optional: let the model answer freely, then read out where it names a room ----
    if cfg.target_mode == "generate":
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        full = gen[0].tolist()
        generated_text = tokenizer.decode(full[len(input_ids):], skip_special_tokens=True)
        hit = next((j for j in range(len(input_ids), len(full))
                    if full[j] in (correct_id, stale_id)), None)
        if hit is not None:
            # truncate so the readout position is the one that PREDICTED the room token.
            # `offsets`/`text` still describe the prompt prefix, which is all the room
            # mentions live in — generated tokens simply get no span attribution.
            ids = torch.tensor([full[:hit]], device=device)
        # if the model never names either room we keep the prefill position

    # INVARIANT: the readout position is always the last token of `ids`. Everything below
    # (logits_to_keep=1, cache slicing, single-token replay) relies on it.
    target = ids.shape[1] - 1

    # ---- token spans of the two competing rooms, for the attention analysis ----
    correct_tokens = token_indices_for_spans(offsets, char_spans(text, record["correct_state"]), target)
    stale_tokens = token_indices_for_spans(offsets, char_spans(text, record["stale_state"]), target)

    # =========================================================================
    # PASS A — one full forward. Pre-hooks harvest resid[0..L] at position `target`;
    #          `use_cache=True` gives us the KV cache every later probe reuses.
    #          logits_to_keep=1 avoids materialising [1, T, 151936] (2.5 GB at T=8k).
    # =========================================================================
    with ResidualCapture(model, target) as cap, torch.no_grad():
        out = forward_with_cache(model, ids)
    final_logits = out.logits[0, -1, :].float()
    resid = cap.resid
    kv_full = _legacy_kv(out.past_key_values)
    # Freeze the prefix: positions 0..target-1. Slices are views; Cache.update() cats onto
    # them without writing back, so the same views are reusable for every probe below.
    kv_prefix = [(k[:, :, :target, :], v[:, :, :target, :]) for (k, v) in kv_full]
    del out
    n_layers = stack.n_layers

    # ---- behaviour: what does the model actually say? ----
    final = readout(final_logits, correct_id, stale_id, tokenizer, cfg.top_k)
    argmax_id = int(final_logits.argmax())
    if argmax_id == correct_id:
        outcome = "correct"
    elif argmax_id == stale_id:
        outcome = "stale"
    else:
        outcome = "other"

    # =========================================================================
    # PASS B — single-token replay with eager attention, hooks grab the query row.
    # =========================================================================
    attn_rows: Dict[int, torch.Tensor] = {}
    if cfg.capture_attention:
        try:
            with eager_attention(model), AttentionCapture(model) as acap, torch.no_grad():
                stack.forward_from(0, resid[0].unsqueeze(1), kv_prefix, target)
            attn_rows = acap.attn
        except Exception as exc:
            print(f"    [attn] capture failed ({type(exc).__name__}: {exc}); "
                  f"attention columns will be NaN")

    # =========================================================================
    # PASS C — the lenses, per residual depth.
    # =========================================================================
    depths = cfg.layers if cfg.layers is not None else list(range(n_layers + 1))
    depths = [d for d in depths if 0 <= d <= n_layers]

    layer_rows: List[Dict[str, Any]] = []
    for depth in depths:
        if depth not in resid:                             # hook never fired for this depth
            continue
        h = resid[depth].unsqueeze(1)                      # [1, 1, d_model]
        row: Dict[str, Any] = {"layer": depth}

        ll = readout(logit_lens(stack, h)[0, 0], correct_id, stale_id, tokenizer, cfg.top_k)
        row.update({f"logitlens_{k}": v for k, v in ll.items()})

        if jlens_mode != "off":
            # depth == n_layers is legal: forward_from() then runs zero decoder layers and
            # the remaining map is just final_norm + lm_head, so J-Lens degenerates towards
            # the logit lens (they differ only by RMSNorm's nonlinearity — a free sanity
            # check that the two columns converge at the top of the stack).
            prev = resid[depth - 1].unsqueeze(1) if depth > 0 else None
            scores, jlens_mode = jacobian_lens(
                stack, depth, h, kv_prefix, target, jlens_mode,
                cfg.jlens_baseline, cfg.jlens_eps, prev,
            )
            jl = readout(scores[0], correct_id, stale_id, tokenizer, cfg.top_k)
            row.update({f"jlens_{k}": v for k, v in jl.items()})

        if depth < n_layers and depth in attn_rows:
            row.update(attention_stats(attn_rows[depth], correct_tokens, stale_tokens))

        layer_rows.append(row)

    result: Dict[str, Any] = {
        "example_id": f"ex{record['_dataset_index']:06d}",
        "dataset_index": record["_dataset_index"],
        # --- stimulus metadata, duplicated onto every layer row for easy groupby ---
        "compression": record.get("compression"),
        "entity_type": record.get("entity_type"),
        "entity": record.get("entity"),
        "complexity_level": record.get("complexity_level"),
        "distractor_position": record.get("distractor_position"),
        "distractor_role": record.get("distractor_role"),
        "n_distractors": record.get("n_distractors"),
        # --- stimulus ---
        "correct_state": record["correct_state"],
        "stale_state": record["stale_state"],
        "correct_token": correct_tok,
        "stale_token": stale_tok,
        "correct_token_id": correct_id,
        "stale_token_id": stale_id,
        "first_token_collision": collision,
        "token_alignment_ok": bool(ok_c and ok_s),
        "n_tokens": int(ids.shape[1]),
        "target_position": target,
        "n_correct_mentions": len(correct_tokens),
        "n_stale_mentions": len(stale_tokens),
        "generated_text": generated_text,
        # --- behaviour at the output layer ---
        "outcome": outcome,
        "output_is_stale": outcome == "stale",
        "final_p_correct": final["p_correct"],
        "final_p_stale": final["p_stale"],
        "final_rank_correct": final["rank_correct"],
        "final_rank_stale": final["rank_stale"],
        "final_margin": final["margin"],
        "final_top1_token": final["top1_token"],
        # --- per-depth results ---
        "layers": layer_rows,
        "prompt_text": text,
    }
    result.update(summarise_layers(layer_rows, n_layers))

    del resid, kv_full, kv_prefix
    return result, jlens_mode


def summarise_layers(layer_rows: List[Dict[str, Any]], n_layers: int) -> Dict[str, Any]:
    """Per-example headline numbers — this is what answers the research question.

    `*_crossover_layer` = shallowest depth from which the correct room outranks the stale
    room *and keeps doing so* is deliberately NOT what we compute; we report the first
    crossover and the best (lowest) rank, so a transient internal win is still visible.
    """
    out: Dict[str, Any] = {}
    for lens in ("logitlens", "jlens"):
        rc = [(r["layer"], r.get(f"{lens}_rank_correct")) for r in layer_rows
              if r.get(f"{lens}_rank_correct") is not None]
        rs = {r["layer"]: r.get(f"{lens}_rank_stale") for r in layer_rows}
        if not rc:
            continue
        best_layer, best_rank = min(rc, key=lambda x: x[1])
        cross = next((l for l, r in rc
                      if rs.get(l) is not None and r < rs[l]), None)
        margins = [(r["layer"], r.get(f"{lens}_margin")) for r in layer_rows
                   if r.get(f"{lens}_margin") is not None]
        best_m_layer, best_margin = max(margins, key=lambda x: x[1]) if margins else (None, None)
        out[f"{lens}_best_rank_correct"] = best_rank
        out[f"{lens}_best_rank_layer"] = best_layer
        out[f"{lens}_best_rank_depth_frac"] = round(best_layer / max(n_layers, 1), 4)
        out[f"{lens}_crossover_layer"] = cross if cross is not None else -1
        out[f"{lens}_best_margin"] = best_margin
        out[f"{lens}_best_margin_layer"] = best_m_layer
        out[f"{lens}_n_layers_correct_wins"] = sum(
            1 for l, r in rc if rs.get(l) is not None and r < rs[l])
    return out


# =============================================================================
# 10. EXPORT
# =============================================================================

EXAMPLE_COLUMNS = [
    "example_id", "dataset_index", "compression", "entity_type", "entity", "complexity_level",
    "distractor_position", "distractor_role", "n_distractors",
    "correct_state", "stale_state", "correct_token", "stale_token",
    "first_token_collision", "token_alignment_ok",
    "n_tokens", "target_position", "n_correct_mentions", "n_stale_mentions",
    "outcome", "output_is_stale", "final_p_correct", "final_p_stale",
    "final_rank_correct", "final_rank_stale", "final_margin", "final_top1_token",
    "logitlens_best_rank_correct", "logitlens_best_rank_layer", "logitlens_best_rank_depth_frac",
    "logitlens_crossover_layer", "logitlens_best_margin", "logitlens_best_margin_layer",
    "logitlens_n_layers_correct_wins",
    "jlens_best_rank_correct", "jlens_best_rank_layer", "jlens_best_rank_depth_frac",
    "jlens_crossover_layer", "jlens_best_margin", "jlens_best_margin_layer",
    "jlens_n_layers_correct_wins",
    "generated_text",
]

LAYER_METRIC_COLUMNS = [
    "logitlens_rank_correct", "logitlens_rank_stale", "logitlens_score_correct",
    "logitlens_score_stale", "logitlens_p_correct", "logitlens_p_stale",
    "logitlens_margin", "logitlens_top1_token",
    "jlens_rank_correct", "jlens_rank_stale", "jlens_score_correct", "jlens_score_stale",
    "jlens_p_correct", "jlens_p_stale", "jlens_margin", "jlens_top1_token",
    "attn_correct_mass", "attn_stale_mass", "attn_correct_update_mass",
    "attn_stale_anchor_mass", "attn_correct_minus_stale",
    "attn_max_head_correct", "attn_max_head_stale", "attn_argmax_head_correct", "attn_entropy",
]

LAYER_CONTEXT_COLUMNS = [
    "example_id", "dataset_index", "layer", "compression", "entity_type", "complexity_level",
    "distractor_position", "n_distractors", "outcome", "output_is_stale",
    "correct_state", "stale_state",
]


def write_outputs(results: List[Dict[str, Any]], cfg: ProbeConfig, meta: Dict[str, Any]) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- full fidelity, including per-layer top-k vocabulary ---
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- one row per example ---
    with (out_dir / "examples.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EXAMPLE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # --- one row per (example, layer): the main analysis table ---
    with (out_dir / "layers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LAYER_CONTEXT_COLUMNS + LAYER_METRIC_COLUMNS,
                           extrasaction="ignore")
        w.writeheader()
        for r in results:
            ctx = {k: r.get(k) for k in LAYER_CONTEXT_COLUMNS if k != "layer"}
            for lr in r["layers"]:
                row = dict(ctx)
                row["layer"] = lr["layer"]
                for k in LAYER_METRIC_COLUMNS:
                    row[k] = lr.get(k)
                w.writerow(row)

    # --- optional per-head attention table ---
    if cfg.dump_heads:
        with (out_dir / "heads.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["example_id", "compression", "entity_type", "complexity_level",
                        "outcome", "layer", "head", "attn_correct_mass", "attn_stale_mass"])
            for r in results:
                for lr in r["layers"]:
                    pc = lr.get("_per_head_correct") or []
                    ps = lr.get("_per_head_stale") or []
                    for hd in range(max(len(pc), len(ps))):
                        w.writerow([
                            r["example_id"], r["compression"], r["entity_type"],
                            r["complexity_level"], r["outcome"], lr["layer"], hd,
                            pc[hd] if hd < len(pc) else "",
                            ps[hd] if hd < len(ps) else "",
                        ])

    with (out_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(cfg), "run": meta}, fh, indent=2)

    print(f"\n[export] {out_dir.resolve()}")
    for name in ("layers.csv", "examples.csv", "heads.csv", "records.jsonl", "config.json"):
        p = out_dir / name
        if p.exists():
            print(f"         {name:<16} {p.stat().st_size / 1024:8.1f} KB")


# =============================================================================
# 11. CONSOLE SUMMARY
# =============================================================================

def print_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    n = len(results)
    stale = [r for r in results if r["outcome"] == "stale"]
    correct = [r for r in results if r["outcome"] == "correct"]

    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 78)
    print(f"SUMMARY over {n} examples")
    print("=" * 78)
    print(f"  output = correct room : {len(correct):4d}  ({len(correct)/n:6.1%})")
    print(f"  output = STALE room   : {len(stale):4d}  ({len(stale)/n:6.1%})")
    print(f"  output = something else: {n - len(correct) - len(stale):3d}")

    if stale:
        print("\n  --- among the STALE-output failures (the cases we care about) ---")
        for lens in ("logitlens", "jlens"):
            # NB: explicit None-check — `(0 or -1)` is -1, which would drop a crossover at layer 0.
            wins = [r for r in stale
                    if r.get(f"{lens}_crossover_layer") is not None
                    and r[f"{lens}_crossover_layer"] >= 0]
            print(f"  [{lens:9s}] correct room outranks stale in >=1 internal layer: "
                  f"{len(wins):4d}/{len(stale)}  ({len(wins)/len(stale):6.1%})")
            print(f"  [{lens:9s}] best internal rank of correct room (mean): "
                  f"{mean([r.get(f'{lens}_best_rank_correct') for r in stale]):8.1f}"
                  f"   at depth frac {mean([r.get(f'{lens}_best_rank_depth_frac') for r in stale]):.2f}")
        print("\n  A large gap between 'internally decodable' and 'emitted' is the")
        print("  retrieval-failure signature Axis 2 is testing for.")

    by_comp: Dict[Any, List[Dict[str, Any]]] = {}
    for r in results:
        by_comp.setdefault(r.get("compression"), []).append(r)
    if len(by_comp) > 1:
        print("\n  --- stale rate by compression mode ---")
        for k, rs in sorted(by_comp.items(), key=lambda kv: str(kv[0])):
            s = sum(1 for r in rs if r["outcome"] == "stale")
            print(f"    {str(k):<12} n={len(rs):4d}  stale={s/len(rs):6.1%}  "
                  f"jlens best rank(correct)={mean([r.get('jlens_best_rank_correct') for r in rs]):7.1f}")

    by_ent: Dict[Any, List[Dict[str, Any]]] = {}
    for r in results:
        by_ent.setdefault(r.get("entity_type"), []).append(r)
    if len(by_ent) > 1:
        print("\n  --- stale rate by entity_type ---")
        for k, rs in sorted(by_ent.items(), key=lambda kv: str(kv[0])):
            s = sum(1 for r in rs if r["outcome"] == "stale")
            print(f"    {str(k):<12} n={len(rs):4d}  stale={s/len(rs):6.1%}")
    print("=" * 78)


# =============================================================================
# 12. SELF-CHECK
# =============================================================================

def run_self_check(model, tokenizer, stack: TruncatedStack, record: Dict[str, Any],
                   cfg: ProbeConfig) -> None:
    """Replay the whole stack from layer 0 through TruncatedStack and compare with the real
    forward pass. Validates cache slicing, RoPE positions, layer kwargs and device routing
    in one number. Anything above ~1e-1 (fp16) means the replay is not faithful."""
    device = next(model.parameters()).device
    text = build_text(tokenizer, record, cfg)
    ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], device=device)
    t = ids.shape[1] - 1

    with ResidualCapture(model, t) as cap, torch.no_grad():
        out = forward_with_cache(model, ids)
    real = out.logits[0, -1, :].float()
    kv = _legacy_kv(out.past_key_values)
    kv_prefix = [(k[:, :, :t, :], v[:, :, :t, :]) for (k, v) in kv]

    with torch.no_grad():
        replay = stack.forward_from(0, cap.resid[0].unsqueeze(1), kv_prefix, t)[0].float()

    diff = (real - replay).abs().max().item()
    corr = torch.corrcoef(torch.stack([real, replay]))[0, 1].item()
    agree = int(real.argmax()) == int(replay.argmax())
    print(f"[self-check] max|Δlogit| = {diff:.4f} | corr = {corr:.6f} | argmax agrees = {agree}")
    if diff > 0.5 or not agree:
        print("[self-check] WARNING: the truncated-stack replay does not match the model. "
              "J-Lens numbers are unreliable — check your transformers version.")
    else:
        print("[self-check] OK — truncated stack faithfully reproduces the model.")


# =============================================================================
# 13. DRY RUN (tokenizer only — no GPU needed)
# =============================================================================

def dry_run(cfg: ProbeConfig, records: List[Dict[str, Any]]) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    print("\n=== ROOM VOCABULARY -> FIRST TOKEN (collisions would break rank metrics) ===")
    rooms = sorted({r["correct_state"] for r in records} | {r["stale_state"] for r in records})
    ctx = "The object is currently in the"
    table = {}
    for room in rooms:
        tid, tstr, ok = first_continuation_token(tok, ctx, " " + room)
        table.setdefault(tid, []).append(room)
        print(f"  {room:<14} -> id={tid:<8} token={tstr!r:<16} aligned={ok}")
    for tid, names in table.items():
        if len(names) > 1:
            print(f"  !! COLLISION: {names} all start with token id {tid}")

    print("\n=== CONSTRUCTED INPUTS (first 3) ===")
    for rec in records[:3]:
        text = build_text(tok, rec, cfg)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        print("-" * 78)
        print(f"  level={rec.get('complexity_level')} compression={rec.get('compression')} "
              f"entity_type={rec.get('entity_type')} | {len(ids)} tokens")
        print(f"  correct={rec['correct_state']!r}  stale={rec['stale_state']!r}")
        print(f"  ...{text[-220:]!r}")
        print(f"  readout position = {len(ids) - 1} (last token = "
              f"{tok.convert_ids_to_tokens(ids[-1])!r})")
        for f in ("correct_state", "stale_state"):
            spans = char_spans(text, rec[f])
            print(f"  {f} appears {len(spans)}x at char offsets {spans}")
    print("\n[dry-run] no model was loaded. Drop --dry-run to run the real probe.")


# =============================================================================
# 14. MAIN
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> ProbeConfig:
    p = argparse.ArgumentParser(
        description="J-Lens / logit-lens stale-state probe for Qwen3-8B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = ProbeConfig()

    g = p.add_argument_group("data")
    g.add_argument("--jsonl", default=d.jsonl, help="input .jsonl from the stale-state generator")
    g.add_argument("--out-dir", default=d.out_dir)
    g.add_argument("--limit", type=int, default=d.limit)
    g.add_argument("--shuffle", action="store_true")
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--filter-compression", nargs="+", default=None,
                   choices=["none", "light", "aggressive"])
    g.add_argument("--filter-entity-type", nargs="+", default=None, choices=["object", "agent"])
    g.add_argument("--filter-level", nargs="+", type=int, default=None, choices=[1, 2, 3])

    g = p.add_argument_group("model")
    g.add_argument("--model-id", default=d.model_id)
    g.add_argument("--load-in", default=d.load_in, choices=["auto", "4bit", "8bit", "bf16", "fp16"])
    g.add_argument("--device-map", default=d.device_map)
    g.add_argument("--attn-impl", default=d.attn_impl, choices=["sdpa", "eager", "flash_attention_2"])
    g.add_argument("--max-prompt-tokens", type=int, default=d.max_prompt_tokens)

    g = p.add_argument_group("prompt / readout")
    g.add_argument("--prompt-format", default=d.prompt_format, choices=["chat", "raw"])
    g.add_argument("--enable-thinking", action="store_true",
                   help="keep Qwen3's <think> block (off by default: it moves the readout position)")
    g.add_argument("--answer-template", default=d.answer_template)
    g.add_argument("--target-mode", default=d.target_mode, choices=["prefill", "generate"])
    g.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)

    g = p.add_argument_group("lenses")
    g.add_argument("--jlens-mode", default=d.jlens_mode, choices=["auto", "autograd", "fd", "off"])
    g.add_argument("--jlens-baseline", default=d.jlens_baseline, choices=["zero", "prev_layer", "none"])
    g.add_argument("--jlens-eps", type=float, default=d.jlens_eps)
    g.add_argument("--layers", default=None,
                   help="comma list / ranges of residual depths, e.g. '0,8,16,24,32' or '0-36:4'")
    g.add_argument("--top-k", type=int, default=d.top_k)

    g = p.add_argument_group("attention")
    g.add_argument("--no-attention", action="store_true")
    g.add_argument("--dump-heads", action="store_true", help="also write per-head attention CSV")

    g = p.add_argument_group("misc")
    g.add_argument("--self-check", action="store_true",
                   help="verify the truncated-stack replay reproduces the model's logits")
    g.add_argument("--dry-run", action="store_true", help="tokenizer-only sanity check, no GPU")
    g.add_argument("--quiet", action="store_true")

    a = p.parse_args(argv)

    layers = None
    if a.layers:
        layers = []
        for part in a.layers.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                rng, _, step = part.partition(":")
                lo, _, hi = rng.partition("-")
                layers.extend(range(int(lo), int(hi) + 1, int(step) if step else 1))
            else:
                layers.append(int(part))
        layers = sorted(set(layers))

    return ProbeConfig(
        jsonl=a.jsonl, out_dir=a.out_dir, limit=a.limit, shuffle=a.shuffle, seed=a.seed,
        filter_compression=a.filter_compression, filter_entity_type=a.filter_entity_type,
        filter_level=a.filter_level,
        model_id=a.model_id, load_in=a.load_in, device_map=a.device_map,
        attn_impl=a.attn_impl, max_prompt_tokens=a.max_prompt_tokens,
        prompt_format=a.prompt_format, enable_thinking=a.enable_thinking,
        answer_template=a.answer_template, target_mode=a.target_mode,
        max_new_tokens=a.max_new_tokens,
        jlens_mode=a.jlens_mode, jlens_baseline=a.jlens_baseline, jlens_eps=a.jlens_eps,
        layers=layers, top_k=a.top_k,
        capture_attention=not a.no_attention, dump_heads=a.dump_heads,
        self_check=a.self_check, dry_run=a.dry_run, verbose=not a.quiet,
    )


def run(cfg: ProbeConfig) -> List[Dict[str, Any]]:
    records = load_records(cfg)
    print(f"[data] {len(records)} records from {cfg.jsonl}")
    if not records:
        print("[data] nothing to do after filtering.")
        return []

    if cfg.dry_run:
        dry_run(cfg, records)
        return []

    model, tokenizer, precision, is_quantized = load_model(cfg)
    stack = TruncatedStack(model)

    jlens_mode = cfg.jlens_mode
    if jlens_mode == "auto":
        jlens_mode = "fd" if is_quantized else "autograd"
    print(f"[lens] logit-lens + J-Lens(mode={jlens_mode}, baseline={cfg.jlens_baseline}"
          + (f", eps={cfg.jlens_eps}" if jlens_mode == "fd" else "") + ")")

    if cfg.self_check:
        run_self_check(model, tokenizer, stack, records[0], cfg)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            res, jlens_mode = probe_example(model, tokenizer, stack, rec, cfg, jlens_mode)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i+1}/{len(records)}] OOM — skipping. Try --load-in 4bit or --layers.")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as exc:
            print(f"  [{i+1}/{len(records)}] FAILED ({type(exc).__name__}: {exc}) — skipping")
            continue

        results.append(res)
        if cfg.verbose:
            el = time.time() - t0
            print(f"  [{i+1}/{len(records)}] {res['example_id']} "
                  f"lvl{res['complexity_level']} {res['compression']:<10} "
                  f"out={res['outcome']:<7} "
                  f"final rank(correct)={res['final_rank_correct']:<5} "
                  f"jlens best rank(correct)={res.get('jlens_best_rank_correct')} "
                  f"@L{res.get('jlens_best_rank_layer')} "
                  f"| {el/(i+1):.1f}s/ex")

        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    meta = {
        "n_examples": len(results),
        "precision": precision,
        "jlens_mode_used": jlens_mode,
        "elapsed_sec": round(time.time() - t0, 1),
        "torch": torch.__version__,
    }
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except Exception:
        pass

    write_outputs(results, cfg, meta)
    print_summary(results)
    return results


# Swap in a different J-Lens formulation here if the team standardises on one.
LENS_REGISTRY = {
    "logit_lens": logit_lens,
    "jacobian_lens": jacobian_lens,
}


if __name__ == "__main__":
    run(parse_args())
