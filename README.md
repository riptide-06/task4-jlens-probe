# Task 4 — J-Lens Stale-State Probe for Qwen3-8B

**Team:** tommy-hjra · Algoverse AI Research, Summer 2026
**Axis:** 2 — State Tracking and Belief Update in Long-Horizon Tasks

Mechanistic-interpretability pipeline that reads Qwen2.5-7B-Instruct internal activations at the
moment it answers *"what room is the object in now?"* and asks whether the **correct,
updated** room is decodable inside the network **even on the trials where the model emits
the stale room**.

```
representation failure  ->  correct state never appears at any depth
retrieval    failure    ->  correct state is decodable mid-stack, then loses at the top
```

Consumes the `.jsonl` produced by `../stale-state-data-generator/` directly — same field
names, same invariants, no conversion step.

---

## Files

| file | what it is |
|---|---|
| `jlens_probe.py` | the pipeline. Importable module *and* CLI. |
| `qwen3_jlens_colab.ipynb` | Colab driver — GPU check, install, dry run, probe, analysis, download. |
| `analyze_results.py` | offline analysis: behaviour, encode-vs-retrieve, depth profile, attention. |
| `test_offline.py` | 46 assertions over every non-GPU code path. Run it after any edit. |
| `data/sample_probe_input.jsonl` | 35 rows stratified over level × compression × entity_type, so the pipeline runs out of the box. |
| `outputs/` | run artefacts land here. |

---

## Quickstart

**Colab (intended path).** Open `qwen3_jlens_colab.ipynb`, set the runtime to a T4 or
A100, upload `jlens_probe.py`, `analyze_results.py` and your `.jsonl`, run the cells.

**CLI.**

```bash
pip install "transformers>=4.51.0" accelerate bitsandbytes pandas matplotlib

# 0. no GPU needed — check tokenisation and readout positions first
python jlens_probe.py --jsonl data/sample_probe_input.jsonl --dry-run

# 1. the probe
python jlens_probe.py \
    --jsonl data/sample_probe_input.jsonl \
    --out-dir outputs/run1 \
    --limit 20 --load-in auto --dump-heads --self-check

# 2. analysis
python analyze_results.py outputs/run1 --plot
```

`transformers >= 4.51.0` is a hard requirement — that release added the Qwen3 architecture.

---

## Verified GPU configuration

This is the configuration that has actually been run end-to-end on a GPU. Prefer it over
the `auto` defaults above when you want results comparable to the tracked runs in
`outputs/`.

```bash
python3 -u jlens_probe.py \
    --jsonl data/staleness_5k.jsonl \
    --out-dir outputs/<name> \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --load-in bf16 \
    --attn-impl eager \
    --jlens-mode autograd \
    --dump-heads
```

> ### ⚠️ `--load-in bf16` and `--attn-impl eager` are load-bearing
>
> **`--load-in 4bit` and `--attn-impl sdpa` both break the autograd JVP and fall back to
> finite differences.** The run still completes and still writes J-Lens columns — but they
> are the noisier finite-difference estimate, not the exact Jacobian, so numbers from such
> a run are **not** comparable to a `--jlens-mode autograd` run.
>
> Why: bitsandbytes' quantised matmul and SDPA's fused kernel are not twice
> differentiable, so the double-backward identity `d/du[v·(Jᵀu)] = Jv` raises on the second
> `autograd.grad` call.
>
> **How to check a run after the fact:** `jlens_mode_used` in that run's `config.json` records
> the mode actually used, not the one requested. Confirm it says `autograd`:
>
> ```bash
> grep jlens_mode_used outputs/<name>/config.json
> ```
>
> A one-line fallback warning is also printed to stdout the first time it happens — easy to
> miss in a long Colab log, which is why `config.json` is the authority.

### CLI gotchas

| gotcha | detail |
|---|---|
| the subsample flag is **`--limit`** | not `--max-records`. |
| `analyze_results.py` takes the run dir | `python analyze_results.py --plot <run_dir>` |
| **`staleness_5k.jsonl` is ordered by complexity level** | the file is exactly three contiguous blocks — rows 0–1666 are level 1, 1667–3333 level 2, 3334–4999 level 3. A bare `--limit 500` therefore samples **100 % level-1** rows. Always pair it with `--shuffle`: `--limit 500 --shuffle`. |
| **`staleness_5k.jsonl` has no compression variation** | all 5000 rows are `compression=none` (also `distractor_role=auto` throughout, and `distractor_position` only `none`/`middle`). Runs over this file cannot speak to the compression axis at all — regenerate with `--compression mixed` in `stale-state-data-generator` before making any claim about it. |

---

## Method

### Readout position

The generator's prompts end with a *question*, so the next token is not the answer. The
probe appends an entity-aware prefill that ends exactly where the room name must appear:

```
<|im_start|>assistant
<think>

</think>

The encrypted laptop is currently in the      <- readout position t
```

Objects get the article (`The rusty key`), agents do not (`Drone R-7`), matching the
generator's `entity_type` convention. `--target-mode generate` instead lets the model
answer freely and puts the readout at whichever position actually emitted a room name.

Candidate tokens are resolved **in context** (tokenise `prefix` and `prefix + " Laboratory"`
and take the first divergent id) rather than by encoding `" Laboratory"` standalone, because
byte-level BPE merges across the boundary. Multi-token rooms such as `Server Room` are
scored on their first token; `--dry-run` prints the table and flags any collision between
two rooms' first tokens.

### The two lenses

Let `resid[i]` be the residual stream entering decoder layer `i` at position `t`
(`resid[0]` = embeddings, `resid[L]` = last layer output), and let `f_i` be the model's own
remaining computation — layers `i..L-1`, final RMSNorm, `lm_head` — evaluated at `t` with
the real, frozen KV cache of positions `< t`.

**Logit lens** (baseline)

```
logitlens_i = lm_head(final_norm(resid[i]))
```

Cheap, but it skips every remaining layer, so it under-reads states whose meaning is only
realised by later attention/MLP reads.

**J-Lens / Jacobian lens** (main instrument)

```
jlens_i = f_i(b) + J · (resid[i] - b),      J = df_i/dh evaluated at h = resid[i]
```

with baseline `b = 0` by default. This is a first-order Taylor readout of the *actual*
downstream network at its true operating point, so the contribution of every later layer is
retained. Output is in logit units, directly comparable to the logit lens.

`J` is never materialised (it would be 151936 × 4096). Only the Jacobian-**vector** product
`J·v` is needed, computed either way:

| `--jlens-mode` | how | notes |
|---|---|---|
| `autograd` | double-backward identity: `g(u) = Jᵀu`, then `d/du[v·g(u)] = Jv` | exact, reverse-mode only |
| `fd` | central differences `[f(h+εv) − f(h−εv)] / 2ε` | works through bitsandbytes kernels |
| `auto` *(default)* | `autograd` unquantised, `fd` quantised, auto-fallback on failure | |

> **Naming.** "J-Lens" has no canonical public implementation, so the projection is
> implemented here from the definition above rather than imported. It is isolated in
> `jacobian_lens()` (see `LENS_REGISTRY`) — swap that one function if the team standardises
> on a different formulation, and nothing else changes.

### Hooks

| what | mechanism | why this way |
|---|---|---|
| residual stream | `register_forward_pre_hook` on every `layers[i]` — a pre-hook's input **is** `resid[i]`; a post-hook on the last layer supplies `resid[L]` | the hook slices column `t` immediately, so memory is `O(L × d)` not `O(L × T × d)`; `output_hidden_states=True` would keep the whole sequence |
| attention scores | `register_forward_hook` on every `layers[i].self_attn`, keeping `output[1]` | weights only exist under the **eager** kernel (sdpa/flash fuse the softmax and return `None`), so the pass runs inside `eager_attention()`, which flips `config._attn_implementation` for that pass only |
| truncated stack | `TruncatedStack.forward_from(i, h, cache, t)` calls `layers[i:]` by hand | makes each J-Lens probe **one token** through ≤ L layers instead of a full re-run |

Two properties make the single-token replay exact rather than approximate:

1. **Causality.** Perturbing position `t` cannot change positions `< t`, so their cached
   keys/values are valid unchanged — that is why the prefix cache can be frozen and reused.
2. **Query length 1.** A lone query at the end of the sequence legitimately attends to every
   cached key, so `attention_mask=None` is correct under every kernel — no causal-mask
   plumbing, and the captured attention tensor is `[1, H, 1, t+1]` (~19 MB at 8k context)
   instead of the `[1, H, T, T]` blow-up that `output_attentions=True` over a full long
   prompt would cost.

`--self-check` validates all of it in one number: it replays from layer 0 and compares
against the model's real logits.

```
[self-check] max|Δlogit| = 0.0043 | corr = 0.999998 | argmax agrees = True
```

If that diverges, the transformers version changed the decoder-layer contract and the
J-Lens numbers should not be trusted.

---

## Outputs

### `layers.csv` — one row per (example, depth). The main analysis table.

| column | meaning |
|---|---|
| `example_id`, `dataset_index`, `layer` | keys (`layer` 0 = embeddings … L = last layer output) |
| `compression`, `entity_type`, `complexity_level`, `distractor_position`, `n_distractors` | stimulus metadata, duplicated for direct `groupby` |
| `outcome`, `output_is_stale` | what the model finally emitted |
| `logitlens_rank_correct` / `_stale` | 1-based vocabulary rank under the logit lens (1 = argmax) |
| `logitlens_score_*`, `_p_*`, `_margin`, `_top1_token` | logit, softmax prob, correct−stale gap, argmax token |
| `jlens_*` | the same six under J-Lens |
| `attn_correct_mass` / `attn_stale_mass` | attention from the readout position onto **all** mentions of each room, averaged over heads |
| `attn_correct_update_mass` | mass on the **last** mention of the correct room — i.e. the update event |
| `attn_stale_anchor_mass` | mass on the **first** mention of the stale room — i.e. the original anchor |
| `attn_max_head_correct` / `_stale`, `attn_argmax_head_correct` | strongest single head, and which one |
| `attn_entropy` | entropy of the head-averaged attention row |

### `examples.csv` — one row per example

Everything above collapsed, plus the behavioural result (`final_p_correct`, `final_rank_stale`, …)
and the summary statistics:

| column | meaning |
|---|---|
| `{lens}_best_rank_correct` | best (lowest) rank the correct room reached at any depth |
| `{lens}_best_rank_layer`, `_depth_frac` | where that happened, absolute and as a fraction of the stack |
| `{lens}_crossover_layer` | shallowest depth where correct outranks stale; `-1` = never |
| `{lens}_n_layers_correct_wins` | how many depths correct outranked stale |
| `{lens}_best_margin`, `_best_margin_layer` | largest correct−stale logit gap and where |

### `heads.csv` (`--dump-heads`) — one row per (example, layer, head)

### `records.jsonl` — everything, including the per-depth top-k vocabulary and the exact prompt text

---

## Reading the result

`analyze_results.py` prints four tables. The one that answers Axis 2 is **T2**, computed
over the trials where the model **emitted the stale room**:

```
[jlens]
  correct room outranked stale in >=1 internal layer :   19/19  (100.0%)   <- RETRIEVAL failure
  correct room never won at any depth                :    0/19  (0.0%)     <- REPRESENTATION failure
  best internal rank of correct room  (median)       : 33
  depth of that best rank (fraction of stack, median): 0.61
```

and **T3**, the depth profile. A rank curve for the correct room that dips deep mid-stack
and then rises again towards the output is the *encoded-then-discarded* signature:

```
-> J-Lens rank of the CORRECT room bottoms out at layer 22 (rank 35.6)
   then degrades to 88129.9 at the output.
```

Cross-reference with `attn_correct_update_mass` vs `attn_stale_anchor_mass` split by
`output_is_stale` to see whether the failure co-occurs with attention drifting off the
update event and onto the original anchor.

---

## Cost and knobs

Per example the pipeline runs **one** full forward, one single-token attention replay, and
~3 single-token partial forwards per probed depth. On short prompts the depth sweep costs
roughly a few full forward passes; on long contexts the single-token replays stay cheap
because only the KV-cache read scales with `T`.

| situation | flag |
|---|---|
| T4 (16 GB, no bf16) | `--load-in 4bit` (≈5.5 GB) |
| A100 | `--load-in bf16` — also enables the exact `autograd` J-Lens |
| too slow | `--layers 0-36:2` or `0-36:4` to probe fewer depths |
| memory pressure | `--no-attention`, or drop `--dump-heads` |
| target one condition | `--filter-compression aggressive --filter-level 3` |

---

## Caveats

* **First-token scoring.** Multi-token rooms (`Server Room`) are ranked on their first
  token. `--dry-run` flags first-token collisions between rooms; drop any rows where
  `first_token_collision` is true.
* **Finite differences in fp16.** Under quantisation, `fd` mode subtracts two nearby fp16
  logit vectors, so the J-Lens scores carry a few percent of noise. Ranks are robust to it;
  raw `jlens_score_*` values should be treated as approximate. `--jlens-eps` trades
  truncation error against cancellation noise.
* **Linearisation.** J-Lens is a *first-order* readout. It reports what the downstream
  network does to small changes around the true activation, which is not the same as what a
  full nonlinear patch would do. Causal confirmation still needs activation patching — the
  natural next step, and the layers J-Lens flags are where to patch.
* **Correlational.** Both lenses are decoding, not intervention. "Decodable at layer 22"
  does not by itself prove the model *could* have used it.
* **Attention aggregation.** `attn_*` columns average over heads; a single head carrying
  the signal will look weak there. Use `heads.csv` for head-level claims.
* **Chat template.** `--prompt-format chat` suppresses Qwen3's `<think>` block so the
  readout position is deterministic. With thinking enabled the model reasons before
  answering, and the single-position readout no longer captures the decision.

---

## Development

```bash
python test_offline.py     # 46 checks, no GPU, no transformers required
```

Covers JSONL loading and invariant guards, prompt construction, context-aware token
resolution, char-span → token-index mapping (including multi-token rooms), the layer
summariser, all four export formats, and CLI parsing.
