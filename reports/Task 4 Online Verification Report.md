# Task 4 Online Verification Report: J-Lens Probing on GPU

**Project:** Mechanistic interpretability of stale-state retention in LLM agents
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, d_model 3584, vocab 152064)
**Hardware:** Google Colab, NVIDIA A100-SXM4-40GB, bf16 precision
**Dataset:** `data/staleness_5k.jsonl` (5,000 records from the synthetic stale-state generator)
**Date:** July 30, 2026
**Author:** TN

## Executive summary

The full probing pipeline (`jlens_probe.py` + `analyze_results.py`) was executed end to end on GPU across the complete 5,000-record dataset, a shuffled 500-record full-layer sweep, and a targeted full-layer sweep of every stale-output failure. The headline result: **in 100% of stale-output failures (137/137), the correct updated state is decodable from the model's internal representations while the model emits the stale one.** In no case was the updated state absent internally. The updated state is encoded but goes unused at the output: an expression (readout) failure, not a loss of the information and not a retrieval failure in the memory-systems sense, since the update is fully present in context and in the internal representations. (Terminology note: `analyze_results.py` labels these categories "RETRIEVAL failure" vs "REPRESENTATION failure" in its printed output; this report and the paper use "encoded-but-unused" vs "not encoded" to avoid collision with context/memory retrieval.)

Two secondary findings sharpen the picture. First, staleness is entirely concentrated: every failure occurs on complexity-level-3 records (8.2% stale rate at level 3, 0.0% at levels 1 and 2) and entirely on agent entities (5.5% agent vs 0.0% object). The model never loses track of objects; it loses track of agents in the hardest condition. Second, attention at the readout position differs systematically between successes and failures: correct answers attend more mass to the update mention (0.053 vs 0.042) and far less to the stale anchor (0.009 vs 0.030), giving a first mechanistic hint at where readout goes wrong.

This report also corrects several errors in the prior offline verification report (see Corrections section). The command recommended there would silently reintroduce a 5-6x slowdown and a broken gradient path; use the configurations documented here.

## Working configuration

The validated fast configuration for autograd J-Lens on A100:

```bash
python3 -u jlens_probe.py \
  --jsonl data/staleness_5k.jsonl \
  --out-dir outputs/<run_name> \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --load-in bf16 \
  --attn-impl eager \
  --jlens-mode autograd \
  --dump-heads \
  [--layers 5,8,13] [--limit N] [--shuffle --seed 42]
```

Why these flags matter. Forward-mode JVPs are not supported by either the `flash_attention_2` package or PyTorch SDPA's flash backend, so `--attn-impl eager` is required for `--jlens-mode autograd`; with sdpa the script silently falls back to finite differences (see the exception handler around line 715 of `jlens_probe.py`), which is both slower and less exact. Similarly, `--load-in 4bit` (bitsandbytes) breaks the double-backward path and triggers the same fallback; on an A100 with 40 GB there is no reason to quantize, and bf16 loads the 7B model in ~5s using 14.2 GB of VRAM. Restricting `--layers` avoids the quadratic cost of the per-layer re-run loop (line 613): probing all 28 layers costs ~3.5 s/example, while a 3-layer restriction runs at ~0.6-0.7 s/example, about 100 records/minute.

## Runs executed

**Run A, behavioral baseline (all 5,000 records, layers 5,8,13):** `outputs/colab_run_qwen_5k_L5-8-13`. ~55 minutes. Provides the behavioral statistics (stale rates, breakdowns) on the full dataset. Its internal-signature statistics are an undercount by construction (see Methodology notes) and should not be quoted.

**Run B, shuffled full-layer sweep (500 records, all 28 layers, seed 42):** `outputs/colab_run_qwen_fullsweep_500`. ~30 minutes. Confirms the behavioral rates on a random sample (2.4% stale vs 2.7% on the full set; agent 4.5% / object 0.0%) and provides unrestricted layer coverage for localization.

**Run C, failure-population full sweep (all 137 stale failures, all 28 layers):** `outputs/colab_run_qwen_failures_fullsweep`. ~8 minutes. The 137 failure records were extracted by row index from Run A's `examples.csv` (the jsonl has no id field; `exNNNNNN` ids map to line numbers, valid because Run A did not shuffle). All 137 records reproduced their stale output deterministically on this second pass (137/137 stale), ruling out sampling noise as the failure mechanism.

## Behavioral results (Run A, N=5000)

The model outputs the correct room in 4,853 cases (97.1%), the stale room in 137 (2.7%), and something else in 10 (0.2%). The stale failures are fully explained by two factors. By complexity level: 0.0% at level 1 (n=1667), 0.0% at level 2 (n=1667), 8.2% at level 3 (n=1666). By entity type: 5.5% for agents (n=2506), 0.0% for objects (n=2494). Every failure in the dataset is a level-3 agent record. For the dataset workstream this means levels 1 and 2 currently produce no failure signal for this model, and difficulty scaling (or harder distractor conditions) is needed if a larger failure population is wanted.

## Core internal result (Run C, N=137, full layer coverage)

For every one of the 137 stale-output failures, both lenses find at least one internal layer where the correct room outranks the emitted stale room:

- Logit-lens: 137/137 (100.0%) encoded-but-unused (script label: "retrieval failure"), 0/137 (0.0%) not-encoded (script label: "representation failure"). Median best internal rank of the correct room: 2, at median depth fraction 1.00 (mean 2.3 at 0.99). On average the correct room beats the stale room in 10.9 layers.
- J-Lens: 137/137 (100.0%) encoded-but-unused, 0/137 (0.0%) not-encoded. Mean best rank 11.3 at mean depth fraction 0.74. Correct beats stale in 8.4 layers on average.

The logit-lens depth profile makes the failure mode vivid: by the final layer, the correct room's mean rank collapses to ~2.4 against the stale room's 1.0. The model arrives at the output with the correct answer in approximately second place and emits the stale one anyway. The J-Lens profile shows the complementary signature the analyzer flags as "encoded-then-discarded": the correct room's decodability dips (improves) sharply in the early-mid stack and then degrades toward the output, while the stale room becomes extremely decodable mid-stack (mean rank ~291 at layer 10, ~973 at layer 12 in the failure population) and wins at the end.

## Attention at the readout position (Run B)

Comparing successes and failures on the shuffled 500: when the model answers correctly, attention mass at the readout position on the update mention is 0.053 versus 0.042 when it answers stale, and mass on the stale anchor is 0.009 versus 0.030. Failures attend roughly 3.5x more to the stale anchor. By layer, attention to the update mention concentrates in the later-mid stack, with layer 22 carrying the largest share (0.25 update mass in correct cases). This is correlational, not causal, but it points the Task 4 causal analysis (patching/ablation) at late-mid attention as the candidate readout circuit.

## Methodology notes

**Layer restriction undercounts the internal signature.** Run A, restricted to layers 5, 8, 13, detects the encoded-but-unused signature in only 57.7% (logit-lens) / 46.0% (J-Lens) of failures, versus 100% with full coverage in Run C. The best-decodable depth simply lies outside the restricted set for many records (J-Lens mean depth 0.74, ~layer 21). Consequence: use restricted runs for behavior and throughput, full sweeps for any internal claim. The original 5/8/13 hypothesis should be revised toward the mid-late stack.

**The dataset file is ordered.** `staleness_5k.jsonl` is written in complexity order (level 1 records first). Any subsample must use `--shuffle --seed <s>`; the first-N records are all level 1 and produce zero failures, which misled early smoke tests into reporting a 0% stale rate.

**Determinism.** All 137 failures reproduced exactly on rerun. Failure is a stable property of the record, not decoding noise.

**The dataset does not exercise the compression axis.** All 5,000 rows of `staleness_5k.jsonl` have `compression=none` (and `distractor_role=auto` throughout, with `distractor_position` limited to none/middle). None of the three runs can therefore support any claim about compression effects, one of the two metadata axes the probe was designed around. The dataset needs regeneration with mixed compression before that axis can be analyzed; this folds into the difficulty-scaling work already planned for the generator.

## Corrections to the prior offline verification report

The offline report (AI-generated) contained configuration details that do not match this project and must not be followed: it names Qwen3-8B-Instruct (actual model: Qwen2.5-7B-Instruct), a Colab T4 (actual: A100), a dataset file `5k_synthetic_dataset.jsonl` (actual: `staleness_5k.jsonl`), 12 layers / 24 heads (test-suite numbers; the real model has 28 layers), and a recommended command using `--load-in 4bit --attn-impl sdpa`, which breaks autograd JVPs and silently degrades to the slow finite-difference path as described above. It also cites `--max-records` (the real flag is `--limit`) and an `analyze_results.py --in-dir/--out-dir` interface (the real interface is `analyze_results.py --plot <run_dir>`, writing `depth_profile.png` into the run directory). The offline test suite results themselves (46/46 passing, tokenizer/vocabulary checks) remain valid.

## Analysis commands

```bash
python3 analyze_results.py --plot outputs/colab_run_qwen_failures_fullsweep
python3 analyze_results.py --plot outputs/colab_run_qwen_fullsweep_500
python3 analyze_results.py --plot outputs/colab_run_qwen_5k_L5-8-13
```

Each writes `depth_profile.png` into its run directory alongside `layers.csv`, `examples.csv`, `heads.csv`, `records.jsonl`, and `config.json`.

## Next steps

The immediate follow-ups suggested by these results: (1) causal validation of the late-mid attention hypothesis via activation patching or head ablation around layers 19-22 on the 137-failure set; (2) generator difficulty scaling, since only level-3 agent records fail, to grow the failure population for statistics; (3) a batched rewrite of the JVP loop in `jlens_probe.py` (currently single-record) to bring full-sweep 5k runs down from ~5 hours to well under one, validated against Run C outputs; (4) cross-model replication once the pipeline is batched.
