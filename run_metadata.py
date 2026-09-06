#!/usr/bin/env python3
"""
run_metadata.py -- extract compute/provenance facts from the committed per-run
config.json files, so the paper can cite them without anyone retyping a number.

Writes reports/run_metadata.json and prints a table.

NOTE ON WHAT IS AND IS NOT RECORDED: the run blocks carry n_examples, precision,
elapsed_sec, and the torch/transformers versions. They do NOT record the GPU
model. Per-record timing is therefore derivable, but hardware is not, and the
paper says only what is here.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "run_metadata.json")

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "outputs", "*", "config.json"))):
    d = json.load(open(f))
    run, cfg = d.get("run", {}), d.get("config", {})
    if run.get("test") or "elapsed_sec" not in run:
        continue                      # skip the offline unit-test fixture
    n = run["n_examples"]
    depths = len(cfg["layers"]) if cfg.get("layers") else 29
    rows.append(dict(
        run=os.path.basename(os.path.dirname(f)),
        n_examples=n,
        depths_swept=depths,
        elapsed_sec=run["elapsed_sec"],
        sec_per_record=round(run["elapsed_sec"] / n, 3),
        precision=run["precision"],
        attn_impl=cfg.get("attn_impl"),
        model_id=cfg.get("model_id"),
        jlens_mode=run.get("jlens_mode_used"),
        self_check=cfg.get("self_check"),
        torch=run.get("torch"),
        transformers=run.get("transformers"),
        gpu_model=None,               # not logged by the probe; see docstring
    ))

total = sum(r["elapsed_sec"] for r in rows)
summary = dict(runs=rows,
               total_elapsed_sec=round(total, 1),
               total_elapsed_hours=round(total / 3600, 2),
               gpu_model_recorded=False,
               self_check_enabled_in_any_run=any(r["self_check"] for r in rows))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=2)

print("=" * 78)
print("RUN METADATA (from committed outputs/*/config.json)")
print("=" * 78)
print(f"  {'run':34s} {'n':>5} {'depths':>7} {'elapsed_s':>10} {'s/record':>9}")
for r in rows:
    print(f"  {r['run']:34s} {r['n_examples']:>5} {r['depths_swept']:>7} "
          f"{r['elapsed_sec']:>10.1f} {r['sec_per_record']:>9.3f}")
print(f"\n  total GPU wall-clock: {summary['total_elapsed_sec']:,.1f} s "
      f"({summary['total_elapsed_hours']} h)")
print(f"  GPU model recorded in configs: {summary['gpu_model_recorded']}")
print(f"  self-check enabled in any run: {summary['self_check_enabled_in_any_run']}")
print(f"\n  saved -> reports/run_metadata.json")
