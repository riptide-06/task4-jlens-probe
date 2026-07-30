#!/usr/bin/env python3
"""
test_offline.py — validate everything in jlens_probe.py that does NOT need a GPU.

Covers: JSONL loading + invariant guards, prompt construction (chat & raw), the
context-aware first-token resolver, char-span -> token-index mapping (incl. the
multi-token "Server Room"), the per-example layer summariser, and all CSV/JSONL exports.

    python test_offline.py                 # uses data/sample_probe_input.jsonl

Runs anywhere: if torch is not installed it is stubbed, since none of the tested code
paths touch it. Exit code 0 = all pass.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# torch is imported at module scope by jlens_probe but unused by anything tested here.
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("torch")
        stub.Tensor = type("Tensor", (), {})
        stub.__version__ = "stub"
        cuda = types.ModuleType("torch.cuda")
        cuda.is_available = lambda: False
        stub.cuda = cuda
        sys.modules["torch"] = stub
        sys.modules["torch.cuda"] = cuda

sys.path.insert(0, str(HERE))
import jlens_probe as jp  # noqa: E402

FAILURES: list = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{extra}]" if extra != "" else ""))
    if not cond:
        FAILURES.append(name)


class FakeTokenizer:
    """Whitespace-chunk stand-in with GPT-2/Qwen-style leading-space tokens and offsets."""
    pad_token_id = 0
    is_fast = True

    @staticmethod
    def _parts(text):
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\s*\S+|\s+$", text)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        parts = self._parts(text)
        out = {"input_ids": [abs(hash(p[0])) % 100_000 for p in parts]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(p[1], p[2]) for p in parts]
        return out

    def convert_ids_to_tokens(self, i):
        return f"<{i}>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        think = "" if enable_thinking else "<think>\n\n</think>\n\n"
        return (f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n{think}")


def main() -> int:
    data = HERE / "data" / "sample_probe_input.jsonl"
    tmp = HERE / "outputs" / "_offline_test"

    # ---------------- 1. data loading + invariants ----------------
    print("\n[1] data loading")
    recs = jp.load_records(jp.ProbeConfig(jsonl=str(data)))
    check("records load", len(recs) > 0, len(recs))
    check("dataset_index stamped", all("_dataset_index" in r for r in recs))
    check("metadata defaulted", all(f in recs[0] for f in jp.META_FIELDS))

    f = jp.load_records(jp.ProbeConfig(jsonl=str(data), filter_compression=["aggressive"]))
    check("compression filter", f and all(r["compression"] == "aggressive" for r in f), len(f))
    f = jp.load_records(jp.ProbeConfig(jsonl=str(data), filter_level=[3],
                                       filter_entity_type=["agent"]))
    check("level+entity filter",
          all(r["complexity_level"] == 3 and r["entity_type"] == "agent" for r in f), len(f))
    check("limit", len(jp.load_records(jp.ProbeConfig(jsonl=str(data), limit=4))) == 4)

    bad = tmp.parent / "_bad.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(json.dumps({"prompt": "it is in the Kitchen",
                               "correct_state": "Garage", "stale_state": "Kitchen"}) + "\n")
    try:
        jp.load_records(jp.ProbeConfig(jsonl=str(bad)))
        check("guard: state must appear in prompt", False)
    except ValueError as exc:
        check("guard: state must appear in prompt", "not present verbatim" in str(exc))
    bad.write_text(json.dumps({"prompt": "Kitchen", "correct_state": "Kitchen",
                               "stale_state": "Kitchen"}) + "\n")
    try:
        jp.load_records(jp.ProbeConfig(jsonl=str(bad)))
        check("guard: correct != stale", False)
    except ValueError as exc:
        check("guard: correct != stale", "correct_state == stale_state" in str(exc))
    bad.unlink()

    # ---------------- 2. prompt construction ----------------
    print("\n[2] prompt construction")
    tok = FakeTokenizer()
    check("subject: object gets article",
          jp.subject_phrase({"entity": "rusty key", "entity_type": "object"}) == "The rusty key")
    check("subject: agent stays bare",
          jp.subject_phrase({"entity": "Drone R-7", "entity_type": "agent"}) == "Drone R-7")

    rec = recs[0]
    raw = jp.build_text(tok, rec, jp.ProbeConfig(prompt_format="raw"))
    chat = jp.build_text(tok, rec, jp.ProbeConfig(prompt_format="chat"))
    check("raw embeds the prompt", rec["prompt"] in raw)
    check("raw ends on the readout cue", raw.endswith("is currently in the"))
    check("chat opens an assistant turn", chat.rstrip().endswith("in the")
          and "<|im_start|>assistant" in chat)
    check("chat suppresses <think>", "<think>\n\n</think>" in chat)

    # ---------------- 3. candidate tokens ----------------
    print("\n[3] candidate token resolution")
    tid, _, aligned = jp.first_continuation_token(tok, chat, " " + rec["correct_state"])
    ctx = tok(chat)["input_ids"]
    full = tok(chat + " " + rec["correct_state"])["input_ids"]
    check("continuation is context-aware", aligned is True)
    check("continuation id correct", tid == full[len(ctx)])
    cid, _, _ = jp.first_continuation_token(tok, chat, " " + rec["correct_state"])
    sid, _, _ = jp.first_continuation_token(tok, chat, " " + rec["stale_state"])
    check("correct/stale tokens differ", cid != sid)

    # ---------------- 4. span mapping ----------------
    print("\n[4] room mention -> token indices")
    enc = tok(chat, return_offsets_mapping=True)
    offs, tgt = enc["offset_mapping"], len(enc["input_ids"]) - 1
    spans = jp.char_spans(chat, rec["correct_state"])
    idx = jp.token_indices_for_spans(offs, spans, tgt)
    check("one index group per mention", len(idx) == len(spans), f"{len(spans)} mentions")
    check("indices within readout window", all(0 <= i <= tgt for g in idx for i in g))
    text_back = "".join(chat[offs[i][0]:offs[i][1]] for i in idx[0])
    check("span text round-trips", rec["correct_state"] in text_back, repr(text_back))
    check("clipping respects `upto`", jp.token_indices_for_spans(offs, spans, 0) == [])

    mw = next((r for r in recs if "Server Room" in (r["correct_state"], r["stale_state"])), None)
    if mw is not None:
        t2 = jp.build_text(tok, mw, jp.ProbeConfig(prompt_format="raw"))
        e2 = tok(t2, return_offsets_mapping=True)
        g = jp.token_indices_for_spans(e2["offset_mapping"], jp.char_spans(t2, "Server Room"),
                                       len(e2["input_ids"]) - 1)
        back = "".join(t2[e2["offset_mapping"][i][0]:e2["offset_mapping"][i][1]] for i in g[0])
        check("multi-token room spans >1 token", len(g[0]) >= 2, g[0])
        check("multi-token room round-trips", "Server Room" in back)

    # ---------------- 5. layer summariser ----------------
    print("\n[5] per-example summary")
    rows = [
        {"layer": 0, "logitlens_rank_correct": 900, "logitlens_rank_stale": 40,
         "logitlens_margin": -4.0, "jlens_rank_correct": 800, "jlens_rank_stale": 30,
         "jlens_margin": -3.0},
        {"layer": 1, "logitlens_rank_correct": 12, "logitlens_rank_stale": 3,
         "logitlens_margin": -1.0, "jlens_rank_correct": 2, "jlens_rank_stale": 5,
         "jlens_margin": 0.8},
        {"layer": 2, "logitlens_rank_correct": 4, "logitlens_rank_stale": 9,
         "logitlens_margin": 0.5, "jlens_rank_correct": 3, "jlens_rank_stale": 1,
         "jlens_margin": -0.2},
        {"layer": 3, "logitlens_rank_correct": 7, "logitlens_rank_stale": 1,
         "logitlens_margin": -2.0, "jlens_rank_correct": 7, "jlens_rank_stale": 1,
         "jlens_margin": -2.0},
    ]
    s = jp.summarise_layers(rows, 3)
    check("best internal rank", s["logitlens_best_rank_correct"] == 4)
    check("best rank layer", s["logitlens_best_rank_layer"] == 2)
    check("logit-lens crossover", s["logitlens_crossover_layer"] == 2, s["logitlens_crossover_layer"])
    check("J-Lens finds it earlier", s["jlens_crossover_layer"] == 1, s["jlens_crossover_layer"])
    check("win count", s["jlens_n_layers_correct_wins"] == 1)
    check("depth fraction", abs(s["jlens_best_rank_depth_frac"] - 1 / 3) < 1e-3,
          s["jlens_best_rank_depth_frac"])
    check("no metrics -> empty summary", jp.summarise_layers([{"layer": 0}], 1) == {})

    # ---------------- 6. export ----------------
    print("\n[6] export")
    results = []
    for k in range(3):
        r = {c: None for c in jp.EXAMPLE_COLUMNS}
        r.update({
            "example_id": f"ex{k:06d}", "dataset_index": k, "compression": "aggressive",
            "entity_type": "object", "complexity_level": 2, "outcome": "stale",
            "output_is_stale": True, "correct_state": "Garage", "stale_state": "Kitchen",
            "layers": [dict(x, **{"attn_correct_mass": 0.15, "attn_stale_mass": 0.35,
                                  "_per_head_correct": [0.1, 0.2],
                                  "_per_head_stale": [0.3, 0.4]}) for x in rows],
            "prompt_text": "...",
        })
        r.update(jp.summarise_layers(rows, 3))
        results.append(r)

    jp.write_outputs(results, jp.ProbeConfig(out_dir=str(tmp), dump_heads=True),
                     {"n_examples": 3, "test": True})

    lr = list(csv.DictReader((tmp / "layers.csv").open()))
    er = list(csv.DictReader((tmp / "examples.csv").open()))
    hr = list(csv.DictReader((tmp / "heads.csv").open()))
    check("layers.csv = examples x depths", len(lr) == 12, len(lr))
    check("layers.csv carries groupby keys",
          lr[0]["compression"] == "aggressive" and lr[0]["outcome"] == "stale")
    check("layers.csv has both lenses",
          "logitlens_rank_correct" in lr[0] and "jlens_rank_correct" in lr[0])
    check("layers.csv has attention", lr[0]["attn_correct_mass"] == "0.15")
    check("examples.csv one row each", len(er) == 3)
    check("examples.csv has summary", er[0]["jlens_crossover_layer"] == "1")
    check("heads.csv = examples x depths x heads", len(hr) == 24, len(hr))
    check("records.jsonl complete", sum(1 for _ in (tmp / "records.jsonl").open()) == 3)
    check("private keys stay out of CSV", "_per_head_correct" not in lr[0])
    check("config.json written", (tmp / "config.json").exists())

    # ---------------- 7. CLI ----------------
    print("\n[7] CLI parsing")
    check("--layers list", jp.parse_args(["--layers", "0,4,8"]).layers == [0, 4, 8])
    check("--layers range:step", jp.parse_args(["--layers", "0-12:4,36"]).layers
          == [0, 4, 8, 12, 36])
    c = jp.parse_args(["--no-attention", "--dump-heads", "--self-check"])
    check("flags wire through", c.capture_attention is False and c.dump_heads and c.self_check)
    check("multi-value filter",
          jp.parse_args(["--filter-compression", "aggressive", "light"]).filter_compression
          == ["aggressive", "light"])
    check("explicit precision honoured",
          jp.resolve_precision(jp.ProbeConfig(load_in="8bit")) == "8bit")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
