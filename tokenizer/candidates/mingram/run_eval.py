# -*- coding: utf-8 -*-
"""Harness metrics, hard gates G1-G5 and properties R1/R2 for the MinGram candidate (PLAN 4.2, 4.3).

    python run_eval.py [--run runs/mingram_P1_D1_16384] [--g4 runs/mingram_P1_D1_16384_g4] [--name NAME]
                       [--data dev_strict] [--gates all|g1] [--tokenizer-json PATH]

Runs eval/harness.run on the HF-native tokenizer.json (stock tokenizers Unigram) with train_D1 as the R1
support view and the independent retrain as the G4 twin. Results go to results/<data>/<name>/ (docs.jsonl +
summary.json). The harness refuses any dataset containing test-split documents.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.join("runs", "mingram_P1_D1_16384"))
ap.add_argument("--tokenizer-json", default=None, help="evaluate this file instead of <run>/tokenizer.json")
ap.add_argument("--g4", default=os.path.join("runs", "mingram_P1_D1_16384_g4"))
ap.add_argument("--name", default="A10-mingram-P1-D1-16k")
ap.add_argument("--data", default="dev_strict")
ap.add_argument("--gates", default="all", choices=["all", "g1"])
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--custom-ref", action="store_true", help="evaluate the pure-Python reference encoder instead")
ap.add_argument("--force", action="store_true")
A = ap.parse_args()
os.environ["RAYON_NUM_THREADS"] = str(A.threads)
os.environ["TOKENIZERS_PARALLELISM"] = "true" if A.threads > 1 else "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HOME", r"F:\Hindko\_tokenizer\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, HERE)
import mingram as M  # noqa: E402
H = M.H
import adapters as AD  # noqa: E402


def absp(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def main():
    run = absp(A.run)
    tok = absp(A.tokenizer_json) if A.tokenizer_json else os.path.join(run, "tokenizer.json")
    g4 = os.path.join(absp(A.g4), "tokenizer.json") if A.g4 and A.gates == "all" else None
    if g4 and not os.path.exists(g4):
        raise SystemExit("G4 twin missing: %s" % g4)
    extra = {"label": M.LABEL}
    eq = os.path.join(run, "equivalence.json")
    if os.path.exists(eq) and not A.tokenizer_json:
        v = json.load(open(eq, encoding="utf-8"))
        extra["encoder_equivalence"] = {"verdict": v.get("verdict"), "tokenizer_sha256": v.get("tokenizer_sha256")}
    tl = os.path.join(run, "train_log.json")
    if os.path.exists(tl) and not A.tokenizer_json:
        t = json.load(open(tl, encoding="utf-8"))
        extra["training"] = {"sizes": t.get("sizes"), "params_argv": t.get("argv"), "prune": t.get("prune"),
                             "outputs_sha256": t.get("outputs_sha256")}
    if A.custom_ref:
        ad = AD.CustomAdapter(M.MinGramRef(os.path.join(run, "mingram_model.json"), "hf"), name=A.name)
        ad.path = os.path.join(run, "mingram_model.json")
    else:
        ad = AD.HFAdapter(path=tok, name=A.name)
    out = os.path.join(HERE, "results", A.data, H.safe_name(A.name))
    s = H.run(ad, data=A.data, gates=A.gates, train="train_D1", g4_retrain=g4, out_dir=out, force=A.force,
              extra=extra)
    g = s["gates"]
    print(json.dumps({k: g[k].get("pass") for k in g}, ensure_ascii=False))
    ov = s["metrics"]["overall"]
    print(json.dumps({k: ov.get(k) for k in ("bytes_per_token", "chars_per_token", "fertility", "continued_word_rate",
                                             "strr", "renyi_eff_a2.5", "vocab_utilisation")}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
