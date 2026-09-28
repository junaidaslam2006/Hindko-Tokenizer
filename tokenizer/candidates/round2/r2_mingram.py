# -*- coding: utf-8 -*-
"""Round 2: A10 MinGram ("MinGram as reimplemented from the paper", Land 2026) with P1r3 on D2, at 32k and 48k.

    python r2_mingram.py train --size 32k [--g4]      train (official run, or the independent G4 twin)
    python r2_mingram.py equiv --size 32k             encoder-equivalence test, HF-native vs reference, 100% of dev
    python r2_mingram.py eval  --size 32k             harness --gates all (G4 twin), R1 on train_D1 and on D2, NSL

The reviewed code in candidates/mingram is used unchanged: train_mingram.py and equiv_test.py are executed with
runpy after ONE patch of the imported mingram module:
  * mingram.P1 (the pre-tokenizer of the seed BPE, of the pretoken types used by EM/prune, of the HF export's
    Split and of the reference encoder MinGramRef) := eval/harness.PRETOKENIZERS['P1r3'], the Stage 1 choice;
    the cached compiled Python twin (mingram._RX) is reset so it is compiled from P1r3;
  * mingram.build_model is wrapped only to label the model file's pretokenizer id 'P1r3' (the stock function
    writes the id 'P1' next to the regex it actually used).
Training data: --data train_D2 (strict train), total = 32,768 / 49,152. Every run uses --no-cache, because the
pretoken-type cache of train_mingram.py lives in candidates/mingram/work (outside round 2); the G4 twin is an
independent second process with identical inputs. The report-only --ablation (dev N_em sweep) is not run.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_common as RC  # noqa: E402

OUT = os.path.join(RC.R2, "mingram")


def run_dir(size, g4=False):
    return os.path.join(OUT, "runs", "mingram_P1r3_D2_%d%s" % (RC.SIZES[size], "_g4" if g4 else ""))


def patch_mingram():
    sys.path.insert(0, RC.MING)
    import mingram as M
    p1r3 = M.H.PRETOKENIZERS["P1r3"]
    M.P1 = p1r3
    M._RX = None
    orig = M.build_model

    def build_model(*a, **k):
        m = orig(*a, **k)
        assert m["pretokenizer"]["oniguruma"] == p1r3
        m["pretokenizer"] = {"id": "P1r3", "oniguruma": p1r3}
        return m
    M.build_model = build_model
    return M


def check_model_pretok(M, run):
    m = json.load(open(os.path.join(run, "mingram_model.json"), encoding="utf-8"))
    assert m["pretokenizer"] == {"id": "P1r3", "oniguruma": M.P1}, m["pretokenizer"]
    tj = json.load(open(os.path.join(run, "tokenizer.json"), encoding="utf-8"))
    assert tj["pre_tokenizer"]["pattern"]["Regex"] == M.P1, tj["pre_tokenizer"]
    assert tj["normalizer"] is None
    return m


def cmd_train(a):
    M = patch_mingram()
    RC.verify_frozen()
    run = run_dir(a.size, a.g4)
    os.makedirs(run, exist_ok=True)
    argv = ["--out", run, "--data", "train_D2", "--total", str(RC.SIZES[a.size]), "--no-cache",
            "--threads", str(a.threads)]
    t0 = time.time()
    sys.argv = [os.path.join(RC.MING, "train_mingram.py")] + argv
    runpy.run_path(os.path.join(RC.MING, "train_mingram.py"), run_name="__main__")
    check_model_pretok(M, run)
    RC.dump({"what": "round-2 provenance of this MinGram run", "id": RC.IDS["A10-" + a.size], "g4_twin": a.g4,
             "train_mingram_argv": argv, "pretokenizer": {"id": "P1r3", "oniguruma": M.P1},
             "patch": "mingram.P1 := harness P1r3; mingram._RX reset; build_model labels pretokenizer id P1r3",
             "code_sha256": {"mingram.py": RC.sha256_file(os.path.join(RC.MING, "mingram.py")),
                             "train_mingram.py": RC.sha256_file(os.path.join(RC.MING, "train_mingram.py")),
                             **RC.code_sha("r2_mingram.py", "r2_common.py")},
             "wall_seconds": round(time.time() - t0, 1)}, os.path.join(run, "r2_meta.json"))


def cmd_equiv(a):
    M = patch_mingram()
    RC.verify_frozen()
    run = run_dir(a.size)
    check_model_pretok(M, run)
    out = os.path.join(run, "equivalence.json")
    sys.argv = [os.path.join(RC.MING, "equiv_test.py"), "--run", run, "--out", out]
    runpy.run_path(os.path.join(RC.MING, "equiv_test.py"), run_name="__main__")
    v = json.load(open(out, encoding="utf-8"))
    v["round2"] = {"pretokenizer": "P1r3 (mingram.P1 patched; reference MinGramRef and HF file both use it)",
                   "code_sha256": {"equiv_test.py": RC.sha256_file(os.path.join(RC.MING, "equiv_test.py")),
                                   "mingram.py": RC.sha256_file(os.path.join(RC.MING, "mingram.py")),
                                   **RC.code_sha("r2_mingram.py", "r2_common.py")}}
    RC.dump(v, out)
    RC.log("equivalence verdict", v["verdict"])


def cmd_eval(a):
    M = patch_mingram()
    RC.verify_frozen()
    import adapters as AD
    H = M.H
    cid = RC.IDS["A10-" + a.size]
    run, twin = run_dir(a.size), run_dir(a.size, True)
    check_model_pretok(M, run)
    tok = os.path.join(run, "tokenizer.json")
    g4 = os.path.join(twin, "tokenizer.json")
    eq = json.load(open(os.path.join(run, "equivalence.json"), encoding="utf-8"))
    tl = json.load(open(os.path.join(run, "train_log.json"), encoding="utf-8"))
    tl2 = json.load(open(os.path.join(twin, "train_log.json"), encoding="utf-8"))
    extra = {"label": M.LABEL, "round2_id": cid,
             "encoder_equivalence": {"verdict": eq.get("verdict"), "tokenizer_sha256": eq.get("tokenizer_sha256")},
             "training": {"sizes": tl.get("sizes"), "params_argv": tl.get("argv"), "prune": tl.get("prune"),
                          "outputs_sha256": tl.get("outputs_sha256")}}
    ad = AD.HFAdapter(path=tok, name=cid)
    res = os.path.join(OUT, "results", "dev_strict", cid)
    t0 = time.time()
    s = H.run(ad, data="dev_strict", gates="all", train="train_D1", g4_retrain=g4, out_dir=res, extra=extra)
    nsl = RC.add_nsl(res)
    r1d2 = RC.r1_own_mix(ad, "train_D2")
    g1p = {"docs": 0, "fail": 0}
    docs, _ = H.load_docs("dev_permissive")
    for d in docs:
        g1p["docs"] += 1
        g1p["fail"] += ad.decode(ad.encode(d["text"])) != d["text"]
    outs_equal = {k: tl["outputs_sha256"][k] == tl2["outputs_sha256"].get(k) for k in tl["outputs_sha256"]}
    g = s["gates"]
    chk = {"id": cid, "tokenizer_json": tok, "tokenizer_sha256": RC.sha256_file(tok),
           "gates": {k: g[k].get("pass") for k in ("G1", "G2", "G3", "G4", "G5")},
           "G1_dev_permissive": g1p,
           "G4_all_outputs_identical": outs_equal,
           "encoder_equivalence_verdict": eq.get("verdict"),
           "equivalence_sets": [{k: x.get(k) for k in ("set", "docs", "docs_identical", "tokens_hf", "tokens_ref",
                                                       "g1_roundtrip_hf", "g1_roundtrip_ref",
                                                       "paper_tie_rule_docs_differing",
                                                       "paper_tie_rule_docs_with_token_count_difference",
                                                       "min_token_check_failures")} for x in eq["sets"]],
           "R1_train_D1": s["properties"]["R1"], "R1_own_training_mix_train_D2": r1d2,
           "R2": {k: s["properties"]["R2"][k] for k in ("partial_utf8_tokens", "with_train_freq_0")},
           "nsl_vs_A1-P1-D1-16k": nsl.get("overall"),
           "train_log": {"sizes": tl["sizes"], "iterations": tl["iterations"], "prune": tl["prune"],
                         "final_on_train": tl["final_on_train"], "timings_s": tl["timings_s"]},
           "code_sha256": RC.code_sha("r2_mingram.py", "r2_common.py"), "seconds": round(time.time() - t0, 1)}
    RC.dump(chk, os.path.join(res, "r2_checks.json"))
    RC.log(cid, "gates", chk["gates"], "bytes/token", s["metrics"]["overall"]["bytes_per_token"],
           "G4 outputs", outs_equal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "equiv", "eval"])
    ap.add_argument("--size", required=True, choices=sorted(RC.SIZES))
    ap.add_argument("--g4", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if a.threads > 1 else "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    {"train": cmd_train, "equiv": cmd_equiv, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
