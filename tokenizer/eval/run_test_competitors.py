# -*- coding: utf-8 -*-
"""PLAN Stage 5 (one-shot final test), step 5: INTRINSIC metrics on the strict TEST split, computed ONCE,
for the chosen tokenizer and every working external baseline of baselines/manifest.json.

    set PYTHONIOENCODING=utf-8
    python eval\\run_test_competitors.py --shard K --of 3      (K = 0, 1, 2 in parallel; 1 thread each)
    python eval\\report_test_competitors.py                    (-> eval/TEST_COMPETITORS.md, eval/test_competitors.json)

Same code path as the Stage 0 dev baselines (eval/run_baselines.py): harness.run(adapter, data, gates='g1')
-> bytes/token, chars/token, fertility, STRR, G1 (+ UNK, lines bytes/token, robustness, report-only
morphology), with the ONE difference that the data is data/test_strict.jsonl and allow_test=True.
The test file is checked against data/test_manifest.json before anything is encoded.
The chosen tokenizer (fixed in analysis/decision.json before the test split was opened) runs in shard 0.
Results: eval/results/test_strict/<name>/{docs.jsonl, summary.json}. Resumable: harness.run() skips a result
whose tokenizer, harness code and dataset sha256 match. Errors are logged and the shard moves on.
These numbers are reported, never used to change a decision.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import harness as H  # noqa: E402

H._setup_threads(1)
import adapters as A  # noqa: E402

sys.path.insert(0, A.BASELINES_DIR)
from load_baselines import list_baselines  # noqa: E402

TEST = os.path.join(TOK, "data", "test_strict.jsonl")
TEST_MANIFEST = os.path.join(TOK, "data", "test_manifest.json")
CHOSEN = "R2-A10-MinGram-P1r3-D2-48k"
# measured in the Stage 0 smoke runs: these take 2-3x longer than a tokenizer.json baseline
SLOW = {"mt5", "muril", "roberta-urdu", "hindko-probe-bpe32k", "xlm-r", "sindhi-xlmr", "indicbert-v2", "nllb-200"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chosen_adapter():
    dec = json.load(open(os.path.join(TOK, "analysis", "decision.json"), encoding="utf-8"))
    if dec["test_candidates_fixed"]["ids"][0] != CHOSEN:
        raise SystemExit("the chosen tokenizer in analysis/decision.json is not %s" % CHOSEN)
    w = json.load(open(os.path.join(TOK, "lm", "WAVES_final_test.json"), encoding="utf-8"))
    e = next(x for x in w["waves"] if x["id"] == CHOSEN)
    if e["final_test_role"] != "chosen" or sha(e["tokenizer_path"]) != e["tokenizer_sha256"]:
        raise SystemExit("WAVES_final_test.json entry of %s does not match its tokenizer file" % CHOSEN)
    return A.HFAdapter(path=e["tokenizer_path"], name=CHOSEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    a = ap.parse_args()
    tm = json.load(open(TEST_MANIFEST, encoding="utf-8"))
    if sha(TEST) != tm["views"]["test_strict"]["sha256"]:
        raise SystemExit("data/test_strict.jsonl does not match data/test_manifest.json")
    names = list_baselines(working_only=True)
    order = [n for n in names if n in SLOW] + [n for n in names if n not in SLOW]   # spread slow ones over shards
    mine = order[a.shard::a.of]
    H.log("shard %d/%d: %d baselines%s: %s" % (a.shard, a.of, len(mine), " + the chosen tokenizer" if a.shard == 0 else "",
                                               " ".join(mine)))
    err_path = os.path.join(HERE, "logs", "test_competitors_errors_shard%d.jsonl" % a.shard)
    os.makedirs(os.path.dirname(err_path), exist_ok=True)
    jobs = ([("chosen", CHOSEN)] if a.shard == 0 else []) + [("baseline", n) for n in mine]
    for kind, name in jobs:
        t0 = time.time()
        try:
            ad = chosen_adapter() if kind == "chosen" else A.from_baseline(name)
        except Exception as e:  # noqa: BLE001
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tokenizer": name, "stage": "load", "error": repr(e),
                                    "trace": traceback.format_exc()}) + "\n")
            H.log("LOAD ERROR", name, repr(e))
            continue
        try:
            H.run(ad, data=TEST, gates="g1", allow_test=True,
                  extra={"one_shot_final_test": "PLAN 7.6 / Stage 5; reported, never used to change a decision",
                         "role": kind})
        except Exception as e:  # noqa: BLE001
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tokenizer": name, "error": repr(e), "trace": traceback.format_exc()}) + "\n")
            H.log("RUN ERROR", name, repr(e))
        H.log("done", name, "%.0fs" % (time.time() - t0))
    H.log("shard %d finished" % a.shard)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
