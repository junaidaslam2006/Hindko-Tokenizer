# -*- coding: utf-8 -*-
"""PLAN Stage 0, step 5: run the harness on EVERY working external baseline of baselines/manifest.json
over dev_strict (the evaluation set) and dev_permissive (reporting), gates = G1 (+ R2, morphology).

    python run_baselines.py --shard K --of N      one process per shard (K = 0..N-1), 1 thread each
Resumable: harness.run() skips a (baseline, dataset) whose summary.json matches the tokenizer file,
the harness code and the dataset. Errors are logged and the shard moves on.
Results: eval/results/<dataset>/<baseline>/{docs.jsonl, summary.json}; the report is report_baselines.py.
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402

H._setup_threads(1)
import adapters as A  # noqa: E402

sys.path.insert(0, A.BASELINES_DIR)
from load_baselines import list_baselines  # noqa: E402

# measured in the smoke runs: these take 2-3x longer than a tokenizer.json baseline
SLOW = {"mt5", "muril", "roberta-urdu", "hindko-probe-bpe32k", "xlm-r", "sindhi-xlmr", "indicbert-v2", "nllb-200"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--data", nargs="+", default=["dev_strict", "dev_permissive"])
    a = ap.parse_args()
    names = list_baselines(working_only=True)
    order = [n for n in names if n in SLOW] + [n for n in names if n not in SLOW]   # spread slow ones over shards
    mine = order[a.shard::a.of]
    H.log("shard %d/%d: %d baselines: %s" % (a.shard, a.of, len(mine), " ".join(mine)))
    err_path = os.path.join(HERE, "logs", "baselines_errors_shard%d.jsonl" % a.shard)
    os.makedirs(os.path.dirname(err_path), exist_ok=True)
    for name in mine:
        try:
            ad = A.from_baseline(name)
        except Exception as e:  # noqa: BLE001
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"baseline": name, "stage": "load", "error": repr(e),
                                    "trace": traceback.format_exc()}) + "\n")
            H.log("LOAD ERROR", name, repr(e))
            continue
        for data in a.data:
            t0 = time.time()
            try:
                H.run(ad, data=data, gates="g1")
            except Exception as e:  # noqa: BLE001
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"baseline": name, "data": data, "error": repr(e),
                                        "trace": traceback.format_exc()}) + "\n")
                H.log("RUN ERROR", name, data, repr(e))
            H.log("done", name, data, "%.0fs" % (time.time() - t0))
    H.log("shard %d finished" % a.shard)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
