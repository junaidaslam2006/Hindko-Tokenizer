# -*- coding: utf-8 -*-
"""Run the Stage-0 evaluation harness (eval/harness.py, unchanged) on dev_strict for the A1 reference and for
PickyBPE, with all gates (G1-G5), R1 on train_D1, R2, morphology (silver, report-only) and NSL vs the A1
reference. Results are written under candidates/pickybpe/results/dev_strict/<name>/ (docs.jsonl + summary.json).

    python run_harness.py [--only a1|picky]
G4 needs a second training with identical inputs: models/*.retrain.json (made by train_pickybpe.py /
build_a1_ref.py with the same arguments).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = r"F:\Hindko\_tokenizer\eval"
sys.path.insert(0, HERE)
sys.path.insert(0, EVAL)
os.environ.setdefault("HF_HOME", r"F:\Hindko\_tokenizer\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import harness as H  # noqa: E402

H._setup_threads(1)
import adapters as A  # noqa: E402
import pickybpe_factory as F  # noqa: E402

RES = os.path.join(HERE, "results", "dev_strict")
A1 = os.path.join(HERE, "models", "a1_ref_P1_D1_16k.json")
A1_RETRAIN = os.path.join(HERE, "models", "a1_ref_P1_D1_16k.retrain.json")
A1_NAME = "A1ref_bpe_P1_D1_16k"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["a1", "picky"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    a1_dir = os.path.join(RES, A1_NAME)
    if a.only in (None, "a1"):
        ad = A.HFAdapter(path=A1, name=A1_NAME)
        H.run(ad, data="dev_strict", gates="all", train="train_D1",
              g4_retrain=A1_RETRAIN if os.path.exists(A1_RETRAIN) else None,
              nsl_ref=None, out_dir=a1_dir, force=a.force,
              extra={"note": "A1 reference built by candidates/pickybpe/build_a1_ref.py (PLAN A1 recipe) for the "
                             "PickyBPE comparison; NSL reference = itself"})
        H.main(["nsl", "--ref", a1_dir, a1_dir])
    if a.only in (None, "picky"):
        tk = F.load_16k()
        ad = A.CustomAdapter(tk, name=F.NAME_16K)
        ad.path = tk.path                      # harness G4 reads adapter.path
        retrain = F.MODEL_16K[:-5] + ".retrain.json"
        H.run(ad, data="dev_strict", gates="all", train="train_D1",
              g4_retrain=retrain if os.path.exists(retrain) else None,
              nsl_ref=a1_dir, out_dir=os.path.join(RES, F.NAME_16K), force=a.force,
              extra={"note": "custom event-ordered PickyBPE encoder; NSL reference = A1 reference (%s)" % A1})


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
