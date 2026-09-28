# -*- coding: utf-8 -*-
"""Round 2 (exploratory refinement, PLAN 2.1 'LM stage: {8k,16k,32k} plus refinements'): shared constants.

Every round-2 tokenizer is trained on D2 (strict train, data/train_D2.jsonl or its lines file) with the Stage-1
pre-tokenizer P1r3 (A4 SentencePiece: its native pre-tokenization with split_digits, as in the Stage 1 sweep),
the 64-token special block at ids 0..63, no normalizer, and a total vocabulary of exactly the nominal size.
The implementations are the reviewed ones in candidates/standard, candidates/superbpe, candidates/mingram and
eval/harness.py; the r2_*.py drivers only point them at P1r3 / D2 and at output paths under candidates/round2.
Nothing here reads the test split (harness.load_docs and superbpe common.load_view refuse it).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

TOK = r"F:\Hindko\_tokenizer"
R2 = os.path.join(TOK, "candidates", "round2")
EVAL = os.path.join(TOK, "eval")
STD = os.path.join(TOK, "candidates", "standard")
SBPE = os.path.join(TOK, "candidates", "superbpe")
MING = os.path.join(TOK, "candidates", "mingram")
LOGS = os.path.join(R2, "logs")
MANIFEST_SHA256 = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"
NSL_REF = os.path.join(STD, "results", "dev_strict", "A1-P1-D1-16k")          # the sweep's NSL reference
BASELINE_ID = "A1-P1r3-D2-16k"                                                   # Stage 3 baseline (rank 1)

# the stage-1 decision (SWEEP.md 1.1) as stored in eval/harness.py; SuperBPE's stage-2 regex is DERIVED from the
# reviewed P1-based S2 by the same digit substitution that turns P1 into P1r3 (asserted in r2_superbpe.py)
P1_DIGIT = r"| ?\p{N}|"
P1R3_DIGIT = r"| ?\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|"

IDS = {
    "A10-32k": "R2-A10-MinGram-P1r3-D2-32k",
    "A10-48k": "R2-A10-MinGram-P1r3-D2-48k",
    "A4-32k": "R2-A4-SPnat-D2-32k",
    "A4-48k": "R2-A4-SPnat-D2-48k",
    "A6-32k": "R2-A6-SBPE-P1r3-D2-32k-t080",
    "A1-48k": "R2-A1-P1r3-D2-48k",
}
SIZES = {"32k": 32768, "48k": 49152}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def verify_frozen():
    fr = json.load(open(os.path.join(TOK, "FROZEN.json"), encoding="utf-8"))
    got = sha256_file(fr["split_manifest"]["path"])
    if got != MANIFEST_SHA256 or fr["split_manifest"]["sha256"] != MANIFEST_SHA256:
        raise SystemExit("split manifest sha256 %s != frozen %s" % (got, MANIFEST_SHA256))
    if sha256_file(fr["normalize"]["path"]) != fr["normalize"]["sha256"]:
        raise SystemExit("normalize.py differs from FROZEN.json")
    return {"split_manifest_sha256": got, "normalize_sha256": fr["normalize"]["sha256"],
            "normalize_version": fr["normalize"]["version"]}


def code_sha(*names):
    return {n: sha256_file(os.path.join(R2, n)) for n in names}


def dump(obj, path):
    sys.path.insert(0, EVAL)
    import harness as H
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(H.clean(obj), f, ensure_ascii=False, indent=1)


def r1_own_mix(adapter, view="train_D2"):
    """PLAN 4.3 R1 on the tokenizer's own training text (D2), as run_sweep.process does for D2/D3 tokenizers.
    The harness summary's R1 is on train_D1, the LM stage's training stream (colab/build_bundle.py)."""
    sys.path.insert(0, EVAL)
    import harness as H
    cnt, _, info = H.train_counts(adapter, view)
    return H.prop_r1(adapter, cnt, info)


def add_nsl(summary_dir):
    """harness.py nsl --ref <A1-P1-D1-16k> <dir>: NSL against the sweep's reference, written into summary.json."""
    sys.path.insert(0, EVAL)
    import harness as H
    sp = os.path.join(summary_dir, "summary.json")
    s = json.load(open(sp, encoding="utf-8"))
    s["nsl"] = H.nsl_from_docs(os.path.join(summary_dir, "docs.jsonl"), os.path.join(NSL_REF, "docs.jsonl"))
    json.dump(H.clean(s), open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return s["nsl"]
