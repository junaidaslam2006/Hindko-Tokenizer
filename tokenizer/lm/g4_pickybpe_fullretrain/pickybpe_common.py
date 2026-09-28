# -*- coding: utf-8 -*-
"""Shared constants and verified data access for the PickyBPE candidate (PLAN.md A7).

Everything here is read-only with respect to the released data. The test split is never read:
the materialised views used here (data/train_D1.jsonl, data/dev_strict.jsonl) contain no test rows,
and load_view() re-checks every uid against the frozen split manifest and refuses test uids.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

TOK = r"F:\Hindko\_tokenizer"
HERE = os.path.join(TOK, "candidates", "pickybpe")
DATA_DIR = os.path.join(TOK, "data")
FROZEN_PATH = os.path.join(TOK, "FROZEN.json")
MANIFEST_SHA256 = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"
PIPELINE_DIR = r"F:\Hindko\_pipeline"

# PLAN 2.1 P1, Oniguruma syntax (HF tokenizers) and its Python `regex` twin. The Python form is built with
# chr() so that no tool can turn the escapes into literal characters by accident.
P1_ONIG = r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
P1_PY = " ?[\\p{L}\\p{M}" + "\\u200C" + "\\u200D" + "]+| ?\\p{N}| ?[^\\s\\p{L}\\p{N}\\p{M}]+|\\s+(?!\\S)|\\s+"
assert "\u200c" not in P1_PY and "\\u200C" in P1_PY

# PLAN 2.1: the special-token block, 64 in total, identical for every candidate
SPECIAL_TOKENS = (["<|endoftext|>", "<|bos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]
                  + ["<|reserved_%d|>" % i for i in range(59)])
assert len(SPECIAL_TOKENS) == 64


def bytes_to_unicode():
    """GPT-2 byte <-> printable-unicode table (the HF ByteLevel alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


_SPLIT = None


def verify_frozen() -> dict:
    """Split manifest and normalize.py must match FROZEN.json (and the manifest the task names)."""
    fr = json.load(open(FROZEN_PATH, encoding="utf-8"))
    man = fr["split_manifest"]["path"]
    got = sha256_file(man)
    if got != MANIFEST_SHA256 or got != fr["split_manifest"]["sha256"]:
        raise SystemExit("split manifest sha256 %s != frozen %s" % (got, MANIFEST_SHA256))
    norm = fr["normalize"]["path"]
    ns = sha256_file(norm)
    if ns != fr["normalize"]["sha256"]:
        raise SystemExit("normalize.py sha256 %s != frozen %s" % (ns, fr["normalize"]["sha256"]))
    return {"split_manifest_sha256": got, "normalize_sha256": ns, "normalize_version": fr["normalize"]["version"]}


def manifest_split() -> dict:
    global _SPLIT
    if _SPLIT is None:
        verify_frozen()
        fr = json.load(open(FROZEN_PATH, encoding="utf-8"))
        _SPLIT = {}
        with open(fr["split_manifest"]["path"], encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _SPLIT[r["uid"]] = r["split"]
    return _SPLIT


def load_view(name: str, expect_split: str) -> (list, dict):
    """A materialised view (train_D1 / dev_strict): sha256 checked against data/data_manifest.json, every uid
    checked against the frozen manifest (must be `expect_split`; a test uid aborts)."""
    if "test" in name:
        raise SystemExit("REFUSED: test data is not used in this run")
    path = os.path.join(DATA_DIR, name + ".jsonl")
    dm = json.load(open(os.path.join(DATA_DIR, "data_manifest.json"), encoding="utf-8"))
    sha = sha256_file(path)
    if dm["views"][name]["sha256"] != sha:
        raise SystemExit("%s sha256 does not match data_manifest.json" % path)
    split = manifest_split()
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = split.get(r["uid"])
            if s == "test":
                raise SystemExit("REFUSED: %s contains a test uid" % path)
            if s != expect_split:
                raise SystemExit("%s: uid %s is in split %r, expected %r" % (path, r["uid"], s, expect_split))
            docs.append(r)
    return docs, {"view": name, "path": path, "sha256": sha, "docs": len(docs),
                  "normalize_version_of_view": dm["normalize_version"], "manifest_sha256": dm["split_manifest_sha256"]}


def check_canonical(docs) -> dict:
    """PLAN/task: every text must be in the canonical form of hp.normalize 1.0.1. The views were materialised
    through normalize(); re-apply it and require a fixed point (normalize is idempotent)."""
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    from hp import normalize as N
    if N.NORMALIZATION_VERSION != "1.0.1":
        raise SystemExit("normalize version %s != 1.0.1" % N.NORMALIZATION_VERSION)
    t0 = time.time()
    bad = [d["uid"] for d in docs if N.normalize(d["text"]) != d["text"]]
    if bad:
        raise SystemExit("%d documents are not in canonical form, e.g. %s" % (len(bad), bad[:5]))
    return {"normalize_version": N.NORMALIZATION_VERSION, "docs_checked": len(docs), "not_canonical": 0,
            "seconds": round(time.time() - t0, 1)}
