# -*- coding: utf-8 -*-
"""Shared constants and helpers for the SuperBPE candidate (PLAN.md Stage 2, A6).

SuperBPE as reimplemented from Liu et al. 2025, "SuperBPE: Space Travel for Language Models" (COLM 2025).
No third-party SuperBPE code is used (the authors' HF fork needs Rust and is not installed; nothing was
cloned or pip-installed).

Pre-tokenizers (Oniguruma syntax, as stored in tokenizer.json; the Python `regex` twin is derived with
onig_to_py(), which turns \\x{HHHH} into \\uHHHH):
  P1  stage 1 (whitespace-aware, PLAN 2.1 default)
  S2  stage 2: identical to P1 except that its first alternative matches a RUN of letter-words joined by
      single spaces. Chunks therefore end only at punctuation/symbols, digits, line breaks (and at runs of
      two or more spaces, which the canonical form does not contain). Superword tokens can span spaces but
      never punctuation (sentence boundaries), digits or line breaks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

TOK = r"F:\Hindko\_tokenizer"
HERE = os.path.join(TOK, "candidates", "superbpe")
DATA_DIR = os.path.join(TOK, "data")
EVAL_DIR = os.path.join(TOK, "eval")
FROZEN_PATH = os.path.join(TOK, "FROZEN.json")
MANIFEST_SHA256 = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"
NORMALIZE_VERSION = "1.0.1"

os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

P1 = r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
S2 = (r" ?[\p{L}\p{M}\x{200C}\x{200D}]+(?: [\p{L}\p{M}\x{200C}\x{200D}]+)*"
      r"| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+")

SPECIAL_TOKENS = (["<|endoftext|>", "<|bos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]
                  + ["<|reserved_%d|>" % i for i in range(59)])          # PLAN 2.1: 64 in total
assert len(SPECIAL_TOKENS) == 64
N_SPECIAL = 64
N_BYTES = 256
MAX_WORDS = 4                     # Liu et al. 2025: superword tokens hold at most 4 words
PUA_BASE = 0xF0000                # plane-15 Private Use Area (65,534 code points)
MIN_FREQUENCY = 2                 # as A1 (PLAN 2.1; gate_audit.py / test_harness.py)


def onig_to_py(p: str) -> str:
    """Oniguruma \\x{HHHH} escapes -> Python `regex` \\uHHHH escapes (same rule as eval/harness.py)."""
    return re.sub(r"\\x\{([0-9A-Fa-f]{4})\}", lambda m: "\\u" + m.group(1), p)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def bytes_to_unicode():
    """GPT-2 byte <-> printable-unicode table (HF ByteLevel)."""
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


def bl_to_bytes(s: str) -> bytes:
    return bytes(U2B[c] for c in s)


def bytes_to_bl(b: bytes) -> str:
    return "".join(B2U[x] for x in b)


def n_words(b: bytes) -> int:
    """Whitespace-delimited words in a token's bytes (runs of non-space bytes)."""
    return len([w for w in b.split(b" ") if w])


# ------------------------------------------------------------------------------------------ frozen inputs
def verify_frozen() -> dict:
    """Split manifest + normalize.py + data views must match FROZEN.json / data_manifest.json."""
    fr = json.load(open(FROZEN_PATH, encoding="utf-8"))
    man = fr["split_manifest"]["path"]
    got = sha256_file(man)
    if got != MANIFEST_SHA256 or fr["split_manifest"]["sha256"] != MANIFEST_SHA256:
        raise SystemExit("split manifest sha256 %s != frozen %s" % (got, MANIFEST_SHA256))
    norm = fr["normalize"]["path"]
    if sha256_file(norm) != fr["normalize"]["sha256"]:
        raise SystemExit("normalize.py differs from FROZEN.json")
    dm = json.load(open(os.path.join(DATA_DIR, "data_manifest.json"), encoding="utf-8"))
    if dm["split_manifest_sha256"] != MANIFEST_SHA256 or dm["normalize_version"] != NORMALIZE_VERSION:
        raise SystemExit("data_manifest.json was not built on the frozen manifest / normalize version")
    return {"split_manifest_sha256": got, "normalize_version": fr["normalize"]["version"],
            "normalize_sha256": fr["normalize"]["sha256"], "data_manifest": dm}


_SPLIT = None


def manifest_split():
    global _SPLIT
    if _SPLIT is None:
        path = json.load(open(FROZEN_PATH, encoding="utf-8"))["split_manifest"]["path"]
        _SPLIT = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _SPLIT[r["uid"]] = r["split"]
    return _SPLIT


def load_view(name: str, check_normalized: bool = True):
    """Materialised, normalised view (train_D1 / dev_strict / dev_permissive). Refuses test-split rows,
    checks the file against data_manifest.json and (optionally) that normalize(text) == text."""
    if "test" in name:
        raise SystemExit("REFUSED: test split is used once, in PLAN Stage 5")
    path = os.path.join(DATA_DIR, name + ".jsonl")
    dm = json.load(open(os.path.join(DATA_DIR, "data_manifest.json"), encoding="utf-8"))
    sha = sha256_file(path)
    if dm["views"][name]["sha256"] != sha:
        raise SystemExit("%s does not match data_manifest.json" % path)
    docs = [json.loads(l) for l in open(path, encoding="utf-8")]
    split = manifest_split()
    want = "train" if name.startswith("train") else "validation"
    bad = [d["uid"] for d in docs if split.get(d["uid"]) != want]
    if bad:
        raise SystemExit("%s: %d rows not in split %s (e.g. %s)" % (name, len(bad), want, bad[:3]))
    n_changed = None
    if check_normalized:
        sys.path.insert(0, r"F:\Hindko\_pipeline")
        from hp.normalize import normalize, NORMALIZATION_VERSION
        assert NORMALIZATION_VERSION == NORMALIZE_VERSION, NORMALIZATION_VERSION
        n_changed = sum(1 for d in docs if normalize(d["text"]) != d["text"])
        if n_changed:
            raise SystemExit("%s: %d texts are not in canonical form" % (name, n_changed))
    return docs, {"view": name, "path": path, "sha256": sha, "docs": len(docs),
                  "normalize_fixed_point_violations": n_changed}


def hf_bytelevel_bpe_json(vocab: dict, merges: list, pre_regex: str) -> dict:
    """tokenizer.json dict of an A1-style byte-level BPE (Split(regex, isolated) + ByteLevel(use_regex=False),
    ByteLevel decoder, no normalizer, no post-processor) with the 64-special block at ids 0..63."""
    added = [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for i, s in enumerate(SPECIAL_TOKENS)]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pre_regex}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [list(m) for m in merges]}}
