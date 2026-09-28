# -*- coding: utf-8 -*-
"""Shared code of the out-of-distribution / robustness study (sota/robustness/).

Tokenizers (11): the released Hindko tokenizer, the runner-up MinGram-48k, and 9 external tokenizers loaded through the
study's own baselines/load_baselines.py (offline files, native encoders, no BOS/EOS). Metrics reuse eval/harness.py
(word geometry, fertility from in-context offsets, segmentation comparison) and eval/perturb.py unchanged.
All codepoints are built with chr(); this file stays ASCII-safe.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = r"F:\Hindko\_tokenizer"
EVAL = os.path.join(TOK, "eval")
BASELINES = os.path.join(TOK, "baselines")
PIPELINE = r"F:\Hindko\_pipeline"
RELEASE = r"F:\Hindko\tokenizer"
SETS = os.path.join(HERE, "sets")
RESULTS = os.path.join(HERE, "results")

os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

for p in (EVAL, BASELINES, PIPELINE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

RELEASED_JSON = os.path.join(RELEASE, "tokenizer.json")
RELEASED_SHA = "49f301c52363a09a1fc1925359af3ddabd7887495de02354cfa4092cba51da41"
MINGRAM48_JSON = os.path.join(TOK, "candidates", "round2", "mingram", "runs", "mingram_P1r3_D2_49152", "tokenizer.json")
TEST = os.path.join(TOK, "data", "test_strict.jsonl")
TEST_SHA = "a74c33ad008f4b0e4a6f7779b0de24e265b0f4edf6f94e89885c7943d3a08468"

# (key, display name, family / provider, loader)
TOKENIZERS = [
    ("released", "Hindko SP-32k (released)", "this study: SentencePiece Unigram 32,768 (R2-A4-SPnat-D2-32k)", "release"),
    ("mingram48k", "MinGram-48k (runner-up)", "this study: MinGram 49,152 (R2-A10-MinGram-P1r3-D2-48k), pre-registered pick", "mingram"),
    ("gpt-4o", "GPT-4o (o200k)", "OpenAI o200k_base, byte-level BPE 200k", "baseline"),
    ("gemma-3", "Gemma 3", "Google Gemma 3 SentencePiece BPE 262k (= Gemma 4 encodings)", "baseline"),
    ("llama-3", "Llama 3", "Meta Llama 3 tiktoken BPE 128k", "baseline"),
    ("llama-4", "Llama 4", "Meta Llama 4 BPE 202k", "baseline"),
    ("qwen-3.5", "Qwen 3.5", "Alibaba Qwen3.5 byte-level BPE 248k", "baseline"),
    ("deepseek-v3", "DeepSeek-V3", "DeepSeek-V3 byte-level BPE 128k (= V4 encodings)", "baseline"),
    ("bloom", "BLOOM", "BigScience BLOOM byte-level BPE 250k", "baseline"),
    ("roberta-urdu", "RoBERTa-Urdu", "UrduHack RoBERTa-Urdu byte-level BPE 52k (best external on Hindko test)", "baseline"),
    ("mt5", "mT5", "Google mT5 SentencePiece Unigram 250k (normalizes: not lossless)", "baseline"),
]
TOK_KEYS = [t[0] for t in TOKENIZERS]
DISPLAY = {t[0]: t[1] for t in TOKENIZERS}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_tokenizer(key: str):
    import adapters as A
    kind = dict((t[0], t[3]) for t in TOKENIZERS)[key]
    if kind == "release":
        if sha256_file(RELEASED_JSON) != RELEASED_SHA:
            raise SystemExit("released tokenizer.json sha256 differs from RELEASE_MANIFEST.json")
        return A.HFAdapter(path=RELEASED_JSON, name=key)
    if kind == "mingram":
        return A.HFAdapter(path=MINGRAM48_JSON, name=key)
    return A.from_baseline(key)


def normalize(text: str) -> str:
    from hp.normalize import normalize as _n
    return _n(text)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_test():
    if sha256_file(TEST) != TEST_SHA:
        raise SystemExit("test_strict.jsonl sha256 differs from data/test_manifest.json")
    return read_jsonl(TEST)


# --------------------------------------------------------------------------------------- 8-gram keys
HARAKAT = frozenset(list(range(0x064B, 0x0660)) + [0x0670])
KEY_FOLD = {chr(0x064A): chr(0x06CC), chr(0x0649): chr(0x06CC), chr(0x0643): chr(0x06A9),
            chr(0x0647): chr(0x06C1), chr(0x0629): chr(0x06C3), chr(0x0640): ""}
WORD_KEY_RE = re.compile(r"[^\W\d_]+")


def key_words(text: str):
    """Words for the n-gram overlap test: NFC, harakat / format characters / kashida removed, Arabic-keyboard
    letters folded to Urdu ones, then maximal runs of letters (digits and punctuation are separators)."""
    t = unicodedata.normalize("NFC", text)
    out = []
    for c in t:
        o = ord(c)
        if o in HARAKAT or unicodedata.category(c) == "Cf":
            continue
        out.append(KEY_FOLD.get(c, c))
    return WORD_KEY_RE.findall("".join(out))


def ngram_hashes(words, n=8):
    if len(words) < n:
        return np.zeros(0, dtype=np.uint64)
    hs = [int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode("utf-8"), digest_size=8).digest(), "little")
          for i in range(len(words) - n + 1)]
    return np.asarray(hs, dtype=np.uint64)
