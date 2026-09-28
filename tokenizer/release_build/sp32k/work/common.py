# -*- coding: utf-8 -*-
"""Shared helpers for the sp32k release build (R2-A4-SPnat-D2-32k)."""
import hashlib
import json
import os
import sys

TOK = r"F:\Hindko\_tokenizer"
sys.path.insert(0, os.path.join(TOK, "eval"))
CAND = os.path.join(TOK, r"candidates\round2\standard\tok\R2-A4-SPnat-D2-32k")
OLD_MODEL = os.path.join(CAND, "sp.model")
OLD_JSON = os.path.join(CAND, "tokenizer.json")
OUT = os.path.join(TOK, r"release_build\sp32k")
WORK = os.path.join(OUT, "work")
DATA = os.path.join(TOK, "data")
SETS = ["dev_strict", "dev_permissive", "test_strict", "train_D2"]
SP_MARK = "\u2581"
NL = "\n"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_set(name):
    """[(uid, text)] of a data view. test_strict is read as TEXT only (encoding-equivalence checks)."""
    p = os.path.join(DATA, name + ".jsonl")
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            out.append((r["uid"], r["text"]))
    return out, sha256_file(p)


def load_proto(path):
    from sentencepiece import sentencepiece_model_pb2 as spb
    m = spb.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m, spb.ModelProto.SentencePiece
