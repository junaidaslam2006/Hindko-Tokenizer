# -*- coding: utf-8 -*-
"""Independent third encoder + float32 bound audit for the sp32k release.

A pure-Python Viterbi over the release sp.model that mirrors sentencepiece 0.2.1 unigram EncodeOptimized:
  line -> U+2581 + line with ' ' -> U+2581 (empty line -> no tokens); lattice over characters; every NORMAL piece
  (max 16 chars) is a candidate; a character without a single-character piece gets an <unk> node with score
  min_normal_score - 10, emitted as its UTF-8 byte pieces (byte_fallback); at each end position the FIRST candidate
  (smallest start) is kept unless a later one is STRICTLY greater.
Scores are multiples of 2**-14, so float64 arithmetic here is exact. For every line it records the largest
magnitude of any value SentencePiece compares (every candidate score and every node score). While that stays below
2**10 = 1024, every such value is exact in float32 too, so sentencepiece's float32 Viterbi and this exact one (and
HF tokenizers' float64 one) must return the same path.
Outputs release_build/sp32k/exact_check.json. Usage: python exact_check.py  (3 worker processes)
"""
import json
import os
import sys
from multiprocessing import Pool

TOK = r"F:\Hindko\_tokenizer"
REL = os.path.join(TOK, r"release_build\sp32k")
DATA = os.path.join(TOK, "data")
SETS = ["dev_strict", "dev_permissive", "test_strict", "train_D2"]
BOUND = 1024.0
MARK = chr(0x2581)
G = {}


def init():
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as spb
    m = spb.ModelProto()
    m.ParseFromString(open(os.path.join(REL, "sp.model"), "rb").read())
    T = spb.ModelProto.SentencePiece
    G["score"] = {p.piece: (i, p.score) for i, p in enumerate(m.pieces) if p.type == T.NORMAL}
    G["byte"] = {int(p.piece[3:5], 16): i for i, p in enumerate(m.pieces) if p.type == T.BYTE}
    G["unk"] = min(s for _, s in G["score"].values()) - 10.0
    G["maxlen"] = max(len(p) for p in G["score"])
    G["sp"] = spm.SentencePieceProcessor(model_file=os.path.join(REL, "sp.model"))


def viterbi(line):
    if not line:
        return [], 0.0
    s = MARK + line.replace(" ", MARK)
    n = len(s)
    sc, unk, ML = G["score"], G["unk"], G["maxlen"]
    best = [None] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0.0
    mx = 0.0
    for st in range(n):
        b0 = best[st]
        single = False
        for L in range(1, min(ML, n - st) + 1):
            e = sc.get(s[st:st + L])
            if e is None:
                continue
            if L == 1:
                single = True
            c = b0 + e[1]
            if abs(c) > mx:
                mx = abs(c)
            en = st + L
            if best[en] is None or c > best[en]:
                best[en] = c; back[en] = (st, e[0])
        if not single:
            c = b0 + unk
            if abs(c) > mx:
                mx = abs(c)
            if best[st + 1] is None or c > best[st + 1]:
                best[st + 1] = c; back[st + 1] = (st, None)
    ids = []
    en = n
    while en > 0:
        st, i = back[en]
        if i is None:
            ids.extend(reversed([G["byte"][b] for b in s[st:en].encode("utf-8")]))
        else:
            ids.append(i)
        en = st
    return ids[::-1], mx


def work(lines):
    out = []
    enc = G["sp"].encode(lines)
    for line, e in zip(lines, enc):
        ids, mx = viterbi(line)
        out.append((ids == e, mx, len(line)))
    return out


def main():
    res = {"bound": BOUND, "rule": "max |value compared by the Viterbi| < 2^10 => float32 exact => sentencepiece == "
                                   "exact arithmetic == HF tokenizers"}
    with Pool(3, initializer=init) as pool:
        for name in SETS:
            with open(os.path.join(DATA, name + ".jsonl"), encoding="utf-8") as f:
                lines = [l for r in f for l in json.loads(r)["text"].split("\n")]
            chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
            rows = [x for part in pool.map(work, chunks) for x in part]
            eq = sum(1 for r in rows if r[0])
            over = [r for r in rows if r[1] >= BOUND]
            res[name] = {"lines": len(rows), "exact_viterbi_equals_sentencepiece": eq,
                         "max_abs_compared_value": max(r[1] for r in rows),
                         "lines_within_bound": len(rows) - len(over), "lines_over_bound": len(over),
                         "lines_over_bound_min_chars": min((r[2] for r in over), default=None),
                         "lines_over_bound_all_equal": all(r[0] for r in over)}
            print(name, json.dumps(res[name]), flush=True)
    with open(os.path.join(REL, "exact_check.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
