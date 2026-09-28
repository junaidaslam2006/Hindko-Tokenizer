# -*- coding: utf-8 -*-
"""Fit a per-character score offset w(c) (a 'gauge' shift: s'(p) = s(p) + sum_{c in p} w(c)) that keeps the running
Viterbi path score near 0, and measure the resulting magnitude of best-path partial sums on every set.
Fit: least squares over the piece occurrences of the train_D2 best paths, min sum_t (s(t) + sum_{c in t} w(c))^2.
Writes work/gauge_w.json."""
import json, os, sys, time
import numpy as np
from common import *
import sentencepiece as spm

m, T = load_proto(OLD_MODEL)
pieces = list(m.pieces)
normal_ids = [i for i, p in enumerate(pieces) if p.type == T.NORMAL]
chars = sorted({c for i in normal_ids for c in pieces[i].piece})
cidx = {c: k for k, c in enumerate(chars)}
C = len(chars)
F = np.zeros((len(pieces), C))
for i in normal_ids:
    for c in pieces[i].piece:
        F[i, cidx[c]] += 1
score = np.array([p.score for p in pieces], dtype=np.float64)
is_normal = np.zeros(len(pieces), bool); is_normal[normal_ids] = True
sp = spm.SentencePieceProcessor(model_file=OLD_MODEL)

docs, _ = load_set("train_D2")
lines = [l for _, t in docs for l in t.split("\n") if l]
enc = sp.encode(lines)
cnt = np.zeros(len(pieces))
for e in enc:
    np.add.at(cnt, e, 1)
cnt[~is_normal] = 0
XtX = (F * cnt[:, None]).T @ F
Xty = (F * cnt[:, None]).T @ (-score)
w = np.linalg.solve(XtX + 1e-6 * np.eye(C), Xty)
json.dump({"chars": ["U+%04X" % ord(c) for c in chars], "w": w.tolist()}, open(os.path.join(WORK, "gauge_w_fit.json"), "w"))
shifted = score + F @ w
print("shifted NORMAL score range:", shifted[is_normal].min(), shifted[is_normal].max())
print("weighted mean residual per token:", float((cnt * shifted).sum() / cnt.sum()))


def unk_score(sc):
    return sc[is_normal].min() - 10.0


def line_partials(e, sc, unk):
    """partial path scores at every token boundary of the SentencePiece best path (byte pieces of one unknown
    character count as one <unk> node)."""
    out = []
    s = 0.0
    k = 0
    n = len(e)
    while k < n:
        i = e[k]
        if is_normal[i]:
            s += sc[i]; k += 1
        elif pieces[i].type == T.BYTE:
            b0 = int(pieces[i].piece[3:5], 16)
            L = 1 if b0 < 0x80 else 2 if b0 < 0xE0 else 3 if b0 < 0xF0 else 4
            s += unk; k += L
        else:
            s += 0.0; k += 1
        out.append(s)
    return out


for name in SETS:
    docs, _ = load_set(name)
    ls = [l for _, t in docs for l in t.split("\n") if l]
    en = sp.encode(ls)
    mx_raw = mx_sh = 0.0
    for e in en:
        a = line_partials(e, score, unk_score(score)); b = line_partials(e, shifted, unk_score(shifted))
        mx_raw = max(mx_raw, max(abs(x) for x in a)); mx_sh = max(mx_sh, max(abs(x) for x in b))
    print(name, "max |partial| raw %.1f  shifted %.1f" % (mx_raw, mx_sh), flush=True)
