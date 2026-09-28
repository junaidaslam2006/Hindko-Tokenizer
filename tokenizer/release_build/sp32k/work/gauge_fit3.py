# -*- coding: utf-8 -*-
"""Gauge fit v2: least squares on CHUNK sums (chunks of <= 8 tokens along the train_D2 best paths), which targets
the drift of partial sums directly. Reports max |partial| per set, and the top lines with their unknown-char count."""
import json, os, sys, collections
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
is_byte = np.array([p.type == T.BYTE for p in pieces])
sp = spm.SentencePieceProcessor(model_file=OLD_MODEL)
CH = int(sys.argv[1]) if len(sys.argv) > 1 else 8

docs, _ = load_set(os.environ.get("FITSET","train_D2"))
lines = [l for _, t in docs for l in t.split("\n") if l]
enc = sp.encode(lines)
XtX = np.zeros((C, C)); Xty = np.zeros(C)
for e in enc:
    e = [i for i in e if is_normal[i]]
    for k in range(0, len(e), CH):
        ch = e[k:k + CH]
        f = F[ch].sum(0); y = -score[ch].sum()
        XtX += np.outer(f, f); Xty += f * y
w = np.linalg.solve(XtX + 1e-6 * np.eye(C), Xty)
shifted = score + F @ w
json.dump({"method": "chunk LS, chunk=%d tokens" % CH, "chars": [ord(c) for c in chars], "w": w.tolist()},
          open(os.path.join(WORK, "gauge_w_%s_chunk%d.json" % (os.environ.get("FITSET","train_D2"), CH)), "w"))
print("chunk", CH, "shifted NORMAL range: %.2f %.2f" % (shifted[is_normal].min(), shifted[is_normal].max()))
unk = shifted[is_normal].min() - 10.0


def partials(e, sc, unk):
    out = []; s = 0.0; k = 0; nunk = 0
    while k < len(e):
        i = e[k]
        if is_normal[i]:
            s += sc[i]; k += 1
        elif is_byte[i]:
            b0 = int(pieces[i].piece[3:5], 16)
            L = 1 if b0 < 0x80 else 2 if b0 < 0xE0 else 3 if b0 < 0xF0 else 4
            s += unk; k += L; nunk += 1
        else:
            k += 1
        out.append(s)
    return out, nunk


for name in SETS:
    docs, _ = load_set(name)
    ls = [l for _, t in docs for l in t.split("\n") if l]
    en = sp.encode(ls)
    rows = []
    for li, e in enumerate(en):
        p, nunk = partials(e, shifted, unk)
        rows.append((max(abs(x) for x in p), li, nunk, len(e)))
    rows.sort(reverse=True)
    q = np.array([r[0] for r in rows])
    print(name, "max %.1f  p99.9 %.1f  lines>256: %d  >512: %d  >1024: %d" % (
        q.max(), np.quantile(q, 0.999), (q > 256).sum(), (q > 512).sum(), (q > 1024).sum()))
    for r in rows[:4]:
        print("   |max| %.1f  unk_chars %d  tokens %d  %r" % (r[0], r[2], r[3], ls[r[1]][:60]))
