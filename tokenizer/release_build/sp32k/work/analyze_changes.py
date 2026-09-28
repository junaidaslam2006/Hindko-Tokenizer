# -*- coding: utf-8 -*-
"""For grid K: every line whose encoding changed old->new. Report differing spans and the exact score gap between
the old and the new segmentation under the OLD scores (exact rationals) and under the NEW scores.
Usage: python analyze_changes.py K"""
import json, os, sys, collections
from fractions import Fraction
from common import *
import sentencepiece as spm

K = int(sys.argv[1])
d = os.path.join(WORK, "k%d" % K)
mo, T = load_proto(OLD_MODEL); mn, _ = load_proto(os.path.join(d, "sp.model"))
so = [Fraction(p.score) for p in mo.pieces]; sn = [Fraction(p.score) for p in mn.pieces]
old = spm.SentencePieceProcessor(model_file=OLD_MODEL)
new = spm.SentencePieceProcessor(model_file=os.path.join(d, "sp.model"))
tie = json.load(open(os.path.join(WORK, "old_mismatches.json"), encoding="utf-8"))
tie_lines = {(r["set"], r["line"]) for r in tie}
rows = []
for name in SETS:
    docs, _ = load_set(name)
    lines = [l for _, t in docs for l in t.split("\n")]
    eo = old.encode(lines); en = new.encode(lines)
    for i in range(len(lines)):
        a, b = eo[i], en[i]
        if a == b:
            continue
        p = 0
        while p < min(len(a), len(b)) and a[p] == b[p]:
            p += 1
        s = 0
        while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
            s += 1
        da, db = a[p:len(a) - s], b[p:len(b) - s]
        gap_old = sum(so[x] for x in da) - sum(so[x] for x in db)   # >0: old path better under old scores
        gap_new = sum(sn[x] for x in db) - sum(sn[x] for x in da)   # >0: new path better under new scores
        rows.append({"set": name, "line": i, "old_tie_line": (name, i) in tie_lines,
                     "old_span": [old.id_to_piece(x) for x in da], "new_span": [old.id_to_piece(x) for x in db],
                     "permutation": collections.Counter(da) == collections.Counter(db),
                     "gap_old_exact": float(gap_old), "gap_new_exact": float(gap_new),
                     "prefix_old_score": float(sum(so[x] for x in a[:p]))})
for r in rows:
    if not r["old_tie_line"]:
        print(r["set"], r["line"], "perm=%s" % r["permutation"], "gap_old=%.3g" % r["gap_old_exact"],
              "gap_new=%.3g" % r["gap_new_exact"], "prefix=%.1f" % r["prefix_old_score"], r["old_span"], "->", r["new_span"])
print("changed:", len(rows), "of which old tie lines:", sum(r["old_tie_line"] for r in rows))
json.dump(rows, open(os.path.join(d, "changes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
