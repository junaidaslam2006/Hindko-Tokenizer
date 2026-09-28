# -*- coding: utf-8 -*-
"""Every line where old SP (f32 Viterbi) and the old HF export (f64 Viterbi) differ: exact path scores,
the differing spans, and whether the two segmentations use the same multiset of pieces (permutation tie)."""
import json, os, collections
from fractions import Fraction
import numpy as np
from common import *
import sentencepiece as spm
from tokenizers import Tokenizer

m, T = load_proto(OLD_MODEL)
sc = [p.score for p in m.pieces]
sp = spm.SentencePieceProcessor(model_file=OLD_MODEL)
hf = Tokenizer.from_file(OLD_JSON)
out = []
for name in SETS:
    docs, _ = load_set(name)
    lines = [l for _, t in docs for l in t.split("\n")]
    enc = sp.encode(lines)
    hfe = [e.ids for e in hf.encode_batch(lines, add_special_tokens=False)]
    for i, (a, b) in enumerate(zip(enc, hfe)):
        if a == b:
            continue
        # common prefix / suffix
        p = 0
        while p < min(len(a), len(b)) and a[p] == b[p]:
            p += 1
        s = 0
        while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
            s += 1
        da, db = a[p:len(a) - s], b[p:len(b) - s]
        ea = sum(Fraction(sc[x]) for x in a); eb = sum(Fraction(sc[x]) for x in b)
        f32a = np.float32(0); f32b = np.float32(0)
        for x in a: f32a = np.float32(f32a + np.float32(sc[x]))
        for x in b: f32b = np.float32(f32b + np.float32(sc[x]))
        out.append({"set": name, "line": i, "chars": len(lines[i]), "tokens": len(a),
                    "sp_span": [sp.id_to_piece(x) for x in da], "hf_span": [sp.id_to_piece(x) for x in db],
                    "sp_ids": da, "hf_ids": db,
                    "permutation": collections.Counter(da) == collections.Counter(db),
                    "exact_score_sp_minus_hf": float(ea - eb), "exact_equal": ea == eb,
                    "prefix_score": float(sum(sc[x] for x in a[:p]))})
for r in out:
    print(r["set"], r["line"], r["chars"], "perm=%s" % r["permutation"], "exact_eq=%s" % r["exact_equal"],
          "d=%.3g" % r["exact_score_sp_minus_hf"], "prefix=%.1f" % r["prefix_score"], r["sp_span"], r["hf_span"])
json.dump(out, open(os.path.join(WORK, "old_mismatches.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "mismatching lines;", sum(r["exact_equal"] for r in out), "exact ties;",
      sum(r["permutation"] for r in out), "permutation ties")
