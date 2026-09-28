# -*- coding: utf-8 -*-
"""Locate the long_lines stress mismatch: position, pieces, exact scores and the running path score there."""
import json, os, sys
from fractions import Fraction
import numpy as np
sys.path.insert(0, r"F:\Hindko\_tokenizer\eval")
import adapters as A
from tokenizers import Tokenizer
REL = r"F:\Hindko\_tokenizer\release_build\sp32k"
can = A.SPAdapter(path=os.path.join(REL, "sp.model"))
hf = Tokenizer.from_file(os.path.join(REL, "tokenizer.json"))
from sentencepiece import sentencepiece_model_pb2 as spb
m = spb.ModelProto(); m.ParseFromString(open(os.path.join(REL, "sp.model"), "rb").read())
sc = [p.score for p in m.pieces]
for l in open(os.path.join(REL, r"stress\stress.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r["cat"] != "long_lines":
        continue
    t = r["text"]
    a = can.encode(t); b = hf.encode(t).ids
    if a == b:
        continue
    k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    s = 0.0; f = np.float32(0)
    for i in a[:k]:
        s += sc[i]; f = np.float32(f + np.float32(sc[i]))
    print(r["id"], "chars", len(t), "tokens", len(a), "first diff at token", k)
    print(" canonical", [can.sp.id_to_piece(i) for i in a[k - 2:k + 4]])
    print(" hf       ", [can.sp.id_to_piece(i) for i in b[k - 2:k + 4]])
    print(" running path score before the diff: exact %.4f  float32 %.4f  (bound 1024)" % (s, float(f)))
    ea = sum(Fraction(sc[i]) for i in a); eb = sum(Fraction(sc[i]) for i in b)
    print(" exact total canonical - hf:", float(ea - eb))
