# -*- coding: utf-8 -*-
"""Exact-arithmetic encodings under round_K(score) (via HF tokenizers, float64 = exact on the 2**-K grid here)
vs the OLD SentencePiece encodings, for several K. Counts changed lines outside the old tie set.
Usage: python flips_vs_k.py K1 K2 ..."""
import json, os, sys
from common import *
import build_lib as B
import sentencepiece as spm
from tokenizers import Tokenizer

old = spm.SentencePieceProcessor(model_file=OLD_MODEL)
tie = json.load(open(os.path.join(WORK, "old_mismatches.json"), encoding="utf-8"))
tie_lines = {(r["set"], r["line"]) for r in tie}
sets = {}
for name in SETS:
    docs, _ = load_set(name)
    lines = [l for _, t in docs for l in t.split("\n")]
    sets[name] = (lines, old.encode(lines))
res = {}
for K in map(int, sys.argv[1:]):
    m, T, ch = B.rounded_proto(K)
    tmp = os.path.join(WORK, "tmp_k%d.model" % K)
    open(tmp, "wb").write(m.SerializeToString())
    j = B.hf_json_dict(tmp)
    hf = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    os.remove(tmp)
    r = {}
    for name, (lines, eo) in sets.items():
        eh = [e.ids for e in hf.encode_batch(lines, add_special_tokens=False)]
        diff = [i for i in range(len(lines)) if eh[i] != eo[i]]
        r[name] = {"changed": len(diff), "changed_nontie": [i for i in diff if (name, i) not in tie_lines]}
    res[K] = r
    print("K=%d" % K, {n: (v["changed"], len(v["changed_nontie"])) for n, v in r.items()}, flush=True)
json.dump(res, open(os.path.join(WORK, "flips_vs_k_%s.json" % "_".join(sys.argv[1:])), "w"), indent=1)
