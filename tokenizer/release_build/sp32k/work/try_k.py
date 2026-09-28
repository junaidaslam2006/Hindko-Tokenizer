# -*- coding: utf-8 -*-
"""For a grid exponent K: build sp.model + tokenizer.json and compare, per line of every set,
old SP vs new SP, and new SP vs new HF. Usage: python try_k.py K"""
import json, os, sys, time
from common import *
import build_lib as B
import sentencepiece as spm
from tokenizers import Tokenizer

K = int(sys.argv[1])
d = os.path.join(WORK, "k%d" % K)
mp, info = B.write_model(K, d)
jp = B.write_hf_json(mp, os.path.join(d, "tokenizer.json"))
old = spm.SentencePieceProcessor(model_file=OLD_MODEL)
new = spm.SentencePieceProcessor(model_file=mp)
hf = Tokenizer.from_file(jp)
tie = json.load(open(os.path.join(WORK, "old_mismatches.json"), encoding="utf-8"))
tie_lines = {(r["set"], r["line"]) for r in tie}
res = {"K": K, "score_change": info}
for name in SETS:
    t0 = time.time()
    docs, _ = load_set(name)
    lines = [l for _, t in docs for l in t.split("\n")]
    eo = old.encode(lines); en = new.encode(lines)
    eh = [e.ids for e in hf.encode_batch(lines, add_special_tokens=False)]
    changed = [i for i in range(len(lines)) if eo[i] != en[i]]
    changed_nontie = [i for i in changed if (name, i) not in tie_lines]
    sp_hf = [i for i in range(len(lines)) if en[i] != eh[i]]
    res[name] = {"lines": len(lines), "old_vs_new_changed_lines": len(changed),
                 "changed_lines_not_in_old_tie_set": len(changed_nontie),
                 "changed_nontie_examples": [lines[i][:80] for i in changed_nontie[:5]],
                 "new_sp_vs_new_hf_mismatch_lines": len(sp_hf), "sec": round(time.time() - t0, 1)}
    print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
json.dump(res, open(os.path.join(d, "try_k.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
