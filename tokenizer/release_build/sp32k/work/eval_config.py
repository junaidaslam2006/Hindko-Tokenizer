# -*- coding: utf-8 -*-
"""Build (K, gauge) -> work/<tag>/sp.model + tokenizer.json, then per line of every set:
 old SP vs new SP (changes outside the old tie set), new SP vs new HF, and the largest |partial path score| of the
 new SP best path (float32 exactness holds while every compared value is < 2**(24-K) in magnitude).
Usage: python eval_config.py K gauge_json|none tag"""
import json, os, sys, time
import numpy as np
from common import *
import build_lib as B
import sentencepiece as spm
from tokenizers import Tokenizer

K = int(sys.argv[1]); gpath = sys.argv[2]; tag = sys.argv[3]
gauge = None if gpath == "none" else B.load_gauge(gpath, K)
d = os.path.join(WORK, tag)
mp, info = B.write_model(K, d, gauge)
jp = B.write_hf_json(mp, os.path.join(d, "tokenizer.json"))
mn, T = load_proto(mp)
pieces = list(mn.pieces)
sc = np.array([p.score for p in pieces])
is_normal = np.array([p.type == T.NORMAL for p in pieces]); is_byte = np.array([p.type == T.BYTE for p in pieces])
unk = float(sc[is_normal].min() - 10.0)
old = spm.SentencePieceProcessor(model_file=OLD_MODEL)
new = spm.SentencePieceProcessor(model_file=mp)
hf = Tokenizer.from_file(jp)
tie = json.load(open(os.path.join(WORK, "old_mismatches.json"), encoding="utf-8"))
tie_lines = {(r["set"], r["line"]) for r in tie}


def max_partial(e):
    s = 0.0; mx = 0.0; k = 0
    while k < len(e):
        i = e[k]
        if is_normal[i]:
            s += sc[i]; k += 1
        elif is_byte[i]:
            b0 = int(pieces[i].piece[3:5], 16)
            s += unk; k += 1 if b0 < 0x80 else 2 if b0 < 0xE0 else 3 if b0 < 0xF0 else 4
        else:
            k += 1
        mx = max(mx, abs(s))
    return mx


res = {"K": K, "gauge": gpath, "score_change": info, "exact_limit": 2.0 ** (24 - K),
       "normal_score_range": [float(sc[is_normal].min()), float(sc[is_normal].max())], "unk_score": unk}
for name in SETS:
    t0 = time.time()
    docs, _ = load_set(name)
    lines = [l for _, t in docs for l in t.split("\n")]
    eo = old.encode(lines); en = new.encode(lines)
    eh = [e.ids for e in hf.encode_batch(lines, add_special_tokens=False)]
    changed = [i for i in range(len(lines)) if eo[i] != en[i]]
    mp_ = np.array([max_partial(e) for e in en])
    res[name] = {"lines": len(lines), "old_vs_new_changed_lines": len(changed),
                 "changed_lines_in_old_tie_set": sum((name, i) in tie_lines for i in changed),
                 "changed_lines_NOT_in_old_tie_set": [i for i in changed if (name, i) not in tie_lines],
                 "old_tie_lines_unchanged": sum(1 for (s, i) in tie_lines if s == name and i not in set(changed)),
                 "new_sp_vs_new_hf_mismatch_lines": [i for i in range(len(lines)) if en[i] != eh[i]],
                 "max_abs_partial": float(mp_.max()),
                 "lines_over_exact_limit": int((mp_ >= 2.0 ** (24 - K) - 64).sum()),
                 "sec": round(time.time() - t0, 1)}
    print(name, json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res[name].items()}), flush=True)
json.dump(res, open(os.path.join(d, "eval_config.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("score_change", "normal_score_range", "unk_score")}))
