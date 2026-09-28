# -*- coding: utf-8 -*-
"""Line statistics and old SP vs old HF export mismatches on all sets (per line)."""
import json, os, sys, time
import numpy as np
from common import *
import sentencepiece as spm
from tokenizers import Tokenizer

m, T = load_proto(OLD_MODEL)
score = np.array([p.score for p in m.pieces], dtype=np.float64)
sp = spm.SentencePieceProcessor(model_file=OLD_MODEL)
hf = Tokenizer.from_file(OLD_JSON)
res = {}
for name in SETS:
    t0 = time.time()
    docs, sha = load_set(name)
    lines = [l for _, t in docs for l in t.split("\n")]
    lens = np.array([len(l) for l in lines])
    blens = np.array([len(l.encode("utf-8")) for l in lines])
    enc = sp.encode(lines)
    ntok = np.array([len(e) for e in enc])
    tot = np.array([score[e].sum() if e else 0.0 for e in enc])
    hfe = [e.ids for e in hf.encode_batch(lines, add_special_tokens=False)]
    mism = [i for i, (a, b) in enumerate(zip(enc, hfe)) if a != b]
    nonempty = [i for i in range(len(lines)) if lines[i]]
    res[name] = {"docs": len(docs), "lines": len(lines), "nonempty_lines": len(nonempty),
                 "max_line_chars": int(lens.max()), "max_line_bytes": int(blens.max()),
                 "max_line_tokens": int(ntok.max()), "min_line_score": float(tot.min()),
                 "lines_score_below_-1024": int((tot < -1024).sum()), "lines_score_below_-4096": int((tot < -4096).sum()),
                 "mean_score_per_byte": float(tot.sum() / max(1, blens.sum())),
                 "sp_vs_hfexport_mismatch_lines": len(mism),
                 "mismatch_examples": [{"line_chars": int(lens[i]), "tokens": int(ntok[i]), "score": float(tot[i])} for i in mism[:10]],
                 "sec": round(time.time() - t0, 1)}
    print(name, json.dumps(res[name], ensure_ascii=False))
json.dump(res, open(os.path.join(WORK, "stats_old.json"), "w"), indent=1)
