# -*- coding: utf-8 -*-
"""train_D1 (the LM training view of every Colab bundle): release canonical vs HF tokenizer.json vs transformers,
decode round trip, and old->new changes with the exact-tie test per changed line. -> lm_view_check.json"""
import collections
import json
import os
import sys
from fractions import Fraction

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
TOK = r"F:\Hindko\_tokenizer"
sys.path.insert(0, os.path.join(TOK, "eval"))
import adapters as A  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REL = os.path.join(TOK, r"release_build\sp32k")
CAND = os.path.join(TOK, r"candidates\round2\standard\tok\R2-A4-SPnat-D2-32k")
can = A.SPAdapter(path=os.path.join(REL, "sp.model"))
old = A.SPAdapter(path=os.path.join(CAND, "sp.model"))
hf = Tokenizer.from_file(os.path.join(REL, "tokenizer.json"))
tf = AutoTokenizer.from_pretrained(REL)
so = [Fraction(p.score) if p.type == 1 else Fraction(0) for p in old.proto.pieces]
path = os.path.join(TOK, r"data\train_D1.jsonl")
docs = [(json.loads(l)["uid"], json.loads(l)["text"]) for l in open(path, encoding="utf-8")]
texts = [t for _, t in docs]
ih = [e.ids for e in hf.encode_batch(texts)]
it = tf(texts, add_special_tokens=False)["input_ids"]
r = collections.Counter()
changed = []
for k, (uid, t) in enumerate(docs):
    c = can.encode(t)
    r["docs"] += 1
    r["hf_ids_equal"] += ih[k] == c
    r["tf_ids_equal"] += list(it[k]) == c
    r["hf_decode_exact"] += hf.decode(ih[k], skip_special_tokens=False) == t
    r["canonical_decode_exact"] += can.decode(c) == t
    if old.encode(t) != c:
        r["changed_vs_old"] += 1
        lines = []
        for line in t.split("\n"):
            a, b = old.sp.encode(line), can.sp.encode(line)
            if a != b:
                gap = sum(so[x] for x in a) - sum(so[x] for x in b)
                lines.append({"exact_old_gap": str(gap), "perm": collections.Counter(a) == collections.Counter(b)})
        r["changed_lines"] += len(lines)
        r["changed_lines_exact_ties"] += sum(Fraction(x["exact_old_gap"]) == 0 for x in lines)
        changed.append({"uid": uid, "lines": lines})
out = {"view": "train_D1", "sha256": A.sha256_file(path), **dict(r), "changed_docs": changed}
json.dump(out, open(os.path.join(REL, "lm_view_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "changed_docs"}))
