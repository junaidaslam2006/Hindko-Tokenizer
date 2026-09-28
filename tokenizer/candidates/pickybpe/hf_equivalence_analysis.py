# -*- coding: utf-8 -*-
"""Can the PickyBPE vocabulary be served by a stock HF `tokenizers` BPE model with the SAME vocabulary?

An HF BPE model emits a non-byte token only as the product of a merge (l, r) -> t with l and r in its vocabulary.
So a present learned token t whose byte string has no split into two tokens that are both in the PickyBPE vocabulary
can never be emitted by any HF BPE model over that vocabulary, whatever merges and ranks are chosen. If the
event-ordered PickyBPE encoder emits such a t on a dev document, exact equivalence on 100% of dev is impossible
(for an HF model with exactly this vocabulary and these ids). This script counts such tokens and their dev
occurrences. Dev = dev_strict + dev_permissive (both 'validation'); the test split is not read.

    python hf_equivalence_analysis.py --model models/X.json --out verify/X.hf_equivalence.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402
import pickybpe as PB  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path = a.model if os.path.isabs(a.model) else os.path.join(HERE, a.model)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    tk = PB.PickyBPETokenizer(path)
    vb = {tk.token_bytes(i): i for i in range(PB.N_SPECIAL, tk.vocab_size)}
    unsplittable = []
    for i in range(PB.N_BASE, tk.vocab_size):
        b = tk.token_bytes(i)
        if not any(b[:p] in vb and b[p:] in vb for p in range(1, len(b))):
            unsplittable.append(i)
    us = set(unsplittable)
    rep = {"model": path, "model_sha256": tk.sha256, "present_learned": tk.vocab_size - PB.N_BASE,
           "unsplittable_present_tokens": len(unsplittable)}
    for view in ("dev_strict", "dev_permissive"):
        docs, info = C.load_view(view, "validation")
        occ = collections.Counter()
        docs_hit = 0
        for d in docs:
            ids = tk.encode(d["text"])
            hit = [i for i in ids if i in us]
            if hit:
                docs_hit += 1
                occ.update(hit)
        rep[view] = {"docs": len(docs), "docs_emitting_an_unsplittable_token": docs_hit,
                     "unsplittable_token_occurrences": sum(occ.values()),
                     "distinct_unsplittable_tokens_emitted": len(occ),
                     "most_common": [{"text": tk.token_bytes(i).decode("utf-8", "replace"), "id": i, "dev_count": c,
                                      "removed_parts_of_its_creating_pair": [
                                          tk.itok[x].decode("utf-8", "replace") for x in _creating_pair(tk, i)
                                          if x in tk.removed_at_end]}
                                     for i, c in occ.most_common(15)]}
    rep["conclusion"] = ("exact HF BPE equivalence with this vocabulary is impossible on dev_strict"
                         if rep["dev_strict"]["docs_emitting_an_unsplittable_token"] else
                         "no unsplittable token is emitted on dev_strict: this test does not rule out equivalence")
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "most_common"})
                      for k, v in rep.items()}, ensure_ascii=False, indent=1))
    print(json.dumps(rep["dev_strict"]["most_common"][:8], ensure_ascii=False))


def _creating_pair(tk, i):
    internal = tk.final_ids[i]
    last = None
    for e in tk.m["events"]:
        if e[0] == 0 and e[3] == internal:
            last = (e[1], e[2])
    return last or ()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
