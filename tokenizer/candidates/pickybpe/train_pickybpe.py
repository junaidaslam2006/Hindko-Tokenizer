# -*- coding: utf-8 -*-
"""Train PickyBPE (PLAN.md A7) on D1 train with the P1 pre-tokenizer, byte-level.

    python train_pickybpe.py --tau 0.9 --vocab 16384 --out models/pickybpe_P1_D1_16k_tau0.9.json
    python train_pickybpe.py --tau none ...      plain BPE with the same code (A1-equivalence check)

Inputs (verified before use): FROZEN.json split-manifest hash 76582d3a..., normalize.py 1.0.1 hash,
data/train_D1.jsonl sha256 from data/data_manifest.json, every uid in the 'train' split, every text a fixed
point of hp.normalize.normalize(). The test split is never read.
Outputs: the model JSON (events + tokens), <out>.train.json (stats, hashes, parameters).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402

CACHE = os.path.join(HERE, "cache")


def code_hashes():
    return {f: C.sha256_file(os.path.join(HERE, f)) for f in ("pickybpe_common.py", "pickybpe.py", "train_pickybpe.py")}


def canonical_check(docs, info):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "canonical_%s.json" % info["view"])
    if os.path.exists(p):
        c = json.load(open(p, encoding="utf-8"))
        if c.get("sha256") == info["sha256"]:
            return c
    c = C.check_canonical(docs)
    c["sha256"] = info["sha256"]
    json.dump(c, open(p, "w", encoding="utf-8"), indent=1)
    return c


def word_counts(docs, info):
    """P1 pretoken types of the training documents with their counts (Python `regex` twin of the Oniguruma
    pattern; PLAN 1.3 differential test: identical pretokens on dev). Cached by data sha256 + regex."""
    import regex
    os.makedirs(CACHE, exist_ok=True)
    key = C.sha256_bytes((info["sha256"] + "|" + C.P1_PY).encode("utf-8"))[:16]
    p = os.path.join(CACHE, "wordcounts_%s_%s.pkl" % (info["view"], key))
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f), key
    rx = regex.compile(C.P1_PY)
    cnt = collections.Counter()
    for d in docs:
        t = d["text"]
        pos = 0
        for m in rx.finditer(t):
            if m.start() > pos:                          # gaps (never happen with P1: \s+ closes every gap)
                cnt[t[pos:m.start()]] += 1
            if m.end() > m.start():
                cnt[m.group()] += 1
            pos = m.end()
        if pos < len(t):
            cnt[t[pos:]] += 1
    items = sorted(((w.encode("utf-8"), c) for w, c in cnt.items()), key=lambda x: x[0])
    with open(p, "wb") as f:
        pickle.dump(items, f, protocol=4)
    return items, key


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", required=True, help="IoS threshold, or 'none' for plain BPE")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--view", default="train_D1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tau = None if a.tau.lower() == "none" else float(a.tau)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    frozen = C.verify_frozen()
    C.log("frozen inputs verified", frozen)
    docs, info = C.load_view(a.view, "train")
    C.log("loaded %s: %d docs" % (a.view, len(docs)))
    canon = canonical_check(docs, info)
    C.log("canonical form check:", canon)
    items, wkey = word_counts(docs, info)
    n_pretokens = sum(c for _, c in items)
    C.log("pretoken types %d, pretokens %d, bytes %d" % (len(items), n_pretokens,
                                                          sum(len(w) * c for w, c in items)))
    del docs
    import pickybpe as PB
    tr = PB.Trainer(items, a.vocab, tau, min_frequency=a.min_frequency, log=C.log)
    tr.train()
    C.log("trained: %s" % json.dumps({k: v for k, v in tr.stats.items()}, ensure_ascii=False))
    # the model file holds deterministic content only (G4 compares file hashes); timings go to .train.json
    meta = {"view": info, "frozen": frozen,
            "canonical_check": {k: v for k, v in canon.items() if k != "seconds"}, "pretoken_types": len(items),
            "pretokens": n_pretokens, "wordcount_cache_key": wkey, "code_sha256": code_hashes(),
            "python": sys.version.split()[0], "regex": __import__("regex").__version__,
            "params": {"tau": tau, "vocab_size": a.vocab, "min_frequency": a.min_frequency,
                       "tie_break": "count desc, then (id_a, id_b) asc"}}
    model = tr.to_model(meta)
    sha = PB.save_model(model, out)
    rep = {"model": out, "model_sha256": sha, "summary": model["summary"], "meta": meta,
           "train_seconds": tr.stats.get("seconds"), "canonical_check_seconds": canon.get("seconds"),
           "platform": platform.platform(), "wall_seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(out[:-5] + ".train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # final training-state token frequencies (internal ids) for the trainer/encoder consistency check
    with open(out[:-5] + ".trainstate.pkl", "wb") as f:
        pickle.dump({"tfreq": tr.final_state_counts(), "words": tr.words, "word_bytes": [w for w, _ in items],
                     "counts": tr.counts}, f, protocol=4)
    C.log("saved %s sha256 %s (%.1fs)" % (out, sha, time.time() - t0))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
