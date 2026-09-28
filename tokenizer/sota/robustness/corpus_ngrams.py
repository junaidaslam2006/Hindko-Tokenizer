# -*- coding: utf-8 -*-
"""Word 8-gram index of the released Hindko corpus, for the out-of-corpus check of build_sets.py.

Corpus = every record of F:\\Hindko\\hindko_dataset_permissive.jsonl and F:\\Hindko\\hindko_dataset.jsonl (all splits,
all tiers, all sources; the strict file is a subset, read anyway). Keys: rcommon.key_words (NFC, harakat and format
characters removed, Arabic-keyboard letters folded), 8 consecutive words, blake2b-64 hash.
Output: sets/corpus_8grams.npy (sorted unique uint64) + sets/corpus_8grams.json (provenance).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import rcommon as C  # noqa: E402

FILES = [r"F:\Hindko\hindko_dataset_permissive.jsonl", r"F:\Hindko\hindko_dataset.jsonl"]


def main():
    t0 = time.time()
    parts, n_docs, n_words = [], 0, 0
    info = {"files": []}
    for p in FILES:
        info["files"].append({"path": p, "sha256": C.sha256_file(p), "bytes": os.path.getsize(p)})
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                w = C.key_words(r.get("text") or "")
                n_docs += 1
                n_words += len(w)
                h = C.ngram_hashes(w, 8)
                if len(h):
                    parts.append(np.unique(h))
        print(p, n_docs, n_words, "%.0fs" % (time.time() - t0), flush=True)
    allh = np.unique(np.concatenate(parts))
    os.makedirs(C.SETS, exist_ok=True)
    np.save(os.path.join(C.SETS, "corpus_8grams.npy"), allh)
    info.update({"records_read": n_docs, "key_words": n_words, "unique_8grams": int(len(allh)),
                 "key": "rcommon.key_words + ngram_hashes(n=8), blake2b digest_size=8 little-endian",
                 "seconds": round(time.time() - t0, 1)})
    json.dump(info, open(os.path.join(C.SETS, "corpus_8grams.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
