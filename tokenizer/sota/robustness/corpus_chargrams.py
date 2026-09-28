# -*- coding: utf-8 -*-
"""Spacing-insensitive near-duplicate index of the released Hindko corpus (second out-of-corpus check).

PDF text layers often join or split words, so a word 8-gram test can miss a book that IS in the corpus with
normal spacing. This index removes spacing from the picture: text -> rcommon.key_words (NFC, harakat / format
characters removed, Arabic-keyboard letters folded, letters only) -> words concatenated WITHOUT spaces -> every
character 24-gram (stride 1) -> 64-bit polynomial rolling hash (numpy, wrap-around arithmetic).
Output: sets/corpus_c24.npy (sorted unique uint64) + sets/corpus_c24.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import rcommon as C  # noqa: E402

FILES = [r"F:\Hindko\hindko_dataset_permissive.jsonl", r"F:\Hindko\hindko_dataset.jsonl"]
K = 24
BASE = np.uint64(1099511628211)   # FNV prime, odd


def chargram_hashes(text: str, k: int = K) -> np.ndarray:
    s = "".join(C.key_words(text))
    if len(s) < k:
        return np.zeros(0, dtype=np.uint64)
    cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64) + np.uint64(1)
    L = len(cp) - k + 1
    h = np.zeros(L, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * BASE + cp[j:j + L]
    return h


def main():
    t0 = time.time()
    parts, n_docs, n_chars = [], 0, 0
    info = {"files": []}
    for p in FILES:
        info["files"].append({"path": p, "sha256": C.sha256_file(p), "bytes": os.path.getsize(p)})
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                h = chargram_hashes(r.get("text") or "")
                n_docs += 1
                n_chars += len(h)
                if len(h):
                    parts.append(np.unique(h))
        print(p, n_docs, n_chars, "%.0fs" % (time.time() - t0), flush=True)
    allh = np.unique(np.concatenate(parts))
    np.save(os.path.join(C.SETS, "corpus_c24.npy"), allh)
    info.update({"records_read": n_docs, "chargrams": n_chars, "unique_chargrams": int(len(allh)), "k": K,
                 "key": "rcommon.key_words joined without spaces; char %d-grams; h = h*1099511628211 + (codepoint+1) "
                        "mod 2^64" % K, "seconds": round(time.time() - t0, 1)})
    json.dump(info, open(os.path.join(C.SETS, "corpus_c24.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
