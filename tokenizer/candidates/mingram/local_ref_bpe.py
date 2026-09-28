# -*- coding: utf-8 -*-
"""LOCAL CONTEXT REFERENCE ONLY (not a PLAN candidate, not the Stage-1 A2 artifact).

Char-level BPE with the SAME pretokenizer (P1), training view (train_D1), alphabet (every train character),
byte_fallback and 64-token special block as the MinGram candidate, at 16,384 in total:
    ids 0-63 special block, 64-319 <0x00>..<0xFF>, then the BPE vocabulary (alphabet + merges).
It exists so that MINGRAM.md can state MinGram's intrinsic numbers next to a BPE built under identical
conditions. PLAN's A1 (byte-level BPE-P1-16k) remains the reference for NSL and every claim.

    python local_ref_bpe.py --out runs/localref_charbpe_P1_D1_16384
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join("runs", "localref_charbpe_P1_D1_16384"))
ap.add_argument("--total", type=int, default=16384)
ap.add_argument("--threads", type=int, default=2)
A = ap.parse_args()
os.environ["RAYON_NUM_THREADS"] = str(A.threads)
sys.path.insert(0, HERE)
import mingram as M  # noqa: E402
H = M.H
sys.path.insert(0, r"F:\Hindko\_pipeline")
from hp.normalize import normalize  # noqa: E402


def main():
    out = A.out if os.path.isabs(A.out) else os.path.join(HERE, A.out)
    os.makedirs(out, exist_ok=True)
    docs, dinfo = H.load_docs("train_D1")
    texts = [normalize(d["text"]) for d in docs]
    alphabet = sorted({c for t in texts for c in t})
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(M.P1), behavior="isolated", invert=False)
    n_bpe = A.total - len(M.SPECIALS) - 256
    trn = trainers.BpeTrainer(vocab_size=n_bpe, min_frequency=2, show_progress=False, special_tokens=[])
    t0 = time.time()
    tok.train_from_iterator(texts, trainer=trn)
    dt = time.time() - t0
    j = json.loads(tok.to_str())
    old = sorted(j["model"]["vocab"].items(), key=lambda kv: kv[1])
    assert sorted(s for s, _ in old if len(s) == 1) == alphabet
    vocab = {}
    for s in M.SPECIALS:
        vocab[s] = len(vocab)
    for b in M.BYTE_PIECES:
        vocab[b] = len(vocab)
    for s, _ in old:
        vocab[s] = len(vocab)
    j["model"]["vocab"] = vocab
    j["model"]["byte_fallback"] = True
    j["decoder"] = {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
    j["added_tokens"] = [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for i, s in enumerate(M.SPECIALS)]
    path = os.path.join(out, "tokenizer.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(j, ensure_ascii=False, indent=1))
    tk = Tokenizer.from_file(path)
    info = {"what": "LOCAL CONTEXT REFERENCE ONLY: char-level BPE-P1 + byte_fallback on train_D1, same alphabet, "
                    "specials and total size as the MinGram candidate; not a PLAN candidate",
            "train_s": round(dt, 1), "vocab_total": tk.get_vocab_size(True), "alphabet": len(alphabet),
            "merges": len(j["model"]["merges"]), "data_sha256": dinfo["sha256"], "sha256": M.sha256_file(path)}
    assert info["vocab_total"] == A.total, info
    json.dump(info, open(os.path.join(out, "build_info.json"), "w", encoding="utf-8"), indent=1)
    print(info)


if __name__ == "__main__":
    main()
