# -*- coding: utf-8 -*-
"""SuperBPE stage 1 = A1 (PLAN 2.1): HF byte-level BPE, Split(P1, isolated) + ByteLevel(use_regex=False),
BpeTrainer(min_frequency=2, initial_alphabet=ByteLevel.alphabet(), the 64-special block), trained on the
normalised train_D1 documents (permissive train, whole documents).

    python build_stage1.py VOCAB OUT.json [--threads N]

VOCAB counts everything (64 specials + 256 bytes + merges). Writes OUT.json and OUT.json.info.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def train_a1(texts, vocab_size, pre_regex=C.P1):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pre_regex), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    trn = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=C.MIN_FREQUENCY, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                              special_tokens=C.SPECIAL_TOKENS)
    t0 = time.time()
    tk.train_from_iterator(texts, trainer=trn)
    return tk, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab", type=int)
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    fz = C.verify_frozen()
    docs, info = C.load_view("train_D1")
    texts = [d["text"] for d in docs]
    tk, dt = train_a1(texts, a.vocab)
    tk.save(a.out)
    j = json.load(open(a.out, encoding="utf-8"))
    n_merges = len(j["model"]["merges"])
    V = tk.get_vocab_size(with_added_tokens=True)
    meta = {"what": "A1 byte-level BPE-P1 (SuperBPE stage 1)", "vocab_size_target": a.vocab, "vocab_size": V,
            "merges": n_merges, "train_seconds": round(dt, 1), "rayon_threads": a.threads,
            "pre_tokenizer_regex": C.P1, "min_frequency": C.MIN_FREQUENCY, "train_view": info,
            "split_manifest_sha256": fz["split_manifest_sha256"], "normalize_version": fz["normalize_version"],
            "tokenizers_version": __import__("tokenizers").__version__,
            "sha256": C.sha256_file(a.out), "code_sha256": C.sha256_file(os.path.abspath(__file__))}
    json.dump(meta, open(a.out + ".info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("A1-P1 vocab %d merges %d in %.1fs -> %s sha256 %s" % (V, n_merges, dt, a.out, meta["sha256"][:16]))


if __name__ == "__main__":
    main()
