# -*- coding: utf-8 -*-
"""A1-16k reference for the PickyBPE comparison (PLAN.md 2.1 A1: HF BpeTrainer, Split(P1) then
ByteLevel(use_regex=False), byte-level, D1 train, 16,384 = 64 specials + 256 bytes + 16,064 merges,
min_frequency 2, normalizer None). Built here because no Stage-1 A1 file existed under
F:\\Hindko\\_tokenizer when this task ran; it is a reference for this task, labelled as such.

    python build_a1_ref.py [--out models/a1_ref_P1_D1_16k.json] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "models", "a1_ref_P1_D1_16k.json"))
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if a.threads > 1 else "false"
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
    import tokenizers
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    frozen = C.verify_frozen()
    docs, info = C.load_view("train_D1", "train")
    texts = [d["text"] for d in docs]
    tok = Tokenizer(models.BPE())
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(C.P1_ONIG), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trn = trainers.BpeTrainer(vocab_size=a.vocab, min_frequency=2, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                              special_tokens=C.SPECIAL_TOKENS)
    t0 = time.time()
    tok.train_from_iterator(texts, trainer=trn)
    dt = time.time() - t0
    tok.save(a.out)
    sha = C.sha256_file(a.out)
    rep = {"model": a.out, "sha256": sha, "train_seconds": round(dt, 1), "vocab_size": tok.get_vocab_size(True),
           "tokenizers": tokenizers.__version__, "view": info, "frozen": frozen,
           "recipe": "BpeTrainer(vocab_size=%d, min_frequency=2, initial_alphabet=ByteLevel.alphabet(), "
                     "special_tokens=64 PLAN block); pre_tokenizer Sequence[Split(P1, isolated), "
                     "ByteLevel(add_prefix_space=False, use_regex=False)]; decoder ByteLevel; normalizer None"
                     % a.vocab,
           "threads": a.threads,
           "code_sha256": C.sha256_file(os.path.abspath(__file__))}
    json.dump(rep, open(a.out[:-5] + ".train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("A1 reference: %s sha256 %s, %d tokens, %.1fs" % (a.out, sha, rep["vocab_size"], dt))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
