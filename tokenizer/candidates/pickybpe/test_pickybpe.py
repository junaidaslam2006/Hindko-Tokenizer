# -*- coding: utf-8 -*-
"""Self-tests of the PickyBPE reimplementation (no test-split data is used).

  1. toy corpora with aggressive tau: the event-ordered encoder reproduces the trainer's final state for every
     training word; round trip is exact; no removed token is ever emitted; removal events really happen,
     including re-additions where the toy data produces them.
  2. tau=None on a 1,500-document slice of train_D1: merges identical to HF BpeTrainer (same specials,
     same byte alphabet), i.e. the BPE core and tie-breaking match the A1 recipe.
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402
import pickybpe as PB  # noqa: E402


def pretok_counts(texts):
    import regex
    rx = regex.compile(C.P1_PY)
    cnt = collections.Counter()
    for t in texts:
        for m in rx.finditer(t):
            cnt[m.group()] += 1
    return sorted(((w.encode("utf-8"), c) for w, c in cnt.items()), key=lambda x: x[0])


def check_consistency(items, V, tau, label):
    tr = PB.Trainer(items, V, tau, min_frequency=2, log=lambda *a: None).train()
    model = tr.to_model({"test": label})
    fd, path = tempfile.mkstemp(suffix=".json", dir=os.path.join(HERE, "cache"))
    os.close(fd)
    PB.save_model(model, path)
    tk = PB.PickyBPETokenizer(path)
    bad = 0
    for (wb, _), seq in zip(items, tr.words):
        got = list(tk.encode_word_internal(wb))
        if got != seq:
            bad += 1
            if bad < 3:
                print("  MISMATCH", wb, got, seq)
        assert not (set(got) & tk.removed_at_end)
        s = wb.decode("utf-8")
        ids = tk.encode(s)
        assert tk.decode(ids) == s
    st = model["summary"]
    os.remove(path)
    print("%-28s V=%d tau=%s: words %d, mismatches %d, removals %d, removed at end %d, readded %d, "
          "merges %d (new %d, existing %d, readd %d)" % (
              label, V, tau, len(items), bad, st["stats"]["removals"], st["removed_at_end"], st["readded"],
              st["stats"]["merges"], st["stats"]["merges_new_token"], st["stats"]["merges_existing_present"],
              st["stats"]["merges_readd"]))
    assert bad == 0
    return tr, model


def hf_merges(texts, V):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(C.P1_ONIG), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trn = trainers.BpeTrainer(vocab_size=V, min_frequency=2, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=C.SPECIAL_TOKENS)
    tok.train_from_iterator(texts, trainer=trn)
    j = json.loads(tok.to_str())
    mg = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in j["model"]["merges"]]
    return mg, j["model"]["vocab"]


def main():
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    # ---- 1. toy corpora
    rng = random.Random(7)
    alpha = "abcde"
    words = collections.Counter()
    for _ in range(4000):
        L = rng.randint(1, 9)
        w = "".join(rng.choice(alpha[:rng.randint(2, 5)]) for _ in range(L))
        words[(" " if rng.random() < .5 else "") + w] += rng.randint(1, 5)
    items = sorted(((w.encode(), c) for w, c in words.items()), key=lambda x: x[0])
    for V, tau in ((N, t) for N in (PB.N_BASE + 40, PB.N_BASE + 150, PB.N_BASE + 400) for t in (None, 0.5, 0.7, 0.9)):
        check_consistency(items, V, tau, "toy")
    # ---- real-text slice (train_D1 only)
    docs, info = C.load_view("train_D1", "train")
    texts = [d["text"] for d in docs[:1500]]
    items = pretok_counts(texts)
    for tau in (0.6, 0.9):
        check_consistency(items, PB.N_BASE + 1500, tau, "train_D1[:1500]")
    # ---- 2. tau=None vs HF
    V = 4096
    tr, model = check_consistency(items, V, None, "train_D1[:1500] plain")
    mine = []
    for e in model["events"]:
        a, b = e[1], e[2]
        mine.append(("".join(C.B2U[x] for x in tr.tok_bytes[a]), "".join(C.B2U[x] for x in tr.tok_bytes[b])))
    hf, hv = hf_merges(texts, V)
    same = mine == hf
    first = next((k for k, (x, y) in enumerate(zip(mine, hf)) if x != y), None)
    ids_same = all(hv.get("".join(C.B2U[x] for x in tr.tok_bytes[i])) == i for i in range(PB.N_SPECIAL, len(tr.tok_bytes)))
    print("plain BPE vs HF BpeTrainer on train_D1[:1500], V=%d: merges %d vs %d, identical %s, first diff %s, "
          "ids identical %s" % (V, len(mine), len(hf), same, first, ids_same))
    if first is not None:
        print("  mine", mine[first:first + 3], "\n  hf  ", hf[first:first + 3])
    assert same and ids_same
    print("ALL OK")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
