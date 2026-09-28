# -*- coding: utf-8 -*-
"""Toy correctness test of mingram.Lattice (MinGram as reimplemented from the paper).

On random 3-letter vocabularies (two score regimes: spread-out scores, and only two score values so that exact
ties are frequent) it checks
  * best_hf and best_paper are optimal (fewest tokens, then largest summed score) against brute-force enumeration;
  * best_paper returns the optimal segmentation with lexicographically longest leading tokens;
  * min_count equals the brute-force minimum token count;
  * a stock tokenizers Unigram with scores -C + score_q/2^Q_BITS returns exactly best_hf's segmentation.
    python test_lattice.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mingram as M  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402


def run(seed, score_fn, n_brute=200, n_hf=2000):
    rnd = random.Random(seed)
    alpha = "abc"
    pieces = set(alpha)
    while len(pieces) < 14:
        pieces.add("".join(rnd.choice(alpha) for _ in range(rnd.randint(2, 4))))
    table = {p: score_fn(rnd) for p in sorted(pieces)}
    lat = M.Lattice(table, min(table.values()) - int(M.HF_UNK_PENALTY * (1 << M.Q_BITS)))

    def brute(s):
        best = None

        def rec(i, acc):
            nonlocal best
            if i == len(s):
                key = (len(acc), -sum(table[p] for p in acc))
                if best is None or key < best[0]:
                    best = (key, [list(acc)])
                elif key == best[0]:
                    best[1].append(list(acc))
                return
            for j in range(i + 1, len(s) + 1):
                if s[i:j] in table:
                    acc.append(s[i:j]); rec(j, acc); acc.pop()
        rec(0, [])
        return best

    bad = ties = 0
    for _ in range(n_brute):
        s = "".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 10)))
        key, segs = brute(s)
        hf = [p for _, _, p in lat.best_hf(s)]
        pa = [p for _, _, p in lat.best_paper(s)]
        if hf not in segs or pa not in segs or lat.min_count(s) != key[0]:
            bad += 1
        if len(segs) > 1:
            ties += 1
            if pa != max(segs, key=lambda sg: [len(p) for p in sg]):
                bad += 1
    voc = [(p, M.model_float_score(q)) for p, q in table.items()]
    voc.append(("<unk>", M.model_float_score(min(table.values()))))
    tk = Tokenizer(models.Unigram(voc, unk_id=len(voc) - 1, byte_fallback=False))
    mis = 0
    for _ in range(n_hf):
        s = "".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 30)))
        if tk.encode(s, add_special_tokens=False).tokens != [p for _, _, p in lat.best_hf(s)]:
            mis += 1
    return {"seed": seed, "brute_failures": bad, "tie_cases": ties, "hf_mismatches": mis}


if __name__ == "__main__":
    rows = [run(1, lambda r: -r.randint(1, 50) * (1 << (M.Q_BITS - 2))),
            run(7, lambda r: -r.randint(1, 2) * (1 << M.Q_BITS))]
    for r in rows:
        print(r)
    assert all(r["brute_failures"] == 0 and r["hf_mismatches"] == 0 for r in rows)
    print("OK")
