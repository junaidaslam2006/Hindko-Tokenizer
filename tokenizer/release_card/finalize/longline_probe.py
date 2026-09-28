# -*- coding: utf-8 -*-
"""Where do sp.model (float32 Viterbi) and tokenizer.json (float64 Viterbi) start to differ on long single lines?

Real text only: every document of dev_strict (or another view) is joined into one stream, "\\n" -> " ", and cut into
consecutive non-overlapping windows of L characters (a window is one line: no newline). For each L the script counts
windows whose SentencePiece ids differ from tokenizer.json ids, and classifies every difference with the exact score
of both segmentations (scores are multiples of 2^-14, so float64 sums are exact):
  exact tie  : equal total score (the two encoders resolved an exact tie differently)
  sp lower   : SentencePiece returned a path with a strictly lower exact score (float32 rounding flipped a near-tie)
Reports the shortest divergent window. Report-only; changes no file of the release.

    set PYTHONIOENCODING=utf-8
    python release_card\\finalize\\longline_probe.py [--view dev_strict] [--max-windows 400]
"""
import argparse
import datetime
import json
import os
import time

REL = r"F:\Hindko\tokenizer"
DATA = r"F:\Hindko\_tokenizer\data"
OUT_DIR = r"F:\Hindko\_tokenizer\release_card\finalize"
LENGTHS = [1000, 2000, 2500, 3000, 3400, 4000, 4500, 5000, 6000, 8000, 10000, 16000, 32000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="dev_strict")
    ap.add_argument("--max-windows", type=int, default=400, help="windows per length (0 = all)")
    a = ap.parse_args()
    import sentencepiece as spm
    from tokenizers import Tokenizer
    sp = spm.SentencePieceProcessor(model_file=os.path.join(REL, "sp.model"))
    tk = Tokenizer.from_file(os.path.join(REL, "tokenizer.json"))
    tj = json.load(open(os.path.join(REL, "tokenizer.json"), encoding="utf-8"))
    score = [s for _, s in tj["model"]["vocab"]]
    with open(os.path.join(DATA, a.view + ".jsonl"), encoding="utf-8") as f:
        stream = " ".join(json.loads(l)["text"].replace("\n", " ") for l in f)
    res = {"what": __doc__.split("\n")[0], "view": a.view, "stream_chars": len(stream),
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "files": {"sp.model": "F:/Hindko/tokenizer/sp.model", "tokenizer.json": "F:/Hindko/tokenizer/tokenizer.json"},
           "by_length": [], "shortest_divergent_window_chars": None, "examples": []}
    t0 = time.time()
    shortest = None
    for L in LENGTHS:
        n = len(stream) // L
        if a.max_windows:
            n = min(n, a.max_windows)
        wins = [stream[k * L:(k + 1) * L] for k in range(n)]
        hf = [e.ids for e in tk.encode_batch(wins)]
        spi = sp.encode(wins)
        diff = tie = lower = higher = same_len = 0
        for w, x, y in zip(wins, spi, hf):
            if x == y:
                continue
            diff += 1
            sx, sy = sum(score[i] for i in x), sum(score[i] for i in y)
            if sx == sy:
                tie += 1
            elif sx < sy:
                lower += 1
            else:
                higher += 1
            same_len += len(x) == len(y)
            if shortest is None or L < shortest:
                shortest = L
            if len(res["examples"]) < 8:
                k = next(i for i, (p, q) in enumerate(zip(x, y)) if p != q) if any(
                    p != q for p, q in zip(x, y)) else min(len(x), len(y))
                res["examples"].append({"L": L, "first_diff_token": k, "sp_tokens": len(x), "hf_tokens": len(y),
                                        "sp_pieces": [sp.id_to_piece(i) for i in x[k:k + 4]],
                                        "hf_pieces": [sp.id_to_piece(i) for i in y[k:k + 4]],
                                        "score_gap_hf_minus_sp": sy - sx})
        row = {"L": L, "windows": n, "differ": diff, "differ_pct": round(100.0 * diff / n, 2) if n else None,
               "exact_tie": tie, "sp_lower_exact_score": lower, "sp_higher_exact_score": higher,
               "same_token_count": same_len}
        res["by_length"].append(row)
        print(json.dumps(row), round(time.time() - t0, 1), flush=True)
    res["shortest_divergent_window_chars"] = shortest
    res["seconds"] = round(time.time() - t0, 1)
    out = os.path.join(OUT_DIR, "longline_probe_%s.json" % a.view)
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
