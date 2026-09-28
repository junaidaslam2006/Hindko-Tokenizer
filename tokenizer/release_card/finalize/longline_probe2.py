# -*- coding: utf-8 -*-
"""Long single lines, part 2: (a) at which character position does the running Viterbi path score of real text leave
the float32-exact range |score| < 1024? (b) how often do sp.model and tokenizer.json differ when a tie-prone run
(three copies of a piece X for which XX is also a piece, e.g. '۔۔۔', '۱۱۱', '«««') follows L characters of real text
on the same line?

(a) uses the tokenizer.json best path (exact, float64) of consecutive non-overlapping 32,000-character windows of the
    stream of dev_strict documents ("\\n" -> " "); the running score is the prefix sum of piece scores along that path.
(b) text = window[:L] + " " + run, for every window and every run of tie_families.json (X+X+X).
Report-only; changes no file of the release.

    set PYTHONIOENCODING=utf-8
    python release_card\\finalize\\longline_probe2.py
"""
import datetime
import json
import os
import time

import numpy as np

REL = r"F:\Hindko\tokenizer"
DATA = r"F:\Hindko\_tokenizer\data"
TIES = r"F:\Hindko\_tokenizer\release_build\sp32k\stress\tie_families.json"
OUT = r"F:\Hindko\_tokenizer\release_card\finalize\longline_probe2_dev_strict.json"
BOUND = 1024.0
LENGTHS = [500, 1000, 2000, 3000, 3400, 4000, 5000, 6000, 8000, 12000, 16000, 24000, 31000]


def main():
    import sentencepiece as spm
    from tokenizers import Tokenizer
    sp = spm.SentencePieceProcessor(model_file=os.path.join(REL, "sp.model"))
    tk = Tokenizer.from_file(os.path.join(REL, "tokenizer.json"))
    tj = json.load(open(os.path.join(REL, "tokenizer.json"), encoding="utf-8"))
    score = np.array([s for _, s in tj["model"]["vocab"]], dtype=np.float64)
    with open(os.path.join(DATA, "dev_strict.jsonl"), encoding="utf-8") as f:
        stream = " ".join(json.loads(l)["text"].replace("\n", " ") for l in f)
    W = 32000
    wins = [stream[k * W:(k + 1) * W] for k in range(len(stream) // W)]
    t0 = time.time()
    # (a) first character position at which the running best-path score leaves the float32-exact range
    cross = []
    for w in wins:
        e = tk.encode(w)
        run = np.cumsum(score[e.ids])
        k = np.nonzero(np.abs(run) >= BOUND)[0]
        cross.append(int(e.offsets[k[0]][1]) if len(k) else None)
    got = sorted(c for c in cross if c is not None)
    qa = {"windows": len(wins), "window_chars": W, "windows_crossing": len(got),
          "first_crossing_char_min": got[0] if got else None,
          "first_crossing_char_p10": int(np.percentile(got, 10)) if got else None,
          "first_crossing_char_median": int(np.median(got)) if got else None,
          "first_crossing_char_p90": int(np.percentile(got, 90)) if got else None,
          "first_crossing_char_max": got[-1] if got else None}
    print("(a)", json.dumps(qa), flush=True)
    # (b) tie-prone runs after L characters of real text
    fams = json.load(open(TIES, encoding="utf-8"))
    runs = [x * 3 for x in fams]
    rows = []
    for L in LENGTHS:
        texts = [w[:L] + " " + r for w in wins for r in runs]
        hf = [e.ids for e in tk.encode_batch(texts)]
        spi = sp.encode(texts)
        d = sum(1 for x, y in zip(spi, hf) if x != y)
        ties = sum(1 for x, y in zip(spi, hf) if x != y and score[x].sum() == score[y].sum())
        rows.append({"L": L, "items": len(texts), "differ": d, "differ_pct": round(100.0 * d / len(texts), 2),
                     "differ_exact_tie": ties})
        print("(b)", json.dumps(rows[-1]), round(time.time() - t0, 1), flush=True)
    res = {"what": __doc__.split("\n")[0], "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "bound": BOUND, "a_running_score_bound_crossing": qa,
        "b_tie_runs_after_L_chars": {"runs": runs, "rows": rows}, "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
