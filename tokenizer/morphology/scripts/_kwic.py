"""Review helper: keyword-in-context lines from the STRICT corpus (read-only).

    python scripts/_kwic.py N WORD [WORD ...]

Prints up to N occurrences per word, evenly spaced over all its occurrences in
corpus order (deterministic), each with 6 tokens of left and right context
inside the sentence (same tokenisation as context_profiles.py).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from context_profiles import STRICT, sentences  # noqa: E402

W = 6


def main():
    n = int(sys.argv[1])
    words = sys.argv[2:]
    want = set(words)
    hits = {w: [] for w in words}
    for line in open(STRICT, encoding='utf-8'):
        r = json.loads(line)
        for toks in sentences(r['text']):
            for i, t in enumerate(toks):
                if t in want:
                    hits[t].append((' '.join(toks[max(0, i - W):i]), ' '.join(toks[i + 1:i + 1 + W]), r['source']))
    for w in words:
        h = hits[w]
        print(f'=== {w}  ({len(h)} occurrences)')
        if not h:
            continue
        step = max(1, len(h) / n)
        idx = sorted({int(k * step) for k in range(min(n, len(h)))})
        for k in idx:
            left, right, src = h[k]
            print(f'  {left:>45} [{w}] {right}   <{src[:4]}>')


if __name__ == '__main__':
    main()
