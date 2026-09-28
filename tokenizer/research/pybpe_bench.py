# -*- coding: utf-8 -*-
"""How fast is a pure-Python incremental BPE trainer on this corpus?

Calibrates the PickyBPE / Scaffold-BPE / SuperBPE-stage-2 reimplementation
estimates in PLAN.md. Character-level BPE over pretoken types (P1 regex),
incremental pair counts + inverted index + lazy max-heap. Deterministic
tie-break (count desc, then pair lexicographic). Single process, 1 thread.
Usage: python pybpe_bench.py <n_merges>
"""
import heapq, json, os, sys, time, collections
import regex

SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
# Python `regex` spells code points ‌; Oniguruma (HF tokenizers) spells them \x{200C}.
P1 = r" ?[\p{L}\p{M}‌‍]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
N_MERGES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

def main():
    t0 = time.time()
    pat = regex.compile(P1)
    counts = collections.Counter()
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            counts.update(pat.findall(json.loads(line)["text"]))
    t_pre = time.time() - t0
    items = sorted(counts.items())                     # deterministic order
    words = [list(w) for w, _ in items]; freq = [c for _, c in items]
    n_symbols = sum(len(w) * c for w, c in zip(words, freq))
    pc = collections.Counter(); index = collections.defaultdict(set)
    for i, w in enumerate(words):
        c = freq[i]
        for a, b in zip(w, w[1:]):
            pc[(a, b)] += c; index[(a, b)].add(i)
    heap = [(-c, p) for p, c in pc.items()]; heapq.heapify(heap)
    t_init = time.time() - t0 - t_pre
    marks = {}; t1 = time.time(); merges = 0; touched = 0
    while merges < N_MERGES and heap:
        negc, p = heapq.heappop(heap)
        if pc.get(p, 0) != -negc or -negc < 2:
            continue                                  # stale heap entry
        a, b = p; ab = a + b; merges += 1
        changed = collections.Counter()
        for i in sorted(index.pop(p, ())):
            w = words[i]; c = freq[i]; touched += 1
            j = 0; nw = []
            while j < len(w):
                if j + 1 < len(w) and w[j] == a and w[j + 1] == b:
                    nw.append(ab); j += 2
                else:
                    nw.append(w[j]); j += 1
            if len(nw) == len(w):
                continue
            for x, y in zip(w, w[1:]):
                changed[(x, y)] -= c
            for x, y in zip(nw, nw[1:]):
                changed[(x, y)] += c; index[(x, y)].add(i)
            words[i] = nw
        for q, dc in changed.items():
            if dc == 0: continue
            nc = pc.get(q, 0) + dc
            if nc > 0:
                pc[q] = nc; heapq.heappush(heap, (-nc, q))
            else:
                pc.pop(q, None)
        pc.pop(p, None)
        if merges in (500, 1000, 2000, 4000, 8000, 16000, 24000, 32000):
            marks[merges] = round(time.time() - t1, 1)
            print(json.dumps({"merges": merges, "elapsed_s": marks[merges], "words_touched": touched}), flush=True)
    res = {"pretoken_types": len(words), "symbols_in_corpus": n_symbols, "pretokenize_s": round(t_pre, 1),
           "init_index_s": round(t_init, 1), "merge_elapsed_s_at": marks, "merges_done": merges,
           "note": "pure CPython, 1 thread, char-level, laptop i7-10610U shared with other jobs"}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pybpe_bench.json"), "w") as g:
        json.dump(res, g, indent=1)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
