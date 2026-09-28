# -*- coding: utf-8 -*-
"""SuperBPE stage 2 (reimplemented from Liu et al. 2025) on the stock HF `tokenizers` Rust trainer, via the
Private-Use-Area code-point trick of SOTA_TOKENIZATION.md section 4, plus an independent pure-Python trainer.

    python build_stage2.py --stage1 stage1/a1_p1_14746.json --T 16384 --out sbpe_16k_t090 [--threads 3]
                           [--py-check] [--force-py]

Stage 1 is A1-P1 trained to t (the vocabulary size at the transition, 64 specials + 256 bytes included).
Stage 2 continues BPE to T WITHOUT whitespace pre-tokenization:
  1. every train_D1 document is split with the stage-2 pre-tokenizer S2 (common.py): runs of letter-words
     separated by single spaces form ONE chunk; chunks end only at punctuation/symbols, digits and line
     breaks. The chunk is encoded with the stage-1 merges (HF BPE, merges by rank inside the chunk);
     verify: this equals the stage-1 (P1) encoding of the same document, i.e. no stage-1 merge crosses a
     P1 boundary inside a chunk.
  2. every stage-1 id i becomes the code point U+F0000+i; each chunk becomes a PUA string.
  3. the stock BpeTrainer (no normalizer, no pre-tokenizer: one chunk = one word; min_frequency 2;
     initial_alphabet = the t-64 PUA characters of the non-special stage-1 ids) learns the new tokens
     until the PUA vocabulary holds (t-64) + (T-t) entries.
  4. PUA merges are translated back to byte-level strings and appended (ranks after all stage-1 merges)
     to the stage-1 merges; the final tokenizer.json uses S2 as its pre-tokenizer.
  Superword cap (Liu et al.: at most 4 words per token): the stock trainer has no such option. The
  pure-Python trainer (py_train) implements the same algorithm (HF tie-break: highest count, then the
  smallest id pair; left-to-right non-overlapping merges; min_frequency 2) with and without the cap.
  --py-check runs it uncapped and requires identical merges to the Rust run (an independent
  cross-check). If the uncapped result contains no token with more than 4 words, the cap never binds and
  the capped result is identical, so the Rust result is kept. Otherwise the capped pure-Python result is
  used (recorded in build_info.json).
Outputs in OUT/: tokenizer.json, stage2_pua.json (the PUA-space merges as stage-1 id sequences, for the
reference encoder), build_info.json. Work files (chunk counts) in work/.
"""
import argparse
import collections
import heapq
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

WORK = os.path.join(C.HERE, "work")


# ------------------------------------------------------------------------------------------ chunks
def stage1_s2_json(stage1_json: dict) -> dict:
    j = json.loads(json.dumps(stage1_json))
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = C.S2
    return j


def build_chunks(stage1_path: str, texts, threads: int):
    """Counter of chunk id-tuples (stage-1 ids, S2 chunks, >= 2 tokens) + statistics + P1 equivalence."""
    from tokenizers import Tokenizer
    s1 = json.load(open(stage1_path, encoding="utf-8"))
    tk_p1 = Tokenizer.from_file(stage1_path)
    tk_s2 = Tokenizer.from_str(json.dumps(stage1_s2_json(s1), ensure_ascii=False))
    cnt = collections.Counter()
    st = collections.Counter()
    len_hist = collections.Counter()
    words_hist = collections.Counter()
    mismatch_docs = 0
    B = 400
    t0 = time.time()
    for k in range(0, len(texts), B):
        batch = texts[k:k + B]
        e2 = tk_s2.encode_batch(batch, add_special_tokens=False)
        e1 = tk_p1.encode_batch(batch, add_special_tokens=False)
        for a, b in zip(e1, e2):
            ids = b.ids
            if a.ids != ids:
                mismatch_docs += 1
            wid = b.word_ids
            st["tokens"] += len(ids)
            start = 0
            n = len(ids)
            for i in range(1, n + 1):
                if i == n or wid[i] != wid[start]:
                    L = i - start
                    st["chunks"] += 1
                    len_hist[min(L, 64)] += 1
                    if L >= 2:
                        cnt[tuple(ids[start:i])] += 1
                        st["chunks_ge2"] += 1
                        st["tokens_in_chunks_ge2"] += L
                    start = i
    st["docs"] = len(texts)
    st["unique_chunks_ge2"] = len(cnt)
    st["seconds"] = round(time.time() - t0, 1)
    st["stage1_S2_vs_P1_mismatch_docs"] = mismatch_docs
    st["chunk_len_tokens_hist_capped64"] = dict(sorted(len_hist.items()))
    return cnt, dict(st)


def load_or_build_chunks(stage1_path, threads, work=WORK):
    os.makedirs(work, exist_ok=True)
    sha = C.sha256_file(stage1_path)
    cache = os.path.join(work, "chunks_%s.pkl" % sha[:16])
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            d = pickle.load(f)
        if d["stage1_sha256"] == sha and d["S2"] == C.S2:
            C.log("chunks from cache", cache)
            return d["counts"], d["stats"], cache
    docs, info = C.load_view("train_D1")
    cnt, st = build_chunks(stage1_path, [d["text"] for d in docs], threads)
    st["train_view"] = info
    with open(cache, "wb") as f:
        pickle.dump({"stage1_sha256": sha, "S2": C.S2, "counts": cnt, "stats": st}, f, protocol=4)
    C.log("chunks built: %s" % {k: v for k, v in st.items() if k not in ("chunk_len_tokens_hist_capped64", "train_view")})
    return cnt, st, cache


# ------------------------------------------------------------------------------------------ HF (Rust) trainer
def pua(i: int) -> str:
    return chr(C.PUA_BASE + i)


def hf_train_pua(cnt, t: int, n_new: int, threads: int):
    """Stock BpeTrainer on PUA strings. Returns merges as [(seqA, seqB)] of stage-1 id tuples, in order."""
    from tokenizers import Tokenizer, models, trainers
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    tk = Tokenizer(models.BPE())
    alphabet = [pua(i) for i in range(C.N_SPECIAL, t)]
    trn = trainers.BpeTrainer(vocab_size=len(alphabet) + n_new, min_frequency=C.MIN_FREQUENCY, show_progress=False,
                              initial_alphabet=alphabet, special_tokens=[])
    items = sorted(cnt.items())                                  # fixed feed order (counts are order-free)

    def gen():
        for seq, c in items:
            s = "".join(pua(i) for i in seq)
            for _ in range(c):
                yield s
    t0 = time.time()
    tk.train_from_iterator(gen(), trainer=trn, length=sum(c for _, c in items))
    dt = time.time() - t0
    j = json.loads(tk.to_str())
    vocab = j["model"]["vocab"]
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in j["model"]["merges"]]
    unpua = lambda s: tuple(ord(ch) - C.PUA_BASE for ch in s)
    out = [(unpua(a), unpua(b)) for a, b in merges]
    return out, {"seconds": round(dt, 1), "pua_vocab": len(vocab), "pua_merges": len(merges),
                 "pua_alphabet": len(alphabet), "target_pua_vocab": len(alphabet) + n_new}


# ------------------------------------------------------------------------------------------ pure-Python trainer
def py_train(cnt, t: int, n_new: int, tok_bytes, max_words=None, min_freq=C.MIN_FREQUENCY, log_every=500):
    """Pure-Python BPE continuation with HF BpeTrainer semantics (see module docstring).
    tok_bytes(stage-1 id) -> bytes. Returns merges [(seqA, seqB)] and stats."""
    t0 = time.time()
    items = sorted(cnt.items())
    words = [list(s) for s, _ in items]
    freq = [c for _, c in items]
    seq_of = {}                                   # id -> stage-1 id sequence (tuple)
    byt = {}                                      # id -> bytes
    for i in range(C.N_SPECIAL, t):
        seq_of[i] = (i,)
        byt[i] = tok_bytes(i)
    id_of_seq = {v: k for k, v in seq_of.items()}
    pc = collections.defaultdict(int)
    index = collections.defaultdict(list)
    for i, w in enumerate(words):
        c = freq[i]
        for p in zip(w, w[1:]):
            pc[p] += c
        for p in set(zip(w, w[1:])):
            index[p].append(i)
    heap = [(-c, p[0], p[1]) for p, c in pc.items()]
    heapq.heapify(heap)
    next_id = t
    new_unique = 0
    merges = []
    forbidden = 0
    forbidden_set = set()
    touched = 0
    while new_unique < n_new and heap:
        negc, a, b = heapq.heappop(heap)
        c = -negc
        cur = pc.get((a, b), 0)
        if cur != c:
            continue                              # stale: every count change pushed a fresh entry
        if c < min_freq:
            break
        if (a, b) in forbidden_set:
            continue
        nb = byt[a] + byt[b]
        if max_words is not None and C.n_words(nb) > max_words:
            forbidden_set.add((a, b))
            forbidden += 1
            continue
        nseq = seq_of[a] + seq_of[b]
        nid = id_of_seq.get(nseq)
        if nid is None:
            nid = next_id
            next_id += 1
            seq_of[nid] = nseq
            byt[nid] = nb
            id_of_seq[nseq] = nid
            new_unique += 1
        merges.append((seq_of[a], seq_of[b]))
        changed = collections.defaultdict(int)
        for i in sorted(set(index.pop((a, b), ()))):
            w = words[i]
            if len(w) < 2:
                continue
            nw = []
            j = 0
            L = len(w)
            hit = False
            while j < L:
                if j + 1 < L and w[j] == a and w[j + 1] == b:
                    nw.append(nid)
                    j += 2
                    hit = True
                else:
                    nw.append(w[j])
                    j += 1
            if not hit:
                continue
            touched += 1
            fc = freq[i]
            for p in zip(w, w[1:]):
                changed[p] -= fc
            for p in zip(nw, nw[1:]):
                changed[p] += fc
                if p[0] == nid or p[1] == nid:
                    index[p].append(i)
            words[i] = nw
        for p, dc in changed.items():
            if dc == 0:
                continue
            v = pc.get(p, 0) + dc
            if v > 0:
                pc[p] = v
                heapq.heappush(heap, (-v, p[0], p[1]))
            else:
                pc.pop(p, None)
        pc.pop((a, b), None)
        if log_every and len(merges) % log_every == 0:
            C.log("py_train merges %d new %d (%.0fs)" % (len(merges), new_unique, time.time() - t0))
    return merges, {"seconds": round(time.time() - t0, 1), "merges": len(merges), "new_unique": new_unique,
                    "max_words": max_words, "pairs_forbidden_by_cap": forbidden, "word_types_touched": touched}


# ------------------------------------------------------------------------------------------ assemble
def assemble(stage1_path, merges_pua, T):
    """Final byte-level tokenizer.json dict from the stage-1 file and the PUA-space merges."""
    s1 = json.load(open(stage1_path, encoding="utf-8"))
    vocab = dict(s1["model"]["vocab"])
    id2tok = {i: s for s, i in vocab.items()}
    s1_merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in s1["model"]["merges"]]
    seq_str = lambda seq: "".join(id2tok[i] for i in seq)
    new_merges = []
    dup_existing = []
    words_hist = collections.Counter()
    over_cap = []
    for sa, sb in merges_pua:
        a, b = seq_str(sa), seq_str(sb)
        new = a + b
        if new in vocab:
            dup_existing.append({"token": new, "seq": list(sa + sb)})
        else:
            vocab[new] = len(vocab)
        nb = C.bl_to_bytes(new)
        nw = C.n_words(nb)
        words_hist[nw] += 1
        if nw > C.MAX_WORDS:
            over_cap.append(nb.decode("utf-8", errors="replace"))
        new_merges.append((a, b))
    pairs = s1_merges + new_merges
    dup_pairs = len(pairs) - len(set(pairs))
    j = C.hf_bytelevel_bpe_json(vocab, pairs, C.S2)
    info = {"vocab_size": len(vocab), "target_T": T, "stage1_merges": len(s1_merges), "stage2_merges": len(new_merges),
            "stage2_new_tokens": len(vocab) - len(s1["model"]["vocab"]) - 0,
            "stage2_tokens_duplicating_existing": len(dup_existing), "duplicate_examples": dup_existing[:10],
            "duplicate_merge_pairs": dup_pairs, "stage2_token_words_hist": dict(sorted(words_hist.items())),
            "stage2_tokens_over_cap": len(over_cap), "over_cap_examples": over_cap[:20]}
    return j, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--py-check", action="store_true", help="run the pure-Python trainer uncapped and compare")
    ap.add_argument("--force-py", action="store_true", help="use the capped pure-Python trainer's merges")
    ap.add_argument("--work", default=WORK, help="chunk-cache directory (G4 retrains use a fresh one)")
    a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    t_all = time.time()
    fz = C.verify_frozen()
    out = a.out if os.path.isabs(a.out) else os.path.join(C.HERE, a.out)
    os.makedirs(out, exist_ok=True)
    s1 = json.load(open(a.stage1, encoding="utf-8"))
    t = len(s1["model"]["vocab"])
    assert len(s1["model"]["merges"]) == t - C.N_SPECIAL - C.N_BYTES, "stage-1 has duplicate merges"
    n_new = a.T - t
    C.log("stage1 %s: t=%d (t/T=%.4f), stage-2 new tokens %d" % (a.stage1, t, t / a.T, n_new))
    id2tok = {i: s for s, i in s1["model"]["vocab"].items()}
    tok_bytes = lambda i: C.bl_to_bytes(id2tok[i])

    t0 = time.time()
    cnt, cstats, cache = load_or_build_chunks(a.stage1, a.threads, a.work)
    t_chunks = time.time() - t0
    if cstats["stage1_S2_vs_P1_mismatch_docs"]:
        raise SystemExit("stage-1 merges applied inside S2 chunks differ from the P1 encoding in %d docs"
                         % cstats["stage1_S2_vs_P1_mismatch_docs"])

    merges_hf, hf_info = hf_train_pua(cnt, t, n_new, a.threads)
    C.log("HF PUA trainer: %s" % hf_info)
    _, pre_info = assemble(a.stage1, merges_hf, a.T)
    route = "hf_rust_pua"
    py_info = None
    py_capped_info = None
    merges_final = merges_hf
    if a.py_check:
        merges_py, py_info = py_train(cnt, t, n_new, tok_bytes, max_words=None)
        py_info["identical_to_hf_rust_merges"] = merges_py == merges_hf
        if not py_info["identical_to_hf_rust_merges"]:
            k = next((i for i, (x, y) in enumerate(zip(merges_py, merges_hf)) if x != y), min(len(merges_py), len(merges_hf)))
            py_info["first_difference_at"] = k
            py_info["py_at"] = [list(map(list, merges_py[k]))] if k < len(merges_py) else None
            py_info["hf_at"] = [list(map(list, merges_hf[k]))] if k < len(merges_hf) else None
        C.log("pure-Python uncapped: %s" % py_info)
    if pre_info["stage2_tokens_over_cap"] or a.force_py:
        merges_final, py_capped_info = py_train(cnt, t, n_new, tok_bytes, max_words=C.MAX_WORDS)
        route = "pure_python_capped"
        C.log("cap binds (%d tokens > %d words): using capped pure-Python merges %s"
              % (pre_info["stage2_tokens_over_cap"], C.MAX_WORDS, py_capped_info))
    j, info = assemble(a.stage1, merges_final, a.T)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    path = os.path.join(out, "tokenizer.json")
    tk.save(path)
    vs = tk.get_vocab_size(with_added_tokens=True)
    pua_path = os.path.join(out, "stage2_pua.json")
    with open(pua_path, "w", encoding="utf-8") as f:
        json.dump({"stage1": os.path.abspath(a.stage1), "stage1_sha256": C.sha256_file(a.stage1), "t": t, "T": a.T,
                   "route": route, "merges": [[list(x), list(y)] for x, y in merges_final]}, f)
    bi = {"what": "SuperBPE (reimplemented from Liu et al. 2025) - stage 1 A1-P1 to t, stage 2 to T on S2 chunks",
          "t": t, "T": a.T, "t_over_T": t / a.T, "vocab_size_loaded": vs, "route": route,
          "stage1": {"path": os.path.abspath(a.stage1), "sha256": C.sha256_file(a.stage1),
                     "info": json.load(open(a.stage1 + ".info.json", encoding="utf-8")) if os.path.exists(a.stage1 + ".info.json") else None},
          "S2_regex_oniguruma": C.S2, "P1_regex_oniguruma": C.P1, "max_words_cap": C.MAX_WORDS,
          "pua_base": hex(C.PUA_BASE), "min_frequency": C.MIN_FREQUENCY, "rayon_threads": a.threads,
          "chunks": {k: v for k, v in cstats.items()}, "chunks_seconds_this_run": round(t_chunks, 1), "chunk_cache": cache,
          "hf_rust_pua_trainer": hf_info, "hf_rust_uncapped_assembly": pre_info,
          "pure_python_uncapped": py_info, "pure_python_capped": py_capped_info,
          "assembly": info, "tokenizer_json": path, "tokenizer_sha256": C.sha256_file(path),
          "stage2_pua_json_sha256": C.sha256_file(pua_path),
          "code_sha256": {"build_stage2.py": C.sha256_file(os.path.abspath(__file__)),
                          "common.py": C.sha256_file(os.path.join(C.HERE, "common.py"))},
          "split_manifest_sha256": fz["split_manifest_sha256"], "normalize_version": fz["normalize_version"],
          "tokenizers_version": __import__("tokenizers").__version__, "total_seconds": round(time.time() - t_all, 1)}
    json.dump(bi, open(os.path.join(out, "build_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("wrote %s vocab %d sha256 %s route %s (%.0fs)" % (path, vs, bi["tokenizer_sha256"][:16], route, bi["total_seconds"]))
    if vs != a.T:
        C.log("WARNING: loaded vocab %d != T %d" % (vs, a.T))


if __name__ == "__main__":
    main()
