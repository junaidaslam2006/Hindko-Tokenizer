# -*- coding: utf-8 -*-
"""Verification of a trained PickyBPE model (PLAN.md A7 / Stage 2). Dev only; the test split is never read.

    python verify_pickybpe.py --model models/X.json --a1 models/a1_ref_P1_D1_16k.json [--plain models/P.json]
                              [--hf-attempt] --out verify/X.verify.json

Checks
  V1 trainer/encoder consistency: the event-ordered encoder reproduces the trainer's final segmentation of every
     one of the training pretoken types (from <model>.trainstate.pkl), hence identical train token counts.
  V2 removed tokens never appear: encode_internal() of every dev document (dev_strict and dev_permissive, both
     'validation') and of every training pretoken type contains no token that is removed at the end.
  V3 lossless: decode(encode(doc)) == doc on dev_strict and dev_permissive (G1 itself is run by the harness).
  V4 (with --plain) the tau=None run of the same code has exactly the merges and ids of the HF A1 reference, so
     PickyBPE differs from A1 only through its removals.
  V5 support profile R1 on train_D1 for PickyBPE and for A1 (both encoders), split into leaves and intermediate
     merge-graph nodes, and what happened to A1's under-supported tokens.
  V6 (with --hf-attempt) conversion to a stock HF BPE tokenizer.json: present tokens only, one merge per token
     in the order of its last creation, the merge being the 2-token split the growing merge list gives the
     token's own string; then HF encodings vs the event-ordered encoder on 100% of dev documents. The file is
     shipped only if every dev document is identical (task rule).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402
import pickybpe as PB  # noqa: E402


def load_hf(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


# --------------------------------------------------------------------------------------------------- V1/V2
def v1_consistency(tk, model_path):
    st = pickle.load(open(model_path[:-5] + ".trainstate.pkl", "rb"))
    words, wbytes, counts, tfreq = st["words"], st["word_bytes"], st["counts"], st["tfreq"]
    bad, bad_ex = 0, []
    removed_hits = 0
    enc_freq = collections.Counter()
    t0 = time.time()
    for wb, seq, n in zip(wbytes, words, counts):
        got = list(tk.encode_word_internal(wb))
        if got != seq:
            bad += 1
            if len(bad_ex) < 5:
                bad_ex.append({"word": wb.decode("utf-8", "replace"), "encoder": got, "trainer": seq})
        if set(got) & tk.removed_at_end:
            removed_hits += 1
        for t in got:
            enc_freq[t] += n
    tf_ok = all(enc_freq.get(i, 0) == f for i, f in enumerate(tfreq))
    return {"pretoken_types": len(wbytes), "mismatching_types": bad, "examples": bad_ex,
            "train_token_frequencies_identical": tf_ok,
            "types_containing_a_removed_token": removed_hits, "seconds": round(time.time() - t0, 1),
            "pass": bad == 0 and tf_ok and removed_hits == 0}


def v2_v3_dev(tk):
    out = {}
    for view in ("dev_strict", "dev_permissive"):
        docs, info = C.load_view(view, "validation")
        rt_fail, removed_docs, removed_toks, ntok = [], 0, 0, 0
        used = collections.Counter()
        for d in docs:
            t = d["text"]
            internal = tk.encode_internal(t)
            hits = [i for i in internal if i >= 0 and i in tk.removed_at_end]
            if hits:
                removed_docs += 1
                removed_toks += len(hits)
            ids = tk.encode(t)
            ntok += len(ids)
            used.update(ids)
            if tk.decode(ids) != t:
                rt_fail.append(d["uid"])
        out[view] = {"docs": len(docs), "sha256": info["sha256"], "tokens": ntok,
                     "docs_with_removed_token": removed_docs, "removed_token_occurrences": removed_toks,
                     "roundtrip_failures": len(rt_fail), "roundtrip_fail_uids": rt_fail[:10],
                     "distinct_ids_used": len(used)}
    out["pass"] = all(out[v]["docs_with_removed_token"] == 0 and out[v]["roundtrip_failures"] == 0
                      for v in ("dev_strict", "dev_permissive"))
    return out


# ------------------------------------------------------------------------------------------------------ V4
def v4_plain_vs_hf(plain_path, a1_path):
    m = json.load(open(plain_path, encoding="utf-8"))
    tb = [None if h is None else bytes.fromhex(h) for h in m["tokens"]]
    s = lambda i: "".join(C.B2U[x] for x in tb[i])
    mine = [(s(e[1]), s(e[2])) for e in m["events"]]
    j = json.load(open(a1_path, encoding="utf-8"))
    hf = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in j["model"]["merges"]]
    vocab = j["model"]["vocab"]
    ids_same = all(vocab.get(s(i)) == i for i in range(PB.N_SPECIAL, len(tb)))
    first = next((k for k, (x, y) in enumerate(zip(mine, hf)) if x != y), None)
    return {"plain_model": plain_path, "a1_reference": a1_path, "merges_plain": len(mine), "merges_hf": len(hf),
            "identical_merges": mine == hf, "first_difference": first, "identical_ids": ids_same,
            "removal_events_in_plain": sum(1 for e in m["events"] if e[0] == 1),
            "pass": mine == hf and ids_same}


# ------------------------------------------------------------------------------------------------------ V5
def train_counts(encode_batch, n_ids):
    docs, info = C.load_view("train_D1", "train")
    cnt = np.zeros(n_ids, np.int64)
    for i in range(0, len(docs), 500):
        for ids in encode_batch([d["text"] for d in docs[i:i + 500]]):
            a = np.asarray(ids, dtype=np.int64)
            if a.size:
                cnt += np.bincount(a, minlength=n_ids)[:n_ids]
    return cnt, info


def profile(freq, inter):
    freq = np.asarray(freq)
    inter = np.asarray(inter, dtype=bool)
    return {"learned": int(len(freq)), "freq_eq0": int((freq == 0).sum()), "freq_lt20": int((freq < 20).sum()),
            "freq_lt20_intermediate": int(((freq < 20) & inter).sum()),
            "freq_lt20_leaf": int(((freq < 20) & ~inter).sum()),
            "freq_lt100": int((freq < 100).sum()), "pct_lt20": round(100 * float((freq < 20).mean()), 3),
            "pct_lt100": round(100 * float((freq < 100).mean()), 2), "median": float(np.median(freq)),
            "intermediate_nodes": int(inter.sum())}


def v5_support(tk, a1_path, v1_ok):
    # PickyBPE: the train counts of the final training state equal the encoder's (V1), so use the encoder
    pc, info = train_counts(tk.encode_batch, tk.vocab_size)
    learned = list(range(PB.N_BASE, tk.vocab_size))
    comps = {c for a, b, c2 in tk.merges for c in (a, b)}
    p_inter = [i in comps for i in learned]
    p_freq = pc[learned]
    # A1 reference (HF native encoder)
    hf = load_hf(a1_path)
    j = json.load(open(a1_path, encoding="utf-8"))
    vocab = j["model"]["vocab"]
    ac, _ = train_counts(lambda ts: [e.ids for e in hf.encode_batch(ts, add_special_tokens=False)], len(vocab))
    a_learned = list(range(PB.N_BASE, len(vocab)))
    a_comps = set()
    for x in j["model"]["merges"]:
        a, b = x.split(" ", 1) if isinstance(x, str) else x
        a_comps.add(vocab[a]); a_comps.add(vocab[b])
    a_inter = [i in a_comps for i in a_learned]
    a_freq = ac[a_learned]
    inv = {i: s for s, i in vocab.items()}
    a_bytes = {i: bytes(C.U2B[ch] for ch in inv[i]) for i in a_learned}
    p_bytes = {i: tk.token_bytes(i) for i in learned}
    p_set = set(p_bytes.values())
    a_set = set(a_bytes.values())
    removed_bytes = {tk.itok[i] for i in tk.removed_at_end}
    a_lt20 = [i for i, f in zip(a_learned, a_freq) if f < 20]
    a_lt20_removed = [i for i in a_lt20 if a_bytes[i] in removed_bytes]
    a_lt20_still = [i for i in a_lt20 if a_bytes[i] in p_set]
    p_lt20 = [i for i, f in zip(learned, p_freq) if f < 20]
    # frequency (in PickyBPE's own train counts) of the tokens PickyBPE has that A1 does not
    only_p = [i for i in learned if p_bytes[i] not in a_set]
    only_a = [i for i in a_learned if a_bytes[i] not in p_set]
    fmt = lambda b: b.decode("utf-8", "replace")
    return {
        "train": info,
        "pickybpe": profile(p_freq, p_inter),
        "a1_reference": profile(a_freq, a_inter),
        "vocab_overlap": {"common_learned": len(p_set & a_set), "only_pickybpe": len(only_p), "only_a1": len(only_a),
                          "only_a1_removed_by_pickybpe": sum(1 for i in only_a if a_bytes[i] in removed_bytes),
                          "only_pickybpe_train_freq_median": float(np.median(pc[only_p])) if only_p else None,
                          "only_pickybpe_train_freq_lt20": int((pc[only_p] < 20).sum()) if only_p else 0,
                          "only_a1_train_freq_in_a1_median": float(np.median(ac[only_a])) if only_a else None,
                          "only_a1_train_freq_in_a1_lt20": int((ac[only_a] < 20).sum()) if only_a else 0},
        "a1_lt20_tokens": len(a_lt20), "a1_lt20_removed_by_pickybpe": len(a_lt20_removed),
        "a1_lt20_still_in_pickybpe": len(a_lt20_still),
        "a1_lt20_still_examples": [{"text": fmt(a_bytes[i]), "a1_train_freq": int(ac[i]),
                                    "pickybpe_train_freq": int(pc[[k for k in learned if p_bytes[k] == a_bytes[i]][0]])}
                                   for i in a_lt20_still[:15]],
        "pickybpe_lt20_examples": [{"text": fmt(p_bytes[i]), "train_freq": int(pc[i]),
                                    "intermediate": bool(i in comps)} for i in
                                   sorted(p_lt20, key=lambda k: pc[k])[:25]],
        "train_counts_note": "PickyBPE counts from the event-ordered encoder over train_D1 documents (equal to the "
                             "trainer's final state, V1=%s); A1 counts from the HF native encoder" % v1_ok,
    }


# ------------------------------------------------------------------------------------------------------ V6
def hf_bpe_sim(word, ranks):
    """Rank-ordered BPE on a list of final ids (HF semantics: lowest rank first, left to right)."""
    seq = list(word)
    while len(seq) > 1:
        best, br = None, None
        for j in range(len(seq) - 1):
            r = ranks.get((seq[j], seq[j + 1]))
            if r is not None and (br is None or r[0] < br[0]):
                best, br = (seq[j], seq[j + 1]), r
        if best is None:
            break
        a, b = best
        new, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new.append(br[1]); i += 2
            else:
                new.append(seq[i]); i += 1
        seq = new
    return seq


def v6_hf_attempt(tk, out_json):
    from tokenizers import Tokenizer
    m = tk.m
    last = {}
    for k, e in enumerate(m["events"]):
        if e[0] == 0:
            last[e[3]] = k
    order = sorted((k, c) for c, k in last.items() if c in tk.i2f)
    fbyte = {PB.BYTE_ID[x]: tk.i2f[PB.BYTE_ID[x]] for x in range(256)}
    ranks, merges, unreachable = {}, [], []
    for _, c in order:
        f = tk.i2f[c]
        word = [fbyte[PB.BYTE_ID[x]] for x in tk.itok[c]]
        seq = hf_bpe_sim(word, ranks)
        if len(seq) == 2:
            ranks[(seq[0], seq[1])] = (len(merges), f)
            merges.append((seq[0], seq[1]))
        else:
            unreachable.append(f)
    vocab = {tk.id_to_token(i): i for i in range(tk.vocab_size)}
    j = {"version": "1.0", "truncation": None, "padding": None,
         "added_tokens": [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                           "normalized": False, "special": True} for i, s in enumerate(m["special_tokens"])],
         "normalizer": None,
         "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
             {"type": "Split", "pattern": {"Regex": C.P1_ONIG}, "behavior": "Isolated", "invert": False},
             {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
         "post_processor": None,
         "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
         "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                   "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                   "vocab": vocab,
                   "merges": [[tk.id_to_token(a), tk.id_to_token(b)] for a, b in merges]}}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)
    hf = Tokenizer.from_file(out_json)
    res = {"merges": len(merges), "hf_unreachable_present_tokens": len(unreachable),
           "hf_unreachable_examples": [tk.token_bytes(i).decode("utf-8", "replace") for i in unreachable[:10]]}
    for view in ("dev_strict", "dev_permissive"):
        docs, _ = C.load_view(view, "validation")
        diff_docs, diff_pt, n_pt, ex = 0, 0, 0, []
        for d in docs:
            t = d["text"]
            a = tk.encode(t)
            b = hf.encode(t, add_special_tokens=False).ids
            if a != b:
                diff_docs += 1
            for pt in tk.pretokens(t):
                n_pt += 1
                pa = tk.encode(pt)
                pb = hf.encode(pt, add_special_tokens=False).ids
                if pa != pb:
                    diff_pt += 1
                    if len(ex) < 8:
                        ex.append({"pretoken": pt, "pickybpe": [tk.id_to_token(i) for i in pa],
                                   "hf": [tk.id_to_token(i) for i in pb]})
        res[view] = {"docs": len(docs), "docs_different": diff_docs, "pretokens": n_pt,
                     "pretokens_different": diff_pt, "examples": ex}
    res["exact_on_100pct_dev"] = all(res[v]["docs_different"] == 0 for v in ("dev_strict", "dev_permissive"))
    res["file_sha256"] = C.sha256_file(out_json)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--plain")
    ap.add_argument("--hf-attempt", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path = a.model if os.path.isabs(a.model) else os.path.join(HERE, a.model)
    a1 = a.a1 if os.path.isabs(a.a1) else os.path.join(HERE, a.a1)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rep = {"model": path, "model_sha256": C.sha256_file(path), "a1_reference": a1, "a1_sha256": C.sha256_file(a1),
           "frozen": C.verify_frozen()}
    tk = PB.PickyBPETokenizer(path)
    C.log("V1 trainer/encoder consistency")
    rep["V1_consistency"] = v1_consistency(tk, path)
    C.log(json.dumps({k: v for k, v in rep["V1_consistency"].items() if k != "examples"}))
    C.log("V2/V3 dev encodings")
    rep["V2_V3_dev"] = v2_v3_dev(tk)
    C.log(json.dumps(rep["V2_V3_dev"], ensure_ascii=False)[:600])
    if a.plain:
        pp = a.plain if os.path.isabs(a.plain) else os.path.join(HERE, a.plain)
        rep["V4_plain_vs_hf"] = v4_plain_vs_hf(pp, a1)
        C.log(json.dumps(rep["V4_plain_vs_hf"]))
    C.log("V5 support profiles")
    rep["V5_support"] = v5_support(tk, a1, rep["V1_consistency"]["pass"])
    C.log(json.dumps({k: rep["V5_support"][k] for k in ("pickybpe", "a1_reference", "vocab_overlap")}))
    if a.hf_attempt:
        C.log("V6 HF conversion attempt")
        hp = os.path.join(HERE, "cache", "hf_attempt_" + os.path.basename(path))
        rep["V6_hf_attempt"] = v6_hf_attempt(tk, hp)
        v6 = rep["V6_hf_attempt"]
        C.log(json.dumps({k: v for k, v in v6.items() if k not in ("dev_strict", "dev_permissive")}, ensure_ascii=False))
        for v in ("dev_strict", "dev_permissive"):
            C.log(v, json.dumps({k: x for k, x in v6[v].items() if k != "examples"}))
        if not v6["exact_on_100pct_dev"]:
            os.remove(hp)
            v6["file"] = "deleted: not exact on 100% of dev (task rule)"
        else:
            v6["file"] = hp
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("wrote", out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
