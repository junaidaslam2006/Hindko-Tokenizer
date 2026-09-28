# -*- coding: utf-8 -*-
"""Encoder-equivalence test for the HF-native MinGram candidate (PLAN.md 3, Stage 2 condition iii).

    python equiv_test.py [--run runs/mingram_P1_D1_16384]

Compares, document by document, the ids of
    HF   tokenizers.Tokenizer.from_file(tokenizer.json).encode(text, add_special_tokens=False)   (stock Unigram)
    REF  mingram.MinGramRef(mingram_model.json, tie='hf')   the pure-Python reference Viterbi (lexicographic:
         fewest tokens, then the summed integer log p; exact ties: first path found left to right)
on 100% of dev_strict (the decision set), plus 100% of dev_permissive, the four perturbed versions of
dev_strict (eval/perturb.py), and synthetic stress strings (unknown characters -> byte fallback, special-block
strings, runs of one repeated character, random concatenations of vocabulary pieces).
Also measured (report only):
  * min-token check: for every dev pretoken, the reference path length equals an independent count-only DP;
  * tie rule: how often the paper's exact-tie rule ("longest leading tokens", REF tie='paper') segments a
    document differently from the HF rule (token counts are equal by construction; checked);
  * G1 lossless: decode(encode(doc)) == doc for HF and REF.
Test split: never read (harness.load_docs refuses it).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, HERE)
import mingram as M  # noqa: E402
H = M.H
import perturb as P  # noqa: E402  (eval/, on sys.path through harness)
sys.path.insert(0, r"F:\Hindko\_pipeline")
from hp.normalize import normalize  # noqa: E402


def compare(name, texts, tk, ref, ref_paper=None, min_check_lat=None):
    t_hf = t_ref = 0.0
    n_eq = n_rt_hf = n_rt_ref = 0
    tok_hf = tok_ref = 0
    mism = []
    tie_docs = tie_tokcount_diff = 0
    pre_checked = pre_bad = 0
    longest = 0
    for k, t in enumerate(texts):
        a = time.perf_counter()
        e = tk.encode(t, add_special_tokens=False).ids
        b = time.perf_counter()
        r = ref.encode(t)
        c = time.perf_counter()
        t_hf += b - a
        t_ref += c - b
        tok_hf += len(e)
        tok_ref += len(r)
        if e == r:
            n_eq += 1
        elif len(mism) < 10:
            i = next(x for x, (u, v) in enumerate(zip(e + [-1], r + [-1])) if u != v)
            mism.append({"doc": k, "first_diff": i, "hf": e[max(0, i - 2):i + 4], "ref": r[max(0, i - 2):i + 4]})
        n_rt_hf += tk.decode(e, skip_special_tokens=False) == t
        n_rt_ref += ref.decode(r) == t
        if ref_paper is not None:
            rp = ref_paper.encode(t)
            if rp != r:
                tie_docs += 1
            if len(rp) != len(r):
                tie_tokcount_diff += 1
        if min_check_lat is not None:
            for pt in M.pretokenize(t):
                longest = max(longest, len(pt))
                pre_checked += 1
                if len(min_check_lat.best_hf(pt)) != min_check_lat.min_count(pt):
                    pre_bad += 1
    out = {"set": name, "docs": len(texts), "docs_identical": n_eq, "docs_differing": len(texts) - n_eq,
           "identical_all": n_eq == len(texts), "tokens_hf": tok_hf, "tokens_ref": tok_ref,
           "g1_roundtrip_hf": n_rt_hf, "g1_roundtrip_ref": n_rt_ref, "mismatch_examples": mism,
           "seconds_hf": round(t_hf, 2), "seconds_ref": round(t_ref, 2)}
    if ref_paper is not None:
        out["paper_tie_rule_docs_differing"] = tie_docs
        out["paper_tie_rule_docs_with_token_count_difference"] = tie_tokcount_diff
    if min_check_lat is not None:
        out["min_token_check_pretokens"] = pre_checked
        out["min_token_check_failures"] = pre_bad
        out["longest_pretoken_chars"] = longest
    M.log("%-22s %4d docs: identical %d/%d, tokens hf %d ref %d, G1 hf %d ref %d%s" % (
        name, len(texts), n_eq, len(texts), tok_hf, tok_ref, n_rt_hf, n_rt_ref,
        (", paper-tie docs differing %d" % tie_docs) if ref_paper is not None else ""))
    return out


def quantisation_effect(ref, texts, fine_bits=20):
    """REPORT ONLY: segment every dev_strict pretoken type with the model's 2^-Q_BITS scores and with the same
    log p on a 2^-fine_bits grid; count types and pretoken occurrences whose segmentation differs."""
    fine = {}
    for p in ref.m["pieces"]:
        if p["kind"] in ("char", "learned"):
            fine[p["piece"]] = int(round(p["logp"] * (1 << fine_bits)))
    mn = min(fine.values())
    for p in ref.m["pieces"]:
        if p["kind"] not in ("char", "learned"):
            fine[p["piece"]] = mn
    lat_f = M.Lattice(fine, mn - int(M.HF_UNK_PENALTY * (1 << fine_bits)))
    types = collections.Counter()
    for t in texts:
        types.update(M.pretokenize(t))
    d_types = d_occ = d_len = 0
    for s, f in types.items():
        a = [(i, j) for i, j, _ in ref.lat.best_hf(s)]
        b = [(i, j) for i, j, _ in lat_f.best_hf(s)]
        if a != b:
            d_types += 1
            d_occ += f
            d_len += len(a) != len(b)
    return {"what": "REPORT ONLY: dev_strict pretoken segmentations with the shipped 2^-%d log p grid vs a 2^-%d grid "
                    "of the same log p" % (M.Q_BITS, fine_bits), "pretoken_types": len(types),
            "pretoken_occurrences": sum(types.values()), "types_differing": d_types, "occurrences_differing": d_occ,
            "types_with_token_count_difference": d_len}


def synthetic(ref, seed=20260926):
    rnd = random.Random(seed)
    learned = [p for p, k in zip(ref.pieces, ref.kind) if k == "learned"]
    chars = [p for p, k in zip(ref.pieces, ref.kind) if k == "char"]
    unseen = [chr(0x4E2D), chr(0x6587), chr(0x1F600), chr(0x00E9), chr(0x200C), chr(0x0915), chr(0x05D0),
              chr(0x10348), chr(0xFFFD) if chr(0xFFFD) not in chars else chr(0x2603)]
    unseen = [u for u in unseen if u not in chars]
    out = []
    for s in M.SPECIALS[:8]:
        out.append("اب " + s + " پت" + s + s)
    for u in unseen:
        out.append("کتاب" + u + "دا " + u * 3 + " " + u + "۔")
    for ch in chars:
        if not ch.isspace():
            for L in (2, 3, 5, 8, 13, 40):
                out.append(ch * L)
    for L in (1, 2, 3, 7, 20, 100):
        out.append("a" + " " * L + "b" + "\n" * L + "c")
    for _ in range(3000):
        k = rnd.randint(1, 8)
        s = "".join(rnd.choice(learned) if rnd.random() < 0.8 else rnd.choice(chars) for _ in range(k))
        out.append(s)
    return out, unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join("runs", "mingram_P1_D1_16384"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    from tokenizers import Tokenizer
    tok_path = os.path.join(run, "tokenizer.json")
    model_path = os.path.join(run, "mingram_model.json")
    tk = Tokenizer.from_file(tok_path)
    ref = M.MinGramRef(model_path, "hf")
    ref_paper = M.MinGramRef(model_path, "paper")
    m = ref.m
    kmax = M.C_SHIFT / (abs(m["unk_score_q"]) / (1 << M.Q_BITS))
    res = {"label": M.LABEL, "tokenizer_json": tok_path, "tokenizer_sha256": M.sha256_file(tok_path),
           "model_json": model_path, "model_sha256": M.sha256_file(model_path),
           "tokenizers_version": __import__("tokenizers").__version__,
           "count_dominance_bound_tokens": kmax,
           "count_dominance_note": "the HF scores -C + log p make the stock Viterbi minimise the token count first for "
                                   "any pretoken whose minimum segmentation has fewer than C/|min node score| = %.0f "
                                   "nodes; every dev pretoken is far shorter (longest_pretoken_chars)" % kmax,
           "sets": []}
    for ds in ("dev_strict", "dev_permissive"):
        docs, dinfo = H.load_docs(ds)                          # refuses test-split uids
        texts = [normalize(d["text"]) for d in docs]
        assert sum(1 for d, t in zip(docs, texts) if d["text"] != t) == 0
        r = compare(ds, texts, tk, ref, ref_paper, ref.lat if ds == "dev_strict" else None)
        r["dataset_sha256"] = dinfo["sha256"]
        res["sets"].append(r)
        if ds == "dev_strict":
            res["quantisation_effect"] = quantisation_effect(ref, texts)
            M.log("quantisation effect", res["quantisation_effect"])
            for p in H.PERT_NAMES:
                pt = [P.PERTURBATIONS[p](t).text for t in texts]
                res["sets"].append(compare("dev_strict+" + p, pt, tk, ref))
    syn, unseen = synthetic(ref)
    r = compare("synthetic_stress", syn, tk, ref, ref_paper)
    r["unseen_chars_used"] = ["U+%04X" % ord(u) for u in unseen]
    res["sets"].append(r)
    d0 = res["sets"][0]
    res["verdict"] = {"dev_strict_docs": d0["docs"], "dev_strict_identical": d0["docs_identical"],
                      "hf_native_exact_on_100pct_dev_strict": d0["identical_all"],
                      "all_sets_identical": all(s["identical_all"] for s in res["sets"])}
    out = a.out or os.path.join(run, "equivalence.json")
    json.dump(H.clean(res), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    M.log("verdict", res["verdict"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
