# -*- coding: utf-8 -*-
"""Exactness checks for a SuperBPE build (PLAN 3 "after Stage 2", SOTA section 4 mandatory verification).

    python verify.py OUT_DIR [--train-every K]      writes OUT_DIR/verify.json
    python verify.py --s2-diff                      Oniguruma vs Python-regex S2 differential -> s2_differential.json

For OUT_DIR:
  1. encoder equivalence: HF-native tokenizer.json ids == reference encoder ids (ref_encoder.py) on 100% of
     dev_strict and dev_permissive documents (+ every K-th train_D1 document);
  2. phase 1 of the reference == HF stage-1 tokenizer (P1) on the same documents;
  3. G1 round trip (HF decode(encode(doc)) == doc) on 100% of dev_strict and dev_permissive;
  4. every final token boundary is a stage-1 boundary (stage-2 tokens are unions of stage-1 tokens);
  5. transformers PreTrainedTokenizerFast(tokenizer_file=...) gives the same ids (HF-native loading);
  6. superword usage statistics on dev_strict.
Never reads the test split (common.load_view refuses it).
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import ref_encoder as R  # noqa: E402


def s2_differential():
    import regex
    from tokenizers import pre_tokenizers, Regex
    sys.path.insert(0, C.EVAL_DIR)
    import perturb as P
    hf = pre_tokenizers.Split(Regex(C.S2), behavior="isolated")
    rx = regex.compile(C.onig_to_py(C.S2))
    out = {"S2_oniguruma": C.S2, "S2_python": C.onig_to_py(C.S2), "regex_version": regex.__version__,
           "tokenizers_version": __import__("tokenizers").__version__, "results": {}}
    for name in ("dev_strict", "dev_permissive", "dev_strict+zwnj", "train_D1"):
        docs, _ = C.load_view(name.split("+")[0], check_normalized=False)
        nd, nt, nmw, ex = 0, 0, 0, []
        for d in docs:
            t = d["text"] if "+" not in name else P.perturb_zwnj(d["text"]).text
            a = [s for s, _ in hf.pre_tokenize_str(t)]
            b = R.py_pretokens(rx, t)
            nt += len(a)
            nmw += sum(1 for s in a if " " in s.strip())
            if a != b:
                nd += 1
                if len(ex) < 3:
                    k = next(i for i, (x, y) in enumerate(zip(a + [None], b + [None])) if x != y)
                    ex.append({"uid": d["uid"], "at": k, "onig": a[k:k + 3], "python": b[k:k + 3]})
        out["results"][name] = {"docs": len(docs), "docs_differing": nd, "chunks": nt, "multiword_chunks": nmw,
                                "examples": ex}
        C.log("S2 differential %s: %d/%d docs differ, %d chunks (%d multi-word)" % (name, nd, len(docs), nt, nmw))
    out["identical"] = all(v["docs_differing"] == 0 for v in out["results"].values())
    json.dump(out, open(os.path.join(C.HERE, "s2_differential.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def boundaries(ids, tb):
    s, pos = set(), 0
    for i in ids:
        pos += len(tb(i))
        s.add(pos)
    return s


def check_docs(name, texts, tk, tk1, ref, final_tb, s1_tb, do_rt=True):
    res = collections.Counter()
    ex = []
    t_hf = t_ref = 0.0
    for k, t in enumerate(texts):
        t0 = time.perf_counter()
        ids = tk.encode(t, add_special_tokens=False).ids
        t_hf += time.perf_counter() - t0
        t0 = time.perf_counter()
        rid = ref.encode(t)
        t_ref += time.perf_counter() - t0
        res["docs"] += 1
        res["tokens"] += len(ids)
        if ids != rid:
            res["hf_vs_ref_mismatch_docs"] += 1
            if len(ex) < 5:
                j = next(i for i, (x, y) in enumerate(zip(ids + [None], rid + [None])) if x != y)
                ex.append({"doc": k, "at": j, "hf": ids[j:j + 4], "ref": rid[j:j + 4]})
        p1 = tk1.encode(t, add_special_tokens=False).ids
        if p1 != ref.encode_phase1(t):
            res["phase1_vs_hf_stage1_mismatch_docs"] += 1
        if not boundaries(ids, final_tb) <= boundaries(p1, s1_tb):
            res["final_boundary_not_stage1_boundary_docs"] += 1
        if do_rt:
            if tk.decode(ids, skip_special_tokens=False) != t:
                res["g1_fail_docs"] += 1
            if ref.decode(rid) != t:
                res["ref_roundtrip_fail_docs"] += 1
    out = dict(res)
    for key in ("hf_vs_ref_mismatch_docs", "phase1_vs_hf_stage1_mismatch_docs", "final_boundary_not_stage1_boundary_docs",
                "g1_fail_docs", "ref_roundtrip_fail_docs"):
        out.setdefault(key, 0)
    out["examples"] = ex
    out["hf_encode_s"] = round(t_hf, 2)
    out["ref_encode_s"] = round(t_ref, 2)
    C.log("%s: %s" % (name, {k: v for k, v in out.items() if k != "examples"}))
    return out


def superword_stats(texts, tk, final_tb):
    n = mw = mwb = allb = 0
    words_hist = collections.Counter()
    top = collections.Counter()
    for t in texts:
        for i in tk.encode(t, add_special_tokens=False).ids:
            b = final_tb(i)
            n += 1
            allb += len(b)
            w = C.n_words(b)
            core = b.strip(b" ")
            if b" " in core:
                mw += 1
                mwb += len(b)
                words_hist[w] += 1
                top[i] += 1
    return {"tokens": n, "multiword_tokens": mw, "multiword_token_share": mw / n if n else None,
            "bytes_in_multiword_tokens_share": mwb / allb if allb else None,
            "multiword_tokens_by_words": dict(sorted(words_hist.items())),
            "distinct_multiword_ids_used": len(top),
            "top_multiword": [{"id": i, "text": final_tb(i).decode("utf-8", "replace"), "dev_count": c}
                              for i, c in top.most_common(30)]}


def verify(out_dir, train_every):
    from tokenizers import Tokenizer
    path = os.path.join(out_dir, "tokenizer.json")
    pua = json.load(open(os.path.join(out_dir, "stage2_pua.json"), encoding="utf-8"))
    tk = Tokenizer.from_file(path)
    tk1 = Tokenizer.from_file(pua["stage1"])
    ref = R.SuperBPERef(out_dir)
    fv = tk.get_vocab(with_added_tokens=True)
    fid = {i: s for s, i in fv.items()}
    specials = set(C.SPECIAL_TOKENS)
    final_tb = lambda i: fid[i].encode("utf-8") if fid[i] in specials else C.bl_to_bytes(fid[i])
    v1 = tk1.get_vocab(with_added_tokens=True)
    id1 = {i: s for s, i in v1.items()}
    s1_tb = lambda i: id1[i].encode("utf-8") if id1[i] in specials else C.bl_to_bytes(id1[i])
    out = {"tokenizer": path, "tokenizer_sha256": C.sha256_file(path), "stage1": pua["stage1"],
           "reference_encoder": "ref_encoder.py sha256 " + C.sha256_file(os.path.join(C.HERE, "ref_encoder.py")),
           "checks": {}}
    for view in ("dev_strict", "dev_permissive"):
        docs, info = C.load_view(view)
        out["checks"][view] = check_docs(view, [d["text"] for d in docs], tk, tk1, ref, final_tb, s1_tb)
        out["checks"][view]["view"] = info
    if train_every:
        docs, info = C.load_view("train_D1", check_normalized=False)
        sub = [d["text"] for d in docs[::train_every]]
        out["checks"]["train_D1_every_%d" % train_every] = check_docs("train_D1/%d" % train_every, sub, tk, tk1, ref,
                                                                      final_tb, s1_tb)
    # special-token edge cases (G3-style) through both encoders
    edge = ["<|endoftext|>", "اب <|im_start|> پت", "<|bos|>اب پت۔<|endoftext|>", "", " ", "\n", "اب\n\nپت", "12۳۴ اب"]
    out["checks"]["edge_strings"] = {"all_equal": all(tk.encode(e, add_special_tokens=False).ids == ref.encode(e) for e in edge),
                                     "all_roundtrip": all(tk.decode(tk.encode(e, add_special_tokens=False).ids,
                                                                    skip_special_tokens=False) == e for e in edge)}
    try:
        from transformers import PreTrainedTokenizerFast
        ptf = PreTrainedTokenizerFast(tokenizer_file=path)
        docs, _ = C.load_view("dev_strict", check_normalized=False)
        bad = sum(1 for d in docs if ptf(d["text"], add_special_tokens=False)["input_ids"] !=
                  tk.encode(d["text"], add_special_tokens=False).ids)
        out["checks"]["transformers_PreTrainedTokenizerFast"] = {"version": __import__("transformers").__version__,
                                                                 "dev_strict_mismatch_docs": bad, "docs": len(docs)}
    except Exception as e:  # noqa: BLE001
        out["checks"]["transformers_PreTrainedTokenizerFast"] = {"error": repr(e)}
    docs, _ = C.load_view("dev_strict", check_normalized=False)
    out["superwords_dev_strict"] = superword_stats([d["text"] for d in docs], tk, final_tb)
    dv = [out["checks"][v] for v in ("dev_strict", "dev_permissive")]
    out["hf_native_exact_on_100pct_dev"] = all(c["hf_vs_ref_mismatch_docs"] == 0 for c in dv)
    out["g1_dev_pass"] = all(c["g1_fail_docs"] == 0 for c in dv)
    out["all_pass"] = (out["hf_native_exact_on_100pct_dev"] and out["g1_dev_pass"]
                       and all(c.get("phase1_vs_hf_stage1_mismatch_docs", 0) == 0 and
                               c.get("final_boundary_not_stage1_boundary_docs", 0) == 0
                               for c in out["checks"].values() if isinstance(c, dict) and "docs" in c and "view" in c or
                               (isinstance(c, dict) and "hf_vs_ref_mismatch_docs" in c))
                       and all(c.get("hf_vs_ref_mismatch_docs", 0) == 0 for c in out["checks"].values() if isinstance(c, dict))
                       and out["checks"]["edge_strings"]["all_equal"] and out["checks"]["edge_strings"]["all_roundtrip"])
    json.dump(out, open(os.path.join(out_dir, "verify.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("verify %s: hf_native_exact=%s g1=%s all_pass=%s" % (out_dir, out["hf_native_exact_on_100pct_dev"],
                                                               out["g1_dev_pass"], out["all_pass"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--train-every", type=int, default=0)
    ap.add_argument("--s2-diff", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    C.verify_frozen()
    if a.s2_diff:
        s2_differential()
    if a.out_dir:
        d = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(C.HERE, a.out_dir)
        verify(d, a.train_every)


if __name__ == "__main__":
    main()
