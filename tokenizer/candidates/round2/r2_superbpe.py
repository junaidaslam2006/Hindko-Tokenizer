# -*- coding: utf-8 -*-
"""Round 2: A6 SuperBPE (reimplemented from Liu et al. 2025) with P1r3 on D2, T = 32,768, t = 26,214 (t/T = 0.8,
the same t and T as the Stage 3 build A6-SBPE-P1-D1-32k-t080).

    python r2_superbpe.py build [--g4] [--py-check]   stage 1 (A1-P1r3-D2 to t) + stage 2 (to T on S2r3 chunks)
    python r2_superbpe.py verify                       HF-native vs reference encoder on 100% of dev (+ train/20)
    python r2_superbpe.py s2diff                       Oniguruma vs Python-regex differential of S2r3 (and P1r3)
    python r2_superbpe.py g2                           multi-word G2 (candidates/superbpe/gates_sbpe.py)
    python r2_superbpe.py eval                         harness --gates all (G4 twin), R1 on D1 and D2, NSL

Reused unchanged (candidates/superbpe): build_stage1.train_a1, build_stage2.build_chunks / hf_train_pua /
py_train / assemble, ref_encoder.SuperBPERef, verify.check_docs / superword_stats, gates_sbpe.main.
What round 2 changes, and how:
  * common.P1 := eval/harness P1r3 (the Stage 1 choice), and common.S2 := S2r3, derived from the reviewed S2 by
    the same digit substitution that turns P1 into P1r3 (asserted below): letter-word runs joined by single
    spaces form one stage-2 chunk; digits are right-to-left groups of 3, as in stage 1. build_chunks, assemble,
    SuperBPERef and gates_sbpe read common.P1/S2 at call time.
  * training text = train_D2 (common.load_view('train_D2'): manifest, split and normalize fixed-point checks).
    The drivers of build_stage2.main / verify.verify hard-code train_D1, so their few driver lines are
    re-stated here with the view as a parameter; gates_sbpe.main is run with common.load_view redirecting its
    'train_D1' to 'train_D2' and common.HERE pointing at round2/superbpe (where the chunk cache lives).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_common as RC  # noqa: E402

sys.path.insert(0, RC.SBPE)
import common as C  # noqa: E402
import build_stage1 as B1  # noqa: E402
import build_stage2 as B2  # noqa: E402
import ref_encoder as R  # noqa: E402
import verify as V  # noqa: E402

sys.path.insert(0, RC.EVAL)
import harness as H  # noqa: E402

OUT = os.path.join(RC.R2, "superbpe")
BUILD = "sbpe_32k_t080_p1r3_d2"
CID = RC.IDS["A6-32k"]
TRAIN = "train_D2"
T = 32768
t = int(round(0.8 * T))                                   # 26,214, as in the Stage 3 32k build
P1R3 = H.PRETOKENIZERS["P1r3"]
assert H.PRETOKENIZERS["P1"] == C.P1 and C.P1.replace(RC.P1_DIGIT, RC.P1R3_DIGIT) == P1R3
S2R3 = C.S2.replace(RC.P1_DIGIT, RC.P1R3_DIGIT)
assert S2R3 != C.S2 and S2R3.count(r"\p{N}{1,3}") == 1
S2_ORIG = C.S2
C.P1, C.S2 = P1R3, S2R3
STD_A1_32K = os.path.join(RC.STD, "tok", "A1-P1r3-D2-32k", "tokenizer.json")


def root(g4):
    return os.path.join(OUT, "g4") if g4 else OUT


# ------------------------------------------------------------------------------------------------- build
def prefix_check(s1_path, full_path):
    """Report only: the stage-1 file at t is a strict prefix (ids and merges) of A1-P1r3-D2 trained to 32,768."""
    a = json.load(open(s1_path, encoding="utf-8"))["model"]
    b = json.load(open(full_path, encoding="utf-8"))["model"]
    va, vb = a["vocab"], b["vocab"]
    ids_ok = all(vb.get(s) == i for s, i in va.items())
    ma = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in a["merges"]]
    mb = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in b["merges"]]
    return {"reference": full_path, "reference_sha256": RC.sha256_file(full_path), "ids_identical": ids_ok,
            "merges_prefix": mb[:len(ma)] == ma, "stage1_vocab": len(va), "reference_vocab": len(vb)}


def cmd_build(a):
    t_all = time.time()
    fz = C.verify_frozen()
    rt = root(a.g4)
    out, work, s1dir = os.path.join(rt, BUILD), os.path.join(rt, "work"), os.path.join(rt, "stage1")
    for d in (out, work, s1dir):
        os.makedirs(d, exist_ok=True)
    docs, info = C.load_view(TRAIN)
    texts = [d["text"] for d in docs]
    del docs
    # ---- stage 1: A1-P1r3-D2 to t (build_stage1.train_a1 with the P1r3 regex)
    s1 = os.path.join(s1dir, "a1_p1r3_d2_%d.json" % t)
    tk, dt = B1.train_a1(texts, t, pre_regex=P1R3)
    tk.save(s1)
    j = json.load(open(s1, encoding="utf-8"))
    s1_meta = {"what": "A1 byte-level BPE-P1r3 on train_D2 (SuperBPE stage 1)", "vocab_size_target": t,
               "vocab_size": tk.get_vocab_size(with_added_tokens=True), "merges": len(j["model"]["merges"]),
               "train_seconds": round(dt, 1), "rayon_threads": a.threads, "pre_tokenizer_regex": P1R3,
               "min_frequency": C.MIN_FREQUENCY, "train_view": info,
               "split_manifest_sha256": fz["split_manifest_sha256"], "normalize_version": fz["normalize_version"],
               "tokenizers_version": __import__("tokenizers").__version__, "sha256": RC.sha256_file(s1),
               "code_sha256": {"build_stage1.py": RC.sha256_file(os.path.join(RC.SBPE, "build_stage1.py")),
                               **RC.code_sha("r2_superbpe.py", "r2_common.py")}}
    s1_meta["prefix_of_A1-P1r3-D2-32k"] = prefix_check(s1, STD_A1_32K) if os.path.exists(STD_A1_32K) else None
    json.dump(s1_meta, open(s1 + ".info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("stage 1: vocab %d merges %d in %.1fs sha %s prefix %s" % (
        s1_meta["vocab_size"], s1_meta["merges"], dt, s1_meta["sha256"][:16], s1_meta["prefix_of_A1-P1r3-D2-32k"]))
    assert s1_meta["vocab_size"] == t and len(j["model"]["merges"]) == t - C.N_SPECIAL - C.N_BYTES
    # ---- chunks (build_stage2.build_chunks on the D2 texts; cache in build_stage2's format for gates_sbpe)
    t0 = time.time()
    sha = RC.sha256_file(s1)
    cnt, cstats = B2.build_chunks(s1, texts, a.threads)
    cstats["train_view"] = info
    cache = os.path.join(work, "chunks_%s.pkl" % sha[:16])
    with open(cache, "wb") as f:
        pickle.dump({"stage1_sha256": sha, "S2": C.S2, "counts": cnt, "stats": cstats}, f, protocol=4)
    t_chunks = time.time() - t0
    C.log("chunks: %s" % {k: v for k, v in cstats.items() if k not in ("chunk_len_tokens_hist_capped64", "train_view")})
    if cstats["stage1_S2_vs_P1_mismatch_docs"]:
        raise SystemExit("stage-1 merges applied inside S2r3 chunks differ from the P1r3 encoding in %d docs"
                         % cstats["stage1_S2_vs_P1_mismatch_docs"])
    del texts
    id2tok = {i: s for s, i in j["model"]["vocab"].items()}
    tok_bytes = lambda i: C.bl_to_bytes(id2tok[i])  # noqa: E731
    n_new = T - t
    # ---- stage 2 (build_stage2.main's route logic)
    merges_hf, hf_info = B2.hf_train_pua(cnt, t, n_new, a.threads)
    C.log("HF PUA trainer: %s" % hf_info)
    _, pre_info = B2.assemble(s1, merges_hf, T)
    route, py_info, py_capped_info, merges_final = "hf_rust_pua", None, None, merges_hf
    if a.py_check:
        merges_py, py_info = B2.py_train(cnt, t, n_new, tok_bytes, max_words=None)
        py_info["identical_to_hf_rust_merges"] = merges_py == merges_hf
        if not py_info["identical_to_hf_rust_merges"]:
            k = next((i for i, (x, y) in enumerate(zip(merges_py, merges_hf)) if x != y),
                     min(len(merges_py), len(merges_hf)))
            py_info["first_difference_at"] = k
        C.log("pure-Python uncapped: %s" % py_info)
    if pre_info["stage2_tokens_over_cap"]:
        merges_final, py_capped_info = B2.py_train(cnt, t, n_new, tok_bytes, max_words=C.MAX_WORDS)
        route = "pure_python_capped"
        C.log("cap binds (%d tokens > %d words): capped pure-Python merges %s"
              % (pre_info["stage2_tokens_over_cap"], C.MAX_WORDS, py_capped_info))
    jj, ainfo = B2.assemble(s1, merges_final, T)
    from tokenizers import Tokenizer
    tkf = Tokenizer.from_str(json.dumps(jj, ensure_ascii=False))
    path = os.path.join(out, "tokenizer.json")
    tkf.save(path)
    vs = tkf.get_vocab_size(with_added_tokens=True)
    pua_path = os.path.join(out, "stage2_pua.json")
    with open(pua_path, "w", encoding="utf-8") as f:
        json.dump({"stage1": os.path.abspath(s1), "stage1_sha256": sha, "t": t, "T": T, "route": route,
                   "merges": [[list(x), list(y)] for x, y in merges_final]}, f)
    bi = {"what": "SuperBPE (reimplemented from Liu et al. 2025) - stage 1 A1-P1r3-D2 to t, stage 2 to T on S2r3 "
                  "chunks (round 2)", "id": CID, "g4_twin": a.g4,
          "t": t, "T": T, "t_over_T": t / T, "vocab_size_loaded": vs, "route": route,
          "stage1": {"path": os.path.abspath(s1), "sha256": sha, "info": s1_meta},
          "S2_regex_oniguruma": C.S2, "S2_reviewed_P1_version": S2_ORIG, "P1_regex_oniguruma": C.P1,
          "max_words_cap": C.MAX_WORDS, "pua_base": hex(C.PUA_BASE), "min_frequency": C.MIN_FREQUENCY,
          "rayon_threads": a.threads, "chunks": cstats, "chunks_seconds_this_run": round(t_chunks, 1),
          "chunk_cache": cache, "hf_rust_pua_trainer": hf_info, "hf_rust_uncapped_assembly": pre_info,
          "pure_python_uncapped": py_info, "pure_python_capped": py_capped_info, "assembly": ainfo,
          "tokenizer_json": path, "tokenizer_sha256": RC.sha256_file(path),
          "stage2_pua_json_sha256": RC.sha256_file(pua_path),
          "code_sha256": {"build_stage1.py": RC.sha256_file(os.path.join(RC.SBPE, "build_stage1.py")),
                          "build_stage2.py": RC.sha256_file(os.path.join(RC.SBPE, "build_stage2.py")),
                          "common.py": RC.sha256_file(os.path.join(RC.SBPE, "common.py")),
                          **RC.code_sha("r2_superbpe.py", "r2_common.py")},
          "split_manifest_sha256": fz["split_manifest_sha256"], "normalize_version": fz["normalize_version"],
          "tokenizers_version": __import__("tokenizers").__version__, "total_seconds": round(time.time() - t_all, 1)}
    RC.dump(bi, os.path.join(out, "build_info.json"))
    C.log("wrote %s vocab %d sha256 %s route %s (%.0fs)" % (path, vs, bi["tokenizer_sha256"][:16], route,
                                                            bi["total_seconds"]))
    if vs != T:
        raise SystemExit("loaded vocab %d != T %d" % (vs, T))


# ------------------------------------------------------------------------------------------------ verify
def cmd_verify(a):
    """verify.verify with the train view as a parameter (train_D2 every 20th document)."""
    C.verify_frozen()
    from tokenizers import Tokenizer
    out_dir = os.path.join(OUT, BUILD)
    path = os.path.join(out_dir, "tokenizer.json")
    pua = json.load(open(os.path.join(out_dir, "stage2_pua.json"), encoding="utf-8"))
    tk = Tokenizer.from_file(path)
    tk1 = Tokenizer.from_file(pua["stage1"])
    ref = R.SuperBPERef(out_dir)
    assert ref.rx.pattern == C.onig_to_py(S2R3)
    fv = tk.get_vocab(with_added_tokens=True)
    fid = {i: s for s, i in fv.items()}
    specials = set(C.SPECIAL_TOKENS)
    final_tb = lambda i: fid[i].encode("utf-8") if fid[i] in specials else C.bl_to_bytes(fid[i])  # noqa: E731
    v1 = tk1.get_vocab(with_added_tokens=True)
    id1 = {i: s for s, i in v1.items()}
    s1_tb = lambda i: id1[i].encode("utf-8") if id1[i] in specials else C.bl_to_bytes(id1[i])  # noqa: E731
    out = {"tokenizer": path, "tokenizer_sha256": RC.sha256_file(path), "stage1": pua["stage1"],
           "reference_encoder": "ref_encoder.py sha256 " + RC.sha256_file(os.path.join(RC.SBPE, "ref_encoder.py")),
           "S2r3": S2R3, "checks": {}}
    for view in ("dev_strict", "dev_permissive"):
        docs, info = C.load_view(view)
        out["checks"][view] = V.check_docs(view, [d["text"] for d in docs], tk, tk1, ref, final_tb, s1_tb)
        out["checks"][view]["view"] = info
    docs, info = C.load_view(TRAIN, check_normalized=False)
    sub = [d["text"] for d in docs[::a.train_every]]
    key = "%s_every_%d" % (TRAIN, a.train_every)
    out["checks"][key] = V.check_docs(key, sub, tk, tk1, ref, final_tb, s1_tb)
    out["checks"][key]["view"] = {k: info[k] for k in ("view", "sha256", "docs")}
    edge = ["<|endoftext|>", "اب <|im_start|> پت", "<|bos|>اب پت۔<|endoftext|>", "", " ", "\n", "اب\n\nپت",
            "12۳۴ اب", "1234567 اب", "اب 12345، 9"]
    out["checks"]["edge_strings"] = {"all_equal": all(tk.encode(e, add_special_tokens=False).ids == ref.encode(e)
                                                      for e in edge),
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
    out["superwords_dev_strict"] = V.superword_stats([d["text"] for d in docs], tk, final_tb)
    doc_checks = [c for c in out["checks"].values() if isinstance(c, dict) and "hf_vs_ref_mismatch_docs" in c]
    dv = [out["checks"][v] for v in ("dev_strict", "dev_permissive")]
    out["hf_native_exact_on_100pct_dev"] = all(c["hf_vs_ref_mismatch_docs"] == 0 for c in dv)
    out["g1_dev_pass"] = all(c["g1_fail_docs"] == 0 and c["ref_roundtrip_fail_docs"] == 0 for c in dv)
    out["all_pass"] = (out["hf_native_exact_on_100pct_dev"] and out["g1_dev_pass"]
                       and all(c["hf_vs_ref_mismatch_docs"] == 0 and c["phase1_vs_hf_stage1_mismatch_docs"] == 0
                               and c["final_boundary_not_stage1_boundary_docs"] == 0 and c["g1_fail_docs"] == 0
                               for c in doc_checks)
                       and out["checks"]["edge_strings"]["all_equal"] and out["checks"]["edge_strings"]["all_roundtrip"])
    out["code_sha256"] = {"verify.py": RC.sha256_file(os.path.join(RC.SBPE, "verify.py")),
                          **RC.code_sha("r2_superbpe.py", "r2_common.py")}
    RC.dump(out, os.path.join(out_dir, "verify.json"))
    C.log("verify: hf_native_exact=%s g1=%s all_pass=%s" % (out["hf_native_exact_on_100pct_dev"], out["g1_dev_pass"],
                                                            out["all_pass"]))


# ------------------------------------------------------------------------------------------- differential
def cmd_s2diff(a):
    """PLAN 1.3 differential (Oniguruma vs Python regex) for S2r3, and for P1r3 on the new training view."""
    import regex
    from tokenizers import pre_tokenizers, Regex
    sys.path.insert(0, RC.EVAL)
    import perturb as P
    C.verify_frozen()
    res = {"regex_version": regex.__version__, "tokenizers_version": __import__("tokenizers").__version__}
    for name, onig in (("S2r3", S2R3), ("P1r3", P1R3)):
        hf = pre_tokenizers.Split(Regex(onig), behavior="isolated")
        rx = regex.compile(C.onig_to_py(onig))
        r = {"oniguruma": onig, "python": C.onig_to_py(onig), "results": {}}
        for view in ("dev_strict", "dev_permissive", "dev_strict+zwnj", TRAIN):
            docs, _ = C.load_view(view.split("+")[0], check_normalized=False)
            nd = nt = nmw = 0
            ex = []
            for d in docs:
                tx = d["text"] if "+" not in view else P.perturb_zwnj(d["text"]).text
                x = [s for s, _ in hf.pre_tokenize_str(tx)]
                y = R.py_pretokens(rx, tx)
                nt += len(x)
                nmw += sum(1 for s in x if " " in s.strip())
                if x != y:
                    nd += 1
                    if len(ex) < 3:
                        k = next(i for i, (u, v) in enumerate(zip(x + [None], y + [None])) if u != v)
                        ex.append({"uid": d["uid"], "at": k, "onig": x[k:k + 3], "python": y[k:k + 3]})
            r["results"][view] = {"docs": len(docs), "docs_differing": nd, "pretokens": nt, "multiword": nmw,
                                  "examples": ex}
            C.log("%s differential %s: %d/%d docs differ, %d pretokens" % (name, view, nd, len(docs), nt))
        r["identical"] = all(v["docs_differing"] == 0 for v in r["results"].values())
        res[name] = r
    res["code_sha256"] = RC.code_sha("r2_superbpe.py", "r2_common.py")
    RC.dump(res, os.path.join(RC.R2, "pretok_differential_round2.json"))


# ------------------------------------------------------------------------------------------------ G2
def cmd_g2(a):
    """gates_sbpe.main on the round-2 build: its train view (hard-coded 'train_D1') is redirected to train_D2, the
    tokenizer's own training text, whose S2r3 chunk cache lies in round2/superbpe/work."""
    import gates_sbpe as G
    orig = C.load_view

    def load_view(name, check_normalized=True):
        return orig(TRAIN if name == "train_D1" else name, check_normalized)
    C.load_view = load_view
    C.HERE = OUT
    sys.argv = ["gates_sbpe.py", os.path.join(OUT, BUILD)]
    G.main()
    p = os.path.join(OUT, BUILD, "g2_multiword.json")
    g = json.load(open(p, encoding="utf-8"))
    assert g["train_view"]["view"] == TRAIN
    g["round2"] = {"train_view_redirect": "gates_sbpe's 'train_D1' -> train_D2 (own training text)",
                   "chunk_cache_dir": os.path.join(OUT, "work"), "S2r3": C.S2,
                   "code_sha256": RC.code_sha("r2_superbpe.py", "r2_common.py")}
    RC.dump(g, p)


# ------------------------------------------------------------------------------------------------ eval
def cmd_eval(a):
    C.verify_frozen()
    import adapters as AD
    out_dir = os.path.join(OUT, BUILD)
    tok = os.path.join(out_dir, "tokenizer.json")
    g4 = os.path.join(OUT, "g4", BUILD, "tokenizer.json")
    vj = json.load(open(os.path.join(out_dir, "verify.json"), encoding="utf-8"))
    g2 = json.load(open(os.path.join(out_dir, "g2_multiword.json"), encoding="utf-8"))
    bi = json.load(open(os.path.join(out_dir, "build_info.json"), encoding="utf-8"))
    bi4 = json.load(open(os.path.join(OUT, "g4", BUILD, "build_info.json"), encoding="utf-8"))
    pa = json.load(open(os.path.join(out_dir, "stage2_pua.json"), encoding="utf-8"))
    pb = json.load(open(os.path.join(OUT, "g4", BUILD, "stage2_pua.json"), encoding="utf-8"))
    ad = AD.HFAdapter(path=tok, name=CID)
    res = os.path.join(OUT, "results", "dev_strict", CID)
    t0 = time.time()
    s = H.run(ad, data="dev_strict", gates="all", train="train_D1", g4_retrain=g4, out_dir=res,
              extra={"round2_id": CID, "t": t, "T": T, "route": bi["route"],
                     "verify_all_pass": vj["all_pass"], "g2_multiword_pass": g2["pass"]})
    nsl = RC.add_nsl(res)
    r1d2 = RC.r1_own_mix(ad, TRAIN)
    g = s["gates"]
    gates = {k: g[k].get("pass") for k in ("G1", "G2", "G3", "G4", "G5")}
    chk = {"id": CID, "tokenizer_json": tok, "tokenizer_sha256": RC.sha256_file(tok), "gates_harness": gates,
           "G2_multiword": {k: g2[k] for k in ("multiword_tokens", "pass", "unreachable", "isolated_failures",
                                               "chunk_aligned_failures", "chunk_raw_failures_literal_PLAN_clause",
                                               "chunk_raw_failures_that_pass_isolated", "train_freq_eq0",
                                               "train_freq_lt20", "train_freq_lt100", "words_hist")},
           "G2_final": bool(gates["G2"]) and bool(g2["pass"]),
           "G4_twin": {"tokenizer_identical": bi["tokenizer_sha256"] == bi4["tokenizer_sha256"],
                       "stage1_identical": bi["stage1"]["sha256"] == bi4["stage1"]["sha256"],
                       "stage2_pua_file_identical": bi["stage2_pua_json_sha256"] == bi4["stage2_pua_json_sha256"],
                       "stage2_pua_identical_except_stage1_path": (
                           {k: v for k, v in pa.items() if k != "stage1"} == {k: v for k, v in pb.items() if k != "stage1"}),
                       "stage2_pua_note": "stage2_pua.json stores the absolute stage-1 path (stage1/ vs g4/stage1/), so the "
                                          "files differ in that field only; merges, t, T, route and stage-1 sha256 compared"},
           "verify": {"all_pass": vj["all_pass"], "hf_native_exact_on_100pct_dev": vj["hf_native_exact_on_100pct_dev"],
                      "checks": {k: {kk: vv for kk, vv in v.items() if kk not in ("examples", "view")}
                                 for k, v in vj["checks"].items()},
                      "superwords_dev_strict": {k: vj["superwords_dev_strict"][k] for k in
                                                ("multiword_token_share", "bytes_in_multiword_tokens_share",
                                                 "distinct_multiword_ids_used")}},
           "build": {"route": bi["route"], "chunks": {k: v for k, v in bi["chunks"].items()
                                                      if k not in ("chunk_len_tokens_hist_capped64", "train_view")},
                     "stage2_token_words_hist": bi["assembly"]["stage2_token_words_hist"],
                     "stage2_tokens_duplicating_existing": bi["assembly"]["stage2_tokens_duplicating_existing"],
                     "duplicate_merge_pairs": bi["assembly"]["duplicate_merge_pairs"],
                     "uncapped_tokens_over_cap": bi["hf_rust_uncapped_assembly"]["stage2_tokens_over_cap"],
                     "py_uncapped_identical_to_rust": (bi["pure_python_uncapped"] or {}).get(
                         "identical_to_hf_rust_merges"),
                     "stage1_prefix_of_A1-P1r3-D2-32k": bi["stage1"]["info"].get("prefix_of_A1-P1r3-D2-32k"),
                     "total_seconds": bi["total_seconds"]},
           "R1_train_D1": s["properties"]["R1"], "R1_own_training_mix_train_D2": r1d2,
           "R2": {k: s["properties"]["R2"][k] for k in ("partial_utf8_tokens", "with_train_freq_0")},
           "nsl_vs_A1-P1-D1-16k": nsl.get("overall"),
           "code_sha256": RC.code_sha("r2_superbpe.py", "r2_common.py"), "seconds": round(time.time() - t0, 1)}
    RC.dump(chk, os.path.join(res, "r2_checks.json"))
    C.log(CID, "gates", gates, "G2 multi-word", g2["pass"], "bytes/token", s["metrics"]["overall"]["bytes_per_token"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "s2diff", "g2", "eval"])
    ap.add_argument("--g4", action="store_true")
    ap.add_argument("--py-check", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--train-every", type=int, default=20)
    a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    {"build": cmd_build, "verify": cmd_verify, "s2diff": cmd_s2diff, "g2": cmd_g2, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
