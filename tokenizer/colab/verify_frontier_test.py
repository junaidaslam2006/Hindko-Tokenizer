# -*- coding: utf-8 -*-
"""verify_frontier_test.py - independent checks of the REPORT-ONLY frontier bundle (colab/build_frontier_test).

    set PYTHONIOENCODING=utf-8 & set HF_HOME=F:\\Hindko\\_tokenizer\\hf_cache & python colab\\verify_frontier_test.py

 V1 bundle: manifest + PARTS kind 'final_test_report'; manifest sha256 == PARTS.json == the completed
    FINAL_TEST_LOG entry; every part <= 9.5 MB and re-verified (size + sha256); the concatenation == archive
    sha256; reassembled + extracted by run_all.reassemble into a scratch dir and every bundle file verified.
 V2 candidates, order and tokenizer sha256 == lm/WAVES_frontier_test.json; baseline hindko-1.0.0; the released
    tokenizer.json sha256 == F:/Hindko/tokenizer/RELEASE_MANIFEST.json; hk_lm.py == the one-shot test's
    (3f78ad79...), so every model is trained by the same code as the published test runs.
 V3 the evaluation set is the one-shot test's: dev_docs / dev_bytes / dev_subset / dev_parity and the shared
    train arrays == bundle f54c929ba1ab; dev_docs uids == data/test_strict.jsonl (491 docs, 1,451,026 bytes).
 V4 anchor R2-A4-SPnat-D2-32k: train + test arrays, n_vocab, ctx, EOT/BOS == bundle f54c929ba1ab (so its
    seeds 1-3 must reproduce the published test runs); release hindko-1.0.0: test arrays == the anchor's,
    train arrays differ in exactly the 29 tie documents of release_build/sp32k/EQUIVALENCE.md.
 V5 G1 on test_strict with the NATIVE library of each tokenizer (not the harness adapters): decode(bundle
    tokens) == text for all 491 documents, re-encoding reproduces the bundle tokens, no EOT/BOS id inside a
    document, every id < n_vocab; uint32 arrays for n_vocab > 65,536.
 V6 test token counts == eval/test_competitors.json (the one-shot intrinsic table) for every external tokenizer,
    so the LM sees exactly the tokenization those bytes/token numbers describe.
Writes colab/build_frontier_test/VERIFY_FRONTIER_TEST.json. Computes no model or LM metric.
"""
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "build_frontier_test")
OLD = os.path.join(HERE, "build_final_test", "staging")          # one-shot bundle f54c929ba1ab
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
RES = {}


def check(name, cond, detail=""):
    RES[name] = {"pass": bool(cond), "detail": detail}
    print("%s %-76s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def native_encoder(c):
    """(encode, decode) from the tokenizer's own library, independent of eval/adapters.py."""
    kind, path = c["kind"], c["tokenizer_path"]
    if kind == "hf" or (kind == "baseline" and c["baseline_spec"]["loader"] in ("hf_json", "tiktoken_ranks")):
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(path)
        tk.no_truncation()
        tk.no_padding()
        return (lambda t: tk.encode(t, add_special_tokens=False).ids,
                lambda ids: tk.decode(list(ids), skip_special_tokens=False), "tokenizers %s" % __import__("tokenizers").__version__)
    if kind == "sentencepiece":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=path)
        nl = sp.piece_to_id("\n")
        assert sp.id_to_piece(nl) == "\n"

        def enc(t):
            out = []
            for k, line in enumerate(t.split("\n")):
                if k:
                    out.append(nl)
                if line:
                    out.extend(sp.encode(line))
            return out

        def dec(ids):
            segs, cur = [], []
            for i in ids:
                if i == nl:
                    segs.append(cur)
                    cur = []
                else:
                    cur.append(i)
            segs.append(cur)
            return "\n".join(sp.decode(s) if s else "" for s in segs)
        return enc, dec, "sentencepiece %s + PLAN 1.1 newline convention" % spm.__version__
    if kind == "baseline" and c["baseline_spec"]["loader"] == "auto":
        # the class the model repo ships (manifest hf_class; AutoTokenizer cannot build this directory in
        # transformers 5.x), plus an INDEPENDENT re-implementation from vocab.json + merges.txt with the
        # `tokenizers` ByteLevel BPE (same GPT-2 split regex): both must reproduce the bundle tokens
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import transformers
        import build_bundle as bb
        from tokenizers import ByteLevelBPETokenizer
        cls = getattr(transformers, bb.baseline_spec(c["baseline_name"]).get("hf_class") or "AutoTokenizer")
        tok = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        bl = ByteLevelBPETokenizer(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt"),
                                   add_prefix_space=False)

        def enc(t):
            a = tok.encode(t, add_special_tokens=False)
            b = bl.encode(t).ids
            return a if a == b else ("MISMATCH", a, b)
        return (enc, lambda ids: tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "transformers %s %s == tokenizers ByteLevelBPE(vocab.json, merges.txt)" % (transformers.__version__,
                                                                                         type(tok).__name__))
    raise ValueError(kind)


def main():
    t0 = time.time()
    import numpy as np
    import hk_lm
    import run_all as ra
    up, st = os.path.join(OUT, "upload"), os.path.join(OUT, "staging")
    P = json.load(open(os.path.join(up, "PARTS.json"), encoding="utf-8"))
    m = json.load(open(os.path.join(st, "manifest.json"), encoding="utf-8"))
    msha = sha(os.path.join(st, "manifest.json"))
    L = json.load(open(os.path.join(TOK, "FINAL_TEST_LOG.json"), encoding="utf-8"))
    e = L["entries"][m["final_test"]["log_entry"]]

    # ---------------------------------------------------------------- V1
    parts_ok = all(os.path.getsize(os.path.join(up, p["name"])) == p["bytes"] <= 9_500_000
                   and sha(os.path.join(up, p["name"])) == p["sha256"] for p in P["parts"])
    h = hashlib.sha256()
    for p in P["parts"]:
        with open(os.path.join(up, p["name"]), "rb") as f:
            h.update(f.read())
    work = os.path.join(OUT, "_verify_work")
    shutil.rmtree(work, ignore_errors=True)
    P2, bdir = ra.reassemble(up, work)
    nfiles = hk_lm.Bundle(bdir).verify()
    same_extract = all(sha(os.path.join(bdir, i["path"])) == sha(os.path.join(st, i["path"])) for i in m["files"].values())
    shutil.rmtree(work, ignore_errors=True)
    check("V1 kind final_test_report; manifest == PARTS == completed log entry (report_only, purpose)",
          m["kind"] == P["kind"] == e["kind"] == "final_test_report" and P["bundle_manifest_sha256"] == msha
          and e["bundle_manifest_sha256"] == msha and e["status"] == "completed" and e["report_only"] is True
          and m["final_test"]["purpose"] == e["purpose"], "log entry #%d, bundle %s" % (m["final_test"]["log_entry"], msha[:12]))
    check("V1 parts <= 9.5 MB + sha256; archive sha256; reassembled, extracted, %d files verified" % nfiles,
          parts_ok and h.hexdigest() == P["archive_sha256"] and same_extract,
          "%d parts, %.1f MB" % (len(P["parts"]), sum(p["bytes"] for p in P["parts"]) / 1e6))

    # ---------------------------------------------------------------- V2
    W = json.load(open(os.path.join(TOK, "lm", "WAVES_frontier_test.json"), encoding="utf-8"))
    rm = json.load(open(r"F:\Hindko\tokenizer\RELEASE_MANIFEST.json", encoding="utf-8"))
    C = {c["id"]: c for c in m["candidates"]}
    ids = [c["id"] for c in m["candidates"]]
    ok = (ids == [w["id"] for w in W["waves"]] and m["baseline_id"] == "hindko-1.0.0"
          and all(C[w["id"]]["tokenizer_sha256"] == w["tokenizer_sha256"] == sha(w["tokenizer_path"])
                  for w in W["waves"] if os.path.isfile(w["tokenizer_path"]))
          and C["hindko-1.0.0"]["tokenizer_sha256"] == rm["tokenizer_files"]["tokenizer.json"]
          and C["roberta-urdu"]["tokenizer_sha256"] == W["waves"][ids.index("roberta-urdu")]["tokenizer_sha256"])
    check("V2 candidates/order/sha256 == WAVES_frontier_test.json; release sha == RELEASE_MANIFEST", ok, ", ".join(ids))
    om = json.load(open(os.path.join(OLD, "manifest.json"), encoding="utf-8"))
    check("V2 hk_lm.py identical to the one-shot test bundle's (%s)" % m["files"]["hk_lm.py"]["sha256"][:12],
          m["files"]["hk_lm.py"]["sha256"] == om["files"]["hk_lm.py"]["sha256"] == sha(os.path.join(HERE, "hk_lm.py")))

    # ---------------------------------------------------------------- V3
    shared = ("dev_bytes", "dev_docs", "dev_subset_idx", "dev_parity_idx", "train_bytes", "train_uids")
    same = {k: m["files"][k]["sha256"] == om["files"][k]["sha256"] for k in shared}
    docs = json.load(open(os.path.join(st, m["files"]["dev_docs"]["path"]), encoding="utf-8"))
    test = [json.loads(l) for l in open(os.path.join(TOK, "data", "test_strict.jsonl"), encoding="utf-8")]
    test.sort(key=lambda r: r["uid"])
    tm = json.load(open(os.path.join(TOK, "data", "test_manifest.json"), encoding="utf-8"))
    ok = (all(same.values()) and [d["uid"] for d in docs] == [r["uid"] for r in test] and len(test) == 491
          and sum(len(r["text"].encode("utf-8")) for r in test) == 1451026 == m["data"]["dev_bytes"]
          and sha(os.path.join(TOK, "data", "test_strict.jsonl")) == tm["views"]["test_strict"]["sha256"] == m["data"]["dev_sha256"])
    check("V3 eval set == one-shot bundle f54c929ba1ab (docs, bytes, subset, parity slice, shared train arrays)",
          ok, str({k: v for k, v in same.items() if not v}) or "all 6 shared arrays equal")

    # ---------------------------------------------------------------- V4
    oc = next(c for c in om["candidates"] if c["id"] == "R2-A4-SPnat-D2-32k")
    a = C["R2-A4-SPnat-D2-32k"]
    arr_same = all(m["files"]["R2-A4-SPnat-D2-32k/%s" % k]["sha256"] == om["files"]["R2-A4-SPnat-D2-32k/%s" % k]["sha256"]
                   for k in ("train_tokens", "train_offsets", "dev_tokens", "dev_offsets"))
    check("V4 anchor R2-A4-SPnat-D2-32k: all 4 arrays + n_vocab/ctx/EOT/BOS == one-shot bundle",
          arr_same and all(a[k] == oc[k] for k in ("n_vocab", "ctx_tokens", "eot_id", "bos_id", "train_tokens", "dev_tokens",
                                                     "tokenizer_sha256")), "ctx %d, %d test tokens" % (a["ctx_tokens"], a["dev_tokens"]))

    def arr(cid, k):
        return np.load(os.path.join(st, m["files"]["%s/%s" % (cid, k)]["path"]))
    r = C["hindko-1.0.0"]
    test_same = np.array_equal(arr("hindko-1.0.0", "dev_tokens"), arr("R2-A4-SPnat-D2-32k", "dev_tokens")) and \
        np.array_equal(arr("hindko-1.0.0", "dev_offsets"), arr("R2-A4-SPnat-D2-32k", "dev_offsets"))
    rt_, ro_ = arr("hindko-1.0.0", "train_tokens"), arr("hindko-1.0.0", "train_offsets")
    at_, ao_ = arr("R2-A4-SPnat-D2-32k", "train_tokens"), arr("R2-A4-SPnat-D2-32k", "train_offsets")
    ndiff = sum(1 for k in range(len(ro_) - 1) if not np.array_equal(rt_[ro_[k]:ro_[k + 1]], at_[ao_[k]:ao_[k + 1]]))
    check("V4 release hindko-1.0.0: test arrays == anchor's; train differs in 29 tie docs only",
          test_same and ndiff == 29 and all(r[k] == a[k] for k in ("n_vocab", "ctx_tokens", "eot_id", "bos_id")),
          "%d of %d train docs differ; train tokens %d vs %d" % (ndiff, len(ro_) - 1, r["train_tokens"], a["train_tokens"]))

    # ---------------------------------------------------------------- V5 + V6
    texts = [x["text"] for x in test]
    comp = json.load(open(os.path.join(TOK, "eval", "test_competitors.json"), encoding="utf-8"))
    ctoks = {t["name"]: t["test_strict"]["overall"]["tokens"] for t in comp["tokenizers"]}
    g1, v6 = {}, {}
    for cid in ids:
        c = C[cid]
        c = dict(c, kind=c["kind"])
        enc, dec, lib = native_encoder(c)
        toks, offs = arr(cid, "dev_tokens"), arr(cid, "dev_offsets")
        special = {int(c["eot_id"]), int(c["bos_id"])}
        n_rt = n_re = 0
        for k, t in enumerate(texts):
            ids_k = toks[offs[k]:offs[k + 1]].astype(np.int64).tolist()
            n_rt += dec(ids_k) == t
            n_re += enc(t) == ids_k
        tt = arr(cid, "train_tokens")
        dtype_ok = str(toks.dtype) == c["dtype"] == ("uint16" if c["n_vocab"] <= 65536 else "uint32") == str(tt.dtype)
        rng_ok = int(toks.max()) < c["n_vocab"] and int(tt.max()) < c["n_vocab"]
        no_spec = not np.isin(toks, list(special)).any() and not np.isin(tt, list(special)).any()
        g1[cid] = {"roundtrip": "%d/491" % n_rt, "reencode_equal": "%d/491" % n_re, "library": lib,
                   "dtype": c["dtype"], "n_vocab": c["n_vocab"], "ids_in_range": rng_ok, "no_special_inside": no_spec,
                   "test_tokens": c["dev_tokens"], "test_bytes_per_token": c["dev_bytes_per_token"]}
        check("V5 G1 native %-20s decode 491/491, re-encode 491/491, ids ok" % cid,
              n_rt == 491 and n_re == 491 and dtype_ok and rng_ok and no_spec,
              "%d/%d, %s, V=%d, %s" % (n_rt, n_re, c["dtype"], c["n_vocab"], lib))
        if c["kind"] == "baseline":
            v6[cid] = (c["dev_tokens"], ctoks.get(c["baseline_name"]))
    check("V6 test token counts == eval/test_competitors.json for every external tokenizer",
          all(x == y for x, y in v6.values()) and len(v6) == 8,
          "; ".join("%s %d" % (k, v[0]) for k, v in v6.items()))

    n_fail = sum(1 for v in RES.values() if not v["pass"])
    out = {"passed": len(RES) - n_fail, "failed": n_fail, "seconds": round(time.time() - t0, 1),
           "bundle_manifest_sha256": msha, "tests": RES, "g1_native": g1,
           "verify_sha256": sha(os.path.abspath(__file__)),
           "note": "reads data/test_strict.jsonl as text only (G1, uids, bytes); computes no model or LM metric"}
    with open(os.path.join(OUT, "VERIFY_FRONTIER_TEST.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\n%d passed, %d failed in %.0fs" % (out["passed"], n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
