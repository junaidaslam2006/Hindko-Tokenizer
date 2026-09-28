# -*- coding: utf-8 -*-
"""verify_final_test.py - independent checks of the one-shot FINAL TEST bundle (colab/build_final_test).

    set PYTHONIOENCODING=utf-8 & python colab\\verify_final_test.py

 V1 bundle: manifest kind 'final_test'; manifest sha256 == PARTS.json == the completed FINAL_TEST_LOG entry;
    every part re-verified (size + sha256) and the concatenation == archive sha256; every staged bundle file
    verified (hk_lm.Bundle.verify).
 V2 candidates == analysis/decision.json test_candidates_fixed (ids, sha256 of the id list), roles, baseline.
 V3 training side identical to the dev bundles: per candidate tokenizer sha256, n_vocab, EOT/BOS ids, dtype,
    train token count, train bytes/token, ctx_tokens and the sha256 of train_tokens.npy / train_offsets.npy
    equal the round-1 / round-2 bundle manifests (1d24425d2d64, 77e1368773fc); the shared train_bytes /
    train_uids arrays and hk_lm.py are identical too.
 V4 evaluation side = the strict test split: dev_docs uids == data/test_strict.jsonl uids (sorted),
    bytes == UTF-8 bytes, the file sha256 == data/test_manifest.json, 27 bootstrap clusters (15/6/6).
 V5 G1 on test_strict for each candidate: decode(bundle tokens) == text for every document, with the harness
    encoder (the exact encoder of the sweep) AND with the native library encoder; re-encoding the text with
    both reproduces the bundle tokens; no EOT/BOS id inside a document; every id < n_vocab.
Writes colab/build_final_test/VERIFY_FINAL_TEST.json. Computes no model or LM metric.
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "build_final_test")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
RES = {}


def check(name, cond, detail=""):
    RES[name] = {"pass": bool(cond), "detail": detail}
    print("%s %-72s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    import numpy as np
    import build_bundle as bb
    stage, upload = os.path.join(OUT, "staging"), os.path.join(OUT, "upload")
    mp = os.path.join(stage, "manifest.json")
    m = json.load(open(mp, encoding="utf-8"))
    msha = sha(mp)
    P = json.load(open(os.path.join(upload, "PARTS.json"), encoding="utf-8"))
    L = json.load(open(os.path.join(TOK, "FINAL_TEST_LOG.json"), encoding="utf-8"))
    done = [e for e in L["entries"] if e.get("status") == "completed"]
    # ------------------------------------------------------------ V1
    h = hashlib.sha256()
    bad_parts = []
    for p in P["parts"]:
        fp = os.path.join(upload, p["name"])
        blk = open(fp, "rb").read()
        h.update(blk)
        if len(blk) != p["bytes"] or hashlib.sha256(blk).hexdigest() != p["sha256"]:
            bad_parts.append(p["name"])
    sys.path.insert(0, stage)
    import hk_lm
    b = hk_lm.Bundle(stage)
    n_files = b.verify()
    check("V1 manifest kind final_test; sha256 == PARTS.json == completed log entry",
          m.get("kind") == "final_test" and P.get("kind") == "final_test" and P["bundle_manifest_sha256"] == msha
          and len(L["entries"]) == 1 and len(done) == 1 and done[0]["bundle_manifest_sha256"] == msha,
          "bundle %s, %d log entr%s" % (msha[:12], len(L["entries"]), "y" if len(L["entries"]) == 1 else "ies"))
    check("V1 parts + archive sha256; every bundle file verified",
          not bad_parts and h.hexdigest() == P["archive_sha256"] and n_files == len(m["files"]),
          "%d parts, %d files" % (len(P["parts"]), n_files))
    # ------------------------------------------------------------ V2
    dec = json.load(open(os.path.join(TOK, "analysis", "decision.json"), encoding="utf-8"))
    fx = dec["test_candidates_fixed"]
    ids_log = done[0]["candidate_ids_input_order"] if done else None
    ids = [c["id"] for c in m["candidates"]]
    roles = {c["id"]: c["waves_fields"].get("final_test_role") for c in m["candidates"]}
    check("V2 candidates == decision.json test_candidates_fixed (ids, order, sha256)",
          ids_log == fx["ids"] and sorted(ids) == sorted(fx["ids"])
          and hashlib.sha256("\n".join(ids_log or []).encode("utf-8")).hexdigest() == fx["sha256_of_ids"]
          and m["final_test"]["candidate_ids_sha256"] == fx["sha256_of_ids"],
          "%s" % fx["sha256_of_ids"][:12])
    check("V2 roles: chosen R2-A10-MinGram-P1r3-D2-48k, baseline A1-P1r3-D2-16k, 2 report-only",
          roles == {"R2-A10-MinGram-P1r3-D2-48k": "chosen", "A1-P1r3-D2-16k": "baseline",
                    "R2-A4-SPnat-D2-32k": "report-only", "R2-A10-MinGram-P1r3-D2-32k": "report-only"}
          and m["baseline_id"] == "A1-P1r3-D2-16k", "%s" % roles)
    # ------------------------------------------------------------ V3
    old = {}
    for d, want in (("build", "1d24425d2d64"), ("build_r2", "77e1368773fc")):
        p = os.path.join(HERE, d, "staging", "manifest.json")
        if sha(p)[:12] != want:
            raise SystemExit("%s is not bundle %s" % (p, want))
        old[want] = json.load(open(p, encoding="utf-8"))
    rows = []
    keys = ("tokenizer_sha256", "n_vocab", "eot_id", "bos_id", "dtype", "train_tokens", "train_bytes_per_token",
            "ctx_tokens", "kind", "sp_newline_wrapper")
    all_same = True
    for c in m["candidates"]:
        refs = []
        for bid, om in old.items():
            oc = next((x for x in om["candidates"] if x["id"] == c["id"]), None)
            if oc is None:
                continue
            same_meta = all(oc.get(k) == c.get(k) for k in keys)
            same_arr = {k: om["files"]["%s/%s" % (c["id"], k)]["sha256"] == m["files"]["%s/%s" % (c["id"], k)]["sha256"]
                        for k in ("train_tokens", "train_offsets")}
            refs.append({"bundle": bid, "metadata_equal": same_meta, "train_tokens_sha256_equal": same_arr["train_tokens"],
                         "train_offsets_sha256_equal": same_arr["train_offsets"]})
            all_same &= same_meta and all(same_arr.values())
        all_same &= bool(refs)
        rows.append({"id": c["id"], "train_tokens_sha256": m["files"]["%s/train_tokens" % c["id"]]["sha256"],
                     "compared_with": refs})
    shared = all(om["files"][k]["sha256"] == m["files"][k]["sha256"] for om in old.values() for k in ("train_bytes", "train_uids"))
    code = all(om["builder"]["hk_lm_sha256"] == m["builder"]["hk_lm_sha256"] for om in old.values())
    data_same = all(om["data"]["train_sha256"] == m["data"]["train_sha256"] for om in old.values())
    check("V3 train encodings == round-1/2 bundles (sha256 of train_tokens / train_offsets, metadata)",
          all_same, "; ".join("%s: %s" % (r["id"], ",".join(x["bundle"] for x in r["compared_with"])) for r in rows))
    check("V3 shared train arrays, train_D1 sha256 and hk_lm.py sha256 == dev bundles", shared and code and data_same,
          "hk_lm %s" % m["builder"]["hk_lm_sha256"][:12])
    # ------------------------------------------------------------ V4
    tp = os.path.join(TOK, "data", "test_strict.jsonl")
    tm = json.load(open(os.path.join(TOK, "data", "test_manifest.json"), encoding="utf-8"))
    man, _ = bb.load_split_manifest()
    test = bb.read_jsonl(tp, "test", man, require_strict=True, final_test=True)
    docs = b.dev_docs()
    texts = [r["text"] for r in test]
    dvb = b.dev_bytes()
    cl = {}
    for d in docs:
        cl.setdefault(d["source"], set()).add(d["cluster"])
    check("V4 'dev' set of the bundle == data/test_strict.jsonl (uids, bytes, sha256, clusters 15/6/6)",
          [d["uid"] for d in docs] == [r["uid"] for r in test] and m["data"]["dev_sha256"] == sha(tp)
          == tm["views"]["test_strict"]["sha256"] and all(int(x) == len(t.encode("utf-8")) for x, t in zip(dvb, texts))
          and {k: len(v) for k, v in cl.items()} == {"newspaper": 15, "book": 6, "web": 6},
          "%d docs, %d bytes, clusters %s" % (len(docs), int(dvb.sum()), {k: len(v) for k, v in sorted(cl.items())}))
    # ------------------------------------------------------------ V5
    g1 = {}
    for c in m["candidates"]:
        toks, offs = b.tokens(c["id"], "dev")
        seqs = [toks[offs[i]:offs[i + 1]].astype(np.int64).tolist() for i in range(len(texts))]
        res = {}
        for src in ("harness", "builtin"):
            enc = bb.load_encoder(c["tokenizer_path"], c["kind"], src, cid=c["id"])
            rt_fail = sum(1 for s, t in zip(seqs, texts) if enc.decode(s) != t)
            same = sum(1 for s, t in zip(seqs, texts) if enc.encode(t) == s)
            res[src] = {"g1_fail_docs": rt_fail, "reencode_identical_docs": same, "encoder": enc.source}
        spec = int(np.isin(toks, [c["eot_id"], c["bos_id"]]).sum())
        rng = bool(len(toks) == 0 or (int(toks.max()) < c["n_vocab"]))
        g1[c["id"]] = dict(res, docs=len(texts), special_ids_inside_docs=spec, ids_in_range=rng,
                           test_tokens=int(len(toks)), test_bytes_per_token=round(float(dvb.sum()) / len(toks), 6),
                           dev_bytes_per_token_from_dev_bundle=next(
                               (x["dev_bytes_per_token"] for om in old.values() for x in om["candidates"]
                                if x["id"] == c["id"]), None),
                           builder_roundtrip_failures=c["roundtrip_dev_failures"])
        ok = all(r["g1_fail_docs"] == 0 and r["reencode_identical_docs"] == len(texts) for r in res.values()) \
            and spec == 0 and rng and c["roundtrip_dev_failures"] == 0
        check("V5 G1 on test_strict: %s" % c["id"], ok,
              "harness %d/%d, native %d/%d round trips; re-encode identical %d/%d" % (
                  len(texts) - res["harness"]["g1_fail_docs"], len(texts), len(texts) - res["builtin"]["g1_fail_docs"],
                  len(texts), res["harness"]["reencode_identical_docs"], len(texts)))
    n_fail = sum(1 for v in RES.values() if not v["pass"])
    out = {"what": "Independent verification of the one-shot final-test bundle (colab/verify_final_test.py)",
           "bundle_manifest_sha256": msha, "passed": len(RES) - n_fail, "failed": n_fail, "tests": RES,
           "train_identity": rows, "g1_test_strict": g1,
           "test_file": {"path": tp, "sha256": sha(tp), "docs": len(texts), "bytes": int(dvb.sum())},
           "verify_py_sha256": sha(os.path.abspath(__file__)), "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "VERIFY_FINAL_TEST.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n%d passed, %d failed in %.0fs" % (out["passed"], n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
