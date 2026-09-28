# -*- coding: utf-8 -*-
"""frontier_dryrun.py - exercise the new code paths of build_bundle.py (encoder_kind 'baseline', uint32 token
arrays, --final-test --report-only) and run_all.py (kind 'final_test_report') WITHOUT the real test text,
before the one real frontier build (colab/FRONTIER_LM.md).

    set PYTHONIOENCODING=utf-8 & set HF_HOME=F:\\Hindko\\_tokenizer\\hf_cache & python colab\\frontier_dryrun.py

 F1 build_bundle.main refuses --report-only without --final-test, with an empty purpose, and (inherited) with
    --max-dev-docs; an unknown baseline name is refused.
 F2 a complete --final-test --report-only build on a FAKE test file (40 strict TEST uids from the split manifest,
    metadata only, carrying the texts of the first 40 dev_strict documents) with 3 candidates: the released
    tokenizer.json (baseline), gpt-4o (200k, uint32) and roberta-urdu (transformers 'auto' loader, a directory).
    FINAL_TEST_LOG is redirected to colab/_frontier_dryrun/. Checks: manifest/PARTS kind 'final_test_report',
    report-only fields, log entry (kind, purpose, test_reuse, baseline names, started -> completed), dtypes,
    native EOT/BOS ids, round trip 40/40, tokenizer sha256 = the load_path's sha256, the real log untouched.
 F3 the released tokenizer.json's full-train encoding vs the anchor candidate's arrays of the one-shot bundle
    f54c929ba1ab: identical except for the documents with exact Viterbi ties (EQUIVALENCE.md: 29 train_D1 docs).
 F4 run_all.py on that bundle: default stages, lr, large, --smoke, --final-test-large and confirm without --lr
    are refused before anything runs; --estimate works; --stages parity reproduces the bundle's PARITY.json.
 F5 stage_confirm with a stubbed run(): the report bundle runs seeds 1-3 only even when the power check would add
    seeds (and says so in power_check.json), seeds 1-5 with --force-extra-seeds, never a re-sweep; the real
    one-shot final-test bundle (kind 'final_test') still runs seeds 1-5 with --force-extra-seeds (unchanged).
 F6 CPU smoke run of one EXTERNAL tokenizer (gpt-4o, n_vocab 200,019, uint32 arrays) through hk_lm.run_one:
    'confirm' model, 3 steps, finite losses, bpb == sum(bits)/sum(bytes); the windowed evaluation of one short
    document equals a token-by-token brute-force recomputation.
Writes only under colab/_frontier_dryrun/ (fake data and bundle are deleted at the end; the manifest, the
redirected log and the result are kept). No real test text is read.
"""
import hashlib
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
DRY = os.path.join(HERE, "_frontier_dryrun")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
RES = {}


def check(name, cond, detail=""):
    RES[name] = {"pass": bool(cond), "detail": detail}
    print("%s %-74s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


def refused(fn, must=()):
    try:
        fn()
        return False, "not refused"
    except SystemExit as e:
        return all(m in str(e) for m in must), str(e)[:160]


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
    import run_all as ra
    shutil.rmtree(DRY, ignore_errors=True)
    os.makedirs(os.path.join(DRY, "data"))
    real_log = bb.FINAL_TEST_LOG
    real_log_sha = sha(real_log) if os.path.isfile(real_log) else None
    real_parity = sha(os.path.join(HERE, "PARITY.json")) if os.path.isfile(os.path.join(HERE, "PARITY.json")) else None
    man, man_sha = bb.load_split_manifest()
    test_uids = sorted(u for u, (sp, tier) in man.items() if sp == "test" and tier == "strict")[:40]
    dev = [json.loads(l) for l in open(os.path.join(TOK, "data", "dev_strict.jsonl"), encoding="utf-8")][:40]

    def write(p, recs):
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fake = os.path.join(DRY, "data", "test_strict.jsonl")
    write(fake, [dict(d, uid=u) for u, d in zip(test_uids, dev)])
    with open(os.path.join(DRY, "data", "test_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"split_manifest_sha256": man_sha, "views": {"test_strict": {"sha256": sha(fake)}}}, f)
    W = json.load(open(os.path.join(TOK, "lm", "WAVES_frontier_test.json"), encoding="utf-8"))
    keep = ("hindko-1.0.0", "gpt-4o", "roberta-urdu")
    W["waves"] = [w for w in W["waves"] if w["id"] in keep]
    wp = os.path.join(DRY, "WAVES_dryrun.json")
    json.dump(W, open(wp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    purpose = "DRY RUN of the frontier comparison (fake test file)"
    common = ["--final-test", "--waves", wp, "--baseline", "hindko-1.0.0", "--dev-jsonl", fake,
              "--encoder-source", "harness", "--workers", "2"]

    # ---------------------------------------------------------------- F1
    bb.FINAL_TEST_LOG = os.path.join(DRY, "FINAL_TEST_LOG.json")
    try:
        ok1, _ = refused(lambda: bb.main(["--report-only", "x", "--waves", wp, "--no-parity",
                                          "--out", os.path.join(DRY, "x1")]), ["needs --final-test"])
        ok2, _ = refused(lambda: bb.main(common + ["--report-only", "  ", "--out", os.path.join(DRY, "x2")]),
                         ["non-empty PURPOSE"])
        ok3, _ = refused(lambda: bb.main(common + ["--report-only", purpose, "--max-dev-docs", "5",
                                                   "--out", os.path.join(DRY, "x3")]), ["--final-test encodes the whole"])
        ok4, m4 = refused(lambda: bb.main(["--candidates", "no-such-tokenizer", "baseline", "--max-train-docs", "5",
                                           "--max-dev-docs", "5", "--no-parity", "--out", os.path.join(DRY, "x4")]),
                          ["unknown baseline"])
        check("F1 --report-only needs --final-test + purpose; slicing refused; unknown baseline refused",
              ok1 and ok2 and ok3 and ok4, m4[:80])

        # ------------------------------------------------------------ F2
        out = os.path.join(DRY, "build")
        res = bb.main(common + ["--report-only", purpose, "--out", out])
    finally:
        bb.FINAL_TEST_LOG = real_log
    m = json.load(open(os.path.join(res["stage"], "manifest.json"), encoding="utf-8"))
    P = json.load(open(os.path.join(res["upload"], "PARTS.json"), encoding="utf-8"))
    L = json.load(open(os.path.join(DRY, "FINAL_TEST_LOG.json"), encoding="utf-8"))
    e = L["entries"][-1]
    C = {c["id"]: c for c in m["candidates"]}
    check("F2 manifest + PARTS kind 'final_test_report', report_only + purpose, 'dev' = fake test uids",
          m["kind"] == "final_test_report" and list(m)[:2] == ["format", "kind"] and P.get("kind") == "final_test_report"
          and m["final_test"]["report_only"] is True and m["final_test"]["purpose"] == purpose
          and "REPORT-ONLY" in m["test_split"] and m["data"]["n_dev_docs"] == 40
          and json.load(open(os.path.join(res["stage"], "shared", "dev_docs.json"), encoding="utf-8"))[0]["uid"] == test_uids[0],
          "%s, %d dev docs" % (m["kind"], m["data"]["n_dev_docs"]))
    check("F2 log entry: kind, report_only, purpose, test_reuse, baseline names, started -> completed",
          len(L["entries"]) == 1 and e["kind"] == "final_test_report" and e["report_only"] is True
          and e["purpose"] == purpose and "used AGAIN" in e["test_reuse"] and e["status"] == "completed"
          and e["bundle_manifest_sha256"] == res["manifest_sha256"] and e["test_file"]["sha256"] == sha(fake)
          and [c.get("baseline_name") for c in e["candidates"]] == [None, "gpt-4o", "roberta-urdu"]
          and set(e["g1_roundtrip_test_failures"].values()) == {0}, str(e["g1_roundtrip_test_failures"]))
    exp_sha = {w["id"]: w["tokenizer_sha256"] for w in W["waves"]}
    check("F2 dtypes uint16 (32k, 52k) / uint32 (200k); n_vocab; native EOT/BOS; tokenizer sha256 = load_path",
          C["hindko-1.0.0"]["dtype"] == "uint16" and C["roberta-urdu"]["dtype"] == "uint16"
          and C["gpt-4o"]["dtype"] == "uint32" and C["gpt-4o"]["n_vocab"] == 200020
          and C["gpt-4o"]["eot_token"] == "<|endoftext|>" and C["gpt-4o"]["specials_appended"] == [{"role": "bos", "id": 200019}]
          and (C["roberta-urdu"]["eot_token"], C["roberta-urdu"]["bos_token"]) == ("</s>", "<s>")
          and (C["hindko-1.0.0"]["eot_id"], C["hindko-1.0.0"]["bos_id"]) == (0, 1)
          and all(C[i]["tokenizer_sha256"] == exp_sha[i] for i in C)
          and C["gpt-4o"]["baseline_spec"]["repo"] == "Xenova/gpt-4o"
          and C["gpt-4o"]["encoder_source"].endswith("HFAdapter") and C["roberta-urdu"]["encoder_source"].endswith("TransformersAdapter"),
          "; ".join("%s %s V=%d eot %d bos %d" % (i, C[i]["dtype"], C[i]["n_vocab"], C[i]["eot_id"], C[i]["bos_id"]) for i in C))
    check("F2 round trip on every (fake) test document, every candidate",
          all(C[i]["roundtrip_dev_failures"] == 0 for i in C), str({i: C[i]["roundtrip_dev_failures"] for i in C}))
    check("F2 real FINAL_TEST_LOG.json and colab/PARITY.json untouched",
          (sha(real_log) if os.path.isfile(real_log) else None) == real_log_sha
          and (real_parity is None or sha(os.path.join(HERE, "PARITY.json")) == real_parity))

    # ---------------------------------------------------------------- F3
    old = os.path.join(HERE, "build_final_test", "staging")
    om = json.load(open(os.path.join(old, "manifest.json"), encoding="utf-8"))
    oc = next(c for c in om["candidates"] if c["id"] == "R2-A4-SPnat-D2-32k")
    ot = np.load(os.path.join(old, om["files"]["R2-A4-SPnat-D2-32k/train_tokens"]["path"]))
    oo = np.load(os.path.join(old, om["files"]["R2-A4-SPnat-D2-32k/train_offsets"]["path"]))
    nt = np.load(os.path.join(res["stage"], m["files"]["hindko-1.0.0/train_tokens"]["path"]))
    no = np.load(os.path.join(res["stage"], m["files"]["hindko-1.0.0/train_offsets"]["path"]))
    ndiff = sum(1 for k in range(len(oo) - 1) if not np.array_equal(ot[oo[k]:oo[k + 1]], nt[no[k]:no[k + 1]]))
    same_shared = all(om["files"][k]["sha256"] == m["files"][k]["sha256"] for k in ("train_bytes", "train_uids"))
    check("F3 release tokenizer.json vs anchor arrays (bundle f54c929ba1ab): 29 train docs differ (ties)",
          ndiff == 29 and same_shared and len(oo) == len(no) and C["hindko-1.0.0"]["n_vocab"] == oc["n_vocab"],
          "%d of %d docs differ; tokens %d vs %d; ctx %d vs %d" % (ndiff, len(oo) - 1, len(nt), len(ot),
                                                                   C["hindko-1.0.0"]["ctx_tokens"], oc["ctx_tokens"]))

    # ---------------------------------------------------------------- F4
    work = os.path.join(DRY, "work")
    P2, bdir = ra.reassemble(res["upload"], work)
    rout = os.path.join(DRY, "out")
    base = ["--bundle-dir", bdir, "--out", rout, "--device", "cpu", "--threads", "2", "--no-zip"]
    cases = [([], ["only --stages parity,confirm"]),
             (["--stages", "lr"], ["only --stages parity,confirm"]),
             (["--stages", "parity,confirm,large", "--lr", "1e-3"], ["only --stages parity,confirm"]),
             (["--stages", "parity,confirm", "--smoke", "--lr", "1e-3"], ["--smoke"]),
             (["--stages", "parity,confirm", "--lr", "1e-3", "--final-test-large", "--large-lr", "5e-4"],
              ["--final-test-large"]),
             (["--stages", "parity,confirm"], ["needs --lr 1e-3"])]
    oks = []
    for extra, must in cases:
        ok, msg = refused(lambda: ra.main(base + extra), must)
        oks.append(ok)
    rdirs = [os.path.join(rout, res["manifest_sha256"][:12] + s, "results") for s in ("", "_smoke")]
    n_res = sum(len(os.listdir(d)) for d in rdirs if os.path.isdir(d))
    rc_est = ra.main(base + ["--estimate"])
    check("F4 run_all refuses every non-report invocation before running anything; --estimate works",
          all(oks) and n_res == 0 and rc_est == 0, "%s, %d result files" % (oks, n_res))
    rc = ra.main(base + ["--stages", "parity"])
    pr = json.load(open(os.path.join(rout, res["manifest_sha256"][:12], "parity_report.json"), encoding="utf-8"))
    check("F4 --stages parity runs; CPU re-run reproduces the bundle's PARITY.json bitwise",
          rc == 0 and pr["runs"]["fp32"]["bitwise_identical_to_cpu"], "bpb %.6f" % pr["runs"]["fp32"]["bpb"])

    # ---------------------------------------------------------------- F5
    def stub(bundle_dir, argv, pooled):
        R = ra.Runner(ra.build_parser().parse_args(["--bundle-dir", bundle_dir, "--out", os.path.join(DRY, "stub"),
                                                    "--device", "cpu"] + argv), bundle_dir, None)
        calls = []
        R.run = lambda label, recipe, cand, seed, lr: calls.append((label, cand, seed, float(lr))) or {}
        R._pooled_rel_sd = lambda label, lr: (pooled, {})
        c = R.cands
        R._stage_means = lambda label, lr, recipe=None: {c[0]: (1.000, 0.0, 3), c[1]: (1.005, 0.0, 3)}
        R.stage_confirm()
        return R, calls
    R, calls = stub(bdir, ["--lr", "1e-3"], 0.004)          # high s.d.: the power check WOULD add seeds
    conf = sorted((cc, s) for lab, cc, s, lr in calls if lab == "confirm")
    pc = json.load(open(os.path.join(R.root, "power_check.json"), encoding="utf-8"))
    ok_a = (R.report_only and R.final_test and conf == sorted((cc, s) for cc in R.cands for s in (1, 2, 3))
            and not any(lab == "confirm_resweep" for lab, *_ in calls) and pc["extra_seeds_run"] == []
            and pc["power_check_would_add_seeds"] is True and "report_only_bundle" in pc)
    R, calls = stub(bdir, ["--lr", "1e-3", "--force-extra-seeds"], 0.0005)
    conf5 = sorted((cc, s) for lab, cc, s, lr in calls if lab == "confirm")
    ok_b = conf5 == sorted((cc, s) for cc in R.cands for s in (1, 2, 3, 4, 5)) and \
        not any(lab == "confirm_resweep" for lab, *_ in calls)
    check("F5 report bundle: seeds 1-3 even if the power check asks (recorded); 1-5 if forced; no re-sweep",
          ok_a and ok_b, "%d / %d confirm runs" % (len(conf), len(conf5)))
    ft = os.path.join(HERE, "build_final_test", "staging")        # the real one-shot bundle dir (read only)
    R, calls = stub(ft, ["--lr", "1e-3", "--force-extra-seeds"], 0.0005)
    conf = sorted((cc, s) for lab, cc, s, lr in calls if lab == "confirm")
    check("F5 one-shot final-test bundle (kind final_test): unchanged, seeds 1-5 forced, no re-sweep",
          R.final_test and not R.report_only and conf == sorted((cc, s) for cc in R.cands for s in (1, 2, 3, 4, 5))
          and not any(lab == "confirm_resweep" for lab, *_ in calls), "%d confirm runs" % len(conf))
    shutil.rmtree(os.path.join(DRY, "stub"), ignore_errors=True)

    # ---------------------------------------------------------------- F6
    import hk_lm
    b = hk_lm.Bundle(bdir, verify=True)
    rt = hk_lm.Runtime("cpu", threads=2)
    t6 = time.time()
    rec = hk_lm.run_one(b, "gpt-4o", "confirm", 1, 1e-3, rt, label="smoke_external",
                        overrides={"max_steps": 3, "eval": ("first", 4), "curve": False, "keep_model": True},
                        log=lambda *x: print(*x, flush=True))
    model = rec.pop("_model")
    bits = np.asarray(rec["dev"]["bits"])
    nbytes = np.asarray(rec["dev"]["bytes"])
    c = b.cand("gpt-4o")
    ok_run = (rec["status"] == "ok" and len(rec["train_loss"]) == 3 and all(math.isfinite(x) for x in rec["train_loss"])
              and rec["model"]["n_vocab"] == 200020 and rec["model"]["params"]["token_embedding_tied"] == 200020 * 192
              and abs(rec["bpb"] - float(bits.sum() / nbytes.sum())) < 1e-12 and rec["data"]["ctx_tokens"] == c["ctx_tokens"])
    check("F6 gpt-4o smoke (confirm model, 3 steps, V=200,020): finite loss, bpb = sum(bits)/sum(bytes)",
          ok_run, "loss %s, bpb %.4f on 4 docs, ctx %d, %.0fs" % ([round(x, 3) for x in rec["train_loss"]], rec["bpb"],
                                                                   c["ctx_tokens"], time.time() - t6))
    toks, offs = b.tokens("gpt-4o", "dev")
    lens = np.diff(offs)
    k = int(np.argmin(lens))
    seq = np.concatenate([[c["bos_id"]], toks[offs[k]:offs[k + 1]].astype(np.int64), [c["eot_id"]]])
    ev = hk_lm.evaluate(model, b, "gpt-4o", np.array([k]), int(c["ctx_tokens"]), rt)
    bf = hk_lm.brute_force_doc_bits(model, seq, int(c["ctx_tokens"]), rt)
    rel = abs(ev["sum_bits"] - bf) / bf
    check("F6 gpt-4o: windowed evaluation of the shortest doc == token-by-token brute force",
          rel < 1e-5, "doc %d, %d targets, %.4f vs %.4f bits (rel %.1e)" % (k, len(seq) - 1, ev["sum_bits"], bf, rel))

    # ---------------------------------------------------------------- clean up + result
    keep_files = {"manifest.json": os.path.join(res["stage"], "manifest.json"),
                  "PARITY.json": os.path.join(res["stage"], "PARITY.json"),
                  "PARTS.json": os.path.join(res["upload"], "PARTS.json")}
    for nm, p in keep_files.items():
        shutil.copy2(p, os.path.join(DRY, "DRYRUN_" + nm))
    os.replace(os.path.join(DRY, "FINAL_TEST_LOG.json"), os.path.join(DRY, "DRYRUN_FINAL_TEST_LOG.json"))
    for d in ("build", "work", "out", "data", "x1", "x2", "x3", "x4"):
        shutil.rmtree(os.path.join(DRY, d), ignore_errors=True)
    n_fail = sum(1 for v in RES.values() if not v["pass"])
    out = {"passed": len(RES) - n_fail, "failed": n_fail, "seconds": round(time.time() - t0, 1), "tests": RES,
           "build_bundle_sha256": sha(os.path.join(HERE, "build_bundle.py")),
           "run_all_sha256": sha(os.path.join(HERE, "run_all.py")),
           "hk_lm_sha256": sha(os.path.join(HERE, "hk_lm.py")),
           "note": "fake test file: strict TEST uids (manifest metadata) with dev_strict texts; no real test text "
                   "read; the fake data, bundle and outputs were deleted, the manifest / PARITY / PARTS / log kept "
                   "with the prefix DRYRUN_"}
    with open(os.path.join(DRY, "dryrun_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\n%d passed, %d failed in %.0fs" % (out["passed"], n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
