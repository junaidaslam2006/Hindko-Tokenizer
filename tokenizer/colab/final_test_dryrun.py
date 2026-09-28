# -*- coding: utf-8 -*-
"""final_test_dryrun.py - exercise the --final-test code paths of build_bundle.py and run_all.py WITHOUT
the real test text, before the one real final-test build.

    set PYTHONIOENCODING=utf-8 & python colab\\final_test_dryrun.py

 D1 read_jsonl guards: without the flag a test uid and a *test* file name are refused (as before); with
    final_test=True a file named test_strict.jsonl whose uids are all strict TEST uids is accepted, and a
    validation uid in it is refused; final_test with want_split != 'test' is refused.
 D2 build_bundle.main refuses --final-test with --max-dev-docs / --allow-roundtrip-failures, and without a
    matching test_manifest.json.
 D3 a complete --final-test build on a FAKE test file: 40 strict test uids from the split manifest
    (metadata only) carrying the texts of the first 40 dev_strict documents. FINAL_TEST_LOG is redirected
    to colab/_final_dryrun/, so the real log only ever holds the real build. Checks: manifest kind,
    PARTS.json kind, log entry started -> completed, train-token sha256 per candidate equal to the
    round-1/2 bundle manifests.
 D4 run_all.py on that bundle: default stages, lr/screen/large, --smoke, confirm without --lr or without
    --force-extra-seeds are all refused before anything runs; --stages parity runs on the CPU and
    reproduces the bundle's PARITY.json bitwise.
 D5 stage_confirm logic with a stubbed run(): final bundle -> seeds 1-5 for every candidate, no LR
    re-sweep even when the top two are within 1 %; a normal bundle with --force-extra-seeds -> seeds 1-5
    and the re-sweep as before; without the flag and a low power-check s.d. -> seeds 1-3 only.
Writes only under colab/_final_dryrun/. No real test text is read.
"""
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
DRY = os.path.join(HERE, "_final_dryrun")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
RES = {}


def check(name, cond, detail=""):
    RES[name] = {"pass": bool(cond), "detail": detail}
    print("%s %-70s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


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
    import build_bundle as bb
    import run_all as ra
    shutil.rmtree(DRY, ignore_errors=True)
    os.makedirs(os.path.join(DRY, "data"))
    man, man_sha = bb.load_split_manifest()
    test_uids = sorted(u for u, (sp, tier) in man.items() if sp == "test" and tier == "strict")[:40]
    val_uid = next(u for u, (sp, tier) in sorted(man.items()) if sp == "validation" and tier == "strict")
    dev = [json.loads(l) for l in open(os.path.join(TOK, "data", "dev_strict.jsonl"), encoding="utf-8")][:40]

    def write(p, recs):
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fake_recs = [dict(d, uid=u) for u, d in zip(test_uids, dev)]
    fake = os.path.join(DRY, "data", "test_strict.jsonl")
    write(fake, fake_recs)

    # ---------------------------------------------------------------- D1
    ok1, m1 = refused(lambda: bb.read_jsonl(fake, "validation", man, require_strict=True), ["never read"])
    other = os.path.join(DRY, "data", "fake_eval.jsonl")
    write(other, fake_recs[:1])
    ok2, m2 = refused(lambda: bb.read_jsonl(other, "validation", man, require_strict=True), ["TEST"])
    check("D1 no flag: *test* file name refused; test uid refused (unchanged guards)", ok1 and ok2, m2[:90])
    got = bb.read_jsonl(fake, "test", man, require_strict=True, final_test=True)
    check("D1 final_test: all-test file accepted", len(got) == 40 and [r["uid"] for r in got] == test_uids)
    mixed = os.path.join(DRY, "data", "mixed_test.jsonl")
    write(mixed, fake_recs[:3] + [dict(dev[0], uid=val_uid)])
    ok3, m3 = refused(lambda: bb.read_jsonl(mixed, "test", man, require_strict=True, final_test=True),
                      ["is split validation, expected test"])
    ok4, m4 = refused(lambda: bb.read_jsonl(fake, "validation", man, require_strict=True, final_test=True),
                      ["reads only the test split"])
    ok5, m5 = refused(lambda: bb.read_jsonl(fake, "test", None, require_strict=True, final_test=True),
                      ["split manifest"])
    check("D1 final_test: validation uid refused; want_split!=test refused; no manifest refused", ok3 and ok4 and ok5, m3[:90])

    # ---------------------------------------------------------------- D2
    waves = os.path.join(TOK, "lm", "WAVES_final_test.json")
    common = ["--final-test", "--waves", waves, "--baseline", "A1-P1r3-D2-16k", "--dev-jsonl", fake]
    ok6, m6 = refused(lambda: bb.main(common + ["--max-dev-docs", "5"]), ["--final-test encodes the whole"])
    ok7, m7 = refused(lambda: bb.main(common + ["--allow-roundtrip-failures"]), ["--final-test encodes the whole"])
    ok8, m8 = refused(lambda: bb.main(common + ["--no-parity", "--out", os.path.join(DRY, "nomanifest")]),
                      ["test_manifest.json"])
    with open(os.path.join(DRY, "data", "test_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"split_manifest_sha256": man_sha, "views": {"test_strict": {"sha256": "0" * 64}}}, f)
    ok9, m9 = refused(lambda: bb.main(common + ["--no-parity", "--out", os.path.join(DRY, "badsha")]),
                      ["does not match"])
    check("D2 --final-test refuses slicing, round-trip waiver, missing / mismatching test_manifest",
          ok6 and ok7 and ok8 and ok9, m9[:90])

    # ---------------------------------------------------------------- D3
    with open(os.path.join(DRY, "data", "test_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"split_manifest_sha256": man_sha, "views": {"test_strict": {"sha256": sha(fake)}}}, f)
    real_log = bb.FINAL_TEST_LOG
    bb.FINAL_TEST_LOG = os.path.join(DRY, "FINAL_TEST_LOG.json")
    real_parity = sha(os.path.join(HERE, "PARITY.json")) if os.path.isfile(os.path.join(HERE, "PARITY.json")) else None
    out = os.path.join(DRY, "build")
    try:
        res = bb.main(common + ["--out", out])
    finally:
        bb.FINAL_TEST_LOG = real_log
    m = json.load(open(os.path.join(res["stage"], "manifest.json"), encoding="utf-8"))
    P = json.load(open(os.path.join(res["upload"], "PARTS.json"), encoding="utf-8"))
    L = json.load(open(os.path.join(DRY, "FINAL_TEST_LOG.json"), encoding="utf-8"))
    e = L["entries"][-1]
    check("D3 manifest kind=final_test, PARTS.json kind, 'dev' = the fake test uids",
          m["kind"] == "final_test" and list(m)[:2] == ["format", "kind"] and P.get("kind") == "final_test"
          and m["data"]["n_dev_docs"] == 40 and m["final_test"]["log_entry"] == 0,
          "%s, %d dev docs" % (m["kind"], m["data"]["n_dev_docs"]))
    check("D3 log entry: started -> completed, candidates + tokenizer sha256 + test sha256 + code sha256",
          len(L["entries"]) == 1 and e["status"] == "completed" and e["bundle_manifest_sha256"] == res["manifest_sha256"]
          and e["test_file"]["sha256"] == sha(fake) and all(c["tokenizer_sha256"] for c in e["candidates"])
          and e["code"]["build_bundle_sha256"] == sha(os.path.join(HERE, "build_bundle.py"))
          and set(e["g1_roundtrip_test_failures"].values()) == {0},
          "roles %s" % [c["role"] for c in e["candidates"]])
    check("D3 real FINAL_TEST_LOG.json untouched, colab/PARITY.json untouched",
          not os.path.exists(real_log) and (real_parity is None or sha(os.path.join(HERE, "PARITY.json")) == real_parity))
    old = {}
    for d in ("build", "build_r2"):
        mm = json.load(open(os.path.join(HERE, d, "staging", "manifest.json"), encoding="utf-8"))
        for c in mm["candidates"]:
            for k in ("train_tokens", "train_offsets"):
                old.setdefault((c["id"], k), set()).add(mm["files"]["%s/%s" % (c["id"], k)]["sha256"])
        for k in ("train_bytes", "train_uids"):
            old.setdefault(("shared", k), set()).add(mm["files"][k]["sha256"])
    same = []
    for c in m["candidates"]:
        for k in ("train_tokens", "train_offsets"):
            same.append(old.get((c["id"], k)) == {m["files"]["%s/%s" % (c["id"], k)]["sha256"]})
    for k in ("train_bytes", "train_uids"):
        same.append(old[("shared", k)] == {m["files"][k]["sha256"]})
    check("D3 train arrays sha256 == round-1/2 bundles (every candidate, shared arrays)", all(same) and len(same) == 10,
          "%d/%d equal" % (sum(same), len(same)))

    # ---------------------------------------------------------------- D4
    work = os.path.join(DRY, "work")
    P2, bdir = ra.reassemble(res["upload"], work)
    rout = os.path.join(DRY, "out")
    base = ["--bundle-dir", bdir, "--out", rout, "--device", "cpu", "--threads", "2", "--no-zip"]
    cases = [([], ["only --stages parity,confirm"]),
             (["--stages", "lr"], ["only --stages parity,confirm"]),
             (["--stages", "parity,confirm,large"], ["only --stages parity,confirm"]),
             (["--stages", "parity,confirm", "--smoke", "--lr", "1e-3", "--force-extra-seeds"], ["--smoke"]),
             (["--stages", "parity,confirm", "--force-extra-seeds"], ["--lr 1e-3"]),
             (["--stages", "parity,confirm", "--lr", "1e-3"], ["--force-extra-seeds"])]
    oks = []
    for extra, must in cases:
        ok, msg = refused(lambda: ra.main(base + extra), must)
        oks.append(ok)
    rdirs = [os.path.join(rout, res["manifest_sha256"][:12] + s, "results") for s in ("", "_smoke")]
    n_res = sum(len(os.listdir(d)) for d in rdirs if os.path.isdir(d))
    check("D4 run_all refuses every non-final-test invocation before running anything", all(oks) and n_res == 0,
          "%s, %d result files" % (oks, n_res))
    rc = ra.main(base + ["--stages", "parity"])
    pr = json.load(open(os.path.join(rout, res["manifest_sha256"][:12], "parity_report.json"), encoding="utf-8"))
    check("D4 --stages parity runs; CPU re-run reproduces the bundle's PARITY.json bitwise",
          rc == 0 and pr["runs"]["fp32"]["bitwise_identical_to_cpu"], "bpb %.6f" % pr["runs"]["fp32"]["bpb"])

    # ---------------------------------------------------------------- D5
    def stub(bundle_dir, argv, pooled):
        R = ra.Runner(ra.build_parser().parse_args(["--bundle-dir", bundle_dir, "--out", os.path.join(DRY, "stub"),
                                                    "--device", "cpu"] + argv), bundle_dir, None)
        calls = []
        R.run = lambda label, recipe, cand, seed, lr: calls.append((label, cand, seed, float(lr))) or {}
        R._pooled_rel_sd = lambda label, lr: (pooled, {})
        c = R.cands
        R._stage_means = lambda label, lr, recipe=None: {c[0]: (1.000, 0.0, 5), c[1]: (1.005, 0.0, 5)}
        R.stage_confirm()
        return R, calls
    R, calls = stub(bdir, ["--lr", "1e-3", "--force-extra-seeds"], 0.0005)
    conf = sorted((cc, s) for lab, cc, s, lr in calls if lab == "confirm")
    check("D5 final bundle: seeds 1-5 for all 4 candidates, no re-sweep (top two within 0.5 %)",
          R.final_test and conf == sorted((cc, s) for cc in R.cands for s in (1, 2, 3, 4, 5))
          and not any(lab == "confirm_resweep" for lab, *_ in calls), "%d confirm runs" % len(conf))
    # a normal (dev) bundle: the round-2 staging dir is a complete bundle directory
    dev_b = os.path.join(HERE, "build_r2", "staging")
    R, calls = stub(dev_b, ["--lr", "1e-3", "--force-extra-seeds"], 0.0005)
    n_conf = sum(1 for lab, *_ in calls if lab == "confirm")
    n_rs = sum(1 for lab, *_ in calls if lab == "confirm_resweep")
    check("D5 dev bundle + --force-extra-seeds: seeds 1-5, re-sweep unchanged",
          not R.final_test and n_conf == 5 * len(R.cands) and n_rs == 4, "%d confirm, %d resweep" % (n_conf, n_rs))
    pc = json.load(open(os.path.join(R.root, "power_check.json"), encoding="utf-8"))
    R, calls = stub(dev_b, ["--lr", "1e-3"], 0.0005)
    n_conf = sum(1 for lab, *_ in calls if lab == "confirm")
    pc2 = json.load(open(os.path.join(R.root, "power_check.json"), encoding="utf-8"))
    check("D5 dev bundle without the flag, low s.d.: seeds 1-3 only (as before)",
          n_conf == 3 * len(R.cands) and pc["extra_seeds_run"] == [4, 5] and pc.get("extra_seeds_forced")
          and pc2["extra_seeds_run"] == [] and "extra_seeds_forced" not in pc2, "%d confirm" % n_conf)
    R, calls = stub(dev_b, ["--lr", "1e-3"], 0.004)
    n_conf = sum(1 for lab, *_ in calls if lab == "confirm")
    check("D5 dev bundle without the flag, high s.d.: power check adds seeds 4, 5 (as before)",
          n_conf == 5 * len(R.cands), "%d confirm" % n_conf)
    shutil.rmtree(os.path.join(DRY, "stub"), ignore_errors=True)

    n_fail = sum(1 for v in RES.values() if not v["pass"])
    out = {"passed": len(RES) - n_fail, "failed": n_fail, "seconds": round(time.time() - t0, 1), "tests": RES,
           "build_bundle_sha256": sha(os.path.join(HERE, "build_bundle.py")),
           "run_all_sha256": sha(os.path.join(HERE, "run_all.py")),
           "note": "fake test file: strict TEST uids (manifest metadata) with dev_strict texts; no real test text read"}
    with open(os.path.join(DRY, "dryrun_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\n%d passed, %d failed in %.0fs" % (out["passed"], n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
