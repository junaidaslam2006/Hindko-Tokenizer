# -*- coding: utf-8 -*-
"""test_local.py - small CPU tests of the Colab arbiter code (<= 2 threads, small slices).

    set PYTHONIOENCODING=utf-8 & python test_local.py            (all tests, ~10-15 min on the busy CPU)
    python test_local.py --skip-smoke                            (without the run_all end-to-end smoke)

 T1 bundle: build from 2 tokenizers on a slice (train 700 / dev 80 docs), parts of 0.25 MB; sha256 of
    every file; reassembly from parts; a corrupted part is rejected; independent round trip and
    byte counts from the extracted bundle; the test-split guard.
 T2 training: 'screen' for 30 steps on both candidates: loss decreases, equal step counts, bpb equals
    sum(bits)/sum(bytes) recomputed from the per-document arrays and from the raw text bytes.
 T3 bpb brute force: one short document (no windows: full prefix, token by token) and one document
    longer than ctx (protocol windows), each recomputed one forward pass per token.
 T4 determinism: two identical CPU runs give identical losses and per-document bits.
 T5 windows: every target scored exactly once with >= ctx/2 context after the first window.
 T6 run_all.py --smoke end to end on CPU (parity vs PARITY.json, LR choice, stages, summary, zip).
 T7 resume safety: lr_tag is lossless; on a copy of the T6 results, --lr 5.8e-3 trains new runs instead
    of reusing the lr 6e-3 ones; a stored record whose LR, budget, steps/overrides or AMP dtype differs
    from the requested run is refused (dtype: reused only with --allow-mixed); stage means select
    records by their own lr field.   (python test_local.py --t7-only re-runs T7 alone.)
Writes colab/_test/test_local_result.json and (via the build) colab/PARITY.json.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
TEST = os.path.join(HERE, "_test")
PILOT = os.path.join(TOK, "research", "pilot")
sys.path.insert(0, HERE)

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"pass": bool(cond), "detail": detail}
    print("%s %-58s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)
    return cond


def _expect_refusal(ra, argv, must):
    """ra.main(argv) must stop with SystemExit whose text contains every string in `must`."""
    try:
        rc = ra.main(argv)
        return False, "not refused (rc %s)" % rc
    except SystemExit as e:
        msg = str(e)
        fields = [" ".join(x.split()) for x in msg.splitlines() if x.startswith("  ")]
        return all(m in msg for m in must), "; ".join(fields)[:200] or msg[:200]


def t7_resume(ra, bdir, b, smoke_root, base):
    """T7 resume safety, on a copy of the T6 smoke results (chosen LR 6e-3, grid runs present).
    Every refusal case also passes --max-hours ~0: if a check wrongly let a run through, the script
    would stop at the deadline instead of training."""
    out2 = os.path.join(TEST, "out_resume")
    shutil.rmtree(out2, ignore_errors=True)
    root2 = os.path.join(out2, os.path.basename(smoke_root))
    shutil.copytree(smoke_root, root2)
    res2 = os.path.join(root2, "results")
    common = ["--bundle-dir", bdir, "--out", out2, "--device", "cpu", "--threads", "2", "--smoke", "--no-zip",
              "--stages", "screen"]
    stop = ["--max-hours", "1e-9"]

    def load(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def dump(p, obj):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    # (1) the reviewer's case: --lr 5.8e-3 next to lr 6e-3 results must train new runs, not reuse them
    before = set(os.listdir(res2))
    rc = ra.main(common + ["--lr", "5.8e-3"])
    new = sorted(set(os.listdir(res2)) - before)
    log = open(os.path.join(root2, "progress.log"), encoding="utf-8").read()
    tail = log[log.rfind("===== stage screen"):]
    recs = [load(os.path.join(res2, f)) for f in new]
    pc = load(os.path.join(root2, "power_check_screen.json"))
    b58 = [r["bpb"] for r in recs if r["candidate"] == base]
    check("T7 --lr 5.8e-3 beside lr 6e-3 results: 6 new runs keyed lr5.8e-3, none reused",
          rc == 0 and len(new) == 6 and all("__lr5.8e-3__s" in f for f in new) and all(r["lr"] == 5.8e-3 for r in recs)
          and "skip screen" not in tail and "(6 new runs)" in tail and len(b58) == 3
          and abs(pc["mean_bpb"] - sum(b58) / 3) < 1e-12,
          "%s ...; power check mean %.5f = mean of the 5.8e-3 baseline runs" % (new[0] if new else "-", pc["mean_bpb"]))

    # (2) a record with another LR stored under the requested key is refused (no skip, nothing written)
    src = os.path.join(res2, "screen__%s__lr6e-3__s1.json" % base)
    planted = os.path.join(res2, "screen__%s__lr2.5e-3__s1.json" % base)
    shutil.copy2(src, planted)
    n0 = len(os.listdir(res2))
    ok, msg = _expect_refusal(ra, common + ["--lr", "2.5e-3"] + stop,
                              ["REFUSING TO RESUME", "stored 0.006, requested 0.0025"])
    check("T7 stored lr 6e-3 under the requested lr 2.5e-3 key -> refused", ok and len(os.listdir(res2)) == n0, msg)
    os.remove(planted)

    # (3) same LR, different budget in the stored record -> refused
    p58 = os.path.join(res2, "screen__%s__lr5.8e-3__s1.json" % base)
    orig = open(p58, encoding="utf-8").read()
    r = json.loads(orig)
    r["data"]["budget_bytes"] = 123
    dump(p58, r)
    ok, msg = _expect_refusal(ra, common + ["--lr", "5.8e-3"] + stop, ["REFUSING TO RESUME", "data.budget_bytes"])
    check("T7 stored budget_bytes differs -> refused", ok, msg)

    # (4) stored AMP dtype differs -> refused, unless --allow-mixed (then reused with a warning)
    r = json.loads(orig)
    r["runtime"]["amp_dtype"] = "bf16"
    dump(p58, r)
    ok, msg = _expect_refusal(ra, common + ["--lr", "5.8e-3"] + stop,
                              ["REFUSING TO RESUME", "runtime.amp_dtype", "--allow-mixed"])
    rc = ra.main(common + ["--lr", "5.8e-3", "--allow-mixed"])
    log = open(os.path.join(root2, "progress.log"), encoding="utf-8").read()
    tail = log[log.rfind("===== stage screen"):]
    check("T7 stored AMP dtype differs -> refused; --allow-mixed reuses it with a warning",
          ok and rc == 0 and "WARNING: --allow-mixed: reusing" in tail and "(0 new runs)" in tail, msg)
    with open(p58, "w", encoding="utf-8") as f:
        f.write(orig)

    # (5) a smoke record (12 steps, overrides) offered to a full-budget run -> refused
    out3 = os.path.join(TEST, "out_resume_full")
    shutil.rmtree(out3, ignore_errors=True)
    res3 = os.path.join(out3, b.manifest_sha256[:12], "results")
    os.makedirs(res3)
    shutil.copy2(src, os.path.join(res3, os.path.basename(src)))
    ok, msg = _expect_refusal(ra, ["--bundle-dir", bdir, "--out", out3, "--device", "cpu", "--threads", "2", "--no-zip",
                                   "--stages", "screen", "--lr", "6e-3"] + stop,
                              ["REFUSING TO RESUME", "data.steps", "overrides"])
    check("T7 smoke record offered to the full-budget run -> refused (steps, overrides)", ok, msg)
    shutil.rmtree(out3, ignore_errors=True)

    # (6) stage means select records by their own lr field, not by the file-name prefix
    fake = load(src)
    fake["lr"] = 5.8e-3
    fake["seed"] = 9
    planted = os.path.join(res2, "screen__%s__lr6e-3__s9.json" % base)
    dump(planted, fake)
    R = ra.Runner(ra.build_parser().parse_args(common + ["--lr", "6e-3"]), bdir, None)
    n_prefix = sum(1 for f in os.listdir(res2) if f.startswith("screen__%s__lr6e-3__s" % base))
    n6 = R._stage_means("screen", 6e-3)[base][2]
    check("T7 _stage_means filters on the record's lr (not the file name)", n6 == 3 and n_prefix == 4,
          "%d files match the old prefix, %d records at lr 6e-3 used" % (n_prefix, n6))
    os.remove(planted)
    shutil.rmtree(out2, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--t7-only", action="store_true", help="only T7, on the existing _test bundle + smoke output")
    a = ap.parse_args()
    t0 = time.time()
    import numpy as np
    import build_bundle as bb
    import run_all as ra
    if a.t7_only:
        P = json.load(open(os.path.join(TEST, "build", "upload", "PARTS.json")))
        bdir = os.path.join(TEST, "work", P["bundle_manifest_sha256"][:12], "bundle")
        b = ra.import_bundled_hk_lm(bdir).Bundle(bdir)
        t7_resume(ra, bdir, b, os.path.join(TEST, "out", b.manifest_sha256[:12] + "_smoke"), b.baseline_id)
        n_fail = sum(1 for v in RESULTS.values() if not v["pass"])
        print("\nT7 only: %d passed, %d failed in %.0fs" % (len(RESULTS) - n_fail, n_fail, time.time() - t0))
        return 1 if n_fail else 0

    cands = [(os.path.join(PILOT, "hf_bpe_P1_16000.json"), "hf", "pilotA1_P1_16k"),
             (os.path.join(PILOT, "sp_unigram_16000.model"), "sentencepiece", "pilotSPuni_16k")]
    build_out = os.path.join(TEST, "build")
    # ------------------------------------------------------------------ T1
    if not a.skip_build:
        args = ["--candidates"] + sum([[p, k] for p, k, _ in cands], []) + ["--ids"] + [i for _, _, i in cands] + \
               ["--baseline", "pilotA1_P1_16k", "--max-train-docs", "700", "--max-dev-docs", "80",
                "--out", build_out, "--part-mb", "0.25", "--workers", "2", "--no-newline-wrapper"]
        bb.main(args)
    upload = os.path.join(build_out, "upload")
    P = json.load(open(os.path.join(upload, "PARTS.json")))
    check("T1 bundle split into several parts <= limit", len(P["parts"]) >= 2 and
          all(p["bytes"] <= P["part_max_bytes"] for p in P["parts"]),
          "%d parts, max %d bytes" % (len(P["parts"]), max(p["bytes"] for p in P["parts"])))
    work = os.path.join(TEST, "work")
    if os.path.isdir(work):
        shutil.rmtree(work)
    P2, bdir = ra.reassemble(upload, work)
    import hk_lm
    b = hk_lm.Bundle(bdir, verify=True)
    check("T1 reassembled from parts, manifest sha256 matches PARTS.json",
          b.manifest_sha256 == P["bundle_manifest_sha256"], b.manifest_sha256[:16])
    check("T1 every bundled file verified by sha256", b.verify() == len(b.m["files"]), "%d files" % len(b.m["files"]))
    # corrupted part must be rejected
    bad_dir = os.path.join(TEST, "corrupt")
    if os.path.isdir(bad_dir):
        shutil.rmtree(bad_dir)
    shutil.copytree(upload, bad_dir)
    pth = os.path.join(bad_dir, P["parts"][1]["name"])
    with open(pth, "r+b") as f:
        f.seek(100)
        c = f.read(1)
        f.seek(100)
        f.write(bytes([c[0] ^ 0xFF]))
    try:
        ra.reassemble(bad_dir, os.path.join(TEST, "work_corrupt"))
        ok = False
        msg = "not rejected"
    except SystemExit as e:
        ok = "PART VERIFICATION FAILED" in str(e)
        msg = str(e)[:90]
    check("T1 corrupted part rejected", ok, msg)
    shutil.rmtree(bad_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(TEST, "work_corrupt"), ignore_errors=True)
    # independent round trip + bytes from the extracted bundle
    dev_recs = {r["uid"]: r for r in bb.read_jsonl(os.path.join(TOK, "data", "dev_strict.jsonl"), "validation",
                                                     bb.load_split_manifest()[0], require_strict=True)}
    docs = b.dev_docs()
    texts = [dev_recs[d["uid"]]["text"] for d in docs]
    check("T1 dev bytes == UTF-8 bytes of the text", all(int(x) == len(t.encode("utf-8")) for x, t in zip(b.dev_bytes(), texts)),
          "%d docs" % len(texts))
    for (p, k, cid) in cands:
        cm = b.cand(cid)
        src = "harness" if cm["encoder_source"].startswith("eval/harness.py") else "builtin"
        nnw = k == "sentencepiece"
        enc = bb.load_encoder(p, k, src, cid=cid, no_newline_wrapper=nnw)
        toks, offs = b.tokens(cid, "dev")
        bad = sum(1 for i, t in enumerate(texts) if enc.decode(toks[offs[i]:offs[i + 1]].tolist()) != t)
        same = sum(1 for i, t in enumerate(texts) if enc.encode(t) == toks[offs[i]:offs[i + 1]].tolist())
        check("T1 %s: decode(bundle tokens) == text, re-encode identical" % cid, bad == 0 and same == len(texts),
              "encoder %s; rt failures %d, identical encodings %d/%d, dtype %s" % (src, bad, same, len(texts), toks.dtype))
        if src == "harness":
            eb = bb.load_encoder(p, k, "builtin", no_newline_wrapper=nnw)
            same_b = sum(1 for i, t in enumerate(texts) if eb.encode(t) == toks[offs[i]:offs[i + 1]].tolist())
            check("T1 %s: harness encoding == native library encoding" % cid, same_b == len(texts),
                  "%d/%d docs" % (same_b, len(texts)))
    # test-split guard (uses one TEST uid from the manifest - metadata only, no test text is read)
    man, _ = bb.load_split_manifest()
    test_uid = next(u for u, (sp, _) in man.items() if sp == "test")
    fake = os.path.join(TEST, "fake_dev.jsonl")
    with open(fake, "w", encoding="utf-8") as f:
        f.write(json.dumps({"uid": test_uid, "text": "x", "tier": "strict"}) + "\n")
    try:
        bb.read_jsonl(fake, "validation", man, require_strict=True)
        g1 = False
    except SystemExit as e:
        g1 = "TEST" in str(e)
    fake2 = os.path.join(TEST, "dev_test_named.jsonl")
    shutil.copy2(fake, fake2)
    try:
        bb.read_jsonl(fake2, "validation", man)
        g2 = False
    except SystemExit as e:
        g2 = "never read" in str(e)
    os.remove(fake)
    os.remove(fake2)
    check("T1 test-split guard (test uid rejected, *test* file never opened)", g1 and g2)
    # WAVES.json path (synthetic file in the expected shape: id, tokenizer_path, encoder_kind)
    wdir = os.path.join(TEST, "waves")
    os.makedirs(wdir, exist_ok=True)
    wp = os.path.join(wdir, "WAVES.json")
    with open(wp, "w", encoding="utf-8") as f:
        json.dump({"waves": [{"wave": 1, "finalists": [
            {"id": "SPuni", "tokenizer_path": os.path.relpath(cands[1][0], TOK), "encoder_kind": "sentencepiece",
             "newline_wrapper": False},
            {"id": "A1_P1_16k", "tokenizer_path": cands[0][0], "encoder_kind": "hf", "baseline": True}]}]}, f)
    wr = bb.main(["--waves", wp, "--max-train-docs", "40", "--max-dev-docs", "8", "--no-parity", "--workers", "1",
                  "--out", os.path.join(wdir, "build")])
    wm = json.load(open(os.path.join(wr["stage"], "manifest.json"), encoding="utf-8"))
    check("T1 WAVES.json parsed (nested, relative path, baseline flag, SP wrapper flag)",
          [c["id"] for c in wm["candidates"]] == ["A1_P1_16k", "SPuni"] and wm["baseline_id"] == "A1_P1_16k"
          and wm["candidates"][1]["sp_newline_wrapper"] is False and wm["waves_sha256"],
          "baseline %s (%s)" % (wm["baseline_id"], wm["baseline_rule"]))
    shutil.rmtree(wdir, ignore_errors=True)

    # ------------------------------------------------------------------ T2 / T4
    rt = hk_lm.Runtime("cpu", threads=2)
    ov = {"max_steps": 30, "eval": ("first", 24), "curve_eval": ("first", 8)}
    recs = {}
    for _, _, cid in cands:
        recs[cid] = hk_lm.run_one(b, cid, "screen", 1, 3e-3, rt, overrides=dict(ov, keep_model=True))
    steps = {c: r["data"]["steps"] for c, r in recs.items()}
    full_steps = {c: hk_lm.steps_for_budget(hk_lm.STAGES["screen"]["budget_bytes"]) for c in recs}
    check("T2 equal step counts across candidates (30 here; 814 at 10 MB)",
          len(set(steps.values())) == 1 and len(set(full_steps.values())) == 1, "%s / %s" % (steps, full_steps))
    for c, r in recs.items():
        L = r["train_loss"]
        f5, l5 = sum(L[:5]) / 5, sum(L[-5:]) / 5
        check("T2 %s loss decreases" % c, l5 < f5 - 0.5, "first5 %.3f -> last5 %.3f" % (f5, l5))
        bits, byt = r["dev"]["bits"], r["dev"]["bytes"]
        re_bpb = math.fsum(bits) / sum(byt)
        idx = r["dev"]["doc_index"]
        raw_bytes = sum(len(texts[i].encode("utf-8")) for i in idx)
        check("T2 %s bpb == sum(bits)/sum(bytes)" % c, abs(re_bpb - r["bpb"]) <= 1e-12 * r["bpb"] and raw_bytes == sum(byt),
              "record %.9f recomputed %.9f, bytes %d" % (r["bpb"], re_bpb, raw_bytes))
        check("T2 %s tokens seen = steps*8*ctx, ctx = round(1536/bpt)" % c,
              r["data"]["tokens_seen"] == 30 * 8 * r["data"]["ctx_tokens"] and
              r["data"]["ctx_tokens"] == round(1536 / b.cand(c)["train_bytes_per_token"]),
              "ctx %d, tokens %d, bytes seen %.0f (target %d)" % (r["data"]["ctx_tokens"], r["data"]["tokens_seen"],
                                                               r["data"]["bytes_seen"], 30 * hk_lm.BYTES_PER_STEP))
        check("T2 %s curve points 25/50/75/100%%" % c, [p["frac"] for p in r["curve"]] == [0.25, 0.5, 0.75, 1.0],
              " ".join("%.2f:%.4f" % (p["frac"], p["bpb"]) for p in r["curve"]))
    # identical document order across candidates (same permutation, different tokens)
    o1 = hk_lm.doc_order(len(b.train_bytes()), 1, 0)
    o2 = hk_lm.doc_order(len(b.train_bytes()), 1, 0)
    o3 = hk_lm.doc_order(len(b.train_bytes()), 2, 0)
    check("T2 document order depends on the seed only", (o1 == o2).all() and not (o1 == o3).all())

    # ------------------------------------------------------------------ T3 brute force
    base = cands[0][2]
    model = recs[base]["_model"]
    model.eval()
    ctx = recs[base]["data"]["ctx_tokens"]
    toks, offs = b.tokens(base, "dev")
    c = b.cand(base)
    lens = np.diff(offs)
    short = int(np.argmin(np.where(lens >= 8, lens, 10 ** 9)))
    longc = [i for i in range(len(lens)) if ctx + 10 < lens[i] + 1 <= int(2.2 * ctx)]
    long_ = longc[0] if longc else int(np.argmax(lens))
    for name, i in (("short (m <= ctx, no windows)", short), ("long (m > ctx, windows)", long_)):
        seq = np.concatenate([[c["bos_id"]], toks[offs[i]:offs[i + 1]].astype(np.int64), [c["eot_id"]]])
        ev = hk_lm.evaluate(model, b, base, np.array([i]), ctx, rt)
        bf = hk_lm.brute_force_doc_bits(model, seq, ctx, rt)
        rel = abs(ev["sum_bits"] - bf) / bf
        check("T3 brute-force bits, %s" % name, rel < 1e-5,
              "doc %d m=%d ctx=%d windows=%d: eval %.6f brute %.6f bits (rel %.1e); bpb %.6f" %
              (i, len(seq) - 1, ctx, ev["n_windows"], ev["sum_bits"], bf, rel, bf / int(b.dev_bytes()[i])))

    # ------------------------------------------------------------------ T4 determinism
    again = hk_lm.run_one(b, base, "screen", 1, 3e-3, rt, overrides=ov)
    check("T4 identical CPU runs -> identical losses and bits",
          again["train_loss"] == recs[base]["train_loss"] and again["dev"]["bits"] == recs[base]["dev"]["bits"],
          "final loss %.8f vs %.8f" % (again["train_loss"][-1], recs[base]["train_loss"][-1]))

    # ------------------------------------------------------------------ T5 windows
    okw = True
    for m in list(range(1, 40)) + [255, 256, 257, 511, 512, 513, 1000, 4097]:
        for cx in (4, 5, 16, 251, 256, 259):
            cov = [0] * (m + 1)
            for a_, end, first in hk_lm.eval_windows(m, cx):
                for rel in range(first, end - a_):
                    j = a_ + rel + 1
                    cov[j] += 1
                    ctx_len = rel + 1
                    wbf = max(0, -(-(j - cx) // max(1, cx // 2)))
                    if a_ != wbf * max(1, cx // 2):
                        okw = False
                    if a_ > 0 and ctx_len < cx - max(1, cx // 2):
                        okw = False
            if cov[1:] != [1] * m:
                okw = False
    check("T5 windows: each target once, >= ctx/2 context, matches brute-force formula", okw)

    # ------------------------------------------------------------------ T7a lossless LR keys
    named = [1e-3, 3e-3, 6e-3, 5e-4, 2e-3, 1.5e-3, 2.5e-3, 2.9e-3, 3.4e-3, 5.8e-3, 1e-5, 0.1 + 0.2]
    rnd = np.random.default_rng(0).uniform(1e-4, 1e-2, 2000).tolist()
    nbr = [float(np.nextafter(x, 1.0)) for x in (1e-3, 3e-3, 6e-3)]        # adjacent doubles
    vals = sorted(set(named + rnd + nbr))
    tags = [hk_lm.lr_tag(v) for v in vals]
    grid = [hk_lm.lr_tag(x) for x in (1e-3, 3e-3, 6e-3, 5e-4, 2e-3)]
    check("T7 lr_tag lossless: distinct LRs -> distinct keys, round-trips, grid tags unchanged",
          len(set(tags)) == len(vals) and all(float(t) == v for v, t in zip(vals, tags))
          and grid == ["1e-3", "3e-3", "6e-3", "5e-4", "2e-3"],
          "%d LRs; 2.5e-3 -> %s, 2.9e-3 -> %s, 3.4e-3 -> %s, 5.8e-3 -> %s, 1.5e-3 -> %s" %
          (len(vals), hk_lm.lr_tag(2.5e-3), hk_lm.lr_tag(2.9e-3), hk_lm.lr_tag(3.4e-3), hk_lm.lr_tag(5.8e-3),
           hk_lm.lr_tag(1.5e-3)))

    # ------------------------------------------------------------------ T6 smoke
    if not a.skip_smoke:
        out = os.path.join(TEST, "out")
        if os.path.isdir(out):
            shutil.rmtree(out)
        rc = ra.main(["--bundle-dir", bdir, "--out", out, "--device", "cpu", "--threads", "2", "--smoke"])
        root = os.path.join(out, b.manifest_sha256[:12] + "_smoke")
        S = json.load(open(os.path.join(root, "summary.json"), encoding="utf-8"))
        pr = json.load(open(os.path.join(root, "parity_report.json")))
        check("T6 run_all smoke exit 0, summary + zip written", rc == 0 and os.path.isfile(root + "_results.zip"),
              "stages %s" % sorted(S["stages"]))
        check("T6 parity re-run on CPU reproduces PARITY.json bitwise", pr["runs"]["fp32"]["bitwise_identical_to_cpu"],
              "bpb %.6f vs %.6f" % (pr["runs"]["fp32"]["bpb"], pr["cpu_reference"]["bpb"]))
        det = S.get("determinism_check", {})
        check("T6 lrsweep vs screen re-run bitwise identical", det.get("bitwise_identical") is True, det.get("verdict", ""))
        # resume: a second call must skip every run
        rc2 = ra.main(["--bundle-dir", bdir, "--out", out, "--device", "cpu", "--threads", "2", "--smoke", "--no-zip"])
        log = open(os.path.join(root, "progress.log"), encoding="utf-8").read()
        check("T6 re-run resumes (0 new runs)", rc2 == 0 and "(0 new runs)" in log.strip().splitlines()[-1],
              log.strip().splitlines()[-1][-40:])
        t7_resume(ra, bdir, b, root, cands[0][2])

    par = json.load(open(os.path.join(HERE, "PARITY.json"), encoding="utf-8"))
    check("PARITY.json written for this bundle", par["bundle_manifest_sha256"] == b.manifest_sha256,
          "cpu bpb %.6f, %d steps" % (par["cpu"]["bpb"], par["config"]["steps"]))
    n_fail = sum(1 for v in RESULTS.values() if not v["pass"])
    out = {"passed": len(RESULTS) - n_fail, "failed": n_fail, "seconds": round(time.time() - t0, 1),
           "bundle_manifest_sha256": b.manifest_sha256, "tests": RESULTS,
           "note": "pilot tokenizers (research/pilot), train slice 700 docs, dev slice 80 docs; encoders from "
                   "eval/harness.py build_adapter when present (SentencePiece pilot without '\\n' piece: "
                   "--no-newline-wrapper, as harness.py run would need)"}
    with open(os.path.join(TEST, "test_local_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\n%d passed, %d failed in %.0fs" % (out["passed"], n_fail, out["seconds"]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
