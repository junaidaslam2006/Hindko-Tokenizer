# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 5: the full eval/harness.py pass on test_strict (491 documents), exactly as
eval/run_test_competitors.py does it (harness.run(adapter, data=test_strict, gates='g1', allow_test=True)), for
  - the RELEASED tokenizer (F:\\Hindko\\tokenizer\\tokenizer.json, sha256 checked), and
  - one representative of every distinct behaviour fingerprint found by screen.py that is NOT already one of the
    66 baselines (whose test results already exist in eval/results/test_strict with the same harness code).
    python run_refresh.py --shard K --of 3        (K = 0, 1, 2; 1 thread each; resumable)
Output: ../results/<name>/{docs.jsonl, summary.json}; ../_groups.json (written by --plan)
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refresh_lib as R  # noqa: E402
import harness as H  # noqa: E402

H._setup_threads(1)
ROOT = R.ROOT
RES = os.path.join(ROOT, "results")
GROUPS = os.path.join(ROOT, "_groups.json")
BASE_RES = os.path.join(R.TOK, "eval", "results", "test_strict")
RELEASED_NAME = "hindko-tokenizer-1.0.0-released"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fp_of_docs(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return hashlib.sha256(json.dumps([[r["tokens"], r["rt"]] for r in rows]).encode()).hexdigest()


def plan():
    """Group every screened tokenizer by fingerprint; map fingerprints to the 66 baselines."""
    base_fp = {}
    for d in sorted(glob.glob(os.path.join(BASE_RES, "*", "docs.jsonl"))):
        name = os.path.basename(os.path.dirname(d))
        base_fp.setdefault(fp_of_docs(d), []).append(name)
    groups = {}
    screens = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(ROOT, "screen", "*.json")))]
    for s in screens:
        if s.get("status") != "ok":
            continue
        fp = s["screen"]["fingerprint"]
        g = groups.setdefault(fp, {"fingerprint": fp, "baselines": base_fp.get(fp, []), "members": []})
        g["members"].append({"name": s["name"], "repo": s["repo"], "dir": s.get("dir"), "downloads": s.get("downloads") or 0,
                             "likes": s.get("likes") or 0, "bytes_per_token": s["screen"]["bytes_per_token"],
                             "g1_pass_docs": s["screen"]["g1_pass_docs"]})
    for g in groups.values():
        g["members"].sort(key=lambda m: -(m["downloads"] + 50 * m["likes"]))
        g["representative"] = None if g["baselines"] else g["members"][0]["name"]
    out = {"n_screened_ok": sum(1 for s in screens if s.get("status") == "ok"), "n_screened": len(screens),
           "n_groups": len(groups), "n_groups_equal_to_a_baseline": sum(1 for g in groups.values() if g["baselines"]),
           "n_to_run": sum(1 for g in groups.values() if g["representative"]),
           "groups": sorted(groups.values(), key=lambda g: -g["members"][0]["bytes_per_token"])}
    json.dump(out, open(GROUPS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: v for k, v in out.items() if k != "groups"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only-missing", action="store_true")
    a = ap.parse_args()
    if a.plan:
        return plan()
    tm = json.load(open(R.TEST_MANIFEST, encoding="utf-8"))
    if sha(R.TEST) != tm["views"]["test_strict"]["sha256"]:
        raise SystemExit("data/test_strict.jsonl does not match data/test_manifest.json")
    G = json.load(open(GROUPS, encoding="utf-8"))
    reps = [g["representative"] for g in G["groups"] if g["representative"]]
    if "--only-missing" in sys.argv:          # later passes: only representatives without a result yet
        reps = [n for n in reps if not os.path.exists(os.path.join(RES, H.safe_name(n), "summary.json"))]
    mine = reps[a.shard::a.of]
    screens = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(ROOT, "screen", "*.json"))}
    jobs = ([RELEASED_NAME] if a.shard == 0 else []) + mine
    err = os.path.join(ROOT, "logs", "harness_errors_shard%d.jsonl" % a.shard)
    H.log("shard %d/%d: %d tokenizers" % (a.shard, a.of, len(jobs)))
    for name in jobs:
        out_dir = os.path.join(RES, H.safe_name(name))
        t0 = time.time()
        try:
            if name == RELEASED_NAME:
                ad = R.released_adapter()
                role = "released"
            else:
                s = json.load(open(screens[name], encoding="utf-8"))
                ad = R.adapter(s["spec"], extra_meta={"source_repo": s["repo"], "dir": s.get("dir"), "kind": s["kind"]})
                role = "external (refresh 2026-09-27)"
            H.run(ad, data=R.TEST, gates="g1", allow_test=True, out_dir=out_dir,
                  extra={"refresh": "sota/competitors 2026-09-27; reported only, changes no decision", "role": role})
        except Exception as e:  # noqa: BLE001
            with open(err, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tokenizer": name, "error": repr(e)[:300], "trace": traceback.format_exc()[-3000:]}) + "\n")
            H.log("ERROR", name, repr(e)[:200])
        H.log("done", name, "%.0fs" % (time.time() - t0))
    H.log("shard %d finished" % a.shard)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
