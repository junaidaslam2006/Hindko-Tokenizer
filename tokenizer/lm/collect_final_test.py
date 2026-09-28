#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_final_test.py - copy the one-shot FINAL TEST LM results (PLAN 7.6) into the project and validate every
record the way lm/collect_colab.py validated the dev records, plus the final-test-specific checks.

    PYTHONIOENCODING=utf-8 python lm/collect_final_test.py            # copy from Google Drive + validate
    PYTHONIOENCODING=utf-8 python lm/collect_final_test.py --no-copy  # re-validate the local copy only

Inputs (read-only):
  G:/My Drive/hindko_lm_out_final_test/<bundle>/...   Colab output (run_all.py) of the two final-test bundles
  colab/build_final_test/staging, colab/build_final_test_supp/staging   the bundles as built (manifest sha = bundle id)
  colab/hk_lm.py                                     the arbiter module (sha256 must equal the records')
  lm/WAVES_final_test.json, lm/WAVES_final_test_supp.json, FINAL_TEST_LOG.json
  lm/colab_results/results_table.json + the dev records   (check (b) of colab/FINAL_TEST.md: same models as dev)
  data/test_strict.jsonl: only the uid / group / cluster / source fields are read (never the text)
  splits/split_manifest.jsonl: only the rows of the 491 test uids (cluster cross-check)
Outputs (all under lm/colab_results/final_test/):
  <bundle>/...                        byte-exact copies of every file (sha256-verified), 'large' records included
  results_table_final_test.json/.csv  one row per bundle x stage x candidate x seed
  validation_final_test.json          every check, per record and per stage; copy manifest with sha256
  test_clusters.csv                   the 491 test documents: doc_index, uid, bytes, source, group, cluster

Per-record checks: validate_run_ft() is collect_colab.validate_run() generalised to a stage spec (seeds, exact LR,
pre-registration flag, expected steps) and to the test constants (491 documents, 1,451,026 bytes, the bundle's
0.5 MB subset). For every 'confirm' record, collect_colab.validate_run() itself is also run (with its three dev
constants N_DEV / DEV_BYTES / SUBSET_BYTES set to the test values) and must give the identical check list.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import sys

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
LM = os.path.join(TOK, "lm")
sys.path.insert(0, LM)
import collect_colab as CC  # noqa: E402  (module-level code only defines constants and functions)

COLAB = os.path.join(TOK, "colab")
DEV_RES = os.path.join(LM, "colab_results")
OUT = os.path.join(DEV_RES, "final_test")
SRC_DEFAULT = r"G:\My Drive\hindko_lm_out_final_test"
TEST_JSONL = os.path.join(TOK, "data", "test_strict.jsonl")
TEST_JSONL_SHA = "a74c33ad008f4b0e4a6f7779b0de24e265b0f4edf6f94e89885c7943d3a08468"
FINAL_TEST_LOG = os.path.join(TOK, "FINAL_TEST_LOG.json")
SPLIT_MANIFEST = CC.SPLIT_MANIFEST
SPLIT_SHA = CC.SPLIT_SHA
HK_LM_SHA = CC.HK_LM_SHA
RUN_ALL_BUNDLED_SHA = "8fbae15a0473bf651e912d36a6f6633fdf7c024022f4e0c8899c8e35b96e568b"

BUNDLES = collections.OrderedDict([
    ("f54c929ba1ab", {"sha": "f54c929ba1ab8f17c001d7fec10deb938f822785587a82bd6c837b1ea861961c",
                      "staging": os.path.join(COLAB, "build_final_test", "staging"),
                      "waves": os.path.join(LM, "WAVES_final_test.json"), "log_entry": 0,
                      "role": "the pre-registered one-shot test list (decision.json test_candidates_fixed)"}),
    ("94175d26497f", {"sha": "94175d26497f5c91bbc16c56860e52e6923cfd1fd78019352231479e81fe64ac",
                      "staging": os.path.join(COLAB, "build_final_test_supp", "staging"),
                      "waves": os.path.join(LM, "WAVES_final_test_supp.json"), "log_entry": 1,
                      "role": "supplement added after the adversarial audit, before any test LM number existed"}),
])
BASELINE = "A1-P1r3-D2-16k"

N_TEST = 491
TEST_BYTES = 1451026
TRAIN_BYTES = CC.TRAIN_BYTES
TRAIN_DOCS = CC.TRAIN_DOCS
BYTES_SEEN_TOL = CC.BYTES_SEEN_TOL
TOLERANCE_CHECKS = CC.TOLERANCE_CHECKS

# stage -> (recipe, allowed seeds, exact LR, expected 'preregistered' flag, expected epochs, role)
STAGE_SPEC_FT = {
    "confirm": ("confirm", {1, 2, 3, 4, 5}, 1e-3, True, 1,
                "one-shot TEST, Stage 4 'confirm' recipe (PLAN 7.6; seeds 1-5 forced)"),
    "large": ("large", {1, 2}, 5e-4, False, 2,
              "REPORT-ONLY large-arbiter test run (AMENDMENT_1.md item 4; LR fixed from dev, no sweep on test)"),
}
STAGE_ORDER = ["confirm", "large"]
EXPECTED_LARGE = [BASELINE, "R2-A10-MinGram-P1r3-D2-48k", "R2-A4-SPnat-D2-32k", "R2-A10-MinGram-P1r3-D2-32k"]


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


sha256_file = CC.sha256_file
load_json = CC.load_json


# ------------------------------------------------------------------ copy
def copy_bundle(src_root, bid, dst_root, do_copy):
    """Every file of the Colab folder, byte-exact ('.tmp' leftovers of atomic writes excepted)."""
    src = os.path.join(src_root, bid)
    dst = os.path.join(dst_root, bid)
    copied, excluded = [], []
    if do_copy:
        if not os.path.isdir(src):
            raise SystemExit("source folder missing: %s" % src)
        if os.path.isdir(dst):
            shutil.rmtree(dst)          # this script's own output; rebuilt from the source every run
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames.sort()
            for fn in sorted(filenames):
                sp = os.path.join(dirpath, fn)
                rel = os.path.relpath(sp, src).replace("\\", "/")
                with open(sp, "rb") as f:
                    data = f.read()
                ent = {"path": rel, "bytes": len(data), "sha256": CC.sha256_bytes(data),
                       "source_mtime_utc": datetime.datetime.fromtimestamp(os.path.getmtime(sp), datetime.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ")}
                if ".tmp" in fn:
                    ent["why"] = "temporary file of an atomic write"
                    excluded.append(ent)
                    continue
                dp = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(dp), exist_ok=True)
                with open(dp + ".part", "wb") as f:
                    f.write(data)
                os.replace(dp + ".part", dp)
                ent["dest_sha256_verified"] = sha256_file(dp) == ent["sha256"]
                copied.append(ent)
    else:
        for dirpath, dirnames, filenames in os.walk(dst):
            for fn in sorted(filenames):
                dp = os.path.join(dirpath, fn)
                rel = os.path.relpath(dp, dst).replace("\\", "/")
                sp = os.path.join(src, rel)
                s_sha = sha256_file(sp) if os.path.isfile(sp) else None
                d_sha = sha256_file(dp)
                copied.append({"path": rel, "bytes": os.path.getsize(dp), "sha256": d_sha,
                               "dest_sha256_verified": (s_sha == d_sha) if s_sha else None})
    return copied, excluded


# ------------------------------------------------------------------ test documents (no text)
def test_docs_meta():
    """uid -> {group, cluster, source} from test_strict.jsonl; the text field is dropped unread (never stored)."""
    out = collections.OrderedDict()
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out[x["uid"]] = {"group": x["group"], "cluster": x["cluster"], "source": x["source"], "tier": x["tier"]}
    return out


def split_manifest_rows(uids):
    want = set(uids)
    got = {}
    uid_re = re.compile(r'"uid"\s*:\s*"([^"]+)"')
    with open(SPLIT_MANIFEST, encoding="utf-8") as f:
        for line in f:
            m = uid_re.search(line)
            if not m or m.group(1) not in want:
                continue
            x = json.loads(line)
            got[x["uid"]] = x
    return got


# ------------------------------------------------------------------ per-record validation
def validate_run_ft(hk, B, bctx, rel, rec, pin):
    """collect_colab.validate_run() generalised to STAGE_SPEC_FT and the test constants (same check names)."""
    ck = CC.Checks()
    bid, spec_b = bctx["bid"], bctx["spec"]
    fn = rel.split("/")[-1]
    key = CC.parse_key(fn)
    stage = rec.get("stage")
    cid = rec.get("candidate")
    seed = rec.get("seed")
    lr = rec.get("lr")
    ck.add("status_ok", rec.get("status") == "ok", rec.get("status"))
    ck.add("protocol", rec.get("protocol") == hk.PROTOCOL, rec.get("protocol"))
    ck.add("bundle_sha", rec.get("bundle_manifest_sha256") == spec_b["sha"] == B.manifest_sha256
           and spec_b["sha"].startswith(bid), rec.get("bundle_manifest_sha256"))
    ck.add("hk_lm_sha", rec.get("hk_lm_sha256") == HK_LM_SHA == B.m["builder"]["hk_lm_sha256"], rec.get("hk_lm_sha256"))
    ok_key = bool(key) and key["label"] == stage and key["slug"] == CC.slug(str(cid)) and int(key["seed"]) == seed \
        and lr is not None and key["lr"] == hk.lr_tag(lr)
    ck.add("file_key_matches_record", ok_key, key)
    spec = STAGE_SPEC_FT.get(stage)
    ck.add("stage_known", spec is not None, stage)
    ck.add("candidate_in_bundle", cid in B.candidate_ids, cid)
    ck.add("candidate_in_waves", cid in bctx["waves_ids"], cid)
    if spec:
        recipe, seeds, lr_exact, prereg, epochs, _ = spec
        ck.add("stage_recipe", rec.get("stage_recipe") == recipe, rec.get("stage_recipe"))
        ck.add("seed_allowed", seed in seeds, seed)
        lr_ok = lr is not None and float(lr) == lr_exact
        ck.add("lr_exact", lr_ok and rec["optimizer"]["peak_lr"] == lr,
               {"lr": lr, "peak_lr": rec["optimizer"]["peak_lr"], "required": lr_exact})
        ck.add("preregistered_flag", rec.get("preregistered") is prereg, rec.get("preregistered"))
        if cid in B.candidate_ids:
            want = hk.run_identity(B, cid, recipe, seed, lr, "cuda", "fp16", label=stage, overrides=None)
            bad = hk.identity_mismatches(rec, want)
            ck.add("hk_lm_identity", not bad, [list(map(str, b)) for b in bad] or None)
            c = B.cand(cid)
            mc = hk.MODELS[hk.STAGES[recipe]["model"]]
            m = rec["model"]
            ck.add("model_arch", m["name"] == hk.STAGES[recipe]["model"] and m["d"] == mc["d"] and m["L"] == mc["L"]
                   and m["H"] == mc["H"] and m["n_vocab"] == c["n_vocab"] and m["ctx_tokens"] == c["ctx_tokens"]
                   and m["tied_embeddings"] is True and m["dropout"] == 0.0,
                   {"name": m["name"], "d": m["d"], "L": m["L"], "n_vocab": m["n_vocab"], "ctx": m["ctx_tokens"]})
            ck.add("ctx_bytes_matched", rec["data"]["ctx_tokens"] == hk.ctx_tokens(rec["data"]["train_bytes_per_token"])
                   == c["ctx_tokens"], rec["data"]["ctx_tokens"])
            expected_steps = hk.steps_for_budget(epochs * TRAIN_BYTES)
            ck.add("steps_expected", rec["data"]["steps"] == expected_steps and rec["data"]["epochs"] == epochs,
                   {"steps": rec["data"]["steps"], "expected": expected_steps, "epochs": rec["data"]["epochs"]})
    o = rec["optimizer"]
    ck.add("optimizer_protocol", o["name"] == "AdamW" and o["betas"] == [0.9, 0.95] and o["eps"] == 1e-8
           and o["weight_decay"] == 0.1 and o["grad_clip"] == 1.0 and o["warmup_steps"] == 50
           and o["schedule"] == "cosine to 10% of peak")
    d = rec["data"]
    tl = rec.get("train_loss") or []
    ck.add("steps_done_eq_steps", d["steps_done"] == d["steps"] and len(tl) == d["steps"],
           {"steps": d["steps"], "steps_done": d["steps_done"], "len_train_loss": len(tl)})
    ck.add("train_loss_finite", all(v is not None and math.isfinite(v) for v in tl))
    ck.add("byte_matching_constants", d["bytes_per_step"] == 12288 and d["seqs_per_step"] == 8
           and d["train_total_bytes"] == TRAIN_BYTES and d["train_docs"] == TRAIN_DOCS)
    rel_seen = d["bytes_seen"] / d["budget_bytes"] - 1.0
    ck.add("bytes_seen_within_1pct_of_budget", abs(rel_seen) <= BYTES_SEEN_TOL,
           {"bytes_seen": d["bytes_seen"], "budget": d["budget_bytes"], "rel": rel_seen})
    rt = rec["runtime"]
    ck.add("device_dtype", rt["device"] == "cuda" and rt["amp_dtype"] == "fp16" and rt["eval_dtype"] == "fp32"
           and rt["deterministic"] == "strict" and rt["tf32"] is False and rt["flush_denormal"] is True,
           {k: rt.get(k) for k in ("device", "gpu_name", "amp_dtype", "eval_dtype", "deterministic", "tf32")})
    ck.add("matches_runtime_pin", pin is not None and rt["device"] == pin["device"] and rt["amp_dtype"] == pin["amp_dtype"]
           and rt["gpu_name"] == pin["gpu_name"], pin)
    # per-document arrays ('dev' = TEST in a final-test bundle) and bpb (PLAN 4.1)
    dv = rec.get("dev") or {}
    arrays = {k: len(dv.get(k) or []) for k in ("bits", "bytes", "ntok", "uids", "doc_index")}
    ck.add("test_doc_set_all_491", dv.get("doc_set") == "all" and dv.get("n_docs") == N_TEST, dv.get("n_docs"))
    ck.add("test_arrays_len_491", all(v == N_TEST for v in arrays.values()), arrays)
    if all(v == N_TEST for v in arrays.values()):
        bits = np.asarray(dv["bits"], dtype=np.float64)
        nb = np.asarray(dv["bytes"], dtype=np.int64)
        ck.add("test_doc_index_is_0_490", dv["doc_index"] == list(range(N_TEST)))
        ck.add("test_uids_eq_bundle", dv["uids"] == bctx["dev_uids"])
        ck.add("test_uids_eq_test_strict_jsonl", sorted(dv["uids"]) == dv["uids"] and set(dv["uids"]) == bctx["test_uids"])
        ck.add("test_bytes_eq_bundle", bool(np.array_equal(nb, bctx["dev_bytes"])) and int(nb.sum()) == TEST_BYTES
               == dv["sum_bytes"], int(nb.sum()))
        if cid in B.candidate_ids:
            ck.add("test_ntok_eq_bundle_tokens_plus_eot",
                   bool(np.array_equal(np.asarray(dv["ntok"]), bctx["dev_ntok"][cid])))
        ck.add("test_bits_finite_positive", bool(np.all(np.isfinite(bits)) and np.all(bits > 0)))
        bpb_np = float(bits.sum() / int(nb.sum()))            # exactly hk_lm.evaluate's expression
        bpb_fsum = math.fsum(dv["bits"]) / int(nb.sum())
        ck.add("bpb_eq_sum_bits_over_sum_bytes", bpb_np == rec["bpb"] == dv["bpb"] and float(bits.sum()) == dv["sum_bits"],
               {"reported": rec["bpb"], "recomputed": bpb_np, "exact": bpb_np == rec["bpb"],
                "rel_diff_fsum": (bpb_fsum - rec["bpb"]) / rec["bpb"]})
        w = dv.get("window") or {}
        ck.add("eval_window_protocol", w.get("ctx") == d["ctx_tokens"] and w.get("stride") == max(1, d["ctx_tokens"] // 2)
               and w.get("bos_given") is True and w.get("eot_predicted") is True, w)
        sub = rec.get("subset") or {}
        cur = rec.get("curve") or []
        fr = [c["frac"] for c in cur]
        sidx = np.asarray(sub.get("doc_index") or [], dtype=np.int64)
        ok_sub = sub.get("doc_set") == "subset" and sub.get("bytes") == bctx["subset_bytes"] \
            and np.array_equal(sidx, bctx["subset_idx"])
        ck.add("subset_is_bundle_0.5MB", ok_sub, {"bytes": sub.get("bytes"), "n": len(sidx)})
        exp_steps = [max(1, int(round(f * d["steps"]))) for f in (0.25, 0.5, 0.75)] + [d["steps"]]
        ok_c = fr == [0.25, 0.5, 0.75, 1.0] and [c["step"] for c in cur] == exp_steps
        if ok_c and ok_sub:
            b100 = float(np.sum(bits[sidx]) / np.sum(nb[sidx]))
            ok_c = ok_c and b100 == cur[-1]["bpb"] and all(c["sum_bytes"] == bctx["subset_bytes"] for c in cur)
        ck.add("curve_protocol", ok_c, {"fracs": fr})
    return ck


def dev_validator_same_verdict(hk, B, bctx, rel, rec, pin, mine):
    """collect_colab.validate_run itself on a confirm record (its dev constants set to the test values)."""
    saved = (CC.N_DEV, CC.DEV_BYTES, CC.SUBSET_BYTES)
    try:
        CC.N_DEV, CC.DEV_BYTES, CC.SUBSET_BYTES = N_TEST, TEST_BYTES, bctx["subset_bytes"]
        cc = CC.validate_run(hk, B, bctx, rel, rec, pin)
    finally:
        CC.N_DEV, CC.DEV_BYTES, CC.SUBSET_BYTES = saved
    rename = {"dev_doc_set_all_836": "test_doc_set_all_491", "dev_arrays_len_836": "test_arrays_len_491",
              "dev_doc_index_is_0_835": "test_doc_index_is_0_490", "dev_uids_eq_bundle": "test_uids_eq_bundle",
              "dev_bytes_eq_bundle": "test_bytes_eq_bundle",
              "dev_ntok_eq_bundle_tokens_plus_eot": "test_ntok_eq_bundle_tokens_plus_eot",
              "dev_bits_finite_positive": "test_bits_finite_positive"}
    theirs = [(rename.get(c["check"], c["check"]), c["ok"]) for c in cc.items]
    ours = [(c["check"], c["ok"]) for c in mine.items if c["check"] != "test_uids_eq_test_strict_jsonl"]
    return {"n_checks_collect_colab": len(cc.items), "failed_collect_colab": [c["check"] for c in cc.failed],
            "identical_check_list_and_verdicts": theirs == ours}


def validate_parity(hk, B, bctx, rel, rec, parity_ref):
    ck = CC.Checks()
    amp = rel.split("_")[-1].replace(".json", "")
    ck.add("status_ok", rec.get("status") == "ok")
    ck.add("bundle_sha", rec.get("bundle_manifest_sha256") == bctx["spec"]["sha"] == B.manifest_sha256)
    ck.add("hk_lm_sha", rec.get("hk_lm_sha256") == HK_LM_SHA)
    want = hk.run_identity(B, parity_ref["candidate"], "parity", 1, parity_ref["config"]["lr"], "cuda", amp,
                           label="parity_%s" % amp, overrides=None)
    bad = hk.identity_mismatches(rec, want)
    ck.add("hk_lm_identity", not bad, [list(map(str, b)) for b in bad] or None)
    dv = rec["dev"]
    n = len(bctx["parity_idx"])
    ck.add("arrays_len_parity", all(len(dv[k]) == n for k in ("bits", "bytes", "ntok", "uids", "doc_index"))
           and dv["doc_index"] == bctx["parity_idx"].tolist(), n)
    bits = np.asarray(dv["bits"], dtype=np.float64)
    nb = np.asarray(dv["bytes"], dtype=np.int64)
    bpb = float(bits.sum() / int(nb.sum()))
    ck.add("bpb_eq_sum_bits_over_sum_bytes", bpb == rec["bpb"], {"reported": rec["bpb"], "recomputed": bpb})
    rel_cpu = (rec["bpb"] - parity_ref["cpu"]["bpb"]) / parity_ref["cpu"]["bpb"]
    ck.add("within_1pct_of_cpu", abs(rel_cpu) <= parity_ref["tolerance_rel_bpb"], {"rel_diff_vs_cpu": rel_cpu})
    ck.add("device_dtype", rec["runtime"]["device"] == "cuda" and rec["runtime"]["amp_dtype"] == amp)
    return ck, rel_cpu


# ------------------------------------------------------------------ same models as dev (FINAL_TEST.md check (b))
def dev_record_index():
    """(stage, candidate, seed) -> dev record path: non-duplicate Stage 4 rows of results_table + the dev large runs."""
    rows = load_json(os.path.join(DEV_RES, "results_table.json"))
    idx = {}
    for r in rows:
        if r["stage"] == "confirm" and not r["duplicate_of"]:
            k = ("confirm", r["candidate"], int(r["seed"]))
            assert k not in idx, k
            idx[k] = {"path": os.path.join(DEV_RES, r["file"]), "file": r["file"], "file_sha256": r["file_sha256"]}
    for fn in sorted(os.listdir(os.path.join(DEV_RES, "77e1368773fc", "results"))):
        m = re.match(r"^large__(.+)__lr5e-4__s(\d+)\.json$", fn)
        if m:
            idx[("large", m.group(1), int(m.group(2)))] = {
                "path": os.path.join(DEV_RES, "77e1368773fc", "results", fn), "file": "77e1368773fc/results/" + fn,
                "file_sha256": None}
    return idx


def same_model_as_dev(rec, dev_idx):
    k = (rec["stage"], rec["candidate"], int(rec["seed"]))
    ent = dev_idx.get(k)
    if ent is None:
        return {"dev_record": None, "same_model": False}
    sha = sha256_file(ent["path"])
    dr = load_json(ent["path"])
    out = {"dev_record": ent["file"], "dev_record_sha256": sha,
           "dev_record_sha256_matches_results_table": (sha == ent["file_sha256"]) if ent["file_sha256"] else None,
           "train_loss_identical": dr["train_loss"] == rec["train_loss"],
           "r3_lowest_norm_embeddings_identical": dr["r3_lowest_norm_embeddings"] == rec["r3_lowest_norm_embeddings"],
           "data_identical": all(dr["data"][f] == rec["data"][f] for f in
                                 ("steps", "ctx_tokens", "tokens_seen", "bytes_seen", "targets_masked_bos",
                                  "stream_tokens_built", "train_bytes_per_token")),
           "model_params_identical": dr["model"]["params"] == rec["model"]["params"],
           "lr_identical": dr["lr"] == rec["lr"] and dr["optimizer"] == rec["optimizer"],
           "runtime_identical": {f: dr["runtime"].get(f) == rec["runtime"].get(f)
                                 for f in ("device", "gpu_name", "torch", "cuda", "amp_dtype", "eval_dtype",
                                           "deterministic", "tf32", "ce_chunk_rows", "eval_window_batch")},
           "dev_bundle": dr["bundle_manifest_sha256"][:12], "dev_bpb": dr["bpb"]}
    out["same_model"] = bool(out["train_loss_identical"] and out["r3_lowest_norm_embeddings_identical"]
                             and out["data_identical"] and out["model_params_identical"] and out["lr_identical"])
    return out


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--no-copy", action="store_true", help="validate the existing local copy only")
    a = ap.parse_args(argv)
    t0 = utcnow()
    hk = CC.import_hk_lm()
    if sha256_file(hk.__file__) != HK_LM_SHA:
        raise SystemExit("local colab/hk_lm.py is not the version the records were produced with")
    os.makedirs(OUT, exist_ok=True)
    log = load_json(FINAL_TEST_LOG)
    report = {"what": "one-shot FINAL TEST LM results: copy manifest + protocol validation (lm/collect_final_test.py)",
              "collected_utc": t0, "source_root": a.src, "dest_root": OUT, "hk_lm_sha256_local": sha256_file(hk.__file__),
              "run_all_local_sha256": sha256_file(os.path.join(COLAB, "run_all.py")),
              "test_jsonl": {"path": TEST_JSONL, "sha256": sha256_file(TEST_JSONL)},
              "split_manifest": {"path": SPLIT_MANIFEST, "sha256": sha256_file(SPLIT_MANIFEST)},
              "bundles": {}, "cross_bundle": {}, "other_source_folders": {}}
    assert report["test_jsonl"]["sha256"] == TEST_JSONL_SHA
    assert report["split_manifest"]["sha256"] == SPLIT_SHA
    if os.path.isdir(a.src):
        for name in sorted(os.listdir(a.src)):
            if name not in BUNDLES:
                report["other_source_folders"][name] = "not part of this collection"

    tmeta = test_docs_meta()
    test_uids = set(tmeta)
    assert len(test_uids) == N_TEST
    dev_idx = dev_record_index()

    rows = []
    recs_by = {}
    parity_by = {}
    for bid, spec in BUNDLES.items():
        print("== bundle %s" % bid, flush=True)
        copied, excluded = copy_bundle(a.src, bid, OUT, not a.no_copy)
        dst = os.path.join(OUT, bid)
        B = hk.Bundle(spec["staging"])
        n_verified = B.verify()
        waves = load_json(spec["waves"])
        wmeta = {w["id"]: w for w in waves["waves"]}
        dev_docs = B.dev_docs()
        dtoks = {}
        for cid in B.candidate_ids:
            _, offs = B.tokens(cid, "dev")
            dtoks[cid] = np.diff(offs.astype(np.int64)) + 1
        sidx = B.eval_doc_index("subset")
        bctx = {"bid": bid, "spec": spec, "waves_ids": set(wmeta), "dev_uids": [x["uid"] for x in dev_docs],
                "dev_bytes": B.dev_bytes().astype(np.int64), "dev_ntok": dtoks, "subset_idx": sidx,
                "subset_bytes": int(B.dev_bytes().astype(np.int64)[sidx].sum()),
                "parity_idx": B.eval_doc_index("parity"), "test_uids": test_uids}
        # the bundle's 'dev' docs are exactly test_strict (uids, clusters, sources)
        docs_ok = (len(dev_docs) == N_TEST and set(x["uid"] for x in dev_docs) == test_uids
                   and all(x["cluster"] == tmeta[x["uid"]]["cluster"] and x["source"] == tmeta[x["uid"]]["source"]
                           and x["group"] == tmeta[x["uid"]]["group"] for x in dev_docs)
                   and [x["bytes"] for x in dev_docs] == bctx["dev_bytes"].tolist())
        pin_p = os.path.join(dst, "runtime_pin.json")
        pin = load_json(pin_p) if os.path.isfile(pin_p) else None
        parity_ref = load_json(os.path.join(spec["staging"], "PARITY.json"))
        le = log["entries"][spec["log_entry"]]
        ok_manifest = (B.manifest_sha256 == spec["sha"] and sha256_file(os.path.join(spec["staging"], "hk_lm.py")) == HK_LM_SHA
                       and B.m["data"]["split_manifest_sha256"] == SPLIT_SHA and B.m.get("kind") == "final_test"
                       and B.m["data"]["dev_sha256"] == TEST_JSONL_SHA and B.m["data"]["n_dev_docs"] == N_TEST
                       and B.m["data"]["dev_bytes"] == TEST_BYTES
                       and le["bundle_manifest_sha256"] == spec["sha"] and le["status"] == "completed"
                       and sorted(c["id"] for c in le["candidates"]) == sorted(B.candidate_ids)
                       and sorted(B.candidate_ids) == sorted(wmeta))
        brep = {"bundle_manifest_sha256": spec["sha"], "role": spec["role"], "staging_manifest_sha256": B.manifest_sha256,
                "staging_files_verified": n_verified, "kind": B.m.get("kind"),
                "final_test_log_entry": spec["log_entry"], "final_test_log_utc": le["utc"],
                "final_test_log_completed_utc": le.get("completed_utc"),
                "baseline": B.baseline_id, "candidates": B.candidate_ids, "runtime_pin": pin,
                "bundle_identity_ok": bool(ok_manifest), "bundle_test_docs_equal_test_strict": bool(docs_ok),
                "subset": {"n_docs": int(len(sidx)), "bytes": bctx["subset_bytes"]},
                "copied": copied, "excluded": excluded,
                "progress_log": progress_trace(os.path.join(dst, "progress.log")),
                "records": {}, "parity": {}, "stage_checks": {}, "same_model_as_dev": {},
                "collect_colab_validator_agreement": {}}
        for amp in ("fp16", "fp32"):
            rel = "parity/parity_cuda_%s.json" % amp
            p = os.path.join(dst, rel)
            if os.path.isfile(p):
                rec = load_json(p)
                ck, rel_cpu = validate_parity(hk, B, bctx, rel, rec, parity_ref)
                parity_by[(bid, amp)] = rec
                brep["parity"][rel] = {"bpb": rec["bpb"], "cpu_bpb": parity_ref["cpu"]["bpb"], "rel_diff_vs_cpu": rel_cpu,
                                       "n_checks": len(ck.items), "failed": ck.failed}
        pr = load_json(os.path.join(dst, "parity_report.json"))
        brep["parity_report_pass_all"] = pr.get("pass_all")
        res_dir = os.path.join(dst, "results")
        for fn in sorted(os.listdir(res_dir)):
            if not fn.endswith(".json"):
                continue
            rel = "results/" + fn
            rec = load_json(os.path.join(res_dir, fn))
            ck = validate_run_ft(hk, B, bctx, rel, rec, pin)
            if rec["stage"] == "confirm":
                brep["collect_colab_validator_agreement"][rel] = dev_validator_same_verdict(hk, B, bctx, rel, rec, pin, ck)
            k = (bid, rec["stage"], rec["candidate"], rec["seed"])
            if k in recs_by:
                ck.add("unique_key", False, "duplicate record for %s" % (k,))
            smd = same_model_as_dev(rec, dev_idx)
            ck.add("same_model_as_dev_record", smd["same_model"], smd.get("dev_record"))
            brep["same_model_as_dev"][rel] = smd
            recs_by[k] = rec
            brep["records"][rel] = {"n_checks": len(ck.items), "failed": ck.failed}
            rows.append(make_row(bid, B, wmeta, rel, rec, ck, os.path.join(dst, rel), smd))
        brep["stage_checks"] = stage_checks(hk, bid, B, recs_by, dst)
        report["bundles"][bid] = brep
        nfail = sum(1 for r in brep["records"].values() if r["failed"])
        print("   copied %d files (%d excluded); %d result records, %d with failed checks; staging files verified: %d"
              % (len(copied), len(excluded), len(brep["records"]), nfail, n_verified), flush=True)

    report["cross_bundle"] = cross_bundle(recs_by, parity_by, rows)
    report["test_clusters"], cl_rows = test_clusters(recs_by, tmeta)
    write_table(rows)
    with open(os.path.join(OUT, "test_clusters.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cl_rows[0].keys()))
        w.writeheader()
        w.writerows(cl_rows)
    report["summary"] = summarize(report, rows)
    report["finished_utc"] = utcnow()
    with open(os.path.join(OUT, "validation_final_test.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(report["summary"], indent=1, default=str))


def progress_trace(p):
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    heads = [(i, ln) for i, ln in enumerate(lines) if "===== stage" in ln]
    out = {"n_lines": len(lines), "stage_headers": [ln.strip() for _, ln in heads],
           "finished_lines": [ln.strip() for ln in lines if " finished in " in ln],
           "warnings": [ln.strip() for ln in lines if "WARNING" in ln],
           "last_line": lines[-1].strip() if lines else None}
    large = [i for i, ln in heads if "stage large" in ln]
    if large:
        tail = lines[large[-1]:]
        out["large"] = {"header": lines[large[-1]].strip(),
                        "runs_done": [ln.strip() for ln in tail if " DONE " in ln],
                        "last_step_line": next((ln.strip() for ln in reversed(tail) if " step " in ln), None),
                        "finished_line_after_header": any(" finished in " in ln for ln in tail)}
    return out


def make_row(bid, B, wmeta, rel, rec, ck, path, smd):
    c = B.cand(rec["candidate"]) if rec["candidate"] in B.candidate_ids else {}
    w = wmeta.get(rec["candidate"], {})
    d, dv, rt = rec["data"], rec.get("dev") or {}, rec["runtime"]
    curve = {c_["frac"]: c_["bpb"] for c_ in rec.get("curve") or []}
    bits = np.asarray(dv.get("bits") or [np.nan], dtype=np.float64)
    nb = np.asarray(dv.get("bytes") or [1], dtype=np.int64)
    return collections.OrderedDict([
        ("bundle", bid), ("stage", rec["stage"]), ("stage_recipe", rec["stage_recipe"]),
        ("preregistered_flag", rec.get("preregistered")), ("candidate", rec["candidate"]), ("seed", rec["seed"]),
        ("lr", rec["lr"]), ("final_test_role", w.get("final_test_role")), ("is_baseline", rec["candidate"] == B.baseline_id),
        ("duplicate_of", ""), ("encoder_kind", c.get("kind")), ("n_vocab", rec["model"]["n_vocab"]),
        ("tokenizer_sha256", c.get("tokenizer_sha256")), ("train_bytes_per_token", d["train_bytes_per_token"]),
        ("test_bytes_per_token", c.get("dev_bytes_per_token")), ("model", rec["model"]["name"]),
        ("d_model", rec["model"]["d"]), ("n_layer", rec["model"]["L"]), ("ctx_tokens", d["ctx_tokens"]),
        ("params_total", rec["model"]["params"]["total"]), ("params_non_embedding", rec["model"]["params"]["non_embedding"]),
        ("steps", d["steps"]), ("steps_done", d["steps_done"]), ("epochs", d["epochs"]), ("budget_bytes", d["budget_bytes"]),
        ("bytes_seen", d["bytes_seen"]), ("bytes_seen_rel_budget", d["bytes_seen"] / d["budget_bytes"] - 1.0),
        ("tokens_seen", d["tokens_seen"]), ("test_n_docs", dv.get("n_docs")), ("test_sum_bits", dv.get("sum_bits")),
        ("test_sum_bytes", dv.get("sum_bytes")), ("test_sum_ntok_scored", int(np.sum(dv.get("ntok") or [0]))),
        ("bpb", rec["bpb"]), ("bpb_recomputed", float(bits.sum() / int(nb.sum()))),
        ("subset_bpb_25", curve.get(0.25)), ("subset_bpb_50", curve.get(0.5)), ("subset_bpb_75", curve.get(0.75)),
        ("subset_bpb_100", curve.get(1.0)), ("status", rec["status"]),
        ("device", rt["device"]), ("gpu_name", rt["gpu_name"]), ("amp_dtype", rt["amp_dtype"]),
        ("eval_dtype", rt["eval_dtype"]), ("torch", rt["torch"]), ("train_s", rec["wall_time_s"]["train"]),
        ("eval_s", rec["wall_time_s"]["eval"]), ("created_utc", rec["created_utc"]),
        ("bundle_manifest_sha256", rec["bundle_manifest_sha256"]), ("hk_lm_sha256", rec["hk_lm_sha256"]),
        ("same_model_as_dev", smd["same_model"]), ("dev_record", smd.get("dev_record")), ("dev_bpb", smd.get("dev_bpb")),
        ("file", "final_test/" + bid + "/" + rel), ("file_sha256", sha256_file(path)),
        ("n_checks", len(ck.items)), ("valid", not ck.failed),
        ("hard_valid", not [x for x in ck.failed if x["check"] not in TOLERANCE_CHECKS]),
        ("tolerance_flags", ";".join(x["check"] for x in ck.failed if x["check"] in TOLERANCE_CHECKS)),
        ("failed_checks", ";".join(x["check"] for x in ck.failed)),
    ])


def stage_checks(hk, bid, B, recs_by, dst):
    out = {}
    mine = {k: v for k, v in recs_by.items() if k[0] == bid}
    by_stage = collections.defaultdict(list)
    for k, r in mine.items():
        by_stage[k[1]].append(r)
    cands = B.candidate_ids
    for st in STAGE_ORDER:
        rs = by_stage.get(st, [])
        if not rs:
            out[st] = {"present": False}
            continue
        steps = sorted({r["data"]["steps"] for r in rs})
        steps_done = sorted({r["data"]["steps_done"] for r in rs})
        seen = [r["data"]["bytes_seen"] for r in rs]
        rel_b = [r["data"]["bytes_seen"] / r["data"]["budget_bytes"] - 1 for r in rs]
        nonemb = sorted({r["model"]["params"]["non_embedding"] for r in rs})
        keys = sorted((r["candidate"], r["seed"]) for r in rs)
        if st == "confirm":
            expected = sorted((c, s) for c in cands for s in (1, 2, 3, 4, 5))
        else:
            expected = sorted((c, s) for c in EXPECTED_LARGE for s in (1, 2))
        missing = [list(map(str, k)) for k in expected if k not in keys]
        unexpected = [list(map(str, k)) for k in keys if k not in expected]
        per_seed = collections.defaultdict(list)
        for r in rs:
            per_seed[r["seed"]].append(r["data"]["bytes_seen"])
        same_seed = {"s%d" % s: max(v) / min(v) - 1 for s, v in sorted(per_seed.items()) if len(v) > 1}
        out[st] = {"present": True, "n_records": len(rs), "n_candidates": len({r["candidate"] for r in rs}),
                   "lrs": sorted({r["lr"] for r in rs}), "steps": steps, "steps_done": steps_done,
                   "steps_identical_across_candidates": len(steps) == 1 and steps == steps_done,
                   "bytes_seen_max_abs_rel_vs_budget": max(abs(x) for x in rel_b),
                   "bytes_seen_n_outside_1pct_of_budget": sum(1 for x in rel_b if abs(x) > BYTES_SEEN_TOL),
                   "bytes_seen_same_seed_spread_max_rel": max(same_seed.values()) if same_seed else 0.0,
                   "non_embedding_params": nonemb, "non_embedding_identical": len(nonemb) == 1,
                   "complete": not missing and not unexpected, "missing": missing, "unexpected": unexpected}
    p = os.path.join(dst, "power_check.json")
    if os.path.isfile(p):
        pc = load_json(p)
        per = {}
        for c in cands:
            xs = [mine[(bid, "confirm", c, s)]["bpb"] for s in (1, 2, 3)]
            m, sd = CC.mean_sd(xs)
            per[c] = sd / m
        pooled = math.sqrt(sum(v ** 2 for v in per.values()) / len(per))
        out["power_check_recomputed"] = {"file_pooled_rel_sd": pc["pooled_rel_sd"], "recomputed_pooled_rel_sd": pooled,
                                         "agree": abs(pooled - pc["pooled_rel_sd"]) < 1e-15,
                                         "seeds_needed_0.5pct": math.ceil(15.7 * pooled ** 2 / 0.005 ** 2),
                                         "file_seeds_needed": pc["seeds_needed_for_delta_0.5pct"],
                                         "extra_seeds_forced": pc.get("extra_seeds_forced"),
                                         "extra_seeds_run": pc["extra_seeds_run"]}
    p = os.path.join(dst, "summary.json")
    if os.path.isfile(p):
        sj = load_json(p)
        agree, n = True, 0
        for key, srows in sj.get("stages", {}).items():
            st, lrt = key.split("@lr")
            for x in srows:
                xs = [r["bpb"] for (b_, s_, c_, sd_), r in mine.items()
                      if s_ == st and c_ == x["candidate"] and hk.lr_tag(r["lr"]) == lrt]
                m, _ = CC.mean_sd(xs)
                n += 1
                agree = agree and m is not None and abs(m - x["mean_bpb"]) <= 1e-15 and len(xs) == x["n_ok"]
        out["summary_json_crosscheck"] = {"n_rows": n, "means_agree": agree, "stage_keys": list(sj.get("stages", {}))}
    return out


def cross_bundle(recs_by, parity_by, rows):
    b1, b2 = list(BUNDLES)
    c1 = {k[2] for k in recs_by if k[0] == b1}
    c2 = {k[2] for k in recs_by if k[0] == b2}
    refs = sorted(c1 & c2)
    pairs = []
    for k, r2 in sorted(recs_by.items(), key=lambda kv: str(kv[0])):
        if k[0] != b2 or k[2] not in refs:
            continue
        r1 = recs_by.get((b1,) + k[1:])
        if r1 is None:
            pairs.append({"stage": k[1], "candidate": k[2], "seed": k[3], "bundle1_record": None})
            continue
        a = np.asarray(r1["dev"]["bits"])
        b = np.asarray(r2["dev"]["bits"])
        pairs.append({"stage": k[1], "candidate": k[2], "seed": k[3],
                      "bits_identical": r1["dev"]["bits"] == r2["dev"]["bits"],
                      "bytes_identical": r1["dev"]["bytes"] == r2["dev"]["bytes"],
                      "uids_identical": r1["dev"]["uids"] == r2["dev"]["uids"],
                      "n_docs_differing": int(np.sum(a != b)), "max_abs_diff_bits": float(np.max(np.abs(a - b))),
                      "ntok_identical": r1["dev"]["ntok"] == r2["dev"]["ntok"],
                      "train_loss_identical": r1["train_loss"] == r2["train_loss"],
                      "curve_identical": r1["curve"] == r2["curve"],
                      "r3_identical": r1["r3_lowest_norm_embeddings"] == r2["r3_lowest_norm_embeddings"],
                      "bpb": r1["bpb"], "bpb_identical": r1["bpb"] == r2["bpb"]})
    par = {}
    for amp in ("fp16", "fp32"):
        p1, p2 = parity_by.get((b1, amp)), parity_by.get((b2, amp))
        if p1 and p2:
            par[amp] = {"bits_identical": p1["dev"]["bits"] == p2["dev"]["bits"], "bpb": [p1["bpb"], p2["bpb"]]}
    m1 = load_json(os.path.join(BUNDLES[b1]["staging"], "manifest.json"))
    m2 = load_json(os.path.join(BUNDLES[b2]["staging"], "manifest.json"))
    shared = {}
    for name, info in m1["files"].items():
        if "/" not in name or name.split("/")[0] in refs:
            if name in m2["files"]:
                shared[name] = info["sha256"] == m2["files"][name]["sha256"]
    ok = [p for p in pairs if p.get("bits_identical")]
    ident = {(p["stage"], p["candidate"], p["seed"]): p.get("bits_identical") for p in pairs}
    for r in rows:
        if r["bundle"] == b2 and ident.get((r["stage"], r["candidate"], r["seed"])):
            r["duplicate_of"] = b1
    return {"reference_arms": refs, "n_pairs": len(pairs), "n_bitwise_identical": len(ok),
            "all_identical": len(ok) == len(pairs) and len(pairs) > 0 and all(
                p["train_loss_identical"] and p["curve_identical"] and p["ntok_identical"] and p["r3_identical"]
                and p["bytes_identical"] and p["uids_identical"] for p in pairs),
            "summary": "%d/%d run pairs with bitwise identical per-document bits" % (len(ok), len(pairs)),
            "pairs": pairs, "parity_across_bundles": par, "input_arrays_identical_in_both_manifests": shared}


def test_clusters(recs_by, tmeta):
    any_rec = recs_by[(list(BUNDLES)[0], "confirm", BASELINE, 1)]
    uids, nbytes = any_rec["dev"]["uids"], any_rec["dev"]["bytes"]
    sm = split_manifest_rows(uids)
    out_rows = []
    agree_manifest = True
    for i, (u, b) in enumerate(zip(uids, nbytes)):
        x = tmeta[u]
        s = sm.get(u, {})
        g = s.get("group")
        m = re.match(r"^web:([^:]+):record:", g or "")
        rule_cluster = ("web:" + m.group(1)) if m else g
        agree_manifest = agree_manifest and s.get("split") == "test" and s.get("quality_tier") == "strict" \
            and g == x["group"] and rule_cluster == x["cluster"] and s.get("source") == x["source"]
        out_rows.append(collections.OrderedDict([("doc_index", i), ("uid", u), ("bytes", b), ("source", x["source"]),
                                                 ("group", x["group"]), ("cluster", x["cluster"])]))
    by_src = collections.Counter(r["source"] for r in {(r["source"], r["cluster"]): r for r in out_rows}.values())
    by_src_raw = collections.Counter(r["source"] for r in {(r["source"], r["group"]): r for r in out_rows}.values())
    cl_bytes = collections.Counter()
    for r in out_rows:
        cl_bytes[r["cluster"]] += r["bytes"]
    big = cl_bytes.most_common(1)[0]
    rep = {"n_docs": len(uids), "n_bytes": int(sum(nbytes)), "n_found_in_split_manifest": len(sm),
           "cluster_field_equals_manifest_group_rule": bool(agree_manifest),
           "rule": "cluster = manifest 'group'; 'web:<site>:record:<uid>' -> 'web:<site>' (as on dev)",
           "n_raw_groups": len({r["group"] for r in out_rows}), "raw_groups_by_source": dict(by_src_raw),
           "n_clusters": len({r["cluster"] for r in out_rows}), "clusters_by_source": dict(by_src),
           "plan6_expects": "27 clusters (15 newspaper, 6 book, 6 web)",
           "largest_cluster": {"cluster": big[0], "bytes": big[1], "share": big[1] / TEST_BYTES},
           "all_records_same_uid_order": all(r["dev"]["uids"] == uids for r in recs_by.values())}
    return rep, out_rows


def write_table(rows):
    cols = list(rows[0].keys())
    rows.sort(key=lambda r: (list(BUNDLES).index(r["bundle"]), STAGE_ORDER.index(r["stage"]), r["candidate"], r["seed"]))
    with open(os.path.join(OUT, "results_table_final_test.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(OUT, "results_table_final_test.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)


def summarize(rep, rows):
    agree = [v for b in rep["bundles"].values() for v in b["collect_colab_validator_agreement"].values()]
    smd = [v for b in rep["bundles"].values() for v in b["same_model_as_dev"].values()]
    return {"n_records": len(rows), "n_hard_valid": sum(1 for r in rows if r["hard_valid"]),
            "n_valid_all_checks": sum(1 for r in rows if r["valid"]),
            "n_checks_total": sum(r["n_checks"] for r in rows),
            "failed": {r["file"]: r["failed_checks"] for r in rows if r["failed_checks"]},
            "by_stage": dict(collections.Counter("%s/%s" % (r["bundle"], r["stage"]) for r in rows)),
            "bpb_bit_exact_all": all(r["bpb"] == r["bpb_recomputed"] for r in rows),
            "same_model_as_dev_all": all(v["same_model"] for v in smd), "n_same_model_checked": len(smd),
            "collect_colab_validator_identical_on_all_confirm_records": all(v["identical_check_list_and_verdicts"]
                                                                            for v in agree),
            "n_confirm_records_cross_validated": len(agree),
            "bundle_identity_ok": {b: v["bundle_identity_ok"] for b, v in rep["bundles"].items()},
            "bundle_test_docs_equal_test_strict": {b: v["bundle_test_docs_equal_test_strict"] for b, v in rep["bundles"].items()},
            "parity_pass": {b: all(not p["failed"] for p in v["parity"].values()) for b, v in rep["bundles"].items()},
            "copied_all_sha_verified": {b: all(x.get("dest_sha256_verified") for x in v["copied"]) for b, v in rep["bundles"].items()},
            "cross_bundle": rep["cross_bundle"]["summary"], "cross_bundle_all_identical": rep["cross_bundle"]["all_identical"],
            "test_clusters": {k: rep["test_clusters"][k] for k in ("n_clusters", "clusters_by_source",
                                                                   "cluster_field_equals_manifest_group_rule")},
            "stage_complete": {"%s/%s" % (b, st): v["stage_checks"][st].get("complete")
                               for b, v in rep["bundles"].items() for st in STAGE_ORDER
                               if v["stage_checks"].get(st, {}).get("present")}}


if __name__ == "__main__":
    main()
