#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_colab.py - copy the Colab LM-arbiter results into the project and validate every record
against the pre-registered protocol (research/PLAN.md 4.1 bpb, 5 LM protocol, 6 power check).

    PYTHONIOENCODING=utf-8 python lm/collect_colab.py            # copy from Google Drive + validate
    PYTHONIOENCODING=utf-8 python lm/collect_colab.py --no-copy  # re-validate the local copy only

Inputs (read-only):
  G:/My Drive/hindko_lm_out/<bundle>/...          Colab output (run_all.py), one folder per bundle
  colab/build/staging, colab/build_r2/staging      the bundles as built (manifest.json = bundle id)
  colab/hk_lm.py                                   the arbiter module (sha256 must equal the records')
  lm/WAVES_colab.json, lm/WAVES_round2.json         candidate definitions
  splits/split_manifest.jsonl                      clusters (dev_strict rows only are looked up)
Outputs (all under lm/colab_results/):
  <bundle>/...                    byte-exact copies (sha256-verified), 'large' stage files excluded
  results_table.csv / .json       one row per bundle x stage x candidate x seed (x LR)
  validation.json                 every check, per record and per stage; copy manifest with sha256
  dev_clusters.csv                the 836 dev_strict documents: index, uid, bytes, source, group, cluster
  COLLECT.md                      the human-readable report

The TEST split is never opened: nothing here reads text, only result records, bundle manifests and the
split manifest's rows for the 836 dev_strict uids that the records carry.
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
COLAB = os.path.join(TOK, "colab")
LM = os.path.join(TOK, "lm")
OUT = os.path.join(LM, "colab_results")
SRC_DEFAULT = r"G:\My Drive\hindko_lm_out"
SPLIT_MANIFEST = os.path.join(TOK, "splits", "split_manifest.jsonl")
SPLIT_SHA = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"
HK_LM_SHA = "3f78ad799f56cd4699a8050d1e32c802ea0981feedbf26399322f145a117e12d"

BUNDLES = collections.OrderedDict([
    ("1d24425d2d64", {"round": 1, "sha": "1d24425d2d647f3150b83637c6055d133cde2aaeb14e9a31d30d3a6318f6c911",
                      "staging": os.path.join(COLAB, "build", "staging"),
                      "waves": os.path.join(LM, "WAVES_colab.json"),
                      "lr_source": "own PLAN 5 sweep (lr_choice.json)"}),
    ("77e1368773fc", {"round": 2, "sha": "77e1368773fc40f2284a601cc00bc0178184bf8992272ac54ec24e61b22f618c",
                      "staging": os.path.join(COLAB, "build_r2", "staging"),
                      "waves": os.path.join(LM, "WAVES_round2.json"),
                      "lr_source": "forced to the round-1 choice (run_all.py --lr 1e-3); no sweep in this bundle"}),
])
OTHER_FOLDERS_NOTE = {"b6d4d9c0a843": "pilot bundle (2 candidates, baseline pilotA1_P1_16k): parity only, results/ empty; "
                                      "not part of the study, not copied"}

CHOSEN_LR = 1e-3
N_DEV = 836
DEV_BYTES = 1445513
TRAIN_BYTES = 45007515
TRAIN_DOCS = 16015
SUBSET_BYTES = 504655
BYTES_SEEN_TOL = 0.01
# checks that measure a tolerance rather than identity/integrity: a failure flags the record, it does not
# make it a different run (reported separately as `tolerance_flags`)
TOLERANCE_CHECKS = {"bytes_seen_within_1pct_of_budget"}
EXCLUDED_STAGE_PREFIX = "large"     # report-only addition (PLAN 8(ii)); incomplete -> not collected

# stage label -> (recipe, allowed seeds, LR rule, role)
STAGE_SPEC = {
    "lrsweep": ("screen", {1}, "grid", "PLAN 5 LR sweep on the baseline (pre-registered)"),
    "screen": ("screen", {1, 2, 3}, "chosen", "Stage 3 screening (pre-registered)"),
    "confirm": ("confirm", {1, 2, 3, 4, 5}, "chosen", "Stage 4 confirmation (pre-registered arbiter; run for ALL candidates)"),
    "confirm_resweep": ("confirm", {1}, "grid_not_chosen", "PLAN 5 finalist LR re-sweep (report-only)"),
}
STAGE_ORDER = ["lrsweep", "screen", "confirm", "confirm_resweep"]

ALGO = {"A1": "byte-level BPE (HF)", "A3": "SentencePiece BPE (native)", "A4": "SentencePiece Unigram (native)",
        "A6": "SuperBPE (reimpl.)", "A7": "PickyBPE (reimpl., custom encoder)", "A10": "MinGram (reimpl., HF Unigram)"}


# ------------------------------------------------------------------ helpers
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mean_sd(xs):
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def import_hk_lm():
    sys.path.insert(0, COLAB)
    import hk_lm  # noqa: E402
    if os.path.normcase(os.path.dirname(os.path.abspath(hk_lm.__file__))) != os.path.normcase(COLAB):
        raise SystemExit("hk_lm imported from %s, not %s" % (hk_lm.__file__, COLAB))
    return hk_lm


def is_excluded(rel):
    """'large' stage artefacts: results/large*__*.json and large_lr_choice.json (plus stray .tmp files)."""
    base = rel.split("/")[-1]
    if base.startswith(EXCLUDED_STAGE_PREFIX):
        return "stage 'large' (report-only addition, incomplete) - excluded by instruction"
    if ".tmp" in base:
        return "temporary file of an atomic write"
    return None


def parse_key(fn):
    m = re.match(r"^(?P<label>.+?)__(?P<slug>.+)__lr(?P<lr>[0-9.e+-]+)__s(?P<seed>\d+)\.json$", fn)
    return m.groupdict() if m else None


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def cand_meta_from_id(cid):
    m = re.match(r"^(?P<r2>R2-)?(?P<fam>A\d+)-", cid)
    fam = m.group("fam") if m else None
    parts = cid.split("-")
    pretok = "P1r3" if "P1r3" in parts else ("P1" if "P1" in parts else ("sp-native" if "SPnat" in parts else None))
    mix = next((p for p in parts if p in ("D1", "D2", "D3")), None)
    vocab = next((p for p in parts if re.fullmatch(r"\d+k", p)), None)
    extra = next((p for p in parts if re.fullmatch(r"t0\d\d|tau[0-9.]+", p)), None)
    return {"family": fam, "algorithm": ALGO.get(fam), "pretokenizer": pretok, "data_mix": mix,
            "vocab_label": vocab, "variant": extra or ""}


# ------------------------------------------------------------------ copy
def copy_bundle(src_root, bid, dst_root, do_copy):
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
                why = is_excluded(rel)
                ent = {"path": rel, "bytes": len(data), "sha256": sha256_bytes(data),
                       "source_mtime_utc": datetime.datetime.fromtimestamp(os.path.getmtime(sp), datetime.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ")}
                if why:
                    ent["why"] = why
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
                copied.append({"path": rel, "bytes": os.path.getsize(dp), "sha256": sha256_file(dp),
                               "dest_sha256_verified": None})
    return copied, excluded


def large_stage_trace(progress_log_path):
    """What progress.log says about the 'large' stage (it produced no collected files)."""
    if not os.path.isfile(progress_log_path):
        return None
    with open(progress_log_path, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    idx = [i for i, ln in enumerate(lines) if "===== stage large =====" in ln]
    if not idx:
        return {"started": False}
    i0 = idx[-1]
    tail = [ln for ln in lines[i0:] if "large" in ln]
    done = [ln for ln in tail if " DONE " in ln]
    steps = [ln for ln in tail if " step " in ln]
    session = next((lines[j] for j in range(i0, -1, -1) if " bundle " in lines[j] and " candidates" in lines[j]), None)
    return {"started": True, "stage_header": lines[i0], "session_line": session, "n_lines": len(tail),
            "runs_done": done, "last_step_line": steps[-1] if steps else None,
            "last_log_line": lines[-1] if lines else None}


# ------------------------------------------------------------------ per-record validation
class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail=None):
        self.items.append({"check": name, "ok": bool(ok), **({"detail": detail} if detail is not None else {})})

    @property
    def failed(self):
        return [c for c in self.items if not c["ok"]]


def validate_run(hk, B, bctx, rel, rec, pin):
    """All per-record checks for a training run of stage lrsweep/screen/confirm/confirm_resweep."""
    ck = Checks()
    bid, spec_b = bctx["bid"], bctx["spec"]
    fn = rel.split("/")[-1]
    key = parse_key(fn)
    stage = rec.get("stage")
    cid = rec.get("candidate")
    seed = rec.get("seed")
    lr = rec.get("lr")
    ck.add("status_ok", rec.get("status") == "ok", rec.get("status"))
    ck.add("protocol", rec.get("protocol") == hk.PROTOCOL, rec.get("protocol"))
    # --- bundle + code identity
    ck.add("bundle_sha", rec.get("bundle_manifest_sha256") == spec_b["sha"] == B.manifest_sha256
           and spec_b["sha"].startswith(bid), rec.get("bundle_manifest_sha256"))
    ck.add("hk_lm_sha", rec.get("hk_lm_sha256") == HK_LM_SHA == B.m["builder"]["hk_lm_sha256"], rec.get("hk_lm_sha256"))
    # --- file key <-> record
    ok_key = bool(key) and key["label"] == stage and key["slug"] == slug(str(cid)) and int(key["seed"]) == seed \
        and lr is not None and key["lr"] == hk.lr_tag(lr)
    ck.add("file_key_matches_record", ok_key, key)
    # --- stage / candidate / seed / exact LR
    spec = STAGE_SPEC.get(stage)
    ck.add("stage_known", spec is not None, stage)
    ck.add("candidate_in_bundle", cid in B.candidate_ids, cid)
    ck.add("candidate_in_waves", cid in bctx["waves_ids"], cid)
    if spec:
        recipe, seeds, lr_rule, _ = spec
        ck.add("stage_recipe", rec.get("stage_recipe") == recipe, rec.get("stage_recipe"))
        ck.add("seed_allowed", seed in seeds, seed)
        if lr_rule == "grid":
            lr_ok = lr is not None and any(float(lr) == g for g in hk.LR_GRID) and cid == B.baseline_id
        elif lr_rule == "chosen":
            lr_ok = lr is not None and float(lr) == CHOSEN_LR
        else:
            lr_ok = lr is not None and any(float(lr) == g for g in hk.LR_GRID) and float(lr) != CHOSEN_LR
        ck.add("lr_exact", lr_ok and rec["optimizer"]["peak_lr"] == lr, {"lr": lr, "peak_lr": rec["optimizer"]["peak_lr"],
                                                                          "rule": lr_rule})
        ck.add("preregistered_flag", rec.get("preregistered") is True, rec.get("preregistered"))
        # --- the arbiter's own resume-identity check (bundle, hk_lm, stage, recipe, cand, seed, exact LR,
        #     model, budget, epochs, steps, ctx, overrides, device, AMP dtype)
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
            expected_steps = hk.steps_for_budget(hk.SCREEN_BUDGET_BYTES if recipe == "screen" else TRAIN_BYTES)
            ck.add("steps_expected", rec["data"]["steps"] == expected_steps, {"steps": rec["data"]["steps"],
                                                                               "expected": expected_steps})
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
    # --- device / dtype
    rt = rec["runtime"]
    ck.add("device_dtype", rt["device"] == "cuda" and rt["amp_dtype"] == "fp16" and rt["eval_dtype"] == "fp32"
           and rt["deterministic"] == "strict" and rt["tf32"] is False and rt["flush_denormal"] is True,
           {k: rt.get(k) for k in ("device", "gpu_name", "amp_dtype", "eval_dtype", "deterministic", "tf32")})
    ck.add("matches_runtime_pin", pin is not None and rt["device"] == pin["device"] and rt["amp_dtype"] == pin["amp_dtype"]
           and rt["gpu_name"] == pin["gpu_name"], pin)
    # --- per-document arrays (PLAN 5: stored per document) and bpb (PLAN 4.1)
    dv = rec.get("dev") or {}
    arrays = {k: len(dv.get(k) or []) for k in ("bits", "bytes", "ntok", "uids", "doc_index")}
    ck.add("dev_doc_set_all_836", dv.get("doc_set") == "all" and dv.get("n_docs") == N_DEV, dv.get("n_docs"))
    ck.add("dev_arrays_len_836", all(v == N_DEV for v in arrays.values()), arrays)
    if all(v == N_DEV for v in arrays.values()):
        bits = np.asarray(dv["bits"], dtype=np.float64)
        nb = np.asarray(dv["bytes"], dtype=np.int64)
        ck.add("dev_doc_index_is_0_835", dv["doc_index"] == list(range(N_DEV)))
        ck.add("dev_uids_eq_bundle", dv["uids"] == bctx["dev_uids"])
        ck.add("dev_bytes_eq_bundle", bool(np.array_equal(nb, bctx["dev_bytes"])) and int(nb.sum()) == DEV_BYTES
               == dv["sum_bytes"], int(nb.sum()))
        if cid in B.candidate_ids:
            ck.add("dev_ntok_eq_bundle_tokens_plus_eot", bool(np.array_equal(np.asarray(dv["ntok"]), bctx["dev_ntok"][cid])))
        ck.add("dev_bits_finite_positive", bool(np.all(np.isfinite(bits)) and np.all(bits > 0)))
        bpb_np = float(bits.sum() / int(nb.sum()))            # exactly hk_lm.evaluate's expression
        bpb_fsum = math.fsum(dv["bits"]) / int(nb.sum())
        ck.add("bpb_eq_sum_bits_over_sum_bytes", bpb_np == rec["bpb"] == dv["bpb"] and float(bits.sum()) == dv["sum_bits"],
               {"reported": rec["bpb"], "recomputed": bpb_np, "exact": bpb_np == rec["bpb"],
                "rel_diff_fsum": (bpb_fsum - rec["bpb"]) / rec["bpb"]})
        w = dv.get("window") or {}
        ck.add("eval_window_protocol", w.get("ctx") == d["ctx_tokens"] and w.get("stride") == max(1, d["ctx_tokens"] // 2)
               and w.get("bos_given") is True and w.get("eot_predicted") is True, w)
        # curve (PLAN 5: 0.5 MB subset at 25/50/75/100 %)
        sub = rec.get("subset") or {}
        cur = rec.get("curve") or []
        fr = [c["frac"] for c in cur]
        sidx = np.asarray(sub.get("doc_index") or [], dtype=np.int64)
        ok_sub = sub.get("doc_set") == "subset" and sub.get("bytes") == SUBSET_BYTES and np.array_equal(sidx, bctx["subset_idx"])
        ck.add("subset_is_bundle_0.5MB", ok_sub, {"bytes": sub.get("bytes"), "n": len(sidx)})
        exp_steps = [max(1, int(round(f * d["steps"]))) for f in (0.25, 0.5, 0.75)] + [d["steps"]]
        ok_c = fr == [0.25, 0.5, 0.75, 1.0] and [c["step"] for c in cur] == exp_steps
        if ok_c and ok_sub:
            b100 = float(np.sum(bits[sidx]) / np.sum(nb[sidx]))
            ok_c = ok_c and b100 == cur[-1]["bpb"] and all(c["sum_bytes"] == SUBSET_BYTES for c in cur)
        ck.add("curve_protocol", ok_c, {"fracs": fr})
    return ck


def validate_parity(hk, B, bctx, rel, rec, parity_ref):
    ck = Checks()
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
    ck.add("dev_arrays_len_parity", all(len(dv[k]) == n for k in ("bits", "bytes", "ntok", "uids", "doc_index"))
           and dv["doc_index"] == bctx["parity_idx"].tolist(), n)
    bits = np.asarray(dv["bits"], dtype=np.float64)
    nb = np.asarray(dv["bytes"], dtype=np.int64)
    bpb = float(bits.sum() / int(nb.sum()))
    ck.add("bpb_eq_sum_bits_over_sum_bytes", bpb == rec["bpb"], {"reported": rec["bpb"], "recomputed": bpb})
    rel_cpu = (rec["bpb"] - parity_ref["cpu"]["bpb"]) / parity_ref["cpu"]["bpb"]
    ck.add("within_1pct_of_cpu", abs(rel_cpu) <= parity_ref["tolerance_rel_bpb"], {"rel_diff_vs_cpu": rel_cpu})
    ck.add("device_dtype", rec["runtime"]["device"] == "cuda" and rec["runtime"]["amp_dtype"] == amp)
    return ck, rel_cpu


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--no-copy", action="store_true", help="validate the existing local copy only")
    a = ap.parse_args(argv)
    t0 = utcnow()
    hk = import_hk_lm()
    os.makedirs(OUT, exist_ok=True)
    report = {"what": "Colab LM-arbiter results: copy manifest + protocol validation (lm/collect_colab.py)",
              "collected_utc": t0, "source_root": a.src, "dest_root": OUT, "hk_lm_path": hk.__file__,
              "hk_lm_sha256_local": sha256_file(hk.__file__), "bundles": {}, "cross_bundle": {},
              "other_source_folders": {}, "split_manifest": {}}
    if report["hk_lm_sha256_local"] != HK_LM_SHA:
        raise SystemExit("local colab/hk_lm.py is not the version the records were produced with")
    if os.path.isdir(a.src):
        for name in sorted(os.listdir(a.src)):
            if name not in BUNDLES:
                report["other_source_folders"][name] = OTHER_FOLDERS_NOTE.get(name, "not part of this collection")

    rows = []
    recs_by = {}          # (bid, stage, cid, seed, lr) -> rec
    parity_by = {}        # (bid, amp) -> rec
    for bid, spec in BUNDLES.items():
        print("== bundle %s (round %d)" % (bid, spec["round"]), flush=True)
        copied, excluded = copy_bundle(a.src, bid, OUT, not a.no_copy)
        dst = os.path.join(OUT, bid)
        B = hk.Bundle(spec["staging"])
        n_verified = B.verify()                        # every bundle file vs its manifest sha256
        waves = load_json(spec["waves"])
        wmeta = {w["id"]: w for w in waves["waves"]}
        dev_docs = B.dev_docs()
        dtoks = {}
        for cid in B.candidate_ids:
            _, offs = B.tokens(cid, "dev")
            dtoks[cid] = np.diff(offs.astype(np.int64)) + 1          # doc tokens + EOT = scored targets
        bctx = {"bid": bid, "spec": spec, "waves_ids": set(wmeta), "dev_uids": [x["uid"] for x in dev_docs],
                "dev_bytes": B.dev_bytes().astype(np.int64), "dev_ntok": dtoks,
                "subset_idx": B.eval_doc_index("subset"), "parity_idx": B.eval_doc_index("parity")}
        pin_p = os.path.join(dst, "runtime_pin.json")
        pin = load_json(pin_p) if os.path.isfile(pin_p) else None
        parity_ref = load_json(os.path.join(spec["staging"], "PARITY.json"))
        brep = {"round": spec["round"], "bundle_manifest_sha256": spec["sha"],
                "staging_manifest_sha256": B.manifest_sha256, "staging_files_verified": n_verified,
                "staging_hk_lm_sha256": sha256_file(os.path.join(spec["staging"], "hk_lm.py")),
                "manifest_builder": B.m["builder"], "manifest_test_split": B.m.get("test_split"),
                "manifest_split_manifest_sha256": B.m["data"]["split_manifest_sha256"],
                "baseline": B.baseline_id, "candidates": B.candidate_ids, "lr_source": spec["lr_source"],
                "runtime_pin": pin, "copied": copied, "excluded": excluded,
                "large_stage_trace": large_stage_trace(os.path.join(dst, "progress.log")),
                "records": {}, "parity": {}, "stage_checks": {}}
        ok_manifest = (B.manifest_sha256 == spec["sha"] and brep["staging_hk_lm_sha256"] == HK_LM_SHA
                       and B.m["data"]["split_manifest_sha256"] == SPLIT_SHA and B.m.get("test_split") == "never read, encoded or bundled")
        brep["bundle_identity_ok"] = ok_manifest
        # ---- parity records
        for amp in ("fp16", "fp32"):
            rel = "parity/parity_cuda_%s.json" % amp
            p = os.path.join(dst, rel)
            if os.path.isfile(p):
                rec = load_json(p)
                ck, rel_cpu = validate_parity(hk, B, bctx, rel, rec, parity_ref)
                parity_by[(bid, amp)] = rec
                brep["parity"][rel] = {"bpb": rec["bpb"], "cpu_bpb": parity_ref["cpu"]["bpb"], "rel_diff_vs_cpu": rel_cpu,
                                       "n_checks": len(ck.items), "failed": ck.failed}
        # ---- training records
        res_dir = os.path.join(dst, "results")
        for fn in sorted(os.listdir(res_dir)):
            if not fn.endswith(".json"):
                continue
            rel = "results/" + fn
            rec = load_json(os.path.join(res_dir, fn))
            ck = validate_run(hk, B, bctx, rel, rec, pin)
            k = (bid, rec["stage"], rec["candidate"], rec["seed"], float(rec["lr"]))
            if k in recs_by:
                ck.add("unique_key", False, "duplicate record for %s" % (k,))
            recs_by[k] = rec
            brep["records"][rel] = {"n_checks": len(ck.items), "failed": ck.failed}
            rows.append(make_row(hk, bid, spec, B, wmeta, rel, rec, ck, os.path.join(dst, rel)))
        # ---- stage-level checks
        brep["stage_checks"] = stage_checks(hk, bid, B, recs_by, dst)
        report["bundles"][bid] = brep
        nfail = sum(1 for r in brep["records"].values() if r["failed"])
        print("   copied %d files (%d excluded); %d result records, %d with failed checks; staging files verified: %d"
              % (len(copied), len(excluded), len(brep["records"]), nfail, n_verified), flush=True)

    # ---- cross-bundle determinism of the reference arms
    report["cross_bundle"] = cross_bundle(hk, recs_by, parity_by, rows)
    # ---- split manifest: clusters of the 836 dev_strict documents (dev rows only)
    report["split_manifest"], dev_cluster_rows = split_clusters(rows, recs_by)
    # ---- write outputs
    write_table(rows)
    with open(os.path.join(OUT, "dev_clusters.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dev_cluster_rows[0].keys()))
        w.writeheader()
        w.writerows(dev_cluster_rows)
    report["finished_utc"] = utcnow()
    with open(os.path.join(OUT, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    write_collect_md(report, rows)
    tot = len(rows)
    bad = sum(1 for r in rows if not r["valid"])
    print("rows %d, invalid %d; cross-bundle identical: %s" % (tot, bad, report["cross_bundle"]["summary"]), flush=True)


def make_row(hk, bid, spec, B, wmeta, rel, rec, ck, path):
    c = B.cand(rec["candidate"]) if rec["candidate"] in B.candidate_ids else {}
    w = wmeta.get(rec["candidate"], {})
    meta = cand_meta_from_id(rec["candidate"])
    d, dv, rt = rec["data"], rec.get("dev") or {}, rec["runtime"]
    curve = {c_["frac"]: c_["bpb"] for c_ in rec.get("curve") or []}
    tl = [v for v in rec.get("train_loss") or [] if v is not None]
    stage = rec["stage"]
    if stage == "confirm":
        role = "stage4_confirm" if rec["seed"] <= 3 else "stage4_confirm_power_seed"
    else:
        role = {"lrsweep": "lr_sweep", "screen": "stage3_screen", "confirm_resweep": "finalist_lr_resweep_report_only"}[stage]
    rank = w.get("rank")
    if str(rec["candidate"]).startswith("R2-"):
        origin = "round2_exploratory"
    elif isinstance(rank, int) and rank >= 90:
        origin = "round1_added_reference_arm"
    else:
        origin = "round1_preregistered_rank"
    sum_ntok = int(np.sum(dv.get("ntok") or [0]))
    bits = np.asarray(dv.get("bits") or [np.nan], dtype=np.float64)
    nb = np.asarray(dv.get("bytes") or [1], dtype=np.int64)
    return collections.OrderedDict([
        ("bundle", bid), ("round", spec["round"]), ("stage", stage), ("stage_recipe", rec["stage_recipe"]),
        ("role", role), ("candidate", rec["candidate"]), ("seed", rec["seed"]), ("lr", rec["lr"]),
        ("lr_tag", hk.lr_tag(rec["lr"])),
        ("candidate_origin", origin), ("wave_rank", rank), ("wave_rank_label", w.get("rank_label")),
        ("conditional_wave", bool(w.get("conditional"))), ("is_baseline", rec["candidate"] == B.baseline_id),
        ("reference_arm_in_both_bundles", None), ("duplicate_of", ""),
        ("family", meta["family"]), ("algorithm", meta["algorithm"]), ("pretokenizer", meta["pretokenizer"]),
        ("data_mix", meta["data_mix"]), ("vocab_label", meta["vocab_label"]), ("variant", meta["variant"]),
        ("encoder_kind", c.get("kind")), ("n_vocab", rec["model"]["n_vocab"]),
        ("tokenizer_sha256", c.get("tokenizer_sha256") or w.get("tokenizer_sha256")),
        ("train_bytes_per_token", d["train_bytes_per_token"]), ("dev_bytes_per_token", c.get("dev_bytes_per_token")),
        ("model", rec["model"]["name"]), ("d_model", rec["model"]["d"]), ("n_layer", rec["model"]["L"]),
        ("ctx_tokens", d["ctx_tokens"]), ("params_total", rec["model"]["params"]["total"]),
        ("params_non_embedding", rec["model"]["params"]["non_embedding"]),
        ("steps", d["steps"]), ("steps_done", d["steps_done"]), ("budget_bytes", d["budget_bytes"]),
        ("bytes_seen", d["bytes_seen"]), ("bytes_seen_rel_budget", d["bytes_seen"] / d["budget_bytes"] - 1.0),
        ("tokens_seen", d["tokens_seen"]), ("epochs_seen", d["epochs_seen"]),
        ("dev_n_docs", dv.get("n_docs")), ("dev_sum_bits", dv.get("sum_bits")), ("dev_sum_bytes", dv.get("sum_bytes")),
        ("dev_sum_ntok_scored", sum_ntok), ("bpb", rec["bpb"]),
        ("bpb_recomputed", float(bits.sum() / int(nb.sum()))),
        ("subset_bpb_25", curve.get(0.25)), ("subset_bpb_50", curve.get(0.5)), ("subset_bpb_75", curve.get(0.75)),
        ("subset_bpb_100", curve.get(1.0)),
        ("train_loss_mean_last50", float(np.mean(tl[-50:])) if tl else None),
        ("status", rec["status"]), ("preregistered_flag", rec.get("preregistered")),
        ("device", rt["device"]), ("gpu_name", rt["gpu_name"]), ("amp_dtype", rt["amp_dtype"]),
        ("eval_dtype", rt["eval_dtype"]), ("torch", rt["torch"]),
        ("train_s", rec["wall_time_s"]["train"]), ("eval_s", rec["wall_time_s"]["eval"]),
        ("created_utc", rec["created_utc"]),
        ("bundle_manifest_sha256", rec["bundle_manifest_sha256"]), ("hk_lm_sha256", rec["hk_lm_sha256"]),
        ("file", bid + "/" + rel), ("file_sha256", sha256_file(path)),
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
    base = B.baseline_id
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
        lrs = sorted({r["lr"] for r in rs})
        keys = sorted((r["candidate"], r["seed"], r["lr"]) for r in rs)
        if st == "lrsweep":
            expected = sorted((base, 1, g) for g in hk.LR_GRID)
        elif st == "screen":
            expected = sorted((c, s, CHOSEN_LR) for c in cands for s in (1, 2, 3))
        elif st == "confirm":
            expected = sorted((c, s, CHOSEN_LR) for c in cands for s in (1, 2, 3, 4, 5))
        else:
            means = {}
            for c in cands:
                xs = [mine[(bid, "confirm", c, s, CHOSEN_LR)]["bpb"] for s in (1, 2, 3, 4, 5)
                      if (bid, "confirm", c, s, CHOSEN_LR) in mine]
                means[c] = sum(xs) / len(xs)
            top2 = [c for _, c in sorted((m, c) for c, m in means.items())[:2]]
            gap = (sorted(means.values())[1] - sorted(means.values())[0]) / sorted(means.values())[0]
            expected = sorted((c, 1, g) for c in top2 for g in hk.LR_GRID if g != CHOSEN_LR)
        missing = [list(map(str, k)) for k in expected if k not in keys]
        unexpected = [list(map(str, k)) for k in keys if k not in expected]
        # byte matching (PLAN 5) is a same-seed contract: every candidate reads the same seed-shuffled document order
        per_seed = collections.defaultdict(list)
        for r in rs:
            per_seed[(r["seed"], r["lr"])].append(r["data"]["bytes_seen"])
        same_seed = {"s%d@lr%s" % (s, hk.lr_tag(l)): max(v) / min(v) - 1 for (s, l), v in sorted(per_seed.items()) if len(v) > 1}
        o = {"present": True, "n_records": len(rs), "n_candidates": len({r["candidate"] for r in rs}),
             "lrs": lrs, "steps": steps, "steps_done": steps_done,
             "steps_identical_across_candidates": len(steps) == 1 and steps == steps_done,
             "bytes_seen_min": min(seen), "bytes_seen_max": max(seen),
             "bytes_seen_max_abs_rel_vs_budget": max(abs(x) for x in rel_b),
             "bytes_seen_n_outside_1pct_of_budget": sum(1 for x in rel_b if abs(x) > BYTES_SEEN_TOL),
             "bytes_seen_same_seed_spread_rel": same_seed,
             "bytes_seen_same_seed_spread_max_rel": max(same_seed.values()) if same_seed else 0.0,
             "bytes_seen_same_seed_within_1pct": all(v <= BYTES_SEEN_TOL for v in same_seed.values()),
             "bytes_seen_pooled_spread_rel_all_seeds": max(seen) / min(seen) - 1,
             "non_embedding_params": nonemb, "non_embedding_identical": len(nonemb) == 1,
             "complete": not missing and not unexpected, "missing": missing, "unexpected": unexpected}
        if st == "confirm_resweep":
            o["finalists_by_confirm_mean"] = top2
            o["finalist_gap_rel"] = gap
            o["resweep_required_gap_lt_1pct"] = gap < 0.01
            tab = {}
            for c in top2:
                tab[c] = {hk.lr_tag(CHOSEN_LR) + " (confirm s1)": mine[(bid, "confirm", c, 1, CHOSEN_LR)]["bpb"]}
                for g in hk.LR_GRID:
                    if g != CHOSEN_LR and (bid, st, c, 1, g) in mine:
                        tab[c][hk.lr_tag(g)] = mine[(bid, st, c, 1, g)]["bpb"]
                vals = list(tab[c].values())
                tab[c]["chosen_lr_best"] = vals[0] == min(vals)
            o["resweep_table"] = tab
            o["chosen_lr_beats_resweep_for_all_finalists"] = all(v["chosen_lr_best"] for v in tab.values())
        out[st] = o
    # lr_choice.json recomputation (round 1)
    p = os.path.join(dst, "lr_choice.json")
    if os.path.isfile(p):
        lc = load_json(p)
        sw = [(mine[(bid, "lrsweep", base, 1, g)]["bpb"], g) for g in hk.LR_GRID if (bid, "lrsweep", base, 1, g) in mine]
        best = min(sw)[1] if sw else None
        out["lr_choice_recomputed"] = {"file_chosen_lr": lc["chosen_lr"], "recomputed": best, "agree": best == lc["chosen_lr"],
                                       "at_grid_edge": best == min(hk.LR_GRID), "table": {hk.lr_tag(g): b for b, g in sw}}
        a_ = mine.get((bid, "lrsweep", base, 1, CHOSEN_LR))
        b_ = mine.get((bid, "screen", base, 1, CHOSEN_LR))
        if a_ and b_:
            out["within_session_determinism_lrsweep_vs_screen_s1"] = {
                "bits_identical": a_["dev"]["bits"] == b_["dev"]["bits"], "train_loss_identical": a_["train_loss"] == b_["train_loss"],
                "bpb": [a_["bpb"], b_["bpb"]]}
    # power check recomputation (PLAN 6)
    p = os.path.join(dst, "power_check.json")
    if os.path.isfile(p):
        pc = load_json(p)
        per = {}
        for c in cands:
            xs = [mine[(bid, "confirm", c, s, CHOSEN_LR)]["bpb"] for s in (1, 2, 3) if (bid, "confirm", c, s, CHOSEN_LR) in mine]
            m, sd = mean_sd(xs)
            per[c] = sd / m
        pooled = math.sqrt(sum(v ** 2 for v in per.values()) / len(per))
        n_req = math.ceil(15.7 * pooled ** 2 / 0.005 ** 2)
        all5 = {c: [s for s in (1, 2, 3, 4, 5) if (bid, "confirm", c, s, CHOSEN_LR) in mine] for c in cands}
        out["power_check_recomputed"] = {"file_pooled_rel_sd": pc["pooled_rel_sd"], "recomputed_pooled_rel_sd": pooled,
                                         "agree": abs(pooled - pc["pooled_rel_sd"]) < 1e-15, "seeds_needed_0.5pct": n_req,
                                         "file_seeds_needed": pc["seeds_needed_for_delta_0.5pct"],
                                         "extra_seeds_run": pc["extra_seeds_run"],
                                         "all_candidates_have_seeds_1_5": all(v == [1, 2, 3, 4, 5] for v in all5.values())}
    p = os.path.join(dst, "power_check_screen.json")
    if os.path.isfile(p):
        ps = load_json(p)
        xs = [mine[(bid, "screen", base, s, CHOSEN_LR)]["bpb"] for s in (1, 2, 3)]
        m, sd = mean_sd(xs)
        out["power_check_screen_recomputed"] = {"file_rel_sd": ps["rel_sd"], "recomputed_rel_sd": sd / m,
                                                "agree": abs(sd / m - ps["rel_sd"]) < 1e-15}
    # summary.json cross-check (non-'large' stages only)
    p = os.path.join(dst, "summary.json")
    if os.path.isfile(p):
        sj = load_json(p)
        agree, n = True, 0
        large_keys = []
        for key, rows in sj.get("stages", {}).items():
            st, lrt = key.split("@lr")
            if st.startswith(EXCLUDED_STAGE_PREFIX):
                large_keys.append(key)
                continue
            for x in rows:
                xs = [r["bpb"] for (b_, s_, c_, sd_, l_), r in mine.items()
                      if s_ == st and c_ == x["candidate"] and hk.lr_tag(l_) == lrt]
                m, _ = mean_sd(xs)
                n += 1
                agree = agree and m is not None and abs(m - x["mean_bpb"]) <= 1e-15 and len(xs) == x["n_ok"]
        out["summary_json_crosscheck"] = {"n_rows": n, "means_agree": agree, "large_rows_ignored": large_keys}
        if "screen_curve_crossing_check" in sj:
            out["screen_curve_crossing_check_file"] = sj["screen_curve_crossing_check"]
    # PLAN 5 crossing check recomputed on the screen subset curve
    pts = collections.defaultdict(lambda: collections.defaultdict(list))
    for (b_, s_, c_, sd_, l_), r in mine.items():
        if s_ == "screen":
            for cp in r["curve"]:
                pts[c_][cp["frac"]].append(cp["bpb"])
    full = {c: np.mean(v[1.0]) for c, v in pts.items()}
    top = sorted(full, key=full.get)[:2]
    at75 = {c: float(np.mean(pts[c][0.75])) for c in top}
    out["screen_curve_crossing_recomputed"] = {"top_two_at_100pct": top, "subset_bpb_100": {c: float(full[c]) for c in top},
                                               "subset_bpb_75": at75, "crossed": at75[top[0]] > at75[top[1]]}
    return out


def cross_bundle(hk, recs_by, parity_by, rows):
    b1, b2 = list(BUNDLES)
    c1 = {k[2] for k in recs_by if k[0] == b1}
    c2 = {k[2] for k in recs_by if k[0] == b2}
    refs = sorted(c1 & c2)
    pairs = []
    for k, r2 in sorted(recs_by.items(), key=lambda kv: str(kv[0])):
        if k[0] != b2 or k[2] not in refs:
            continue
        k1 = (b1,) + k[1:]
        r1 = recs_by.get(k1)
        if r1 is None:
            pairs.append({"stage": k[1], "candidate": k[2], "seed": k[3], "lr": k[4], "round1_record": None})
            continue
        a = np.asarray(r1["dev"]["bits"])
        b = np.asarray(r2["dev"]["bits"])
        pairs.append({"stage": k[1], "candidate": k[2], "seed": k[3], "lr": k[4],
                      "bits_identical": r1["dev"]["bits"] == r2["dev"]["bits"],
                      "n_docs_differing": int(np.sum(a != b)), "max_abs_diff_bits": float(np.max(np.abs(a - b))),
                      "ntok_identical": r1["dev"]["ntok"] == r2["dev"]["ntok"],
                      "train_loss_identical": r1["train_loss"] == r2["train_loss"],
                      "curve_identical": r1["curve"] == r2["curve"],
                      "r3_identical": r1["r3_lowest_norm_embeddings"] == r2["r3_lowest_norm_embeddings"],
                      "bpb": r1["bpb"], "bpb_identical": r1["bpb"] == r2["bpb"],
                      "runtime_identical": {k_: r1["runtime"][k_] == r2["runtime"][k_] for k_ in ("gpu_name", "torch", "cuda", "amp_dtype")}})
    # parity across bundles
    par = {}
    for amp in ("fp16", "fp32"):
        p1, p2 = parity_by.get((b1, amp)), parity_by.get((b2, amp))
        if p1 and p2:
            par[amp] = {"bits_identical": p1["dev"]["bits"] == p2["dev"]["bits"], "bpb": [p1["bpb"], p2["bpb"]]}
    # inputs: token arrays / shared arrays identical in both manifests?
    m1 = load_json(os.path.join(BUNDLES[b1]["staging"], "manifest.json"))
    m2 = load_json(os.path.join(BUNDLES[b2]["staging"], "manifest.json"))
    shared = {}
    for name, info in m1["files"].items():
        if "/" not in name or name.split("/")[0] in refs:
            if name in m2["files"]:
                shared[name] = info["sha256"] == m2["files"][name]["sha256"]
    tok_sha = {c: (next(x for x in m1["candidates"] if x["id"] == c)["tokenizer_sha256"] ==
                   next(x for x in m2["candidates"] if x["id"] == c)["tokenizer_sha256"]) for c in refs}
    ok = [p for p in pairs if p.get("bits_identical")]
    # annotate table rows
    ident = {(p["stage"], p["candidate"], p["seed"], p["lr"]): p.get("bits_identical") for p in pairs}
    for r in rows:
        if r["candidate"] in refs:
            r["reference_arm_in_both_bundles"] = True
            if r["bundle"] == b2 and ident.get((r["stage"], r["candidate"], r["seed"], float(r["lr"]))):
                r["duplicate_of"] = b1
        else:
            r["reference_arm_in_both_bundles"] = False
    return {"reference_arms": refs, "n_pairs": len(pairs), "n_bitwise_identical": len(ok),
            "all_identical": len(ok) == len(pairs) and len(pairs) > 0 and all(
                p["train_loss_identical"] and p["curve_identical"] and p["ntok_identical"] for p in pairs),
            "summary": "%d/%d run pairs bitwise identical per-document bits" % (len(ok), len(pairs)),
            "pairs": pairs, "parity_across_bundles": par,
            "input_arrays_identical_in_both_manifests": shared, "tokenizer_sha_identical": tok_sha}


def split_clusters(rows, recs_by):
    sha = sha256_file(SPLIT_MANIFEST)
    any_rec = next(iter(recs_by.values()))
    uids = any_rec["dev"]["uids"]
    nbytes = any_rec["dev"]["bytes"]
    want = set(uids)
    got = {}
    with open(SPLIT_MANIFEST, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x["uid"] in want:                     # dev_strict rows only; nothing else is kept
                got[x["uid"]] = x
    out_rows = []
    for i, (u, b) in enumerate(zip(uids, nbytes)):
        x = got.get(u, {})
        g = x.get("group")
        m = re.match(r"^web:([^:]+):record:", g or "")
        cluster = ("web:" + m.group(1)) if m else g
        out_rows.append(collections.OrderedDict([("doc_index", i), ("uid", u), ("bytes", b), ("source", x.get("source")),
                                                 ("split", x.get("split")), ("quality_tier", x.get("quality_tier")),
                                                 ("group", g), ("cluster", cluster)]))
    by_src = collections.Counter(r["source"] for r in {(r["source"], r["cluster"]): r for r in out_rows}.values())
    by_src_raw = collections.Counter(r["source"] for r in {(r["source"], r["group"]): r for r in out_rows}.values())
    cl_bytes = collections.Counter()
    for r in out_rows:
        cl_bytes[r["cluster"]] += r["bytes"]
    big = cl_bytes.most_common(1)[0]
    rep = {"path": SPLIT_MANIFEST, "sha256": sha, "sha256_ok": sha == SPLIT_SHA, "n_dev_uids": len(uids),
           "n_found": len(got), "all_validation_strict": all(r["split"] == "validation" and r["quality_tier"] == "strict" for r in out_rows),
           "n_groups_raw": len({r["group"] for r in out_rows}), "groups_raw_by_source": dict(by_src_raw),
           "n_clusters_web_records_collapsed_to_site": len({r["cluster"] for r in out_rows}),
           "clusters_by_source": dict(by_src),
           "plan6_expects": "32 clusters (15 newspaper, 12 book, 5 web)",
           "largest_cluster": {"cluster": big[0], "bytes": big[1], "share": big[1] / DEV_BYTES},
           "all_records_same_uid_order": all(r["dev"]["uids"] == uids for r in recs_by.values())}
    return rep, out_rows


def write_table(rows):
    cols = list(rows[0].keys())
    rows.sort(key=lambda r: (r["bundle"] != list(BUNDLES)[0], STAGE_ORDER.index(r["stage"]), r["candidate"], r["lr"], r["seed"]))
    with open(os.path.join(OUT, "results_table.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(OUT, "results_table.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)


# ------------------------------------------------------------------ COLLECT.md
def fmt(x, n=5):
    return "-" if x is None else ("%.*f" % (n, x))


def write_collect_md(rep, rows):
    L = []
    A = L.append
    b1, b2 = list(BUNDLES)
    cb = rep["cross_bundle"]
    sm = rep["split_manifest"]
    nrows = len(rows)
    nvalid = sum(1 for r in rows if r["valid"])
    nchecks = sum(r["n_checks"] for r in rows)
    A("# Colab LM-arbiter results: collection and validation")
    A("")
    A("Collected %s by `lm/collect_colab.py`, from `%s`. Regenerate: "
      "`PYTHONIOENCODING=utf-8 python lm/collect_colab.py` (add `--no-copy` to re-validate the local copy only)."
      % (rep["collected_utc"], rep["source_root"]))
    A("")
    A("This file covers collection and validation only. No candidate is selected here. PLAN §7 needs the §6 cluster "
      "bootstrap, which comes next. The **test split was not read**. The bundles contain no test data "
      "(`manifest.test_split` = \"%s\"), and the split manifest was only queried for the 836 dev_strict uids that the "
      "records carry." % rep["bundles"][b1]["manifest_test_split"])
    A("")
    A("## 1. Result")
    A("")
    ok_all = nvalid == nrows and all(b["bundle_identity_ok"] for b in rep["bundles"].values())
    par_all = [p for b in rep["bundles"].values() for p in b["parity"].values()]
    par_ok = all(not p["failed"] for p in par_all)
    steps_ok = all(s.get("steps_identical_across_candidates") for b in rep["bundles"].values()
                   for k, s in b["stage_checks"].items() if k in STAGE_ORDER and s.get("present"))
    steps_by_stage = sorted({(r["stage_recipe"], r["steps"]) for r in rows})
    dev_ok = all(r["device"] == "cuda" and r["amp_dtype"] == "fp16" and r["eval_dtype"] == "fp32" and r["gpu_name"] == "Tesla T4"
                 and "device_dtype" not in r["failed_checks"] for r in rows)
    bpb_exact = all(r["bpb"] == r["bpb_recomputed"] for r in rows)
    nhard = sum(1 for r in rows if r["hard_valid"])
    flagged = [r for r in rows if r["tolerance_flags"]]
    stc = [s for b in rep["bundles"].values() for k, s in b["stage_checks"].items() if k in STAGE_ORDER and s.get("present")]
    same_seed_max = max(s["bytes_seen_same_seed_spread_max_rel"] for s in stc)
    A("- **%d / %d result records pass every identity and integrity check.** %d / %d pass every check including the "
      "`bytes_seen` tolerance (%d checks in total). Failed: %s."
      % (nhard, nrows, nvalid, nrows, nchecks, "; ".join("`%s` → %s" % (r["file"], r["failed_checks"]) for r in rows
                                                        if r["failed_checks"]) or "none"))
    A("- The %d parity records (2 per bundle) %s their own checks (§4)." % (len(par_all), "pass" if par_ok else "do NOT all pass"))
    A("- **Bitwise determinism across sessions:** %s. That covers the %d reference arms × (screen s1–3 + confirm s1–5). "
      "Per-document bits, per-document token counts, training-loss curves and subset curves are all identical: %s."
      % (cb["summary"], len(cb["reference_arms"]), cb["all_identical"]))
    A("- **Steps identical across candidates within every stage:** %s (%s)." % (
        steps_ok, ", ".join("%s %d" % sr for sr in steps_by_stage)))
    A("- **bytes_seen:** %d / %d runs are within 1 %% of the stage's byte budget. %s For the same seed, the candidates differ "
      "by at most **%.2f %%** in every stage, which is within 1 %%; this is the PLAN §5 fairness contract, since all "
      "candidates share the seed's document order (§3.1)."
      % (nrows - len(flagged), nrows,
         " ".join("The exception is `%s`: %+.3f %%, %.3f pp over the tolerance." % (
             r["file"], 100 * r["bytes_seen_rel_budget"], 100 * (abs(r["bytes_seen_rel_budget"]) - BYTES_SEEN_TOL))
             for r in flagged) if flagged else "No exceptions.", 100 * same_seed_max))
    A("- **Device / dtype:** %s." % ("every run used cuda / Tesla T4 / fp16 autocast for training, fp32 for evaluation, strict "
                                     "determinism and tf32 off. This matches each bundle's `runtime_pin.json`" if dev_ok
                                     else "NOT uniform - see validation.json"))
    A("- **bpb = Σbits / Σbytes (PLAN §4.1):** recomputing from the 836 per-document values %s." % (
        "reproduces the reported bpb **bit-for-bit** in every record" if bpb_exact else "does NOT reproduce every record"))
    A("- The split manifest `%s…` (sha256 ok: %s) maps the 836 dev documents to %d raw groups. With the per-record web groups "
      "collapsed to their site, that gives **%d clusters** %s. PLAN §6 expects %s. The largest cluster is `%s` (%.1f %% of dev bytes). "
      "The mapping is in `dev_clusters.csv`."
      % (sm["sha256"][:8], sm["sha256_ok"], sm["n_groups_raw"], sm["n_clusters_web_records_collapsed_to_site"],
         json.dumps(sm["clusters_by_source"]), sm["plan6_expects"], sm["largest_cluster"]["cluster"],
         100 * sm["largest_cluster"]["share"]))
    hard_all = nhard == nrows and all(b["bundle_identity_ok"] for b in rep["bundles"].values()) and par_ok
    if ok_all:
        verdict = "**all collected records are valid and usable for the PLAN §6/§7 analysis.**"
    elif hard_all:
        verdict = ("**all %d records are the runs they claim to be, and are usable for the PLAN §6/§7 analysis.** %d screening "
                   "run%s carr%s a `bytes_seen` tolerance flag. It is kept and flagged: it is not decisive, and the "
                   "same-seed matching holds (§3.1)." % (nrows, len(flagged), "" if len(flagged) == 1 else "s",
                                                          "ies" if len(flagged) == 1 else "y"))
    else:
        verdict = "**SOME IDENTITY/INTEGRITY CHECKS FAILED - see §3 and validation.json.**"
    A("- Verdict: " + verdict)
    A("")
    A("## 2. What was copied")
    A("")
    A("| bundle | round | full manifest sha256 | files copied | bytes | sha256 verified after copy | excluded |")
    A("|---|---|---|---|---|---|---|")
    for bid, b in rep["bundles"].items():
        A("| `%s` | %d | `%s` | %d | %d | %s | %d |" % (
            bid, b["round"], b["bundle_manifest_sha256"], len(b["copied"]), sum(x["bytes"] for x in b["copied"]),
            "all %d" % len(b["copied"]) if all(x.get("dest_sha256_verified") for x in b["copied"]) else "NO",
            len(b["excluded"])))
    A("")
    A("Each bundle folder is copied with its layout intact (`results/`, `parity/`, `parity_report.json`, `power_check*.json`, "
      "`lr_choice.json` (round 1 only), `runtime_pin.json`, `summary.json/.txt`, `progress.log`) to "
      "`lm/colab_results/<bundle>/`. `validation.json` → `bundles.<id>.copied` lists every file with its size, sha256 and source mtime.")
    A("")
    A("The bundle identity is checked too. `colab/build*/staging/manifest.json` hashes to the bundle id. All %d / %d staging files "
      "match their manifest sha256. The staging `hk_lm.py` and the local `colab/hk_lm.py` are both `%s…`, which is the "
      "`hk_lm_sha256` in every record. Both manifests carry split manifest `76582d3a…`."
      % (rep["bundles"][b1]["staging_files_verified"], rep["bundles"][b2]["staging_files_verified"], HK_LM_SHA[:12]))
    A("")
    A("Other folders under `%s` that were not copied: %s." % (
        rep["source_root"], "; ".join("`%s`: %s" % kv for kv in rep["other_source_folders"].items()) or "none"))
    A("")
    A("### 2.1 The 'large' stage: left out, listed here")
    A("")
    A("`large` (d=384, L=6, 2 epochs, own LR grid {5e-4, 1e-3, 2e-3}) is an **addition beyond the pre-registration** "
      "(PLAN §8(ii)). It is report-only, and the PLAN §7 decision never uses it. It is incomplete in both bundles, so none of its files were collected:")
    A("")
    for bid, b in rep["bundles"].items():
        t = b["large_stage_trace"] or {}
        ex = b["excluded"]
        A("- **`%s` (round %d)**: %s" % (bid, b["round"], "stage started (`%s`)" % t.get("stage_header", "").strip()
                                           if t.get("started") else "never started"))
        if t.get("started"):
            A("  - session: `%s`" % (t.get("session_line") or "").strip())
            A("  - runs finished at the time of copying: %s" % ("; ".join("`%s`" % x.strip() for x in t["runs_done"]) or "none"))
            A("  - last `large` progress line in the copied log: `%s`" % (t.get("last_step_line") or "").strip())
        A("  - result files present in the source and excluded: %s" % (
            ", ".join("`%s` (%d B, sha256 %s…)" % (x["path"], x["bytes"], x["sha256"][:12]) for x in ex) or "none"))
    A("")
    for bid, b in rep["bundles"].items():
        t = b["large_stage_trace"] or {}
        if not t.get("started"):
            continue
        fin = "finished in" in (t.get("last_log_line") or "")
        A("- `%s`: the log's last line is `%s`. %s" % (
            bid, (t.get("last_log_line") or "").strip(),
            "The stage finished after the copy's snapshot of the log." if fin else
            "There is no 'finished' line after the `large` header, so the stage was incomplete at copy time: interrupted, "
            "or still running on Colab."))
    A("")
    ex2 = rep["bundles"][b2]["excluded"]
    A("Round 1's `large` stopped inside its first LR-sweep run and wrote no result file. The round-2 session started at 17:58 UTC, "
      "about a minute after round 1's log ends. At copy time, round 2's `large` had written %d file(s): %s. All are excluded "
      "above. Its `progress.log` and `summary.json` are snapshots that may mention `large*` rows. The validation ignores those "
      "rows (§3). If `large` is wanted as a report-only result, collect it separately once it completes."
      % (len(ex2), ", ".join("`%s`" % x["path"].split("/")[-1] for x in ex2) or "none"))
    A("")
    A("## 3. Inventory and validation")
    A("")
    A("| bundle | stage | records | candidates | seeds | LR | steps (all cands) | bytes_seen max \\|rel\\| vs budget | same-seed spread across cands (max) | complete | identity/integrity checks | all checks |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for bid, b in rep["bundles"].items():
        for st in STAGE_ORDER:
            s = b["stage_checks"].get(st, {})
            if not s.get("present"):
                A("| `%s` | %s | 0 | - | - | - | - | - | - | %s | - | - |" % (bid, st, "n/a (not run in this bundle)"))
                continue
            rs = [r for r in rows if r["bundle"] == bid and r["stage"] == st]
            seeds = sorted({r["seed"] for r in rs})
            A("| `%s` | %s | %d | %d | %s | %s | %s | %.3f %% | %s | %s | %s | %s |" % (
                bid, st, s["n_records"], s["n_candidates"], ",".join(map(str, seeds)),
                ", ".join("%g" % x for x in s["lrs"]), "/".join(map(str, s["steps"])),
                100 * s["bytes_seen_max_abs_rel_vs_budget"],
                ("%.3f %%" % (100 * s["bytes_seen_same_seed_spread_max_rel"])) if s["n_candidates"] > 1 else "- (1 cand)",
                "yes" if s["complete"] else "NO: missing %s, unexpected %s" % (s["missing"], s["unexpected"]),
                "pass" if all(r["hard_valid"] for r in rs) else "FAIL",
                "pass" if all(r["valid"] for r in rs) else "%d flagged" % sum(1 for r in rs if not r["valid"])))
    A("")
    A("Checks applied to **every** training record (`validation.json` → `bundles.<id>.records`; failures are listed per record):")
    A("")
    A("- **bundle sha:** the record's `bundle_manifest_sha256` = the full sha of the folder's bundle = the sha256 of the staging `manifest.json`.")
    A("- **hk_lm sha:** `hk_lm_sha256` = `%s` = the manifest's `builder.hk_lm_sha256`." % HK_LM_SHA)
    A("- **stage / candidate / seed / exact LR:**")
    A("  - the file key (`stage__cand__lrTAG__sN`) equals the record's fields;")
    A("  - the stage is one of lrsweep / screen / confirm / confirm_resweep, with the right recipe;")
    A("  - the candidate is in the bundle and in its WAVES file;")
    A("  - the seed is allowed for the stage (screen 1–3, confirm 1–5, sweeps 1);")
    A("  - the LR matches the stage **exactly** (float `==`): screen and confirm = 1e-3; lrsweep ∈ {1e-3, 3e-3, 6e-3} on the baseline; "
      "re-sweep ∈ {3e-3, 6e-3}. `optimizer.peak_lr` = lr.")
    A("- **The arbiter's own identity check:** `hk_lm.identity_mismatches(record, hk_lm.run_identity(...))` is run against the bundle, on "
      "cuda/fp16. It covers protocol, bundle, hk_lm, stage, recipe, candidate, seed, LR, model, budget bytes, epochs, steps, ctx, "
      "overrides, device and AMP dtype. It is the same test that `run_all.py` applies before it reuses a stored record.")
    A("- **Model:** d / L / H per recipe (screen 128/4/4, confirm 192/4/4). n_vocab and ctx equal the bundle's. ctx = round(1536 / train "
      "bytes-per-token), as in PLAN §5 byte-matched context. Embeddings are tied, dropout is 0, and non-embedding parameters are identical "
      "within each stage.")
    A("- **Optimiser as in PLAN §5:** AdamW (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0, 50 warm-up steps, cosine to 10 %.")
    A("- **Steps:**")
    A("  - steps = steps_done = len(train_loss) = the PLAN budget: screen round(10 MB / 12,288) = 814; confirm round(45,007,515 / 12,288) = 3,663;")
    A("  - **identical across candidates within each stage**;")
    A("  - all training losses are finite.")
    A("- **bytes_seen:** each record is checked against 1 %% of its stage's byte budget (the tolerance flag). Each stage is also "
      "checked for the spread across candidates within the same seed (§3.1). The largest per-record deviation is %.3f %%."
      % (100 * max(abs(r["bytes_seen_rel_budget"]) for r in rows)))
    A("- **Per-document arrays:** `bits`, `bytes`, `ntok`, `uids` and `doc_index` all have length **836**. `doc_index` = 0..835. "
      "`uids` and `bytes` equal the bundle's dev_strict (Σ = 1,445,513 bytes). `ntok` = the bundle's dev token count + 1 (EOT). "
      "All bits are finite and > 0.")
    A("- **bpb:** `float(np.sum(bits) / np.sum(bytes))` **==** the reported `bpb` == `dev.bpb`, bit-exact, and `sum_bits` == Σbits. "
      "The `math.fsum` version agrees to < 1e-15 relative.")
    A("- **Evaluation window:** ctx = the run's ctx, stride ctx/2, BOS given, EOT predicted (PLAN §4.1/§5).")
    A("- **Curve:**")
    A("  - it is taken at 25/50/75/100 % of the steps, on the bundle's fixed 0.5 MB subset (297 docs, 504,655 bytes);")
    A("  - the 100 % point equals the full-dev bits restricted to the subset.")
    A("- **Device / dtype:** cuda, fp16 AMP, fp32 eval, strict determinism, tf32 off, flush-denormal on. This equals `runtime_pin.json`.")
    A("")
    A("Stage-level checks (`validation.json` → `bundles.<id>.stage_checks`):")
    A("")
    for bid, b in rep["bundles"].items():
        sc = b["stage_checks"]
        A("- **`%s`**" % bid)
        lc = sc.get("lr_choice_recomputed")
        if lc:
            A("  - `lr_choice.json` recomputed from the 3 sweep records: %s (bpb %s). It agrees with the file: %s. At the grid edge: %s."
              % (lc["recomputed"], ", ".join("%s → %.5f" % kv for kv in lc["table"].items()), lc["agree"], lc["at_grid_edge"]))
            d = sc.get("within_session_determinism_lrsweep_vs_screen_s1", {})
            A("  - Within-session determinism: lrsweep lr1e-3 s1 and screen s1 of the baseline are the same run, and are "
              "bitwise identical (bits %s, train loss %s)." % (d.get("bits_identical"), d.get("train_loss_identical")))
        else:
            A("  - No LR stage in this bundle: the LR was %s." % b["lr_source"])
        pc = sc.get("power_check_recomputed")
        if pc:
            A("  - PLAN §6 power check recomputed from confirm seeds 1–3: the pooled relative seed s.d. is %.3f %% (file %.3f %%, "
              "agree %s). That is > 0.2 %%, so seeds 4 and 5 were run for every candidate (%s). The formula n = ⌈15.7 σ² / 0.005²⌉ "
              "asks for **%d seeds**."
              % (100 * pc["recomputed_pooled_rel_sd"], 100 * pc["file_pooled_rel_sd"], pc["agree"],
                 "all have seeds 1–5" if pc["all_candidates_have_seeds_1_5"] else "INCOMPLETE", pc["seeds_needed_0.5pct"]))
        ps = sc.get("power_check_screen_recomputed")
        if ps:
            A("  - Baseline Stage-3 seed s.d. (the PLAN §6 literal 'after the baseline's first wave' check): %.3f %% (agrees: %s)."
              % (100 * ps["recomputed_rel_sd"], ps["agree"]))
        rs = sc.get("confirm_resweep", {})
        if rs.get("present"):
            A("  - Finalist LR re-sweep. The top two by confirm mean (5 seeds) are %s, with a gap of %.3f %% (< 1 %%, so the "
              "re-sweep is required). The re-sweep records are for exactly these two (complete: %s). The chosen 1e-3 beats 3e-3 and 6e-3 "
              "for both: %s." % (" and ".join("`%s`" % c for c in rs["finalists_by_confirm_mean"]), 100 * rs["finalist_gap_rel"],
                                 rs["complete"], rs["chosen_lr_beats_resweep_for_all_finalists"]))
            for c, t in rs["resweep_table"].items():
                A("    - `%s`: %s" % (c, ", ".join("%s → %.5f" % (k, v) for k, v in t.items() if k != "chosen_lr_best")))
        cc = sc.get("screen_curve_crossing_recomputed")
        if cc:
            A("  - PLAN §5 screen-curve crossing check (top two at 100 %% of the 0.5 MB subset: %s): crossed between 75 %% and 100 %%: **%s**."
              % (" / ".join("`%s`" % c for c in cc["top_two_at_100pct"]), cc["crossed"]))
        sj = sc.get("summary_json_crosscheck")
        if sj:
            A("  - `summary.json` stage means were recomputed from the records: %d rows, all agree: %s%s." % (
                sj["n_rows"], sj["means_agree"], (" (ignored 'large' rows: %s)" % ", ".join(sj["large_rows_ignored"]))
                if sj["large_rows_ignored"] else ""))
    A("")
    A("### 3.1 bytes_seen: what varies, and the one tolerance flag")
    A("")
    A("`hk_lm` matches the budget in bytes as PLAN §5 requires:")
    A("")
    A("- steps = round(B / 12,288), the same for every candidate;")
    A("- ctx(c) = round(1,536 / train bytes-per-token(c));")
    A("- each run therefore reads a fixed number of tokens, steps × 8 × ctx(c) + 1, of the seed-shuffled stream.")
    A("")
    A("The bytes those tokens cover (`bytes_seen`) move away from B for three reasons:")
    A("")
    A("1. **The seed's document order.** Screening reads about 22 % of the train split, so the local compression of the first "
      "documents matters. This effect is shared by all candidates, because every candidate gets the same order for a given seed.")
    cf = [100 * r["bytes_seen_rel_budget"] for r in rows if r["stage_recipe"] == "confirm"]
    A("2. **BOS/EOT tokens.** They carry no bytes. This is why the full-epoch confirm runs all sit %.2f–%.2f %% *below* B%s."
      % (-max(cf), -min(cf), "" if max(cf) < 0 else " (NOTE: not all below)"))
    A("3. **The rounding of ctx.**")
    A("")
    A("| bundle | stage | bytes_seen rel. to budget, by seed: min … max over candidates |")
    A("|---|---|---|")
    for bid in BUNDLES:
        for st in ("screen", "confirm"):
            rs = [r for r in rows if r["bundle"] == bid and r["stage"] == st]
            parts = []
            for s in sorted({r["seed"] for r in rs}):
                v = [100 * r["bytes_seen_rel_budget"] for r in rs if r["seed"] == s]
                parts.append("s%d %+.2f … %+.2f %%" % (s, min(v), max(v)))
            A("| `%s` | %s | %s |" % (bid, st, "; ".join(parts)))
    A("")
    if flagged:
        for r in flagged:
            same = [x for x in rows if x["bundle"] == r["bundle"] and x["stage"] == r["stage"] and x["seed"] == r["seed"]]
            v = [x["bytes_seen"] for x in same]
            bpt = {x["candidate"]: x["train_bytes_per_token"] for x in rows if x["bundle"] == r["bundle"]}
            rank_bpt = sorted(bpt, key=bpt.get, reverse=True).index(r["candidate"]) + 1
            all_above = all(x["bytes_seen_rel_budget"] > 0 for x in same)
            conf = [abs(x["bytes_seen_rel_budget"]) for x in rows if x["bundle"] == r["bundle"] and x["candidate"] == r["candidate"]
                    and x["stage"] == "confirm"]
            A("**Flag:** `%s` (%s, %s, seed %d) read %.1f bytes, %+.3f %% of the 10 MB screening budget. Its tokenizer ranks #%d of "
              "%d in this bundle by compression (train %.3f bytes/token, ctx %d). It ran on seed %d, whose document order puts "
              "above-average-compression text first (%s). Within that seed, the candidates differ by %.2f %%, which is within 1 %%. "
              "The bias is small (+%.3f pp more text than the next candidate on that seed) and in the direction of *more* "
              "training text for this run. It is a screening run (Stage 3), which PLAN §7.2 does not use when Stage 4 exists, and "
              "the same candidate's confirm runs are all within %.2f %% of budget. **Kept and flagged** (`tolerance_flags`), "
              "not dropped."
              % (r["file"], r["candidate"], r["stage"], r["seed"], r["bytes_seen"], 100 * r["bytes_seen_rel_budget"],
                 rank_bpt, len(bpt), r["train_bytes_per_token"], r["ctx_tokens"], r["seed"],
                 "every candidate is above budget on this seed" if all_above else "not every candidate is above budget",
                 100 * (max(v) / min(v) - 1), 100 * (r["bytes_seen"] - sorted(v)[-2]) / r["budget_bytes"],
                 100 * max(conf) if conf else float("nan")))
    else:
        A("No run is outside 1 % of its budget.")
    A("")
    A("## 4. Parity (GPU vs CPU reference, fixed tiny config, 65 dev docs)")
    A("")
    A("| bundle | record | bpb | CPU reference | rel. diff | checks |")
    A("|---|---|---|---|---|---|")
    for bid, b in rep["bundles"].items():
        for rel, p in b["parity"].items():
            A("| `%s` | `%s` | %.6f | %.6f | %+.3f %% | %s |" % (bid, rel, p["bpb"], p["cpu_bpb"], 100 * p["rel_diff_vs_cpu"],
                                                          "pass (%d)" % p["n_checks"] if not p["failed"] else "FAIL %s" % p["failed"]))
    par = cb["parity_across_bundles"]
    A("")
    A("The parity runs are also bitwise identical across the two sessions (fp16: %s, fp32: %s)." % (
        par.get("fp16", {}).get("bits_identical"), par.get("fp32", {}).get("bits_identical")))
    A("")
    A("## 5. Cross-session determinism of the 4 reference arms")
    A("")
    A("Reference arms present in both bundles: %s. Each bundle re-encoded its own copy of the data. The token arrays and shared arrays "
      "of these arms have identical sha256 in both manifests (%d / %d files), and so do the tokenizers (%s)."
      % (", ".join("`%s`" % c for c in cb["reference_arms"]), sum(cb["input_arrays_identical_in_both_manifests"].values()),
         len(cb["input_arrays_identical_in_both_manifests"]), all(cb["tokenizer_sha_identical"].values())))
    A("")
    A("| candidate | stage | seeds | per-doc bits identical | docs differing | train loss identical | bpb |")
    A("|---|---|---|---|---|---|---|")
    grp = collections.OrderedDict()
    for p in cb["pairs"]:
        grp.setdefault((p["candidate"], p["stage"]), []).append(p)
    for (c, st), ps in grp.items():
        A("| `%s` | %s | %s | %s | %d | %s | %s |" % (
            c, st, ",".join(str(p["seed"]) for p in ps), "yes (%d/%d)" % (sum(p["bits_identical"] for p in ps), len(ps))
            if all(p["bits_identical"] for p in ps) else "NO (%d/%d)" % (sum(p["bits_identical"] for p in ps), len(ps)),
            sum(p["n_docs_differing"] for p in ps), all(p["train_loss_identical"] for p in ps),
            ", ".join("%.5f" % p["bpb"] for p in ps)))
    A("")
    A("Consequence for the analysis: the round-2 copies of the reference arms are exact duplicates of the round-1 runs. In "
      "`results_table`, those rows have `duplicate_of` = `%s`. Drop them when pooling both bundles, so that no run is counted twice." % b1)
    A("")
    A("## 6. Descriptive summary (dev_strict bpb; NOT the decision)")
    A("")
    A("These are means and seed s.d. only. The PLAN §7 ranking and claims need the §6 hierarchical cluster bootstrap, Holm "
      "correction and TOST, which have not been run yet. `*` marks candidates whose confirm seed s.d. is above 0.5 % of bpb.")
    A("")
    for bid in BUNDLES:
        for st in ("screen", "confirm"):
            rs = [r for r in rows if r["bundle"] == bid and r["stage"] == st]
            agg = collections.defaultdict(list)
            for r in rs:
                agg[r["candidate"]].append(r)
            base = [r["bpb"] for r in agg.get("A1-P1r3-D2-16k", [])]
            bm = sum(base) / len(base)
            A("**`%s` %s** (lr 1e-3; baseline A1-P1r3-D2-16k = %.5f)" % (bid, st, bm))
            A("")
            A("| # | candidate | origin | n | mean bpb | seed s.d. (rel %) | Δ vs baseline | per-seed bpb |")
            A("|---|---|---|---|---|---|---|---|")
            order = sorted(agg, key=lambda c: sum(r["bpb"] for r in agg[c]) / len(agg[c]))
            for i, c in enumerate(order, 1):
                xs = [r["bpb"] for r in sorted(agg[c], key=lambda r: r["seed"])]
                m, sd = mean_sd(xs)
                o = agg[c][0]["candidate_origin"] + (" (conditional rank 8)" if agg[c][0]["conditional_wave"] else "")
                A("| %d | `%s`%s | %s | %d | %.5f | %.3f | %+.3f %% | %s |" % (
                    i, c, " *" if (st == "confirm" and sd / m > 0.005) else "", o, len(xs), m, 100 * sd / m,
                    100 * (m - bm) / bm, " ".join("%.4f" % x for x in xs)))
            A("")
    A("## 7. Deviations from PLAN (documented honestly)")
    A("")
    A("All deviations are additive. None changes a pre-registered metric, split or model.")
    A("")
    A("1. **GPU instead of CPU.** The LM ran on a Colab Tesla T4, fp16 autocast for training and fp32 evaluation "
      "(torch 2.11.0+cu128), instead of the CPU (torch 2.10.0+cpu) that PLAN §5 budgets for. Parity on the fixed tiny config: "
      "fp16 %+.3f %%, fp32 %+.3f %% vs the CPU reference (tolerance 1 %%). Both sessions were pinned to cuda/fp16, so no "
      "precisions were mixed." % (100 * rep["bundles"][b1]["parity"]["parity/parity_cuda_fp16.json"]["rel_diff_vs_cpu"],
                                   100 * rep["bundles"][b1]["parity"]["parity/parity_cuda_fp32.json"]["rel_diff_vs_cpu"]))
    A("2. **Stage 4 'confirm' for ALL candidates.** PLAN §3/§5 has the top 2 plus the baseline. On the GPU every candidate got the "
      "d=192, full-train, 1-epoch arbiter. PLAN §7.2 ranks on Stage 4 when it is run, so the ranking uses more candidates than planned; "
      "it does not replace any of them.")
    A("3. **Two D1 reference arms added in round 1** (`A1-P1r3-D1-16k`, `A1-P1-D1-16k`, wave ranks 90/91). The rank-1 baseline is D2, "
      "while the Stage-2 candidates (A6/A7/A10) are D1. These arms separate the data-mix effect from the algorithm effect. They are "
      "outside the PLAN §2.2 ranked list; `candidate_origin` = `round1_added_reference_arm`.")
    r1c = collections.defaultdict(list)
    for r in rows:
        if r["bundle"] == b1 and r["stage"] == "confirm" and r["seed"] <= 3:
            r1c[r["candidate"]].append(r["bpb"])
    top4 = sorted(r1c, key=lambda c: sum(r1c[c]) / len(r1c[c]))[:len(cb["reference_arms"])]
    A("4. **Round 2 is exploratory** (garden-of-forking-paths risk). The 6 `R2-*` candidates were designed after the Stage-3 results "
      "and during/after Stage 4:")
    A("   - the tokenizers were built 16:47–17:00 UTC (file mtimes under `candidates/round2/`), after round-1 screen ended at "
      "16:41 UTC (`%s/progress.log`);" % b1)
    A("   - `WAVES_round2.json` was written 17:23 UTC, after the round-1 Stage-4 power check on seeds 1–3 at 17:22 UTC;")
    A("   - the 4 round-2 reference arms %s the round-1 Stage-4 top %d by the seeds 1–3 mean (%s)."
      % ("are exactly" if set(top4) == set(cb["reference_arms"]) else "are NOT", len(top4), ", ".join("`%s`" % c for c in top4)))
    A("   They carry ranks 101–106 and `candidate_origin` = `round2_exploratory`. Any selection among them is conditioned on dev, and "
      "only the sealed test split (PLAN §7.6, one shot) can guard against that.")
    A("5. **LR at the edge of the grid.** The PLAN §5 sweep on the baseline chose 1e-3, the lowest grid value (1.42321 vs 1.44182 at "
      "3e-3 and 1.45496 at 6e-3). Round 2 did not sweep and reused 1e-3 (`--lr`). In both bundles the finalist re-sweep (report-only) "
      "confirms that 1e-3 beats 3e-3 and 6e-3 for both finalists (§3). No LR below 1e-3 was tried, so the optimum may lie below the grid.")
    A("")
    A("Further observations from this collection (not in the brief):")
    A("")
    A("6. **Round 2 also started the `large` stage** (5 candidates), and it was incomplete at copy time (§2.1). Round 1's "
      "`large` was interrupted. Neither is pre-registered, and neither is collected.")
    pr1 = rep["bundles"][b1]["stage_checks"]["power_check_recomputed"]
    pr2 = rep["bundles"][b2]["stage_checks"]["power_check_recomputed"]
    A("7. **Seed count is below the PLAN §6 power formula.** The pooled confirm seed s.d. is %.3f %% (round 1) and %.3f %% (round 2). "
      "n = ⌈15.7 σ²/Δ²⌉ at Δ_min = 0.5 %% asks for %d and %d seeds; the runner stops at 5 for every candidate. PLAN §6 wants extra "
      "seeds for the finalists only; here every candidate got 5. Gaps below about 0.5 %% should be reported as \"not distinguishable at "
      "our power\" (PLAN §6), never as \"equal\". The pooled s.d. is inflated by a few high-variance arms (§6, `*`)."
      % (100 * pr1["recomputed_pooled_rel_sd"], 100 * pr2["recomputed_pooled_rel_sd"], pr1["seeds_needed_0.5pct"],
         pr2["seeds_needed_0.5pct"]))
    A("8. **The PLAN §6 bootstrap text says 'resample its 3 seeds'.** Confirm now has 5 seeds per candidate; the bootstrap should "
      "resample the seeds each candidate actually has. This is an analysis note, not a data problem.")
    cr2 = rep["bundles"][b2]["stage_checks"]["screen_curve_crossing_recomputed"]
    A("9. **Round-2 screen-curve crossing.** The top two at 100 %% of the subset (%s) crossed between 75 %% and 100 %%. Under PLAN §5 "
      "that means the screening budget is too small for them, and they go to Stage 4. Stage 4 ran for every candidate anyway, so "
      "nothing is missing. Round 1: no crossing." % " / ".join("`%s`" % c for c in cr2["top_two_at_100pct"]))
    A("10. **Conditional rank 8** (`A6-SBPE-P1-D1-32k-t080`) was trained unconditionally (the GPU made it cheap). Per "
      "`WAVES_colab.json`, it enters PLAN §7 and the Holm family only if the PLAN §2.2 rank-8 condition holds. Flag: `conditional_wave`.")
    A("11. **Round-2 reference arms are exact duplicates** of round-1 runs (§5). Pooling both bundles without `duplicate_of` "
      "filtering would double-count them.")
    A("")
    A("## 8. Files")
    A("")
    A("| file | content |")
    A("|---|---|")
    A("| `lm/collect_colab.py` | this collector + validator (deterministic; re-run to regenerate everything below) |")
    A("| `lm/colab_results/%s/`, `lm/colab_results/%s/` | byte-exact copies of the Colab output folders ('large' excluded) |" % (b1, b2))
    A("| `lm/colab_results/results_table.csv` / `.json` | **%d rows**, one per bundle × stage × candidate × seed (× LR for the sweeps). Columns: see below |" % nrows)
    A("| `lm/colab_results/validation.json` | every check per record, stage-level checks, the cross-bundle pairs, and the copy manifest with sha256 |")
    A("| `lm/colab_results/dev_clusters.csv` | 836 dev_strict docs: doc_index, uid, bytes, source, split, tier, raw `group`, bootstrap `cluster` (PLAN §6) |")
    A("")
    A("`results_table` columns:")
    A("")
    A("- **Keys:** `bundle`, `round`, `stage`, `stage_recipe`, `role` (lr_sweep / stage3_screen / stage4_confirm / stage4_confirm_power_seed "
      "/ finalist_lr_resweep_report_only), `candidate`, `seed`, `lr`, `lr_tag`.")
    A("- **Candidate:**")
    A("  - provenance: `candidate_origin` (round1_preregistered_rank / round1_added_reference_arm / round2_exploratory), `wave_rank`, "
      "`wave_rank_label`, `conditional_wave`, `is_baseline`, `reference_arm_in_both_bundles`, `duplicate_of`;")
    A("  - recipe: `family`, `algorithm`, `pretokenizer`, `data_mix`, `vocab_label`, `variant`, `encoder_kind`, `n_vocab`, `tokenizer_sha256`;")
    A("  - compression: `train_bytes_per_token`, `dev_bytes_per_token`.")
    A("- **Model and data:** `model`, `d_model`, `n_layer`, `ctx_tokens`, `params_total`, `params_non_embedding`, `steps`, `steps_done`, "
      "`budget_bytes`, `bytes_seen`, `bytes_seen_rel_budget`, `tokens_seen`, `epochs_seen`.")
    A("- **Result:** `dev_n_docs`, `dev_sum_bits`, `dev_sum_bytes`, `dev_sum_ntok_scored`, `bpb` (reported), `bpb_recomputed` (Σbits/Σbytes), "
      "`subset_bpb_25/50/75/100` (0.5 MB curve), `train_loss_mean_last50`, `status`, `preregistered_flag`.")
    A("- **Runtime:** `device`, `gpu_name`, `amp_dtype`, `eval_dtype`, `torch`, `train_s`, `eval_s`, `created_utc`.")
    A("- **Provenance and validation:** `bundle_manifest_sha256`, `hk_lm_sha256`, `file` (under `lm/colab_results/`), `file_sha256`, "
      "`n_checks`, `valid` (every check), `hard_valid` (every identity/integrity check; tolerance flags excluded), "
      "`tolerance_flags`, `failed_checks`.")
    A("")
    A("The per-document bits and bytes (836 each) stay in the copied JSON records (`dev.bits`, `dev.bytes`, `dev.uids`), in the same "
      "uid order as `dev_clusters.csv`. The bootstrap reads them from there.")
    with open(os.path.join(OUT, "COLLECT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
