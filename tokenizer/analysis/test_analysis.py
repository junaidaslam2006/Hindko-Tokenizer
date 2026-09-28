"""One-shot TEST analysis (PLAN §7.6; AMENDMENT_1.md §4-§5): the final-test LM results under the DECISION.md bootstrap.

Reads:
  lm/colab_results/final_test/results_table_final_test.json + validation_final_test.json + the per-run records
      (written by lm/collect_final_test.py: byte-exact copies of G:\\My Drive\\hindko_lm_out_final_test\\<bundle>)
  data/test_strict.jsonl: only the uid / cluster / source fields (the text is never kept or used)
  lm/colab_results/results_table.json + the dev Stage 4 records (dev side of the dev-vs-test comparison, via decide.py)
  lm/colab_results/77e1368773fc/results/large__*.json (dev side of the report-only large comparison)
  analysis/decision.json, analysis/amendment1_stats.json (numbers of record, cross-checks), AMENDMENT_1.sha.json
It changes no decision: decision.json, DECISION.md and AMENDMENT_1.md are only read (AMENDMENT_1.md's sha256 is verified).

Method: decide.py's own code, imported unchanged (decide.Boot, decide.compare, decide.holm, decide.load_stage,
decide.dev_clusters): 10,000 replicates, numpy default_rng(12345); the 27 test clusters (manifest 'group', per-record web
groups collapsed to their site; the 'cluster' field of test_strict.jsonl) resampled with replacement within each source
(book 6, newspaper 15, web 6); each candidate's seeds resampled with replacement, independently per candidate;
bpb = sum(bits)/sum(bytes) over the resampled documents, averaged over the resampled seeds; RNG order = cluster draws
(sources sorted), then seed draws (candidates in sorted id order). Primary test bootstrap: the 6 confirm-scale test
candidates of both final-test bundles, 5 seeds each (the baseline, scored in both bundles bitwise identically, once).

Run: PYTHONIOENCODING=utf-8 python analysis/test_analysis.py      (single process; well under a minute)
"""
import collections
import csv
import datetime
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide as D0  # noqa: E402  (decide.py only defines functions and constants at import time)

ROOT = D0.ROOT
FT = os.path.join(ROOT, "lm", "colab_results", "final_test")
TABLE = os.path.join(FT, "results_table_final_test.json")
VALID = os.path.join(FT, "validation_final_test.json")
TEST_CLUSTERS_CSV = os.path.join(FT, "test_clusters.csv")
TEST_JSONL = os.path.join(ROOT, "data", "test_strict.jsonl")
TEST_JSONL_SHA = "a74c33ad008f4b0e4a6f7779b0de24e265b0f4edf6f94e89885c7943d3a08468"
DEV_LARGE_DIR = os.path.join(D0.RES, "77e1368773fc", "results")
DECISION_JSON = os.path.join(D0.OUT, "decision.json")
AMEND = os.path.join(D0.OUT, "AMENDMENT_1.md")
AMEND_SHA = os.path.join(D0.OUT, "AMENDMENT_1.sha.json")
AMEND_STATS = os.path.join(D0.OUT, "amendment1_stats.json")
EQUIV = os.path.join(ROOT, "release_build", "sp32k", "equivalence.json")
FINAL_TEST_LOG = os.path.join(ROOT, "FINAL_TEST_LOG.json")
OUT_JSON = os.path.join(D0.OUT, "test_results.json")

BASE = D0.BASELINE                      # A1-P1r3-D2-16k
A132 = "A1-P1r3-D2-32k"
A148 = "R2-A1-P1r3-D2-48k"
SP32 = "R2-A4-SPnat-D2-32k"
MG32 = "R2-A10-MinGram-P1r3-D2-32k"
MG48 = "R2-A10-MinGram-P1r3-D2-48k"
CANDS = sorted([BASE, A132, A148, SP32, MG32, MG48])      # fixed RNG order: sorted ids (as decide.py)
F54C = sorted([BASE, MG48, SP32, MG32])                  # the pre-registered test list (bundle f54c929ba1ab)
TOP = [SP32, MG48, MG32]                                 # pre-registered top set (DECISION.md §0)
SEEDS = (1, 2, 3, 4, 5)
LARGE_CANDS = sorted([BASE, MG48, SP32, MG32])
LARGE2 = sorted([BASE, MG48, SP32])                      # the three with both large test seeds
LARGE_LR = 5e-4
SENS_SEED = 12346                                        # parametric seed-noise variant only (as amendment1_stats.py)
ROLE = {MG48: "pre-registered pick (DECISION.md)", SP32: "released default (AMENDMENT_1.md)",
        MG32: "top-set member (Amendment 1 fallback)", A132: "round-1 confirmatory winner (report-only)",
        A148: "equal-vocabulary BPE control for the 48k pick (report-only)", BASE: "baseline (standard recipe)"}
SHORT = {MG48: "MinGram-48k", SP32: "SP-32k", MG32: "MinGram-32k", A132: "BPE-32k", A148: "BPE-48k", BASE: "baseline"}

# (id, section, a, b, what)   Delta = (bpb_a - bpb_b) / bpb_b ; negative = a is better
CONTRASTS = [
    ("a_prereg", "a", MG48, BASE, "PRE-REGISTERED (PLAN §7.6): the pre-registered pick vs the baseline"),
    ("b_released", "b", SP32, BASE, "the released default (Amendment 1) vs the baseline"),
    ("c_mingram32", "c", MG32, BASE, "MinGram-32k vs the baseline"),
    ("d_mg48_vs_r1", "d", MG48, A132, "forking paths: pre-registered pick vs the round-1 confirmatory winner"),
    ("d_sp32_vs_r1", "d", SP32, A132, "forking paths (and equal vocabulary 32k): released default vs BPE-32k"),
    ("d_mg32_vs_r1", "d", MG32, A132, "forking paths (and equal vocabulary 32k): MinGram-32k vs BPE-32k"),
    ("d_r1_vs_base", "d", A132, BASE, "the round-1 confirmatory winner vs the baseline (its own dev claim)"),
    ("e_mg48_vs_bpe48", "e", MG48, A148, "equal vocabulary 48k: pre-registered pick vs BPE-48k"),
    ("o_bpe48_vs_base", "o", A148, BASE, "BPE-48k vs the baseline"),
    ("o_bpe48_vs_bpe32", "o", A148, A132, "vocabulary size, BPE 32k -> 48k"),
    ("o_mg48_vs_mg32", "o", MG48, MG32, "vocabulary size, MinGram 32k -> 48k (top set)"),
    ("o_sp32_vs_mg48", "o", SP32, MG48, "top set: released default vs pre-registered pick"),
    ("o_sp32_vs_mg32", "o", SP32, MG32, "top set: released default vs MinGram-32k"),
]
REPORT_ONLY_FAMILY = ["b_released", "c_mingram32", "d_mg48_vs_r1", "d_sp32_vs_r1", "d_mg32_vs_r1", "d_r1_vs_base",
                      "e_mg48_vs_bpe48"]
LARGE_PAIRS = [(SP32, BASE), (MG32, BASE), (MG48, BASE), (SP32, MG32), (SP32, MG48), (MG48, MG32)]


def pkey(a, b):
    return "%s - %s" % (a, b)


def slim(c):
    keep = ("delta", "delta_pct", "ci95_pct", "ci90", "p", "p_plus1", "se_pct", "ci95_excludes_0", "tost_equivalent",
            "mde80_pct")
    return {k: c[k] for k in keep}


def pvalue(x):
    return min(1.0, 2 * min(int(np.sum(x <= 0)), int(np.sum(x >= 0))) / x.size)


def reps_mask(boot, mask):
    """decide.Boot.reps() restricted to the clusters in `mask` (a 0/1 vector over clusters); several sources allowed."""
    W = boot.W * np.asarray(mask, dtype=np.float64)[None, :]
    T = np.einsum("csk,rk->csr", boot.CB, W)
    num = np.einsum("crs,csr->cr", boot.M, T)
    return num / (boot.S * (W @ boot.KB))[None, :]


def point_mask(boot, mask):
    """Point bpb per candidate (mean over seeds of per-seed bpb) on the clusters in `mask`."""
    m = np.asarray(mask, dtype=np.float64)
    return ((boot.CB * m).sum(axis=2) / (boot.KB * m).sum()).mean(axis=1)


def index_clusters(labels, sources):
    names = sorted(set(labels))
    pos = {n: i for i, n in enumerate(names)}
    src_of = {}
    for lab, s in zip(labels, sources):
        assert src_of.setdefault(lab, s) == s
    return np.array([pos[x] for x in labels]), names, [src_of[n] for n in names]


# ----------------------------------------------------------------------------------------------- inputs
def load_test(checks):
    rows = D0.jload(TABLE)
    val = D0.jload(VALID)
    s = val["summary"]
    checks["collector_summary"] = {k: s[k] for k in ("n_records", "n_hard_valid", "n_valid_all_checks", "n_checks_total",
                                                     "bpb_bit_exact_all", "same_model_as_dev_all",
                                                     "collect_colab_validator_identical_on_all_confirm_records",
                                                     "cross_bundle", "cross_bundle_all_identical", "stage_complete")}
    assert s["n_hard_valid"] == s["n_records"] == len(rows) and s["same_model_as_dev_all"] and s["cross_bundle_all_identical"]
    conf = collections.defaultdict(dict)
    large = collections.defaultdict(dict)
    dups = []
    for r in rows:
        assert r["hard_valid"] is True, r["file"]
        p = os.path.join(D0.RES, r["file"])
        assert D0.sha256_file(p) == r["file_sha256"], p
        if r["duplicate_of"]:
            dups.append(r["file"])
            continue
        rec = D0.jload(p)
        dv = rec["dev"]
        bits = np.asarray(dv["bits"], dtype=np.float64)
        nb = np.asarray(dv["bytes"], dtype=np.float64)
        assert len(bits) == 491 and np.all(np.isfinite(bits)) and np.all(bits > 0)
        assert float(np.sum(bits) / np.sum(nb)) == rec["bpb"], p                 # bit-exact
        ent = {"bits": bits, "bytes": nb, "uids": list(dv["uids"]), "bpb": rec["bpb"], "file": r["file"],
               "created_utc": rec["created_utc"], "params": rec["model"]["params"], "ctx": rec["data"]["ctx_tokens"],
               "n_vocab": rec["model"]["n_vocab"],
               "dev_bpb_same_model": r["dev_bpb"], "curve": rec["curve"]}
        if r["stage"] == "confirm":
            assert rec["lr"] == 0.001 and rec["stage_recipe"] == "confirm" and rec["preregistered"] is True
            conf[r["candidate"]][int(r["seed"])] = ent
        elif r["stage"] == "large":
            assert rec["lr"] == LARGE_LR and rec["stage_recipe"] == "large" and rec["preregistered"] is False
            large[r["candidate"]][int(r["seed"])] = ent
    checks["duplicates_dropped"] = dups
    checks["duplicates_are_baseline_copies_in_94175d26497f"] = (
        len(dups) == 5 and all(d.startswith("final_test/94175d26497f/results/confirm__%s__" % BASE) for d in dups))
    assert checks["duplicates_are_baseline_copies_in_94175d26497f"]
    assert sorted(conf) == CANDS and all(sorted(conf[c]) == list(SEEDS) for c in CANDS)
    assert sorted(large) == LARGE_CANDS
    return conf, large, val


def test_clusters(uids, checks):
    meta = {}
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            meta[x["uid"]] = (x["cluster"], x["source"], x["group"])       # the text is not kept
    checks["test_jsonl_sha256"] = D0.sha256_file(TEST_JSONL)
    assert checks["test_jsonl_sha256"] == TEST_JSONL_SHA
    assert set(meta) == set(uids) and len(uids) == 491
    with open(TEST_CLUSTERS_CSV, encoding="utf-8") as f:
        cc = list(csv.DictReader(f))
    checks["test_clusters_csv_agrees"] = all(c["uid"] == u and c["cluster"] == meta[u][0] and c["source"] == meta[u][1]
                                             for c, u in zip(cc, uids)) and len(cc) == len(uids)
    assert checks["test_clusters_csv_agrees"]
    col = [meta[u][0] for u in uids]
    raw = [meta[u][2] for u in uids]
    src = [meta[u][1] for u in uids]
    return index_clusters(col, src), index_clusters(raw, src), src


def load_dev_large(checks, uids_ref, bytes_ref):
    out = collections.defaultdict(dict)
    for c in LARGE_CANDS:
        for s in (1, 2):
            p = os.path.join(DEV_LARGE_DIR, "large__%s__lr5e-4__s%d.json" % (c, s))
            rec = D0.jload(p)
            dv = rec["dev"]
            bits = np.asarray(dv["bits"], dtype=np.float64)
            assert rec["stage"] == "large" and rec["lr"] == LARGE_LR and rec["seed"] == s and rec["candidate"] == c
            assert list(dv["uids"]) == uids_ref and np.array_equal(np.asarray(dv["bytes"], dtype=np.float64), bytes_ref)
            assert float(np.sum(bits) / np.sum(bytes_ref)) == rec["bpb"]
            out[c][s] = {"bits": bits, "bpb": rec["bpb"]}
    checks["dev_large_records"] = {c: [os.path.relpath(os.path.join(DEV_LARGE_DIR, "large__%s__lr5e-4__s%d.json" % (c, s)),
                                                       D0.RES).replace("\\", "/") for s in (1, 2)] for c in LARGE_CANDS}
    return out


def decision_of_record(dec):
    """(a, b) -> confirm-scale comparison of record in decision.json (screen-stage entries skipped)."""
    look = {}
    best = dec["decision"]["best"]
    for r in dec["ranking"]:
        if r["vs_baseline"]:
            look[(r["id"], BASE)] = dict(r["vs_baseline"], where="decision.json ranking[%s].vs_baseline" % r["id"])
        if r["vs_best"]:
            look.setdefault((r["id"], best), dict(r["vs_best"], where="decision.json ranking[%s].vs_best" % r["id"]))

    def walk(x, path):
        if isinstance(x, dict):
            if "a" in x and "b" in x and "delta_pct" in x and "ci95_pct" in x and "screen" not in path:
                look.setdefault((x["a"], x["b"]), dict(x, where="decision.json " + path))
            for k, v in x.items():
                walk(v, path + "." + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + "[%d]" % i)
    walk(dec["contrasts"], "contrasts")
    walk(dec["stage3_vs_stage4"].get("confirm_vs_baseline", {}), "stage3_vs_stage4.confirm_vs_baseline")
    return look


# ----------------------------------------------------------------------------------------------- main
def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    checks = collections.OrderedDict()
    amend_rec = D0.jload(AMEND_SHA)
    checks["amendment_1_sha256_now"] = D0.sha256_file(AMEND)
    checks["amendment_1_unchanged"] = checks["amendment_1_sha256_now"] == amend_rec["sha256"]
    assert checks["amendment_1_unchanged"]
    dec = D0.jload(DECISION_JSON)
    checks["decision_json_sha256"] = D0.sha256_file(DECISION_JSON)
    checks["decide_py_sha256_now"] = D0.sha256_file(os.path.join(D0.OUT, "decide.py"))
    checks["decide_py_unchanged_since_decision"] = checks["decide_py_sha256_now"] == dec["code"]["decide.py"]
    assert checks["decide_py_unchanged_since_decision"]
    assert dec["decision"]["chosen"] == MG48 and dec["decision"]["top_set"] == [SP32, MG48, MG32]
    amst = D0.jload(AMEND_STATS)
    equiv = D0.jload(EQUIV)
    ftlog = D0.jload(FINAL_TEST_LOG)

    # ---------------- test records (validated by lm/collect_final_test.py; re-verified here)
    conf, large, val = load_test(checks)
    uids = conf[BASE][1]["uids"]
    nbytes = conf[BASE][1]["bytes"]
    for c in CANDS:
        for s in SEEDS:
            assert conf[c][s]["uids"] == uids and np.array_equal(conf[c][s]["bytes"], nbytes)
    for c in large:
        for s in large[c]:
            assert large[c][s]["uids"] == uids and np.array_equal(large[c][s]["bytes"], nbytes)
    (col_idx, col_names, col_src), (raw_idx, raw_names, raw_src), doc_src = test_clusters(uids, checks)
    cl_info = {"n_docs": len(uids), "n_bytes": int(nbytes.sum()), "n_clusters": len(col_names),
               "clusters_by_source": dict(collections.Counter(col_src)), "n_raw_groups": len(raw_names),
               "raw_groups_by_source": dict(collections.Counter(raw_src))}
    assert cl_info["n_clusters"] == 27 and cl_info["clusters_by_source"] == {"book": 6, "newspaper": 15, "web": 6}
    idx = {c: i for i, c in enumerate(CANDS)}

    # ---------------- (1) primary test bootstrap: 6 candidates x 5 seeds, decide.Boot as is
    BT = np.stack([np.stack([conf[c][s]["bits"] for s in SEEDS]) for c in CANDS])        # (6, 5, 491)
    bootT = D0.Boot(BT, nbytes, col_idx, col_src)
    repsT = bootT.reps()
    ptT = {c: float(bootT.point[idx[c]]) for c in CANDS}
    for c in CANDS:
        assert abs(ptT[c] - np.mean([conf[c][s]["bpb"] for s in SEEDS])) < 1e-12
    deltaT = D0.DELTA_REL * ptT[BASE]

    def CT(a, b, reps=repsT, point=bootT.point, ix=idx, delta=deltaT):
        return D0.compare(ix[a], ix[b], reps, point, delta)

    test_cmp = {cid: dict(CT(a, b), a=a, b=b, section=sec, what=w) for cid, sec, a, b, w in CONTRASTS}
    test_vs_base = {c: slim(CT(c, BASE)) for c in CANDS if c != BASE}
    holm_ro = D0.holm({k: test_cmp[k]["p"] for k in REPORT_ONLY_FAMILY})
    for k in REPORT_ONLY_FAMILY:
        test_cmp[k]["holm_p_report_only_family"] = holm_ro[k]

    orderT = sorted(CANDS, key=lambda c: ptT[c])
    ti = [idx[c] for c in TOP]
    am_top = np.argmin(repsT[ti], axis=0)
    rk_top = np.argsort(np.argsort(repsT[ti], axis=0), axis=0) + 1
    top_rank_dist = {c: {str(r): float(np.mean(rk_top[j] == r)) for r in (1, 2, 3)} for j, c in enumerate(TOP)}
    p_first_all = {c: float(np.mean(np.argmin(repsT, axis=0) == idx[c])) for c in CANDS}
    bestT = orderT[0]
    vs_bestT = {c: slim(CT(c, bestT)) for c in CANDS if c != bestT}
    holm_bestT = D0.holm({c: vs_bestT[c]["p"] for c in vs_bestT})
    # PLAN §7.3's top-set rule applied to the 6 test tokenizers: REPORT-ONLY (test never re-selects, PLAN §7.6)
    top_rule_T = [bestT] + [c for c in orderT[1:] if not (holm_bestT[c] <= D0.ALPHA and vs_bestT[c]["delta"] > 0)
                            or vs_bestT[c]["tost_equivalent"]]

    # per source (same replicates restricted to one source's clusters, as decide.py)
    per_source = collections.OrderedDict()
    for src in sorted(set(col_src)):
        rs = bootT.reps(source=src)
        ps = bootT.point_source(src)
        per_source[src] = {"n_clusters": int(sum(1 for s in col_src if s == src)),
                           "bytes": int(sum(bootT.KB[k] for k in range(len(col_src)) if col_src[k] == src)),
                           "mean_bpb": {c: float(ps[idx[c]]) for c in CANDS},
                           "contrasts": {cid: slim(D0.compare(idx[a], idx[b], rs, ps, deltaT))
                                         for cid, _, a, b, _ in CONTRASTS}}

    # leave-one-cluster-out (point estimates, seed-mean) and clusters where a is better
    CB, KB = bootT.CB, bootT.KB
    cl_bpb = CB.mean(axis=1) / KB[None, :]
    loco, cview = {}, {}
    for cid, _, a, b, _ in CONTRASTS:
        vals = []
        for k in range(len(KB)):
            m = np.ones(len(KB))
            m[k] = 0.0
            pa = float((CB[idx[a]] @ m).mean() / (KB @ m))
            pb = float((CB[idx[b]] @ m).mean() / (KB @ m))
            vals.append(100 * (pa - pb) / pb)
        full = test_cmp[cid]["delta_pct"]
        loco[cid] = {"min_pct": min(vals), "max_pct": max(vals),
                     "sign_flips": int(sum(1 for v in vals if (v < 0) != (full < 0))),
                     "cluster_at_min": col_names[int(np.argmin(vals))], "cluster_at_max": col_names[int(np.argmax(vals))]}
        better = cl_bpb[idx[a]] < cl_bpb[idx[b]]
        cview[cid] = {"n_clusters_a_better": int(np.sum(better)), "n_clusters": int(len(KB)),
                      "bytes_share_a_better": float(KB[better].sum() / KB.sum())}

    # ---------------- sensitivities of (a) and (b)
    sens = collections.OrderedDict()

    def sens_row(name, what, cands_s, bits_s, doc_cl, cl_src, pkey_="p"):
        ix = {c: i for i, c in enumerate(cands_s)}
        bt = D0.Boot(bits_s, nbytes, doc_cl, cl_src)
        rp = bt.reps()
        d_ = D0.DELTA_REL * float(bt.point[ix[BASE]])
        row = {"what": what, "candidates_rng_order": cands_s}
        for cid in ("a_prereg", "b_released", "c_mingram32"):
            a, b = next((a, b) for k, _, a, b, _ in CONTRASTS if k == cid)
            if a in ix and b in ix:
                row[cid] = slim(D0.compare(ix[a], ix[b], rp, bt.point, d_))
        sens[name] = row

    sens_row("bundle_f54c_only", "Boot over the 4 candidates of the pre-registered test bundle f54c929ba1ab only "
             "(its own seed-draw stream)", F54C, np.stack([BT[idx[c]] for c in F54C]), col_idx, col_src)
    sens_row("pair_only", "Boot over the pre-registered pair only (PLAN §7.6: the chosen tokenizer and the baseline)",
             sorted([BASE, MG48]), np.stack([BT[idx[c]] for c in sorted([BASE, MG48])]), col_idx, col_src)
    sens_row("raw_manifest_groups", "PLAN §6 sensitivity: manifest groups as they are (per-record web groups kept): "
             "%d groups" % len(raw_names), CANDS, BT, raw_idx, raw_src)
    sens_row("seeds_1to3", "only seeds 1-3 (the 3 seeds PLAN §5 planned)", CANDS, BT[:, :3, :], col_idx, col_src)
    sens["p_plus1"] = {"what": "p = 2 min((#{D*<=0}+1)/(R+1), (#{D*>=0}+1)/(R+1))",
                       **{cid: {"p_plus1": test_cmp[cid]["p_plus1"]} for cid in ("a_prereg", "b_released", "c_mingram32")}}

    # ---------------- (2) dev side: the same 6 candidates, decide.py's loaders, clusters and Boot
    rows_dev, c0 = D0.load_inputs()
    checks["dev_inputs"] = {k: v for k, v in c0.items()}
    conf_dev_all = D0.load_stage(rows_dev, "confirm", checks)
    confD = {c: conf_dev_all[c] for c in CANDS}
    uidsD = confD[BASE][1]["uids"]
    nbytesD = confD[BASE][1]["bytes"]
    for c in CANDS:
        assert sorted(confD[c]) == list(SEEDS)
        for s in SEEDS:
            assert confD[c][s]["uids"] == uidsD and np.array_equal(confD[c][s]["bytes"], nbytesD)
            # the test model is the dev model: same training (collector), and the dev bpb it carries is this record's
            assert conf[c][s]["dev_bpb_same_model"] == confD[c][s]["bpb"]
    clD = D0.dev_clusters(uidsD, nbytesD)
    assert clD["info"]["n_clusters"] == 32 and clD["info"]["agrees_with_dev_clusters_csv"]
    dcol_idx, dcol_names, dcol_src = clD["collapsed"]
    BD = np.stack([np.stack([confD[c][s]["bits"] for s in SEEDS]) for c in CANDS])        # (6, 5, 836)
    bootD = D0.Boot(BD, nbytesD, dcol_idx, dcol_src)
    repsD = bootD.reps()
    ptD = {c: float(bootD.point[idx[c]]) for c in CANDS}
    deltaD = D0.DELTA_REL * ptD[BASE]
    look = decision_of_record(dec)
    dev_cmp = {}
    xcheck = []
    for cid, _, a, b, _ in CONTRASTS:
        mine = D0.compare(idx[a], idx[b], repsD, bootD.point, deltaD)
        ent = {"recomputed": slim(mine), "of_record": None}
        rec = look.get((a, b))
        if rec is not None:
            assert abs(rec["delta_pct"] - mine["delta_pct"]) < 1e-9, (a, b)
            ent["of_record"] = {"delta_pct": rec["delta_pct"], "ci95_pct": rec["ci95_pct"], "p": rec["p"],
                                "where": rec["where"]}
            xcheck.append({"contrast": cid, "max_ci_endpoint_diff_pp": max(abs(mine["ci95_pct"][0] - rec["ci95_pct"][0]),
                                                                           abs(mine["ci95_pct"][1] - rec["ci95_pct"][1]))})
        dev_cmp[cid] = ent

    # ---------------- (3) joint dev+test bootstrap: Delta_test - Delta_dev, paired over the shared models (seeds)
    names_j = ["dev|" + n for n in dcol_names] + ["test|" + n for n in col_names]
    src_j = ["dev:" + s for s in dcol_src] + ["test:" + s for s in col_src]
    Kd = len(dcol_names)
    doc_j = np.concatenate([dcol_idx, col_idx + Kd])
    BJ = np.concatenate([BD, BT], axis=2)
    nbJ = np.concatenate([nbytesD, nbytes])
    bootJ = D0.Boot(BJ, nbJ, doc_j, src_j)
    checks["joint_dev_cluster_draws_equal_dev_bootstrap"] = bool(np.array_equal(bootJ.W[:, :Kd], bootD.W))
    assert checks["joint_dev_cluster_draws_equal_dev_bootstrap"]
    mD = np.array([1.0] * Kd + [0.0] * len(col_names))
    repsJD = reps_mask(bootJ, mD)
    repsJT = reps_mask(bootJ, 1.0 - mD)
    chk = reps_mask(bootJ, np.array([1.0 if s == "dev:book" else 0.0 for s in src_j]))
    assert np.allclose(chk, bootJ.reps(source="dev:book"), rtol=0, atol=1e-14)
    consistency = {}
    for cid, _, a, b, _ in CONTRASTS:
        rD = (repsJD[idx[a]] - repsJD[idx[b]]) / repsJD[idx[b]]
        rT = (repsJT[idx[a]] - repsJT[idx[b]]) / repsJT[idx[b]]
        x = 100 * (rT - rD)
        lo, hi = np.percentile(x, [2.5, 97.5])
        dpt = dev_cmp[cid]["recomputed"]["delta_pct"]
        tpt = test_cmp[cid]["delta_pct"]
        dci = (dev_cmp[cid]["of_record"] or dev_cmp[cid]["recomputed"])["ci95_pct"]
        tci = test_cmp[cid]["ci95_pct"]
        consistency[cid] = {
            "dev_pct": dpt, "test_pct": tpt, "change_pp": tpt - dpt, "change_ci95_pp": [float(lo), float(hi)],
            "p_change": pvalue(x), "change_ci95_excludes_0": bool(lo > 0 or hi < 0),
            "sign_agrees": bool((dpt < 0) == (tpt < 0)),
            "test_ci_excludes_0": bool(tci[0] > 0 or tci[1] < 0), "dev_ci_excludes_0": bool(dci[0] > 0 or dci[1] < 0),
            "test_point_inside_dev_ci95": bool(dci[0] <= tpt <= dci[1]),
            "dev_point_inside_test_ci95": bool(tci[0] <= dpt <= tci[1]),
            "ratio_test_over_dev": (tpt / dpt) if dpt != 0 else None}

    # ---------------- (4) report-only large test run (d=384, L=6, 2 epochs, LR 5e-4 fixed from dev)
    devL = load_dev_large(checks, uidsD, nbytesD)
    large_seeds = {c: sorted(large[c]) for c in LARGE_CANDS}
    assert large_seeds[MG32] == [1] and all(large_seeds[c] == [1, 2] for c in LARGE2)
    # 4a: the three candidates with 2 seeds, decide.Boot as is
    ix2 = {c: i for i, c in enumerate(LARGE2)}
    BL2 = np.stack([np.stack([large[c][s]["bits"] for s in (1, 2)]) for c in LARGE2])
    bootL2 = D0.Boot(BL2, nbytes, col_idx, col_src)
    repsL2 = bootL2.reps()
    ptL2 = {c: float(bootL2.point[ix2[c]]) for c in LARGE2}
    deltaL = D0.DELTA_REL * ptL2[BASE]
    L2 = {pkey(a, b): slim(D0.compare(ix2[a], ix2[b], repsL2, bootL2.point, deltaL))
          for a, b in LARGE_PAIRS if a in ix2 and b in ix2}
    # 4b: all four on seed 1 only (clusters resampled; no seed noise: 1 seed each)
    ix1 = {c: i for i, c in enumerate(LARGE_CANDS)}
    BL1 = np.stack([np.stack([large[c][1]["bits"]]) for c in LARGE_CANDS])
    bootL1 = D0.Boot(BL1, nbytes, col_idx, col_src)
    repsL1 = bootL1.reps()
    L1 = {pkey(a, b): slim(D0.compare(ix1[a], ix1[b], repsL1, bootL1.point, D0.DELTA_REL * float(bootL1.point[ix1[BASE]])))
          for a, b in LARGE_PAIRS}
    assert np.array_equal(bootL1.W, bootL2.W) and np.array_equal(bootL1.W, bootT.W)   # same test cluster draws
    # 4c: all available seeds (MinGram-32k n = 1) + parametric seed noise (amendment1_stats.py's 'own s.d.' variant)
    amc = amst["candidates"]
    sig = {c: max(amc[c]["large_seed_sd_rel_pct"], amc[c]["confirm_seed_sd_rel_pct"]) / 100.0 for c in LARGE_CANDS}
    nL = {c: len(large_seeds[c]) for c in LARGE_CANDS}
    CB1 = np.stack([np.mean([large[c][s]["bits"] for s in large_seeds[c]], axis=0) @ bootL1_onehot(col_idx, len(col_names))
                    for c in LARGE_CANDS])                                                    # (4, K)
    base_reps = (CB1 @ bootL1.W.T) / (bootL1.W @ bootL1.KB)[None, :]
    Z = np.random.default_rng(SENS_SEED).standard_normal(base_reps.shape)
    repsLP = base_reps * (1 + Z * np.array([sig[c] / math.sqrt(nL[c]) for c in LARGE_CANDS])[:, None])
    ptLall = {c: float(np.mean([large[c][s]["bpb"] for s in large_seeds[c]])) for c in LARGE_CANDS}
    pointLall = np.array([ptLall[c] for c in LARGE_CANDS])
    for c in LARGE_CANDS:
        assert abs(ptLall[c] - float(CB1[ix1[c]].sum() / bootL1.KB.sum())) < 1e-12
    LP = {pkey(a, b): slim(D0.compare(ix1[a], ix1[b], repsLP, pointLall, D0.DELTA_REL * ptLall[BASE]))
          for a, b in LARGE_PAIRS}
    am_top_LP = np.argmin(repsLP[[ix1[c] for c in TOP]], axis=0)
    p_first_top_LP = {c: float(np.mean(am_top_LP == j)) for j, c in enumerate(TOP)}
    # 4d: dev large (amendment1_stats.json, numbers of record) vs test large, and joint dev+test change CIs
    dev_large = {}
    for a, b in LARGE_PAIRS:
        rec = amst["large"]["vs_baseline"].get(a) if b == BASE else amst["large"]["pairs"].get(pkey(a, b))
        dev_large[pkey(a, b)] = {k: rec[k] for k in ("delta_pct", "ci95_pct", "p")}
    BJL2 = np.concatenate([np.stack([np.stack([devL[c][s]["bits"] for s in (1, 2)]) for c in LARGE2]), BL2], axis=2)
    bootJL2 = D0.Boot(BJL2, nbJ, doc_j, src_j)
    BJL1 = np.concatenate([np.stack([np.stack([devL[c][1]["bits"]]) for c in LARGE_CANDS]), BL1], axis=2)
    bootJL1 = D0.Boot(BJL1, nbJ, doc_j, src_j)
    large_consistency = {}
    for a, b in LARGE_PAIRS:
        if a in ix2 and b in ix2:
            bj, ixx, how = bootJL2, ix2, "joint dev+test bootstrap, seeds 1-2 resampled (shared models)"
        else:
            bj, ixx, how = bootJL1, ix1, "joint dev+test bootstrap on seed 1 only (MinGram-32k has 1 test seed; no seed noise)"
        rD_ = reps_mask(bj, mD)
        rT_ = reps_mask(bj, 1.0 - mD)
        x = 100 * ((rT_[ixx[a]] - rT_[ixx[b]]) / rT_[ixx[b]] - (rD_[ixx[a]] - rD_[ixx[b]]) / rD_[ixx[b]])
        lo, hi = np.percentile(x, [2.5, 97.5])
        pd_, pt_ = point_mask(bj, mD), point_mask(bj, 1.0 - mD)
        bd = 100 * (pd_[ixx[a]] - pd_[ixx[b]]) / pd_[ixx[b]]
        bt = 100 * (pt_[ixx[a]] - pt_[ixx[b]]) / pt_[ixx[b]]
        dpt = dev_large[pkey(a, b)]["delta_pct"]
        tpt = 100 * (ptLall[a] - ptLall[b]) / ptLall[b]
        large_consistency[pkey(a, b)] = {"dev_pct": dpt, "test_pct_all_seeds": tpt, "sign_agrees": bool((dpt < 0) == (tpt < 0)),
                                         "ci_basis_dev_pct": float(bd), "ci_basis_test_pct": float(bt),
                                         "ci_basis_change_pp": float(bt - bd), "ci_basis_sign_agrees": bool((bd < 0) == (bt < 0)),
                                         "change_ci95_pp": [float(lo), float(hi)], "p_change": pvalue(x), "how": how}
    large_summary = {c: {"seeds": large_seeds[c], "seed_bpb": [large[c][s]["bpb"] for s in large_seeds[c]],
                         "mean_bpb": ptLall[c], "n": nL[c],
                         "seed_sd_rel_pct": (100 * D0.rel_sd([large[c][s]["bpb"] for s in large_seeds[c]])
                                             if nL[c] > 1 else None),
                         "dev_large_mean_bpb": amc[c]["large_mean_bpb"], "dev_large_seed_bpb": amc[c]["large_seed_bpb"],
                         "sigma_rel_pct_used_in_parametric_variant": 100 * sig[c],
                         "params_total": large[c][1]["params"]["total"], "ctx_tokens": large[c][1]["ctx"],
                         "params_non_embedding": large[c][1]["params"]["non_embedding"],
                         "created_utc": [large[c][s]["created_utc"] for s in large_seeds[c]],
                         "vs_baseline_pct_all_seeds": (100 * (ptLall[c] - ptLall[BASE]) / ptLall[BASE]) if c != BASE else None}
                     for c in LARGE_CANDS}
    orderL = sorted(LARGE_CANDS, key=lambda c: ptLall[c])
    # 4e: confirm vs large on test, paired over clusters (amendment1_stats-style scale comparison; the three 2-seed arms)
    import amendment1_stats as A1S   # noqa: E402  (imported for joint_reps only)
    ixc = {c: i for i, c in enumerate(LARGE2)}
    (rL_j, rC_j), Wj = A1S.joint_reps([BL2, np.stack([BT[idx[c]] for c in LARGE2])], nbytes, col_idx, col_src)
    assert np.array_equal(rL_j, repsL2) and np.array_equal(Wj, bootL2.W)
    scale_test = {}
    for a, b in [(SP32, BASE), (MG48, BASE), (SP32, MG48)]:
        dL = (rL_j[ixc[a]] - rL_j[ixc[b]]) / rL_j[ixc[b]]
        dC = (rC_j[ixc[a]] - rC_j[ixc[b]]) / rC_j[ixc[b]]
        x = 100 * (dL - dC)
        lo, hi = np.percentile(x, [2.5, 97.5])
        pC = 100 * (ptT[a] - ptT[b]) / ptT[b]
        pL = 100 * (ptL2[a] - ptL2[b]) / ptL2[b]
        scale_test[pkey(a, b)] = {"confirm_pct": pC, "large_pct": pL, "change_pp": pL - pC,
                                  "change_ci95_pp": [float(lo), float(hi)], "p": pvalue(x)}

    # ---------------- (5) per-candidate table, seeds, power
    cand = {}
    for c in CANDS:
        seeds_t = [conf[c][s]["bpb"] for s in SEEDS]
        seeds_d = [confD[c][s]["bpb"] for s in SEEDS]
        cand[c] = {"role": ROLE[c], "short": SHORT[c], "test_mean_bpb": ptT[c], "test_seed_bpb": seeds_t,
                   "test_seed_sd_rel_pct": 100 * D0.rel_sd(seeds_t), "test_rank_of6": orderT.index(c) + 1,
                   "dev_mean_bpb": ptD[c], "dev_seed_bpb": seeds_d, "dev_seed_sd_rel_pct": 100 * D0.rel_sd(seeds_d),
                   "dev_rank_of6": sorted(CANDS, key=lambda x: ptD[x]).index(c) + 1,
                   "dev_rank_of17_decision": next(r["rank_primary"] for r in dec["ranking"] if r["id"] == c),
                   "test_vs_baseline": test_vs_base.get(c), "test_vs_best_of6": vs_bestT.get(c),
                   "test_holm_p_vs_best_of6": holm_bestT.get(c),
                   "params_total": conf[c][1]["params"]["total"], "ctx_tokens": conf[c][1]["ctx"],
                   "n_vocab": conf[c][1]["n_vocab"], "params_non_embedding": conf[c][1]["params"]["non_embedding"],
                   "test_bytes_per_token": next(r["test_bytes_per_token"] for r in D0.jload(TABLE)
                                                if r["candidate"] == c and r["stage"] == "confirm"),
                   "bundle": conf[c][1]["file"].split("/")[1], "files": [conf[c][s]["file"] for s in SEEDS],
                   "created_utc_range": [min(conf[c][s]["created_utc"] for s in SEEDS),
                                         max(conf[c][s]["created_utc"] for s in SEEDS)]}
    rank_corr = D0.kendall_spearman([ptD[c] for c in CANDS], [ptT[c] for c in CANDS])
    pooled = lambda cs, st: float(math.sqrt(np.mean([D0.rel_sd([st[c][s]["bpb"] for s in SEEDS]) ** 2 for c in cs])))
    power = {"pooled_rel_sd_pct_test": 100 * pooled(CANDS, conf), "pooled_rel_sd_pct_dev_same6": 100 * pooled(CANDS, confD),
             "n_needed_test": D0.n_needed(pooled(CANDS, conf)),
             "mde80_pct": {cid: test_cmp[cid]["mde80_pct"] for cid in test_cmp}}

    # ---------------- (6) the pre-registered claim (PLAN §7.5-7.6; FINAL_TEST.md §5)
    a_t, a_c = test_cmp["a_prereg"], consistency["a_prereg"]
    dev_rec = dec["decision"]["chosen_vs_baseline"]
    claim = {"comparison": "%s - %s" % (MG48, BASE),
             "dev_of_record": {"delta_pct": dev_rec["delta_pct"], "ci95_pct": dev_rec["ci95_pct"], "p": dev_rec["p"]},
             "test": slim(a_t), "test_ci95_excludes_0": a_t["ci95_excludes_0"], "sign_agrees_with_dev": a_c["sign_agrees"],
             "rule": "improvement claim on test iff the test 95 % CI excludes 0 and the test Delta has the dev sign "
                     "(PLAN §7.5-7.6; colab/FINAL_TEST.md §5); if the sign contradicts dev, report it and do not re-select"}
    claim["holds"] = bool(a_t["ci95_excludes_0"] and a_c["sign_agrees"] and a_t["delta_pct"] < 0)

    # ---------------- (7) timing facts for the deviations (from the records and FINAL_TEST_LOG.json)
    conf_times = [conf[c][s]["created_utc"] for c in CANDS for s in SEEDS]
    large_times = [large[c][s]["created_utc"] for c in LARGE_CANDS for s in large_seeds[c]]
    timing = {"final_test_log_builds_utc": [(e["utc"], e.get("completed_utc"), e["bundle_manifest_sha256"][:12])
                                            for e in ftlog["entries"]],
              "test_candidates_fixed_utc": dec["test_candidates_fixed"]["utc"],
              "amendment_1_written_utc": amend_rec["written_utc"],
              "first_test_lm_record_utc": min(conf_times), "last_confirm_test_record_utc": max(conf_times),
              "large_test_records_utc": [min(large_times), max(large_times)],
              "supp_bundle_built_before_first_test_number": ftlog["entries"][1]["completed_utc"] < min(conf_times),
              "amendment_1_before_first_test_number": amend_rec["written_utc"] < min(conf_times),
              "large_test_run_after_confirm_test_numbers": min(large_times) > max(conf_times)}
    assert timing["supp_bundle_built_before_first_test_number"] and timing["amendment_1_before_first_test_number"]

    ft_val = val["bundles"]
    out = collections.OrderedDict()
    out["what"] = ("One-shot TEST analysis (PLAN §7.6; AMENDMENT_1.md §4-§5): the final-test LM results (confirm scale: "
                   "d=192, full train, 1 epoch, LR 1e-3, seeds 1-5; plus the report-only large run) under the DECISION.md "
                   "hierarchical cluster bootstrap on the 27 strict-test clusters. Changes no decision.")
    out["generated_utc"] = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["code"] = {"test_analysis.py": D0.sha256_file(os.path.abspath(__file__)),
                   "decide.py (imported unchanged: Boot, compare, holm, load_stage, dev_clusters)": checks["decide_py_sha256_now"],
                   "amendment1_stats.py (imported: joint_reps)": D0.sha256_file(os.path.join(D0.OUT, "amendment1_stats.py")),
                   "lm/collect_final_test.py": D0.sha256_file(os.path.join(ROOT, "lm", "collect_final_test.py"))}
    out["inputs"] = checks
    out["validation"] = {"summary": val["summary"],
                         "bundles": {b: {k: v[k] for k in ("bundle_manifest_sha256", "role", "candidates", "bundle_identity_ok",
                                                           "bundle_test_docs_equal_test_strict", "parity",
                                                           "parity_report_pass_all", "stage_checks", "subset",
                                                           "final_test_log_utc", "final_test_log_completed_utc")}
                                     for b, v in ft_val.items()},
                         "progress_log_f54c": ft_val["f54c929ba1ab"]["progress_log"],
                         "cross_bundle_pairs": val["cross_bundle"]["pairs"],
                         "cross_bundle_parity": val["cross_bundle"]["parity_across_bundles"],
                         "run_all_local_sha256": val["run_all_local_sha256"]}
    out["release_equivalence"] = {k: equiv["verdict"][k] for k in ("dev_strict_encodings_changed", "test_strict_encodings_changed",
                                                                   "hf_equals_canonical_on_all_required_sets", "limit")}
    ts_eq = equiv["sets"]["test_strict"]
    sp_bundle = next(x for x in D0.jload(os.path.join(ROOT, "colab", "build_final_test", "staging", "manifest.json"))["candidates"]
                     if x["id"] == SP32)
    out["release_equivalence"]["test_strict"] = {k: ts_eq[k] for k in ("n", "hf_tokenizer_json_ids_equal", "transformers_ids_equal",
                                                                       "changed_vs_old_model", "canonical_tokens", "sha256")}
    out["release_equivalence"]["test_strict_tokens_equal_lm_bundle"] = ts_eq["canonical_tokens"] == sp_bundle["dev_tokens"]
    assert ts_eq["sha256"] == TEST_JSONL_SHA and out["release_equivalence"]["test_strict_tokens_equal_lm_bundle"]
    # the run_all.py that ran the report-only large test stage vs the bundled copy
    import difflib
    ra_b = open(os.path.join(ROOT, "colab", "build_final_test", "staging", "run_all.py"), encoding="utf-8").read().splitlines()
    ra_l = open(os.path.join(ROOT, "colab", "run_all.py"), encoding="utf-8").read().splitlines()
    diff = [ln for ln in difflib.unified_diff(ra_b, ra_l, "bundled run_all.py", "run_all.py used for 'large'", lineterm="", n=0)]
    out["validation"]["run_all_diff_bundled_vs_large_run"] = {
        "bundled_sha256": D0.sha256_file(os.path.join(ROOT, "colab", "build_final_test", "staging", "run_all.py")),
        "large_run_sha256": D0.sha256_file(os.path.join(ROOT, "colab", "run_all.py")),
        "changed_lines": [ln for ln in diff if ln[:1] in "+-" and not ln.startswith(("+++", "---"))], "unified_diff": diff}
    kbig = int(np.argmax(bootT.KB))
    cl_info["largest_cluster"] = {"cluster": col_names[kbig], "source": col_src[kbig], "bytes": int(bootT.KB[kbig]),
                                  "share": float(bootT.KB[kbig] / bootT.KB.sum())}
    # references used by the report (numbers of record from other files)
    dev_val = D0.jload(os.path.join(D0.RES, "validation.json"))
    stage_man = D0.jload(os.path.join(ROOT, "colab", "build_final_test", "staging", "manifest.json"))
    ns = amst["confirm_same5"]["per_source"]["newspaper"]
    out["references"] = {
        "dev_parity_rel_diff_vs_cpu": {b: {k: p["rel_diff_vs_cpu"] for k, p in v["parity"].items()}
                                       for b, v in dev_val["bundles"].items()},
        "test_parity_slice": {"n_docs": stage_man["dev_parity"]["n_docs"], "bytes": stage_man["dev_parity"]["bytes"]},
        "amendment1_dev_confirm_newspaper": {pkey(SP32, MG48): ns[pkey(SP32, MG48)]["delta_pct"],
                                             pkey(SP32, MG32): ns[pkey(SP32, MG32)]["delta_pct"],
                                             "source": "analysis/amendment1_stats.json confirm_same5.per_source.newspaper"},
        "amendment1_dev_scale_change": {k: amst["scale_change"]["interaction"][k] for k in
                                        (pkey(MG48, BASE), pkey(SP32, BASE), pkey(SP32, MG48))},
        "decision_k_ranked": dec["decision"]["k"], "decision_n_trained": len(dec["ranking"])}
    out["clusters"] = dict(cl_info, cluster_list=[{"cluster": n, "source": s, "bytes": int(bootT.KB[k]),
                                                   "docs": int(np.sum(col_idx == k))}
                                                  for k, (n, s) in enumerate(zip(col_names, col_src))])
    out["protocol"] = {"n_rep": D0.N_REP, "rng": "numpy.random.default_rng(%d)" % D0.RNG_SEED,
                       "candidate_rng_order": CANDS, "seeds": list(SEEDS),
                       "rng_order": "cluster draws (sources sorted: book, newspaper, web), then seed draws (candidates sorted)",
                       "delta_pct": "(bpb_a - bpb_b) / bpb_b per replicate; 95 % percentile CI",
                       "p_value": "2 min(P(D*<=0), P(D*>=0)); resolution 1e-4",
                       "tost_delta_abs_bpb_test": deltaT,
                       "joint_dev_test": "decide.Boot over dev (32 clusters) + test (27) with 6 strata (split x source): "
                                         "dev and test clusters resampled independently, each candidate's seeds drawn once "
                                         "and applied to both splits (the test models are the dev models); Delta_test - "
                                         "Delta_dev per replicate, in pp",
                       "report_only_family_holm": REPORT_ONLY_FAMILY}
    out["candidates"] = cand
    out["test_order"] = orderT
    out["dev_order_same6"] = sorted(CANDS, key=lambda c: ptD[c])
    out["rank_correlation_dev_vs_test_6"] = rank_corr
    out["pre_registered_claim"] = claim
    out["contrasts"] = {cid: {"section": sec, "a": a, "b": b, "what": w, "test": slim(test_cmp[cid]),
                              "test_holm_p_report_only_family": test_cmp[cid].get("holm_p_report_only_family"),
                              "dev": dev_cmp[cid], "dev_vs_test": consistency[cid], "per_source_test":
                                  {src: per_source[src]["contrasts"][cid] for src in per_source},
                              "leave_one_cluster_out_test": loco[cid], "cluster_view_test": cview[cid]}
                        for cid, sec, a, b, w in CONTRASTS}
    out["dev_crosscheck_vs_decision_json"] = {"n": len(xcheck), "max_ci_endpoint_diff_pp":
                                              max(x["max_ci_endpoint_diff_pp"] for x in xcheck), "items": xcheck}
    out["top_set_on_test"] = {"rank_distribution": top_rank_dist,
                              "p_first_within_top_set": {c: float(np.mean(am_top == j)) for j, c in enumerate(TOP)},
                              "p_first_of_6": p_first_all, "best_of6_by_mean": bestT, "vs_best_of6": vs_bestT,
                              "holm_vs_best_of6": holm_bestT,
                              "plan73_top_set_rule_on_test_report_only": top_rule_T,
                              "same_members_as_dev_top_set": sorted(top_rule_T) == sorted(TOP)}
    out["per_source_test"] = {src: {k: v for k, v in e.items() if k != "contrasts"} for src, e in per_source.items()}
    out["sensitivity_test"] = sens
    out["power"] = power
    out["large_test"] = {"what": "REPORT-ONLY large arbiter on test (d=384, L=6, H=6, 2 epochs, LR 5e-4 fixed from dev, no "
                                 "sweep on test); MinGram-32k has seed 1 only (seed 2 interrupted by a Colab disconnect at "
                                 "step 6588/7325, then the free GPU quota was exhausted)",
                         "candidates": large_summary, "order_by_mean": orderL,
                         "boot_2seeds_3cands": L2, "boot_seed1_4cands": L1,
                         "parametric_all_seeds": {"pairs": LP, "p_first_within_top_set": p_first_top_LP,
                                                  "what": "clusters resampled (the test cluster draws), each candidate's "
                                                          "available seeds averaged, plus relative seed noise N(0, sigma_c^2/n_c); "
                                                          "sigma_c = max(dev large 2-seed s.d., dev confirm 5-seed s.d.) from "
                                                          "amendment1_stats.json; normal draws from default_rng(%d)" % SENS_SEED},
                         "dev_vs_test": large_consistency, "dev_large_of_record": dev_large,
                         "confirm_vs_large_on_test": scale_test}
    out["timing"] = timing
    out["card_placeholders"] = card_values(out)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("wrote", OUT_JSON)
    print("test order:", [(c, round(ptT[c], 5)) for c in orderT])
    for cid, sec, a, b, w in CONTRASTS:
        t, d, k = test_cmp[cid], dev_cmp[cid]["recomputed"], consistency[cid]
        print("%-18s test %+.3f [%+.3f, %+.3f] p %.4f | dev %+.3f [%+.3f, %+.3f] | change %+.3f pp [%+.3f, %+.3f] sign %s"
              % (cid, t["delta_pct"], *t["ci95_pct"], t["p"], d["delta_pct"], *d["ci95_pct"], k["change_pp"],
                 *k["change_ci95_pp"], k["sign_agrees"]))
    print("claim holds:", claim["holds"])
    for k, v in sens.items():
        print("sens", k, {c: (round(x["delta_pct"], 3), [round(y, 3) for y in x["ci95_pct"]]) for c, x in v.items()
                          if isinstance(x, dict) and "ci95_pct" in x})
    print("large order:", [(c, round(ptLall[c], 5), nL[c]) for c in orderL])
    for k, v in L2.items():
        print(" L2", k, round(v["delta_pct"], 3), [round(x, 3) for x in v["ci95_pct"]], v["p"])
    for k, v in LP.items():
        print(" LP", k, round(v["delta_pct"], 3), [round(x, 3) for x in v["ci95_pct"]], v["p"])
    for k, v in large_consistency.items():
        print(" Ldt", k, round(v["dev_pct"], 3), round(v["test_pct_all_seeds"], 3), v["sign_agrees"],
              [round(x, 3) for x in v["change_ci95_pp"]])
    print("top-set rank dist:", top_rank_dist)
    print("done in %.1fs" % (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds())


def bootL1_onehot(doc_cluster, K):
    oh = np.zeros((len(doc_cluster), K))
    oh[np.arange(len(doc_cluster)), doc_cluster] = 1.0
    return oh


# ----------------------------------------------------------------------------------------------- card values
MINUS = chr(0x2212)


def fpct(x, nd=2):
    s = "%+.*f" % (nd, x)
    return s.replace("-", MINUS)


def fci(ci, nd=2):
    return "[%s, %s]" % (fpct(ci[0], nd), fpct(ci[1], nd))


def fp(p):
    return "<1e-4" if p == 0 else ("%.4f" % p)


def card_values(out):
    """Ready-to-paste strings for the test placeholders of F:/Hindko/tokenizer/README.md (source values, one rounding)."""
    c = out["candidates"]
    k = {MG48: "PREREG", SP32: "RELEASED", MG32: "ALT32", A132: "A1_32K", A148: "A1_48K", BASE: "BASELINE"}
    v = {}
    for cid, tag in k.items():
        v["TEST_BPB_" + tag] = "%.5f" % c[cid]["test_mean_bpb"]
        v["TEST_SD_" + tag] = "%.2f %%" % c[cid]["test_seed_sd_rel_pct"]
        if cid != BASE:
            t = c[cid]["test_vs_baseline"]
            v["TEST_DELTA_" + tag] = "%s %%" % fpct(t["delta_pct"])
            v["TEST_CI_" + tag] = fci(t["ci95_pct"])
            v["TEST_P_" + tag] = fp(t["p"])
    val = out["validation"]["summary"]
    v["TEST_CHECK_COMPLETE"] = ("yes: %d confirm records (6 tokenizers x seeds 1-5; the baseline in both bundles), status ok, "
                                "491 test documents each, bpb recomputed bit-exactly" % (val["by_stage"]["f54c929ba1ab/confirm"]
                                                                                    + val["by_stage"]["94175d26497f/confirm"]))
    v["TEST_CHECK_SAME_MODELS"] = ("yes: every test record's training-loss curve (and R3 embedding norms) equals its dev "
                                   "record's (%d/%d)" % (val["n_same_model_checked"], val["n_records"]))
    par = out["validation"]["bundles"]["f54c929ba1ab"]["parity"]
    v["TEST_CHECK_PARITY"] = ("pass in both bundles: fp16 %s %%, fp32 %s %% vs the CPU reference (tolerance 1 %%)"
                              % (fpct(100 * par["parity/parity_cuda_fp16.json"]["rel_diff_vs_cpu"], 3),
                                 fpct(100 * par["parity/parity_cuda_fp32.json"]["rel_diff_vs_cpu"], 3)))
    cl = out["pre_registered_claim"]
    v["TEST_SIGN_AGREES"] = "yes" if cl["sign_agrees_with_dev"] else "NO"
    v["TEST_IMPROVEMENT_CLAIM"] = (("yes: %s %s, p %s" % (fpct(cl["test"]["delta_pct"]) + " %", fci(cl["test"]["ci95_pct"]),
                                                          fp(cl["test"]["p"])))
                                   if cl["holds"] else "no")
    for cid, tag, other in (("o_sp32_vs_mg48", "TEST_RELEASED_VS_PREREG", "MinGram-48k"),
                            ("o_sp32_vs_mg32", "TEST_RELEASED_VS_ALT32", "MinGram-32k")):
        t = out["contrasts"][cid]["test"]
        v[tag] = "SP-32k vs %s %s %% %s, p %s: SP-32k's test bpb is %s; the 95 %% CI %s" % (
            other, fpct(t["delta_pct"]), fci(t["ci95_pct"]), fp(t["p"]), "higher" if t["delta_pct"] > 0 else "lower",
            "excludes 0" if t["ci95_excludes_0"] else "includes 0, so the difference is not significant")
    rk = c[SP32]["test_rank_of6"]
    rk_top = sorted(TOP, key=lambda x: c[x]["test_mean_bpb"]).index(SP32) + 1
    ordn = {1: "1st", 2: "2nd", 3: "3rd"}
    v["TEST_RELEASED_RANK"] = ("not claimable: at the confirm scale the released tokenizer is %s of the 6 test "
                               "tokenizers in mean test bpb (%s of the 3 top-set members)" % (ordn.get(rk, "%dth" % rk),
                                                                                             ordn[rk_top])
                               if rk != 1 else
                               "1st of 6 in mean test bpb at the confirm scale (still not claimable as 'best on test': "
                               "see the within-top-set CIs)")
    v["TEST_ANALYSIS_SCRIPT"] = "`analysis/test_analysis.py` (numbers: `analysis/test_results.json`; report: `analysis/TEST_RESULTS.md`)"
    L = out["large_test"]
    lc = L["candidates"]
    l2 = L["boot_2seeds_3cands"]
    lp = L["parametric_all_seeds"]["pairs"]
    v["TEST_LARGE_RESULT_OR_NOT_RUN"] = (
        "run, report-only (d=384, L=6, 2 epochs, LR 5e-4 fixed from dev; seeds 1-2, but MinGram-32k seed 1 only: its "
        "seed-2 run was cut by a Colab disconnect and the free GPU quota was then exhausted). Mean test bpb: SP-32k %.5f, "
        "MinGram-32k %.5f (n = 1), MinGram-48k %.5f, baseline %.5f. SP-32k vs baseline %s %% %s; SP-32k vs MinGram-48k "
        "%s %% %s (2 seeds each); SP-32k vs MinGram-32k %s %% with only 1 MinGram-32k seed: %s with seed noise added, "
        "p %s, not established" % (
            lc[SP32]["mean_bpb"], lc[MG32]["mean_bpb"], lc[MG48]["mean_bpb"], lc[BASE]["mean_bpb"],
            fpct(l2[pkey(SP32, BASE)]["delta_pct"]), fci(l2[pkey(SP32, BASE)]["ci95_pct"]),
            fpct(l2[pkey(SP32, MG48)]["delta_pct"]), fci(l2[pkey(SP32, MG48)]["ci95_pct"]),
            fpct(lp[pkey(SP32, MG32)]["delta_pct"]), fci(lp[pkey(SP32, MG32)]["ci95_pct"]), fp(lp[pkey(SP32, MG32)]["p"])))
    a = out["pre_registered_claim"]
    t = a["test"]
    d = a["dev_of_record"]
    if a["holds"]:
        v["CLAIM_TEST_PREREG"] = (
            "confirmed. R2-A10-MinGram-P1r3-D2-48k, chosen by the pre-registered rule and selected after an exploratory "
            "second round, improves on the standard BPE recipe A1-P1r3-D2-16k on the sealed strict test split, used once: "
            "Δ = %s %% bpb, 95 %% CI %s, p %s (confirm-scale arbiter, 5 seeds, hierarchical cluster bootstrap over 27 "
            "clusters); dev: %s %% %s" % (fpct(t["delta_pct"]), fci(t["ci95_pct"]), fp(t["p"]), fpct(d["delta_pct"]),
                                          fci(d["ci95_pct"])))
    else:
        v["CLAIM_TEST_PREREG"] = ("NOT confirmed on test: Δ = %s %% bpb, 95 %% CI %s (dev %s %% %s)"
                                  % (fpct(t["delta_pct"]), fci(t["ci95_pct"]), fpct(d["delta_pct"]), fci(d["ci95_pct"])))
    b = out["contrasts"]["b_released"]
    s1, s2 = out["contrasts"]["o_sp32_vs_mg48"]["test"], out["contrasts"]["o_sp32_vs_mg32"]["test"]
    assert max(TOP, key=lambda x: c[x]["test_mean_bpb"]) == SP32            # "highest ... of the three"
    assert not s1["ci95_excludes_0"] and not s2["ci95_excludes_0"]          # "neither CI excludes 0"
    v["CLAIM_TEST_RELEASED"] = (
        "at the confirm scale (5 seeds), R2-A4-SPnat-D2-32k vs the standard BPE recipe A1-P1r3-D2-16k: Δ = %s %% bpb, "
        "95 %% CI %s, p %s (dev %s %% %s). Within the top set it has the highest confirm-scale test bpb of the three "
        "(vs MinGram-48k %s %% %s; vs MinGram-32k %s %% %s; neither CI excludes 0). Report-only large arbiter on test: "
        "%s %% %s vs the baseline" % (
            fpct(b["test"]["delta_pct"]), fci(b["test"]["ci95_pct"]), fp(b["test"]["p"]),
            fpct(b["dev"]["of_record"]["delta_pct"]), fci(b["dev"]["of_record"]["ci95_pct"]),
            fpct(s1["delta_pct"]), fci(s1["ci95_pct"]), fpct(s2["delta_pct"]), fci(s2["ci95_pct"]),
            fpct(l2[pkey(SP32, BASE)]["delta_pct"]), fci(l2[pkey(SP32, BASE)]["ci95_pct"])))
    return v


if __name__ == "__main__":
    main()
