"""Statistics behind Amendment 1 (report-only): the 'large' arbiter's dev_strict results under the DECISION.md bootstrap.

Reads only dev_strict per-document results (never the test split, never a test LM result):
  lm/colab_results/77e1368773fc/results/large__*.json, large_lrsweep__*.json  (copied from the Colab Drive folder
      G:\\My Drive\\hindko_lm_out\\77e1368773fc\\results; the copies are compared with the Drive files by sha256)
  lm/colab_results/results_table.json + the Stage 4 'confirm' records it lists (numbers of record, scale comparison)
  analysis/decision.json (cross-checks against the numbers of record)
  splits/split_manifest.jsonl (clusters, via decide.dev_clusters)
It does not write AMENDMENT_1.md (hashed); it only verifies that file's sha256.

Method: the bootstrap is decide.py's own code (decide.Boot, decide.compare, decide.holm, decide.dev_clusters), so it is
the DECISION.md bootstrap: 10,000 replicates, numpy default_rng(12345), 32 clusters resampled with replacement within
each source (book, newspaper, web), each candidate's seeds resampled with replacement, independently per candidate;
bpb = sum(bits)/sum(bytes) over the resampled documents, averaged over the resampled seeds; RNG order = cluster draws
(sources sorted), then seed draws (candidates in sorted id order). Here: the 5 large candidates x 2 seeds.

Run: PYTHONIOENCODING=utf-8 python analysis/amendment1_stats.py      (single process; a few seconds)
"""
import collections
import datetime
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide as D0  # noqa: E402  (decide.py only defines functions and constants at import time)

BUNDLE = "77e1368773fc"
LARGE_DIR = os.path.join(D0.RES, BUNDLE, "results")
DRIVE_DIR = r"G:\My Drive\hindko_lm_out\77e1368773fc\results"
OUT_JSON = os.path.join(D0.OUT, "amendment1_stats.json")
AMEND = os.path.join(D0.OUT, "AMENDMENT_1.md")
AMEND_SHA = os.path.join(D0.OUT, "AMENDMENT_1.sha.json")
DECISION_JSON = os.path.join(D0.OUT, "decision.json")

BASE = D0.BASELINE                      # A1-P1r3-D2-16k
A132 = "A1-P1r3-D2-32k"
SP32 = "R2-A4-SPnat-D2-32k"
MG32 = "R2-A10-MinGram-P1r3-D2-32k"
MG48 = "R2-A10-MinGram-P1r3-D2-48k"
CANDS = sorted([BASE, A132, SP32, MG32, MG48])      # fixed RNG order: sorted candidate ids (as decide.py)
TOP = [SP32, MG48, MG32]                            # pre-registered top set (DECISION.md §0), confirm-scale order
LARGE_SEEDS = (1, 2)
LARGE_LR = 5e-4
CONF_SEEDS = (1, 2, 3, 4, 5)
SENS_SEED = 12346                                   # separate stream, only for the parametric seed-noise sensitivities

# orientation: (a, b) means Delta = (bpb_a - bpb_b) / bpb_b ; negative = a is better
PAIRS = [(SP32, MG32), (SP32, MG48), (MG48, MG32),
         (SP32, A132), (MG32, A132), (MG48, A132),
         (A132, BASE), (SP32, BASE), (MG32, BASE), (MG48, BASE)]
TOP_PAIRS = PAIRS[:3]


def pkey(a, b):
    return "%s - %s" % (a, b)


def rel_sd(v):
    return D0.rel_sd(v)


# ----------------------------------------------------------------------------------------------- inputs
def load_large(checks, uids_ref, bytes_ref):
    """The 10 large records (5 candidates x seeds 1, 2) plus the baseline's 3 large LR-sweep records."""
    want = {("large__%s__lr5e-4__s%d.json" % (c, s)): (c, s) for c in CANDS for s in LARGE_SEEDS}
    present = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LARGE_DIR, "large__*.json")))
    checks["large_files_present"] = present
    checks["large_file_set_is_exactly_5x2"] = sorted(want) == present
    assert checks["large_file_set_is_exactly_5x2"], present
    drive_ok = os.path.isdir(DRIVE_DIR)
    out = collections.defaultdict(dict)
    files = []
    for fn, (c, s) in sorted(want.items()):
        p = os.path.join(LARGE_DIR, fn)
        sha = D0.sha256_file(p)
        drive_sha = D0.sha256_file(os.path.join(DRIVE_DIR, fn)) if drive_ok else None
        rec = D0.jload(p)
        m, d, dev = rec["model"], rec["data"], rec["dev"]
        assert rec["stage"] == "large" and rec["stage_recipe"] == "large" and rec["status"] == "ok", fn
        assert rec["preregistered"] is False, fn
        assert rec["candidate"] == c and rec["seed"] == s and rec["lr"] == LARGE_LR, fn
        assert (m["name"], m["d"], m["L"], m["H"]) == ("large", 384, 6, 6), fn
        assert d["steps_done"] == d["steps"] and d["epochs"] == 2, fn
        assert rec["bundle_manifest_sha256"].startswith(BUNDLE), fn
        bits = np.asarray(dev["bits"], dtype=np.float64)
        nb = np.asarray(dev["bytes"], dtype=np.float64)
        assert dev["n_docs"] == 836 and len(bits) == 836 and np.all(np.isfinite(bits)) and np.all(bits > 0), fn
        assert float(np.sum(bits) / np.sum(nb)) == rec["bpb"] == dev["bpb"], fn      # bit-exact recomputation
        assert list(dev["uids"]) == uids_ref and np.array_equal(nb, bytes_ref), fn   # same 836 docs, same order
        out[c][s] = {"bits": bits, "bpb": rec["bpb"], "rec": rec}
        files.append({"file": "lm/colab_results/%s/results/%s" % (BUNDLE, fn), "sha256": sha,
                      "equals_drive_copy": (drive_sha == sha) if drive_ok else None,
                      "candidate": c, "seed": s, "bpb": rec["bpb"], "created_utc": rec["created_utc"],
                      "hk_lm_sha256": rec["hk_lm_sha256"], "bundle_manifest_sha256": rec["bundle_manifest_sha256"],
                      "bytes_seen_rel_budget": d["bytes_seen"] / d["budget_bytes"], "epochs_seen": d["epochs_seen"]})
    if drive_ok:
        assert all(f["equals_drive_copy"] for f in files)
    checks["large_records"] = files
    checks["large_records_equal_drive_copies"] = all(f["equals_drive_copy"] for f in files) if drive_ok else None
    checks["large_hk_lm_sha256"] = sorted({f["hk_lm_sha256"] for f in files})
    checks["large_created_utc_range"] = [min(f["created_utc"] for f in files), max(f["created_utc"] for f in files)]

    sweep = []
    for p in sorted(glob.glob(os.path.join(LARGE_DIR, "large_lrsweep__*.json"))):
        rec = D0.jload(p)
        assert rec["stage"] == "large_lrsweep" and rec["candidate"] == BASE and rec["seed"] == 1 and rec["status"] == "ok"
        bits = np.asarray(rec["dev"]["bits"], dtype=np.float64)
        assert float(np.sum(bits) / np.sum(bytes_ref)) == rec["bpb"]
        ent = {"file": "lm/colab_results/%s/results/%s" % (BUNDLE, os.path.basename(p)),
               "sha256": D0.sha256_file(p), "lr": rec["lr"], "bpb": rec["bpb"]}
        if rec["lr"] == LARGE_LR:
            ent["per_doc_bits_identical_to_large_s1"] = bool(np.array_equal(bits, out[BASE][1]["bits"]))
        sweep.append(ent)
    sweep.sort(key=lambda e: e["lr"])
    assert min(sweep, key=lambda e: e["bpb"])["lr"] == LARGE_LR
    return out, sweep


# ----------------------------------------------------------------------------------------------- bootstrap helpers
def joint_reps(bits_list, nbytes, doc_cluster, cl_source, n_rep=D0.N_REP, seed=D0.RNG_SEED):
    """decide.Boot's RNG consumption, extended: cluster draws once, then the seed draws of each block in turn.

    Block 0 reproduces decide.Boot(bits_list[0], ...) exactly (asserted by the caller); block 1 then shares the
    replicate's cluster draw, so a Delta at one scale and a Delta at the other are paired over documents."""
    Cs = [b.shape[0] for b in bits_list]
    K = len(cl_source)
    rng = np.random.default_rng(seed)
    W = np.zeros((n_rep, K), dtype=np.float64)
    rows = np.arange(n_rep)
    for src in sorted(set(cl_source)):
        idx = np.array([k for k in range(K) if cl_source[k] == src])
        draws = idx[rng.integers(0, len(idx), size=(n_rep, len(idx)))]
        for j in range(len(idx)):
            np.add.at(W, (rows, draws[:, j]), 1.0)
    D = bits_list[0].shape[2]
    onehot = np.zeros((D, K))
    onehot[np.arange(D), doc_cluster] = 1.0
    KB = nbytes @ onehot
    out = []
    for bits, C in zip(bits_list, Cs):
        S = bits.shape[1]
        M = np.zeros((C, n_rep, S), dtype=np.float64)
        for c in range(C):
            d = rng.integers(0, S, size=(n_rep, S))
            for j in range(S):
                np.add.at(M[c], (rows, d[:, j]), 1.0)
        CB = bits @ onehot
        T = np.einsum("csk,rk->csr", CB, W)
        num = np.einsum("crs,csr->cr", M, T)
        out.append(num / (S * (W @ KB))[None, :])
    return out, W


def pvalue(x):
    return min(1.0, 2 * min(int(np.sum(x <= 0)), int(np.sum(x >= 0))) / x.size)


def rel_series(reps, ia, ib):
    return (reps[ia] - reps[ib]) / reps[ib]


def slim(cmp_):
    keep = ("delta", "delta_pct", "ci95_pct", "ci90", "p", "p_plus1", "se_pct", "ci95_excludes_0", "tost_equivalent",
            "mde80_pct")
    return {k: cmp_[k] for k in keep}


def ranks(pt, cands):
    order = sorted(cands, key=lambda c: pt[c])
    return {c: order.index(c) + 1 for c in cands}, order


# ----------------------------------------------------------------------------------------------- main
def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    checks = collections.OrderedDict()

    # AMENDMENT_1.md is hashed: verify only, never write
    amend_rec = D0.jload(AMEND_SHA)
    amend_sha_now = D0.sha256_file(AMEND)
    checks["amendment_1_sha256_recorded"] = amend_rec["sha256"]
    checks["amendment_1_sha256_now"] = amend_sha_now
    checks["amendment_1_unchanged"] = amend_sha_now == amend_rec["sha256"]
    assert checks["amendment_1_unchanged"]
    dec = D0.jload(DECISION_JSON)
    checks["decide_py_sha256_now"] = D0.sha256_file(os.path.join(D0.OUT, "decide.py"))
    checks["decide_py_unchanged_since_decision"] = checks["decide_py_sha256_now"] == dec["code"]["decide.py"]

    # Stage 4 'confirm' records of record (the same loader and checks as decide.py)
    rows, c0 = D0.load_inputs()
    checks.update({k: v for k, v in c0.items()})
    conf_all = D0.load_stage(rows, "confirm", checks)
    conf = {c: conf_all[c] for c in CANDS}
    for c in CANDS:
        assert sorted(conf[c]) == list(CONF_SEEDS), c
    uids = conf[BASE][1]["uids"]
    nbytes = conf[BASE][1]["bytes"]
    for c in CANDS:
        for s in CONF_SEEDS:
            assert conf[c][s]["uids"] == uids and np.array_equal(conf[c][s]["bytes"], nbytes)
    checks["confirm_files_of_record"] = {c: [conf[c][s]["row"]["file"] for s in CONF_SEEDS] for c in CANDS}

    large, sweep = load_large(checks, uids, nbytes)

    cl = D0.dev_clusters(uids, nbytes)
    assert cl["info"]["n_clusters"] == 32 and cl["info"]["agrees_with_dev_clusters_csv"]
    col_idx, col_names, col_src = cl["collapsed"]
    idx = {c: i for i, c in enumerate(CANDS)}

    BL = np.stack([np.stack([large[c][s]["bits"] for s in LARGE_SEEDS]) for c in CANDS])     # (5, 2, 836)
    BC = np.stack([np.stack([conf[c][s]["bits"] for s in CONF_SEEDS]) for c in CANDS])       # (5, 5, 836)

    # ---------------- primary: large arbiter, decide.Boot as is
    bootL = D0.Boot(BL, nbytes, col_idx, col_src)
    repsL = bootL.reps()
    ptL = {c: float(bootL.point[idx[c]]) for c in CANDS}
    for c in CANDS:
        assert abs(ptL[c] - np.mean([large[c][s]["bpb"] for s in LARGE_SEEDS])) < 1e-12
    deltaL = D0.DELTA_REL * ptL[BASE]

    def CL(a, b, reps=repsL, point=bootL.point, delta=deltaL):
        x = D0.compare(idx[a], idx[b], reps, point, delta)
        x.update({"a": a, "b": b})
        return x

    vs_base = {c: CL(c, BASE) for c in CANDS if c != BASE}
    pairs = {pkey(a, b): CL(a, b) for a, b in PAIRS}
    rkL, orderL = ranks(ptL, CANDS)
    bestL = orderL[0]
    vs_best = {c: CL(c, bestL) for c in CANDS if c != bestL}
    holm_best = D0.holm({c: vs_best[c]["p"] for c in vs_best})
    holm_top = D0.holm({pkey(a, b): pairs[pkey(a, b)]["p"] for a, b in TOP_PAIRS})

    # how often each member is the lowest (within the top set; within all 5)
    ti = [idx[c] for c in TOP]
    am_top = np.argmin(repsL[ti], axis=0)
    p_first_top = {c: float(np.mean(am_top == j)) for j, c in enumerate(TOP)}
    am_all = np.argmin(repsL, axis=0)
    p_first_all = {c: float(np.mean(am_all == idx[c])) for c in CANDS}
    top_rank_dist = {}
    rk_top = np.argsort(np.argsort(repsL[ti], axis=0), axis=0) + 1
    for j, c in enumerate(TOP):
        top_rank_dist[c] = {str(r): float(np.mean(rk_top[j] == r)) for r in (1, 2, 3)}

    # per source (the same replicates restricted to one source's clusters, as decide.py)
    per_source = collections.OrderedDict()
    for src in sorted(set(col_src)):
        rs = bootL.reps(source=src)
        ps = bootL.point_source(src)
        ent = {"n_clusters": int(sum(1 for s in col_src if s == src)),
               "bytes": int(sum(bootL.KB[k] for k in range(len(col_src)) if col_src[k] == src)),
               "mean_bpb": {c: float(ps[idx[c]]) for c in CANDS}, "pairs": {}}
        for a, b in PAIRS:
            ent["pairs"][pkey(a, b)] = slim(D0.compare(idx[a], idx[b], rs, ps, deltaL))
        per_source[src] = ent

    # cluster-level view and leave-one-cluster-out (point estimates; seed-mean)
    CB = bootL.CB                               # (C, S, K)
    KB = bootL.KB                               # (K,)
    cl_bpb = CB.mean(axis=1) / KB[None, :]      # (C, K) seed-mean bpb per cluster
    cluster_view = {}
    for a, b in TOP_PAIRS + [(SP32, BASE), (MG32, BASE), (MG48, BASE)]:
        better = cl_bpb[idx[a]] < cl_bpb[idx[b]]
        cluster_view[pkey(a, b)] = {"n_clusters_a_better": int(np.sum(better)), "n_clusters": int(len(KB)),
                                    "bytes_share_a_better": float(KB[better].sum() / KB.sum())}
    loco = {}
    for a, b in PAIRS:
        vals = []
        for k in range(len(KB)):
            m = np.ones(len(KB))
            m[k] = 0.0
            pa = float((CB[idx[a]] @ m).mean() / (KB @ m))
            pb = float((CB[idx[b]] @ m).mean() / (KB @ m))
            vals.append(100 * (pa - pb) / pb)
        full = pairs[pkey(a, b)]["delta_pct"]
        loco[pkey(a, b)] = {"min_pct": min(vals), "max_pct": max(vals),
                            "sign_flips": int(sum(1 for v in vals if (v < 0) != (full < 0))),
                            "cluster_at_min": col_names[int(np.argmin(vals))],
                            "cluster_at_max": col_names[int(np.argmax(vals))]}
    loco_order = {}
    for k in range(len(KB)):
        m = np.ones(len(KB))
        m[k] = 0.0
        p_k = {c: float((CB[idx[c]] @ m).mean() / (KB @ m)) for c in TOP}
        loco_order[col_names[k]] = sorted(TOP, key=lambda c: p_k[c])
    loco_top_order_counts = collections.Counter(" < ".join(v) for v in loco_order.values())

    # ---------------- sensitivity to the 2-seed design (same cluster draws W)
    S = BL.shape[1]
    T = np.einsum("csk,rk->csr", CB, bootL.W)                            # (C, S, R)
    base_reps = T.mean(axis=1) / (bootL.W @ KB)[None, :]                   # clusters resampled, seeds averaged
    shrink = {"n_seeds": S, "variance_factor": (S - 1) / S, "se_factor": math.sqrt((S - 1) / S),
              "note": "resampling n seeds with replacement gives the seed mean a variance of (n-1)/n x s^2/n"}
    sens = collections.OrderedDict()
    seed_sd_large = {c: rel_sd([large[c][s]["bpb"] for s in LARGE_SEEDS]) for c in CANDS}
    seed_sd_conf = {c: rel_sd([conf[c][s]["bpb"] for s in CONF_SEEDS]) for c in CANDS}
    pooled18 = dec["power"]["stage4_5seeds_pooled"]["all18"]["pooled_rel_sd_pct"] / 100.0
    rngP = np.random.default_rng(SENS_SEED)
    Z = rngP.standard_normal(base_reps.shape)
    variants = collections.OrderedDict()
    variants["rescaled_seed_bootstrap"] = {
        "reps": base_reps + math.sqrt(S / (S - 1)) * (repsL - base_reps),
        "what": "seed deviation of each replicate scaled by sqrt(n/(n-1)) = sqrt(2), undoing the (n-1)/n shrink"}
    sig_b = np.array([max(seed_sd_large[c], seed_sd_conf[c]) for c in CANDS])
    variants["parametric_own_sd"] = {
        "reps": base_reps * (1 + Z * (sig_b / math.sqrt(S))[:, None]),
        "sigma_rel_pct": {c: 100 * float(sig_b[idx[c]]) for c in CANDS},
        "what": "clusters resampled, seeds averaged, plus N(0, sigma_c^2/2) relative seed noise per candidate; sigma_c "
                "= max(its large 2-seed s.d., its confirm 5-seed s.d.); normal draws from default_rng(%d)" % SENS_SEED}
    variants["parametric_pooled18_stress"] = {
        "reps": base_reps * (1 + Z * (pooled18 / math.sqrt(S))),
        "sigma_rel_pct": 100 * pooled18,
        "what": "as above with sigma = %.3f %% for every candidate: the pooled confirm 5-seed s.d. over all 18 "
                "candidates (DECISION.md §7.6), inflated by the two SentencePiece D1 arms; a stress test"
                % (100 * pooled18)}
    for name, v in variants.items():
        r = v.pop("reps")
        ent = dict(v)
        ent["pairs"] = {pkey(a, b): slim(D0.compare(idx[a], idx[b], r, bootL.point, deltaL)) for a, b in PAIRS}
        am = np.argmin(r[ti], axis=0)
        ent["p_first_within_top_set"] = {c: float(np.mean(am == j)) for j, c in enumerate(TOP)}
        sens[name] = ent

    # ---------------- confirm scale, same 5 candidates, paired with the large replicates over documents
    (repsL_joint, repsC), Wj = joint_reps([BL, BC], nbytes, col_idx, col_src)
    checks["joint_block0_equals_primary_large_bootstrap"] = bool(np.array_equal(repsL_joint, repsL) and
                                                                np.array_equal(Wj, bootL.W))
    assert checks["joint_block0_equals_primary_large_bootstrap"]
    ptC = {c: float(np.mean([conf[c][s]["bpb"] for s in CONF_SEEDS])) for c in CANDS}
    pointC = np.array([ptC[c] for c in CANDS])
    deltaC = D0.DELTA_REL * ptC[BASE]
    rkC, orderC = ranks(ptC, CANDS)
    conf_pairs = {pkey(a, b): slim(D0.compare(idx[a], idx[b], repsC, pointC, deltaC)) for a, b in PAIRS}
    interaction = {}
    for a, b in PAIRS:
        dL = rel_series(repsL, idx[a], idx[b])
        dC = rel_series(repsC, idx[a], idx[b])
        x = 100 * (dL - dC)
        pL = 100 * (ptL[a] - ptL[b]) / ptL[b]
        pC = 100 * (ptC[a] - ptC[b]) / ptC[b]
        lo, hi = np.percentile(x, [2.5, 97.5])
        interaction[pkey(a, b)] = {"confirm_pct": pC, "large_pct": pL, "change_pp": pL - pC,
                                   "change_ci95_pp": [float(lo), float(hi)], "p": pvalue(x),
                                   "ci95_excludes_0": bool(lo > 0 or hi < 0)}

    # cross-check the recomputed confirm numbers against decision.json (numbers of record, 18-candidate stream)
    rk = {r["id"]: r for r in dec["ranking"]}
    xc = []

    def xcheck(label, a, b, rec):
        mine = D0.compare(idx[a], idx[b], repsC, pointC, deltaC)
        xc.append({"what": label, "a": a, "b": b, "decision_delta_pct": rec["delta_pct"],
                   "recomputed_delta_pct": mine["delta_pct"], "decision_ci95_pct": rec["ci95_pct"],
                   "recomputed_ci95_pct": mine["ci95_pct"],
                   "max_ci_endpoint_diff_pp": max(abs(mine["ci95_pct"][0] - rec["ci95_pct"][0]),
                                                  abs(mine["ci95_pct"][1] - rec["ci95_pct"][1]))})
        assert abs(mine["delta_pct"] - rec["delta_pct"]) < 1e-9

    for c in (A132, SP32, MG32, MG48):
        xcheck("vs baseline (ranking[].vs_baseline)", c, BASE, rk[c]["vs_baseline"])
    xcheck("vs best (ranking[].vs_best)", MG48, SP32, rk[MG48]["vs_best"])
    xcheck("vs best (ranking[].vs_best)", MG32, SP32, rk[MG32]["vs_best"])
    xcheck("MinGram 32k->48k (contrasts)", MG48, MG32,
           dec["contrasts"]["vocab_curve_A1_P1r3_D2"]["other_families_32k_to_48k"][0])
    for ent in dec["contrasts"]["algorithm_at_equal_settings"]["P1r3_D2_32k_vs_A1_BPE"]:
        if ent["a"] in (SP32, MG32):
            xcheck("algorithm at 32k (contrasts)", ent["a"], A132, ent)

    # confirm per source (same 5 candidates; decide.Boot on them, its own seed-draw stream)
    bootC = D0.Boot(BC, nbytes, col_idx, col_src)
    per_source_conf = collections.OrderedDict()
    for src in sorted(set(col_src)):
        rs = bootC.reps(source=src)
        ps = bootC.point_source(src)
        per_source_conf[src] = {pkey(a, b): slim(D0.compare(idx[a], idx[b], rs, ps, deltaC)) for a, b in PAIRS}

    # confirm with seeds 1-2 only (like-for-like seed count with the large arbiter)
    bootC2 = D0.Boot(BC[:, :2, :], nbytes, col_idx, col_src)
    repsC2 = bootC2.reps()
    ptC2 = {c: float(bootC2.point[idx[c]]) for c in CANDS}
    rkC2, orderC2 = ranks(ptC2, CANDS)
    conf2_pairs = {pkey(a, b): slim(D0.compare(idx[a], idx[b], repsC2, bootC2.point, D0.DELTA_REL * ptC2[BASE]))
                   for a, b in PAIRS}

    # ---------------- parameters and compute per model, both scales
    params = {}
    for c in CANDS:
        ent = {}
        for scale, recs in (("confirm", [D0.jload(os.path.join(D0.RES, conf[c][s]["row"]["file"])) for s in CONF_SEEDS]),
                            ("large", [large[c][s]["rec"] for s in LARGE_SEEDS])):
            m = recs[0]["model"]
            pr = m["params"]
            for r_ in recs:
                assert r_["model"]["params"] == pr
            tok = float(np.mean([r_["data"]["tokens_seen"] for r_ in recs]))
            ent[scale] = {"d": m["d"], "L": m["L"], "H": m["H"], "n_vocab": m["n_vocab"], "ctx_tokens": m["ctx_tokens"],
                          "params_total": pr["total"], "token_embedding_tied": pr["token_embedding_tied"],
                          "position_embedding": pr["position_embedding"], "non_embedding": pr["non_embedding"],
                          "embedding_share": (pr["token_embedding_tied"] + pr["position_embedding"]) / pr["total"],
                          "tokens_seen_mean": tok,
                          "train_flops_est": 6.0 * (pr["non_embedding"] + pr["token_embedding_tied"]) * tok,
                          "train_s_mean_T4": float(np.mean([r_["wall_time_s"]["train"] for r_ in recs])),
                          "epochs": recs[0]["data"]["epochs"], "peak_lr": recs[0]["optimizer"]["peak_lr"],
                          "n_seeds": len(recs)}
        params[c] = ent
    for c in CANDS:
        for scale in ("confirm", "large"):
            params[c][scale]["params_ratio_vs_baseline"] = params[c][scale]["params_total"] / params[BASE][scale]["params_total"]
            params[c][scale]["flops_ratio_vs_baseline"] = params[c][scale]["train_flops_est"] / params[BASE][scale]["train_flops_est"]

    # ---------------- per-candidate summary
    cand = {}
    for c in CANDS:
        cand[c] = {"large_mean_bpb": ptL[c], "large_seed_bpb": [large[c][s]["bpb"] for s in LARGE_SEEDS],
                   "large_seed_sd_rel_pct": 100 * seed_sd_large[c], "large_rank_of5": rkL[c],
                   "confirm_mean_bpb": ptC[c], "confirm_seed_bpb": [conf[c][s]["bpb"] for s in CONF_SEEDS],
                   "confirm_seed_sd_rel_pct": 100 * seed_sd_conf[c], "confirm_rank_of5": rkC[c],
                   "confirm_rank_of17_decision": rk[c]["rank_primary"],
                   "confirm_seeds12_mean_bpb": ptC2[c], "confirm_seeds12_rank_of5": rkC2[c],
                   "in_preregistered_top_set": c in TOP,
                   "large_vs_baseline": slim(vs_base[c]) if c != BASE else None,
                   "large_vs_best": slim(vs_best[c]) if c != bestL else None,
                   "large_holm_p_vs_best": holm_best.get(c)}

    n_pairs = len(CANDS) * (len(CANDS) - 1) // 2
    flips = []
    for i, a in enumerate(CANDS):
        for b in CANDS[i + 1:]:
            if (ptC[a] < ptC[b]) != (ptL[a] < ptL[b]):
                flips.append(sorted([a, b]))

    out = collections.OrderedDict()
    out["what"] = ("Amendment 1 statistics (report-only): the 'large' arbiter (d=384, L=6, H=6, 2 epochs, LR 5e-4, "
                   "seeds 1-2) on dev_strict under the DECISION.md bootstrap, and its comparison with Stage 4 'confirm'")
    out["generated_utc"] = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["code"] = {"amendment1_stats.py": D0.sha256_file(os.path.abspath(__file__)),
                   "decide.py (imported: Boot, compare, holm, dev_clusters, load_stage)": checks["decide_py_sha256_now"]}
    out["test_split"] = "no test LM result read; the test split is not opened by this script"
    out["inputs"] = checks
    out["clusters"] = cl["info"]
    out["protocol"] = {"n_rep": D0.N_REP, "rng": "numpy.random.default_rng(%d)" % D0.RNG_SEED,
                       "candidate_rng_order": CANDS, "large_seeds": list(LARGE_SEEDS), "confirm_seeds": list(CONF_SEEDS),
                       "rng_order": "cluster draws (sources sorted), then seed draws (candidates sorted); in the joint "
                                    "confirm/large bootstrap the confirm seed draws follow the large ones",
                       "delta_pct": "(bpb_a - bpb_b) / bpb_b per replicate; 95 % percentile CI",
                       "p_value": "2 min(P(D*<=0), P(D*>=0)); resolution 1e-4",
                       "tost_delta_abs_bpb_large": deltaL, "seed_bootstrap_shrink": shrink}
    out["large_lr_sweep"] = sweep
    out["candidates"] = cand
    out["large"] = {"order": orderL, "best": bestL, "vs_baseline": {c: slim(vs_base[c]) for c in vs_base},
                    "pairs": {k: slim(v) for k, v in pairs.items()},
                    "holm_vs_best": holm_best, "holm_within_top_set_pairs": holm_top,
                    "p_first_within_top_set": p_first_top, "rank_distribution_within_top_set": top_rank_dist,
                    "p_first_of_5": p_first_all, "per_source": per_source, "cluster_view": cluster_view,
                    "leave_one_cluster_out": loco,
                    "leave_one_cluster_out_top_set_order_counts": dict(loco_top_order_counts)}
    out["sensitivity_2_seeds"] = sens
    out["confirm_same5"] = {"order": orderC, "pairs": conf_pairs, "per_source": per_source_conf,
                            "crosscheck_vs_decision_json": xc,
                            "max_ci_endpoint_diff_pp": max(x["max_ci_endpoint_diff_pp"] for x in xc)}
    out["confirm_seeds12"] = {"order": orderC2, "pairs": conf2_pairs}
    out["scale_change"] = {"interaction": interaction, "n_pairs": n_pairs, "pairs_changing_order": flips,
                           "kendall_tau_b_confirm_vs_large": D0.kendall_spearman([ptC[c] for c in CANDS],
                                                                                 [ptL[c] for c in CANDS])}
    out["parameters"] = params
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", OUT_JSON)
    print("large order:", orderL)
    print("confirm order:", orderC, " seeds1-2:", orderC2)
    for c in CANDS:
        if c != BASE:
            v = vs_base[c]
            print("  %-30s vs base %+.3f %% [%+.3f, %+.3f] p %.4f" % (c, v["delta_pct"], *v["ci95_pct"], v["p"]))
    for k, v in pairs.items():
        print("  %-60s %+.3f %% [%+.3f, %+.3f] p %.4f" % (k, v["delta_pct"], *v["ci95_pct"], v["p"]))
    for k, v in interaction.items():
        print("  change %-53s C %+.3f  L %+.3f  d %+.3f pp [%+.3f, %+.3f] p %.4f" %
              (k, v["confirm_pct"], v["large_pct"], v["change_pp"], *v["change_ci95_pp"], v["p"]))
    print("P(first in top set):", p_first_top)
    print("done in %.1fs" % (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds())


if __name__ == "__main__":
    main()
