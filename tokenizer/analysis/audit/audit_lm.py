# -*- coding: utf-8 -*-
"""Adversarial audit of the claim 'the chosen tokenizer is the best Hindko tokenizer among those measured'.

LM side (dev_strict only; the test split is never opened). Reads the Stage 4 'confirm' records of both
Colab bundles through lm/colab_results/results_table.json, the dev clusters of lm/colab_results/dev_clusters.csv
and analysis/decision.json. Writes analysis/audit/audit_lm.json.

Independent of analysis/decide.py: own loader, own vectorised bootstrap, own RNG stream (seed 20260927).
"""
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
RES = os.path.join(TOK, "lm", "colab_results")
OUT = os.path.join(TOK, "analysis", "audit", "audit_lm.json")
CHOSEN = "R2-A10-MinGram-P1r3-D2-48k"
BASE = "A1-P1r3-D2-16k"
BEST_R1 = "A1-P1r3-D2-32k"
RANK8 = "A6-SBPE-P1-D1-32k-t080"
R = 10000
RNG_SEED = 20260927


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    table = json.load(open(os.path.join(RES, "results_table.json"), encoding="utf-8"))
    dec = json.load(open(os.path.join(TOK, "analysis", "decision.json"), encoding="utf-8"))
    rows = [r for r in table if r["stage"] == "confirm" and not r.get("duplicate_of")]
    cands = sorted({r["candidate"] for r in rows})
    meta = {}
    recs = {}
    for r in rows:
        p = os.path.join(RES, r["file"].replace("/", os.sep))
        assert sha256(p) == r["file_sha256"], p
        rec = json.load(open(p, encoding="utf-8"))
        assert rec["candidate"] == r["candidate"] and rec["seed"] == r["seed"] and rec["stage"] == "confirm"
        recs[(r["candidate"], r["seed"])] = rec
        meta[r["candidate"]] = {k: r[k] for k in ("n_vocab", "ctx_tokens", "params_total", "params_non_embedding",
                                                  "train_bytes_per_token", "dev_bytes_per_token", "candidate_origin",
                                                  "encoder_kind", "family", "data_mix", "pretokenizer")}
    seeds = [1, 2, 3, 4, 5]
    # clusters
    cl_rows = list(csv.DictReader(open(os.path.join(RES, "dev_clusters.csv"), encoding="utf-8")))
    uids = [x["uid"] for x in cl_rows]
    dbytes = np.array([int(x["bytes"]) for x in cl_rows], dtype=np.float64)
    clusters = [x["cluster"] for x in cl_rows]
    sources = [x["source"] for x in cl_rows]
    cl_names = sorted(set(clusters))
    cl_idx = np.array([cl_names.index(c) for c in clusters])
    cl_src = {c: s for c, s in zip(clusters, sources)}
    K = len(cl_names)
    C = len(cands)
    S = 5
    bits = np.zeros((C, S, len(uids)))
    for i, c in enumerate(cands):
        for j, s in enumerate(seeds):
            rec = recs[(c, s)]
            assert rec["dev"]["uids"] == uids
            assert np.array_equal(np.array(rec["dev"]["bytes"], dtype=np.float64), dbytes)
            bits[i, j] = np.array(rec["dev"]["bits"], dtype=np.float64)
    # cluster sums
    bits_cl = np.zeros((C, S, K))
    for k in range(K):
        bits_cl[:, :, k] = bits[:, :, cl_idx == k].sum(axis=2)
    bytes_cl = np.array([dbytes[cl_idx == k].sum() for k in range(K)])
    seed_bpb = bits.sum(axis=2) / dbytes.sum()            # C x S
    mean_bpb = seed_bpb.mean(axis=1)
    ci = {c: i for i, c in enumerate(cands)}
    out = {"what": "adversarial audit, LM side (dev_strict, Stage 4 confirm, 5 seeds)", "test_split": "never read",
           "n_candidates": C, "n_clusters": K}

    # check against decision.json
    dec_rank = {r["id"]: r for r in dec["ranking"]}
    out["recomputed_mean_bpb_matches_decision"] = all(
        abs(mean_bpb[ci[c]] - dec_rank[c]["mean_bpb"]) < 1e-9 for c in cands if c in dec_rank and "mean_bpb" in dec_rank[c])

    # ---------------------------------------------------------------- parameters / compute
    par = {}
    for c in cands:
        m = meta[c]
        V, ctx = m["n_vocab"], m["ctx_tokens"]
        emb = V * 192
        rec = recs[(c, 1)]
        tok_seen = rec["data"]["tokens_seen"]
        tr_s = np.mean([recs[(c, s)]["wall_time_s"]["train"] for s in seeds])
        # training FLOPs ~ 6 * (non-embedding + output projection) * tokens (input lookup is free)
        flops = 6.0 * (m["params_non_embedding"] + emb) * tok_seen
        par[c] = {"V": V, "ctx_tokens": ctx, "params_total": m["params_total"], "token_embedding": emb,
                  "position_embedding": ctx * 192, "non_embedding": m["params_non_embedding"],
                  "embedding_share": round((emb + ctx * 192) / m["params_total"], 4),
                  "tokens_seen": tok_seen, "train_flops_est": flops, "train_s_mean_T4": round(float(tr_s), 1),
                  "dev_context_bytes": round(ctx * m["dev_bytes_per_token"], 1),
                  "train_context_bytes": round(ctx * m["train_bytes_per_token"], 1),
                  "mean_bpb": float(mean_bpb[ci[c]])}
    for c in cands:
        par[c]["params_ratio_vs_baseline"] = round(par[c]["params_total"] / par[BASE]["params_total"], 3)
        par[c]["flops_ratio_vs_baseline"] = round(par[c]["train_flops_est"] / par[BASE]["train_flops_est"], 3)
    out["parameters"] = par

    # ---------------------------------------------------------------- bootstrap (own stream)
    rng = np.random.default_rng(RNG_SEED)
    src_names = sorted(set(sources))
    Mcount = np.zeros((R, K))
    for s_ in src_names:
        ks = [k for k in range(K) if cl_src[cl_names[k]] == s_]
        draw = rng.integers(0, len(ks), size=(R, len(ks)))
        for col in range(len(ks)):
            np.add.at(Mcount, (np.arange(R), np.array(ks)[draw[:, col]]), 1)
    Scount = np.zeros((R, C, S))
    for i in range(C):
        draw = rng.integers(0, S, size=(R, S))
        for col in range(S):
            np.add.at(Scount, (np.arange(R), i, draw[:, col]), 1)
    den = Mcount @ bytes_cl                                   # R
    num = np.einsum("rk,csk->rcs", Mcount, bits_cl)           # R x C x S
    bpb_b = (num * Scount).sum(axis=2) / S / den[:, None]     # R x C

    def delta(a, b):
        d = (bpb_b[:, ci[a]] - bpb_b[:, ci[b]]) / bpb_b[:, ci[b]] * 100
        obs = (mean_bpb[ci[a]] - mean_bpb[ci[b]]) / mean_bpb[ci[b]] * 100
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return {"delta_pct": round(float(obs), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                "se_pp": round(float(d.std(ddof=1)), 4), "p_boot": float(p), "z": round(float(obs / d.std(ddof=1)), 2)}, d, obs

    family = [c for c in cands if c != RANK8]
    fam_idx = np.array([ci[c] for c in family])
    out["chosen_vs_baseline"] = delta(CHOSEN, BASE)[0]
    # rank-1 probability within the primary family
    win = np.argmin(bpb_b[:, fam_idx], axis=1)
    p_best = {family[j]: float((win == j).mean()) for j in range(len(family))}
    out["p_rank1_bootstrap"] = {k: round(v, 4) for k, v in sorted(p_best.items(), key=lambda kv: -kv[1]) if v > 0}
    ranks = np.argsort(np.argsort(bpb_b[:, fam_idx], axis=1), axis=1) + 1
    jc = family.index(CHOSEN)
    out["chosen_rank_distribution"] = {str(r_): round(float((ranks[:, jc] == r_).mean()), 4) for r_ in range(1, 7)}
    # pairwise among the top 6 by mean
    order = sorted(family, key=lambda c: mean_bpb[ci[c]])
    top6 = order[:6]
    out["pairwise_top6"] = {a + " - " + b: delta(a, b)[0] for ii, a in enumerate(top6) for b in top6[ii + 1:]}
    # max-T simultaneous intervals for every family member vs baseline
    others = [c for c in family if c != BASE]
    Ds, obs_, se_ = [], [], []
    for c in others:
        _, d, o = delta(c, BASE)
        Ds.append(d); obs_.append(o); se_.append(d.std(ddof=1))
    Ds = np.array(Ds); obs_ = np.array(obs_); se_ = np.array(se_)
    T = np.abs((Ds - obs_[:, None]) / se_[:, None]).max(axis=0)
    q = float(np.percentile(T, 95))
    out["simultaneous_vs_baseline_maxT"] = {"q95_maxT": round(q, 3), "k": len(others),
                                            "chosen_simultaneous_ci95": [round(float(obs_[others.index(CHOSEN)] - q * se_[others.index(CHOSEN)]), 4),
                                                                         round(float(obs_[others.index(CHOSEN)] + q * se_[others.index(CHOSEN)]), 4)],
                                            "chosen_z": round(float(obs_[others.index(CHOSEN)] / se_[others.index(CHOSEN)]), 2)}
    # round 2 vs the best round-1 candidate: single and max-T (forking paths: 6 exploratory tries)
    r2 = [c for c in cands if c.startswith("R2-")]
    D2s, o2, s2 = [], [], []
    per = {}
    for c in r2:
        st, d, o = delta(c, BEST_R1)
        per[c] = st
        D2s.append(d); o2.append(o); s2.append(d.std(ddof=1))
    D2s = np.array(D2s); o2 = np.array(o2); s2 = np.array(s2)
    Tn = ((D2s - o2[:, None]) / s2[:, None])            # null-centred
    minT = Tn.min(axis=0)                                 # most negative (best-looking) R2 under H0
    adj = {c: float((minT <= o2[i] / s2[i]).mean()) for i, c in enumerate(r2)}
    out["round2_vs_best_round1"] = {"reference": BEST_R1, "per_candidate": per,
                                    "maxT_adjusted_one_sided_p": {c: round(v, 5) for c, v in adj.items()},
                                    "note": "H0: an R2 tokenizer is no better than the best pre-registered round-1 tokenizer; the adjustment is over the 6 exploratory tries (min-T of null-centred bootstrap)."}
    # chosen vs every other family member (Bonferroni over all pairs involving the chosen)
    vs_all = {}
    for c in family:
        if c == CHOSEN:
            continue
        st = delta(CHOSEN, c)[0]
        st["p_bonferroni_16"] = min(1.0, st["p_boot"] * 16)
        vs_all[c] = st
    out["chosen_vs_each"] = vs_all

    # ---------------------------------------------------------------- winner's-curse proxy for the rule
    # In each replicate: best-mean; 'top set' approximated by the candidates whose replicate delta vs the replicate best
    # is below the Holm-free z<1.645*se threshold (se from the main bootstrap); then tie-breaker (b) = highest dev bytes/token.
    bpt = np.array([meta[c]["dev_bytes_per_token"] for c in family])
    hf_exact = np.array([meta[c]["encoder_kind"] == "hf" for c in family])
    se_pair = {}
    choice = []
    fb = bpb_b[:, fam_idx]
    for rr in range(0, R, 10):
        row = fb[rr]
        b = int(np.argmin(row))
        rel = (row - row[b]) / row[b] * 100
        cand = [j for j in range(len(family)) if rel[j] < 0.30]   # within 0.30 % of the replicate best
        pool = [j for j in cand if hf_exact[j]] or cand
        choice.append(family[max(pool, key=lambda j: bpt[j])])
    vals, cnt = np.unique(np.array(choice), return_counts=True)
    out["rule_stability_proxy"] = {"what": "1,000 replicates; top set proxy = within 0.30 % of the replicate best; (a) HF-native exact; (b) max dev bytes/token",
                                   "choice_freq": {v: round(int(n) / len(choice), 3) for v, n in zip(vals, cnt)}}

    # ---------------------------------------------------------------- leave-one-cluster-out / leave-one-seed-out
    loco = []
    for k in range(K):
        keep = np.ones(K, bool); keep[k] = False
        bp = (bits_cl[:, :, keep].sum(axis=2) / bytes_cl[keep].sum()).mean(axis=1)
        fam_bp = {c: bp[ci[c]] for c in family}
        o = sorted(fam_bp, key=fam_bp.get)
        loco.append({"dropped": cl_names[k], "bytes_share": round(float(bytes_cl[k] / bytes_cl.sum()), 4), "best": o[0],
                     "top3": o[:3], "chosen_rank": o.index(CHOSEN) + 1,
                     "chosen_vs_base_pct": round(float((bp[ci[CHOSEN]] - bp[ci[BASE]]) / bp[ci[BASE]] * 100), 3)})
    out["leave_one_cluster_out"] = {"best_counts": {b: sum(1 for x in loco if x["best"] == b) for b in {x["best"] for x in loco}},
                                    "chosen_rank_counts": {str(r_): sum(1 for x in loco if x["chosen_rank"] == r_) for r_ in sorted({x["chosen_rank"] for x in loco})},
                                    "chosen_vs_base_range_pct": [min(x["chosen_vs_base_pct"] for x in loco), max(x["chosen_vs_base_pct"] for x in loco)],
                                    "rows_where_top3_changes": [x for x in loco if set(x["top3"]) != set(order[:3])]}
    loso = []
    for j in range(S):
        keep = [x for x in range(S) if x != j]
        bp = seed_bpb[:, keep].mean(axis=1)
        o = sorted(family, key=lambda c: bp[ci[c]])
        loso.append({"dropped_seed": j + 1, "best": o[0], "top3": o[:3], "chosen_rank": o.index(CHOSEN) + 1})
    out["leave_one_seed_out"] = loso
    # per-cluster sign consistency (seed-mean per cluster)
    cb = bits_cl.mean(axis=1) / bytes_cl[None, :]
    dcl = (cb[ci[CHOSEN]] - cb[ci[BASE]]) / cb[ci[BASE]] * 100
    out["chosen_vs_base_by_cluster"] = {"n_clusters_chosen_better": int((dcl < 0).sum()), "n_clusters": K,
                                        "worst_cluster": {"cluster": cl_names[int(np.argmax(dcl))], "delta_pct": round(float(dcl.max()), 3),
                                                          "bytes": int(bytes_cl[int(np.argmax(dcl))])},
                                        "median_pct": round(float(np.median(dcl)), 3)}
    d2 = (cb[ci[CHOSEN]] - cb[ci["R2-A4-SPnat-D2-32k"]]) / cb[ci["R2-A4-SPnat-D2-32k"]] * 100
    d3 = (cb[ci[CHOSEN]] - cb[ci["R2-A10-MinGram-P1r3-D2-32k"]]) / cb[ci["R2-A10-MinGram-P1r3-D2-32k"]] * 100
    out["chosen_vs_topset_by_cluster"] = {
        "vs_R2-A4-SPnat-D2-32k": {"n_clusters_chosen_better": int((d2 < 0).sum()), "by_source_bytes_weighted_pct": {
            s_: round(float(np.sum(d2[[cl_src[n] == s_ for n in cl_names]] * bytes_cl[[cl_src[n] == s_ for n in cl_names]]) / bytes_cl[[cl_src[n] == s_ for n in cl_names]].sum()), 3) for s_ in src_names}},
        "vs_R2-A10-MinGram-P1r3-D2-32k": {"n_clusters_chosen_better": int((d3 < 0).sum())}}

    # ---------------------------------------------------------------- seed stability, loss spikes, fp16 traces
    seedinfo = {}
    for c in cands:
        sb = seed_bpb[ci[c]]
        spikes, nonfinite, last50 = [], [], []
        for s in seeds:
            tl = recs[(c, s)]["train_loss"]
            nonfinite.append(sum(1 for v in tl if v is None))
            arr = np.array([np.nan if v is None else v for v in tl], dtype=np.float64)
            # spike = loss above the running median of the previous 25 steps, after warm-up
            sp = []
            for t in range(100, len(arr)):
                w = arr[t - 25:t]
                w = w[np.isfinite(w)]
                if len(w) and np.isfinite(arr[t]):
                    sp.append(arr[t] - np.median(w))
            spikes.append(round(float(np.max(sp)), 3))
            last50.append(float(np.nanmean(arr[-50:])))
        seedinfo[c] = {"seed_bpb": [round(float(x), 5) for x in sb], "rel_sd_pct": round(float(sb.std(ddof=1) / sb.mean() * 100), 3),
                       "range_pct": round(float((sb.max() - sb.min()) / sb.mean() * 100), 3),
                       "max_loss_spike_nats_per_seed": spikes, "nonfinite_train_losses_per_seed": nonfinite,
                       "train_loss_last50_per_seed": [round(x, 4) for x in last50],
                       "corr_trainloss_devbpb_across_seeds": round(float(np.corrcoef(last50, sb)[0, 1]), 3),
                       "curve_subset_bpb_per_seed": [[round(p["bpb"], 4) for p in recs[(c, s)]["curve"]] for s in seeds],
                       "encoder_kind": meta[c]["encoder_kind"]}
    out["seed_stability"] = seedinfo
    # which clusters carry the seed variance of the SentencePiece D1 arms
    var_dec = {}
    for c in ["A3-SPnat-D1-16k", "A4-SPnat-D1-16k", "R2-A4-SPnat-D2-32k", "R2-A4-SPnat-D2-48k", BASE, CHOSEN]:
        per_seed_cl = bits_cl[ci[c]]                           # S x K bits
        tot = per_seed_cl.sum(axis=1)
        dev_ = tot - tot.mean()
        contrib = [(cl_names[k], float(np.sum((per_seed_cl[:, k] - per_seed_cl[:, k].mean()) * dev_) / np.sum(dev_ ** 2)))
                   for k in range(K)]
        contrib.sort(key=lambda kv: -kv[1])
        var_dec[c] = {"top_clusters_share_of_seed_variance": [(n, round(v, 3), cl_src[n]) for n, v in contrib[:4]],
                      "by_source_share": {s_: round(sum(v for n, v in contrib if cl_src[n] == s_), 3) for s_ in src_names}}
    out["seed_variance_by_cluster"] = var_dec
    # pooled s.d. by encoder kind
    kinds = {}
    for c in family:
        kinds.setdefault(meta[c]["encoder_kind"], []).append(seedinfo[c]["rel_sd_pct"])
    out["rel_sd_by_encoder_kind"] = {k: {"n": len(v), "pooled_rel_sd_pct": round(float(np.sqrt(np.mean(np.square(v)))), 3), "values": v}
                                     for k, v in kinds.items()}

    # ---------------------------------------------------------------- equal-parameter comparisons
    groups = {}
    for c in family:
        groups.setdefault(par[c]["V"], []).append(c)
    eq = {}
    for V, cs in sorted(groups.items()):
        cs = sorted(cs, key=lambda c: mean_bpb[ci[c]])
        eq[str(V)] = [{"id": c, "mean_bpb": round(float(mean_bpb[ci[c]]), 5), "params_total": par[c]["params_total"]} for c in cs]
    out["equal_V_groups"] = eq
    # decomposition of the chosen's gain: size (BPE 16k->48k) + algorithm at 48k
    out["gain_decomposition"] = {
        "size_BPE_16k_to_48k": delta("R2-A1-P1r3-D2-48k", BASE)[0],
        "algorithm_at_48k_MinGram_vs_BPE": delta(CHOSEN, "R2-A1-P1r3-D2-48k")[0],
        "chosen_48k_vs_best_32k_SPnat": delta(CHOSEN, "R2-A4-SPnat-D2-32k")[0],
        "chosen_48k_vs_MinGram_32k": delta(CHOSEN, "R2-A10-MinGram-P1r3-D2-32k")[0],
        "MinGram_32k_vs_baseline": delta("R2-A10-MinGram-P1r3-D2-32k", BASE)[0],
        "baseline_vs_literal_standard_A1-P1-D1-16k": delta(BASE, "A1-P1-D1-16k")[0],
        "chosen_vs_literal_standard_A1-P1-D1-16k": delta(CHOSEN, "A1-P1-D1-16k")[0],
    }
    # log-params fit on the BPE size curve (A1-P1r3-D2 8k/16k/32k/48k) and residual of every candidate at its own size
    curve = ["A1-P1r3-D2-8k", BASE, BEST_R1, "R2-A1-P1r3-D2-48k"]
    x = np.log([par[c]["params_total"] for c in curve]); y = np.log([mean_bpb[ci[c]] for c in curve])
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = {}
    for c in family:
        lp = math.log(par[c]["params_total"])
        pred = math.exp(coef[0] + coef[1] * lp + coef[2] * lp * lp)
        resid[c] = round((mean_bpb[ci[c]] - pred) / pred * 100, 3)
    out["residual_vs_BPE_param_curve_pct"] = dict(sorted(resid.items(), key=lambda kv: kv[1]))

    # ---------------------------------------------------------------- bootstrap-variance sanity (seeds)
    # seed component: resampling n=5 seeds with replacement shrinks the variance of the mean by (n-1)/n
    out["seed_bootstrap_shrink_factor"] = {"n_seeds": S, "variance_factor": (S - 1) / S, "se_factor": round(math.sqrt((S - 1) / S), 4)}

    # LR resweep (report-only) and the baseline's own Stage-4 LR
    rs = [r for r in table if r["stage"] == "confirm_resweep"]
    out["lr_resweep_candidates"] = sorted({r["candidate"] for r in rs})
    out["baseline_resweep_at_stage4"] = BASE in out["lr_resweep_candidates"]

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("chosen_vs_baseline", "p_rank1_bootstrap", "chosen_rank_distribution",
                                           "simultaneous_vs_baseline_maxT", "rule_stability_proxy")}, indent=1))


if __name__ == "__main__":
    main()
