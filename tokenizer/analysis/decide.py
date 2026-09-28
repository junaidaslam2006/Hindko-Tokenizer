"""PLAN §6-§7 decision over every LM-arbiter candidate of both Colab rounds (Stage 4 'confirm').

Reads only:
  lm/colab_results/results_table.json + the per-run JSON records it lists (dev_strict per-document bits/bytes)
  splits/split_manifest.jsonl  -> clusters of the 836 dev_strict uids (rows whose uid is not a dev uid are skipped
                                   on the uid alone; no other field of a non-dev row is parsed)
  lm/WAVES_colab.json, lm/WAVES_round2.json, candidates/**/summary.json, candidates/**/meta.json (tie-breakers)
It never opens, encodes or scores the test split.

Pre-registered procedure (research/PLAN.md, sha256 d7a811df...):
  §6 hierarchical cluster bootstrap, 10,000 replicates, numpy default_rng(12345):
     per replicate, clusters (manifest 'group'; per-record web groups collapsed to their site) are resampled with
     replacement within each source (stratified); each candidate's seeds are resampled with replacement,
     independently per candidate; bpb = sum(bits) / sum(bytes) over the resampled documents and seeds.
     RNG consumption order (implementation detail, fixed here): cluster draws for all replicates, source by source in
     sorted order (book, newspaper, web); then seed draws, candidate by candidate in sorted id order.
     Outputs: 95% percentile CI, two-sided p = 2 min(P(D*<=0), P(D*>=0)).
  Holm-Bonferroni over the (k-1) comparisons against the best-mean candidate (alpha 0.05).
  TOST: equivalent if the 90% CI of D lies inside +-delta, delta = 0.3% of the baseline's mean bpb.
  §7: rank by mean dev bpb (Stage 4); top set = best + not significantly worse after Holm + equivalent within delta;
     tie-breakers (a) HF-native exact, (b) higher dev bytes/token, (c) fewer learned tokens with train freq < 20,
     (d) lower robustness sensitivity, (e) smaller vocabulary; improvement claim iff the chosen tokenizer's D vs
     A1-P1r3-D2-16k has a 95% CI excluding 0.

Run: PYTHONIOENCODING=utf-8 python analysis/decide.py   (single process; about a minute)
"""
import collections
import datetime
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = r"F:\Hindko\_tokenizer"
RES = os.path.join(ROOT, "lm", "colab_results")
OUT = os.path.join(ROOT, "analysis")
PLAN = os.path.join(ROOT, "research", "PLAN.md")
SPLIT_MANIFEST = os.path.join(ROOT, "splits", "split_manifest.jsonl")
DEV_CLUSTERS_CSV = os.path.join(RES, "dev_clusters.csv")
WAVES_FILES = [os.path.join(ROOT, "lm", "WAVES_colab.json"), os.path.join(ROOT, "lm", "WAVES_round2.json")]
PLAN_SHA = "d7a811df5165540013e1340d0e2351a20aaa5a01b76adff907781c167d47e1eb"
SPLIT_SHA = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"

BASELINE = "A1-P1r3-D2-16k"
N_REP = 10_000
RNG_SEED = 12345
ALPHA = 0.05
DELTA_REL = 0.003          # PLAN §6 equivalence margin, fraction of baseline bpb
DELTA_MIN_POWER = 0.005    # PLAN §6 power check, Delta_min = 0.5 %
Z_POWER = 1.959963984540054 + 0.8416212335729143   # alpha .05 two-sided, power .8
PERTURBATIONS = ["harakat", "digits", "punct_space", "zwnj"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fnum(x, nd=6):
    return None if x is None else float(round(float(x), nd))


# ----------------------------------------------------------------------------------------------- inputs
def load_inputs():
    checks = collections.OrderedDict()
    checks["plan_sha256"] = sha256_file(PLAN)
    checks["plan_sha256_ok"] = checks["plan_sha256"] == PLAN_SHA
    checks["split_manifest_sha256"] = sha256_file(SPLIT_MANIFEST)
    checks["split_manifest_sha256_ok"] = checks["split_manifest_sha256"] == SPLIT_SHA
    table_path = os.path.join(RES, "results_table.json")
    checks["results_table_sha256"] = sha256_file(table_path)
    rows = jload(table_path)
    assert checks["plan_sha256_ok"] and checks["split_manifest_sha256_ok"], checks
    return rows, checks


def load_stage(rows, stage, checks):
    """candidate -> seed -> record (bits, bytes, uids, bpb, curve, row) for non-duplicate rows of a stage."""
    out = collections.defaultdict(dict)
    n_files = 0
    for r in rows:
        if r["stage"] != stage or r["duplicate_of"]:
            continue
        assert r["hard_valid"] in (True, "True"), (r["file"], r["failed_checks"])
        p = os.path.join(RES, r["file"])
        assert sha256_file(p) == r["file_sha256"], p
        rec = jload(p)
        dev = rec["dev"]
        bits = np.asarray(dev["bits"], dtype=np.float64)
        nbytes = np.asarray(dev["bytes"], dtype=np.float64)
        assert len(bits) == 836 and np.all(np.isfinite(bits)) and np.all(bits > 0)
        assert float(np.sum(bits) / np.sum(nbytes)) == rec["bpb"], p
        assert rec["candidate"] == r["candidate"] and rec["seed"] == int(r["seed"]) and rec["stage"] == stage
        assert rec["lr"] == 0.001, p
        out[r["candidate"]][int(r["seed"])] = {"bits": bits, "bytes": nbytes, "uids": list(dev["uids"]),
                                                "bpb": rec["bpb"], "curve": rec.get("curve"), "row": r,
                                                "bytes_seen_rel": r["bytes_seen_rel_budget"],
                                                "tolerance_flags": r["tolerance_flags"]}
        n_files += 1
    checks["records_loaded_%s" % stage] = n_files
    return out


def verify_duplicates(rows):
    """Round-2 copies of the reference arms must equal the round-1 runs bit for bit; they are then dropped."""
    idx = {(r["bundle"], r["stage"], r["candidate"], int(r["seed"]), r["lr_tag"]): r for r in rows}
    pairs = []
    for r in rows:
        if not r["duplicate_of"]:
            continue
        o = idx[(r["duplicate_of"], r["stage"], r["candidate"], int(r["seed"]), r["lr_tag"])]
        a = jload(os.path.join(RES, r["file"]))["dev"]
        b = jload(os.path.join(RES, o["file"]))["dev"]
        same = (a["bits"] == b["bits"]) and (a["uids"] == b["uids"]) and (a["bytes"] == b["bytes"])
        pairs.append({"stage": r["stage"], "candidate": r["candidate"], "seed": int(r["seed"]),
                      "kept": o["file"], "dropped": r["file"], "bitwise_identical": bool(same)})
    assert all(p["bitwise_identical"] for p in pairs)
    return pairs


def dev_clusters(uids, nbytes):
    """Clusters from the split manifest (dev uids only), per-record web groups collapsed to their site."""
    want = set(uids)
    got = {}
    uid_re = re.compile(r'"uid"\s*:\s*"([^"]+)"')
    with open(SPLIT_MANIFEST, encoding="utf-8") as f:
        for line in f:
            m = uid_re.search(line)
            if not m or m.group(1) not in want:
                continue                      # non-dev row: skipped on its uid alone
            x = json.loads(line)
            got[x["uid"]] = x
    assert len(got) == len(want) == 836
    raw, collapsed, source = [], [], []
    for u in uids:
        x = got[u]
        assert x["split"] == "validation" and x["quality_tier"] == "strict", u
        g = x["group"]
        m = re.match(r"^web:([^:]+):record:", g)
        raw.append(g)
        collapsed.append(("web:" + m.group(1)) if m else g)
        source.append(x["source"])
    # cross-check with the collector's mapping
    import csv
    with open(DEV_CLUSTERS_CSV, encoding="utf-8") as f:
        cc = list(csv.DictReader(f))
    agree = all(c["uid"] == u and c["cluster"] == k and c["group"] == g and int(c["bytes"]) == int(b)
                for c, u, k, g, b in zip(cc, uids, collapsed, raw, nbytes))

    def index(labels):
        names = sorted(set(labels))
        pos = {n: i for i, n in enumerate(names)}
        src_of = {}
        for lab, s in zip(labels, source):
            assert src_of.setdefault(lab, s) == s
        return np.array([pos[l] for l in labels]), names, [src_of[n] for n in names]

    col_idx, col_names, col_src = index(collapsed)
    raw_idx, raw_names, raw_src = index(raw)
    cl_bytes = collections.Counter()
    for k, b in zip(collapsed, nbytes):
        cl_bytes[k] += b
    big = cl_bytes.most_common(1)[0]
    info = {"n_docs": len(uids), "n_bytes": int(np.sum(nbytes)),
            "n_clusters": len(col_names), "clusters_by_source": dict(collections.Counter(col_src)),
            "n_raw_groups": len(raw_names), "raw_groups_by_source": dict(collections.Counter(raw_src)),
            "largest_cluster": {"cluster": big[0], "bytes": int(big[1]), "share": big[1] / float(np.sum(nbytes))},
            "agrees_with_dev_clusters_csv": bool(agree),
            "rule": "cluster = manifest 'group'; 'web:<site>:record:<uid>' -> 'web:<site>'"}
    return {"collapsed": (col_idx, col_names, col_src), "raw": (raw_idx, raw_names, raw_src), "info": info,
            "doc_source": source}


# ----------------------------------------------------------------------------------------------- bootstrap
class Boot:
    """Replicate weights shared by every candidate of one bootstrap run (paired over documents)."""

    def __init__(self, bits, nbytes, doc_cluster, cl_source, n_rep=N_REP, seed=RNG_SEED):
        # bits: (C, S, D)
        C, S, D = bits.shape
        K = len(cl_source)
        rng = np.random.default_rng(seed)
        W = np.zeros((n_rep, K), dtype=np.float64)
        rows = np.arange(n_rep)
        for src in sorted(set(cl_source)):
            idx = np.array([k for k in range(K) if cl_source[k] == src])
            draws = idx[rng.integers(0, len(idx), size=(n_rep, len(idx)))]
            for j in range(len(idx)):
                np.add.at(W, (rows, draws[:, j]), 1.0)
        M = np.zeros((C, n_rep, S), dtype=np.float64)
        for c in range(C):
            d = rng.integers(0, S, size=(n_rep, S))
            for j in range(S):
                np.add.at(M[c], (rows, d[:, j]), 1.0)
        onehot = np.zeros((D, K))
        onehot[np.arange(D), doc_cluster] = 1.0
        self.CB = bits @ onehot                      # (C, S, K) bits per cluster
        self.KB = nbytes @ onehot                    # (K,) bytes per cluster
        self.W, self.M, self.S, self.cl_source = W, M, S, list(cl_source)
        self.point = (bits.sum(axis=2) / nbytes.sum()).mean(axis=1)   # (C,) mean over seeds of per-seed bpb
        self.seed_bpb = bits.sum(axis=2) / nbytes.sum()                # (C, S)

    def reps(self, source=None):
        W = self.W
        CB, KB = self.CB, self.KB
        if source is not None:
            mask = np.array([s == source for s in self.cl_source], dtype=np.float64)
            W = W * mask[None, :]
        T = np.einsum("csk,rk->csr", CB, W)
        num = np.einsum("crs,csr->cr", self.M, T)
        den = self.S * (W @ KB)
        return num / den[None, :]

    def point_source(self, source):
        mask = np.array([s == source for s in self.cl_source], dtype=np.float64)
        return ((self.CB * mask).sum(axis=2) / (self.KB * mask).sum()).mean(axis=1)


def compare(ia, ib, reps, point, delta_abs):
    d = reps[ia] - reps[ib]
    rel = d / reps[ib]
    d0 = float(point[ia] - point[ib])
    R = d.size
    le, ge = int(np.sum(d <= 0)), int(np.sum(d >= 0))
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    rlo, rhi = np.percentile(rel, [2.5, 97.5])
    lo90, hi90 = np.percentile(d, [5, 95])
    return {"delta": d0, "delta_pct": 100 * d0 / float(point[ib]),
            "ci95": [float(lo95), float(hi95)], "ci95_pct": [100 * float(rlo), 100 * float(rhi)],
            "ci90": [float(lo90), float(hi90)],
            "p": min(1.0, 2 * min(le, ge) / R),
            "p_plus1": min(1.0, 2 * min(le + 1, ge + 1) / (R + 1)),
            "se": float(np.std(d, ddof=1)), "se_pct": 100 * float(np.std(rel, ddof=1)),
            "tost_equivalent": bool(lo90 > -delta_abs and hi90 < delta_abs),
            "ci95_excludes_0": bool(lo95 > 0 or hi95 < 0),
            "mde80_pct": 100 * Z_POWER * float(np.std(rel, ddof=1))}


def holm(pv):
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    items = sorted(pv.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


# ----------------------------------------------------------------------------------------------- candidates
def wave_entries():
    ent = collections.OrderedDict()
    for f in WAVES_FILES:
        d = jload(f)
        for w in d["waves"]:
            ent.setdefault(w["id"], dict(w, waves_file=os.path.basename(f)))
    return ent


def summary_path(cid, w):
    p = w.get("dev_strict_summary")
    if p:
        return p
    return os.path.join(ROOT, "candidates", "standard", "results", "dev_strict", cid, "summary.json")


def tiebreak_metrics(cid, w, row):
    sp = summary_path(cid, w)
    s = jload(sp)
    ov = s["metrics"]["overall"]
    R1 = s["properties"]["R1"]
    assert R1["train"]["train"] == "train_D1", cid
    gates = {g: s["gates"][g].get("pass") for g in ("G1", "G2", "G3", "G4", "G5")}
    gates_ok = all(v in (True, None) for v in gates.values())
    ek = w["encoder_kind"]
    tok_sha_ok = s["tokenizer"]["sha256"] == row["tokenizer_sha256"]
    eff_vocab = int(row["n_vocab"])
    meta = None
    if ek == "sentencepiece":
        meta_p = os.path.join(os.path.dirname(w["tokenizer_path"]), "meta.json")
        meta = jload(meta_p)
        he = meta["hf_export"]
        eq = bool(he.get("equivalent"))
        ds, dp = he["dev_strict"]["ids_differ_docs"], he["dev_permissive"]["ids_differ_docs"]
        hf_native = eq
        hf_ev = ("the LM encoder is the native sp.model plus the newline wrapper; its HF tokenizer.json export %s the "
                 "native ids on %d/836 dev_strict and %d/1,358 dev_permissive documents (meta.json hf_export)"
                 % ("reproduces" if eq else "differs from", ds if not eq else 836, dp if not eq else 1358))
        if not eq:
            hf_ev = ("the LM encoder is the native sp.model plus the newline wrapper (custom code), and its HF tokenizer.json "
                     "export differs from the native ids on %d/836 dev_strict and %d/1,358 dev_permissive documents (equal-score "
                     "ties; token counts identical), so it is not exact" % (ds, dp))
        if "g2_remedy" in meta:
            eff_vocab = int(meta["g2_remedy"]["effective_vocab"])
    elif ek == "custom":
        hf_native = False
        hf_ev = "a custom Python encoder (%s) that needs custom code" % w.get("custom")
    else:
        sc = w.get("stage2_checks", {})
        alg = row["family"]
        if alg in ("A6", "A10"):
            ok = (sc.get("hf_vs_reference_mismatch_docs_dev_strict") == 0 and
                  sc.get("hf_vs_reference_mismatch_docs_dev_permissive") == 0) or \
                 (sc.get("hf_vs_reference_identical_docs_dev_strict") == "836/836" and
                  sc.get("hf_vs_reference_identical_docs_dev_permissive") == "1358/1358")
            hf_native = bool(ok)
            hf_ev = ("a stock HF tokenizer.json that equals the reference encoder on 836/836 dev_strict and 1,358/1,358 "
                     "dev_permissive documents (WAVES stage2_checks)" if ok else "HF export not verified exact")
        else:
            hf_native = True
            hf_ev = "a stock HF tokenizers BPE, trained natively by HF BpeTrainer"
    rob = ov["robustness"]
    rob_seg = float(np.mean([rob[p]["seg_change_rate_affected"] for p in PERTURBATIONS]))
    rob_tok = float(np.mean([abs(rob[p]["rel_token_change"]) for p in PERTURBATIONS]))
    return {"summary_path": sp, "tokenizer_sha256_matches_lm_runs": bool(tok_sha_ok),
            "gates": gates, "gates_pass": bool(gates_ok),
            "hf_native_exact": bool(hf_native), "hf_native_evidence": hf_ev,
            "dev_bytes_per_token": float(ov["bytes_per_token"]),
            "train_lt20": int(R1["all_learned"]["train_freq_lt20"]),
            "train_lt20_pct": float(R1["all_learned"]["pct_lt20"]),
            "train_lt20_multichar": int(R1["multichar_learned"]["train_freq_lt20"]),
            "train_eq0": int(R1["all_learned"]["train_freq_eq0"]),
            "learned_tokens": int(R1["all_learned"]["learned_tokens"]),
            "robustness_seg_change_affected_mean": rob_seg,
            "robustness_abs_rel_token_change_mean": rob_tok,
            "robustness_by_perturbation": {p: {"seg_change_rate_affected": rob[p]["seg_change_rate_affected"],
                                               "rel_token_change": rob[p]["rel_token_change"]} for p in PERTURBATIONS},
            "vocab_nominal": int(row["n_vocab"]), "vocab_effective": eff_vocab}


def tb_key(tb):
    return (0 if tb["hf_native_exact"] else 1, -tb["dev_bytes_per_token"], tb["train_lt20"],
            tb["robustness_seg_change_affected_mean"], tb["vocab_effective"])


def apply_tiebreakers(top, TB):
    """Lexicographic PLAN §7.4 order; returns the order and a readable trace of which step decided."""
    steps = [("a", "HF-native exact encoding", lambda c: 0 if TB[c]["hf_native_exact"] else 1),
             ("b", "higher dev bytes/token", lambda c: -TB[c]["dev_bytes_per_token"]),
             ("c", "fewer learned tokens with train_D1 frequency < 20", lambda c: TB[c]["train_lt20"]),
             ("d", "lower robustness sensitivity (mean share of affected words re-segmented, 4 perturbations)",
              lambda c: TB[c]["robustness_seg_change_affected_mean"]),
             ("e", "smaller vocabulary (effective)", lambda c: TB[c]["vocab_effective"])]
    remaining = list(top)
    trace = []
    for tag, name, key in steps:
        if len(remaining) <= 1:
            trace.append({"step": tag, "name": name, "applied": False, "remaining": remaining})
            continue
        best = min(key(c) for c in remaining)
        kept = [c for c in remaining if key(c) == best]
        dropped = [c for c in remaining if key(c) != best]
        trace.append({"step": tag, "name": name, "applied": True, "kept": kept, "dropped": dropped,
                      "values": {c: key(c) for c in remaining}})
        remaining = kept
    order = sorted(top, key=lambda c: tb_key(TB[c]))
    assert order[0] == remaining[0]
    return order, trace


def decide(family, idx, boot, reps, TB, baseline, delta_abs, pkey="p"):
    point = boot.point
    best = min(family, key=lambda c: point[idx[c]])
    others = [c for c in family if c != best]
    comps = {c: compare(idx[c], idx[best], reps, point, delta_abs) for c in others}
    adj = holm({c: comps[c][pkey] for c in others})
    sig_worse = {c: bool(adj[c] <= ALPHA and comps[c]["delta"] > 0) for c in others}
    equiv = {c: comps[c]["tost_equivalent"] for c in others}
    top = [best] + [c for c in sorted(others, key=lambda c: point[idx[c]]) if (not sig_worse[c]) or equiv[c]]
    order, trace = apply_tiebreakers(top, TB)
    chosen = order[0]
    vs_base = compare(idx[chosen], idx[baseline], reps, point, delta_abs) if chosen != baseline else None
    claim = bool(vs_base is not None and vs_base["ci95"][1] < 0)
    return {"family": list(family), "k": len(family), "best": best, "comparisons_vs_best": comps,
            "holm_adjusted_p": adj, "significantly_worse": sig_worse, "tost_equivalent_to_best": equiv,
            "top_set": top, "tiebreak_order": order, "tiebreak_trace": trace, "chosen": chosen,
            "chosen_vs_baseline": vs_base, "improvement_claim": claim}


def kendall_spearman(a, b):
    kt = stats.kendalltau(a, b)
    sr = stats.spearmanr(a, b)
    return {"kendall_tau_b": float(kt.statistic), "kendall_p": float(kt.pvalue),
            "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def rel_sd(v):
    v = np.asarray(v, dtype=np.float64)
    return float(np.std(v, ddof=1) / np.mean(v))


def n_needed(sigma_rel):
    return int(math.ceil(15.7 * sigma_rel ** 2 / DELTA_MIN_POWER ** 2))


# ----------------------------------------------------------------------------------------------- deviations
def deviations(out, pt_s, pt_c, D, TB, sens, contrasts, power, lr_tab, lr_sweep, flags):
    par = {}
    for b in out["inputs"]["bundles"]:
        pr = jload(os.path.join(RES, b, "parity_report.json"))
        par[b] = {k: pr["runs"][k]["rel_diff_vs_cpu"] for k in ("fp16", "fp32")}
    p16 = max(abs(v["fp16"]) for v in par.values())
    p32 = max(abs(v["fp32"]) for v in par.values())
    chosen, best = D["chosen"], D["best"]
    lrs = {e["lr"]: e["bpb"] for e in lr_sweep}
    r1 = sens["round1_only"]
    dm = contrasts["data_mix_and_pretokenizer"]["D2_vs_D1_A1_P1r3_16k"]
    top_mde = [power["achieved_mde80_pct_vs_best"][c] for c in D["top_set"] if c != best]
    within = [D["comparisons_vs_best"][c]["delta_pct"] for c in D["top_set"] if c != best]
    rs = lambda c: "%s: 1e-3 %.5f, 3e-3 %.5f, 6e-3 %.5f" % (c, lr_tab[c]["1e-3 (confirm s1)"], lr_tab[c]["3e-03"],
                                                           lr_tab[c]["6e-03"])
    # every qualitative statement below is checked here
    assert all(v["1e-3 (confirm s1)"] < v["3e-03"] < v["6e-03"] for v in lr_tab.values())
    assert chosen in lr_tab and best in lr_tab
    assert sens["rank8_included"]["top_set"] == D["top_set"] and sens["rank8_included"]["chosen"] == chosen
    assert sens["p_plus1"]["top_set"] == D["top_set"] and sens["p_plus1"]["chosen"] == chosen
    assert sens["confirm_seeds_1to3"]["chosen"] == chosen
    assert max(D["top_set"], key=lambda c: TB[c]["dev_bytes_per_token"]) == chosen
    assert all(c.startswith("R2-") for c in D["top_set"])
    assert all("-D2-" in c for c in D["top_set"])
    assert not any(c.startswith(("A6", "A7", "A10", "R2-A6")) for c in D["top_set"])
    assert all(abs(x) < m for x, m in zip(within, top_mde))
    assert not any(D["tost_equivalent_to_best"].values())
    dev = [
        {"id": 1, "source": "known before this analysis", "title": "LM on a GPU (fp16) instead of the CPU",
         "text": ("PLAN §5 budgets a CPU arbiter; every run used a Colab Tesla T4 with fp16 autocast for training and fp32 "
                  "evaluation, strict determinism, tf32 off. Parity against the CPU reference on the fixed tiny "
                  "configuration: fp16 %+.3f %%, fp32 %+.3f %% (tolerance 1 %%), identical in both bundles."
                  % (100 * max((v["fp16"] for v in par.values()), key=abs), 100 * max((v["fp32"] for v in par.values()), key=abs))),
         "impact": ("All 18 candidates ran on the same device and dtype, so every comparison is like-for-like. Absolute "
                    "bpb is comparable to a CPU run only to about %.2f %%." % (100 * p16))},
        {"id": 2, "source": "known before this analysis", "title": "Stage 4 'confirm' ran for all candidates, not only the top 2",
         "text": "PLAN §3/§5 plan Stage 4 for the top 2 plus the baseline; on the GPU every candidate got the d=192, "
                 "full-train, 1-epoch arbiter with 5 seeds. PLAN §7.2 ranks on Stage 4 when it is run.",
         "impact": ("The ranking and the Holm family cover %d candidates (%d comparisons against the best) instead of 3. "
                    "A larger family makes Holm more conservative, so it can only widen the top set." % (D["k"], D["k"] - 1))},
        {"id": 3, "source": "known before this analysis", "title": "Two D1 reference arms added in round 1",
         "text": "A1-P1r3-D1-16k and A1-P1-D1-16k were added (WAVES_colab.json, written 16:11 UTC, before the first LM "
                 "run at 16:30 UTC) because the rank-1 baseline is D2 while the Stage-2 candidates (A6/A7/A10) are D1.",
         "impact": ("They are ranked with everyone else (%.5f and %.5f; neither is in the top set) and give the clean "
                    "D2-vs-D1 and algorithm contrasts of the report-only sections." % (pt_c["A1-P1r3-D1-16k"], pt_c["A1-P1-D1-16k"]))},
        {"id": 4, "source": "known before this analysis", "title": "Round 2 is exploratory (garden of forking paths)",
         "text": "The 6 R2-* tokenizers were built 16:47-17:00 UTC, after the round-1 screen ended (16:41 UTC), and "
                 "WAVES_round2.json was written 17:23 UTC, after the round-1 Stage-4 power check on seeds 1-3 (17:22 UTC). "
                 "They combine the recipe of the best Stage-3 tokenizer with the algorithms that looked best.",
         "impact": ("The chosen tokenizer (%s), the best-mean candidate (%s) and every top-set member are round-2 "
                    "tokenizers, so the dev ranking is conditioned on dev and the dev Δ is optimistic (winner's "
                    "curse). The sealed test split, used once (PLAN §7.6), is the guard. Confirmatory reading: restricted "
                    "to round 1 the same rule chooses %s, Δ vs baseline %+.3f %% [%+.3f, %+.3f] (improvement claim holds "
                    "there too)." % (chosen, best, r1["chosen"], r1["chosen_vs_baseline_pct"],
                                     *r1["chosen_vs_baseline_ci95_pct"]))},
        {"id": 5, "source": "known before this analysis", "title": "Learning rate at the edge of the grid",
         "text": ("The PLAN §5 sweep on the baseline chose 1e-3, the lowest grid value (bpb %.5f vs %.5f at 3e-3 and %.5f "
                  "at 6e-3); round 2 reused 1e-3 without a sweep. The finalist re-sweep (report-only) confirms 1e-3 beats "
                  "3e-3 and 6e-3 for all four finalists: %s."
                  % (lrs[0.001], lrs[0.003], lrs[0.006], "; ".join(rs(c) for c in lr_tab))),
         "impact": "No LR below 1e-3 was tried, so the optimum may lie below the grid, and it may differ between "
                   "tokenizers. The chosen tokenizer and the best-mean candidate are both among the re-swept finalists."},
        {"id": 6, "source": "found or decided in this analysis", "title": "Conditional rank 8 not admitted to the PLAN §7 ranking",
         "text": ("A6-SBPE-P1-D1-32k-t080 was trained unconditionally on the GPU. Its PLAN §2.2 condition (rank 2 or rank 5 "
                  "beats rank 1 in Stage-3 mean bpb) is false: %.5f and %.5f vs %.5f. Moreover rank 2 beat rank 5 at Stage 3, "
                  "so the valid rank-8 tokenizer would have been a t/T = 0.9 32k build, which does not exist."
                  % (pt_s["A6-SBPE-P1-D1-16k-t090"], pt_s["A6-SBPE-P1-D1-16k-t080"], pt_s[BASELINE])),
         "impact": ("Reported (SuperBPE section) but outside the Holm family (k = %d, not %d). Admitting it anyway changes "
                    "nothing: same best, top set and choice (sensitivity 'rank8_included')." % (D["k"], D["k"] + 1))},
        {"id": 7, "source": "found or decided in this analysis", "title": "5 seeds resampled per candidate, not 3",
         "text": "PLAN §6 says 'resample its 3 seeds'. The power check added seeds 4 and 5 for every candidate, so the "
                 "bootstrap resamples the 5 seeds each candidate has (Stage 3: its 3).",
         "impact": ("Using only seeds 1-3 changes the best-mean candidate to %s and widens the top set to %d members, "
                    "but the chosen tokenizer is the same (%s)." % (sens["confirm_seeds_1to3"]["best"],
                                                                   len(sens["confirm_seeds_1to3"]["top_set"]),
                                                                   sens["confirm_seeds_1to3"]["chosen"]))},
        {"id": 8, "source": "found or decided in this analysis", "title": "Operational definitions PLAN leaves open",
         "text": ("delta = 0.3 %% of the baseline's Stage-4 mean bpb = %.6f bpb (absolute, used for every TOST). "
                  "p = 2 min(P(Δ*≤0), P(Δ*≥0)) has a resolution of 1/10,000; p = 0 means p < 1e-4. "
                  "'Significantly worse' = Holm-adjusted p <= 0.05 with a positive point Δ. Tie-breaker (a): a stock HF "
                  "tokenizer.json reproduces the ids the LM used on 100 %% of dev (dev_strict and dev_permissive); "
                  "(c): all learned tokens with train_D1 frequency < 20 (train_D1 is the Stage-4 training stream; "
                  "summary.json R1); (d): mean over the four PLAN §4.2 perturbations of the share of affected words "
                  "whose segmentation changes; (e): effective vocabulary. Tie-breakers are lexicographic, without "
                  "tolerance." % out["protocol"]["delta_abs_bpb"]),
         "impact": ("Only (a) and (b) were reached. The choice does not depend on (a): %s also has the highest dev "
                    "bytes/token of the three top-set members, so (b) alone gives the same result, including under a "
                    "dev_strict-only reading of (a) (R2-A4-SPnat-D2-32k's export matches on 836/836 dev_strict "
                    "documents and differs on 1/1,358 dev_permissive). The +1-corrected p-value gives the same Holm "
                    "decisions." % chosen)},
        {"id": 9, "source": "found or decided in this analysis", "title": "Seed count below the PLAN §6 power formula",
         "text": ("Pooled 5-seed relative s.d. over all 18 candidates: %.3f %% -> n = %d seeds for Δ_min = 0.5 %%. It is "
                  "driven by two SentencePiece D1 arms (%s); without them it is %.3f %% -> %d seeds."
                  % (power["stage4_5seeds_pooled"]["all18"]["pooled_rel_sd_pct"],
                     power["stage4_5seeds_pooled"]["all18"]["n_needed"],
                     ", ".join("%s %.2f %%" % (c, power["per_candidate"][c]["rel_sd_5seeds_pct"])
                               for c in power["stage4_5seeds_pooled"]["without_high_variance"]["excluded"]),
                     power["stage4_5seeds_pooled"]["without_high_variance"]["pooled_rel_sd_pct"],
                     power["stage4_5seeds_pooled"]["without_high_variance"]["n_needed"])),
         "impact": ("The gaps inside the top set (%s %% vs the best) are below the achieved 80 %%-power detectable "
                    "effect of those comparisons (%s %%): they are 'not distinguishable at our power', never 'equal' "
                    "(TOST did not establish equivalence either)." % (", ".join("%+.3f" % x for x in within),
                                                                      ", ".join("%.2f" % x for x in top_mde)))},
        {"id": 10, "source": "inherited from WAVES pending_user_decisions", "title": "Open user decisions not resolved here",
         "text": ("(1) Data mix D2 was fragile at Stage 1 and awaits confirmation; the LM now favours D2 at 16k: "
                  "A1-P1r3-D2-16k vs A1-P1r3-D1-16k %+.3f %% [%+.3f, %+.3f]. (2) SuperBPE multi-word G2 was passed under the "
                  "reachability reading; under the literal 'shortest train chunk' reading ranks 2, 5, 8 and "
                  "R2-A6-SBPE-P1r3-D2-32k-t080 would be dropped. (3) The Stage-2 candidates (A6/A7/A10 round 1) were built "
                  "on P1/D1, not on rank 1's P1r3/D2." % (dm["delta_pct"], *dm["ci95_pct"])),
         "impact": "None of these changes the top set: no SuperBPE, A7 or round-1 A10 tokenizer is near it, and every "
                   "top-set member is a D2 tokenizer that the LM favours."},
        {"id": 11, "source": "found or decided in this analysis", "title": "G1 on test not yet run",
         "text": "PLAN G1 covers dev and test. Every candidate passed G1 on dev_strict and dev_permissive; the test half is "
                 "deferred to Stage 5 for every candidate (test discipline), as in Stages 1-2 and round 2.",
         "impact": "The TestBundle phase must run G1 on test for the test candidates before scoring them."},
        {"id": 12, "source": "from the collection, COLLECT.md", "title": "Minor data flags",
         "text": ("One screening run is 0.035 pp over the 1 %% bytes_seen tolerance (%s); screening is not used by the "
                  "decision. The round-2 screen curve of the top two crossed between 75 %% and 100 %% (PLAN §5: send them "
                  "to Stage 4), which ran for everyone anyway. The incomplete, non-pre-registered 'large' stage is ignored."
                  % ", ".join(f["file"] for f in flags)),
         "impact": "None."},
    ]
    return dev


# ----------------------------------------------------------------------------------------------- main
def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    rows, checks = load_inputs()
    dup_pairs = verify_duplicates(rows)
    conf = load_stage(rows, "confirm", checks)
    scr = load_stage(rows, "screen", checks)
    cands = sorted(conf)                              # fixed RNG order: sorted candidate ids
    assert sorted(scr) == cands and len(cands) == 18
    for c in cands:
        assert sorted(conf[c]) == [1, 2, 3, 4, 5], c
        assert sorted(scr[c]) == [1, 2, 3], c
    any_rec = conf[BASELINE][1]
    uids, nbytes = any_rec["uids"], any_rec["bytes"]
    for st in (conf, scr):
        for c in st:
            for s in st[c]:
                assert st[c][s]["uids"] == uids and np.array_equal(st[c][s]["bytes"], nbytes)
    cl = dev_clusters(uids, nbytes)
    assert cl["info"]["n_clusters"] == 32 and cl["info"]["agrees_with_dev_clusters_csv"]
    idx = {c: i for i, c in enumerate(cands)}

    # rows / metadata per candidate
    meta_row = {c: conf[c][1]["row"] for c in cands}
    W = wave_entries()
    TB = {c: tiebreak_metrics(c, W[c], meta_row[c]) for c in cands}
    for c in cands:
        assert TB[c]["tokenizer_sha256_matches_lm_runs"], c
        assert abs(TB[c]["dev_bytes_per_token"] - float(meta_row[c]["dev_bytes_per_token"])) < 1e-5, c

    BITS_C = np.stack([np.stack([conf[c][s]["bits"] for s in (1, 2, 3, 4, 5)]) for c in cands])
    BITS_S = np.stack([np.stack([scr[c][s]["bits"] for s in (1, 2, 3)]) for c in cands])
    col_idx, col_names, col_src = cl["collapsed"]
    raw_idx, raw_names, raw_src = cl["raw"]

    boot_c = Boot(BITS_C, nbytes, col_idx, col_src)
    reps_c = boot_c.reps()
    boot_s = Boot(BITS_S, nbytes, col_idx, col_src)
    reps_s = boot_s.reps()
    for c in cands:     # point estimates equal the mean of the reported per-seed bpb
        assert abs(boot_c.point[idx[c]] - np.mean([conf[c][s]["bpb"] for s in conf[c]])) < 1e-12
        assert abs(boot_s.point[idx[c]] - np.mean([scr[c][s]["bpb"] for s in scr[c]])) < 1e-12

    pt_c = {c: float(boot_c.point[idx[c]]) for c in cands}
    pt_s = {c: float(boot_s.point[idx[c]]) for c in cands}
    delta_abs = DELTA_REL * pt_c[BASELINE]

    # ---- PLAN §7.1 gate + conditional rank 8 (applied at analysis time, WAVES_colab.json colab_note)
    by_rank = {}
    for c in cands:
        by_rank.setdefault(str(meta_row[c]["wave_rank_label"]), c)
    r1, r2, r5 = by_rank["1"], by_rank["2"], by_rank["5"]
    assert r1 == BASELINE
    cond_c = [c for c in cands if meta_row[c]["conditional_wave"] in (True, "True")]
    assert len(cond_c) == 1
    rank8 = cond_c[0]
    rank8_condition = bool(pt_s[r2] < pt_s[r1] or pt_s[r5] < pt_s[r1])
    rank8_tT_valid = bool(pt_s[r5] <= pt_s[r2])
    rank8_info = {"candidate": rank8, "condition": W[rank8].get("condition"),
                  "stage3_mean_bpb": {r1: pt_s[r1], r2: pt_s[r2], r5: pt_s[r5]},
                  "condition_holds": rank8_condition,
                  "t_over_T_rule": "rank 8 = t/T 0.8 build is valid only if mean bpb(rank 5) <= mean bpb(rank 2)",
                  "t_over_T_0.8_file_is_valid_rank8": rank8_tT_valid,
                  "admitted_to_plan7_ranking": bool(rank8_condition and rank8_tT_valid),
                  "stage4_mean_bpb": {r1: pt_c[r1], r2: pt_c[r2], r5: pt_c[r5], rank8: pt_c[rank8]}}
    gate_fail = [c for c in cands if not TB[c]["gates_pass"]]
    primary = [c for c in cands if c not in gate_fail and (c != rank8 or rank8_info["admitted_to_plan7_ranking"])]

    D = decide(primary, idx, boot_c, reps_c, TB, BASELINE, delta_abs)
    chosen, best = D["chosen"], D["best"]

    # ---- per-candidate table (confirm)
    table = []
    ranked = sorted(cands, key=lambda c: pt_c[c])
    ranked_primary = [c for c in ranked if c in primary]
    for c in ranked:
        i = idx[c]
        seeds = [conf[c][s]["bpb"] for s in (1, 2, 3, 4, 5)]
        vb = compare(i, idx[BASELINE], reps_c, boot_c.point, delta_abs) if c != BASELINE else None
        vbest = compare(i, idx[best], reps_c, boot_c.point, delta_abs) if c != best else None
        r = meta_row[c]
        table.append({
            "id": c, "in_primary_family": c in primary,
            "rank_primary": (ranked_primary.index(c) + 1) if c in primary else None,
            "rank_all": ranked.index(c) + 1,
            "origin": r["candidate_origin"], "wave_rank_label": r["wave_rank_label"],
            "family": r["family"], "algorithm": r["algorithm"], "pretokenizer": r["pretokenizer"],
            "data_mix": r["data_mix"], "vocab": int(r["n_vocab"]), "encoder_kind": r["encoder_kind"],
            "bundle": r["bundle"], "ctx_tokens": int(r["ctx_tokens"]), "params_total": int(r["params_total"]),
            "params_non_embedding": int(r["params_non_embedding"]),
            "train_bytes_per_token": float(r["train_bytes_per_token"]),
            "n_seeds": 5, "mean_bpb": pt_c[c], "seed_bpb": seeds, "seed_sd_rel_pct": 100 * rel_sd(seeds),
            "vs_baseline": vb, "vs_best": vbest,
            "holm_p_vs_best": D["holm_adjusted_p"].get(c) if c in primary else None,
            "significantly_worse_than_best": D["significantly_worse"].get(c) if c in primary else None,
            "tost_equivalent_to_best": D["tost_equivalent_to_best"].get(c) if c in primary else None,
            "in_top_set": c in D["top_set"],
            "stage3_mean_bpb": pt_s[c], "tiebreak": TB[c]})

    # ---- improvement claim
    vs_base = D["chosen_vs_baseline"]
    claim_text = (("%s improves on the standard recipe A1-P1r3-D2-16k on dev_strict: Δ = %+.3f %% bpb, 95 %% CI "
                   "[%+.3f %%, %+.3f %%] (hierarchical cluster bootstrap, 5 seeds each)")
                  % (chosen, vs_base["delta_pct"], vs_base["ci95_pct"][0], vs_base["ci95_pct"][1])
                  if D["improvement_claim"] else
                  "no measurable difference from standard BPE; chosen for efficiency")

    # ---- test candidates (fixed here, before anything touches test)
    report_only = []
    if best != chosen and best in D["top_set"]:
        report_only.append({"id": best, "why": "best mean dev bpb (rank 1); lost tie-breaker (a) to the chosen tokenizer"})
    r1_top = [c for c in D["tiebreak_order"] if meta_row[c]["round"] in (1, "1") and c != BASELINE
              and c not in [x["id"] for x in report_only] and c != chosen]
    if r1_top and len(report_only) < 2:
        c = sorted(r1_top, key=lambda c: pt_c[c])[0]
        report_only.append({"id": c, "why": "best pre-registered round-1 candidate in the top set; control for the "
                                            "round-2 garden-of-forking-paths selection"})
    rest = [c for c in sorted(D["top_set"], key=lambda c: pt_c[c])
            if c not in (chosen, BASELINE) and c not in [x["id"] for x in report_only]]
    while len(report_only) < 2 and rest:
        report_only.append({"id": rest.pop(0), "why": "next top-set member by mean dev bpb"})
    test_candidates = [{"id": chosen, "role": "chosen (PLAN §7.6 one-shot test)"},
                       {"id": BASELINE, "role": "baseline (PLAN §7.6 one-shot test)"}] + \
                      [{"id": x["id"], "role": "report-only (top-set member; never re-selects)", "why": x["why"]}
                       for x in report_only]

    # ---- sensitivity analyses (report-only)
    sens = collections.OrderedDict()

    def sens_summary(Dx, bx=None):
        cvb = Dx["chosen_vs_baseline"]
        return {"k": Dx["k"], "best": Dx["best"], "top_set": Dx["top_set"], "chosen": Dx["chosen"],
                "improvement_claim": Dx["improvement_claim"],
                "chosen_vs_baseline_pct": None if cvb is None else cvb["delta_pct"],
                "chosen_vs_baseline_ci95_pct": None if cvb is None else cvb["ci95_pct"],
                "holm_adjusted_p": Dx["holm_adjusted_p"]}

    b_raw = Boot(BITS_C, nbytes, raw_idx, raw_src)
    D_raw = decide(primary, idx, b_raw, b_raw.reps(), TB, BASELINE, delta_abs)
    sens["raw_manifest_groups"] = dict(sens_summary(D_raw), what="PLAN §6 sensitivity: manifest groups as they are "
                                       "(per-record web groups kept): %d groups" % len(raw_names))
    D_r8 = decide(sorted(set(primary) | {rank8}), idx, boot_c, reps_c, TB, BASELINE, delta_abs)
    sens["rank8_included"] = dict(sens_summary(D_r8), what="conditional rank 8 admitted to the Holm family anyway")
    D_p1 = decide(primary, idx, boot_c, reps_c, TB, BASELINE, delta_abs, pkey="p_plus1")
    sens["p_plus1"] = dict(sens_summary(D_p1), what="p = 2 min((#{D*<=0}+1)/(R+1), (#{D*>=0}+1)/(R+1))")
    r1_family = [c for c in primary if meta_row[c]["round"] in (1, "1")]
    D_r1 = decide(r1_family, idx, boot_c, reps_c, TB, BASELINE, delta_abs)
    sens["round1_only"] = dict(sens_summary(D_r1), what="confirmatory set: round-1 candidates only (round-2 "
                               "exploratory tokenizers removed; conditional rank 8 excluded as in the primary)")
    b3 = Boot(BITS_C[:, :3, :], nbytes, col_idx, col_src)
    D_3 = decide(primary, idx, b3, b3.reps(), TB, BASELINE, DELTA_REL * float(b3.point[idx[BASELINE]]))
    sens["confirm_seeds_1to3"] = dict(sens_summary(D_3), what="only the 3 seeds PLAN §5 planned (seeds 4-5 dropped)")

    # ---- per-source (report-only)
    per_source = collections.OrderedDict()
    for src in sorted(set(col_src)):
        rs = boot_c.reps(source=src)
        ps = boot_c.point_source(src)
        n_cl = sum(1 for s in col_src if s == src)
        ent = {"n_clusters": n_cl, "bytes": int(sum(boot_c.KB[k] for k in range(len(col_src)) if col_src[k] == src))}
        for c in D["top_set"] + [BASELINE]:
            ent.setdefault("mean_bpb", {})[c] = float(ps[idx[c]])
        for c in D["top_set"]:
            ent.setdefault("vs_baseline", {})[c] = compare(idx[c], idx[BASELINE], rs, ps, delta_abs)
        per_source[src] = ent

    # ---- report-only contrasts (same replicates as the primary decision)
    def C(a, b, R=reps_c, B=boot_c):
        x = compare(idx[a], idx[b], R, B.point, delta_abs)
        x.update({"a": a, "b": b})
        return x

    contrasts = collections.OrderedDict()
    contrasts["vocab_curve_A1_P1r3_D2"] = {
        "sizes": {"8k": "A1-P1r3-D2-8k", "16k": "A1-P1r3-D2-16k", "32k": "A1-P1r3-D2-32k", "48k": "R2-A1-P1r3-D2-48k"},
        "confirm_mean_bpb": {k: pt_c[v] for k, v in {"8k": "A1-P1r3-D2-8k", "16k": "A1-P1r3-D2-16k",
                                                   "32k": "A1-P1r3-D2-32k", "48k": "R2-A1-P1r3-D2-48k"}.items()},
        "screen_mean_bpb": {k: pt_s[v] for k, v in {"8k": "A1-P1r3-D2-8k", "16k": "A1-P1r3-D2-16k",
                                                  "32k": "A1-P1r3-D2-32k", "48k": "R2-A1-P1r3-D2-48k"}.items()},
        "steps": [C("A1-P1r3-D2-16k", "A1-P1r3-D2-8k"), C("A1-P1r3-D2-32k", "A1-P1r3-D2-16k"),
                  C("R2-A1-P1r3-D2-48k", "A1-P1r3-D2-32k")],
        "vs_16k": [C("A1-P1r3-D2-8k", "A1-P1r3-D2-16k"), C("A1-P1r3-D2-32k", "A1-P1r3-D2-16k"),
                   C("R2-A1-P1r3-D2-48k", "A1-P1r3-D2-16k")],
        "screen_steps": [C("A1-P1r3-D2-16k", "A1-P1r3-D2-8k", reps_s, boot_s),
                         C("A1-P1r3-D2-32k", "A1-P1r3-D2-16k", reps_s, boot_s),
                         C("R2-A1-P1r3-D2-48k", "A1-P1r3-D2-32k", reps_s, boot_s)],
        "other_families_32k_to_48k": [C("R2-A10-MinGram-P1r3-D2-48k", "R2-A10-MinGram-P1r3-D2-32k"),
                                      C("R2-A4-SPnat-D2-48k", "R2-A4-SPnat-D2-32k")]}
    contrasts["data_mix_and_pretokenizer"] = {
        "D2_vs_D1_A1_P1r3_16k": C("A1-P1r3-D2-16k", "A1-P1r3-D1-16k"),
        "P1r3_vs_P1_A1_D1_16k": C("A1-P1r3-D1-16k", "A1-P1-D1-16k"),
        "stage1_recipe_P1r3D2_vs_P1D1_A1_16k": C("A1-P1r3-D2-16k", "A1-P1-D1-16k"),
        "screen_D2_vs_D1_A1_P1r3_16k": C("A1-P1r3-D2-16k", "A1-P1r3-D1-16k", reps_s, boot_s),
        "screen_P1r3_vs_P1_A1_D1_16k": C("A1-P1r3-D1-16k", "A1-P1-D1-16k", reps_s, boot_s),
        "recipe_change_confounded": {
            "MinGram_16k_P1D1_to_32k_P1r3D2": C("R2-A10-MinGram-P1r3-D2-32k", "A10-MinGram-P1-D1-16k"),
            "A4_16k_D1_to_32k_D2": C("R2-A4-SPnat-D2-32k", "A4-SPnat-D1-16k"),
            "SuperBPE_32k_P1D1_to_P1r3D2": C("R2-A6-SBPE-P1r3-D2-32k-t080", "A6-SBPE-P1-D1-32k-t080")}}
    contrasts["algorithm_at_equal_settings"] = {
        "P1_D1_16k_vs_A1_BPE": [C(x, "A1-P1-D1-16k") for x in
                                ("A10-MinGram-P1-D1-16k", "A7-PickyBPE-P1-D1-16k-tau0.9",
                                 "A6-SBPE-P1-D1-16k-t090", "A6-SBPE-P1-D1-16k-t080")],
        "SPnat_D1_16k_Unigram_vs_SPBPE": C("A4-SPnat-D1-16k", "A3-SPnat-D1-16k"),
        "D1_16k_SP_vs_HF_BPE_P1": [C("A4-SPnat-D1-16k", "A1-P1-D1-16k"), C("A3-SPnat-D1-16k", "A1-P1-D1-16k")],
        "P1r3_D2_32k_vs_A1_BPE": [C(x, "A1-P1r3-D2-32k") for x in
                                  ("R2-A10-MinGram-P1r3-D2-32k", "R2-A4-SPnat-D2-32k", "R2-A6-SBPE-P1r3-D2-32k-t080")],
        "P1r3_D2_48k_vs_A1_BPE": [C(x, "R2-A1-P1r3-D2-48k") for x in
                                  ("R2-A10-MinGram-P1r3-D2-48k", "R2-A4-SPnat-D2-48k")],
        "MinGram_vs_Unigram_A4": [C("A10-MinGram-P1-D1-16k", "A4-SPnat-D1-16k"),
                                  C("R2-A10-MinGram-P1r3-D2-32k", "R2-A4-SPnat-D2-32k"),
                                  C("R2-A10-MinGram-P1r3-D2-48k", "R2-A4-SPnat-D2-48k")]}

    # SuperBPE: both stages + confirm-curve (0.5 MB subset, descriptive)
    def curve_mean(stage_d, c, frac):
        v = []
        for s in stage_d[c]:
            for pnt in stage_d[c][s]["curve"]:
                if abs(pnt["frac"] - frac) < 1e-9:
                    v.append(pnt["bpb"])
        return float(np.mean(v))

    sbpe_pairs = [("A6-SBPE-P1-D1-16k-t090", "A1-P1-D1-16k"), ("A6-SBPE-P1-D1-16k-t080", "A1-P1-D1-16k"),
                  ("A6-SBPE-P1-D1-32k-t080", "A1-P1r3-D2-32k"), ("R2-A6-SBPE-P1r3-D2-32k-t080", "A1-P1r3-D2-32k"),
                  ("A6-SBPE-P1-D1-16k-t090", "A6-SBPE-P1-D1-16k-t080")]
    sbpe = {"pairs": []}
    for a, b in sbpe_pairs:
        ent = {"a": a, "b": b, "screen": C(a, b, reps_s, boot_s), "confirm": C(a, b),
               "confirm_subset_curve_delta_pct": {}, "screen_subset_curve_delta_pct": {}}
        for fr in (0.25, 0.5, 0.75, 1.0):
            ca, cb = curve_mean(conf, a, fr), curve_mean(conf, b, fr)
            sa, sb = curve_mean(scr, a, fr), curve_mean(scr, b, fr)
            ent["confirm_subset_curve_delta_pct"][str(fr)] = 100 * (ca - cb) / cb
            ent["screen_subset_curve_delta_pct"][str(fr)] = 100 * (sa - sb) / sb
        sbpe["pairs"].append(ent)
    sbpe["ranks"] = {c: {"screen_rank": sorted(cands, key=lambda x: pt_s[x]).index(c) + 1,
                         "confirm_rank": ranked.index(c) + 1, "screen_bpb": pt_s[c], "confirm_bpb": pt_c[c],
                         "ctx_tokens": int(meta_row[c]["ctx_tokens"]),
                         "train_bytes_per_token": float(meta_row[c]["train_bytes_per_token"]),
                         "dev_bytes_per_token": TB[c]["dev_bytes_per_token"]}
                     for c in cands if meta_row[c]["family"] == "A6"}
    sbpe["rank8"] = rank8_info
    contrasts["superbpe"] = sbpe

    # ---- Stage 3 vs Stage 4
    scr_rank = sorted(cands, key=lambda c: pt_s[c])
    s34 = {"screen_rank": {c: scr_rank.index(c) + 1 for c in cands},
           "confirm_rank": {c: ranked.index(c) + 1 for c in cands},
           "screen_mean_bpb": pt_s, "confirm_mean_bpb": pt_c,
           "screen_vs_baseline": {c: C(c, BASELINE, reps_s, boot_s) for c in cands if c != BASELINE},
           "confirm_vs_baseline": {c: C(c, BASELINE) for c in cands if c != BASELINE}}
    s34["rank_correlation_all18"] = kendall_spearman([pt_s[c] for c in cands], [pt_c[c] for c in cands])
    nons = [c for c in cands if meta_row[c]["family"] != "A6"]
    s34["rank_correlation_without_superbpe"] = kendall_spearman([pt_s[c] for c in nons], [pt_c[c] for c in nons])
    rev = []
    n_pairs = 0
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            n_pairs += 1
            cs = C(a, b, reps_s, boot_s)
            cc = C(a, b)
            flip = (cs["delta"] < 0) != (cc["delta"] < 0)
            if not flip:
                continue
            sig_s, sig_c = cs["ci95_excludes_0"], cc["ci95_excludes_0"]
            rev.append({"a": a, "b": b, "screen_delta_pct": cs["delta_pct"], "screen_ci95_pct": cs["ci95_pct"],
                        "confirm_delta_pct": cc["delta_pct"], "confirm_ci95_pct": cc["ci95_pct"],
                        "significant_in_both": bool(sig_s and sig_c),
                        "significant_screen_only": bool(sig_s and not sig_c),
                        "significant_confirm_only": bool(sig_c and not sig_s)})
    s34["n_pairs"] = n_pairs
    s34["reversals"] = sorted(rev, key=lambda r: (not r["significant_in_both"], -abs(r["confirm_delta_pct"])))
    s34["n_reversals"] = len(rev)
    s34["n_significant_both"] = sum(r["significant_in_both"] for r in rev)
    s34["screen_top_confirm_not_top"] = [c for c in scr_rank[:5] if ranked.index(c) >= 5]

    # ---- power check (PLAN §6)
    def pooled(cs, seeds, stage_d):
        v = [rel_sd([stage_d[c][s]["bpb"] for s in seeds]) for c in cs]
        return float(math.sqrt(np.mean(np.square(v))))

    r1c = [c for c in cands if meta_row[c]["round"] in (1, "1")]
    r2_bundle = [c for c in cands if meta_row[c]["round"] in (2, "2")] + \
                [c for c in cands if meta_row[c]["reference_arm_in_both_bundles"] in (True, "True")]
    base_scr_sd = rel_sd([scr[BASELINE][s]["bpb"] for s in (1, 2, 3)])
    per_cand = {c: {"rel_sd_5seeds_pct": 100 * rel_sd([conf[c][s]["bpb"] for s in (1, 2, 3, 4, 5)]),
                    "n_needed_own_sd": n_needed(rel_sd([conf[c][s]["bpb"] for s in (1, 2, 3, 4, 5)]))}
                for c in cands}
    hi_var = [c for c in cands if per_cand[c]["rel_sd_5seeds_pct"] > 0.5]
    mde_vs_base = {c: s34["confirm_vs_baseline"][c]["mde80_pct"] for c in cands if c != BASELINE}
    power = {
        "formula": "n = ceil(15.7 sigma^2 / Delta_min^2), sigma = relative seed s.d., Delta_min = 0.5 %",
        "stage3_baseline_first_wave": {"rel_sd_pct": 100 * base_scr_sd, "n_needed": n_needed(base_scr_sd),
                                       "verdict": "3 seeds suffice" if base_scr_sd <= 0.002 else "more seeds"},
        "stage4_seeds1to3_pooled": {
            "round1_bundle": {"candidates": len(r1c), "pooled_rel_sd_pct": 100 * pooled(r1c, (1, 2, 3), conf),
                              "n_needed": n_needed(pooled(r1c, (1, 2, 3), conf))},
            "round2_bundle": {"candidates": len(r2_bundle), "pooled_rel_sd_pct": 100 * pooled(r2_bundle, (1, 2, 3), conf),
                              "n_needed": n_needed(pooled(r2_bundle, (1, 2, 3), conf))},
            "action": "sigma > 0.2 % -> seeds 4 and 5 run for every candidate"},
        "stage4_5seeds_pooled": {
            "all18": {"pooled_rel_sd_pct": 100 * pooled(cands, (1, 2, 3, 4, 5), conf),
                      "n_needed": n_needed(pooled(cands, (1, 2, 3, 4, 5), conf))},
            "primary17": {"pooled_rel_sd_pct": 100 * pooled(primary, (1, 2, 3, 4, 5), conf),
                          "n_needed": n_needed(pooled(primary, (1, 2, 3, 4, 5), conf))},
            "without_high_variance": {"excluded": hi_var,
                                      "pooled_rel_sd_pct": 100 * pooled([c for c in cands if c not in hi_var],
                                                                        (1, 2, 3, 4, 5), conf),
                                      "n_needed": n_needed(pooled([c for c in cands if c not in hi_var],
                                                                  (1, 2, 3, 4, 5), conf))}},
        "per_candidate": per_cand,
        "achieved_mde80_pct_vs_baseline": mde_vs_base,
        "achieved_mde80_pct_vs_baseline_median": float(np.median(list(mde_vs_base.values()))),
        "achieved_mde80_pct_chosen_vs_baseline": vs_base["mde80_pct"] if vs_base else None,
        "achieved_mde80_pct_vs_best": {c: D["comparisons_vs_best"][c]["mde80_pct"] for c in D["comparisons_vs_best"]},
        "mde_definition": "(z_.975 + z_.8) x bootstrap s.d. of the relative D (clusters + seeds)"}

    # ---- LR re-sweep (report-only)
    lr = []
    for r in rows:
        if r["stage"] == "confirm_resweep" and not r["duplicate_of"]:
            lr.append({"candidate": r["candidate"], "lr": r["lr"], "bpb": r["bpb"], "bundle": r["bundle"]})
    lr_tab = collections.OrderedDict()
    for e in lr:
        lr_tab.setdefault(e["candidate"], {"1e-3 (confirm s1)": conf[e["candidate"]][1]["bpb"]})
        lr_tab[e["candidate"]]["%.0e" % e["lr"]] = e["bpb"]
    lr_sweep = [{"lr": r["lr"], "bpb": r["bpb"]} for r in rows if r["stage"] == "lrsweep"]

    flags = [{"file": conf[c][s]["row"]["file"], "flags": conf[c][s]["tolerance_flags"]}
             for c in cands for s in conf[c] if conf[c][s]["tolerance_flags"]] + \
            [{"file": scr[c][s]["row"]["file"], "flags": scr[c][s]["tolerance_flags"]}
             for c in cands for s in scr[c] if scr[c][s]["tolerance_flags"]]

    out = collections.OrderedDict()
    out["what"] = ("PLAN §6-§7 pre-registered decision over every LM-arbiter candidate of both Colab rounds, on "
                   "Stage 4 'confirm' (d=192, full permissive train split, 1 epoch, LR 1e-3, 5 seeds), dev_strict")
    out["generated_utc"] = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["code"] = {"decide.py": sha256_file(os.path.abspath(__file__))}
    out["test_split"] = "never read, encoded or scored by this analysis"
    out["inputs"] = dict(checks, bundles=sorted({meta_row[c]["bundle"] for c in cands}),
                         duplicates_dropped=len(dup_pairs), duplicates_bitwise_identical=all(p["bitwise_identical"] for p in dup_pairs),
                         tolerance_flags=flags)
    out["clusters"] = cl["info"]
    out["cluster_list"] = [{"cluster": n, "source": s, "bytes": int(boot_c.KB[k]),
                            "docs": int(np.sum(col_idx == k))} for k, (n, s) in enumerate(zip(col_names, col_src))]
    out["protocol"] = {"n_rep": N_REP, "rng": "numpy.random.default_rng(%d)" % RNG_SEED, "alpha": ALPHA,
                       "delta_rel": DELTA_REL, "delta_abs_bpb": delta_abs,
                       "rng_order": "cluster draws (sources in sorted order) then seed draws (candidates in sorted id order)",
                       "seeds_resampled": "each candidate's 5 confirm seeds (3 at screen), with replacement, independently",
                       "p_value": "2 min(P(D*<=0), P(D*>=0)); resolution 1/10,000",
                       "candidate_rng_order": cands}
    out["baseline"] = BASELINE
    out["gate_failures"] = gate_fail
    out["rank8"] = rank8_info
    out["primary_family"] = primary
    out["decision"] = {k: D[k] for k in ("k", "best", "top_set", "tiebreak_order", "tiebreak_trace", "chosen",
                                         "improvement_claim", "holm_adjusted_p", "significantly_worse",
                                         "tost_equivalent_to_best")}
    out["decision"]["chosen_vs_baseline"] = vs_base
    out["decision"]["improvement_claim_text"] = claim_text
    out["decision"]["chosen_vs_best"] = D["comparisons_vs_best"].get(chosen)
    out["test_candidates"] = test_candidates
    out["ranking"] = table
    out["sensitivity"] = sens
    out["per_source"] = per_source
    out["contrasts"] = contrasts
    out["stage3_vs_stage4"] = s34
    out["power"] = power
    out["lr"] = {"stage3_sweep_baseline": lr_sweep, "finalist_resweep": lr_tab}
    out["duplicate_pairs"] = dup_pairs
    out["deviations"] = deviations(out, pt_s, pt_c, D, TB, sens, contrasts, power, lr_tab, lr_sweep, flags)
    out["test_candidates_fixed"] = {
        "utc": out["generated_utc"],
        "ids": [x["id"] for x in test_candidates],
        "sha256_of_ids": hashlib.sha256("\n".join(x["id"] for x in test_candidates).encode("utf-8")).hexdigest(),
        "test_split_touched_before_fixing": False,
        "note": "fixed by this analysis on dev_strict only; the test split has not been read, encoded or scored"}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # console summary
    print("primary family k=%d best=%s chosen=%s claim=%s" % (D["k"], best, chosen, D["improvement_claim"]))
    print("top set:", D["top_set"])
    for t in D["tiebreak_trace"]:
        print(" tie-break", t)
    print("chosen vs baseline: %+.3f%% [%+.3f, %+.3f]" % (vs_base["delta_pct"], *vs_base["ci95_pct"]))
    for r in table:
        vb = r["vs_baseline"]
        vbe = r["vs_best"]
        print("%2d %-30s %.5f sd%.3f%%  vsBase %s  vsBest %s holm=%s top=%s" % (
            r["rank_all"], r["id"], r["mean_bpb"], r["seed_sd_rel_pct"],
            "-" if vb is None else "%+.3f [%+.3f,%+.3f] p=%.4f" % (vb["delta_pct"], *vb["ci95_pct"], vb["p"]),
            "-" if vbe is None else "%+.3f [%+.3f,%+.3f] p=%.4f eq=%s" % (vbe["delta_pct"], *vbe["ci95_pct"], vbe["p"],
                                                                         vbe["tost_equivalent"]),
            None if r["holm_p_vs_best"] is None else round(r["holm_p_vs_best"], 5), r["in_top_set"]))
    for k, v in sens.items():
        print("sens", k, v["best"], v["chosen"], v["top_set"], v["improvement_claim"], v["chosen_vs_baseline_ci95_pct"])
    print("test candidates:", test_candidates)
    print("done in %.1fs" % (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds())


if __name__ == "__main__":
    main()
