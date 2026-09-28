# -*- coding: utf-8 -*-
"""Audit addendum: how much of the uncertainty of the key contrasts is seed noise (reducible by more seeds on a T4)
and how much is dev-cluster heterogeneity (not reducible by seeds). dev_strict only; test never read.
Writes analysis/audit/audit_power.json.
"""
import csv
import json
import math
import os

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
RES = os.path.join(TOK, "lm", "colab_results")
OUT = os.path.join(TOK, "analysis", "audit", "audit_power.json")
PAIRS = [("R2-A10-MinGram-P1r3-D2-48k", "A1-P1r3-D2-16k"),
         ("R2-A10-MinGram-P1r3-D2-48k", "R2-A4-SPnat-D2-32k"),
         ("R2-A10-MinGram-P1r3-D2-48k", "R2-A10-MinGram-P1r3-D2-32k"),
         ("R2-A10-MinGram-P1r3-D2-48k", "R2-A1-P1r3-D2-48k"),
         ("R2-A10-MinGram-P1r3-D2-48k", "A1-P1r3-D2-32k"),
         ("R2-A10-MinGram-P1r3-D2-32k", "A1-P1r3-D2-32k"),
         ("R2-A4-SPnat-D2-32k", "R2-A10-MinGram-P1r3-D2-32k")]
R = 10000


def main():
    table = json.load(open(os.path.join(RES, "results_table.json"), encoding="utf-8"))
    need = {c for p in PAIRS for c in p}
    rows = [r for r in table if r["stage"] == "confirm" and not r.get("duplicate_of") and r["candidate"] in need]
    cl = list(csv.DictReader(open(os.path.join(RES, "dev_clusters.csv"), encoding="utf-8")))
    names = sorted({x["cluster"] for x in cl})
    idx = np.array([names.index(x["cluster"]) for x in cl])
    src = {x["cluster"]: x["source"] for x in cl}
    byt = np.array([int(x["bytes"]) for x in cl], dtype=float)
    K = len(names)
    bcl = np.array([byt[idx == k].sum() for k in range(K)])
    bits = {}
    for r in rows:
        rec = json.load(open(os.path.join(RES, r["file"].replace("/", os.sep)), encoding="utf-8"))
        b = np.array(rec["dev"]["bits"])
        bits.setdefault(r["candidate"], {})[r["seed"]] = np.array([b[idx == k].sum() for k in range(K)])
    X = {c: np.array([bits[c][s] for s in sorted(bits[c])]) for c in bits}   # S x K
    rng = np.random.default_rng(777)
    # cluster draws (stratified) as count matrices
    M = np.zeros((R, K))
    for s_ in sorted(set(src.values())):
        ks = np.array([k for k in range(K) if src[names[k]] == s_])
        d = rng.integers(0, len(ks), size=(R, len(ks)))
        for j in range(len(ks)):
            np.add.at(M, (np.arange(R), ks[d[:, j]]), 1)
    out = {"what": "variance decomposition of key contrasts: seed-only vs cluster-only bootstrap (R=10,000)", "pairs": {}}
    for a, b in PAIRS:
        Xa, Xb = X[a], X[b]
        obs = (Xa.sum(1).mean() - Xb.sum(1).mean()) / Xb.sum(1).mean() * 100
        # seed-only: all clusters fixed, resample seeds
        sa = rng.integers(0, 5, size=(R, 5)); sb = rng.integers(0, 5, size=(R, 5))
        ta = Xa.sum(1)[sa].mean(1); tb = Xb.sum(1)[sb].mean(1)
        d_seed = (ta - tb) / tb * 100
        # cluster-only: seed means fixed, resample clusters
        ma = Xa.mean(0); mb = Xb.mean(0)
        d_cl = (M @ ma - M @ mb) / (M @ mb) * 100
        se_s, se_c = float(d_seed.std(ddof=1)), float(d_cl.std(ddof=1))
        se_tot = math.sqrt(se_s ** 2 + se_c ** 2)

        def n_for(target_se):
            if se_c >= target_se:
                return None
            return int(math.ceil(5 * se_s ** 2 / (target_se ** 2 - se_c ** 2)))
        need80 = abs(obs) / 2.80 if obs != 0 else None       # se needed for 80 % power at alpha 0.05 two-sided
        # TOST at delta = 0.3 % needs the 90 % CI inside +-0.3: |obs| + 1.645 se < 0.3
        se_tost = (0.3 - abs(obs)) / 1.645 if abs(obs) < 0.3 else None
        out["pairs"][a + " vs " + b] = {
            "delta_pct": round(obs, 4), "se_seed_only_pp": round(se_s, 4), "se_cluster_only_pp": round(se_c, 4),
            "se_combined_pp": round(se_tot, 4), "seed_share_of_variance": round(se_s ** 2 / se_tot ** 2, 3),
            "seeds_per_arm_for_80pct_power_at_observed_gap": n_for(need80) if need80 else None,
            "seeds_per_arm_for_TOST_equivalence_at_0.3pct": (n_for(se_tost) if se_tost and se_tost > 0 else None),
            "note": "None = unreachable with more seeds on this dev set (cluster heterogeneity alone exceeds the target)"}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
