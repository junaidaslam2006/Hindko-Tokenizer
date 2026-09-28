# -*- coding: utf-8 -*-
"""Audit addendum: dev groups that are not fully held out (records moved to train by the split's leak rule).
Quantifies them in train_D1 / train_D2 and checks whether the chosen tokenizer's gain differs between dev documents
in shared groups and in fully held-out groups (cluster-bootstrap CI). dev_strict only; no test data.
Writes analysis/audit/audit_groups.json.
"""
import csv
import json
import os

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
DATA = os.path.join(TOK, "data")
RES = os.path.join(TOK, "lm", "colab_results")
OUT = os.path.join(TOK, "analysis", "audit", "audit_groups.json")
CHOSEN, BASE = "R2-A10-MinGram-P1r3-D2-48k", "A1-P1r3-D2-16k"
OTHER = ["R2-A4-SPnat-D2-32k", "R2-A10-MinGram-P1r3-D2-32k", "R2-A1-P1r3-D2-48k"]


def load(n):
    return [json.loads(l) for l in open(os.path.join(DATA, n + ".jsonl"), encoding="utf-8")]


def grams(ws, n=8):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    d1, d2, ds = load("train_D1"), load("train_D2"), load("dev_strict")
    gdev = {r["group"] for r in ds}
    shared1 = sorted(gdev & {r["group"] for r in d1})
    shared2 = sorted(gdev & {r["group"] for r in d2})
    moved1 = [r for r in d1 if r["group"] in gdev]
    moved2 = [r for r in d2 if r["group"] in gdev]
    # 8-gram containment of each dev doc in the moved records of its own group
    by_g = {}
    for r in moved2:
        by_g.setdefault(r["group"], set()).update(grams(r["text"].split()))
    cont = {}
    for r in ds:
        g = grams(r["text"].split())
        if r["group"] in by_g and g:
            cont[r["uid"]] = len(g & by_g[r["group"]]) / len(g)
    out = {"shared_groups_D1": len(shared1), "shared_groups_D2": len(shared2), "dev_groups": len(gdev),
           "moved_records_D1": len(moved1), "moved_bytes_D1": sum(len(r["text"].encode()) for r in moved1),
           "moved_records_D2": len(moved2), "moved_bytes_D2": sum(len(r["text"].encode()) for r in moved2),
           "D1_bytes": sum(len(r["text"].encode()) for r in d1),
           "dev_docs_in_shared_groups": sum(1 for r in ds if r["group"] in shared1),
           "dev_bytes_share_in_shared_groups": round(sum(len(r["text"].encode()) for r in ds if r["group"] in shared1)
                                                     / sum(len(r["text"].encode()) for r in ds), 4),
           "own_group_8gram_containment_in_moved_D2": {
               "docs": len(cont), "max": round(max(cont.values()), 4) if cont else None,
               "mean": round(float(np.mean(list(cont.values()))), 5) if cont else None,
               "docs_ge_0.1": sum(1 for v in cont.values() if v >= 0.1)}}
    # LM: gain in shared vs fully held-out groups, cluster bootstrap within each subset
    table = json.load(open(os.path.join(RES, "results_table.json"), encoding="utf-8"))
    rows = [r for r in table if r["stage"] == "confirm" and not r.get("duplicate_of") and r["candidate"] in [CHOSEN, BASE] + OTHER]
    cl = list(csv.DictReader(open(os.path.join(RES, "dev_clusters.csv"), encoding="utf-8")))
    uid = [x["uid"] for x in cl]
    grp = {r["uid"]: r["group"] for r in ds}
    shared_mask = np.array([grp[u] in shared1 for u in uid])
    clus = [x["cluster"] for x in cl]
    byt = np.array([int(x["bytes"]) for x in cl], float)
    bits = {}
    for r in rows:
        rec = json.load(open(os.path.join(RES, r["file"].replace("/", os.sep)), encoding="utf-8"))
        bits.setdefault(r["candidate"], []).append(np.array(rec["dev"]["bits"]))
    mb = {c: np.mean(v, 0) for c, v in bits.items()}
    rng = np.random.default_rng(99)
    res = {}
    for lab, m in (("shared_groups", shared_mask), ("fully_held_out_groups", ~shared_mask)):
        names = sorted({clus[i] for i in range(len(uid)) if m[i]})
        idx = np.array([names.index(clus[i]) if m[i] else -1 for i in range(len(uid))])
        K = len(names)
        B = np.array([byt[idx == k].sum() for k in range(K)])
        row = {"docs": int(m.sum()), "bytes": int(byt[m].sum()), "clusters": K}
        for c in [CHOSEN] + OTHER:
            A = np.array([mb[c][idx == k].sum() for k in range(K)])
            Z = np.array([mb[BASE][idx == k].sum() for k in range(K)])
            obs = (A.sum() - Z.sum()) / Z.sum() * 100
            draws = rng.integers(0, K, size=(5000, K))
            bs = (A[draws].sum(1) - Z[draws].sum(1)) / Z[draws].sum(1) * 100
            row[c + "_vs_base_pct"] = [round(float(obs), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        res[lab] = row
    out["gain_shared_vs_heldout_groups (seed-mean bits; unstratified cluster bootstrap, 5,000)"] = res
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
