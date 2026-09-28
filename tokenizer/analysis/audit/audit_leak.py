# -*- coding: utf-8 -*-
"""Adversarial audit: leakage between tokenizer-training text (D2 = strict train; D1 = LM training stream) and dev.

Reads data/train_D1.jsonl, data/train_D2.jsonl, data/dev_strict.jsonl, data/dev_permissive.jsonl (no test file
exists there and none is opened) and the Stage 4 confirm records of a few candidates (per-document bits).
Writes analysis/audit/audit_leak.json.
"""
import csv
import hashlib
import json
import os

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
DATA = os.path.join(TOK, "data")
RES = os.path.join(TOK, "lm", "colab_results")
OUT = os.path.join(TOK, "analysis", "audit", "audit_leak.json")
CHOSEN = "R2-A10-MinGram-P1r3-D2-48k"
BASE = "A1-P1r3-D2-16k"
OTHERS = ["R2-A4-SPnat-D2-32k", "R2-A10-MinGram-P1r3-D2-32k", "A1-P1r3-D2-32k", "R2-A1-P1r3-D2-48k", "A1-P1-D1-16k"]
NS = (8, 13)
MIN_LINE = 40


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name + ".jsonl"), encoding="utf-8")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


VOC = {}


def ids_of(text):
    return np.fromiter((VOC.setdefault(w, len(VOC) + 1) for w in text.split()), dtype=np.uint64)


def grams(ids, n):
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    B = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * B + ids[j:j + m]
    return h


def main():
    man = json.load(open(os.path.join(DATA, "data_manifest.json"), encoding="utf-8"))
    files_ok = {v: sha(os.path.join(DATA, man["views"][v]["file"])) == man["views"][v]["sha256"]
                for v in ("train_D1", "train_D2", "dev_strict", "dev_permissive")}
    d1, d2, ds, dp = load("train_D1"), load("train_D2"), load("dev_strict"), load("dev_permissive")
    out = {"what": "leakage audit: tokenizer-training text vs dev", "files_sha_match_manifest": files_ok,
           "test_split": "no test file exists under data/; none opened"}
    u1, u2 = {r["uid"] for r in d1}, {r["uid"] for r in d2}
    us, up = {r["uid"] for r in ds}, {r["uid"] for r in dp}
    g1, g2 = {r["group"] for r in d1}, {r["group"] for r in d2}
    gs, gp = {r["group"] for r in ds}, {r["group"] for r in dp}
    c1, c2 = {r["cluster"] for r in d1}, {r["cluster"] for r in d2}
    cs = {r["cluster"] for r in ds}
    out["id_checks"] = {
        "D2_subset_of_D1": u2 <= u1, "D2_all_strict": all(r["tier"] == "strict" for r in d2),
        "uid_overlap_D1_devstrict": len(u1 & us), "uid_overlap_D1_devpermissive": len(u1 & up),
        "uid_overlap_D2_devstrict": len(u2 & us),
        "group_overlap_D1_devstrict": sorted(g1 & gs), "group_overlap_D1_devpermissive": sorted(g1 & gp),
        "group_overlap_D2_devstrict": sorted(g2 & gs),
        "bootstrap_cluster_overlap_D2_devstrict": sorted(c2 & cs),
        "bootstrap_cluster_overlap_D1_devstrict": sorted(c1 & cs),
        "devstrict_subset_of_devpermissive": us <= up,
    }
    # exact document duplicates
    h = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()
    th1 = {h(r["text"]) for r in d1}
    th2 = {h(r["text"]) for r in d2}
    out["exact_doc_duplicates"] = {"devstrict_in_D1": sum(1 for r in ds if h(r["text"]) in th1),
                                   "devstrict_in_D2": sum(1 for r in ds if h(r["text"]) in th2),
                                   "devpermissive_in_D1": sum(1 for r in dp if h(r["text"]) in th1)}
    # exact long-line overlap
    lines1, lines2 = set(), set()
    for rs, st in ((d1, lines1), (d2, lines2)):
        for r in rs:
            for ln in r["text"].split("\n"):
                ln = ln.strip()
                if len(ln) >= MIN_LINE:
                    st.add(h(ln))
    tot_b = sum(len(r["text"].encode("utf-8")) for r in ds)
    lb1 = lb2 = 0
    per_doc_line = {}
    for r in ds:
        b1 = b2 = 0
        for ln in r["text"].split("\n"):
            s = ln.strip()
            if len(s) >= MIN_LINE:
                hh = h(s)
                nb = len(s.encode("utf-8"))
                if hh in lines1:
                    b1 += nb
                if hh in lines2:
                    b2 += nb
        lb1 += b1; lb2 += b2
        per_doc_line[r["uid"]] = b2 / max(1, len(r["text"].encode("utf-8")))
    out["exact_line_overlap"] = {"min_line_chars": MIN_LINE, "devstrict_bytes": tot_b,
                                 "bytes_in_lines_seen_in_D1": lb1, "share_D1": round(lb1 / tot_b, 5),
                                 "bytes_in_lines_seen_in_D2": lb2, "share_D2": round(lb2 / tot_b, 5),
                                 "docs_with_any_D2_line": sum(1 for v in per_doc_line.values() if v > 0)}
    # word n-gram containment
    per_doc = {r["uid"]: {"bytes": len(r["text"].encode("utf-8")), "source": None} for r in ds}
    for n in NS:
        train_h = {}
        for name, rs in (("D1", d1), ("D2", d2)):
            arr = [grams(ids_of(r["text"]), n) for r in rs]
            train_h[name] = np.unique(np.concatenate(arr)) if arr else np.zeros(0, np.uint64)
        agg = {"D1": [0, 0], "D2": [0, 0]}
        short = 0
        for r in ds:
            g = grams(ids_of(r["text"]), n)
            if len(g) == 0:
                short += 1
                for name in ("D1", "D2"):
                    per_doc[r["uid"]]["c%d_%s" % (n, name)] = None
                continue
            for name in ("D1", "D2"):
                arr_ = train_h[name]
                pos = np.searchsorted(arr_, g)
                pos[pos >= len(arr_)] = len(arr_) - 1
                hit = arr_[pos] == g
                per_doc[r["uid"]]["c%d_%s" % (n, name)] = float(hit.mean())
                agg[name][0] += int(hit.sum()); agg[name][1] += len(g)
        cont = {}
        for name in ("D1", "D2"):
            vals = [(v["c%d_%s" % (n, name)], v["bytes"]) for v in per_doc.values() if v["c%d_%s" % (n, name)] is not None]
            cont[name] = {"gram_containment_overall": round(agg[name][0] / agg[name][1], 5),
                          "docs_ge_0.3": sum(1 for c, _ in vals if c >= 0.3), "docs_ge_0.5": sum(1 for c, _ in vals if c >= 0.5),
                          "docs_ge_0.8": sum(1 for c, _ in vals if c >= 0.8),
                          "bytes_share_docs_ge_0.3": round(sum(b for c, b in vals if c >= 0.3) / tot_b, 5),
                          "bytes_share_docs_ge_0.5": round(sum(b for c, b in vals if c >= 0.5) / tot_b, 5)}
        cont["docs_shorter_than_n"] = short
        out["ngram_%d" % n] = cont
        del train_h
    # link to the LM: does the chosen tokenizer gain more on dev text that overlaps its training text?
    table = json.load(open(os.path.join(RES, "results_table.json"), encoding="utf-8"))
    rows = [r for r in table if r["stage"] == "confirm" and not r.get("duplicate_of") and r["candidate"] in [CHOSEN, BASE] + OTHERS]
    bits = {}
    uids = None
    for r in rows:
        rec = json.load(open(os.path.join(RES, r["file"].replace("/", os.sep)), encoding="utf-8"))
        uids = rec["dev"]["uids"]
        bits.setdefault(r["candidate"], []).append(np.array(rec["dev"]["bits"]))
    mb = {c: np.mean(v, axis=0) for c, v in bits.items()}
    cl = {x["uid"]: x for x in csv.DictReader(open(os.path.join(RES, "dev_clusters.csv"), encoding="utf-8"))}
    byt = np.array([int(cl[u]["bytes"]) for u in uids], dtype=np.float64)
    c13 = np.array([per_doc[u]["c13_D2"] if per_doc[u]["c13_D2"] is not None else -1 for u in uids])
    c8 = np.array([per_doc[u]["c8_D2"] if per_doc[u]["c8_D2"] is not None else -1 for u in uids])
    bins = [(-1.01, -0.5, "shorter than 13 words"), (-0.5, 0.01, "c13 < 0.01"), (0.01, 0.1, "0.01-0.1"),
            (0.1, 0.3, "0.1-0.3"), (0.3, 1.01, ">= 0.3")]
    tab = []
    for lo, hi, lab in bins:
        m = (c13 > lo) & (c13 < hi) if lab != ">= 0.3" else (c13 >= 0.3)
        if lab == "c13 < 0.01":
            m = (c13 >= 0) & (c13 < 0.01)
        if not m.any():
            tab.append({"bin": lab, "docs": 0}); continue
        rowd = {"bin": lab, "docs": int(m.sum()), "bytes_share": round(float(byt[m].sum() / byt.sum()), 4),
                "baseline_bpb": round(float(mb[BASE][m].sum() / byt[m].sum()), 4)}
        for c in [CHOSEN] + OTHERS:
            rowd[c + "_vs_base_pct"] = round(float((mb[c][m].sum() - mb[BASE][m].sum()) / mb[BASE][m].sum() * 100), 3)
        tab.append(rowd)
    out["gain_by_D2_overlap_bin_13gram"] = tab
    # bytes-weighted correlation of the per-doc gain with 8-gram containment
    ok = c8 >= 0
    g = (mb[CHOSEN] - mb[BASE]) / byt
    w = byt[ok]
    x = c8[ok]; y = g[ok]
    xm = np.sum(w * x) / w.sum(); ym = np.sum(w * y) / w.sum()
    corr = np.sum(w * (x - xm) * (y - ym)) / np.sqrt(np.sum(w * (x - xm) ** 2) * np.sum(w * (y - ym) ** 2))
    out["weighted_corr_gain_vs_c8_D2"] = round(float(corr), 4)
    clean = (c13 >= 0) & (c13 < 0.01)
    out["clean_docs_chosen_vs_base_pct"] = round(float((mb[CHOSEN][clean].sum() - mb[BASE][clean].sum()) / mb[BASE][clean].sum() * 100), 3)
    out["clean_docs_bytes_share"] = round(float(byt[clean].sum() / byt.sum()), 4)
    # per-variety view (dev_strict is 97 % Hindko by bytes)
    var = {r["uid"]: r["variety"] for r in ds}
    vv = np.array([var[u] for u in uids])
    out["by_variety"] = {v: {"docs": int((vv == v).sum()), "bytes": int(byt[vv == v].sum()),
                             "chosen_vs_base_pct": round(float((mb[CHOSEN][vv == v].sum() - mb[BASE][vv == v].sum()) / mb[BASE][vv == v].sum() * 100), 3)}
                         for v in sorted(set(vv))}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ("gain_by_D2_overlap_bin_13gram",)}, ensure_ascii=False, indent=1)[:6000])
    print(json.dumps(out["gain_by_D2_overlap_bin_13gram"], indent=1))


if __name__ == "__main__":
    main()
