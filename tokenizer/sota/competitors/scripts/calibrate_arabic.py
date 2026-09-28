# -*- coding: utf-8 -*-
"""Calibration for the bandwidth triage: how large must a tokenizer's Perso-Arabic vocabulary share be before it gets
anywhere near the released tokenizer on Hindko?  For every MEASURED tokenizer (the 66 baselines and every screened
refresh tokenizer with bytes/token >= 2.5) this computes the share of vocabulary entries containing an Arabic-script
character, both over the whole vocabulary and with the triage's own sampler (strings in the middle 16 KB and last
32 KB of tokenizer.json), next to its measured test bytes/token.   Output: ../_calibration.json"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import range_triage as T  # noqa: E402
import triage2 as T2  # noqa: E402

ROOT = T.ROOT


def vocab_share(path):
    j = json.load(open(path, encoding="utf-8"))
    v = j["model"].get("vocab")
    toks = list(v.keys()) if isinstance(v, dict) else [x[0] for x in (v or [])]
    bl = '"ByteLevel"' in json.dumps(j.get("pre_tokenizer")) or '"ByteLevel"' in json.dumps(j.get("decoder"))
    n = 0
    for t in toks:
        s = T2.bl_decode(t) if bl else t
        n += any(T2.is_ar(c) for c in s)
    return n / max(1, len(toks)), len(toks)


def sampled_share(path):
    data = open(path, "rb").read()
    off = T.model_offset(data[:4 * 1024 * 1024])
    if off is None:
        return None
    L = len(data) - off
    mid = data[off + L // 2: off + L // 2 + 16 * 1024]
    tail = data[-32 * 1024:]
    samp = T2.strings(mid)[1:-1] + T2.strings(tail)[1:-1]
    dec = [T2.bl_decode(x) if any(c in x for c in "ĠĊÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß") else x for x in samp]
    return sum(1 for x in dec if any(T2.is_ar(c) for c in x)) / max(1, len(dec))


def main():
    rows = []
    base = json.load(open(os.path.join(T.TOK, "baselines", "manifest.json"), encoding="utf-8"))["baselines"]
    bres = os.path.join(T.TOK, "eval", "results", "test_strict")
    for e in base:
        lp = e.get("load_path") or ""
        if not lp.endswith(".json"):
            continue
        p = lp if os.path.isabs(lp) else os.path.join(T.TOK, "baselines", lp)
        s = json.load(open(os.path.join(bres, e["name"], "summary.json"), encoding="utf-8"))["metrics"]["overall"]
        vs, n = vocab_share(p)
        rows.append({"name": e["name"], "origin": "baseline", "bytes_per_token": s["bytes_per_token"], "g1": s["g1_pass_docs"],
                     "vocab": n, "vocab_arabic_share": vs, "sampled_arabic_share": sampled_share(p)})
    for p in glob.glob(os.path.join(ROOT, "screen", "*.json")):
        s = json.load(open(p, encoding="utf-8"))
        sc = s.get("screen")
        if not sc or sc["bytes_per_token"] < 2.5 or sc["bytes_per_token"] > 100 or s["spec"]["loader"] != "hf_json":
            continue
        try:
            vs, n = vocab_share(s["spec"]["load_path"])
            rows.append({"name": s["name"], "origin": "refresh", "bytes_per_token": sc["bytes_per_token"], "g1": sc["g1_pass_docs"],
                         "vocab": n, "vocab_arabic_share": vs, "sampled_arabic_share": sampled_share(s["spec"]["load_path"])})
        except Exception as ex:  # noqa: BLE001
            print("skip", s["name"], ex)
    rows.sort(key=lambda r: -r["bytes_per_token"])
    json.dump(rows, open(os.path.join(ROOT, "_calibration.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rows[:40]:
        print("%.3f G1 %3d vocab %7d vocab-AR %.3f sampled-AR %s %s" % (r["bytes_per_token"], r["g1"], r["vocab"], r["vocab_arabic_share"],
                                                                     "-" if r["sampled_arabic_share"] is None else "%.3f" % r["sampled_arabic_share"],
                                                                     r["name"]))
    low = [r for r in rows if (r["sampled_arabic_share"] or 0) < 0.03 and r["g1"] >= 400]
    print("max bytes/token among lossless-ish tokenizers with sampled Arabic share < 3%:",
          max(low, key=lambda r: r["bytes_per_token"]) if low else None)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
