# -*- coding: utf-8 -*-
"""Collect the SuperBPE results (harness summaries, verify.json, g2_multiword.json, build_info.json) into
results_superbpe.json and print the Markdown tables used in SUPERBPE.md. dev_strict only; no test split.

    python report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

STD = os.path.join(C.TOK, "candidates", "standard", "results", "dev_strict")
RES = os.path.join(C.HERE, "results", "dev_strict")
ROWS = [
    ("A1-P1-D1-16k", os.path.join(STD, "A1-P1-D1-16k"), None),
    ("A6-SBPE-P1-D1-16k-t090", os.path.join(RES, "A6-SBPE-P1-D1-16k-t090"), "sbpe_16k_t090"),
    ("A6-SBPE-P1-D1-16k-t080", os.path.join(RES, "A6-SBPE-P1-D1-16k-t080"), "sbpe_16k_t080"),
    ("A1-P1-D1-32k", os.path.join(RES, "A1-P1-D1-32k"), None),
    ("A6-SBPE-P1-D1-32k-t080", os.path.join(RES, "A6-SBPE-P1-D1-32k-t080"), "sbpe_32k_t080"),
]
TRAIN_BYTES = 45007515          # data_manifest.json train_D1 bytes_utf8


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def row(name, rdir, bdir):
    s = load(os.path.join(rdir, "summary.json"))
    if s is None:
        return None
    o = s["metrics"]["overall"]
    r1 = s["properties"]["R1"]
    out = {"name": name, "tokenizer": s["tokenizer"]["path"], "tokenizer_sha256": s["tokenizer"]["sha256"],
           "harness_code": s["harness"]["code"], "vocab_size": s["tokenizer"]["vocab_size"],
           "gates": {g: v.get("pass") for g, v in s["gates"].items()},
           "g2_harness_tested": s["gates"]["G2"].get("tested"), "g2_harness_failures": s["gates"]["G2"].get("failures"),
           "g2_superword_untested_by_harness": s["gates"]["G2"].get("superword_tokens_untested"),
           "overall": {k: o.get(k) for k in ("docs", "bytes", "tokens", "bytes_per_token", "chars_per_token", "fertility",
                                             "tokens_per_word", "strr", "continued_word_rate", "renyi_eff_a2.5",
                                             "renyi_eff_a2.0", "vocab_used", "vocab_utilisation",
                                             "bytes_per_token_lines")},
           "by_source_bytes_per_token": {k: v["bytes_per_token"] for k, v in s["metrics"]["by_source"].items()},
           "by_variety_bytes_per_token": {k: v["bytes_per_token"] for k, v in s["metrics"]["by_variety"].items()},
           "nsl": s.get("nsl"),
           "robustness": {k: {"rel_token_change": v["rel_token_change"], "seg_change_rate": v["seg_change_rate"],
                              "seg_change_rate_affected": v["seg_change_rate_affected"]}
                          for k, v in o["robustness"].items()},
           "R1": r1.get("all_learned"), "R1_multichar": r1.get("multichar_learned"),
           "R1_train_tokens": r1.get("train", {}).get("tokens"),
           "train_bytes_per_token": TRAIN_BYTES / r1["train"]["tokens"] if r1.get("train") else None,
           "R2": {k: v for k, v in s["properties"]["R2"].items() if k != "list"},
           "morphology": {k: {m: s.get("morphology", {}).get(k, {}).get(m) for m in ("boundary_f1", "morphscore", "stem_intact")}
                          for k in ("silver_high", "silver_low")},
           "speed_docs_per_s": s["speed"]["docs_per_s"], "speed_MB_per_s": s["speed"]["MB_per_s"]}
    if bdir:
        d = os.path.join(C.HERE, bdir)
        b = load(os.path.join(d, "build_info.json"))
        v = load(os.path.join(d, "verify.json"))
        g = load(os.path.join(d, "g2_multiword.json"))
        out["build"] = {"t": b["t"], "T": b["T"], "route": b["route"], "stage2_merges": b["assembly"]["stage2_merges"],
                        "stage2_words_hist": b["assembly"]["stage2_token_words_hist"],
                        "uncapped_words_hist": b["hf_rust_uncapped_assembly"]["stage2_token_words_hist"],
                        "uncapped_over_cap": b["hf_rust_uncapped_assembly"]["stage2_tokens_over_cap"],
                        "duplicates": b["assembly"]["stage2_tokens_duplicating_existing"],
                        "py_uncapped_identical_to_rust": (b.get("pure_python_uncapped") or {}).get("identical_to_hf_rust_merges"),
                        "seconds": {"stage1_train": (b["stage1"]["info"] or {}).get("train_seconds"),
                                    "chunks": b["chunks"]["seconds"], "rust_pua_trainer": b["hf_rust_pua_trainer"]["seconds"],
                                    "py_uncapped": (b.get("pure_python_uncapped") or {}).get("seconds"),
                                    "py_capped": (b.get("pure_python_capped") or {}).get("seconds"),
                                    "total_build": b["total_seconds"]},
                        "chunks": {k: b["chunks"].get(k) for k in ("chunks", "chunks_ge2", "unique_chunks_ge2", "tokens",
                                                                    "stage1_S2_vs_P1_mismatch_docs")}}
        if v:
            out["verify"] = {"hf_native_exact_on_100pct_dev": v["hf_native_exact_on_100pct_dev"], "all_pass": v["all_pass"],
                             "checks": {k: {kk: vv for kk, vv in c.items() if kk not in ("examples", "view")}
                                        for k, c in v["checks"].items()},
                             "superwords": {k: vv for k, vv in v["superwords_dev_strict"].items() if k != "top_multiword"},
                             "top_superwords": v["superwords_dev_strict"]["top_multiword"][:12]}
        if g:
            out["g2_multiword"] = {k: g[k] for k in ("multiword_tokens", "pass", "unreachable", "isolated_failures",
                                                     "chunk_aligned_failures", "chunk_raw_failures_literal_PLAN_clause",
                                                     "chunk_raw_failures_that_pass_isolated",
                                                     "chunk_raw_failures_train_freq_min_median_max",
                                                     "chunk_aligned_failures_train_freq_min_median_max",
                                                     "train_freq_eq0", "train_freq_lt20", "train_freq_lt100")}
    return out


def f(x, d=3):
    return "—" if x is None else ("%.*f" % (d, x))


def main():
    rows = [r for r in (row(*x) for x in ROWS) if r]
    ref = next(r for r in rows if r["name"] == "A1-P1-D1-16k")
    res = {"what": "SuperBPE (A6) intrinsic results on dev_strict vs A1-P1-D1-16k; PLAN Stage 2",
           "split_manifest_sha256": C.MANIFEST_SHA256, "rows": rows}
    json.dump(res, open(os.path.join(C.HERE, "results_superbpe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("| tokenizer | vocab | dev tokens | bytes/token | Δ vs A1-16k | NSL | chars/token | fertility | STRR | Rényi α2.5 | vocab used | train bytes/token |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o = r["overall"]
        print("| %s | %d | %d | %s | %+.2f%% | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["vocab_size"], o["tokens"], f(o["bytes_per_token"]),
            100 * (o["bytes_per_token"] / ref["overall"]["bytes_per_token"] - 1),
            f((r["nsl"] or {}).get("overall"), 4), f(o["chars_per_token"]), f(o["fertility"]), f(o["strr"]),
            f(o["renyi_eff_a2.5"], 4), f(o["vocab_utilisation"], 3), f(r["train_bytes_per_token"])))
    print()
    print("| tokenizer | book | newspaper | web | NSL book | NSL newspaper | NSL web | bytes/token, line by line |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = r["by_source_bytes_per_token"]
        n = r["nsl"] or {}
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (r["name"], f(b.get("book")), f(b.get("newspaper")), f(b.get("web")),
                                                  f(n.get("source:book"), 4), f(n.get("source:newspaper"), 4),
                                                  f(n.get("source:web"), 4), f(r["overall"]["bytes_per_token_lines"])))
    print()
    print("| tokenizer | G1 | G2 (harness) | G2 multi-word | G3 | G4 | G5 | R1 learned | freq=0 | <20 | <100 | median | R2 partial-UTF-8 (freq 0) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = r["gates"]
        gm = r.get("g2_multiword")
        R1 = r["R1"]
        print("| %s | %s | %s (%s tested, %s fail) | %s | %s | %s | %s | %d | %d | %d | %d | %s | %d (%s) |" % (
            r["name"], g["G1"], g["G2"], r["g2_harness_tested"], r["g2_harness_failures"],
            ("%s (%d tokens, %d unreachable)" % (gm["pass"], gm["multiword_tokens"], gm["unreachable"])) if gm else "n/a",
            g["G3"], g["G4"], "n/a" if g["G5"] is None else g["G5"], R1["learned_tokens"], R1["train_freq_eq0"],
            R1["train_freq_lt20"], R1["train_freq_lt100"], f(R1["median_train_freq"], 0), r["R2"]["partial_utf8_tokens"],
            r["R2"]["with_train_freq_0"]))
    print()
    print("| tokenizer | harakat Δtok | harakat seg-chg (affected) | digits Δtok | punct_space Δtok | punct_space seg-chg (affected) | zwnj Δtok | zwnj seg-chg (affected) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        q = r["robustness"]
        print("| %s | %+.4f | %s | %+.6f | %+.4f | %s | %+.4f | %s |" % (
            r["name"], q["harakat"]["rel_token_change"], f(q["harakat"]["seg_change_rate_affected"]),
            q["digits"]["rel_token_change"], q["punct_space"]["rel_token_change"],
            f(q["punct_space"]["seg_change_rate_affected"]), q["zwnj"]["rel_token_change"],
            f(q["zwnj"]["seg_change_rate_affected"])))
    print()
    print("| tokenizer | morph silver-high F1 / MorphScore | silver-low F1 / MorphScore | encode docs/s |")
    print("|---|---|---|---|")
    for r in rows:
        m = r["morphology"]
        print("| %s | %s / %s | %s / %s | %.0f |" % (r["name"], f(m["silver_high"]["boundary_f1"]), f(m["silver_high"]["morphscore"]),
                                                f(m["silver_low"]["boundary_f1"]), f(m["silver_low"]["morphscore"]),
                                                r["speed_docs_per_s"]))
    print()
    for r in rows:
        if "verify" in r:
            sw = r["verify"]["superwords"]
            print(r["name"], "superword share of dev tokens %.4f, of bytes %.4f, by words %s, distinct used %d" % (
                sw["multiword_token_share"], sw["bytes_in_multiword_tokens_share"], sw["multiword_tokens_by_words"],
                sw["distinct_multiword_ids_used"]))


if __name__ == "__main__":
    main()
