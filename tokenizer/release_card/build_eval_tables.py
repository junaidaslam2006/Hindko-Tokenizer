"""Build the result tables of the release folder F:\\Hindko\\tokenizer\\eval\\ (JSON + CSV, no per-document arrays).

Reads only files that already exist (dev LM results, the one-shot intrinsic competitor table on test, the
tokenizer summaries, Track B). It never reads a test-split LM result: the one-shot test LM run lands in
G:\\My Drive\\hindko_lm_out_final_test\\, which this script does not open.

Also computes one new report-only statistic: the hierarchical cluster bootstrap (the PLAN 6 method, via
analysis/decide.py's Boot/compare) for the 'large' dev arbiter (2 seeds each). It enters no decision:
AMENDMENT_1.md fixed the released tokenizer before this script existed, using the point means.

Run: PYTHONIOENCODING=utf-8 python release_card/build_eval_tables.py   (from F:\\Hindko\\_tokenizer)
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = r"F:\Hindko\_tokenizer"
REL = r"F:\Hindko\tokenizer"
OUT = os.path.join(REL, "eval")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import decide as D  # noqa: E402  (Boot, compare: the PLAN 6 bootstrap used for DECISION.md)

RELEASED = "R2-A4-SPnat-D2-32k"
PREREG = "R2-A10-MinGram-P1r3-D2-48k"
ALT32 = "R2-A10-MinGram-P1r3-D2-32k"
BASELINE = "A1-P1r3-D2-16k"
TOP_SET = [RELEASED, PREREG, ALT32]
LARGE_DIR = os.path.join(ROOT, "lm", "colab_results", "77e1368773fc", "results")
FORBIDDEN = "hindko_lm_out_final_test"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    assert FORBIDDEN not in p
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def wjson(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return p


def wcsv(name, rows, cols):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
    return p


def r6(x):
    return None if x is None else round(float(x), 6)


SOURCES = {}


def src(key, path):
    SOURCES[key] = {"path": path, "sha256": sha(path)}
    return path


# --------------------------------------------------------------------------------------------- 1. confirm ranking
def confirm_ranking(dec):
    rows, full = [], []
    for x in dec["ranking"]:
        vb, vbest, tb = x.get("vs_baseline") or {}, x.get("vs_best") or {}, x["tiebreak"]
        r = {"rank": x.get("rank_primary"), "id": x["id"], "origin": x["origin"], "algorithm": x["algorithm"],
             "pretokenizer": x["pretokenizer"], "data_mix": x["data_mix"], "vocab": x["vocab"],
             "encoder_kind": x["encoder_kind"], "params_total": x["params_total"],
             "params_non_embedding": x["params_non_embedding"], "ctx_tokens": x["ctx_tokens"],
             "n_seeds": x["n_seeds"], "mean_bpb": r6(x["mean_bpb"]),
             "seed_sd_rel_pct": r6(x["seed_sd_rel_pct"]),
             "delta_vs_baseline_pct": r6(vb.get("delta_pct")),
             "delta_vs_baseline_ci95_lo": r6((vb.get("ci95_pct") or [None, None])[0]),
             "delta_vs_baseline_ci95_hi": r6((vb.get("ci95_pct") or [None, None])[1]),
             "p_vs_baseline": r6(vb.get("p")),
             "delta_vs_best_pct": r6(vbest.get("delta_pct")),
             "delta_vs_best_ci95_lo": r6((vbest.get("ci95_pct") or [None, None])[0]),
             "delta_vs_best_ci95_hi": r6((vbest.get("ci95_pct") or [None, None])[1]),
             "holm_p_vs_best": r6(x.get("holm_p_vs_best")), "in_top_set": x["in_top_set"],
             "stage3_mean_bpb": r6(x.get("stage3_mean_bpb")),
             "hf_native_exact": tb["hf_native_exact"], "dev_bytes_per_token": r6(tb["dev_bytes_per_token"]),
             "learned_tokens": tb["learned_tokens"], "train_D1_freq_lt20": tb["train_lt20"],
             "train_D1_freq_lt20_pct": r6(tb["train_lt20_pct"]), "train_D1_freq_eq0": tb["train_eq0"],
             "robustness_seg_change_affected_mean": r6(tb["robustness_seg_change_affected_mean"]),
             "gates_pass": tb["gates_pass"]}
        rows.append(r)
        full.append(dict(r, seed_bpb=[r6(v) for v in x["seed_bpb"]]))
    cols = list(rows[0].keys())
    wcsv("dev_lm_confirm_ranking.csv", rows, cols)
    wjson("dev_lm_confirm_ranking.json", {
        "what": "PLAN 7 ranking on dev_strict, Stage 4 'confirm' arbiter (d=192, L=4, H=4; full permissive train "
                "split, 1 epoch; LR 1e-3; 5 seeds). Copied from analysis/decision.json (ranking). Deltas in %% of "
                "the reference's bpb, 95%% percentile CI of the hierarchical cluster bootstrap (10,000 replicates, "
                "32 clusters, seeds resampled). rank = PLAN 7 rank (null = conditional rank 8, not admitted).",
        "baseline": BASELINE, "best_mean": dec["ranking"][0]["id"],
        "top_set": [x["id"] for x in dec["ranking"] if x["in_top_set"]],
        "preregistered_choice": PREREG, "released_default_amendment_1": RELEASED,
        "rows": full})
    return {x["id"]: x for x in dec["ranking"]}


# --------------------------------------------------------------------------------------------- 2. large arbiter
def large_tables(rank_by_id):
    files = sorted(glob.glob(os.path.join(LARGE_DIR, "large__*.json")))
    sweep = sorted(glob.glob(os.path.join(LARGE_DIR, "large_lrsweep__*.json")))
    recs = {}
    meta = {}
    for p in files:
        r = jl(p)
        assert r["status"] == "ok" and r["stage"] == "large", p
        c, s = r["candidate"], r["seed"]
        recs.setdefault(c, {})[s] = r
        m = r["model"]
        meta[c] = {"d": m["d"], "L": m["L"], "H": m["H"], "n_vocab": m["n_vocab"], "ctx_tokens": m["ctx_tokens"],
                   "params_total": m["params"]["total"], "params_non_embedding": m["params"]["non_embedding"],
                   "steps": r["data"]["steps"], "epochs_seen": r["data"]["epochs_seen"],
                   "budget_bytes": r["data"]["budget_bytes"], "lr": r["lr"],
                   "device": r["runtime"]["gpu_name"], "amp_dtype": r["runtime"]["amp_dtype"],
                   "eval_dtype": r["runtime"]["eval_dtype"], "hk_lm_sha256": r["hk_lm_sha256"],
                   "bundle_manifest_sha256": r["bundle_manifest_sha256"], "preregistered": r["preregistered"]}
        SOURCES["large/" + os.path.basename(p)] = {"path": p, "sha256": sha(p), "created_utc": r["created_utc"]}
    cands = sorted(recs)
    seeds = sorted({s for c in cands for s in recs[c]})
    assert all(sorted(recs[c]) == seeds for c in cands), "every large candidate must have the same seeds"
    uids = recs[cands[0]][seeds[0]]["dev"]["uids"]
    nbytes = np.array(recs[cands[0]][seeds[0]]["dev"]["bytes"], dtype=np.float64)
    for c in cands:
        for s in seeds:
            dv = recs[c][s]["dev"]
            assert dv["uids"] == uids and dv["n_docs"] == 836
            assert np.array_equal(np.array(dv["bytes"], dtype=np.float64), nbytes)
            assert abs(sum(dv["bits"]) / sum(dv["bytes"]) - dv["bpb"]) < 1e-12
    # clusters: the collector's mapping, identical to decide.py's (checked there against the manifest)
    with open(src("dev_clusters_csv", os.path.join(ROOT, "lm", "colab_results", "dev_clusters.csv")),
              encoding="utf-8") as f:
        cc = list(csv.DictReader(f))
    assert [c["uid"] for c in cc] == uids and [int(c["bytes"]) for c in cc] == [int(b) for b in nbytes]
    names = sorted({c["cluster"] for c in cc})
    pos = {n: i for i, n in enumerate(names)}
    src_of = {}
    for c in cc:
        assert src_of.setdefault(c["cluster"], c["source"]) == c["source"]
    doc_cluster = np.array([pos[c["cluster"]] for c in cc])
    cl_source = [src_of[n] for n in names]
    assert len(names) == 32
    bits = np.array([[recs[c][s]["dev"]["bits"] for s in seeds] for c in cands], dtype=np.float64)
    boot = D.Boot(bits, nbytes, doc_cluster, cl_source, n_rep=D.N_REP, seed=D.RNG_SEED)
    reps = boot.reps()
    point = boot.point
    ib = cands.index(BASELINE)
    delta_abs = D.DELTA_REL * float(point[ib])
    rows = []
    order = sorted(range(len(cands)), key=lambda i: point[i])
    for rank, i in enumerate(order, 1):
        c = cands[i]
        vb = D.compare(i, ib, reps, point, delta_abs) if i != ib else None
        ir = cands.index(RELEASED)
        vr = D.compare(i, ir, reps, point, delta_abs) if i != ir else None
        conf = rank_by_id.get(c)
        rows.append({"rank_large": rank, "id": c, "n_seeds": len(seeds),
                     "seed_bpb": [r6(boot.seed_bpb[i, k]) for k in range(len(seeds))],
                     "mean_bpb": r6(point[i]),
                     "delta_vs_baseline_pct": r6(vb["delta_pct"]) if vb else 0.0,
                     "delta_vs_baseline_ci95_pct": [r6(v) for v in vb["ci95_pct"]] if vb else None,
                     "p_vs_baseline": r6(vb["p"]) if vb else None,
                     "delta_vs_released_pct": r6(vr["delta_pct"]) if vr else 0.0,
                     "delta_vs_released_ci95_pct": [r6(v) for v in vr["ci95_pct"]] if vr else None,
                     "p_vs_released": r6(vr["p"]) if vr else None,
                     "confirm_mean_bpb": r6(conf["mean_bpb"]) if conf else None,
                     "confirm_rank": conf.get("rank_primary") if conf else None,
                     "in_confirm_top_set": conf["in_top_set"] if conf else None,
                     **meta[c]})
    # per source, vs baseline (report-only)
    per_source = {}
    for srcname in sorted(set(cl_source)):
        rs = boot.reps(source=srcname)
        ps = boot.point_source(srcname)
        per_source[srcname] = {cands[i]: {"bpb": r6(ps[i]),
                                          **({k: v for k, v in D.compare(i, ib, rs, ps, delta_abs).items()
                                              if k in ("delta_pct", "ci95_pct", "p")} if i != ib else {})}
                               for i in range(len(cands))}
    lr = []
    for p in sweep:
        r = jl(p)
        SOURCES["large_lrsweep/" + os.path.basename(p)] = {"path": p, "sha256": sha(p), "created_utc": r["created_utc"]}
        lr.append({"candidate": r["candidate"], "seed": r["seed"], "lr": r["lr"], "dev_bpb": r6(r["dev"]["bpb"])})
    lr.sort(key=lambda x: x["lr"])
    out = {
        "what": "'Large' LM arbiter on dev_strict (report-only; not pre-registered; added beyond PLAN 8(ii) and "
                "run 2026-09-26 19:44-20:53 UTC, before AMENDMENT_1.md). d=384, L=6, H=6 (10.6M non-embedding "
                "parameters), 2 epochs of the full permissive train split (train_D1), LR 5e-4 from its own 3-point "
                "sweep on the baseline, seeds 1-2, Colab Tesla T4, fp16 training / fp32 evaluation.",
        "amendment_1_rule": "Within the pre-registered, statistically tied top set, the released default is the "
                            "member with the lowest large-arbiter dev bpb (point mean). AMENDMENT_1.md fixed this "
                            "before any test LM number existed.",
        "bootstrap": "computed by release_card/build_eval_tables.py with analysis/decide.py Boot/compare: 10,000 "
                     "replicates, default_rng(12345), 32 dev clusters resampled within source (book 12, newspaper "
                     "15, web 5), the 2 seeds of each candidate resampled with replacement. Report-only; with 2 "
                     "seeds the seed component of the CI is crude. It was computed after AMENDMENT_1 and changes "
                     "nothing.",
        "baseline": BASELINE, "released": RELEASED, "tost_delta_abs_bpb": r6(delta_abs),
        "rows": rows, "per_source_vs_baseline": per_source,
        "top_set_pairs": {
            "%s - %s" % (a, b): {k: (r6(v) if not isinstance(v, (list, bool)) else
                                     ([r6(t) for t in v] if isinstance(v, list) else v))
                                 for k, v in D.compare(cands.index(a), cands.index(b), reps, point,
                                                       delta_abs).items()}
            for a, b in [(RELEASED, PREREG), (RELEASED, ALT32), (ALT32, PREREG)]},
        "lr_sweep_baseline": lr}
    wjson("dev_lm_large.json", out)
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, list)}
        f["seed1_bpb"], f["seed2_bpb"] = r["seed_bpb"][0], r["seed_bpb"][1]
        for key in ("delta_vs_baseline_ci95_pct", "delta_vs_released_ci95_pct"):
            v = r[key] or [None, None]
            f[key + "_lo"], f[key + "_hi"] = v[0], v[1]
        flat.append(f)
    cols = ["rank_large", "id", "mean_bpb", "seed1_bpb", "seed2_bpb", "delta_vs_baseline_pct",
            "delta_vs_baseline_ci95_pct_lo", "delta_vs_baseline_ci95_pct_hi", "p_vs_baseline",
            "delta_vs_released_pct", "delta_vs_released_ci95_pct_lo", "delta_vs_released_ci95_pct_hi",
            "p_vs_released", "confirm_mean_bpb", "confirm_rank", "in_confirm_top_set", "n_vocab", "ctx_tokens",
            "params_total", "params_non_embedding", "steps", "epochs_seen", "lr", "device", "amp_dtype"]
    wcsv("dev_lm_large.csv", flat, cols)
    return out


# --------------------------------------------------------------------------------------------- 3. test intrinsic
def test_intrinsic():
    tc = jl(src("test_competitors_json", os.path.join(ROOT, "eval", "test_competitors.json")))
    rel = tc["in_study_bundle_rows"][RELEASED]
    rel_tok = rel["test_tokens"]
    keep = ["docs", "bytes", "chars", "words", "tokens", "bytes_per_token", "chars_per_token", "fertility",
            "tokens_per_word", "continued_word_rate", "strr", "g1_pass_docs", "g1_fail_docs", "unk_tokens",
            "bytes_per_token_lines", "vocab_used"]
    rows = []
    for t in tc["tokenizers"]:
        o = t["test_strict"]["overall"]
        r = {"name": t["name"], "role": t["role"], "provider": t["provider"], "family": t["family"],
             "algorithm": t["algorithm"], "vocab_size": t["vocab_size"], "tokenizer_sha256": t["tokenizer_sha256"],
             "newline_handling": t["newline_handling"],
             "same_encodings_as": ", ".join(t["same_encodings_as"]) if t.get("same_encodings_as") else None,
             "rank_test_bytes_per_token": t.get("rank_test_bytes_per_token"),
             "rank_among_lossless": t.get("rank_among_lossless")}
        for k in keep:
            v = o.get(k)
            r["test_" + k] = r6(v) if isinstance(v, float) else v
        for s, v in (t["test_strict"].get("by_source") or {}).items():
            r["test_bytes_per_token_" + s] = r6(v.get("bytes_per_token"))
        r["lossless_on_test"] = o.get("g1_fail_docs") == 0
        r["dev_bytes_per_token"] = r6((t.get("dev_strict_reference") or {}).get("bytes_per_token")) \
            if isinstance(t.get("dev_strict_reference"), dict) else None
        if t["role"] != "chosen":
            r["prereg_pick_uses_fewer_tokens_pct"] = r6(t["test_strict"].get("chosen_uses_fewer_tokens_pct"))
            r["released_uses_fewer_tokens_pct"] = r6(100.0 * (1.0 - rel_tok / o["tokens"]))
            r["tokens_vs_released"] = r6(o["tokens"] / rel_tok)
        rows.append(r)
    ext = [r for r in rows if r["role"] != "chosen" and r["role"] != "leaky"]
    n_better_bpt = sum(1 for r in ext if r["test_bytes_per_token"] > rel["test_bytes_per_token"])
    lossless = [r for r in ext if r["lossless_on_test"]]
    best_ll = max(lossless, key=lambda r: r["test_bytes_per_token"])
    released_row = {
        "name": RELEASED, "role": "released default (AMENDMENT_1)",
        "test_tokens": rel_tok, "test_bytes_per_token": r6(rel["test_bytes_per_token"]),
        "test_bytes": 1451026, "test_docs": rel["docs"],
        "g1_harness_fail_docs": rel["harness"]["g1_fail_docs"], "g1_native_fail_docs": rel["builtin"]["g1_fail_docs"],
        "encoder_harness": rel["harness"]["encoder"], "encoder_native": rel["builtin"]["encoder"],
        "dev_bytes_per_token": r6(rel["dev_bytes_per_token_from_dev_bundle"]),
        "source": "eval/test_competitors.json -> in_study_bundle_rows (the final-test bundle encoding, "
                  "colab/build_final_test/VERIFY_FINAL_TEST.json); fertility/STRR of this tokenizer were not "
                  "computed on test",
        "ranked_set": "the %d external model tokenizers (role 'external') + this one; the in-house leaky probe "
                      "hindko-probe-bpe32k (trained on all data incl. test) is not ranked, as in TEST_COMPETITORS.md; "
                      "external + probe = the %d tokenizers of baselines/manifest.json evaluated on test"
                      % (len(ext), tc["n_external_evaluated"]),
        "rank_by_test_bytes_per_token": 1 + n_better_bpt, "n_ranked": len(ext) + 1,
        "rank_among_lossless": 1 + sum(1 for r in lossless if r["test_bytes_per_token"] > rel["test_bytes_per_token"]),
        "n_ranked_lossless": len(lossless) + 1,
        "best_lossless_external": best_ll["name"],
        "best_lossless_external_bytes_per_token": best_ll["test_bytes_per_token"],
        "fewer_tokens_than_best_lossless_external_pct": r6(100.0 * (1.0 - rel_tok / best_ll["test_tokens"]))}
    in_study = {k: {"test_tokens": v["test_tokens"], "test_bytes_per_token": r6(v["test_bytes_per_token"]),
                    "dev_bytes_per_token": r6(v["dev_bytes_per_token_from_dev_bundle"]),
                    "g1_harness_fail_docs": v["harness"]["g1_fail_docs"],
                    "g1_native_fail_docs": v["builtin"]["g1_fail_docs"]}
                for k, v in tc["in_study_bundle_rows"].items()}
    wjson("test_intrinsic_competitors.json", {
        "what": "One-shot intrinsic table on the strict TEST split (491 documents, 1,451,026 UTF-8 bytes), from "
                "eval/test_competitors.json (generated 2026-09-26T21:07:58Z by eval/run_test_competitors.py; the "
                "tokenizer choices were fixed on dev before it ran). No LM number. Per-document arrays and "
                "robustness blocks are left out. 'released_uses_fewer_tokens_pct' = 1 - tokens(released) / "
                "tokens(external) on the same 491 documents, with the released tokenizer's token count from "
                "the final-test bundle encoding (canonical SentencePiece + newline encoder). For tokenizers that "
                "are not lossless (g1_fail_docs > 0) the token counts cover different text.",
        "generated_from": tc["generated_utc"], "n_external": tc["n_external_evaluated"],
        "released": released_row, "in_study_bundle_rows": in_study, "rows": rows})
    cols = ["name", "role", "provider", "family", "algorithm", "vocab_size", "test_bytes_per_token",
            "test_chars_per_token", "test_fertility", "test_strr", "test_tokens", "test_g1_pass_docs",
            "test_g1_fail_docs", "lossless_on_test", "test_unk_tokens", "dev_bytes_per_token",
            "released_uses_fewer_tokens_pct", "tokens_vs_released", "prereg_pick_uses_fewer_tokens_pct",
            "rank_test_bytes_per_token", "rank_among_lossless", "test_bytes_per_token_book",
            "test_bytes_per_token_newspaper", "test_bytes_per_token_web", "same_encodings_as", "newline_handling",
            "tokenizer_sha256"]
    wcsv("test_intrinsic_competitors.csv", rows, cols)
    return released_row, rows


# --------------------------------------------------------------------------------------------- 4. tokenizer summaries
def summaries(rank_by_id):
    ids = [RELEASED, PREREG, ALT32, BASELINE]
    out, gate_rows = {}, []
    for c in ids:
        p = rank_by_id[c]["tiebreak"]["summary_path"]
        s = jl(src("summary/" + c, p))
        m = s["metrics"]

        def slim(d):
            return {k: v for k, v in d.items() if k not in ("encode_seconds",)}
        out[c] = {"tokenizer": s["tokenizer"], "frozen": s["frozen"], "dataset": s["dataset"],
                  "harness_code": s["harness"]["code"],
                  "dev_strict": {"overall": slim(m["overall"]),
                                 "by_source": {k: slim(v) for k, v in m["by_source"].items()},
                                 "by_variety": {k: slim(v) for k, v in m["by_variety"].items()}},
                  "nsl_vs_A1-P1-D1-16k": s.get("nsl"), "gates": s["gates"], "R1_support": s["properties"]["R1"],
                  "R2_partial_utf8": s["properties"]["R2"], "morphology_silver": s.get("morphology"),
                  "stage1_config": (s.get("extra") or {}).get("stage1_config")}
        g, r1 = s["gates"], s["properties"]["R1"]
        al = r1["all_learned"]
        gate_rows.append({
            "id": c, "role": {RELEASED: "released default", PREREG: "pre-registered pick", ALT32: "alternative",
                              BASELINE: "baseline (standard recipe)"}[c],
            "G1_dev_strict": "%d/%d" % (g["G1"]["docs"] - g["G1"]["fail_docs"], g["G1"]["docs"]),
            "G2_pass": g["G2"]["pass"], "G2_tested": g["G2"]["tested"], "G2_failures": g["G2"]["failures"],
            "G3_pass": g["G3"]["pass"], "G3_corpus_marker_occurrences": g["G3"]["corpus_marker_occurrences"],
            "G4_pass": g["G4"]["pass"], "G5_pass": g["G5"]["pass"], "G5_sub_character_tokens":
                g["G5"].get("sub_character_tokens"),
            "learned_tokens": al["learned_tokens"], "train_D1_freq_eq0": al["train_freq_eq0"],
            "train_D1_freq_lt20": al["train_freq_lt20"], "train_D1_freq_lt20_pct": r6(al["pct_lt20"]),
            "train_D1_freq_lt100": al["train_freq_lt100"], "train_D1_freq_lt100_pct": r6(al["pct_lt100"]),
            "train_D1_median_freq": al["median_train_freq"], "train_D1_tokens": r1["train"]["tokens"],
            "partial_utf8_tokens": s["properties"]["R2"]["partial_utf8_tokens"],
            "dev_bytes_per_token": r6(m["overall"]["bytes_per_token"]), "dev_fertility": r6(m["overall"]["fertility"]),
            "dev_strr": r6(m["overall"]["strr"]), "tokenizer_sha256": s["tokenizer"]["sha256"]})
    # the pre-remedy G2 failures of the released model (PLAN 4.3 remedy)
    pre = jl(src("summary/pre_remedy", os.path.join(ROOT, "candidates", "round2", "standard", "results",
                                                     "dev_strict", RELEASED + "__pre_remedy", "summary.json")))
    meta = jl(src("meta/" + RELEASED, os.path.join(ROOT, "candidates", "round2", "standard", "tok", RELEASED,
                                                    "meta.json")))
    out[RELEASED]["g2_remedy"] = {"pre_remedy_G2": pre["gates"]["G2"], "after": meta["g2_remedy"],
                                  "note": "PLAN 4.3 remedy: the 5 pieces that lose the Viterbi search on their own "
                                          "string were turned into CONTROL placeholders at their ids (never "
                                          "produced by encoding), so every other id is unchanged; effective "
                                          "vocabulary 32,763."}
    out[RELEASED]["trainer"] = {"trainer": meta["trainer"], "trainer_params": meta["trainer_params"],
                                "train_data": meta["train_data"], "versions": meta["versions"],
                                "special_block": meta["special_block"], "hf_export_of_candidate": meta["hf_export"],
                                "train_seconds": meta["train_seconds"]}
    wjson("tokenizer_summaries_dev.json", {
        "what": "Intrinsic metrics on dev_strict (836 documents), hard gates G1-G5 and the support profile R1 "
                "(on train_D1, the LM training stream) of the three top-set members and the baseline, from each "
                "candidate's summary.json (eval/harness.py). Per-document arrays left out.",
        "tokenizers": out})
    cols = list(gate_rows[0].keys())
    wcsv("health_gates_and_support.csv", gate_rows, cols)
    wjson("health_gates_and_support.json", {"what": "PLAN 4.3 hard gates (dev) and R1/R2 properties", "rows": gate_rows})
    return out, gate_rows


# --------------------------------------------------------------------------------------------- 5. Track B
def trackb():
    rows, curves = [], []
    summ = jl(src("trackb_summary", os.path.join(ROOT, "trackb", "results", "summary.json")))
    for p in sorted(glob.glob(os.path.join(ROOT, "trackb", "deliver", "*", "EXTENSION.json"))):
        e = jl(src("trackb/" + os.path.basename(os.path.dirname(p)), p))
        base0 = summ[e["base"]]["0"]["dev"]
        d, r1 = e["dev_strict"], e["R1_new"]
        ni = e["non_interference"]
        rows.append({"folder": e["name"], "base": e["base"], "base_repo": e["base_repo"],
                     "base_models": e["base_models"], "base_license": e["base_license"], "k_new": e["k_new_tokens"],
                     "first_new_id": e["first_new_id"], "len_tokenizer": e["len_tokenizer"],
                     "dev_bytes_per_token_base": r6(base0["bytes_per_token"]),
                     "dev_bytes_per_token_ext": r6(d["bytes_per_token"]),
                     "dev_token_reduction_pct": r6(100 * (1 - d["nsl_vs_base"])),
                     "dev_fertility_base": r6(base0["fertility"]), "dev_fertility_ext": r6(d["fertility"]),
                     "dev_strr_base": r6(base0["strr"]), "dev_strr_ext": r6(d["strr"]),
                     "train_D1_tokens_base": e["train_D1_tokens"]["base"],
                     "train_D1_tokens_ext": e["train_D1_tokens"]["extended"],
                     "new_lt100": r1["train_freq_lt100"], "new_lt20": r1["train_freq_lt20"],
                     "new_eq0": r1["train_freq_eq0"], "partial_utf8_new": r1.get("partial_utf8_new"),
                     "G1": e["gates"]["G1"], "G2_new": e["gates"]["G2_new"], "equivalence": e["gates"]["equivalence"],
                     "english_changed": ni["english_wiki"]["changed"], "code_changed": ni["python_code"]["changed"],
                     "other_scripts_changed": ni["other_scripts"]["changed"],
                     "urdu_news_dev_token_reduction_pct": r6(100 * ni["urdu_news_dev"]["token_reduction"]),
                     "urdu_book_dev_token_reduction_pct": r6(100 * ni["urdu_book_dev"]["token_reduction"]),
                     "delivery_check_pass": e["verification"]["pass"],
                     "tokenizer_json_sha256": e["files_sha256"]["tokenizer.json"]})
    for base, ks in summ.items():
        for k, v in sorted(ks.items(), key=lambda kv: int(kv[0])):
            dv = v["dev"]
            r1 = v.get("R1_new") or {}
            curves.append({"base": base, "k": int(k), "vocab": v["vocab"], "dev_tokens": dv["tokens"],
                           "dev_bytes_per_token": r6(dv["bytes_per_token"]), "dev_fertility": r6(dv["fertility"]),
                           "dev_strr": r6(dv["strr"]), "nsl_vs_base": r6(dv["nsl_vs_base"]),
                           "train_D1_tokens": v["train_tokens"], "new_lt100": r1.get("train_freq_lt100"),
                           "new_lt20": r1.get("train_freq_lt20"), "new_eq0": r1.get("train_freq_eq0"),
                           "eligible": (v.get("eligibility") or {}).get("eligible"),
                           "tokenizer_sha256": v["tokenizer_sha256"]})
    wjson("trackb_extensions.json", {
        "what": "Track B (PLAN 9): continued-BPE Hindko extensions of five open LLM tokenizers at the knee k = 2,048 "
                "(KNEE_RULE.md, pre-registered). dev_strict numbers; k was chosen on dev. No model was trained.",
        "delivered": rows, "k_grid_curves": curves})
    wcsv("trackb_extensions.csv", rows, list(rows[0].keys()))
    wcsv("trackb_k_grid.csv", curves, list(curves[0].keys()))
    return rows


# --------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    dec = jl(src("decision_json", os.path.join(ROOT, "analysis", "decision.json")))
    am = jl(src("amendment_sha_json", os.path.join(ROOT, "analysis", "AMENDMENT_1.sha.json")))
    am_path = src("amendment_md", os.path.join(ROOT, "analysis", "AMENDMENT_1.md"))
    assert sha(am_path) == am["sha256"], "AMENDMENT_1.md changed after its sha was recorded"
    rank_by_id = confirm_ranking(dec)
    large = large_tables(rank_by_id)
    released_row, _ = test_intrinsic()
    summ, gates = summaries(rank_by_id)
    tb = trackb()
    # verbatim copies of the decision documents (reports/)
    copies = {"DECISION.md": os.path.join(ROOT, "analysis", "DECISION.md"),
              "decision.json": os.path.join(ROOT, "analysis", "decision.json"),
              "AMENDMENT_1.md": am_path,
              "AMENDMENT_1.sha.json": os.path.join(ROOT, "analysis", "AMENDMENT_1.sha.json"),
              "TEST_COMPETITORS.md": os.path.join(ROOT, "eval", "TEST_COMPETITORS.md"),
              "PLAN.md": os.path.join(ROOT, "research", "PLAN.md"),
              "FROZEN.json": os.path.join(ROOT, "FROZEN.json")}
    for name, p in copies.items():
        shutil.copyfile(p, os.path.join(OUT, "reports", name))
        src("copy/" + name, p)
    man = {"what": "Files of F:\\Hindko\\tokenizer\\eval\\ with sha256, and the sha256 of every source file read. "
                   "No test-split LM result was read (G:\\My Drive\\hindko_lm_out_final_test\\ was not opened).",
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script": {"path": os.path.abspath(__file__), "sha256": sha(os.path.abspath(__file__))},
           "files": {}, "sources": SOURCES}
    for dp, _, fns in os.walk(OUT):
        for fn in sorted(fns):
            if fn == "MANIFEST.json":
                continue
            p = os.path.join(dp, fn)
            man["files"][os.path.relpath(p, OUT).replace("\\", "/")] = sha(p)
    wjson("MANIFEST.json", man)
    # console digest for the card
    print("LARGE:")
    for r in large["rows"]:
        print("  %d %-30s mean %.5f seeds %s dBase %+.3f%% CI %s p %s | dRel %+.3f%% CI %s p %s | confirm %.5f"
              % (r["rank_large"], r["id"], r["mean_bpb"], r["seed_bpb"], r["delta_vs_baseline_pct"],
                 r["delta_vs_baseline_ci95_pct"], r["p_vs_baseline"], r["delta_vs_released_pct"],
                 r["delta_vs_released_ci95_pct"], r["p_vs_released"], r["confirm_mean_bpb"]))
    print("  pairs:", json.dumps(large["top_set_pairs"], ensure_ascii=False))
    print("  per source:", json.dumps({s: {c: v.get("delta_pct") and round(v["delta_pct"], 3) for c, v in d.items()}
                                       for s, d in large["per_source_vs_baseline"].items()}, ensure_ascii=False))
    print("  lr:", large["lr_sweep_baseline"])
    print("RELEASED TEST INTRINSIC:", json.dumps(released_row, ensure_ascii=False))
    print("GATES:", json.dumps(gates, ensure_ascii=False))
    print("TRACKB:", len(tb))
    print("files:", len(man["files"]))


if __name__ == "__main__":
    main()
