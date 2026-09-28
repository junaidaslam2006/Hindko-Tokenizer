# -*- coding: utf-8 -*-
"""Write the release folder's test tables and manifests (after copy_release.py and the checks have run).

Inputs (read only): analysis/test_results.json, release_card/finalize/test_intrinsic/*/summary.json,
release_card/finalize/longline_probe*_dev_strict.json, release_card/finalize/test_tokenizer_*.json,
release_card/finalize/release_files.json, eval/test_intrinsic_competitors.json.
Outputs (F:\\Hindko\\tokenizer\\):
  eval/test_lm_confirm.csv/.json   one-shot test LM, confirm scale, 6 tokenizers (5 seeds each)
  eval/test_lm_large.csv/.json     report-only large arbiter on test (MinGram-32k: 1 seed)
  eval/test_intrinsic_released.json  test bytes/token, fertility, STRR of the released files (harness, report-only)
  eval/float32_long_lines.json     sp.model vs tokenizer.json on long single lines (probes)
  eval/release_checks.json         examples/test_tokenizer.py results on the final folder
  eval/MANIFEST.json               + section 'release_finalization' (earlier sections unchanged)
  RELEASE_MANIFEST.json            sha256 of every file of the release folder (written last)

    set PYTHONIOENCODING=utf-8
    python release_card\\finalize\\make_release_tables.py
"""
import csv
import datetime
import hashlib
import json
import os

TOK = r"F:\Hindko\_tokenizer"
REL = r"F:\Hindko\tokenizer"
FIN = os.path.join(TOK, r"release_card\finalize")
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
VERSION = "1.0.0"

CONTRAST_VS_BASE = {"R2-A10-MinGram-P1r3-D2-48k": "a_prereg", "R2-A4-SPnat-D2-32k": "b_released",
                    "R2-A10-MinGram-P1r3-D2-32k": "c_mingram32", "A1-P1r3-D2-32k": "d_r1_vs_base",
                    "R2-A1-P1r3-D2-48k": "o_bpe48_vs_base"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r6(x):
    """full precision (no re-rounding: rounding is done once, where a number is displayed)"""
    return None if x is None else float(x)


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def check_no_build_artefacts():
    """The card calls RELEASE_MANIFEST.json the sha256 of every other file of the folder, so no build artefacts."""
    stray = [os.path.relpath(os.path.join(root, f), REL) for root, _, fs in os.walk(REL) for f in fs
             if "__pycache__" in root or f.endswith((".pyc", ".pyo"))]
    if stray:
        raise SystemExit("remove build artefacts from the release folder first (run the examples with python -B): "
                         + ", ".join(stray))


def main():
    check_no_build_artefacts()
    tr_path = os.path.join(TOK, r"analysis\test_results.json")
    tr = json.load(open(tr_path, encoding="utf-8"))
    src = {"analysis/test_results.json": sha(tr_path)}

    # ---- confirm-scale test table ------------------------------------------------------------------------------
    rows = []
    for rank, cid in enumerate(tr["test_order"], 1):
        c = tr["candidates"][cid]
        key = CONTRAST_VS_BASE.get(cid)
        con = tr["contrasts"][key] if key else None
        t = con["test"] if con else None
        dev = (con["dev"]["of_record"] or con["dev"]["recomputed"]) if con else None
        dvt = con["dev_vs_test"] if con else None
        best = c["test_vs_best_of6"]
        rows.append({
            "test_rank_of6": rank, "id": cid, "role": c["role"], "bundle": c["bundle"], "n_vocab": c["n_vocab"],
            "params_total": c["params_total"], "params_non_embedding": c["params_non_embedding"],
            "test_bytes_per_token": c["test_bytes_per_token"], "n_seeds": len(c["test_seed_bpb"]),
            "test_mean_bpb": r6(c["test_mean_bpb"]), "test_seed_bpb": [r6(x) for x in c["test_seed_bpb"]],
            "test_seed_sd_rel_pct": r6(c["test_seed_sd_rel_pct"]),
            "test_delta_vs_baseline_pct": r6(t["delta_pct"]) if t else None,
            "test_delta_vs_baseline_ci95_pct": [r6(x) for x in t["ci95_pct"]] if t else None,
            "test_p_vs_baseline": t["p"] if t else None,
            "dev_mean_bpb": r6(c["dev_mean_bpb"]), "dev_rank_of17": c["dev_rank_of17_decision"],
            "dev_delta_vs_baseline_pct": r6(dev["delta_pct"]) if dev else None,
            "dev_delta_vs_baseline_ci95_pct": [r6(x) for x in dev["ci95_pct"]] if dev else None,
            "change_dev_to_test_pp": r6(dvt["change_pp"]) if dvt else None,
            "change_dev_to_test_ci95_pp": [r6(x) for x in dvt["change_ci95_pp"]] if dvt else None,
            "test_delta_vs_best_of6_pct": r6(best["delta_pct"]) if best else None,
            "test_delta_vs_best_of6_ci95_pct": [r6(x) for x in best["ci95_pct"]] if best else None,
            "test_holm_p_vs_best_of6": c.get("test_holm_p_vs_best_of6"),
        })
    pairs = {}
    for k in ("o_sp32_vs_mg48", "o_sp32_vs_mg32", "o_mg48_vs_mg32", "e_mg48_vs_bpe48", "d_mg48_vs_r1",
              "d_sp32_vs_r1", "d_mg32_vs_r1"):
        con = tr["contrasts"][k]
        pairs["%s - %s" % (con["a"], con["b"])] = {
            "what": con["what"], "test_delta_pct": r6(con["test"]["delta_pct"]),
            "test_ci95_pct": [r6(x) for x in con["test"]["ci95_pct"]], "test_p": con["test"]["p"],
            "tost_equivalent_0.3pct": con["test"]["tost_equivalent"]}
    conf = {"what": "One-shot TEST LM results at the confirm scale (d=192, L=4, H=4, full permissive train split, "
                    "1 epoch, LR 1e-3, seeds 1-5), strict test split (491 documents, 1,451,026 bytes, 27 clusters). "
                    "Copied from analysis/test_results.json (DECISION.md hierarchical cluster bootstrap, 10,000 "
                    "replicates, default_rng(12345)); p = 0 means p < 1e-4. Only "
                    "R2-A10-MinGram-P1r3-D2-48k vs A1-P1r3-D2-16k is pre-registered; every other row is report-only.",
            "source": src, "baseline": "A1-P1r3-D2-16k", "pre_registered_pick": "R2-A10-MinGram-P1r3-D2-48k",
            "released_default": "R2-A4-SPnat-D2-32k", "rows": rows, "pairs": pairs,
            "pre_registered_claim": tr["pre_registered_claim"],
            "top_set_on_test": tr["top_set_on_test"], "rank_correlation_dev_vs_test_6":
                tr["rank_correlation_dev_vs_test_6"], "timing": tr["timing"]}
    json.dump(conf, open(os.path.join(REL, r"eval\test_lm_confirm.json"), "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    write_csv(os.path.join(REL, r"eval\test_lm_confirm.csv"), rows)

    # ---- large arbiter on test (report-only) ------------------------------------------------------------------
    L = tr["large_test"]
    lrows = []
    for cid in L["order_by_mean"]:
        c = L["candidates"][cid]
        lrows.append({"id": cid, "n_seeds": c["n"], "seeds": c["seeds"], "test_seed_bpb": [r6(x) for x in c["seed_bpb"]],
                      "test_mean_bpb": r6(c["mean_bpb"]),
                      "test_delta_vs_baseline_pct_means": r6(c["vs_baseline_pct_all_seeds"]),
                      "dev_large_seed_bpb": [r6(x) for x in c["dev_large_seed_bpb"]],
                      "dev_large_mean_bpb": r6(c["dev_large_mean_bpb"]), "params_total": c["params_total"],
                      "note": "seed 2 missing: Colab disconnected at step 6588/7325 and the free GPU quota was then "
                              "exhausted" if c["n"] == 1 else ""})
    large = {"what": L["what"], "source": src, "rows": lrows,
             "boot_2seeds_3cands": L["boot_2seeds_3cands"], "boot_seed1_4cands": L["boot_seed1_4cands"],
             "parametric_all_seeds": L["parametric_all_seeds"], "dev_vs_test": L["dev_vs_test"],
             "dev_large_of_record": L["dev_large_of_record"], "confirm_vs_large_on_test": L["confirm_vs_large_on_test"]}
    json.dump(large, open(os.path.join(REL, r"eval\test_lm_large.json"), "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    write_csv(os.path.join(REL, r"eval\test_lm_large.csv"), lrows)

    # ---- test intrinsic metrics of the released files (report-only) --------------------------------------------
    intr = {"what": "Intrinsic metrics of the released tokenizer on the strict test split, computed for the release "
                    "card on %s with eval/harness.py (same metric code as TEST_COMPETITORS.md), after every test LM "
                    "result existed. Report-only; no decision depends on it. Both release encoders give identical "
                    "per-document rows." % NOW[:10], "runs": {}}
    for n in ("tokenizer_json", "sp_model"):
        p = os.path.join(FIN, "test_intrinsic", n, "summary.json")
        s = json.load(open(p, encoding="utf-8"))
        o = s["metrics"]["overall"]
        keep = ("docs", "bytes", "chars", "words", "tokens", "bytes_per_token", "chars_per_token", "fertility",
                "tokens_per_word", "strr", "continued_word_rate", "g1_pass_docs", "g1_fail_docs", "unk_tokens",
                "bytes_per_token_lines")
        intr["runs"][n] = {"tokenizer": s["tokenizer"], "dataset": s["dataset"], "harness": s["harness"],
                           "overall": {k: o[k] for k in keep},
                           "by_source": {k: {m: v[m] for m in ("docs", "bytes", "tokens", "bytes_per_token",
                                                               "fertility", "strr", "g1_pass_docs")}
                                         for k, v in s["metrics"]["by_source"].items()},
                           "summary_json_sha256": sha(p)}
    json.dump(intr, open(os.path.join(REL, r"eval\test_intrinsic_released.json"), "w", encoding="utf-8",
                         newline="\n"), ensure_ascii=False, indent=1)

    # ---- float32 long-line probes -------------------------------------------------------------------------------
    f32 = {"what": "sp.model (SentencePiece, float32 Viterbi) vs tokenizer.json (HF tokenizers, float64 Viterbi) on "
                   "long single lines. Release-build limit: release_build/sp32k/EQUIVALENCE.md section 8.1; "
                   "probes: release_card/finalize/longline_probe.py and longline_probe2.py (report-only).",
           "equivalence_long_lines_stress": json.load(open(os.path.join(REL, r"eval\reports\equivalence.json"),
                                                            encoding="utf-8"))["stress"]["categories"]["long_lines"],
           "exact_check": json.load(open(os.path.join(REL, r"eval\reports\exact_check.json"), encoding="utf-8")),
           "probe_real_text_windows": json.load(open(os.path.join(FIN, "longline_probe_dev_strict.json"),
                                                     encoding="utf-8")),
           "probe_bound_crossing_and_tie_runs": json.load(open(os.path.join(FIN, "longline_probe2_dev_strict.json"),
                                                               encoding="utf-8"))}
    json.dump(f32, open(os.path.join(REL, r"eval\float32_long_lines.json"), "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)

    # ---- release checks -------------------------------------------------------------------------------------------
    chk = {"what": "examples/test_tokenizer.py run on the final release folder F:\\Hindko\\tokenizer\\ (%s), one entry "
                   "per run: 'default' = no arguments (8 built-in samples); 'dev_strict_hp' = --jsonl "
                   "data/dev_strict.jsonl --hp-path F:\\Hindko\\_pipeline; 'full' = --jsonl dev_strict, dev_permissive, "
                   "test_strict --stress stress.jsonl --hp-path F:\\Hindko\\_pipeline. The stress set is "
                   "release_build/sp32k/stress/stress.jsonl (86,516 items; not redistributed: it contains corpus "
                   "text)." % NOW[:10], "runs": {}}
    for n in ("full", "dev_strict_hp", "default"):
        p = os.path.join(FIN, "test_tokenizer_%s.json" % n)
        if os.path.exists(p):
            chk["runs"][n] = json.load(open(p, encoding="utf-8"))
    p = os.path.join(FIN, "train_views_check.json")
    if os.path.exists(p):
        chk["train_views"] = json.load(open(p, encoding="utf-8"))
    json.dump(chk, open(os.path.join(REL, r"eval\release_checks.json"), "w", encoding="utf-8", newline="\n"),
              ensure_ascii=True, indent=1)

    # ---- eval/MANIFEST.json: add a release_finalization section ---------------------------------------------------
    mp = os.path.join(REL, r"eval\MANIFEST.json")
    man = json.load(open(mp, encoding="utf-8"))
    files_now = {}
    for root, _, fs in os.walk(os.path.join(REL, "eval")):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, os.path.join(REL, "eval")).replace("\\", "/")
            if rel != "MANIFEST.json":
                files_now[rel] = sha(p)
    man["release_finalization"] = {
        "what": "Release %s finalization (%s): test LM tables, release-build records and checks added; README.md "
                "edited. The 'files' section above is the state of %s and is kept unchanged; 'files_now' is the "
                "current state." % (VERSION, NOW, man.get("generated_utc")),
        "generated_utc": NOW,
        "script": {"path": os.path.abspath(__file__), "sha256": sha(os.path.abspath(__file__))},
        "sources": {
            "analysis/test_results.json": src["analysis/test_results.json"],
            "release_card/finalize/release_files.json": sha(os.path.join(FIN, "release_files.json")),
            "release_card/finalize/longline_probe_dev_strict.json": sha(os.path.join(FIN,
                                                                                     "longline_probe_dev_strict.json")),
            "release_card/finalize/longline_probe2_dev_strict.json": sha(os.path.join(FIN,
                                                                                      "longline_probe2_dev_strict.json")),
        },
        "added": sorted(k for k in files_now if k not in man["files"]),
        "changed": sorted(k for k in files_now if k in man["files"] and man["files"][k] != files_now[k]),
        "files_now": dict(sorted(files_now.items())),
    }
    json.dump(man, open(mp, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)

    # ---- RELEASE_MANIFEST.json (last) -----------------------------------------------------------------------------
    check_no_build_artefacts()
    allf = {}
    for root, _, fs in os.walk(REL):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, REL).replace("\\", "/")
            if rel == "RELEASE_MANIFEST.json":
                continue
            allf[rel] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
    rf = json.load(open(os.path.join(FIN, "release_files.json"), encoding="utf-8"))
    out = {"release": "Hindko tokenizer %s (R2-A4-SPnat-D2-32k, SentencePiece Unigram, 32,768 ids)" % VERSION,
           "generated_utc": NOW,
           "canonical_encoder": "tokenizer.json",
           "convenience_copy": "sp.model (float32 Viterbi; may resolve exact ties differently on single lines longer "
                               "than about 3,400 characters; README section 2.3)",
           "tokenizer_files": {r["file"]: r["sha256"] for r in rf["files"] if "/" not in r["file"]},
           "copied_from": {r["file"]: r["source"] for r in rf["files"]},
           "files": dict(sorted(allf.items()))}
    json.dump(out, open(os.path.join(REL, "RELEASE_MANIFEST.json"), "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    print("ok: %d files in RELEASE_MANIFEST.json" % len(allf))


if __name__ == "__main__":
    main()
