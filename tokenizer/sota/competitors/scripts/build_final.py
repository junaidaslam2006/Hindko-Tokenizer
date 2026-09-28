# -*- coding: utf-8 -*-
"""Build competitors_final.json + COMPETITORS_FINAL.md from already-measured tokenizers (no search, no download).
Run finalize.py first (vocabulary-family hashes -> logs/_final_family.json)."""
import collections
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finalize import C, ROOT, RELEASED, LEAKY, docs_path, fingerprints  # noqa: E402

N_DOCS = 491


def load(p):
    return json.load(open(os.path.join(C, p), encoding="utf-8"))


def pct(x):
    return "%.1f %%" % (100 * x)


def main():
    A = load("competitors_all.json")
    F = load(os.path.join("logs", "_final_family.json"))
    fam = F["family"]
    T = A["tokenizers"]
    rel = next(t for t in T if t["name"] == RELEASED)
    ro = rel["test_strict"]["overall"]
    R_BPT, R_FERT, R_TOK = ro["bytes_per_token"], ro["fertility"], ro["tokens"]
    ext = [t for t in T if t["name"] not in (RELEASED, LEAKY)]

    # repos represented by each harnessed row (refresh behaviour groups: members + byte-identical file repos)
    group_members = {}
    for g in A["groups"]:
        reps = [g["representative"]] if g["representative"] else []
        for b in g.get("equals_baselines") or []:
            reps.append(b)
        repos = []
        for m in g["members"]:
            repos.extend(m.get("same_file_repos") or [m["repo"]])
        for r in reps:
            group_members.setdefault(r, set()).update(repos)

    rows = []
    for t in ext:
        o = t["test_strict"]["overall"]
        fx, ftok, nd = fingerprints(docs_path(t["name"], t["origin"]))
        assert nd == N_DOCS
        h, src = fam[t["name"]]
        repos = set(group_members.get(t["name"], set()))
        if t.get("repo") and not str(t["repo"]).startswith("local:"):
            repos.add(t["repo"])
        rows.append({
            "name": t["name"], "origin": "baseline" if t["origin"].startswith("baseline") else "new",
            "provider": t.get("provider"), "repo": t.get("repo"), "family_text": t.get("family"),
            "model_type": t.get("model_type"), "vocab_size": t.get("vocab_size_harness") or t.get("vocab_size"),
            "created": t.get("created"), "downloads": t.get("downloads"), "found_by": t.get("found_by"),
            "bytes_per_token": o["bytes_per_token"], "fertility": o["fertility"], "strr": o["strr"],
            "tokens": o["tokens"], "g1_pass_docs": o["g1_pass_docs"],
            "g1_pass_docs_modulo_whitespace": N_DOCS - o.get("g1_fail_docs_modulo_whitespace", 0),
            "unk_tokens": o.get("unk_tokens", 0), "unk_share": (o.get("unk_tokens", 0) or 0) / o["tokens"],
            "nonws_chars_lost": o.get("nonws_chars_lost", 0),
            "lossless": o["g1_pass_docs"] == N_DOCS,
            "released_uses_fewer_tokens": 1 - R_TOK / o["tokens"],
            "fp_behaviour": fx, "fp_token_counts": ftok, "vocab_family": h, "vocab_family_source": src,
            "repos": sorted(repos),
        })

    # ---- vocabulary-family labels
    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["vocab_family"]].append(r)
    fam_label = {}
    for h, rs in by_fam.items():
        bl = [r["name"] for r in rs if r["origin"] == "baseline"]
        if bl:
            lab = " / ".join(bl[:4]) + (" (+%d)" % (len(bl) - 4) if len(bl) > 4 else "")
        else:
            top = max(rs, key=lambda r: (r["downloads"] or 0))
            lab = "%s (%s %s)" % (top["repo"], top["model_type"], "{:,}".format(top["vocab_size"] or 0))
        fam_label[h] = lab
    for r in rows:
        r["vocab_family_label"] = fam_label[r["vocab_family"]]

    # ---- behaviour groups (identical per-document (tokens, round-trip) vectors)
    by_fp = collections.defaultdict(list)
    for r in rows:
        by_fp[r["fp_behaviour"]].append(r)
    groups = []
    for fp, rs in by_fp.items():
        rs.sort(key=lambda r: (r["origin"] != "baseline", r["name"]))
        rep = rs[0]
        repos = sorted(set(x for r in rs for x in r["repos"]))
        groups.append({"representative": rep["name"], "aliases": [r["name"] for r in rs[1:]],
                       "n_harness_rows": len(rs), "n_repos": len(repos), "repos": repos,
                       "origin": "baseline" if any(r["origin"] == "baseline" for r in rs) else "new",
                       "provider": rep["provider"], "repo": rep["repo"], "family_text": rep["family_text"],
                       "vocab_family": rep["vocab_family"], "vocab_family_label": rep["vocab_family_label"],
                       "model_type": rep["model_type"], "vocab_size": rep["vocab_size"],
                       **{k: rep[k] for k in ("bytes_per_token", "fertility", "strr", "tokens", "g1_pass_docs",
                                              "g1_pass_docs_modulo_whitespace", "unk_tokens", "unk_share",
                                              "nonws_chars_lost", "lossless", "released_uses_fewer_tokens",
                                              "fp_behaviour", "fp_token_counts")}})
    groups.sort(key=lambda g: -g["bytes_per_token"])
    tokprof = collections.Counter(g["fp_token_counts"] for g in groups)
    same_counts_diff_rt = [[g["representative"] for g in groups if g["fp_token_counts"] == k] for k, n in tokprof.items() if n > 1]

    # baseline 'same_encodings_as' (study labels) vs fingerprint: record fingerprint-identical pairs the label missed
    fp_of = {r["name"]: r["fp_behaviour"] for r in rows}
    missed = []
    for t in ext:
        if t["origin"].startswith("baseline"):
            same = set(t.get("same_encodings_as") or [])
            for r in by_fp[fp_of[t["name"]]]:
                if r["name"] != t["name"] and r["origin"] == "baseline" and r["name"] not in same:
                    missed.append(sorted([t["name"], r["name"]]))
    missed = sorted(set(map(tuple, missed)))

    LL = [g for g in groups if g["lossless"]]
    LY = [g for g in groups if not g["lossless"]]
    threats_bpt = [g["representative"] for g in LL if g["bytes_per_token"] >= R_BPT]
    threats_fert = [g["representative"] for g in LL if g["fertility"] <= R_FERT]
    # families: best lossless member per vocabulary family
    fam_best = {}
    for g in LL:
        h = g["vocab_family"]
        if h not in fam_best or g["bytes_per_token"] > fam_best[h]["bytes_per_token"]:
            fam_best[h] = g
    fam_rank = sorted(fam_best.values(), key=lambda g: -g["bytes_per_token"])
    lossy_above = [g for g in LY if g["bytes_per_token"] >= R_BPT]
    lossy_fert = [g for g in LY if g["fertility"] <= R_FERT]
    clean_fail = [g for g in LY if g["unk_tokens"] == 0 and g["nonws_chars_lost"] == 0]
    ws_only = [g for g in LY if g["g1_pass_docs_modulo_whitespace"] == N_DOCS and g["unk_tokens"] == 0]
    lossy_next = [g for g in LY if g["bytes_per_token"] < R_BPT][:8]

    # ---- search coverage (what the stopped search did cover)
    s1, s2, s2f, s3, s3f = (load(f) for f in ("_search.json", "_search2.json", "_search2_filtered.json",
                                               "_search3.json", "_search3_filtered.json"))
    dl = load("_download_log.json")
    snap = A["generated_utc"]
    dlv = list(dl.values())
    late = [x for x in dlv if (x.get("time_utc") or "") > snap]
    n_probe_cache = sum(1 for _ in open(os.path.join(C, "_probe_cache.jsonl"), encoding="utf-8"))
    sw = A["sweep"]
    coverage = {
        "ranking_snapshot_utc": snap,
        "pass1": {"file": "_search.json", "generated_utc": s1["generated_utc"], "n_keywords": len(s1["keywords"]),
                  "keywords": s1["keywords"], "n_lang_tags": len(s1["lang_tags"]), "lang_tags": s1["lang_tags"],
                  "n_authors": len(s1["authors"]), "n_named_repos": len(s1["named"]),
                  "n_named_missing": len(s1["named_missing"]), "n_queries": len(s1["queries"]),
                  "n_repos": s1["n_repos"], "n_with_tokenizer_files": s1["n_with_tok_files"]},
        "pass2": {"file": "_search2.json", "generated_utc": s2["generated_utc"], "n_keywords": len(s2["keywords"]),
                  "keywords": s2["keywords"], "n_authors": len(s2["authors"]), "authors": s2["authors"],
                  "named": s2["named"], "n_queries": len(s2["queries"]), "n_repos": s2["n_repos"],
                  "n_kept_after_filter": s2f["n_repos"], "filter": s2f["filter"]},
        "pass3": {"file": "_search3.json", "generated_utc": s3["generated_utc"], "authors": s3["authors"],
                  "n_queries": len(s3["queries"]), "n_repos": s3["n_repos"], "n_with_tokenizer_files": s3["n_with_tok_files"],
                  "n_kept_after_filter": s3f["n_repos"], "filter": s3f["filter"],
                  "status": "STOPPED: started after the ranking snapshot; the probe log (logs/probe_pass3.log) ends with 458 "
                            "repos still to probe; nothing from this pass was screened or measured"},
        "sweep_totals_at_snapshot": sw,
        "download_log": {"entries": len(dlv), "after_snapshot": len(late),
                         "after_snapshot_repos": sorted(x["repo"] for x in late),
                         "errors": [{"repo": x["repo"], "error": x["error"][:160]} for x in dlv if x.get("error")],
                         "note": "downloads logged after the snapshot were not screened or measured"},
        "probe_cache_lines": n_probe_cache,
        "github_and_web": "GitHub search API + web search (refresh, before the snapshot): no Hindko tokenizer or Hindko LM "
                          "found; Urdu tokenizer projects on GitHub publish code only and were not run (no third-party code).",
        "not_collected": ["gated repos (not logged into): %d skipped at search, 65 unmatched after metadata check" % sw["skipped"]["gated"],
                          "closed tokenizers: Gemini, Claude 3+, OpenAI encodings newer than o200k_base / o200k_harmony",
                          "tiktoken rank files whose split regex exists only in repo Python code; xAI tokenizer.tok.json (Grok 2)",
                          "392 large files classified by range reads as identical / expected-identical to a measured tokenizer (not measured themselves)",
                          "13 downloaded tokenizers that could not be loaded"],
    }

    counts = {
        "harness_rows_external": len(rows),
        "harness_rows_new": sum(r["origin"] == "new" for r in rows),
        "harness_rows_study_baselines": sum(r["origin"] == "baseline" for r in rows),
        "excluded": {LEAKY: "trained on the whole Hindko corpus incl. test (leaky probe; listed in study, never ranked)"},
        "distinct_encoding_behaviours": len(groups),
        "distinct_behaviours_lossless": len(LL),
        "distinct_behaviours_lossy": len(LY),
        "distinct_per_doc_token_count_profiles": len(tokprof),
        "vocabulary_families": len(by_fam),
        "vocabulary_families_with_a_lossless_member": len(fam_best),
        "repos_represented_by_measured_rows": len(set(x for r in rows for x in r["repos"])),
        "tokenizer_files_screened_ok_in_refresh": sw.get("n_screened_ok"),
        "new_files_equal_in_behaviour_to_a_baseline": sw.get("n_groups_equal_to_a_baseline"),
    }

    def slim(g, extra=()):
        keys = ["representative", "aliases", "n_repos", "origin", "provider", "repo", "vocab_family_label", "model_type",
                "vocab_size", "bytes_per_token", "fertility", "strr", "tokens", "g1_pass_docs", "unk_share",
                "nonws_chars_lost", "released_uses_fewer_tokens"] + list(extra)
        return {k: g[k] for k in keys}

    out = {
        "what": "Final competitor ranking for the released Hindko tokenizer 1.0.0 on test_strict (491 docs, 1,451,026 bytes, "
                "171,769 words): intrinsic token-count metrics of every external tokenizer ALREADY measured "
                "(refresh results/ + the study baselines), deduplicated by encoding behaviour and grouped by vocabulary family. "
                "No new search or download was done for this file.",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": A["dataset"], "harness_code_sha256": A["harness_code_sha256"],
        "method": {
            "metrics": "bytes/token = UTF-8 bytes / tokens; fertility = tokens / whitespace word; each tokenizer with its own native encoder, no BOS/EOS/CLS (eval/harness.py)",
            "lossless": "G1: decode(encode(doc)) == doc for all 491 test documents",
            "dedup": "behaviour = sha256 of the per-document (token count, round-trip ok) vector over the 491 docs (same key as scripts/screen.py); "
                     "identical vectors are treated as identical encodings (a strong proxy, not a token-by-token proof)",
            "family": "vocabulary family = sha256 of vocabulary entries at ids 0-299 and 1000-1099 (heuristic lineage key; "
                      "fine-tunes and appended vocab extensions stay in the base family)",
            "metric_check": "all %d rows re-read from their own summary.json: %d mismatches with competitors_all.json" % (len(rows), len(F["metric_mismatch"])),
        },
        "released": {"name": RELEASED, "sha256_tokenizer_json": rel.get("tokenizer_sha256"), "vocab_size": 32768,
                     "bytes_per_token": R_BPT, "fertility": R_FERT, "strr": ro["strr"], "tokens": R_TOK,
                     "g1": "%d/%d" % (ro["g1_pass_docs"], N_DOCS)},
        "verdict": {
            "any_lossless_external_beats_released_on_bytes_per_token": bool(threats_bpt),
            "any_lossless_external_beats_released_on_fertility": bool(threats_fert),
            "lossless_threats_bytes_per_token": threats_bpt, "lossless_threats_fertility": threats_fert,
            "released_rank_among_lossless_behaviours": 1 + sum(g["bytes_per_token"] > R_BPT for g in LL),
            "n_lossless_behaviours": len(LL),
            "best_lossless_external": slim(LL[0]),
            "best_lossless_external_by_fertility": slim(min(LL, key=lambda g: g["fertility"])),
            "lossy_with_higher_raw_bytes_per_token": len(lossy_above),
            "lossy_with_lower_or_equal_fertility": len(lossy_fert),
            "best_lossy_with_zero_unk_and_zero_nonspace_loss": slim(clean_fail[0]) if clean_fail else None,
            "best_lossy_whitespace_only_failures": slim(ws_only[0]) if ws_only else None,
            "scope": "Token counts on the strict Hindko test split only. A tokenizer is not a model: this says nothing about "
                     "generation, translation or downstream quality (LM bits-per-byte evidence is in analysis/TEST_RESULTS.md). "
                     "Covers only tokenizers found and measured by 2026-09-27; closed tokenizers (Gemini, Claude 3+, newer OpenAI) are not public.",
        },
        "counts": counts,
        "top15_lossless_behaviours": [slim(g) for g in LL[:15]],
        "top15_lossless_families": [dict(slim(g), family_size=len(by_fam[g["vocab_family"]])) for g in fam_rank[:15]],
        "notable_lossy": {
            "higher_raw_bytes_per_token_than_released": [slim(g) for g in lossy_above],
            "next_best_lossy_below_released": [slim(g) for g in lossy_next],
            "best_zero_unk_zero_loss_but_not_round_trip": [slim(g) for g in clean_fail[:5]],
        },
        "study_baselines_deduplicated": [slim(g) for g in groups if g["origin"] == "baseline"],
        "baseline_pairs_identical_by_fingerprint_but_not_flagged_in_study": missed,
        "groups_same_token_counts_different_round_trip": same_counts_diff_rt,
        "search_coverage": coverage,
        "all_behaviours": [dict(slim(g, ("vocab_family", "g1_pass_docs_modulo_whitespace", "unk_tokens", "fp_behaviour")),
                                repos=g["repos"]) for g in groups],
    }
    json.dump(out, open(os.path.join(C, "competitors_final.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"R_BPT": R_BPT, "R_FERT": R_FERT, "R_TOK": R_TOK}, open(os.path.join(C, "logs", "_final_released.json"), "w"))
    print(json.dumps({"counts": counts, "threats": [threats_bpt, threats_fert], "missed": missed,
                      "same_counts": same_counts_diff_rt[:10], "late": len(late), "probe_cache": n_probe_cache}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
