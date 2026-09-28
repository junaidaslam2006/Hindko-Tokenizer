# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 6: the UPDATED full ranking on test_strict (491 documents).

Rows ranked (one per distinct encoding behaviour where possible):
  - the released tokenizer (sota/competitors/results/hindko-tokenizer-1.0.0-released, harness run 2026-09-27)
  - the 66 external baselines of baselines/manifest.json (eval/results/test_strict, same harness code; checked)
  - one representative of every NEW behaviour fingerprint found by this refresh (sota/competitors/results/<name>)
Every other downloaded tokenizer is listed with its screen numbers (bytes/token, G1) and the row it encodes like.
Outputs: ../competitors_all.json, ../COMPETITORS_REFRESH.md
"""
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refresh_lib as R  # noqa: E402
import harness as H  # noqa: E402

ROOT = R.ROOT
BASE_RES = os.path.join(R.TOK, "eval", "results", "test_strict")
RELEASED_NAME = "hindko-tokenizer-1.0.0-released"
N_DOCS = 491


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def f3(x):
    return "–" if x is None else "%.3f" % x


def pct(x):
    return "–" if x is None else "%.1f%%" % (100 * x)


def slim(o):
    keys = ["docs", "bytes", "chars", "words", "tokens", "bytes_per_token", "chars_per_token", "fertility", "tokens_per_word",
            "strr", "continued_word_rate", "g1_pass_docs", "g1_fail_docs", "g1_fail_docs_modulo_whitespace", "unk_tokens",
            "nonws_chars_lost", "nonws_chars_added", "bytes_per_token_lines"]
    return {k: o.get(k) for k in keys}


def main():
    rel = jload(os.path.join(ROOT, "results", RELEASED_NAME, "summary.json"))
    code = rel["harness"]["code"]
    ro = rel["metrics"]["overall"]
    rel_tok = ro["tokens"]
    rel_src = {k: v["tokens"] for k, v in rel["metrics"]["by_source"].items()}
    man = {e["name"]: e for e in jload(os.path.join(R.BASE, "manifest.json"))["baselines"]}
    dl = jload(os.path.join(ROOT, "_download_log.json"))
    dl_by_name = {v["local_dir"].split("/", 1)[1]: v for v in dl.values() if "local_dir" in v}
    G = jload(os.path.join(ROOT, "_groups.json"))
    screens = {os.path.basename(p)[:-5]: jload(p) for p in glob.glob(os.path.join(ROOT, "screen", "*.json"))}
    code_mismatch = []
    lp = os.path.join(ROOT, "_licenses.json")
    lic = jload(lp) if os.path.exists(lp) else {}

    def row(name, s, origin, meta):
        if s["harness"]["code"] != code:
            code_mismatch.append(name)
        o = s["metrics"]["overall"]
        return dict(meta, name=name, origin=origin, tokenizer_sha256=s["tokenizer"].get("sha256"),
                    model_type=s["tokenizer"].get("model_type"), vocab_size_harness=s["tokenizer"].get("vocab_size"),
                    test_strict={"overall": slim(o), "by_source": {k: slim(v) for k, v in s["metrics"]["by_source"].items()},
                                 "G1": s["gates"]["G1"], "tokens_vs_released": o["tokens"] / rel_tok,
                                 "released_uses_fewer_tokens_pct": 100 * (1 - rel_tok / o["tokens"]),
                                 "released_uses_fewer_tokens_pct_by_source": {
                                     k: 100 * (1 - rel_src[k] / v["tokens"]) for k, v in s["metrics"]["by_source"].items()
                                     if rel_src.get(k) and v["tokens"]}})

    rows = [row(RELEASED_NAME, rel, "released", {"provider": "this study", "family": "Hindko tokenizer 1.0.0 "
                                                 "(R2-A4-SPnat-D2-32k), SentencePiece Unigram, byte fallback",
                                                 "vocab_size": 32768, "repo": "F:/Hindko/tokenizer/tokenizer.json",
                                                 "algorithm": "Unigram (SentencePiece-trained; tokenizer.json)"})]
    for d in sorted(glob.glob(os.path.join(BASE_RES, "*", "summary.json"))):
        name = os.path.basename(os.path.dirname(d))
        if name not in man:
            continue          # the study's own earlier candidate (R2-A10-...) is not an external tokenizer
        e = man[name]
        rows.append(row(name, jload(d), "baseline (66, 2026-09-26)",
                        {k: e.get(k) for k in ("provider", "family", "tier", "repo", "revision", "license", "year", "algorithm",
                                               "vocab_size", "newline_handling", "models")} |
                        {"leaky": name == "hindko-probe-bpe32k", "same_encodings_as": e.get("same_encodings_as")}))
    new_members_of_baselines, groups_out = [], []
    for g in G["groups"]:
        mem = []
        for m in g["members"]:
            s = screens[m["name"]]
            v = dl_by_name.get(m["name"], {})
            mem.append({"name": m["name"], "repo": s["repo"], "dir": s.get("dir"), "revision": s.get("revision"),
                        "kind": s["kind"], "license": v.get("license") or lic.get(s["repo"]),
                        "reconstructed_from": v.get("reconstructed_from"), "downloads": s.get("downloads"), "likes": s.get("likes"),
                        "created": s.get("created"), "found_by": s.get("how"), "vocab_size": s.get("vocab_size"),
                        "model_type": s.get("model_type"), "bytes_per_token": s["screen"]["bytes_per_token"],
                        "tokens": s["screen"]["tokens"], "g1_pass_docs": s["screen"]["g1_pass_docs"],
                        "unk_tokens": s["screen"]["unk_tokens"], "same_file_repos": s.get("members"),
                        "n_same_file_repos": s.get("n_members")})
        go = {"fingerprint": g["fingerprint"], "equals_baselines": g["baselines"], "representative": g["representative"],
              "members": mem}
        groups_out.append(go)
        if g["baselines"]:
            new_members_of_baselines.append(go)
            continue
        rep = g["representative"]
        p = os.path.join(ROOT, "results", H.safe_name(rep), "summary.json")
        if not os.path.exists(p):
            go["harness"] = "missing"
            continue
        m0 = mem[0]
        owner = m0["repo"].split("/")[0]
        rows.append(row(rep, jload(p), "new (refresh 2026-09-27)",
                        {"provider": owner, "family": "%s %s (%s)" % (m0["model_type"] or "?", format(m0["vocab_size"] or 0, ","),
                                                                     m0["kind"]),
                         "repo": m0["repo"] + ("/" + m0["dir"] if m0["dir"] else ""), "revision": m0["revision"],
                         "license": m0["license"], "vocab_size": m0["vocab_size"], "created": m0["created"],
                         "downloads": m0["downloads"], "found_by": m0["found_by"],
                         "same_encodings_as": [x["name"] for x in mem[1:]], "n_group": len(mem)}))
    ranked = sorted([r for r in rows if not r.get("leaky")], key=lambda r: -r["test_strict"]["overall"]["bytes_per_token"])
    k = 0
    for i, r in enumerate(ranked, 1):
        r["rank_bytes_per_token"] = i
        if r["test_strict"]["G1"]["fail_docs"] == 0:
            k += 1
            r["rank_among_lossless"] = k
        else:
            r["rank_among_lossless"] = None
    relrow = ranked[[r["name"] for r in ranked].index(RELEASED_NAME)]
    ext = [r for r in ranked if r["origin"] != "released"]
    ll = [r for r in ext if r["test_strict"]["G1"]["fail_docs"] == 0]
    thr_bpt = [r["name"] for r in ll if r["test_strict"]["overall"]["bytes_per_token"] >= ro["bytes_per_token"]]
    thr_fert = [r["name"] for r in ll if r["test_strict"]["overall"]["fertility"] <= ro["fertility"]]
    lossy_above = [r["name"] for r in ext if r["test_strict"]["G1"]["fail_docs"] and
                   r["test_strict"]["overall"]["bytes_per_token"] >= ro["bytes_per_token"]]
    lossy_fert = [r["name"] for r in ext if r["test_strict"]["G1"]["fail_docs"] and
                  r["test_strict"]["overall"]["fertility"] <= ro["fertility"]]
    # every screened tokenizer (incl. members of groups): the screen numbers cover all of them
    all_screen = [s for s in screens.values() if s.get("status") == "ok"]
    scr_above = sorted([(s["screen"]["bytes_per_token"], s["name"], s["screen"]["g1_pass_docs"]) for s in all_screen
                        if s["screen"]["bytes_per_token"] and s["screen"]["bytes_per_token"] >= ro["bytes_per_token"]],
                       reverse=True)
    verdict = {"released": {"bytes_per_token": ro["bytes_per_token"], "fertility": ro["fertility"], "strr": ro["strr"],
                            "g1": "%d/%d" % (ro["g1_pass_docs"], ro["docs"]), "tokens": rel_tok},
               "sota_threats_lossless_bytes_per_token": thr_bpt, "sota_threats_lossless_fertility": thr_fert,
               "lossy_tokenizers_at_or_above_released_bytes_per_token": lossy_above,
               "lossy_tokenizers_at_or_below_released_fertility": lossy_fert,
               "screened_tokenizers_at_or_above_released_bytes_per_token": scr_above,
               "best_lossless_external": ll[0]["name"] if ll else None,
               "best_lossless_external_bytes_per_token": ll[0]["test_strict"]["overall"]["bytes_per_token"] if ll else None,
               "released_uses_fewer_tokens_than_best_lossless_pct": ll[0]["test_strict"]["released_uses_fewer_tokens_pct"] if ll else None,
               "best_lossless_external_by_fertility": min(ll, key=lambda r: r["test_strict"]["overall"]["fertility"])["name"] if ll else None,
               "rank_released_bytes_per_token": relrow["rank_bytes_per_token"], "n_ranked": len(ranked),
               "rank_released_among_lossless": relrow["rank_among_lossless"], "n_lossless_ranked": k}
    out = {"what": "Refreshed competitor sweep (2026-09-27): every external tokenizer found for Hindko vs the released Hindko "
                   "tokenizer 1.0.0, intrinsic metrics on the strict TEST split with eval/harness.py (same code as "
                   "eval/TEST_COMPETITORS.md). Reporting only; no decision depends on it.",
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "dataset": rel["dataset"], "harness_code_sha256": code, "code_mismatch": code_mismatch,
           "verdict": verdict, "sweep": sweep_stats(G, screens, dl),
           "ranking": "bytes/token on test_strict, descending; the leaky probe (hindko-probe-bpe32k) is listed but not ranked",
           "tokenizers": ranked + [r for r in rows if r.get("leaky")],
           "new_tokenizers_identical_in_behaviour_to_a_baseline": new_members_of_baselines,
           "groups": groups_out,
           "not_loadable_or_failed": [{"name": s["name"], "repo": s["repo"], "kind": s["kind"], "status": s["status"]}
                                      for s in screens.values() if s.get("status") != "ok"],
           "download_errors": [{"repo": v["repo"], "dir": v.get("dir"), "kind": v["kind"], "error": v["error"],
                                "n_members": v.get("n_members")} for v in dl.values() if "error" in v]}
    json.dump(H.clean(out), open(os.path.join(ROOT, "competitors_all.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(verdict, indent=1)[:3000])
    print("ranked", len(ranked), "code mismatch", code_mismatch)


def sweep_stats(G, screens, dl):
    srch = jload(os.path.join(ROOT, "_search.json"))
    s2 = jload(os.path.join(ROOT, "_search2.json"))
    s2f = jload(os.path.join(ROOT, "_search2_filtered.json"))
    pr = jload(os.path.join(ROOT, "_probe.json"))
    p2 = jload(os.path.join(ROOT, "_probe2.json"))
    sigs = {g["sig"] for g in pr["groups"]} | {g["sig"] for g in p2["groups"]}
    return {"n_repos_found": srch["n_repos"] + s2["n_repos"], "n_repos_found_pass1": srch["n_repos"],
            "n_repos_found_pass2": s2["n_repos"], "n_repos_kept_pass2": s2f["n_repos"], "pass2_filter": s2f["filter"],
            "n_repos_with_tokenizer_files": srch["n_with_tok_files"] + s2f["n_with_tok_files"],
            "n_queries": len(srch["queries"]) + len(s2["queries"]), "n_probed": pr["n_probed"] + p2["n_probed"],
            "skipped": {k: pr["skipped"][k] + p2["skipped"][k] for k in pr["skipped"]},
            "n_probe_errors": pr["n_errors"] + p2["n_errors"],
            "n_repos_all_files_already_known": pr["n_repos_all_known"] + p2["n_repos_all_known"],
            "n_new_file_signatures": len(sigs), "n_downloaded": sum(1 for v in dl.values() if "error" not in v and
                                                                    not v.get("reconstructed_from")),
            "n_reconstructed": sum(1 for v in dl.values() if v.get("reconstructed_from")),
            "n_download_errors": sum(1 for v in dl.values() if "error" in v), "n_screened_ok": G["n_screened_ok"],
            "n_screened": G["n_screened"], "n_behaviour_groups": G["n_groups"],
            "n_groups_equal_to_a_baseline": G["n_groups_equal_to_a_baseline"], "n_new_behaviours_harnessed": G["n_to_run"]}

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
