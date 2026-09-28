# -*- coding: utf-8 -*-
"""Collect the ONE-SHOT test-split intrinsic results (eval/run_test_competitors.py) into
eval/test_competitors.json and eval/TEST_COMPETITORS.md: the final competitor table of the chosen tokenizer
against every working external tokenizer (GPT-4o, Gemma, Llama, Qwen, DeepSeek, ...).

Reads eval/results/test_strict/<name>/summary.json. Dev reference columns come from eval/baselines_dev.json
(externals) and the chosen tokenizer's dev_strict summary. The in-study rows (bytes/token and G1 of the four
final-test LM candidates) come from colab/build_final_test/VERIFY_FINAL_TEST.json, i.e. from the final-test
bundle encoding, not from a new pass over the test text. Reporting only: nothing here changes a decision.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import harness as H  # noqa: E402
import adapters as A  # noqa: E402
import report_baselines as RB  # noqa: E402  (slim, f3, pct)

sys.path.insert(0, A.BASELINES_DIR)
from load_baselines import list_baselines, manifest  # noqa: E402

DATA = "test_strict"
CHOSEN = "R2-A10-MinGram-P1r3-D2-48k"
CHOSEN_DEV = os.path.join(TOK, "candidates", "round2", "mingram", "results", "dev_strict", CHOSEN, "summary.json")
VERIFY = os.path.join(TOK, "colab", "build_final_test", "VERIFY_FINAL_TEST.json")
HEADLINE = ["gpt-4o", "gpt-4", "gemma-4", "gemma-2", "llama-4", "llama-3", "llama-2", "qwen-3.5", "qwen-3",
            "deepseek-v4.1", "deepseek-v3", "mistral-nemo", "command-a-plus", "phi-4", "grok-1", "kimi-k2", "glm-5",
            "falcon-h1", "bloom", "roberta-urdu", "tiny-aya", "claude-legacy"]
SRCS = ["newspaper", "book", "web"]
f3, pct = RB.f3, RB.pct


def load(name):
    p = os.path.join(H.RESULTS, DATA, H.safe_name(name), "summary.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt_int(x):
    return "–" if x is None else format(int(x), ",")


def main():
    man = {e["name"]: e for e in manifest()["baselines"]}
    names = list_baselines(working_only=True)
    dev_b = {r["name"]: r for r in json.load(open(os.path.join(HERE, "baselines_dev.json"), encoding="utf-8"))["baselines"]}
    cs = load(CHOSEN)
    if cs is None:
        raise SystemExit("no test result for the chosen tokenizer yet")
    cdev = json.load(open(CHOSEN_DEV, encoding="utf-8"))
    tok_c = cs["metrics"]["overall"]["tokens"]
    tok_c_src = {s: v["tokens"] for s, v in cs["metrics"]["by_source"].items()}
    codes, missing, rows = set(), [], []

    def row(name, s, role, meta, dev_over, dev_g1):
        o = s["metrics"]["overall"]
        codes.add(json.dumps(s["harness"]["code"], sort_keys=True))
        return {"name": name, "role": role, "provider": meta.get("provider"), "family": meta.get("family"),
                "algorithm": meta.get("algorithm"), "vocab_size": meta.get("vocab_size"),
                "tokenizer_sha256": s["tokenizer"].get("sha256"),
                "newline_handling": meta.get("newline_handling"),
                "same_encodings_as": meta.get("same_encodings_as") or meta.get("identical_to"),
                "test_strict": {"overall": RB.slim(o),
                                "by_source": {k: RB.slim(v) for k, v in s["metrics"]["by_source"].items()},
                                "by_variety": {k: RB.slim(v) for k, v in s["metrics"]["by_variety"].items()},
                                "G1": s["gates"]["G1"],
                                "tokens_vs_chosen": o["tokens"] / tok_c,
                                "tokens_vs_chosen_by_source": {k: v["tokens"] / tok_c_src[k]
                                                               for k, v in s["metrics"]["by_source"].items() if tok_c_src.get(k)},
                                "chosen_uses_fewer_tokens_pct": 100 * (1 - tok_c / o["tokens"]),
                                "offset_id_mismatch_docs": s["tokenizer"].get("offset_id_mismatch_docs")},
                "dev_strict_reference": ({"bytes_per_token": dev_over["bytes_per_token"], "fertility": dev_over["fertility"],
                                          "strr": dev_over["strr"], "g1_fail_docs": dev_g1["fail_docs"]}
                                         if dev_over else None)}

    ctok = cs["tokenizer"]
    rows.append(row(CHOSEN, cs, "chosen", {"provider": "this study", "vocab_size": ctok.get("vocab_size"),
                                           "family": "MinGram (A10), pre-tokenizer P1r3, trained on strict train D2; "
                                                     "chosen on dev by the pre-registered rule",
                                           "algorithm": "MinGram / Unigram (HF tokenizer.json, byte fallback)",
                                           "newline_handling": "preserved"},
                    cdev["metrics"]["overall"], cdev["gates"]["G1"]))
    for n in names:
        s = load(n)
        if s is None:
            missing.append(n)
            continue
        d = dev_b.get(n)
        rows.append(row(n, s, "leaky" if n in A.LEAKY_BASELINES else "external", man[n],
                        d["dev_strict"]["overall"] if d else None, d["dev_strict"]["G1"] if d else None))
    comp = sorted([r for r in rows if r["role"] != "leaky"], key=lambda r: -r["test_strict"]["overall"]["bytes_per_token"])
    leaky = [r for r in rows if r["role"] == "leaky"]
    k = 0
    for i, r in enumerate(comp, 1):
        r["rank_test_bytes_per_token"] = i
        if r["test_strict"]["G1"]["fail_docs"] == 0:
            k += 1
            r["rank_among_lossless"] = k
        else:
            r["rank_among_lossless"] = None
    by = {r["name"]: r for r in rows}
    ver = json.load(open(VERIFY, encoding="utf-8")) if os.path.exists(VERIFY) else None
    tm = json.load(open(os.path.join(TOK, "data", "test_manifest.json"), encoding="utf-8"))
    out = {"what": "ONE-SHOT intrinsic competitor table on the strict TEST split (PLAN 7.6 / Stage 5): the chosen "
                   "tokenizer vs every working external tokenizer. Reported, never used to change a decision.",
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "frozen": H._frozen_short(), "dataset": cs["dataset"],
           "test_manifest": {"sha256_of_file": H.sha256_file(os.path.join(TOK, "data", "test_manifest.json")),
                             "stats": tm["views"]["test_strict"]["total"],
                             "bootstrap_clusters": tm["views"]["test_strict"]["bootstrap_clusters"]},
           "harness_code_sha256": [json.loads(x) for x in sorted(codes)],
           "chosen": CHOSEN, "ranking": "test_strict bytes/token, descending; the leaky probe is excluded",
           "n_external_evaluated": len(rows) - 1, "n_ranked": len(comp), "missing": missing,
           "headline": [n for n in HEADLINE if n in by],
           "in_study_bundle_rows": (ver or {}).get("g1_test_strict"),
           "tokenizers": comp + leaky}
    with open(os.path.join(HERE, "test_competitors.json"), "w", encoding="utf-8") as f:
        json.dump(H.clean(out), f, ensure_ascii=False, indent=1)
    write_md(out, comp, leaky, by, ver)
    print("ranked", len(comp), "leaky", len(leaky), "missing", missing)


def write_md(out, comp, leaky, by, ver):
    L = []
    a = L.append
    ds = out["dataset"]
    c = by[CHOSEN]
    co = c["test_strict"]["overall"]
    n_docs = ds["docs"]
    n_ll = sum(1 for r in comp if r["test_strict"]["G1"]["fail_docs"] == 0)
    ext_ll = [r for r in comp if r["role"] == "external" and r["test_strict"]["G1"]["fail_docs"] == 0]
    ext_all = [r for r in comp if r["role"] == "external"]
    a("# Final competitor table on the Hindko TEST split (one shot)")
    a("")
    a("Generated %s by `eval/report_test_competitors.py` from `eval/run_test_competitors.py` (harness code as in "
      "Stage 0). **These are TEST numbers, computed once, after the tokenizer was chosen on dev** "
      "(`analysis/decision.json`, fixed %s). They are reported; they change no decision." %
      (out["generated_utc"], "before the test split was opened"))
    a("")
    a("- **Text:** `data/test_strict.jsonl`, the strict test split of manifest `%s…`: %d documents, %s UTF-8 bytes, "
      "%s whitespace words, %d bootstrap clusters (%s), sha256 `%s…`, in canonical form (`hp.normalize` %s). It was "
      "materialised once by `data/materialize_test.py`, exactly like dev_strict (`data/test_manifest.json`)." % (
          out["frozen"]["split_manifest_sha256"][:8], n_docs, format(co["bytes"], ","), format(co["words"], ","),
          sum(out["test_manifest"]["bootstrap_clusters"].values()),
          ", ".join("%s %d" % (k, v) for k, v in sorted(out["test_manifest"]["bootstrap_clusters"].items())),
          ds["sha256"][:12], out["frozen"]["normalize_version"]))
    a("- **Tokenizers:** the chosen tokenizer `%s` and all %d working external tokenizers of `baselines/manifest.json`, "
      "each with its own native encoder, no BOS/EOS/CLS added. Same metric code and definitions as "
      "`eval/BASELINES_DEV.md`." % (CHOSEN, out["n_external_evaluated"]))
    best_ext = ext_ll[0] if ext_ll else None
    a("- **Chosen tokenizer on test:** %s bytes/token, fertility %s, STRR %s, G1 %d/%d (lossless). Rank **%d of %d** by "
      "bytes/token, **%d of %d** among the tokenizers that round-trip every test document." % (
          f3(co["bytes_per_token"]), f3(co["fertility"]), pct(co["strr"]), n_docs - c["test_strict"]["G1"]["fail_docs"],
          n_docs, c["rank_test_bytes_per_token"], len(comp), c["rank_among_lossless"] or 0, n_ll))
    if best_ext:
        bo = best_ext["test_strict"]["overall"]
        a("  - Best lossless external tokenizer: `%s` (%s) at %s bytes/token; the chosen tokenizer uses **%.1f %% fewer "
          "tokens** on the same text." % (best_ext["name"], best_ext["provider"], f3(bo["bytes_per_token"]),
                                         best_ext["test_strict"]["chosen_uses_fewer_tokens_pct"]))
    lossy_above = [r for r in comp if r["rank_test_bytes_per_token"] < c["rank_test_bytes_per_token"]
                   and r["test_strict"]["G1"]["fail_docs"]]
    if lossy_above:
        a("  - Ranked above it: %s. %s not lossless: they drop line breaks, fold or strip characters, or emit UNK, so they "
          "encode less text than they were given (see *lines b/tok* and the G1 column)." % (
              ", ".join("`%s`" % r["name"] for r in lossy_above), "They are" if len(lossy_above) > 1 else "It is"))
    a("- **Ranking:** bytes/token on test_strict, descending. Bytes/token is a screening metric (PLAN §4.2); the "
      "decision metric is LM bits-per-byte, which the one-shot LM run (`colab/FINAL_TEST.md`) measures for the four "
      "test candidates only. `hindko-probe-bpe32k` is LEAKY (trained on all data, including this test text) and is "
      "never ranked.")
    a("- **Claim this supports (PLAN §8):** \"uses x % fewer tokens per byte of Hindko than the tokenizers of <external "
      "models measured>\" — read x off the headline table. It says nothing about downstream quality.")
    a("")
    a("## Headline: chosen tokenizer vs the big model families (test_strict)")
    a("")
    a("*fewer tokens* = 1 − tokens(chosen) ÷ tokens(competitor) on the same %d documents; *× tokens* = tokens(competitor) "
      "÷ tokens(chosen). *dev b/tok* is the Stage 0 dev_strict value, for reference." % n_docs)
    a("")
    a("| tokenizer | provider — family | vocab | test bytes/tok | dev b/tok | fertility | STRR | G1 | × tokens | chosen uses fewer tokens |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    a("| **`%s`** (chosen) | this study | %s | **%s** | %s | %s | %s | %d/%d | 1.000 | – |" % (
        CHOSEN, fmt_int(c["vocab_size"]), f3(co["bytes_per_token"]), f3(c["dev_strict_reference"]["bytes_per_token"]),
        f3(co["fertility"]), pct(co["strr"]), n_docs - c["test_strict"]["G1"]["fail_docs"], n_docs))
    for n in out["headline"]:
        r = by[n]
        o = r["test_strict"]["overall"]
        g = r["test_strict"]["G1"]
        a("| `%s` | %s — %s | %s | %s | %s | %s | %s | %d/%d%s | %.3f | %.1f %% |" % (
            n, r["provider"] or "", r["family"] or "", fmt_int(r["vocab_size"]), f3(o["bytes_per_token"]),
            f3((r["dev_strict_reference"] or {}).get("bytes_per_token")), f3(o["fertility"]), pct(o["strr"]),
            g["docs"] - g["fail_docs"], g["docs"], "" if not g["fail_docs"] else " ⚠", r["test_strict"]["tokens_vs_chosen"],
            r["test_strict"]["chosen_uses_fewer_tokens_pct"]))
    a("")
    a("⚠ = not lossless on test (fails `decode(encode(doc)) == doc`), so its token count covers less text than the chosen "
      "tokenizer's. Tokenizers with identical encodings (e.g. `gemma-3` = `gemma-4`, `gpt-oss`/`phi-4-mini` = `gpt-4o`) "
      "are in the full table.")
    a("")
    a("## Full ranking on test_strict")
    a("")
    a("*#* = rank by bytes/token; *ll #* = rank among the tokenizers that round-trip all %d documents; *dev b/tok* is "
      "the dev_strict value (Stage 0 for the externals, the candidate's dev summary for the chosen tokenizer), for "
      "reference; *× tokens* = tokens ÷ the chosen tokenizer's tokens on the same documents." % n_docs)
    a("")
    a("| # | ll # | tokenizer | provider / family | vocab | algorithm | bytes/tok | dev b/tok | chars/tok | × tokens | fertility | STRR | G1 | UNK | lines b/tok | note |")
    a("|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in comp:
        o = r["test_strict"]["overall"]
        g = r["test_strict"]["G1"]
        note = []
        if r["role"] == "chosen":
            note.append("**chosen (this study)**")
        if g["fail_docs"]:
            note.append("**not lossless**")
        if r["newline_handling"] and r["newline_handling"] != "preserved":
            note.append("line breaks: " + r["newline_handling"])
        if r["same_encodings_as"]:
            same = r["same_encodings_as"] if isinstance(r["same_encodings_as"], list) else [r["same_encodings_as"]]
            note.append("same encodings as " + ", ".join("`%s`" % x for x in same))
        nm = "**`%s`**" % r["name"] if r["role"] == "chosen" else "`%s`" % r["name"]
        a("| %d | %s | %s | %s — %s | %s | %s | **%s** | %s | %s | %.3f | %s | %s | %d/%d | %s | %s | %s |" % (
            r["rank_test_bytes_per_token"], r["rank_among_lossless"] or "–", nm, r["provider"] or "", r["family"] or "",
            fmt_int(r["vocab_size"]), r["algorithm"] or "", f3(o["bytes_per_token"]),
            f3((r["dev_strict_reference"] or {}).get("bytes_per_token")), f3(o["chars_per_token"]),
            r["test_strict"]["tokens_vs_chosen"], f3(o["fertility"]), pct(o["strr"]), g["docs"] - g["fail_docs"], g["docs"],
            fmt_int(o["unk_tokens"]), f3(o["bytes_per_token_lines"]), "; ".join(note)))
    a("")
    a("## Leaky reference (not a competitor)")
    a("")
    a("| tokenizer | why it is not ranked | vocab | bytes/tok | fertility | STRR | G1 | lines b/tok |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in leaky:
        o = r["test_strict"]["overall"]
        g = r["test_strict"]["G1"]
        a("| `%s` | **LEAKY**: trained on the whole corpus, including this test text | %s | %s | %s | %s | %d/%d | %s |" % (
            r["name"], fmt_int(r["vocab_size"]), f3(o["bytes_per_token"]), f3(o["fertility"]), pct(o["strr"]),
            g["docs"] - g["fail_docs"], g["docs"], f3(o["bytes_per_token_lines"])))
    a("")
    a("## Per source (test_strict)")
    a("")
    sz = c["test_strict"]["by_source"]
    a("test_strict holds %s. Web is small, so its numbers are noisy." % ", ".join(
        "%s %d docs / %s bytes" % (s, sz[s]["docs"], format(sz[s]["bytes"], ",")) for s in SRCS if s in sz))
    a("")
    a("| tokenizer | " + " | ".join("%s b/tok | %s fert. | %s × tok" % (s, s, s) for s in SRCS) + " |")
    a("|---|" + "---:|---:|---:|" * len(SRCS))
    for n in [CHOSEN] + out["headline"]:
        r = by[n]
        bs = r["test_strict"]["by_source"]
        tv = r["test_strict"]["tokens_vs_chosen_by_source"]
        a("| %s | " % ("**`%s`**" % n if n == CHOSEN else "`%s`" % n) + " | ".join(
            "%s | %s | %s" % (f3(bs.get(s, {}).get("bytes_per_token")), f3(bs.get(s, {}).get("fertility")),
                              "%.3f" % tv[s] if s in tv else "–") for s in SRCS) + " |")
    a("")
    # lossy
    a("## Tokenizers that are not lossless on test_strict")
    a("")
    a("| tokenizer | G1 fail docs | fail modulo whitespace | chars lost | chars added | UNK |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in comp + leaky:
        g = r["test_strict"]["G1"]
        if g["fail_docs"] or g["unk_tokens"]:
            a("| `%s`%s | %d | %d | %s | %s | %s |" % (r["name"], " (leaky)" if r["role"] == "leaky" else "", g["fail_docs"],
                                                   g["fail_docs_modulo_whitespace"], format(g["nonws_chars_lost"], ","),
                                                   format(g["nonws_chars_added"], ","), format(g["unk_tokens"], ",")))
    a("")
    a("The failure causes are the ones diagnosed on dev (`eval/BASELINES_DEV.md`, `eval/lossy_diagnosis.json`): NFKC "
      "rewriting, dropped line breaks, character folding, UNK tokens. They were not re-diagnosed on test.")
    a("")
    if ver:
        a("## In-study LM candidates on test (from the final-test bundle encoding)")
        a("")
        a("These four are the LM test candidates of `colab/FINAL_TEST.md`. Their bytes/token and G1 below come from the "
          "encoding already made for the final-test bundle (`colab/build_final_test/VERIFY_FINAL_TEST.json`), not from "
          "the harness. For the chosen tokenizer both routes must agree.")
        a("")
        a("| tokenizer | role | test bytes/tok (bundle) | dev bytes/tok (dev bundle) | test tokens | G1 harness encoder | G1 native encoder |")
        a("|---|---|---:|---:|---:|---:|---:|")
        roles = {"R2-A10-MinGram-P1r3-D2-48k": "chosen", "A1-P1r3-D2-16k": "baseline"}
        for n, g in ver["g1_test_strict"].items():
            a("| `%s` | %s | %s | %s | %s | %d/%d | %d/%d |" % (
                n, roles.get(n, "report-only"), f3(g["test_bytes_per_token"]), f3(g["dev_bytes_per_token_from_dev_bundle"]),
                format(g["test_tokens"], ","), g["docs"] - g["harness"]["g1_fail_docs"], g["docs"],
                g["docs"] - g["builtin"]["g1_fail_docs"], g["docs"]))
        hb = ver["g1_test_strict"][CHOSEN]["test_bytes_per_token"]
        a("")
        a("Cross-check: chosen tokenizer bytes/token on test = %s (harness) vs %s (bundle) → %s." % (
            f3(co["bytes_per_token"], 6), f3(hb, 6), "identical" if abs(hb - co["bytes_per_token"]) < 5e-6 else "DIFFERENT"))
        a("")
    a("## What this page does not cover")
    a("")
    a("- **Bits-per-byte on test** comes from the one-shot LM run (`colab/FINAL_TEST.md`), for the four LM test "
      "candidates only; external tokenizers have no LM in this study.")
    a("- **Robustness and morphology** were computed by the same harness run and are in `eval/test_competitors.json` "
      "(robustness) and the per-tokenizer `summary.json` files; they are report-only, as on dev.")
    a("- **G2–G5, R1:** gates for the project's own candidates, run on dev (PLAN §4.3); not repeated here.")
    a("")
    a("## Reproduce")
    a("")
    a("```")
    a("set PYTHONIOENCODING=utf-8")
    a("python eval\\run_test_competitors.py --shard 0 --of 3   (and --shard 1, --shard 2, in parallel)")
    a("python eval\\report_test_competitors.py")
    a("```")
    a("")
    a("Results: `eval/results/test_strict/<name>/`. Harness code sha256: " + "; ".join(
        ", ".join("%s `%s…`" % (k, v[:10]) for k, v in hc.items()) for hc in out["harness_code_sha256"]) + ".")
    with open(os.path.join(HERE, "TEST_COMPETITORS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
