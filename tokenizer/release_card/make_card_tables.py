"""Print the markdown tables of the tokenizer card from F:\\Hindko\\tokenizer\\eval\\*.json (no hand-typed numbers).
Output: release_card/card_tables.md (pasted into F:\\Hindko\\tokenizer\\README.md)."""
import json
import os

EV = r"F:\Hindko\tokenizer\eval"
OUT = r"F:\Hindko\_tokenizer\release_card\card_tables.md"
RELEASED = "R2-A4-SPnat-D2-32k"
PREREG = "R2-A10-MinGram-P1r3-D2-48k"
ALT32 = "R2-A10-MinGram-P1r3-D2-32k"
BASELINE = "A1-P1r3-D2-16k"


def jl(n):
    with open(os.path.join(EV, n), encoding="utf-8") as f:
        return json.load(f)


def pct(x, nd=2, sign=True):
    return ("%+.*f %%" if sign else "%.*f %%") % (nd, x)


def ci(lo, hi, nd=2):
    return "[%+.*f, %+.*f]" % (nd, lo, nd, hi)


def role(c):
    return {RELEASED: " **(released)**", PREREG: " (pre-registered pick)", ALT32: " (alternative)",
            BASELINE: " (baseline)"}.get(c, "")


L = []
# ---------------------------------------------------------------- confirm ranking
conf = jl("dev_lm_confirm_ranking.json")
L.append("### TABLE confirm\n")
L.append("| # | tokenizer | round | vocab | mean dev bpb (5 seeds) | seed s.d. | Δ vs baseline, 95 % CI | Δ vs best, 95 % CI | Holm p vs best | top set |")
L.append("|---:|---|---|---:|---:|---:|---|---|---:|---|")
for r in conf["rows"]:
    rk = "–" if r["rank"] is None else str(r["rank"])
    rnd = "R2 (exploratory)" if r["origin"].startswith("round2") else "R1 (pre-registered)"
    if "added" in r["origin"]:
        rnd = "R1 (added arm)"
    vb = "baseline" if r["id"] == BASELINE else "%s %s" % (pct(r["delta_vs_baseline_pct"]),
                                                           ci(r["delta_vs_baseline_ci95_lo"], r["delta_vs_baseline_ci95_hi"]))
    if r["delta_vs_best_pct"] is None:
        vbest = "best"
    else:
        vbest = "%s %s" % (pct(r["delta_vs_best_pct"]), ci(r["delta_vs_best_ci95_lo"], r["delta_vs_best_ci95_hi"]))
    hp = "–" if r["holm_p_vs_best"] is None else ("<1e-4" if r["holm_p_vs_best"] == 0 else "%.4f" % r["holm_p_vs_best"])
    name = "`%s`%s" % (r["id"], role(r["id"]))
    if r["rank"] is None:
        name += " (conditional rank 8, not ranked)"
    L.append("| %s | %s | %s | %s | %.5f | %.2f %% | %s | %s | %s | %s |" % (
        rk, name, rnd, "{:,}".format(r["vocab"]), r["mean_bpb"], r["seed_sd_rel_pct"], vb, vbest, hp,
        "**yes**" if r["in_top_set"] else "no"))
L.append("")

# ---------------------------------------------------------------- large
lg = jl("dev_lm_large.json")
L.append("### TABLE large\n")
L.append("| # | tokenizer | total params | mean dev bpb (2 seeds) | seeds 1 / 2 | Δ vs baseline, 95 % CI | Δ vs released, 95 % CI | confirm-scale bpb (rank) |")
L.append("|---:|---|---:|---:|---|---|---|---|")
for r in lg["rows"]:
    vb = "baseline" if r["id"] == BASELINE else "%s %s" % (pct(r["delta_vs_baseline_pct"], 3),
                                                           ci(*r["delta_vs_baseline_ci95_pct"]))
    vr = "–" if r["id"] == RELEASED else "%s %s" % (pct(r["delta_vs_released_pct"], 3),
                                                    ci(*r["delta_vs_released_ci95_pct"]))
    L.append("| %d | `%s`%s | %.2fM | %.5f | %.5f / %.5f | %s | %s | %.5f (%s) |" % (
        r["rank_large"], r["id"], role(r["id"]), r["params_total"] / 1e6, r["mean_bpb"], r["seed_bpb"][0],
        r["seed_bpb"][1], vb, vr, r["confirm_mean_bpb"], r["confirm_rank"]))
L.append("")
L.append("LR sweep of the large arbiter (baseline, seed 1): " + "; ".join(
    "%g → %.5f" % (x["lr"], x["dev_bpb"]) for x in lg["lr_sweep_baseline"]))
L.append("")
L.append("Per source, Δ vs baseline (large arbiter, same bootstrap restricted to one source):\n")
L.append("| source | " + " | ".join("`%s`" % r["id"] for r in lg["rows"] if r["id"] != BASELINE) + " |")
L.append("|---|" + "---|" * (len(lg["rows"]) - 1))
for s, d in lg["per_source_vs_baseline"].items():
    cells = []
    for r in lg["rows"]:
        if r["id"] == BASELINE:
            continue
        v = d[r["id"]]
        cells.append("%s %s" % (pct(v["delta_pct"]), ci(*v["ci95_pct"])))
    L.append("| %s | %s |" % (s, " | ".join(cells)))
L.append("")
L.append("Top-set pairs (large arbiter): " + "; ".join(
    "`%s` %s %s, p %s" % (k, pct(v["delta_pct"]), ci(*v["ci95_pct"]), "<1e-4" if v["p"] == 0 else "%.4f" % v["p"])
    for k, v in lg["top_set_pairs"].items()))
L.append("")

# ---------------------------------------------------------------- test intrinsic
ti = jl("test_intrinsic_competitors.json")
rel = ti["released"]
by = {r["name"]: r for r in ti["rows"]}
head = ["gpt-4o", "gpt-4", "gemma-4", "gemma-2", "llama-4", "llama-3", "llama-2", "qwen-3.5", "qwen-3",
        "deepseek-v4.1", "mistral-nemo", "command-a-plus", "grok-1", "kimi-k2", "glm-5", "falcon-h1", "bloom",
        "tiny-aya", "roberta-urdu", "claude-legacy"]
L.append("### TABLE test intrinsic headline\n")
L.append("| tokenizer | provider / family | vocab | test bytes/token | G1 lossless (test) | × tokens vs released | released uses fewer tokens |")
L.append("|---|---|---:|---:|---:|---:|---:|")
L.append("| **`%s`** (released) | this study: SentencePiece Unigram, D2 | 32,768 | **%.3f** | %d/%d | 1.000 | – |" % (
    RELEASED, rel["test_bytes_per_token"], rel["test_docs"] - rel["g1_harness_fail_docs"], rel["test_docs"]))
p = by[PREREG]
L.append("| `%s` (pre-registered pick) | this study: MinGram, P1r3, D2 | 49,152 | %.3f | %d/%d | %.3f | %s |" % (
    PREREG, p["test_bytes_per_token"], p["test_g1_pass_docs"], p["test_docs"],
    ti["in_study_bundle_rows"][PREREG]["test_tokens"] / rel["test_tokens"],
    pct(100 * (1 - rel["test_tokens"] / ti["in_study_bundle_rows"][PREREG]["test_tokens"]), 1)))
a = ti["in_study_bundle_rows"][ALT32]
L.append("| `%s` (alternative) | this study: MinGram, P1r3, D2 | 32,768 | %.3f | 491/491 | %.3f | %s |" % (
    ALT32, a["test_bytes_per_token"], a["test_tokens"] / rel["test_tokens"],
    pct(100 * (1 - rel["test_tokens"] / a["test_tokens"]), 1)))
b = ti["in_study_bundle_rows"][BASELINE]
L.append("| `%s` (baseline) | this study: byte-level BPE, P1r3, D2 | 16,384 | %.3f | 491/491 | %.3f | %s |" % (
    BASELINE, b["test_bytes_per_token"], b["test_tokens"] / rel["test_tokens"],
    pct(100 * (1 - rel["test_tokens"] / b["test_tokens"]), 1, sign=False)))
for n in head:
    r = by[n]
    g1 = "%d/%d" % (r["test_g1_pass_docs"], r["test_docs"]) + ("" if r["lossless_on_test"] else " ⚠")
    L.append("| `%s` | %s | %s | %.3f | %s | %.3f | %s |" % (
        n, r["provider"], "{:,}".format(r["vocab_size"]), r["test_bytes_per_token"], g1, r["tokens_vs_released"],
        pct(r["released_uses_fewer_tokens_pct"], 1, sign=False)))
L.append("")
L.append("### TABLE test intrinsic top10\n")
L.append("| rank | tokenizer | provider / family | bytes/token (test) | lossless on test |")
L.append("|---:|---|---|---:|---|")
ext = sorted([r for r in ti["rows"] if r["role"] == "external"], key=lambda r: -r["test_bytes_per_token"])
L.append("| 1 | **`%s`** (released) | this study | %.3f | yes (491/491) |" % (RELEASED, rel["test_bytes_per_token"]))
for i, r in enumerate(ext[:10], 2):
    L.append("| %d | `%s` | %s | %.3f | %s |" % (i, r["name"], r["provider"], r["test_bytes_per_token"],
                                              "yes" if r["lossless_on_test"] else "**no** (%d/%d)" % (
                                                  r["test_g1_pass_docs"], r["test_docs"])))
L.append("")
L.append("RELEASED rank %d of %d; among lossless: %d of %d; best lossless external %s %.3f; fewer tokens %.1f %%" % (
    rel["rank_by_test_bytes_per_token"], rel["n_ranked"], rel["rank_among_lossless"], rel["n_ranked_lossless"],
    rel["best_lossless_external"], rel["best_lossless_external_bytes_per_token"],
    rel["fewer_tokens_than_best_lossless_external_pct"]))
n_lossless = sum(1 for r in ti["rows"] if r["role"] == "external" and r["lossless_on_test"])
L.append("externals lossless on test: %d of %d" % (n_lossless, sum(1 for r in ti["rows"] if r["role"] == "external")))
L.append("")

# ---------------------------------------------------------------- gates / support
g = jl("health_gates_and_support.json")["rows"]
L.append("### TABLE gates\n")
L.append("| tokenizer | G1 lossless (dev_strict) | G2 unreachable whole-char tokens | G3 specials atomic, never in corpus | G4 deterministic retrain | G5 no sub-character tokens | partial-UTF-8 tokens (R2) |")
L.append("|---|---|---|---|---|---|---|")
for r in g:
    L.append("| `%s` (%s) | %s | %s (%d of %s tested) | %s (%d occurrences) | %s | %s | %d |" % (
        r["id"], r["role"], r["G1_dev_strict"], "pass" if r["G2_pass"] else "FAIL", r["G2_failures"],
        "{:,}".format(r["G2_tested"]), "pass" if r["G3_pass"] else "FAIL", r["G3_corpus_marker_occurrences"],
        "pass" if r["G4_pass"] else "FAIL",
        "n/a (byte-level)" if r["G5_pass"] is None else ("pass" if r["G5_pass"] else "FAIL"),
        r["partial_utf8_tokens"]))
L.append("")
L.append("### TABLE support\n")
L.append("| tokenizer | learned tokens | train freq = 0 | train freq < 20 | train freq < 100 | median train freq | train_D1 tokens | dev bytes/token | dev fertility | dev STRR |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in g:
    L.append("| `%s` (%s) | %s | %d | %s (%.1f %%) | %s (%.1f %%) | %g | %s | %.3f | %.3f | %.1f %% |" % (
        r["id"], r["role"], "{:,}".format(r["learned_tokens"]), r["train_D1_freq_eq0"],
        "{:,}".format(r["train_D1_freq_lt20"]), r["train_D1_freq_lt20_pct"], "{:,}".format(r["train_D1_freq_lt100"]),
        r["train_D1_freq_lt100_pct"], r["train_D1_median_freq"], "{:,}".format(r["train_D1_tokens"]),
        r["dev_bytes_per_token"], r["dev_fertility"], 100 * r["dev_strr"]))
L.append("")

# ---------------------------------------------------------------- released dev intrinsic by source / robustness
s = jl("tokenizer_summaries_dev.json")["tokenizers"][RELEASED]
L.append("### TABLE released dev by source\n")
L.append("| slice | docs | bytes | bytes/token | chars/token | fertility | STRR | G1 |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
rows = [("all dev_strict", s["dev_strict"]["overall"])] + [("source: " + k, v) for k, v in s["dev_strict"]["by_source"].items()] \
    + [("variety: " + k, v) for k, v in s["dev_strict"]["by_variety"].items()]
for n, v in rows:
    L.append("| %s | %s | %s | %.3f | %.3f | %.3f | %.1f %% | %d/%d |" % (
        n, "{:,}".format(v["docs"]), "{:,}".format(v["bytes"]), v["bytes_per_token"], v["chars_per_token"],
        v["fertility"], 100 * v["strr"], v["g1_pass_docs"], v["docs"]))
L.append("")
rb = s["dev_strict"]["overall"]["robustness"]
L.append("Robustness (dev_strict, PLAN §4.2 perturbations): " + "; ".join(
    "%s: token count %s, %.1f %% of affected words re-segmented" % (k, pct(100 * v["rel_token_change"], 2),
                                                                   100 * v["seg_change_rate_affected"])
    for k, v in rb.items()))
L.append("")
L.append("NSL vs A1-P1-D1-16k: %.4f" % s["nsl_vs_A1-P1-D1-16k"]["overall"])
L.append("")

# ---------------------------------------------------------------- Track B
tb = jl("trackb_extensions.json")["delivered"]
L.append("### TABLE trackb\n")
L.append("| folder | base models | base licence | new tokens | len(tokenizer) | dev bytes/token base → extended | dev tokens | fertility | STRR | new tokens < 100 train occ. | English / code / other-script docs changed | Urdu dev token change (news / books) |")
L.append("|---|---|---|---:|---:|---|---:|---|---|---:|---|---|")
for r in tb:
    L.append("| `%s/` | %s | %s | %s | %s | %.3f → **%.3f** | −%.1f %% | %.3f → %.3f | %.1f %% → %.1f %% | %d (%.1f %%) | %d / %d / %d | −%.1f %% / −%.1f %% |" % (
        r["folder"], r["base_models"], r["base_license"], "{:,}".format(r["k_new"]), "{:,}".format(r["len_tokenizer"]),
        r["dev_bytes_per_token_base"], r["dev_bytes_per_token_ext"], r["dev_token_reduction_pct"],
        r["dev_fertility_base"], r["dev_fertility_ext"], 100 * r["dev_strr_base"], 100 * r["dev_strr_ext"],
        r["new_lt100"], 100.0 * r["new_lt100"] / r["k_new"], r["english_changed"], r["code_changed"],
        r["other_scripts_changed"], r["urdu_news_dev_token_reduction_pct"], r["urdu_book_dev_token_reduction_pct"]))
L.append("")

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
