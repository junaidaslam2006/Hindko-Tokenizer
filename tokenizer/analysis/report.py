"""Render analysis/DECISION.md from analysis/decision.json (written by analysis/decide.py). No computation of its own
beyond formatting and counting. Run: PYTHONIOENCODING=utf-8 python analysis/report.py"""
import hashlib
import json
import os

OUT = r"F:\Hindko\_tokenizer\analysis"
d = json.load(open(os.path.join(OUT, "decision.json"), encoding="utf-8"))
L = []
A = L.append

BASE = d["baseline"]
DEC = d["decision"]
CH, BEST = DEC["chosen"], DEC["best"]
RK = {r["id"]: r for r in d["ranking"]}
TB = {r["id"]: r["tiebreak"] for r in d["ranking"]}
C = d["contrasts"]
S34 = d["stage3_vs_stage4"]
PW = d["power"]


def p_(p):
    return "<1e-4" if p == 0 else ("%.4f" % p)


def ci(x, nd=2):
    return "[%+.*f, %+.*f]" % (nd, x[0], nd, x[1])


def dl(x, nd=2):
    """Delta % with its 95 % CI."""
    return "%+.*f %% %s" % (nd, x["delta_pct"], ci(x["ci95_pct"], nd))


def dlp(x, nd=2):
    return "%s, p %s" % (dl(x, nd), p_(x["p"]))


def ci90pct(x, ref_bpb):
    return "[%+.2f, %+.2f]" % (100 * x["ci90"][0] / ref_bpb, 100 * x["ci90"][1] / ref_bpb)


def short(c):
    return "`%s`" % c


base_bpb = RK[BASE]["mean_bpb"]
best_bpb = RK[BEST]["mean_bpb"]
vsb = DEC["chosen_vs_baseline"]
delta_abs = d["protocol"]["delta_abs_bpb"]
top = DEC["top_set"]
ranked_primary = [r for r in d["ranking"] if r["in_primary_family"]]
k = DEC["k"]

A("# Hindko tokenizer: the pre-registered decision (PLAN §6–§7) on Stage 4, both rounds")
A("")
A("Generated %s by `analysis/decide.py` (sha256 `%s…`) and `analysis/report.py`. PLAN.md sha256 `%s…` (equals "
  "FROZEN.json). Every number below is on **dev_strict** (836 documents, 1,445,513 bytes). **The test split was not "
  "read, encoded or scored.** Machine-readable twin: `analysis/decision.json`."
  % (d["generated_utc"], d["code"]["decide.py"][:12], d["inputs"]["plan_sha256"][:8]))
A("")

# ------------------------------------------------------------------ 0
A("## 0. Decision")
A("")
A("- **Chosen tokenizer: %s.** It is MinGram (A10, as reimplemented from the paper), with pre-tokenizer P1r3, trained on "
  "the strict-train mix D2, with 49,152 tokens including the 64 specials. It is a stock HF `tokenizer.json`, and its "
  "ids equal the reference encoder on every dev document." % short(CH))
A("- **Ranking (PLAN §7.2)** is on the Stage 4 'confirm' arbiter (d=192, full permissive train split, 1 epoch, LR 1e-3, "
  "5 seeds each). There are %d ranked candidates: all 18 LM candidates of both rounds, minus the conditional rank 8, "
  "whose pre-registered condition is false (§8, deviation 6). The best mean is %s at %.5f bpb. The chosen tokenizer "
  "is ranked %d, at %.5f bpb: %s against the best."
  % (k, short(BEST), best_bpb, RK[CH]["rank_primary"], RK[CH]["mean_bpb"], dlp(DEC["chosen_vs_best"])))
A("- **Top set (PLAN §7.3): %d members**, %s. Every other candidate is significantly worse than the best after Holm "
  "correction (%d comparisons). No candidate is TOST-equivalent to the best within ±0.3 %% (δ = %.6f bpb)."
  % (len(top), ", ".join(short(c) for c in top), k - 1, delta_abs))
tr = {t["step"]: t for t in DEC["tiebreak_trace"]}
A("- **Tie-breakers (PLAN §7.4)**, applied in order:")
for _c in tr["a"]["dropped"]:
    A("  - (a) HF-native exact encoding drops %s: %s." % (short(_c), TB[_c]["hf_native_evidence"]))
A("  - (b) Higher dev bytes/token keeps %s (%.4f) over %s (%.4f)."
  % (short(CH), TB[CH]["dev_bytes_per_token"], short(tr["b"]["dropped"][0]), TB[tr["b"]["dropped"][0]]["dev_bytes_per_token"]))
A("  - (c)–(e) were not reached.")
A("  - The choice does not hinge on (a): the chosen tokenizer also has the highest bytes/token of the three.")
A("- **Improvement claim (PLAN §7.5): allowed on dev.** Against the standard recipe %s, Δ = %s (p %s). The 95 %% CI "
  "excludes 0. %s is %.5f bpb against %.5f for the baseline."
  % (short(BASE), dl(vsb), p_(vsb["p"]), short(CH), RK[CH]["mean_bpb"], base_bpb))
A("  - The chosen tokenizer and the whole top set come from the **exploratory round 2**, which was built after the "
  "Stage 3 and Stage 4 results were seen. The dev Δ is therefore conditioned on dev, and the one-shot test (PLAN §7.6) "
  "decides whether it holds.")
A("  - A confirmatory reading supports an improvement too. Restricted to the pre-registered round 1, the same rule "
  "chooses %s, with Δ = %+.2f %% %s against the baseline."
  % (short(d["sensitivity"]["round1_only"]["chosen"]), d["sensitivity"]["round1_only"]["chosen_vs_baseline_pct"],
     ci(d["sensitivity"]["round1_only"]["chosen_vs_baseline_ci95_pct"])))
A("- **Test candidates, fixed now and before anything touches test** (sha256 of the id list `%s…`):"
  % d["test_candidates_fixed"]["sha256_of_ids"][:12])
A("")
A("| # | tokenizer | role |")
A("|---|---|---|")
for i, t in enumerate(d["test_candidates"], 1):
    A("| %d | %s | %s%s |" % (i, short(t["id"]), t["role"], ("; " + t["why"]) if t.get("why") else ""))
A("")

# ------------------------------------------------------------------ 1
A("## 1. Inputs and checks")
A("")
inp = d["inputs"]
A("- **Results:** `lm/colab_results/results_table.json` (sha256 `%s…`), with the per-run JSON records it lists. Each "
  "record's sha256 was re-verified, and each record's bpb was recomputed bit-exactly as Σbits/Σbytes."
  % inp["results_table_sha256"][:12])
A("  - Records used: %d Stage-4 'confirm' records (18 candidates × 5 seeds) and %d Stage-3 'screen' records "
  "(18 × 3, report-only)." % (inp["records_loaded_confirm"], inp["records_loaded_screen"]))
A("  - Bundles: `1d24425d2d64` (round 1) and `77e1368773fc` (round 2).")
A("- **Reference arms in both bundles are counted once.** The %d round-2 copies (4 arms × (5 confirm + 3 screen) seeds) "
  "were compared with the round-1 runs. Their per-document bits, uids and bytes are identical: %s. The copies were "
  "dropped." % (inp["duplicates_dropped"], inp["duplicates_bitwise_identical"]))
cl = d["clusters"]
A("- **Clusters (PLAN §6)** come from `splits/split_manifest.jsonl`, whose sha256 `%s…` matches the frozen hash." % inp["split_manifest_sha256"][:8])
A("  - Only the 836 dev_strict uids were looked up. Every other row was skipped on its uid alone.")
A("  - Cluster = the manifest `group`, with the per-record web groups (`web:<site>:record:<uid>`) collapsed to "
  "`web:<site>`.")
A("  - Result: **%d clusters** (%s). The raw manifest has %d groups (%s). The largest cluster is `%s`, with %.1f %% of "
  "dev bytes. The mapping agrees with `lm/colab_results/dev_clusters.csv`: %s."
  % (cl["n_clusters"], ", ".join("%s %d" % kv for kv in sorted(cl["clusters_by_source"].items())), cl["n_raw_groups"],
     ", ".join("%s %d" % kv for kv in sorted(cl["raw_groups_by_source"].items())), cl["largest_cluster"]["cluster"],
     100 * cl["largest_cluster"]["share"], cl["agrees_with_dev_clusters_csv"]))
A("- **Gates (PLAN §7.1):** G1–G5 pass for all 18 candidates, after the pre-declared G2 remedy (read from each "
  "`summary.json`; G5 is n/a for byte-level vocabularies). The tokenizer sha256 in every summary equals the one the LM "
  "runs used. Gate failures: %s." % (d["gate_failures"] or "none"))
A("")

# ------------------------------------------------------------------ 2
A("## 2. Method, as run")
A("")
pr = d["protocol"]
A("- **Hierarchical cluster bootstrap:** %s replicates, `%s`. In each replicate:" % (format(pr["n_rep"], ","), pr["rng"]))
A("  - the clusters are resampled with replacement *within* each source (book 12, newspaper 15, web 5), keeping all "
  "their documents;")
A("  - each candidate's 5 seeds are resampled with replacement, independently for each candidate;")
A("  - bpb = Σbits ÷ Σbytes over the resampled documents, averaged over the resampled seeds.")
A("  - All candidates share the replicate's cluster draw, so every Δ is paired over documents.")
A("  - RNG order: the cluster draws come first (sources in sorted order), then the seed draws (candidates in sorted id "
  "order).")
A("  - An independent per-replicate loop implementation with another RNG stream reproduced the CIs to within ±0.01 pp.")
A("- **Outputs:**")
A("  - Δ% is computed per replicate as (bpb_A − bpb_B)/bpb_B, with the 95 % percentile CI.")
A("  - p = 2·min(P(Δ*≤0), P(Δ*≥0)). Its resolution is 1e-4, and p = 0 is shown as <1e-4.")
A("- **Holm–Bonferroni** covers the %d comparisons of every ranked candidate against the best-mean candidate, at "
  "α = 0.05. 'Significantly worse' means a Holm-adjusted p ≤ 0.05 with a positive Δ." % (k - 1))
A("- **TOST:** a candidate is equivalent if the 90 %% CI of Δ lies inside ±δ, with δ = 0.3 %% × baseline mean bpb = "
  "%.6f bpb." % delta_abs)
A("- **Top set** = the best, plus every candidate not significantly worse, plus every candidate equivalent within δ.")
A("- **Tie-breakers** are lexicographic, in PLAN order, without tolerance. Each tokenizer's metrics come from its "
  "`summary.json` (harness, dev_strict), plus `meta.json` for the SentencePiece HF-export check.")
A("  - (a) A stock HF `tokenizer.json` reproduces the ids the LM used on 100 % of dev (strict and permissive).")
A("  - (b) Dev bytes/token.")
A("  - (c) Learned tokens with train_D1 frequency < 20. train_D1 is the Stage 4 training stream.")
A("  - (d) The mean over the four PLAN §4.2 perturbations of the share of affected words that get re-segmented.")
A("  - (e) The effective vocabulary.")
A("- **Improvement claim:** the chosen tokenizer's Δ against %s must have a 95 %% CI that excludes 0. This is the "
  "single pre-registered comparison, and it is not Holm-corrected." % short(BASE))
A("")

# ------------------------------------------------------------------ 3
A("## 3. Ranking (Stage 4 'confirm', 5 seeds each)")
A("")
A("Δ%% is relative to the reference's bpb, with the 95 %% bootstrap CI in brackets. The TOST column shows the 90 %% CI "
  "of Δ against the best, in %% of the best's bpb. δ = ±0.300 %% of the baseline, which is ±%.3f %% of the best's bpb."
  % (100 * delta_abs / best_bpb))
A("")
A("| # | candidate | origin | V | mean bpb | seed s.d. | Δ vs baseline | Δ vs best | p vs best | Holm p | 90 % CI vs best | top set |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
orig = {"round1_preregistered_rank": "R1 pre-reg", "round1_added_reference_arm": "R1 added arm",
        "round2_exploratory": "R2 exploratory"}
for r in d["ranking"]:
    rank = str(r["rank_primary"]) if r["in_primary_family"] else "–"
    vb = "baseline" if r["id"] == BASE else dl(r["vs_baseline"])
    vbe = "best" if r["id"] == BEST else dl(r["vs_best"])
    pb = "–" if r["id"] == BEST else p_(r["vs_best"]["p"])
    hp = "–" if r["holm_p_vs_best"] is None else p_(r["holm_p_vs_best"])
    t90 = "–" if r["id"] == BEST else ci90pct(r["vs_best"], best_bpb)
    name = short(r["id"]) + ("" if r["in_primary_family"] else " (conditional rank 8, not ranked)")
    A("| %s | %s | %s | %dk | %.5f | %.2f %% | %s | %s | %s | %s | %s | %s |"
      % (rank, name, orig[r["origin"]] + " (" + str(r["wave_rank_label"]) + ")", r["vocab"] // 1024, r["mean_bpb"],
         r["seed_sd_rel_pct"], vb, vbe, pb, hp, t90, "**yes**" if r["in_top_set"] else "no"))
A("")
A("Notes:")
A("- 'Δ vs baseline' is descriptive for every row except the chosen one. PLAN pre-registers only the chosen-vs-baseline "
  "comparison, and it corrects only the family against the best.")
_rej = [r for r in ranked_primary if r["id"] != BEST and r["significantly_worse_than_best"]]
_acc = [r for r in ranked_primary if r["id"] != BEST and not r["significantly_worse_than_best"]]
_last_rej = max(_rej, key=lambda r: r["vs_best"]["p"])
_first_acc = min(_acc, key=lambda r: r["vs_best"]["p"])
A("- The Holm step-down rejects everything down to %s (raw p %s, adjusted %s). It stops at %s (raw p %s, adjusted %s)."
  % (short(_last_rej["id"]), p_(_last_rej["vs_best"]["p"]), p_(_last_rej["holm_p_vs_best"]),
     short(_first_acc["id"]), p_(_first_acc["vs_best"]["p"]), p_(_first_acc["holm_p_vs_best"])))
A("- The per-seed bpb of every run is in `decision.json` → `ranking[].seed_bpb`.")
A("")

# ------------------------------------------------------------------ 4
A("## 4. Top set and tie-breakers")
A("")
A("| candidate | Δ vs best | (a) HF-native exact | (b) dev bytes/token | (c) train_D1 freq < 20 | (d) robustness | (e) vocab (effective) | order |")
A("|---|---|---|---|---|---|---|---|")
for i, c in enumerate(DEC["tiebreak_order"], 1):
    t = TB[c]
    A("| %s | %s | %s | %.4f | %d (%.1f %%) | %.3f | %d | %d |"
      % (short(c), "best" if c == BEST else dl(RK[c]["vs_best"]), "yes" if t["hf_native_exact"] else "**no**",
         t["dev_bytes_per_token"], t["train_lt20"], t["train_lt20_pct"], t["robustness_seg_change_affected_mean"],
         t["vocab_effective"], i))
A("")
A("Evidence for (a):")
for c in DEC["tiebreak_order"]:
    A("- %s: %s." % (short(c), TB[c]["hf_native_evidence"]))
A("")
A("How the decision falls out:")
A("- (a) removes %s." % ", ".join(short(c) for c in tr["a"]["dropped"]))
A("- (b) separates the two MinGram builds.")
A("- (c)–(e) were never reached.")
A("- (c) and (e) would favour the 32k MinGram (%d vs %d tokens below 20; 32k vs 48k), but they come after (b)."
  % (TB["R2-A10-MinGram-P1r3-D2-32k"]["train_lt20"], TB[CH]["train_lt20"]))
A("- If (a) were read on dev_strict only, %s would count as HF-native: its export matches the native ids on 836/836 "
  "dev_strict documents. (b) would still choose %s, since 7.2237 > 7.1258 > 7.0764 bytes/token."
  % (short("R2-A4-SPnat-D2-32k"), short(CH)))
A("")
A("Cost of the choice, report-only:")
_m32 = "R2-A10-MinGram-P1r3-D2-32k"
assert RK[CH]["params_non_embedding"] == RK[_m32]["params_non_embedding"] == RK[BASE]["params_non_embedding"]
A("- The chosen 48k tokenizer has %.2fM total parameters in the arbiter, against %.2fM at 32k and %.2fM for the "
  "16k baseline. The non-embedding parameters are the same %.2fM in all three."
  % (RK[CH]["params_total"] / 1e6, RK[_m32]["params_total"] / 1e6, RK[BASE]["params_total"] / 1e6,
     RK[CH]["params_non_embedding"] / 1e6))
A("- %d of its %d learned tokens (%.1f %%) are seen fewer than 20 times in train_D1. That is the data-scarce regime of "
  "PLAN §4.3, and R3 (embedding norms) is where it would show."
  % (TB[CH]["train_lt20"], TB[CH]["learned_tokens"], TB[CH]["train_lt20_pct"]))
A("")

# ------------------------------------------------------------------ 5
A("## 5. Improvement claim (PLAN §7.5)")
A("")
A("- %s vs %s: **Δ = %+.3f %%, 95 %% CI %s, p %s** (bpb %.5f vs %.5f; bootstrap s.e. %.3f pp)."
  % (short(CH), short(BASE), vsb["delta_pct"], ci(vsb["ci95_pct"], 3), p_(vsb["p"]), RK[CH]["mean_bpb"], base_bpb,
     vsb["se_pct"]))
A("- The CI excludes 0, so the rule allows the claim. On dev its wording is: *%s*." % DEC["improvement_claim_text"])
A("- Per source (report-only; the same replicates, restricted to one source's clusters):")
A("")
A("| source | clusters | bytes | " + " | ".join(short(c) + " vs baseline" for c in top) + " |")
A("|---|---|---|" + "---|" * len(top))
for src, e in d["per_source"].items():
    A("| %s | %d | %s | %s |" % (src, e["n_clusters"], format(e["bytes"], ","),
                                 " | ".join(dl(e["vs_baseline"][c]) for c in top)))
A("")
assert all(e["vs_baseline"][c]["ci95_pct"][1] < 0 for e in d["per_source"].values() for c in top)
A("No source regresses: every per-source CI lies below 0. The web stratum has only 5 clusters and 34 KB, so its CIs "
  "are wide.")
A("")

# ------------------------------------------------------------------ 6
A("## 6. Test candidates (fixed before test; PLAN §7.6)")
A("")
A("- **The list:** %s." % ", ".join(short(t["id"]) for t in d["test_candidates"]))
A("  - It was frozen in `decision.json` → `test_candidates_fixed` at %s. The sha256 of the newline-joined ids is `%s`."
  % (d["test_candidates_fixed"]["utc"], d["test_candidates_fixed"]["sha256_of_ids"]))
A("  - It is the chosen tokenizer, the baseline, and the two other top-set members, labelled report-only. The two "
  "other members are the whole rest of the top set.")
A("- **Why these report-only members:**")
A("  - %s is the best-mean candidate. It lost only on tie-breaker (a)." % short(BEST))
A("  - %s is the remaining top-set member." % short("R2-A10-MinGram-P1r3-D2-32k"))
A("  - The best pre-registered round-1 tokenizer (%s) is **not** in the top set: it is %s against the best. It "
  "therefore cannot be a report-only test candidate under this rule. The baseline comparison is the forking-paths "
  "guard on test." % (short("A1-P1r3-D2-32k"), dlp(RK["A1-P1r3-D2-32k"]["vs_best"])))
A("- **What the TestBundle phase must do:**")
A("  - Run G1 on test for the four tokenizers.")
A("  - Score each of them once on strict test with the Stage 4 recipe: d=192, full permissive train, 1 epoch, LR 1e-3, "
  "seeds 1–5, GPU fp16, the same `hk_lm`. Re-use the Stage 4 models if the bundle kept them; otherwise retrain them. "
  "The runs were bitwise reproducible across sessions (COLLECT.md §5).")
A("  - Report the chosen tokenizer's Δ against the baseline with the same bootstrap. The test split has 27 clusters "
  "(15 newspaper, 6 book, 6 web).")
A("  - The report-only rows are never used to re-select. If test contradicts dev in sign, report it and do not "
  "re-select (PLAN §7.6).")
A("")

# ------------------------------------------------------------------ 7
A("## 7. Report-only analyses (same bootstrap; none of this enters the decision)")
A("")
A("### 7.1 Vocabulary-size curve (A1 byte-level BPE, P1r3, D2)")
A("")
vc = C["vocab_curve_A1_P1r3_D2"]
sizes = ["8k", "16k", "32k", "48k"]
A("| size | tokenizer | total params | ctx tokens | Stage 3 bpb | Stage 4 bpb | Stage 4 Δ vs previous size | Stage 3 Δ vs previous size | Stage 4 Δ vs 16k |")
A("|---|---|---|---|---|---|---|---|---|")
for i, s in enumerate(sizes):
    c = vc["sizes"][s]
    st4 = "–" if i == 0 else dlp(vc["steps"][i - 1])
    st3 = "–" if i == 0 else dl(vc["screen_steps"][i - 1])
    v16 = {"8k": dl(vc["vs_16k"][0]), "16k": "–", "32k": dl(vc["vs_16k"][1]), "48k": dl(vc["vs_16k"][2])}[s]
    A("| %s | %s | %.2fM | %d | %.5f | %.5f | %s | %s | %s |"
      % (s, short(c), RK[c]["params_total"] / 1e6, RK[c]["ctx_tokens"], vc["screen_mean_bpb"][s], vc["confirm_mean_bpb"][s],
         st4, st3, v16))
A("")
o = vc["other_families_32k_to_48k"]
A("- **The curve flattens after 32k.**")
assert vc["steps"][0]["ci95_excludes_0"] and vc["steps"][1]["ci95_excludes_0"] and not vc["steps"][2]["ci95_excludes_0"]
A("  - 8k→16k gains %s and 16k→32k gains %s at Stage 4. Both steps are significant."
  % (dl(vc["steps"][0]), dl(vc["steps"][1])))
A("  - 32k→48k gains %s, which is not significant." % dlp(vc["steps"][2]))
A("  - The same holds in the other families: MinGram 32k→48k gives %s, and SentencePiece Unigram 32k→48k gives %s, "
  "which is *worse*." % (dlp(o[0]), dlp(o[1])))
A("- Stage 3 exaggerates the small-vocabulary penalty: 8k→16k is %s there." % dl(vc["screen_steps"][0]))
A("- Total parameters grow with the vocabulary, because PLAN §5 fixes non-embedding size, not total size. The curve "
  "is therefore not iso-parameter.")
A("")
A("### 7.2 Data mix D2 vs D1, and pre-tokenizer P1r3 vs P1 (A1 BPE, 16k)")
A("")
m = C["data_mix_and_pretokenizer"]
A("| contrast | Stage 4 | Stage 3 |")
A("|---|---|---|")
A("| D2 vs D1 (`A1-P1r3-D2-16k` − `A1-P1r3-D1-16k`) | %s | %s |" % (dlp(m["D2_vs_D1_A1_P1r3_16k"]), dlp(m["screen_D2_vs_D1_A1_P1r3_16k"])))
A("| P1r3 vs P1 (`A1-P1r3-D1-16k` − `A1-P1-D1-16k`) | %s | %s |" % (dlp(m["P1r3_vs_P1_A1_D1_16k"]), dlp(m["screen_P1r3_vs_P1_A1_D1_16k"])))
A("| Stage-1 recipe P1r3+D2 vs P1+D1 (`A1-P1r3-D2-16k` − `A1-P1-D1-16k`) | %s | – |" % dlp(m["stage1_recipe_P1r3D2_vs_P1D1_A1_16k"]))
rc = m["recipe_change_confounded"]
A("| SuperBPE 32k t/T 0.8: P1r3+D2 vs P1+D1 (`R2-A6…` − `A6-…-32k-t080`) | %s | – |" % dlp(rc["SuperBPE_32k_P1D1_to_P1r3D2"]))
A("| *confounded with size:* MinGram P1+D1 16k → P1r3+D2 32k | %s | – |" % dlp(rc["MinGram_16k_P1D1_to_32k_P1r3D2"]))
A("| *confounded with size:* SP-Unigram D1 16k → D2 32k | %s | – |" % dlp(rc["A4_16k_D1_to_32k_D2"]))
A("")
A("- **D2 (strict-only train text) beats D1 by about 0.7 % bpb at 16k in the LM.** This is the only clean data-mix "
  "contrast with LM runs, and it supports the Stage 1 choice of D2, which was fragile on bytes/token (WAVES pending "
  "decision 1).")
A("- Recall that D2 made SentencePiece Unigram 0.9–1.1 % *less* compressive than D1 at 32k–48k (ROUND2.md). There is "
  "no LM run of A4 on D1 at 32k, so the data-mix effect for Unigram at 32k is not measured.")
A("- P1r3 (3-digit groups) is neutral in bpb at Stage 4. It is kept for numeracy, as PLAN §3 intends.")
A("")
A("### 7.3 Algorithm effect at equal settings")
A("")
g = C["algorithm_at_equal_settings"]
A("| setting | contrast | Stage 4 Δ, 95 % CI, p |")
A("|---|---|---|")
for x in g["P1_D1_16k_vs_A1_BPE"]:
    A("| P1, D1, 16k (same pre-tokenizer and data) | %s − `A1-P1-D1-16k` | %s |" % (short(x["a"]), dlp(x)))
x = g["SPnat_D1_16k_Unigram_vs_SPBPE"]
A("| SentencePiece native, D1, 16k | %s − %s (Unigram vs SP-BPE) | %s |" % (short(x["a"]), short(x["b"]), dlp(x)))
for x in g["D1_16k_SP_vs_HF_BPE_P1"]:
    A("| D1, 16k (pre-tokenizer differs) | %s − %s | %s |" % (short(x["a"]), short(x["b"]), dlp(x)))
for x in g["P1r3_D2_32k_vs_A1_BPE"]:
    A("| P1r3 (A4: SP-native), D2, 32k | %s − %s | %s |" % (short(x["a"]), short(x["b"]), dlp(x)))
for x in g["P1r3_D2_48k_vs_A1_BPE"]:
    A("| P1r3 (A4: SP-native), D2, 48k | %s − %s | %s |" % (short(x["a"]), short(x["b"]), dlp(x)))
for x in g["MinGram_vs_Unigram_A4"]:
    A("| MinGram vs SentencePiece Unigram | %s − %s | %s |" % (short(x["a"]), short(x["b"]), dlp(x)))
A("")
assert all(x["ci95_pct"][1] < 0 for x in (g["P1_D1_16k_vs_A1_BPE"][0], g["P1r3_D2_32k_vs_A1_BPE"][0],
                                            g["P1r3_D2_48k_vs_A1_BPE"][0]))
A("- **MinGram beats byte-level BPE at equal settings, at every size tested:** %s at P1-D1-16k, %s at P1r3-D2-32k, "
  "and %s at 48k."
  % (dl(g["P1_D1_16k_vs_A1_BPE"][0]), dl(g["P1r3_D2_32k_vs_A1_BPE"][0]), dl(g["P1r3_D2_48k_vs_A1_BPE"][0])))
A("- **SentencePiece Unigram beats it at 16k and 32k:** %s at D1-16k and %s at D2-32k. There its pre-tokenization is "
  "SentencePiece's own. At 48k the difference is not significant (%s)."
  % (dl(g["D1_16k_SP_vs_HF_BPE_P1"][0]), dl(g["P1r3_D2_32k_vs_A1_BPE"][1]), dl(g["P1r3_D2_48k_vs_A1_BPE"][1])))
A("- The Unigram-family advantage matches Land (2026), not Yavuz et al. (2026) (SOTA §1.1).")
A("- **MinGram and SentencePiece Unigram are not distinguishable** at any size: %s."
  % "; ".join(dl(x) for x in g["MinGram_vs_Unigram_A4"]))
pk = g["P1_D1_16k_vs_A1_BPE"][1]
A("- **PickyBPE vs BPE:** %s, and %s. Removing the under-trained intermediate tokens did not measurably change bpb "
  "(PLAN §4.3 hypothesis)." % (dl(pk), "TOST-equivalent within ±0.3 %" if pk["tost_equivalent"] else "not TOST-equivalent"))
A("- **SentencePiece-BPE (A3) is not better than HF BPE** (%s, with seed s.d. %.2f %%). It is clearly worse than "
  "SentencePiece Unigram (%s)." % (dl(g["D1_16k_SP_vs_HF_BPE_P1"][1]), RK["A3-SPnat-D1-16k"]["seed_sd_rel_pct"],
                                  dl(g["SPnat_D1_16k_Unigram_vs_SPBPE"])))
A("")
A("### 7.4 SuperBPE")
A("")
sb = C["superbpe"]
A("| contrast | Stage 3 Δ | Stage 4 Δ | Stage 4 subset curve Δ at 25 / 50 / 75 / 100 % of steps | Stage 3 subset curve Δ at 25 / 50 / 75 / 100 % |")
A("|---|---|---|---|---|")
for x in sb["pairs"]:
    cc = x["confirm_subset_curve_delta_pct"]
    sc = x["screen_subset_curve_delta_pct"]
    A("| %s − %s | %s | %s | %s | %s |" % (short(x["a"]), short(x["b"]), dlp(x["screen"]), dlp(x["confirm"]),
                                         " / ".join("%+.2f" % cc[f] for f in ("0.25", "0.5", "0.75", "1.0")),
                                         " / ".join("%+.2f" % sc[f] for f in ("0.25", "0.5", "0.75", "1.0"))))
A("")
A("| SuperBPE tokenizer | Stage 3 rank (of 18) | Stage 4 rank (of 18) | ctx tokens | train_D1 bytes/token | dev bytes/token |")
A("|---|---|---|---|---|---|")
for c, x in sorted(sb["ranks"].items(), key=lambda kv: kv[1]["screen_rank"]):
    A("| %s | %d | %d | %d | %.3f | %.3f |" % (short(c), x["screen_rank"], x["confirm_rank"], x["ctx_tokens"],
                                            x["train_bytes_per_token"], x["dev_bytes_per_token"]))
A("")
r8 = sb["rank8"]
A("- **The SuperBPE results:**")
A("  - At Stage 4, every SuperBPE tokenizer is significantly worse than its own stage-1 BPE base: +1.6 % (16k, "
  "t/T 0.9), +2.6 % (16k, 0.8) and +3.1 % (32k, 0.8, P1r3+D2).")
A("  - More superwords is worse: t/T 0.9 beats 0.8 by %s." % dl(sb["pairs"][4]["confirm"]))
A("  - The largest documented gain of the literature (PLAN §2.2 rank 2) does not appear with these small, byte-matched "
  "arbiters.")
A("- **The Stage 3 picture was different:**")
A("  - The P1-D1 builds were level with their base: +0.39 % and +0.42 %, both not significant.")
A("  - The round-2 build was *better* than A1-P1r3-D2-32k (%s)." % dlp(sb["pairs"][3]["screen"]))
A("  - The two 32k SuperBPE builds ranked 2nd and 6th of 18.")
A("  - In the Stage 3 subset curves, SuperBPE leads early (−0.9 to −2.7 % at 25 % of steps) and loses most of its "
  "lead by 100 %.")
A("  - At Stage 4 it is already behind at 25 % of the steps, and falls further behind to the end.")
fewer = [100 * (1 - RK[x["b"]]["train_bytes_per_token"] / RK[x["a"]]["train_bytes_per_token"]) for x in sb["pairs"][:4]]
A("- **Hypothesis (not tested):** with byte-matched budgets, SuperBPE gives the model %.0f–%.0f %% fewer prediction "
  "targets per training byte than its BPE comparator (train_D1 bytes/token). Its multi-word tokens also need more data "
  "per embedding. A superword vocabulary helps a small model that is far from convergence, and the larger, "
  "longer-trained Stage 4 arbiter gains more from the finer segmentation." % (min(fewer), max(fewer)))
A("- **Conditional rank 8 (%s)** is not admitted to the PLAN §7 ranking:" % short(r8["candidate"]))
A("  - At Stage 3, neither rank 2 (%.5f) nor rank 5 (%.5f) beat rank 1 (%.5f)."
  % (r8["stage3_mean_bpb"]["A6-SBPE-P1-D1-16k-t090"], r8["stage3_mean_bpb"]["A6-SBPE-P1-D1-16k-t080"],
     r8["stage3_mean_bpb"][BASE]))
A("  - Rank 2 beat rank 5, so the valid rank-8 build would have been t/T = 0.9, not the 0.8 file that ran.")
A("  - Its Stage 4 result (%.5f, 17th of 18) is reported above." % r8["stage4_mean_bpb"][r8["candidate"]])
A("")
A("### 7.5 Stage 3 vs Stage 4: ranking reversals")
A("")
A("| candidate | Stage 3 rank | Stage 4 rank | Stage 3 Δ vs baseline | Stage 4 Δ vs baseline |")
A("|---|---|---|---|---|")
for c in sorted(S34["screen_rank"], key=lambda c: S34["screen_rank"][c]):
    sv = S34["screen_vs_baseline"].get(c)
    cv = S34["confirm_vs_baseline"].get(c)
    A("| %s | %d | %d | %s | %s |" % (short(c), S34["screen_rank"][c], S34["confirm_rank"][c],
                                     "baseline" if sv is None else dl(sv), "baseline" if cv is None else dl(cv)))
A("")
rc1, rc2 = S34["rank_correlation_all18"], S34["rank_correlation_without_superbpe"]
sig = [r for r in S34["reversals"] if r["significant_in_both"]]
sig_sbpe = [r for r in sig if "SBPE" in r["a"] or "SBPE" in r["b"]]
sig_other = [r for r in sig if r not in sig_sbpe]
A("- **Rank agreement between the two stages:**")
A("  - All 18 candidates: Kendall τ_b = %.2f and Spearman ρ = %.2f." % (rc1["kendall_tau_b"], rc1["spearman_rho"]))
A("  - Without the 4 SuperBPE tokenizers: τ_b = %.2f and ρ = %.2f." % (rc2["kendall_tau_b"], rc2["spearman_rho"]))
A("- **Reversals:** %d of %d candidate pairs change order between the stages. %d of them are significant in both "
  "stages (the 95 %% CIs exclude 0 in opposite directions), and %d of those involve a SuperBPE tokenizer."
  % (S34["n_reversals"], S34["n_pairs"], len(sig), len(sig_sbpe)))
A("  - The largest SuperBPE reversal: %s vs %s goes from %s at Stage 3 to %s at Stage 4."
  % (short(sig_sbpe[0]["a"]), short(sig_sbpe[0]["b"]), "%+.2f %% %s" % (sig_sbpe[0]["screen_delta_pct"], ci(sig_sbpe[0]["screen_ci95_pct"])),
     "%+.2f %% %s" % (sig_sbpe[0]["confirm_delta_pct"], ci(sig_sbpe[0]["confirm_ci95_pct"]))))
_c = "R2-A4-SPnat-D2-32k"
assert len(sig_other) >= 1 and all(_c in (r["a"], r["b"]) for r in sig_other)
A("  - The significant reversals that do not involve SuperBPE: %s. All of them concern %s, which rises from %dth at "
  "Stage 3 to %dst at Stage 4."
  % ("; ".join("%s vs %s (%+.2f %% → %+.2f %%)" % (short(r["a"]), short(r["b"]), r["screen_delta_pct"], r["confirm_delta_pct"])
               for r in sig_other), short(_c), S34["screen_rank"][_c], S34["confirm_rank"][_c]))
A("- **Other large moves:**")
A("  - `A1-P1r3-D2-8k` goes from 18th to 13th, because its penalty shrinks from +2.8 % to +0.9 %.")
A("  - `A6-SBPE-P1-D1-32k-t080` goes from 6th to 17th, and `R2-A6-SBPE-P1r3-D2-32k-t080` from 2nd to 15th.")
A("- **Consequence:** the 10 MB, d=128 screening stage is a poor proxy for SuperBPE, and for the size of the "
  "small-vocabulary penalty. The decision uses Stage 4, as PLAN §7.2 prescribes when Stage 4 exists.")
A("")
A("### 7.6 Power check (PLAN §6)")
A("")
pw = PW
A("- **The PLAN's literal check** (after the baseline's first wave, Stage 3): the baseline's seed s.d. is %.3f %%, so "
  "n = %d, and 3 seeds suffice." % (pw["stage3_baseline_first_wave"]["rel_sd_pct"], pw["stage3_baseline_first_wave"]["n_needed"]))
A("- **Stage 4, seeds 1–3** (the collector's rule):")
A("  - The pooled s.d. is %.3f %% in round 1 (n = %d) and %.3f %% in round 2 (n = %d)."
  % (pw["stage4_seeds1to3_pooled"]["round1_bundle"]["pooled_rel_sd_pct"], pw["stage4_seeds1to3_pooled"]["round1_bundle"]["n_needed"],
     pw["stage4_seeds1to3_pooled"]["round2_bundle"]["pooled_rel_sd_pct"], pw["stage4_seeds1to3_pooled"]["round2_bundle"]["n_needed"]))
A("  - Both are > 0.2 %, so seeds 4 and 5 were run for every candidate.")
A("- **Stage 4, all 5 seeds:**")
A("  - The pooled s.d. is %.3f %% over all 18 candidates, which gives n = %d."
  % (pw["stage4_5seeds_pooled"]["all18"]["pooled_rel_sd_pct"], pw["stage4_5seeds_pooled"]["all18"]["n_needed"]))
A("  - Two arms drive it: %s. Without them the pooled s.d. is %.3f %% and n = %d."
  % (", ".join("%s %.2f %%" % (c, pw["per_candidate"][c]["rel_sd_5seeds_pct"]) for c in pw["stage4_5seeds_pooled"]["without_high_variance"]["excluded"]),
     pw["stage4_5seeds_pooled"]["without_high_variance"]["pooled_rel_sd_pct"], pw["stage4_5seeds_pooled"]["without_high_variance"]["n_needed"]))
A("  - For each top-set member, the seed s.d. and the n its own s.d. requires: %s."
  % "; ".join("%s %.2f %% (n = %d)" % (short(c), pw["per_candidate"][c]["rel_sd_5seeds_pct"], pw["per_candidate"][c]["n_needed_own_sd"]) for c in top))
A("- **Achieved resolution** (80 %-power detectable effect, from the bootstrap s.d. of Δ, which includes cluster and "
  "seed noise):")
A("  - chosen vs baseline: %.2f %%;" % pw["achieved_mde80_pct_chosen_vs_baseline"])
A("  - median over the candidates vs baseline: %.2f %%;" % pw["achieved_mde80_pct_vs_baseline_median"])
A("  - the top-set members vs the best: %s."
  % ", ".join("%s %.2f %%" % (short(c), pw["achieved_mde80_pct_vs_best"][c]) for c in top if c != BEST))
A("- **Reading:** the chosen tokenizer's gain over the baseline (%.2f %%) is about %.0f× the detectable effect."
  % (abs(vsb["delta_pct"]), abs(vsb["delta_pct"]) / pw["achieved_mde80_pct_chosen_vs_baseline"]))
A("- The gaps inside the top set (%s vs the best) are below it. They are **not distinguishable at our power**, never "
  "'equal'. TOST did not establish equivalence either."
  % ", ".join("%+.2f %%" % RK[c]["vs_best"]["delta_pct"] for c in top if c != BEST))
A("")
A("### 7.7 Learning rate")
A("")
A("| tokenizer | 1e-3 | 3e-3 | 6e-3 |")
A("|---|---|---|---|")
A("| %s (Stage 3 sweep, seed 1) | %.5f | %.5f | %.5f |" % ((short(BASE),) + tuple(e["bpb"] for e in sorted(d["lr"]["stage3_sweep_baseline"], key=lambda e: e["lr"]))))
for c, v in d["lr"]["finalist_resweep"].items():
    A("| %s (Stage 4 re-sweep, seed 1) | %.5f | %.5f | %.5f |" % (short(c), v["1e-3 (confirm s1)"], v["3e-03"], v["6e-03"]))
A("")
A("- 1e-3 wins everywhere, including for the chosen tokenizer and the best-mean candidate.")
A("- No LR below 1e-3 was tried (§8, deviation 5).")
A("")
A("### 7.8 Sensitivity analyses")
A("")
A("| analysis | k | best mean | top set | chosen | chosen vs baseline |")
A("|---|---|---|---|---|---|")
A("| **primary** (PLAN §6–§7 as run) | %d | %s | %s | **%s** | %+.2f %% %s |"
  % (k, short(BEST), ", ".join(short(c) for c in top), short(CH), vsb["delta_pct"], ci(vsb["ci95_pct"])))
for key, s in d["sensitivity"].items():
    A("| %s: %s | %d | %s | %s | %s | %+.2f %% %s |"
      % (key, s["what"], s["k"], short(s["best"]), ", ".join(short(c) for c in s["top_set"]), short(s["chosen"]),
         s["chosen_vs_baseline_pct"], ci(s["chosen_vs_baseline_ci95_pct"])))
A("")
assert all(s["chosen"] == CH for key, s in d["sensitivity"].items() if key != "round1_only")
assert all(s["improvement_claim"] for s in d["sensitivity"].values())
A("- **The choice is the same under every sensitivity analysis except the confirmatory 'round 1 only' family.** That "
  "family excludes the exploratory candidates by construction. It chooses `A1-P1r3-D2-32k`, whose improvement claim "
  "also holds.")
A("- The improvement claim holds in every row.")
A("")

# ------------------------------------------------------------------ 8
A("## 8. Deviations from PLAN")
A("")
A("All deviations are additive: none changes a pre-registered metric, split, model or decision rule. Items 1–5 were "
  "known before this analysis; the others were found or decided in it.")
A("")
for x in d["deviations"]:
    A("%d. **%s** (%s)" % (x["id"], x["title"], x["source"]))
    A("   - What: %s" % x["text"])
    A("   - Consequence: %s" % x["impact"])
A("")

# ------------------------------------------------------------------ 9
A("## 9. What can be claimed now, and what waits for the test")
A("")
A("**Now (dev_strict, Stage 4 arbiter):**")
A("")
A("- Among the %d tokenizers ranked under PLAN §7 (18 trained, with the conditional rank 8 excluded), the lowest "
  "mean bpb is %s's." % (k, short(BEST)))
A("- %s is chosen from the top set by the pre-registered tie-breakers. It beats the standard recipe %s by %+.2f %% "
  "bpb, 95 %% CI %s (cluster bootstrap, 5 seeds each)." % (short(CH), short(BASE), vsb["delta_pct"], ci(vsb["ci95_pct"])))
A("- The dev evidence also supports these general statements:")
A("  - Unigram-family vocabularies beat BPE at equal settings.")
A("  - 32k beats 16k and 8k, and 48k adds nothing significant.")
A("  - The strict-only train mix D2 beats D1 (at 16k).")
A("  - SuperBPE and PickyBPE do not help these small arbiters.")
A("")
A("**After the one-shot test (PLAN §7.6/§8):**")
A("")
A("- The PLAN §8 sentence becomes claimable only if test agrees in sign. Its template is: *'Among %d tokenizers "
  "trained and compared under one protocol, %s achieved the lowest held-out bits-per-byte … significantly better "
  "than the standard BPE recipe (Δ = x %%, 95 %% CI [a, b], 5 seeds, cluster bootstrap)'*." % (k, short(CH)))
A("  - Strictly, the chosen tokenizer is 2nd in mean dev bpb and 1st after the tie-breakers, so the wording should say "
  "*'chosen by the pre-registered rule'*, not *'lowest bpb'*, unless test puts it first among the test candidates.")
A("- Round 2 was exploratory. Every public statement should name it: *'selected after an exploratory second round; "
  "confirmed on a sealed test split used once'*.")
A("")
A("**Cannot be claimed (PLAN §8, plus this analysis):**")
A("")
A("- Transfer to 1B+ models or to CPT.")
A("- That MinGram-48k, MinGram-32k and SentencePiece-Unigram-32k are 'equal'. They are only not distinguishable at "
  "our power.")
A("- That 48k is better than 32k: the step is not significant.")
A("- Anything about LR below 1e-3.")
A("")

# ------------------------------------------------------------------ 10
A("## 10. Files and reproduction")
A("")
A("| file | content |")
A("|---|---|")
A("| `analysis/decide.py` | the whole computation: input checks, clusters, bootstrap, Holm, TOST, top set, tie-breakers, contrasts, power, sensitivity, deviations |")
A("| `analysis/report.py` | renders this file from `decision.json` |")
A("| `analysis/decision.json` | every number in this file, plus per-seed bpb, the tie-breaker metrics of all 18 tokenizers, the 32 clusters, and `test_candidates_fixed` |")
A("| `analysis/DECISION.md` | this report |")
A("")
A("Reproduce (single process, a few seconds; deterministic): `PYTHONIOENCODING=utf-8 python analysis/decide.py && "
  "PYTHONIOENCODING=utf-8 python analysis/report.py`, from `F:\\Hindko\\_tokenizer`.")
A("")

# ------------------------------------------------------------------ appendix
A("## Appendix A. Tie-breaker metrics of every candidate (dev_strict harness; R1 on train_D1)")
A("")
A("| candidate | encoder | HF-native exact | dev bytes/token | learned tokens | train freq < 20 | freq = 0 | robustness (seg. change, affected) | mean abs. token change | vocab nominal / effective | gates |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for r in d["ranking"]:
    t = r["tiebreak"]
    A("| %s | %s | %s | %.4f | %d | %d (%.1f %%) | %d | %.3f | %.4f | %d / %d | %s |"
      % (short(r["id"]), r["encoder_kind"], "yes" if t["hf_native_exact"] else "no", t["dev_bytes_per_token"],
         t["learned_tokens"], t["train_lt20"], t["train_lt20_pct"], t["train_eq0"], t["robustness_seg_change_affected_mean"],
         t["robustness_abs_rel_token_change_mean"], t["vocab_nominal"], t["vocab_effective"],
         "pass" if t["gates_pass"] else "FAIL"))
A("")

with open(os.path.join(OUT, "DECISION.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L))
print("wrote DECISION.md, %d lines" % len(L))
