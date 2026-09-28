"""Render analysis/TEST_RESULTS.md from analysis/test_results.json (written by analysis/test_analysis.py).

Every qualitative statement in the report is asserted against the numbers here before it is written.
Run: PYTHONIOENCODING=utf-8 python analysis/test_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "test_results.json")
OUT = os.path.join(HERE, "TEST_RESULTS.md")

BASE = "A1-P1r3-D2-16k"
A132 = "A1-P1r3-D2-32k"
A148 = "R2-A1-P1r3-D2-48k"
SP32 = "R2-A4-SPnat-D2-32k"
MG32 = "R2-A10-MinGram-P1r3-D2-32k"
MG48 = "R2-A10-MinGram-P1r3-D2-48k"
TOP = [SP32, MG48, MG32]


def pct(x, nd=2):
    return "%+.*f %%" % (nd, x)


def ci(c, nd=2):
    return "[%+.*f, %+.*f]" % (nd, c[0], nd, c[1])


def pv(p):
    return "<1e-4" if p == 0 else "%.4f" % p


def dci(d, nd=2):
    """'-1.32 % [-1.47, -1.16]'"""
    return "%s %s" % (pct(d["delta_pct"], nd), ci(d["ci95_pct"], nd))


def pk(a, b):
    return "%s - %s" % (a, b)


def main():
    R = json.load(open(IN, encoding="utf-8"))
    C = R["candidates"]
    K = R["contrasts"]
    V = R["validation"]
    VS = V["summary"]
    L = R["large_test"]
    T = R["timing"]
    cl = R["pre_registered_claim"]
    S = {c: C[c]["short"] for c in C}
    L_ = []
    A = L_.append

    def dev_of(cid):
        d = K[cid]["dev"]
        return d["of_record"] or d["recomputed"]

    def dev_str(cid):
        d = K[cid]["dev"]
        if d["of_record"]:
            return dci(d["of_record"])
        return dci(d["recomputed"]) + " (r)"

    # ------------------------------------------------------------------ assertions behind the prose
    a, b, c_ = K["a_prereg"], K["b_released"], K["c_mingram32"]
    assert cl["holds"] and a["test"]["ci95_excludes_0"] and a["dev_vs_test"]["sign_agrees"]
    assert b["test"]["ci95_excludes_0"] and b["test"]["delta_pct"] < 0 and c_["test"]["ci95_excludes_0"]
    order = R["test_order"]
    assert order[:3] == [MG48, MG32, SP32] and order[3:] == [A148, A132, BASE]
    assert not K["o_sp32_vs_mg48"]["test"]["ci95_excludes_0"] and not K["o_sp32_vs_mg32"]["test"]["ci95_excludes_0"]
    assert K["o_sp32_vs_mg48"]["test"]["delta_pct"] > 0 and K["o_sp32_vs_mg32"]["test"]["delta_pct"] > 0
    ts = R["top_set_on_test"]
    assert ts["same_members_as_dev_top_set"]
    for cid in ("d_mg48_vs_r1", "d_sp32_vs_r1", "d_mg32_vs_r1", "d_r1_vs_base", "e_mg48_vs_bpe48"):
        assert K[cid]["test"]["ci95_excludes_0"] and K[cid]["test"]["delta_pct"] < 0, cid
    flips = [cid for cid in K if not K[cid]["dev_vs_test"]["sign_agrees"]]
    assert sorted(flips) == ["o_sp32_vs_mg32", "o_sp32_vs_mg48"]
    sig_change = [cid for cid in K if K[cid]["dev_vs_test"]["change_ci95_excludes_0"]]
    assert all(K[x]["test"]["ci95_excludes_0"] for x in ("a_prereg", "b_released", "c_mingram32"))
    assert VS["n_hard_valid"] == VS["n_records"] == VS["n_valid_all_checks"] and VS["same_model_as_dev_all"]
    assert VS["bpb_bit_exact_all"] and VS["cross_bundle_all_identical"]
    assert VS["collect_colab_validator_identical_on_all_confirm_records"]
    lc = L["candidates"]
    assert L["order_by_mean"] == [SP32, MG32, MG48, BASE] and lc[MG32]["n"] == 1
    l2, l1, lp = L["boot_2seeds_3cands"], L["boot_seed1_4cands"], L["parametric_all_seeds"]["pairs"]
    assert l2[pk(SP32, MG48)]["ci95_excludes_0"] and l2[pk(SP32, BASE)]["ci95_excludes_0"]
    assert not lp[pk(SP32, MG32)]["ci95_excludes_0"]
    assert all(v["sign_agrees"] for v in L["dev_vs_test"].values())
    sc = L["confirm_vs_large_on_test"]
    assert sc[pk(MG48, BASE)]["change_ci95_pp"][0] > 0 and sc[pk(SP32, MG48)]["change_ci95_pp"][1] < 0
    assert T["supp_bundle_built_before_first_test_number"] and T["amendment_1_before_first_test_number"]
    assert T["large_test_run_after_confirm_test_numbers"]
    assert R["inputs"]["amendment_1_unchanged"] and R["inputs"]["decide_py_unchanged_since_decision"]
    assert R["release_equivalence"]["test_strict_encodings_changed"] == 0

    n_con = len(K)
    shrink_ab = a["dev_vs_test"]["change_pp"]
    code = R["code"]

    # ------------------------------------------------------------------ header
    A("# Hindko tokenizer: the one-shot test result (PLAN §7.6)")
    A("")
    A("Generated %s by `analysis/test_analysis.py` (sha256 `%s…`) and `analysis/test_report.py`. The bootstrap is "
      "`analysis/decide.py`'s own code, imported unchanged (sha256 `%s…`, the same as in `decision.json`). "
      "Machine-readable twin: `analysis/test_results.json`." % (
          R["generated_utc"], code["test_analysis.py"][:12],
          code["decide.py (imported unchanged: Boot, compare, holm, load_stage, dev_clusters)"][:12]))
    A("")
    A("- **The strict test split, used once.** 491 documents, 1,451,026 bytes, **27 clusters** (15 newspaper, 6 book, "
      "6 web). These are the only LM numbers on test in this study.")
    re_ = R["release_equivalence"]
    assert re_["hf_equals_canonical_on_all_required_sets"]
    A("- **Nothing is re-selected (PLAN §7.6).** The pre-registered pick stays `%s` (`DECISION.md`). The released "
      "default stays `%s` (`AMENDMENT_1.md`, whose sha256 is still `%s…`); its exact-HF-export condition is met "
      "(`release_build/sp32k/equivalence.json`). Test results are reported whatever they are." % (
          MG48, SP32, R["inputs"]["amendment_1_sha256_now"][:12]))
    A("- **Scale.** Every number is at the **confirm scale** unless it is marked *large*: d=192, L=4, H=4, the full "
      "permissive train split, 1 epoch, LR 1e-3, seeds 1–5. That is the dev Stage 4 recipe that ranked the candidates.")
    A("- **The test models are the dev models.** Every test record's training-loss curve is identical to its dev record's "
      "(%d/%d records, §1). Only the evaluation text differs." % (VS["n_same_model_checked"], VS["n_records"]))
    A("- Δ = (bpb_a − bpb_b)/bpb_b; negative means a is better. Brackets are 95 % bootstrap CIs. p is two-sided, and "
      "<1e-4 means 0 of 10,000 replicates. \"(r)\" marks a dev CI recomputed here because `decision.json` has no entry "
      "in that orientation.")
    A("")

    # ------------------------------------------------------------------ 0 summary
    A("## 0. Summary")
    A("")
    A("| tokenizer | role | test bpb (5 seeds) | seed s.d. | Δ vs baseline, test | p | Δ vs baseline, dev | change dev → test (pp) |")
    A("|---|---|---|---|---|---|---|---|")
    for cc in order:
        e = C[cc]
        if cc == BASE:
            A("| `%s` | %s | %.5f | %.2f %% | – | – | – | – |" % (cc, e["role"], e["test_mean_bpb"], e["test_seed_sd_rel_pct"]))
            continue
        cid = next(k for k, v in K.items() if v["a"] == cc and v["b"] == BASE)
        A("| `%s` | %s | %.5f | %.2f %% | %s | %s | %s | %+.2f %s |" % (
            cc, e["role"], e["test_mean_bpb"], e["test_seed_sd_rel_pct"], dci(K[cid]["test"]), pv(K[cid]["test"]["p"]),
            dev_str(cid), K[cid]["dev_vs_test"]["change_pp"], ci(K[cid]["dev_vs_test"]["change_ci95_pp"])))
    A("")
    A("- **(a) The pre-registered claim holds on test.** `%s` vs `%s`: **Δ = %s, p %s**. Dev: %s. The test CI excludes 0 "
      "and the sign agrees with dev, so the PLAN §7.5–7.6 improvement claim stands. The test gain is smaller than the dev "
      "gain by %.2f pp %s. That fits selection on dev (winner's curse), but every gain over the baseline shrank on test, "
      "round-1 ones included (§9)." % (
          MG48, BASE, dci(a["test"]), pv(a["test"]["p"]), dci(cl["dev_of_record"]), shrink_ab,
          ci(a["dev_vs_test"]["change_ci95_pp"])))
    A("- **(b) The released tokenizer improves on the baseline on test too.** `%s`: Δ = %s, p %s (dev %s). Its gain shrank "
      "the most of the three top-set members: by %.2f pp %s." % (
          SP32, dci(b["test"]), pv(b["test"]["p"]), dev_str("b_released"), b["dev_vs_test"]["change_pp"],
          ci(b["dev_vs_test"]["change_ci95_pp"])))
    A("- **(c)** `%s`: Δ = %s, p %s (dev %s)." % (MG32, dci(c_["test"]), pv(c_["test"]["p"]), dev_str("c_mingram32")))
    A("- **Within the top set, on test, the released tokenizer has the highest bpb of the three**: %s %.5f < %s %.5f < "
      "%s %.5f. SP-32k vs MinGram-48k is %s (p %s), and vs MinGram-32k %s (p %s). Neither CI excludes 0, so the three "
      "stay statistically tied, but SP-32k ranks 3rd of the three in %.1f %% of the replicates. On dev it had the lowest "
      "mean. Applying the PLAN §7.3 top-set rule to the 6 test tokenizers (report-only) gives the same three members." % (
          S[MG48], C[MG48]["test_mean_bpb"], S[MG32], C[MG32]["test_mean_bpb"], S[SP32], C[SP32]["test_mean_bpb"],
          dci(K["o_sp32_vs_mg48"]["test"]), pv(K["o_sp32_vs_mg48"]["test"]["p"]), dci(K["o_sp32_vs_mg32"]["test"]),
          pv(K["o_sp32_vs_mg32"]["test"]["p"]), 100 * ts["rank_distribution"][SP32]["3"]))
    A("- **(d) Forking paths: every top-set member beats the round-1 confirmatory winner `%s` on test.** MinGram-48k %s, "
      "MinGram-32k %s, SP-32k %s. `%s` itself beats the baseline: %s (dev %s)." % (
          A132, dci(K["d_mg48_vs_r1"]["test"]), dci(K["d_mg32_vs_r1"]["test"]), dci(K["d_sp32_vs_r1"]["test"]), A132,
          dci(K["d_r1_vs_base"]["test"]), dev_str("d_r1_vs_base")))
    A("- **(e) At equal vocabulary, and so at equal parameter count, the Unigram-family tokenizers beat BPE on test:** "
      "MinGram-48k vs BPE-48k %s; SP-32k vs BPE-32k %s; MinGram-32k vs BPE-32k %s." % (
          dci(K["e_mg48_vs_bpe48"]["test"]), dci(K["d_sp32_vs_r1"]["test"]), dci(K["d_mg32_vs_r1"]["test"])))
    A("- **(f) Dev vs test:** %d of %d contrasts keep their sign. The %d that flip are SP-32k's two within-top-set "
      "comparisons, whose dev Δs were small and not significant (%s vs MinGram-48k; %s vs MinGram-32k). %d changes "
      "exclude 0 in the paired dev-vs-test bootstrap (§9). The rank order of the 6 correlates at Kendall τ_b = %.2f." % (
          n_con - len(flips), n_con, len(flips), dev_str("o_sp32_vs_mg48"), dev_str("o_sp32_vs_mg32"), len(sig_change),
          R["rank_correlation_dev_vs_test_6"]["kendall_tau_b"]))
    A("- **Large arbiter on test (report-only; 2 seeds, MinGram-32k 1 seed).** Mean test bpb: SP-32k %.5f < MinGram-32k "
      "%.5f (n = 1) < MinGram-48k %.5f < baseline %.5f. That is the same order as the dev large run on which the amended "
      "rule chose SP-32k. SP-32k vs MinGram-48k: %s. SP-32k vs MinGram-32k: %s with only one MinGram-32k seed; once seed "
      "noise is added, %s, p %s, so that gap is **not established**." % (
          lc[SP32]["mean_bpb"], lc[MG32]["mean_bpb"], lc[MG48]["mean_bpb"], lc[BASE]["mean_bpb"],
          dci(l2[pk(SP32, MG48)]), pct(lp[pk(SP32, MG32)]["delta_pct"]), ci(lp[pk(SP32, MG32)]["ci95_pct"]),
          pv(lp[pk(SP32, MG32)]["p"])))
    A("- **Validation:** %d result records (%d confirm + %d large), %s checks, all passed. bpb recomputes bit-exactly. "
      "The baseline is bitwise identical in both final-test bundles (%s). Parity with the CPU reference passes in both bundles." % (
          VS["n_records"], VS["by_stage"]["f54c929ba1ab/confirm"] + VS["by_stage"]["94175d26497f/confirm"],
          VS["by_stage"]["f54c929ba1ab/large"], "{:,}".format(VS["n_checks_total"]), VS["cross_bundle"]))
    A("")

    # ------------------------------------------------------------------ 1 inputs
    bf, bs = V["bundles"]["f54c929ba1ab"], V["bundles"]["94175d26497f"]
    A("## 1. Inputs, collection and validation")
    A("")
    A("`lm/collect_final_test.py` (sha256 `%s…`) copied both Colab folders from `G:\\My Drive\\hindko_lm_out_final_test\\` "
      "to `lm/colab_results/final_test/<bundle>/`. Every file was copied byte for byte and its sha256 re-checked after "
      "the copy, including the `large` records. Outputs: `results_table_final_test.json/.csv`, "
      "`validation_final_test.json`, `test_clusters.csv`." % code["lm/collect_final_test.py"][:12])
    A("")
    A("| bundle | role | candidates | confirm records | large records | built (FINAL_TEST_LOG.json) |")
    A("|---|---|---|---|---|---|")
    A("| `f54c929ba1ab` | %s | %s | %d | %d | %s → %s |" % (
        bf["role"], ", ".join("`%s`" % x for x in bf["candidates"]), VS["by_stage"]["f54c929ba1ab/confirm"],
        VS["by_stage"]["f54c929ba1ab/large"], bf["final_test_log_utc"], bf["final_test_log_completed_utc"]))
    A("| `94175d26497f` | %s | %s | %d | 0 | %s → %s |" % (
        bs["role"], ", ".join("`%s`" % x for x in bs["candidates"]), VS["by_stage"]["94175d26497f/confirm"],
        bs["final_test_log_utc"], bs["final_test_log_completed_utc"]))
    A("")
    A("**Per-record checks.** They are the checks `lm/collect_colab.py` applied to dev, with the test constants:")
    A("")
    A("- **Identity:** bundle sha256 = the staging `manifest.json` = the bundle id; `hk_lm.py` sha256 `3f78ad799f56…` in "
      "every record, the bundle and the local copy; the file key equals the record's stage, candidate, seed and LR; the "
      "candidate is in the bundle and in its WAVES file; `hk_lm.identity_mismatches` against `hk_lm.run_identity(...)` "
      "(protocol, bundle, recipe, candidate, seed, LR, model, budget, epochs, steps, ctx, device, AMP dtype) is empty.")
    A("- **Protocol fields and exact LR:** confirm = recipe `confirm`, seeds 1–5, LR exactly 1e-3 (float `==`, and "
      "`optimizer.peak_lr` equal), `preregistered: true`, 3,663 steps, 1 epoch. Large = recipe `large`, seeds 1–2, LR "
      "exactly 5e-4, `preregistered: false`, 7,325 steps, 2 epochs. Model d/L/H per recipe, n_vocab and ctx equal to the "
      "bundle, ctx = round(1,536 / train bytes per token), tied embeddings, dropout 0. AdamW (0.9, 0.95), eps 1e-8, wd 0.1, "
      "clip 1.0, 50 warm-up steps, cosine to 10 %. steps = steps_done = len(train_loss), all losses finite.")
    A("- **491 per-document arrays:** `bits`, `bytes`, `ntok`, `uids`, `doc_index` all have length 491. `doc_index` = "
      "0..490. `uids` equal the bundle's test documents and the uids of `data/test_strict.jsonl` (sha256 `a74c33ad…`), in "
      "uid order. `bytes` equal the bundle (Σ = 1,451,026). `ntok` = the bundle's test token count + 1 (EOT). All bits "
      "are finite and > 0.")
    A("- **Bit-exact bpb:** `float(np.sum(bits) / np.sum(bytes))` **==** the reported `bpb` == `dev.bpb` in every record, "
      "and `sum_bits` == Σbits.")
    A("- **Evaluation window, curve, device:** ctx and stride ctx/2, BOS given, EOT predicted; the learning curve on the "
      "bundle's fixed %d-document, %s-byte test subset at 25/50/75/100 %%; cuda / Tesla T4 / fp16 training / fp32 "
      "evaluation, strict determinism, tf32 off, equal to `runtime_pin.json`. bytes_seen within 1 %% of the budget "
      "(largest deviation %.2f %%)." % (bf["subset"]["n_docs"], "{:,}".format(bf["subset"]["bytes"]),
                                         100 * max(bf["stage_checks"]["confirm"]["bytes_seen_max_abs_rel_vs_budget"],
                                                   bf["stage_checks"]["large"]["bytes_seen_max_abs_rel_vs_budget"],
                                                   bs["stage_checks"]["confirm"]["bytes_seen_max_abs_rel_vs_budget"])))
    A("- **The dev validator agrees.** `collect_colab.validate_run` itself was also run on all %d confirm records, with its "
      "three dev constants (836 documents, dev bytes, subset bytes) set to the test values. Its check list and verdicts "
      "are identical to the ones above." % VS["n_confirm_records_cross_validated"])
    assert VS["collect_colab_validator_identical_on_all_confirm_records"]
    A("")
    A("**Result: %d / %d records pass every check (%s checks).**" % (VS["n_valid_all_checks"], VS["n_records"],
                                                                    "{:,}".format(VS["n_checks_total"])))
    A("")
    A("Further checks:")
    A("")
    A("- **Same models as dev (colab/FINAL_TEST.md check (b)).** For each of the %d records, the dev record of the same "
      "stage, candidate and seed was found (confirm: the non-duplicate Stage 4 record of `results_table.json`; large: "
      "`77e1368773fc/results/large__*`). The training-loss curve, the R3 lowest-norm embeddings, the step count, ctx, "
      "tokens and bytes seen, the parameter counts, the optimiser and the runtime (GPU, torch, CUDA, dtypes) are "
      "identical in all %d. So each test number is the dev-selected model, evaluated on test." % (
          VS["n_same_model_checked"], VS["n_same_model_checked"]))
    A("- **The baseline is bitwise identical across the two final-test bundles:** %s. Per-document bits, bytes, uids, "
      "token counts, training loss, curve and R3 are all equal. The copy in `94175d26497f` is dropped from the pooled "
      "analysis, so no run counts twice." % VS["cross_bundle"])
    A("- **Completeness.** Confirm: 4 × 5 records in `f54c929ba1ab` and 3 × 5 in `94175d26497f`, complete. Large: 7 of "
      "the 8 records Amendment 1 allows. `large__%s__lr5e-4__s2` is missing: the log's last line is `%s`, with no DONE and "
      "no 'finished' line after it. The Colab session disconnected there, and the free GPU quota was then exhausted, "
      "so MinGram-32k has **one** large test seed." % (MG32, V["progress_log_f54c"]["large"]["last_step_line"]))
    par = bf["parity"]
    ref = R["references"]
    dev_par16 = sorted({v["parity/parity_cuda_fp16.json"] for v in ref["dev_parity_rel_diff_vs_cpu"].values()})
    assert len(dev_par16) == 1
    assert V["cross_bundle_parity"]["fp16"]["bits_identical"] and V["cross_bundle_parity"]["fp32"]["bits_identical"]
    A("- **Parity (CPU ↔ GPU, fixed tiny config, on a %d-document, %s-byte test slice):** fp16 %+.3f %%, fp32 %+.3f %% "
      "against the CPU reference %.6f bpb (tolerance 1 %%). Both bundles pass and give bitwise identical parity runs. On "
      "dev the fp16 figure was %+.3f %%, on a dev slice." % (
          ref["test_parity_slice"]["n_docs"], "{:,}".format(ref["test_parity_slice"]["bytes"]),
          100 * par["parity/parity_cuda_fp16.json"]["rel_diff_vs_cpu"], 100 * par["parity/parity_cuda_fp32.json"]["rel_diff_vs_cpu"],
          par["parity/parity_cuda_fp16.json"]["cpu_bpb"], 100 * dev_par16[0]))
    clu = R["clusters"]
    tcv = VS["test_clusters"]
    assert tcv["cluster_field_equals_manifest_group_rule"] and clu["n_clusters"] == 27
    A("- **Clusters.** The 'cluster' field of `data/test_strict.jsonl` gives %d clusters (book %d, newspaper %d, web %d). "
      "For all 491 documents it equals the split manifest's `group` with per-record web groups collapsed to the site "
      "(the dev rule). The raw manifest has %d groups (web %d). The largest cluster is a %s cluster with %.1f %% of test "
      "bytes." % (clu["n_clusters"], clu["clusters_by_source"]["book"], clu["clusters_by_source"]["newspaper"],
                  clu["clusters_by_source"]["web"], clu["n_raw_groups"], clu["raw_groups_by_source"]["web"],
                  clu["largest_cluster"]["source"], 100 * clu["largest_cluster"]["share"]))
    tse = re_["test_strict"]
    assert tse["hf_tokenizer_json_ids_equal"] == tse["n"] == 491 and tse["changed_vs_old_model"] == 0
    A("- **The released files encode test exactly as the LM saw it.** Per `release_build/sp32k/equivalence.json`, the "
      "released `tokenizer.json` gives the canonical ids on %d/%d test documents, the re-scored model changed %d of them "
      "relative to the `sp.model` the LM bundles used, and its %s test tokens equal the bundle's. SP-32k's test numbers "
      "therefore apply to the released files." % (tse["hf_tokenizer_json_ids_equal"], tse["n"], tse["changed_vs_old_model"],
                                                 "{:,}".format(tse["canonical_tokens"])))
    A("")

    # ------------------------------------------------------------------ 2 method
    A("## 2. Method, as run")
    A("")
    A("- **The DECISION.md bootstrap, unchanged:** `decide.Boot` and `decide.compare`; 10,000 replicates, "
      "`numpy.random.default_rng(12345)`. In each replicate the 27 test clusters are resampled with replacement within "
      "each source (book 6, newspaper 15, web 6), each candidate's 5 seeds are resampled with replacement, independently "
      "per candidate, and bpb = Σbits ÷ Σbytes, averaged over the resampled seeds. RNG order: cluster draws (sources "
      "sorted), then seed draws (candidates in sorted id order).")
    A("- **Primary test bootstrap:** all 6 confirm-scale test tokenizers of both bundles, in sorted id order: %s. The "
      "choice of stream moves the CI endpoints of (a) by at most 0.01 pp (§4.1)." % ", ".join("`%s`" % x for x in R["protocol"]["candidate_rng_order"]))
    A("- **Dev side:** the same 6 candidates' Stage 4 records via `decide.load_stage`, the 32 dev clusters via "
      "`decide.dev_clusters`, and `decide.Boot`. Its point Δs equal `decision.json` exactly wherever it has the same "
      "comparison (%d contrasts). Their CI endpoints differ by at most %.3f pp, because the seed-draw stream differs "
      "with 6 candidates instead of 18. The tables show dev CIs of record where they exist." % (
          R["dev_crosscheck_vs_decision_json"]["n"], R["dev_crosscheck_vs_decision_json"]["max_ci_endpoint_diff_pp"]))
    A("- **Dev vs test (addition, report-only):** one `decide.Boot` over dev and test together, with 6 strata (split × "
      "source). Dev and test clusters are resampled independently. Each candidate's seed draw is shared by the two "
      "splits, because the test models are the dev models. The change is Δ_test − Δ_dev per replicate, in pp. Its dev "
      "cluster draws equal the dev bootstrap's exactly (asserted).")
    A("- **Holm:** PLAN pre-registers one test comparison, (a); it is not corrected. For (b)–(e), a Holm column over those "
      "7 report-only comparisons is shown as a conservative reading. TOST margin δ = 0.3 %% of the test baseline mean = "
      "%.6f bpb." % R["protocol"]["tost_delta_abs_bpb_test"])
    A("")

    # ------------------------------------------------------------------ 3 per candidate
    A("## 3. Every tokenizer on test (confirm scale, 5 seeds each)")
    A("")
    A("| # | tokenizer | role | bundle | V | total params | test bytes/token | test bpb | per-seed test bpb (s1–s5) | seed s.d. | dev bpb (dev rank of 17) | Δ vs best of 6 on test | Holm p (5) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    best6 = ts["best_of6_by_mean"]
    for i, cc in enumerate(order, 1):
        e = C[cc]
        vb = ts["vs_best_of6"].get(cc)
        A("| %d | `%s` | %s | `%s` | %s | %.2fM | %.4f | %.5f | %s | %.2f %% | %.5f (%d) | %s | %s |" % (
            i, cc, e["role"], e["bundle"], "%dk" % (e["n_vocab"] // 1024),
            e["params_total"] / 1e6, e["test_bytes_per_token"], e["test_mean_bpb"],
            " / ".join("%.5f" % x for x in e["test_seed_bpb"]), e["test_seed_sd_rel_pct"], e["dev_mean_bpb"],
            e["dev_rank_of17_decision"], "best" if vb is None else dci(vb), "–" if vb is None else pv(ts["holm_vs_best_of6"][cc])))
    A("")
    dord = R["dev_order_same6"]
    assert set(dord[:3]) == set(order[:3]) == set(TOP) and dord[3:] == order[3:]
    A("- The dev order of these 6 was %s. On test the three top-set members swap places among themselves; the three BPE "
      "tokenizers keep theirs." % " < ".join(S[x] for x in dord))
    A("- Pooled relative seed s.d. over the 6: %.3f %% on test, %.3f %% on dev (same models)." % (
        R["power"]["pooled_rel_sd_pct_test"], R["power"]["pooled_rel_sd_pct_dev_same6"]))
    A("")

    # ------------------------------------------------------------------ 4 (a)
    A("## 4. (a) The pre-registered comparison (PLAN §7.5–7.6)")
    A("")
    A("**`%s` − `%s` on test: Δ = %s, p %s** (bpb %.5f vs %.5f; bootstrap s.e. %.3f pp; 80 %%-power detectable effect "
      "%.2f %%)." % (MG48, BASE, dci(a["test"], 3), pv(a["test"]["p"]), C[MG48]["test_mean_bpb"], C[BASE]["test_mean_bpb"],
                     a["test"]["se_pct"], a["test"]["mde80_pct"]))
    A("")
    A("- Dev, of record: %s, p %s." % (dci(cl["dev_of_record"], 3), pv(cl["dev_of_record"]["p"])))
    A("- **Rule:** %s." % cl["rule"])
    A("- **Verdict: the improvement claim holds on test.** The test CI excludes 0, and the test Δ has the dev sign.")
    A("- The test gain is smaller than the dev gain by %.3f pp %s, p %s (paired dev-vs-test bootstrap). The test point "
      "lies %s the dev 95 %% CI %s." % (a["dev_vs_test"]["change_pp"], ci(a["dev_vs_test"]["change_ci95_pp"], 3),
                                        pv(a["dev_vs_test"]["p_change"]),
                                        "inside" if a["dev_vs_test"]["test_point_inside_dev_ci95"] else "just outside",
                                        ci(cl["dev_of_record"]["ci95_pct"], 3)))
    A("- **Clusters:** MinGram-48k is better in %d of the 27 clusters (%.1f %% of test bytes). Leaving any one cluster out "
      "keeps Δ between %+.3f %% and %+.3f %%." % (
          a["cluster_view_test"]["n_clusters_a_better"], 100 * a["cluster_view_test"]["bytes_share_a_better"],
          a["leave_one_cluster_out_test"]["min_pct"], a["leave_one_cluster_out_test"]["max_pct"]))
    A("- **Per source** (same replicates, restricted to one source's clusters; report-only): %s." % "; ".join(
        "%s (%d clusters) %s" % (s, R["per_source_test"][s]["n_clusters"], dci(v)) for s, v in a["per_source_test"].items()))
    A("")
    A("### 4.1 Sensitivity of (a) and (b)")
    A("")
    A("| analysis | (a) MinGram-48k vs baseline | (b) SP-32k vs baseline |")
    A("|---|---|---|")
    A("| **primary** (6 candidates, 27 clusters, 5 seeds) | %s | %s |" % (dci(a["test"]), dci(b["test"])))
    for name, row in R["sensitivity_test"].items():
        if name == "p_plus1":
            A("| %s | p_plus1 %.4f | p_plus1 %.4f |" % (row["what"], row["a_prereg"]["p_plus1"], row["b_released"]["p_plus1"]))
            continue
        A("| %s: %s | %s | %s |" % (name, row["what"], dci(row["a_prereg"]) if "a_prereg" in row else "–",
                                    dci(row["b_released"]) if "b_released" in row else "–"))
    sens_a = [row["a_prereg"] for n_, row in R["sensitivity_test"].items() if n_ != "p_plus1" and "a_prereg" in row]
    assert all(x["ci95_excludes_0"] and x["delta_pct"] < 0 for x in sens_a)
    rng_rows = [R["sensitivity_test"][n_]["a_prereg"] for n_ in ("bundle_f54c_only", "pair_only")]
    assert max(max(abs(x["ci95_pct"][0] - a["test"]["ci95_pct"][0]), abs(x["ci95_pct"][1] - a["test"]["ci95_pct"][1]))
               for x in rng_rows) < 0.01
    A("")
    s13 = R["sensitivity_test"]["seeds_1to3"]
    assert s13["a_prereg"]["delta_pct"] > a["test"]["delta_pct"] and s13["b_released"]["delta_pct"] > b["test"]["delta_pct"]
    A("The claim holds in every row. With seeds 1–3 only, both Δs are smaller in size, and both CIs still exclude 0.")
    A("")

    # ------------------------------------------------------------------ 5 (b), (c)
    A("## 5. (b) The released tokenizer and (c) MinGram-32k")
    A("")
    A("| contrast | test Δ | p | Holm p (7, report-only) | dev Δ | test book (6) | test newspaper (15) | test web (6) |")
    A("|---|---|---|---|---|---|---|---|")
    for cid in ("b_released", "c_mingram32"):
        k = K[cid]
        ps = k["per_source_test"]
        A("| `%s` − `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
            k["a"], k["b"], dci(k["test"]), pv(k["test"]["p"]), pv(k["test_holm_p_report_only_family"]), dev_str(cid),
            dci(ps["book"]), dci(ps["newspaper"]), dci(ps["web"])))
    A("")
    A("- **SP-32k improves on the baseline on test in every source.** Each per-source CI lies below 0.")
    for s in ("book", "newspaper", "web"):
        assert b["per_source_test"][s]["ci95_pct"][1] < 0 and c_["per_source_test"][s]["ci95_pct"][1] < 0
    A("- **Its dev margin did not fully carry over.** Δ vs baseline went from %s to %s: a change of %+.2f pp %s, the "
      "largest of the 13 contrasts. On dev, SP-32k's seed s.d. was %.2f %%, the largest of the top set; its dev Δ CI "
      "was also the widest of the three." % (dev_str("b_released"), dci(b["test"]), b["dev_vs_test"]["change_pp"],
                                             ci(b["dev_vs_test"]["change_ci95_pp"]), C[SP32]["dev_seed_sd_rel_pct"]))
    assert max(K, key=lambda x: K[x]["dev_vs_test"]["change_pp"]) == "b_released"
    assert max(TOP, key=lambda x: C[x]["dev_seed_sd_rel_pct"]) == SP32
    wid = {x: dev_of(x)["ci95_pct"][1] - dev_of(x)["ci95_pct"][0] for x in ("a_prereg", "b_released", "c_mingram32")}
    assert max(wid, key=wid.get) == "b_released"
    nsd = R["references"]["amendment1_dev_confirm_newspaper"]
    assert nsd[pk(SP32, MG48)] > 0 and nsd[pk(SP32, MG32)] > 0
    A("- **Newspaper text:** on test SP-32k is behind both MinGram builds there (vs MinGram-48k %s; vs MinGram-32k %s). "
      "AMENDMENT_1_STATS.md §4 found the same at the dev confirm scale (%s and %s)." % (
          dci(K["o_sp32_vs_mg48"]["per_source_test"]["newspaper"]), dci(K["o_sp32_vs_mg32"]["per_source_test"]["newspaper"]),
          pct(nsd[pk(SP32, MG48)]), pct(nsd[pk(SP32, MG32)])))
    assert K["o_sp32_vs_mg48"]["per_source_test"]["newspaper"]["ci95_pct"][0] > 0
    assert K["o_sp32_vs_mg32"]["per_source_test"]["newspaper"]["ci95_pct"][0] > 0
    A("")
    A("**The released tokenizer's standing, worded as AMENDMENT_1.md §5 requires:**")
    A("")
    A("1. It is in the statistically tied top set at the confirm scale (DECISION.md §3, on dev). Applying the same PLAN "
      "§7.3 rule to the 6 test tokenizers, report-only, gives the same three members (§8).")
    dlr = L["dev_large_of_record"]
    assert min(TOP, key=lambda x: lc[x]["dev_large_mean_bpb"]) == SP32
    A("2. It was chosen by the amended rule: the lowest large-arbiter dev bpb within that top set (%.5f, against %.5f "
      "for MinGram-32k and %.5f for MinGram-48k; %s and %s, 2 seeds each; AMENDMENT_1_STATS.md §0). The report-only "
      "large run on test keeps that order (§10)." % (
          lc[SP32]["dev_large_mean_bpb"], lc[MG32]["dev_large_mean_bpb"], lc[MG48]["dev_large_mean_bpb"],
          dci(dlr[pk(SP32, MG32)]), dci(dlr[pk(SP32, MG48)])))
    A("3. Its test result: %s vs the baseline at the confirm scale, p %s; highest test bpb of the three top-set members "
      "(not significantly). It must not be called 'best on test'." % (dci(b["test"]), pv(b["test"]["p"])))
    A("")

    # ------------------------------------------------------------------ 6 (d)
    A("## 6. (d) The forking-paths check: the round-2 top set vs the round-1 confirmatory winner")
    A("")
    A("Round 2 was exploratory: its tokenizers were built after the round-1 results were seen (DECISION.md §8, "
      "deviation 4). Restricted to the pre-registered round 1, the rule chooses `%s` (DECISION.md §7.8). If the round-2 "
      "gain were an artefact of selecting on dev, the top set would not beat that tokenizer on sealed test data." % A132)
    A("")
    A("| contrast | test Δ | p | Holm p (7) | dev Δ | change (pp) | clusters where a is better |")
    A("|---|---|---|---|---|---|---|")
    for cid in ("d_mg48_vs_r1", "d_mg32_vs_r1", "d_sp32_vs_r1", "d_r1_vs_base"):
        k = K[cid]
        A("| `%s` − `%s` | %s | %s | %s | %s | %+.2f %s | %d / 27 |" % (
            k["a"], k["b"], dci(k["test"]), pv(k["test"]["p"]), pv(k["test_holm_p_report_only_family"]), dev_str(cid),
            k["dev_vs_test"]["change_pp"], ci(k["dev_vs_test"]["change_ci95_pp"]), k["cluster_view_test"]["n_clusters_a_better"]))
    A("")
    A("- **All three top-set members beat `%s` on test**, and each stays significant after Holm. The round-2 "
      "advantage over the best pre-registered round-1 tokenizer is therefore not only a product of selecting on dev: "
      "it replicates on the sealed test split." % A132)
    assert all(K[x]["test_holm_p_report_only_family"] <= 0.05 for x in ("d_mg48_vs_r1", "d_mg32_vs_r1", "d_sp32_vs_r1"))
    A("- MinGram-48k's and MinGram-32k's margins over `%s` are the same size on test as on dev (changes %+.2f and "
      "%+.2f pp, both CIs include 0). SP-32k's margin shrank from %s to %s; that change's CI %s just includes 0." % (
          A132, K["d_mg48_vs_r1"]["dev_vs_test"]["change_pp"], K["d_mg32_vs_r1"]["dev_vs_test"]["change_pp"],
          dev_str("d_sp32_vs_r1"), dci(K["d_sp32_vs_r1"]["test"]), ci(K["d_sp32_vs_r1"]["dev_vs_test"]["change_ci95_pp"])))
    assert not K["d_mg48_vs_r1"]["dev_vs_test"]["change_ci95_excludes_0"]
    assert not K["d_mg32_vs_r1"]["dev_vs_test"]["change_ci95_excludes_0"]
    assert not K["d_sp32_vs_r1"]["dev_vs_test"]["change_ci95_excludes_0"]
    A("- The confirmatory round-1 claim also holds on test: `%s` vs the baseline %s (dev %s)." % (
        A132, dci(K["d_r1_vs_base"]["test"]), dev_str("d_r1_vs_base")))
    A("")

    # ------------------------------------------------------------------ 7 (e)
    A("## 7. (e) Equal-vocabulary controls")
    A("")
    A("At equal vocabulary the models have the same total parameter count (within 768 parameters, from the ctx-dependent "
      "position table), so these contrasts are free of the vocabulary-size/parameter confound that AMENDMENT_1.md "
      "describes.")
    A("")
    A("| contrast | vocabulary | test Δ | p | dev Δ | change (pp) | test book | test newspaper | test web |")
    A("|---|---|---|---|---|---|---|---|---|")
    for cid, v_ in (("e_mg48_vs_bpe48", "48k"), ("d_sp32_vs_r1", "32k"), ("d_mg32_vs_r1", "32k")):
        k = K[cid]
        ps = k["per_source_test"]
        A("| `%s` − `%s` | %s | %s | %s | %s | %+.2f %s | %s | %s | %s |" % (
            k["a"], k["b"], v_, dci(k["test"]), pv(k["test"]["p"]), dev_str(cid), k["dev_vs_test"]["change_pp"],
            ci(k["dev_vs_test"]["change_ci95_pp"]), dci(ps["book"]), dci(ps["newspaper"]), dci(ps["web"])))
    A("")
    A("- **The algorithm effect holds on test at equal vocabulary.** MinGram beats byte-level BPE at 48k and at 32k, and "
      "SentencePiece Unigram beats it at 32k. All three CIs exclude 0.")
    A("- SP-32k vs BPE-32k is not significant on newspaper text (%s). The two MinGram contrasts are below 0 in every source." % (
        dci(K["d_sp32_vs_r1"]["per_source_test"]["newspaper"])))
    assert not K["d_sp32_vs_r1"]["per_source_test"]["newspaper"]["ci95_excludes_0"]
    for cid in ("e_mg48_vs_bpe48", "d_mg32_vs_r1"):
        assert all(v["ci95_pct"][1] < 0 for v in K[cid]["per_source_test"].values())
    A("- **Vocabulary size at this scale (report-only):** BPE 32k → 48k %s (dev %s); MinGram 32k → 48k %s (dev %s); the "
      "latter is TOST-equivalent within ±0.3 %% on test (its 90 %% CI lies inside ±δ). The 32k→48k step buys little at "
      "the confirm scale, on dev and on test." % (
          dci(K["o_bpe48_vs_bpe32"]["test"]), dev_str("o_bpe48_vs_bpe32"), dci(K["o_mg48_vs_mg32"]["test"]),
          dev_str("o_mg48_vs_mg32")))
    assert K["o_mg48_vs_mg32"]["test"]["tost_equivalent"]
    A("")

    # ------------------------------------------------------------------ 8 top set
    A("## 8. The top set on test (report-only; never re-selects)")
    A("")
    A("| pair (Δ = a vs b) | test Δ | p | 90 % CI (bpb) | TOST ±0.3 % | clusters where a is better | test book | test newspaper | test web | dev Δ |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for cid in ("o_sp32_vs_mg48", "o_sp32_vs_mg32", "o_mg48_vs_mg32"):
        k = K[cid]
        ps = k["per_source_test"]
        A("| `%s` vs `%s` | %s | %s | [%+.5f, %+.5f] | %s | %d / 27 (%.0f %% of bytes) | %s | %s | %s | %s |" % (
            k["a"], k["b"], dci(k["test"]), pv(k["test"]["p"]), k["test"]["ci90"][0], k["test"]["ci90"][1],
            "equivalent" if k["test"]["tost_equivalent"] else "not shown", k["cluster_view_test"]["n_clusters_a_better"],
            100 * k["cluster_view_test"]["bytes_share_a_better"], dci(ps["book"]), dci(ps["newspaper"]), dci(ps["web"]),
            dev_str(cid)))
    A("")
    rd = ts["rank_distribution"]
    A("- **Rank within the top set across the 10,000 test replicates:** %s." % "; ".join(
        "`%s` 1st %.1f %%, 2nd %.1f %%, 3rd %.1f %%" % (c, 100 * rd[c]["1"], 100 * rd[c]["2"], 100 * rd[c]["3"]) for c in TOP))
    A("- **PLAN §7.3's rule applied to the 6 test tokenizers** (best by mean `%s`; Holm over 5 comparisons): the top set "
      "is %s. `%s`, `%s` and `%s` are significantly worse (Holm p %s). This is the same set of three as on dev; it is "
      "reported, not used." % (
          best6, ", ".join("`%s`" % x for x in ts["plan73_top_set_rule_on_test_report_only"]), A148, A132, BASE,
          " / ".join(pv(ts["holm_vs_best_of6"][x]) for x in (A148, A132, BASE))))
    A("- **What the test says about SP-32k vs the MinGram builds at this scale:** SP-32k is behind, by %s and %s, but "
      "neither is significant at 95 %%. The newspaper clusters carry the gap (§5). The test gives no support for "
      "calling SP-32k better than the MinGram builds at the confirm scale, and no significant evidence that it is "
      "worse." % (pct(K["o_sp32_vs_mg48"]["test"]["delta_pct"]), pct(K["o_sp32_vs_mg32"]["test"]["delta_pct"])))
    A("")

    # ------------------------------------------------------------------ 9 (f)
    A("## 9. (f) Dev vs test, every Δ")
    A("")
    A("Change = Δ_test − Δ_dev in pp, with its 95 % CI from the paired dev+test bootstrap (§2). 'Inside' says whether "
      "the test point lies inside the dev 95 % CI.")
    A("")
    A("| sec. | contrast | dev Δ | test Δ | same sign | change (pp) | p (change) | inside dev CI | test / dev |")
    A("|---|---|---|---|---|---|---|---|---|")
    for cid, k in K.items():
        dv = k["dev_vs_test"]
        A("| %s | `%s` − `%s` | %s | %s | %s | %+.3f %s | %s | %s | %s |" % (
            k["section"] if k["section"] != "o" else "–", k["a"], k["b"], dev_str(cid), dci(k["test"]),
            "yes" if dv["sign_agrees"] else "**no**", dv["change_pp"], ci(dv["change_ci95_pp"], 3), pv(dv["p_change"]),
            "yes" if dv["test_point_inside_dev_ci95"] else "no",
            "%.2f" % dv["ratio_test_over_dev"] if dv["ratio_test_over_dev"] is not None else "–"))
    A("")
    shr = [cid for cid in K if K[cid]["dev_vs_test"]["change_ci95_excludes_0"]]
    A("- **Sign:** %d of %d contrasts keep their sign. The flips are `%s` vs `%s` and vs `%s`. Their dev Δs were near 0 "
      "(both CIs spanned 0) and their test Δs are not significant either, so neither flip contradicts a dev finding." % (
          n_con - len(flips), n_con, SP32, MG48, MG32))
    assert all(not K[x]["test"]["ci95_excludes_0"] for x in flips)
    assert all(not (dev_of(x)["ci95_pct"][0] > 0 or dev_of(x)["ci95_pct"][1] < 0) for x in flips)
    A("- **Significant changes (CI of the change excludes 0):** %s. All are in the direction of a smaller advantage on "
      "test for the tokenizer that looked better on dev. Regression after selection on dev predicts this; the next "
      "point qualifies it." % ", ".join("`%s` − `%s` %+.2f pp" % (K[x]["a"], K[x]["b"], K[x]["dev_vs_test"]["change_pp"]) for x in shr))
    for x in shr:
        dv = K[x]["dev_vs_test"]
        assert (dv["dev_pct"] < 0 and dv["change_pp"] > 0) or (dv["dev_pct"] > 0 and dv["change_pp"] < 0) or \
            (x in flips and dv["change_pp"] > 0)
    vb = [x for x in K if K[x]["b"] == BASE]
    assert all(K[x]["dev_vs_test"]["change_pp"] > 0 for x in vb)
    A("- **Every gain over the 16k baseline is smaller on test than on dev** (changes %+.2f to %+.2f pp over the %d "
      "tokenizers), for the round-1 winner BPE-32k (%+.2f pp %s) as well as for the round-2 tokenizers. Selection on dev "
      "(winner's curse) predicts a shrink for the selected tokenizers. A shrink shared by all of them is also consistent "
      "with the baseline simply doing relatively better on the test text. These data do not separate the two." % (
          min(K[x]["dev_vs_test"]["change_pp"] for x in vb), max(K[x]["dev_vs_test"]["change_pp"] for x in vb), len(vb),
          K["d_r1_vs_base"]["dev_vs_test"]["change_pp"], ci(K["d_r1_vs_base"]["dev_vs_test"]["change_ci95_pp"])))
    rc = R["rank_correlation_dev_vs_test_6"]
    A("- **Rank order of the 6:** Kendall τ_b = %.2f, Spearman ρ = %.2f between dev and test means." % (
        rc["kendall_tau_b"], rc["spearman_rho"]))
    A("")

    # ------------------------------------------------------------------ 10 large
    A("## 10. Large arbiter on test (report-only)")
    A("")
    import re as _re
    mstep = _re.search(r"step (\d+)/(\d+)", V["progress_log_f54c"]["large"]["last_step_line"])
    lsteps = bf["stage_checks"]["large"]["steps"]
    assert len(lsteps) == 1 and int(mstep.group(2)) == lsteps[0]
    A("AMENDMENT_1.md §4 allowed a report-only large-arbiter test run of the top set plus the baseline, at LR 5e-4 fixed "
      "from dev and 2 seeds. It ran in bundle `f54c929ba1ab` (d=384, L=6, H=6, 2 epochs, %s steps). It cannot change "
      "which tokenizer is released. **MinGram-32k has one seed (n = 1):** its seed-2 run stopped at step %s of %s "
      "when Colab disconnected, and the free GPU quota was then exhausted." % (
          "{:,}".format(lsteps[0]), "{:,}".format(int(mstep.group(1))), "{:,}".format(lsteps[0])))
    A("")
    A("| tokenizer | seeds | test bpb per seed | mean test bpb | Δ vs baseline (means) | dev large bpb (s1 / s2) | dev large mean |")
    A("|---|---|---|---|---|---|---|")
    for cc in L["order_by_mean"]:
        e = lc[cc]
        A("| `%s` | %s | %s | %.5f | %s | %s | %.5f |" % (
            cc, ",".join(map(str, e["seeds"])), " / ".join("%.5f" % x for x in e["seed_bpb"]), e["mean_bpb"],
            "–" if cc == BASE else pct(e["vs_baseline_pct_all_seeds"]), " / ".join("%.5f" % x for x in e["dev_large_seed_bpb"]),
            e["dev_large_mean_bpb"]))
    A("")
    A("Three ways to put a CI on these, each with the test cluster draws of the primary bootstrap:")
    A("")
    A("- **2 seeds:** `decide.Boot` on SP-32k, MinGram-48k and the baseline (seeds 1–2 each).")
    A("- **seed 1 only:** `decide.Boot` on all four with seed 1. Only cluster noise enters; seed noise is ignored.")
    A("- **all seeds + seed noise:** each candidate's available seeds averaged, plus relative seed noise N(0, σ²/n), with "
      "σ = the larger of its dev large and dev confirm seed s.d. (%s) — AMENDMENT_1_STATS.md §5's 'own s.d.' variant." % (
          ", ".join("%s %.3f %%" % (S[c], lc[c]["sigma_rel_pct_used_in_parametric_variant"]) for c in L["order_by_mean"])))
    A("")
    A("| pair (Δ = a vs b) | 2 seeds | seed 1 only | all seeds + seed noise | dev large (2 seeds, of record) | change dev → test (pp) |")
    A("|---|---|---|---|---|---|")
    for a_, b_ in [(SP32, BASE), (MG32, BASE), (MG48, BASE), (SP32, MG32), (SP32, MG48), (MG48, MG32)]:
        key = pk(a_, b_)
        x2 = l2.get(key)
        dv = L["dev_vs_test"][key]
        A("| `%s` vs `%s` | %s | %s | %s, p %s | %s, p %s | %+.2f %s%s |" % (
            a_, b_, "–" if x2 is None else "%s, p %s" % (dci(x2), pv(x2["p"])), "%s, p %s" % (dci(l1[key]), pv(l1[key]["p"])),
            dci(lp[key]), pv(lp[key]["p"]), dci(L["dev_large_of_record"][key]), pv(L["dev_large_of_record"][key]["p"]),
            dv["ci_basis_change_pp"], ci(dv["change_ci95_pp"]), "" if x2 is not None else " (seed 1)"))
    A("")
    A("- **The large-scale order on test is the dev order:** SP-32k < MinGram-32k < MinGram-48k < baseline, which is the "
      "evidence the amended rule used. Every large-scale Δ keeps its dev sign.")
    A("- **SP-32k vs MinGram-48k and vs the baseline are clear** with 2 seeds each (%s; %s), and survive the seed-noise "
      "variant (%s; %s)." % (dci(l2[pk(SP32, MG48)]), dci(l2[pk(SP32, BASE)]), dci(lp[pk(SP32, MG48)]), dci(lp[pk(SP32, BASE)])))
    assert lp[pk(SP32, MG48)]["ci95_excludes_0"] and lp[pk(SP32, BASE)]["ci95_excludes_0"]
    A("- **SP-32k vs MinGram-32k is not established on test.** Seed 1 alone gives %s, but that ignores seed noise; with "
      "it, %s, p %s. On dev (2 seeds) it was %s." % (dci(l1[pk(SP32, MG32)]), dci(lp[pk(SP32, MG32)]), pv(lp[pk(SP32, MG32)]["p"]),
                                                     dci(L["dev_large_of_record"][pk(SP32, MG32)])))
    A("- **The test margins are smaller than the dev ones** (e.g. SP-32k vs baseline %s on test, %s on dev; change "
      "%+.2f pp %s)." % (pct(lc[SP32]["vs_baseline_pct_all_seeds"]), pct(L["dev_large_of_record"][pk(SP32, BASE)]["delta_pct"]),
                         L["dev_vs_test"][pk(SP32, BASE)]["ci_basis_change_pp"], ci(L["dev_vs_test"][pk(SP32, BASE)]["change_ci95_pp"])))
    A("- **Confirm → large on test** (same clusters; AMENDMENT_1_STATS.md §6.2 style, the three 2-seed arms): %s. The "
      "48k vocabulary's confirm-scale edge over the baseline shrinks by about 1 pp at the large scale on test too, as it "
      "did on dev (%+.2f pp there); SP-32k vs MinGram-48k reverses from %s to %s. This is the parameter-confound "
      "pattern of AMENDMENT_1_STATS.md §6.3, now seen on test as well." % (
          "; ".join("`%s` vs `%s`: %s → %s, change %+.2f pp %s" % (
              k_.split(" - ")[0], k_.split(" - ")[1], pct(v_["confirm_pct"]), pct(v_["large_pct"]), v_["change_pp"],
              ci(v_["change_ci95_pp"])) for k_, v_ in sc.items()),
          R["references"]["amendment1_dev_scale_change"][pk(MG48, BASE)]["change_pp"],
          pct(sc[pk(SP32, MG48)]["confirm_pct"]), pct(sc[pk(SP32, MG48)]["large_pct"])))
    assert sc[pk(SP32, MG48)]["confirm_pct"] > 0 > sc[pk(SP32, MG48)]["large_pct"]
    assert 0.8 < sc[pk(MG48, BASE)]["change_pp"] < 1.3
    A("")
    lh = V["progress_log_f54c"]["large"]["header"][:8]
    A("The large test stage started at %s UTC (first record %s UTC), after every confirm test number existed (last "
      "confirm record %s UTC). Amendment 1 had announced it before any test number existed, and it changes nothing." % (
          lh, T["large_test_records_utc"][0][11:19], T["last_confirm_test_record_utc"][11:19]))
    assert lh > T["last_confirm_test_record_utc"][11:19]
    A("")

    # ------------------------------------------------------------------ 11 deviations
    A("## 11. Deviations and caveats (all additive)")
    A("")
    A("1. **Six tokenizers on test, not two.** PLAN §7.6 names the chosen tokenizer and the baseline. The two other "
      "top-set members were fixed as report-only before test (`decision.json` → `test_candidates_fixed`, %s). `%s` "
      "and `%s` were added in bundle `94175d26497f` after the adversarial audit; that bundle was built at %s, before "
      "the first test LM record (%s). None of them can re-select." % (
          T["test_candidates_fixed_utc"], A132, A148, T["final_test_log_builds_utc"][1][1], T["first_test_lm_record_utc"]))
    A("2. **Seeds 1–5, forced** (`--force-extra-seeds`), as on dev; the test power check did not decide the seed count.")
    A("3. **GPU, fp16 training, fp32 evaluation, Tesla T4**, as on dev; the same `hk_lm.py`. Parity fp16 %+.3f %% on the "
      "test slice (dev %+.3f %% on the dev slice), within the 1 %% tolerance." % (
          100 * par["parity/parity_cuda_fp16.json"]["rel_diff_vs_cpu"], 100 * dev_par16[0]))
    rd_ = V["run_all_diff_bundled_vs_large_run"]
    assert rd_["large_run_sha256"] == V["run_all_local_sha256"] and len(rd_["changed_lines"]) <= 12
    assert any("final-test-large" in x or "final_test_large" in x for x in rd_["changed_lines"])
    A("4. **The large test run is report-only and incomplete.** It was unlocked by a `run_all.py` change (sha256 `%s…` "
      "vs the bundled `%s…`; %d changed lines, all implementing the `--final-test-large` switch, which requires a fixed "
      "`--large-lr`; the diff is in `test_results.json` → `validation.run_all_diff_bundled_vs_large_run`). The bundled "
      "`hk_lm.py` trained every model, and its training losses equal the dev large runs. MinGram-32k has 1 seed, not 2." % (
          rd_["large_run_sha256"][:12], rd_["bundled_sha256"][:8], len(rd_["changed_lines"])))
    A("5. **The primary test bootstrap spans both bundles' 6 tokenizers.** PLAN fixes no candidate list for the test "
      "bootstrap. The RNG stream moves the CI of (a) by less than 0.01 pp (§4.1).")
    A("6. **The paired dev-vs-test bootstrap and the Holm column over (b)–(e) are additions** for reading the results; "
      "neither is pre-registered.")
    assert clu["largest_cluster"]["source"] == "book"
    A("7. **5 seeds, 27 clusters.** The book stratum has %d clusters, the largest with %.0f %% of test bytes, and web "
      "%d clusters with %s bytes, so per-source CIs are wide. Gaps below about %.2f %% (the 80 %%-power detectable "
      "effect of SP-32k vs MinGram-48k) are 'not distinguishable at our power', never 'equal'." % (
          clu["clusters_by_source"]["book"], 100 * clu["largest_cluster"]["share"], clu["clusters_by_source"]["web"],
          "{:,}".format(R["per_source_test"]["web"]["bytes"]), K["o_sp32_vs_mg48"]["test"]["mde80_pct"]))
    ne_c = {C[x]["params_non_embedding"] for x in C}
    ne_l = {lc[x]["params_non_embedding"] for x in lc}
    assert len(ne_c) == 1 and len(ne_l) == 1
    A("8. **Tiny proxies.** %.2fM (confirm) and %.2fM (large) non-embedding parameters. Nothing here is evidence about "
      "1B+ models or continued pretraining (PLAN §8)." % (list(ne_c)[0] / 1e6, list(ne_l)[0] / 1e6))
    A("")

    # ------------------------------------------------------------------ 12 claims
    A("## 12. What can be claimed now (PLAN §8, AMENDMENT_1.md §5)")
    A("")
    A("**Pre-registered pick, `%s`:**" % MG48)
    A("")
    A("- *\"Chosen by the pre-registered rule; selected after an exploratory second round; confirmed on a sealed test split "
      "used once: it improves on the standard BPE recipe `%s` by Δ = %s bits-per-byte (95 %% CI %s, p %s; hierarchical "
      "cluster bootstrap over 27 test clusters, 5 seeds; dev %s).\"*" % (
          BASE, pct(a["test"]["delta_pct"]), ci(a["test"]["ci95_pct"]), pv(a["test"]["p"]), dci(cl["dev_of_record"])))
    A("- It has the lowest mean test bpb of the 6 tokenizers scored on test at the confirm scale, but it is not "
      "distinguishable from MinGram-32k (%s, TOST-equivalent within ±0.3 %%) or from SP-32k (%s). \"Lowest held-out "
      "bits-per-byte among the 17 compared\" is not claimable: only 6 were scored on test." % (
          dci(K["o_mg48_vs_mg32"]["test"]), dci(K["o_sp32_vs_mg48"]["test"])))
    A("")
    A("**Released default, `%s`:**" % SP32)
    A("")
    A("- *\"In the statistically tied top set at the confirm scale; chosen by the amended rule (lowest large-arbiter dev "
      "bpb in that set: %s vs MinGram-32k and %s vs MinGram-48k, 2 seeds). On "
      "the strict test split: %s bits-per-byte vs the standard BPE recipe at the confirm scale (p %s, 5 seeds; dev %s); "
      "within the top set its confirm-scale test bpb is the highest of the three, not significantly (%s vs MinGram-48k, "
      "%s vs MinGram-32k). Report-only large arbiter on test: %s vs the baseline, %s vs MinGram-48k (2 seeds); vs "
      "MinGram-32k not established (1 seed).\"*" % (
          dci(dlr[pk(SP32, MG32)]), dci(dlr[pk(SP32, MG48)]),
          dci(b["test"]), pv(b["test"]["p"]), dev_str("b_released"), dci(K["o_sp32_vs_mg48"]["test"]),
          dci(K["o_sp32_vs_mg32"]["test"]), dci(l2[pk(SP32, BASE)]), dci(l2[pk(SP32, MG48)])))
    A("")
    assert all(K[x]["test"]["ci95_pct"][1] < 0 for x in K if K[x]["b"] == BASE)
    A("**Supported in general (test, confirm scale):** Unigram-family vocabularies beat byte-level BPE at equal "
      "vocabulary (§7); the round-2 top set beats the round-1 winner (§6); every tokenizer tested beats the 16k baseline.")
    A("")
    A("**Not claimable:**")
    A("")
    A("- That SP-32k is 'best on test', or 'lowest held-out bpb' on test: at the pre-registered confirm scale it is 3rd "
      "of the 3 top-set members, and the large run that puts it first is report-only, with 1–2 seeds.")
    maxp = max(max(C[x]["params_total"] for x in C), max(lc[x]["params_total"] for x in lc))
    A("- 'SOTA for all AI models' or 'best possible' (PLAN §8): %d LM candidates trained (%d ranked), on proxies of at "
      "most %.1fM parameters." % (R["references"]["decision_n_trained"], R["references"]["decision_k_ranked"], maxp / 1e6))
    A("- That MinGram-48k, MinGram-32k and SP-32k are 'equal'. Only MinGram-48k vs MinGram-32k is TOST-equivalent at "
      "the confirm scale on test; the others are 'not distinguishable at our power'.")
    A("- Transfer to 1B+ models or to continued pretraining; downstream task quality.")
    A("")

    # ------------------------------------------------------------------ 13 files
    A("## 13. Files and reproduction")
    A("")
    A("| file | content |")
    A("|---|---|")
    A("| `lm/collect_final_test.py` | copy + validation of both final-test bundles (deterministic; `--no-copy` re-validates) |")
    A("| `lm/colab_results/final_test/<bundle>/` | byte-exact copies of `G:\\My Drive\\hindko_lm_out_final_test\\<bundle>\\` (incl. `large` records) |")
    A("| `lm/colab_results/final_test/results_table_final_test.json` / `.csv` | one row per bundle × stage × candidate × seed (%d rows) |" % VS["n_records"])
    A("| `lm/colab_results/final_test/validation_final_test.json` | every check per record, stage checks, same-model checks, cross-bundle pairs, copy manifest |")
    A("| `lm/colab_results/final_test/test_clusters.csv` | the 491 test documents: doc_index, uid, bytes, source, group, cluster |")
    A("| `analysis/test_analysis.py` | every computation; imports `decide.py` (bootstrap, loaders, dev clusters) and `amendment1_stats.joint_reps`, unchanged |")
    A("| `analysis/test_results.json` | all numbers, per-seed bpb, per-source, leave-one-cluster-out, sensitivities, large run, and ready strings for the card's test placeholders (`card_placeholders`) |")
    A("| `analysis/test_report.py` | renders this file and asserts each qualitative statement |")
    A("")
    A("Reproduce from `F:\\Hindko\\_tokenizer` (single process, about a minute including the copy): "
      "`PYTHONIOENCODING=utf-8 python lm/collect_final_test.py && PYTHONIOENCODING=utf-8 python analysis/test_analysis.py "
      "&& PYTHONIOENCODING=utf-8 python analysis/test_report.py`.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
