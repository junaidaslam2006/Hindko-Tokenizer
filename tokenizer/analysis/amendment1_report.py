"""Renders analysis/AMENDMENT_1_STATS.md from analysis/amendment1_stats.json (report-only; never touches AMENDMENT_1.md).

The prose states qualitative readings (which CIs exclude 0, which order holds); each one is asserted against the JSON,
so a re-run on changed inputs fails instead of printing a stale sentence.

Run: PYTHONIOENCODING=utf-8 python analysis/amendment1_report.py
"""
import hashlib
import json
import os

ROOT = r"F:\Hindko\_tokenizer"
OUT = os.path.join(ROOT, "analysis")
J = os.path.join(OUT, "amendment1_stats.json")
MD = os.path.join(OUT, "AMENDMENT_1_STATS.md")
AUDIT = os.path.join(OUT, "audit", "audit_lm.json")
DECISION = os.path.join(OUT, "decision.json")

BASE = "A1-P1r3-D2-16k"
A132 = "A1-P1r3-D2-32k"
SP32 = "R2-A4-SPnat-D2-32k"
MG32 = "R2-A10-MinGram-P1r3-D2-32k"
MG48 = "R2-A10-MinGram-P1r3-D2-48k"
TOP = [SP32, MG48, MG32]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def k(a, b):
    return "%s - %s" % (a, b)


def pc(x, nd=2):
    return "%+.*f %%" % (nd, x)


def ci(v, nd=2):
    return "[%+.*f, %+.*f]" % (nd, v[0], nd, v[1])


def pv(p):
    return "<1e-4" if p == 0 else "%.4f" % p


def dci(x, nd=2):
    """'-0.62 % [-1.04, -0.31]' from a slim compare record."""
    return "%s %s" % (pc(x["delta_pct"], nd), ci(x["ci95_pct"], nd))


def t(c):
    return "`%s`" % c


def lrf(x):
    """0.0005 -> '5e-4'"""
    m, e = ("%.0e" % x).split("e")
    return "%se%d" % (m, int(e))


def excl0(x):
    lo, hi = x["ci95_pct"]
    return lo > 0 or hi < 0


def main():
    o = jl(J)
    dec = jl(DECISION)
    aud = jl(AUDIT)
    C = o["candidates"]
    L = o["large"]
    P = L["pairs"]
    CP = o["confirm_same5"]["pairs"]
    C2 = o["confirm_seeds12"]["pairs"]
    X = o["scale_change"]["interaction"]
    S = o["sensitivity_2_seeds"]
    PR = o["parameters"]
    inp = o["inputs"]
    resid = aud["residual_vs_BPE_param_curve_pct"]
    pooled_sd = S["parametric_pooled18_stress"]["sigma_rel_pct"]

    # ---------------- the readings the prose makes, asserted
    assert L["best"] == SP32 and L["order"][:3] == [SP32, MG32, MG48]
    assert excl0(P[k(SP32, MG32)]) and excl0(P[k(SP32, MG48)]) and excl0(P[k(MG48, MG32)])
    assert all(v <= 0.05 for v in L["holm_within_top_set_pairs"].values())
    assert not excl0(P[k(A132, BASE)]) and P[k(A132, BASE)]["tost_equivalent"]
    assert all(excl0(L["vs_baseline"][c]) and L["vs_baseline"][c]["delta_pct"] < 0 for c in TOP)
    assert X[k(A132, BASE)]["ci95_excludes_0"] and X[k(A132, BASE)]["change_pp"] > 0
    assert X[k(MG48, MG32)]["ci95_excludes_0"] and X[k(MG48, MG32)]["change_pp"] > 0
    assert X[k(SP32, A132)]["ci95_excludes_0"] and X[k(SP32, A132)]["change_pp"] < 0
    assert not X[k(MG32, A132)]["ci95_excludes_0"]
    assert CP[k(A132, BASE)]["delta_pct"] < 0 and CP[k(MG48, MG32)]["delta_pct"] < 0
    assert C2[k(A132, BASE)]["delta_pct"] < 0 and C2[k(MG48, MG32)]["delta_pct"] < 0
    assert excl0(S["parametric_own_sd"]["pairs"][k(SP32, MG32)]) and excl0(S["parametric_own_sd"]["pairs"][k(SP32, MG48)])
    assert excl0(S["rescaled_seed_bootstrap"]["pairs"][k(SP32, MG32)])
    assert not excl0(S["parametric_own_sd"]["pairs"][k(MG48, MG32)])
    assert not excl0(S["parametric_pooled18_stress"]["pairs"][k(SP32, MG32)])
    assert not any(S[v]["pairs"][k(A132, BASE)]["tost_equivalent"] for v in S)
    assert list(L["leave_one_cluster_out_top_set_order_counts"]) == [" < ".join([SP32, MG32, MG48])]
    assert all(L["leave_one_cluster_out"][k(a, b)]["sign_flips"] == 0 for a, b in [(SP32, MG32), (SP32, MG48), (MG48, MG32)])
    news = L["per_source"]["newspaper"]["pairs"]
    book = L["per_source"]["book"]["pairs"]
    cnews = o["confirm_same5"]["per_source"]["newspaper"]
    cbook = o["confirm_same5"]["per_source"]["book"]
    assert excl0(book[k(SP32, MG32)]) and not excl0(news[k(SP32, MG32)])
    assert excl0(cnews[k(SP32, MG32)]) and cnews[k(SP32, MG32)]["delta_pct"] > 0
    assert resid[SP32] < resid[MG32] < resid[MG48]
    assert inp["amendment_1_unchanged"] and inp["joint_block0_equals_primary_large_bootstrap"]
    assert inp["large_records_equal_drive_copies"] is True
    assert sorted(o["scale_change"]["pairs_changing_order"]) == sorted([sorted([BASE, A132]), sorted([MG32, MG48])])

    amend_rec = jl(os.path.join(OUT, "AMENDMENT_1.sha.json"))
    base_L = C[BASE]["large_mean_bpb"]
    delta_abs = o["protocol"]["tost_delta_abs_bpb_large"]
    ci90_a132 = [100 * v / base_L for v in P[k(A132, BASE)]["ci90"]]
    W = []
    w = W.append

    # ---------------- header
    w("# Amendment 1: the statistics behind the released default (large arbiter, dev_strict)")
    w("")
    w("Generated %s by `analysis/amendment1_stats.py` (sha256 `%s…`) and `analysis/amendment1_report.py`. The bootstrap "
      "is `analysis/decide.py`'s own code, imported unchanged (sha256 `%s…`, the same as in `decision.json`). "
      "Machine-readable twin: `analysis/amendment1_stats.json`."
      % (o["generated_utc"], o["code"]["amendment1_stats.py"][:12], inp["decide_py_sha256_now"][:12]))
    w("")
    w("- **Report-only.** This file changes neither the pre-registered decision (`DECISION.md`) nor `AMENDMENT_1.md`. "
      "The sha256 of `AMENDMENT_1.md` is still `%s…`, as recorded in `AMENDMENT_1.sha.json` (written %s)."
      % (amend_rec["sha256"][:12], amend_rec["written_utc"]))
    w("- **Dev only.** Every number is on dev_strict (836 documents, 1,445,513 bytes). No test LM result was read.")
    w("- **The large arbiter is not pre-registered.** It is d=384, L=6, H=6, trained for 2 epochs of the full train "
      "split at LR 5e-4, with seeds 1 and 2. Its LR sweep started at 19:44 UTC, and its 10 records were written "
      "between %s and %s, before Amendment 1 was written (%s)."
      % (inp["large_created_utc_range"][0], inp["large_created_utc_range"][1], amend_rec["written_utc"]))
    w("")

    # ---------------- 0. summary
    w("## 0. Summary")
    w("")
    w("**The amended rule's outcome is `R2-A4-SPnat-D2-32k`.** Within the pre-registered top set, it has the lowest "
      "large-arbiter dev bpb:")
    w("")
    w("| candidate | large dev bpb | Δ for SP-32k against it, 95 % CI | p | Holm p (3 pairs) |")
    w("|---|---|---|---|---|")
    w("| `R2-A4-SPnat-D2-32k` (SP-32k) | %.5f | – | – | – |" % C[SP32]["large_mean_bpb"])
    for c in (MG32, MG48):
        x = P[k(SP32, c)]
        w("| %s | %.5f | %s | %s | %s |" % (t(c), C[c]["large_mean_bpb"], dci(x), pv(x["p"]),
                                          pv(L["holm_within_top_set_pairs"][k(SP32, c)])))
    w("")
    w("- **Both CIs exclude 0.** SP-32k is the lowest of the three in %.2f %% of the 10,000 replicates. It keeps that "
      "place when any single one of the 32 clusters is left out."
      % (100 * L["p_first_within_top_set"][SP32]))
    w("- **The rule needs only the lowest mean.** The CIs show that, at this scale, the choice is not a coin flip.")
    w("- **The main caveat is the 2 seeds per candidate** (§5):")
    w("  - with a conservative seed-noise model (each candidate's larger observed seed s.d.), both CIs still exclude 0;")
    w("  - with a pessimistic stress model (%.3f %% seed s.d. for every candidate), they include 0. SP-32k is still "
      "lowest in %.0f %% of replicates." % (pooled_sd, 100 * S["parametric_pooled18_stress"]["p_first_within_top_set"][SP32]))
    w("")
    w("**Δ vs the baseline `A1-P1r3-D2-16k`, large arbiter:**")
    w("")
    for c in (SP32, MG32, MG48, A132):
        x = L["vs_baseline"][c]
        w("- %s: %s, p %s" % (t(c), dci(x), pv(x["p"])))
    w("")
    w("All three top-set members improve on the baseline. The BPE 32k tokenizer does not: its CI straddles 0.")
    w("")
    w("**From the confirm scale to the large scale, vocabulary-size effects shrink.** 'Change' is the large-scale Δ "
      "minus the confirm-scale Δ, in percentage points (pp). Each change has its own paired bootstrap CI.")
    w("")
    w("| contrast | what it varies | confirm Δ | large Δ | change (pp) |")
    w("|---|---|---|---|---|")
    for a, b, what in [(A132, BASE, "vocabulary size (BPE, 16k to 32k)"),
                       (MG48, MG32, "vocabulary size (MinGram, 32k to 48k)"),
                       (SP32, A132, "algorithm at equal size (Unigram vs BPE, 32k)"),
                       (MG32, A132, "algorithm at equal size (MinGram vs BPE, 32k)")]:
        x = X[k(a, b)]
        w("| %s − %s | %s | %s | %s | %+.2f %s |" % (t(a), t(b), what, pc(x["confirm_pct"]), pc(x["large_pct"]),
                                                   x["change_pp"], ci(x["change_ci95_pp"])))
    w("")
    w("- **Vocabulary size:** the effect vanishes for BPE, and it reverses for MinGram 32k to 48k.")
    w("- **Algorithm at equal size:** the effect persists. For Unigram it grows.")
    w("- **This pattern fits the audit's parameter confound** (details in §6.3). The confirm-scale arbiter has 1.77M "
      "non-embedding parameters. There, the embeddings are 64–84 % of all parameters, so a larger vocabulary "
      "buys model capacity. The large arbiter has 10.63M non-embedding parameters, and the extra embedding rows no "
      "longer help.")
    w("- **Model size, epochs and LR changed together** between the two scales, so the pattern supports the confound "
      "but does not isolate it.")
    w("")
    w("**Total parameters per model** (full table in §7):")
    w("")
    w("| candidate | confirm (d=192, L=4) | large (d=384, L=6) |")
    w("|---|---|---|")
    for c in (BASE, A132, SP32, MG32, MG48):
        a, b = PR[c]["confirm"], PR[c]["large"]
        w("| %s | %.2fM (%.2fx) | %.2fM (%.2fx) |" % (t(c), a["params_total"] / 1e6, a["params_ratio_vs_baseline"],
                                                    b["params_total"] / 1e6, b["params_ratio_vs_baseline"]))
    w("")

    # ---------------- 1. inputs and checks
    w("## 1. Inputs and checks")
    w("")
    w("- **Large records.** There are 10 files, `lm/colab_results/77e1368773fc/results/large__<candidate>__lr5e-4__s{1,2}.json`.")
    w("  - They were copied from `G:\\My Drive\\hindko_lm_out\\77e1368773fc\\results\\`. Each copy's sha256 equals "
      "the Drive file's (the per-file hashes are in the JSON).")
    w("  - Every record has status ok, stage `large`, `preregistered: false`, 7,325/7,325 steps and 2 epochs.")
    w("  - Every record has the same `hk_lm.py` (sha256 `%s…`) and the same bundle, `77e1368773fc`."
      % inp["large_hk_lm_sha256"][0][:12])
    w("  - Every record's bpb was recomputed bit-exactly as Σbits/Σbytes.")
    w("  - Every record covers the same 836 dev_strict uids, in the same order and with the same bytes, as the Stage 4 "
      "records of record.")
    w("  - No other `large__*` file exists.")
    sw = o["large_lr_sweep"]
    w("- **Large LR sweep** (baseline, seed 1): %s. The sweep chose 5e-4, which is the **lowest value of its grid**, "
      "as 1e-3 was at Stage 4."
      % ", ".join("%s → %.5f" % (lrf(e["lr"]), e["bpb"]) for e in sw))
    assert next(e for e in sw if e["lr"] == 5e-4)["per_doc_bits_identical_to_large_s1"]
    w("  - The 5e-4 sweep run and `large__A1-P1r3-D2-16k__lr5e-4__s1` have identical per-document bits, so the "
      "large runs are bitwise reproducible.")
    w("  - `large_lr_choice.json` was copied next to the results as well.")
    w("- **Stage 4 'confirm' records of record** come from `lm/colab_results/results_table.json` (sha256 `%s…`), "
      "loaded with `decide.load_stage`."
      % inp["results_table_sha256"][:12])
    w("  - That loader re-verifies each file's sha256 and bit-exact bpb, and it drops duplicate copies.")
    w("  - The baseline and `A1-P1r3-D2-32k` records of record are the round-1 runs (bundle `1d24425d2d64`). Their "
      "round-2 copies in `77e1368773fc`, the bundle the large arbiter used, are bitwise identical (DECISION.md §1).")
    w("- **Clusters:** 32, with book 12, newspaper 15 and web 5. The mapping is the same as in DECISION.md "
      "(`decide.dev_clusters`), and it agrees with `dev_clusters.csv`.")
    w("- **Cross-check against `decision.json`.**")
    w("  - The confirm-scale Δs recomputed here for the same 5 candidates equal the numbers of record.")
    w("  - Their 95 %% CI endpoints differ by at most %.3f pp. The cause is that the seed-draw stream differs with 5 "
      "candidates instead of 18." % o["confirm_same5"]["max_ci_endpoint_diff_pp"])
    w("")

    # ---------------- 2. method
    w("## 2. Method, as run")
    w("")
    w("- **The bootstrap is DECISION.md §2 unchanged**, via `decide.Boot`, `decide.compare` and `decide.holm`:")
    w("  - 10,000 replicates with `numpy.random.default_rng(12345)`;")
    w("  - the 32 clusters are resampled with replacement within each source;")
    w("  - each candidate's seeds are resampled with replacement, independently;")
    w("  - bpb = Σbits ÷ Σbytes, averaged over the resampled seeds;")
    w("  - the RNG order is the cluster draws, then the seed draws, with candidates in sorted id order: %s."
      % ", ".join(t(c) for c in o["protocol"]["candidate_rng_order"]))
    w("  - Δ% = (bpb_A − bpb_B)/bpb_B per replicate, with a 95 % percentile CI and p = 2·min(P(Δ*≤0), P(Δ*≥0)).")
    w("- **What differs from DECISION.md:**")
    w("  - There are 5 candidates with 2 seeds each, because the large stage ran only seeds 1 and 2.")
    w("  - The TOST margin is δ = 0.3 %% of the large baseline mean = %.6f bpb." % delta_abs)
    w("  - Holm is applied (report-only) to two families: the 3 pairs inside the top set, and the 4 comparisons "
      "against the large-scale best.")
    w("- **Scale comparison:**")
    w("  - One joint bootstrap draws the clusters once, then the seed draws for the large runs, then those for the "
      "confirm runs.")
    w("  - Its large block reproduces the primary large replicates exactly (asserted), so every large-scale number "
      "here comes from the same replicates.")
    w("  - The confirm and large Δs of a pair are paired over documents. The change is Δ_large − Δ_confirm per "
      "replicate, in pp.")
    w("- **A 2-seed caveat.** Resampling n seeds with replacement shrinks the seed variance of the mean by (n−1)/n:")
    w("  - 0.8 at Stage 4's 5 seeds (the audit's figure);")
    w("  - **0.5 at 2 seeds.**")
    w("  - §5 therefore repeats the key numbers under three seed-noise variants.")
    w("")

    # ---------------- 3. vs baseline
    w("## 3. Large arbiter: every candidate against the baseline")
    w("")
    w("| # | candidate | top set | V | total params | bpb s1 / s2 | mean bpb | seed s.d. | Δ vs baseline, 95 % CI | p | Δ vs best, 95 % CI | Holm p vs best (4) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(L["order"]):
        e = C[c]
        vb = L["vs_baseline"].get(c)
        vbest = e["large_vs_best"]
        w("| %d | %s | %s | %dk | %.2fM | %.5f / %.5f | %.5f | %.3f %% | %s | %s | %s | %s |" % (
            i + 1, t(c), "**yes**" if c in TOP else "no", PR[c]["large"]["n_vocab"] // 1024,
            PR[c]["large"]["params_total"] / 1e6, e["large_seed_bpb"][0], e["large_seed_bpb"][1], e["large_mean_bpb"],
            e["large_seed_sd_rel_pct"], "baseline" if vb is None else dci(vb), "–" if vb is None else pv(vb["p"]),
            "best" if vbest is None else dci(vbest), "–" if vbest is None else pv(e["large_holm_p_vs_best"])))
    w("")
    w("- **BPE 32k vs BPE 16k:** %s, p %s." % (dci(P[k(A132, BASE)]), pv(P[k(A132, BASE)]["p"])))
    w("  - In the primary bootstrap it is also TOST-equivalent within ±0.3 %%: the 90 %% CI is [%+.2f, %+.2f] %% of the "
      "baseline bpb." % tuple(ci90_a132))
    w("  - Under the seed-noise variants of §5, it is not TOST-equivalent. Read it as 'no detectable difference', "
      "not 'equal'.")
    w("- **The seed s.d.s here come from 2 seeds each, so they are very imprecise.** They range from 0.011 % (the "
      "baseline) to 0.207 % (`A1-P1r3-D2-32k`).")
    w("")

    # ---------------- 4. pairwise
    w("## 4. Pairwise comparisons inside the top set")
    w("")
    w("| pair (Δ = a vs b) | large Δ, 95 % CI | p | Holm p (3) | clusters where a is better (of 32) | their share of dev bytes | leave-one-cluster-out range | confirm Δ, 95 % CI (same 5) |")
    w("|---|---|---|---|---|---|---|---|")
    for a, b in [(SP32, MG32), (SP32, MG48), (MG48, MG32)]:
        x = P[k(a, b)]
        cv = L["cluster_view"][k(a, b)]
        lo = L["leave_one_cluster_out"][k(a, b)]
        w("| %s vs %s | %s | %s | %s | %d | %.1f %% | %s to %s | %s |" % (
            t(a), t(b), dci(x), pv(x["p"]), pv(L["holm_within_top_set_pairs"][k(a, b)]), cv["n_clusters_a_better"],
            100 * cv["bytes_share_a_better"], pc(lo["min_pct"]), pc(lo["max_pct"]), dci(CP[k(a, b)])))
    w("")
    rd = L["rank_distribution_within_top_set"]
    w("- **Rank within the top set across the 10,000 replicates:**")
    for c in (SP32, MG32, MG48):
        w("  - %s: 1st %.2f %%, 2nd %.2f %%, 3rd %.2f %%." % (t(c), 100 * rd[c]["1"], 100 * rd[c]["2"], 100 * rd[c]["3"]))
    w("- **Leaving any one of the 32 clusters out never changes the order** SP-32k < MinGram-32k < MinGram-48k.")
    w("- **The PLAN §6–§7 comparison family, applied at the large scale** (report-only; every candidate against the "
      "best, Holm over 4): all four are "
      "significantly worse than SP-32k. The largest adjusted p is %s, for `R2-A10-MinGram-P1r3-D2-32k`."
      % pv(max(L["holm_vs_best"].values())))
    w("  - At the confirm scale, by contrast, the whole top set was statistically tied (DECISION.md §3).")
    w("")
    w("**By source** (the same replicates restricted to one source's clusters; report-only):")
    w("")
    w("| source | clusters | bytes | SP-32k vs MinGram-32k: confirm → large | SP-32k vs MinGram-48k: confirm → large | MinGram 48k vs 32k: confirm → large |")
    w("|---|---|---|---|---|---|")
    for src in ("book", "newspaper", "web"):
        e = L["per_source"][src]
        cs = o["confirm_same5"]["per_source"][src]
        cells = []
        for a, b in [(SP32, MG32), (SP32, MG48), (MG48, MG32)]:
            cells.append("%s → %s" % (dci(cs[k(a, b)]), dci(e["pairs"][k(a, b)])))
        w("| %s | %d | %s | %s |" % (src, e["n_clusters"], format(e["bytes"], ","), " | ".join(cells)))
    w("")
    book_share = 100 * L["per_source"]["book"]["bytes"] / o["clusters"]["n_bytes"]
    w("- **Books:** SP-32k's large-scale lead comes mostly from the books, which are 12 clusters and %.0f %% of dev "
      "bytes. Against MinGram-32k it is %s there." % (book_share, dci(book[k(SP32, MG32)])))
    w("- **Newspapers:** at the confirm scale, SP-32k was significantly *worse* than both MinGram builds (%s vs "
      "MinGram-32k). At the large scale that gap closes to a tie: %s."
      % (dci(cnews[k(SP32, MG32)]), dci(news[k(SP32, MG32)])))
    w("- **Implication:** for news-like text, the released default has no demonstrated edge over MinGram-32k at "
      "either scale. At the confirm scale it was worse there.")
    w("- **Web:** 5 clusters and 34 KB, so its CIs are wide.")
    w("")

    # ---------------- 5. seed sensitivity
    w("## 5. Sensitivity to having only 2 seeds")
    w("")
    w("Each variant reuses the primary replicates' cluster draws. They differ only in how seed noise enters:")
    w("")
    w("- **rescaled:** each replicate's seed deviation is multiplied by √(n/(n−1)) = √2, which undoes the 0.5 "
      "variance shrink.")
    w("- **own s.d.:** clusters are resampled, seeds are averaged, and relative seed noise N(0, σ_c²/2) is added per "
      "candidate.")
    w("  - σ_c is the larger of the candidate's large 2-seed s.d. and its confirm 5-seed s.d.: %s."
      % ", ".join("%s %.3f %%" % (t(c), S["parametric_own_sd"]["sigma_rel_pct"][c])
                  for c in (BASE, A132, SP32, MG32, MG48)))
    w("  - The normal draws come from `default_rng(12346)`.")
    w("- **stress:** the same, with σ = %.3f %% for every candidate." % pooled_sd)
    w("  - That value is the pooled confirm 5-seed s.d. over all 18 candidates (DECISION.md §7.6).")
    w("  - It is inflated by the two SentencePiece D1 arms, and it is about 2 to 40 times the large-scale seed s.d.s "
      "observed here. It is a deliberately pessimistic bound.")
    w("")
    w("| variant | SP-32k vs MinGram-32k | SP-32k vs MinGram-48k | MinGram 48k vs 32k | BPE 32k vs 16k | P(SP-32k lowest in top set) |")
    w("|---|---|---|---|---|---|")
    rows = [("primary (DECISION.md bootstrap)", {"pairs": P, "p_first_within_top_set": L["p_first_within_top_set"]})]
    rows += [("rescaled seed bootstrap", S["rescaled_seed_bootstrap"]), ("own s.d. (parametric)", S["parametric_own_sd"]),
             ("stress, %.3f %% (parametric)" % pooled_sd, S["parametric_pooled18_stress"])]
    for name, e in rows:
        cells = ["%s, p %s" % (dci(e["pairs"][k(a, b)]), pv(e["pairs"][k(a, b)]["p"]))
                 for a, b in [(SP32, MG32), (SP32, MG48), (MG48, MG32), (A132, BASE)]]
        w("| %s | %s | %.2f %% |" % (name, " | ".join(cells), 100 * e["p_first_within_top_set"][SP32]))
    w("")
    w("- **SP-32k vs both MinGram builds survives** the rescaled and the own-s.d. variants.")
    w("- **MinGram 48k vs 32k (%s) does not survive** the own-s.d. variant: p %s there."
      % (pc(P[k(MG48, MG32)]["delta_pct"]), pv(S["parametric_own_sd"]["pairs"][k(MG48, MG32)]["p"])))
    w("- **Under the stress bound, none of the within-top-set pairs is significant.** SP-32k is still lowest in "
      "%.0f %% of replicates. In short: SP-32k is lowest under every variant, and its lead is significant unless the "
      "large-scale seed noise is far above what was observed."
      % (100 * S["parametric_pooled18_stress"]["p_first_within_top_set"][SP32]))
    w("")

    # ---------------- 6. scale change
    w("## 6. Confirm scale vs large scale")
    w("")
    w("### 6.1 Ranking")
    w("")
    w("| candidate | V | confirm bpb (5 seeds) | confirm rank of 5 (of 17) | confirm bpb, seeds 1–2 only (rank) | large bpb (seeds 1–2) | large rank | Δ vs baseline: confirm → large |")
    w("|---|---|---|---|---|---|---|---|")
    for c in o["confirm_same5"]["order"]:
        e = C[c]
        chg = "baseline" if c == BASE else "%s → %s" % (pc(X[k(c, BASE)]["confirm_pct"]), pc(X[k(c, BASE)]["large_pct"]))
        w("| %s | %dk | %.5f | %d (%d) | %.5f (%d) | %.5f | %d | %s |" % (
            t(c), PR[c]["large"]["n_vocab"] // 1024, e["confirm_mean_bpb"], e["confirm_rank_of5"],
            e["confirm_rank_of17_decision"], e["confirm_seeds12_mean_bpb"], e["confirm_seeds12_rank_of5"],
            e["large_mean_bpb"], e["large_rank_of5"], chg))
    w("")
    kt = o["scale_change"]["kendall_tau_b_confirm_vs_large"]
    w("- **2 of the 10 candidate pairs change order** (Kendall τ_b = %.2f), and both are vocabulary-size pairs:"
      % kt["kendall_tau_b"])
    w("  - BPE 32k vs BPE 16k;")
    w("  - MinGram 48k vs MinGram 32k.")
    w("  - Every other pair keeps its order.")
    w("- **The change is not an artefact of the seed count.** With Stage 4 restricted to seeds 1–2, like-for-like "
      "with the large arbiter:")
    w("  - BPE 32k vs 16k is %s, and MinGram 48k vs 32k is %s. Both reverse at the large scale."
      % (dci(C2[k(A132, BASE)]), dci(C2[k(MG48, MG32)])))
    w("  - The confirm-scale best on seeds 1–2 is `R2-A10-MinGram-P1r3-D2-48k`, which is 3rd of the top set at the "
      "large scale.")
    w("")
    w("### 6.2 Contrasts, paired across scales")
    w("")
    w("| contrast | type | confirm Δ, 95 % CI | large Δ, 95 % CI | change (pp), 95 % CI | p (change) |")
    w("|---|---|---|---|---|---|")
    rows = [(A132, BASE, "vocabulary size, BPE 16k → 32k"), (MG48, MG32, "vocabulary size, MinGram 32k → 48k"),
            (SP32, A132, "algorithm at 32k: Unigram vs BPE"), (MG32, A132, "algorithm at 32k: MinGram vs BPE"),
            (MG48, A132, "MinGram 48k vs BPE 32k (algorithm + size)"),
            (SP32, MG32, "top set: SP-32k vs MinGram-32k"), (SP32, MG48, "top set: SP-32k vs MinGram-48k"),
            (SP32, BASE, "vs baseline"), (MG32, BASE, "vs baseline"), (MG48, BASE, "vs baseline")]
    for a, b, what in rows:
        x = X[k(a, b)]
        w("| %s vs %s | %s | %s | %s | %+.2f %s | %s |" % (t(a), t(b), what, dci(CP[k(a, b)]), dci(P[k(a, b)]),
                                                         x["change_pp"], ci(x["change_ci95_pp"]), pv(x["p"])))
    w("")
    w("The confirm column is recomputed on these 5 candidates. Its point Δs equal `decision.json`, and its CIs are "
      "within %.2f pp of it (§1)." % o["confirm_same5"]["max_ci_endpoint_diff_pp"])
    w("")
    w("### 6.3 Reading: vocabulary-size effects shrink, and the audit's parameter confound fits")
    w("")
    w("- **The audit's point** (`analysis/audit/audit_lm.json`; AMENDMENT_1.md):")
    w("  - PLAN §5 fixes the non-embedding size, not the total size, so the total parameters grow with the vocabulary.")
    w("  - At the confirm scale, the 48k model has %.2fx the baseline's parameters and the 32k models have %.2fx."
      % (PR[MG48]["confirm"]["params_ratio_vs_baseline"], PR[SP32]["confirm"]["params_ratio_vs_baseline"]))
    w("  - The token embedding is %.0f–%.0f %% of all parameters there."
      % (100 * PR[BASE]["confirm"]["embedding_share"], 100 * PR[MG48]["confirm"]["embedding_share"]))
    w("  - Tie-breaker (b), bytes/token, rises with vocabulary size. So within a tied set it picks the largest "
      "vocabulary, and with it the most parameters.")
    w("- **What the large arbiter changes.** The non-embedding part grows 6x, from 1.77M to 10.63M:")
    w("  - the embedding share falls to %.0f–%.0f %%;"
      % (100 * PR[BASE]["large"]["embedding_share"], 100 * PR[MG48]["large"]["embedding_share"]))
    w("  - the parameter ratios fall to %.2fx (48k) and %.2fx (32k)."
      % (PR[MG48]["large"]["params_ratio_vs_baseline"], PR[SP32]["large"]["params_ratio_vs_baseline"]))
    w("- **What happens to the vocabulary-size gains:**")
    w("  - BPE 16k → 32k goes from %s to %s, a change of %+.2f pp %s."
      % (pc(X[k(A132, BASE)]["confirm_pct"]), pc(X[k(A132, BASE)]["large_pct"]), X[k(A132, BASE)]["change_pp"],
         ci(X[k(A132, BASE)]["change_ci95_pp"])))
    w("  - MinGram 32k → 48k goes from %s to %s, a change of %+.2f pp %s."
      % (pc(X[k(MG48, MG32)]["confirm_pct"]), pc(X[k(MG48, MG32)]["large_pct"]), X[k(MG48, MG32)]["change_pp"],
         ci(X[k(MG48, MG32)]["change_ci95_pp"])))
    w("  - Both changes are significant.")
    w("  - At the large scale, each step adds %.2fM embedding parameters (BPE 16k → 32k; MinGram 32k → 48k), and "
      "they buy nothing, or worse."
      % ((PR[A132]["large"]["token_embedding_tied"] - PR[BASE]["large"]["token_embedding_tied"]) / 1e6))
    w("- **At equal vocabulary, where the parameter count is the same, the algorithm effects persist:**")
    w("  - MinGram vs BPE at 32k: %s → %s, a change of %+.2f pp %s, not significant;"
      % (pc(X[k(MG32, A132)]["confirm_pct"]), pc(X[k(MG32, A132)]["large_pct"]), X[k(MG32, A132)]["change_pp"],
         ci(X[k(MG32, A132)]["change_ci95_pp"])))
    w("  - Unigram vs BPE at 32k grows: %s → %s, a change of %+.2f pp %s."
      % (pc(X[k(SP32, A132)]["confirm_pct"]), pc(X[k(SP32, A132)]["large_pct"]), X[k(SP32, A132)]["change_pp"],
         ci(X[k(SP32, A132)]["change_ci95_pp"])))
    w("- **The audit's parameter-adjusted confirm ranking already matched the large one.** The audit measured each "
      "candidate's residual against the BPE bpb-vs-log(parameters) curve at the confirm scale:")
    w("  - SP-32k %s, MinGram-32k %s, MinGram-48k %s;" % (pc(resid[SP32], 3), pc(resid[MG32], 3), pc(resid[MG48], 3)))
    w("  - BPE-32k %s and baseline %s." % (pc(resid[A132], 3), pc(resid[BASE], 3)))
    w("  - That is the order SP-32k < MinGram-32k < MinGram-48k.")
    w("  - Once parameters were accounted for, the confirm scale already gave the large arbiter's order.")
    w("- **Why this is consistent with the confound, not proof of it:**")
    w("  - Three things changed together between the scales: model width and depth, epochs (1 → 2), and peak LR "
      "(1e-3 → 5e-4).")
    w("  - Both LRs were chosen on the 16k baseline, and both sit at the low edge of their grids; the Stage 4 finalist "
      "re-sweep tried only higher LRs. Larger vocabularies may prefer a different LR at either scale.")
    lt20 = {r["id"]: r["tiebreak"]["train_lt20_pct"] for r in dec["ranking"]}
    assert lt20[SP32] > lt20[MG32]
    w("  - A second mechanism may contribute: rare tokens. %.1f %% of MinGram-48k's learned tokens occur fewer than 20 "
      "times in train_D1 (DECISION.md §4), and a second epoch repeats those occurrences rather than adding new ones. "
      "This does not explain SP-32k's lead over MinGram-32k, though: SP-32k has more rare tokens (%.1f %% vs %.1f %%)."
      % (lt20[MG48], lt20[SP32], lt20[MG32]))
    w("  - No iso-parameter run, such as a 48k model with a narrower embedding, was made, so these explanations are "
      "not separated.")
    w("")

    # ---------------- 7. params
    w("## 7. Parameters and compute per model")
    w("")
    w("Tied input/output embeddings, learned positions. The training-compute estimate is 6 × (non-embedding + tied "
      "output) × tokens seen, as in the audit.")
    w("")
    w("| candidate | scale | V | ctx tokens | total | token embedding | position | non-embedding | embedding share | total × baseline | tokens seen | train FLOPs × baseline | T4 train s |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in (BASE, A132, SP32, MG32, MG48):
        for sc in ("confirm", "large"):
            e = PR[c][sc]
            w("| %s | %s (d=%d, L=%d, %d ep, LR %s) | %s | %d | %s | %s | %s | %s | %.1f %% | %.3f | %.2fM | %.3f | %.0f |" % (
                t(c), sc, e["d"], e["L"], e["epochs"], lrf(e["peak_lr"]), format(e["n_vocab"], ","), e["ctx_tokens"],
                format(e["params_total"], ","), format(e["token_embedding_tied"], ","),
                format(e["position_embedding"], ","), format(e["non_embedding"], ","), 100 * e["embedding_share"],
                e["params_ratio_vs_baseline"], e["tokens_seen_mean"] / 1e6, e["flops_ratio_vs_baseline"],
                e["train_s_mean_T4"]))
    w("")
    spread = max(max(PR[c][sc]["params_total"] for c in (A132, SP32, MG32)) -
                 min(PR[c][sc]["params_total"] for c in (A132, SP32, MG32)) for sc in ("confirm", "large"))
    w("- **At both scales, the three 32k models are iso-parameter within %s parameters.** The only difference is "
      "the position table, which depends on ctx." % format(spread, ","))
    w("  - So the SP-32k vs MinGram-32k and algorithm-at-32k contrasts are free of the parameter confound.")
    w("  - The SP-32k vs MinGram-48k contrast is not: MinGram-48k has +6.29M parameters at the large scale.")
    w("")

    # ---------------- 8. claims
    w("## 8. What this supports, and what it does not")
    w("")
    w("**Supported, on dev_strict (report-only arbiter):**")
    w("")
    w("- **The amended rule's outcome.** Within the pre-registered top set, SP-32k has the lowest large-arbiter dev "
      "bpb:")
    w("  - vs MinGram-32k: %s, p %s;" % (dci(P[k(SP32, MG32)]), pv(P[k(SP32, MG32)]["p"])))
    w("  - vs MinGram-48k: %s, p %s;" % (dci(P[k(SP32, MG48)]), pv(P[k(SP32, MG48)]["p"])))
    w("  - 2 seeds each, hierarchical cluster bootstrap.")
    w("  - This is the evidence AMENDMENT_1.md §5 asks the claim to state.")
    w("- **The fallback ordering of AMENDMENT_1.md §3 matches the large arbiter.** If exact HF equivalence fails, the "
      "fallback is MinGram-32k, and MinGram-48k is higher than it at this scale: %s, p %s. The difference does not "
      "survive the own-s.d. seed variant (§5)." % (dci(P[k(MG48, MG32)]), pv(P[k(MG48, MG32)]["p"])))
    w("- **The reason for replacing tie-breaker (b) is borne out.** Bytes/token rewards vocabulary size, and at the "
      "large scale the vocabulary-size gains do not carry over.")
    w("")
    w("**Not supported, or not shown:**")
    w("")
    w("- **Anything on test.** No test LM number was read. The one-shot test result will be reported whatever it is "
      "(AMENDMENT_1.md §4–§5).")
    w("- **A pre-registered claim.** The large arbiter was added beyond PLAN §8(ii), and the top-set members are "
      "exploratory round-2 tokenizers selected on dev, so winner's-curse optimism applies.")
    w("- **Robustness to much larger seed noise.** Under the stress bound (§5), the within-top-set differences are not "
      "significant.")
    w("- **An edge on news-like text.** On the newspaper clusters, SP-32k and MinGram-32k are tied at the large scale "
      "(§4).")
    w("- **That vocabulary size never matters.** Only two model scales were tested, and the LR was tuned on the "
      "baseline only.")
    w("- **The packaging condition of AMENDMENT_1.md §3.** Whether the HF `tokenizer.json` exactly equals the "
      "canonical SentencePiece + newline encoder is a separate check, not assessed here.")
    w("")

    # ---------------- 9. files
    w("## 9. Files and reproduction")
    w("")
    w("| file | content |")
    w("|---|---|")
    w("| `analysis/amendment1_stats.py` | every computation. It imports `decide.py` for the bootstrap, loaders and clusters, and reads no test data |")
    w("| `analysis/amendment1_report.py` | renders this file from the JSON and asserts every qualitative statement it makes |")
    w("| `analysis/amendment1_stats.json` | all numbers: per-seed bpb, all 10 pairs at both scales, per source, leave-one-cluster-out, sensitivities, parameters, input hashes |")
    w("| `lm/colab_results/77e1368773fc/results/large__*.json`, `large_lrsweep__*.json`, `../large_lr_choice.json` | the large-arbiter records, copied from the Colab Drive folder |")
    w("")
    w("To reproduce (single process, about a second, deterministic), run from `F:\\Hindko\\_tokenizer`: "
      "`PYTHONIOENCODING=utf-8 python analysis/amendment1_stats.py && PYTHONIOENCODING=utf-8 python analysis/amendment1_report.py`.")
    w("")

    with open(MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(W))
    print("wrote", MD, len(W), "lines")


if __name__ == "__main__":
    main()
