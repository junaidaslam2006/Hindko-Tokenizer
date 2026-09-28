# Hindko tokenizer: the pre-registered decision (PLAN §6–§7) on Stage 4, both rounds

Generated 2026-09-26T20:20:43Z by `analysis/decide.py` (sha256 `5ad1170722a6…`) and `analysis/report.py`. PLAN.md sha256 `d7a811df…` (equals FROZEN.json). Every number below is on **dev_strict** (836 documents, 1,445,513 bytes). **The test split was not read, encoded or scored.** Machine-readable twin: `analysis/decision.json`.

## 0. Decision

- **Chosen tokenizer: `R2-A10-MinGram-P1r3-D2-48k`.** It is MinGram (A10, as reimplemented from the paper), with pre-tokenizer P1r3, trained on the strict-train mix D2, with 49,152 tokens including the 64 specials. It is a stock HF `tokenizer.json`, and its ids equal the reference encoder on every dev document.
- **Ranking (PLAN §7.2)** is on the Stage 4 'confirm' arbiter (d=192, full permissive train split, 1 epoch, LR 1e-3, 5 seeds each). There are 17 ranked candidates: all 18 LM candidates of both rounds, minus the conditional rank 8, whose pre-registered condition is false (§8, deviation 6). The best mean is `R2-A4-SPnat-D2-32k` at 1.22584 bpb. The chosen tokenizer is ranked 2, at 1.22636 bpb: +0.04 % [-0.40, +0.57], p 0.8366 against the best.
- **Top set (PLAN §7.3): 3 members**, `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k`. Every other candidate is significantly worse than the best after Holm correction (16 comparisons). No candidate is TOST-equivalent to the best within ±0.3 % (δ = 0.003738 bpb).
- **Tie-breakers (PLAN §7.4)**, applied in order:
  - (a) HF-native exact encoding drops `R2-A4-SPnat-D2-32k`: the LM encoder is the native sp.model plus the newline wrapper (custom code), and its HF tokenizer.json export differs from the native ids on 0/836 dev_strict and 1/1,358 dev_permissive documents (equal-score ties; token counts identical), so it is not exact.
  - (b) Higher dev bytes/token keeps `R2-A10-MinGram-P1r3-D2-48k` (7.2237) over `R2-A10-MinGram-P1r3-D2-32k` (7.1258).
  - (c)–(e) were not reached.
  - The choice does not hinge on (a): the chosen tokenizer also has the highest bytes/token of the three.
- **Improvement claim (PLAN §7.5): allowed on dev.** Against the standard recipe `A1-P1r3-D2-16k`, Δ = -1.58 % [-1.87, -1.36] (p <1e-4). The 95 % CI excludes 0. `R2-A10-MinGram-P1r3-D2-48k` is 1.22636 bpb against 1.24607 for the baseline.
  - The chosen tokenizer and the whole top set come from the **exploratory round 2**, which was built after the Stage 3 and Stage 4 results were seen. The dev Δ is therefore conditioned on dev, and the one-shot test (PLAN §7.6) decides whether it holds.
  - A confirmatory reading supports an improvement too. Restricted to the pre-registered round 1, the same rule chooses `A1-P1r3-D2-32k`, with Δ = -0.81 % [-1.00, -0.66] against the baseline.
- **Test candidates, fixed now and before anything touches test** (sha256 of the id list `86e95a4ac05d…`):

| # | tokenizer | role |
|---|---|---|
| 1 | `R2-A10-MinGram-P1r3-D2-48k` | chosen (PLAN §7.6 one-shot test) |
| 2 | `A1-P1r3-D2-16k` | baseline (PLAN §7.6 one-shot test) |
| 3 | `R2-A4-SPnat-D2-32k` | report-only (top-set member; never re-selects); best mean dev bpb (rank 1); lost tie-breaker (a) to the chosen tokenizer |
| 4 | `R2-A10-MinGram-P1r3-D2-32k` | report-only (top-set member; never re-selects); next top-set member by mean dev bpb |

## 1. Inputs and checks

- **Results:** `lm/colab_results/results_table.json` (sha256 `aa315f1bf89e…`), with the per-run JSON records it lists. Each record's sha256 was re-verified, and each record's bpb was recomputed bit-exactly as Σbits/Σbytes.
  - Records used: 90 Stage-4 'confirm' records (18 candidates × 5 seeds) and 54 Stage-3 'screen' records (18 × 3, report-only).
  - Bundles: `1d24425d2d64` (round 1) and `77e1368773fc` (round 2).
- **Reference arms in both bundles are counted once.** The 32 round-2 copies (4 arms × (5 confirm + 3 screen) seeds) were compared with the round-1 runs. Their per-document bits, uids and bytes are identical: True. The copies were dropped.
- **Clusters (PLAN §6)** come from `splits/split_manifest.jsonl`, whose sha256 `76582d3a…` matches the frozen hash.
  - Only the 836 dev_strict uids were looked up. Every other row was skipped on its uid alone.
  - Cluster = the manifest `group`, with the per-record web groups (`web:<site>:record:<uid>`) collapsed to `web:<site>`.
  - Result: **32 clusters** (book 12, newspaper 15, web 5). The raw manifest has 41 groups (book 12, newspaper 15, web 14). The largest cluster is `book:9th Farma Jamda Jeevay Sheen-Shaukat 11-03-16`, with 17.0 % of dev bytes. The mapping agrees with `lm/colab_results/dev_clusters.csv`: True.
- **Gates (PLAN §7.1):** G1–G5 pass for all 18 candidates, after the pre-declared G2 remedy (read from each `summary.json`; G5 is n/a for byte-level vocabularies). The tokenizer sha256 in every summary equals the one the LM runs used. Gate failures: none.

## 2. Method, as run

- **Hierarchical cluster bootstrap:** 10,000 replicates, `numpy.random.default_rng(12345)`. In each replicate:
  - the clusters are resampled with replacement *within* each source (book 12, newspaper 15, web 5), keeping all their documents;
  - each candidate's 5 seeds are resampled with replacement, independently for each candidate;
  - bpb = Σbits ÷ Σbytes over the resampled documents, averaged over the resampled seeds.
  - All candidates share the replicate's cluster draw, so every Δ is paired over documents.
  - RNG order: the cluster draws come first (sources in sorted order), then the seed draws (candidates in sorted id order).
  - An independent per-replicate loop implementation with another RNG stream reproduced the CIs to within ±0.01 pp.
- **Outputs:**
  - Δ% is computed per replicate as (bpb_A − bpb_B)/bpb_B, with the 95 % percentile CI.
  - p = 2·min(P(Δ*≤0), P(Δ*≥0)). Its resolution is 1e-4, and p = 0 is shown as <1e-4.
- **Holm–Bonferroni** covers the 16 comparisons of every ranked candidate against the best-mean candidate, at α = 0.05. 'Significantly worse' means a Holm-adjusted p ≤ 0.05 with a positive Δ.
- **TOST:** a candidate is equivalent if the 90 % CI of Δ lies inside ±δ, with δ = 0.3 % × baseline mean bpb = 0.003738 bpb.
- **Top set** = the best, plus every candidate not significantly worse, plus every candidate equivalent within δ.
- **Tie-breakers** are lexicographic, in PLAN order, without tolerance. Each tokenizer's metrics come from its `summary.json` (harness, dev_strict), plus `meta.json` for the SentencePiece HF-export check.
  - (a) A stock HF `tokenizer.json` reproduces the ids the LM used on 100 % of dev (strict and permissive).
  - (b) Dev bytes/token.
  - (c) Learned tokens with train_D1 frequency < 20. train_D1 is the Stage 4 training stream.
  - (d) The mean over the four PLAN §4.2 perturbations of the share of affected words that get re-segmented.
  - (e) The effective vocabulary.
- **Improvement claim:** the chosen tokenizer's Δ against `A1-P1r3-D2-16k` must have a 95 % CI that excludes 0. This is the single pre-registered comparison, and it is not Holm-corrected.

## 3. Ranking (Stage 4 'confirm', 5 seeds each)

Δ% is relative to the reference's bpb, with the 95 % bootstrap CI in brackets. The TOST column shows the 90 % CI of Δ against the best, in % of the best's bpb. δ = ±0.300 % of the baseline, which is ±0.305 % of the best's bpb.

| # | candidate | origin | V | mean bpb | seed s.d. | Δ vs baseline | Δ vs best | p vs best | Holm p | 90 % CI vs best | top set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `R2-A4-SPnat-D2-32k` | R2 exploratory (R2-3) | 32k | 1.22584 | 0.23 % | -1.62 % [-2.11, -1.24] | best | – | – | – | **yes** |
| 2 | `R2-A10-MinGram-P1r3-D2-48k` | R2 exploratory (R2-5) | 48k | 1.22636 | 0.15 % | -1.58 % [-1.87, -1.36] | +0.04 % [-0.40, +0.57] | 0.8366 | 0.8366 | [-0.34, +0.47] | **yes** |
| 3 | `R2-A10-MinGram-P1r3-D2-32k` | R2 exploratory (R2-2) | 32k | 1.22844 | 0.21 % | -1.41 % [-1.66, -1.19] | +0.21 % [-0.17, +0.65] | 0.2900 | 0.5800 | [-0.11, +0.58] | **yes** |
| 4 | `R2-A4-SPnat-D2-48k` | R2 exploratory (R2-6) | 48k | 1.23205 | 0.38 % | -1.13 % [-1.69, -0.64] | +0.51 % [+0.14, +0.87] | 0.0078 | 0.0234 | [+0.19, +0.80] | no |
| 5 | `R2-A1-P1r3-D2-48k` | R2 exploratory (R2-4) | 48k | 1.23374 | 0.17 % | -0.99 % [-1.21, -0.78] | +0.64 % [+0.25, +1.12] | 0.0020 | 0.0080 | [+0.30, +1.05] | no |
| 6 | `A1-P1r3-D2-32k` | R1 pre-reg (6b) | 32k | 1.23597 | 0.07 % | -0.81 % [-1.00, -0.66] | +0.83 % [+0.47, +1.26] | <1e-4 | <1e-4 | [+0.51, +1.19] | no |
| 7 | `A4-SPnat-D1-16k` | R1 pre-reg (3) | 16k | 1.23925 | 0.81 % | -0.55 % [-1.30, +0.23] | +1.09 % [+0.43, +1.83] | 0.0008 | 0.0040 | [+0.53, +1.69] | no |
| 8 | `A10-MinGram-P1-D1-16k` | R1 pre-reg (4) | 16k | 1.24180 | 0.20 % | -0.34 % [-0.57, -0.10] | +1.30 % [+0.91, +1.79] | <1e-4 | <1e-4 | [+0.97, +1.72] | no |
| 9 | `A1-P1r3-D2-16k` | R1 pre-reg (1) | 16k | 1.24607 | 0.10 % | baseline | +1.65 % [+1.25, +2.15] | <1e-4 | <1e-4 | [+1.29, +2.09] | no |
| 10 | `A1-P1-D1-16k` | R1 added arm (extra-2) | 16k | 1.25430 | 0.16 % | +0.66 % [+0.46, +0.86] | +2.32 % [+1.94, +2.80] | <1e-4 | <1e-4 | [+1.97, +2.74] | no |
| 11 | `A7-PickyBPE-P1-D1-16k-tau0.9` | R1 pre-reg (7) | 16k | 1.25446 | 0.10 % | +0.67 % [+0.51, +0.84] | +2.34 % [+1.97, +2.77] | <1e-4 | <1e-4 | [+1.99, +2.73] | no |
| 12 | `A1-P1r3-D1-16k` | R1 added arm (extra-1) | 16k | 1.25532 | 0.18 % | +0.74 % [+0.51, +0.97] | +2.41 % [+1.98, +2.96] | <1e-4 | <1e-4 | [+2.03, +2.87] | no |
| 13 | `A1-P1r3-D2-8k` | R1 pre-reg (6a) | 8k | 1.25672 | 0.18 % | +0.86 % [+0.37, +1.29] | +2.52 % [+1.88, +3.24] | <1e-4 | <1e-4 | [+2.00, +3.13] | no |
| 14 | `A3-SPnat-D1-16k` | R1 pre-reg (9) | 16k | 1.26651 | 1.46 % | +1.64 % [+0.43, +2.96] | +3.32 % [+2.15, +4.61] | <1e-4 | <1e-4 | [+2.30, +4.39] | no |
| 15 | `R2-A6-SBPE-P1r3-D2-32k-t080` | R2 exploratory (R2-1) | 32k | 1.27375 | 0.12 % | +2.22 % [+1.82, +2.61] | +3.91 % [+3.53, +4.32] | <1e-4 | <1e-4 | [+3.60, +4.23] | no |
| 16 | `A6-SBPE-P1-D1-16k-t090` | R1 pre-reg (2) | 16k | 1.27485 | 0.20 % | +2.31 % [+2.02, +2.63] | +4.00 % [+3.62, +4.47] | <1e-4 | <1e-4 | [+3.65, +4.42] | no |
| – | `A6-SBPE-P1-D1-32k-t080` (conditional rank 8, not ranked) | R1 pre-reg (8) | 32k | 1.27960 | 0.13 % | +2.69 % [+2.25, +3.11] | +4.39 % [+4.04, +4.78] | <1e-4 | – | [+4.12, +4.67] | no |
| 17 | `A6-SBPE-P1-D1-16k-t080` | R1 pre-reg (5) | 16k | 1.28692 | 0.32 % | +3.28 % [+2.88, +3.71] | +4.98 % [+4.54, +5.52] | <1e-4 | <1e-4 | [+4.60, +5.41] | no |

Notes:
- 'Δ vs baseline' is descriptive for every row except the chosen one. PLAN pre-registers only the chosen-vs-baseline comparison, and it corrects only the family against the best.
- The Holm step-down rejects everything down to `R2-A4-SPnat-D2-48k` (raw p 0.0078, adjusted 0.0234). It stops at `R2-A10-MinGram-P1r3-D2-32k` (raw p 0.2900, adjusted 0.5800).
- The per-seed bpb of every run is in `decision.json` → `ranking[].seed_bpb`.

## 4. Top set and tie-breakers

| candidate | Δ vs best | (a) HF-native exact | (b) dev bytes/token | (c) train_D1 freq < 20 | (d) robustness | (e) vocab (effective) | order |
|---|---|---|---|---|---|---|---|
| `R2-A10-MinGram-P1r3-D2-48k` | +0.04 % [-0.40, +0.57] | yes | 7.2237 | 27246 (55.8 %) | 0.295 | 49152 | 1 |
| `R2-A10-MinGram-P1r3-D2-32k` | +0.21 % [-0.17, +0.65] | yes | 7.1258 | 11329 (34.9 %) | 0.316 | 32768 | 2 |
| `R2-A4-SPnat-D2-32k` | best | **no** | 7.0764 | 13182 (40.6 %) | 0.589 | 32763 | 3 |

Evidence for (a):
- `R2-A10-MinGram-P1r3-D2-48k`: a stock HF tokenizer.json that equals the reference encoder on 836/836 dev_strict and 1,358/1,358 dev_permissive documents (WAVES stage2_checks).
- `R2-A10-MinGram-P1r3-D2-32k`: a stock HF tokenizer.json that equals the reference encoder on 836/836 dev_strict and 1,358/1,358 dev_permissive documents (WAVES stage2_checks).
- `R2-A4-SPnat-D2-32k`: the LM encoder is the native sp.model plus the newline wrapper (custom code), and its HF tokenizer.json export differs from the native ids on 0/836 dev_strict and 1/1,358 dev_permissive documents (equal-score ties; token counts identical), so it is not exact.

How the decision falls out:
- (a) removes `R2-A4-SPnat-D2-32k`.
- (b) separates the two MinGram builds.
- (c)–(e) were never reached.
- (c) and (e) would favour the 32k MinGram (11329 vs 27246 tokens below 20; 32k vs 48k), but they come after (b).
- If (a) were read on dev_strict only, `R2-A4-SPnat-D2-32k` would count as HF-native: its export matches the native ids on 836/836 dev_strict documents. (b) would still choose `R2-A10-MinGram-P1r3-D2-48k`, since 7.2237 > 7.1258 > 7.0764 bytes/token.

Cost of the choice, report-only:
- The chosen 48k tokenizer has 11.26M total parameters in the arbiter, against 8.11M at 32k and 4.97M for the 16k baseline. The non-embedding parameters are the same 1.77M in all three.
- 27246 of its 48831 learned tokens (55.8 %) are seen fewer than 20 times in train_D1. That is the data-scarce regime of PLAN §4.3, and R3 (embedding norms) is where it would show.

## 5. Improvement claim (PLAN §7.5)

- `R2-A10-MinGram-P1r3-D2-48k` vs `A1-P1r3-D2-16k`: **Δ = -1.582 %, 95 % CI [-1.868, -1.356], p <1e-4** (bpb 1.22636 vs 1.24607; bootstrap s.e. 0.130 pp).
- The CI excludes 0, so the rule allows the claim. On dev its wording is: *R2-A10-MinGram-P1r3-D2-48k improves on the standard recipe A1-P1r3-D2-16k on dev_strict: Δ = -1.582 % bpb, 95 % CI [-1.868 %, -1.356 %] (hierarchical cluster bootstrap, 5 seeds each)*.
- Per source (report-only; the same replicates, restricted to one source's clusters):

| source | clusters | bytes | `R2-A4-SPnat-D2-32k` vs baseline | `R2-A10-MinGram-P1r3-D2-48k` vs baseline | `R2-A10-MinGram-P1r3-D2-32k` vs baseline |
|---|---|---|---|---|---|
| book | 12 | 878,168 | -1.89 % [-2.76, -1.37] | -1.53 % [-1.92, -1.19] | -1.29 % [-1.55, -1.05] |
| newspaper | 15 | 532,872 | -1.04 % [-1.42, -0.67] | -1.70 % [-1.99, -1.45] | -1.62 % [-1.96, -1.31] |
| web | 5 | 34,473 | -2.10 % [-3.63, -0.70] | -1.49 % [-2.72, -0.07] | -2.01 % [-3.45, -0.65] |

No source regresses: every per-source CI lies below 0. The web stratum has only 5 clusters and 34 KB, so its CIs are wide.

## 6. Test candidates (fixed before test; PLAN §7.6)

- **The list:** `R2-A10-MinGram-P1r3-D2-48k`, `A1-P1r3-D2-16k`, `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-32k`.
  - It was frozen in `decision.json` → `test_candidates_fixed` at 2026-09-26T20:20:43Z. The sha256 of the newline-joined ids is `86e95a4ac05dfa1401369f0cc1f0af7f88e809177dc0bf46ed22db217820648a`.
  - It is the chosen tokenizer, the baseline, and the two other top-set members, labelled report-only. The two other members are the whole rest of the top set.
- **Why these report-only members:**
  - `R2-A4-SPnat-D2-32k` is the best-mean candidate. It lost only on tie-breaker (a).
  - `R2-A10-MinGram-P1r3-D2-32k` is the remaining top-set member.
  - The best pre-registered round-1 tokenizer (`A1-P1r3-D2-32k`) is **not** in the top set: it is +0.83 % [+0.47, +1.26], p <1e-4 against the best. It therefore cannot be a report-only test candidate under this rule. The baseline comparison is the forking-paths guard on test.
- **What the TestBundle phase must do:**
  - Run G1 on test for the four tokenizers.
  - Score each of them once on strict test with the Stage 4 recipe: d=192, full permissive train, 1 epoch, LR 1e-3, seeds 1–5, GPU fp16, the same `hk_lm`. Re-use the Stage 4 models if the bundle kept them; otherwise retrain them. The runs were bitwise reproducible across sessions (COLLECT.md §5).
  - Report the chosen tokenizer's Δ against the baseline with the same bootstrap. The test split has 27 clusters (15 newspaper, 6 book, 6 web).
  - The report-only rows are never used to re-select. If test contradicts dev in sign, report it and do not re-select (PLAN §7.6).

## 7. Report-only analyses (same bootstrap; none of this enters the decision)

### 7.1 Vocabulary-size curve (A1 byte-level BPE, P1r3, D2)

| size | tokenizer | total params | ctx tokens | Stage 3 bpb | Stage 4 bpb | Stage 4 Δ vs previous size | Stage 3 Δ vs previous size | Stage 4 Δ vs 16k |
|---|---|---|---|---|---|---|---|---|
| 8k | `A1-P1r3-D2-8k` | 3.40M | 286 | 1.46235 | 1.25672 | – | – | +0.86 % [+0.37, +1.29] |
| 16k | `A1-P1r3-D2-16k` | 4.97M | 265 | 1.42254 | 1.24607 | -0.85 % [-1.27, -0.37], p 0.0010 | -2.72 % [-3.23, -2.30] | – |
| 32k | `A1-P1r3-D2-32k` | 8.11M | 251 | 1.40039 | 1.23597 | -0.81 % [-1.00, -0.66], p <1e-4 | -1.56 % [-1.79, -1.37] | -0.81 % [-1.00, -0.66] |
| 48k | `R2-A1-P1r3-D2-48k` | 11.26M | 245 | 1.39758 | 1.23374 | -0.18 % [-0.36, +0.04], p 0.0944 | -0.20 % [-0.56, +0.09] | -0.99 % [-1.21, -0.78] |

- **The curve flattens after 32k.**
  - 8k→16k gains -0.85 % [-1.27, -0.37] and 16k→32k gains -0.81 % [-1.00, -0.66] at Stage 4. Both steps are significant.
  - 32k→48k gains -0.18 % [-0.36, +0.04], p 0.0944, which is not significant.
  - The same holds in the other families: MinGram 32k→48k gives -0.17 % [-0.47, +0.11], p 0.2562, and SentencePiece Unigram 32k→48k gives +0.51 % [+0.14, +0.87], p 0.0078, which is *worse*.
- Stage 3 exaggerates the small-vocabulary penalty: 8k→16k is -2.72 % [-3.23, -2.30] there.
- Total parameters grow with the vocabulary, because PLAN §5 fixes non-embedding size, not total size. The curve is therefore not iso-parameter.

### 7.2 Data mix D2 vs D1, and pre-tokenizer P1r3 vs P1 (A1 BPE, 16k)

| contrast | Stage 4 | Stage 3 |
|---|---|---|
| D2 vs D1 (`A1-P1r3-D2-16k` − `A1-P1r3-D1-16k`) | -0.74 % [-0.96, -0.51], p <1e-4 | -1.05 % [-1.40, -0.73], p <1e-4 |
| P1r3 vs P1 (`A1-P1r3-D1-16k` − `A1-P1-D1-16k`) | +0.08 % [-0.13, +0.30], p 0.4138 | +0.34 % [+0.02, +0.70], p 0.0384 |
| Stage-1 recipe P1r3+D2 vs P1+D1 (`A1-P1r3-D2-16k` − `A1-P1-D1-16k`) | -0.66 % [-0.85, -0.46], p <1e-4 | – |
| SuperBPE 32k t/T 0.8: P1r3+D2 vs P1+D1 (`R2-A6…` − `A6-…-32k-t080`) | -0.46 % [-0.66, -0.26], p <1e-4 | – |
| *confounded with size:* MinGram P1+D1 16k → P1r3+D2 32k | -1.08 % [-1.39, -0.81], p <1e-4 | – |
| *confounded with size:* SP-Unigram D1 16k → D2 32k | -1.08 % [-1.79, -0.43], p 0.0008 | – |

- **D2 (strict-only train text) beats D1 by about 0.7 % bpb at 16k in the LM.** This is the only clean data-mix contrast with LM runs, and it supports the Stage 1 choice of D2, which was fragile on bytes/token (WAVES pending decision 1).
- Recall that D2 made SentencePiece Unigram 0.9–1.1 % *less* compressive than D1 at 32k–48k (ROUND2.md). There is no LM run of A4 on D1 at 32k, so the data-mix effect for Unigram at 32k is not measured.
- P1r3 (3-digit groups) is neutral in bpb at Stage 4. It is kept for numeracy, as PLAN §3 intends.

### 7.3 Algorithm effect at equal settings

| setting | contrast | Stage 4 Δ, 95 % CI, p |
|---|---|---|
| P1, D1, 16k (same pre-tokenizer and data) | `A10-MinGram-P1-D1-16k` − `A1-P1-D1-16k` | -1.00 % [-1.22, -0.76], p <1e-4 |
| P1, D1, 16k (same pre-tokenizer and data) | `A7-PickyBPE-P1-D1-16k-tau0.9` − `A1-P1-D1-16k` | +0.01 % [-0.16, +0.19], p 0.9254 |
| P1, D1, 16k (same pre-tokenizer and data) | `A6-SBPE-P1-D1-16k-t090` − `A1-P1-D1-16k` | +1.64 % [+1.39, +1.91], p <1e-4 |
| P1, D1, 16k (same pre-tokenizer and data) | `A6-SBPE-P1-D1-16k-t080` − `A1-P1-D1-16k` | +2.60 % [+2.24, +2.99], p <1e-4 |
| SentencePiece native, D1, 16k | `A4-SPnat-D1-16k` − `A3-SPnat-D1-16k` (Unigram vs SP-BPE) | -2.15 % [-3.48, -0.89], p 0.0002 |
| D1, 16k (pre-tokenizer differs) | `A4-SPnat-D1-16k` − `A1-P1-D1-16k` | -1.20 % [-1.92, -0.44], p 0.0016 |
| D1, 16k (pre-tokenizer differs) | `A3-SPnat-D1-16k` − `A1-P1-D1-16k` | +0.97 % [-0.21, +2.25], p 0.1172 |
| P1r3 (A4: SP-native), D2, 32k | `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-32k` | -0.61 % [-0.80, -0.41], p <1e-4 |
| P1r3 (A4: SP-native), D2, 32k | `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-32k` | -0.82 % [-1.24, -0.46], p <1e-4 |
| P1r3 (A4: SP-native), D2, 32k | `R2-A6-SBPE-P1r3-D2-32k-t080` − `A1-P1r3-D2-32k` | +3.06 % [+2.72, +3.40], p <1e-4 |
| P1r3 (A4: SP-native), D2, 48k | `R2-A10-MinGram-P1r3-D2-48k` − `R2-A1-P1r3-D2-48k` | -0.60 % [-0.88, -0.37], p <1e-4 |
| P1r3 (A4: SP-native), D2, 48k | `R2-A4-SPnat-D2-48k` − `R2-A1-P1r3-D2-48k` | -0.14 % [-0.70, +0.36], p 0.6028 |
| MinGram vs SentencePiece Unigram | `A10-MinGram-P1-D1-16k` − `A4-SPnat-D1-16k` | +0.21 % [-0.54, +0.93], p 0.5756 |
| MinGram vs SentencePiece Unigram | `R2-A10-MinGram-P1r3-D2-32k` − `R2-A4-SPnat-D2-32k` | +0.21 % [-0.17, +0.65], p 0.2900 |
| MinGram vs SentencePiece Unigram | `R2-A10-MinGram-P1r3-D2-48k` − `R2-A4-SPnat-D2-48k` | -0.46 % [-0.97, +0.12], p 0.1120 |

- **MinGram beats byte-level BPE at equal settings, at every size tested:** -1.00 % [-1.22, -0.76] at P1-D1-16k, -0.61 % [-0.80, -0.41] at P1r3-D2-32k, and -0.60 % [-0.88, -0.37] at 48k.
- **SentencePiece Unigram beats it at 16k and 32k:** -1.20 % [-1.92, -0.44] at D1-16k and -0.82 % [-1.24, -0.46] at D2-32k. There its pre-tokenization is SentencePiece's own. At 48k the difference is not significant (-0.14 % [-0.70, +0.36]).
- The Unigram-family advantage matches Land (2026), not Yavuz et al. (2026) (SOTA §1.1).
- **MinGram and SentencePiece Unigram are not distinguishable** at any size: +0.21 % [-0.54, +0.93]; +0.21 % [-0.17, +0.65]; -0.46 % [-0.97, +0.12].
- **PickyBPE vs BPE:** +0.01 % [-0.16, +0.19], and TOST-equivalent within ±0.3 %. Removing the under-trained intermediate tokens did not measurably change bpb (PLAN §4.3 hypothesis).
- **SentencePiece-BPE (A3) is not better than HF BPE** (+0.97 % [-0.21, +2.25], with seed s.d. 1.46 %). It is clearly worse than SentencePiece Unigram (-2.15 % [-3.48, -0.89]).

### 7.4 SuperBPE

| contrast | Stage 3 Δ | Stage 4 Δ | Stage 4 subset curve Δ at 25 / 50 / 75 / 100 % of steps | Stage 3 subset curve Δ at 25 / 50 / 75 / 100 % |
|---|---|---|---|---|
| `A6-SBPE-P1-D1-16k-t090` − `A1-P1-D1-16k` | +0.39 % [-0.05, +0.70], p 0.0770 | +1.64 % [+1.39, +1.91], p <1e-4 | +1.11 / +1.52 / +1.59 / +1.65 | -0.99 / -0.05 / +0.23 / +0.31 |
| `A6-SBPE-P1-D1-16k-t080` − `A1-P1-D1-16k` | +0.42 % [-0.02, +0.75], p 0.0578 | +2.60 % [+2.24, +2.99], p <1e-4 | +1.58 / +2.33 / +2.52 / +2.58 | -0.92 / -0.07 / +0.26 / +0.35 |
| `A6-SBPE-P1-D1-32k-t080` − `A1-P1r3-D2-32k` | +0.35 % [-0.20, +0.73], p 0.1860 | +3.53 % [+3.14, +3.91], p <1e-4 | +1.60 / +2.54 / +3.15 / +3.60 | -1.51 / -0.21 / +0.24 / +0.44 |
| `R2-A6-SBPE-P1r3-D2-32k-t080` − `A1-P1r3-D2-32k` | -0.58 % [-1.24, -0.09], p 0.0154 | +3.06 % [+2.72, +3.40], p <1e-4 | +0.92 / +2.18 / +2.69 / +3.12 | -2.70 / -1.17 / -0.72 / -0.54 |
| `A6-SBPE-P1-D1-16k-t090` − `A6-SBPE-P1-D1-16k-t080` | -0.03 % [-0.29, +0.20], p 0.9112 | -0.94 % [-1.26, -0.62], p <1e-4 | -0.46 / -0.80 / -0.91 / -0.91 | -0.07 / +0.02 / -0.03 / -0.04 |

| SuperBPE tokenizer | Stage 3 rank (of 18) | Stage 4 rank (of 18) | ctx tokens | train_D1 bytes/token | dev bytes/token |
|---|---|---|---|---|---|
| `R2-A6-SBPE-P1r3-D2-32k-t080` | 2 | 15 | 230 | 6.687 | 8.399 |
| `A6-SBPE-P1-D1-32k-t080` | 6 | 17 | 220 | 6.988 | 8.190 |
| `A6-SBPE-P1-D1-16k-t090` | 15 | 16 | 242 | 6.343 | 7.359 |
| `A6-SBPE-P1-D1-16k-t080` | 16 | 18 | 240 | 6.403 | 7.465 |

- **The SuperBPE results:**
  - At Stage 4, every SuperBPE tokenizer is significantly worse than its own stage-1 BPE base: +1.6 % (16k, t/T 0.9), +2.6 % (16k, 0.8) and +3.1 % (32k, 0.8, P1r3+D2).
  - More superwords is worse: t/T 0.9 beats 0.8 by -0.94 % [-1.26, -0.62].
  - The largest documented gain of the literature (PLAN §2.2 rank 2) does not appear with these small, byte-matched arbiters.
- **The Stage 3 picture was different:**
  - The P1-D1 builds were level with their base: +0.39 % and +0.42 %, both not significant.
  - The round-2 build was *better* than A1-P1r3-D2-32k (-0.58 % [-1.24, -0.09], p 0.0154).
  - The two 32k SuperBPE builds ranked 2nd and 6th of 18.
  - In the Stage 3 subset curves, SuperBPE leads early (−0.9 to −2.7 % at 25 % of steps) and loses most of its lead by 100 %.
  - At Stage 4 it is already behind at 25 % of the steps, and falls further behind to the end.
- **Hypothesis (not tested):** with byte-matched budgets, SuperBPE gives the model 6–12 % fewer prediction targets per training byte than its BPE comparator (train_D1 bytes/token). Its multi-word tokens also need more data per embedding. A superword vocabulary helps a small model that is far from convergence, and the larger, longer-trained Stage 4 arbiter gains more from the finer segmentation.
- **Conditional rank 8 (`A6-SBPE-P1-D1-32k-t080`)** is not admitted to the PLAN §7 ranking:
  - At Stage 3, neither rank 2 (1.43837) nor rank 5 (1.43877) beat rank 1 (1.42254).
  - Rank 2 beat rank 5, so the valid rank-8 build would have been t/T = 0.9, not the 0.8 file that ran.
  - Its Stage 4 result (1.27960, 17th of 18) is reported above.

### 7.5 Stage 3 vs Stage 4: ranking reversals

| candidate | Stage 3 rank | Stage 4 rank | Stage 3 Δ vs baseline | Stage 4 Δ vs baseline |
|---|---|---|---|---|
| `R2-A10-MinGram-P1r3-D2-48k` | 1 | 2 | -2.26 % [-2.56, -1.93] | -1.58 % [-1.87, -1.36] |
| `R2-A6-SBPE-P1r3-D2-32k-t080` | 2 | 15 | -2.13 % [-2.85, -1.60] | +2.22 % [+1.82, +2.61] |
| `R2-A10-MinGram-P1r3-D2-32k` | 3 | 3 | -1.92 % [-2.28, -1.53] | -1.41 % [-1.66, -1.19] |
| `R2-A1-P1r3-D2-48k` | 4 | 5 | -1.75 % [-2.13, -1.45] | -0.99 % [-1.21, -0.78] |
| `A1-P1r3-D2-32k` | 5 | 6 | -1.56 % [-1.79, -1.37] | -0.81 % [-1.00, -0.66] |
| `A6-SBPE-P1-D1-32k-t080` | 6 | 17 | -1.21 % [-1.83, -0.77] | +2.69 % [+2.25, +3.11] |
| `R2-A4-SPnat-D2-48k` | 7 | 4 | -1.08 % [-1.68, -0.54] | -1.13 % [-1.69, -0.64] |
| `R2-A4-SPnat-D2-32k` | 8 | 1 | -0.93 % [-1.59, -0.26] | -1.62 % [-2.11, -1.24] |
| `A10-MinGram-P1-D1-16k` | 9 | 8 | -0.27 % [-0.66, +0.23] | -0.34 % [-0.57, -0.10] |
| `A4-SPnat-D1-16k` | 10 | 7 | -0.09 % [-0.68, +0.46] | -0.55 % [-1.30, +0.23] |
| `A1-P1r3-D2-16k` | 11 | 9 | baseline | baseline |
| `A1-P1-D1-16k` | 12 | 10 | +0.72 % [+0.36, +1.01] | +0.66 % [+0.46, +0.86] |
| `A7-PickyBPE-P1-D1-16k-tau0.9` | 13 | 11 | +0.89 % [+0.60, +1.17] | +0.67 % [+0.51, +0.84] |
| `A1-P1r3-D1-16k` | 14 | 12 | +1.06 % [+0.73, +1.42] | +0.74 % [+0.51, +0.97] |
| `A6-SBPE-P1-D1-16k-t090` | 15 | 16 | +1.11 % [+0.66, +1.46] | +2.31 % [+2.02, +2.63] |
| `A6-SBPE-P1-D1-16k-t080` | 16 | 18 | +1.14 % [+0.72, +1.45] | +3.28 % [+2.88, +3.71] |
| `A3-SPnat-D1-16k` | 17 | 14 | +1.66 % [+0.87, +2.42] | +1.64 % [+0.43, +2.96] |
| `A1-P1r3-D2-8k` | 18 | 13 | +2.80 % [+2.36, +3.34] | +0.86 % [+0.37, +1.29] |

- **Rank agreement between the two stages:**
  - All 18 candidates: Kendall τ_b = 0.52 and Spearman ρ = 0.57.
  - Without the 4 SuperBPE tokenizers: τ_b = 0.80 and ρ = 0.91.
- **Reversals:** 37 of 153 candidate pairs change order between the stages. 22 of them are significant in both stages (the 95 % CIs exclude 0 in opposite directions), and 20 of those involve a SuperBPE tokenizer.
  - The largest SuperBPE reversal: `R2-A4-SPnat-D2-32k` vs `R2-A6-SBPE-P1r3-D2-32k-t080` goes from +1.23 % [+0.33, +2.29] at Stage 3 to -3.76 % [-4.14, -3.41] at Stage 4.
  - The significant reversals that do not involve SuperBPE: `A1-P1r3-D2-32k` vs `R2-A4-SPnat-D2-32k` (-0.64 % → +0.83 %); `R2-A1-P1r3-D2-48k` vs `R2-A4-SPnat-D2-32k` (-0.84 % → +0.64 %). All of them concern `R2-A4-SPnat-D2-32k`, which rises from 8th at Stage 3 to 1st at Stage 4.
- **Other large moves:**
  - `A1-P1r3-D2-8k` goes from 18th to 13th, because its penalty shrinks from +2.8 % to +0.9 %.
  - `A6-SBPE-P1-D1-32k-t080` goes from 6th to 17th, and `R2-A6-SBPE-P1r3-D2-32k-t080` from 2nd to 15th.
- **Consequence:** the 10 MB, d=128 screening stage is a poor proxy for SuperBPE, and for the size of the small-vocabulary penalty. The decision uses Stage 4, as PLAN §7.2 prescribes when Stage 4 exists.

### 7.6 Power check (PLAN §6)

- **The PLAN's literal check** (after the baseline's first wave, Stage 3): the baseline's seed s.d. is 0.043 %, so n = 1, and 3 seeds suffice.
- **Stage 4, seeds 1–3** (the collector's rule):
  - The pooled s.d. is 0.509 % in round 1 (n = 17) and 0.394 % in round 2 (n = 10).
  - Both are > 0.2 %, so seeds 4 and 5 were run for every candidate.
- **Stage 4, all 5 seeds:**
  - The pooled s.d. is 0.435 % over all 18 candidates, which gives n = 12.
  - Two arms drive it: A3-SPnat-D1-16k 1.46 %, A4-SPnat-D1-16k 0.81 %. Without them the pooled s.d. is 0.196 % and n = 3.
  - For each top-set member, the seed s.d. and the n its own s.d. requires: `R2-A4-SPnat-D2-32k` 0.23 % (n = 4); `R2-A10-MinGram-P1r3-D2-48k` 0.15 % (n = 2); `R2-A10-MinGram-P1r3-D2-32k` 0.21 % (n = 3).
- **Achieved resolution** (80 %-power detectable effect, from the bootstrap s.d. of Δ, which includes cluster and seed noise):
  - chosen vs baseline: 0.36 %;
  - median over the candidates vs baseline: 0.43 %;
  - the top-set members vs the best: `R2-A10-MinGram-P1r3-D2-48k` 0.68 %, `R2-A10-MinGram-P1r3-D2-32k` 0.59 %.
- **Reading:** the chosen tokenizer's gain over the baseline (1.58 %) is about 4× the detectable effect.
- The gaps inside the top set (+0.04 %, +0.21 % vs the best) are below it. They are **not distinguishable at our power**, never 'equal'. TOST did not establish equivalence either.

### 7.7 Learning rate

| tokenizer | 1e-3 | 3e-3 | 6e-3 |
|---|---|---|---|
| `A1-P1r3-D2-16k` (Stage 3 sweep, seed 1) | 1.42321 | 1.44182 | 1.45496 |
| `A1-P1r3-D2-32k` (Stage 4 re-sweep, seed 1) | 1.23576 | 1.27560 | 1.33084 |
| `A4-SPnat-D1-16k` (Stage 4 re-sweep, seed 1) | 1.23155 | 1.25919 | 1.38023 |
| `R2-A10-MinGram-P1r3-D2-48k` (Stage 4 re-sweep, seed 1) | 1.22387 | 1.25139 | 1.38592 |
| `R2-A4-SPnat-D2-32k` (Stage 4 re-sweep, seed 1) | 1.22995 | 1.25739 | 1.40877 |

- 1e-3 wins everywhere, including for the chosen tokenizer and the best-mean candidate.
- No LR below 1e-3 was tried (§8, deviation 5).

### 7.8 Sensitivity analyses

| analysis | k | best mean | top set | chosen | chosen vs baseline |
|---|---|---|---|---|---|
| **primary** (PLAN §6–§7 as run) | 17 | `R2-A4-SPnat-D2-32k` | `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k` | **`R2-A10-MinGram-P1r3-D2-48k`** | -1.58 % [-1.87, -1.36] |
| raw_manifest_groups: PLAN §6 sensitivity: manifest groups as they are (per-record web groups kept): 41 groups | 17 | `R2-A4-SPnat-D2-32k` | `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k` | `R2-A10-MinGram-P1r3-D2-48k` | -1.58 % [-1.86, -1.36] |
| rank8_included: conditional rank 8 admitted to the Holm family anyway | 18 | `R2-A4-SPnat-D2-32k` | `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k` | `R2-A10-MinGram-P1r3-D2-48k` | -1.58 % [-1.87, -1.36] |
| p_plus1: p = 2 min((#{D*<=0}+1)/(R+1), (#{D*>=0}+1)/(R+1)) | 17 | `R2-A4-SPnat-D2-32k` | `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k` | `R2-A10-MinGram-P1r3-D2-48k` | -1.58 % [-1.87, -1.36] |
| round1_only: confirmatory set: round-1 candidates only (round-2 exploratory tokenizers removed; conditional rank 8 excluded as in the primary) | 11 | `A1-P1r3-D2-32k` | `A1-P1r3-D2-32k`, `A4-SPnat-D1-16k` | `A1-P1r3-D2-32k` | -0.81 % [-1.00, -0.66] |
| confirm_seeds_1to3: only the 3 seeds PLAN §5 planned (seeds 4-5 dropped) | 17 | `R2-A10-MinGram-P1r3-D2-48k` | `R2-A10-MinGram-P1r3-D2-48k`, `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-32k`, `R2-A4-SPnat-D2-48k`, `A4-SPnat-D1-16k` | `R2-A10-MinGram-P1r3-D2-48k` | -1.57 % [-1.97, -1.27] |

- **The choice is the same under every sensitivity analysis except the confirmatory 'round 1 only' family.** That family excludes the exploratory candidates by construction. It chooses `A1-P1r3-D2-32k`, whose improvement claim also holds.
- The improvement claim holds in every row.

## 8. Deviations from PLAN

All deviations are additive: none changes a pre-registered metric, split, model or decision rule. Items 1–5 were known before this analysis; the others were found or decided in it.

1. **LM on a GPU (fp16) instead of the CPU** (known before this analysis)
   - What: PLAN §5 budgets a CPU arbiter; every run used a Colab Tesla T4 with fp16 autocast for training and fp32 evaluation, strict determinism, tf32 off. Parity against the CPU reference on the fixed tiny configuration: fp16 +0.447 %, fp32 -0.018 % (tolerance 1 %), identical in both bundles.
   - Consequence: All 18 candidates ran on the same device and dtype, so every comparison is like-for-like. Absolute bpb is comparable to a CPU run only to about 0.45 %.
2. **Stage 4 'confirm' ran for all candidates, not only the top 2** (known before this analysis)
   - What: PLAN §3/§5 plan Stage 4 for the top 2 plus the baseline; on the GPU every candidate got the d=192, full-train, 1-epoch arbiter with 5 seeds. PLAN §7.2 ranks on Stage 4 when it is run.
   - Consequence: The ranking and the Holm family cover 17 candidates (16 comparisons against the best) instead of 3. A larger family makes Holm more conservative, so it can only widen the top set.
3. **Two D1 reference arms added in round 1** (known before this analysis)
   - What: A1-P1r3-D1-16k and A1-P1-D1-16k were added (WAVES_colab.json, written 16:11 UTC, before the first LM run at 16:30 UTC) because the rank-1 baseline is D2 while the Stage-2 candidates (A6/A7/A10) are D1.
   - Consequence: They are ranked with everyone else (1.25532 and 1.25430; neither is in the top set) and give the clean D2-vs-D1 and algorithm contrasts of the report-only sections.
4. **Round 2 is exploratory (garden of forking paths)** (known before this analysis)
   - What: The 6 R2-* tokenizers were built 16:47-17:00 UTC, after the round-1 screen ended (16:41 UTC), and WAVES_round2.json was written 17:23 UTC, after the round-1 Stage-4 power check on seeds 1-3 (17:22 UTC). They combine the recipe of the best Stage-3 tokenizer with the algorithms that looked best.
   - Consequence: The chosen tokenizer (R2-A10-MinGram-P1r3-D2-48k), the best-mean candidate (R2-A4-SPnat-D2-32k) and every top-set member are round-2 tokenizers, so the dev ranking is conditioned on dev and the dev Δ is optimistic (winner's curse). The sealed test split, used once (PLAN §7.6), is the guard. Confirmatory reading: restricted to round 1 the same rule chooses A1-P1r3-D2-32k, Δ vs baseline -0.810 % [-1.001, -0.657] (improvement claim holds there too).
5. **Learning rate at the edge of the grid** (known before this analysis)
   - What: The PLAN §5 sweep on the baseline chose 1e-3, the lowest grid value (bpb 1.42321 vs 1.44182 at 3e-3 and 1.45496 at 6e-3); round 2 reused 1e-3 without a sweep. The finalist re-sweep (report-only) confirms 1e-3 beats 3e-3 and 6e-3 for all four finalists: A1-P1r3-D2-32k: 1e-3 1.23576, 3e-3 1.27560, 6e-3 1.33084; A4-SPnat-D1-16k: 1e-3 1.23155, 3e-3 1.25919, 6e-3 1.38023; R2-A10-MinGram-P1r3-D2-48k: 1e-3 1.22387, 3e-3 1.25139, 6e-3 1.38592; R2-A4-SPnat-D2-32k: 1e-3 1.22995, 3e-3 1.25739, 6e-3 1.40877.
   - Consequence: No LR below 1e-3 was tried, so the optimum may lie below the grid, and it may differ between tokenizers. The chosen tokenizer and the best-mean candidate are both among the re-swept finalists.
6. **Conditional rank 8 not admitted to the PLAN §7 ranking** (found or decided in this analysis)
   - What: A6-SBPE-P1-D1-32k-t080 was trained unconditionally on the GPU. Its PLAN §2.2 condition (rank 2 or rank 5 beats rank 1 in Stage-3 mean bpb) is false: 1.43837 and 1.43877 vs 1.42254. Moreover rank 2 beat rank 5 at Stage 3, so the valid rank-8 tokenizer would have been a t/T = 0.9 32k build, which does not exist.
   - Consequence: Reported (SuperBPE section) but outside the Holm family (k = 17, not 18). Admitting it anyway changes nothing: same best, top set and choice (sensitivity 'rank8_included').
7. **5 seeds resampled per candidate, not 3** (found or decided in this analysis)
   - What: PLAN §6 says 'resample its 3 seeds'. The power check added seeds 4 and 5 for every candidate, so the bootstrap resamples the 5 seeds each candidate has (Stage 3: its 3).
   - Consequence: Using only seeds 1-3 changes the best-mean candidate to R2-A10-MinGram-P1r3-D2-48k and widens the top set to 5 members, but the chosen tokenizer is the same (R2-A10-MinGram-P1r3-D2-48k).
8. **Operational definitions PLAN leaves open** (found or decided in this analysis)
   - What: delta = 0.3 % of the baseline's Stage-4 mean bpb = 0.003738 bpb (absolute, used for every TOST). p = 2 min(P(Δ*≤0), P(Δ*≥0)) has a resolution of 1/10,000; p = 0 means p < 1e-4. 'Significantly worse' = Holm-adjusted p <= 0.05 with a positive point Δ. Tie-breaker (a): a stock HF tokenizer.json reproduces the ids the LM used on 100 % of dev (dev_strict and dev_permissive); (c): all learned tokens with train_D1 frequency < 20 (train_D1 is the Stage-4 training stream; summary.json R1); (d): mean over the four PLAN §4.2 perturbations of the share of affected words whose segmentation changes; (e): effective vocabulary. Tie-breakers are lexicographic, without tolerance.
   - Consequence: Only (a) and (b) were reached. The choice does not depend on (a): R2-A10-MinGram-P1r3-D2-48k also has the highest dev bytes/token of the three top-set members, so (b) alone gives the same result, including under a dev_strict-only reading of (a) (R2-A4-SPnat-D2-32k's export matches on 836/836 dev_strict documents and differs on 1/1,358 dev_permissive). The +1-corrected p-value gives the same Holm decisions.
9. **Seed count below the PLAN §6 power formula** (found or decided in this analysis)
   - What: Pooled 5-seed relative s.d. over all 18 candidates: 0.435 % -> n = 12 seeds for Δ_min = 0.5 %. It is driven by two SentencePiece D1 arms (A3-SPnat-D1-16k 1.46 %, A4-SPnat-D1-16k 0.81 %); without them it is 0.196 % -> 3 seeds.
   - Consequence: The gaps inside the top set (+0.043, +0.213 % vs the best) are below the achieved 80 %-power detectable effect of those comparisons (0.68, 0.59 %): they are 'not distinguishable at our power', never 'equal' (TOST did not establish equivalence either).
10. **Open user decisions not resolved here** (inherited from WAVES pending_user_decisions)
   - What: (1) Data mix D2 was fragile at Stage 1 and awaits confirmation; the LM now favours D2 at 16k: A1-P1r3-D2-16k vs A1-P1r3-D1-16k -0.737 % [-0.963, -0.512]. (2) SuperBPE multi-word G2 was passed under the reachability reading; under the literal 'shortest train chunk' reading ranks 2, 5, 8 and R2-A6-SBPE-P1r3-D2-32k-t080 would be dropped. (3) The Stage-2 candidates (A6/A7/A10 round 1) were built on P1/D1, not on rank 1's P1r3/D2.
   - Consequence: None of these changes the top set: no SuperBPE, A7 or round-1 A10 tokenizer is near it, and every top-set member is a D2 tokenizer that the LM favours.
11. **G1 on test not yet run** (found or decided in this analysis)
   - What: PLAN G1 covers dev and test. Every candidate passed G1 on dev_strict and dev_permissive; the test half is deferred to Stage 5 for every candidate (test discipline), as in Stages 1-2 and round 2.
   - Consequence: The TestBundle phase must run G1 on test for the test candidates before scoring them.
12. **Minor data flags** (from the collection, COLLECT.md)
   - What: One screening run is 0.035 pp over the 1 % bytes_seen tolerance (77e1368773fc/results/screen__R2-A6-SBPE-P1r3-D2-32k-t080__lr1e-3__s2.json); screening is not used by the decision. The round-2 screen curve of the top two crossed between 75 % and 100 % (PLAN §5: send them to Stage 4), which ran for everyone anyway. The incomplete, non-pre-registered 'large' stage is ignored.
   - Consequence: None.

## 9. What can be claimed now, and what waits for the test

**Now (dev_strict, Stage 4 arbiter):**

- Among the 17 tokenizers ranked under PLAN §7 (18 trained, with the conditional rank 8 excluded), the lowest mean bpb is `R2-A4-SPnat-D2-32k`'s.
- `R2-A10-MinGram-P1r3-D2-48k` is chosen from the top set by the pre-registered tie-breakers. It beats the standard recipe `A1-P1r3-D2-16k` by -1.58 % bpb, 95 % CI [-1.87, -1.36] (cluster bootstrap, 5 seeds each).
- The dev evidence also supports these general statements:
  - Unigram-family vocabularies beat BPE at equal settings.
  - 32k beats 16k and 8k, and 48k adds nothing significant.
  - The strict-only train mix D2 beats D1 (at 16k).
  - SuperBPE and PickyBPE do not help these small arbiters.

**After the one-shot test (PLAN §7.6/§8):**

- The PLAN §8 sentence becomes claimable only if test agrees in sign. Its template is: *'Among 17 tokenizers trained and compared under one protocol, `R2-A10-MinGram-P1r3-D2-48k` achieved the lowest held-out bits-per-byte … significantly better than the standard BPE recipe (Δ = x %, 95 % CI [a, b], 5 seeds, cluster bootstrap)'*.
  - Strictly, the chosen tokenizer is 2nd in mean dev bpb and 1st after the tie-breakers, so the wording should say *'chosen by the pre-registered rule'*, not *'lowest bpb'*, unless test puts it first among the test candidates.
- Round 2 was exploratory. Every public statement should name it: *'selected after an exploratory second round; confirmed on a sealed test split used once'*.

**Cannot be claimed (PLAN §8, plus this analysis):**

- Transfer to 1B+ models or to CPT.
- That MinGram-48k, MinGram-32k and SentencePiece-Unigram-32k are 'equal'. They are only not distinguishable at our power.
- That 48k is better than 32k: the step is not significant.
- Anything about LR below 1e-3.

## 10. Files and reproduction

| file | content |
|---|---|
| `analysis/decide.py` | the whole computation: input checks, clusters, bootstrap, Holm, TOST, top set, tie-breakers, contrasts, power, sensitivity, deviations |
| `analysis/report.py` | renders this file from `decision.json` |
| `analysis/decision.json` | every number in this file, plus per-seed bpb, the tie-breaker metrics of all 18 tokenizers, the 32 clusters, and `test_candidates_fixed` |
| `analysis/DECISION.md` | this report |

Reproduce (single process, a few seconds; deterministic): `PYTHONIOENCODING=utf-8 python analysis/decide.py && PYTHONIOENCODING=utf-8 python analysis/report.py`, from `F:\Hindko\_tokenizer`.

## Appendix A. Tie-breaker metrics of every candidate (dev_strict harness; R1 on train_D1)

| candidate | encoder | HF-native exact | dev bytes/token | learned tokens | train freq < 20 | freq = 0 | robustness (seg. change, affected) | mean abs. token change | vocab nominal / effective | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` | sentencepiece | no | 7.0764 | 32441 | 13182 (40.6 %) | 2 | 0.589 | 0.0192 | 32768 / 32763 | pass |
| `R2-A10-MinGram-P1r3-D2-48k` | hf | yes | 7.2237 | 48831 | 27246 (55.8 %) | 1 | 0.295 | 0.0124 | 49152 / 49152 | pass |
| `R2-A10-MinGram-P1r3-D2-32k` | hf | yes | 7.1258 | 32447 | 11329 (34.9 %) | 0 | 0.316 | 0.0125 | 32768 / 32768 | pass |
| `R2-A4-SPnat-D2-48k` | sentencepiece | no | 7.1346 | 48792 | 28858 (59.1 %) | 58 | 0.598 | 0.0194 | 49152 / 49114 | pass |
| `R2-A1-P1r3-D2-48k` | hf | yes | 7.2017 | 48832 | 27273 (55.9 %) | 386 | 0.382 | 0.0120 | 49152 / 49152 | pass |
| `A1-P1r3-D2-32k` | hf | yes | 7.0949 | 32448 | 11512 (35.5 %) | 172 | 0.427 | 0.0122 | 32768 / 32768 | pass |
| `A4-SPnat-D1-16k` | sentencepiece | no | 6.8668 | 16061 | 68 (0.4 %) | 0 | 0.503 | 0.0174 | 16384 / 16383 | pass |
| `A10-MinGram-P1-D1-16k` | hf | yes | 6.7702 | 16063 | 66 (0.4 %) | 0 | 0.205 | 0.0118 | 16384 / 16384 | pass |
| `A1-P1r3-D2-16k` | hf | yes | 6.7883 | 16064 | 577 (3.6 %) | 33 | 0.426 | 0.0119 | 16384 / 16384 | pass |
| `A1-P1-D1-16k` | hf | yes | 6.7117 | 16064 | 447 (2.8 %) | 25 | 0.292 | 0.0115 | 16384 / 16384 | pass |
| `A7-PickyBPE-P1-D1-16k-tau0.9` | custom | no | 6.7188 | 16064 | 249 (1.6 %) | 0 | 0.297 | 0.0115 | 16384 / 16384 | pass |
| `A1-P1r3-D1-16k` | hf | yes | 6.7227 | 16064 | 434 (2.7 %) | 24 | 0.410 | 0.0118 | 16384 / 16384 | pass |
| `A1-P1r3-D2-8k` | hf | yes | 6.3076 | 7872 | 125 (1.6 %) | 15 | 0.351 | 0.0113 | 8192 / 8192 | pass |
| `A3-SPnat-D1-16k` | sentencepiece | yes | 6.8765 | 15812 | 405 (2.6 %) | 9 | 0.489 | 0.0120 | 16384 / 16384 | pass |
| `R2-A6-SBPE-P1r3-D2-32k-t080` | hf | yes | 8.3990 | 32448 | 6057 (18.7 %) | 119 | 0.483 | 0.0149 | 32768 / 32768 | pass |
| `A6-SBPE-P1-D1-16k-t090` | hf | yes | 7.3590 | 16064 | 369 (2.3 %) | 22 | 0.331 | 0.0129 | 16384 / 16384 | pass |
| `A6-SBPE-P1-D1-32k-t080` | hf | yes | 8.1900 | 32448 | 1838 (5.7 %) | 68 | 0.352 | 0.0142 | 32768 / 32768 | pass |
| `A6-SBPE-P1-D1-16k-t080` | hf | yes | 7.4651 | 16064 | 326 (2.0 %) | 19 | 0.296 | 0.0130 | 16384 / 16384 | pass |
