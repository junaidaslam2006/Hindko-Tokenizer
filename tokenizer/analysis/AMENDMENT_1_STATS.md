# Amendment 1: the statistics behind the released default (large arbiter, dev_strict)

Generated 2026-09-26T21:28:41Z by `analysis/amendment1_stats.py` (sha256 `62150c222efb…`) and `analysis/amendment1_report.py`. The bootstrap is `analysis/decide.py`'s own code, imported unchanged (sha256 `5ad1170722a6…`, the same as in `decision.json`). Machine-readable twin: `analysis/amendment1_stats.json`.

- **Report-only.** This file changes neither the pre-registered decision (`DECISION.md`) nor `AMENDMENT_1.md`. The sha256 of `AMENDMENT_1.md` is still `9a966033ffdb…`, as recorded in `AMENDMENT_1.sha.json` (written 2026-09-26T21:14:48Z).
- **Dev only.** Every number is on dev_strict (836 documents, 1,445,513 bytes). No test LM result was read.
- **The large arbiter is not pre-registered.** It is d=384, L=6, H=6, trained for 2 epochs of the full train split at LR 5e-4, with seeds 1 and 2. Its LR sweep started at 19:44 UTC, and its 10 records were written between 2026-09-26T20:01:17Z and 2026-09-26T20:53:12Z, before Amendment 1 was written (2026-09-26T21:14:48Z).

## 0. Summary

**The amended rule's outcome is `R2-A4-SPnat-D2-32k`.** Within the pre-registered top set, it has the lowest large-arbiter dev bpb:

| candidate | large dev bpb | Δ for SP-32k against it, 95 % CI | p | Holm p (3 pairs) |
|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` (SP-32k) | 1.13373 | – | – | – |
| `R2-A10-MinGram-P1r3-D2-32k` | 1.14086 | -0.62 % [-1.04, -0.31] | 0.0004 | 0.0008 |
| `R2-A10-MinGram-P1r3-D2-48k` | 1.14407 | -0.90 % [-1.35, -0.56] | <1e-4 | <1e-4 |

- **Both CIs exclude 0.** SP-32k is the lowest of the three in 99.98 % of the 10,000 replicates. It keeps that place when any single one of the 32 clusters is left out.
- **The rule needs only the lowest mean.** The CIs show that, at this scale, the choice is not a coin flip.
- **The main caveat is the 2 seeds per candidate** (§5):
  - with a conservative seed-noise model (each candidate's larger observed seed s.d.), both CIs still exclude 0;
  - with a pessimistic stress model (0.435 % seed s.d. for every candidate), they include 0. SP-32k is still lowest in 90 % of replicates.

**Δ vs the baseline `A1-P1r3-D2-16k`, large arbiter:**

- `R2-A4-SPnat-D2-32k`: -1.34 % [-1.78, -1.03], p <1e-4
- `R2-A10-MinGram-P1r3-D2-32k`: -0.72 % [-0.91, -0.52], p <1e-4
- `R2-A10-MinGram-P1r3-D2-48k`: -0.44 % [-0.62, -0.29], p <1e-4
- `A1-P1r3-D2-32k`: +0.03 % [-0.30, +0.32], p 0.9080

All three top-set members improve on the baseline. The BPE 32k tokenizer does not: its CI straddles 0.

**From the confirm scale to the large scale, vocabulary-size effects shrink.** 'Change' is the large-scale Δ minus the confirm-scale Δ, in percentage points (pp). Each change has its own paired bootstrap CI.

| contrast | what it varies | confirm Δ | large Δ | change (pp) |
|---|---|---|---|---|
| `A1-P1r3-D2-32k` − `A1-P1r3-D2-16k` | vocabulary size (BPE, 16k to 32k) | -0.81 % | +0.03 % | +0.84 [+0.52, +1.19] |
| `R2-A10-MinGram-P1r3-D2-48k` − `R2-A10-MinGram-P1r3-D2-32k` | vocabulary size (MinGram, 32k to 48k) | -0.17 % | +0.28 % | +0.45 [+0.16, +0.71] |
| `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-32k` | algorithm at equal size (Unigram vs BPE, 32k) | -0.82 % | -1.36 % | -0.54 [-0.88, -0.22] |
| `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-32k` | algorithm at equal size (MinGram vs BPE, 32k) | -0.61 % | -0.74 % | -0.13 [-0.45, +0.19] |

- **Vocabulary size:** the effect vanishes for BPE, and it reverses for MinGram 32k to 48k.
- **Algorithm at equal size:** the effect persists. For Unigram it grows.
- **This pattern fits the audit's parameter confound** (details in §6.3). The confirm-scale arbiter has 1.77M non-embedding parameters. There, the embeddings are 64–84 % of all parameters, so a larger vocabulary buys model capacity. The large arbiter has 10.63M non-embedding parameters, and the extra embedding rows no longer help.
- **Model size, epochs and LR changed together** between the two scales, so the pattern supports the confound but does not isolate it.

**Total parameters per model** (full table in §7):

| candidate | confirm (d=192, L=4) | large (d=384, L=6) |
|---|---|---|
| `A1-P1r3-D2-16k` | 4.97M (1.00x) | 17.02M (1.00x) |
| `A1-P1r3-D2-32k` | 8.11M (1.63x) | 23.31M (1.37x) |
| `R2-A4-SPnat-D2-32k` | 8.11M (1.63x) | 23.31M (1.37x) |
| `R2-A10-MinGram-P1r3-D2-32k` | 8.11M (1.63x) | 23.31M (1.37x) |
| `R2-A10-MinGram-P1r3-D2-48k` | 11.26M (2.27x) | 29.59M (1.74x) |

## 1. Inputs and checks

- **Large records.** There are 10 files, `lm/colab_results/77e1368773fc/results/large__<candidate>__lr5e-4__s{1,2}.json`.
  - They were copied from `G:\My Drive\hindko_lm_out\77e1368773fc\results\`. Each copy's sha256 equals the Drive file's (the per-file hashes are in the JSON).
  - Every record has status ok, stage `large`, `preregistered: false`, 7,325/7,325 steps and 2 epochs.
  - Every record has the same `hk_lm.py` (sha256 `3f78ad799f56…`) and the same bundle, `77e1368773fc`.
  - Every record's bpb was recomputed bit-exactly as Σbits/Σbytes.
  - Every record covers the same 836 dev_strict uids, in the same order and with the same bytes, as the Stage 4 records of record.
  - No other `large__*` file exists.
- **Large LR sweep** (baseline, seed 1): 5e-4 → 1.14899, 1e-3 → 1.18164, 2e-3 → 1.22207. The sweep chose 5e-4, which is the **lowest value of its grid**, as 1e-3 was at Stage 4.
  - The 5e-4 sweep run and `large__A1-P1r3-D2-16k__lr5e-4__s1` have identical per-document bits, so the large runs are bitwise reproducible.
  - `large_lr_choice.json` was copied next to the results as well.
- **Stage 4 'confirm' records of record** come from `lm/colab_results/results_table.json` (sha256 `aa315f1bf89e…`), loaded with `decide.load_stage`.
  - That loader re-verifies each file's sha256 and bit-exact bpb, and it drops duplicate copies.
  - The baseline and `A1-P1r3-D2-32k` records of record are the round-1 runs (bundle `1d24425d2d64`). Their round-2 copies in `77e1368773fc`, the bundle the large arbiter used, are bitwise identical (DECISION.md §1).
- **Clusters:** 32, with book 12, newspaper 15 and web 5. The mapping is the same as in DECISION.md (`decide.dev_clusters`), and it agrees with `dev_clusters.csv`.
- **Cross-check against `decision.json`.**
  - The confirm-scale Δs recomputed here for the same 5 candidates equal the numbers of record.
  - Their 95 % CI endpoints differ by at most 0.019 pp. The cause is that the seed-draw stream differs with 5 candidates instead of 18.

## 2. Method, as run

- **The bootstrap is DECISION.md §2 unchanged**, via `decide.Boot`, `decide.compare` and `decide.holm`:
  - 10,000 replicates with `numpy.random.default_rng(12345)`;
  - the 32 clusters are resampled with replacement within each source;
  - each candidate's seeds are resampled with replacement, independently;
  - bpb = Σbits ÷ Σbytes, averaged over the resampled seeds;
  - the RNG order is the cluster draws, then the seed draws, with candidates in sorted id order: `A1-P1r3-D2-16k`, `A1-P1r3-D2-32k`, `R2-A10-MinGram-P1r3-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A4-SPnat-D2-32k`.
  - Δ% = (bpb_A − bpb_B)/bpb_B per replicate, with a 95 % percentile CI and p = 2·min(P(Δ*≤0), P(Δ*≥0)).
- **What differs from DECISION.md:**
  - There are 5 candidates with 2 seeds each, because the large stage ran only seeds 1 and 2.
  - The TOST margin is δ = 0.3 % of the large baseline mean = 0.003447 bpb.
  - Holm is applied (report-only) to two families: the 3 pairs inside the top set, and the 4 comparisons against the large-scale best.
- **Scale comparison:**
  - One joint bootstrap draws the clusters once, then the seed draws for the large runs, then those for the confirm runs.
  - Its large block reproduces the primary large replicates exactly (asserted), so every large-scale number here comes from the same replicates.
  - The confirm and large Δs of a pair are paired over documents. The change is Δ_large − Δ_confirm per replicate, in pp.
- **A 2-seed caveat.** Resampling n seeds with replacement shrinks the seed variance of the mean by (n−1)/n:
  - 0.8 at Stage 4's 5 seeds (the audit's figure);
  - **0.5 at 2 seeds.**
  - §5 therefore repeats the key numbers under three seed-noise variants.

## 3. Large arbiter: every candidate against the baseline

| # | candidate | top set | V | total params | bpb s1 / s2 | mean bpb | seed s.d. | Δ vs baseline, 95 % CI | p | Δ vs best, 95 % CI | Holm p vs best (4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `R2-A4-SPnat-D2-32k` | **yes** | 32k | 23.31M | 1.13319 / 1.13428 | 1.13373 | 0.068 % | -1.34 % [-1.78, -1.03] | <1e-4 | best | – |
| 2 | `R2-A10-MinGram-P1r3-D2-32k` | **yes** | 32k | 23.31M | 1.14016 / 1.14157 | 1.14086 | 0.088 % | -0.72 % [-0.91, -0.52] | <1e-4 | +0.63 % [+0.31, +1.05] | 0.0004 |
| 3 | `R2-A10-MinGram-P1r3-D2-48k` | **yes** | 48k | 29.59M | 1.14336 / 1.14477 | 1.14407 | 0.088 % | -0.44 % [-0.62, -0.29] | <1e-4 | +0.91 % [+0.56, +1.37] | <1e-4 |
| 4 | `A1-P1r3-D2-16k` | no | 16k | 17.02M | 1.14899 / 1.14917 | 1.14908 | 0.011 % | baseline | – | +1.35 % [+1.04, +1.81] | <1e-4 |
| 5 | `A1-P1r3-D2-32k` | no | 32k | 23.31M | 1.15108 / 1.14771 | 1.14940 | 0.207 % | +0.03 % [-0.30, +0.32] | 0.9080 | +1.38 % [+1.02, +1.82] | <1e-4 |

- **BPE 32k vs BPE 16k:** +0.03 % [-0.30, +0.32], p 0.9080.
  - In the primary bootstrap it is also TOST-equivalent within ±0.3 %: the 90 % CI is [-0.24, +0.28] % of the baseline bpb.
  - Under the seed-noise variants of §5, it is not TOST-equivalent. Read it as 'no detectable difference', not 'equal'.
- **The seed s.d.s here come from 2 seeds each, so they are very imprecise.** They range from 0.011 % (the baseline) to 0.207 % (`A1-P1r3-D2-32k`).

## 4. Pairwise comparisons inside the top set

| pair (Δ = a vs b) | large Δ, 95 % CI | p | Holm p (3) | clusters where a is better (of 32) | their share of dev bytes | leave-one-cluster-out range | confirm Δ, 95 % CI (same 5) |
|---|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-32k` | -0.62 % [-1.04, -0.31] | 0.0004 | 0.0008 | 24 | 83.5 % | -0.68 % to -0.49 % | -0.21 % [-0.64, +0.19] |
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-48k` | -0.90 % [-1.35, -0.56] | <1e-4 | <1e-4 | 28 | 96.3 % | -1.00 % to -0.76 % | -0.04 % [-0.55, +0.40] |
| `R2-A10-MinGram-P1r3-D2-48k` vs `R2-A10-MinGram-P1r3-D2-32k` | +0.28 % [+0.07, +0.47] | 0.0110 | 0.0110 | 6 | 19.4 % | +0.25 % to +0.34 % | -0.17 % [-0.47, +0.11] |

- **Rank within the top set across the 10,000 replicates:**
  - `R2-A4-SPnat-D2-32k`: 1st 99.98 %, 2nd 0.02 %, 3rd 0.00 %.
  - `R2-A10-MinGram-P1r3-D2-32k`: 1st 0.02 %, 2nd 99.43 %, 3rd 0.55 %.
  - `R2-A10-MinGram-P1r3-D2-48k`: 1st 0.00 %, 2nd 0.55 %, 3rd 99.45 %.
- **Leaving any one of the 32 clusters out never changes the order** SP-32k < MinGram-32k < MinGram-48k.
- **The PLAN §6–§7 comparison family, applied at the large scale** (report-only; every candidate against the best, Holm over 4): all four are significantly worse than SP-32k. The largest adjusted p is 0.0004, for `R2-A10-MinGram-P1r3-D2-32k`.
  - At the confirm scale, by contrast, the whole top set was statistically tied (DECISION.md §3).

**By source** (the same replicates restricted to one source's clusters; report-only):

| source | clusters | bytes | SP-32k vs MinGram-32k: confirm → large | SP-32k vs MinGram-48k: confirm → large | MinGram 48k vs 32k: confirm → large |
|---|---|---|---|---|---|
| book | 12 | 878,168 | -0.61 % [-1.37, -0.13] → -0.90 % [-1.69, -0.46] | -0.37 % [-1.35, +0.28] → -1.17 % [-2.00, -0.68] | -0.24 % [-0.66, +0.12] → +0.27 % [+0.02, +0.49] |
| newspaper | 15 | 532,872 | +0.59 % [+0.20, +1.01] → -0.02 % [-0.35, +0.27] | +0.68 % [+0.34, +1.03] → -0.32 % [-0.64, +0.00] | -0.09 % [-0.34, +0.19] → +0.30 % [+0.00, +0.53] |
| web | 5 | 34,473 | -0.08 % [-1.56, +1.20] → -0.72 % [-2.67, +0.89] | -0.62 % [-1.84, +0.39] → -1.07 % [-2.64, +0.13] | +0.53 % [+0.05, +1.16] → +0.36 % [-0.33, +1.18] |

- **Books:** SP-32k's large-scale lead comes mostly from the books, which are 12 clusters and 61 % of dev bytes. Against MinGram-32k it is -0.90 % [-1.69, -0.46] there.
- **Newspapers:** at the confirm scale, SP-32k was significantly *worse* than both MinGram builds (+0.59 % [+0.20, +1.01] vs MinGram-32k). At the large scale that gap closes to a tie: -0.02 % [-0.35, +0.27].
- **Implication:** for news-like text, the released default has no demonstrated edge over MinGram-32k at either scale. At the confirm scale it was worse there.
- **Web:** 5 clusters and 34 KB, so its CIs are wide.

## 5. Sensitivity to having only 2 seeds

Each variant reuses the primary replicates' cluster draws. They differ only in how seed noise enters:

- **rescaled:** each replicate's seed deviation is multiplied by √(n/(n−1)) = √2, which undoes the 0.5 variance shrink.
- **own s.d.:** clusters are resampled, seeds are averaged, and relative seed noise N(0, σ_c²/2) is added per candidate.
  - σ_c is the larger of the candidate's large 2-seed s.d. and its confirm 5-seed s.d.: `A1-P1r3-D2-16k` 0.100 %, `A1-P1r3-D2-32k` 0.207 %, `R2-A4-SPnat-D2-32k` 0.227 %, `R2-A10-MinGram-P1r3-D2-32k` 0.207 %, `R2-A10-MinGram-P1r3-D2-48k` 0.151 %.
  - The normal draws come from `default_rng(12346)`.
- **stress:** the same, with σ = 0.435 % for every candidate.
  - That value is the pooled confirm 5-seed s.d. over all 18 candidates (DECISION.md §7.6).
  - It is inflated by the two SentencePiece D1 arms, and it is about 2 to 40 times the large-scale seed s.d.s observed here. It is a deliberately pessimistic bound.

| variant | SP-32k vs MinGram-32k | SP-32k vs MinGram-48k | MinGram 48k vs 32k | BPE 32k vs 16k | P(SP-32k lowest in top set) |
|---|---|---|---|---|---|
| primary (DECISION.md bootstrap) | -0.62 % [-1.04, -0.31], p 0.0004 | -0.90 % [-1.35, -0.56], p <1e-4 | +0.28 % [+0.07, +0.47], p 0.0110 | +0.03 % [-0.30, +0.32], p 0.9080 | 99.98 % |
| rescaled seed bootstrap | -0.62 % [-1.05, -0.27], p 0.0004 | -0.90 % [-1.37, -0.52], p <1e-4 | +0.28 % [+0.02, +0.52], p 0.0334 | +0.03 % [-0.35, +0.38], p 0.9210 | 99.98 % |
| own s.d. (parametric) | -0.62 % [-1.21, -0.10], p 0.0202 | -0.90 % [-1.46, -0.40], p 0.0004 | +0.28 % [-0.09, +0.67], p 0.1426 | +0.03 % [-0.37, +0.41], p 0.9514 | 98.99 % |
| stress, 0.435 % (parametric) | -0.62 % [-1.57, +0.28], p 0.1804 | -0.90 % [-1.83, +0.01], p 0.0512 | +0.28 % [-0.58, +1.16], p 0.5286 | +0.03 % [-0.86, +0.91], p 0.9848 | 89.76 % |

- **SP-32k vs both MinGram builds survives** the rescaled and the own-s.d. variants.
- **MinGram 48k vs 32k (+0.28 %) does not survive** the own-s.d. variant: p 0.1426 there.
- **Under the stress bound, none of the within-top-set pairs is significant.** SP-32k is still lowest in 90 % of replicates. In short: SP-32k is lowest under every variant, and its lead is significant unless the large-scale seed noise is far above what was observed.

## 6. Confirm scale vs large scale

### 6.1 Ranking

| candidate | V | confirm bpb (5 seeds) | confirm rank of 5 (of 17) | confirm bpb, seeds 1–2 only (rank) | large bpb (seeds 1–2) | large rank | Δ vs baseline: confirm → large |
|---|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` | 32k | 1.22584 | 1 (1) | 1.22794 (2) | 1.13373 | 1 | -1.62 % → -1.34 % |
| `R2-A10-MinGram-P1r3-D2-48k` | 48k | 1.22636 | 2 (2) | 1.22566 (1) | 1.14407 | 3 | -1.58 % → -0.44 % |
| `R2-A10-MinGram-P1r3-D2-32k` | 32k | 1.22844 | 3 (3) | 1.22829 (3) | 1.14086 | 2 | -1.41 % → -0.72 % |
| `A1-P1r3-D2-32k` | 32k | 1.23597 | 4 (6) | 1.23611 (4) | 1.14940 | 5 | -0.81 % → +0.03 % |
| `A1-P1r3-D2-16k` | 16k | 1.24607 | 5 (9) | 1.24523 (5) | 1.14908 | 4 | baseline |

- **2 of the 10 candidate pairs change order** (Kendall τ_b = 0.60), and both are vocabulary-size pairs:
  - BPE 32k vs BPE 16k;
  - MinGram 48k vs MinGram 32k.
  - Every other pair keeps its order.
- **The change is not an artefact of the seed count.** With Stage 4 restricted to seeds 1–2, like-for-like with the large arbiter:
  - BPE 32k vs 16k is -0.73 % [-0.90, -0.58], and MinGram 48k vs 32k is -0.21 % [-0.59, +0.07]. Both reverse at the large scale.
  - The confirm-scale best on seeds 1–2 is `R2-A10-MinGram-P1r3-D2-48k`, which is 3rd of the top set at the large scale.

### 6.2 Contrasts, paired across scales

| contrast | type | confirm Δ, 95 % CI | large Δ, 95 % CI | change (pp), 95 % CI | p (change) |
|---|---|---|---|---|---|
| `A1-P1r3-D2-32k` vs `A1-P1r3-D2-16k` | vocabulary size, BPE 16k → 32k | -0.81 % [-1.00, -0.66] | +0.03 % [-0.30, +0.32] | +0.84 [+0.52, +1.19] | <1e-4 |
| `R2-A10-MinGram-P1r3-D2-48k` vs `R2-A10-MinGram-P1r3-D2-32k` | vocabulary size, MinGram 32k → 48k | -0.17 % [-0.47, +0.11] | +0.28 % [+0.07, +0.47] | +0.45 [+0.16, +0.71] | 0.0038 |
| `R2-A4-SPnat-D2-32k` vs `A1-P1r3-D2-32k` | algorithm at 32k: Unigram vs BPE | -0.82 % [-1.24, -0.46] | -1.36 % [-1.79, -1.01] | -0.54 [-0.88, -0.22] | 0.0008 |
| `R2-A10-MinGram-P1r3-D2-32k` vs `A1-P1r3-D2-32k` | algorithm at 32k: MinGram vs BPE | -0.61 % [-0.80, -0.41] | -0.74 % [-0.98, -0.50] | -0.13 [-0.45, +0.19] | 0.4564 |
| `R2-A10-MinGram-P1r3-D2-48k` vs `A1-P1r3-D2-32k` | MinGram 48k vs BPE 32k (algorithm + size) | -0.78 % [-0.99, -0.57] | -0.46 % [-0.77, -0.15] | +0.31 [+0.04, +0.58] | 0.0282 |
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-32k` | top set: SP-32k vs MinGram-32k | -0.21 % [-0.64, +0.19] | -0.62 % [-1.04, -0.31] | -0.41 [-0.75, -0.12] | 0.0058 |
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-48k` | top set: SP-32k vs MinGram-48k | -0.04 % [-0.55, +0.40] | -0.90 % [-1.35, -0.56] | -0.86 [-1.15, -0.59] | <1e-4 |
| `R2-A4-SPnat-D2-32k` vs `A1-P1r3-D2-16k` | vs baseline | -1.62 % [-2.11, -1.24] | -1.34 % [-1.78, -1.03] | +0.29 [+0.01, +0.54] | 0.0414 |
| `R2-A10-MinGram-P1r3-D2-32k` vs `A1-P1r3-D2-16k` | vs baseline | -1.41 % [-1.67, -1.19] | -0.72 % [-0.91, -0.52] | +0.70 [+0.44, +0.97] | <1e-4 |
| `R2-A10-MinGram-P1r3-D2-48k` vs `A1-P1r3-D2-16k` | vs baseline | -1.58 % [-1.87, -1.36] | -0.44 % [-0.62, -0.29] | +1.15 [+0.89, +1.41] | <1e-4 |

The confirm column is recomputed on these 5 candidates. Its point Δs equal `decision.json`, and its CIs are within 0.02 pp of it (§1).

### 6.3 Reading: vocabulary-size effects shrink, and the audit's parameter confound fits

- **The audit's point** (`analysis/audit/audit_lm.json`; AMENDMENT_1.md):
  - PLAN §5 fixes the non-embedding size, not the total size, so the total parameters grow with the vocabulary.
  - At the confirm scale, the 48k model has 2.27x the baseline's parameters and the 32k models have 1.63x.
  - The token embedding is 64–84 % of all parameters there.
  - Tie-breaker (b), bytes/token, rises with vocabulary size. So within a tied set it picks the largest vocabulary, and with it the most parameters.
- **What the large arbiter changes.** The non-embedding part grows 6x, from 1.77M to 10.63M:
  - the embedding share falls to 38–64 %;
  - the parameter ratios fall to 1.74x (48k) and 1.37x (32k).
- **What happens to the vocabulary-size gains:**
  - BPE 16k → 32k goes from -0.81 % to +0.03 %, a change of +0.84 pp [+0.52, +1.19].
  - MinGram 32k → 48k goes from -0.17 % to +0.28 %, a change of +0.45 pp [+0.16, +0.71].
  - Both changes are significant.
  - At the large scale, each step adds 6.29M embedding parameters (BPE 16k → 32k; MinGram 32k → 48k), and they buy nothing, or worse.
- **At equal vocabulary, where the parameter count is the same, the algorithm effects persist:**
  - MinGram vs BPE at 32k: -0.61 % → -0.74 %, a change of -0.13 pp [-0.45, +0.19], not significant;
  - Unigram vs BPE at 32k grows: -0.82 % → -1.36 %, a change of -0.54 pp [-0.88, -0.22].
- **The audit's parameter-adjusted confirm ranking already matched the large one.** The audit measured each candidate's residual against the BPE bpb-vs-log(parameters) curve at the confirm scale:
  - SP-32k -0.857 %, MinGram-32k -0.646 %, MinGram-48k -0.582 %;
  - BPE-32k -0.037 % and baseline +0.034 %.
  - That is the order SP-32k < MinGram-32k < MinGram-48k.
  - Once parameters were accounted for, the confirm scale already gave the large arbiter's order.
- **Why this is consistent with the confound, not proof of it:**
  - Three things changed together between the scales: model width and depth, epochs (1 → 2), and peak LR (1e-3 → 5e-4).
  - Both LRs were chosen on the 16k baseline, and both sit at the low edge of their grids; the Stage 4 finalist re-sweep tried only higher LRs. Larger vocabularies may prefer a different LR at either scale.
  - A second mechanism may contribute: rare tokens. 55.8 % of MinGram-48k's learned tokens occur fewer than 20 times in train_D1 (DECISION.md §4), and a second epoch repeats those occurrences rather than adding new ones. This does not explain SP-32k's lead over MinGram-32k, though: SP-32k has more rare tokens (40.6 % vs 34.9 %).
  - No iso-parameter run, such as a 48k model with a narrower embedding, was made, so these explanations are not separated.

## 7. Parameters and compute per model

Tied input/output embeddings, learned positions. The training-compute estimate is 6 × (non-embedding + tied output) × tokens seen, as in the audit.

| candidate | scale | V | ctx tokens | total | token embedding | position | non-embedding | embedding share | total × baseline | tokens seen | train FLOPs × baseline | T4 train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A1-P1r3-D2-16k` | confirm (d=192, L=4, 1 ep, LR 1e-3) | 16,384 | 265 | 4,969,536 | 3,145,728 | 50,880 | 1,772,928 | 64.3 % | 1.000 | 7.77M | 1.000 | 62 |
| `A1-P1r3-D2-16k` | large (d=384, L=6, 2 ep, LR 5e-4) | 16,384 | 265 | 17,020,032 | 6,291,456 | 101,760 | 10,626,816 | 37.6 % | 1.000 | 15.53M | 1.000 | 237 |
| `A1-P1r3-D2-32k` | confirm (d=192, L=4, 1 ep, LR 1e-3) | 32,768 | 251 | 8,112,576 | 6,291,456 | 48,192 | 1,772,928 | 78.1 % | 1.632 | 7.36M | 1.553 | 95 |
| `A1-P1r3-D2-32k` | large (d=384, L=6, 2 ep, LR 5e-4) | 32,768 | 251 | 23,306,112 | 12,582,912 | 96,384 | 10,626,816 | 54.4 % | 1.369 | 14.71M | 1.299 | 318 |
| `R2-A4-SPnat-D2-32k` | confirm (d=192, L=4, 1 ep, LR 1e-3) | 32,768 | 249 | 8,112,192 | 6,291,456 | 47,808 | 1,772,928 | 78.1 % | 1.632 | 7.30M | 1.541 | 94 |
| `R2-A4-SPnat-D2-32k` | large (d=384, L=6, 2 ep, LR 5e-4) | 32,768 | 249 | 23,305,344 | 12,582,912 | 95,616 | 10,626,816 | 54.4 % | 1.369 | 14.59M | 1.289 | 317 |
| `R2-A10-MinGram-P1r3-D2-32k` | confirm (d=192, L=4, 1 ep, LR 1e-3) | 32,768 | 250 | 8,112,384 | 6,291,456 | 48,000 | 1,772,928 | 78.1 % | 1.632 | 7.33M | 1.547 | 94 |
| `R2-A10-MinGram-P1r3-D2-32k` | large (d=384, L=6, 2 ep, LR 5e-4) | 32,768 | 250 | 23,305,728 | 12,582,912 | 96,000 | 10,626,816 | 54.4 % | 1.369 | 14.65M | 1.294 | 317 |
| `R2-A10-MinGram-P1r3-D2-48k` | confirm (d=192, L=4, 1 ep, LR 1e-3) | 49,152 | 244 | 11,256,960 | 9,437,184 | 46,848 | 1,772,928 | 84.3 % | 2.265 | 7.15M | 2.098 | 129 |
| `R2-A10-MinGram-P1r3-D2-48k` | large (d=384, L=6, 2 ep, LR 5e-4) | 49,152 | 244 | 29,594,880 | 18,874,368 | 93,696 | 10,626,816 | 64.1 % | 1.739 | 14.30M | 1.606 | 405 |

- **At both scales, the three 32k models are iso-parameter within 768 parameters.** The only difference is the position table, which depends on ctx.
  - So the SP-32k vs MinGram-32k and algorithm-at-32k contrasts are free of the parameter confound.
  - The SP-32k vs MinGram-48k contrast is not: MinGram-48k has +6.29M parameters at the large scale.

## 8. What this supports, and what it does not

**Supported, on dev_strict (report-only arbiter):**

- **The amended rule's outcome.** Within the pre-registered top set, SP-32k has the lowest large-arbiter dev bpb:
  - vs MinGram-32k: -0.62 % [-1.04, -0.31], p 0.0004;
  - vs MinGram-48k: -0.90 % [-1.35, -0.56], p <1e-4;
  - 2 seeds each, hierarchical cluster bootstrap.
  - This is the evidence AMENDMENT_1.md §5 asks the claim to state.
- **The fallback ordering of AMENDMENT_1.md §3 matches the large arbiter.** If exact HF equivalence fails, the fallback is MinGram-32k, and MinGram-48k is higher than it at this scale: +0.28 % [+0.07, +0.47], p 0.0110. The difference does not survive the own-s.d. seed variant (§5).
- **The reason for replacing tie-breaker (b) is borne out.** Bytes/token rewards vocabulary size, and at the large scale the vocabulary-size gains do not carry over.

**Not supported, or not shown:**

- **Anything on test.** No test LM number was read. The one-shot test result will be reported whatever it is (AMENDMENT_1.md §4–§5).
- **A pre-registered claim.** The large arbiter was added beyond PLAN §8(ii), and the top-set members are exploratory round-2 tokenizers selected on dev, so winner's-curse optimism applies.
- **Robustness to much larger seed noise.** Under the stress bound (§5), the within-top-set differences are not significant.
- **An edge on news-like text.** On the newspaper clusters, SP-32k and MinGram-32k are tied at the large scale (§4).
- **That vocabulary size never matters.** Only two model scales were tested, and the LR was tuned on the baseline only.
- **The packaging condition of AMENDMENT_1.md §3.** Whether the HF `tokenizer.json` exactly equals the canonical SentencePiece + newline encoder is a separate check, not assessed here.

## 9. Files and reproduction

| file | content |
|---|---|
| `analysis/amendment1_stats.py` | every computation. It imports `decide.py` for the bootstrap, loaders and clusters, and reads no test data |
| `analysis/amendment1_report.py` | renders this file from the JSON and asserts every qualitative statement it makes |
| `analysis/amendment1_stats.json` | all numbers: per-seed bpb, all 10 pairs at both scales, per source, leave-one-cluster-out, sensitivities, parameters, input hashes |
| `lm/colab_results/77e1368773fc/results/large__*.json`, `large_lrsweep__*.json`, `../large_lr_choice.json` | the large-arbiter records, copied from the Colab Drive folder |

To reproduce (single process, about a second, deterministic), run from `F:\Hindko\_tokenizer`: `PYTHONIOENCODING=utf-8 python analysis/amendment1_stats.py && PYTHONIOENCODING=utf-8 python analysis/amendment1_report.py`.
