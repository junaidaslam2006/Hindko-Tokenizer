# Hindko tokenizer: the one-shot test result (PLAN §7.6)

Generated 2026-09-27T00:43:50Z by `analysis/test_analysis.py` (sha256 `ab3de1546fd2…`) and `analysis/test_report.py`. The bootstrap is `analysis/decide.py`'s own code, imported unchanged (sha256 `5ad1170722a6…`, the same as in `decision.json`). Machine-readable twin: `analysis/test_results.json`.

- **The strict test split, used once.** 491 documents, 1,451,026 bytes, **27 clusters** (15 newspaper, 6 book, 6 web). These are the only LM numbers on test in this study.
- **Nothing is re-selected (PLAN §7.6).** The pre-registered pick stays `R2-A10-MinGram-P1r3-D2-48k` (`DECISION.md`). The released default stays `R2-A4-SPnat-D2-32k` (`AMENDMENT_1.md`, whose sha256 is still `9a966033ffdb…`); its exact-HF-export condition is met (`release_build/sp32k/equivalence.json`). Test results are reported whatever they are.
- **Scale.** Every number is at the **confirm scale** unless it is marked *large*: d=192, L=4, H=4, the full permissive train split, 1 epoch, LR 1e-3, seeds 1–5. That is the dev Stage 4 recipe that ranked the candidates.
- **The test models are the dev models.** Every test record's training-loss curve is identical to its dev record's (42/42 records, §1). Only the evaluation text differs.
- Δ = (bpb_a − bpb_b)/bpb_b; negative means a is better. Brackets are 95 % bootstrap CIs. p is two-sided, and <1e-4 means 0 of 10,000 replicates. "(r)" marks a dev CI recomputed here because `decision.json` has no entry in that orientation.

## 0. Summary

| tokenizer | role | test bpb (5 seeds) | seed s.d. | Δ vs baseline, test | p | Δ vs baseline, dev | change dev → test (pp) |
|---|---|---|---|---|---|---|---|
| `R2-A10-MinGram-P1r3-D2-48k` | pre-registered pick (DECISION.md) | 1.20543 | 0.06 % | -1.32 % [-1.47, -1.16] | <1e-4 | -1.58 % [-1.87, -1.36] | +0.27 [+0.02, +0.55] |
| `R2-A10-MinGram-P1r3-D2-32k` | top-set member (Amendment 1 fallback) | 1.20620 | 0.21 % | -1.25 % [-1.48, -1.04] | <1e-4 | -1.41 % [-1.66, -1.19] | +0.16 [-0.04, +0.39] |
| `R2-A4-SPnat-D2-32k` | released default (AMENDMENT_1.md) | 1.20911 | 0.20 % | -1.01 % [-1.33, -0.72] | <1e-4 | -1.62 % [-2.11, -1.24] | +0.61 [+0.17, +1.12] |
| `R2-A1-P1r3-D2-48k` | equal-vocabulary BPE control for the 48k pick (report-only) | 1.21111 | 0.12 % | -0.85 % [-1.05, -0.65] | <1e-4 | -0.99 % [-1.21, -0.78] | +0.14 [-0.10, +0.38] |
| `A1-P1r3-D2-32k` | round-1 confirmatory winner (report-only) | 1.21459 | 0.09 % | -0.57 % [-0.75, -0.38] | <1e-4 | -0.81 % [-1.00, -0.66] | +0.24 [+0.03, +0.49] |
| `A1-P1r3-D2-16k` | baseline (standard recipe) | 1.22150 | 0.11 % | – | – | – | – |

- **(a) The pre-registered claim holds on test.** `R2-A10-MinGram-P1r3-D2-48k` vs `A1-P1r3-D2-16k`: **Δ = -1.32 % [-1.47, -1.16], p <1e-4**. Dev: -1.58 % [-1.87, -1.36]. The test CI excludes 0 and the sign agrees with dev, so the PLAN §7.5–7.6 improvement claim stands. The test gain is smaller than the dev gain by 0.27 pp [+0.02, +0.55]. That fits selection on dev (winner's curse), but every gain over the baseline shrank on test, round-1 ones included (§9).
- **(b) The released tokenizer improves on the baseline on test too.** `R2-A4-SPnat-D2-32k`: Δ = -1.01 % [-1.33, -0.72], p <1e-4 (dev -1.62 % [-2.11, -1.24]). Its gain shrank the most of the three top-set members: by 0.61 pp [+0.17, +1.12].
- **(c)** `R2-A10-MinGram-P1r3-D2-32k`: Δ = -1.25 % [-1.48, -1.04], p <1e-4 (dev -1.41 % [-1.66, -1.19]).
- **Within the top set, on test, the released tokenizer has the highest bpb of the three**: MinGram-48k 1.20543 < MinGram-32k 1.20620 < SP-32k 1.20911. SP-32k vs MinGram-48k is +0.31 % [-0.02, +0.63] (p 0.0648), and vs MinGram-32k +0.24 % [-0.13, +0.60] (p 0.1896). Neither CI excludes 0, so the three stay statistically tied, but SP-32k ranks 3rd of the three in 89.6 % of the replicates. On dev it had the lowest mean. Applying the PLAN §7.3 top-set rule to the 6 test tokenizers (report-only) gives the same three members.
- **(d) Forking paths: every top-set member beats the round-1 confirmatory winner `A1-P1r3-D2-32k` on test.** MinGram-48k -0.75 % [-0.93, -0.58], MinGram-32k -0.69 % [-0.92, -0.46], SP-32k -0.45 % [-0.71, -0.20]. `A1-P1r3-D2-32k` itself beats the baseline: -0.57 % [-0.75, -0.38] (dev -0.81 % [-1.00, -0.66]).
- **(e) At equal vocabulary, and so at equal parameter count, the Unigram-family tokenizers beat BPE on test:** MinGram-48k vs BPE-48k -0.47 % [-0.64, -0.31]; SP-32k vs BPE-32k -0.45 % [-0.71, -0.20]; MinGram-32k vs BPE-32k -0.69 % [-0.92, -0.46].
- **(f) Dev vs test:** 11 of 13 contrasts keep their sign. The 2 that flip are SP-32k's two within-top-set comparisons, whose dev Δs were small and not significant (-0.04 % [-0.55, +0.40] (r) vs MinGram-48k; -0.21 % [-0.65, +0.18] (r) vs MinGram-32k). 4 changes exclude 0 in the paired dev-vs-test bootstrap (§9). The rank order of the 6 correlates at Kendall τ_b = 0.73.
- **Large arbiter on test (report-only; 2 seeds, MinGram-32k 1 seed).** Mean test bpb: SP-32k 1.11898 < MinGram-32k 1.12240 (n = 1) < MinGram-48k 1.12477 < baseline 1.12767. That is the same order as the dev large run on which the amended rule chose SP-32k. SP-32k vs MinGram-48k: -0.51 % [-0.86, -0.22]. SP-32k vs MinGram-32k: -0.30 % with only one MinGram-32k seed; once seed noise is added, [-0.95, +0.27], p 0.2996, so that gap is **not established**.
- **Validation:** 42 result records (35 confirm + 7 large), 1,512 checks, all passed. bpb recomputes bit-exactly. The baseline is bitwise identical in both final-test bundles (5/5 run pairs with bitwise identical per-document bits). Parity with the CPU reference passes in both bundles.

## 1. Inputs, collection and validation

`lm/collect_final_test.py` (sha256 `71c7690c7556…`) copied both Colab folders from `G:\My Drive\hindko_lm_out_final_test\` to `lm/colab_results/final_test/<bundle>/`. Every file was copied byte for byte and its sha256 re-checked after the copy, including the `large` records. Outputs: `results_table_final_test.json/.csv`, `validation_final_test.json`, `test_clusters.csv`.

| bundle | role | candidates | confirm records | large records | built (FINAL_TEST_LOG.json) |
|---|---|---|---|---|---|
| `f54c929ba1ab` | the pre-registered one-shot test list (decision.json test_candidates_fixed) | `A1-P1r3-D2-16k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A4-SPnat-D2-32k`, `R2-A10-MinGram-P1r3-D2-32k` | 20 | 7 | 2026-09-26T21:01:52Z → 2026-09-26T21:07:11Z |
| `94175d26497f` | supplement added after the adversarial audit, before any test LM number existed | `A1-P1r3-D2-16k`, `A1-P1r3-D2-32k`, `R2-A1-P1r3-D2-48k` | 15 | 0 | 2026-09-26T21:10:36Z → 2026-09-26T21:13:01Z |

**Per-record checks.** They are the checks `lm/collect_colab.py` applied to dev, with the test constants:

- **Identity:** bundle sha256 = the staging `manifest.json` = the bundle id; `hk_lm.py` sha256 `3f78ad799f56…` in every record, the bundle and the local copy; the file key equals the record's stage, candidate, seed and LR; the candidate is in the bundle and in its WAVES file; `hk_lm.identity_mismatches` against `hk_lm.run_identity(...)` (protocol, bundle, recipe, candidate, seed, LR, model, budget, epochs, steps, ctx, device, AMP dtype) is empty.
- **Protocol fields and exact LR:** confirm = recipe `confirm`, seeds 1–5, LR exactly 1e-3 (float `==`, and `optimizer.peak_lr` equal), `preregistered: true`, 3,663 steps, 1 epoch. Large = recipe `large`, seeds 1–2, LR exactly 5e-4, `preregistered: false`, 7,325 steps, 2 epochs. Model d/L/H per recipe, n_vocab and ctx equal to the bundle, ctx = round(1,536 / train bytes per token), tied embeddings, dropout 0. AdamW (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0, 50 warm-up steps, cosine to 10 %. steps = steps_done = len(train_loss), all losses finite.
- **491 per-document arrays:** `bits`, `bytes`, `ntok`, `uids`, `doc_index` all have length 491. `doc_index` = 0..490. `uids` equal the bundle's test documents and the uids of `data/test_strict.jsonl` (sha256 `a74c33ad…`), in uid order. `bytes` equal the bundle (Σ = 1,451,026). `ntok` = the bundle's test token count + 1 (EOT). All bits are finite and > 0.
- **Bit-exact bpb:** `float(np.sum(bits) / np.sum(bytes))` **==** the reported `bpb` == `dev.bpb` in every record, and `sum_bits` == Σbits.
- **Evaluation window, curve, device:** ctx and stride ctx/2, BOS given, EOT predicted; the learning curve on the bundle's fixed 159-document, 505,702-byte test subset at 25/50/75/100 %; cuda / Tesla T4 / fp16 training / fp32 evaluation, strict determinism, tf32 off, equal to `runtime_pin.json`. bytes_seen within 1 % of the budget (largest deviation 0.60 %).
- **The dev validator agrees.** `collect_colab.validate_run` itself was also run on all 35 confirm records, with its three dev constants (836 documents, dev bytes, subset bytes) set to the test values. Its check list and verdicts are identical to the ones above.

**Result: 42 / 42 records pass every check (1,512 checks).**

Further checks:

- **Same models as dev (colab/FINAL_TEST.md check (b)).** For each of the 42 records, the dev record of the same stage, candidate and seed was found (confirm: the non-duplicate Stage 4 record of `results_table.json`; large: `77e1368773fc/results/large__*`). The training-loss curve, the R3 lowest-norm embeddings, the step count, ctx, tokens and bytes seen, the parameter counts, the optimiser and the runtime (GPU, torch, CUDA, dtypes) are identical in all 42. So each test number is the dev-selected model, evaluated on test.
- **The baseline is bitwise identical across the two final-test bundles:** 5/5 run pairs with bitwise identical per-document bits. Per-document bits, bytes, uids, token counts, training loss, curve and R3 are all equal. The copy in `94175d26497f` is dropped from the pooled analysis, so no run counts twice.
- **Completeness.** Confirm: 4 × 5 records in `f54c929ba1ab` and 3 × 5 in `94175d26497f`, complete. Large: 7 of the 8 records Amendment 1 allows. `large__R2-A10-MinGram-P1r3-D2-32k__lr5e-4__s2` is missing: the log's last line is `23:00:35 [large|R2-A10-MinGram-P1r3-D2-32k|s2|lr5e-4] step 6588/7325 loss 5.1614 lr 6.13e-05 292.3s`, with no DONE and no 'finished' line after it. The Colab session disconnected there, and the free GPU quota was then exhausted, so MinGram-32k has **one** large test seed.
- **Parity (CPU ↔ GPU, fixed tiny config, on a 29-document, 98,441-byte test slice):** fp16 +0.534 %, fp32 +0.008 % against the CPU reference 1.689204 bpb (tolerance 1 %). Both bundles pass and give bitwise identical parity runs. On dev the fp16 figure was +0.447 %, on a dev slice.
- **Clusters.** The 'cluster' field of `data/test_strict.jsonl` gives 27 clusters (book 6, newspaper 15, web 6). For all 491 documents it equals the split manifest's `group` with per-record web groups collapsed to the site (the dev rule). The raw manifest has 34 groups (web 13). The largest cluster is a book cluster with 18.2 % of test bytes.
- **The released files encode test exactly as the LM saw it.** Per `release_build/sp32k/equivalence.json`, the released `tokenizer.json` gives the canonical ids on 491/491 test documents, the re-scored model changed 0 of them relative to the `sp.model` the LM bundles used, and its 199,390 test tokens equal the bundle's. SP-32k's test numbers therefore apply to the released files.

## 2. Method, as run

- **The DECISION.md bootstrap, unchanged:** `decide.Boot` and `decide.compare`; 10,000 replicates, `numpy.random.default_rng(12345)`. In each replicate the 27 test clusters are resampled with replacement within each source (book 6, newspaper 15, web 6), each candidate's 5 seeds are resampled with replacement, independently per candidate, and bpb = Σbits ÷ Σbytes, averaged over the resampled seeds. RNG order: cluster draws (sources sorted), then seed draws (candidates in sorted id order).
- **Primary test bootstrap:** all 6 confirm-scale test tokenizers of both bundles, in sorted id order: `A1-P1r3-D2-16k`, `A1-P1r3-D2-32k`, `R2-A1-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k`, `R2-A10-MinGram-P1r3-D2-48k`, `R2-A4-SPnat-D2-32k`. The choice of stream moves the CI endpoints of (a) by at most 0.01 pp (§4.1).
- **Dev side:** the same 6 candidates' Stage 4 records via `decide.load_stage`, the 32 dev clusters via `decide.dev_clusters`, and `decide.Boot`. Its point Δs equal `decision.json` exactly wherever it has the same comparison (10 contrasts). Their CI endpoints differ by at most 0.014 pp, because the seed-draw stream differs with 6 candidates instead of 18. The tables show dev CIs of record where they exist.
- **Dev vs test (addition, report-only):** one `decide.Boot` over dev and test together, with 6 strata (split × source). Dev and test clusters are resampled independently. Each candidate's seed draw is shared by the two splits, because the test models are the dev models. The change is Δ_test − Δ_dev per replicate, in pp. Its dev cluster draws equal the dev bootstrap's exactly (asserted).
- **Holm:** PLAN pre-registers one test comparison, (a); it is not corrected. For (b)–(e), a Holm column over those 7 report-only comparisons is shown as a conservative reading. TOST margin δ = 0.3 % of the test baseline mean = 0.003665 bpb.

## 3. Every tokenizer on test (confirm scale, 5 seeds each)

| # | tokenizer | role | bundle | V | total params | test bytes/token | test bpb | per-seed test bpb (s1–s5) | seed s.d. | dev bpb (dev rank of 17) | Δ vs best of 6 on test | Holm p (5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `R2-A10-MinGram-P1r3-D2-48k` | pre-registered pick (DECISION.md) | `f54c929ba1ab` | 48k | 11.26M | 7.3734 | 1.20543 | 1.20479 / 1.20590 / 1.20634 / 1.20541 / 1.20470 | 0.06 % | 1.22636 (2) | best | – |
| 2 | `R2-A10-MinGram-P1r3-D2-32k` | top-set member (Amendment 1 fallback) | `f54c929ba1ab` | 32k | 8.11M | 7.2750 | 1.20620 | 1.20588 / 1.20631 / 1.20939 / 1.20694 / 1.20246 | 0.21 % | 1.22844 (3) | +0.06 % [-0.15, +0.30] | 0.5822 |
| 3 | `R2-A4-SPnat-D2-32k` | released default (AMENDMENT_1.md) | `f54c929ba1ab` | 32k | 8.11M | 7.2773 | 1.20911 | 1.21299 / 1.20906 / 1.20803 / 1.20625 / 1.20919 | 0.20 % | 1.22584 (1) | +0.31 % [-0.02, +0.63] | 0.1296 |
| 4 | `R2-A1-P1r3-D2-48k` | equal-vocabulary BPE control for the 48k pick (report-only) | `94175d26497f` | 48k | 11.26M | 7.3526 | 1.21111 | 1.20930 / 1.21196 / 1.21295 / 1.21101 / 1.21034 | 0.12 % | 1.23374 (5) | +0.47 % [+0.31, +0.64] | <1e-4 |
| 5 | `A1-P1r3-D2-32k` | round-1 confirmatory winner (report-only) | `94175d26497f` | 32k | 8.11M | 7.2460 | 1.21459 | 1.21574 / 1.21529 / 1.21508 / 1.21360 / 1.21324 | 0.09 % | 1.23597 (6) | +0.76 % [+0.59, +0.94] | <1e-4 |
| 6 | `A1-P1r3-D2-16k` | baseline (standard recipe) | `f54c929ba1ab` | 16k | 4.97M | 6.9485 | 1.22150 | 1.22096 / 1.22014 / 1.22208 / 1.22082 / 1.22351 | 0.11 % | 1.24607 (9) | +1.33 % [+1.18, +1.49] | <1e-4 |

- The dev order of these 6 was SP-32k < MinGram-48k < MinGram-32k < BPE-48k < BPE-32k < baseline. On test the three top-set members swap places among themselves; the three BPE tokenizers keep theirs.
- Pooled relative seed s.d. over the 6: 0.142 % on test, 0.163 % on dev (same models).

## 4. (a) The pre-registered comparison (PLAN §7.5–7.6)

**`R2-A10-MinGram-P1r3-D2-48k` − `A1-P1r3-D2-16k` on test: Δ = -1.316 % [-1.467, -1.164], p <1e-4** (bpb 1.20543 vs 1.22150; bootstrap s.e. 0.077 pp; 80 %-power detectable effect 0.22 %).

- Dev, of record: -1.582 % [-1.868, -1.356], p <1e-4.
- **Rule:** improvement claim on test iff the test 95 % CI excludes 0 and the test Delta has the dev sign (PLAN §7.5-7.6; colab/FINAL_TEST.md §5); if the sign contradicts dev, report it and do not re-select.
- **Verdict: the improvement claim holds on test.** The test CI excludes 0, and the test Δ has the dev sign.
- The test gain is smaller than the dev gain by 0.266 pp [+0.019, +0.553], p 0.0356 (paired dev-vs-test bootstrap). The test point lies just outside the dev 95 % CI [-1.868, -1.356].
- **Clusters:** MinGram-48k is better in 24 of the 27 clusters (98.8 % of test bytes). Leaving any one cluster out keeps Δ between -1.361 % and -1.284 %.
- **Per source** (same replicates, restricted to one source's clusters; report-only): book (6 clusters) -1.19 % [-1.34, -1.02]; newspaper (15 clusters) -1.55 % [-1.80, -1.29]; web (6 clusters) -1.45 % [-2.19, -0.70].

### 4.1 Sensitivity of (a) and (b)

| analysis | (a) MinGram-48k vs baseline | (b) SP-32k vs baseline |
|---|---|---|
| **primary** (6 candidates, 27 clusters, 5 seeds) | -1.32 % [-1.47, -1.16] | -1.01 % [-1.33, -0.72] |
| bundle_f54c_only: Boot over the 4 candidates of the pre-registered test bundle f54c929ba1ab only (its own seed-draw stream) | -1.32 % [-1.47, -1.17] | -1.01 % [-1.34, -0.72] |
| pair_only: Boot over the pre-registered pair only (PLAN §7.6: the chosen tokenizer and the baseline) | -1.32 % [-1.47, -1.17] | – |
| raw_manifest_groups: PLAN §6 sensitivity: manifest groups as they are (per-record web groups kept): 34 groups | -1.32 % [-1.47, -1.17] | -1.01 % [-1.33, -0.72] |
| seeds_1to3: only seeds 1-3 (the 3 seeds PLAN §5 planned) | -1.26 % [-1.42, -1.10] | -0.90 % [-1.25, -0.57] |
| p = 2 min((#{D*<=0}+1)/(R+1), (#{D*>=0}+1)/(R+1)) | p_plus1 0.0002 | p_plus1 0.0002 |

The claim holds in every row. With seeds 1–3 only, both Δs are smaller in size, and both CIs still exclude 0.

## 5. (b) The released tokenizer and (c) MinGram-32k

| contrast | test Δ | p | Holm p (7, report-only) | dev Δ | test book (6) | test newspaper (15) | test web (6) |
|---|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-16k` | -1.01 % [-1.33, -0.72] | <1e-4 | <1e-4 | -1.62 % [-2.11, -1.24] | -1.02 % [-1.53, -0.64] | -0.97 % [-1.28, -0.63] | -1.44 % [-2.31, -0.32] |
| `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-16k` | -1.25 % [-1.48, -1.04] | <1e-4 | <1e-4 | -1.41 % [-1.66, -1.19] | -1.14 % [-1.36, -0.91] | -1.47 % [-1.75, -1.18] | -1.35 % [-2.37, -0.54] |

- **SP-32k improves on the baseline on test in every source.** Each per-source CI lies below 0.
- **Its dev margin did not fully carry over.** Δ vs baseline went from -1.62 % [-2.11, -1.24] to -1.01 % [-1.33, -0.72]: a change of +0.61 pp [+0.17, +1.12], the largest of the 13 contrasts. On dev, SP-32k's seed s.d. was 0.23 %, the largest of the top set; its dev Δ CI was also the widest of the three.
- **Newspaper text:** on test SP-32k is behind both MinGram builds there (vs MinGram-48k +0.59 % [+0.24, +0.97]; vs MinGram-32k +0.51 % [+0.14, +0.89]). AMENDMENT_1_STATS.md §4 found the same at the dev confirm scale (+0.68 % and +0.59 %).

**The released tokenizer's standing, worded as AMENDMENT_1.md §5 requires:**

1. It is in the statistically tied top set at the confirm scale (DECISION.md §3, on dev). Applying the same PLAN §7.3 rule to the 6 test tokenizers, report-only, gives the same three members (§8).
2. It was chosen by the amended rule: the lowest large-arbiter dev bpb within that top set (1.13373, against 1.14086 for MinGram-32k and 1.14407 for MinGram-48k; -0.62 % [-1.04, -0.31] and -0.90 % [-1.35, -0.56], 2 seeds each; AMENDMENT_1_STATS.md §0). The report-only large run on test keeps that order (§10).
3. Its test result: -1.01 % [-1.33, -0.72] vs the baseline at the confirm scale, p <1e-4; highest test bpb of the three top-set members (not significantly). It must not be called 'best on test'.

## 6. (d) The forking-paths check: the round-2 top set vs the round-1 confirmatory winner

Round 2 was exploratory: its tokenizers were built after the round-1 results were seen (DECISION.md §8, deviation 4). Restricted to the pre-registered round 1, the rule chooses `A1-P1r3-D2-32k` (DECISION.md §7.8). If the round-2 gain were an artefact of selecting on dev, the top set would not beat that tokenizer on sealed test data.

| contrast | test Δ | p | Holm p (7) | dev Δ | change (pp) | clusters where a is better |
|---|---|---|---|---|---|---|
| `R2-A10-MinGram-P1r3-D2-48k` − `A1-P1r3-D2-32k` | -0.75 % [-0.93, -0.58] | <1e-4 | <1e-4 | -0.78 % [-1.00, -0.57] (r) | +0.02 [-0.22, +0.28] | 25 / 27 |
| `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-32k` | -0.69 % [-0.92, -0.46] | <1e-4 | <1e-4 | -0.61 % [-0.80, -0.41] | -0.08 [-0.28, +0.10] | 27 / 27 |
| `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-32k` | -0.45 % [-0.71, -0.20] | 0.0002 | 0.0002 | -0.82 % [-1.24, -0.46] | +0.37 [-0.01, +0.79] | 18 / 27 |
| `A1-P1r3-D2-32k` − `A1-P1r3-D2-16k` | -0.57 % [-0.75, -0.38] | <1e-4 | <1e-4 | -0.81 % [-1.00, -0.66] | +0.24 [+0.03, +0.49] | 23 / 27 |

- **All three top-set members beat `A1-P1r3-D2-32k` on test**, and each stays significant after Holm. The round-2 advantage over the best pre-registered round-1 tokenizer is therefore not only a product of selecting on dev: it replicates on the sealed test split.
- MinGram-48k's and MinGram-32k's margins over `A1-P1r3-D2-32k` are the same size on test as on dev (changes +0.02 and -0.08 pp, both CIs include 0). SP-32k's margin shrank from -0.82 % [-1.24, -0.46] to -0.45 % [-0.71, -0.20]; that change's CI [-0.01, +0.79] just includes 0.
- The confirmatory round-1 claim also holds on test: `A1-P1r3-D2-32k` vs the baseline -0.57 % [-0.75, -0.38] (dev -0.81 % [-1.00, -0.66]).

## 7. (e) Equal-vocabulary controls

At equal vocabulary the models have the same total parameter count (within 768 parameters, from the ctx-dependent position table), so these contrasts are free of the vocabulary-size/parameter confound that AMENDMENT_1.md describes.

| contrast | vocabulary | test Δ | p | dev Δ | change (pp) | test book | test newspaper | test web |
|---|---|---|---|---|---|---|---|---|
| `R2-A10-MinGram-P1r3-D2-48k` − `R2-A1-P1r3-D2-48k` | 48k | -0.47 % [-0.64, -0.31] | <1e-4 | -0.60 % [-0.88, -0.37] | +0.13 [-0.10, +0.40] | -0.50 % [-0.75, -0.29] | -0.40 % [-0.55, -0.24] | -0.63 % [-1.10, -0.31] |
| `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-32k` | 32k | -0.45 % [-0.71, -0.20] | 0.0002 | -0.82 % [-1.24, -0.46] | +0.37 [-0.01, +0.79] | -0.64 % [-1.10, -0.36] | -0.07 % [-0.40, +0.30] | -0.61 % [-1.23, +0.24] |
| `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-32k` | 32k | -0.69 % [-0.92, -0.46] | <1e-4 | -0.61 % [-0.80, -0.41] | -0.08 [-0.28, +0.10] | -0.76 % [-1.06, -0.48] | -0.57 % [-0.80, -0.36] | -0.52 % [-1.11, -0.13] |

- **The algorithm effect holds on test at equal vocabulary.** MinGram beats byte-level BPE at 48k and at 32k, and SentencePiece Unigram beats it at 32k. All three CIs exclude 0.
- SP-32k vs BPE-32k is not significant on newspaper text (-0.07 % [-0.40, +0.30]). The two MinGram contrasts are below 0 in every source.
- **Vocabulary size at this scale (report-only):** BPE 32k → 48k -0.29 % [-0.46, -0.12] (dev -0.18 % [-0.36, +0.04]); MinGram 32k → 48k -0.06 % [-0.29, +0.15] (dev -0.17 % [-0.47, +0.11]); the latter is TOST-equivalent within ±0.3 % on test (its 90 % CI lies inside ±δ). The 32k→48k step buys little at the confirm scale, on dev and on test.

## 8. The top set on test (report-only; never re-selects)

| pair (Δ = a vs b) | test Δ | p | 90 % CI (bpb) | TOST ±0.3 % | clusters where a is better | test book | test newspaper | test web | dev Δ |
|---|---|---|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-48k` | +0.31 % [-0.02, +0.63] | 0.0648 | [+0.00043, +0.00688] | not shown | 8 / 27 (8 % of bytes) | +0.17 % [-0.42, +0.59] | +0.59 % [+0.24, +0.97] | +0.01 % [-0.43, +0.70] | -0.04 % [-0.55, +0.40] (r) |
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-32k` | +0.24 % [-0.13, +0.60] | 0.1896 | [-0.00084, +0.00651] | not shown | 9 / 27 (24 % of bytes) | +0.12 % [-0.50, +0.56] | +0.51 % [+0.14, +0.89] | -0.10 % [-0.72, +1.02] | -0.21 % [-0.65, +0.18] (r) |
| `R2-A10-MinGram-P1r3-D2-48k` vs `R2-A10-MinGram-P1r3-D2-32k` | -0.06 % [-0.29, +0.15] | 0.5822 | [-0.00312, +0.00144] | equivalent | 16 / 27 (54 % of bytes) | -0.05 % [-0.35, +0.21] | -0.08 % [-0.30, +0.15] | -0.10 % [-0.51, +0.54] | -0.17 % [-0.47, +0.11] |

- **Rank within the top set across the 10,000 test replicates:** `R2-A4-SPnat-D2-32k` 1st 2.3 %, 2nd 8.1 %, 3rd 89.6 %; `R2-A10-MinGram-P1r3-D2-48k` 1st 69.1 %, 2nd 29.5 %, 3rd 1.4 %; `R2-A10-MinGram-P1r3-D2-32k` 1st 28.6 %, 2nd 62.5 %, 3rd 9.0 %.
- **PLAN §7.3's rule applied to the 6 test tokenizers** (best by mean `R2-A10-MinGram-P1r3-D2-48k`; Holm over 5 comparisons): the top set is `R2-A10-MinGram-P1r3-D2-48k`, `R2-A10-MinGram-P1r3-D2-32k`, `R2-A4-SPnat-D2-32k`. `R2-A1-P1r3-D2-48k`, `A1-P1r3-D2-32k` and `A1-P1r3-D2-16k` are significantly worse (Holm p <1e-4 / <1e-4 / <1e-4). This is the same set of three as on dev; it is reported, not used.
- **What the test says about SP-32k vs the MinGram builds at this scale:** SP-32k is behind, by +0.31 % and +0.24 %, but neither is significant at 95 %. The newspaper clusters carry the gap (§5). The test gives no support for calling SP-32k better than the MinGram builds at the confirm scale, and no significant evidence that it is worse.

## 9. (f) Dev vs test, every Δ

Change = Δ_test − Δ_dev in pp, with its 95 % CI from the paired dev+test bootstrap (§2). 'Inside' says whether the test point lies inside the dev 95 % CI.

| sec. | contrast | dev Δ | test Δ | same sign | change (pp) | p (change) | inside dev CI | test / dev |
|---|---|---|---|---|---|---|---|---|
| a | `R2-A10-MinGram-P1r3-D2-48k` − `A1-P1r3-D2-16k` | -1.58 % [-1.87, -1.36] | -1.32 % [-1.47, -1.16] | yes | +0.266 [+0.019, +0.553] | 0.0356 | no | 0.83 |
| b | `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-16k` | -1.62 % [-2.11, -1.24] | -1.01 % [-1.33, -0.72] | yes | +0.609 [+0.172, +1.117] | 0.0036 | no | 0.63 |
| c | `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-16k` | -1.41 % [-1.66, -1.19] | -1.25 % [-1.48, -1.04] | yes | +0.161 [-0.042, +0.386] | 0.1170 | yes | 0.89 |
| d | `R2-A10-MinGram-P1r3-D2-48k` − `A1-P1r3-D2-32k` | -0.78 % [-1.00, -0.57] (r) | -0.75 % [-0.93, -0.58] | yes | +0.023 [-0.223, +0.278] | 0.9166 | yes | 0.97 |
| d | `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-32k` | -0.82 % [-1.24, -0.46] | -0.45 % [-0.71, -0.20] | yes | +0.368 [-0.010, +0.786] | 0.0566 | no | 0.55 |
| d | `R2-A10-MinGram-P1r3-D2-32k` − `A1-P1r3-D2-32k` | -0.61 % [-0.80, -0.41] | -0.69 % [-0.92, -0.46] | yes | -0.082 [-0.275, +0.101] | 0.4210 | yes | 1.14 |
| d | `A1-P1r3-D2-32k` − `A1-P1r3-D2-16k` | -0.81 % [-1.00, -0.66] | -0.57 % [-0.75, -0.38] | yes | +0.244 [+0.025, +0.486] | 0.0302 | no | 0.70 |
| e | `R2-A10-MinGram-P1r3-D2-48k` − `R2-A1-P1r3-D2-48k` | -0.60 % [-0.88, -0.37] | -0.47 % [-0.64, -0.31] | yes | +0.129 [-0.102, +0.397] | 0.2826 | yes | 0.78 |
| – | `R2-A1-P1r3-D2-48k` − `A1-P1r3-D2-16k` | -0.99 % [-1.21, -0.78] | -0.85 % [-1.05, -0.65] | yes | +0.139 [-0.100, +0.385] | 0.2452 | yes | 0.86 |
| – | `R2-A1-P1r3-D2-48k` − `A1-P1r3-D2-32k` | -0.18 % [-0.36, +0.04] | -0.29 % [-0.46, -0.12] | yes | -0.106 [-0.327, +0.102] | 0.3056 | yes | 1.59 |
| – | `R2-A10-MinGram-P1r3-D2-48k` − `R2-A10-MinGram-P1r3-D2-32k` | -0.17 % [-0.47, +0.11] | -0.06 % [-0.29, +0.15] | yes | +0.106 [-0.160, +0.383] | 0.4738 | yes | 0.37 |
| – | `R2-A4-SPnat-D2-32k` − `R2-A10-MinGram-P1r3-D2-48k` | -0.04 % [-0.55, +0.40] (r) | +0.31 % [-0.02, +0.63] | **no** | +0.348 [-0.148, +0.882] | 0.1694 | yes | -7.17 |
| – | `R2-A4-SPnat-D2-32k` − `R2-A10-MinGram-P1r3-D2-32k` | -0.21 % [-0.65, +0.18] (r) | +0.24 % [-0.13, +0.60] | **no** | +0.454 [+0.019, +0.898] | 0.0400 | no | -1.14 |

- **Sign:** 11 of 13 contrasts keep their sign. The flips are `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-48k` and vs `R2-A10-MinGram-P1r3-D2-32k`. Their dev Δs were near 0 (both CIs spanned 0) and their test Δs are not significant either, so neither flip contradicts a dev finding.
- **Significant changes (CI of the change excludes 0):** `R2-A10-MinGram-P1r3-D2-48k` − `A1-P1r3-D2-16k` +0.27 pp, `R2-A4-SPnat-D2-32k` − `A1-P1r3-D2-16k` +0.61 pp, `A1-P1r3-D2-32k` − `A1-P1r3-D2-16k` +0.24 pp, `R2-A4-SPnat-D2-32k` − `R2-A10-MinGram-P1r3-D2-32k` +0.45 pp. All are in the direction of a smaller advantage on test for the tokenizer that looked better on dev. Regression after selection on dev predicts this; the next point qualifies it.
- **Every gain over the 16k baseline is smaller on test than on dev** (changes +0.14 to +0.61 pp over the 5 tokenizers), for the round-1 winner BPE-32k (+0.24 pp [+0.03, +0.49]) as well as for the round-2 tokenizers. Selection on dev (winner's curse) predicts a shrink for the selected tokenizers. A shrink shared by all of them is also consistent with the baseline simply doing relatively better on the test text. These data do not separate the two.
- **Rank order of the 6:** Kendall τ_b = 0.73, Spearman ρ = 0.83 between dev and test means.

## 10. Large arbiter on test (report-only)

AMENDMENT_1.md §4 allowed a report-only large-arbiter test run of the top set plus the baseline, at LR 5e-4 fixed from dev and 2 seeds. It ran in bundle `f54c929ba1ab` (d=384, L=6, H=6, 2 epochs, 7,325 steps). It cannot change which tokenizer is released. **MinGram-32k has one seed (n = 1):** its seed-2 run stopped at step 6,588 of 7,325 when Colab disconnected, and the free GPU quota was then exhausted.

| tokenizer | seeds | test bpb per seed | mean test bpb | Δ vs baseline (means) | dev large bpb (s1 / s2) | dev large mean |
|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` | 1,2 | 1.11856 / 1.11940 | 1.11898 | -0.77 % | 1.13319 / 1.13428 | 1.13373 |
| `R2-A10-MinGram-P1r3-D2-32k` | 1 | 1.12240 | 1.12240 | -0.47 % | 1.14016 / 1.14157 | 1.14086 |
| `R2-A10-MinGram-P1r3-D2-48k` | 1,2 | 1.12475 / 1.12480 | 1.12477 | -0.26 % | 1.14336 / 1.14477 | 1.14407 |
| `A1-P1r3-D2-16k` | 1,2 | 1.12831 / 1.12702 | 1.12767 | – | 1.14899 / 1.14917 | 1.14908 |

Three ways to put a CI on these, each with the test cluster draws of the primary bootstrap:

- **2 seeds:** `decide.Boot` on SP-32k, MinGram-48k and the baseline (seeds 1–2 each).
- **seed 1 only:** `decide.Boot` on all four with seed 1. Only cluster noise enters; seed noise is ignored.
- **all seeds + seed noise:** each candidate's available seeds averaged, plus relative seed noise N(0, σ²/n), with σ = the larger of its dev large and dev confirm seed s.d. (SP-32k 0.227 %, MinGram-32k 0.207 %, MinGram-48k 0.151 %, baseline 0.100 %) — AMENDMENT_1_STATS.md §5's 'own s.d.' variant.

| pair (Δ = a vs b) | 2 seeds | seed 1 only | all seeds + seed noise | dev large (2 seeds, of record) | change dev → test (pp) |
|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` vs `A1-P1r3-D2-16k` | -0.77 % [-1.15, -0.40], p <1e-4 | -0.86 % [-1.25, -0.52], p <1e-4 | -0.77 % [-1.27, -0.28], p 0.0018 | -1.34 % [-1.78, -1.03], p <1e-4 | +0.57 [+0.07, +1.13] |
| `R2-A10-MinGram-P1r3-D2-32k` vs `A1-P1r3-D2-16k` | – | -0.52 % [-0.73, -0.25], p 0.0002 | -0.47 % [-0.94, +0.04], p 0.0708 | -0.72 % [-0.91, -0.52], p <1e-4 | +0.24 [-0.01, +0.56] (seed 1) |
| `R2-A10-MinGram-P1r3-D2-48k` vs `A1-P1r3-D2-16k` | -0.26 % [-0.44, -0.08], p 0.0040 | -0.32 % [-0.49, -0.16], p <1e-4 | -0.26 % [-0.55, +0.03], p 0.0854 | -0.44 % [-0.62, -0.29], p <1e-4 | +0.18 [-0.05, +0.44] |
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-32k` | – | -0.34 % [-0.74, -0.10], p <1e-4 | -0.30 % [-0.95, +0.27], p 0.2996 | -0.62 % [-1.04, -0.31], p 0.0004 | +0.27 [-0.21, +0.72] (seed 1) |
| `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-48k` | -0.51 % [-0.86, -0.22], p 0.0014 | -0.55 % [-0.89, -0.25], p 0.0006 | -0.51 % [-1.02, -0.04], p 0.0356 | -0.90 % [-1.35, -0.56], p <1e-4 | +0.39 [-0.10, +0.90] |
| `R2-A10-MinGram-P1r3-D2-48k` vs `R2-A10-MinGram-P1r3-D2-32k` | – | +0.21 % [-0.06, +0.44], p 0.1366 | +0.21 % [-0.31, +0.72], p 0.4218 | +0.28 % [+0.07, +0.47], p 0.0110 | -0.07 [-0.37, +0.20] (seed 1) |

- **The large-scale order on test is the dev order:** SP-32k < MinGram-32k < MinGram-48k < baseline, which is the evidence the amended rule used. Every large-scale Δ keeps its dev sign.
- **SP-32k vs MinGram-48k and vs the baseline are clear** with 2 seeds each (-0.51 % [-0.86, -0.22]; -0.77 % [-1.15, -0.40]), and survive the seed-noise variant (-0.51 % [-1.02, -0.04]; -0.77 % [-1.27, -0.28]).
- **SP-32k vs MinGram-32k is not established on test.** Seed 1 alone gives -0.34 % [-0.74, -0.10], but that ignores seed noise; with it, -0.30 % [-0.95, +0.27], p 0.2996. On dev (2 seeds) it was -0.62 % [-1.04, -0.31].
- **The test margins are smaller than the dev ones** (e.g. SP-32k vs baseline -0.77 % on test, -1.34 % on dev; change +0.57 pp [+0.07, +1.13]).
- **Confirm → large on test** (same clusters; AMENDMENT_1_STATS.md §6.2 style, the three 2-seed arms): `R2-A4-SPnat-D2-32k` vs `A1-P1r3-D2-16k`: -1.01 % → -0.77 %, change +0.24 pp [-0.04, +0.51]; `R2-A10-MinGram-P1r3-D2-48k` vs `A1-P1r3-D2-16k`: -1.32 % → -0.26 %, change +1.06 pp [+0.89, +1.24]; `R2-A4-SPnat-D2-32k` vs `R2-A10-MinGram-P1r3-D2-48k`: +0.31 % → -0.51 %, change -0.82 pp [-1.05, -0.62]. The 48k vocabulary's confirm-scale edge over the baseline shrinks by about 1 pp at the large scale on test too, as it did on dev (+1.15 pp there); SP-32k vs MinGram-48k reverses from +0.31 % to -0.51 %. This is the parameter-confound pattern of AMENDMENT_1_STATS.md §6.3, now seen on test as well.

The large test stage started at 22:16:40 UTC (first record 22:20:48 UTC), after every confirm test number existed (last confirm record 22:14:47 UTC). Amendment 1 had announced it before any test number existed, and it changes nothing.

## 11. Deviations and caveats (all additive)

1. **Six tokenizers on test, not two.** PLAN §7.6 names the chosen tokenizer and the baseline. The two other top-set members were fixed as report-only before test (`decision.json` → `test_candidates_fixed`, 2026-09-26T20:20:43Z). `A1-P1r3-D2-32k` and `R2-A1-P1r3-D2-48k` were added in bundle `94175d26497f` after the adversarial audit; that bundle was built at 2026-09-26T21:13:01Z, before the first test LM record (2026-09-26T21:16:22Z). None of them can re-select.
2. **Seeds 1–5, forced** (`--force-extra-seeds`), as on dev; the test power check did not decide the seed count.
3. **GPU, fp16 training, fp32 evaluation, Tesla T4**, as on dev; the same `hk_lm.py`. Parity fp16 +0.534 % on the test slice (dev +0.447 % on the dev slice), within the 1 % tolerance.
4. **The large test run is report-only and incomplete.** It was unlocked by a `run_all.py` change (sha256 `a11c2d7e9ab2…` vs the bundled `8fbae15a…`; 11 changed lines, all implementing the `--final-test-large` switch, which requires a fixed `--large-lr`; the diff is in `test_results.json` → `validation.run_all_diff_bundled_vs_large_run`). The bundled `hk_lm.py` trained every model, and its training losses equal the dev large runs. MinGram-32k has 1 seed, not 2.
5. **The primary test bootstrap spans both bundles' 6 tokenizers.** PLAN fixes no candidate list for the test bootstrap. The RNG stream moves the CI of (a) by less than 0.01 pp (§4.1).
6. **The paired dev-vs-test bootstrap and the Holm column over (b)–(e) are additions** for reading the results; neither is pre-registered.
7. **5 seeds, 27 clusters.** The book stratum has 6 clusters, the largest with 18 % of test bytes, and web 6 clusters with 34,842 bytes, so per-source CIs are wide. Gaps below about 0.46 % (the 80 %-power detectable effect of SP-32k vs MinGram-48k) are 'not distinguishable at our power', never 'equal'.
8. **Tiny proxies.** 1.77M (confirm) and 10.63M (large) non-embedding parameters. Nothing here is evidence about 1B+ models or continued pretraining (PLAN §8).

## 12. What can be claimed now (PLAN §8, AMENDMENT_1.md §5)

**Pre-registered pick, `R2-A10-MinGram-P1r3-D2-48k`:**

- *"Chosen by the pre-registered rule; selected after an exploratory second round; confirmed on a sealed test split used once: it improves on the standard BPE recipe `A1-P1r3-D2-16k` by Δ = -1.32 % bits-per-byte (95 % CI [-1.47, -1.16], p <1e-4; hierarchical cluster bootstrap over 27 test clusters, 5 seeds; dev -1.58 % [-1.87, -1.36])."*
- It has the lowest mean test bpb of the 6 tokenizers scored on test at the confirm scale, but it is not distinguishable from MinGram-32k (-0.06 % [-0.29, +0.15], TOST-equivalent within ±0.3 %) or from SP-32k (+0.31 % [-0.02, +0.63]). "Lowest held-out bits-per-byte among the 17 compared" is not claimable: only 6 were scored on test.

**Released default, `R2-A4-SPnat-D2-32k`:**

- *"In the statistically tied top set at the confirm scale; chosen by the amended rule (lowest large-arbiter dev bpb in that set: -0.62 % [-1.04, -0.31] vs MinGram-32k and -0.90 % [-1.35, -0.56] vs MinGram-48k, 2 seeds). On the strict test split: -1.01 % [-1.33, -0.72] bits-per-byte vs the standard BPE recipe at the confirm scale (p <1e-4, 5 seeds; dev -1.62 % [-2.11, -1.24]); within the top set its confirm-scale test bpb is the highest of the three, not significantly (+0.31 % [-0.02, +0.63] vs MinGram-48k, +0.24 % [-0.13, +0.60] vs MinGram-32k). Report-only large arbiter on test: -0.77 % [-1.15, -0.40] vs the baseline, -0.51 % [-0.86, -0.22] vs MinGram-48k (2 seeds); vs MinGram-32k not established (1 seed)."*

**Supported in general (test, confirm scale):** Unigram-family vocabularies beat byte-level BPE at equal vocabulary (§7); the round-2 top set beats the round-1 winner (§6); every tokenizer tested beats the 16k baseline.

**Not claimable:**

- That SP-32k is 'best on test', or 'lowest held-out bpb' on test: at the pre-registered confirm scale it is 3rd of the 3 top-set members, and the large run that puts it first is report-only, with 1–2 seeds.
- 'SOTA for all AI models' or 'best possible' (PLAN §8): 18 LM candidates trained (17 ranked), on proxies of at most 29.6M parameters.
- That MinGram-48k, MinGram-32k and SP-32k are 'equal'. Only MinGram-48k vs MinGram-32k is TOST-equivalent at the confirm scale on test; the others are 'not distinguishable at our power'.
- Transfer to 1B+ models or to continued pretraining; downstream task quality.

## 13. Files and reproduction

| file | content |
|---|---|
| `lm/collect_final_test.py` | copy + validation of both final-test bundles (deterministic; `--no-copy` re-validates) |
| `lm/colab_results/final_test/<bundle>/` | byte-exact copies of `G:\My Drive\hindko_lm_out_final_test\<bundle>\` (incl. `large` records) |
| `lm/colab_results/final_test/results_table_final_test.json` / `.csv` | one row per bundle × stage × candidate × seed (42 rows) |
| `lm/colab_results/final_test/validation_final_test.json` | every check per record, stage checks, same-model checks, cross-bundle pairs, copy manifest |
| `lm/colab_results/final_test/test_clusters.csv` | the 491 test documents: doc_index, uid, bytes, source, group, cluster |
| `analysis/test_analysis.py` | every computation; imports `decide.py` (bootstrap, loaders, dev clusters) and `amendment1_stats.joint_reps`, unchanged |
| `analysis/test_results.json` | all numbers, per-seed bpb, per-source, leave-one-cluster-out, sensitivities, large run, and ready strings for the card's test placeholders (`card_placeholders`) |
| `analysis/test_report.py` | renders this file and asserts each qualitative statement |

Reproduce from `F:\Hindko\_tokenizer` (single process, about a minute including the copy): `PYTHONIOENCODING=utf-8 python lm/collect_final_test.py && PYTHONIOENCODING=utf-8 python analysis/test_analysis.py && PYTHONIOENCODING=utf-8 python analysis/test_report.py`.
