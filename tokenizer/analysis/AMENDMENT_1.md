# Amendment 1 to the decision (written 2026-09-27 before any LM number on the test split existed)

## Status at the time of writing

- **Test split.** It has been opened only to build the two final-test bundles and to compute intrinsic metrics:
  - `f54c929ba1ab`, with 4 candidates;
  - `94175d26497f`, with 3 candidates including the baseline.

  No language model has been trained or scored on it. `FINAL_TEST_LOG.json` lists both builds.
- **Pre-registered decision** (`analysis/DECISION.md`, `analysis/decision.json`, PLAN §7), made on the Stage 4 'confirm' dev results:
  - Top set, statistically tied: {R2-A4-SPnat-D2-32k (best mean, 1.22584), R2-A10-MinGram-P1r3-D2-48k (1.22636), R2-A10-MinGram-P1r3-D2-32k (1.22844)}.
  - Tie-breaker (a), HF-native exact encoding, removed the SentencePiece candidate.
  - Tie-breaker (b), higher bytes/token, chose **R2-A10-MinGram-P1r3-D2-48k**.
- **Adversarial audit of that decision.** It found that tie-breaker (b) rises mechanically with vocabulary size. Within a tied set, (b) therefore picks the largest vocabulary, and the 48k model has 2.27x the total parameters of the 16k baseline, so the confirm-scale gain is partly a parameter effect.
- **'Large' arbiter on dev.** d=384, L=6, H=6, 2 epochs of the full train split, LR 5e-4 from its own sweep on the baseline, 2 seeds. Report-only, added beyond PLAN §8(ii), run 19:44-20:53 UTC on 2026-09-26 before this amendment:

  | candidate | large dev bpb | Δ vs baseline |
  |---|---|---|
  | R2-A4-SPnat-D2-32k | 1.13373 | -1.336% |
  | R2-A10-MinGram-P1r3-D2-32k | 1.14086 | -0.715% |
  | R2-A10-MinGram-P1r3-D2-48k | 1.14407 | -0.436% |
  | A1-P1r3-D2-16k (baseline) | 1.14908 | 0 |
  | A1-P1r3-D2-32k | 1.14940 | +0.027% |

## Amendment

1. **The pre-registered choice stays on record.** The pre-registered choice, R2-A10-MinGram-P1r3-D2-48k, and its claim are reported unchanged, together with their one-shot test result.
2. **The released default tokenizer is chosen by an amended rule, fixed now.** Within the pre-registered, statistically tied top set, prefer the candidate with the lowest *large-arbiter* dev bpb. This replaces tie-breaker (b), which is confounded with vocabulary size and parameter count. The large arbiter is closer to the scale at which the tokenizer will actually be used.
   - Under this rule, **R2-A4-SPnat-D2-32k** is chosen. It is also the best-mean candidate at the confirm scale.
3. **Tie-breaker (a) becomes an engineering requirement.** Being "HF-native exact" becomes a packaging requirement, no longer a reason to drop a candidate. The released tokenizer must ship:
   - the SentencePiece model as the canonical encoder, used with the newline convention of PLAN §1.1;
   - a Hugging Face `tokenizer.json` whose encodings equal the canonical encoder on 100% of dev and test documents. Viterbi ties may be broken by a deterministic score perturbation that changes no encoding outside tie cases; this must be verified.

   If exact equivalence cannot be achieved, the released default falls back to the best HF-exact member of the top set at the large scale, **R2-A10-MinGram-P1r3-D2-32k**, and this document records why.
4. **The test run is unchanged.** It still evaluates, at the confirm scale, 5 seeds, LR 1e-3, the fixed candidate lists of both final-test bundles:
   - the baseline;
   - both choices (pre-registered and amended);
   - the other top-set member;
   - the confirmatory round-1 winner, A1-P1r3-D2-32k;
   - the equal-vocabulary BPE control, R2-A1-P1r3-D2-48k.

   A report-only large-arbiter test run of the top set plus the baseline may be added (fixed LR 5e-4, 2 seeds). Neither run can change which tokenizer is released: this amendment fixes that now.
5. **Claims for the released tokenizer are worded as follows:**
   - it is in the statistically tied top set at the confirm scale;
   - it was chosen by the amended rule, with the large-arbiter evidence stated;
   - its test-split result is reported whatever it is.
