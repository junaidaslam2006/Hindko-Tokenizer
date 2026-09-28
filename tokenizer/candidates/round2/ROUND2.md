# Round 2: tokenizers that combine the Stage 3 findings

Built 2026-09-26 in `F:\Hindko\_tokenizer\candidates\round2\`. This is an **exploratory refinement** under PLAN §2.1 ("LM stage: {8k,16k,32k} plus refinements"). It is outside the pre-registered PLAN §2.2 ranking, so the waves carry ranks 101–106.

**Test discipline.** Nothing here read the test split. Every data file was loaded through `eval/harness.load_docs` or `candidates/superbpe/common.load_view`; both verify the frozen split manifest (`76582d3a…`) and refuse test rows. All numbers are on dev_strict (836 documents, 1,445,513 bytes), plus dev_permissive where a check says so.

## 0. Result at a glance

Stage 3 (dev bpb, lower is better) found three things: 32k beats 16k, D2 beats D1 by about 1%, and MinGram/Unigram beat BPE at equal data. Round 2 therefore rebuilds the Stage 2 candidates on the recipe of the best tokenizer (A1-P1r3-D2-32k), and extends the vocabulary to 48k.

**All six requested tokenizers were built.** None was infeasible; the slowest build (SuperBPE with its cross-check) took 134 s. All six pass G1–G5 after the pre-declared G2 remedy.

| rank | id | tokenizer file (under `round2\`) | sha256 | dev bytes/token | encoder |
|---|---|---|---|---|---|
| 101 | R2-A6-SBPE-P1r3-D2-32k-t080 | `superbpe\sbpe_32k_t080_p1r3_d2\tokenizer.json` | `98a568ee8b3a9bdc3614b5c42772a5301acd84633fbe210f92766b0162901863` | **8.3990** | HF-native, exact (100% of dev) |
| 102 | R2-A10-MinGram-P1r3-D2-32k | `mingram\runs\mingram_P1r3_D2_32768\tokenizer.json` | `12fcd260b8eaee8c514e16ad6ae9b3c912ce1d2fce1447963a56b56cdf409119` | **7.1258** | HF-native, exact (100% of dev) |
| 103 | R2-A4-SPnat-D2-32k | `standard\tok\R2-A4-SPnat-D2-32k\sp.model` | `1496e9a7471b5548730fbff8787d9de9a6c0bc06e0ecc0c1f8d2c676ea512d8f` | **7.0764** | SentencePiece native + newline wrapper |
| 104 | R2-A1-P1r3-D2-48k | `standard\tok\R2-A1-P1r3-D2-48k\tokenizer.json` | `7c4c76092e63b96c2124f6b2f65d3f92d4de8a1391db33a94ab0c17c8441bdbb` | **7.2017** | HF-native |
| 105 | R2-A10-MinGram-P1r3-D2-48k | `mingram\runs\mingram_P1r3_D2_49152\tokenizer.json` | `4dd12e627f770ce734c635f657ff12e15816fdc3ca817b67ba0d3cd4bd2fd3cc` | **7.2237** | HF-native, exact (100% of dev) |
| 106 | R2-A4-SPnat-D2-48k | `standard\tok\R2-A4-SPnat-D2-48k\sp.model` | `a04a54cdad44fa0575cb54b521f92599e365fb01843c5373e0f4759c1ed7000b` | **7.1346** | SentencePiece native + newline wrapper |

- **Wave order** (the rank, and the cut order from the bottom): the 32k combinations first, then the 48k probes. Among the 32k builds, SuperBPE comes first because its P1-D1 version was the second-best tokenizer in Stage 3 (1.4053).
- **`round2_waves.json`** holds the six entries in lm/WAVES.json's format. `colab/build_bundle.load_waves` parses it: 6 candidates in this order, kinds hf/sentencepiece, no custom encoder, so no `pythonpath_required`.
- **It has no top-level `baseline` key.** The Stage 3 baseline is not a round-2 wave, and build_bundle.py stops if a named baseline is missing from the list. On its own the file would fall back to "first candidate" as baseline. To compare against Stage 3 in one bundle, merge these entries with lm/WAVES.json's `A1-P1r3-D2-16k` and `A1-P1r3-D2-32k` entries, or pass `--baseline` with an id that is in the list.

## 1. What was built, and how the reviewed code was reused

**Recipe for every build:**
- training text D2 (`train_D2`; SentencePiece: `train_D2.lines.txt`);
- pre-tokenizer P1r3 (`eval/harness.PRETOKENIZERS['P1r3']`); A4 uses SentencePiece's native pre-tokenization with `split_digits=true`, exactly as in the Stage 1 sweep;
- the 64-token special block at ids 0–63;
- no normalizer;
- total vocabulary exactly 32,768 or 49,152 (checked on every file, and again when build_bundle's loader reads it).

The algorithms are the reviewed implementations. The `r2_*.py` drivers only redirect inputs and output paths:

| family | reused unchanged | what round 2 changes |
|---|---|---|
| A1, A4 (`r2_standard.py`) | `sweep_lib` trainers, SentencePiece→HF export and its equivalence test, G2 remedy; `run_sweep.process` (double training for G4, harness `--gates all`, G1 on dev_permissive, remedy, R1 on own mix) | `sweep_lib.TOKDIR/RESDIR` → `round2\standard`. `run_sweep.train_config`'s SentencePiece branch hard-codes `train_D1`. `train_config_r2` is that branch verbatim with the view taken from the config (`train_D2`). The A1 branch already reads the data mix and is used as is. |
| A10 MinGram (`r2_mingram.py`) | `train_mingram.py`, `equiv_test.py`, `mingram.py` (run through `runpy`) | one patch: `mingram.P1` := P1r3, and the compiled Python twin is reset. It drives the seed BPE, the pretoken types for EM and prune, the HF export's Split and the reference encoder. `build_model` is wrapped only to label the model's pretokenizer id `P1r3`. Arguments: `--data train_D2 --total 32768/49152 --no-cache`. |
| A6 SuperBPE (`r2_superbpe.py`) | `build_stage1.train_a1`; `build_stage2.build_chunks / hf_train_pua / py_train / assemble`; `ref_encoder.SuperBPERef`; `verify.check_docs / superword_stats`; `gates_sbpe.main` | `common.P1` := P1r3. `common.S2` := S2r3, derived from the reviewed S2 by the digit substitution that turns P1 into P1r3 (asserted in code). The drivers of `build_stage2.main` and `verify.verify` hard-code train_D1, so their few driver lines are re-stated with the view as a parameter. `gates_sbpe.main` runs with its `train_D1` redirected to `train_D2`. |

**S2r3** (stage-2 pre-tokenizer, Oniguruma):
` ?[\p{L}\p{M}\x{200C}\x{200D}]+(?: [\p{L}\p{M}\x{200C}\x{200D}]+)*| ?\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+`

- It is the Stage 2 S2 with P1r3's digit rule.
- Superwords still end at punctuation, digits and line breaks.
- Digits form right-to-left groups of 3 in both stages, which is what Liu et al. do.

**Pre-tokenizer differential** (PLAN §1.3; `pretok_differential_round2.json`). The Oniguruma and Python-`regex` versions give 0 differing documents:

| regex | dev_strict | dev_permissive | ZWNJ-perturbed dev_strict | all of train_D2 |
|---|---|---|---|---|
| S2r3 | 0/836 | 0/1,358 | 0/836 | 0/11,066 (762,833 chunks) |
| P1r3 | 0/836 | 0/1,358 | 0/836 | 0/11,066 (3,515,913 pretokens) |

## 2. Intrinsic results (dev_strict)

*vs baseline* = bytes/token relative to A1-P1r3-D2-16k (6.7883). *vs refined* = relative to the Stage 1/2 tokenizer that the build refines. NSL = tokens ÷ tokens of A1-P1-D1-16k. The *train_D1 bytes/token* column is what build_bundle uses for the byte-matched context (ctx_tokens ≈ 1,536 / it).

| rank | id | V (effective) | dev bytes/token | vs baseline | vs refined | NSL | train_D1 bytes/token | book / newspaper / web | fertility | STRR |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | R2-A6-SBPE-P1r3-D2-32k-t080 | 32,768 | 8.3990 | +23.73% | A6-SBPE-P1-D1-32k-t080 +2.55% | 0.7991 | 6.6868 | 7.833 / 9.739 / 6.538 | 1.156 | 0.876 |
| 102 | R2-A10-MinGram-P1r3-D2-32k | 32,768 | 7.1258 | +4.97% | A10-MinGram-P1-D1-16k +5.25% | 0.9419 | 6.1513 | 6.872 / 7.673 / 6.144 | 1.138 | 0.887 |
| 103 | R2-A4-SPnat-D2-32k | 32,768 (32,763) | 7.0764 | +4.25% | A4-SPnat-D1-32k **−0.86%** | 0.9485 | 6.1612 | 6.858 / 7.531 / 6.310 | 1.141 | 0.886 |
| 104 | R2-A1-P1r3-D2-48k | 49,152 | 7.2017 | +6.09% | A1-P1r3-D1-48k +0.14% | 0.9320 | 6.2634 | 6.955 / 7.738 / 6.177 | 1.126 | 0.896 |
| 105 | R2-A10-MinGram-P1r3-D2-48k | 49,152 | 7.2237 | +6.41% | A10-MinGram-P1-D1-16k +6.70% | 0.9291 | 6.2875 | 6.978 / 7.756 / 6.210 | 1.122 | 0.898 |
| 106 | R2-A4-SPnat-D2-48k | 49,152 (49,114) | 7.1346 | +5.10% | A4-SPnat-D1-48k **−1.05%** | 0.9407 | 6.2797 | 6.923 / 7.571 / 6.416 | 1.132 | 0.891 |

**References** (existing files, re-read from their harness summaries):

| reference | dev bytes/token | Stage 3 mean dev bpb | train_D1 bytes/token |
|---|---|---|---|
| A1-P1r3-D2-16k (baseline) | 6.7883 | 1.4225 | 5.7984 |
| A1-P1r3-D2-32k (Stage 3 best) | 7.0949 | 1.4004 | 6.1155 |
| A6-SBPE-P1-D1-32k-t080 | 8.1900 | 1.4053 | 6.9885 |
| A10-MinGram-P1-D1-16k | 6.7702 | 1.4187 | 6.0067 |
| A4-SPnat-D1-16k | 6.8668 | 1.4213 | 6.2235 |
| A4-SPnat-D1-32k / -48k (Stage 1 grid) | 7.1378 / 7.2101 | – | 6.5258 / 6.6527 |
| A1-P1r3-D1-48k (Stage 1 grid) | 7.1919 | – | 6.4267 |

**Observations.** These are intrinsic only. Bytes/token screens candidates; the LM bpb decides (PLAN §4.2).

- **Against the Stage 3 best (A1-P1r3-D2-32k, 7.0949):**
  - SuperBPE-32k needs 15.5% fewer tokens (+18.38% bytes/token);
  - MinGram-32k: +0.44%;
  - A4-D2-32k: −0.26%;
  - A1-48k: +1.51%, while MinGram-48k is +0.31% and A4-D2-48k −0.93% relative to A1-48k itself.
- **D2 does not help SentencePiece Unigram in bytes/token.**
  - A4 on D2 is 0.86% (32k) and 1.05% (48k) *worse* than A4 on D1 at the same size, on book, newspaper and web alike.
  - For A1 the D2 gain shrinks with size: +0.95% at 16k, +0.49% at 32k, +0.14% at 48k.
  - A plausible reason: D2 has 58% of D1's bytes, and at 32k–48k the data-scarcity effect outweighs the in-domain gain. That explanation was not tested.
  - Stage 3's "D2 beats D1" was measured for A1 BPE at 16k in LM bpb. It does not transfer automatically to Unigram at 32k and above. The A4 waves are the place to test it.
- **SuperBPE on P1r3 + D2 is more compressive than the P1-D1 build** (8.399 vs 8.190 on dev).
  - On *train_D1*, the LM's training stream, it is less compressive (6.687 vs 6.989), because D2 excludes the permissive-tier text.
  - Its byte-matched context is therefore about 230 tokens against 220 for the Stage 3 build.
  - Superwords make up 17.0% of dev tokens and 31.1% of dev bytes (P1-D1-32k: 15.7% / 28.8%). The most frequent are ` دے نال`, ` اس دے`, ` اس دی`, ` اے کہ`, ` اُناں نے`.
- **Robustness (report only).**
  - Removing harakat re-segments 27.4% of affected words under SuperBPE, against 6–10% for the others; this was already seen in Stage 2.
  - Toggling the space before ۔/، re-segments 57–59% of affected words under A4 and 2.8% under MinGram.
  - Full tables are in each `summary.json`.

## 3. Gates and checks

**Hard gates** (harness `eval/harness.py`, `--gates all`, dev_strict; G1 also on dev_permissive):

| id | G1 strict / permissive | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|
| R2-A6-SBPE-P1r3-D2-32k-t080 | 836/836 / 1,358/1,358 | pass: 25,880 tested, 0 fail (14 partial-UTF-8 exempt); multi-word part 6,554, 0 unreachable | pass | pass: full-pipeline retrain gives a byte-identical tokenizer.json and stage-1 file. `stage2_pua.json` differs only in its stored stage-1 path (`stage1\` vs `g4\stage1\`); merges, t, T, route and stage-1 sha256 are identical | n/a (byte-level) |
| R2-A10-MinGram-P1r3-D2-32k | 836/836 / 1,358/1,358 | pass | pass | pass: tokenizer, model, seed and vocab.tsv all byte-identical | pass |
| R2-A4-SPnat-D2-32k | 836/836 / 1,358/1,358 | pass after the remedy: 5 pieces deleted, dev re-encoding identical | pass | pass (sp.model byte-identical) | pass |
| R2-A1-P1r3-D2-48k | 836/836 / 1,358/1,358 | pass (0 failures) | pass | pass (byte-identical) | n/a |
| R2-A10-MinGram-P1r3-D2-48k | 836/836 / 1,358/1,358 | pass | pass | pass: all four outputs byte-identical | pass |
| R2-A4-SPnat-D2-48k | 836/836 / 1,358/1,358 | pass after the remedy: 38 pieces deleted, dev re-encoding identical | pass | pass | pass |

**Encoder equivalence, MinGram** (`equiv_test.py` unchanged; `runs\…\equivalence.json`). Stock HF Unigram ids equal the pure-Python reference Viterbi (`MinGramRef`, HF tie rule) on every document of every set:

| set | 32k | 48k |
|---|---|---|
| dev_strict (100%) | 836/836 | 836/836 |
| dev_permissive (100%) | 1,358/1,358 | 1,358/1,358 |
| dev_strict with harakat removed / digit script swapped / punct-space toggled / ZWNJ inserted | 836/836 each | 836/836 each |
| synthetic stress | 4,289/4,289 | 4,289/4,289 |
| min-token check (dev_strict pretokens) | 0 failures | 0 failures |

- The paper's exact-tie rule, report only, changes 3 dev_strict / 7 dev_permissive documents at 32k and 4 / 11 at 48k, with identical token counts.
- The 2⁻⁹ score grid versus a 2⁻²⁰ grid: 0 dev pretoken types differ at 32k and 2 at 48k, with identical token counts (report only).
- The count-dominance bound is about 20,300 tokens per pretoken; the longest train_D2 pretoken has 29 characters.

**Encoder equivalence, SuperBPE** (`superbpe\sbpe_32k_t080_p1r3_d2\verify.json`). HF-native ids equal `SuperBPERef` (two-phase: stage-1 merges, then PUA-space stage-2 merges):

| set | HF vs reference | phase 1 vs HF stage 1 | final boundary not a stage-1 boundary | G1 (HF / reference) |
|---|---|---|---|---|
| dev_strict (100%) | 0/836 mismatching | 0 | 0 | 0 / 0 failures |
| dev_permissive (100%) | 0/1,358 | 0 | 0 | 0 / 0 |
| train_D2, every 20th (554) | 0/554 | 0 | 0 | 0 / 0 |

Edge strings (special markers in text, empty string, newlines, digit runs) are equal and round-trip.

**SuperBPE build facts** (`build_info.json`):
- The stage-1 file (t = 26,214) is a strict prefix of `A1-P1r3-D2-32k`: identical ids, and its merges are a prefix.
- Stage-1 merges inside S2r3 chunks reproduce the P1r3 encoding in 11,066/11,066 train documents.
- The pure-Python trainer, run uncapped, reproduces the Rust PUA trainer's merges exactly.
- The 4-word cap binds: 156 uncapped tokens had more than 4 words. The shipped merges are therefore the capped pure-Python ones (route `pure_python_capped`, as in Stage 2).
- Stage-2 tokens: 5,448 with 2 words, 894 with 3, 212 with 4. None duplicates an existing token, and no merge pair repeats.

**SuperBPE multi-word G2** (`g2_multiword.json`). This uses Stage 2's reachability reading (WAVES pending decision 2).
- Isolated self-tokenization fails for 0 of 6,554 tokens, so 0 are unreachable.
- The literal "shortest train chunk" clause fails for 1,329 tokens (all of which pass in isolation); the stage-1-aligned version fails for 945.
- The train frequencies in this file are counted on train_D2, the tokenizer's own training text.

**R1 (support) and R2:**

| id | R1 on train_D1: =0 / <20 (%) / <100 / median | R1 on train_D2 (own): =0 / <20 (%) / median | R2 partial-UTF-8 (train freq 0) |
|---|---|---|---|
| R2-A6-SBPE-P1r3-D2-32k-t080 | 119 / 6,057 (18.67) / 23,897 / 46 | 209 / 11,411 (35.17) / 30 | 14 (5) |
| R2-A10-MinGram-P1r3-D2-32k | 0 / 11,329 (34.92) / 25,148 / 32 | 2 / 18,062 (55.67) / 16 | 0 |
| R2-A4-SPnat-D2-32k | 2 / 13,182 (40.63) / 25,413 / 27 | 4 / 19,321 (59.56) / 14 | 0 |
| R2-A1-P1r3-D2-48k | 386 / 27,273 (55.85) / 41,867 / 16 | 741 / 35,136 (71.95) / 8 | 16 (8) |
| R2-A10-MinGram-P1r3-D2-48k | 1 / 27,246 (55.80) / 41,802 / 16 | 5 / 35,286 (72.26) / 8 | 0 |
| R2-A4-SPnat-D2-48k | 58 / 28,858 (59.14) / 41,959 / 13 | 132 / 36,293 (74.38) / 6 | 0 |

For comparison, A1-P1r3-D2-32k has 11,512 learned tokens (35.48%) below 20 on train_D1.
- **MinGram's "almost no rare tokens" does not hold at these sizes on D2.** At 16k on D1 it had 0 multi-character tokens below 20. Here, multi-character tokens below 20 on train_D1 number 11,284 at 32k and 27,196 at 48k.
- **SuperBPE-32k has the smallest rare tail of the six** (18.7% below 20 on train_D1). As in Stage 2, it spends its last 6,554 ids on frequent superwords.

**HF-native loading.** `transformers.PreTrainedTokenizerFast(tokenizer_file=…)` gives the same ids as `tokenizers` on 836/836 dev_strict documents for the four tokenizer.json candidates, and round-trips all of them (`transformers_check.json`, transformers 5.3.0).

**Bundle loader check.** `r2_report.py` loads every wave with `colab/build_bundle.load_encoder(…, 'harness')`, the encoder the bundle will use. For all six:
- n_vocab is the nominal size;
- `<|endoftext|>` = 0 and `<|bos|>` = 1;
- lm/make_waves.py's synthetic string round-trips;
- the file's sha256 equals the harness summary's tokenizer identity.

## 4. Before running these waves: what to know

1. **A4 is not HF-native exact** (PLAN §7 tie-breaker a).
   - Its `tokenizer.json` export differs from the native ids on 0 dev_strict / 1 dev_permissive documents at 32k and 1 / 1 at 48k, on equal-score ties with identical token counts.
   - The waves therefore use the native `sp.model` with the newline wrapper (`encoder_kind: sentencepiece`), as for Stage 3's A4.
   - After the G2 remedy the deleted slots are CONTROL placeholders `<|unused_k|>`. The effective vocabularies are 32,763 and 49,114, while `vocab_size` stays nominal.
2. **The MinGram prune boundary sits inside large ties.**
   - At 32k the boundary count is 5 and 2,667 tokens share it; at 48k it is 3 with 8,736 tokens.
   - The tie-break order that Stage 2 fixed (previous-pass count, BPE-init count, later merge first) therefore decides a large part of the tail. At 16k on D1 the boundary was 37 with 212 tied tokens.
   - The builds are deterministic (G4), but the tail is not "chosen by log p" in any strong sense.
3. **Every 32k/48k build on D2 has a large rare-token tail.** R1 on train_D1 below 20 ranges from 19% (SuperBPE) to 59% (A4-48k). This is the data-scarce regime of PLAN §4.3, and R3 (embedding norms) after the LM runs is where it would show.
4. **These waves combine two changes at once** (recipe and size) for A6, A10 and A4. They are refinements, not controlled ablations.
   - A clean read of "MinGram vs BPE at 32k" is R2-A10-32k vs A1-P1r3-D2-32k: same data and pre-tokenizer.
   - "Unigram vs BPE" is R2-A4-32k vs A1-P1r3-D2-32k. There the pre-tokenizer differs by design, as in Stage 3.
   - "SuperBPE vs BPE" is R2-A6-32k vs A1-P1r3-D2-32k: same stage 1.
5. **Run cost.** The 48k waves have 1.5× the embedding and softmax rows of 32k. No LM run time was measured in this round.

## 5. Deviations and interpretations

- **t/T for SuperBPE** is 26,214/32,768, the Stage 3 build's values, as requested. PLAN §2.2 rank 8 would take t/T from the better of ranks 2 and 5 in Stage 3. That comparison is not in the task's Stage 3 summary, so it was not applied.
- **MinGram runs used `--no-cache`** (both the official run and the G4 twin), because train_mingram.py's pretoken cache lives in `candidates\mingram\work`, outside round 2.
  - The G4 twin is an independent second process with identical inputs. Stage 2's twin differed from its official run by cache use; here neither run uses the cache.
  - The report-only `--ablation` (N_em sweep on dev) was not run.
- **R1 is reported twice:** on train_D1 (harness, the LM stream in the bundle) and on train_D2 (the tokenizer's own training text; run_sweep does the same for D2 tokenizers). SuperBPE's multi-word G2 reachability witness also uses train_D2.
- **SuperBPE verify** checks every 20th *train_D2* document instead of train_D1.
- **G1 on test** is not run (as in Stages 1–2); it belongs to Stage 5.
- **The G2 remedy for A4** is the pre-declared one (Unigram family: delete, then verify identical dev re-encoding), run by the unchanged `run_sweep.process`. The deleted pieces are all Viterbi-dominated word+punctuation or rare forms, e.g. `▁والے۔`, `▁کہیا۔`, `▁دا۔`, `▁وچ۔`.
- **One code edit after the builds.** `r2_superbpe.py` was edited only in `cmd_eval`, to record the G4 comparison of `stage2_pua.json` without its stored path. The eval was then re-run; the harness pass was cached as up to date. `build_info.json`, `verify.json` and `g2_multiword.json` therefore record the earlier driver hash `ca594cb6…`; the build, verify and g2 code paths did not change. `r2_report.py` was fixed (a dictionary key) before its first successful run.
- **No third-party code** was downloaded; nothing outside `candidates\round2\` was written. Python ran with `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` appeared in the reused directories.

## 6. Files

| path | content |
|---|---|
| `round2_waves.json` | the six LM-wave entries (lm/WAVES.json format; ranks 101–106) |
| `round2_results.json` | every metric, check and file hash per candidate, plus the reference rows |
| `standard\tok\<id>\` | A1/A4: `tokenizer.json` (A1), `sp.model` + `sp.vocab` + HF export `tokenizer.json` (A4), `meta.json`, `g4\`, `pre_remedy\` (A4) |
| `standard\results\dev_strict\<id>\` | `summary.json`, `docs.jsonl` (per document, for the PLAN §6 bootstrap), `stage1.json` |
| `mingram\runs\mingram_P1r3_D2_{32768,49152}[_g4]\` | `tokenizer.json`, `mingram_model.json`, `seed_bpe.json`, `vocab.tsv`, `train_log.json`, `r2_meta.json`, `equivalence.json` |
| `mingram\results\dev_strict\<id>\` | `summary.json`, `docs.jsonl`, `r2_checks.json` |
| `superbpe\stage1\`, `superbpe\sbpe_32k_t080_p1r3_d2\` | stage-1 file (+ `.info.json`); `tokenizer.json`, `stage2_pua.json`, `build_info.json`, `verify.json`, `g2_multiword.json` |
| `superbpe\g4\` | the from-scratch retrain used by G4; `superbpe\work\` holds the chunk caches |
| `superbpe\results\dev_strict\<id>\` | `summary.json`, `docs.jsonl`, `r2_checks.json` |
| `pretok_differential_round2.json`, `transformers_check.json` | regex differentials; HF-native loading check |
| `r2_common.py`, `r2_standard.py`, `r2_mingram.py`, `r2_superbpe.py`, `r2_transformers_check.py`, `r2_report.py`; `logs\` | code and console logs |

**Reproduce** (run from anywhere, with `PYTHONIOENCODING=utf-8` and `PYTHONDONTWRITEBYTECODE=1`):
```
python r2_standard.py R2-A4-SPnat-D2-32k      (likewise R2-A4-SPnat-D2-48k, R2-A1-P1r3-D2-48k)
python r2_mingram.py train --size 32k ; python r2_mingram.py train --size 32k --g4
python r2_mingram.py equiv --size 32k ; python r2_mingram.py eval --size 32k        (likewise 48k)
python r2_superbpe.py build --py-check ; python r2_superbpe.py build --g4
python r2_superbpe.py verify ; python r2_superbpe.py s2diff ; python r2_superbpe.py g2 ; python r2_superbpe.py eval
python r2_transformers_check.py ; python r2_report.py
```
Wall clock on this CPU: every training ran in 11–134 s, and every harness pass took under 1 minute.

## 7. Not done

- **No LM run and no bpb.** Round 2 only builds and screens; the Colab waves decide.
- Nothing ran on the test split.
- No other t/T for SuperBPE, no A4 or MinGram at 16k on D2, and no D1 counterparts of the 48k builds, except where the Stage 1 grid already has them.
- No R3 (it needs trained embeddings).
- Morphology (SILVER sets) is in each `summary.json` but is not discussed here; it is report-only.
