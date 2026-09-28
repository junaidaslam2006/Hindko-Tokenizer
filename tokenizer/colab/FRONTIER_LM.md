# Released Hindko tokenizer vs frontier tokenizers: the same small LM, the same Hindko bytes (report-only, test split)

This page describes one Colab GPU run, prepared but **not yet run**. The bundle in `colab/build_frontier_test/upload/`
is bundle **`527e1023adb5`** (manifest sha256 `527e1023adb54743c57d705904ed8f70363699c27373b829bfebf33e1b8eaf5a`,
kind **`final_test_report`**). It was built once by `colab/build_bundle.py --final-test --report-only` and recorded in
`F:\Hindko\_tokenizer\FINAL_TEST_LOG.json` as **entry #2** (status `completed`, `"report_only": true`).

> **In this bundle, "dev" means TEST.** The code keeps its names, so these are all on the **strict test split**:
> the arrays `*/dev_tokens`, `shared/dev_docs.json`, `rec["dev"]`, every `dev bpb` log line and every `mean bpb`
> in `summary.txt` / `summary.json`. It is the test split of the one-shot final test (491 documents, 1,451,026
> UTF-8 bytes, 27 clusters: 15 newspaper, 6 book, 6 web), with identical documents, order, bytes, 0.5 MB subset
> and parity slice (checked, §6).

## 0. What this measures, and what it does not

- **The question.** If you train the *same* small language model on the *same* Hindko text, and change only the
  tokenizer, which tokenizer lets the model predict held-out Hindko best? The score is **bits per UTF-8 byte (bpb)**
  of the test text. bpb does not depend on how a tokenizer cuts the text, so every tokenizer is on one scale; lower
  is better.
- **The model** is the study's Stage 4 'confirm' arbiter, unchanged (`hk_lm.py` sha256 `3f78ad79…`, the file that
  trained every published test run):
  - GPT with d=192, L=4, H=4, tied input/output embeddings, and 1.77M non-embedding parameters;
  - trained from scratch on the full permissive train split: 45,007,515 bytes, 16,015 documents, 1 epoch = 3,663
    steps of 8 × ~1,536 bytes;
  - AdamW with peak LR **1e-3** (the dev choice), 50 warm-up steps and cosine decay to 10 %;
  - **seeds 1, 2, 3**; fp16 training and fp32 evaluation on a T4;
  - every one of the 491 test documents scored with windows of ctx tokens (stride ctx/2).
- **Byte-matched.** Every tokenizer gets the same documents in the same order, the same number of optimiser steps,
  the same bytes per step, and the same ~1,536 bytes of context per sequence (ctx in tokens = 1,536 ÷ its train
  bytes/token).
- **It is not a comparison with GPT-4o, Gemma, Llama, Qwen or DeepSeek *as models*.** Only their **tokenizers** are
  used, inside a 1.8M-parameter LM trained here. Nothing here says how well those models handle Hindko.
- **Report-only.** The released tokenizer is fixed (`analysis/AMENDMENT_1.md`). No number from this run selects,
  tunes or changes anything. It is a second, logged use of the test split, after the one-shot final test
  (`analysis/TEST_RESULTS.md`).

## 1. The tokenizers (run order = priority)

| # | id | what | vocab (model rows) | ids used on train | train b/tok | **test b/tok** | ctx | token arrays | tokenizer sha256 |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `hindko-1.0.0` | **the released tokenizer** (`F:\Hindko\tokenizer\tokenizer.json`, R2-A4-SPnat-D2-32k); baseline of this bundle | 32,768 | 32,480 (99.1 %) | 6.161 | **7.277** | 249 | uint16 | `49f301c52363…` |
| 2 | `R2-A4-SPnat-D2-32k` | reproduction anchor: the candidate `sp.model` scored in the one-shot test | 32,768 | 32,480 (99.1 %) | 6.161 | 7.277 | 249 | uint16 | `1496e9a7471b…` |
| 3 | `gpt-4o` | OpenAI o200k_base (GPT-4o; same encodings as gpt-oss, phi-4-mini) | 200,020 | 12,113 (6.1 %) | 4.112 | 4.338 | 374 | uint32 | `43a3ad4618a6…` |
| 4 | `gemma-3` | Google Gemma 3 SentencePiece 262k (same encodings as Gemma 4) | 262,145 | 12,486 (4.8 %) | 4.314 | 4.610 | 356 | uint32 | `4667f2089529…` |
| 5 | `llama-3` | Meta Llama 3.1 tiktoken-BPE 128k (also Alif-1.0, Qalb-1.0) | 128,256 | 9,955 (7.8 %) | 2.622 | 2.641 | 586 | uint32 | `79e3e522635f…` |
| 6 | `qwen-3.5` | Alibaba Qwen 3.5 byte-level BPE 248k (same encodings as Qwen 3.8) | 248,071 | 10,851 (4.4 %) | 3.481 | 3.598 | 441 | uint32 | `5f9e4d4901a9…` |
| 7 | `deepseek-v3` | DeepSeek-V3 byte-level BPE 128k (same encodings as R1, V4, V4.1) | 128,815 | 9,467 (7.3 %) | 3.249 | 3.372 | 473 | uint32 | `621ac2e32d0d…` |
| 8 | `llama-4` | Meta Llama 4 BPE 202k | 201,135 | 10,660 (5.3 %) | 3.641 | 3.727 | 422 | uint32 | `172c9eb4beaf…` |
| 9 | `roberta-urdu` | UrduHack RoBERTa-Urdu byte-level BPE 52k (best lossless external tokenizer on test by bytes/token) | 52,000 | 38,973 (74.9 %) | 5.012 | 5.709 | 306 | uint16 | `ad848f502226…` (directory) |
| 10 | `bloom` | BigScience BLOOM byte-level BPE 250k | 250,680 | 18,174 (7.2 %) | 4.796 | 5.122 | 320 | uint32 | `3fa39cd4b150…` |

Notes on the table:
- **External tokenizers** are loaded by name through `baselines/load_baselines.py`, the same objects as the intrinsic
  tables (`eval/TEST_COMPETITORS.md`). Their test token counts equal that table's exactly (§6, V6).
- **Special tokens.** Each tokenizer's own document-boundary tokens are used where they exist (`<|endoftext|>`,
  `<eos>`/`<bos>`, `<|begin_of_text|>`/`<|end_of_text|>`, DeepSeek's sentence tokens, `<s>`/`</s>`). GPT-4o and
  Qwen 3.5 have no BOS token, so one row is appended for it (vocab 200,019 → 200,020; 248,070 → 248,071).
- **The anchor and the release are the same tokenizer.** They have the same 32,768 pieces and ids, and identical
  test encodings. Their train encodings differ in exactly **29 of 16,015 documents**, all exact Viterbi ties
  (`release_build/sp32k/EQUIVALENCE.md`; re-checked here, §6 V4).
  - The anchor's train and test arrays are **byte-identical** to the one-shot test bundle `f54c929ba1ab`, so its
    seeds 1–3 must reproduce the published test runs (§5 b).

## 2. The run

1. **Runtime → Change runtime type → GPU: a T4**, as in every earlier run (the precision pin is `cuda / fp16`).
2. Upload the 9 `hk_bundle.tar.xz.part000…008` files, `PARTS.json` and `run_all.py` from
   `colab/build_frontier_test/upload/` into **one** folder:
   - 82.8 MB in total; every part is ≤ 9.5 MB;
   - `/content/upload` works, or a Drive folder `MyDrive/hindko_lm_upload`;
   - mount Drive (cell 2, `USE_DRIVE = True`) so that results survive a disconnect.
3. Open **`hindko_lm_arbiter_FRONTIER_TEST.ipynb`** from the upload folder. Run cells 1–3; cell 3 must show bundle
   `527e1023adb5` and the 10 tokenizers.
4. Cell 4 runs `--stages parity`, the CPU↔GPU sanity check (100 steps, seed 1, baseline `hindko-1.0.0`, on a ~100 KB
   test slice). The CPU reference in `PARITY.json` is 1.617155 bpb, with a 1 % tolerance.
5. **Cell 5** is already set:

   ```python
   import os
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
   rc = run_all('--stages', 'parity,confirm', '--lr', '1e-3', '--no-zip')
   ```

The same run from a shell (the exact command):

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_all.py --parts <folder> --stages parity,confirm --lr 1e-3 --no-zip
```

- **Refusals.** `run_all.py` refuses, before anything runs, every other use of this bundle: the default stages, `lr`,
  `screen`, `large`, `--smoke`, `--final-test-large`, and `confirm` without `--lr`.
- **Seeds.** Seeds 1–3 are fixed now. The PLAN §6 power check is written to `power_check.json`, but on this bundle it
  never adds seeds. `--force-extra-seeds` would add seeds 4 and 5 (+67 % time); decide that before the run, not after
  seeing numbers.
- **Resume.** A disconnect loses at most the run in progress. Re-run the same cell or command: finished runs are
  skipped after their identity is checked. Never delete a result file to "re-run" a tokenizer.
- **Shorter run.** Candidates run one after another in the table's order, so stopping early drops the least
  important ones first. To leave some out on purpose, use `--only` (the baseline `hindko-1.0.0` is always kept):
  - without BLOOM, RoBERTa-Urdu and Llama 4 (≈ 3.7 h):
    `--only R2-A4-SPnat-D2-32k,gpt-4o,gemma-3,llama-3,qwen-3.5,deepseek-v3`
  - `--max-hours 11` stops cleanly before a session limit.
- **Not on a T4?** Add `--amp fp16` to keep the precision of every earlier run (an L4 or A100 would otherwise pick
  bf16). The anchor's bitwise reproduction (§5 b) is expected only on a T4.

## 3. Time and memory on a T4 (estimate)

`colab/frontier_estimate.py` (→ `build_frontier_test/FRONTIER_ESTIMATE.json`) fits a time model on **110 measured T4
'confirm' runs** of this study (same `hk_lm.py`, same 3,663 steps):
- train ms/step = −0.83 + 0.298 × (rows × V)/10⁶ + 3.67 × rows/10³, with rows = 8 × ctx (max residual 1.3 ms);
- eval s = 1.46 + 0.401 × (windows × ctx × V)/10⁹ (max residual 0.2 s).

The calibrated vocabularies are 8k–49k, so **the 128k–262k rows are an extrapolation**. The output layer is
bandwidth-bound and linear in rows × V, but a pessimistic ×1.5 is shown next to each estimate.

| # | tokenizer | V | rows/step | min per run | 3 seeds | ×1.5 | peak GPU train / eval |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `hindko-1.0.0` | 32,768 | 1,992 | 1.7 | 5 min | 8 min | 1.6 / 1.7 GB |
| 2 | `R2-A4-SPnat-D2-32k` | 32,768 | 1,992 | 1.7 | 5 min | 8 min | 1.6 / 1.7 GB |
| 3 | `gpt-4o` | 200,020 | 2,992 | 12.4 | 37 min | 56 min | 9.6 / 7.7 GB |
| 4 | `gemma-3` | 262,145 | 2,848 | 15.3 | 46 min | 69 min | 11.8 / 9.9 GB |
| 5 | `llama-3` | 128,256 | 4,688 | 12.9 | 39 min | 58 min | 9.4 / 5.1 GB |
| 6 | `qwen-3.5` | 248,071 | 3,528 | 18.0 | 54 min | 81 min | 13.6 / 9.4 GB |
| 7 | `deepseek-v3` | 128,815 | 3,784 | 10.4 | 31 min | 47 min | 7.8 / 5.1 GB |
| 8 | `llama-4` | 201,135 | 3,376 | 14.1 | 42 min | 64 min | 10.7 / 7.7 GB |
| 9 | `roberta-urdu` | 52,000 | 2,448 | 3.1 | 9 min | 14 min | 2.5 / 2.4 GB |
| 10 | `bloom` | 250,680 | 2,560 | 13.2 | 40 min | 59 min | 10.3 / 9.5 GB |
| | **total** (30 runs + parity + unpacking) | | | | **≈ 5.2 h** | **≈ 7.8 h** | |

**Every 200k+ vocabulary adds a large softmax.**
- Per step, the output layer handles rows × V logits: 65M for the release and 747M–875M for Gemma 3 and Qwen 3.5,
  11–13× more.
- That is why one release run takes 1.7 min and one Qwen 3.5 run about 18 min, although the model body is identical.

**Memory.**
- Training peaks at about 14 bytes per logit: the fp16 logits, and the fp32 log-softmax, its gradient and the input
  gradient, all alive in the cross-entropy backward. Add 16 bytes per embedding parameter.
- Evaluation uses 4,096-row fp32 chunks.
- Everything fits a T4 (15 GB, about 15.8 GB visible to PyTorch). **Qwen 3.5 is the tightest, at about 13.6 GB.**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation. It changes no arithmetic.
- **If a tokenizer runs out of memory:**
  - the run stops with a traceback, and finished runs are kept;
  - resume with `--only` without that tokenizer, or run it on an L4 or A100 with `--amp fp16`;
  - write down which GPU ran which tokenizer.

**Space.**
- The upload is 82.8 MB; the extracted bundle (410 MB) goes to Colab's local disk.
- The results are about 5 MB.
- `G:` (Drive) showed 0.77 GB free on 2026-09-27, so there is room. No tokenizer had to be dropped for space.

**Recommendation: run all 10.** About 5.2 h (up to 7.8 h) fits one long Colab session, or two with resume.
- If the session budget is shorter, the order already puts the three least important tokenizers last, in drop
  order: BLOOM first, then RoBERTa-Urdu, then Llama 4.
- The `--only` line in §2 gives the ≈ 3.7 h core run.

## 4. Fairness: what this comparison favours, plainly

- **A larger vocabulary means more embedding parameters, and d=192 is tiny.**
  - The transformer body is identical for every tokenizer: 1.77M parameters.
  - The tied embedding is V × 192: 6.3M for the release, but 24.6M (Llama 3) to 50.3M (Gemma 3) for the frontier
    tokenizers. That is 93–97 % of their parameters.
  - Most of those rows are **never used on Hindko**: GPT-4o uses 12,113 of 200,020 ids on the train split, and Gemma
    3 uses 12,486 of 262,145 (table §1). The release uses 99.1 % of its ids.
  - So the 128k–262k tokenizers get **3.3–6.4× the release's total parameters (8.1M), but mostly dead ones**.
    RoBERTa-Urdu gets 1.5×. The model must still learn to put near-zero probability on 118k–250k unused outputs,
    through a rank-192 softmax. That is a real, if probably small, cost to them.
  - We do not separate these two effects here. A vocabulary-trimmed control would be a separate, later experiment.
- **Tokens per byte.** Byte-matching gives every tokenizer the same text, steps and bytes of context. But a tokenizer
  with fewer bytes per token (Llama 3: 2.62 on train) runs the model at ~2.4× more positions per byte of text than
  the release (6.16).
  - That means more computation per byte, and more but easier predictions.
  - This can help or hurt at this size. It is part of what a tokenizer *is*, and it is not corrected for.
- **LR.** 1e-3 was chosen on dev for the 16k BPE baseline of the study, not for the release and not for any external
  tokenizer. Nothing is tuned per tokenizer, because no tuning may happen on test. A tokenizer that would prefer a
  different LR is at some disadvantage; that includes the release.
- **Domain.** The release was trained on Hindko train text (strict split D2); the external tokenizers were trained on
  large multilingual corpora.
  - The test split is group-disjoint from train, so this is a held-out comparison.
  - It measures exactly what a language-specific tokenizer is for: fit to the language. It does **not** show that the
    release's *algorithm* beats BPE in general. The study's equal-vocabulary controls (TEST_RESULTS §7) speak to that
    at 32k/48k.
- **Scale.**
  - One small arbiter size is used, and 3 seeds (the one-shot test used 5).
  - The pooled seed s.d. of this recipe on test was 0.14 % of bpb. With 5 seeds and 27 clusters, the one-shot test
    could not distinguish gaps below about 0.46 %; with 3 seeds, small gaps are resolved less well still. A gap that
    is not significant is "not distinguishable at our power", never "equal".
  - Nothing here transfers automatically to 1B+ models, to continued pretraining of GPT-4o / Gemma / Llama, or to
    downstream tasks (PLAN §8).

## 5. Checks after the run (before reading any comparison)

- **(a) Completeness.**
  - 30 records `results/confirm__<id>__lr1e-3__s{1,2,3}.json`, each with `status == "ok"` and 491 documents.
  - Their `uids` equal the uids of `data/test_strict.jsonl` in uid order.
  - Nothing diverged, and nothing ran out of memory.
- **(b) Reproduction.** On a T4 in fp16, `R2-A4-SPnat-D2-32k` seeds 1–3 must equal the published one-shot test runs
  **bitwise**: the same `train_loss` and the same per-document `bits`.
  - The published runs are `lm/colab_results/final_test/f54c929ba1ab/results/confirm__R2-A4-SPnat-D2-32k__lr1e-3__s{1,2,3}.json`.
  - Their bpb: s1 1.2129939093, s2 1.2090638021, s3 1.2080341771.
  - The same GPU, code, arrays, seeds and LR gave bitwise-identical runs across bundles before (TEST_RESULTS §0).
  - The published runs recorded torch `2.11.0+cu128`, CUDA 12.8, a Tesla T4, fp16, strict determinism and sdpa
    attention (`rec["runtime"]`). Bitwise equality is expected only with that same stack. If Colab's torch or CUDA
    version has changed, expect equality within seed noise instead, and report the versions.
  - **If they do not reproduce on the same stack, stop and investigate before reading anything else.**
- **(c) Parity.** `parity_report.json` must pass (fp32 within 1 % of 1.617155).
- **(d)** `hindko-1.0.0` should be within seed noise of the anchor. Its train encodings differ in 29 tie documents,
  so bitwise equality is not expected.

## 6. How the bundle was made, and what was verified

1. **Candidates.** `lm/make_waves_frontier_test.py` writes `lm/WAVES_frontier_test.json` (sha256 `7530b789…`).
   - It is deterministic.
   - It checks the release sha256 against `F:\Hindko\tokenizer\RELEASE_MANIFEST.json` and the anchor's against
     `WAVES_final_test.json`.
   - It requires every external tokenizer to be `lossless` in `baselines/manifest.json`.
2. **Build.** The command, with output `colab/build_frontier_test/` and log `colab/build_bundle_frontier_test.log`, 876 s:

   ```
   python colab\build_bundle.py --final-test --report-only "<purpose>" --waves lm\WAVES_frontier_test.json --baseline hindko-1.0.0 --encoder-source harness --workers 2 --out colab\build_frontier_test
   ```

   - All `--final-test` guards apply. The test file must match `data/test_manifest.json`, every uid must be
     split=test and tier=strict, and there is no slicing.
   - The builder's own G1 (decode(encode(doc)) == doc on all 491 test documents) passed for all 10 tokenizers.
   - Log entry #2 records the purpose, the test re-use, every tokenizer's sha256 and the code sha256.
   - The CPU parity reference is 1.617155 bpb on the test parity slice. Its training losses equal the dry run's
     step for step.
3. **Independent verification**: `colab/verify_frontier_test.py` → `build_frontier_test/VERIFY_FRONTIER_TEST.json`,
   **18/18 passed**.
   - The parts, the archive and the reassembled bundle (48 files) match their sha256.
   - The candidates, their order and every sha256 equal the WAVES file. `hk_lm.py` equals the one-shot bundle's.
   - **The evaluation set is the one-shot test's:** the six shared arrays are identical to `f54c929ba1ab`'s.
   - **The anchor's 4 arrays are identical to `f54c929ba1ab`'s.** The release's test arrays equal the anchor's, and
     its train arrays differ in the 29 tie documents only.
   - **G1 with each tokenizer's own library** (`tokenizers`, `sentencepiece` with the newline convention, and for
     RoBERTa-Urdu `transformers` RobertaTokenizer cross-checked against a `tokenizers` ByteLevel BPE built from
     `vocab.json` + `merges.txt`):
     - decode(bundle tokens) == text for **491/491** test documents, for all 10 tokenizers;
     - re-encoding reproduces the bundle tokens (491/491);
     - no EOT/BOS id occurs inside a document, every id < V, and uint32 is used wherever V > 65,536.
   - **Test token counts equal `eval/test_competitors.json`** for all 8 external tokenizers.
4. **Code checks.**
   - `colab/frontier_dryrun.py` → `colab/_frontier_dryrun/dryrun_result.json`, **13/13 passed**, on the same code.
     It used a *fake* test file (strict test uids carrying dev texts) and a redirected log, so no real test text was
     read and no real log entry was written. It checked:
     - the new refusals in the builder and in run_all;
     - the report kind in the manifest, PARTS.json and log;
     - uint32 arrays and native EOT/BOS ids;
     - the 29-document release/anchor difference;
     - run_all's seed logic: seeds 1–3 on this kind, 1–5 only if forced, never a re-sweep. The one-shot bundle kind
       is unchanged.
     - **a CPU smoke run of the external GPT-4o tokenizer** through `hk_lm.run_one`: the confirm model with V=200,020,
       3 steps; finite losses; bpb = Σbits/Σbytes; and the windowed evaluation of one document equal to a
       token-by-token brute force (rel 7e-9).
   - `colab/test_local.py`, the existing suite, re-ran on the changed code: **37/37 passed** in 844 s, the same 37
     checks as before, including the unchanged test-split guard (`colab/_test_frontier_regression.log`).
     `colab/PARITY.json` was backed up and restored byte for byte (sha256 `2a35582e…`), because the suite rewrites it.

**Code changes** (backups in `colab/_pre_frontier_test/`; `hk_lm.py` unchanged):
- `build_bundle.py` (`9c8b9d17…` → `f64847ba…`):
  - encoder kind `baseline` (external tokenizers by name, through the harness → `adapters.from_baseline` →
    `load_baselines`);
  - `--final-test --report-only PURPOSE` (kind `final_test_report`, plus log fields);
  - the Llama and DeepSeek BOS/EOT names, appended after the existing ones, so every earlier candidate resolves as
    before.
- `run_all.py` (`a11c2d7e…` → `4431bc0b…`): the `final_test_report` kind.
  - parity and confirm only; `--lr` required; seeds 1–3; the power check is reported but never acted on;
    `--force-extra-seeds` optional;
  - no large stage, no `--smoke`;
  - `--estimate` lists only confirm for test bundles.
  - The `final_test` kind behaves exactly as before.

## 7. Analysis plan (fixed now, before any number exists) and allowed wording

- **Report.** For each tokenizer: the mean test bpb over seeds 1–3, the per-seed values, test bytes/token, vocab, and
  the ids used.
- **Δ vs `hindko-1.0.0`.** Δ = (bpb_x − bpb_release) / bpb_release, with a 95 % CI and a two-sided p.
  - The bootstrap is the study's hierarchical one, `analysis/decide.py`, imported unchanged:
    - 10,000 replicates, `numpy.random.default_rng(12345)`;
    - the 27 test clusters resampled within source;
    - each tokenizer's 3 seeds resampled independently.
  - Per-source numbers (newspaper / book / web) are descriptive only.
- **Holm correction** over the 8 external comparisons, reported next to the raw p.
- **Allowed wording**, when a CI excludes 0 (PLAN §8, AMENDMENT_1 §5, TEST_RESULTS §12):
  - *"With the same small LM (1.8M non-embedding parameters, d=192) trained on the same 45 MB of Hindko, the released
    tokenizer gives x % lower test bits-per-byte than **the tokenizer of** <model> (95 % CI […], 3 seeds, 27 test
    clusters; report-only)."*
  - Always name the scale and "tokenizer".
- **Not claimable from this run:**
  - that the release is better than GPT-4o / Gemma / Llama / Qwen / DeepSeek **as models**;
  - "SOTA", "best on every metric" or "best possible";
  - transfer to large models or continued pretraining;
  - downstream quality.
- **Intrinsic claim, unchanged.** The release uses 40 % fewer tokens than GPT-4o's tokenizer on Hindko test text
  (7.277 vs 4.338 bytes/token), 37 % fewer than Gemma 3/4's, and 22 % fewer than RoBERTa-Urdu's. It is lossless on
  491/491 documents.
