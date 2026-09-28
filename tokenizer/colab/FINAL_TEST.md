# One-shot FINAL TEST on the Colab GPU (PLAN §7.6)

This page describes the only LM run on the strict **test** split in this study. The bundle in
`colab/build_final_test/upload/` is bundle **`f54c929ba1ab`** (manifest sha256 `f54c929ba1ab8f17c001d7fec10deb938f822785587a82bd6c837b1ea861961c`). It was built once, by
`colab/build_bundle.py --final-test`, and recorded in `F:\Hindko\_tokenizer\FINAL_TEST_LOG.json` (entry #0, status
`completed`). The decision was fixed on dev before this bundle existed (`analysis/DECISION.md`,
`analysis/decision.json` → `test_candidates_fixed`, 2026-09-26T20:20:43Z). **No test number may change it.**

> **In this bundle, "dev" means TEST.** The code keeps its names, so these are all on the **strict test split**:
> - the arrays `*/dev_tokens` and `shared/dev_docs.json`;
> - the result field `rec["dev"]` and the `dev bpb` log lines;
> - every `mean bpb` in `summary.txt` and `summary.json`.
>
> The split has 491 documents, 1,451,026 UTF-8 bytes and 27 bootstrap clusters (15 newspaper, 6 book,
> 6 web). `manifest.json` and `PARTS.json` carry `"kind": "final_test"`.

## 1. What is scored

| # | candidate | role (fixed before test) | vocab | train b/tok | ctx | tokenizer sha256 |
|---:|---|---|---:|---:|---:|---|
| 1 | `R2-A10-MinGram-P1r3-D2-48k` | **chosen** | 49,152 | 6.2875 | 244 | `4dd12e627f77…` |
| 2 | `A1-P1r3-D2-16k` | **baseline** (standard recipe) | 16,384 | 5.7984 | 265 | `a725ff94390b…` |
| 3 | `R2-A4-SPnat-D2-32k` | report-only (best mean dev bpb; lost tie-breaker (a)) | 32,768 | 6.1612 | 249 | `1496e9a7471b…` |
| 4 | `R2-A10-MinGram-P1r3-D2-32k` | report-only (next top-set member) | 32,768 | 6.1513 | 250 | `12fcd260b8ea…` |

- **Chosen vs baseline** is the single pre-registered test comparison (PLAN §7.5–7.6).
- The two report-only rows are the rest of the dev top set. They are scored in the same run, but they **never
  re-select**.
- If test contradicts dev in sign, report it and do not re-select (PLAN §7.6).

## 2. The notebook run

Use **`hindko_lm_arbiter_FINAL_TEST.ipynb`** from the upload folder, written by
`colab/make_final_test_notebook.py`. It is the usual notebook with two changes:
- a FINAL TEST warning in the first cell;
- **cell 5** replaced by the final command below.

The generic notebook was removed from this upload folder. If you use it anyway, replace its cell 5 as shown in
step 5.

1. **Runtime → Change runtime type → GPU. Use a T4**, like every dev run (the dev runtime pin was `cuda / fp16 /
   Tesla T4`).
2. Upload every `hk_bundle.tar.xz.part###`, `PARTS.json` and `run_all.py` from `colab/build_final_test/upload/`
   into **one** folder, for example `/content/upload`. Mounting Drive (cell 2, `USE_DRIVE = True`) is recommended:
   results then survive a disconnect.
3. Run cells 1–3 as usual. Cell 3 must show bundle `f54c929ba1ab` and the four candidates.
4. Run cell 4 (`--stages parity`): the CPU↔GPU parity check, exactly as before (100 steps, seed 1, baseline).
   - In this bundle the parity evaluation slice is about 100 KB of **test** documents, because the bundle has no
     other evaluation text.
   - It is a sanity check of the GPU, not a result. The CPU reference is in the bundle's `PARITY.json`.
5. **Cell 5** (already set in the FINAL_TEST notebook; in the generic notebook, replace `rc = run_all()` with it):

   ```python
   rc = run_all('--stages', 'parity,confirm', '--lr', '1e-3', '--no-zip', '--force-extra-seeds')
   ```

   - The unchanged `rc = run_all()` is **refused** by run_all.py for this bundle, and nothing runs. So are
     `lr`, `screen`, `large`, `--smoke`, and `confirm` without `--lr` or without `--force-extra-seeds`.
   - The notebook's helper adds `--parts`, `--work /content/work` and `--out <OUT>`.
6. Run cell 6 to zip the results, or take the folder `<OUT>/f54c929ba1ab/` from Drive.

The same run from a shell (the exact command line):

```
python run_all.py --parts <folder> --stages parity,confirm --lr 1e-3 --no-zip --force-extra-seeds
```

- **`--force-extra-seeds`** is the flag that makes the power-check seeds 4 and 5 always run. It is new in
  run_all.py.
  - Without it, run_all.py adds seeds 4 and 5 only if the pooled seed s.d. of seeds 1–3 exceeds 0.2 % of bpb
    (PLAN §6).
  - The dev decision used 5 seeds per candidate. The test must use the same 5, whatever the power check says on
    test. `power_check.json` is still written, report-only, with `"extra_seeds_forced"`.
- **Not on a T4?** Add `--amp fp16` to keep the dev precision. Bitwise identity with the dev models (§4, check b)
  is then not expected, and that must be reported.

## 3. What run_all.py runs on this bundle

| step | what | runs |
|---|---|---:|
| parity | fixed tiny config (100 steps, seed 1, baseline), device AMP dtype + fp32, vs the CPU reference (1 % tolerance) | 2 |
| confirm, seeds 1–3 | every candidate × seeds 1, 2, 3 | 12 |
| confirm, seeds 4–5 | every candidate × seeds 4, 5 (forced) | 8 |
| **not run** | LR sweep (`lr` stage refused; `--lr 1e-3` fixed), `screen`, `large`, the finalist LR re-sweep (skipped for final-test bundles) | 0 |

**The confirm recipe is the dev Stage 4 recipe, unchanged** (the same `hk_lm.py`, sha256 `3f78ad799f56cd4699a8050d1e32c802ea0981feedbf26399322f145a117e12d`):
- model d=192, L=4, H=4, tied embeddings, 1.77M non-embedding parameters;
- the full permissive train split (train_D1, 16,015 documents, 45,007,515 bytes), 1 epoch = 3,663 steps of
  8 sequences;
- context = round(1,536 ÷ train bytes/token) tokens (the *ctx* column of §1);
- AdamW, peak LR **1e-3**, 50 warm-up steps, cosine decay to 10 %;
- fp16 autocast on the T4 for training, fp32 for evaluation;
- evaluation of all 491 test documents with windows of ctx tokens and stride ctx/2 (PLAN §5), storing
  per-document bits and bytes;
- the learning curve at 25/50/75/100 % on a fixed 0.5 MB subset of the test documents.

**Expected time on a T4:** about 35–40 minutes, from the dev Stage 4 timings of the same candidates (per run:
67 s at 16k, 101 s at 32k, 138 s at 48k; 5 seeds each; plus parity and unpacking). Evaluation takes about as
long as on dev: test has 1.451 MB, dev 1.446 MB.

**Resuming is safe.** A disconnect loses at most the run in progress. Re-running the same command skips every
finished run after checking its identity (bundle, hk_lm.py, recipe, seed, exact LR, device and dtype), and
re-trains the interrupted run, deterministically. Never delete or edit a result file to "re-run" a candidate: the
test is scored once.

## 4. Results and the checks to run first

- **Results:** `<OUT>/f54c929ba1ab/results/confirm__<candidate>__lr1e-3__s<seed>.json`, 20 files.
  - `rec["dev"]["bits"]`, `["bytes"]` and `["uids"]` are per **test** document.
  - The bootstrap clusters (the manifest `group` collapsed to `cluster`) are in the bundle's
    `shared/dev_docs.json`, and in `data/test_strict.jsonl`.
- **`summary.txt`**: the `confirm (lr 1e-3)` block is the TEST table.

Check before any analysis:
- **(a) Completeness:** 20 records, `status == "ok"`, 491 documents each, and `uids` equal to the uids of
  `data/test_strict.jsonl` in uid order.
- **(b) Same models as dev:** each record's `train_loss` must equal the dev Stage 4 record of the same
  candidate and seed.
  - The dev records are `G:\My Drive\hindko_lm_out\77e1368773fc\results\confirm__<candidate>__lr1e-3__s<seed>.json`
    (all four candidates), and `...\1d24425d2d64\...` too for `A1-P1r3-D2-16k`.
  - Training never reads the evaluation set: the train token arrays are verified identical (§5), and so are the
    seeds, the LR, the steps and `hk_lm.py`.
  - The dry run confirms this on the CPU: the parity training losses on the fake-test bundle equal those of the
    dev bundles, step for step.
  - Equal losses therefore mean the test models **are** the dev-selected models.
- **(c) Parity:** `parity_report.json` must pass, as on dev (dev: fp16 within 0.45 %, fp32 within 0.02 %).

## 5. Analysis (pre-registered; after the run)

- **Primary:** Δ = bpb(chosen) − bpb(baseline) on test, `R2-A10-MinGram-P1r3-D2-48k` − `A1-P1r3-D2-16k`.
- **Bootstrap:** the same hierarchical bootstrap as dev (`analysis/decision.json` → `protocol`):
  - 10,000 replicates, `numpy.random.default_rng(12345)`;
  - clusters resampled with replacement within each source, over the 27 test clusters (15 / 6 / 6);
  - each candidate's 5 seeds resampled independently;
  - 95 % percentile CI, and two-sided p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)).
- **Improvement claim** (PLAN §7.5–7.6, DECISION.md §9): only if the test CI excludes 0 **and** the test Δ
  agrees in sign with dev (dev: −1.58 %, 95 % CI [−1.87, −1.36]; chosen 1.22636 bpb against 1.24607 for the
  baseline).
- **Report-only:** the two report-only rows, per-source numbers, and the equivalence (TOST) readings. They
  never re-select.
- **Wording** (DECISION.md §9):
  - "chosen by the pre-registered rule; selected after an exploratory second round; confirmed on a sealed test
    split used once" if test agrees;
  - otherwise report the contradiction as it is.

## 6. How the bundle was made, and what was verified

1. **Test text.** `data/materialize_test.py` wrote `data/test_strict.jsonl` and `data/test_manifest.json`:
   - It is `data/materialize.py` for one view: the same FROZEN.json checks, `iter_split` with the manifest check,
     `hp.normalize` 1.0.1, the same fields and cluster rule.
   - Before reading any test row, the same row function re-made `validation/strict` byte for byte equal to
     `dev_strict.jsonl` (sha256 `b6553953…`).
   - The result: 491 documents, every one split=test and tier=strict in manifest `76582d3a…`; 1,451,026
     bytes; 27 clusters (15/6/6), as in `research/split_facts.json`; 0 documents changed by
     normalisation.
   - File sha256 `a74c33ad008f4b0e4a6f7779b0de24e265b0f4edf6f94e89885c7943d3a08468`.
   - `data/data_manifest.json` was not touched: its sha256 is recorded in the dev bundles.
2. **Candidates.** `lm/make_waves_final_test.py` wrote `lm/WAVES_final_test.json`: the four ids of
   `decision.json` (sha256 of the id list `86e95a4a…`, checked), with the entries copied verbatim from
   `lm/WAVES_round2.json` and every tokenizer file re-hashed.
3. **Build.** Command:
   `python colab\build_bundle.py --final-test --waves lm\WAVES_final_test.json --baseline A1-P1r3-D2-16k --encoder-source harness --workers 1`
   (output `colab/build_final_test/`, log `colab/build_bundle_final_test.log`).
   - `--workers 1` only kept the machine at 4 processes while the harness ran. The encodings do not depend on it.
   - The CPU parity reference is `PARITY.json` inside the bundle: 1.689204 bpb on its test slice. Its training
     losses equal the dev bundles' parity runs step for step. `colab/PARITY.json` still holds the round-1
     reference.
   - `read_jsonl` lifts its two test guards for this one file only; every uid must then be split=test.
   - The file must match `data/test_manifest.json`.
   - The builder's own G1 (decode(encode(doc)) == doc on every test document) must pass for every candidate.
4. **Independent verification** (`colab/verify_final_test.py` → `colab/build_final_test/VERIFY_FINAL_TEST.json`,
   11/11 checks passed):
   - parts, archive and bundle files match their sha256;
   - the candidates and their order equal `test_candidates_fixed`;
   - **the train encodings are identical to the dev bundles:** for every candidate, the sha256 of
     `train_tokens.npy` and `train_offsets.npy`, the token count, train bytes/token, ctx, n_vocab and EOT/BOS ids
     equal the round-1 (`1d24425d2d64`) and/or round-2 (`77e1368773fc`) manifests; the shared train arrays and
     hk_lm.py are equal too;
   - the bundle's "dev" set equals `test_strict.jsonl` (uids, bytes, clusters);
   - **G1 on test_strict for each candidate:** every document round-trips, with the harness encoder and with the
     native library encoder, and re-encoding reproduces the bundle tokens.

| candidate | G1 harness | G1 native | test tokens | test bytes/token | dev bytes/token |
|---|---:|---:|---:|---:|---:|
| `R2-A10-MinGram-P1r3-D2-48k` | 491/491 | 491/491 | 196,793 | 7.3734 | 7.2237 |
| `A1-P1r3-D2-16k` | 491/491 | 491/491 | 208,826 | 6.9485 | 6.7883 |
| `R2-A4-SPnat-D2-32k` | 491/491 | 491/491 | 199,390 | 7.2773 | 7.0764 |
| `R2-A10-MinGram-P1r3-D2-32k` | 491/491 | 491/491 | 199,455 | 7.2750 | 7.1258 |

5. **Code checks.**
   - `colab/final_test_dryrun.py` exercised every `--final-test` path on a fake test file: strict test uids
     (manifest metadata) carrying dev texts. No real test text was read. Result:
     `colab/_final_dryrun/dryrun_result.json`, 14/14 passed.
   - Its fake bundle, fake data and outputs were deleted afterwards, so they cannot be uploaded by mistake. Its
     manifest and its log (`DRYRUN_FINAL_TEST_LOG.json`) are kept.
   - `colab/test_local.py` re-ran on the changed code and passed 37/37 (the same 37 checks as before). It
     includes the unchanged test guard: a test uid and a `*test*` file name are refused without the flag.

**Code changes for this step** (backups of the previous files are in `colab/_pre_final_test/`):
- `build_bundle.py` (`392466b5…` → `9c8b9d17…`): the `--final-test` flag only.
  - It adds the test guards' lift for one file, the test_manifest check, the log record, `kind='final_test'` in
    the manifest and PARTS.json, the default `--out build_final_test`, and no copy of PARITY.json to `colab/`.
  - It refuses `--max-*-docs` and `--allow-roundtrip-failures`.
  - Without the flag the behaviour is unchanged.
- `run_all.py` (`91b48ce6…` → `8fbae15a…`) gains two things:
  - `--force-extra-seeds`;
  - the final-test rails: only parity and confirm, `--lr` and `--force-extra-seeds` required, no `--smoke`, no
    finalist re-sweep.
  - For any bundle without `kind` (all dev bundles) and without the new flag, behaviour is unchanged. The dev
    bundles also carry their own copies of run_all.py.
- `hk_lm.py` is unchanged (`3f78ad79…`), so the training recipe is the dev one.

## 7. Deviations from PLAN (all additive; to be reported with the test result)

Known before this step:
1. **GPU instead of CPU.** The LM ran on a Colab T4 in fp16 (evaluation in fp32). CPU parity: within 0.45 % in
   fp16 and 0.02 % in fp32.
2. **Stage 4 'confirm' ran for all candidates**, not only the top 2 plus the baseline.
3. **Two D1 reference arms were added in round 1**, because the rank-1 baseline was D2 while the Stage 2
   candidates were D1.
4. **Round 2 is exploratory.** Its 6 candidates were built after the Stage 3 and Stage 4 results were seen, and
   the chosen tokenizer and the whole top set come from it.
   - This is a garden-of-forking-paths risk: the dev Δ is conditioned on dev (winner's curse).
   - The sealed test split, used once here, is the guard.
5. **The LR is at the edge of the grid.** The sweep {1e-3, 3e-3, 6e-3} on the baseline chose 1e-3, the lower edge.
   The finalist re-sweep confirms that 1e-3 beats 3e-3 and 6e-3. No LR below 1e-3 was tried.

Specific to this final test:

6. **Four candidates, not two.** PLAN §7.6 names the chosen tokenizer and the baseline. The two other top-set
   members are scored too, labelled report-only before the test (`decision.json`). They never re-select.
7. **Seeds 1–5, forced.** PLAN §7.6 says "same protocol". The dev decision used 5 seeds per candidate, because
   the power check added seeds 4 and 5. The test uses the same 5 unconditionally (`--force-extra-seeds`), so the
   seed count cannot depend on test data.
8. **Stage 4 recipe.** Test is scored with the Stage 4 'confirm' arbiter that ranked on dev (DECISION.md §6). No
   Stage 3 'screen' and no 'large' run is made on test.
9. **Parity slice on test text.** The CPU↔GPU parity check evaluates its 100-step model on about 100 KB of test
   documents: the bundle's only evaluation text. It is a device sanity check with no bearing on any comparison.
10. **Retrained, not reused, models.** The dev runs did not keep model weights, so the 20 models are retrained
    with the identical deterministic recipe. Check (b) of §4 proves they are the same models, if the GPU type and
    dtype match dev.

## 8. Files

| file | what |
|---|---|
| `colab/build_final_test/upload/` | **what to upload**: 4 parts, `PARTS.json`, `run_all.py`; plus the notebook `hindko_lm_arbiter_FINAL_TEST.ipynb` |
| `colab/build_final_test/staging/manifest.json` | bundle manifest, `kind: final_test` (sha256 `f54c929ba1ab8f17c001d7fec10deb938f822785587a82bd6c837b1ea861961c`) |
| `colab/build_final_test/VERIFY_FINAL_TEST.json` | independent verification (train identity, G1 on test) |
| `F:\Hindko\_tokenizer\FINAL_TEST_LOG.json` | the audit record of this build: time, candidates, tokenizer sha256, test sha256, code sha256. Its top-level note lists the other one-shot readers of the test text in this phase: materialisation, verification and the intrinsic table |
| `data/test_strict.jsonl`, `data/test_manifest.json`, `data/materialize_test.py` | the normalised strict test view |
| `lm/WAVES_final_test.json`, `lm/make_waves_final_test.py` | the four fixed candidates |
| `eval/TEST_COMPETITORS.md`, `eval/test_competitors.json` | the one-shot intrinsic table on test: chosen vs every external tokenizer |
