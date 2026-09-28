# Tokenizer LM arbiter on a Colab GPU

The LM arbiter of `research/PLAN.md` §5 runs on a Google Colab GPU instead of this CPU.
- One module, `hk_lm.py`, holds the model, the data pipeline, the optimiser and the evaluation. The same file runs on the GPU and on this CPU.
- Encoding happens locally with each finalist's exact shipped encoder. Colab receives token arrays only; it never tokenizes anything and never sees the test split.
- Every number called an *estimate* below was computed, not measured on a GPU.

## Files (`F:\Hindko\_tokenizer\colab\`)

| file | role |
|---|---|
| `hk_lm.py` | the arbiter: GPT, byte-matched data, AdamW schedule, per-document evaluation, result records (torch + numpy only) |
| `build_bundle.py` | encodes train D1 and dev_strict with every finalist and packs the upload (parts ≤ 9.5 MB, `PARTS.json`) |
| `run_all.py` | on Colab: reassembles and verifies the parts, then runs parity → LR sweep → Stage 3 → Stage 4 → large; resumable |
| `make_notebook.py` → `hindko_lm_arbiter.ipynb` | the Colab notebook (7 cells, pure Python, installs nothing) |
| `test_local.py` | small CPU tests (§9) |
| `PARITY.json` | CPU reference of the fixed tiny config for the **current** bundle. `run_all.py` re-runs it on the GPU first |
| `build\staging\` / `build\upload\` | unpacked bundle / what goes to Colab (written by `build_bundle.py`; created by the real build) |
| `_dryrun_pilot\` | full-size dry-run bundle made with the pilot tokenizers (§2.1): for testing the Colab path only |
| `_test\` | test outputs (`test_local_result.json`), the test bundle, `amp_path_check.py`, `bytes_seen_check.py` |
| `FINAL_TEST.md`, `build_final_test\` | the ONE-SHOT final test (PLAN §7.6): `build_bundle.py --final-test` bundle whose 'dev' set is the strict TEST split; run_all.py allows only `--stages parity,confirm --lr 1e-3 --force-extra-seeds` on it. Read `FINAL_TEST.md` |

## 1. Build the bundle (on this machine, once `lm\WAVES.json` exists)

```bat
set PYTHONIOENCODING=utf-8
python F:\Hindko\_tokenizer\colab\make_notebook.py
python F:\Hindko\_tokenizer\colab\build_bundle.py --encoder-source harness
```

- **Candidates** come from `lm\WAVES.json`. Every entry that has `tokenizer_path` and `encoder_kind` is used, at any nesting depth, in file order. Optional per-entry fields:
  - `baseline: true` marks the baseline;
  - `newline_wrapper: false` for a SentencePiece model without a `\n` piece;
  - `custom: "MODULE:FACTORY"` for custom encoders.
- **Baseline:** `--baseline ID`, else a top-level `"baseline"` entry, else a `baseline: true` flag, else an id matching `A1…P1…16`. It is recorded in the manifest.
- **Encoders:** `--encoder-source harness` makes `eval\harness.py` `build_adapter()` mandatory. That call is the one `harness.py run` makes, so the result is exactly the adapter the intrinsic sweep evaluated:
  - `HFAdapter`;
  - `SPAdapter` with the PLAN §1.1 newline wrapper;
  - `CustomAdapter`.

  `adapter.identity()` and `harness.code_hashes()` are stored per candidate.
- **Text:** `data\train_D1.jsonl` (all 16,015 permissive-train documents) and `data\dev_strict.jsonl` (836 documents). Both are sorted by uid, and their sha256 is checked against `data\data_manifest.json`.
- **Split guard:**
  - every uid is checked against the split manifest: train rows must be `train`, dev rows `validation` and `strict`;
  - a test uid aborts the build;
  - a file whose name contains "test" is never opened.
- **Hard checks.** The build stops on any of these:
  - `decode(encode(doc)) == doc` fails for any dev document (G1);
  - an EOT/BOS id occurs inside a document (G3);
  - an id is ≥ n_vocab.
- **Then:**
  - it runs the parity config on the CPU (100 steps; 65–120 s measured on this loaded machine);
  - it writes `PARITY.json` into the bundle and copies it to `colab\PARITY.json`;
  - it packs `build\upload\`: `hk_bundle.tar.xz.part###` (each ≤ 9,500,000 bytes), `PARTS.json`, `run_all.py` and the notebook.

## 2. What gets uploaded

Upload **every** `hk_bundle.tar.xz.part###`, plus `PARTS.json` and `run_all.py`, into **one** Colab folder, preferably `/content/upload`. Open `hindko_lm_arbiter.ipynb` in Colab (File → Upload notebook).

Inside the archive:

| path | content |
|---|---|
| `manifest.json` | sha256 and size of every file, candidate metadata (tokenizer path + sha256, encoder identity, n_vocab, EOT/BOS ids, train/dev bytes per token, ctx_tokens, round-trip result), data sha256s, split-manifest sha256 |
| `shared/train_bytes.npy`, `shared/dev_bytes.npy` | UTF-8 bytes per document (int64), shared by all candidates |
| `shared/train_uids.txt`, `shared/dev_docs.json` | uids; for dev also group, bootstrap cluster, source and variety (for PLAN §6) |
| `shared/dev_subset_idx.npy`, `shared/dev_parity_idx.npy` | the fixed 0.5 MB learning-curve subset; the parity slice |
| `cand/<id>/{train,dev}_{tokens,offsets}.npy` | uint16 tokens (uint32 if an id > 65,535) and int64 document offsets |
| `hk_lm.py`, `run_all.py` | the code (their sha256 are in the manifest) |
| `PARITY.json` | CPU reference for the parity check |

**Size** (measured on the full-size dry-run build, §2.1): about **7.8 MB of xz per 16k candidate** (15.6 MB for two, from a 31.7 MB tar). Nine finalists come to ≈ 70 MB ≈ 8 parts of ≤ 9.5 MB, plus `PARTS.json` (1 KB), `run_all.py` (40 KB) and the notebook (7 KB). Every file is under the 10 MB upload limit.

### 2.1 Dry-run bundle (available now, before `WAVES.json` exists)

`colab\_dryrun_pilot\upload\` is a complete full-size bundle: the full D1 train and dev_strict, encoded with the two pilot tokenizers through the harness adapters.
- It has 2 parts (9,500,000 + 6,119,224 bytes), and `colab\PARITY.json` currently belongs to it (bundle `b6d4d9c0a843`, CPU parity bpb 1.637265, unchanged from the earlier builds). It was rebuilt after the resume-safety fix, so it carries the fixed `hk_lm.py` and `run_all.py`.
- Build time on this loaded CPU: 160 s, 226 s and 549 s in three builds (the last one on a much busier machine). Encoding in 2 worker processes took 39–144 s for HF and 4–33 s for SentencePiece; CPU parity took 86–311 s; hashing, tar and xz took ≈ 30–60 s, by difference.
- Uploading it to Colab measures the real GPU speed and exercises the whole path before the finalists are known.
- Its **numbers must not be used**: the pilot tokenizers were trained on a split that is not group-disjoint.
- The real build (`build\`) makes its own `PARITY.json` and overwrites `colab\PARITY.json`.

No tokenizer files, no raw text and no test data are uploaded.

## 3. Run on Colab

Set Runtime → Change runtime type → GPU. Then run the notebook's cells in order:

1. **GPU check.** It prints the GPU and the precision `run_all.py` will use: bf16 on compute capability ≥ 8 (L4, A100), fp16 + GradScaler on a T4.
2. **Optional Drive mount.** Off by default, because mounting opens a Google consent dialog that the account owner must approve. With Drive mounted, results are written to `MyDrive/hindko_lm_out/` after every run and survive a disconnect.
3. **Locate the parts.** It checks that every part is present with the right size.
4. **Parity.** `run_all.py --stages parity` verifies the parts (sha256), reassembles them, verifies the archive and every file (sha256), then trains the fixed tiny config on the GPU twice, in AMP and in fp32. It prints both bpb values next to the CPU value from `PARITY.json`.
   - Tolerance: 1 % bpb.
   - A fp32 mismatch stops everything; an AMP mismatch only warns.
5. **Full run.** Plain `run_all.py`, with live output and resumable: every finished (stage, candidate, seed, lr) is skipped on a re-run.
   - A stored result is reused only if it is the run being requested. Its record must match on: bundle, `hk_lm.py` sha256, stage label, recipe, candidate, seed, **exact** peak LR, model, budget bytes, steps, ctx, overrides, and device + AMP dtype. On any mismatch `run_all.py` stops with `REFUSING TO RESUME` and lists the fields; it never skips silently. A device/dtype difference alone is accepted only with `--allow-mixed`, with a warning.
   - `--lr X` or `--large-lr X` with a value off the grid (e.g. 2.5e-3) gets its own result files (`lr2.5e-3`); it never reuses the grid runs (`lr3e-3`).
   - `--max-hours H` stops cleanly before a session limit.
   - `run_all('--estimate')` prints a FLOP-based estimate for the GPU it finds.
6. **Zip.** It zips the results and prints `summary.txt`.

**Precision is pinned.** The first real training run writes `runtime_pin.json` (device and AMP dtype); parity runs never pin. If a later session lands on another GPU type, it reuses the pinned dtype, and a device change is refused. Candidates are therefore never trained in different precisions (`--allow-mixed` overrides this, and the override is recorded).

**No accidental CPU runs.** If no GPU is found, `run_all.py` runs only the parity stage and refuses the arbiter stages, unless `--device cpu` is given explicitly.

## 4. Expected GPU runtime per stage (**ESTIMATE**, not measured)

The estimate is FLOP-based: training ≈ 6·N_matmul·tokens + causal attention, with N_matmul including the tied output layer; evaluation ≈ ⅓ of that per token.
- Assumed *effective* throughput for these tiny models: T4 4 TFLOP/s, L4 8, A100 20.
- Floor per optimiser step, because tiny models are bound by kernel launches: 8 / 6 / 4 ms.
- Plus 10 s per run for set-up and evaluation overhead.
- The slate is PLAN §2.2 ranks 1–7 plus conditional rank 8 (9 candidates: six at 16k, one at 8k, two at 32k), with the pilot bytes/token.

| stage | runs | steps / run | training PFLOP / run (16k) | T4 | L4 | A100 |
|---|---|---|---|---|---|---|
| LR sweep (baseline, 3 LRs) | 3 | 814 | 0.030 | 1 min | 1 min | 1 min |
| Stage 3 screen (9 × 3 seeds) | 27 | 814 | 0.030 | 9 min | 7 min | 6 min |
| Stage 4 confirm (9 × 3 seeds) | 27 | 3,663 | 0.230 | 34 min | 19 min | 12 min |
| Stage 4 seeds 4, 5 (only if the power check asks) | 18 | 3,663 | 0.230 | 23 min | 13 min | 8 min |
| large LR sweep (baseline, 3 LRs) | 3 | 7,325 | 1.575 | 21 min | 11 min | 5 min |
| large (9 × 2 seeds) | 18 | 7,325 | 1.575 | 129 min | 66 min | 28 min |
| **total** | 96 | | | **≈ 3.6 h** | **≈ 1.9 h** | **≈ 1.0 h** |

For comparison, PLAN §5 estimated ≈ 5.4 h of CPU time for Stage 3 alone and ≈ 8–12 h for a three-candidate Stage 4. The 'large' stage dominates the GPU budget. It is last and report-only, so an interrupted session loses nothing that the decision needs.

## 5. How results come back

`<out>/<bundle sha256[:12]>/` (Drive or `/content/out`), zipped by `run_all.py` and by cell 6:

| file | content |
|---|---|
| `results/<stage>__<candidate>__lr<lr>__s<seed>.json` | one record per run (below). `<lr>` is lossless: the shortest form that parses back to the same number (`3e-3`, `2.5e-3`, `5.8e-3`), so two LRs never share a file. Stage means and power checks select records by their own `lr` field, not by file name |
| `summary.txt`, `summary.json` | per stage: mean bpb, seed s.d., Δ % vs baseline, ranks, plus the checks below |
| `lr_choice.json`, `large_lr_choice.json` | the sweeps and the chosen LRs |
| `power_check_screen.json`, `power_check.json` | PLAN §6 power checks: the literal one on the baseline's Stage 3 wave, and the pooled one on Stage 4 that decides seeds 4 and 5 |
| `parity_report.json`, `parity/` | GPU vs CPU parity |
| `runtime_pin.json`, `progress.log` | precision pin; every progress line |

**Each run record contains:**
- identity: protocol version, status (`ok` or `diverged`), stage, candidate, seed, peak LR;
- `preregistered`, and the model config with parameter counts (total, tied embedding, positions, non-embedding);
- the optimiser and schedule;
- data accounting: budget bytes, steps, ctx_tokens, tokens seen, bytes actually seen, epochs seen, BOS targets masked, and the doc-order and sequence-order rules;
- runtime: device, GPU name, AMP dtype, eval dtype fp32, deterministic mode, attention kernel, torch/CUDA versions;
- wall time (train and eval), the full training-loss curve, and the curve points at 25/50/75/100 % on the 0.5 MB subset;
- PLAN §4.3 R3: the 50 lowest-norm output embeddings with their train frequencies;
- the sha256 of the bundle manifest and of `hk_lm.py`;
- `dev`: `bpb`, `sum_bits`, `sum_bytes`, and **per document** `uids`, `bits`, `bytes`, `ntok`.

The PLAN §6 cluster bootstrap joins `dev.uids` with the `cluster` field of `shared/dev_docs.json` (or `data\dev_strict.jsonl`). The decision rule (PLAN §7) reads the **Stage 4 `confirm`** records at the chosen LR.

**`summary.json` also reports:**
- the Stage 3 **curve-crossing check** (PLAN §5: top two on the subset at 75 % vs 100 %);
- a **GPU determinism check**: the baseline's seed-1 run at the chosen LR is trained twice, once in the sweep and once in Stage 3, and must be bitwise identical.

## 6. Protocol mapping (PLAN §5 → `hk_lm.py`)

| PLAN | implementation |
|---|---|
| GPT, pre-LN, GELU MLP 4×, learned positions, tied embeddings, no dropout | `GPT`/`Block`. Linear layers are bias-free and LayerNorm has a bias: 0.79M / 1.77M non-embedding parameters at d = 128 / 192, as in PLAN |
| screen d=128 L=4 H=4; confirm d=192 L=4 H=4 | `MODELS["screen"]`, `MODELS["confirm"]` |
| chunked output loss (512 rows) | `train_step`: the hidden state is detached and the logits are built chunk by chunk. 512 rows on CPU; on GPU the whole 8-sequence batch is one chunk (same arithmetic, bigger kernel) |
| same training bytes: first B bytes of the seed-shuffled stream | document order = `PCG64(SeedSequence([seed, epoch, 0x484B])).permutation(n_docs)`, identical for every candidate. steps = round(B / 12,288) is identical for every candidate; candidate c reads the first steps·8·ctx(c)+1 tokens of its stream |
| ctx_tokens(c) = round(1536 / train bytes/token) | `ctx_tokens()`; bytes/token = Σ train bytes / Σ train document tokens |
| 8 sequences per step, ≈ 12 KB | `SEQS_PER_STEP = 8` |
| documents joined with `<\|endoftext\|>` | stream = `[BOS] doc [EOT] [BOS] doc [EOT] …` |
| AdamW 0.9/0.95, ε 1e-8, wd 0.1 on matrices, clip 1.0, warm-up 50, cosine to 10 % | `make_optimizer`, `lr_at`, `train_step` |
| LR {1e-3, 3e-3, 6e-3} on the baseline, 1 seed, screening budget, then fixed | `run_all.py` stage `lr` → `lr_choice.json`; used by screen, confirm and the re-sweep reference |
| `set_flush_denormal`, 1 thread, deterministic algorithms, fixed seeds | `Runtime`: flush-denormal always; `--threads` on CPU. On CUDA: `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, cuDNN deterministic. A start-up probe falls back to math attention, then to warn-only, and records which |
| evaluation: every dev_strict document alone, windows ctx / stride ctx/2, BOS given, EOT predicted | `eval_windows`, `evaluate` (fp32 forward, float64 sums) |
| per-document NLL (bits) and bytes | `rec["dev"]` |
| 0.5 MB dev subset at 25/50/75/100 % | subset = `PCG64(12345).permutation(n_dev)`, taking documents until ≥ 0.5 MB (fixed, candidate-independent). The 100 % point is read from the full evaluation |
| bpb = Σ bits / Σ bytes (§4.1) | `bpb_of`; bytes = UTF-8 bytes of the normalised text, no separators |
| power check σ > 0.2 % → more seeds (§6) | `run_all.py` `stage_confirm` (pooled over candidates) and `stage_screen` (baseline, literal §6) |

## 7. Deviations from PLAN, and why they only add evidence

### 7.1 Deviations

1. **Hardware.** The arbiter runs on a GPU, not this CPU.
   - Training uses mixed precision: bf16 autocast, or fp16 + GradScaler on GPUs without native bf16. Weights, optimiser state and **evaluation are fp32**.
   - The model, data, optimiser, schedule, budgets and evaluation are unchanged.
   - Every candidate in a study trains on the same device type and dtype (the pin).
   - CPU↔GPU agreement is measured before any real run (parity, 1 % tolerance).
2. **Stage 4 for every candidate.** PLAN has the top 2 + baseline, "optional", 8–12 h of CPU. On the GPU all candidates cost ≈ 34 min on a T4 (estimate, §4).
3. **Seeds 4 and 5 for every candidate**, not only the finalists, when the pooled Stage 4 seed s.d. exceeds 0.2 % of bpb. This keeps the seed counts balanced across the comparison. The PLAN §6 formula's seed count is printed; the script stops at 5.
4. **'large' arbiter, an ADDITION beyond the pre-registration.**
   - Model: d=384, L=6, H=6, 10.6M non-embedding parameters (≈ 17M total at 16k, ≈ 23M at 32k). This is the strengthening step PLAN §8(ii) suggests, at its lower end.
   - Data: the full train split for **2 epochs**, chosen because one pass of ~7.5M tokens under-trains a 10M-parameter model. Repeating data a few times costs little (Muennighoff et al. 2023), and dev is disjoint.
   - LR: its own sweep {5e-4, 1e-3, 2e-3} on the baseline (seed 1, full large budget). The d=128 LR would be too high for d=384.
   - Every candidate × seeds {1, 2}. **Report-only.**
5. **The finalist LR re-sweep of PLAN §5** ("re-sweep the finalists in Stage 4 if their bpb gap is < 1 %") is implemented as report-only. For the top two by Stage 4 mean, seed 1 is run at the other two grid LRs (`confirm_resweep`). PLAN does not say how a re-sweep would enter the decision, so it only shows whether the ranking depends on the LR.

### 7.2 Why none of this changes what decides

- PLAN §7 step 2 ranks by "mean dev bpb (**Stage 4 results if run**, otherwise Stage 3)". Stage 4 is run, for every candidate, so the ranking uses Stage 4 `confirm` records at the pre-registered LR (chosen by the pre-registered sweep), with the pre-registered model, budget and evaluation.
- Running Stage 4 for every candidate removes a selection step and adds nothing new to the decision:
  - no candidate is dropped on the smaller Stage 3 budget before the larger arbiter sees it;
  - all ranked numbers come from one stage, never a mix of Stage 3 and Stage 4;
  - the Holm correction of PLAN §6 already covers all k−1 comparisons against the best candidate.
- Stage 3 still runs as pre-registered. It supplies the §5 crossing check and a second, smaller-budget ranking.
- Extra seeds only narrow the bootstrap CIs; the test and the margins are unchanged.
- `large`, `large_lrsweep` and `confirm_resweep` are labelled report-only (`preregistered: false` or a separate stage name) and are never read by the decision. If `large` disagrees with `confirm`, that is reported as a limitation of the claim (PLAN §8), not used to re-select.
- The hardware change is covered by the parity check and recorded per run. Every candidate within a study shares one device type and dtype.

### 7.3 Choices PLAN left open, fixed here identically for every candidate

- **BOS/EOT.**
  - A training document is `[BOS] doc [EOT]`. BOS is the tokenizer's `<|bos|>`, else `<s>`, else an appended id. EOT is `<|endoftext|>`, else `</s>`, else an appended id.
  - The EOT → BOS target is masked (ignore_index), because it is trivial and is not text.
  - Evaluation feeds `[BOS] doc [EOT]` and predicts every document token plus EOT; it never predicts BOS.
  - PLAN §2.1 requires `<|endoftext|>` and `<|bos|>` in every candidate's special block, so normally nothing is appended. The manifest records `specials_appended` per candidate; the pilot HF tokenizer used in the tests had no BOS, so one was appended.
- **Sequence order.** The steps·8 packed sequences are presented in a seeded permutation (`PCG64(SeedSequence([seed, 0x5E0, 0x484B]))`) rather than in stream order, so each batch mixes documents. It depends only on (seed, steps·8), so it is identical across candidates.
- **"First B bytes".** B = 10,000,000 bytes (PLAN: ≈ 1.66M tokens at 6.009 bytes/token). The step count is round(B / 12,288) = 814, identical for all.
  - Because ctx is rounded, each document adds 2 special tokens and the local bytes/token of the sampled documents varies, the bytes actually seen differ slightly from B. Measured on the full-size dry-run bundle (`_test\bytes_seen_check.py`, 2 candidates × 3 seeds):
    - screen: 99.79–100.60 % of B (it varies with the seed's documents; at a given seed the candidates differ by ≤ 0.13 points);
    - confirm and large: 99.68–99.73 % (candidates within 0.05 points).

    Every run records it (`data.bytes_seen`, `data.epochs_seen`).
  - Full-train stages use B = epochs × 45,007,515 bytes.
- **Init:**
  - normal(0, 0.02);
  - residual projections scaled by 0.02/√(2L);
  - drawn from a CPU generator, so CPU and GPU start from identical weights.
- **Weight decay** covers every parameter with ≥ 2 dimensions (all linear layers, the tied embedding and the positions). LayerNorm parameters are not decayed.
- **GELU** is exact (erf). The warm-up is linear from peak/50. The cosine ends at 10 % of peak on the last step.
- **Divergence:** a run whose last 20 losses are non-finite or have a mean above 50 nats is recorded as `diverged` and gets no bpb. The LR sweeps ignore it.

## 8. Reproducing on this CPU

```bat
set PYTHONIOENCODING=utf-8
python F:\Hindko\_tokenizer\colab\hk_lm.py --bundle F:\Hindko\_tokenizer\colab\build\staging --cand <id> --stage screen --seed 1 --lr <chosen> --device cpu --threads 1 --out r.json
```

This is the same code on the same arrays, in fp32.
- Two identical CPU runs are bitwise identical (tested).
- A CPU run matches a GPU run within the parity tolerance, not bit for bit, because summation order and AMP differ.
- `run_all.py --bundle-dir build\staging --device cpu` runs the whole schedule on the CPU. It runs one process at a time; PLAN §5 budgeted ≈ 5.4 h for Stage 3 with three concurrent processes.

## 9. Local tests (`test_local.py`, CPU, ≤ 2 threads)

**Last run, on the final code (after the resume-safety fix): 37 passed, 0 failed, 2,342 s on a heavily loaded CPU** (`_test\test_local_result.json`; log `_test_run4.log`). The earlier run took 570 s. Every T2–T6 number is bitwise identical to that run, so the fix did not change the training or evaluation arithmetic.
- Tokenizers: the pilot `hf_bpe_P1_16000.json` and `sp_unigram_16000.model` (`research\pilot\`).
- Encoders: from `eval\harness.py` `build_adapter`.
- Data: a slice of 700 train and 80 dev documents, split into parts of 0.25 MB to force a multi-part archive.

| test | result |
|---|---|
| T1 split into 4 parts; reassembled; manifest sha256 = PARTS.json; all 16 files sha256-verified | pass |
| T1 a part with one flipped byte is rejected ("PART VERIFICATION FAILED … part001 sha256 mismatch") | pass |
| T1 dev bytes = UTF-8 bytes of the text; for both tokenizers `decode(bundle tokens) == text` and re-encoding is identical (80/80); harness encoding = native library encoding (80/80) | pass |
| T1 test-split guard: a test uid aborts; a file named `*test*` is never opened | pass |
| T1 `WAVES.json` parsing (nested list, relative path, `baseline: true`, `newline_wrapper: false`) | pass |
| T2 `screen`, 30 steps, both tokenizers: equal step counts (30; 814 at the 10 MB budget) | pass |
| T2 loss decreases: A1 9.645 → 7.796, SP-Unigram 9.622 → 7.889 (mean of first 5 / last 5 steps) | pass |
| T2 bpb equals Σbits/Σbytes recomputed from the per-document arrays and from the raw text bytes (1.707634417 / 1.689121804, exact) | pass |
| T2 tokens seen = steps·8·ctx and ctx = round(1536 / bytes per token) (259 and 251); curve points at 25/50/75/100 % | pass |
| T3 brute force, token by token: a short document (m = 57 ≤ ctx, no windows) matches to 5.4e-9 relative; a 3-window document (m = 430, ctx = 259) to 3.6e-9 | pass |
| T4 two identical CPU runs → identical losses and per-document bits | pass |
| T5 windows: every target scored once, ≥ ctx/2 context after the first window, same window start as the brute-force formula (m up to 4,097; ctx 4–259) | pass |
| T6 `run_all.py --smoke` end to end on CPU: parity reproduces `PARITY.json` bitwise; LR choice; Stage 3; Stage 4 with the power check triggering seeds 4 and 5; finalist re-sweep; large sweep + runs; summary; zip | pass |
| T6 the lrsweep and Stage 3 re-run of the same config are bitwise identical; a second invocation resumes with 0 new runs | pass |
| T7 `lr_tag` is lossless: 2,015 LRs, including adjacent doubles, give 2,015 distinct keys that parse back exactly; the grid tags are unchanged (`1e-3` `3e-3` `6e-3` `5e-4` `2e-3`) | pass |
| T7 the reviewer's case, on a copy of the T6 results (chosen LR 6e-3): `--stages screen --lr 5.8e-3` trains 6 new runs keyed `lr5.8e-3` and reuses none. `power_check_screen.json` is computed from the 5.8e-3 runs | pass |
| T7 refusals, all with no skip and nothing written: an lr 6e-3 record under the requested `lr2.5e-3` key ("lr stored 0.006, requested 0.0025"); a stored `budget_bytes` that differs; a smoke record offered to the full-budget run (steps 12 vs 814, overrides); a stored AMP dtype that differs (bf16 vs fp32), which `--allow-mixed` then reuses with a warning | pass |
| T7 stage means select records by their own `lr`: a planted `screen__…__lr6e-3__s9.json` holding lr 5.8e-3 is excluded (4 files match the old prefix, 3 records used) | pass |

**Extra checks run separately:**
- `_test\amp_path_check.py`: the GPU-only code paths on the CPU, bf16 autocast and fp16 autocast + GradScaler (unscale, clip, step), through the real `run_one`/`train_step` with the model shrunk to d=32. Both run and give finite losses and bpb. CPU bf16/fp16 is emulated and too slow at full size.
- `_test\bytes_seen_check.py`: the bytes-seen figures in §7.3.
- The full-size dry-run build (§2.1), reassembled from its two real parts and sha256-verified by `run_all.py --estimate`, which reassembles and verifies before estimating.

## 10. Known limitations

- GPU runtimes are FLOP-based estimates. `run_all.py --estimate` on the actual GPU is also an estimate. Real times appear in every record (`wall_time_s`).
- **GPU determinism depends on the kernels torch picks.** It is verified only on the GPU itself:
  - the start-up probe records strict or warn-only mode per run (`runtime.deterministic`);
  - the lrsweep/screen re-run check in `summary.txt` reports "bitwise identical", or the relative difference.

  If it is not bitwise, compare that difference with the seed s.d. before trusting sub-seed-noise gaps.
- The tests used the pilot tokenizers (`research\pilot`), trained on a split that is **not** group-disjoint, on slices of train/dev. They check the code, not any tokenizer. The real bundle must be built from `WAVES.json`.
- **Custom encoders** (e.g. PickyBPE) go through `harness.build_adapter` with `custom: "MODULE:FACTORY"`. That path is implemented but untested, because no custom finalist exists yet.
