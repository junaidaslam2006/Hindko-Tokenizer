# Track B: Hindko vocabulary extension of open LLM tokenizers (continued BPE)

Written 2026-09-26 20:32 by `make_trackb_md.py`; every number below is read from `results/`, `work/`, `deliver/` and `archive_v1.0/`, except the few marked as coming from the session log. PLAN.md section 9 (tokenizer side). **Nothing in this run was computed on the test split**: training used `train_D1` (permissive train), every measurement used `dev_strict` (validation) or the non-Hindko check set; `dev_permissive` (validation) supplied Urdu and English check documents only. Frozen inputs were verified before every step: split manifest `76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2`, `hp.normalize` 1.0.1 (`normalize()` re-applied to every text and asserted to be a no-op on the materialised views).

## 1. Result at the chosen k (pre-registered knee rule, `KNEE_RULE.md`)

| base (models) | license | k new | len(tokenizer) base → extended | dev_strict bytes/token base → ext. | tokens | fertility | STRR | train_D1 tokens base → ext. | new tokens < 100 train occ. | delivered folder |
|---|---|---:|---|---|---:|---|---|---|---:|---|
| **qwen-3** (Qwen3 (0.6B-235B), Qwen2/2.5 share the encodings) | apache-2.0 | 2,048 | 151,669 → 153,717 | 2.684 → **5.162** | **−48.0%** | 2.998 → 1.595 | 3.1% → 59.9% | 17.29M → 9.55M | 34 (1.7%) | `deliver/qwen3-hindko-cbpe2048/` |
| **llama-3** (Llama 3/3.1/3.2/3.3 (also Alif-1.0, Qalb-1.0)) | llama3.1 | 2,048 | 128,256 → 130,304 | 2.736 → **5.470** | **−50.0%** | 2.981 → 1.504 | 19.0% → 64.3% | 17.16M → 9.08M | 40 (2.0%) | `deliver/llama3-hindko-cbpe2048/` |
| **gemma-3** (Gemma 3 (1B-27B)) | gemma | 2,048 | 262,145 → 264,193 | 4.640 → **5.995** | **−22.6%** | 1.765 → 1.357 | 49.6% → 74.0% | 10.43M → 8.33M | 51 (2.5%) | `deliver/gemma3-hindko-cbpe2048/` |
| **qwen-3.5** (Qwen3.5 (0.8B-397B), Qwen3.8 shares the encodings) | apache-2.0 | 2,048 | 248,077 → 250,125 | 3.631 → **5.601** | **−35.2%** | 2.268 → 1.465 | 26.6% → 67.0% | 12.93M → 8.87M | 37 (1.8%) | `deliver/qwen3.5-hindko-cbpe2048/` |
| **gemma-4** (Gemma 4 (E2B-31B); same pieces and merges as Gemma 3, different control tokens) | apache-2.0 | 2,048 | 262,144 → 264,192 | 4.640 → **5.995** | **−22.6%** | 1.765 → 1.357 | 49.6% → 74.0% | 10.43M → 8.33M | 51 (2.5%) | `deliver/gemma4-hindko-cbpe2048/` |

- *tokens* = change in the number of tokens of dev_strict (1 − NSL against the unextended base). Bytes/token is Σ UTF-8 bytes ÷ Σ tokens of the 836 dev_strict documents; fertility and STRR as in PLAN 4.2 (in-context offsets, `eval/harness.py`).
- Over all 25 extended tokenizers (5 bases × the k grid): G1 lossless on dev_strict: all; G2, no unreachable new whole-character token: all; HF-native encoding = reference continued-BPE encoding on every dev document: all; no English / code / other-script check document changed: all (sections 5 and 6).
- Dev was used to choose k, so the dev gains at the chosen k are selection-biased by the choice among 5 grid points; the test split has not been evaluated (PLAN Stage 5 is a later run).

## 2. What was done, and deviations from PLAN 9

- **Bases** (tokenizer files already in `baselines/files/`, no download): Qwen3 and Llama-3.x (byte-level BPE) and Gemma-3 (SentencePiece BPE, used through its HF `tokenizer.json`), as PLAN 9.1 lists; added as 'other strong open bases': **Qwen3.5** (Apache-2.0, byte-level BPE, 248k) and **Gemma-4** (Apache-2.0; same pieces and merges as Gemma-3, different control tokens). Llama-3.x and Gemma-3 come from the byte-identical ungated mirrors recorded in `baselines/manifest.json`.
- **Method**: continued BPE (Purason et al. 2026, arXiv:2512.03989), **reimplemented from the paper**; their toolkit was not downloaded or run. Grid k ∈ {1,024, 2,048, 4,096, 8,192, 16,384} (PLAN's '1k…16k' read as powers of two, as PLAN 2.1 does for vocabulary sizes).
- **Training text**: `train_D1` (16,015 documents, 45.0 MB; permissive train split), `min_frequency` = 2 as in every BpeTrainer run of this project.
- **Deviation 1, scope filter (added):** merges are learned only from training units that contain an Arabic-script character (Unicode Script_Extensions = Arabic, or a code point in an Arabic block; section 3.1 item 5). The Hindko corpus also holds English, digits-only, whitespace and code-like units; learning merges from them could change how the base encodes English or code. Share of the base tokens of train_D1 that the filter left out: qwen-3 3.5%, llama-3 3.4%, gemma-3 4.2%, qwen-3.5 5.2%, gemma-4 4.2%. Its effect is measured, not assumed (section 6).
- **Deviation 2, Gemma (SentencePiece):** the continued merges are learned and appended on Gemma's HF `tokenizer.json`, which is a BPE merge-list model, exactly as for the byte-level bases; the training units follow Gemma's SentencePiece `trainer_spec` (section 3.2). SentencePiece's own BPE encoder has no merge list (it repeatedly merges the adjacent pair whose concatenation is a piece with the best score, whatever the split), so pair-specific appended merges are not guaranteed to be expressible in a `.model`. The analogous `.model` (the k new pieces appended as NORMAL pieces with scores below every base piece, in learning order; Gemma-3's HF-only `<image_soft_token>` inserted first so that ids line up) was built and measured: k = 0: 836/836 dev_strict documents identical to the tokenizer.json; k = 1,024: 836/836 dev_strict documents identical to the tokenizer.json; k = 2,048: 836/836 dev_strict documents identical to the tokenizer.json; k = 4,096: 836/836 dev_strict documents identical to the tokenizer.json; k = 8,192: 836/836 dev_strict documents identical to the tokenizer.json; k = 16,384: 836/836 dev_strict documents identical to the tokenizer.json (`results/gemma-3/sp_model_check.json`). At the delivered k it also agreed on every train_D1 document (16,015) and every check-set document, so the Gemma-3 folder includes it as `tokenizer.model` (measured agreement, not a proof for all inputs). Gemma-4's base folder has no `.model`, so none is built for it.
- **Deviation 3, token ids (implementation fact):** HF `tokenizers` re-derives the id of every added token that is absent from the model vocabulary as len(model vocab), len + 1, … when it loads a file. Appending model tokens after Qwen's/Llama's special-token block therefore silently moved every special token onto a new token's id (first attempt, caught by G1: 397/836 dev documents round-tripped; `<tool_call>` became id 152,681). The delivered files write the added tokens into the model vocabulary at their own ids (the layout Gemma's file already has); `materialize()` asserts that every special token keeps its id. (The two first-attempt figures are from the session log, not from a results file.)
- **Deviation 3b, tokens defined only in `tokenizer_config.json`:** qwen-3.5's config defines 7 added tokens that its tokenizer.json lacks (`<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`, `<tts_text_bos_single>`, `<|audio_pad|>`, ids 248,070–248,076). The new tokens therefore start after them, and the extended `tokenizer.json` carries them as added tokens at their config ids, so that `tokenizers` and `transformers` give the same ids (first found by the delivery check: `transformers` saw 7 extra tokens and encoded only 224/836 dev documents like `tokenizers`).
- **Deviation 4, gates:** G3 of PLAN 4.3 checks this project's own 64-token special block, which the bases do not have; it is replaced by 'every base special/added token keeps its id' (asserted). G4 is run as a rebuild from scratch (section 3.4). The PLAN 4.3 G2 remedy was not needed: no new token failed G2 (section 5).
- **Deviation 7, Qwen3.5 `tokenizer_config.json`:** the base config names `Qwen2Tokenizer`. In `transformers` 5.3.0 that class rebuilds the pre-tokenizer with the Qwen2 regex (combining marks split off) instead of the Qwen3.5 regex of the `tokenizer.json` (marks kept inside words), so even the unextended Qwen3.5 folder encodes only 650/836 dev_strict documents like its own `tokenizer.json`. The delivered config sets `tokenizer_class` to `PreTrainedTokenizerFast`, which loads `tokenizer.json` as it is: 836/836. This is the only change to a copied config file.
- The knee rule was written to `KNEE_RULE.md` (sha256 `1f0b3d3675f2c079…`) before any extended tokenizer was evaluated on dev.
- **Deviation 5, reading of the rule's non-interference condition:** the bulk English set (package READMEs, assembled together with the rule) turned out to contain a few documents with Arabic-script words (language names such as العربية in the transformers README). These change, like Urdu. The condition is applied as: no document of the curated English, code and other-script sets changes, and no bulk document *without* Arabic-script characters changes. The bulk sets are run at k = 16,384 only; no base chose k = 16,384 (section 7), so the reading changes no choice.
- **Deviation 6, method revised during the run (1.0.0 → 1.1.0), merge-level scope:** with only the unit-level filter, byte-level bases also learned merges *inside* Arabic-containing pretokens whose result contains no Arabic character: `’’` (Qwen3 rank 460; Hindko writes Urdu-style double quotes after `۔`), `)\n` (Qwen3.5 rank 4,639) and continuation-byte pairs such as `a2 bf`. The check set caught Qwen3.5 at k ≥ 8,192 (all 10 stdlib modules and 585 of 678 bulk .py files changed; `archive_v1.0/results/qwen-3.5/`), and an exhaustive single-code-point test (run on Qwen3 only before it was stopped; `archive_v1.0/results/audit.json`) then showed Qwen3 1.0.0 changing the encoding of 1,036 / 3,094 / 3,094 / 3,094 / 3,094 code points outside Script_Extensions=Arabic (k = 1,024 … 16,384; e.g. U+08E2, U+18BE, U+18BF, U+1DFA). Static audit of the 1.0.0 merge lists (`archive_v1.0/audit_static_v1.0.json`): qwen-3: 3 non-Arabic and 5 partial-UTF-8 new tokens among 16,384, first non-Arabic at rank 460; llama-3: 3 non-Arabic and 5 partial-UTF-8 new tokens among 16,384, first non-Arabic at rank 3,037; gemma-3: 0 non-Arabic and 0 partial-UTF-8 new tokens among 16,384, first non-Arabic at rank none; qwen-3.5: 3 non-Arabic and 5 partial-UTF-8 new tokens among 16,384, first non-Arabic at rank 3,042; gemma-4: 0 non-Arabic and 0 partial-UTF-8 new tokens among 16,384, first non-Arabic at rank none. Method 1.1.0 keeps a learned merge only if its token can occur only in text that contains an Arabic-script character (its bytes contain a complete Arabic-script character, or end in a byte prefix that only Arabic-script characters complete); a rejected merge is dropped with every later merge that needs it (counts in section 3.2). This is a post-filter on the unconstrained greedy run, not a constrained re-count: merges after a dropped one were chosen with counts in which the dropped merge had happened. **All numbers in this document are 1.1.0**; Gemma's lists contained no such merge and are byte-identical under both versions. 1.0.0 outputs are kept in `archive_v1.0/` (its delivered folders are superseded and must not be used).

## 3. Method

### 3.1 Continued BPE, and why the base behaviour is kept

1. Every training unit is encoded with the **base** tokenizer (its own normalizer, pre-tokenizer and merges).
2. New merges are learned greedily on those base-token sequences: the most frequent adjacent pair of tokens becomes a new token, the counts are updated, repeat until k new tokens exist.
3. The new merges are appended **after** all base merges. At encoding time BPE applies the lowest-ranked available merge first, so the base merges run to completion (the unchanged base encoding) before any new merge can fire, and the new merges then fire in the order they were learned. Consequences: a text in which no new merge fires is encoded exactly as by the base; otherwise every extended token is a concatenation of consecutive base tokens (checked: 'coarsening' in section 6).
4. **Code-point (PUA) trick** (SOTA_TOKENIZATION 4, route a): each base-token id that occurs in a unit is mapped to one Supplementary Private Use Area code point (U+F0000…); each unit becomes a string of these characters; the stock Rust `tokenizers.trainers.BpeTrainer` learns the merges on them (fed as whitespace-separated words with their counts); the PUA merges are translated back into base-token strings. Merges whose result string already exists reuse its id (counted as 'merge to existing': qwen-3 0, llama-3 0, gemma-3 0, qwen-3.5 4, gemma-4 0). Because the trainer is greedy, the k-token extension is the prefix of the 16,384-token one; this is checked by an independent training at k = 1,024 (section 3.4).
5. **Scope** (section 2, deviations 1 and 6): merges are learned only from units that contain an Arabic-script character, and a learned merge is kept only if its token can occur only in text containing an Arabic-script character. 'Arabic-script character' = Unicode Script_Extensions contains Arabic, or the code point lies in an Arabic block (this adds Arabic-only Common-script signs such as U+0605 and U+08E2; Syriac stays out).

### 3.2 Training units per base family

| base | kind | units | unit occurrences used (≥ 2 base tokens, Arabic script) | base tokens of train_D1 | distinct base tokens in the units (PUA alphabet) | PUA training time | PUA merges used for 16,384 new tokens | dropped: out of scope / needing a dropped token / too long |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen-3 | bytelevel | pretokens of the base pre-tokenizer | 5,232,421 | 17,290,332 | 2,051 | 16 s | 16,412 | 6 / 22 / 0 |
| llama-3 | bytelevel | pretokens of the base pre-tokenizer | 4,540,615 | 17,163,156 | 2,826 | 13 s | 16,418 | 6 / 28 / 0 |
| gemma-3 | spbpe | SentencePiece word units (below) | 2,638,096 | 10,432,840 | 5,039 | 11 s | 16,384 | 0 / 0 / 0 |
| qwen-3.5 | bytelevel | pretokens of the base pre-tokenizer | 3,858,893 | 12,930,717 | 4,098 | 13 s | 16,426 | 6 / 32 / 0 |
| gemma-4 | spbpe | SentencePiece word units (below) | 2,638,096 | 10,432,840 | 5,039 | 11 s | 16,384 | 0 / 0 / 0 |

- **Byte-level bases** (Qwen3, Qwen3.5, Llama-3): a unit is a pretoken of the base's own pre-tokenizer (normalizer + Split regex + ByteLevel), encoded by the base BPE model; merges can never cross a pretoken, as in the base. Checked on 300 train documents per base: concatenated per-pretoken encodings = the full-pipeline encoding (0 mismatches). Llama-3's `ignore_merges` (a pretoken that is a vocabulary entry is emitted whole) is kept and covered by the equivalence test.
- **Gemma (SentencePiece BPE, byte fallback)**: its `tokenizer.json` has no effective pre-tokenizer (the normalizer turns spaces into U+2581 before the no-op `Split(' ')`), so the whole document is one BPE word. Units are cut from the base encoding of each document following Gemma's `trainer_spec` (read from its `tokenizer.model`: `split_by_whitespace` false but the vocabulary has one piece with an inner U+2581, `split_by_unicode_script` true, `split_digits` true, `max_sentencepiece_length` default 16): a unit starts at every token beginning with U+2581; newline and whitespace-run pieces, `<0xNN>` byte-fallback pieces, added tokens and ASCII-digit pieces are never merged; a unit is cut where the Unicode script changes (Inherited marks take the preceding script); a merge whose piece would exceed 16 characters is dropped with every later merge that needs it (gemma-3: 0 dropped, gemma-4: 0 dropped).

### 3.3 Files

- New tokens are ordinary **model** tokens (vocabulary entry + merge), not added tokens: added tokens are matched greedily before BPE, which is the 'naive appending' that Purason et al. show creates unreachable tokens.
- New ids are contiguous from the first free id (table in section 1). Merges are written in the base file's own format (list of pairs or 'a b' strings). The k = 0 file written by the same code is byte-identical to the base file (sha256 equal), so the serialisation adds nothing.

### 3.4 Verification

- **Equivalence** (every k, every dev_strict document): the HF-native encoding equals a pure-Python reference: the base tokenizer's encoding followed by the extended merge table applied in tokenizers' own (rank, left position) order (`evaluate.py: Reference`).
- **G4 determinism** (rebuild from scratch, compare the ordered merge list and the sha256 of the k = 16,384 tokenizer.json): gemma-3 identical; gemma-4 identical; qwen-3.5 identical; qwen-3 identical; llama-3 identical.
- **Prefix property**: an independent PUA training for 1,024 new tokens on qwen-3 gives the same merges as the first 1,024 of the 16,384-token run (pass).
- **Gemma-3 vs Gemma-4**: independent builds give identical new-token strings in the same order; ids differ by [-1] (Gemma-3 has its extra `<image_soft_token>` at 262,144).

## 4. Curves

![Track B curves](figures/trackb_curves.png)

(`figures/trackb_curves.svg` is the vector version; x axis symlog in k.)

### Dev curves (dev_strict, 836 documents, 1,445,513 bytes)

**qwen-3** (Qwen3 (0.6B-235B), Qwen2/2.5 share the encodings)

| k | vocab | bytes/token | vs base | NSL | fertility | STRR | cont. words | new tokens on dev | train_D1 tokens | new < 20 | new < 100 | new = 0 | < 100 intermediate | partial UTF-8 | G1 | G2 new | equiv. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 151,669 | 2.684 | – | 1.000 | 2.998 | 3.1% | 96.9% | – | 17,290,332 | – | – | – | – | – | 836/836 | – | – |
| 1,024 | 152,693 | 4.649 | +73.2% | 0.577 | 1.774 | 51.2% | 48.8% | 97.2% (52.7% of dev tokens) | 10,467,042 | 2 (0.2%) | 10 (1.0%) | 1 | 10 | 0 | 836/836 | 0 fail / 1024 | 836/836 |
| 2,048 | 153,717 | 5.162 | +92.3% | 0.520 | 1.595 | 59.9% | 40.1% | 96.3% (62.2% of dev tokens) | 9,547,158 | 7 (0.3%) | 34 (1.7%) | 1 | 34 | 2 | 836/836 | 0 fail / 2046 | 836/836 |
| 4,096 | 155,765 | 5.707 | +112.6% | 0.470 | 1.440 | 68.6% | 31.4% | 93.3% (70.3% of dev tokens) | 8,737,862 | 27 (0.7%) | 151 (3.7%) | 2 | 151 | 2 | 836/836 | 0 fail / 4094 | 836/836 |
| 8,192 | 159,861 | 6.227 | +132.0% | 0.431 | 1.317 | 76.1% | 23.9% | 85.1% (76.8% of dev tokens) | 8,080,432 | 150 (1.8%) | 637 (7.8%) | 6 | 635 | 2 | 836/836 | 0 fail / 8190 | 836/836 |
| 16,384 | 168,053 | 6.631 | +147.1% | 0.405 | 1.235 | 81.5% | 18.5% | 67.4% (81.0% of dev tokens) | 7,602,587 | 582 (3.6%) | 9466 (57.8%) | 22 | 1805 | 2 | 836/836 | 0 fail / 16382 | 836/836 |

**llama-3** (Llama 3/3.1/3.2/3.3 (also Alif-1.0, Qalb-1.0))

| k | vocab | bytes/token | vs base | NSL | fertility | STRR | cont. words | new tokens on dev | train_D1 tokens | new < 20 | new < 100 | new = 0 | < 100 intermediate | partial UTF-8 | G1 | G2 new | equiv. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 128,256 | 2.736 | – | 1.000 | 2.981 | 19.0% | 81.0% | – | 17,163,156 | – | – | – | – | – | 836/836 | – | – |
| 1,024 | 129,280 | 5.065 | +85.1% | 0.540 | 1.626 | 57.6% | 42.4% | 95.6% (44.0% of dev tokens) | 9,726,456 | 2 (0.2%) | 7 (0.7%) | 0 | 7 | 1 | 836/836 | 0 fail / 1023 | 836/836 |
| 2,048 | 130,304 | 5.470 | +99.9% | 0.500 | 1.504 | 64.3% | 35.7% | 94.1% (50.6% of dev tokens) | 9,083,798 | 12 (0.6%) | 40 (2.0%) | 0 | 40 | 2 | 836/836 | 0 fail / 2046 | 836/836 |
| 4,096 | 132,352 | 5.905 | +115.8% | 0.463 | 1.391 | 71.1% | 28.9% | 90.9% (57.1% of dev tokens) | 8,478,222 | 33 (0.8%) | 165 (4.0%) | 2 | 165 | 2 | 836/836 | 0 fail / 4094 | 836/836 |
| 8,192 | 136,448 | 6.335 | +131.5% | 0.432 | 1.294 | 77.3% | 22.7% | 81.5% (62.9% of dev tokens) | 7,946,165 | 149 (1.8%) | 1523 (18.6%) | 3 | 590 | 2 | 836/836 | 0 fail / 8190 | 836/836 |
| 16,384 | 144,640 | 6.692 | +144.6% | 0.409 | 1.224 | 82.0% | 18.0% | 64.0% (67.3% of dev tokens) | 7,527,054 | 528 (3.2%) | 10253 (62.6%) | 12 | 1714 | 2 | 836/836 | 0 fail / 16382 | 836/836 |

**gemma-3** (Gemma 3 (1B-27B))

| k | vocab | bytes/token | vs base | NSL | fertility | STRR | cont. words | new tokens on dev | train_D1 tokens | new < 20 | new < 100 | new = 0 | < 100 intermediate | partial UTF-8 | G1 | G2 new | equiv. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 262,145 | 4.640 | – | 1.000 | 1.765 | 49.6% | 50.4% | – | 10,432,840 | – | – | – | – | – | 836/836 | – | – |
| 1,024 | 263,169 | 5.718 | +23.2% | 0.811 | 1.425 | 69.7% | 30.3% | 87.6% (19.2% of dev tokens) | 8,691,420 | 3 (0.3%) | 15 (1.5%) | 1 | 15 | 0 | 836/836 | 0 fail / 1024 | 836/836 |
| 2,048 | 264,193 | 5.995 | +29.2% | 0.774 | 1.357 | 74.0% | 26.0% | 87.1% (24.0% of dev tokens) | 8,327,229 | 8 (0.4%) | 51 (2.5%) | 1 | 51 | 0 | 836/836 | 0 fail / 2048 | 836/836 |
| 4,096 | 266,241 | 6.319 | +36.2% | 0.734 | 1.285 | 78.7% | 21.3% | 82.9% (29.2% of dev tokens) | 7,927,429 | 35 (0.9%) | 168 (4.1%) | 2 | 168 | 0 | 836/836 | 0 fail / 4096 | 836/836 |
| 8,192 | 270,337 | 6.644 | +43.2% | 0.698 | 1.221 | 83.3% | 16.7% | 72.9% (34.1% of dev tokens) | 7,523,392 | 146 (1.8%) | 2830 (34.5%) | 5 | 534 | 0 | 836/836 | 0 fail / 8192 | 836/836 |
| 16,384 | 278,529 | 6.961 | +50.0% | 0.667 | 1.163 | 87.4% | 12.6% | 56.7% (38.4% of dev tokens) | 7,157,865 | 498 (3.0%) | 11361 (69.3%) | 17 | 1463 | 0 | 836/836 | 0 fail / 16384 | 836/836 |

**qwen-3.5** (Qwen3.5 (0.8B-397B), Qwen3.8 shares the encodings)

| k | vocab | bytes/token | vs base | NSL | fertility | STRR | cont. words | new tokens on dev | train_D1 tokens | new < 20 | new < 100 | new = 0 | < 100 intermediate | partial UTF-8 | G1 | G2 new | equiv. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 248,070 | 3.631 | – | 1.000 | 2.268 | 26.6% | 73.4% | – | 12,930,717 | – | – | – | – | – | 836/836 | – | – |
| 1,024 | 249,101 | 5.201 | +43.2% | 0.698 | 1.581 | 60.4% | 39.6% | 93.4% (34.4% of dev tokens) | 9,498,590 | 2 (0.2%) | 8 (0.8%) | 1 | 8 | 2 | 836/836 | 0 fail / 1022 | 836/836 |
| 2,048 | 250,125 | 5.601 | +54.3% | 0.648 | 1.465 | 67.0% | 33.0% | 92.3% (41.7% of dev tokens) | 8,867,566 | 8 (0.4%) | 37 (1.8%) | 1 | 37 | 2 | 836/836 | 0 fail / 2046 | 836/836 |
| 4,096 | 252,173 | 6.046 | +66.5% | 0.601 | 1.355 | 73.8% | 26.2% | 89.0% (49.1% of dev tokens) | 8,262,565 | 32 (0.8%) | 146 (3.6%) | 2 | 146 | 2 | 836/836 | 0 fail / 4094 | 836/836 |
| 8,192 | 256,269 | 6.480 | +78.5% | 0.560 | 1.262 | 80.0% | 20.0% | 79.6% (55.6% of dev tokens) | 7,728,668 | 137 (1.7%) | 1457 (17.8%) | 5 | 543 | 2 | 836/836 | 0 fail / 8190 | 836/836 |
| 16,384 | 264,461 | 6.842 | +88.5% | 0.531 | 1.194 | 84.8% | 15.2% | 62.4% (60.3% of dev tokens) | 7,300,143 | 485 (3.0%) | 10137 (61.9%) | 15 | 1569 | 2 | 836/836 | 0 fail / 16382 | 836/836 |

**gemma-4** (Gemma 4 (E2B-31B); same pieces and merges as Gemma 3, different control tokens)

| k | vocab | bytes/token | vs base | NSL | fertility | STRR | cont. words | new tokens on dev | train_D1 tokens | new < 20 | new < 100 | new = 0 | < 100 intermediate | partial UTF-8 | G1 | G2 new | equiv. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 262,144 | 4.640 | – | 1.000 | 1.765 | 49.6% | 50.4% | – | 10,432,840 | – | – | – | – | – | 836/836 | – | – |
| 1,024 | 263,168 | 5.718 | +23.2% | 0.811 | 1.425 | 69.7% | 30.3% | 87.6% (19.2% of dev tokens) | 8,691,420 | 3 (0.3%) | 15 (1.5%) | 1 | 15 | 0 | 836/836 | 0 fail / 1024 | 836/836 |
| 2,048 | 264,192 | 5.995 | +29.2% | 0.774 | 1.357 | 74.0% | 26.0% | 87.1% (24.0% of dev tokens) | 8,327,229 | 8 (0.4%) | 51 (2.5%) | 1 | 51 | 0 | 836/836 | 0 fail / 2048 | 836/836 |
| 4,096 | 266,240 | 6.319 | +36.2% | 0.734 | 1.285 | 78.7% | 21.3% | 82.9% (29.2% of dev tokens) | 7,927,429 | 35 (0.9%) | 168 (4.1%) | 2 | 168 | 0 | 836/836 | 0 fail / 4096 | 836/836 |
| 8,192 | 270,336 | 6.644 | +43.2% | 0.698 | 1.221 | 83.3% | 16.7% | 72.9% (34.1% of dev tokens) | 7,523,392 | 146 (1.8%) | 2830 (34.5%) | 5 | 534 | 0 | 836/836 | 0 fail / 8192 | 836/836 |
| 16,384 | 278,528 | 6.961 | +50.0% | 0.667 | 1.163 | 87.4% | 12.6% | 56.7% (38.4% of dev tokens) | 7,157,865 | 498 (3.0%) | 11361 (69.3%) | 17 | 1463 | 0 | 836/836 | 0 fail / 16384 | 836/836 |

### Per source at the chosen k (bytes/token; base → extended)

| base | k | book | newspaper | web |
|---|---:|---|---|---|
| qwen-3 | 2,048 | 2.656 → 5.070 (+90.9%) | 2.739 → 5.346 (+95.1%) | 2.568 → 4.814 (+87.5%) |
| llama-3 | 2,048 | 2.670 → 5.345 (+100.2%) | 2.872 → 5.722 (+99.3%) | 2.494 → 5.024 (+101.5%) |
| gemma-3 | 2,048 | 4.520 → 5.803 (+28.4%) | 4.879 → 6.392 (+31.0%) | 4.284 → 5.346 (+24.8%) |
| qwen-3.5 | 2,048 | 3.561 → 5.471 (+53.6%) | 3.768 → 5.870 (+55.8%) | 3.410 → 5.073 (+48.8%) |
| gemma-4 | 2,048 | 4.520 → 5.803 (+28.4%) | 4.879 → 6.392 (+31.0%) | 4.284 → 5.346 (+24.8%) |

### Non-interference (documents whose ids differ from the base; bulk sets tested at k = 16,384)

| base | k | English (12 gen.) | English in corpus dev (7) | Python stdlib (10) | other scripts (7) | bulk stdlib .py (678) | bulk English READMEs (228) | Urdu generated (9) | Urdu news dev (22) | Urdu books dev (358) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| qwen-3 | 1,024 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 40.9% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 42.4% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 37.6% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3 | 2,048 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 46.8% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 48.4% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 42.0% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3 | 4,096 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 54.7% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 53.4% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 46.2% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3 | 8,192 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 59.4% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 57.1% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 49.5% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3 | 16,384 | 0 | 0 | 0 | 0 | 0 | 0 of 224 without Arabic script (4 of 4 with it) | 9 changed, 61.4% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 59.5% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 52.3% fewer tokens, coarsening 358/358, round trip 358/358 |
| llama-3 | 1,024 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 43.2% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 44.7% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 37.8% fewer tokens, coarsening 358/358, round trip 358/358 |
| llama-3 | 2,048 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 48.8% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 49.2% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 41.3% fewer tokens, coarsening 358/358, round trip 358/358 |
| llama-3 | 4,096 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 53.8% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 52.9% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 44.2% fewer tokens, coarsening 358/358, round trip 358/358 |
| llama-3 | 8,192 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 57.5% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 55.7% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 47.2% fewer tokens, coarsening 358/358, round trip 358/358 |
| llama-3 | 16,384 | 0 | 0 | 0 | 0 | 0 | 0 of 224 without Arabic script (3 of 4 with it) | 9 changed, 60.0% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 58.1% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 50.2% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-3 | 1,024 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 6 changed, 3.7% fewer tokens, coarsening 6/6, round trip 6/6 | 21 changed, 7.8% fewer tokens, coarsening 21/21, round trip 21/21 | 358 changed, 5.1% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-3 | 2,048 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 8 changed, 7.7% fewer tokens, coarsening 8/8, round trip 8/8 | 22 changed, 11.4% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 7.9% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-3 | 4,096 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 11.0% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 15.0% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 11.6% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-3 | 8,192 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 14.5% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 19.0% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 15.0% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-3 | 16,384 | 0 | 0 | 0 | 0 | 0 | 0 of 224 without Arabic script (3 of 4 with it) | 9 changed, 18.2% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 22.2% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 19.2% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3.5 | 1,024 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 14.2% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 20.7% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 19.3% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3.5 | 2,048 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 22.0% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 26.5% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 23.4% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3.5 | 4,096 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 28.6% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 31.6% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 27.3% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3.5 | 8,192 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 33.8% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 35.7% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 31.2% fewer tokens, coarsening 358/358, round trip 358/358 |
| qwen-3.5 | 16,384 | 0 | 0 | 0 | 0 | 0 | 0 of 224 without Arabic script (3 of 4 with it) | 9 changed, 36.5% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 39.1% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 34.3% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-4 | 1,024 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 6 changed, 3.7% fewer tokens, coarsening 6/6, round trip 6/6 | 21 changed, 7.8% fewer tokens, coarsening 21/21, round trip 21/21 | 358 changed, 5.1% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-4 | 2,048 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 8 changed, 7.7% fewer tokens, coarsening 8/8, round trip 8/8 | 22 changed, 11.4% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 7.9% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-4 | 4,096 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 11.0% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 15.0% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 11.6% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-4 | 8,192 | 0 | 0 | 0 | 0 | (= 16k) | (= 16k) | 9 changed, 14.5% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 19.0% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 15.0% fewer tokens, coarsening 358/358, round trip 358/358 |
| gemma-4 | 16,384 | 0 | 0 | 0 | 0 | 0 | 0 of 224 without Arabic script (3 of 4 with it) | 9 changed, 18.2% fewer tokens, coarsening 9/9, round trip 9/9 | 22 changed, 22.2% fewer tokens, coarsening 22/22, round trip 22/22 | 358 changed, 19.2% fewer tokens, coarsening 358/358, round trip 358/358 |

### Audit (`audit.py`): scope of every new merge, and every Unicode code point

Static: new tokens by class of their bytes (*Arabic* = contains a complete Arabic-script character or ends in a byte prefix that only Arabic-script characters complete; *other* = can occur in text without Arabic script). Code points: all 1,112,064 Unicode scalar values, alone and after a space (2,224,128 strings), encoded by base and extension; strings whose ids change.

| base | k | new tokens: Arabic / partial-UTF-8 with Arabic-only prefix / other partial-UTF-8 / other | code points changed: Arabic-script / other |
|---|---:|---|---|
| qwen-3 | 1,024 | 1024 / 0 / 0 / 0 | 37 / **0** |
| qwen-3 | 2,048 | 2046 / 2 / 0 / 0 | 171 / **0** |
| qwen-3 | 4,096 | 4094 / 2 / 0 / 0 | 187 / **0** |
| qwen-3 | 8,192 | 8190 / 2 / 0 / 0 | 196 / **0** |
| qwen-3 | 16,384 | 16382 / 2 / 0 / 0 | 200 / **0** |
| llama-3 | 1,024 | 1023 / 1 / 0 / 0 | 88 / **0** |
| llama-3 | 2,048 | 2046 / 2 / 0 / 0 | 153 / **0** |
| llama-3 | 4,096 | 4094 / 2 / 0 / 0 | 165 / **0** |
| llama-3 | 8,192 | 8190 / 2 / 0 / 0 | 176 / **0** |
| llama-3 | 16,384 | 16382 / 2 / 0 / 0 | 181 / **0** |
| gemma-3 | 1,024 | 1024 / 0 / 0 / 0 | 5 / **0** |
| gemma-3 | 2,048 | 2048 / 0 / 0 / 0 | 10 / **0** |
| gemma-3 | 4,096 | 4096 / 0 / 0 / 0 | 11 / **0** |
| gemma-3 | 8,192 | 8192 / 0 / 0 / 0 | 15 / **0** |
| gemma-3 | 16,384 | 16384 / 0 / 0 / 0 | 21 / **0** |
| qwen-3.5 | 1,024 | 1022 / 2 / 0 / 0 | 161 / **0** |
| qwen-3.5 | 2,048 | 2046 / 2 / 0 / 0 | 168 / **0** |
| qwen-3.5 | 4,096 | 4094 / 2 / 0 / 0 | 183 / **0** |
| qwen-3.5 | 8,192 | 8190 / 2 / 0 / 0 | 193 / **0** |
| qwen-3.5 | 16,384 | 16382 / 2 / 0 / 0 | 196 / **0** |
| gemma-4 | 1,024 | 1024 / 0 / 0 / 0 | 5 / **0** |
| gemma-4 | 2,048 | 2048 / 0 / 0 / 0 | 10 / **0** |
| gemma-4 | 4,096 | 4096 / 0 / 0 / 0 | 11 / **0** |
| gemma-4 | 8,192 | 8192 / 0 / 0 / 0 | 15 / **0** |
| gemma-4 | 16,384 | 16384 / 0 / 0 / 0 | 21 / **0** |

### Knee (KNEE_RULE.md)

| base | Kneedle y − x at 1k / 2k / 4k / 8k / 16k | Kneedle knee | largest eligible k with ≤ 10% new tokens < 100 | chosen k |
|---|---|---:|---:|---:|
| qwen-3 | 0.648 / 0.681 / 0.640 / 0.456 / 0.000 | 2,048 | 8,192 | **2,048** |
| llama-3 | 0.715 / 0.720 / 0.658 / 0.461 / 0.000 | 2,048 | 4,096 | **2,048** |
| gemma-3 | 0.503 / 0.553 / 0.547 / 0.405 / 0.000 | 2,048 | 4,096 | **2,048** |
| qwen-3.5 | 0.581 / 0.624 / 0.601 / 0.437 / 0.000 | 2,048 | 4,096 | **2,048** |
| gemma-4 | 0.503 / 0.553 / 0.547 / 0.405 / 0.000 | 2,048 | 4,096 | **2,048** |


## 5. Gates and the support profile of the new tokens

| base | k | G1 dev_strict | G2: new whole-character tokens failing self-tokenization | partial-UTF-8 new tokens (exempt, R2) | equivalence | base vocabulary's own G2 failures (k = 0) → at k = 16,384 |
|---|---:|---|---|---:|---|---|
| qwen-3 | 1,024 | 836/836 | 0 of 1,024 | 0 | 836/836 |  |
| qwen-3 | 2,048 | 836/836 | 0 of 2,046 | 2 | 836/836 |  |
| qwen-3 | 4,096 | 836/836 | 0 of 4,094 | 2 | 836/836 |  |
| qwen-3 | 8,192 | 836/836 | 0 of 8,190 | 2 | 836/836 |  |
| qwen-3 | 16,384 | 836/836 | 0 of 16,382 | 2 | 836/836 | 1,909 → 1,909 |
| llama-3 | 1,024 | 836/836 | 0 of 1,023 | 1 | 836/836 |  |
| llama-3 | 2,048 | 836/836 | 0 of 2,046 | 2 | 836/836 |  |
| llama-3 | 4,096 | 836/836 | 0 of 4,094 | 2 | 836/836 |  |
| llama-3 | 8,192 | 836/836 | 0 of 8,190 | 2 | 836/836 |  |
| llama-3 | 16,384 | 836/836 | 0 of 16,382 | 2 | 836/836 | 0 → 0 |
| gemma-3 | 1,024 | 836/836 | 0 of 1,024 | 0 | 836/836 |  |
| gemma-3 | 2,048 | 836/836 | 0 of 2,048 | 0 | 836/836 |  |
| gemma-3 | 4,096 | 836/836 | 0 of 4,096 | 0 | 836/836 |  |
| gemma-3 | 8,192 | 836/836 | 0 of 8,192 | 0 | 836/836 |  |
| gemma-3 | 16,384 | 836/836 | 0 of 16,384 | 0 | 836/836 | 4 → 4 |
| qwen-3.5 | 1,024 | 836/836 | 0 of 1,022 | 2 | 836/836 |  |
| qwen-3.5 | 2,048 | 836/836 | 0 of 2,046 | 2 | 836/836 |  |
| qwen-3.5 | 4,096 | 836/836 | 0 of 4,094 | 2 | 836/836 |  |
| qwen-3.5 | 8,192 | 836/836 | 0 of 8,190 | 2 | 836/836 |  |
| qwen-3.5 | 16,384 | 836/836 | 0 of 16,382 | 2 | 836/836 | 203 → 199 |
| gemma-4 | 1,024 | 836/836 | 0 of 1,024 | 0 | 836/836 |  |
| gemma-4 | 2,048 | 836/836 | 0 of 2,048 | 0 | 836/836 |  |
| gemma-4 | 4,096 | 836/836 | 0 of 4,096 | 0 | 836/836 |  |
| gemma-4 | 8,192 | 836/836 | 0 of 8,192 | 0 | 836/836 |  |
| gemma-4 | 16,384 | 836/836 | 0 of 16,384 | 0 | 836/836 | 6,302 → 6,302 |

- qwen-3.5: the base's own G2 failures go from 203 to 199 at k = 16,384. Its extension contains 4 merges whose result is an existing base token (section 3.1 item 4), which makes base tokens that the base alone never produces reachable; these merges are in scope (they fire only in Arabic-script text).
- The base vocabularies' own G2 failures (examples: qwen-3: `１０`, `２０`, `珊󠄁`; gemma-3: ` YYYY`, ` yyyy`, ` diffformul`; qwen-3.5: `１０`, `２０`, `俱乐部`; gemma-4: `[multimodal]`, `<unused0>`, `<unused1>`) belong to the base (tokens its own pre-tokenizer or merge order never produces) and are listed only to show that the extension adds none. G2 of the new tokens tests every new token with valid UTF-8 text, including those with an inner newline such as `۔\n`, which the harness's whole-vocabulary G2 skips as 'superword'.
- **Support (R1)** is counted by encoding train_D1 with the extended tokenizer itself. New tokens with < 100 occurrences, split into intermediate merge nodes (later absorbed into longer new tokens, like the pilot BPE's rare tokens in PLAN 4.3) and leaves (tokens no later merge uses; at large k these are the last, rarest merges):

| base | k | < 100 | of which intermediate | of which leaves | never used (= 0) | of which intermediate |
|---|---:|---:|---:|---:|---:|---:|
| qwen-3 | 1,024 | 10 | 10 | 0 | 1 | 1 |
| qwen-3 | 2,048 | 34 | 34 | 0 | 1 | 1 |
| qwen-3 | 4,096 | 151 | 151 | 0 | 2 | 2 |
| qwen-3 | 8,192 | 637 | 635 | 2 | 6 | 6 |
| qwen-3 | 16,384 | 9466 | 1805 | 7661 | 22 | 22 |
| llama-3 | 1,024 | 7 | 7 | 0 | 0 | 0 |
| llama-3 | 2,048 | 40 | 40 | 0 | 0 | 0 |
| llama-3 | 4,096 | 165 | 165 | 0 | 2 | 2 |
| llama-3 | 8,192 | 1523 | 590 | 933 | 3 | 3 |
| llama-3 | 16,384 | 10253 | 1714 | 8539 | 12 | 12 |
| gemma-3 | 1,024 | 15 | 15 | 0 | 1 | 1 |
| gemma-3 | 2,048 | 51 | 51 | 0 | 1 | 1 |
| gemma-3 | 4,096 | 168 | 168 | 0 | 2 | 2 |
| gemma-3 | 8,192 | 2830 | 534 | 2296 | 5 | 5 |
| gemma-3 | 16,384 | 11361 | 1463 | 9898 | 17 | 17 |
| qwen-3.5 | 1,024 | 8 | 8 | 0 | 1 | 1 |
| qwen-3.5 | 2,048 | 37 | 37 | 0 | 1 | 1 |
| qwen-3.5 | 4,096 | 146 | 146 | 0 | 2 | 2 |
| qwen-3.5 | 8,192 | 1457 | 543 | 914 | 5 | 5 |
| qwen-3.5 | 16,384 | 10137 | 1569 | 8568 | 15 | 15 |
| gemma-4 | 1,024 | 15 | 15 | 0 | 1 | 1 |
| gemma-4 | 2,048 | 51 | 51 | 0 | 1 | 1 |
| gemma-4 | 4,096 | 168 | 168 | 0 | 2 | 2 |
| gemma-4 | 8,192 | 2830 | 534 | 2296 | 5 | 5 |
| gemma-4 | 16,384 | 11361 | 1463 | 9898 | 17 | 17 |

  Rarest new tokens at the chosen k (train_D1 occurrences):

  - qwen-3 k = 2,048: `ندھارا` 0, `اکٹر` 2, `␣ڈیس` 6, `تمل` 6, `␣تروی` 7, `␣مضم` 9, `عث` 14, `␣سلسل` 25
  - llama-3 k = 2,048: `ریر` 3, `ندھارا` 4, `سطے` 5, `␣ذریع` 6, `اکٹر` 7, `عقاد` 8, `رویج` 9, `یسک` 11
  - gemma-3 k = 2,048: `ندھارا` 0, `␣تروی` 7, `␣تریم` 10, `زنوی` 12, `یسک` 15, `پروف` 16, `ستاد` 18, `قاد` 18
  - qwen-3.5 k = 2,048: `ندھارا` 0, `اکٹر` 3, `سطے` 3, `تمل` 6, `␣ذریع` 9, `رویج` 9, `یسک` 14, `کردگی` 17
  - gemma-4 k = 2,048: `ندھارا` 0, `␣تروی` 7, `␣تریم` 10, `زنوی` 12, `یسک` 15, `پروف` 16, `ستاد` 18, `قاد` 18

## 6. Is the base encoding of other text unchanged?

Check set (`checkset/make_checkset.py`; nothing downloaded; all text through `hp.normalize`):

| set | documents | bytes | source |
|---|---:|---:|---|
| english_wiki | 12 | 4,316 | 12 Wikipedia-style English paragraphs written for this check |
| python_code | 10 | 107,752 | 10 whole CPython 3.11 stdlib modules (json/encoder, json/decoder, textwrap, heapq, bisect, fnmatch, colorsys, shlex, string, contextlib) |
| other_scripts | 7 | 2,080 | generated Hindi (Devanagari), Russian, Greek, Chinese, Japanese, Korean, symbols/emoji |
| urdu_generated | 9 | 3,488 | 9 Urdu news paragraphs written for this check |
| urdu_news_dev | 22 | 40,268 | Urdu-labelled newspaper/web documents of dev_permissive (validation) |
| urdu_book_dev | 358 | 811,834 | Urdu-labelled book documents of dev_permissive (validation) |
| english_dev | 7 | 3,079 | English-labelled documents of dev_permissive (validation) |
| bulk_code | 678 | 11,772,501 | every .py of the local CPython 3.11 stdlib outside test/, tests/, idlelib/ |
| bulk_english | 228 | 2,808,636 | every `*.dist-info/METADATA` (package READMEs) in the local site-packages |

The table 'Non-interference' in section 4 gives, per base and k, the number of documents whose ids differ from the base's. Bulk sets are tested at k = 16,384 only: if no new merge fires there, none fires at a smaller k (the smaller extensions are prefixes). For changed documents two properties are checked: the extended tokens are concatenations of consecutive base tokens (**coarsening**) and the round trip is exact.

**Reading:** no document without Arabic-script characters changed, for any base at any k (English, Python code, other scripts, bulk code and READMEs). The exhaustive code-point audit (every Unicode scalar value alone and after a space) found no non-Arabic code point whose encoding changes, for any base at any k. What this establishes: a new merge can only fire in text that contains an Arabic-script character (by construction, section 3.1 item 5), and the tests found no exception. What it does not establish: that text *with* Arabic script keeps its encoding. **Urdu is not unchanged, by design**: it shares the script and much of the vocabulary with Hindko, so Urdu text gets fewer tokens (section 4), and a model extended this way also sees new tokens in Urdu, and presumably in other Arabic-script languages (Persian, Pashto, Shahmukhi Punjabi, Arabic; not measured). Any text that merely contains one Arabic-script word (a language name in an English README) changes around that word.

## 7. Choice of k

Rule (`KNEE_RULE.md`, written before the dev evaluation): eligible = G1, G2, equivalence and non-interference pass; support cap = at most 10% of the k new tokens with < 100 train_D1 occurrences; compression knee = Kneedle on the linear k axis over the relative dev_strict token reduction; chosen k = min(knee, largest eligible k within the cap). Values: table 'Knee' in section 4.

- qwen-3: Kneedle knee 2,048; largest eligible k within the cap 8,192; **chosen 2,048**.
- llama-3: Kneedle knee 2,048; largest eligible k within the cap 4,096; **chosen 2,048**.
- gemma-3: Kneedle knee 2,048; largest eligible k within the cap 4,096; **chosen 2,048**.
- qwen-3.5: Kneedle knee 2,048; largest eligible k within the cap 4,096; **chosen 2,048**.
- gemma-4: Kneedle knee 2,048; largest eligible k within the cap 4,096; **chosen 2,048**.

The curves keep rising after the knee; a user who values compression more than embedding-row count or token support can take any k from `sweep/<base>/k<k>/tokenizer.json` (sha256 in `sweep/<base>/sweep_manifest.json`), all of which passed the same gates.

## 8. Deliverables

| folder | base | k | tokenizer.json sha256 | loads in `tokenizers` and `transformers` 5.3.0 | ids identical between them on dev_strict | same check for the unextended base folder | special ids kept | chat template renders |
|---|---|---:|---|---|---|---|---|---|
| `deliver/qwen3-hindko-cbpe2048/` | qwen-3 | 2,048 | `d118f96c28788bf1…` | yes (Qwen2Tokenizer, len 153,717) | 836/836 | 836/836 (Qwen2Tokenizer) | yes (26 checked) | yes |
| `deliver/llama3-hindko-cbpe2048/` | llama-3 | 2,048 | `d6b8e4ee157c1c72…` | yes (TokenizersBackend, len 130,304) | 836/836 | 836/836 (TokenizersBackend) | yes (256 checked) | yes |
| `deliver/gemma3-hindko-cbpe2048/` | gemma-3 | 2,048 | `a41e84bdf79fe04c…` | yes (GemmaTokenizer, len 264,193) | 836/836 | 836/836 (GemmaTokenizer) | yes (6415 checked) | yes |
| `deliver/qwen3.5-hindko-cbpe2048/` | qwen-3.5 | 2,048 | `6837822f7cab959f…` | yes (TokenizersBackend, len 250,125) | 836/836 | 650/836 (Qwen2Tokenizer) | yes (33 checked) | yes |
| `deliver/gemma4-hindko-cbpe2048/` | gemma-4 | 2,048 | `a9d17bcea7befa1f…` | yes (GemmaTokenizer, len 264,192) | 836/836 | 836/836 (GemmaTokenizer) | yes (24 checked) | no template in base |

Each folder: `tokenizer.json`; `tokenizer_config.json` (and `special_tokens_map.json` / `added_tokens.json` where the base has them) copied byte-identically from the base (except Qwen3.5's `tokenizer_class`, deviation 7), because the new tokens are model tokens and need no config entry; `new_tokens.jsonl` (id, vocabulary string, text, the merge, the base-token decomposition and byte lengths used by the initialisation, train/dev frequency, leaf flag); `init_embeddings.py`; `EXTENSION.json` (provenance, sha256 of every file, metrics and gates at this k); `README.md`. The Gemma-3 folder also has the extended SentencePiece `tokenizer.model` if it passed the agreement check (section 2, deviation 2); the base's own `.model` is never copied, because it lacks the new pieces. Only `transformers` 5.3.0 was available to test loading; older versions (4.x `PreTrainedTokenizerFast`) read the same `tokenizer.json` but were not tested. The Qwen3 and Gemma merge lists use the list-of-pairs format of their base files, which needs `tokenizers` ≥ 0.20 (as the bases do).

## 9. Embedding initialisation for continued pretraining (PLAN 9.3; for whoever runs CPT on a GPU)

Nothing of this section was run on a real checkpoint here (no GPU; no model weights downloaded).

1. **Resize** the embedding matrix (and the untied LM head) to at least `len(tokenizer)` rows, padded to a multiple of 64. Qwen3 checkpoints already have 151,936 rows for 151,669 ids: the first 267 new ids fall into these untrained padding rows, which must be initialised like every other new row. Llama-3.1 has 128,256 rows. Gemma-3 1B has 262,144 rows, so its tokenizer's `<image_soft_token>` (id 262,144) lies outside the matrix and is created by the resize (it is not a text token; new Gemma-3 ids start at 262,145); Gemma-3 4B+ has 262,208 rows. The row counts of Qwen3.5 and Gemma-4 checkpoints were not checked here (no model config downloaded); new Qwen3.5 ids start at 248,077, after the 7 config-only audio/TTS tokens (section 2, deviation 3b).
2. **Input rows, uniform subword mean + script-norm calibration** (Joshi et al. 2026): e_in(t) = mean of the input rows of D(t), the base tokens that formed t in training (`base_ids` in `new_tokens.jsonl`), then rescaled to ν = mean norm of the input rows of all base tokens occurring in any D(t) (the Arabic-script base tokens Hindko is written with). The exact calibration target of Joshi et al. is this project's reading of the paper; `--no-norm-calibration` switches it off.
3. **Output rows, character-length-weighted subword mean** (untied models only): e_out(t) = Σ w_b E_out[b] / Σ w_b over b ∈ D(t), w_b = UTF-8 byte length of b (∝ characters for Arabic script). Tied models (all Gemma; Qwen3 ≤ 4B; Llama-3.2 1B/3B) have one matrix: only step 2 applies, unless the matrix is untied for CPT.
4. **Choose the initialisation with 50-step CPT probes**, not with initialisation loss or bpb (Joshi et al. 2026): same data order and learning rate, validation loss after 50 steps on held-out Hindko (dev_strict; never the test split). Alternatives to include: mean of all embeddings (+ small noise), FOCUS (Dobler & de Melo 2023; needs only a fastText model trained on train_D1), Token Distillation (Dobler et al. 2026).
5. **Schedule** (Yamaguchi et al. 2026): first train the embedding and LM-head rows together with the top-2 and bottom-2 transformer layers ('2x2 LS'), then full CPT. Watch the norms of the new rows (PLAN 4.3 R3: the 50 lowest-norm output rows and their train frequencies) to find under-trained new tokens.
6. **Data volume**: train_D1 in tokens for each base and k is in section 4 (column train_D1 tokens); e.g. at the chosen k the extended tokenizers need the token counts in section 1.

Reference implementation: `init_embeddings.py` (`--model`, `--tokenizer <delivered folder>`, `--out`). 
Unit test `test_init_embeddings.py` on tiny random-weight models with the real embedding row counts (Qwen3 151,936; Llama-3.1 128,256; Gemma-3 1B and Gemma-4 262,144; Qwen3.5 assumed = its tokenizer length rounded up to 64, 248,128; Qwen3.5 tested with the Qwen3 architecture class): **all pass** (8 runs: qwen3-hindko-cbpe2048 Qwen3ForCausalLM tied, qwen3-hindko-cbpe2048 Qwen3ForCausalLM untied, llama3-hindko-cbpe2048 LlamaForCausalLM tied, llama3-hindko-cbpe2048 LlamaForCausalLM untied, gemma3-hindko-cbpe2048 Gemma3ForCausalLM tied, qwen3.5-hindko-cbpe2048 Qwen3ForCausalLM tied, qwen3.5-hindko-cbpe2048 Qwen3ForCausalLM untied, gemma4-hindko-cbpe2048 Gemma3ForCausalLM tied). It checks resizing, bit-identical base rows, both formulas against an independent NumPy recomputation, tied rows, and a forward pass on a Hindko sentence that uses new ids. It says nothing about the quality of the initialisation.

## 10. Claims

**Allowed** (measured here; fill in the base and its numbers from section 1):

- "On the Hindko strict validation split (836 documents, 1.45 MB), the <base> tokenizer extended with k tokens by continued BPE uses x% fewer tokens than the unextended <base> tokenizer (bytes/token a → b), is lossless on every document, and none of its k new whole-character tokens is unreachable by self-tokenization." (Validation-split numbers; k was chosen on this split.)
- "y (z%) of the new tokens occur fewer than 100 times in the Hindko training split; w never occur (all of them intermediate merge nodes)." Say this instead of 'all well-supported': PLAN 9.4's phrase 'well-supported' holds only in this quantified sense.
- "On the tested English, Python-code and other-script texts (N documents, M MB), the extended tokenizer returns exactly the base tokenizer's ids."
- "Urdu text also gets fewer tokens (u% on Urdu news of the corpus's validation split); every extended token is a concatenation of consecutive base tokens."
- "The files load with stock `tokenizers` and `transformers` (5.3.0 tested) and encode identically in both."

**Not allowed:**

- Anything about model quality (perplexity, bpb of a CPT model, downstream tasks, chat quality): no model was trained or initialised; PLAN 9.4.
- 'The base behaviour on all non-Hindko text is preserved': false for Urdu (measured) and unmeasured for other Arabic-script languages; true only for text in which no new merge fires, and measured only on the check set.
- Test-split numbers of any kind (not computed), or 'best k' beyond the stated rule.
- 'Equivalent to Purason et al.'s toolkit': our reimplementation was not compared with it (not run).
- For Gemma-3: that the extended SentencePiece `tokenizer.model` encodes like `tokenizer.json` on all inputs; only the measured agreement (section 2) may be stated.
- That the initialisation recipe works: it is a literature-based recipe with a unit-tested implementation only.

## 11. Not done

- No model side: no embedding initialisation of a real checkpoint, no CPT, no 50-step probes (GPU needed).
- No test-split evaluation (reserved for the one-shot Stage 5 run).
- No comparison with Purason et al.'s own toolkit, AdaptBPE or naive 'train-and-append' extension (the latter is the baseline in Purason et al.; not needed for the tokenizer deliverable and not run).
- Leaf-based pruning of rare intermediate new tokens (Purason et al.) was not applied: PLAN 4.3 measured that leaf pruning cannot remove intermediate nodes, which is where the rare new tokens are.
- No GGUF / llama.cpp conversion check of the Gemma-3 `tokenizer.model`; no SentencePiece model for Gemma-4 (its base folder has none).
- Other Arabic-script languages (Persian, Pashto, Arabic, Shahmukhi) were not in the check set.
- Aya/Command, gated originals and bases without local tokenizer files were not added.

## 12. Reproduce (F:\Hindko\_tokenizer\trackb, Python 3.11, tokenizers 0.22.2)

```
python checkset/make_checkset.py
python continued_bpe.py all --base <qwen-3|llama-3|gemma-3|qwen-3.5|gemma-4>   # build + sweep/<base>/k*/tokenizer.json
python evaluate.py --bases <base ...>                                         # results/<base>/k*/
python report.py                                                              # knee.json, tables.md, figures/
python deliver.py                                                             # deliver/<name>/
python determinism.py --bases ... --prefix --gemma-cross; python sp_model_check.py --base gemma-3
python test_init_embeddings.py; python make_trackb_md.py
```

Code sha256 at build time: qwen-3: continued_bpe.py `947189b88698…`; llama-3: continued_bpe.py `947189b88698…`; gemma-3: continued_bpe.py `947189b88698…`; qwen-3.5: continued_bpe.py `fd1b6fa966e0…`; gemma-4: continued_bpe.py `947189b88698…`. Final `continued_bpe.py`: `fd1b6fa966e0…`; the G4 rebuilds (section 3.4) ran with the final code and reproduced every base's 16,384-token file byte for byte, so the earlier hashes differ only by the Qwen3.5 config-token fix, which changes no other base.

