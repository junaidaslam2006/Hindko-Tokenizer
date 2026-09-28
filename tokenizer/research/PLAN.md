# Hindko tokenizer: ranked candidates, evaluation protocol, decision rule

Written 2026-09-26. Evidence and citations: `SOTA_TOKENIZATION.md` (section numbers below refer to it).
Every runtime in this plan was measured on this machine or is marked *estimate*.

---

## 0. What "state of the art" can honestly mean here

- No Hindko tokenizer exists (§1.5), so any carefully built one is the first.
- "Best" must be *measured*, and the claim scoped to what was measured. The plan below produces exactly this claim (§8):

  > "Among N tokenizers trained and compared under one protocol, T achieved the lowest held-out bits-per-byte on
  > Hindko (strict test split) with a small from-scratch LM, significantly better than the standard BPE recipe
  > (Δ = x%, 95% CI [a, b], 3+ seeds, cluster bootstrap). It also uses y% fewer tokens than the tokenizers of
  > <external models> on the same text, is lossless, has no unreachable whole-character tokens, and publishes the
  > training frequency of every token."

- Two different deliverables, because Hindko models will be built two ways:
  - **Track A: a from-scratch Hindko tokenizer.** Used for Hindko-only models, the benchmark release, and research. This track is fully executable on this CPU and is the subject of §§1–8.
  - **Track B: vocabulary extension of an existing LLM for continued pretraining (CPT).** This is the likeliest route to a *useful* Hindko chat model (§3.5). The tokenizer side runs on this CPU; the model side (embedding initialisation, CPT, 50-step probes) needs a GPU. See §9.

---

## 1. Prerequisites (blocking; do these first)

### 1.1 Freeze the canonical text form, and make the tokenizer normalise nothing

- **Corpus side** (settled already): NFC; ZWNJ preserved; kashida removed.
- **Decisions to write down** (my recommendation in brackets):
  - Presentation forms: [keep; ﷺ is 98% of them].
  - Digit script: [keep as written; ASCII and Extended Arabic-Indic both kept].
  - U+0095 × 50: [ask the corpus pipeline; it is probably a mis-decoded cp1252 bullet].
  - U+FFFD × 49 in 20 book documents (found by the gate audit, §4.3 R2): [ask the corpus pipeline; replacement characters mark text lost in an upstream decoding step].
  - Harakat: [keep; they are content].
- **Tokenizer side:** HF `normalizer=None`. SentencePiece: `normalization_rule_name=identity`, `remove_extra_whitespaces=False`, `max_sentence_length=65536` (783 lines exceed the 4,192-byte default and would be silently dropped).
- **SentencePiece newline handling** (found in the post-review audit, §4.3): the pilot trained SentencePiece on single lines but encoded whole documents. Every `\n` then became a byte-fallback piece (322,852 in the pilot train set), and the first word of each line lost its `▁` prefix. SentencePiece candidates must encode the way they were trained: split each document at `\n`, encode the lines, and join them with one dedicated newline piece (`user_defined_symbols=["\n"]`). This wrapper is part of the encoder's identity, and G1 round-trip covers it. It is untested so far; verify it in Stage 0, including that SentencePiece accepts a newline user symbol under `normalization_rule_name=identity`.
- **Hard gate:** `decode(encode(doc)) == doc` for 100% of dev and test documents, for every candidate. bpb is only comparable across tokenizers when all of them encode the identical text.

### 1.2 Splits (`F:\Hindko\_tokenizer\splits\`)

*Revised 2026-09-26 after review.* The first version keyed newspaper clusters on `issue` without checking that field. `issue` is **null for 4,345 of the 6,098 newspaper records: 78.5% of newspaper bytes, and 78.6% (8.26 MB) of strict newspaper bytes** (`split_facts.py` → `split_facts.json`). Keyed on `issue` alone, those records would form either one unsplittable 8 MB group, larger than dev and test together, or one cluster per document, which is not group-disjoint.
- **Where the null-issue records come from** (measured; this corrects a detail of the review): 4,337 of them lie in the dated tree `Hindkowan Newspaper Data/Hindko wan data 2024 to 2026/<year>/<month>/<day folder>/` and 8 in `2023-2024-2025/<Month-Year>/`. They span 120 distinct folders. `date_precision` is `day` for 3,251, `month` for 1,037, and absent for 57.
- **Fallback key (adopted from the split builder, `splits/make_splits.py` and `SPLITS.md`):** newspaper group = *edition* = `issue` if present, else the `source_path` folder, which is one edition per day folder. Editions sharing a day-precision `date` are merged (union-find), because the numbered and dated trees hold the same editions. Same-source groups sharing ≥ 25% of the smaller group's distinct word 8-grams are merged as well.
  - Result in the manifest used here: 256 newspaper groups (149 issue-keyed, 107 folder-keyed); the largest is 0.196 MB, the median 0.025 MB.
  - Both the split (§1.2) and the bootstrap (§6) use this key, read from the manifest's `group` field. They never use the raw `issue` field.

The splits now exist and have the required properties:
- **Group-disjoint:**
  - newspaper by edition (above);
  - book by `book_folder`;
  - web by `site`, except Omnilingual transcripts (by `web_meta.speaker_id`), Common Voice (by record) and five large sites (by record, with leakage-checked boundaries; see `SPLITS.md`).
- **Near-duplicate leakage checked** across the boundaries: MinHash LSH + exact Jaccard ≥ 0.5, plus word-8-gram containment ≥ 0.3. There are 0 leaked evaluation records after resolution (`splits_report.json`).
- **Sizes:** 90/5/5 by words per source (tolerance ±1.5 points). Measured on manifest sha256 `76582d3a…`:
  - strict dev: 836 documents, **1.446 MB**;
  - strict test: 491 documents, 1.451 MB;
  - permissive train: 16,015 documents, 45.0 MB;
  - strict train: 26.3 MB.
- **Evaluation text:** dev and test contain only `quality_tier == "strict"` documents. Tokenizer and LM *training* may use permissive-tier documents from the train groups (data-mix axis, §2.1).
- **Manifest:** `split_manifest.jsonl` (uid → split, group). Every result records its sha256. The builder re-ran several times on 2026-09-26 (hashes `6f0da737…`, then `76582d3a…`), so **Stage 0 freezes one hash** and re-runs `split_facts.py` on it.
- **Test discipline:** the test split is used **once**, at the end (§3, Stage 5). All selection happens on dev.

### 1.3 One evaluation harness (`_tokenizer/eval/`, to be written)

- Accepts HF `tokenizer.json`, SentencePiece `.model`, and custom Python encoders (PickyBPE).
- Computes every metric in §4 per document, so that the bootstrap in §6 can resample documents.
- Encoding must be each algorithm's *native* encoder (Uzan et al. 2024), recorded as part of the tokenizer's identity.
- Differential test: the Python `regex` and Oniguruma versions of each pre-tokenizer regex must produce identical pretokens on dev. The escape syntax differs: `\u200C` vs `\x{200C}`.

---

## 2. Candidate configurations

### 2.1 Building blocks

**Pre-tokenizers** (all verified to compile and behave as intended in HF/Oniguruma on 2026-09-26):

| id | regex (Oniguruma) | role |
|---|---|---|
| **P1** (default) | ` ?[\p{L}\p{M}\x{200C}\x{200D}]+\| ?\p{N}\| ?[^\s\p{L}\p{N}\p{M}]+\|\s+(?!\S)\|\s+` | marks, ZWNJ and ZWJ stay inside words; single digits; punctuation runs separate |
| P1r3 | as P1, with ` ?\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))` for digits | right-to-left groups of 3 (`1234567` → `1` `234` `567`) |
| P0 | GPT-2 regex | **negative control only.** Measured 6–7% worse compression, because marks get split off. |
| Pm | ` ?[\p{L}\x{200C}\x{200D}]+\|\p{M}+\| ?\p{N}\| …` | marks as separate pretokens. The diacritic-consistency hypothesis; intrinsic-only probe, low prior. |
| SPnat | SentencePiece's own ▁/script splitting + `split_digits=true` | native SentencePiece behaviour |

**Algorithms and implementations:**

| id | algorithm | implementation | encoder | runs here now? |
|---|---|---|---|---|
| A1 | BPE, byte-level | HF `BpeTrainer`, `ByteLevel(use_regex=False)` after `Split(P)` | HF native | yes (15 s) |
| A2 | BPE, char-level + `byte_fallback` | HF `BPE(byte_fallback=True)`; add the 256 `<0xNN>` tokens | HF native | yes (16 s; built and audited in `gate_audit.py a2`) |
| A3 | BPE (SentencePiece) | `spm` `model_type=bpe`, `byte_fallback` | SP native | yes (minutes, *estimate*) |
| A4 | Unigram (SentencePiece) | `spm` `model_type=unigram`, `byte_fallback` | SP native | yes (65 s) |
| A5 | Unigram, with P1 | HF `UnigramTrainer` after `Split(P1)` | HF native | yes |
| A6 | **SuperBPE** (stage 1 = A1 up to t, stage 2 to T without whitespace splitting) | stage 2 via the code-point (PUA) trick on the stock Rust trainer, or pure Python (§4 of SOTA) | **HF native** after conversion; verify | to build (≈½ day of coding) |
| A7 | **PickyBPE** (IoS ≥ τ deletions) | pure-Python reimplementation, with its event-ordered encoder | custom | to build; ≈1 min per training (estimated from the measured 20 s / 16k merges) |
| A8 | Unigram, superword | `spm` `split_by_whitespace=false` | SP native | yes; low prior (no curriculum) |
| (A9) | BoundlessBPE | kensho `fastboundlessbpe` pure-Python reference | custom `.model` | only if you approve the git download; near-equivalent to A6 per Schmidt et al. 2026 |
| A10 | **MinGram** (Land 2026): BPE seed at 1.15·n, 2 hard-EM iterations on the minimum-token path, one flat score prune; P1 pretokens, `byte_fallback` | route (a): the author's pure-Python `sanderland/script_tok` (git clone, **needs your approval**); route (b): our reimplementation from the paper (≈150 lines: HF `BpeTrainer` seed + Python Viterbi over the ≈250k pretoken types) | reference encoder; **HF-native candidate** as an HF `Unigram` with scores −C + log p (fewest tokens first, score as tie-break), verified by an encoder-equivalence test on 100% of dev | to build (≈½ day for route b); training cost *estimate* a few minutes |

Deliberately **not** candidates:
- WordPiece: built for encoders.
- Length-MAX, Scaffold-BPE, LiteToken: no usable code. PickyBPE covers the idea of removing intermediate tokens.
- MinGram-PP (the compression-oriented MinGram variant): in Land's Table 4 it was not better in bpb than plain MinGram (0.7125 vs 0.7123). Plain MinGram is the Unigram-family representative beside standard Unigram.
- MorphBPE, MorphPiece: there is no Hindko morphological segmenter.
- SaGe, TreeTok: heavy, and no evidence of bpb gains.
- Over-tokenized input vocabularies: an architecture change.

**Vocabulary sizes:**
- Intrinsic sweep: {8k, 12k, 16k, 24k, 32k, 48k}.
- LM stage: at most {8k, 16k, 32k} plus refinements.
- Every tokenizer reserves the same **special-token block**:
  - `<|endoftext|>`, `<|bos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, and `<|reserved_0..58|>` (64 in total);
  - the total is padded to a multiple of 64. "16k" therefore means 16,384 including specials.

**Training-data mixes** (all from the *train* groups only):

| id | contents |
|---|---|
| D1 | permissive train (all tiers; default, because the tokenizer is far from data saturation, Reddy et al. 2025) |
| D2 | strict train only |
| D3 | D1 with book text down-sampled to ≤50% of characters (fixed seed). It tests whether dictionary- and poetry-heavy books skew the vocabulary. |

### 2.2 Ranked list (the priors: what I expect to matter most)

*Re-ranked 2026-09-26 after review.* The BPE-vs-Unigram evidence for bpb is mixed (SOTA §1.1: Yavuz et al. 2026 vs Land 2026), so the Unigram family moves up and MinGram is added. **Ranks 1–7 are 8 LM waves** (rank 6 is two waves). Rank 8 is a conditional extra wave. The rank order is also the pre-declared **cut order** when compute runs short (§3).

| rank | configuration | why it is on the list | stage |
|---|---|---|---|
| 1 | **A1 BPE-P1-D1 @16k**, the "standard recipe" | baseline every claim is measured against; HF-native; the Llama-3/Qwen family format | 1, 3 |
| 2 | **A6 SuperBPE-P1 @16k, t/T = 0.9** | the largest documented gain (Liu et al. 2025; Arnett et al. 2025), and Hindko's frequent short function words are its ideal case. A late transition beat the most compressive setting in the original paper. | 2, 3 |
| 3 | **Unigram @16k** (the better of A4-SPnat and A5-P1 on dev bytes/token) | best pilot compression (+2.6% over A1); far fewer under-supported tokens (§4.3 R1: 51 multi-character pieces below 20 train occurrences vs 404 learned tokens for A1); Unigram-family beat BPE in bpb in Land 2026, lost in Yavuz et al. 2026 | 1, 3 |
| 4 | **A10 MinGram @16k** | Unigram-family variant with the best bpb in Land 2026 (20 seeds, English, 32k) and better compression than BPE/Unigram in 6 languages incl. Arabic; cheap to train | 2, 3 |
| 5 | **A6 SuperBPE @16k, t/T = 0.8** | brackets the transition point | 2, 3 |
| 6 | **A1 @8k** and **A1 @32k** (two waves) | vocabulary-size curve in the data-scarce regime (Tao et al. 2024); 32k has 22.7% of its learned tokens below 20 train occurrences (§4.3) | 1, 3 |
| 7 | **A7 PickyBPE @16k, τ = 0.9** (0.8 if 0.9 removes <1% of tokens) | removes exactly the under-trained intermediate tokens that §4.3 found (all 404 of A1-16k's); the gain was largest at small vocabularies (Chizhov et al. 2024) | 2, 3 |
| 8 | A6 SuperBPE @32k (t/T from the better of ranks 2 and 5) | superwords with more headroom. **Conditional extra wave** (≈ 55 min), run only if rank 2 or 5 beats rank 1 | 3 |
| 9 | A2 char-level BPE, A3 SentencePiece BPE | library and base-alphabet effects (Ali et al. 2024: SentencePiece-BPE > HF-BPE). A2 matches A1 on pilot compression (6.007 vs 6.009 bytes/token) and has no partial-UTF-8 tokens | 1 (intrinsic); LM only if intrinsically ≥1% better than A1 |
| 10 | A8 SentencePiece superword Unigram; Pm; P1r3; D2/D3 | cheap probes | 1 (intrinsic only) |

---

## 3. Stages and budget (wall-clock at ≤3 worker processes on this shared laptop CPU)

| stage | what | budget |
|---|---|---|
| 0 | prerequisites §1 (freeze the split-manifest hash); harness; gate audit (`gate_audit.py` generalised); external baselines (intrinsic; needs approval, §10) | 30–60 min, mostly coding |
| 1 | **intrinsic sweep**: A1 × {P1, P0, P1r3, Pm} × 6 sizes × D1, plus A1-P1 × {D2, D3} × {16k, 32k}, plus A2, A3, A4, A5, A8 × 6 sizes × D1. About 70 tokenizers at 15–120 s each, plus gate and support-profile audits (≈ 1 min each, measured: 37 s per train-split encode) | **≈ 2 h CPU** (*estimate* from the measured per-tokenizer times) |
| 2 | build and verify A6 (SuperBPE), A7 (PickyBPE) and A10 (MinGram); train their variants; intrinsic metrics | coding ≈ 1 day; compute < 45 min (*estimate*) |
| 3 | **LM screening**: ranks 1–7 = 8 candidate waves + 1 learning-rate wave, 3 seeds each (3 simultaneous 1-thread processes) | **≈ 5.4 h** (§5, from measured concurrent throughput). Core session (LR wave + ranks 1–5): ≈ 3.5 h; second session (ranks 6–7): ≈ 1.9 h; conditional rank 8: +≈ 55 min |
| 4 | *optional*: confirmation of the top 2 plus baseline on the **full** train split with the larger arbiter | **≈ 8–12 h** for three 16k waves (≈ 2.6–3.3 h per 16k wave; ≈ 5.4 h for a 32k wave). Reduced variant, winner vs baseline only: ≈ 5.3–6.5 h |
| 5 | one-shot test-set evaluation, health checks, packaging, tokenizer card | 30 min |

*Budget corrected 2026-09-26 after review.* The first version gave ≈ 3–3.5 h for Stage 3 and ≈ 6–8 h for Stage 4, based on solo single-thread throughput and an unmeasured "about 2×" parallel speed-up. Measured concurrent throughput is 0.54–0.76× of solo (§5), so the stages take ≈ 1.3–1.7× longer than first stated. I kept the candidate list and the 10 MB screening budget. Cutting training bytes would make the rankings of larger vocabularies less stable, so the cut is by rank instead.

**Selection between stages.**
- **After Stage 1:**
  - Fix the pre-tokenizer. Expected: P1. Choose P1r3 only if it is not worse by more than 0.2% in dev bytes/token; digits are 0.43% of characters, so this mostly matters for numeracy (TokEval).
  - Fix the data mix. Choose D1 unless D2 or D3 is ≥0.5% better on *strict* dev bytes/token *and* no source is worse by >1%.
  - Choose the Unigram implementation for rank 3 (A4-SPnat or A5-P1).
  - Drop anything that fails a hard gate G1–G5 (§4.3) after the pre-declared G2 remedy. The reported properties R1–R2 never drop a candidate.
- **After Stage 2:** keep an A6, A7 or A10 variant only if (i) it passes the hard gates G1–G5, (ii) it round-trips exactly, and (iii) for A6 and A10, its HF-native encoding reproduces the reference encoder on 100% of dev documents (otherwise it stays in with its custom encoder and loses tie-breaker a).

---

## 4. Metrics (exact definitions; all computed per document on dev, then aggregated)

### 4.1 Primary (decides): LM bits-per-byte (§5)

bpb(c) = Σ_docs Σ_tokens −log₂ p(token | prefix) ÷ Σ_docs UTF-8 bytes(doc)

- Every token is predicted, including each document's end-of-text token.
- BOS is given, not predicted.
- Bytes are counted on the canonical text, excluding separators.

### 4.2 Secondary (screening and tie-breaking; **never decisive alone**)

| metric | definition |
|---|---|
| bytes/token (compression) | Σ bytes ÷ Σ tokens. Also reported per source (newspaper/book/web) and per `language_variety`. |
| chars/token | Σ chars ÷ Σ tokens |
| NSL | tokens(candidate) ÷ tokens(A1-P1-16k) on the same dev text |
| fertility | tokens per whitespace word, computed from *in-context* encoding via offsets. Do not encode isolated words: in the pilot that inflated SentencePiece fertility by +1. |
| continued-word rate / STRR | share of word tokens split into ≥2 tokens / kept as exactly 1 token |
| Rényi efficiency | H_α(p)/log₂\|V\| of dev token frequencies, α = 2.5 (Zouhar et al.) and α = 2 (TokEval). **Report only.** It rated the worse GPT-2 regex higher in our pilot. |
| vocabulary utilisation and tail | share of vocabulary ids seen on dev; the support profile R1 of §4.3 (learned tokens with **train** frequency 0, < 20, < 100; leaves vs intermediate tokens) |
| robustness (TokSuite-style) | relative change in token count, and share of words whose segmentation changes, when dev text is perturbed by (i) removing all harakat, (ii) swapping digit script, (iii) deleting the space before ۔/، (iv) inserting ZWNJ inside compounds |
| morphological alignment | report-only. MorphScore-`urd_arab` (download needs approval) and/or a native-speaker Hindko gold set of 300–500 words. MorphScore is not predictive of LM quality (Arnett et al. 2025). |
| encode speed | documents per second (sanity only) |

### 4.3 Health checks: hard gates and reported properties

*Rewritten 2026-09-26 after review.* The first version had seven hard gates. Gates 2–4 rejected the plan's own baseline A1, and the stated remedy for gate 4 does nothing. Every number below was measured by `gate_audit.py` (outputs in `gates/`) on the pilot tokenizers: pilot split, 17,333 training / 950 held-out documents, **not** group-disjoint. The gates now separate *correctness* (hard) from *vocabulary quality* (reported and compared, never grounds for discarding a candidate).

**Hard gates.** Every LM candidate must pass them, and the pilot baseline A1-16k does.

| # | gate | exact test | pilot result |
|---|---|---|---|
| G1 | lossless | `decode(encode(doc)) == doc` for 100% of dev and test documents | 0 / 950 failures for A1-16k, A1-32k, SP-Unigram-16k/32k, A2-16k |
| G2 | no unreachable complete-character tokens | self-tokenization, `encode(decode([id])) == [id]`, for every learned non-special id whose bytes are **valid UTF-8**. Exempt: the 256 base bytes / `<0xNN>` pieces, and byte-level tokens that are partial UTF-8 sequences (they cannot be written as text on their own; see R2). SuperBPE multi-word tokens are tested on the shortest train chunk that contains them | A1-16k 0 failures; A1-32k 0; A2-16k 0; SP-Unigram-16k 0; **SP-Unigram-32k 5 failures** (pieces such as `▁نے۔`, which Viterbi always splits into `▁نے` + `۔`; train frequency 0) |
| G3 | special tokens atomic | the literal marker strings never occur in corpus text, and the encoder never splits them | to run in Stage 0 |
| G4 | determinism | two trainings with identical inputs give identical tokenizer files | A1-16k retrained: JSON identical to the saved pilot file (1 repetition) |
| G5 | no sub-character tokens in *character-level* vocabularies (A2–A5, A8, A10) | every token except the 256 `<0xNN>` fallback pieces is a whole-character string. A token that is not would be a bug | 0 for A2-16k and SP-Unigram, by construction |

- **Remedy for G2, fixed in advance.** Delete the failing tokens and report the effective vocabulary size.
  - Unigram family: a piece that loses the Viterbi search on its own string never wins it inside a longer string, so deleting it cannot change any encoding. Verify by re-encoding dev: the ids must be identical up to renumbering.
  - BPE family: delete, then run the same dev re-encoding check. If any encoding changes, retrain instead.

**Reported properties.** They are measured for every candidate, printed in the tokenizer card, and used as tie-breakers in §7, but never used to discard a candidate:

- **R1. Support profile on the train split:** the number and share of learned tokens with train frequency 0, < 20 and < 100, split (BPE family) into leaves and intermediate nodes of the merge graph. Pilot values:

| tokenizer | learned tokens | freq = 0 | freq < 20 | of which intermediate | freq < 100 | held-out bytes/token |
|---|---|---|---|---|---|---|
| A1 byte-level BPE-P1 16k | 15,743 | 20 | 404 (2.57%) | **404 (all)** | 7,463 (47.4%) | 6.009 |
| A1 leaf-pruned to 16k from 18k | 15,743 | 20 | 403 (2.56%) | 403 (all) | 7,463 | 6.009 |
| A1 leaf-pruned to 16k from 20k | 15,743 | 20 | 403 (2.56%) | 403 (all) | 7,463 | 6.009 |
| A2 char-level BPE + byte_fallback 16k | 15,478 | 15 | 419 (2.71%) | 419 (all) | 7,248 (46.8%) | 6.007 |
| A1 byte-level BPE-P1 32k | 31,743 | 78 | 7,218 (22.7%) | 1,820 | 24,242 (76.4%) | 6.318 |
| SP Unigram 16k (all normal pieces) | 15,741 | 0 | 127 (0.81%); multi-character pieces: **51** | — (76 are single characters that `character_coverage=1.0` must keep) | 7,285 (46.3%) | 6.168 |
| SP Unigram 32k | 31,741 | 30 | 8,437 (26.6%) | — | 23,962 (75.5%) | 6.497 |

  SentencePiece support is counted on whole training documents, as an LM sees them. Counted on the non-empty *lines* the pilot models were trained on, the figures are 79 (1 multi-character) at 16k and 8,385 (6 at 0) at 32k. The two differ because the pilot trained SentencePiece on lines but encodes whole documents. Inside a document, every `\n` becomes a byte-fallback piece (322,852 in train), and a word after a newline gets no `▁` prefix. That train/use mismatch is fixed in §1.1.

- **R2. Partial-UTF-8 tokens (byte-level BPE only):** count and train frequency.
  - Pilot: 9 at 16k and 10 at 32k. Examples: id 258 = `20 d8` (space + Arabic lead byte, 52 train occurrences), id 282 = `d8 a7 d8`, id 1148 = `e0 a2` (lead bytes of the U+08xx Hindko letters), id 15887 = `bf bd` (tail of U+FFFD).
  - 3 of the 9 at 16k have train frequency 0.
  - Every byte-level BPE (GPT-2, Llama-3, Qwen) has such tokens. Generation must therefore detokenise at byte level.
  - `bf bd` exists because the corpus contains **49 U+FFFD replacement characters in 20 book documents**. That is a corpus defect, reported with U+0095 in §1.1.
- **R3. After LM training** (Stages 3–4): the 50 lowest-norm output embeddings and their train frequencies, to catch under-trained tokens (Land & Bartolo 2024).

**Why old gates 3 and 4 are not hard gates any more (measured):**
- **Old gate 3 (no partial UTF-8) fails for every byte-level BPE.** BPE merges bytes before a character is complete whenever such a pair is frequent, for example space + the lead byte `d8`. Only character-level vocabularies can meet it, and for them it is G5.
- **Old gate 4 (every learned token seen ≥ 20 times) is not attainable by BPE with any standard remedy:**
  - At 16k, *all* 404 under-supported tokens, and all 20 never-used ones, are **intermediate** tokens (e.g. ` تحقی`, absorbed into ` تحقیق`). Later merges consume almost all their occurrences. No leaf of the 16k merge graph occurs fewer than 20 times.
  - **`min_frequency` does nothing:** `BpeTrainer` applies it to pair counts *at merge time*. Retraining with `min_frequency` = 20 gave merges identical to `min_frequency` = 2. At 100 the trainer stops early, at 9,500 merges (vocabulary 9,757); at 400, 3,685 merges; at 1,000, 1,903. Training stops at the first merge whose pair count is below the threshold, so a threshold that binds only shrinks the vocabulary. All 15,743 merges of the 16k vocabulary had a pair count ≥ 20 when they were made; the rarity appears only afterwards, in the final encoding.
  - **Leaf-pruning to a fixed size does nothing either** (only leaves of the merge graph are removed, as in Purason et al.'s pruning, so no token becomes unreachable; train at 16k + m, then repeatedly delete the lowest-frequency leaf tokens and their merges, re-counting after each batch). From 18k and from 20k it converges to the same tokenizer, which differs from the plain 16k tokenizer in exactly one token (and one merge): 403 tokens below 20 instead of 404, bytes/token 6.0089 vs 6.0088. The rarest leaves are simply the last merges.
  - **What does remove rare intermediates:** deleting them from the vocabulary with an encoder that tolerates deletions (PickyBPE, A7; custom encoder), or a Unigram-family vocabulary, where pruning is top-down (SP-Unigram-16k: 51 multi-character pieces below 20, 0 never used; MinGram paper, Table 4: 9 rare tokens vs 172 for BPE at 32k).
  - Whether removing them *helps bits-per-byte* is exactly what the A1-vs-A7 and A1-vs-Unigram/MinGram LM comparisons measure. It is a hypothesis to test, not a precondition.
- The 32k rows show the data-scarcity point of §1.6 of SOTA_TOKENIZATION: at 32k, 23–27% of the learned vocabulary is seen fewer than 20 times in about 7.3–7.5M training tokens.

---

## 5. LM-based arbiter protocol

**Model ("hk-tiny").**
- GPT decoder: pre-LayerNorm, GELU MLP (4×), learned positions, **tied input/output embeddings**, no dropout in screening.
- Screening arbiter: d = 128, L = 4, H = 4 (0.79M non-embedding parameters).
- Confirmation arbiter: d = 192, L = 4, H = 4 (1.77M).
- Justification: the whole corpus is ≈ 8M tokens. A compute-optimal from-scratch model for that data is < 10M parameters, so these are proxies of the right order. They are **not** evidence about 7B CPT behaviour.
- Embedding size varies with vocabulary; non-embedding size is fixed. Report total parameters.

**Fairness controls (byte-matched):**
- Same training *bytes* for every candidate: the first B bytes of the seed-shuffled train document stream.
- Same context in *bytes*: ctx_tokens(c) = round(1,536 / bytes_per_token(c) on train), which is ≈ 256 tokens at 6 bytes/token.
- Same bytes per optimiser step: 8 sequences, ≈ 12 KB/step. The step count is therefore identical across tokenizers (SuperBPE matched context in bytes the same way).
- Documents are shuffled with the run seed and joined with `<|endoftext|>`.

**Optimisation:**
- AdamW (β = 0.9/0.95, ε = 1e-8, weight decay 0.1 on matrices only), gradient clip 1.0.
- Warm-up 50 steps, cosine decay to 10%.
- Peak learning rate chosen **once** on the baseline A1-16k by a 3-point sweep {1e-3, 3e-3, 6e-3} (1 seed, screening budget), then used for all candidates. This is a documented limitation; re-sweep the finalists in Stage 4 if their bpb gap is < 1%.

**CPU settings (measured to matter):**
- `torch.set_flush_denormal(True)`. Without it, steps were 2–11× slower and erratic.
- One thread per process (`OMP_NUM_THREADS=1`), the three seeds of a wave running simultaneously. Measured (`bench_concurrent.json`): each of three simultaneous 1-thread processes runs at 0.54–0.76× of a solo 1-thread process, so a wave's aggregate is 1.6–2.3× one solo process, **not** the 3× that the earlier "about 2×" claim implicitly assumed. Against one 3-thread job per seed run in sequence, the parallel wave is **1.46× faster** at 16k (d=128: 2,821 vs 1,926 tok/s aggregate; d=192: 2,011 vs 1,382). The reviewer measured 1.28× for d=192. The earlier "about 2×" was arithmetic (3 × solo 1-thread rate), never measured, and is withdrawn.
- Chunked output-layer loss (512 rows).
- `torch.use_deterministic_algorithms(True)`; fixed seeds for torch, numpy and python.

**Seeds and budget.** Seeds {1, 2, 3} run simultaneously as one "wave" per candidate. *Re-derived 2026-09-26 after review, from **concurrently** measured throughput* (`bench_concurrent.json`: three simultaneous 1-thread processes on the same configuration, under load from other agents) by `budget_estimate.py` → `budget_estimate.json`:

| stage | arbiter | train bytes / run | per-process train tok/s, 3 concurrent (8k / 16k / 32k) | minutes per wave incl. evaluation (8k / 16k / 32k) | waves | total |
|---|---|---|---|---|---|---|
| 3 screening | d=128 | 10 MB (≈ 1.66M tokens at 16k) | 1,487 / 940 / 573 | **24.5 / 35.3 / 55.1** (train 20.0 / 29.5 / 46.0 + eval 3.4 / 4.8 / 8.1 + 1 overhead) | 8 candidate waves (ranks 1–7) + 1 LR-sweep wave; conditional rank 8 (+55 min) | **≈ 5.4 h** (core LR + ranks 1–5 ≈ 3.5 h; ranks 6–7 ≈ 1.9 h) |
| 4 confirmation | d=192 | full permissive train split, 1 epoch (45.0 MB) | 1,087 / 670 / 381 (reviewer's run at 16k: 832) | 130 / 158–195 / 326 | 3 (baseline + top 2) | **≈ 8–12 h**; winner vs baseline only ≈ 5.3–6.5 h |

How the estimates were derived:
- train minutes = train bytes ÷ (bytes/token × per-process concurrent tok/s);
- evaluation minutes = (2 × 1.446 MB strict dev + 3 × 2 × 0.5 MB subset) ÷ bytes/token ÷ concurrent eval tok/s (d=128: 5,123 / 3,388 / 1,921; d=192: 3,323 / 2,105 / 1,134). The factor 2 is the stride-ctx/2 overlap;
- plus 1 min per wave for set-up (*estimate*);
- bytes/token: 16k and 32k from the pilot (6.009, 6.318); **8k measured** by training BPE-P1 at 8,192 on the pilot split (5.597, `pilot/hf_bpe_P1_8192.json`); Unigram and MinGram use the pilot Unigram value 6.168; SuperBPE and PickyBPE use A1's value. SuperBPE compresses more, so its figure is an upper bound on time;
- throughput on this shared machine varied by up to ≈ 25% between measurements. The Stage 4 range spans this run (670 tok/s at 16k) and the reviewer's (832).

**Evaluation:**
- Full strict dev (836 documents, 1.446 MB on manifest `76582d3a…`). Each document is scored independently, with windows of ctx tokens and stride ctx/2, so that every token after the first window has ≥ ctx/2 context.
- Store per-document NLL (bits) and bytes for every run.
- Evaluate a fixed 0.5 MB dev subset at 25/50/75/100% of training. If candidate rankings cross between 75% and 100% for the top two, the screening budget is too small, and those two go straight to Stage 4.

---

## 6. Significance testing

- **Unit of resampling:** the source cluster, not the document. Documents of one book or edition are correlated, and treating them as independent would overstate confidence.
  - *Revised after review:* the cluster is the split manifest's `group` field (§1.2), **not** the raw `issue` field, which is null for 78.5% of newspaper bytes. Newspaper = edition (issue, else the `source_path` day folder, with date and overlap merges); book = `book_folder`; web = site or Omnilingual speaker.
  - Per-record web groups (Common Voice and the five large sites split by record) are collapsed to their site for resampling. That is coarser, and so conservative.
  - Strict dev then has **32 clusters** (15 newspaper, 12 book, 5 web); strict test has 27 (15 / 6 / 6) (`split_facts.json`, manifest `76582d3a…`).
  - The largest dev cluster is a single book of 0.245 MB (17% of dev bytes). With this few clusters, the CIs will be wide; that is honest.
  - Sensitivity analysis, reported but not decisive: the same bootstrap with the manifest groups as they are (per-record web groups kept).
- **Hierarchical bootstrap** (10,000 replicates, RNG seed 12345). In each replicate:
  - resample clusters with replacement **within each source** (stratified, so that the newspaper/book/web mix of dev is fixed), keeping all their documents;
  - independently for each candidate, resample its 3 seeds with replacement;
  - compute Δ = bpb(A) − bpb(B) on the replicate.
- **Outputs:** the 95% percentile CI and a two-sided p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)).
- **Multiple comparisons:** Holm–Bonferroni over the (k − 1) comparisons of every candidate against the best-mean candidate. Separately, a single pre-registered comparison of the winner against the baseline A1-16k.
- **Equivalence margin:** δ = 0.3% of baseline bpb. Two candidates are *equivalent* if the 90% CI of Δ lies inside ±δ (TOST).
- **Power check after the baseline's first wave:**
  - σ = standard deviation of run-level bpb across the 3 seeds;
  - seeds needed to detect Δ_min = 0.5% at α = 0.05 and power 0.8: n = ⌈15.7 · σ² / Δ_min²⌉.
  - σ ≤ 0.2% → 3 seeds suffice. σ ≈ 0.35% → about 8 seeds; add them only for the finalists in Stage 4.
  - Reference: Yavuz et al. (2026) report seed s.d. of 0.01–0.3% of bpb for their 1B / 128k-vocabulary runs; tiny models are likely noisier.
  - Land (2026) needed 20 seeds to resolve gaps of 0.03–0.22% of bpb between tokenizers (BPE vs MinGram: 0.22%). Gaps that small are below this plan's Δ_min, and would be reported as "not distinguishable at our power", never as "equal".

---

## 7. Decision rule (pre-registered)

1. **Gate:** discard any candidate that fails a hard gate G1–G5 of §4.3 after the pre-declared G2 remedy. The pilot baseline A1-16k passes all five; the reported properties R1–R3 never discard a candidate.
2. **Rank** survivors by mean dev bpb (Stage 4 results if run, otherwise Stage 3). Only candidates whose waves completed are ranked. If the cut order of §2.2 was applied, N counts only completed candidates.
3. The **top set** is the best candidate plus every candidate not significantly worse after Holm correction, and every candidate equivalent within δ.
4. **Tie-breakers inside the top set, in order:**
   - (a) HF-native exact encoding (loads with `AutoTokenizer`, no custom code) over a custom encoder;
   - (b) higher dev bytes/token (cheaper inference, longer effective context);
   - (c) fewer learned tokens with train frequency < 20 (§4.3 R1: fewer under-trained embeddings);
   - (d) lower robustness sensitivity (§4.2);
   - (e) smaller vocabulary.
5. **Improvement claim:** the chosen tokenizer is claimed to *improve on the standard recipe* only if its Δ against A1-16k has a 95% CI that excludes 0. Otherwise the claim is "no measurable difference from standard BPE; chosen for efficiency".
6. **Test:** evaluate the chosen tokenizer and the baseline once on the strict test split (same protocol) and report Δ with its CI. If test contradicts dev in sign, report it and do not re-select.
7. **Deliverables:**
   - `tokenizer.json`/`.model`;
   - `tokenizer_config.json` with the special tokens and a Jinja `chat_template` (ChatML-style using `<|im_start|>`/`<|im_end|>`);
   - a tokenizer card with every parameter, split-manifest hash, code hash, and the metric table.

---

## 8. What we can and cannot claim afterwards

**Can** (if the numbers support it):
- "Lowest held-out bits-per-byte on Hindko (strict test split, group-disjoint) among the N tokenizers compared with small from-scratch LMs (3+ seeds each, cluster-bootstrap CIs), and among ≈ 70 on intrinsic metrics." N = 8 if all rank 1–7 waves ran (9 with the conditional rank 8); fewer if the cut order was applied.
- "Uses x% fewer tokens per byte of Hindko than the tokenizers of <external models measured>." This needs the external tokenizers (§10).
- "Lossless on the canonical form; no unreachable whole-character tokens; y learned tokens (z%) were seen fewer than 20 times in training (published support profile)." For a byte-level winner, add: "w partial-UTF-8 byte tokens, as in every byte-level BPE."
- **Not claimable for a BPE winner:** "every learned token seen ≥ 20 times". The pilot shows that this does not hold for BPE at 16k (404 tokens) and that `min_frequency` and leaf-pruning do not change it (§4.3).

**Cannot:**
- "SOTA for all AI models" or "best possible": only N configurations were tested, with ≤ 5M-parameter proxies.
- That the ranking transfers to 1B+ models or to CPT of a multilingual LLM. Lotz et al. 2025 found good small-to-large transfer, but from 350M, not from 1M.
- Downstream task quality: no Hindko benchmark exists.
- Parity or tokenization premium against other languages: there is no parallel Hindko text (not in FLORES+).
- Morphological quality: there is no gold segmentation.

**To strengthen the claim later:**
- (i) Stage 4 with more seeds;
- (ii) a 20–50M-parameter confirmation on a GPU;
- (iii) native-speaker morphology and perturbation sets;
- (iv) a small parallel Hindko–Urdu–English test set, which would enable parity.

---

## 9. Track B: vocabulary extension for CPT (tokenizer side on this CPU; model side needs a GPU)

1. **Pick base models to screen.** Urdu-capable open models: Qwen3 (byte-level BPE, ungated), Gemma-3 (SentencePiece, 262k; gated), Llama-3.x (tiktoken BPE, 128k; gated), Aya/Command (gated), XLM-R and mT5 (ungated references). Qalb and Alif are Llama-3.1 derivatives, so they share Llama's tokenizer.
   - Measure bytes/token, fertility and STRR on strict dev with the §4 harness.
   - Downloading tokenizer files (a few MB each) needs your approval. Gated repositories need you to accept their licences yourself.
2. **Continued-BPE extension** (Purason et al. 2026): resume merge learning on Hindko train *from the base tokenizer's merges*, for k ∈ {1k, 2k, 4k, 8k, 16k} new tokens.
   - Implementation: their toolkit (git download, needs approval), or our code-point trick (map base token ids to Private-Use-Area code points, then run the stock `BpeTrainer` for k merges).
   - Gate: 0 unreachable whole-character tokens (G2 of §4.3; Purason et al. report that naive appending creates them). Reported, not gated: the support profile of the new tokens (R1), because continued BPE creates intermediate tokens exactly like A1 does (§4.3). Prefer the k at which the share of new tokens with < 100 train occurrences stays small.
   - Deliverable: the dev bytes/token-vs-k curve, and the tokenizer at the knee. Yamaguchi et al. (2026) suggest 100–1k when target data is 0.01 GB; our 50 MB justifies testing up to 16k.
3. **Initialisation recipe to hand over** (not runnable here):
   - asymmetric subword-mean: input embeddings = uniform mean of the base-tokenizer subtokens, with Urdu-script norm calibration; output embeddings = character-length-weighted mean (Joshi et al. 2026);
   - alternatives to compare: FOCUS (needs only a fastText model on our corpus), mean-of-all, Token Distillation;
   - choose with **50-step CPT probes**, not initialisation loss (Joshi et al. 2026);
   - train embeddings plus top-2/bottom-2 layers first (Yamaguchi et al. 2026).
4. **Claim for Track B:** "The extended <base> tokenizer encodes Hindko with x% fewer tokens, with k new tokens all reachable and well-supported." Model quality claims wait for GPU CPT.

---

## 10. Approvals needed before the steps that use them

| item | source | size | used for |
|---|---|---|---|
| External tokenizer files: o200k (tiktoken), Qwen3, XLM-R, mT5, BLOOM (ungated); Gemma-3, Llama-3.x, Aya (gated, licence acceptance by you) | HF Hub / openaipublic | ≈ 2–35 MB each | intrinsic baselines (Stage 0), Track B |
| `kensho-technologies/fastboundlessbpe` (git clone) + `pip install heapdict` | GitHub / PyPI | < 5 MB | optional A9, cross-check of A6 |
| `taidopurason/tokenizer-extension` (git clone) | GitHub | small | Track B (optional; our own route exists) |
| MorphScore data (`urd_arab`) | HF Hub | small | report-only morphology proxy |
| `cimeister/tokenizer-intrinsic-evals` (TokEval) | GitHub | small | optional cross-check of our metric implementations |
| `sanderland/script_tok` (git clone; pure Python, managed with `uv`, Apache-2.0) | GitHub | small | A10 MinGram reference implementation, and the reference for the encoder-equivalence test of our reimplementation. `uv` is installed locally, but it would download the repository's dependencies into a new environment: point `UV_CACHE_DIR` and the environment at the data drive and approve the dependency download too |

Nothing in Stages 1–3 *requires* any download: HF `tokenizers`, `sentencepiece` and `torch` are installed, and A6/A7/A10 can be reimplemented. Without the `script_tok` download, A10 rests on our reimplementation alone and must be labelled "MinGram as reimplemented from the paper", not "MinGram".

---

## 11. Risks and mitigations

| risk | mitigation |
|---|---|
| Splits not ready or leaky | §1.2 gates; the plan cannot start Stage 3 without the manifest |
| A tiny-LM ranking does not reflect larger models | report as a proxy; Stage 4 at a larger size and full data; the claim wording in §8 |
| Learning rate favours one tokenizer | single-LR protocol plus a finalist re-sweep if the gap is < 1% |
| SuperBPE conversion is not exact in HF | mandatory encoder-equivalence test on 100% of dev; fall back to a custom encoder (loses tie-breaker a) |
| CPU contention from other agents slows runs | throughput never enters a decision; FTZ on; runs are resumable per seed. Budgets use *concurrently* measured throughput, which varied by ≈ 25% between runs |
| Compute runs out before all waves finish (Stage 3 ≈ 5.4 h, not the 3–3.5 h first stated) | waves run in rank order (§2.2), and each wave is self-contained; the decision covers completed waves only, and the claim's N says how many |
| Newspaper cluster key undefined (`issue` null for 78.5% of newspaper bytes) | edition key from the split manifest's `group` field (§1.2) for both the split and the bootstrap |
| A gate that the baseline cannot pass makes the decision rule unrunnable | gates re-derived from the pilot audit (§4.3): the baseline passes G1–G5; vocabulary-quality properties are reported and used as tie-breakers |
| Book and dictionary text (71% of characters) dominates the vocabulary | D3 mix; per-source metrics; a no-source-regresses-by->1% rule |
| Superword tokens hurt word-level tasks and prompts ending in a space | keep the best non-superword tokenizer as a documented fallback |
| Overfitting the decision to dev | test used once; everything pre-registered here |
