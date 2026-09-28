# A10 MinGram @16k: MinGram as reimplemented from the paper

PLAN.md Stage 2, candidate A10 (rank 4 of PLAN §2.2). Built and evaluated on 2026-09-26 in
`F:\Hindko\_tokenizer\candidates\mingram\`.

**Label.** Every artefact here is **"MinGram as reimplemented from the paper"**. The paper is Land (2026), *MinGram: A
Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment*, arXiv 2606.27019 v2. I
read it in the arXiv HTML on 2026-09-26. I did not download, read or run the author's code (`sanderland/script_tok`).
This is PLAN route (b). Do not call the tokenizer "MinGram" without that qualifier (PLAN §10).

**Test discipline.** Nothing in this run read the test split. Every data file was loaded through
`eval/harness.load_docs`, which verifies the frozen split manifest (`76582d3a…`) and refuses any file that contains a
test uid. Training used `train_D1` only. Evaluation used `dev_strict`, plus `dev_permissive` for the equivalence test.

---

## 1. Result at a glance (all numbers measured)

| item | value |
|---|---|
| candidate file | `runs/mingram_P1_D1_16384/tokenizer.json`, sha256 `a7df59a194ea3f2741c0b1d928e5f877a4e94210d81e60b5f7bf0e5c14c3d2ec` |
| format | stock HF `tokenizers` **Unigram** (0.22.2), `byte_fallback=true`, P1 `Split` pre-tokenizer, no normalizer, ByteFallback+Fuse decoder; loads with `Tokenizer.from_file`, no custom code. `transformers` 5.3.0 `PreTrainedTokenizerFast(tokenizer_file=…)` gives identical ids on 836/836 dev_strict documents and round-trips all of them (`logs/transformers_check.log`). No `tokenizer_config.json` yet, which `AutoTokenizer` needs (Stage 5) |
| vocabulary | **16,384** = 64 special + 1 `<unk>` + 256 `<0xNN>` + 251 characters + **15,812 learned** |
| hard gates G1–G5 (dev_strict) | **all pass**: G1 836/836; G2 0/16,063 failures; G3 pass; G4 byte-identical retrain; G5 0 sub-character tokens |
| HF-native = reference encoder | **836/836 dev_strict documents identical** (and 100% of every other set tested, §5) |
| dev_strict bytes/token | **6.770** (chars/token 3.796; 213,512 tokens for 1,445,513 bytes) |
| fertility / STRR | 1.200 tokens per word / 84.3% of words are one token |
| R1 support on train_D1 | **0 learned multi-character tokens seen < 20 times** (0 seen 0 times). The 66 tokens under 20 are all single characters of the training alphabet (R1 table, §7) |
| R2 partial-UTF-8 tokens | 0 (character-level vocabulary) |
| train_D1 bytes/token | 6.0067 (7,492,889 tokens) |
| training time | 130 s wall-clock, one process with 2 threads, on a shared CPU at 100% load from other agents |

**PLAN §3 "after Stage 2" conditions:**

- (i) passes G1–G5;
- (ii) round-trips exactly;
- (iii) its HF-native encoding reproduces the reference encoder on 100% of dev documents.

All three hold. A10 therefore stays in as an **HF-native** candidate and keeps tie-breaker (a) of PLAN §7.

---

## 2. The algorithm as implemented

The paper's Algorithm 1, line by line, and what this code does (`mingram.py`, `train_mingram.py`).

| paper (Algorithm 1) | here |
|---|---|
| input: corpus *C*, target size *n*, overshoot *f*, *N_em* | `train_D1` (16,015 documents, 45.0 MB, hp.normalize 1.0.1; normalize() re-applied, 0 documents changed). *n* = 15,812 learned tokens (§3); *f* = 1.15; *N_em* = 2 |
| 1: train BPE on *C* to size ⌈f·n⌉ | HF `BpeTrainer`, character-level (no ByteLevel), P1 `Split`, `min_frequency=2`, no special tokens. It learns exactly **⌈1.15 × 15,812⌉ = 18,184** merges on top of the 251-character alphabet (the ceiling is computed in integers). Time: 20–25 s |
| 2: initialise *V* and log p from the BPE vocabulary and token frequencies | *V* = 251 characters + 18,184 BPE tokens. log p(t) = log(count(t)/Σ count), where count(t) is t's frequency when the BPE seed encodes the 6,651,777 train pretokens (7,474,115 tokens) |
| 3–7: *N_em* times: encode every sequence as s* = argmin (\|s\| − δ Σ log p(s_j)); add the tokens of s* to the counts | "sequence" = P1 pretoken. The 233,337 pretoken types are each segmented once and weighted by their frequency. The objective is **lexicographic**: fewest tokens first, then the largest Σ log p. That is the paper's δ → 0 limit, and it is exact because the scores are integers (§4 item 5) |
| 8: re-estimate log p by normalised frequencies | log p(t) = log(count(t)/Σ count) over the whole working vocabulary |
| 9: prune to \|V\| = n by lowest log p, preserving atomic tokens | one flat step: the 2,372 learned tokens with the lowest final log p are dropped. Atomic tokens (the 251 characters and the 256 byte pieces) are never candidates |
| inference: fewest tokens, Unigram score as tie-break | the same lexicographic Viterbi per P1 pretoken. HF-native as a stock Unigram with scores −C + log p (§5) |

---

## 3. Vocabulary layout (16,384 ids)

| ids | count | what |
|---|---|---|
| 0–63 | 64 | the PLAN §2.1 special block: `<\|endoftext\|>`, `<\|bos\|>`, `<\|pad\|>`, `<\|im_start\|>`, `<\|im_end\|>`, `<\|reserved_0..58\|>` (special AddedTokens, `normalized=false`) |
| 64 | 1 | `<unk>` (HF Unigram requires an unk id; with byte_fallback it is never emitted: 0 UNK tokens on dev) |
| 65–320 | 256 | `<0x00>`…`<0xFF>` byte-fallback pieces |
| 321–571 | 251 | the atomic alphabet: **every character that occurs in train_D1** (character coverage 1.0) |
| 572–16,383 | 15,812 | learned tokens, sorted by descending score |

**Learned tokens (report only):**

- 11,274 of them (71%) are word-initial, i.e. they start with a space;
- length 2–14 characters, mean 4.75;
- 19 contain a digit; 210 contain a Latin letter; none is whitespace-only.

---

## 4. Decisions the paper leaves open, and deviations (each recorded, most measured)

1. **Route (b), no reference code.** This is a reimplementation from the paper text only. There is no check against the
   author's implementation, so every item below is my reading.
2. **What *n* counts.** *n* = learned, non-atomic tokens. This follows the paper, which counts its 32,768 learned tokens
   on top of the atomic tokens. *n* = 16,384 − 64 − 1 − 256 − 251 = **15,812**. The seed has ⌈1.15·n⌉ = 18,184 learned tokens.
3. **Atomic alphabet.** The paper uses SCRIPT encoding with character-boundary constraints as its atomic alphabet. Here
   (as PLAN A10 specifies) the atomic set is the 251 train characters plus 256 `<0xNN>` byte-fallback pieces. A character
   never seen in training is encoded as its UTF-8 bytes.
4. **Zero counts.** The paper does not say what log p a token with count 0 gets. Here it gets the floor log(0.5/Σ count),
   below every counted token. The token remains usable by the min-token objective (as in the lexicographic reading),
   and it has the lowest tie-break score of any token. Zero-count learned tokens: 25 after the BPE-seed encoding, 75 after EM pass 1, 89 after
   pass 2. **None survives the prune** (the prune boundary is at count 37).
5. **Integer scores.** log p is quantised to multiples of 2⁻⁹ nats (`score_q`), in training and at inference. The reference
   Viterbi then compares (token count, Σ score_q) in exact integer arithmetic. The grid is 2⁻⁹ rather than something finer
   because of the JSON float issue explained in §5.
   - Report-only effect: with the same log p on a 2⁻²⁰ grid, **1 of 194,101** dev_strict pretoken occurrences
     (1 of 18,751 types) is segmented differently.
   - Token counts are identical either way.
6. **Exact-tie rule (deviation).** The paper's implementation "prefers the segmentation with longest leading tokens". The
   stock HF Viterbi cannot express that rule. Its rule is to keep the first path found left to right, which prefers the
   longest *final* token.
   - The reference encoder and the training E-step use HF's rule, so that the shipped HF file is exactly the model.
   - The paper's rule is implemented too (`Lattice.best_paper`) and measured, report only:
     - At inference it segments **2 of 836** dev_strict documents differently (5 of 1,358 in dev_permissive). The
       token counts are identical in every document.
     - Used in training (run C, `runs/mingram_P1_D1_16384_tiepaper`), it changes **1 of 15,812** learned tokens.
       `الس` (count 38) is replaced by ` پج` (count 37); both sit at the prune boundary. 127 of 16,384 scores differ.
7. **Prune ties.** The flat prune ranks tokens by final log p, i.e. by final count. The boundary falls inside a group of
   **212 tokens with count 37**. The ties are broken, in order, by:
   - (a) the count after the previous EM pass;
   - (b) the BPE-init count;
   - (c) the later BPE merge first.

   Lowest is dropped first. The paper does not specify a tie-break.
8. **No re-estimation after the prune** (as in Algorithm 1). The shipped log p are the final EM pass's values over the
   un-pruned working vocabulary. The kept tokens hold 99.29% of that mass.
   - Renormalising would shift every token by the same constant, so it could not change any encoding. It was not done.
9. **Seed trainer details.** HF `BpeTrainer` (not the paper's BPE), `min_frequency=2`, trained on whole documents
   through the P1 pre-tokenizer. The seed file is kept (`seed_bpe.json`).
10. **`<unk>` inside the 16,384.** One slot of the learned budget goes to `<unk>`, so that the total stays a multiple of 64
    with the fixed special block.

---

## 5. HF-native export and the encoder-equivalence test

**Export.** Every piece gets the float score −C + score_q/2⁹ with **C = 2¹⁹**. Special, `<unk>` and byte pieces get the
minimum learned/character score, so that HF's unknown-character score (min − 10) is defined by it. With these scores
the stock HF Unigram Viterbi, which maximises the summed score, behaves as follows:

- **Fewest tokens first.** For any pretoken whose shortest segmentation has fewer than
  C / |min node score| = **20,304 nodes**, HF minimises the token count before anything else. The longest pretoken is
  93 characters in train_D1 and 16 in dev_strict.
- **Exact arithmetic.** C is a power of two and the scores sit on a 2⁻⁹ grid, so every path sum is an exact double.
  HF's float comparisons are therefore exactly the reference's integer lexicographic comparisons, including exact
  ties, which both break with the same left-to-right rule.

**A pitfall found and fixed (measured).** The first export used a 2⁻²⁰ grid with C = 2¹⁷. The file was written
correctly, but `tokenizers` parsed **353 of 16,384 scores 1 ulp off**: serde_json's default float parser is exact only
for decimal significands below 2⁵³, and those scores had 17 significant digits. The shipped grid has at most 6 integer
and 9 decimal digits (15 significant digits). `check_hf_scores` verifies bit-exact round trip of all 16,384 scores
after `Tokenizer.from_file` (0 mismatches), and training asserts it.

**Equivalence test** (`equiv_test.py` → `runs/mingram_P1_D1_16384/equivalence.json`). It compares HF
`Tokenizer.from_file(...).encode(text, add_special_tokens=False).ids` with the pure-Python reference
`MinGramRef(tie='hf')`, document by document:

| set | documents | identical ids | tokens (HF = ref) | G1 round trip HF / ref |
|---|---:|---:|---:|---:|
| **dev_strict (100%)** | **836** | **836** | 213,512 | 836 / 836 |
| dev_permissive (100%) | 1,358 | 1,358 | 414,765 | 1,358 / 1,358 |
| dev_strict, harakat removed | 836 | 836 | 212,660 | 836 / 836 |
| dev_strict, digit script swapped | 836 | 836 | 213,513 | 836 / 836 |
| dev_strict, space before ۔/، toggled | 836 | 836 | 221,692 | 836 / 836 |
| dev_strict, ZWNJ inserted (unseen character → byte fallback) | 836 | 836 | 214,577 | 836 / 836 |
| synthetic stress: special-block strings in text, 9 unseen characters (CJK, emoji, U+10348, Devanagari, ZWNJ…), runs of every character ×2…×40, whitespace runs, 3,000 random piece concatenations | 4,517 | 4,517 | 31,171 | 4,517 / 4,517 |

**Independent min-token check.** For all 194,101 dev_strict pretokens, the reference path length equals a separate
count-only dynamic programme. There were 0 failures.

**Toy test** (`test_lattice.py`, log in `logs/test_lattice.log`). Two random 3-letter vocabularies: one with spread-out
scores, one with only two score values, which gives 37 exact-tie strings out of 200. The lattice matched brute-force
enumeration in every case, including the paper tie rule, with 0 failures. The stock HF Unigram matched `best_hf` on
2 × 2,000 random strings, with 0 mismatches.

---

## 6. Training trace (`runs/mingram_P1_D1_16384/train_log.json`)

| pass | what | train tokens | learned tokens with count 0 |
|---|---|---:|---:|
| 0 | BPE seed (251 chars + 18,184 learned) encodes train | 7,474,115 | 25 |
| 1 | hard EM, min-token path | 7,441,563 | 75 |
| 2 | hard EM, min-token path | 7,441,563 | 89 |
| prune | 18,184 → 15,812 learned (2,372 dropped; boundary count 37) | 7,492,889 | 0 |

**Observations:**

- **EM does not change the training token count** (7,441,563 in every pass). With the working vocabulary fixed, the
  minimum number of tokens per pretoken depends only on the vocabulary, not on the scores.
  - EM only moves counts between equally short segmentations.
  - That decides which tokens reach the bottom of the ranking, and so which ones the flat prune removes.
- **Cost of the prune.** It costs +0.69% train tokens against the un-pruned working vocabulary. At 15,812 learned
  tokens, the final model needs only 0.25% more train tokens than the 18,184-token BPE seed (7,492,889 vs 7,474,115).
- **Timings** (s): load + normalise 19.4; seed BPE 24.5; seed counts 5.8; EM passes 9.0 / 9.1; final train encode 10.7.

**N_em ablation.** Report only; it selects nothing, because PLAN fixes N_em = 2. `--ablation` prunes after k passes:

| N_em | dev_strict bytes/token | train bytes/token | learned tokens shared with the N_em = 2 vocabulary |
|---:|---:|---:|---:|
| 0 (prune on BPE-init log p) | 6.7623 | 5.9976 | 15,257 |
| 1 | 6.7696 | 6.0060 | 15,725 |
| **2 (candidate)** | **6.7702** | **6.0067** | 15,812 |
| 3 | 6.7702 | 6.0068 | 15,786 |

This matches the paper's report that gains stop after about 2 iterations (its Table 6).

---

## 7. Hard gates and reported properties (harness, `results/dev_strict/A10-mingram-P1-D1-16k/summary.json`)

| gate | result |
|---|---|
| G1 lossless | **pass**: 836/836 dev_strict documents; 0 UNK; 0 characters lost or added. Also 1,358/1,358 dev_permissive (equivalence test) |
| G2 self-tokenization | **pass**: 16,063 learned ids tested, 0 failures. By construction: a token's own string always has the 1-token path, which min-token encoding cannot beat. So the G2 remedy was not needed |
| G3 special block atomic | **pass**: all 64 present, atomic alone and in context; 0 marker strings in train/dev text |
| G4 determinism | **pass**: an independent retrain (`runs/mingram_P1_D1_16384_g4`, same command, pretoken counts recomputed from text instead of the cache) is **byte-identical** in every output: tokenizer.json, mingram_model.json, seed_bpe.json, vocab.tsv |
| G5 no sub-character tokens | **pass**: 0 (only the 256 `<0xNN>` pieces are sub-character, by design) |

**R1: support profile.** Measured on train_D1 as the harness encodes whole documents: 7,492,889 tokens, identical to
the training-side count.

| subset | tokens | freq = 0 | freq < 20 | freq < 100 | median train freq |
|---|---:|---:|---:|---:|---:|
| all "learned" in the harness definition (learned tokens + the 251 characters) | 16,063 | 0 | 66 (0.41%) | 7,868 (49.0%) | 102 |
| multi-character learned tokens (text minus surrounding spaces longer than 1 character) | 15,679 | **0** | **0** | 7,750 (49.4%) | 101 |

- The 66 tokens under 20 are all **single characters** of the atomic alphabet, with train frequency 1–19: Vietnamese
  vowels, `Δ`, `©`, Devanagari `थ`, `。`, and so on. They exist because the alphabet keeps every train character.
- For comparison, PLAN §4.3's pilot values: A1 byte-level BPE-16k had 404 learned tokens < 20; SP-Unigram-16k had 51
  multi-character pieces < 20. That was a different, non-disjoint split, so the numbers are indicative only.
- The final EM counts of the kept learned tokens range from 37 (minimum) to a median of 99.

**R2:** 0 partial-UTF-8 tokens.

---

## 8. Harness metrics on dev_strict (836 documents, 1,445,513 bytes; sha256 `b6553953…`)

**Overall:**

| metric | value |
|---|---|
| bytes/token | **6.7702** |
| chars/token | 3.7956 |
| bytes/token, each line encoded alone | 6.9607 |
| fertility (tokens overlapping a word, in context) | 1.2000 |
| tokens per word | 1.2397 |
| continued-word rate / STRR | 15.68% / 84.32% |
| Rényi efficiency α = 2.5 / α = 2.0 (report only) | 0.4977 / 0.5432 |
| vocabulary used on dev | 11,442 of 16,384 (69.8%) |
| encode speed (HF, 1 thread, shared CPU at 100% load; sanity only) | 301 documents/s, 0.52 MB/s |
| NSL vs A1-P1-16k | pending: A1-P1-16k does not exist yet. Add it with `harness.py nsl --ref <A1 dir> results/dev_strict/A10-mingram-P1-D1-16k` |

**By source:**

| source | documents | bytes | bytes/token | chars/token | fertility | STRR |
|---|---:|---:|---:|---:|---:|---:|
| book | 539 | 878,168 | 6.529 | 3.666 | 1.204 | 83.5% |
| newspaper | 280 | 532,872 | 7.294 | 4.078 | 1.181 | 86.6% |
| web | 17 | 34,473 | 5.786 | 3.258 | 1.368 | 72.8% |

**By language_variety:**

| variety | documents | bytes | bytes/token | fertility |
|---|---:|---:|---:|---:|
| hindko | 751 | 1,405,500 | 6.796 | 1.199 |
| mixed | 83 | 39,031 | 5.993 | 1.229 |
| no_signal | 2 | 982 | 5.115 | 1.510 |

**Robustness (PLAN §4.2, report only):**

| perturbation | documents affected | relative token change | extra tokens per affected word | segmentation change (affected words) |
|---|---:|---:|---:|---:|
| harakat removed | 688 | −0.40% | −0.118 | 11.0% |
| digit script swapped | 268 | +0.0005% | +0.001 | 0.0% |
| space before ۔/، toggled | 633 | +3.83% | +1.047 | 2.6% |
| ZWNJ inserted in compounds | 159 | +0.50% | **+3.79** | 68.3% |

The ZWNJ cost is structural: train_D1 contains no ZWNJ, so U+200C is not in the alphabet, and each one becomes 3 byte
tokens.

**Morphology.** SILVER sets, report only, not predictive of LM quality (PLAN §4.2). Words are encoded in context after
`۔ `.

| set | words | boundary P / R / F1 | MorphScore | stem intact | stem boundary respected | single-token rate |
|---|---:|---|---:|---:|---:|---:|
| silver_high | 518 | 0.833 / 0.074 / 0.135 | 0.804 | 98.6% | 6.8% | 91.1% |
| silver_low | 236 | 0.695 / 0.160 / 0.260 | 0.656 | 93.6% | 16.9% | 74.2% |

MinGram at 16k keeps 91% of the silver_high words whole. Among the words it splits, 80% have the gold stem|suffix
boundary among their token boundaries (MorphScore).

---

## 9. Context (report only; nothing here selects or ranks A10)

**Local char-BPE reference built under identical conditions** (`local_ref_bpe.py`). This is **not** a PLAN candidate and
**not** the Stage-1 A1/A2 artefact. It is HF `BpeTrainer`, character level, P1, train_D1, `min_frequency=2`, the same
251-character alphabet, byte_fallback, the same 64-token special block, and 16,384 in total (15,813 merges). Its sha256
is `85d4dde8…7c3d`. It went through the same harness (`results/dev_strict/LOCALREF-charbpe-P1-D1-16k/`) and passes
G1, G2, G3 and G5; G4 was not run.

| dev_strict (836 documents) | MinGram (A10) | local char-BPE |
|---|---:|---:|
| tokens | **213,512** | 215,390 |
| bytes/token | **6.770** | 6.711 (MinGram uses 0.87% fewer tokens) |
| book / newspaper / web bytes/token | 6.529 / 7.294 / 5.786 | 6.472 / 7.232 / 5.739 |
| fertility / STRR | 1.200 / 84.3% | 1.211 / 83.8% |
| train_D1 tokens | 7,492,889 | 7,563,086 (−0.93% for MinGram) |
| documents where MinGram needs fewer / more / equal tokens | 555 / 62 / 219 | |
| bootstrap clusters (32) where MinGram needs fewer tokens | 28 | |
| multi-character learned tokens with train frequency < 20 (R1) | **0** | 468 (all intermediate merge nodes; 21 at frequency 0) |
| harakat removal: relative token change / affected words re-segmented | −0.40% / 11.0% | −0.44% / 12.2% |
| silver_high MorphScore / stem boundary respected | 0.804 / 6.8% | 0.620 / 6.0% |
| learned vocabulary overlap | 14,360 shared (Jaccard 0.83) | |

- This reproduces, on Hindko at 16k, the two properties the paper reports: slightly better compression than BPE, and
  almost no rare learned tokens (paper Table 4: 9 rare tokens for MinGram vs 172 for BPE at 32k, English).
- Whether it translates into lower bits-per-byte is exactly what Stage 3 measures. The paper's own LM gap was 0.22%,
  below this study's detectable Δ_min of 0.5% (PLAN §6).
- The MinGram vocabulary keeps 14,359 of the seed's first 15,812 BPE merges. 1,453 kept tokens come from seed merges
  ranked beyond 15,812 (the highest kept rank is 18,179).

**External tokenizers on the same dev_strict** (`eval/BASELINES_DEV.md`). A 16k tokenizer trained on Hindko is expected to
beat general-purpose vocabularies on Hindko text, so these rows are context, not a claim.

| tokenizer | vocab | bytes/token | A10 needs … fewer tokens |
|---|---:|---:|---:|
| roberta-urdu (best lossless external) | 52,000 | 5.771 | 14.8% |
| bloom | 250,680 | 5.215 | 23.0% |
| gemma-3 | 262,145 | 4.640 | 31.5% |
| gpt-4o (o200k) | 200,000 | 4.384 | 35.2% |
| qwen-3 | 151,669 | 2.684 | 60.4% |
| llama-3 | 128,256 | 2.736 | 59.6% |
| hindko-probe-bpe32k: **LEAKY** (trained on dev and test), never a competitor | 32,000 | 7.529 | — |

---

## 10. Files (all under `F:\Hindko\_tokenizer\candidates\mingram\`)

| file | what | sha256 |
|---|---|---|
| `runs/mingram_P1_D1_16384/tokenizer.json` | **the candidate** (HF-native) | `a7df59a194ea3f2741c0b1d928e5f877a4e94210d81e60b5f7bf0e5c14c3d2ec` |
| `runs/mingram_P1_D1_16384/mingram_model.json` | reference model: every piece with kind, integer score, unquantised log p, final EM count, BPE-init count, count after each pass, BPE merge rank; all parameters | `759c88c877f4fc4b5c979ea9441f2e34b327b72716e71061f4114df125868a75` |
| `runs/mingram_P1_D1_16384/vocab.tsv` | the same per token (published training frequency of every token) | `3d7844c8714ca898c2bd07e89a4e1b66b203d8cb0e460c4e717308c2b822c7ae` |
| `runs/mingram_P1_D1_16384/seed_bpe.json` | the BPE seed (18,184 learned) | `b02328298d4e22a79015a08d354796efa24e0b6182c93cc8c5bbfb19447ad36f` |
| `runs/mingram_P1_D1_16384/train_log.json` | parameters, sizes, iterations, prune, ablation, timings, input and output hashes | – |
| `runs/mingram_P1_D1_16384/equivalence.json` | encoder-equivalence results (§5) | – |
| `runs/mingram_P1_D1_16384_g4/` | G4 retrain (byte-identical outputs) | same as above |
| `runs/mingram_P1_D1_16384_tiepaper/` | sensitivity run C, paper tie rule in training (report only; not a candidate) | tokenizer.json `c145d873…7a21` |
| `runs/localref_charbpe_P1_D1_16384/` | local char-BPE context reference (§9; not a candidate), results in `results/dev_strict/LOCALREF-charbpe-P1-D1-16k/` | tokenizer.json `85d4dde87f8b15ed7955d9e9415179cf56b3a0ffff8ba0e28ed3dbe5b55e7c3d` |
| `results/dev_strict/A10-mingram-P1-D1-16k/` | harness `summary.json` + per-document `docs.jsonl` (for the PLAN §6 bootstrap) | – |
| `results/extras.json` | tie-rule sensitivity, vocabulary profile, overlaps | – |
| `mingram.py` | library: pretokenizer twin, M-step, lattice (HF and paper tie rules, count-only check), model build, HF export and score check, reference encoder `MinGramRef` (CustomAdapter-compatible) | `0cf1d05c39a547f6dd51768364ebdcebbb0ccb793bb2427937cec1569f95206f` |
| `train_mingram.py` | training driver | `be07e03f0673ff4db41068d6e3ff122edd1202edaf226ee0f2d75cfc5523f2fb` |
| `equiv_test.py`, `test_lattice.py`, `run_eval.py`, `extras.py`, `local_ref_bpe.py` | equivalence test, toy lattice test, harness runner, report-only extras, local reference | – |
| `logs/` | console logs of every run | – |

Inputs: split manifest `76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2` (verified before use);
`hp.normalize` 1.0.1 `037f3582…`; `train_D1.jsonl` `b17308aa…788b`; `dev_strict.jsonl` `b6553953…7e20`; harness.py
`31db30dd…b83a`. Software: Python 3.11.9, tokenizers 0.22.2, regex 2026.2.28.

**Reproduce:**
```
cd F:\Hindko\_tokenizer\candidates\mingram
set PYTHONIOENCODING=utf-8
python train_mingram.py --out runs/mingram_P1_D1_16384 --ablation          # 2-3 min
python train_mingram.py --out runs/mingram_P1_D1_16384_g4 --ablation --no-cache
python equiv_test.py
python run_eval.py                                                          # gates=all, G4 twin = the _g4 run
python test_lattice.py
python train_mingram.py --out runs/mingram_P1_D1_16384_tiepaper --tie paper  # report-only sensitivity run C
python local_ref_bpe.py && python run_eval.py --tokenizer-json runs/localref_charbpe_P1_D1_16384/tokenizer.json --name LOCALREF-charbpe-P1-D1-16k --g4 ""
python extras.py
```

---

## 11. What was not done, and limits

- **No LM run.** bits-per-byte (the decision metric) is Stage 3. Nothing here says A10 is better or worse than any
  other candidate.
- **NSL is pending.** A1-P1-16k, the reference, does not exist yet.
- **Nothing ran on test.**
- **Not checked against the author's code** (not downloaded, by rule). Items 2–10 of §4 are my readings of an
  underspecified text. The tie rule (item 6) is a deliberate deviation, measured at 2/836 dev_strict documents with
  identical token counts.
- **Only the pre-registered configuration was built:** 16k, P1, D1, f = 1.15, N_em = 2.
  - No 8k/32k, no D2/D3, and no MinGram-PP, which PLAN §2.1 excludes.
  - The N_em ablation is report only.
- **The exactness guarantee of the HF export has a range.** It holds for pretokens whose minimum segmentation has fewer
  than 20,304 tokens. A longer pretoken, e.g. an adversarial run of tens of thousands of characters with no whitespace
  or class change, could be encoded differently by HF than by the reference. The file would still be lossless.
- **Unseen characters cost bytes.** Characters absent from train_D1 (ZWNJ, most non-Arabic scripts) are encoded as UTF-8
  byte pieces. That is lossless, but it costs 2–4 tokens each (the ZWNJ probe above).
- **Speed numbers are sanity only.** The CPU was at 100% load from other agents during every run.
- **Not written:** `tokenizer_config.json` and the chat template, which are Stage 5 deliverables for the chosen tokenizer.
