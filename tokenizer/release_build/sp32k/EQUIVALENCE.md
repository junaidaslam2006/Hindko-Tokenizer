# R2-A4-SPnat-D2-32k: an exact HF `tokenizer.json` for the released default

Built 2026-09-27 under AMENDMENT_1 §3. The released default needs an HF `tokenizer.json` that equals its canonical encoder exactly. The canonical encoder is the SentencePiece model plus the PLAN §1.1 newline convention, as `eval/adapters.py` `SPAdapter` implements it. No test-split LM number was read or used; test_strict was used as text only.

## Verdict

- **Exact equivalence is achieved.** `tokenizer.json` (tokenizers 0.22.2, called with default arguments) and `transformers.AutoTokenizer` (5.3.0) give the canonical ids on:
  - **100% of dev_strict, dev_permissive, test_strict and train_D2**;
  - also on train_D1, the view the LM bundles were trained on;
  - **100% of every required stress category**.

  `decode(encode(x)) == x` holds in all of them.
- **No blocker.**
  - The fix changes **no encoding of dev_strict (0 of 836 documents)** and **none of test_strict (0 of 491)**. All LM results, and the test run now in progress, therefore apply to the released model unchanged.
  - The fix changes 1 dev_permissive document, 22 train_D2 documents and 29 train_D1 documents. **Every changed line is an exact Viterbi tie** under the old scores: the old and new segmentations have identical exact scores.
- **AMENDMENT_1's fallback is not triggered.** R2-A4-SPnat-D2-32k stays the released default. The MinGram-32k folder was not built.
- **One inherent limit, stated precisely (§8.1).** SentencePiece accumulates Viterbi scores in float32, so exactness is guaranteed only while every value compared on a line stays below 1024 in magnitude.
  - In the four data sets, 10 of 169,111 lines go past that bound, and all 10 are still equal.
  - One synthetic probe of 80,010 characters on a single line differs. It is not part of any required set.

  No score assignment can remove this limit for arbitrarily long lines, because no score can break the ties involved (§1). The limit applies equally to the original model.

## 0. Files (`F:\Hindko\_tokenizer\release_build\sp32k\`)

| file | what |
|---|---|
| `sp.model`, `sp.vocab` | **Canonical encoder** with tie-safe scores. The 32,768 pieces, their ids and types, and the trainer, normalizer and denormalizer specs are identical to the candidate. The effective vocabulary is still 32,763: the 5 G2-remedied pieces stay CONTROL `<\|unused_0..4\|>`. |
| `gauge.json` | The exact per-character score offset w(c) (§2.2): rule, fit and values for 212 characters. |
| `tokenizer.json` | HF export. The newline convention is implemented inside it (§4), so no custom code is needed. |
| `tokenizer_config.json`, `special_tokens_map.json` | transformers config: special tokens, ChatML chat template, no automatic BOS/EOS (§5). |
| `equivalence.json` | **Counts per set and per stress category**, verdict, and the evidence for choosing the grid. |
| `equivalence_full.json`, `exact_check.json`, `lm_view_check.json` | Full verification outputs, including examples and every changed item. |
| `build_manifest.json` | sha256 of every input and output, plus versions. |
| `stress/stress.jsonl` (86,516 items), `stress/stress_summary.json`, `stress/tie_families.json` | Synthetic stress set. |
| `scripts/` | `build_release.py`, `make_stress.py`, `verify.py`, `exact_check.py`, `lm_view_check.py`, `make_report.py` |
| `work/` | The diagnosis and grid-selection runs (`old_mismatches.json`, `k*/`, `flips_vs_k_*.json`, `gauge_fit*.py`). |

Hashes of the release files:

| file | sha256 |
|---|---|
| `sp.model` | `b91854fedcb25c9d520a444a2c902f148d0ffe65a73330d7b4483ad2d594cc92` |
| `tokenizer.json` | `49f301c52363a09a1fc1925359af3ddabd7887495de02354cfa4092cba51da41` |

The candidate's `sp.model` (`1496e9a7…`) and the old export (`617f2ba6…`) are left untouched.

## 1. What caused the old mismatch

The old export differed from the canonical encoder on 1 of 1,358 dev_permissive documents. Line by line over all four sets, the old SentencePiece encoder and the old export (exact arithmetic) disagree on **26 lines**:
- **dev_permissive:** 1 line;
- **train_D2:** 25 lines;
- dev_strict and test_strict: none.

**All 26 are the same kind of tie.** A run of three copies of a character X can be segmented as XX+X or as X+XX. The two segmentations use the same multiset of pieces, so their exact scores are equal:

| where | old SentencePiece (float32) | exact arithmetic (old HF export) |
|---|---|---|
| dev_permissive line 5786 (running score at the tie: -4143) | `۱۱` `۱` | `۱` `۱۱` |
| train_D2, 11 lines | `۔۔` `۔` | `۔` `۔۔` |
| train_D2, 7 lines | `««` `«` | `«` `««` |
| train_D2, 4 lines | `’’` `’` | `’` `’’` |
| train_D2, 1 line each | `ww` `w`, `۴۴` `۴`, `!!` `!` | `w` `ww`, `۴` `۴۴`, `!` `!!` |

The mechanism:
- **Both implementations use the same tie rule.** SentencePiece 0.2.1 (`unigram_model.cc`, `EncodeOptimized`) and HF tokenizers (`models/unigram`, `encode_optimized`) both keep the first candidate at a lattice end position, the one with the smallest start, unless a later candidate is *strictly* greater.
- **Their arithmetic differs.** SentencePiece accumulates the path score in **float32**; HF accumulates in **float64**, which is exact for these scores.
- **So SentencePiece decides the tie by rounding.** It compares `fl(fl(P+s(XX))+s(X))` with `fl(fl(P+s(X))+s(XX))`, and rounding decides the result. This happens even at a running score of P = -5.3.
- **HF applies the tie rule.** It keeps X+XX, the longer piece last.

**Consequence: the requested fix (an epsilon ordering of piece scores) cannot work here.** Both segmentations contain the same pieces, so any change of piece scores changes both sums by the same amount, and the tie can never be broken by scores. What *can* be fixed is the arithmetic: make SentencePiece's float32 sums exact, so that both encoders compare the same numbers and apply their identical tie rule.

## 2. The fix: new scores for the NORMAL pieces

For every NORMAL piece p with old score s(p):

    s'(p) = round_half_even(s(p) * 2^14) / 2^14  +  sum over the characters c of p of w(c)

- Every w(c) is itself a multiple of 2^-14.
- Non-NORMAL pieces (USER_DEFINED, CONTROL, BYTE, UNKNOWN) keep their scores, because SentencePiece keeps them out of its lattice.
- Nothing else in the model proto changes (asserted in `build_release.py`).

### 2.1 Rounding: the only part that can change an encoding

- **Size.** Each score moves by at most 2^-15 = 3.05e-5 (mean 1.53e-5; 31,957 of 32,441 scores move).
- **Effect.** A decision can flip only if the old exact score gap between two competing segmentations is smaller than the sum of the rounding moves of the pieces involved. For a 2-against-2 piece swap that is ≤ 1.2e-4.
- **Grid choice.** 2^-14 is the **coarsest grid with no such flip in any of the four sets**. The coarsest grid is wanted because it gives the widest float32-exact range. Measured changed lines outside the old tie set:

  | grid | 2^-10 | 2^-12 | 2^-13 | **2^-14** | 2^-15 | 2^-16, 2^-18, 2^-20 |
  |---|---|---|---|---|---|---|
  | changed non-tie lines (four sets) | 6 (1 in test_strict) | 5 | 4 | **0** | 0 | 0 |

  Each non-tie change at 2^-10…2^-13 was a real near-tie: the old exact gaps were 1.2e-4 to 1.9e-4, for example `▁اووُن`+`۔` against `▁او`+`وُن۔`. Rounding had made these exact ties. At 2^-14 none remain.

### 2.2 Gauge: exact, and cannot change any encoding

Every candidate compared at one lattice node covers the same characters `[0, pos)`. Adding w(c) for each character of each piece therefore shifts every candidate at that node by the same exact amount, so **no comparison, no tie and no encoding can change**. The shift is a gauge transformation of the Viterbi potentials, and it also leaves SentencePiece's sampling and n-best distributions unchanged.

Its only purpose is to keep the running path score small, so that float32 stays exact:
- **The problem.** With the raw scores (-15.9 to -3.6), path scores grow by about -1.06 per byte, reaching -13,118 in train_D2. On a 2^-14 grid, float32 is exact only below 2^10.
- **The fit.** w(c) is fitted by least squares on train_D2 only (the tokenizer's own training view). The objective is the squared sums of 8 consecutive pieces of the old best paths, so that path scores stay near 0. Examples: w(`▁`) = 3.284, w(`ا`) = 0.709, w(`۔`) = 3.945, w(`۱`) = 4.538, w(`ࢾ`) = 13.367.
- **The result.** The largest value compared anywhere in the four sets' lattices falls from about 13,118 to 1,745.
- **Side effects.** New NORMAL scores range from -14.47 to +44.38. The SentencePiece `<unk>` node score is min - 10 = -24.47.
  - The `<unk>` score never decides a comparison: every character inside a NORMAL piece has its own single-character piece (`work/coverage_check.json`), so an `<unk>` node is only ever forced.
  - **The scores are no longer log-probabilities.** Recover the rounded log-probability of a piece as `score - sum of w(c) over its characters`.

### 2.3 Why the two encoders now agree

- All scores are multiples of 2^-14, so every partial path score is too.
- **float32 represents such a value exactly while |value| < 2^24 · 2^-14 = 1024.** float64 does so while |value| < 2^39.
- **Therefore**, on any line where every value compared stays below 1024 in magnitude:
  - SentencePiece's float32 Viterbi, HF's float64 Viterbi, and exact rational arithmetic compute identical numbers;
  - both implementations use the same tie rule;
  - so they return the same path.
- **Exact ties are now resolved by that rule:** at each end position the smallest start wins. For runs this gives X+XX, the longer piece last, e.g. `(۱۱۱) «««` → `▁(` `۱` `۱۱` `)` `▁` `«` `««`.

## 3. Proof that nothing but ties changed

Old canonical encoder = SPAdapter on the candidate `sp.model`; new = SPAdapter on the release `sp.model`. For every line whose encoding changed, `verify.py` computes the exact rational old score of both segmentations:

| set | docs | docs changed | changed lines | lines that are exact ties (old gap = 0) | changed, not an exact tie |
|---|---|---|---|---|---|
| **dev_strict** | 836 | **0** | 0 | – | 0 |
| dev_permissive | 1,358 | 1 | 1 | 1 | 0 |
| **test_strict** | 491 | **0** | 0 | – | 0 |
| train_D2 | 11,066 | 22 | 25 | 25 | 0 |
| train_D1 (LM training view) | 16,015 | 29 | 45 | 45 | 0 |
| stress (86,516 items) | – | 476 | – | all | 0 |

- **The changed lines are exactly the old ties.** The dev_permissive line and all 25 train_D2 lines are the 26 lines of §1. Each now takes the exact-arithmetic answer, which the old HF export had already given.
- **Every other document keeps a byte-identical id sequence.**
- **The LM runs are unaffected.** dev_strict (the LM evaluation text) and test_strict (the final-test bundles) have 0 changed documents. The train_D1 bundles differ from the release encoder only by the order of 45 run segmentations such as `۔`+`۔۔`, all with the same token count.

## 4. `tokenizer.json`: how the newline convention is implemented

The canonical convention: split the text at `\n`, encode each line with SentencePiece (identity normalization, `' '`→`▁`, one `▁` dummy prefix, empty line → no tokens), and join the lines with the `\n` piece (id 64). In `tokenizer.json`:

- **added_tokens.**
  - `\n` (id 64: `special=false`, `normalized=false`). It is extracted from the raw text first, so every remaining segment is one line.
  - The 64-token block, ids 0..63 (`<|endoftext|>`, `<|bos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|reserved_0..58|>`), and the 5 `<|unused_k|>` CONTROL placeholders (ids 7183, 16359, 16629, 16924, 19063): all `special=true`, `normalized=false`.
- **normalizer.** `Sequence[Replace(' '→'▁'), Prepend('▁')]`, applied to each line segment. `Prepend` skips empty strings, as SentencePiece encodes `''` to `[]`. There is no Unicode normalization.
- **pre_tokenizer: none.** The Viterbi runs over the whole line, as SentencePiece's does. No NORMAL piece contains an interior `▁`.
- **model.** `Unigram` with byte_fallback and unk_id 65, and ids equal to the SentencePiece ids.
  - The scores are the release scores.
  - The non-NORMAL pieces get -2^20: they are in HF's trie, but can never win against the normal pieces SentencePiece uses. -2^20 is on the grid, so float64 stays exact.
- **post_processor: none.** `Tokenizer.encode(text).ids` with default arguments is the canonical id sequence.
- **decoder.** `ByteFallback`, `Fuse`, `Replace('\n▁'→'\n')`, then one `Replace(S+'▁'→S)` for each of the 69 special strings S (§5), then `Strip('▁', 1, 0)` and `Replace('▁'→' ')`.

## 5. Special tokens, BOS/EOS, chat template

- **Named tokens.**
  - `bos_token` = `<|bos|>` (1), `eos_token` = `<|endoftext|>` (0), `pad_token` = `<|pad|>` (2).
  - `additional_special_tokens` = `<|im_start|>`, `<|im_end|>`, `<|reserved_0..58|>`, `<|unused_0..4|>`.
  - transformers reports 69 special ids. `len(tokenizer)` = 32,768.
- **`unk_token` is left unset (null) on purpose.** transformers turns every named special token into a matchable added token. A declared `<unk>` would make the literal text `<unk>` encode as id 65, whereas SentencePiece encodes it as `<`,`unk`,`>`. With byte_fallback, `<unk>` is never produced anyway.
- **No automatic BOS/EOS.** `tokenizer.json` has no post-processor, and `tokenizer_config.json` sets `add_bos_token=false` and `add_eos_token=false`.
  - transformers ignores these two keys when a `tokenizer.json` exists, and then installs an identity template.
  - So `tokenizer(text)`, `tokenizer.encode(text)` and `Tokenizer.encode(text)` all return exactly the canonical ids (verified).
  - The **LM convention** (`colab/build/staging/hk_lm.py`) is to add them explicitly: each document is `[<|bos|>] + ids + [<|endoftext|>]`.
  - To add BOS automatically at runtime, set `tokenizer.add_bos_token = True`.
- **Chat template (ChatML with a leading BOS).**

  ```
  {{ bos_token }}{% for m in messages %}<|im_start|>{{role}}\n{{content}}<|im_end|>\n{% endfor %}[<|im_start|>assistant\n]
  ```

  This follows the LM's `[BOS] document` convention. The following are all verified:
  - `apply_chat_template(..., tokenize=True)` equals `tokenizer.json` on the rendered text;
  - it equals the split-canonical encoding below;
  - decode is exact.
- **Text that contains literal special-token strings.**
  - HF extracts them as special tokens, as standard HF does. Each text segment between them is then encoded as a line.
  - So HF equals **split-canonical**: cut out the special strings, encode each segment with the canonical encoder, and insert the special ids. Verified on **626/626** stress items, with exact decode on 626/626.
  - Plain SentencePiece applied to the same raw string differs: it also emits `▁` before a line-initial special, and gives no dummy prefix to text right after a special. It equals HF on only 128 of the 626 items.
  - **Recommendation:** insert special tokens as ids, or via the chat template, never as text through the plain canonical encoder.
  - The corpus contains no such strings. `|` is not in the vocabulary at all (byte fallback), so none of these strings can arise from a normal segmentation.

## 6. Equivalence results

### 6.1 Data sets (per document)

Columns:
- **hf ids =**: `Tokenizer.from_file(tokenizer.json).encode(doc).ids` with default arguments.
- **transformers ids =**: `AutoTokenizer(doc, add_special_tokens=False)`.
- **decode exact**: counted for HF, transformers and canonical alike.

| set | docs | canonical tokens | hf ids = canonical | transformers ids = canonical | decode exact | lines | independent exact Viterbi = SentencePiece (lines) | lines within float32 bound |
|---|---|---|---|---|---|---|---|---|
| dev_strict | 836 | 204,272 | **836** | **836** | 836 | 7,662 | 7,662 | 7,662 |
| dev_permissive | 1,358 | 390,460 | **1,358** | **1,358** | 1,358 | 13,733 | 13,733 | 13,730 |
| test_strict (text only) | 491 | 199,390 | **491** | **491** | 491 | 4,118 | 4,118 | 4,117 |
| train_D2 | 11,066 | 3,686,994 | **11,066** | **11,066** | 11,066 | 143,598 | 143,598 | 143,592 |
| train_D1 (LM view, extra) | 16,015 | – | **16,015** | **16,015** | 16,015 | – | – | – |

**Independent exact Viterbi** (`exact_check.py`): a pure-Python lattice that mirrors `EncodeOptimized`. Its values are exact because they sit on the 2^-14 grid, and it records the largest magnitude of every value compared.
- It agrees with SentencePiece on **169,111 of 169,111 lines**, so three implementations agree.
- The largest compared magnitude is 683 in dev_strict and 1,745 overall.

### 6.2 Stress set (86,516 items, seed PCG64(20260927), `make_stress.py`)

Required categories:

| category | items | hf = canonical | transformers = canonical | decode exact | contents |
|---|---|---|---|---|---|
| newlines | 419 | 419 | 419 | 419 | `''`, runs of 1–10 `\n`, leading/trailing/empty lines, `\n \n`, `\r\n`, `\r`, 400 random mixes |
| spaces | 209 | 209 | 209 | 209 | 1–8 leading/trailing/inner spaces, only spaces, tab, VT, FF, NBSP, U+1680, U+2000–200A, U+2028/2029, U+202F, U+205F, U+3000, U+FEFF |
| digits | 1,133 | 1,133 | 1,133 | 1,133 | ASCII, Extended Arabic-Indic, Arabic-Indic: single digits × runs of 1–16, random numbers of 1–40 digits, separators |
| latin | 1,090 | 1,090 | 1,090 | 1,090 | English sentences, URL, e-mail, code, accented Latin, every printable ASCII character alone, in context and ×3, letter runs 1–12 |
| codepoints | 2,520 | 2,520 | 2,520 | 2,520 | **every code point in any data view** (253, including U+0095 and U+FFFD); each ×10 contexts (alone, spaced, ×2, ×3, ×7, inside and around words and sentences) |
| zwnj | 622 | 622 | 622 | 622 | ZWNJ/ZWJ alone, doubled, word-initial, word-internal and word-final, around spaces, and 600 random insertions into dev lines. ZWNJ does not occur in the corpus. |
| tone_letters | 538 | 538 | 538 | 538 | U+08BE–U+08C2 and U+0768: alone, ×2, ×3, in words, with harakat and ZWNJ; decomposed base+U+065A; the 400 most frequent tone-letter words, alone and in sentences |

Additional categories:

| category | items | hf = canonical | transformers = canonical | decode exact | notes |
|---|---|---|---|---|---|
| ties | 2,821 | 2,821 | 2,821 | 2,821 | runs X^2…X^8 of all 31 pieces X for which X+X is a piece, alone and after real-text prefixes of 40–5,600 characters; 435 were old float32 ties |
| pieces | 57,898 | 57,898 | 57,898 | 57,898 | every NORMAL piece string, with and without its leading space |
| fuzz (4 kinds) | 18,000 | 18,000 | 18,000 | 18,000 | random code-point strings (uniform and corpus-frequency), random piece concatenations with newlines, shuffled dev words |
| unknown_scripts | 569 | 569 | 569 | 569 | CJK, emoji and ZWJ sequences, Devanagari, Cyrillic, Greek, Hebrew, C0/C1 controls, private use, U+FFFF, U+10FFFF, bidi marks, `\|`, `<unk>`, `<0x41>`, `[UNK]` |
| long_lines | 65 | **64** | **64** | 65 | single lines of 8k–80k characters, past the float32 bound, see §8.1 |
| u2581 | 6 | 6 | 6 | **0 (all encoders)** | literal U+2581 decodes to a space in the canonical encoder too (SentencePiece escaping); U+2581 is not in the corpus |
| specials | 626 | 626 (split-canonical) | 626 | 626 | literal special strings and chat text, §5 |

### 6.3 transformers (5.3.0, `AutoTokenizer.from_pretrained(release_build\sp32k)`)

- **Loaded object.** Class `TokenizersBackend` (`PreTrainedTokenizerFast`); 32,768 tokens.
- **Ids.** bos, eos and pad are 1, 0 and 2. `convert_tokens_to_ids('\n')` is 64.
- **Calls.** The default `__call__` and `encode` add nothing and equal the canonical ids.
- **Decoding.** `decode(..., skip_special_tokens=False)` is exact on every item above. Chat template checks are in §5.

## 7. What this means for the decision records

- **The pre-registered claims are unaffected.**
  - Every dev_strict encoding the LMs were evaluated on is identical under the release model.
  - Every test_strict encoding in the final-test bundles is identical as well.
  - So the confirm-scale, large-arbiter and pending test results of R2-A4-SPnat-D2-32k are results of the released tokenizer.
- **The release `sp.model` becomes the canonical encoder** (sha256 `b91854fe…`). The candidate's model differs from it only on exact-tie lines, 26 in the four sets. Its scores are the old log-probabilities rounded to 2^-14 plus the gauge.

## 8. Limits

1. **float32 bound.**
   - **What is guaranteed.** Equality holds whenever every value the SentencePiece Viterbi compares on a line has magnitude < 1024, in the gauge-shifted scores. `exact_check.py` measures this per line.
   - **What happens past the bound.** SentencePiece's float32 accumulator rounds. It can then resolve an *exact* tie, or a near-tie below its own float32 resolution, differently from exact arithmetic.
   - **Observed in the data.** 10 of 169,111 lines in the four sets pass the bound, the shortest at 1,054 characters, and all 10 are still equal.
   - **Observed in the stress probe.** One synthetic line differs: `long_lines-000062`, 80,010 characters with no newline. At the tie `«««` its running score is 3,868, and float32 has already drifted to 3,868.40 against the exact 3,868.47.
   - **Why no rescoring fixes it.** These ties use the same pieces in a different order, so no score change can separate them. A float32 accumulator loses exactness on long enough lines under any scoring. So universal equivalence with any float64 implementation is impossible for this SentencePiece model, as it was for the original one.
   - **Why this grid.** The chosen grid is the coarsest one that changes no non-tie encoding, which makes the bound as large as that constraint allows.
2. **U+2581 in input** is lossy in the canonical encoder itself: it decodes as a space. It is absent from the corpus.
3. **Literal special-token strings** follow §5 (split-canonical), not plain SentencePiece.
4. **Normalization.** Neither encoder normalizes anything (PLAN §1.1). Text must already be in the frozen canonical form: NFC, kashida removed.

## 9. Reproduce

    set PYTHONIOENCODING=utf-8
    python scripts\build_release.py     # sp.model, gauge.json, tokenizer.json, configs, build_manifest.json
    python scripts\make_stress.py       # stress\stress.jsonl
    python scripts\verify.py            # equivalence_full.json   (sets, stress, transformers, old->new tie proof)
    python scripts\exact_check.py       # exact_check.json        (independent exact Viterbi, float32 bound)
    python scripts\lm_view_check.py     # lm_view_check.json      (train_D1)
    python scripts\make_report.py       # equivalence.json

Environment: Python 3.11.9, sentencepiece 0.2.1, tokenizers 0.22.2, transformers 5.3.0, numpy 2.4.2. CPU, at most 3 processes. The build is deterministic: re-running `build_release.py` reproduces `sp.model` byte for byte, the same file that was evaluated as `work/k14g`.
