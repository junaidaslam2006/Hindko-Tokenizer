# A7: PickyBPE @16k, τ = 0.9 (PLAN.md Stage 2)

*PickyBPE as reimplemented from Chizhov, Arnett, Korotkova, Yamshchikov (2024), "BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training", EMNLP 2024, arXiv:2409.04599.*

- No code from the authors' repository (`pchizhov/picky_bpe`) was downloaded or run.
- The algorithm was written from the paper's description; §2 lists the points the paper leaves implicit.

Written 2026-09-26. Every number below was measured in this folder. No number was computed on the test split.

---

## 0. Result in one paragraph

A byte-level PickyBPE tokenizer with the P1 pre-tokenizer was trained on D1 train at 16,384 tokens (64 specials + 256 bytes + 16,064 learned).
- **τ = 0.9 stands.** Training made 276 removals; 24 of the removed tokens came back later, so 252 tokens are removed at the end. That is 1.57% of the learned slots, above the 1% at which PLAN switches to 0.8.
- **Gates.** It passes G1–G4. G5 does not apply to a byte-level vocabulary. No removed token appears in any dev encoding, on either dev view.
- **The event-ordered encoder** reproduces the trainer's final segmentation of all 233,337 training pretoken types.
- **Plain-BPE check.** The same code with removals switched off gives exactly the merges and ids of the official A1-16k.
- **Support profile (R1) against A1-16k.**
  - Learned tokens seen < 20 times in train: **447 → 249 (−44%)**.
  - Never-seen tokens: **25 → 0**.
  - Partial-UTF-8 tokens: 9 → 5, and none of the 5 has train frequency 0 (A1: 4 do).
  - So there are far fewer under-trained tokens, but they are **not eliminated**. 245 of the 249 remaining ones are still intermediate merge-graph nodes (§6).
- **Compression on dev_strict** is marginally better: 6.7188 vs 6.7117 bytes/token (NSL 0.99896).
- **Shipped:** a custom Python encoder only.
- **Not shipped: an HF `tokenizer.json`.** Exact equivalence on 100% of dev is **provably impossible** with this vocabulary (§8).

---

## 1. Deliverables

| what | file | sha256 |
|---|---|---|
| **PickyBPE-16k model** (tokens + training events; read by the encoder) | `models/pickybpe_P1_D1_16k_tau0.9.json` | `1508d094532db24ca3d74704d8be91ba126e2e6f2f648ee3c29c6fcbce4dd6cc` |
| encoder + trainer | `pickybpe.py` | in `MANIFEST.json` (also recorded inside the model's `meta.code_sha256`, checked equal) |
| loader for the harness and the LM bundle | `pickybpe_factory.py` → `load_16k()` (refuses a model file whose sha256 differs) | in `MANIFEST.json` |
| A1-16k reference (HF), byte-identical to Stage 1's `candidates/standard/tok/A1-P1-D1-16k/tokenizer.json` | `models/a1_ref_P1_D1_16k.json` | `4d018f829024928aa03e1b8ba7463a43f6f66ef0103a4384541c76713c552490` |
| harness results (dev_strict) | `results/dev_strict/A7_pickybpe_P1_D1_16k_tau0.9/{summary.json,docs.jsonl}`, `results/dev_strict/A1ref_bpe_P1_D1_16k/…` | in `MANIFEST.json` |
| verification outputs | `verify/*.json` | in `MANIFEST.json` |
| every file with its sha256 | `MANIFEST.json` | — |

**Using it:**

```python
import sys; sys.path.insert(0, r"F:\Hindko\_tokenizer\candidates\pickybpe")
import pickybpe_factory as F
tk = F.load_16k()                  # checks the pinned sha256
ids = tk.encode(text)              # no BOS/EOS added; special-token strings are matched first, as HF added tokens are
text == tk.decode(ids)             # lossless on the canonical form
```

- Ids 0–63 are the PLAN special block (`<|endoftext|>` = 0, `<|bos|>` = 1, …).
- Ids 64–319 are the 256 bytes, in HF ByteLevel alphabet order; they are the same ids as in A1.
- Ids 320–16383 are the learned tokens.
- `tk.id_to_token(i)` gives the GPT-2 byte-level string, and `tk.token_bytes(i)` the raw bytes.

**Stage 3 (LM) entry for `lm\WAVES.json`:**
- entry: `{"id": "A7-pickybpe-P1-D1-16k-tau0.9", "tokenizer_path": "F:\\Hindko\\_tokenizer\\candidates\\pickybpe\\models\\pickybpe_P1_D1_16k_tau0.9.json", "encoder_kind": "custom", "custom": "pickybpe_factory:load_16k"}`;
- before running `build_bundle.py`, `set PYTHONPATH=F:\Hindko\_tokenizer\candidates\pickybpe`.

That path was smoke-tested:
- `build_bundle.load_encoder(..., "custom", "harness", custom_spec="pickybpe_factory:load_16k")` gives n_vocab 16,384, EOT id 0, BOS id 1, and an exact round trip.
- Train bytes/token = 45,007,515 / 7,550,428 = **5.961** (A1: 5.955), so ctx_tokens = round(1536 / 5.961) = 258, the same as A1.
- Encoding train_D1 took 116 s single-threaded with a cold cache (§9).

---

## 2. The algorithm as implemented (`pickybpe.py`)

**Training** runs on the P1 pretoken types of D1 train (233,337 types, 6,651,777 pretokens, 45,007,515 bytes), as byte sequences with counts.

1. **Start state.** Ids 0–63 are the specials and ids 64–319 the 256 bytes, sorted by their GPT-2 byte-level character, which is HF's alphabet order.
2. **Each step.** Pop the most frequent adjacent pair (a, b).
   - Pair counts are over all adjacent positions, as in HF.
   - Ties go to the smallest (id_a, id_b), as HF `BpeTrainer`.
   - Create c = a+b, or re-activate it if it was removed earlier, and merge every occurrence left to right. → **MERGE event.**
3. **Removal check.** Then, for x = a, and for x = b if b ≠ a:
   - compute IoS(x | a, b) = f_p(a, b) / f_t(x), with both frequencies taken **before** the merge, and f_p = the number of merges made;
   - if x is a learned token, is still present and has IoS ≥ τ, remove it and split its remaining occurrences. → **REMOVE event.**
   - Bytes and specials are never removed.
4. **Stop** when the number of *present* tokens reaches 16,384.

**Encoder** ("event-ordered", the paper's Algorithm 2), per pretoken:
- start from the bytes and repeatedly apply the earliest training event, later than the last one applied, that can act on the current sequence;
- a merge event merges all its pairs left to right, and a removal event replaces the token by its stored split;
- results are cached per pretoken.

Special-token strings are split out first, exactly as HF added tokens are. Decoding concatenates token bytes.

**Confirmed against the paper text** (arXiv HTML, read 2026-09-26):
- the IoS definition;
- both parts of a merge are checked, the second only if x₂ ≠ x₁;
- alphabet symbols are never removed;
- removed tokens may be merged again later ("Any removed token may be merged again…");
- inference greedily picks the earliest possible event in training order.

**Reimplementation choices where the paper is implicit.** Each is recorded in the model file:

| point | choice | why |
|---|---|---|
| what a removed token's remaining occurrences become | `walk(x)`: x's most recent creating pair, each part expanded recursively until it is a present token (bytes always are). The split is stored in the REMOVE event, so the encoder replays it exactly. | Any split that spells the same bytes keeps the tokenizer lossless; storing it makes training and encoding identical. The 276 removals split into 2 tokens (270 times) or 3 (6 times). |
| when f_t and f_p are measured | before the merge | Only then is IoS an "intersection over self" in [0, 1]. Measured after the merge, f_t(x) would be the remainder and IoS could exceed 1. |
| stopping rule | present-token count = target (16,384) | PLAN needs exactly 16,384 (64 specials, padded to a multiple of 64). The paper claims a vocabulary "of the desired size". This needs 16,340 merges at τ = 0.9. |
| IoS when a = b | literal formula (merges / f_t), checked once | Each such merge consumes 2 occurrences, so IoS ≤ 0.5: such tokens are never removed at τ = 0.9. |
| a merge whose product already exists | re-use the id (re-activate it if removed) | This is what HF does for duplicate strings. It happened 24 times at τ = 0.9 (re-additions), plus once in the report-only τ = 0.7 run. |
| a and b both qualify for removal | remove a, then b | deterministic |
| `min_frequency` | 2, as A1 | It never binds: the last merge had pair count 42. |

**Faithfulness checks of the implementation:**
- `test_pickybpe.py` (`logs/test_pickybpe.log`):
  - The encoder equals the trainer's final state on a 1,500-document train slice at τ = 0.6 (335 removals, 55 re-additions) and at τ = 0.9.
  - With τ = None, 3,776 merges and ids are identical to HF `BpeTrainer` on the same slice.
- On the full data:
  - V1 of `verify_pickybpe.py`: 0 of 233,337 train types differ between encoder and trainer, and the train token frequencies are identical.
  - V4: with removals switched off, the same code gives **exactly** the 16,064 merges and all ids of the HF A1 reference. PickyBPE therefore differs from the baseline *only* through its removals.

---

## 3. Inputs (verified before use; the test split is never read)

- **Split manifest:** sha256 `76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2`. It is checked against `FROZEN.json` and against the hash this task names, on every run (`pickybpe_common.verify_frozen`).
- **`hp/normalize.py` 1.0.1:** sha256 `037f3582…`, checked against `FROZEN.json`.
- **Train data:** `data/train_D1.jsonl` (16,015 documents, sha256 `b17308aa…`, matching `data/data_manifest.json`).
  - Every uid is checked to be in the `train` split.
  - Every text is a fixed point of `normalize()` (16,015/16,015). The canonical form is therefore the training form.
- **Dev data:** `data/dev_strict.jsonl` (836 documents, sha256 `b6553953…`) and `data/dev_permissive.jsonl` (1,358 documents). Every uid is checked to be `validation`.
- **Pre-tokenizer P1:** the Python `regex` twin of the Oniguruma pattern. It is identical to the pair that the harness differential test (`eval/pretok_differential.json`) found to give identical pretokens on dev. The ZWNJ/ZWJ escapes are built without literal characters.

---

## 4. The τ decision (PLAN §2.2 rank 7: "τ = 0.9, 0.8 if 0.9 removes < 1% of tokens")

| | τ = 0.9 |
|---|---|
| removal events during training | 276 (IoS 0.90–0.92: 80, 0.92–0.94: 66, 0.94–0.96: 44, 0.96–0.98: 42, 0.98–1.0: 19, exactly 1: 25) |
| removed tokens later re-added (still present at the end) | 24 (e.g. `کومت`, the intermediate of ` حکومت`; the partial-UTF-8 token `20 d9`) |
| **removed at the end** | **252 = 1.57% of the 16,064 learned slots** (1.54% of 16,384) |
| decision | ≥ 1%, so **τ = 0.9** (0.8 not needed) |

The removed tokens are typical intermediates: `ندکو` → `ہندکو`; `احب` → ` صاحب`; `روع` → ` شروع`; `ڈمی` → ` اکیڈمی`. Six removal events hit partial-UTF-8 byte tokens (`verify/*.removals.json`):
- **Four stay removed, all with IoS = 1:** `b7 ba` inside ﷺ, the lead bytes `e0 a2`/`e0 a3` of the U+08xx Hindko letters, and `bf bd`, the tail of U+FFFD. These are exactly the four zero-frequency R2 tokens of A1.
- **Two were re-added later:** `20 d9` (space + lead byte) and `e2 80`.

---

## 5. Hard gates (PLAN §4.3), dev_strict, harness `eval/harness.py` unchanged

| gate | PickyBPE-16k τ = 0.9 | A1-16k (reference) |
|---|---|---|
| G1 lossless | **pass**: 836/836 (also dev_permissive 1,358/1,358 in V3) | pass 836/836 |
| G2 self-tokenization of whole-character tokens | **pass**: 16,059 tested, 0 failures (5 partial-UTF-8 exempt) | pass: 16,055, 0 failures (9 exempt) |
| G3 special block atomic, absent from corpus | **pass**: 64/64 atomic, 0 marker occurrences in train/dev | pass |
| G4 determinism | **pass**: retrained file byte-identical (`…retrain.json`, same sha256) | pass (byte-identical retrain) |
| G5 no sub-character tokens | not applicable: byte-level vocabulary (PLAN: G5 is for A2–A5, A8, A10). Partial-UTF-8 tokens are R2: **5**, all with train frequency > 0 | not applicable; R2 = 9, 4 with train frequency 0 |

**Task-specific checks** (`verify/pickybpe_P1_D1_16k_tau0.9.verify.json`):

| check | result |
|---|---|
| **removed tokens never appear in any encoding of dev** | **0** occurrences in dev_strict (215,146 tokens) and dev_permissive (419,376 tokens), checked on the encoder's internal ids *before* the mapping to final ids. They also never appear in the final segmentation of any of the 233,337 train types. |
| V1 encoder = trainer, all train types | 0 mismatches; train token frequencies identical |
| V4 τ = None run of the same code = HF A1 | identical merges (16,064) and ids |

The same V1, removed-token and round-trip checks also pass for the report-only τ = 0.8 and τ = 0.7 models (§9).

---

## 6. Support profile R1 against A1-16k (train_D1, the whole training split, as an LM sees it)

| | learned | train freq = 0 | < 20 | < 20 that are intermediate nodes | < 20 that are leaves | < 100 | median train freq |
|---|---|---|---|---|---|---|---|
| A1-16k (HF BPE, official Stage-1 file) | 16,064 | 25 | **447 (2.78%)** | 447 (all) | 0 | 8,125 (50.6%) | 98 |
| **PickyBPE-16k, τ = 0.9** | 16,064 | **0** | **249 (1.55%)** | 245 | 4 | 8,153 (50.8%) | 97 |

(Intermediate = a component of a merge whose product is in the final vocabulary. For PickyBPE, the last creating merge of each present token counts.)

**What changed:**
- The two vocabularies share 15,805 learned tokens.
- **A1's 259 extra tokens:** 251 of them were removed by PickyBPE. Their median train frequency in A1 is **8**, and 206 are below 20.
- **PickyBPE's 259 replacement tokens:** they are later merges with a median train frequency of **41**; 2 are below 20.

**What did not change:**
- Of A1's 447 under-supported tokens, PickyBPE removed 205 and **kept 241**, e.g. `فاظ` (12), ` یونیور` (12), ` مرک` (16).
- A token survives when it is absorbed into several longer tokens and no single merge takes ≥ 90% of its remaining occurrences, for example 50% into one word, then 88% of the rest into another. IoS is evaluated merge by merge, so τ = 0.9 misses this gradual absorption.
- PLAN §2.2 expected PickyBPE to remove "exactly the under-trained intermediate tokens … all 404 of A1-16k's". At τ = 0.9 that holds for about half of them (447 → 249). The report-only τ sweep in §9 shows that τ = 0.8 would leave 84 and τ = 0.7 would leave 9.

The share of tokens below 100 is unchanged (≈ 51%): that is the data-scarcity property of any 16k vocabulary on 7.5M train tokens (PLAN §4.3). PickyBPE does not change it.

---

## 7. Harness metrics, dev_strict (836 documents, 1,445,513 bytes; per-document rows in `docs.jsonl`)

| metric | PickyBPE-16k τ = 0.9 | A1-16k |
|---|---|---|
| tokens | 215,146 | 215,371 |
| **bytes/token** | **6.7188** | 6.7117 |
| NSL vs A1 | **0.99896** (book 0.99899, newspaper 0.99891, web 0.99867) | 1 |
| bytes/token by source: book / newspaper / web | 6.4772 / 7.2428 / 5.7513 | 6.4706 / 7.2349 / 5.7436 |
| bytes/token, hindko variety (751 docs) / mixed (83) | 6.7441 / 5.9580 | 6.7375 / 5.9408 |
| chars/token | 3.7668 | 3.7628 |
| fertility (in-context, tokens per word) | 1.2095 | 1.2108 |
| STRR (words kept as 1 token) / continued-word rate | 0.8393 / 0.1607 | 0.8380 / 0.1620 |
| bytes/token, line by line | 6.9061 | 6.8987 |
| Rényi efficiency α = 2.5 / 2.0 (report only) | 0.49936 / 0.54555 | 0.49954 / 0.54573 |
| vocabulary ids used on dev | 11,418 (69.7%) | 11,355 (69.3%) |
| robustness, relative token change: harakat removed / digits swapped / space before ۔، deleted / ZWNJ inserted | −0.449% / 0 / +3.802% / +0.350% | −0.449% / 0 / +3.798% / +0.346% |
| robustness, segmentation change among affected words: harakat / punct-space / ZWNJ | 12.02% / 34.92% / 71.89% | 12.12% / 34.92% / 69.75% |
| morphology, SILVER high (report only): boundary F1 / MorphScore / single-token rate | 0.1014 / 0.592 / 0.905 | 0.1046 / 0.600 / 0.904 |
| morphology, SILVER low: boundary F1 / MorphScore | 0.2609 / 0.612 | 0.2609 / 0.603 |
| round-trip failures / UNK | 0 / 0 | 0 / 0 |

Intrinsically, PickyBPE-16k at τ = 0.9 is A1 with about 1.6% of its vocabulary exchanged. Compression is 0.1% better, the other metrics move by less than their resolution, and the support profile is clearly better. Whether this changes bits-per-byte is the Stage 3 question (PLAN §2.2 rank 7).

---

## 8. HF `tokenizer.json`: not shipped (exact equivalence is impossible)

**The rule:** ship an HF file only with exact equivalence on 100% of dev.

**Why it cannot be met** (`hf_equivalence_analysis.py` → `verify/*.hf_equivalence.json`):
- An HF BPE model emits a non-byte token only as the product of a merge (l, r) → t, with l and r in its vocabulary.
- **156 present tokens** of the PickyBPE vocabulary have **no** split into two tokens that are both in the vocabulary. The reason is that their creating part was removed.
  - Example: ` ذریعے` was built from ` ذری` + `عے`, and ` ذری` is removed.
- No HF BPE model with this vocabulary and these ids can ever emit these 156 tokens, whatever its merges or ranks.
- The event-ordered encoder emits **128** of them, **1,042** times, in **337 of the 836** dev_strict documents. Examples: ` ذریعے` 47, ` علاقیاں` 39, ` ہسپتال` 33, ` واضح` 32, ` اضافے` 31.
- Equivalence on 100% of dev is therefore impossible.

**A best-effort conversion as well** (V6):
- method: present tokens in the order of their last creation, each given the 2-token split that the growing merge list produces;
- result: it differs on **781/836** dev_strict documents (13,966 of 194,101 pretokens) and leaves 1,061 tokens unreachable;
- the file was deleted, as the rule requires.

**Consequence for the decision rule:** A7 stays a custom-encoder candidate and loses tie-breaker (a) of PLAN §7. This was anticipated in PLAN §2.1 and SOTA §1.3.

---

## 9. Report-only extras (not used for any selection)

**τ sensitivity** (`tau_sensitivity.py` → `verify/tau_sensitivity.json`; models in `models/sensitivity/`):

| τ | removal events | removed at end (% of learned) | re-added | train freq = 0 | < 20 | < 20 intermediate | dev_strict bytes/token |
|---|---|---|---|---|---|---|---|
| none (plain BPE = A1) | 0 | 0 | 0 | 25 | 447 | 447 | 6.7117 |
| **0.9 (the candidate)** | 276 | 252 (1.57%) | 24 | 0 | 249 | 245 | 6.7188 |
| 0.8 | 679 | 558 (3.47%) | 97 | 0 | 84 | 82 | 6.7267 |
| 0.7 | 1,095 | 809 (5.04%) | 222 | 0 | 9 | 8 | 6.7281 |

- **Lower τ** removes more of the under-supported tokens and compresses slightly better.
- **PLAN pre-registered τ = 0.9**, with 0.8 only as a fallback that did not trigger, and the paper found no universal best (0.7 and 0.9 both gave gains at 8k).
- The rows above are dev metrics. Choosing τ from them now would be post-hoc selection on dev. The candidate therefore stays τ = 0.9. Whether a τ = 0.7 wave should be *added* is left as an open question for the plan owner.
- All four models pass V1 (encoder = trainer on every train type), emit no removed token on dev_strict, and round-trip dev_strict exactly.

**Encode speed** (`verify/speed_check.json`; one thread, shared CPU, sanity only):

| data | PickyBPE, cold cache | PickyBPE, warm cache | A1 via HF, one call per document |
|---|---|---|---|
| dev_strict | 0.22 MB/s | 0.60 MB/s | 0.33 MB/s |
| train_D1 (45 MB) | 116 s | 101 s | 112 s |

Training takes 61 s for 16,340 merges, plus ≈ 12 s of loading and checks with the cached pretoken counts (115 s on the first run, which counts pretokens and checks the canonical form). PLAN's estimate was "≈ 1 min".

---

## 10. Deviations from PLAN, and additions

- **None in the algorithm, data, pre-tokenizer, vocabulary size or τ rule.**
- **Additions:**
  - The A1-16k reference was rebuilt here with PLAN's recipe, because the Stage-1 file did not exist yet when this task started. It turned out byte-identical to Stage 1's `A1-P1-D1-16k/tokenizer.json`, and so are the harness metrics and per-document token counts.
  - The report-only τ sweep (§9).
  - The HF impossibility analysis (§8).
- **Harness outputs** are written to `candidates/pickybpe/results/dev_strict/…` rather than `eval/results/…`, to keep this task inside its folder. The harness code is unchanged, and nothing was written under `eval/`.
- **Definition used for the 1% rule:** removed at the end ÷ learned slots (16,064). Every other reading also exceeds 1%: 276 removal events, or ÷ 16,384.

## 11. What I did not do

- **No LM was trained** (Stage 3), so nothing here says whether PickyBPE improves bits-per-byte.
- **No test-split data** was read or scored. The harness refuses test uids, and so does `pickybpe_common.load_view`.
- **I did not run or read the authors' code** (`pchizhov/picky_bpe`). Equivalence to it is therefore **not verified**. The implementation is checked for internal consistency, against HF BPE with removals off, and against the paper's text. The choices in §2 may differ from the reference implementation.
- **No HF `tokenizer.json`** is shipped (§8). No other export (tiktoken, SentencePiece) was attempted; they have the same limitation.
- **Only 16k was built.** PLAN §2.2 ranks A7 @16k only; the paper's largest gains were at 8k, which was not built.
- **Morphology** is the SILVER set, report-only. MorphScore `urd_arab` was not downloaded.
- **R3** (embedding norms) needs an LM, so it is not done.

## 12. Reproduce

```bat
set PYTHONIOENCODING=utf-8
set HF_HOME=F:\Hindko\_tokenizer\hf_cache
cd /d F:\Hindko\_tokenizer\candidates\pickybpe
python test_pickybpe.py
python build_a1_ref.py --threads 2
python build_a1_ref.py --threads 1 --out models\a1_ref_P1_D1_16k.retrain.json
python train_pickybpe.py --tau 0.9 --vocab 16384 --out models\pickybpe_P1_D1_16k_tau0.9.json
python train_pickybpe.py --tau 0.9 --vocab 16384 --out models\pickybpe_P1_D1_16k_tau0.9.retrain.json
python train_pickybpe.py --tau none --vocab 16384 --out models\plainbpe_P1_D1_16k_samecode.json
python verify_pickybpe.py --model models\pickybpe_P1_D1_16k_tau0.9.json --a1 models\a1_ref_P1_D1_16k.json --plain models\plainbpe_P1_D1_16k_samecode.json --hf-attempt --out verify\pickybpe_P1_D1_16k_tau0.9.verify.json
python hf_equivalence_analysis.py --model models\pickybpe_P1_D1_16k_tau0.9.json --out verify\pickybpe_P1_D1_16k_tau0.9.hf_equivalence.json
python removal_report.py --model models\pickybpe_P1_D1_16k_tau0.9.json --out verify\pickybpe_P1_D1_16k_tau0.9.removals.json
python run_harness.py
python tau_sensitivity.py
python speed_check.py
python make_manifest.py
```

- The model file is deterministic: it holds no timings. Timings go to `*.train.json`.
- `cache/` holds the pretoken counts (keyed by the data sha256 and the regex) and the canonical-form check.
- `*.trainstate.pkl` holds the trainer's final state, which V1 compares with the encoder.
