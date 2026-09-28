---
language:
- hno
license: cc-by-4.0
library_name: tokenizers
tags:
- tokenizer
- sentencepiece
- unigram
- hindko
- urdu-script
---

# Hindko tokenizer: SentencePiece Unigram, 32,768 ids

Release 1.0.0, 2026-09-27. Internal id: `R2-A4-SPnat-D2-32k`.

> **Licence: CC BY 4.0** for the tokenizer files. The licence does not cover the training texts, which are not distributed (Section 10).
>
> Every number on this page names its source. Files marked `eval/…` are in this folder. Other paths are relative to the study folder `F:\Hindko\_tokenizer\`.

---

## 1. What it is

- **A tokenizer for Hindko**, the Indo-Aryan language of Peshawar, Hazara (Abbottabad, Mansehra) and neighbouring districts of Pakistan, written in Urdu (Perso-Arabic) script. As far as we know, it is the first tokenizer trained for Hindko.
- **SentencePiece Unigram with byte fallback, 32,768 ids** (layout in Section 2.5; checked by `examples/test_tokenizer.py`, `eval/release_checks.json`; G2 remedy in `eval/tokenizer_summaries_dev.json`):

  | block | ids | count |
  |---|---|---:|
  | reserved special tokens | 0–63 | 64 |
  | the newline piece `\n` | 64 | 1 |
  | `<unk>` (never produced) | 65 | 1 |
  | UTF-8 byte pieces `<0x00>`…`<0xFF>` | 66–321 | 256 |
  | learned pieces | 322–32,767, minus the 5 below | 32,441 |
  | never-produced placeholders `<\|unused_0..4\|>` (G2 remedy, Section 5) | 7183, 16359, 16629, 16924, 19063 | 5 |
  | **total** | | **32,768** |

- **The canonical encoder is `tokenizer.json`** (Hugging Face `tokenizers`). `sp.model` is a SentencePiece convenience copy. With the newline convention of Section 2.3 it gives the same ids on every verified data set, with two documented exceptions:
  - very long single lines, where its float32 arithmetic can resolve an exact tie differently (Section 3.2);
  - literal special-token strings such as `<|im_start|>` typed inside the text. `tokenizer.json` cuts them out as special tokens. Plain SentencePiece does not, and matches it on only 128 of the 626 such stress items unless the text is first split with `split_canonical()` (Section 3.4).
- **Lossless on the verified text** (Section 3.1). `decode(encode(x)) == x` holds on:
  - every document of the dev split (dev_strict and dev_permissive), of the strict test split (test_strict) and of the train split (train_D2 and train_D1), all in the canonical form. The study's test view is strict-only (PLAN §1.2), so the 419 permissive-tier test documents were never encoded;
  - 86,510 of the 86,516 items of a synthetic stress set, which also covers non-canonical input.

  Byte fallback covers any code point. The known exception is U+2581 (`▁`), which decodes as a space. Losslessness on *all* possible text is not claimed.
- **Trained on the strict tier of the train split of the Hindko corpus** (D2: 11,066 documents, 26.3 MB; Section 4).
- **Chosen from 18 tokenizers** (17 ranked) that were trained and compared under one protocol, using the held-out bits-per-byte of small language models trained from scratch (Section 6).
  - It is one of three tokenizers in the statistically tied top set at the pre-registered decision scale.
  - It was chosen within that set by **Amendment 1**, an amended rule that uses the lowest dev bpb of a larger, report-only arbiter. The pre-registered rule chose a different member, `R2-A10-MinGram-P1r3-D2-48k`. That tokenizer stays on record in [`alternatives/`](alternatives/).
  - **On the sealed test split, at the pre-registered confirm scale** (5 seeds), it beats the standard BPE recipe: −1.01 % bpb [−1.33, −0.72]. At that scale it has the highest test bpb of the three top-set members, though not significantly (Section 7.3).
  - In the report-only large run on test (2 seeds; MinGram 32k only 1, because the Colab quota ran out) the order within the top set reverses, and it has the lowest mean test bpb of the four arms run, 1.11898 (Section 7.3).
  - It is **not** the best tokenizer on test at the pre-registered scale, and no "best on test" claim is made (Section 9).
- **Compression** (screening metric only; Section 7.4). On the strict test split it gives **7.277 bytes/token**. That is:
  - **21.6 % fewer tokens** than the best lossless external tokenizer measured (UrduHack RoBERTa-Urdu);
  - 40.4 % fewer than GPT-4o's o200k.

What it is **not**:
- It is not a language model.
- It is not a vocabulary extension for an existing LLM. For continued pretraining of Qwen, Llama or Gemma, use [`for_llm_extension/`](for_llm_extension/).

| file | what | sha256 |
|---|---|---|
| `tokenizer.json` | **Canonical encoder** (HF `tokenizers` Unigram, byte fallback). It contains the newline convention of Section 2.3, needs no custom code, and computes the Viterbi exactly. | `49f301c52363a09a1fc1925359af3ddabd7887495de02354cfa4092cba51da41` |
| `tokenizer_config.json` | `transformers` configuration: special tokens, the ChatML chat template, no automatic BOS/EOS | `2d51e3597741b9d1f4b4fa450000ae76d400f4d3f0b2c3a9f8aec546e58e31b3` |
| `special_tokens_map.json` | special-token roles | `f91a7115d847b2c524407f8da7eaff58392ea046f654400e894a585bde09c8a2` |
| `sp.model` | SentencePiece convenience copy with the release scores. Use it with the newline convention of Section 2.3. It differs from the model the LMs used (`1496e9a7…`) only by score rounding and a gauge shift (Section 3.3). | `b91854fedcb25c9d520a444a2c902f148d0ffe65a73330d7b4483ad2d594cc92` |
| `sp.vocab` | the piece list with release scores (text) | `c40b7e1846a4a0daefdeab6cfebd5e355223d27a934833a09e49d14d0d89548f` |
| `gauge.json` | per-character score offsets w(c). Recover the rounded log-probability of a piece as score − Σ w(c) (Section 3.3). | `b136a83222425872f3332e408eaf75c4e44493e970238c9ec28ba6b8cb2b6b16` |
| `RELEASE_MANIFEST.json` | sha256 of every other file in this folder | – |
| [`alternatives/`](alternatives/) | pointers to the two other members of the tied top set: MinGram 48k (the pre-registered pick) and MinGram 32k | |
| [`for_llm_extension/`](for_llm_extension/) | Track B: continued-BPE Hindko extensions of the Qwen3, Qwen3.5, Llama-3, Gemma-3 and Gemma-4 tokenizers, plus the embedding-initialisation recipe | |
| [`examples/`](examples/) | `usage.py` (encode, decode, chat template, the three loaders) and `test_tokenizer.py` (id layout, special tokens, round trip, chat template, loader agreement) | |
| [`eval/`](eval/) | result tables (JSON/CSV), the decision documents, the release-build equivalence records and the test LM results | |

The six tokenizer files were copied from `release_build/sp32k/`, and their sha256 was checked against `release_build/sp32k/build_manifest.json` (copy record `release_card/finalize/release_files.json`).

---

## 2. How to use it

Install `tokenizers` and, optionally, `transformers` and `sentencepiece`. `examples/usage.py` runs every snippet below, and `examples/test_tokenizer.py` checks the files (Section 3.5).

### 2.1 transformers

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("path/to/this/folder")
ids = tok(text)["input_ids"]                       # adds nothing: no BOS/EOS
back = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
assert back == text                                # for text without U+2581 (Section 3.1)
```

- **Nothing is added automatically.** `tokenizer.json` has no post-processor, and `tokenizer_config.json` sets `add_bos_token = add_eos_token = false`. So `tok(text)`, `tok(text, add_special_tokens=False)` and `tok.encode(text)` all return the canonical ids (verified: `eval/reports/EQUIVALENCE.md` §5; `eval/release_checks.json`).
- **Add document markers yourself.** The language models of this study saw every document as `[<|bos|>] + ids + [<|endoftext|>]`, that is id 1, then the text, then id 0. To add BOS automatically, set `tok.add_bos_token = True`.
- **Decoding.** `clean_up_tokenization_spaces` is already `false` in the config. Pass `skip_special_tokens=False` to get special tokens back as text.
- **Decode whole sequences, not one token at a time.** The decoder removes the word-initial `▁` at the start of each decode call, as SentencePiece-style decoders do. So `decode([349]) + decode([369])` loses the space between the two words that `decode([349, 369])` keeps. For streaming, decode all ids so far and emit only the new suffix, as transformers' `TextStreamer` does.
- **The loaded class** is `TokenizersBackend`, a fast tokenizer (transformers 5.3.0), with `len(tok) == 32,768`.

### 2.2 tokenizers (the canonical encoder)

```python
from tokenizers import Tokenizer

tk = Tokenizer.from_file("tokenizer.json")
enc = tk.encode(text)                              # default arguments give the canonical ids
enc.ids, enc.tokens
tk.decode(enc.ids, skip_special_tokens=False) == text
```

`tk.decode` also strips the leading `▁` of each call, so decode whole sequences (Section 2.1).

### 2.3 sentencepiece: the convenience copy, and the newline convention

The study trained SentencePiece on single lines and encodes a document line by line (PLAN §1.1):
1. split the text at `\n`;
2. encode each line;
3. join the lines with the newline piece (id 64).

`tokenizer.json` implements this convention inside the file:
- `\n` is an added token, so it is cut out first;
- each remaining line gets `' '`→`▁` and one `▁` prefix, and an empty line gets no tokens;
- there is no pre-tokenizer, so the Unigram Viterbi runs over the whole line.

With `sp.model` you must apply the convention yourself:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="sp.model")
NL = sp.piece_to_id("\n")                      # 64

def encode(text):
    out = []
    for k, ids in enumerate(sp.encode(text.split("\n"))):
        if k:
            out.append(NL)
        out.extend(ids)
    return out

def decode(ids):
    segs, cur = [], []
    for i in ids:
        if i == NL:
            segs.append(cur); cur = []
        else:
            cur.append(i)
    segs.append(cur)
    return "\n".join(sp.decode(s) if s else "" for s in segs)
```

- Do **not** call `sp.encode(text)` on a multi-line text. Every `\n` would become a byte-fallback piece, and the first word of each later line would lose its word-initial marker `▁`. Those are not the ids the models saw.
- `sp.model` with this wrapper gives the ids of `tokenizer.json` on:
  - 100 % of the documents of dev_strict, dev_permissive, test_strict, train_D2 and train_D1;
  - 86,515 of 86,516 stress items.

  Special-token strings in text need `usage.split_canonical` (Section 3.4). The one stress exception is a single 80,010-character line (Section 3.2).
- **Why `tokenizer.json` and not `sp.model` is canonical.** SentencePiece adds up Viterbi scores in float32, and `tokenizers` in float64.
  - With the release scores both are exact while every compared value stays below 1024 in magnitude.
  - On long single lines, float32 rounding can resolve an exact tie differently. `tokenizer.json` is the implementation that applies the shared tie rule exactly on any line length.
  - Amendment 1 §3 had named the SentencePiece model the canonical encoder. This release designates `tokenizer.json` instead. On every document the amendment's condition covers, the two give identical ids: 100 % of dev (dev_strict and dev_permissive) and of the strict test split (test_strict).
  - The study's dev and test evaluation text is strict-tier only (PLAN §1.2), so test_strict is the test split in the amendment's sense. The 419 permissive-tier test documents were never encoded or checked.

### 2.4 Normalizing input: `normalize()` (recommended, optional)

The tokenizer normalizes **nothing**. The SentencePiece normalization rule is `identity`, and `tokenizer.json` only escapes spaces as `▁`. Text round-trips as described in Section 3.1, including non-canonical input.

All training and evaluation text was in the canonical *data form* of `hp.normalize` 1.0.1 (`normalize.py` sha256 `037f3582f1656f80341447b84792d017b0ae5bb29f38c55657ef1babe9a74bf5`; `FROZEN.json`). To give the tokenizer input in the form it was trained on, apply the same function first:

```python
from hp.normalize import normalize        # corpus pipeline, F:\Hindko\_pipeline\hp\normalize.py (not bundled here)
ids = tk.encode(normalize(raw_text)).ids
```

What `normalize()` does (specification: `normalization/NORMALIZATION.md`):
- It is deterministic and idempotent, and its output is NFC.
- It folds only encoding noise:
  - Arabic presentation forms (the Allah ligature becomes the corpus spelling with HEH GOAL);
  - control characters and invisible characters (ZWSP, bidi marks, BOM, soft hyphen, …);
  - kashida;
  - ZWJ/ZWNJ outside letters (ZWNJ between letters is kept);
  - non-standard spaces and tabs, which become one space; spaces at line ends are stripped; 3 or more line breaks become 2;
  - CR/CRLF, which become LF;
  - Arabic-keyboard YEH/KAF/TEH MARBUTA, but only inside words that are provably Urdu/Hindko orthography;
  - Arabic-Indic digits U+0660–0669, which become Extended Arabic-Indic U+06F0–06F9;
  - decomposed Hindko tone letters, which become U+08BE–U+08C2.
- It never merges letters that are distinct in Hindko or Urdu, such as HEH GOAL and HEH DOACHASHMEE.
- It never touches harakat, punctuation, quotes, Latin text, ASCII vs Urdu digits, or U+FFFD.

Effects:
- On the released corpus it changes 233 characters in 20 web records (`normalization/NORMALIZATION.md`). Newspaper and book text is already in canonical form.
- It matters mainly for text typed or pasted by users. In the example of `examples/test_tokenizer.py`, a line with doubled spaces, a ZWSP and a CRLF takes 31 tokens raw and 18 after `normalize()` (`eval/release_checks.json`).

`normalize()` is not reversible: after it, decoding returns `normalize(x)`, not `x`. Do not apply it where whitespace must be preserved exactly (code, tables, verbatim quotes).

### 2.5 Special tokens and chat

| ids | pieces | notes |
|---|---|---|
| 0 | `<\|endoftext\|>` | end of document (EOT), `eos_token`. It was predicted by the LMs. |
| 1 | `<\|bos\|>` | start of document, `bos_token`. It was given to the LMs and never predicted. |
| 2 | `<\|pad\|>` | `pad_token` |
| 3, 4 | `<\|im_start\|>`, `<\|im_end\|>` | ChatML turn markers |
| 5–63 | `<\|reserved_0\|>` … `<\|reserved_58\|>` | free for future use (the PLAN §2.1 block, 64 in total) |
| 64 | `\n` | the newline piece of Section 2.3 (an added token, not special) |
| 65 | `<unk>` | never produced, because byte fallback covers every character. `unk_token` is deliberately unset (Section 3.4). |
| 66–321 | `<0x00>` … `<0xFF>` | UTF-8 byte fallback for characters that are not in the vocabulary |
| 322–32,767 | learned pieces (`▁` marks a word-initial piece) | 32,441 learned pieces. The 5 placeholders `<\|unused_0\|>`…`<\|unused_4\|>` (7183, 16359, 16629, 16924, 19063) are special tokens that ordinary text never produces. |

transformers reports 69 special ids: the 64-token block plus the 5 placeholders. Source: `eval/reports/EQUIVALENCE.md` §5.

- **Special-token strings typed inside text are matched as special tokens** (Section 3.4). If you encode untrusted text, remove or escape literal `<|…|>` strings first, or insert special tokens only as ids.
- **The chat template** (`tokenizer_config.json`) renders `<|bos|>` first, then the ChatML layout of PLAN §7.7 for each message (`<|im_start|>` role, newline, content, `<|im_end|>`, newline). With `add_generation_prompt=True` it appends `<|im_start|>assistant` and a newline. For one user message it renders:

  ```
  <|bos|><|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n
  ```

  - `apply_chat_template(..., tokenize=True)` equals `tokenizer.json` on the rendered text and equals the split-canonical SentencePiece encoding.
  - Decoding the ids returns the rendered string exactly.
  - Verified in `eval/reports/EQUIVALENCE.md` §5 and by `examples/test_tokenizer.py`, whose chat checks all pass (`eval/release_checks.json`).
  - The role name after `<|im_start|>` gets a word-initial `▁`, as any text segment does. For example, `user` is encoded as `▁`, `us`, `er`. The decoder removes that `▁` again.
- **Chat formatting was not part of the evaluated protocol.** The LMs saw plain documents only.

### 2.6 Using it for a new model

- **Embedding matrix.** It needs 32,768 rows, a multiple of 64. Ordinary text never produces ids 5–63 (reserved), 65 (`<unk>`) or the 5 placeholders.
- **Rare pieces.** Many pieces are rare in the corpus:
  - 13,182 of the 32,441 learned pieces (40.6 %) occur fewer than 20 times in the 7.3M-token LM training stream (train_D1), and 2 never occur (`eval/health_gates_and_support.csv`).
  - With this little Hindko text their embeddings will be under-trained. Checking the lowest-norm output embeddings after training is recommended (PLAN §4.3, R3).
- **Context length.** On average, one token covers:
  - dev_strict: 7.076 UTF-8 bytes, or 3.967 characters (`eval/tokenizer_summaries_dev.json`);
  - test_strict: 7.277 bytes, or 4.068 characters (`eval/test_intrinsic_released.json`).

---

## 3. Exactness: what "canonical" and "lossless" mean here

### 3.1 Lossless: scope and exceptions

What was verified on the final folder (sources: `eval/reports/EQUIVALENCE.md` §6 and `eval/reports/equivalence.json` for the release build; `eval/release_checks.json` for the re-run on this folder, including `train_views` for train_D2 and train_D1):

| text | items | `tokenizer.json` decode exact | transformers decode exact | `sp.model` decode exact |
|---|---:|---:|---:|---:|
| dev_strict (canonical form) | 836 docs | 836 | 836 | 836 |
| dev_permissive (canonical form) | 1,358 docs | 1,358 | 1,358 | 1,358 |
| test_strict (canonical form) | 491 docs | 491 | 491 | 491 |
| train_D2 (canonical form) | 11,066 docs | 11,066 | 11,066 | 11,066 |
| train_D1, the LM training view | 16,015 docs | 16,015 | 16,015 | 16,015 |
| stress set, 17 categories | 86,516 items | 86,510 | 86,510 | 86,510 |

- **The stress set** (`release_build/sp32k/stress/stress.jsonl`; categories in `eval/reports/stress_summary.json`) goes beyond the canonical form:
  - newlines, including `\r\n` and `\r`, and every Unicode space including tab, NBSP and U+3000;
  - ASCII, Urdu and Arabic-Indic digit runs;
  - Latin text, URLs and code;
  - every code point in any data view, in 10 contexts;
  - ZWNJ/ZWJ, the tone letters and tie-prone runs;
  - every piece string, and 18,000 fuzz strings;
  - CJK, emoji, Devanagari, Cyrillic, controls, private use and U+10FFFF;
  - single lines of 8k–80k characters;
  - literal special-token strings and chat text.

  It is not shipped here, because it contains corpus text (Section 10).
- **The exception: U+2581 (`▁`, 6 stress items).** A literal U+2581 in the input decodes as a space in every encoder, SentencePiece included, because `▁` is the space marker. U+2581 does not occur in the corpus.
- **Conditions for an exact round trip:**
  - decode with `skip_special_tokens=False`. With `True`, special-token strings in the text are dropped;
  - with transformers, keep `clean_up_tokenization_spaces=False`, which is the shipped default;
  - with `sp.model`, use the newline convention (Section 2.3).
- **Not claimed:** losslessness on every possible string. Byte fallback gives every code point a representation, but only the sets above were checked.

### 3.2 The float32 long-line limitation of `sp.model`

- **The mechanism** (`eval/reports/EQUIVALENCE.md` §1–§2, §8.1).
  - SentencePiece 0.2.1 accumulates Viterbi path scores in float32; `tokenizers` accumulates in float64. Both libraries keep the candidate with the smallest start at a lattice end position unless a later one is strictly better.
  - The release scores are multiples of 2^-14. So float32 is exact, and both encoders compute the same numbers, only while every value compared on a line is below 1024 in magnitude. float64 stays exact up to 2^39.
  - Past that bound, SentencePiece can resolve an **exact tie** differently. Exact ties arise in runs such as `۔۔۔`, `۱۱۱` or `«««`, which can be segmented XX+X or X+XX with an identical score. Near-ties below float32 resolution can in principle flip as well.
  - Newlines reset the score, so only the length of a *single line* matters.
- **Where the bound is crossed on real text** (`eval/float32_long_lines.json`; probe `release_card/finalize/longline_probe2.py`):
  - The probe cut the dev_strict text, joined into one line, into 25 windows of 32,000 characters. The running best-path score first reaches 1,024 after **3,418 characters at the earliest** (median 15,190; 5 of the 25 windows never reach it).
  - In the four data sets, 10 of 169,111 lines cross the bound (shortest 1,054 characters). All 10 are still encoded identically (`eval/reports/exact_check.json`).
- **How often the encoders differ** (same file):
  - **Plain real text:** 0 differences in windows of 1,000 to 32,000 characters (2,344 windows; `probe_real_text_windows`).
  - **Real text followed by a tie-prone run on the same line:** no differences up to 3,400 characters. The rate then rises with line length: 0.26 % at 4,000–5,000 characters, 0.52 % at 6,000–8,000, 2.45 % at 12,000 and 3.1–5.9 % at 16,000–31,000.
  - **Every observed difference is an exact tie:** the same total score, the same token count and the same pieces in a different order.
  - **In the stress set**, 1 of 65 long lines differs: `long_lines-000062`, 80,010 characters, 18,737 tokens both ways (`eval/release_checks.json` → `runs.full.sp_exact_tie_differences`; `eval/reports/EQUIVALENCE.md` §6.2; the 64 of 65 equal items are also in `eval/float32_long_lines.json` → `equivalence_long_lines_stress`).
- **What it means in practice:**
  - The round trip is unaffected, because every segmentation decodes to the same text.
  - The token count was equal in every observed difference, all of which were exact ties. A near-tie flip below float32 resolution, which was neither observed nor probed, could change it.
  - Documents with ordinary line lengths are unaffected. The longest line in the tokenizer's own training text (train_D2) is 14,186 bytes (`data/data_manifest.json`).
  - When ids must be canonical, for example to reproduce training data, use `tokenizer.json` or transformers. No rescoring can remove the limit, because the tied segmentations contain the same pieces (EQUIVALENCE.md §8.1). The limit applies to the original model (`1496e9a7…`) as well.

### 3.3 Score rounding (release scores ≠ trained scores)

The release model is the trained model (`candidates/round2/standard/tok/R2-A4-SPnat-D2-32k/sp.model`, sha256 `1496e9a7…`) with new scores for its 32,441 NORMAL pieces (`eval/reports/EQUIVALENCE.md` §2; `eval/reports/build_manifest.json`):

    s'(p) = round_half_even(s(p) · 2^14) / 2^14  +  Σ_{c in p} w(c)

- **The gauge** w(c) (`gauge.json`, 212 characters) adds the same exact amount to every candidate at a lattice node, so it cannot change any encoding. It only keeps SentencePiece's running score small. After it the scores (−14.47 to +44.38) are no longer log-probabilities. Recover the rounded log-probability of a piece as score − Σ w(c).
- **The rounding can change an encoding.** Each score moves by at most 2^-15 = 3.05e-5, and 31,957 of the 32,441 scores moved.
  - A decision flips if the exact score gap between two competing segmentations is smaller than the sum of their rounding moves. For a swap of 2 pieces against 2 pieces that sum is at most about 1.2e-4.
  - 2^-14 is the coarsest grid at which **no non-tie encoding changed** in the four data sets (dev_strict, dev_permissive, test_strict, train_D2). At 2^-14 no non-tie encoding changed in train_D1 or the stress set either. The coarser grids 2^-13, 2^-12 and 2^-10 changed 4, 5 and 6 non-tie lines in the four sets; at 2^-10 one of them was in test_strict.
  - **Outside those sets, near-ties with gaps below about 1.2e-4 may be resolved differently from the trained model.** `tokenizer.json` and `sp.model` carry the same rounded scores, so this is a difference from the original model, not between the two release files.
- **What changed relative to the trained model.** Only exact-tie lines changed:
  - 0 of 836 dev_strict documents and 0 of 491 test_strict documents;
  - 1 dev_permissive document, 22 train_D2 documents and 29 train_D1 documents (45 lines);
  - 476 stress items.

  Every changed line was verified to be an exact tie under the old scores. So every dev and test LM number in this card applies to the released files unchanged. The train_D1 stream that the LMs were trained on differs from the release encoder only in the order of 45 tie segmentations such as `۔`+`۔۔`, with the same token count.

### 3.4 Special-token strings in text

`tokenizer.json` (and therefore transformers) treats the 69 special strings as added tokens, as any Hugging Face tokenizer does (`eval/reports/EQUIVALENCE.md` §5):
- **How HF encodes them.** It cuts every literal `<|endoftext|>`, `<|bos|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|reserved_k|>` or `<|unused_k|>` out of the text and emits its id. Each text segment between them is encoded as a line, with its own `▁` prefix. This is **split-canonical**, implemented as `split_canonical()` in `examples/usage.py`. It was verified on 626/626 stress items, with an exact decode on 626/626.
- **Plain SentencePiece differs.** It matches only the user-defined symbols (ids 0–64), emits `▁` before a line-initial special, and gives no dummy prefix to the text right after a special. It equals HF on only 128 of the 626 items. The `<|unused_k|>` placeholders are CONTROL pieces in SentencePiece, so SentencePiece encodes their strings as ordinary characters.
- **Other strings.** `<unk>`, `<0x41>` and `[UNK]` in text are ordinary characters: `unk_token` is unset on purpose, so the literal `<unk>` is not id 65.
- **Decoding** with `skip_special_tokens=True` drops every special string, including one that was typed as text.
- **Recommendation:** insert special tokens as ids, or through the chat template. Remove or escape literal `<|…|>` in untrusted input. The corpus contains none of these strings (gate G3), and `|` is not in the vocabulary (byte fallback), so no ordinary segmentation can produce one.

### 3.5 Checks run on this folder

`examples/test_tokenizer.py` was run on the final folder on 2026-09-27 with Python 3.11.9, tokenizers 0.22.2, transformers 5.3.0 and sentencepiece 0.2.1. Source: `eval/release_checks.json` → `runs` (one entry per row).

| run | `runs` key | texts | checks passed | failed |
|---|---|---:|---:|---:|
| default (8 built-in samples) | `default` | 8 | 276 | 0 |
| `--jsonl dev_strict.jsonl --hp-path …` | `dev_strict_hp` | 844 | 3,623 | 0 |
| `--jsonl` dev_strict, dev_permissive, test_strict `--stress stress.jsonl --hp-path …` | `full` | 89,209 | 356,457 | 0 |

- **What the full run covers:**
  - the id layout;
  - all 69 special tokens (one id each, at the reserved id, in `tokenizers` and transformers);
  - bos/eos/pad = 1/0/2, `unk_token` unset, no automatic BOS/EOS;
  - the round trip in all three loaders;
  - transformers ids = `tokenizer.json` ids on every text;
  - `sp.model` ids = `tokenizer.json` ids;
  - the chat template, with and without a generation prompt: rendered string, ids, atomic markers, exact decode, and equality with split-canonical;
  - `hp.normalize`.
- **The two documented exceptions are tested as such:**
  - the 6 U+2581 items must decode with `▁` → space;
  - the one `sp.model` difference (`long_lines-000062`) must be an exact score tie.
- **AutoTokenizer** loads the folder as `TokenizersBackend` with 32,768 tokens.

---

## 4. Training data (D2 = strict train split)

Source for every number in this section: `data/data_manifest.json` (view `train_D2`), unless another file is named.

- **Source:** the Hindko text corpus of 2026-09-26: `hindko_dataset.jsonl` (strict tier), sha256 `1f62e946…` (`FROZEN.json`). Its texts are:
  - *Weekly Hindkowan* (Peshawar) newspaper issues;
  - the *Gandhara Hindko Academy* book series;
  - a small amount of web text.
- **Split:** manifest `split_manifest.jsonl`, sha256 `76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2` (`splits/SPLITS.md`, `splits/splits_report.json`).
  - The split is 90/5/5 by words per source.
  - It is group-disjoint by rule: newspaper editions, books and web sites (or speakers) are assigned whole.
  - One exception to that rule: 230 leak-prone records from 40 evaluation groups were moved to train, so those groups are not 100 % held out. The moved records duplicate text from other groups.
  - Near-duplicate leakage across split boundaries was checked: after that move, 0 evaluation records leak (containment ≥ 0.30 or Jaccard ≥ 0.50).
- **D2** is every `quality_tier == "strict"` document of the train split, in `hp.normalize` 1.0.1 form, which changed 3 documents. No test or dev document is in it.

| D2 | documents | characters | UTF-8 bytes | whitespace words | line breaks |
|---|---:|---:|---:|---:|---:|
| **total** | **11,066** | **14,731,915** | **26,330,463** | **3,083,541** | 132,532 |
| book | 6,037 | 9,110,441 | 16,279,682 | 1,918,570 | 113,442 |
| newspaper | 4,748 | 5,279,372 | 9,444,696 | 1,091,091 | 15,594 |
| web | 281 | 342,102 | 606,085 | 73,880 | 3,496 |

- **Composition:**
  - Books are 61.8 % of the characters. Much of that is poetry, dictionaries and literary prose.
  - Variety labels: Hindko 10,458 documents, mixed 552, no signal 56.
  - The documents come from 447 split groups.
- **Training units:** the 141,146 non-empty lines of `train_D2.lines.txt` (sha256 `50ea1ec79cd591b9b912d68fcedab6212141c9585a7f7d3cbd9d30423de251a7`; the longest is 14,186 bytes). Every line was used, with no sampling and no shuffling.
- **Why D2 and not the permissive train split D1** (45.0 MB, which adds permissive-tier text, including Urdu-labelled documents):
  - The Stage 1 rule chose D2.
  - In the LM, D2 beat D1 by −0.74 % bpb [−0.96, −0.51] at equal settings (BPE 16k; `eval/reports/DECISION.md` §7).
- **Held-out text** was used only for evaluation:
  - dev_strict: 836 documents, 1,445,513 bytes, 32 bootstrap clusters (`eval/reports/decision.json` → `clusters`);
  - test_strict: 491 documents, 1,451,026 bytes, 27 clusters (15 newspaper, 6 book, 6 web; `eval/reports/TEST_RESULTS.md` §1).
  - The arbiter LMs trained on train_D1 (16,015 documents, 45.0 MB), the same stream for every tokenizer.

---

## 5. Algorithm

- **SentencePiece 0.2.1 Unigram**, trained in about 20 s on 2 threads. Parameters (`eval/tokenizer_summaries_dev.json` → trainer):

  | parameter | value | parameter | value |
  |---|---|---|---|
  | `model_type` | `unigram` | `vocab_size` | 32,768 |
  | `character_coverage` | 1.0 | `byte_fallback` | true |
  | `split_by_whitespace` | true | `split_by_unicode_script` | true |
  | `split_by_number` | true | `split_digits` | true (one digit per token) |
  | `max_sentencepiece_length` | 16 | `normalization_rule_name` | `identity` |
  | `remove_extra_whitespaces` | false | `add_dummy_prefix` | true (per line) |
  | `max_sentence_length` | 65,536 | `input_sentence_size` / `shuffle_input_sentence` | 0 / false (every line) |
  | `user_defined_symbols` | the 64-token block + `"\n"` (ids 0–64) | `unk_id` / `bos_id` / `eos_id` / `pad_id` | 65 / −1 / −1 / −1 |

- **Pre-tokenization** is SentencePiece's own ("SPnat" in the study):
  - spaces become the word-initial marker `▁`;
  - pieces never cross a script change or a number boundary;
  - digits are single tokens.
  - Harakat are not split off from their letters. For example, `▁تُساں` is one piece.
- **Vocabulary.** 32,441 learned pieces.
  - Every character seen in training has its own piece (`character_coverage` 1.0). Anything else falls back to UTF-8 bytes.
  - The vocabulary has no sub-character pieces (gate G5) and no partial-UTF-8 pieces.
- **G2 remedy** (pre-declared in PLAN §4.3):
  - 5 learned pieces could never be produced, because Viterbi always splits them: ` وچہ`, ` والے۔`, ` چھوڑاں`, ` کہیا۔`, ` اُتے۔`.
  - Their slots became the never-produced placeholders `<|unused_0..4|>`, so no other id moved. The effective vocabulary is 32,763.
  - The remedy cannot change any encoding, because a Unigram piece that loses on its own string never wins inside a longer one. This was verified by re-encoding dev.
- **Determinism (G4).** Retraining from the same input gives a byte-identical model.
- **Release scores.** The trained scores were rounded to 2^-14 and gauge-shifted for the exact HF export (Section 3.3).

---

## 6. How it was chosen

The whole protocol was written and frozen before any LM result existed: `eval/reports/PLAN.md`, sha256 `d7a811df…` in `eval/reports/FROZEN.json`. Its elements:

- **Decision metric:** held-out **bits per byte** (bpb) of small GPT-style LMs trained from scratch with each tokenizer.
  - The comparison is byte-matched: the same training bytes, the same context in bytes and the same bytes per step for every tokenizer.
  - bpb is measured on the strict dev split.
- **Statistics:**
  - a hierarchical cluster bootstrap over newspaper editions, books and web sites, which also resamples seeds;
  - Holm correction against the best candidate;
  - an equivalence margin of ±0.3 %.
- **Top set:** the best candidate, plus every candidate not significantly worse or equivalent to it.
- **Tie-breakers inside the top set, in order:** (a) exact Hugging Face-native encoding; (b) higher bytes/token; (c) fewer rare tokens; (d) robustness; (e) smaller vocabulary.
- **Test split:** used once, at the end.

What happened, in order (sources: `eval/reports/DECISION.md`, `eval/reports/AMENDMENT_1.md`, `eval/dev_lm_large.csv`, `eval/reports/TEST_RESULTS.md`):

1. **Intrinsic sweep (Stage 1–2).** About 70 tokenizers were trained and measured on dev:
   - algorithms: BPE, SentencePiece BPE/Unigram, SuperBPE, PickyBPE, MinGram;
   - 4 pre-tokenizers, 3 data mixes, vocabularies of 8k–48k.

   This fixed the pre-tokenizer P1r3 (3-digit groups) and the data mix D2.
2. **Round 1** (pre-registered): 12 LM candidates.
   - They are ranks 1–8 of the plan's list (rank 6 is two sizes, rank 8 is conditional), the rank-9 SentencePiece BPE, and 2 D1 reference arms added before the first LM run.
   - The baseline "standard recipe" is byte-level BPE, P1r3, D2, 16k: `A1-P1r3-D2-16k`.
3. **Round 2** (exploratory): 6 more tokenizers, built after the round-1 LM results had been seen.
   - They combine the best round-1 recipe with the algorithms that looked best, at 32k and 48k.
   - This is a garden-of-forking-paths risk: the dev results of round 2 are conditioned on dev. The sealed test split is the guard (Section 7.3 (d)).
4. **LM arbiters.**
   - Stage 3 'screen': d=128, 10 MB, 3 seeds.
   - Stage 4 **'confirm'**: d=192, L=4, 1.77M non-embedding parameters, the full permissive train split, 1 epoch, 5 seeds, LR 1e-3.
   - All runs were on a Colab T4 GPU. The decision uses Stage 4.
5. **Pre-registered decision** (`eval/reports/DECISION.md`, 2026-09-26 20:20 UTC). 17 tokenizers were ranked.
   - **The top set has 3 statistically tied members:**
     - `R2-A4-SPnat-D2-32k` (this tokenizer; the best mean, 1.22584 bpb);
     - `R2-A10-MinGram-P1r3-D2-48k` (1.22636);
     - `R2-A10-MinGram-P1r3-D2-32k` (1.22844).
   - The gaps inside the top set (+0.04 %, +0.21 %) are below the achievable resolution (≈ 0.6–0.7 %). The three are *not distinguishable at our power*, which does not make them equal.
   - Tie-breaker (a) removed this tokenizer. Its automatic HF export was not exact: it differed on 1 of 1,358 dev_permissive documents, an equal-score tie with the same token count.
   - Tie-breaker (b) then chose **MinGram 48k**, which has the highest bytes/token.
   - The pre-registered improvement claim holds on dev: MinGram 48k vs the baseline is **−1.58 % bpb [−1.87, −1.36]**.
6. **Adversarial audit.** Tie-breaker (b) rises mechanically with vocabulary size, so inside a tied set it simply picks the largest vocabulary. The 48k model has 2.27× the total parameters of the 16k baseline, so part of its confirm-scale gain is a parameter effect.
7. **The 'large' arbiter** (report-only, not pre-registered, run 2026-09-26 19:44–20:53 UTC):
   - setup: d=384, L=6, 10.6M non-embedding parameters, 2 epochs of the full train split, LR 5e-4 from its own sweep on the baseline, 2 seeds;
   - it covered the top set plus the baseline and BPE 32k;
   - result: this tokenizer had the lowest dev bpb, **1.13373**, against 1.14086 (MinGram 32k) and 1.14407 (MinGram 48k). The baseline had 1.14908 (Section 7.2).
8. **Amendment 1** (`eval/reports/AMENDMENT_1.md`, sha256 `9a966033…`, written 2026-09-26 21:14:48 UTC, **before any test LM number existed**):
   - The pre-registered choice and its claim stay on record unchanged, with their test result.
   - The **released default** is chosen by an amended rule: *within the pre-registered, statistically tied top set, the lowest large-arbiter dev bpb*. This replaces tie-breaker (b), which is confounded with parameter count. The rule selects `R2-A4-SPnat-D2-32k`, which is also the best-mean candidate at the confirm scale.
   - Tie-breaker (a) becomes a packaging requirement: the release must ship a `tokenizer.json` whose ids equal the canonical encoder on 100 % of dev and test documents, with MinGram 32k as the fallback.
   - **Status: the condition is met.**
     - The exact export was built (`release_build/sp32k/`; `eval/reports/EQUIVALENCE.md`). No test LM number was used for it; test_strict was used as text only.
     - Its ids equal the SentencePiece encoder on 836/836 dev_strict, 1,358/1,358 dev_permissive and 491/491 test_strict documents. test_strict is the study's whole test view (PLAN §1.2: evaluation text is strict-tier only); the 419 permissive-tier test documents were never encoded. The fallback was not used.
     - This release names `tokenizer.json` rather than `sp.model` as the canonical encoder, for the reason in Section 2.3.
   - The test run was not changed by the amendment, and no test result can change which tokenizer is released.
9. **One-shot test** (2026-09-26 21:16–22:55 UTC; `eval/reports/TEST_RESULTS.md`):
   - Bundle `f54c929ba1ab` holds the 4 candidates fixed in `decision.json` → `test_candidates_fixed` (`colab/FINAL_TEST.md`, `lm/WAVES_final_test.json`).
   - Bundle `94175d26497f` holds 2 report-only BPE controls and the baseline. It was added after the audit and built 2026-09-26 21:10–21:13 UTC, before the first test LM number (`lm/WAVES_final_test_supp.json`; `FINAL_TEST_LOG.json` entry 2).
   - Results: Section 7.3.

**Read this choice for what it is.** The released tokenizer was selected by a rule written *after* the dev results at two model scales were known. It was not selected by the pre-registered rule. Its test result (Section 7.3) is reported as it is: at the pre-registered confirm scale it beats the baseline but has the highest test bpb of the three top-set members, not significantly. The only pre-registered test comparison is MinGram 48k vs the baseline, and it holds.

---

## 7. Results

Conventions for every table in this section:
- Δ % = (bpb_A − bpb_B) / bpb_B.
- CIs are 95 % percentile intervals of the hierarchical cluster bootstrap: 10,000 replicates, `numpy.random.default_rng(12345)`, clusters resampled within source, and each candidate's seeds resampled.
- p is two-sided. "<1e-4" means 0 of 10,000 replicates fell on the other side. Such tail p-values depend on the random stream, so read them as "about 1e-3 or smaller".
- "Baseline" is `A1-P1r3-D2-16k`.

### 7.1 Dev LM bits per byte: Stage 4 'confirm' arbiter (decision scale)

- **Data:** dev_strict, 836 documents and 1,445,513 bytes. Each tokenizer has 5 seeds, and the bootstrap uses 32 clusters.
- **"Round":** R1 = the pre-registered round 1; R2 = the exploratory round 2.
- **Source:** [`eval/dev_lm_confirm_ranking.csv`](eval/dev_lm_confirm_ranking.csv), copied from `analysis/decision.json`.

| # | tokenizer | round | vocab | mean dev bpb (5 seeds) | seed s.d. | Δ vs baseline, 95 % CI | Δ vs best, 95 % CI | Holm p vs best | top set |
|---:|---|---|---:|---:|---:|---|---|---:|---|
| 1 | `R2-A4-SPnat-D2-32k` **(released)** | R2 | 32,768 | 1.22584 | 0.23 % | −1.62 % [−2.11, −1.24] | best | – | **yes** |
| 2 | `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | R2 | 49,152 | 1.22636 | 0.15 % | −1.58 % [−1.87, −1.36] | +0.04 % [−0.40, +0.57] | 0.8366 | **yes** |
| 3 | `R2-A10-MinGram-P1r3-D2-32k` | R2 | 32,768 | 1.22844 | 0.21 % | −1.41 % [−1.66, −1.19] | +0.21 % [−0.17, +0.65] | 0.5800 | **yes** |
| 4 | `R2-A4-SPnat-D2-48k` | R2 | 49,152 | 1.23205 | 0.38 % | −1.13 % [−1.69, −0.64] | +0.51 % [+0.14, +0.87] | 0.0234 | no |
| 5 | `R2-A1-P1r3-D2-48k` | R2 | 49,152 | 1.23374 | 0.17 % | −0.99 % [−1.21, −0.78] | +0.64 % [+0.25, +1.12] | 0.0080 | no |
| 6 | `A1-P1r3-D2-32k` | R1 | 32,768 | 1.23597 | 0.07 % | −0.81 % [−1.00, −0.66] | +0.83 % [+0.47, +1.26] | <1e-4 | no |
| 7 | `A4-SPnat-D1-16k` | R1 | 16,384 | 1.23925 | 0.81 % | −0.55 % [−1.30, +0.23] | +1.09 % [+0.43, +1.83] | 0.0040 | no |
| 8 | `A10-MinGram-P1-D1-16k` | R1 | 16,384 | 1.24180 | 0.20 % | −0.34 % [−0.57, −0.10] | +1.30 % [+0.91, +1.79] | <1e-4 | no |
| 9 | `A1-P1r3-D2-16k` (baseline, "standard recipe") | R1 | 16,384 | 1.24607 | 0.10 % | baseline | +1.65 % [+1.25, +2.15] | <1e-4 | no |
| 10 | `A1-P1-D1-16k` | R1 (added arm) | 16,384 | 1.25430 | 0.16 % | +0.66 % [+0.46, +0.86] | +2.32 % [+1.94, +2.80] | <1e-4 | no |
| 11 | `A7-PickyBPE-P1-D1-16k-tau0.9` | R1 | 16,384 | 1.25446 | 0.10 % | +0.67 % [+0.51, +0.84] | +2.34 % [+1.97, +2.77] | <1e-4 | no |
| 12 | `A1-P1r3-D1-16k` | R1 (added arm) | 16,384 | 1.25532 | 0.18 % | +0.74 % [+0.51, +0.97] | +2.41 % [+1.98, +2.96] | <1e-4 | no |
| 13 | `A1-P1r3-D2-8k` | R1 | 8,192 | 1.25672 | 0.18 % | +0.86 % [+0.37, +1.29] | +2.52 % [+1.88, +3.24] | <1e-4 | no |
| 14 | `A3-SPnat-D1-16k` | R1 | 16,384 | 1.26651 | 1.46 % | +1.64 % [+0.43, +2.96] | +3.32 % [+2.15, +4.61] | <1e-4 | no |
| 15 | `R2-A6-SBPE-P1r3-D2-32k-t080` | R2 | 32,768 | 1.27375 | 0.12 % | +2.22 % [+1.82, +2.61] | +3.91 % [+3.53, +4.32] | <1e-4 | no |
| 16 | `A6-SBPE-P1-D1-16k-t090` | R1 | 16,384 | 1.27485 | 0.20 % | +2.31 % [+2.02, +2.63] | +4.00 % [+3.62, +4.47] | <1e-4 | no |
| – | `A6-SBPE-P1-D1-32k-t080` (conditional rank 8; its condition was false, so it is not ranked) | R1 | 32,768 | 1.27960 | 0.13 % | +2.69 % [+2.25, +3.11] | +4.39 % [+4.04, +4.78] | – | no |
| 17 | `A6-SBPE-P1-D1-16k-t080` | R1 | 16,384 | 1.28692 | 0.32 % | +3.28 % [+2.88, +3.71] | +4.98 % [+4.54, +5.52] | <1e-4 | no |

Id legend:
- A1: byte-level BPE; A3: SentencePiece BPE; A4: SentencePiece Unigram; A6: SuperBPE; A7: PickyBPE; A10: MinGram (reimplemented from Land 2026).
- SPnat: SentencePiece's own pre-tokenization; P1: marks stay inside words, single digits; P1r3: as P1, with right-to-left 3-digit groups.
- D1: permissive train; D2: strict train.

Findings on dev (report-only, `eval/reports/DECISION.md` §7):
- Unigram-family vocabularies beat BPE at equal settings.
- 32k beats 16k and 8k, while 48k adds nothing significant.
- D2 beats D1.
- SuperBPE and PickyBPE did not help these small arbiters.

The released tokenizer's rank 1 on dev is **descriptive**. Round 2 was built after the round-1 results were seen, so its dev Δs are optimistic (winner's curse). On test its Δ vs the baseline is −1.01 % (Section 7.3).

### 7.2 Dev LM bits per byte: 'large' arbiter (report-only; the basis of Amendment 1)

- **Setup:** d=384, L=6, H=6, tied embeddings; 2 epochs of train_D1; LR 5e-4; seeds 1–2; Colab T4, fp16 training and fp32 evaluation.
- **Status:** not pre-registered.
- **The CIs** come from the same hierarchical cluster bootstrap. They were computed for this card, after Amendment 1, by `release_card/build_eval_tables.py`, which reuses `analysis/decide.py`. With 2 seeds the seed part of the CI is crude.
- **bpb values** are rounded once from the full-precision records (`lm/colab_results/77e1368773fc/results/large__*.json`, as in `eval/reports/TEST_RESULTS.md` §10).
- **Source:** [`eval/dev_lm_large.csv`](eval/dev_lm_large.csv) / `.json`, which hold 6-decimal values.

| # | tokenizer | total params | mean dev bpb (2 seeds) | seeds 1 / 2 | Δ vs baseline, 95 % CI | Δ vs released, 95 % CI | confirm-scale bpb (rank) |
|---:|---|---:|---:|---|---|---|---|
| 1 | `R2-A4-SPnat-D2-32k` **(released)** | 23.31M | **1.13373** | 1.13319 / 1.13428 | −1.336 % [−1.78, −1.03] | – | 1.22584 (1) |
| 2 | `R2-A10-MinGram-P1r3-D2-32k` | 23.31M | 1.14086 | 1.14016 / 1.14157 | −0.715 % [−0.91, −0.52] | +0.629 % [+0.31, +1.05] | 1.22844 (3) |
| 3 | `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 29.59M | 1.14407 | 1.14336 / 1.14477 | −0.436 % [−0.62, −0.29] | +0.911 % [+0.56, +1.37] | 1.22636 (2) |
| 4 | `A1-P1r3-D2-16k` (baseline) | 17.02M | 1.14908 | 1.14899 / 1.14917 | baseline | +1.354 % [+1.04, +1.81] | 1.24607 (9) |
| 5 | `A1-P1r3-D2-32k` | 23.31M | 1.14940 | 1.15108 / 1.14771 | +0.027 % [−0.30, +0.32] | +1.381 % [+1.02, +1.82] | 1.23597 (6) |

Per source, Δ vs baseline (large arbiter; source `eval/dev_lm_large.json` → `per_source_vs_baseline`):

| source | `R2-A4-SPnat-D2-32k` (released) | `R2-A10-MinGram-P1r3-D2-32k` | `R2-A10-MinGram-P1r3-D2-48k` | `A1-P1r3-D2-32k` |
|---|---|---|---|---|
| book (12 clusters) | −1.40 % [−2.17, −0.97] | −0.50 % [−0.72, −0.26] | −0.24 % [−0.41, −0.03] | +0.26 % [−0.08, +0.57] |
| newspaper (15) | −1.16 % [−1.43, −0.94] | −1.13 % [−1.39, −0.85] | −0.84 % [−1.18, −0.55] | −0.48 % [−0.88, −0.13] |
| web (5; 34 KB) | −1.72 % [−4.12, +0.07] | −1.01 % [−2.55, +0.21] | −0.66 % [−1.91, +0.65] | +0.11 % [−1.70, +1.30] |

- **At this scale the top set separates** (`eval/dev_lm_large.json` → `top_set_pairs`):
  - Released − MinGram 48k = **−0.90 % [−1.35, −0.56]** (p < 1e-4).
  - Released − MinGram 32k = **−0.62 % [−1.04, −0.31]** (p 0.0004).
  - MinGram 32k − MinGram 48k = −0.28 % [−0.46, −0.07] (p 0.011).
- **The larger vocabulary no longer helps.** BPE 32k is not distinguishable from BPE 16k (+0.03 % [−0.30, +0.32]), and MinGram 48k is behind MinGram 32k. This is consistent with the audit's reading that the confirm-scale advantage of 48k was partly a parameter effect of the small model.
- **LR sweep** of the large arbiter (baseline, seed 1; `eval/dev_lm_large.json` → `lr_sweep_baseline`): 5e-4 → 1.14899; 1e-3 → 1.18164; 2e-3 → 1.22207. The chosen 5e-4 is the lowest grid value (Section 8).

### 7.3 Test LM bits per byte (one shot): every arm, including the pre-registered pick

- **Protocol:** unchanged from dev Stage 4: 5 seeds, LR 1e-3, the same deterministic recipe and the same `hk_lm.py`.
- **Same models as dev.** Every test record's training-loss curve equals its dev record's, so the test models are the dev-selected models.
- **Data:** the strict test split, 491 documents, 1,451,026 bytes, 27 clusters.
- **Bundles:**
  - `f54c929ba1ab`: the 4 candidates fixed in `decision.json` before test (`colab/FINAL_TEST.md`);
  - `94175d26497f`: `A1-P1r3-D2-32k` and `R2-A1-P1r3-D2-48k` plus the baseline, added after the audit and before any test LM number (`lm/WAVES_final_test_supp.json`, `FINAL_TEST_LOG.json`). Its baseline runs are bitwise identical to those of `f54c929ba1ab` and are counted once.
- **Only one comparison is pre-registered:** MinGram 48k vs the baseline. Every other row is report-only and never re-selects.
- **Sources:** `eval/reports/TEST_RESULTS.md` §0–§9, [`eval/test_lm_confirm.csv`](eval/test_lm_confirm.csv) / `.json` (from `analysis/test_results.json`).

| test rank | tokenizer | role | test bytes/token | test bpb (5 seeds) | seed s.d. | Δ vs baseline, test | p | Δ vs baseline, dev | change dev → test (pp) |
|---:|---|---|---:|---:|---:|---|---:|---|---|
| 1 | `R2-A10-MinGram-P1r3-D2-48k` | **pre-registered pick: the primary comparison** | 7.373 | 1.20543 | 0.06 % | **−1.32 % [−1.47, −1.16]** | <1e-4 | −1.58 % [−1.87, −1.36] | +0.27 [+0.02, +0.55] |
| 2 | `R2-A10-MinGram-P1r3-D2-32k` | top-set member, Amendment 1 fallback (report-only) | 7.275 | 1.20620 | 0.21 % | −1.25 % [−1.48, −1.04] | <1e-4 | −1.41 % [−1.66, −1.19] | +0.16 [−0.04, +0.39] |
| 3 | `R2-A4-SPnat-D2-32k` | **released default** (report-only) | 7.277 | 1.20911 | 0.20 % | **−1.01 % [−1.33, −0.72]** | <1e-4 | −1.62 % [−2.11, −1.24] | +0.61 [+0.17, +1.12] |
| 4 | `R2-A1-P1r3-D2-48k` | equal-vocabulary BPE control (report-only; bundle `94175d26497f`) | 7.353 | 1.21111 | 0.12 % | −0.85 % [−1.05, −0.65] | <1e-4 | −0.99 % [−1.21, −0.78] | +0.14 [−0.10, +0.38] |
| 5 | `A1-P1r3-D2-32k` | best round-1 tokenizer (report-only; bundle `94175d26497f`) | 7.246 | 1.21459 | 0.09 % | −0.57 % [−0.75, −0.38] | <1e-4 | −0.81 % [−1.00, −0.66] | +0.24 [+0.03, +0.49] |
| 6 | `A1-P1r3-D2-16k` | baseline | 6.948 | 1.22150 | 0.11 % | – | – | – | – |

**Checks before reading this table** (`colab/FINAL_TEST.md` §4 for `f54c929ba1ab`; the same checks on `94175d26497f`; `eval/reports/TEST_RESULTS.md` §1):
- **Completeness:** 35 confirm records: 6 tokenizers × 5 seeds, plus a second, bitwise-identical copy of the 5 baseline runs in `94175d26497f`. All have status ok, 491 test documents each, and bpb recomputed bit-exactly. All 42 records, including the 7 large ones, pass all 1,512 per-record checks.
- **The test models are the dev models:** every test record's training-loss curve and R3 embedding norms equal its dev record's (42/42).
- **CPU↔GPU parity:** passes in both bundles (fp16 +0.534 %, fp32 +0.008 % vs the CPU reference; tolerance 1 %).

**(a) Primary result** (PLAN §7.5–7.6; `eval/reports/TEST_RESULTS.md` §4):
- **Verdict: the improvement claim holds on test.** MinGram 48k vs the baseline: −1.32 % [−1.47, −1.16], p <1e-4.
  - Test agrees with dev in sign.
  - It holds in every sensitivity run: bundle-only stream, pair-only, raw manifest groups, and seeds 1–3.
- **The gain is smaller on test than on dev**, by 0.27 pp [+0.02, +0.55].
  - MinGram 48k is better than the baseline in 24 of 27 clusters (98.8 % of test bytes).
  - Per source: book −1.19 % [−1.34, −1.02], newspaper −1.55 % [−1.80, −1.29], web −1.45 % [−2.19, −0.70].

**(b) The released tokenizer on test** (`eval/reports/TEST_RESULTS.md` §5, §8):
- **Vs the baseline:** −1.01 % [−1.33, −0.72], p <1e-4, with the CI below 0 in every source:
  - book −1.02 % [−1.53, −0.64];
  - newspaper −0.97 % [−1.28, −0.63];
  - web −1.44 % [−2.31, −0.32].
- **Its dev margin shrank the most of the 13 contrasts:** −1.62 % → −1.01 %, a change of +0.61 pp [+0.17, +1.12].
- **Vs the pre-registered pick:** SP-32k vs MinGram-48k +0.31 % [−0.02, +0.63], p 0.0648. SP-32k's test bpb is higher, but the 95 % CI includes 0, so the difference is not significant.
- **Vs MinGram 32k:** SP-32k vs MinGram-32k +0.24 % [−0.13, +0.60], p 0.1896. SP-32k's test bpb is higher, but the 95 % CI includes 0, so the difference is not significant.
- **Rank within the top set** across the 10,000 test replicates: SP-32k is 3rd of the three in 89.6 %. On newspaper text it is significantly behind both MinGram builds: +0.59 % [+0.24, +0.97] and +0.51 % [+0.14, +0.89].
- **Top set on test.** Applying PLAN §7.3's top-set rule to the 6 test tokenizers (report-only) gives the same three members as on dev.

**(c)–(f) Other report-only contrasts** (`eval/reports/TEST_RESULTS.md` §6, §7, §9):
- **(c)** MinGram 32k vs the baseline: −1.25 % [−1.48, −1.04] (dev −1.41 %).
- **(d) Forking paths.** Every top-set member beats the round-1 winner `A1-P1r3-D2-32k` on test, significantly after Holm:
  - MinGram 48k −0.75 % [−0.93, −0.58];
  - MinGram 32k −0.69 % [−0.92, −0.46];
  - SP-32k −0.45 % [−0.71, −0.20].
- **(e) Equal vocabulary.** At equal vocabulary, and so at equal parameter count, the Unigram-family tokenizers beat BPE:
  - MinGram 48k vs BPE 48k −0.47 % [−0.64, −0.31];
  - SP-32k vs BPE 32k −0.45 % [−0.71, −0.20];
  - MinGram 32k vs BPE 32k −0.69 % [−0.92, −0.46].
- **(f) Dev vs test.**
  - 11 of 13 contrasts keep their sign. The 2 flips are SP-32k's two within-top-set comparisons, which were not significant on dev and are not significant on test.
  - Every gain over the baseline is smaller on test, by +0.14 to +0.61 pp, round-1 tokenizers included.
  - Kendall τ_b between the dev and test orders of the 6 is 0.73.

**Large arbiter on test** (report-only, allowed by Amendment 1 §4; d=384, L=6, 2 epochs, LR 5e-4 fixed from dev; `eval/reports/TEST_RESULTS.md` §10, [`eval/test_lm_large.csv`](eval/test_lm_large.csv) / `.json`):

> **Colab quota note: MinGram 32k has one seed (n = 1).** Its seed-2 run stopped at step 6,588 of 7,325 when the Colab session disconnected. The free GPU quota was then exhausted, so it was not rerun. Every other arm has seeds 1 and 2.

| tokenizer | seeds | test bpb per seed | mean test bpb | Δ vs baseline (means) | dev large mean bpb |
|---|---|---|---:|---:|---:|
| `R2-A4-SPnat-D2-32k` (released) | 1, 2 | 1.11856 / 1.11940 | 1.11898 | −0.77 % | 1.13373 |
| `R2-A10-MinGram-P1r3-D2-32k` | **1 only** | 1.12240 | 1.12240 | −0.47 % | 1.14086 |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 1, 2 | 1.12475 / 1.12480 | 1.12477 | −0.26 % | 1.14407 |
| `A1-P1r3-D2-16k` (baseline) | 1, 2 | 1.12831 / 1.12702 | 1.12767 | – | 1.14908 |

- **The large-scale order on test is the dev order:** SP-32k < MinGram-32k (n = 1) < MinGram-48k < baseline.
- **With 2 seeds each:**
  - SP-32k vs the baseline: −0.77 % [−1.15, −0.40];
  - SP-32k vs MinGram-48k: −0.51 % [−0.86, −0.22], p 0.0014;
  - MinGram-48k vs the baseline: −0.26 % [−0.44, −0.08].

  The first two survive when seed noise is added: −0.77 % [−1.27, −0.28] and −0.51 % [−1.02, −0.04].
- **SP-32k vs MinGram-32k is not established on test.** −0.30 % with only one MinGram-32k seed; with seed noise added, [−0.95, +0.27], p 0.2996.
- **Scale dependence.** From the confirm to the large scale on test, MinGram-48k's edge over the baseline shrinks by 1.06 pp [+0.89, +1.24], and SP-32k vs MinGram-48k reverses from +0.31 % to −0.51 %. This is the parameter-confound pattern that Amendment 1 cites, now also on test.
- **Timing and code.**
  - The large test stage started at 22:16:40 UTC, after every confirm test number existed (the last at 22:14:47 UTC). Amendment 1 had announced it before any test number existed.
  - It ran with a modified `run_all.py` (sha256 `a11c2d7e…`; 11 changed lines, all for the `--final-test-large` switch). `hk_lm.py` was unchanged.

### 7.4 Comparison with frontier model tokenizers on held-out Hindko (test split)

**Compression is a screening metric, not a quality metric.** Fewer tokens per byte says nothing by itself about how well a model will learn or generate Hindko. The LM evidence for this tokenizer is in Sections 7.1–7.3 and is summarised below the table.

- **Sources:**
  - external tokenizers: [`eval/test_intrinsic_competitors.csv`](eval/test_intrinsic_competitors.csv), from `eval/reports/TEST_COMPETITORS.md` (one-shot run, 2026-09-26 21:07 UTC, after the choice was fixed on dev), each with its own native encoder and no BOS/EOS added;
  - this tokenizer: `eval/test_intrinsic_released.json`, computed for this card on 2026-09-27 with the same harness metric code. That was after all test LM results existed; it is report-only, and `tokenizer.json` and `sp.model` give identical rows.
- **Metrics:**
  - **Fertility** = tokens overlapping each whitespace word, averaged over words.
  - **Lossless** = `decode(encode(doc)) == doc` on the 491 test documents (G1).
  - **Released uses fewer tokens** = 1 − tokens(released) ÷ tokens(other), on the same 491 documents (1,451,026 bytes).

| tokenizer | provider | vocab | test bytes/token | test fertility | lossless on test | released uses fewer tokens |
|---|---|---:|---:|---:|---|---:|
| **`R2-A4-SPnat-D2-32k` (released)** | this study | 32,768 | **7.277** | **1.135** | yes (491/491) | – |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | this study | 49,152 | 7.373 | 1.125 | yes (491/491) | −1.3 % (it uses fewer) |
| GPT-4o (o200k; = gpt-oss, Phi-4-mini) | OpenAI | 200,000 | 4.338 | 1.931 | yes (491/491) | 40.4 % |
| Gemma 3 / Gemma 4 (identical encodings) | Google | 262,145 / 262,144 | 4.610 | 1.807 | yes (491/491) | 36.6 % |
| Llama 4 | Meta | 201,135 | 3.727 | 2.251 | yes (491/491) | 48.8 % |
| Llama 3 (= Alif-1.0, Qalb-1.0, ALMAnaCH Llama-3-8B-mono-Urdu, Llama-3.2 + custom Urdu tokens) | Meta | 128,256 | 2.641 | 3.119 | yes (491/491) | 63.7 % |
| Qwen 3.5 (= Qwen 3.8) | Alibaba Qwen | 248,070 | 3.598 | 2.323 | yes (491/491) | 50.6 % |
| Qwen 3 (= Qwen 2.5) | Alibaba Qwen | 151,669 | 2.633 | 3.082 | yes (491/491) | 63.8 % |
| DeepSeek-V3 (= R1, V4, V4.1) | DeepSeek | 128,815 | 3.372 | 2.476 | yes (491/491) | 53.7 % |
| Mistral Tekken (Mistral NeMo = Mistral Small 4) | Mistral AI | 131,072 | 3.901 | 2.149 | yes (491/491) | 46.4 % |
| BLOOM | BigScience | 250,680 | 5.122 | 1.633 | yes (491/491) | 29.6 % |
| XLM-R | Meta | 250,002 | 4.792 | 1.680 | **no** (173/491) ⚠ | 34.2 % |
| mT5 | Google | 250,100 | 3.912 | 1.874 | **no** (175/491) ⚠ | 46.2 % |
| RoBERTa-Urdu (monolingual Urdu) | UrduHack | 52,000 | 5.709 | 1.459 | yes (491/491) | **21.6 %** |
| Urdu BERT WordPiece 64k | community (farahadeeba) | 64,000 | 6.221 | 1.358 | **no** (0/491) ⚠ | 14.5 % |
| Urdu GPT-2 BPE 20k | community (aariciah) | 20,000 | 5.633 | 1.478 | **no** (166/491) ⚠ | 22.6 % |
| Llama + Urdu tokenizer | community (BilalKhan1) | 129,435 | 2.372 | 2.834 | yes (491/491) | 67.4 % |
| ALMAnaCH Llama-2-7B-mono-Urdu (= Llama 2) | Inria ALMAnaCH | 32,000 | 1.714 | 3.937 | yes (491/491) | 76.5 % |

- **⚠ Not lossless.** These tokenizers do not reproduce the text. XLM-R, mT5 and Urdu BERT turn each run of line breaks into one space, among other normalizations. Urdu GPT-2 applies NFKC and lowercasing and turns line breaks into UNK tokens (`eval/reports/TEST_COMPETITORS.md`). So their token counts cover slightly different text, and part of their compression comes from discarding information.
- **Ranking.** By test bytes/token the released tokenizer is 1st of 66 (65 external tokenizers of released models plus this one), and 1st of the 56 that round-trip every test document.
  - The pre-registered pick MinGram 48k, with a 1.5× larger vocabulary, needs 1.3 % fewer tokens still.
  - The 66 ranked tokenizers are split over two files. `eval/test_intrinsic_competitors.csv` has 67 rows: the 65 external tokenizers, MinGram 48k and `hindko-probe-bpe32k`, an in-house probe trained on all data, including this test text, which is never ranked. The released tokenizer is not a row of that CSV. Its values are in `eval/test_intrinsic_competitors.json` → `released` (token count, bytes/token and the two ranks) and in `eval/test_intrinsic_released.json` (fertility, STRR, G1).
- **Fair comparison.** A 32k vocabulary trained only on Hindko is expected to compress Hindko far better than a multilingual vocabulary of 128k–262k that serves hundreds of languages. The table shows the token cost of Hindko for those models; it does not rank them as models.

**The LM evidence, separately.**
- No external tokenizer was scored with a language model in this study. The external rows have compression numbers only.
- Among this study's own tokenizers, higher compression did not reliably mean lower bpb:
  - on test, BPE 48k (7.353 bytes/token) has a *higher* confirm-scale bpb (1.21111) than SP-32k (7.277 bytes/token, 1.20911);
  - MinGram 48k compresses best of all and has the lowest confirm-scale test bpb. At the large scale, though, it is behind both 32k top-set members: significantly on dev, and in means on test (Sections 7.2–7.3).
- The claims this card makes about model quality rest on held-out LM bpb (Sections 7.1–7.3), not on the table above.

### 7.5 Intrinsic metrics on dev (released tokenizer)

Source: `eval/tokenizer_summaries_dev.json` (study harness `F:\Hindko\_tokenizer\eval\harness.py`, not bundled here; SentencePiece encoder with the newline convention; the release files give identical dev_strict ids). Test-split values are in Section 7.4 and `eval/test_intrinsic_released.json`.

| slice | docs | bytes | bytes/token | chars/token | fertility (tokens/word) | STRR (words kept whole) | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all dev_strict | 836 | 1,445,513 | 7.076 | 3.967 | 1.141 | 88.6 % | 836/836 |
| book | 539 | 878,168 | 6.858 | 3.851 | 1.140 | 88.4 % | 539/539 |
| newspaper | 280 | 532,872 | 7.531 | 4.210 | 1.136 | 89.3 % | 280/280 |
| web | 17 | 34,473 | 6.310 | 3.553 | 1.244 | 82.2 % | 17/17 |
| variety: Hindko | 751 | 1,405,500 | 7.099 | 3.980 | 1.141 | 88.7 % | 751/751 |
| variety: mixed | 83 | 39,031 | 6.391 | 3.595 | 1.144 | 87.2 % | 83/83 |
| **test_strict** (report-only) | 491 | 1,451,026 | 7.277 | 4.068 | 1.135 | 89.1 % | 491/491 |

- **Tokens vs the literal standard recipe.** On the same dev text, this tokenizer needs 0.948× the tokens of byte-level BPE-P1-D1-16k (NSL).
- **Robustness** (PLAN §4.2 perturbations, report-only). The tie-breaker (d) metric is 0.589, against 0.295 and 0.316 for the two MinGram tokenizers (`eval/dev_lm_confirm_ranking.csv`). For each perturbation, the table gives the change in token count and the share of affected words that are segmented differently:

  | perturbation | token count change | affected words re-segmented |
  |---|---:|---:|
  | harakat removed | −0.37 % | 9.6 % |
  | digit script swapped | −0.65 % | 80.6 % |
  | space before ۔/، deleted | +6.07 % | 56.6 % |
  | ZWNJ inserted in compounds | +0.57 % | 89.0 % |

### 7.6 Health gates (PLAN §4.3; hard gates, all dev)

Source: [`eval/health_gates_and_support.csv`](eval/health_gates_and_support.csv).

| tokenizer | G1 lossless (dev_strict) | G2 no unreachable whole-character tokens | G3 specials atomic and never in the corpus | G4 deterministic retrain | G5 no sub-character tokens | partial-UTF-8 tokens (R2) |
|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` (released) | 836/836 | pass (0 of 32,441), after the pre-declared remedy of 5 pieces | pass (0 occurrences) | pass (byte-identical) | pass | 0 |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 836/836 | pass (0 of 48,831) | pass (0 occurrences) | pass | pass | 0 |
| `R2-A10-MinGram-P1r3-D2-32k` | 836/836 | pass (0 of 32,447) | pass (0 occurrences) | pass | pass | 0 |
| `A1-P1r3-D2-16k` (baseline) | 836/836 | pass (0 of 16,055) | pass (0 occurrences) | pass | n/a (byte-level) | 9 |

G1 on test_strict (report-only), computed with the harness encoder and the native encoder while the final-test bundles were built, is 491/491 for all four of these tokenizers (`eval/reports/TEST_COMPETITORS.md`). The release files also give 491/491 (Section 3.1).

### 7.7 Support profile (PLAN §4.3 R1: how often each learned token occurs in the LM training stream train_D1)

Source: [`eval/health_gates_and_support.csv`](eval/health_gates_and_support.csv).

| tokenizer | learned tokens | train freq = 0 | train freq < 20 | train freq < 100 | median train freq | train_D1 tokens | dev bytes/token | dev fertility | dev STRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R2-A4-SPnat-D2-32k` (released) | 32,441 | 2 | 13,182 (40.6 %) | 25,413 (78.3 %) | 27 | 7,305,029 | 7.076 | 1.141 | 88.6 % |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 48,831 | 1 | 27,246 (55.8 %) | 41,802 (85.6 %) | 16 | 7,158,278 | 7.224 | 1.122 | 89.8 % |
| `R2-A10-MinGram-P1r3-D2-32k` | 32,447 | 0 | 11,329 (34.9 %) | 25,148 (77.5 %) | 32 | 7,316,797 | 7.126 | 1.138 | 88.7 % |
| `A1-P1r3-D2-16k` (baseline) | 16,064 | 33 | 577 (3.6 %) | 8,491 (52.9 %) | 91 | 7,762,075 | 6.788 | 1.197 | 84.8 % |

- The tokenizer was trained on D2 (26.3 MB). The counts above are on train_D1 (45.0 MB), the stream the LMs trained on, which is the relevant one for embedding training.
- 32k and 48k vocabularies are in the data-scarce regime: most pieces are rare.

---

## 8. Limitations

- **Proxy models.** The arbiters are tiny: the 'confirm' model has 1.77M non-embedding parameters and the 'large' one 10.6M. The ranking is evidence about models of this size trained on about 45 MB of Hindko. It is **not** evidence about 1B+ models or about continued pretraining of a multilingual LLM.
- **Selection after seeing dev.**
  - Round 2, which contains all three top-set members, was built after the round-1 results had been seen.
  - The released tokenizer was then chosen by Amendment 1, written after the 'large' dev results were known.
  - The dev Δ values are therefore optimistic (winner's curse). On test, the released tokenizer's gain over the baseline shrank from −1.62 % to −1.01 %, the largest shrink of any contrast (Section 7.3).
- **At the decision scale the released tokenizer is tied, not ahead, and on test it has the highest bpb of the tied three.** At the confirm scale it is statistically indistinguishable from both MinGram tokenizers on dev and on test. On test it has the highest mean bpb of the three, not significantly. On test newspaper text it is significantly behind both MinGram builds.
  - It is ahead only at the large scale. That arbiter has 2 seeds (1 for MinGram 32k on test, because of the exhausted Colab quota) and is not pre-registered.
  - The large arbiter's LR (5e-4) was tuned on the baseline only and is the lowest value of its sweep, so the optimum may be lower and may differ between tokenizers.
  - The confirm LR (1e-3) is also the lowest value of its grid.
- **Seed noise of SentencePiece arbiters.** SentencePiece-encoded LM runs had a higher seed s.d. than HF-encoded ones at the confirm scale on dev: pooled 0.86 % vs 0.18 % of bpb, the root mean square over 4 and 13 arms (`eval/dev_lm_confirm_ranking.json`, `seed_sd_rel_pct` by `encoder_kind`). It was driven by the two D1 16k arms (1.46 %, 0.81 %). This tokenizer's own s.d. is 0.23 % on dev and 0.20 % on test.
- **`sp.model` is not exact on very long single lines** (Section 3.2). Beyond about 3,400 characters without a newline it may resolve an exact tie differently from `tokenizer.json`. Use `tokenizer.json` when ids must be canonical.
- **Release scores differ from the trained scores** (Section 3.3). No encoding changed on any verified set, but near-ties outside them can be resolved differently from the original trained model.
- **Special-token strings in input text are matched as special tokens** by `tokenizer.json` and transformers. Plain `sp.model` segments them differently unless the text is split with `split_canonical()` first (Section 3.4).
- **Lossless only as verified** (Section 3.1). U+2581 in input decodes as a space.
- **Rare pieces.** 40.6 % of the learned pieces occur fewer than 20 times in the 7.3M-token training stream, and 2 never occur. Expect under-trained embeddings.
- **Segmentation robustness** is lower than MinGram's (Section 7.5):
  - swapping digit script re-segments 81 % of the affected words;
  - deleting the space before `۔`/`،` adds 6 % tokens.

  Unnormalised or inconsistently spaced input costs tokens, and `normalize()` helps only with the noise it folds.
- **Numbers.** Digits are always single tokens (`split_digits`), so long numbers are long in tokens.
- **The newline convention is required for `sp.model`.** Plain `sp.encode()` on multi-line text gives different ids than the models saw (Section 2.3).
- **Chat use.** ChatML is provided, but chat formatting was never evaluated.
- **Domain and dialect coverage.**
  - Books make up 62 % of the training characters, much of it poetry, dictionaries and literary prose. Newspaper text is mostly Peshawari. Web text is tiny (0.6 MB).
  - Hazara, Peshawari and other varieties are unevenly represented, and no native-speaker evaluation was done.
  - Urdu-labelled and Pahari/Pothohari-labelled text is not in D2.
- **Morphology.** Morphology was not measured against gold data; only silver sets exist. The pieces are mostly whole words (87–92 % of the silver-set words are single tokens; `eval/tokenizer_summaries_dev.json`), so no morphological segmentation is claimed.
- **Comparisons we cannot make.** No parallel Hindko text exists, so parity or tokenization-premium comparisons with other languages are not possible.
- **Corpus defects that the tokenizer faithfully encodes.** 49 U+FFFD replacement characters occur in 20 book documents: 48 in 19 train_D1 documents and 1 in 1 dev document, none in test_strict (count on `data/*.jsonl`). `normalize()` leaves U+FFFD as it is (Section 2.4). The C1 control U+0095, about 50 times in one raw permissive web record, is removed by `normalize()` rule R03 (`normalization/NORMALIZATION.md`) and occurs in no data view.
- **Test intrinsic metrics.** For this tokenizer, bytes/token, fertility, STRR and G1 on test were computed for this card, after the test LM run (Section 7.4). Robustness was not computed on test.

---

## 9. What can and cannot be claimed

These rules follow PLAN §8, `DECISION.md` §9 and Amendment 1 §5. The claim texts come from `eval/reports/TEST_RESULTS.md` §12.

**Claims allowed for the released tokenizer (`R2-A4-SPnat-D2-32k`):**
- "In the statistically tied top set at the pre-registered decision (confirm) scale: one of the three of 17 ranked tokenizers that are not significantly worse than the best (5 seeds each, hierarchical cluster bootstrap)."
  - Descriptive, not a confirmatory claim: its mean dev_strict bpb was the lowest of the 17 (1.22584; −1.62 % [−2.11, −1.24] vs the standard BPE recipe). It was built in an exploratory round after the round-1 results were seen.
- "Chosen from that top set by an amended rule (Amendment 1, fixed before any test LM number): the lowest dev bpb with a larger, report-only arbiter."
  - The large-arbiter evidence: 1.13373, against 1.14086 for MinGram 32k and 1.14407 for MinGram 48k; −0.62 % [−1.04, −0.31] and −0.90 % [−1.35, −0.56]; 2 seeds.
- "On the strict test split, at the confirm scale (5 seeds), R2-A4-SPnat-D2-32k vs the standard BPE recipe A1-P1r3-D2-16k: Δ = −1.01 % bpb, 95 % CI [−1.33, −0.72], p <1e-4 (dev −1.62 % [−2.11, −1.24])."
  - Always add, in the same statement: "Within the top set it has the highest confirm-scale test bpb of the three (vs MinGram-48k +0.31 % [−0.02, +0.63]; vs MinGram-32k +0.24 % [−0.13, +0.60]; neither CI excludes 0)."
  - "Report-only large arbiter on test: −0.77 % [−1.15, −0.40] vs the baseline and −0.51 % [−0.86, −0.22] vs MinGram-48k (2 seeds); vs MinGram-32k not established (1 seed)."
  - This is the test result, reported as it is.
- "On the same Hindko test text it uses 21.6 % fewer tokens than the best lossless external tokenizer measured (UrduHack RoBERTa-Urdu), 36.6 % fewer than Gemma 3/4, 40.4 % fewer than GPT-4o (o200k), 50.6 % fewer than Qwen3.5 and 63.7 % fewer than Llama 3. The comparison covers 65 external tokenizers." Always call this a screening metric (Section 7.4).
- "Lossless on the canonical form and on an 86,516-item stress set, with byte fallback: `decode(encode(x)) == x` on every document of the dev split, the strict test split (test_strict) and the train split, and on all stress items except literal U+2581, which decodes as a space."
- "No unreachable whole-character tokens after the pre-declared remedy (5 pieces removed). 13,182 of its 32,441 learned tokens (40.6 %) occur fewer than 20 times in the LM training stream. The full support profile is published."
- "The Hugging Face `tokenizer.json` gives the SentencePiece encoder's ids (with the newline convention) on 100 % of the documents of the dev split, the strict test split (test_strict) and the train split, and on 86,515 of 86,516 stress items. For the 626 stress items with literal special-token strings, the SentencePiece side is the split-canonical encoding; plain SentencePiece matches only 128 of them. The one other exception is an 80,010-character single line, where SentencePiece's float32 arithmetic resolves an exact tie differently."

**Claims for the pre-registered pick (`R2-A10-MinGram-P1r3-D2-48k`), on record:**
- On dev: "improves on the standard recipe `A1-P1r3-D2-16k`: Δ = −1.58 % bpb, 95 % CI [−1.87, −1.36]."
- On test, **confirmed**: "R2-A10-MinGram-P1r3-D2-48k, chosen by the pre-registered rule; selected after an exploratory second round; confirmed on a sealed test split used once: it improves on the standard BPE recipe A1-P1r3-D2-16k by Δ = −1.32 % bits-per-byte (95 % CI [−1.47, −1.16], p <1e-4; confirm-scale arbiter, 5 seeds, hierarchical cluster bootstrap over 27 test clusters; dev −1.58 % [−1.87, −1.36])."
- It has the lowest mean test bpb of the 6 tokenizers scored on test at the confirm scale. It is not distinguishable from MinGram 32k (−0.06 % [−0.29, +0.15], TOST-equivalent within ±0.3 %) or from SP-32k.

**Claims not allowed:**
- **For the released tokenizer: "best on test", "lowest held-out bpb on test" or "best tokenizer for Hindko".** This holds whatever the reading. At the pre-registered confirm scale it is 3rd of the 6 test tokenizers and 3rd of the 3 top-set members. The large run that puts it first is report-only, with 1–2 seeds.
- "State of the art for all models", or "the best possible Hindko tokenizer". Only 18 LM candidates were trained (17 ranked), with proxies of at most 10.6M non-embedding (29.6M total) parameters.
- "The pre-registered choice" for the released tokenizer. It was chosen by Amendment 1, not by the pre-registered rule.
- The dev Δ of the released tokenizer (−1.62 %) as a confirmed effect. It is descriptive; the test value is −1.01 %.
- That the three top-set members are "equal". At the confirm scale only MinGram 48k vs MinGram 32k is TOST-equivalent on test; the others are only not distinguishable at our power.
- That 48k is better than 32k. The step was not significant at the confirm scale and was reversed at the large scale.
- "Lossless on any text", or that `sp.model` and `tokenizer.json` encode identically on all input (Sections 3.1–3.2).
- Transfer to 1B+ models or to continued pretraining.
- Downstream task quality. No Hindko benchmark exists.
- Parity with other languages.
- Morphological quality.
- Anything about learning rates below the tested grids.
- Model quality from the compression table (Section 7.4).

---

## 10. Licence and rights

- **Tokenizer files** (`tokenizer.json`, `sp.model`, `sp.vocab`, `gauge.json`, configs): released on Hugging Face under **CC BY 4.0**.
- **Rights in the training text.** The vocabulary and the piece scores were learned from the Hindko corpus.
  - Its texts belong to *Weekly Hindkowan*, the *Gandhara Hindko Academy* and their authors.
  - The web part has per-record licences. Only Meta's Omnilingual ASR `hno_Arab` (CC BY 4.0) and Mozilla Common Voice `hno` (CC0) are openly licensed.
  - The corpus itself is not distributed.
  - The tokenizer contains no documents. It does contain 32,441 pieces of up to 16 characters and frequency-derived scores. The CC BY 4.0 licence covers the tokenizer files, not the training texts.
- **The stress set** (`release_build/sp32k/stress/stress.jsonl`) contains corpus text and is therefore not bundled.
- **`hp.normalize`** (recommended in Section 2.4) is part of the corpus pipeline in `pipeline/hp/`. The pipeline's InPage decoder (`hp/inpage.py`) uses a GPL-3.0 glyph table (`glyph_map.py`) that is not included in this repository; see `pipeline/README.md`.
- **`for_llm_extension/`** folders are derived from third-party tokenizers and keep their base licences:
  - Qwen3, Qwen3.5 and Gemma 4: Apache-2.0;
  - Llama 3.x: the Llama 3.1 Community License. Among its terms, redistribution requires a copy of the licence and a prominent "Built with Llama" notice;
  - Gemma 3: the Gemma Terms of Use.
  - These extended tokenizers are not part of the public release. Any future release will ship each folder with its base licence text: Apache-2.0; the Llama 3.1 Community License plus the "Built with Llama" notice; the Gemma Terms of Use notice.
- **The 65 external tokenizers in Section 7.4** were only measured locally. None is redistributed here.
- **`examples/`:** CC BY 4.0, like the tokenizer files.

**Suggested citation:** *Hindko tokenizer `R2-A4-SPnat-D2-32k` (SentencePiece Unigram, 32,768 ids), release 1.0.0, 2026-09-27. Trained on the strict train split of the Hindko text corpus (Weekly Hindkowan, Gandhara Hindko Academy, web; corpus release of 2026-09-26).* Authors: Junaid Aslam and Muhammad Ozair.

---

## 11. How to reproduce

All paths are relative to the study folder `F:\Hindko\_tokenizer\`, except in step 11. In this section `eval/…` means the study's `eval/` folder, not the `eval/` of this release. Python 3.11.9, sentencepiece 0.2.1, tokenizers 0.22.2, transformers 5.3.0, numpy 2.4.2. Set `PYTHONIOENCODING=utf-8`.

1. **Frozen inputs** (`FROZEN.json`):
   - split manifest `splits/split_manifest.jsonl`, sha256 `76582d3a…`;
   - `hp.normalize` 1.0.1, sha256 `037f3582…`;
   - datasets `hindko_dataset.jsonl` (`1f62e946…`) and `hindko_dataset_permissive.jsonl` (`01d07443…`).
2. **Data views.** `python data/materialize.py` writes `data/train_D2.lines.txt` (sha256 `50ea1ec7…`) and the dev views (`data/data_manifest.json`).
3. **Train the tokenizer.** The round-2 driver `candidates/round2/r2_standard.py` (sha256 `7bc31379…`) calls `candidates/standard/sweep_lib.py` (`69f49bef…`). The call is equivalent to:
   ```python
   import sentencepiece as spm
   SPECIALS = ["<|endoftext|>", "<|bos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"] + [f"<|reserved_{i}|>" for i in range(59)]
   spm.SentencePieceTrainer.train(
       input="data/train_D2.lines.txt", model_prefix="sp", model_type="unigram", vocab_size=32768,
       character_coverage=1.0, byte_fallback=True, split_digits=True, split_by_whitespace=True,
       split_by_unicode_script=True, split_by_number=True, max_sentencepiece_length=16,
       normalization_rule_name="identity", remove_extra_whitespaces=False, add_dummy_prefix=True,
       max_sentence_length=65536, user_defined_symbols=SPECIALS + ["\n"], unk_id=65,
       bos_id=-1, eos_id=-1, pad_id=-1, num_threads=2, input_sentence_size=0, shuffle_input_sentence=False)
   ```
   Then apply the G2 remedy with `sweep_lib.remedy_sp`: the 5 failing pieces become CONTROL `<|unused_k|>` at the same ids. The result is the trained model `sp.model`, sha256 `1496e9a7…`. A retrain is byte-identical (G4).
4. **Intrinsic metrics and gates.**
   - `eval/harness.py` (`31db30dd…`) with `eval/adapters.py` (`bc377b78…`; `SPAdapter(newline_wrapper=True)` is the SentencePiece encoder of the study) writes `candidates/round2/standard/results/dev_strict/R2-A4-SPnat-D2-32k/summary.json`.
   - The test competitor table comes from `eval/run_test_competitors.py`, then `eval/report_test_competitors.py`.
5. **LM arbiter.**
   - `colab/build_bundle.py` builds the bundles: `1d24425d2d64` (round 1) and `77e1368773fc` (round 2).
   - `run_all.py` and `hk_lm.py` (`3f78ad79…`) run them on a Colab T4.
   - `lm/collect_colab.py` collects the results into `lm/colab_results/`. The runs are bitwise reproducible across sessions on the same GPU type.
6. **Decision.** `python analysis/decide.py && python analysis/report.py` gives `DECISION.md` and `decision.json`. It takes a few seconds and is deterministic.
7. **Amendment 1:** `analysis/AMENDMENT_1.md`, with its sha256 in `AMENDMENT_1.sha.json`.
8. **Release files.**
   - The Hugging Face files, the release scores and the equivalence check come from `release_build/sp32k/scripts/` (`eval/reports/EQUIVALENCE.md` §9):
     - `build_release.py` (sha256 `ce9a63bb…`) writes `sp.model`, `gauge.json`, `tokenizer.json` and the configs;
     - `make_stress.py` writes the stress set; `verify.py`, `exact_check.py` and `lm_view_check.py` check it; `make_report.py` writes the report.
   - The build records are `release_build/sp32k/build_manifest.json` (sha256 of every input and output), `equivalence_full.json`, `exact_check.json` and `lm_view_check.json`. Copies are in `eval/reports/`.
9. **This folder.**
   - `python release_card/build_eval_tables.py` writes the dev tables in `eval/` and adds the large-arbiter bootstrap.
   - `python release_card/finalize/copy_release.py` copies the release files and checks their sha256.
   - `release_card/finalize/longline_probe.py` and `longline_probe2.py` run the float32 probes.
   - `eval/harness.py run --spm sp.model` / `--tokenizer-json tokenizer.json --data data/test_strict.jsonl --allow-test --no-morph` gives the test intrinsic metrics (outputs in `release_card/finalize/test_intrinsic/`).
   - `python release_card/finalize/make_release_tables.py` writes the test tables, `eval/release_checks.json`, the `release_finalization` section of `eval/MANIFEST.json` and `RELEASE_MANIFEST.json`.
10. **Test.**
    - The one-shot run covers bundle `f54c929ba1ab` (`colab/FINAL_TEST.md`, `lm/WAVES_final_test.json`) and the supplementary bundle `94175d26497f` (`lm/WAVES_final_test_supp.json`). Both builds are logged in `FINAL_TEST_LOG.json`.
    - The analysis is `analysis/test_analysis.py` (numbers: `analysis/test_results.json`; report: `analysis/TEST_RESULTS.md`), after `lm/collect_final_test.py`.
11. **Check a copy of this folder.** This step runs in the release folder `F:\Hindko\tokenizer\`, not in the study folder, because `examples/` exists only in the release. The data paths are therefore absolute:
    - `python -B examples/test_tokenizer.py --jsonl F:\Hindko\_tokenizer\data\dev_strict.jsonl --hp-path F:\Hindko\_pipeline`;
    - add `--stress F:\Hindko\_tokenizer\release_build\sp32k\stress\stress.jsonl` for the stress set, and `--summary-json <file>` to write the counts (the runs in `eval/release_checks.json` were written this way);
    - `-B` keeps Python from writing `examples/__pycache__/` into the release folder.
