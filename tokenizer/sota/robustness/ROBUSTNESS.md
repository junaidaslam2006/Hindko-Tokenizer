# Hindko tokenizer: out-of-distribution and robustness tests

Generated 2026-09-27T08:44:17Z by `sota/robustness/report.py` from `results/<tokenizer>.json` (`run_eval.py`) and `sets/sets_manifest.json` (`build_sets.py`). Machine-readable twin: `robustness.json`.

- **What is measured.** Intrinsic tokenizer properties only: bytes per token, fertility (tokens per whitespace word, from in-context offsets), exact round trip, and how segmentations react to realistic spelling and typing variation. No language model was trained for this report. A tokenizer is not a model: it cannot translate, answer or generate. It only decides how text is cut into ids, which sets how much text fits in a context window, what training and inference cost per sentence, and how consistently a model sees the same word.
- **Released tokenizer:** `F:\Hindko\tokenizer\tokenizer.json` (SentencePiece Unigram, 32,768 ids; sha256 `49f301c52363…`, checked at load). Runner-up: `R2-A10-MinGram-P1r3-D2-48k` (the pre-registered pick). External tokenizers: the study's offline copies (`baselines/manifest.json`), each with its own native encoder, no BOS/EOS added.
- **Input form.** Hindko, Urdu, Punjabi, Saraiki and Pashto text is given to every tokenizer after `hp.normalize` 1.0.1, the form the release card recommends (raw-text numbers are in `robustness.json`). English and code are given as found. test_strict is already in that form.
- **Status.** Report-only. test_strict was used once for the LM decision (`analysis/TEST_RESULTS.md`); these intrinsic probes were run afterwards and change nothing about the release.

## 0. Summary

- **Out-of-corpus Hindko.** GotQuestions articles in Southern Hindko are not in the corpus (word-8-gram overlap 0.00 %, spacing-insensitive character-24-gram overlap 0.00 %) and differ in dialect from the training text: 6.7 % of their running words never occur in the training text (test_strict: 1.6 %). On them the released tokenizer gives 6.18 bytes/token (test_strict: 7.28), rank 2 of 11 (behind MinGram-48k (runner-up) 6.54). The best external tokenizer there is RoBERTa-Urdu at 4.74, so the released tokenizer uses 23 % fewer tokens than it (35 % fewer than GPT-4o).
- **Noisy out-of-corpus Hindko** (UI strings, PDF text layer, OCR), bytes/token: translatewiki UI messages: 3.62 vs best external mT5 3.73 (rank 3); 'Chup di Kahani' PDF text: 4.41 vs best external RoBERTa-Urdu 4.20 (rank 2); hindko.org OCR text: 4.97 vs best external BLOOM 4.91 (rank 2). The released tokenizer is ahead of every external tokenizer on 3 of the 4 out-of-corpus sets; on the translatewiki UI messages mT5 is 2.9 % more efficient, but it does not reproduce that text exactly (round trip 0/1 documents). On these three noisy sets the gap is small: the released tokenizer needs -4.6 % to +2.9 % tokens relative to the best external tokenizer.
- **Weak spot: Arabic-keyboard letters.** Typing ي ك ه instead of ی ک ہ (every occurrence) costs the released tokenizer +92 % tokens on test_strict and changes the segmentation of 95 % of the words it touches (GPT-4o: +6 % tokens, 48 % of touched words; RoBERTa-Urdu, the other Urdu-script-specialised vocabulary: +102 %). After that change the released tokenizer needs 383,125 tokens, more than 4 of the 9 external tokenizers (GPT-4o (o200k), Gemma 3, BLOOM, mT5). `normalize()` recovers part of it (+61 % tokens vs clean text), because it folds YEH and KAF only inside provably Urdu-script words and never folds HEH. Folding every ي→ی and ك→ک before encoding (appropriate for Hindko/Urdu text; it also rewrites Arabic quotations) leaves +26 %, and the released tokenizer again needs fewer tokens than every external tokenizer on the same folded text; the rest is the cost of ه, which cannot be folded blindly.
- **The other six probes** (harakat, digits, punctuation spacing, ZWNJ, and the two spellings of the retroflex nasal) change the released tokenizer's token count by -0.5 % to +7.6 %. After each of them it still needs fewer tokens than every external tokenizer. In relative terms it reacts more than the external tokenizers to a space before ۔/، (+7.6 % against at most +1.8 %; the corpus never has that space), because many of its word pieces end in the full stop.
- **Code-mixed Hindko + Urdu + English (100 synthetic lines).** The released tokenizer gives 5.66 bytes/token on the mixed lines (7.86 on the same lines without the insertions); the best external tokenizer gives 5.55 (BLOOM). The inserted English and Urdu material costs the released tokenizer 2.77 bytes per token, against 5.93 for GPT-4o.
- **Other languages (honest characterisation).** Bytes/token of the released tokenizer: Urdu 6.59 (1 of 9 external tokenizers more efficient; best RoBERTa-Urdu 6.96); Punjabi (Shahmukhi) 6.53 (0 of 9 external tokenizers more efficient; best RoBERTa-Urdu 5.83); Saraiki 4.95 (0 of 9 external tokenizers more efficient; best RoBERTa-Urdu 4.42); Pashto 3.21 (8 of 9 external tokenizers more efficient; best GPT-4o (o200k) 4.86); English prose 1.82 (9 of 9 external tokenizers more efficient; best Llama 4 4.76); Python code 1.37 (9 of 9 external tokenizers more efficient; best Llama 3 4.16). It is a Hindko-specialised vocabulary. Because Hindko shares script and much vocabulary with its neighbours, it is also efficient on Urdu, Punjabi (Shahmukhi), Saraiki; it is clearly worse than general-purpose tokenizers on Pashto, English prose, Python code, and it is not suitable as the only tokenizer of a multilingual, English or coding model.
- **Lossless.** The released tokenizer round-trips every document of every set in this report (1692/1692). mT5 does not (it normalizes text, e.g. line breaks become spaces).
- **Consistency check.** The four PLAN §4.2 probes reproduce the study's one-shot test runs exactly for 11 of 11 tokenizers (tokens, words affected, changed segmentations).

## 1. Tokenizers

| tokenizer | family | vocab | test_strict bytes/token | fertility | lossless on test_strict |
|---|---|---:|---:|---:|---:|
| **Hindko SP-32k (released)** | this study: SentencePiece Unigram 32,768 (R2-A4-SPnat-D2-32k) | 32,768 | 7.277 | 1.135 | 491/491 |
| MinGram-48k (runner-up) | this study: MinGram 49,152 (R2-A10-MinGram-P1r3-D2-48k), pre-registered pick | 49,152 | 7.373 | 1.125 | 491/491 |
| GPT-4o (o200k) | OpenAI o200k_base, byte-level BPE 200k | 200,000 | 4.338 | 1.931 | 491/491 |
| Gemma 3 | Google Gemma 3 SentencePiece BPE 262k (= Gemma 4 encodings) | 262,145 | 4.610 | 1.807 | 491/491 |
| Llama 3 | Meta Llama 3 tiktoken BPE 128k | 128,256 | 2.641 | 3.119 | 491/491 |
| Llama 4 | Meta Llama 4 BPE 202k | 201,135 | 3.727 | 2.251 | 491/491 |
| Qwen 3.5 | Alibaba Qwen3.5 byte-level BPE 248k | 248,070 | 3.598 | 2.323 | 491/491 |
| DeepSeek-V3 | DeepSeek-V3 byte-level BPE 128k (= V4 encodings) | 128,815 | 3.372 | 2.476 | 491/491 |
| BLOOM | BigScience BLOOM byte-level BPE 250k | 250,680 | 5.122 | 1.633 | 491/491 |
| RoBERTa-Urdu | UrduHack RoBERTa-Urdu byte-level BPE 52k (best external on Hindko test) | 52,000 | 5.709 | 1.459 | 491/491 |
| mT5 | Google mT5 SentencePiece Unigram 250k (normalizes: not lossless) | 250,100 | 3.912 | 1.874 | 175/491 |

## 2. Out-of-corpus Hindko

### 2.1 Which samples are really outside the corpus

Every survey sample on disk (`F:\Hindko\_web\survey\_samples`, `…\samples`, `…\corpora_work\samples`) that could be Hindko running text was split into documents and compared with the whole released corpus (`hindko_dataset_permissive.jsonl` + `hindko_dataset.jsonl`, all splits and tiers: 5,397,086 8-grams, 18,733,468 character 24-grams). Keys: NFC, harakat and invisible characters removed, Arabic-keyboard letters folded. A document is kept when its word-8-gram overlap is < 10 %, its character-24-gram overlap (spacing-insensitive, catches PDF texts with joined words) is < 20 %, it has ≥ 20 words, ≥ 60 % Arabic-script letters, no replacement characters, and more Hindko/Punjabi than Urdu function words.

| source | role | docs | docs passing the rules | used as out-of-corpus Hindko | word-8-gram overlap (all docs) | char-24-gram overlap (all docs) | note |
|---|---|---:|---:|---:|---:|---:|---|
| `gotquestions_southern_hindko` | candidate | 9 | 9 | 9 | 0.0 % | 0.0 % | GotQuestions.org articles in Southern Hindko (hnd), one article per line |
| `surah_yasin_hindko_pdf` | candidate | 1 | 0 | 0 | 0.0 % | 0.3 % | Surah Yasin with its meaning in Hindko (Internet Archive PDF text layer: joined words, bidi marks); dropped: Urdu function words >= Hindko ones |
| `chup_di_kahani_pdf` | candidate | 1 | 1 | 1 | 0.0 % | 0.0 % | 'Chup di Kahani', Hindko short stories (Internet Archive PDF text layer) |
| `translatewiki_hno` | candidate | 1 | 1 | 1 | 0.0 % | 0.0 % | MediaWiki interface messages in Northern Hindko (OPUS translatewiki), one file |
| `fineweb2_pnb_hindko_candidates` | screened, not used | 85 | 62 | 0 | 46.1 % | 45.9 % | FineWeb-2 pnb_Arab pages flagged as Hindko by the survey (passing rows mostly Punjabi: not used) |
| `fineweb2_pnb_hindko_domains` | screened, not used | 162 | 52 | 0 | 70.0 % | 69.9 % | FineWeb-2 pnb_Arab rows from Hindko-related domains (passing rows mostly Punjabi poetry: not used) |
| `finepdfs_hindko_org` | candidate | 12 | 10 | 10 | 0.0 % | 0.0 % | hindko.org PDFs as OCR'd by FinePDFs (Hindko, but heavily OCR-corrupted letters: a noisy-input set); dropped: Urdu function words >= Hindko ones |
| `bible_for_children_pdf` | candidate | 1 | 0 | 0 | 0.0 % | 0.0 % | Bible for Children in Hindko (PDF text layer); dropped: U+FFFD |
| `manglori_tareekh_djvu` | candidate | 1 | 0 | 0 | 0.0 % | 0.0 % | History of Hindko language and literature (Internet Archive djvu OCR); dropped: Urdu function words >= Hindko ones |
| `ctrl_common_voice_hno` | control: known corpus source | 1 | 0 | 0 | 97.7 % | 97.8 % | Common Voice 26 Hindko sentences (corpus source common_voice_hno) |
| `ctrl_omnilingual_hno` | control: known corpus source | 851 | 0 | 0 | 99.5 % | 99.4 % | Omnilingual ASR Hindko transcripts (corpus source omnilingual_asr_hno) |
| `ctrl_gandhara_posts` | control: known corpus source | 69 | 1 | 0 | 96.5 % | 97.3 % | gandharahindko.com posts (corpus source web_gandharahindko) |
| `ctrl_hindko_org` | control: known corpus source | 16 | 0 | 0 | 56.1 % | 58.7 % | hindko.org pages (corpus source web_hindko_org) |
| `ctrl_aaprihindko` | control: known corpus source | 49 | 0 | 0 | 80.4 % | 80.4 % | aaprihindko.com via the Wayback Machine (corpus source web_aaprihindko) |
| `ctrl_hindkomaza` | control: known corpus source | 2 | 0 | 0 | 94.4 % | 94.2 % | hindkomaza blog (web_hindkomaza) |
| `ctrl_blog_hindko_pk` | control: known corpus source | 2 | 0 | 0 | 84.6 % | 83.2 % | hindko-pk blog |
| `ctrl_blog_hindkopoint` | control: known corpus source | 1 | 0 | 0 | 54.1 % | 49.3 % | hindkopoint blog |
| `urdu_about_hindko` | control: not Hindko | 3 | 0 | 0 | 1.4 % | 1.9 % | Urdu prose about Hindko |
| `urduweb_zeerak` | control: not Hindko | 1 | 1 | 0 | 3.8 % | 6.7 % | urduweb forum thread comparing Punjabi/Saraiki/Pothwari/Hindko words (word lists, not running Hindko) |

- **Controls behave as expected.** 25 random test_strict documents: 100 % word-8-gram and 100 % character-24-gram overlap. The known corpus sources (Common Voice, Omnilingual, gandharahindko.com, aaprihindko, …) show 54 %–99 % word-8-gram overlap; the parts not found are pages or sentences the corpus did not take.
- **Used as out-of-corpus Hindko:** `gotquestions_southern_hindko` (9 docs); `chup_di_kahani_pdf` (1 docs); `translatewiki_hno` (1 docs); `finepdfs_hindko_org` (10 docs). The highest overlap of any kept set with the corpus, on either index, is 0.01 %. The Surah Yasin PDF (Quranic Arabic with an Urdu-leaning translation), the Bible-for-Children PDF (broken font encoding: replacement characters) and the djvu OCR of an Urdu-language history book were dropped by the rules. The FineWeb-2 rows the survey had flagged pass the rules, but by inspection they are mostly Punjabi (Punjabi poetry sites, pnb.wikipedia.org), so they are not used as Hindko.

### 2.2 Results

**Bytes per token** (higher = fewer tokens for the same text; normalized input):

| tokenizer | test_strict (in-distribution) | GotQuestions, Southern Hindko | translatewiki, Northern Hindko UI messages | 'Chup di Kahani' stories | hindko.org PDFs via FinePDFs |
|---|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | **7.28** | **6.18** | **3.62** | **4.41** | **4.97** |
| MinGram-48k (runner-up) | 7.37 | 6.54 | 3.68 | 4.73 | 5.18 |
| GPT-4o (o200k) | 4.34 | 4.00 | 3.22 | 3.40 | 4.34 |
| Gemma 3 | 4.61 | 4.15 | 3.21 | 3.50 | 4.53 |
| Llama 3 | 2.64 | 2.44 | 2.42 | 2.41 | 2.93 |
| Llama 4 | 3.73 | 3.57 | 2.97 | 3.15 | 3.92 |
| Qwen 3.5 | 3.60 | 3.45 | 2.83 | 3.02 | 3.78 |
| DeepSeek-V3 | 3.37 | 3.26 | 2.78 | 2.90 | 3.52 |
| BLOOM | 5.12 | 4.61 | 3.56 | 3.81 | 4.91 |
| RoBERTa-Urdu | 5.71 | 4.74 | 3.57 | 4.20 | 4.89 |
| mT5 | 3.91 | 3.67 | 3.73 | 3.88 | 4.13 |
| *released uses x % fewer tokens than the best external* | 21.6 % (vs RoBERTa-Urdu) | 23.3 % (vs RoBERTa-Urdu) | -2.9 % (vs mT5) | 4.6 % (vs RoBERTa-Urdu) | 1.2 % (vs BLOOM) |

**Fertility** (tokens per whitespace word; lower is better) and **lossless documents**:

| tokenizer | GotQuestions, Southern Hindko | translatewiki, Northern Hindko UI messages | 'Chup di Kahani' stories | hindko.org PDFs via FinePDFs | lossless (all 4 sets) |
|---|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | 1.399 | 1.896 | 5.007 | 1.477 | 21/21 |
| MinGram-48k (runner-up) | 1.337 | 1.893 | 4.655 | 1.435 | 21/21 |
| GPT-4o (o200k) | 2.225 | 2.350 | 6.801 | 1.792 | 21/21 |
| Gemma 3 | 2.146 | 2.271 | 6.475 | 1.693 | 21/21 |
| Llama 3 | 3.535 | 3.211 | 9.507 | 2.647 | 21/21 |
| Llama 4 | 2.489 | 2.574 | 7.317 | 1.983 | 21/21 |
| Qwen 3.5 | 2.587 | 2.640 | 7.545 | 2.045 | 21/21 |
| DeepSeek-V3 | 2.716 | 2.760 | 7.908 | 2.213 | 21/21 |
| BLOOM | 1.928 | 2.030 | 6.045 | 1.578 | 21/21 |
| RoBERTa-Urdu | 1.896 | 1.965 | 5.392 | 1.529 | 21/21 |
| mT5 | 2.151 | 2.085 | 5.751 | 1.710 | 9/21 |

- Set sizes (normalized UTF-8 bytes): GotQuestions, Southern Hindko 70,148; translatewiki, Northern Hindko UI messages 13,784; 'Chup di Kahani' stories 110,089; hindko.org PDFs via FinePDFs 215,859.
- Distance from the training text: share of running words (letters only, harakat removed) never seen in train_D2: test_strict 1.6 %; GotQuestions, Southern Hindko 6.7 %; translatewiki, Northern Hindko UI messages 16.8 %; 'Chup di Kahani' stories 68.1 %; hindko.org PDFs via FinePDFs 20.3 %. The PDF and OCR sets are far from the training text mostly because of extraction damage (joined words, wrong letters), not dialect.
- Raw (un-normalized) input changes the released tokenizer's bytes/token to GotQuestions, Southern Hindko 6.18; translatewiki, Northern Hindko UI messages 3.60; 'Chup di Kahani' stories 3.04; hindko.org PDFs via FinePDFs 4.90 (PDF text carries bidi control characters that `normalize()` removes).
- Byte-fallback pieces used by the released tokenizer (characters outside its vocabulary): GotQuestions, Southern Hindko 2; translatewiki, Northern Hindko UI messages 409; 'Chup di Kahani' stories 269; hindko.org PDFs via FinePDFs 534.

## 3. Perturbation robustness on test_strict

Each probe rewrites all 491 test_strict documents (1,451,026 bytes, 171,769 words). The four PLAN §4.2 probes are `eval/perturb.py`, unchanged; the three new ones are `perturb_extra.py` (same alignment format, so `eval/harness.seg_compare` scores them identically). *Relative token change* = (tokens after − tokens before) / tokens before, over all documents. *Segmentation changed* = share of words whose token boundaries, on the characters the word keeps, are not the same as before; given over the words the probe touched, and over all words.

| probe | what it does | words touched | source |
|---|---|---:|---|
| `harakat` | harakat removed | 8,214 | PLAN 4.2 (i) |
| `digits` | digit script swapped (ASCII <-> Urdu) | 525 | PLAN 4.2 (ii) |
| `punct_space` | space before ۔ / ، toggled | 9,451 | PLAN 4.2 (iii); on this data it inserts the space |
| `zwnj` | ZWNJ inserted in compounds | 230 | PLAN 4.2 (iv) |
| `arabic_keyboard` | Arabic-keyboard letters ي ك ه for ی ک ہ | 88,504 | new: realistic typing |
| `nr_nnu` | retroflex nasal نڑ written ݨ | 3,420 | new: Punjabi/Saraiki orthography |
| `nr_n` | retroflex nasal نڑ written ن | 3,420 | new: Urdu-style spelling |

**Relative token-count change** (closer to 0 is more robust):

| tokenizer | `harakat` | `digits` | `punct_space` | `zwnj` | `arabic_keyboard` | `nr_nnu` | `nr_n` | `arabic_keyboard` then `normalize()` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | -0.45 % | -0.45 % | +7.58 % | +0.49 % | +92.15 % | +2.76 % | +0.40 % | +60.72 % |
| MinGram-48k (runner-up) | -0.33 % | +0.04 % | +5.11 % | +0.46 % | +84.04 % | +2.93 % | +0.30 % | +55.33 % |
| GPT-4o (o200k) | -2.53 % | +0.11 % | +0.34 % | +0.05 % | +6.18 % | +0.76 % | -1.21 % | +1.27 % |
| Gemma 3 | -2.81 % | -0.17 % | +0.54 % | +0.07 % | +9.78 % | -0.15 % | -1.31 % | +3.40 % |
| Llama 3 | -1.69 % | +0.03 % | +1.24 % | +0.04 % | -9.25 % | -0.24 % | -1.56 % | -9.55 % |
| Llama 4 | -2.64 % | +0.07 % | +1.75 % | +0.04 % | +3.73 % | +0.37 % | -1.20 % | +0.28 % |
| Qwen 3.5 | -2.00 % | +0.28 % | +0.00 % | +0.06 % | -1.16 % | +0.28 % | -1.20 % | -2.73 % |
| DeepSeek-V3 | -2.25 % | +0.08 % | +1.58 % | +0.11 % | -1.07 % | +0.50 % | -1.09 % | -2.72 % |
| BLOOM | -2.26 % | +0.39 % | +0.06 % | +0.09 % | +13.70 % | +1.11 % | -1.43 % | +6.39 % |
| RoBERTa-Urdu | -8.61 % | +0.07 % | +0.01 % | +0.16 % | +101.60 % | +1.60 % | -1.38 % | +68.29 % |
| mT5 | -3.28 % | +0.09 % | +1.78 % | +0.01 % | +2.69 % | -0.36 % | -1.10 % | -1.14 % |

**Share of touched words whose segmentation changes** (lower is more robust; all words in brackets):

| tokenizer | `harakat` | `digits` | `punct_space` | `zwnj` | `arabic_keyboard` | `nr_nnu` | `nr_n` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | 13.7 % (0.65 %) | 79.0 % (0.24 %) | 58.3 % (3.21 %) | 90.9 % (0.12 %) | 94.6 % (48.73 %) | 98.3 % (1.96 %) | 37.2 % (0.74 %) |
| MinGram-48k (runner-up) | 9.6 % (0.46 %) | 13.3 % (0.04 %) | 3.6 % (0.20 %) | 87.8 % (0.12 %) | 94.5 % (48.71 %) | 98.5 % (1.96 %) | 23.9 % (0.48 %) |
| GPT-4o (o200k) | 53.0 % (2.53 %) | 67.2 % (0.21 %) | 2.0 % (0.11 %) | 33.9 % (0.05 %) | 47.9 % (24.66 %) | 100.0 % (1.99 %) | 41.9 % (0.83 %) |
| Gemma 3 | 57.3 % (2.74 %) | 35.6 % (0.11 %) | 27.6 % (1.52 %) | 31.3 % (0.04 %) | 49.9 % (25.70 %) | 86.2 % (1.72 %) | 37.8 % (0.75 %) |
| Llama 3 | 50.0 % (2.39 %) | 35.8 % (0.11 %) | 0.0 % (0.00 %) | 37.0 % (0.05 %) | 51.1 % (26.31 %) | 100.0 % (1.99 %) | 43.5 % (0.87 %) |
| Llama 4 | 55.2 % (2.64 %) | 56.2 % (0.17 %) | 0.0 % (0.00 %) | 43.0 % (0.06 %) | 47.0 % (24.23 %) | 100.0 % (1.99 %) | 43.4 % (0.86 %) |
| Qwen 3.5 | 44.0 % (2.11 %) | 100.0 % (0.31 %) | 0.0 % (0.00 %) | 23.5 % (0.03 %) | 45.7 % (23.54 %) | 100.0 % (1.99 %) | 31.9 % (0.64 %) |
| DeepSeek-V3 | 48.0 % (2.29 %) | 64.6 % (0.20 %) | 0.0 % (0.00 %) | 29.6 % (0.04 %) | 52.3 % (26.97 %) | 100.0 % (1.99 %) | 44.1 % (0.88 %) |
| BLOOM | 55.0 % (2.63 %) | 94.7 % (0.29 %) | 1.9 % (0.11 %) | 29.6 % (0.04 %) | 53.7 % (27.68 %) | 100.0 % (1.99 %) | 43.6 % (0.87 %) |
| RoBERTa-Urdu | 81.9 % (3.92 %) | 41.0 % (0.13 %) | 0.3 % (0.02 %) | 77.8 % (0.10 %) | 100.0 % (51.53 %) | 100.0 % (1.99 %) | 47.2 % (0.94 %) |
| mT5 | 69.2 % (3.31 %) | 43.8 % (0.13 %) | 50.6 % (2.79 %) | 38.3 % (0.05 %) | 44.6 % (22.98 %) | 70.2 % (1.40 %) | 32.1 % (0.64 %) |

**Arabic-keyboard input and two fixes** (`ak_mitigation.py`; tokens relative to the clean text; last column: tokens after the YEH/KAF fold ÷ the released tokenizer's):

| tokenizer | typed with ي ك ه | then `normalize()` | then fold ي→ی, ك→ک (ه left) | × released after the fold |
|---|---:|---:|---:|---:|
| Hindko SP-32k (released) | +92.1 % | +60.7 % | +25.8 % | 1.000 |
| MinGram-48k (runner-up) | +84.0 % | +55.3 % | +26.0 % | 0.989 |
| GPT-4o (o200k) | +6.2 % | +1.3 % | -0.7 % | 1.325 |
| Gemma 3 | +9.8 % | +3.4 % | +0.0 % | 1.256 |
| Llama 3 | -9.3 % | -9.6 % | -10.5 % | 1.961 |
| Llama 4 | +3.7 % | +0.3 % | -2.1 % | 1.521 |
| Qwen 3.5 | -1.2 % | -2.7 % | -2.5 % | 1.568 |
| DeepSeek-V3 | -1.1 % | -2.7 % | -4.2 % | 1.644 |
| BLOOM | +13.7 % | +6.4 % | +4.4 % | 1.179 |
| RoBERTa-Urdu | +101.6 % | +68.3 % | +24.3 % | 1.260 |
| mT5 | +2.7 % | -1.1 % | -2.0 % | 1.449 |

**Tokens after the probe, relative to the released tokenizer after the same probe** (× tokens; > 1 means the released tokenizer still needs fewer tokens):

| tokenizer | clean | `harakat` | `digits` | `punct_space` | `zwnj` | `arabic_keyboard` | `nr_nnu` | `nr_n` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MinGram-48k (runner-up) | 0.987 | 0.988 | 0.992 | 0.964 | 0.987 | 0.945 | 0.989 | 0.986 |
| GPT-4o (o200k) | 1.678 | 1.643 | 1.687 | 1.565 | 1.671 | 0.927 | 1.645 | 1.651 |
| Gemma 3 | 1.578 | 1.541 | 1.583 | 1.475 | 1.572 | 0.902 | 1.534 | 1.552 |
| Llama 3 | 2.755 | 2.721 | 2.769 | 2.593 | 2.743 | 1.301 | 2.675 | 2.701 |
| Llama 4 | 1.953 | 1.910 | 1.963 | 1.847 | 1.944 | 1.054 | 1.907 | 1.921 |
| Qwen 3.5 | 2.022 | 1.991 | 2.037 | 1.880 | 2.014 | 1.040 | 1.974 | 1.990 |
| DeepSeek-V3 | 2.158 | 2.119 | 2.169 | 2.038 | 2.150 | 1.111 | 2.110 | 2.126 |
| BLOOM | 1.421 | 1.395 | 1.433 | 1.322 | 1.415 | 0.841 | 1.398 | 1.395 |
| RoBERTa-Urdu | 1.275 | 1.170 | 1.281 | 1.185 | 1.271 | 1.337 | 1.260 | 1.252 |
| mT5 | 1.860 | 1.807 | 1.870 | 1.760 | 1.851 | 0.994 | 1.803 | 1.832 |

Reading the tables:

- **Relative change is not the whole story.** A tokenizer that splits Hindko into many small pieces has little left to break, so its relative change can be small while its absolute token count stays much higher. The last table below shows the absolute effect.
- **Arabic-keyboard letters are the released tokenizer's clearest weakness.** ی, ک and ہ are among the most frequent letters in Hindko, and its training text (train_D2, 14,731,915 characters) writes them in the Urdu codepoints almost without exception: ARABIC YEH 58 times, ARABIC KAF 19, ARABIC HEH 46. Only 9 of the 32,768 vocabulary pieces contain any of ي ك ه. So 95 % of the touched words are re-cut, at 2.08 extra tokens per touched word. The external tokenizers change by -9.3 % to +101.6 % on the same probe. Mitigation for applications: apply `normalize()` (it recovers YEH/KAF inside provably Urdu-script words) and, where the input is known to be Hindko or Urdu, fold ي→ی and ك→ک before encoding; ه cannot be folded blindly, because Arabic-keyboard users also type it for ھ. With the YEH/KAF fold the released tokenizer's cost falls to +25.8 % (table above).
- **The retroflex nasal.** Writing نڑ as ݨ (the Punjabi/Saraiki convention) costs +2.76 % tokens; as plain ن, +0.40 %. Both spellings occur in the corpus, so the vocabulary has pieces for them.
- **Removing harakat and swapping the digit script** change every tokenizer's token count by -8.61 % to +0.39 % (removing harakat shortens the text).
- **Space before ۔ / ،**: the corpus never has it (0 occurrences in train and dev, `eval/perturb.py`). The released tokenizer spends +7.6 % more tokens, its largest change among the six small probes, because many of its word pieces end in the full stop (e.g. `▁آئی۔` in sentence 1 of §6), and a space breaks them off.

Cross-check of the four PLAN probes against the study's one-shot test runs:

| tokenizer | reference file | identical |
|---|---|---|
| Hindko SP-32k (released) | `release_card\finalize\test_intrinsic\tokenizer_json\summary.json` | yes |
| MinGram-48k (runner-up) | `eval\results\test_strict\R2-A10-MinGram-P1r3-D2-48k\summary.json` | yes |
| GPT-4o (o200k) | `eval\results\test_strict\gpt-4o\summary.json` | yes |
| Gemma 3 | `eval\results\test_strict\gemma-3\summary.json` | yes |
| Llama 3 | `eval\results\test_strict\llama-3\summary.json` | yes |
| Llama 4 | `eval\results\test_strict\llama-4\summary.json` | yes |
| Qwen 3.5 | `eval\results\test_strict\qwen-3.5\summary.json` | yes |
| DeepSeek-V3 | `eval\results\test_strict\deepseek-v3\summary.json` | yes |
| BLOOM | `eval\results\test_strict\bloom\summary.json` | yes |
| RoBERTa-Urdu | `eval\results\test_strict\roberta-urdu\summary.json` | yes |
| mT5 | `eval\results\test_strict\mt5\summary.json` | yes |

## 4. Code-mixed text

- **Synthetic set (100 lines, `sets/codemixed.jsonl`).** One test_strict Hindko sentence (8–25 words) from each of 100 randomly chosen documents; 1–3 English words inserted at random word boundaries (from a fixed list of 40 words common in Pakistani code-switching: `meeting`, `online`, `WhatsApp`, `exam`, …); with probability 0.25 an English opener (`Actually,` …), with 0.25 an English closer (`Thank you!` …), with 0.35 an Urdu clause (12 fixed clauses such as `کوئی بات نہیں۔`). Seed `20260927-codemix`. The insertions are placed at random, so the lines are realistic at the level of scripts and words, not always grammatical.
- **Natural set.** The 8 test_strict documents labelled `mixed` (Hindko text quoting Urdu songs, and Urdu ghazals inside Hindko books; 6,579 bytes).
- Example line: `کمپیوٹر آیا تا دفتری internet ملازمتاں دے مستقبل تے class سوال اُٹھے۔`

| tokenizer | synthetic mixed: bytes/token | same lines without insertions | inserted material: bytes/token | tokens / line | lossless | natural mixed: bytes/token |
|---|---:|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | **5.66** | **7.86** | **2.77** | **30.1** | **100/100** | **6.10** |
| MinGram-48k (runner-up) | 5.78 | 7.73 | 2.98 | 29.4 | 100/100 | 6.41 |
| GPT-4o (o200k) | 4.68 | 4.43 | 5.93 | 36.3 | 100/100 | 4.26 |
| Gemma 3 | 5.04 | 4.76 | 6.48 | 33.7 | 100/100 | 4.50 |
| Llama 3 | 2.89 | 2.77 | 3.48 | 58.8 | 100/100 | 2.61 |
| Llama 4 | 4.07 | 3.82 | 5.39 | 41.8 | 100/100 | 3.79 |
| Qwen 3.5 | 3.92 | 3.70 | 5.02 | 43.4 | 100/100 | 3.63 |
| DeepSeek-V3 | 3.67 | 3.50 | 4.49 | 46.4 | 100/100 | 3.33 |
| BLOOM | 5.55 | 5.43 | 6.03 | 30.7 | 100/100 | 4.79 |
| RoBERTa-Urdu | 4.91 | 5.98 | 2.94 | 34.7 | 100/100 | 5.37 |
| mT5 | 4.10 | 3.95 | 4.81 | 41.5 | 100/100 | 4.12 |

- The inserted English words and Urdu clauses cost the released tokenizer 2.77 bytes per token; 9 of the 9 external tokenizers encode that material more cheaply (up to 6.48, Gemma 3). On whole lines, where Hindko is most of the text, the fewest tokens per line are MinGram-48k (runner-up)'s (29.4); the released tokenizer needs 30.1 and the best external tokenizer, BLOOM, 30.7.
- The more English a line contains, the smaller the released tokenizer's advantage; for mostly-English text it is at a disadvantage (§5).

## 5. Other languages (where the released tokenizer is and is not appropriate)

| tokenizer | Urdu | Punjabi (Shahmukhi) | Saraiki | Pashto | English prose | Python code |
|---|---:|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | **6.59** | **6.53** | **4.95** | **3.21** | **1.82** | **1.37** |
| MinGram-48k (runner-up) | 6.91 | 6.80 | 5.15 | 3.34 | 2.04 | 1.49 |
| GPT-4o (o200k) | 5.35 | 4.38 | 3.73 | 4.86 | 4.72 | 4.14 |
| Gemma 3 | 5.67 | 4.64 | 3.93 | 4.43 | 4.64 | 3.62 |
| Llama 3 | 2.79 | 2.68 | 2.39 | 3.29 | 4.69 | 4.16 |
| Llama 4 | 4.54 | 3.78 | 3.30 | 4.11 | 4.76 | 4.14 |
| Qwen 3.5 | 4.28 | 3.60 | 3.15 | 3.64 | 4.64 | 3.86 |
| DeepSeek-V3 | 3.59 | 3.39 | 3.00 | 3.40 | 4.63 | 3.98 |
| BLOOM | 6.49 | 5.10 | 4.09 | 3.48 | 4.70 | 4.15 |
| RoBERTa-Urdu | 6.96 | 5.83 | 4.42 | 3.00 | 2.00 | 1.50 |
| mT5 | 4.54 | 3.99 | 3.67 | 4.18 | 3.84 | 3.62 |
| *released: rank among the 11* | 3 | 2 | 2 | 10 | 11 | 11 |
| *released tokens ÷ best external tokens* | 1.05× (RoBERTa-Urdu) | 0.89× (RoBERTa-Urdu) | 0.89× (RoBERTa-Urdu) | 1.51× (GPT-4o (o200k)) | 2.61× (Llama 4) | 3.05× (Llama 3) |

Bytes per token; higher is better. Fertility and lossless counts are in `robustness.json`.

- **Urdu**: 300 documents, 1,319,154 bytes; `F:\Hindko\_web\ref\urd_Arab.jsonl`; 300 sampled, 0 dropped because they overlap the Hindko corpus
- **Punjabi (Shahmukhi)**: 300 documents, 1,222,473 bytes; `F:\Hindko\_web\ref\pnb_Arab.jsonl`; 300 sampled, 0 dropped because they overlap the Hindko corpus
- **Saraiki**: 200 documents, 1,228,900 bytes; `F:\Hindko\_web\ref\skr_Arab.jsonl`; 200 sampled, 0 dropped because they overlap the Hindko corpus
- **Pashto**: 200 documents, 1,179,034 bytes; `F:\Hindko\_web\ref\pbt_Arab.jsonl`; 200 sampled, 0 dropped because they overlap the Hindko corpus
- **English prose**: 50 documents, 18,536 bytes; `<python 3.11>\Lib\site-packages\*.dist-info\METADATA (long descriptions)`. paragraphs with 250-1500 chars, >=40 words, >=99.5% ASCII, >=88% letters/spaces, ending in '.', no code/list/heading markup, no URL, no licence text; round robin over packages sorted by dist-info name (1st qualifying paragraph of each, then 2nd), first 50
- **Python code**: 30 documents, 648,218 bytes; `<python 3.11>\Lib`

- **Urdu and Punjabi:** the released tokenizer ranks 3 and 2 of 11. Hindko shares script, much vocabulary and many loanwords with both, and its training text (96.8 % Hindko-labelled, 3.1 % mixed Hindko–Urdu by characters) quotes Urdu. This is a side effect, not a design goal: it was not trained or selected on Urdu or Punjabi text.
- **Saraiki and Pashto:** the training text contains the Saraiki implosives ٻ ڄ ݙ ڳ 0 times and the Pashto letters ټ ډ ړ ږ ښ ګ ڼ 2 times in total, so the tokenizer spells these letters with byte-fallback pieces (9.0 % of its Saraiki tokens and 19.0 % of its Pashto tokens are byte pieces). Saraiki still ranks 2 of 11, because the rest of its vocabulary is close to Hindko; Pashto, a different language family, ranks 10 of 11.
- **English and Python code:** only 528 of the 32,768 pieces contain Latin letters (mostly names from the corpus, e.g. botanical terms). The released tokenizer needs 2.6× GPT-4o's tokens on English prose and 3.0× on Python code. It is lossless on both, so nothing breaks, but it should not be used as-is for an English or coding model.

## 6. Example segmentations (for the model card)

Three natural sentences from the strict test split (held out from tokenizer training), each tokenizer with its own encoder, no special tokens. `▁` marks a space the token carries; `▯` stands for one byte of a letter that the token splits (byte-level BPE can cut a 2-byte Urdu letter into two tokens). Token boundaries are shown as `|`. The text is right-to-left, so in most viewers the first token is at the right.

| tokenizer | sentence 1 (14 words) | sentence 2 (20 words) | sentence 3 (20 words) | total | × released |
|---|---:|---:|---:|---:|---:|
| Hindko SP-32k (released) | **14** | **20** | **20** | **54** | **1.00** |
| MinGram-48k (runner-up) | 15 | 21 | 21 | 57 | 1.06 |
| GPT-4o (o200k) | 29 | 39 | 38 | 106 | 1.96 |
| Gemma 3 | 27 | 36 | 35 | 98 | 1.81 |
| Llama 3 | 58 | 66 | 53 | 177 | 3.28 |
| Llama 4 | 35 | 42 | 42 | 119 | 2.20 |
| Qwen 3.5 | 36 | 47 | 45 | 128 | 2.37 |
| DeepSeek-V3 | 41 | 47 | 43 | 131 | 2.43 |
| BLOOM | 25 | 34 | 27 | 86 | 1.59 |
| RoBERTa-Urdu | 21 | 26 | 28 | 75 | 1.39 |
| mT5 | 31 | 40 | 42 | 113 | 2.09 |

**Sentence 1** (book, test_strict): ایہہ بہت کہٹ لوک جانڑدین کہ ڈرامے دی پیدائش کسراں تے کتھے ہوئی آئی۔

*Approximate English:* Very few people know how and where drama was born.

| tokenizer | tokens | segmentation |
|---|---:|---|
| Hindko SP-32k (released) | 14 | ▁ایہہ \| ▁بہت \| ▁کہٹ \| ▁لوک \| ▁جانڑدین \| ▁کہ \| ▁ڈرامے \| ▁دی \| ▁پیدائش \| ▁کسراں \| ▁تے \| ▁کتھے \| ▁ہوئی \| ▁آئی۔ |
| MinGram-48k (runner-up) | 15 | ایہہ \| ▁بہت \| ▁کہٹ \| ▁لوک \| ▁جانڑدین \| ▁کہ \| ▁ڈرامے \| ▁دی \| ▁پیدائش \| ▁کسراں \| ▁تے \| ▁کتھے \| ▁ہوئی \| ▁آئی \| ۔ |
| GPT-4o (o200k) | 29 | ای \| ہ \| ہ \| ▁بہت \| ▁کہ \| ٹ \| ▁لو \| ک \| ▁جان \| ڑ \| د \| ین \| ▁کہ \| ▁ڈ \| رام \| ے \| ▁دی \| ▁پید \| ائش \| ▁کس \| را \| ں \| ▁ت \| ے \| ▁کت \| ھے \| ▁ہوئی \| ▁آئی \| ۔ |
| Gemma 3 | 27 | ایہ \| ہ \| ▁بہت \| ▁کہ \| ٹ \| ▁لو \| ک \| ▁جان \| ڑ \| د \| ین \| ▁کہ \| ▁ڈ \| رام \| ے \| ▁دی \| ▁پید \| ائش \| ▁کس \| را \| ں \| ▁تے \| ▁کت \| ھے \| ▁ہوئی \| ▁آئی \| ۔ |
| Llama 3 | 58 | ای \| ▯ \| ▯ \| ▯ \| ▯ \| ▁ب \| ▯ \| ▯ \| ت \| ▁ک \| ▯ \| ▯ \| ٹ \| ▁لو \| ک \| ▁جان \| ▯ \| ▯ \| د \| ین \| ▁ک \| ▯ \| ▯ \| ▁ \| ▯ \| ▯ \| ر \| ام \| ▯ \| ▯ \| ▁دی \| ▁پ \| ید \| ائ \| ش \| ▁ک \| سر \| ا \| ▯ \| ▯ \| ▁ت \| ▯ \| ▯ \| ▁ک \| ت \| ھ \| ▯ \| ▯ \| ▁ \| ▯ \| ▯ \| وئ \| ی \| ▁آ \| ئ \| ی \| ▯ \| ▯ |
| Llama 4 | 35 | ای \| ہ \| ہ \| ▁بہت \| ▁کہ \| ٹ \| ▁لو \| ک \| ▁جان \| ڑ \| د \| ین \| ▁کہ \| ▁ڈ \| ر \| ام \| ے \| ▁دی \| ▁پید \| ائ \| ش \| ▁ک \| سر \| اں \| ▁ت \| ے \| ▁ک \| ت \| ھ \| ے \| ▁ہو \| ئی \| ▁آ \| ئی \| ۔ |
| Qwen 3.5 | 36 | ای \| ہ \| ہ \| ▁بہ \| ت \| ▁کہ \| ٹ \| ▁لو \| ک \| ▁جان \| ڑ \| د \| ین \| ▁کہ \| ▁ڈ \| رام \| ے \| ▁دی \| ▁پ \| ید \| ائ \| ش \| ▁ک \| س \| را \| ں \| ▁ت \| ے \| ▁ک \| ت \| ھے \| ▁ہو \| ئی \| ▁آ \| ئی \| ۔ |
| DeepSeek-V3 | 41 | ای \| ہ \| ہ \| ▁ب \| ہ \| ت \| ▁ک \| ہ \| ٹ \| ▁لو \| ک \| ▁جان \| ڑ \| د \| ین \| ▁ک \| ہ \| ▁ \| ڈ \| ر \| ام \| ے \| ▁دی \| ▁پید \| ائ \| ش \| ▁ک \| سر \| اں \| ▁ت \| ے \| ▁ک \| ت \| ھ \| ے \| ▁ہ \| وئ \| ی \| ▁آ \| ئی \| ۔ |
| BLOOM | 25 | ا \| یہ \| ہ \| ▁بہت \| ▁کہ \| ٹ \| ▁لوک \| ▁جان \| ڑ \| د \| ین \| ▁کہ \| ▁ڈرام \| ے \| ▁دی \| ▁پیدائش \| ▁ک \| سر \| اں \| ▁تے \| ▁کت \| ھے \| ▁ہوئی \| ▁آئی \| ۔ |
| RoBERTa-Urdu | 21 | ایہ \| ہ \| ▁بہت \| ▁کہ \| ٹ \| ▁لوک \| ▁جان \| ڑ \| دین \| ▁کہ \| ▁ڈرامے \| ▁دی \| ▁پیدائش \| ▁کسر \| اں \| ▁تے \| ▁کت \| ھے \| ▁ہوئی \| ▁آئی \| ۔ |
| mT5 | 31 | ▁ای \| ہ \| ہ \| ▁ب \| ہت \| ▁کہ \| ٹ \| ▁ \| لوک \| ▁جان \| ڑ \| دین \| ▁کہ \| ▁ڈ \| رام \| ے \| ▁دی \| ▁پی \| دائ \| ش \| ▁کس \| ر \| اں \| ▁ \| تے \| ▁کت \| ھے \| ▁ہوئ \| ی \| ▁آئی \| ۔ |

**Sentence 2** (newspaper, test_strict): اج اساں اگر آپڑیں بزرگاں دی انہاں گلاں تے غور نہ کیتا تے پچھتاوے دے سوا کج ہتھ نہ آسی۔

*Approximate English:* If we do not reflect on these words of our elders today, we will be left with nothing but regret.

| tokenizer | tokens | segmentation |
|---|---:|---|
| Hindko SP-32k (released) | 20 | ▁اج \| ▁اساں \| ▁اگر \| ▁آپڑیں \| ▁بزرگاں \| ▁دی \| ▁انہاں \| ▁گلاں \| ▁تے \| ▁غور \| ▁نہ \| ▁کیتا \| ▁تے \| ▁پچھتاوے \| ▁دے \| ▁سوا \| ▁کج \| ▁ہتھ \| ▁نہ \| ▁آسی۔ |
| MinGram-48k (runner-up) | 21 | اج \| ▁اساں \| ▁اگر \| ▁آپڑیں \| ▁بزرگاں \| ▁دی \| ▁انہاں \| ▁گلاں \| ▁تے \| ▁غور \| ▁نہ \| ▁کیتا \| ▁تے \| ▁پچھتاوے \| ▁دے \| ▁سوا \| ▁کج \| ▁ہتھ \| ▁نہ \| ▁آسی \| ۔ |
| GPT-4o (o200k) | 39 | اج \| ▁اس \| اں \| ▁اگر \| ▁آپ \| ڑ \| یں \| ▁بزرگ \| اں \| ▁دی \| ▁انہ \| اں \| ▁گ \| لا \| ں \| ▁ت \| ے \| ▁غور \| ▁نہ \| ▁کی \| تا \| ▁ت \| ے \| ▁پ \| چھ \| تا \| و \| ے \| ▁دے \| ▁س \| وا \| ▁ک \| ج \| ▁ہ \| ت \| ھ \| ▁نہ \| ▁آسی \| ۔ |
| Gemma 3 | 36 | اج \| ▁اس \| اں \| ▁اگر \| ▁آپ \| ڑ \| یں \| ▁بزرگ \| اں \| ▁دی \| ▁انہ \| اں \| ▁گ \| لا \| ں \| ▁تے \| ▁غور \| ▁نہ \| ▁کی \| تا \| ▁تے \| ▁پ \| چھ \| تا \| و \| ے \| ▁دے \| ▁سوا \| ▁ک \| ج \| ▁ہ \| ت \| ھ \| ▁نہ \| ▁آسی \| ۔ |
| Llama 3 | 66 | اج \| ▁اس \| ا \| ▯ \| ▯ \| ▁اگر \| ▁آپ \| ▯ \| ▯ \| ی \| ▯ \| ▯ \| ▁بزرگ \| ا \| ▯ \| ▯ \| ▁دی \| ▁ان \| ▯ \| ▯ \| ا \| ▯ \| ▯ \| ▁گ \| لا \| ▯ \| ▯ \| ▁ت \| ▯ \| ▯ \| ▁غ \| ور \| ▁ن \| ▯ \| ▯ \| ▁ک \| یت \| ا \| ▁ت \| ▯ \| ▯ \| ▁پ \| چ \| ھ \| ت \| او \| ▯ \| ▯ \| ▁د \| ▯ \| ▯ \| ▁س \| وا \| ▁ک \| ج \| ▁ \| ▯ \| ▯ \| ت \| ھ \| ▁ن \| ▯ \| ▯ \| ▁آسی \| ▯ \| ▯ |
| Llama 4 | 42 | اج \| ▁اس \| اں \| ▁اگر \| ▁آپ \| ڑ \| یں \| ▁بزرگ \| اں \| ▁دی \| ▁انہ \| اں \| ▁گ \| لا \| ں \| ▁ت \| ے \| ▁غ \| ور \| ▁نہ \| ▁ک \| یت \| ا \| ▁ت \| ے \| ▁پ \| چھ \| ت \| او \| ے \| ▁د \| ے \| ▁س \| وا \| ▁ک \| ج \| ▁ہ \| ت \| ھ \| ▁نہ \| ▁آسی \| ۔ |
| Qwen 3.5 | 47 | اج \| ▁اسا \| ں \| ▁اگر \| ▁آپ \| ڑ \| یں \| ▁ب \| زر \| گا \| ں \| ▁دی \| ▁ان \| ہ \| ا \| ں \| ▁گ \| لا \| ں \| ▁ت \| ے \| ▁غ \| ور \| ▁نہ \| ▁ک \| یت \| ا \| ▁ت \| ے \| ▁پ \| چ \| ھ \| تا \| و \| ے \| ▁د \| ے \| ▁س \| وا \| ▁ک \| ج \| ▁ہ \| ت \| ھ \| ▁نہ \| ▁آسی \| ۔ |
| DeepSeek-V3 | 47 | اج \| ▁اس \| اں \| ▁اگر \| ▁آ \| پ \| ڑ \| ی \| ں \| ▁بزرگ \| اں \| ▁دی \| ▁ان \| ہ \| اں \| ▁گ \| لا \| ں \| ▁ت \| ے \| ▁غ \| ور \| ▁ن \| ہ \| ▁ک \| یت \| ا \| ▁ت \| ے \| ▁پ \| چ \| ھ \| ت \| او \| ے \| ▁دے \| ▁س \| وا \| ▁ک \| ج \| ▁ہ \| ت \| ھ \| ▁ن \| ہ \| ▁آسی \| ۔ |
| BLOOM | 34 | اج \| ▁اس \| اں \| ▁اگر \| ▁آپ \| ڑ \| یں \| ▁بزرگ \| اں \| ▁دی \| ▁انہ \| اں \| ▁گ \| لا \| ں \| ▁تے \| ▁غور \| ▁نہ \| ▁کی \| تا \| ▁تے \| ▁پچھ \| تا \| وے \| ▁دے \| ▁سوا \| ▁ک \| ج \| ▁ہ \| تھ \| ▁نہ \| ▁آ \| سی \| ۔ |
| RoBERTa-Urdu | 26 | اج \| ▁اساں \| ▁اگر \| ▁آپ \| ڑیں \| ▁بزرگ \| اں \| ▁دی \| ▁انہ \| اں \| ▁گلا \| ں \| ▁تے \| ▁غور \| ▁نہ \| ▁کی \| تا \| ▁تے \| ▁پچھتاوے \| ▁دے \| ▁سوا \| ▁کج \| ▁ہتھ \| ▁نہ \| ▁آسی \| ۔ |
| mT5 | 40 | ▁اج \| ▁اس \| اں \| ▁ \| اگر \| ▁آپ \| ڑ \| یں \| ▁بزرگ \| اں \| ▁دی \| ▁ان \| ہاں \| ▁گل \| اں \| ▁ \| تے \| ▁ \| غور \| ▁نہ \| ▁کی \| تا \| ▁ \| تے \| ▁پ \| چھ \| تا \| و \| ے \| ▁ \| دے \| ▁سوا \| ▁ک \| ج \| ▁ \| ہ \| تھ \| ▁نہ \| ▁آس \| ی۔ |

**Sentence 3** (newspaper, test_strict): اسی طراں خیبر پختونخوا اچ بولی جانڑیں ولی دیگر زباناں دے ادب و ثقافت دی بی نمائندگی کیتی گئی اَئی۔

*Approximate English:* In the same way, the literature and culture of the other languages spoken in Khyber Pakhtunkhwa were also represented.

| tokenizer | tokens | segmentation |
|---|---:|---|
| Hindko SP-32k (released) | 20 | ▁اسی \| ▁طراں \| ▁خیبر \| ▁پختونخوا \| ▁اچ \| ▁بولی \| ▁جانڑیں \| ▁ولی \| ▁دیگر \| ▁زباناں \| ▁دے \| ▁ادب \| ▁و \| ▁ثقافت \| ▁دی \| ▁بی \| ▁نمائندگی \| ▁کیتی \| ▁گئی \| ▁اَئی۔ |
| MinGram-48k (runner-up) | 21 | اسی \| ▁طراں \| ▁خیبر \| ▁پختونخوا \| ▁اچ \| ▁بولی \| ▁جانڑیں \| ▁ولی \| ▁دیگر \| ▁زباناں \| ▁دے \| ▁ادب \| ▁و \| ▁ثقافت \| ▁دی \| ▁بی \| ▁نمائندگی \| ▁کیتی \| ▁گئی \| ▁اَئی \| ۔ |
| GPT-4o (o200k) | 38 | اسی \| ▁ط \| را \| ں \| ▁خی \| بر \| ▁پ \| خت \| ون \| خوا \| ▁اچ \| ▁بول \| ی \| ▁جان \| ڑ \| یں \| ▁ولی \| ▁دیگر \| ▁ز \| ب \| انا \| ں \| ▁دے \| ▁ادب \| ▁و \| ▁ثق \| افت \| ▁دی \| ▁بی \| ▁نمائ \| ندگی \| ▁کی \| تی \| ▁گئی \| ▁ا \| َ \| ئی \| ۔ |
| Gemma 3 | 35 | اسی \| ▁ط \| را \| ں \| ▁خی \| بر \| ▁پ \| خت \| ون \| خوا \| ▁اچ \| ▁بولی \| ▁جان \| ڑ \| یں \| ▁ولی \| ▁دیگر \| ▁زبان \| اں \| ▁دے \| ▁ادب \| ▁و \| ▁ثقاف \| ت \| ▁دی \| ▁بی \| ▁نمائ \| ندگی \| ▁کی \| تی \| ▁گئی \| ▁ا \| َ \| ئی \| ۔ |
| Llama 3 | 53 | اسی \| ▁طر \| ا \| ▯ \| ▯ \| ▁خی \| بر \| ▁پ \| خت \| ون \| خ \| وا \| ▁ا \| چ \| ▁ب \| ولی \| ▁جان \| ▯ \| ▯ \| ی \| ▯ \| ▯ \| ▁ولی \| ▁دیگر \| ▁زبان \| ا \| ▯ \| ▯ \| ▁د \| ▯ \| ▯ \| ▁اد \| ب \| ▁و \| ▁ث \| ق \| افت \| ▁دی \| ▁بی \| ▁نم \| ائ \| ندگی \| ▁ک \| یتی \| ▁گ \| ئ \| ی \| ▁ا \| َ \| ئ \| ی \| ▯ \| ▯ |
| Llama 4 | 42 | اسی \| ▁ط \| را \| ں \| ▁خی \| بر \| ▁پ \| خت \| ون \| خ \| وا \| ▁ا \| چ \| ▁ب \| ولی \| ▁جان \| ڑ \| یں \| ▁ولی \| ▁دیگر \| ▁زبان \| اں \| ▁د \| ے \| ▁اد \| ب \| ▁و \| ▁ث \| ق \| افت \| ▁دی \| ▁بی \| ▁نم \| ائ \| ندگی \| ▁کی \| تی \| ▁گئی \| ▁ا▯ \| ▯ \| ئی \| ۔ |
| Qwen 3.5 | 45 | اس \| ی \| ▁ط \| را \| ں \| ▁خی \| بر \| ▁پ \| خت \| ون \| خوا \| ▁ا \| چ \| ▁ب \| ولی \| ▁جان \| ڑ \| یں \| ▁ولی \| ▁دیگر \| ▁ز \| ب \| انا \| ں \| ▁د \| ے \| ▁اد \| ب \| ▁و \| ▁ثق \| افت \| ▁دی \| ▁بی \| ▁نم \| ائ \| ند \| گی \| ▁ک \| یتی \| ▁گ \| ئی \| ▁ا \| َ \| ئی \| ۔ |
| DeepSeek-V3 | 43 | اسی \| ▁طر \| اں \| ▁خ \| ی \| بر \| ▁پ \| خت \| ون \| خ \| وا \| ▁ا \| چ \| ▁ب \| ولی \| ▁جان \| ڑ \| ی \| ں \| ▁ولی \| ▁دیگر \| ▁زبان \| اں \| ▁دے \| ▁اد \| ب \| ▁و \| ▁ث \| ق \| افت \| ▁دی \| ▁بی \| ▁نم \| ائ \| ندگی \| ▁ک \| یتی \| ▁گ \| ئی \| ▁ا \| َ \| ئی \| ۔ |
| BLOOM | 27 | اسی \| ▁طر \| اں \| ▁خیبر \| ▁پختونخوا \| ▁اچ \| ▁بولی \| ▁جان \| ڑ \| یں \| ▁ولی \| ▁دیگر \| ▁زبان \| اں \| ▁دے \| ▁ادب \| ▁و \| ▁ثقافت \| ▁دی \| ▁بی \| ▁نمائندگی \| ▁کی \| تی \| ▁گئی \| ▁اَ \| ئی \| ۔ |
| RoBERTa-Urdu | 28 | اسی \| ▁طر \| اں \| ▁خیبر \| ▁پختونخوا \| ▁اچ \| ▁بولی \| ▁جان \| ڑیں \| ▁ولی \| ▁دیگر \| ▁زبان \| اں \| ▁دے \| ▁ادب \| ▁و \| ▁ثقافت \| ▁دی \| ▁بی \| ▁نمائندگی \| ▁کی \| تی \| ▁گئی \| ▁ا \| ▯ \| ▯ \| ئی \| ۔ |
| mT5 | 42 | ▁ \| اسی \| ▁طر \| اں \| ▁خی \| بر \| ▁پ \| خت \| و \| نخوا \| ▁ \| اچ \| ▁ب \| ولی \| ▁جان \| ڑ \| یں \| ▁ \| ولی \| ▁دیگر \| ▁ \| زبان \| اں \| ▁ \| دے \| ▁ \| ادب \| ▁و \| ▁ \| ثقاف \| ت \| ▁دی \| ▁بی \| ▁نما \| ئ \| ندگی \| ▁کی \| تی \| ▁گئی \| ▁ا \| َ \| ئی۔ |


## 7. What these tests support

**Supported (intrinsic, report-only):**

- "On Hindko text that is not in its training corpus (Southern Hindko web articles), the released tokenizer uses 23 % fewer tokens than the most efficient of the 9 external tokenizers measured (RoBERTa-Urdu) and 35 % fewer than GPT-4o's o200k, and it round-trips the text exactly."
- "Under the four PLAN §4.2 perturbations and the two alternative spellings of the retroflex nasal, its token count on the test split changes by -0.5 % to +7.6 %, and after each of them it still needs fewer tokens than every external tokenizer measured." The words a probe touches (at most 5.5 % of all words) are often re-segmented, as with every tokenizer, so 'stable segmentation' is not claimed.

**Not supported / must be stated as limitations:**

- Robustness to Arabic-keyboard typing: +92 % tokens as typed, +61 % after `normalize()`, +26 % after also folding ي→ی and ك→ک.
- Efficiency outside Hindko and its close neighbours: Pashto, English prose, Python code are clearly worse than with general tokenizers.
- Relative robustness to a space before ۔/، (+7.6 % vs at most +1.8 % for the external tokenizers).
- Anything about model quality. These are tokenizer measurements; downstream quality needs a model trained or adapted with this tokenizer (PLAN §8).
- 'Best tokenizer for all languages' or 'better than all AI models'. The comparison is 11 tokenizers on Hindko-centred text.

## 8. Files and reproduction

| file | content |
|---|---|
| `rcommon.py` | tokenizer list and loaders, n-gram keys |
| `corpus_ngrams.py / corpus_chargrams.py` | corpus word-8-gram and character-24-gram indexes (sets/corpus_*.npy) |
| `build_sets.py` | every evaluation set + sets/sets_manifest.json (screening of each survey document) |
| `perturb_extra.py` | the three new probes (Arabic keyboard, نڑ→ݨ, نڑ→ن) |
| `run_eval.py` | metrics per tokenizer → results/<key>.json (3 shards, 1 thread each) |
| `ak_mitigation.py` | Arabic-keyboard probe followed by the YEH/KAF fold → results/ak_mitigation.json |
| `report.py` | this file, robustness.json, examples_table.md |

Reproduce from `F:\Hindko\_tokenizer\sota\robustness` with `PYTHONIOENCODING=utf-8`: `python corpus_ngrams.py`, `python corpus_chargrams.py`, `python build_sets.py`, `python run_eval.py --shard K --of 3` for K = 0, 1, 2, `python ak_mitigation.py`, `python report.py`. The set files contain third-party text and are for local use only.
