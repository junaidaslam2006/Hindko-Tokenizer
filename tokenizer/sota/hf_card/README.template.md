---
language:
- hno
- hnd
license: cc-by-4.0
library_name: tokenizers
tags:
- tokenizer
- sentencepiece
- unigram
- hindko
- low-resource
- urdu-script
- perso-arabic
---

# Hindko Tokenizer

> **License: [CC BY 4.0](LICENSE).** Free to use, share and adapt, including commercially, with attribution. See [License](#license).

A 32,768-id SentencePiece Unigram tokenizer for **Hindko** (Northern `hno` and Southern `hnd`), the Indo-Aryan language of Peshawar, Hazara and neighbouring districts of Pakistan, written in Urdu (Perso-Arabic) script. Its algorithm and settings were **selected by language-model experiments**: small GPT-style models were trained from scratch with each candidate tokenizer, and held-out bits per byte decided under a pre-registered protocol with a sealed test split; this tokenizer is in the statistically tied top set and was chosen within it by a documented amendment (the pre-registered pick was MinGram 48k). On held-out Hindko, GPT-4o's tokenizer needs **1.68x as many tokens** for the same text and Qwen 3.5's **2.02x**, and this tokenizer round-trips the text exactly. It is a tokenizer, not a model: it turns text into ids for a model you train or adapt with it.

## At a glance

| | |
|---|---|
| **Type** | SentencePiece Unigram with byte fallback; canonical encoder `tokenizer.json` (Hugging Face `tokenizers`) |
| **Vocabulary** | 32,768 ids: 32,441 learned pieces, 256 byte pieces, 64 reserved special tokens, newline, `<unk>`, 5 placeholders |
| **Languages** | Hindko (`hno`, `hnd`). Also efficient on Urdu, Punjabi (Shahmukhi) and Saraiki; not suited to Pashto, English or code |
| **Training text** | 11,066 documents, 26.3 MB of strict-quality Hindko (newspaper, books, web); dev and test splits held out |
| **Held-out bytes per token** | **7.277** on the strict test split (491 documents, 1.45 MB), i.e. about 4.07 characters per token |
| **Fertility (tokens per word, harness definition)** | **1.135** on the test split (plain tokens / whitespace words: 1.161) |
| **Lossless** | `decode(encode(x)) == x` on 491/491 test documents and on every document of the train and dev splits |
| **External comparison** | Fewest tokens of the 131 distinct lossless tokenizers compared on the test split (550 external tokenizer files, deduplicated, plus this one) |
| **Nearest external tokenizer** | ProximaAI/urnova-95m at 6.061 bytes/token: this tokenizer uses 16.7 % fewer tokens |
| **Frontier tokenizers** | 36.6 % fewer tokens than Gemma 3/4, 40.4 % fewer than GPT-4o, 48.8 % fewer than Llama 4, 50.6 % fewer than Qwen 3.5, 53.7 % fewer than DeepSeek-V3 |
| **LM evidence (test, pre-registered scale)** | −1.01 % bits per byte vs the standard BPE recipe, 95 % CI [−1.33, −0.72] |
| **Known weak spot** | **+92 % tokens** on text typed with an Arabic keyboard layout (Arabic letters in place of the Urdu yeh, kaf and heh: `ي` `ك` `ه` for `ی` `ک` `ہ`); normalize first (see [Quickstart](#normalize-your-input-first)) |
| **Version** | 1.0.0, 2026-09-27 |

## Benchmarks

Every tokenizer below encodes the same held-out Hindko text (491 documents, 1.45 MB, never used in training) with its own encoder. The numbers come from the files in [`eval/`](eval); the images are in [`benchmarks/`](benchmarks) (3200 x 1800 PNG, free to reuse under the licence).

![Scorecard: Hindko Tokenizer vs frontier tokenizers](benchmarks/00_scorecard.png)

![Tokens needed for the same Hindko text](benchmarks/01_tokens_needed.png)

![One Hindko sentence, three tokenizers](benchmarks/02_example_sentence.png)

![Share of Hindko words kept as a single token](benchmarks/03_words_kept_whole.png)

![Hindko words that fit in a 4,096-token context](benchmarks/04_context_window.png)

![Rank among 131 lossless tokenizers](benchmarks/05_rank_of_131.png)

![Out-of-corpus Hindko text](benchmarks/06_unseen_hindko.png)

![Language-model selection](benchmarks/07_language_model_selection.png)

## Why a dedicated Hindko tokenizer

General-purpose tokenizers are trained mostly on English, Chinese and code. They cut Hindko words into short fragments, and byte-level ones sometimes split a single Urdu letter into two tokens. Three natural sentences from the held-out test split:

| Tokenizer | Sentence 1 (14 words) | Sentence 2 (20 words) | Sentence 3 (20 words) | Total | Tokens vs this |
|---|---:|---:|---:|---:|---:|
| **This tokenizer** | **14** | **20** | **20** | **54** | **1.00x** |
| Gemma 3 / Gemma 4 | 27 | 36 | 35 | 98 | 1.81x |
| GPT-4o (o200k) | 29 | 39 | 38 | 106 | 1.96x |
| Llama 4 | 35 | 42 | 42 | 119 | 2.20x |
| Qwen 3.5 | 36 | 47 | 45 | 128 | 2.37x |
| DeepSeek-V3 | 41 | 47 | 43 | 131 | 2.43x |
| Llama 3 | 58 | 66 | 53 | 177 | 3.28x |

Sentence 1: ایہہ بہت کہٹ لوک جانڑدین کہ ڈرامے دی پیدائش کسراں تے کتھے ہوئی آئی۔
*(approximately: "Very few people know how and where drama was born.")*

{{SEG1}}

<details>
<summary>Sentence 2 and sentence 3 (token splits)</summary>

Sentence 2: اج اساں اگر آپڑیں بزرگاں دی انہاں گلاں تے غور نہ کیتا تے پچھتاوے دے سوا کج ہتھ نہ آسی۔
*(approximately: "If we do not reflect on these words of our elders today, we will be left with nothing but regret.")*

{{SEG2}}

Sentence 3: اسی طراں خیبر پختونخوا اچ بولی جانڑیں ولی دیگر زباناں دے ادب و ثقافت دی بی نمائندگی کیتی گئی اَئی۔
*(approximately: "In the same way, the literature and culture of the other languages spoken in Khyber Pakhtunkhwa were also represented.")*

{{SEG3}}

</details>

`▁` marks a word start; `▯` is one byte of a letter that a byte-level tokenizer split in two. The text is right-to-left. Splits for 11 tokenizers are in [`eval/examples_segmentations.json`](eval/examples_segmentations.json).

**What fewer tokens means in practice.** These gains apply to a model that is trained or adapted with this tokenizer. The tokenizer cannot be plugged into an existing model such as GPT-4o or Llama.

| | Effect of needing fewer tokens for the same Hindko text |
|---|---|
| **Cost** | Training compute and per-token API prices scale with token count. The same Hindko text is 1.68x as many tokens with GPT-4o's tokenizer and 2.02x with Qwen 3.5's. |
| **Context length** | A fixed window holds more text: 4,096 tokens cover about 29.8 KB of Hindko here, against about 17.8 KB with GPT-4o's tokenizer. |
| **Speed** | Generation emits one token per step, so a sentence takes fewer steps (14 instead of 29 for sentence 1 vs GPT-4o's tokenizer). The 32,768-row embedding and output layers are also smaller than the 128k–262k rows of frontier vocabularies. |
| **Consistency** | 89.1 % of test words are a single token, so a model sees most words as whole units. |

## Quickstart

```bash
pip install tokenizers transformers huggingface_hub   # sentencepiece is optional
```

The repository is private: pass a Hugging Face read token (`token=...`, or set `HF_TOKEN`).

### transformers

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("junaid008/hindko-tokenizer", token="hf_...")

text = "ایہہ بہت کہٹ لوک جانڑدین کہ ڈرامے دی پیدائش کسراں تے کتھے ہوئی آئی۔"
ids = tok(text)["input_ids"]                  # 14 ids; nothing is added (no BOS/EOS)
assert tok.decode(ids, skip_special_tokens=False) == text
```

- **Nothing is added automatically.** `add_bos_token` and `add_eos_token` are `false`. The study's language models saw each document as `<|bos|>` (id 1) + text + `<|endoftext|>` (id 0). Add them yourself, or set `tok.add_bos_token = True`.
- **Decode whole sequences.** Each decode call strips the leading `▁`, so `decode([a]) + decode([b])` can lose a space that `decode([a, b])` keeps. For streaming, decode everything so far and emit the new suffix (as `TextStreamer` does).

### tokenizers (the canonical encoder)

```python
from tokenizers import Tokenizer

tk = Tokenizer.from_pretrained("junaid008/hindko-tokenizer", token="hf_...")
text = "ایہہ بہت کہٹ لوک جانڑدین کہ ڈرامے دی پیدائش کسراں تے کتھے ہوئی آئی۔"
enc = tk.encode(text)
enc.ids, enc.tokens
assert tk.decode(enc.ids, skip_special_tokens=False) == text
```

### Chat template

`tokenizer_config.json` ships a ChatML template:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("junaid008/hindko-tokenizer", token="hf_...")
msgs = [{"role": "user", "content": "سلام! تساں کیہہ حال اے؟"}]
tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
# '<|bos|><|im_start|>user\nسلام! تساں کیہہ حال اے؟<|im_end|>\n<|im_start|>assistant\n'
```

Chat formatting was not part of the evaluated protocol; the study's models saw plain documents only.

### sentencepiece (convenience copy, with caveats)

`sp.model` gives the same ids as `tokenizer.json` only if you apply the newline convention: split the text at `\n`, encode each line, and join with the newline piece (id 64).

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download

sp = spm.SentencePieceProcessor(model_file=hf_hub_download("junaid008/hindko-tokenizer", "sp.model", token="hf_..."))
NL = sp.piece_to_id("\n")                                   # 64

def sp_encode(text):
    out = []
    for k, ids in enumerate(sp.encode(text.split("\n"))):
        if k:
            out.append(NL)
        out.extend(ids)
    return out
```

- Never call `sp.encode(text)` on multi-line text: each `\n` becomes byte pieces and the next line loses its word-start marker.
- On single lines longer than about 3,400 characters, SentencePiece's float32 arithmetic can resolve an exact tie differently. Use `tokenizer.json` when ids must be canonical.
- Plain SentencePiece does not treat literal `<|...|>` strings as special tokens; `tokenizer.json` does.

### Normalize your input first

The tokenizer itself normalizes nothing, so it is lossless. It was trained and evaluated on text in a canonical form. [`examples/quickstart.py`](examples/quickstart.py) contains a self-contained `normalize()` that produces that form: NFC, CRLF to LF, removal of invisible and control characters (ZWSP, bidi marks, BOM, kashida), collapsed spaces, Arabic presentation forms to letters, and Arabic-keyboard YEH/KAF folded **only inside words that are provably Urdu/Hindko spelling**. It never merges letters that differ in Hindko.

```python
import os, sys
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

qs = hf_hub_download("junaid008/hindko-tokenizer", "examples/quickstart.py", token="hf_...")
sys.path.insert(0, os.path.dirname(qs))
from quickstart import normalize, fold_arabic_yeh_kaf

tok = AutoTokenizer.from_pretrained("junaid008/hindko-tokenizer", token="hf_...")
raw_text = "ایہہ  بہت کہٹ لوک​۔\r\n"   # any input: double space, ZWSP, CRLF are cleaned
ids = tok(normalize(raw_text))["input_ids"]
```

> **Arabic-keyboard caveat.** Text typed on an Arabic keyboard layout, with ي ك ه in place of ی ک ہ, costs this tokenizer **+92 % tokens** on the test split, because its training text almost never uses those code points (only 9 of 32,768 pieces contain them). `normalize()` recovers part of this (+61 % remains). If the input is known to be Hindko or Urdu, also fold every ي→ی and ك→ک with `fold_arabic_yeh_kaf()`: +26 % remains, and the tokenizer again needs fewer tokens than every external tokenizer tested on the same text. ه cannot be folded blindly, because such typists use it for both ہ and ھ.

`normalize()` is not reversible. Do not apply it where whitespace must be kept exactly (code, tables, verbatim quotes). It matches the project's reference implementation (`hp.normalize` 1.0.1) on 2,194 dev documents, 86,516 stress strings and 50,000 fuzz strings.

## Intended uses

- Tokenizer for **new language models trained on Hindko**, or on Hindko with some Urdu, Punjabi (Shahmukhi) or Saraiki.
- Text processing where token count matters: corpus statistics, length budgeting, retrieval chunking, classification or embedding models for Hindko.
- A baseline for research on tokenization of low-resource, Perso-Arabic-script languages.

## Out-of-scope uses

- **Replacing the tokenizer of an existing pretrained model.** Ids are incompatible; an existing model would need retraining or vocabulary adaptation. Extended tokenizers for continued pretraining of Qwen, Llama and Gemma are coming separately.
- **Multilingual, English or coding models** as the only tokenizer: on English prose it needs 2.61x the tokens of Llama 4's tokenizer, and on Python code 3.05x those of Llama 3's.
- **Pashto**: 1.51x the tokens of GPT-4o's tokenizer.
- Any claim about model quality, translation, or understanding. A tokenizer produces ids; it does not understand text.

## Performance

All numbers are on the **strict test split**: 491 documents, 1,451,026 bytes, never used for training or selection of the tokenizer. Every tokenizer runs with its own native encoder and no special tokens added.

### (a) Held-out compression vs other tokenizers

We measured 550 external tokenizer files: 65 study baselines of frontier and regional models, plus 485 found by a search of the Hugging Face Hub (14,856 repositories found, 2,608 represented by a measured tokenizer). Identical encoders were merged, leaving 528 distinct encodings, 130 of them lossless. **This tokenizer ranks first of the 131 lossless ones on both bytes per token and fertility.** Tokenizers that cannot reproduce the text (lossy) are not ranked: several emit `<unk>` for most words, which inflates their raw numbers.

| # | Tokenizer | Provider | Vocab | Bytes/token | Fertility | This uses fewer tokens |
|---:|---|---|---:|---:|---:|---:|
| 1 | **This tokenizer** | junaid008 | 32,768 | **7.277** | **1.135** | – |
| 2 | ProximaAI/urnova-95m | ProximaAI | 50,048 | 6.061 | 1.373 | 16.7 % |
| 3 | DunbaaBERT/DunbaaBERT_52k_base | DunbaaBERT | 52,010 | 5.964 | 1.395 | 18.0 % |
| 4 | orature/ALIF-Base-100M | orature | 32,000 | 5.912 | 1.405 | 18.8 % |
| 5 | DunbaaBERT/DunbaaBERT_32k_base | DunbaaBERT | 32,010 | 5.718 | 1.456 | 21.4 % |
| 6 | urduhack/roberta-urdu-small | UrduHack | 52,000 | 5.709 | 1.459 | 21.6 % |
| 7 | ibm-research/ia-multilingual-original-script-roberta | IBM Research | 110,000 | 5.668 | 1.469 | 22.1 % |
| 8 | mahwizzzz/urdu-dummy | community | 25,000 | 5.586 | 1.491 | 23.2 % |
| 9 | hadidev/robertaurdu | community | 59,458 | 5.523 | 1.501 | 24.1 % |
| 10 | BLOOM | BigScience | 250,680 | 5.122 | 1.633 | 29.6 % |
| 11 | MADLAD-400 | Google (via santhosh/madlad400-3b-ct2) | 256,000 | 5.080 | 1.599 | 30.2 % |
| 12 | K2-Horizon-MoVA | mlx-community | 250,624 | 5.064 | 1.651 | 30.4 % |

Frontier model tokenizers on the same text:

| Tokenizer (identical encoders) | Provider | Vocab | Bytes/token | Fertility | This uses fewer tokens |
|---|---|---:|---:|---:|---:|
| Gemma 3 (= Gemma 4, Sarvam-30B) | Google | 262,145 | 4.610 | 1.807 | 36.6 % |
| GPT-4o o200k (= gpt-oss, Phi-4-mini) | OpenAI | 200,000 | 4.338 | 1.931 | 40.4 % |
| Mistral Tekken (= Mistral NeMo, Mistral Small 4; also Apertus, Nemotron 3) | Mistral AI | 131,072 | 3.901 | 2.149 | 46.4 % |
| Llama 4 | Meta | 201,135 | 3.727 | 2.251 | 48.8 % |
| Qwen 3.5 | Alibaba | 248,070 | 3.598 | 2.323 | 50.6 % |
| Kimi K2 | Moonshot AI | 163,840 | 3.530 | 2.375 | 51.5 % |
| DeepSeek-V3 (= R1, V4, V4.1) | DeepSeek | 128,815 | 3.372 | 2.476 | 53.7 % |
| Llama 3 (= Alif-1.0, Qalb-1.0) | Meta | 128,256 | 2.641 | 3.119 | 63.7 % |

Full table of all 528 encodings: [`eval/competitors_test_strict.csv`](eval/competitors_test_strict.csv); summary and search coverage: [`eval/competitors_summary.json`](eval/competitors_summary.json).

**Scope of this comparison.**
- It covers tokenizers that were public and measured by 2026-09-27. Closed tokenizers (Gemini, Claude 3 and later, OpenAI encodings newer than o200k) are not covered, and the Hub search was stopped before its third pass was screened.
- Some of this study's own candidates compress more but were not better in the language-model evaluation: MinGram 48k gives 7.373 bytes/token on test, and SuperBPE builds reach 8.40 bytes/token on dev. Compression is a screening metric; the selection used bits per byte (b).
- A 32k vocabulary trained only on Hindko is expected to compress Hindko better than a 128k–262k multilingual vocabulary that serves hundreds of languages. This table shows the token cost of Hindko for those models; it does not rank them as models.

### (b) Language-model evidence

Each candidate tokenizer was used to train small GPT-style language models from scratch on the same 45 MB of Hindko, with the same bytes of context and the same bytes per step. The score is held-out **bits per byte** (bpb), which is comparable across tokenizers. Confidence intervals come from a hierarchical cluster bootstrap over newspaper editions, books and web sites (10,000 replicates). The baseline is the standard recipe, byte-level BPE with a 16k vocabulary.

![LM bits per byte on dev and test](eval/chart_lm_bpb_dev_test.png)

**Decision scale (pre-registered; 1.77M non-embedding parameters, 5 seeds).**

| Tokenizer | Dev bpb | Dev Δ vs baseline | Test bpb | Test Δ vs baseline |
|---|---:|---|---:|---|
| **This tokenizer (SP-Unigram 32k)** | **1.22584** (rank 1 of 17) | −1.62 % [−2.11, −1.24] | 1.20911 | **−1.01 % [−1.33, −0.72]** |
| MinGram 48k (the pre-registered pick) | 1.22636 | −1.58 % [−1.87, −1.36] | 1.20543 | −1.32 % [−1.47, −1.16] |
| MinGram 32k | 1.22844 | −1.41 % [−1.66, −1.19] | 1.20620 | −1.25 % [−1.48, −1.04] |
| BPE 48k | 1.23374 | −0.99 % [−1.21, −0.78] | 1.21111 | −0.85 % [−1.05, −0.65] |
| BPE 32k | 1.23597 | −0.81 % [−1.00, −0.66] | 1.21459 | −0.57 % [−0.75, −0.38] |
| Baseline: BPE 16k | 1.24607 | – | 1.22150 | – |

- **A statistically tied top set.** On dev, this tokenizer and the two MinGram tokenizers were not distinguishable (gaps of +0.04 % and +0.21 %, below the achievable resolution of about 0.6–0.7 %). All other candidates were significantly worse.
- **On the sealed test split this tokenizer has the highest bpb of the three**, though not significantly: +0.31 % [−0.02, +0.63] vs MinGram 48k and +0.24 % [−0.13, +0.60] vs MinGram 32k. Its gain over the baseline shrank from −1.62 % on dev to −1.01 % on test, the largest shrink of any contrast.
- **The pre-registered claim holds on test:** MinGram 48k, chosen by the pre-registered rule, beats the baseline by −1.32 % [−1.47, −1.16].

**Large arbiter (report-only; 10.6M non-embedding parameters, 2 seeds).** Amendment 1, written before any test LM number existed, chose the released tokenizer from the tied top set by the lowest dev bpb at this larger scale.

| Tokenizer | Dev bpb | Dev Δ vs baseline | Test bpb | Test Δ vs baseline |
|---|---:|---|---:|---|
| **This tokenizer** | **1.13373** | −1.34 % [−1.78, −1.03] | **1.11898** | −0.77 % [−1.15, −0.40] |
| MinGram 32k | 1.14086 | −0.72 % [−0.91, −0.52] | 1.12240 (1 seed) | −0.47 % (no CI) |
| MinGram 48k (the pre-registered pick) | 1.14407 | −0.44 % [−0.62, −0.29] | 1.12477 | −0.26 % [−0.44, −0.08] |
| BPE 32k | 1.14940 | +0.03 % [−0.30, +0.32] | not run | – |
| Baseline: BPE 16k | 1.14908 | – | 1.12767 | – |

- At this scale the top set separates on dev: this tokenizer vs MinGram 32k −0.62 % [−1.04, −0.31], vs MinGram 48k −0.90 % [−1.35, −0.56].
- On test the order is the same. Vs MinGram 48k: −0.51 % [−0.86, −0.22]. Vs MinGram 32k: not established (only 1 seed, because the free Colab GPU quota ran out).

**What this supports.** The algorithm and settings were selected by language-model experiments, and the released tokenizer is in the statistically tied top set at the pre-registered scale. It is **not** claimed to be the best tokenizer on test: at the pre-registered scale it is third of the three tied members, and the larger-scale result that favours it is report-only, with 1–2 seeds. Full tables: [`eval/dev_lm_confirm_ranking.csv`](eval/dev_lm_confirm_ranking.csv), [`eval/test_lm_confirm.csv`](eval/test_lm_confirm.csv), [`eval/dev_lm_large.csv`](eval/dev_lm_large.csv), [`eval/test_lm_large.csv`](eval/test_lm_large.csv), [`eval/lm_key_contrasts.json`](eval/lm_key_contrasts.json).

### (c) Robustness and out-of-distribution text

Ten comparison tokenizers were re-run on text outside the training corpus and under realistic spelling and typing changes (input normalized as recommended above).

| Test | This tokenizer | Best external | Result |
|---|---:|---:|---|
| Southern Hindko web articles (not in the corpus; 6.7 % of words unseen in training) | **6.18** bytes/token | 4.74 (RoBERTa-Urdu) | 23 % fewer tokens; 35 % fewer than GPT-4o |
| Hindko stories, PDF text layer | **4.41** | 4.20 (RoBERTa-Urdu) | 4.6 % fewer tokens |
| hindko.org PDFs, OCR (noisy letters) | **4.97** | 4.91 (BLOOM) | 1.2 % fewer tokens |
| Northern Hindko interface messages (translatewiki) | 3.62 | **3.73** (mT5, which does not round-trip) | **2.9 % more tokens** |
| Code-mixed Hindko + Urdu + English (100 synthetic lines) | **5.66** | 5.55 (BLOOM) | ahead on whole lines; the inserted English/Urdu costs 2.77 bytes/token vs 5.93 for GPT-4o |
| Lossless on all robustness sets | 1,692/1,692 docs | – | – |

| Spelling or typing change (whole test split) | Token change, this | Range, 9 external |
|---|---:|---:|
| Harakat removed | −0.45 % | −8.61 % to −1.69 % |
| Digit script swapped (ASCII / Urdu) | −0.45 % | −0.17 % to +0.39 % |
| ZWNJ inserted in compounds | +0.49 % | +0.01 % to +0.16 % |
| Retroflex nasal نڑ written ݨ / ن | +2.76 % / +0.40 % | −0.36 % to +1.60 % / −1.56 % to −1.09 % |
| Space before ۔ and ، | **+7.58 %** | 0.00 % to +1.78 % |
| **Arabic-keyboard ي ك ه for ی ک ہ** | **+92.15 %** | −9.25 % to +101.60 % |
| … then `normalize()` | +60.72 % | −9.55 % to +68.29 % |
| … then also fold ي→ی, ك→ک | +25.8 % | −10.5 % to +24.3 % |

After each of the six changes above the Arabic-keyboard row, this tokenizer still needs fewer tokens than every external tokenizer. On Arabic-keyboard text as typed, it needs more tokens than GPT-4o, Gemma 3, BLOOM and mT5; after `normalize()` and the YEH/KAF fold it again needs fewer than all of them.

**Other languages** (bytes/token; higher is better):

| | Urdu | Punjabi (Shahmukhi) | Saraiki | Pashto | English prose | Python code |
|---|---:|---:|---:|---:|---:|---:|
| This tokenizer | 6.59 | 6.53 | 4.95 | 3.21 | 1.82 | 1.37 |
| Best external | 6.96 (RoBERTa-Urdu) | 5.83 (RoBERTa-Urdu) | 4.42 (RoBERTa-Urdu) | 4.86 (GPT-4o) | 4.76 (Llama 4) | 4.16 (Llama 3) |
| Rank among 11 | 3 | 2 | 2 | 10 | 11 | 11 |

Details: [`eval/robustness_summary.json`](eval/robustness_summary.json).

### (d) Frontier tokenizers inside the same language model: planned

A run that trains the same small language model with this tokenizer and with the GPT-4o, Gemma 3, Llama 3 and other frontier tokenizers, and compares held-out bits per byte on the test split, is prepared but **not yet run** (the free GPU quota is exhausted). Until then, the comparison with frontier tokenizers rests on token counts only (a), and no statement is made about which tokenizer would give a better model.

## How it was built

**Data.** The strict-quality tier of the train split of a Hindko text corpus assembled for this project: issues of the *Weekly Hindkowan* (Peshawar), the *Gandhara Hindko Academy* book series (much of it poetry, dictionaries and literary prose; 61.8 % of the characters) and a small amount of web text.

| Training text (D2) | Documents | Characters | UTF-8 bytes |
|---|---:|---:|---:|
| Books | 6,037 | 9,110,441 | 16,279,682 |
| Newspaper | 4,748 | 5,279,372 | 9,444,696 |
| Web | 281 | 342,102 | 606,085 |
| **Total** | **11,066** | **14,731,915** | **26,330,463** |

The split is 90/5/5 by words per source and assigns whole newspaper editions, books and web sites to one side, with one exception: 230 leak-prone records from 40 evaluation groups were moved to train, so those groups are not 100 % held out. Near-duplicate leakage into dev and test was checked (0 leaking records). Contact details in the corpus were masked as `<PHONE>`, `<EMAIL>` and `<ID_NUMBER>` before training.

**Normalization.** All text was put in the canonical form of `normalize()` (see [Quickstart](#normalize-your-input-first)); the tokenizer itself applies no normalization.

**Algorithm.** SentencePiece 0.2.1 Unigram, `vocab_size` 32,768, `character_coverage` 1.0, byte fallback, splits at whitespace, script changes and numbers, maximum piece length 16, identity normalization, every training line used. Pieces never contain a space. Five pieces that the encoder could never produce were replaced in place by never-used placeholders (a pre-declared remedy); no other id moved. Retraining gives a byte-identical model. The Hugging Face `tokenizer.json` was exported and verified to give the SentencePiece ids on 100 % of the dev, test and train documents.

**Selection, in plain terms.**
1. **A plan was written and frozen first.** The decision metric (held-out bits per byte of small LMs trained from scratch), the statistics and the tie-breakers were fixed before any LM result existed.
2. **Broad screen.** 75 tokenizer builds (BPE, SentencePiece BPE and Unigram, SuperBPE, PickyBPE, MinGram; 4 pre-tokenizers, 3 data mixes, 8k–48k vocabularies) were measured on the dev split for compression and hard health gates.
3. **Language-model arbiter.** 18 finalists (17 ranked): 12 in the first round (10 pre-registered plus 2 reference arms added before the first LM run) and 6 built in a second, exploratory round; each trained 5 small LMs on a GPU. Hierarchical cluster bootstrap, Holm correction, ±0.3 % equivalence margin.
4. **Decision.** Three tokenizers formed a statistically tied top set. The pre-registered tie-breaker picked MinGram 48k. An audit found that this tie-breaker favours the largest vocabulary, which also has the most parameters.
5. **Amendment 1**, written and hashed before any test LM number existed, chose within the tied set by the lowest dev bpb of a larger, report-only arbiter. It selected this tokenizer. The pre-registered pick and its claim stay on record.
6. **Sealed test split, used once**, for all finalists together. Results are reported as they came out, including where this tokenizer is behind (Performance b).

## Technical specifications

| Ids | Content | Notes |
|---|---|---|
| 0 | `<\|endoftext\|>` | end of document; `eos_token` |
| 1 | `<\|bos\|>` | start of document; `bos_token` |
| 2 | `<\|pad\|>` | `pad_token` |
| 3, 4 | `<\|im_start\|>`, `<\|im_end\|>` | ChatML turn markers |
| 5–63 | `<\|reserved_0\|>` … `<\|reserved_58\|>` | free for future use |
| 64 | `\n` | newline piece (added token, not special) |
| 65 | `<unk>` | never produced (byte fallback covers every character); `unk_token` is unset |
| 66–321 | `<0x00>` … `<0xFF>` | UTF-8 byte fallback |
| 322–32,767 | 32,441 learned pieces | `▁` marks a word-initial piece |
| 7183, 16359, 16629, 16924, 19063 | `<\|unused_0\|>` … `<\|unused_4\|>` | never produced by text |

- **Vocabulary content.** 90.9 % of learned pieces are Arabic-script words or word parts (90.3 % of test tokens). 530 are Latin-letter pieces. ASCII digits are single-digit pieces; 345 pieces are runs of 2–4 Urdu digits (years such as `▁۱۹۸۵`), so `۲۰۲۴` is one token while `2024` at a word start is five (a standalone `▁` plus four digits).
- **Word + full stop.** `۔` is fused with the preceding word (`▁آئی۔`), because SentencePiece treats it as Arabic script; `،` and `؟` are split off.
- **Embedding matrix.** 32,768 rows (a multiple of 64). 40.6 % of learned pieces occur fewer than 20 times in the 7.3M-token LM training stream, so expect under-trained embeddings for rare pieces.
- **Files.**

| File | Purpose | sha256 |
|---|---|---|
| `tokenizer.json` | canonical encoder | `{{SHA_tokenizer.json}}` |
| `tokenizer_config.json` | transformers config, chat template, no automatic BOS/EOS | `{{SHA_tokenizer_config.json}}` |
| `special_tokens_map.json` | special-token roles | `{{SHA_special_tokens_map.json}}` |
| `sp.model` | SentencePiece convenience copy (see caveats) | `{{SHA_sp.model}}` |
| `sp.vocab` | piece list with scores | `{{SHA_sp.vocab}}` |
| `gauge.json` | per-character score offsets; rounded log-probability of a piece = score − Σ w(c) | `{{SHA_gauge.json}}` |

## Limitations and risks

- **Arabic-keyboard input** is the largest weakness: +92 % tokens as typed, +61 % after `normalize()`, +26 % after the YEH/KAF fold.
- **Formatting sensitivity.** A space before `۔`/`،` costs +7.6 % tokens (external tokenizers: at most +1.8 %), because many pieces end in the full stop. Digit-script swaps and ZWNJ insertion re-segment most of the words they touch.
- **Small proxy models.** The LM evidence comes from models of 1.8M–10.6M non-embedding parameters trained on 45 MB. It is not evidence about 1B+ models or continued pretraining.
- **Selection after seeing dev.** The top set came from an exploratory second round, and the release choice came from Amendment 1, written after the larger dev results were known. The dev gains are optimistic; the test gain is smaller.
- **Tied, not ahead, at the pre-registered scale;** on test newspaper text it is significantly behind both MinGram builds (+0.59 % and +0.51 %).
- **Dialect and domain coverage.** Books are 62 % of the training characters; newspaper text is mostly Peshawari; web text is 0.6 MB. Hazara and other varieties are unevenly represented. No native-speaker evaluation was done.
- **Other languages.** Poor on Pashto, English and code (Out-of-scope uses). The Saraiki implosives and most Pashto-specific letters are absent from training and fall back to bytes.
- **Literal special-token strings in input** (`<|im_start|>` and so on) are matched as special tokens by `tokenizer.json` and transformers; `split_special_tokens=True` does not prevent this with this tokenizer. Strip or escape `<|...|>` in untrusted text, or insert special tokens only as ids.
- **Lossless as verified,** not on all possible text: the character `▁` (U+2581) in input decodes as a space.
- **Corpus artifacts.** 478 learned pieces (1.47 %) are artifacts of the source text, mostly two words fused across a full stop without a space. They decode exactly and cover 0.29 % of test tokens.
- **No downstream benchmark** exists for Hindko, and no parallel text exists for cross-language parity comparisons.

## Ethical considerations and privacy

- **No documents are included.** The files hold 32,441 pieces of at most 16 characters and their scores.
- **Personal data.** Phone numbers, e-mail addresses and ID numbers were masked in the corpus. A vocabulary audit found that no piece can hold a contact identifier: the longest digit run in any piece is 4, pieces never contain spaces, and the only piece with `@` is `@`. The masks have no dedicated piece. One probable unmasked phone number in the training text is encoded with generic digit pieces and is not in the vocabulary. Names in the vocabulary are of public, historical or literary figures, fictional characters, places and common given names; the audit found no piece that singles out a private individual.
- **Sensitive words.** The vocabulary reflects published Hindko literature and news. It contains 13 pieces with an attested vulgar, insulting or slur sense (6 tokens in the whole test split), 17 sensitive group labels (caste, sect, religion, race), mostly used descriptively, and 24 ordinary words that can be used as insults. Nothing was removed: a tokenizer must be able to encode any text, and deleting a piece would not stop the word from being encoded. The list is kept for review by a native speaker and is not published here.
- **Representation.** The tokenizer encodes the spelling conventions of its sources. Text in under-represented varieties or spellings costs more tokens, which can mean higher cost and shorter effective context for those users.
- **Provenance.** The training texts belong to their publishers and authors (*Weekly Hindkowan*, the *Gandhara Hindko Academy* and web sources). They are not included in this repository, and the licence below does not cover them.

## License

**[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**; full text in [`LICENSE`](LICENSE).

- You may use, share and adapt the tokenizer, including in commercial models and products.
- Give credit: "Hindko Tokenizer" by junaid008, with a link to this repository (the citation below is enough).
- The licence covers the files in this repository: the vocabulary, scores, configuration, examples, evaluation outputs and benchmark images. It grants no rights in the texts the tokenizer was trained on, which are not included.
- The external tokenizers in the comparison were measured locally; none is redistributed here.

## Citation

```bibtex
@misc{hindko_tokenizer_2026,
  title        = {Hindko Tokenizer: a 32,768-id SentencePiece Unigram tokenizer for Hindko},
  author       = {junaid008},
  year         = {2026},
  version      = {1.0.0},
  howpublished = {\url{https://huggingface.co/junaid008/hindko-tokenizer}},
  note         = {CC BY 4.0. In the statistically tied top set by held-out bits-per-byte of language models under a pre-registered protocol; chosen within that set by a documented amendment (the pre-registered pick was MinGram 48k)}
}
```

## Contact

**junaid008** on Hugging Face. Please open a discussion on this repository for questions, errors or clearance matters.

## Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0.0 (card update) | 2026-09-27 | Licence set to CC BY 4.0; benchmark images added in `benchmarks/`. Tokenizer files unchanged. |
| 1.0.0 | 2026-09-27 | First release: SentencePiece Unigram, 32,768 ids (`R2-A4-SPnat-D2-32k`); `tokenizer.json` canonical; evaluation tables and charts; `examples/quickstart.py` with `normalize()`. |

{{SOURCES}}
