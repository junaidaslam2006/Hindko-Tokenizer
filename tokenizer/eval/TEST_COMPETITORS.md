# Final competitor table on the Hindko TEST split (one shot)

Generated 2026-09-26T21:07:58Z by `eval/report_test_competitors.py` from `eval/run_test_competitors.py` (harness code as in Stage 0). **These are TEST numbers, computed once, after the tokenizer was chosen on dev** (`analysis/decision.json`, fixed before the test split was opened). They are reported; they change no decision.

- **Text:** `data/test_strict.jsonl`, the strict test split of manifest `76582d3a…`: 491 documents, 1,451,026 UTF-8 bytes, 171,769 whitespace words, 27 bootstrap clusters (book 6, newspaper 15, web 6), sha256 `a74c33ad008f…`, in canonical form (`hp.normalize` 1.0.1). It was materialised once by `data/materialize_test.py`, exactly like dev_strict (`data/test_manifest.json`).
- **Tokenizers:** the chosen tokenizer `R2-A10-MinGram-P1r3-D2-48k` and all 66 working external tokenizers of `baselines/manifest.json`, each with its own native encoder, no BOS/EOS/CLS added. Same metric code and definitions as `eval/BASELINES_DEV.md`.
- **Chosen tokenizer on test:** 7.373 bytes/token, fertility 1.125, STRR 90.1%, G1 491/491 (lossless). Rank **1 of 66** by bytes/token, **1 of 56** among the tokenizers that round-trip every test document.
  - Best lossless external tokenizer: `roberta-urdu` (UrduHack) at 5.709 bytes/token; the chosen tokenizer uses **22.6 % fewer tokens** on the same text.
- **Ranking:** bytes/token on test_strict, descending. Bytes/token is a screening metric (PLAN §4.2); the decision metric is LM bits-per-byte, which the one-shot LM run (`colab/FINAL_TEST.md`) measures for the four test candidates only. `hindko-probe-bpe32k` is LEAKY (trained on all data, including this test text) and is never ranked.
- **Claim this supports (PLAN §8):** "uses x % fewer tokens per byte of Hindko than the tokenizers of <external models measured>" — read x off the headline table. It says nothing about downstream quality.

## Headline: chosen tokenizer vs the big model families (test_strict)

*fewer tokens* = 1 − tokens(chosen) ÷ tokens(competitor) on the same 491 documents; *× tokens* = tokens(competitor) ÷ tokens(chosen). *dev b/tok* is the Stage 0 dev_strict value, for reference.

| tokenizer | provider — family | vocab | test bytes/tok | dev b/tok | fertility | STRR | G1 | × tokens | chosen uses fewer tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`R2-A10-MinGram-P1r3-D2-48k`** (chosen) | this study | 49,152 | **7.373** | 7.224 | 1.125 | 90.1% | 491/491 | 1.000 | – |
| `gpt-4o` | OpenAI — o200k_base (tiktoken) | 200,000 | 4.338 | 4.384 | 1.931 | 41.5% | 491/491 | 1.700 | 41.2 % |
| `gpt-4` | OpenAI — cl100k_base (tiktoken) | 100,263 | 1.919 | 1.967 | 4.275 | 0.7% | 491/491 | 3.842 | 74.0 % |
| `gemma-4` | Google — Gemma 4 SentencePiece 262k | 262,144 | 4.610 | 4.640 | 1.807 | 48.7% | 491/491 | 1.599 | 37.5 % |
| `gemma-2` | Google — Gemma 2 SentencePiece 256k | 256,000 | 3.787 | 3.834 | 2.206 | 30.4% | 491/491 | 1.947 | 48.6 % |
| `llama-4` | Meta — Llama 4 BPE 202k | 201,135 | 3.727 | 3.759 | 2.251 | 27.5% | 491/491 | 1.978 | 49.5 % |
| `llama-3` | Meta — Llama 3 tiktoken-BPE 128k | 128,256 | 2.641 | 2.736 | 3.119 | 17.8% | 491/491 | 2.792 | 64.2 % |
| `llama-2` | Meta — Llama 2 SentencePiece BPE 32k | 32,000 | 1.714 | 1.722 | 3.937 | 0.6% | 491/491 | 4.302 | 76.8 % |
| `qwen-3.5` | Alibaba Qwen — Qwen3.5 byte-level BPE 248k | 248,070 | 3.598 | 3.631 | 2.323 | 25.7% | 491/491 | 2.049 | 51.2 % |
| `qwen-3` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 151,669 | 2.633 | 2.684 | 3.082 | 3.0% | 491/491 | 2.801 | 64.3 % |
| `deepseek-v4.1` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 129,280 | 3.372 | 3.418 | 2.476 | 23.2% | 491/491 | 2.187 | 54.3 % |
| `deepseek-v3` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 128,815 | 3.372 | 3.418 | 2.476 | 23.2% | 491/491 | 2.187 | 54.3 % |
| `mistral-nemo` | Mistral AI — Tekken (tiktoken-style BPE, 131k) | 131,072 | 3.901 | 3.942 | 2.149 | 30.4% | 491/491 | 1.890 | 47.1 % |
| `command-a-plus` | Cohere — Command A+ (05-2026) byte-level BPE 255k | 255,032 | 3.685 | 3.729 | 2.274 | 26.1% | 491/491 | 2.001 | 50.0 % |
| `phi-4` | Microsoft — Phi-4 (cl100k-derived) 100k | 100,352 | 1.919 | 1.967 | 4.275 | 0.7% | 491/491 | 3.842 | 74.0 % |
| `grok-1` | xAI — Grok-1 SentencePiece 131k | 131,072 | 1.618 | 1.646 | 4.822 | 0.6% | 491/491 | 4.557 | 78.1 % |
| `kimi-k2` | Moonshot AI — Kimi tiktoken BPE 160k | 163,840 | 3.530 | 3.543 | 2.375 | 27.8% | 491/491 | 2.089 | 52.1 % |
| `glm-5` | Zhipu AI / Z.ai — GLM-5 byte-level BPE 155k | 154,856 | 3.025 | 3.084 | 2.763 | 13.7% | 491/491 | 2.437 | 59.0 % |
| `falcon-h1` | TII — Falcon-H1 BPE 261k (34B) | 261,120 | 4.506 | 4.563 | 1.846 | 45.6% | 491/491 | 1.636 | 38.9 % |
| `bloom` | BigScience — BLOOM byte-level BPE 250k | 250,680 | 5.122 | 5.215 | 1.633 | 57.2% | 491/491 | 1.440 | 30.5 % |
| `roberta-urdu` | UrduHack — RoBERTa-Urdu byte-level BPE 52k (monolingual Urdu) | 52,000 | 5.709 | 5.771 | 1.459 | 72.1% | 491/491 | 1.292 | 22.6 % |
| `tiny-aya` | Cohere — Tiny Aya byte-level BPE 261k | 261,010 | 4.106 | 4.152 | 2.032 | 38.4% | 491/491 | 1.796 | 44.3 % |
| `claude-legacy` | Anthropic — Claude 1/2 BPE 65k (legacy, public) | 65,000 | 1.554 | 1.579 | 4.945 | 0.6% | 467/491 ⚠ | 4.744 | 78.9 % |

⚠ = not lossless on test (fails `decode(encode(doc)) == doc`), so its token count covers less text than the chosen tokenizer's. Tokenizers with identical encodings (e.g. `gemma-3` = `gemma-4`, `gpt-oss`/`phi-4-mini` = `gpt-4o`) are in the full table.

## Full ranking on test_strict

*#* = rank by bytes/token; *ll #* = rank among the tokenizers that round-trip all 491 documents; *dev b/tok* is the dev_strict value (Stage 0 for the externals, the candidate's dev summary for the chosen tokenizer), for reference; *× tokens* = tokens ÷ the chosen tokenizer's tokens on the same documents.

| # | ll # | tokenizer | provider / family | vocab | algorithm | bytes/tok | dev b/tok | chars/tok | × tokens | fertility | STRR | G1 | UNK | lines b/tok | note |
|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | **`R2-A10-MinGram-P1r3-D2-48k`** | this study — MinGram (A10), pre-tokenizer P1r3, trained on strict train D2; chosen on dev by the pre-registered rule | 49,152 | MinGram / Unigram (HF tokenizer.json, byte fallback) | **7.373** | 7.224 | 4.122 | 1.000 | 1.125 | 90.1% | 491/491 | 0 | 7.493 | **chosen (this study)** |
| 2 | – | `urdu-bert-64k` | community (farahadeeba) — Urdu BERT WordPiece 64k | 64,000 | WordPiece | **6.221** | 6.436 | 3.478 | 1.185 | 1.358 | 73.6% | 0/491 | 2 | 6.206 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 3 | 2 | `roberta-urdu` | UrduHack — RoBERTa-Urdu byte-level BPE 52k (monolingual Urdu) | 52,000 | byte-level BPE | **5.709** | 5.771 | 3.191 | 1.292 | 1.459 | 72.1% | 491/491 | 0 | 5.777 |  |
| 4 | – | `urdu-gpt2-20k` | community (aariciah) — Urdu GPT-2 BPE 20k (Urdu-only vocab, NFKC+lowercase) | 20,000 | BPE (character-level, no byte fallback) | **5.633** | 5.676 | 3.149 | 1.309 | 1.478 | 65.9% | 166/491 | 3,633 | 5.723 | **not lossless**; line breaks: line breaks become UNK tokens |
| 5 | – | `muril` | Google — MuRIL WordPiece 197k | 197,258 | WordPiece | **5.392** | 5.613 | 3.014 | 1.367 | 1.567 | 61.6% | 0/491 | 38 | 5.379 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 6 | – | `indicbert-v2` | AI4Bharat — IndicBERT v2 WordPiece 250k | 250,000 | WordPiece | **5.299** | 5.469 | 2.962 | 1.392 | 1.594 | 58.5% | 0/491 | 2 | 5.285 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 7 | 3 | `bloom` | BigScience — BLOOM byte-level BPE 250k | 250,680 | byte-level BPE | **5.122** | 5.215 | 2.863 | 1.440 | 1.633 | 57.2% | 491/491 | 0 | 5.155 |  |
| 8 | – | `sindhi-xlmr` | community (Kashif786) — XLM-R + Sindhi vocabulary extension (Unigram 265k) | 264,635 | Unigram (SentencePiece) | **5.069** | 5.243 | 2.834 | 1.455 | 1.599 | 58.6% | 172/491 | 14 | 5.057 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 9 | – | `xlm-r` | Meta — XLM-R SentencePiece Unigram 250k | 250,002 | Unigram (SentencePiece) | **4.792** | 4.947 | 2.678 | 1.539 | 1.680 | 56.8% | 173/491 | 6 | 4.780 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 10 | – | `nllb-200` | Meta — NLLB-200 SentencePiece BPE 256k | 256,204 | BPE (character-level, no byte fallback) | **4.617** | 4.737 | 2.581 | 1.597 | 1.824 | 46.6% | 150/491 | 1,777 | 4.606 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 11 | 4 | `gemma-4` | Google — Gemma 4 SentencePiece 262k | 262,144 | BPE (SentencePiece-style, byte fallback) | **4.610** | 4.640 | 2.577 | 1.599 | 1.807 | 48.7% | 491/491 | 0 | 4.652 | same encodings as `gemma-3` |
| 12 | 5 | `gemma-3` | Google — Gemma 3 SentencePiece 262k | 262,145 | BPE (SentencePiece-style, byte fallback) | **4.610** | 4.640 | 2.577 | 1.599 | 1.807 | 48.7% | 491/491 | 0 | 4.652 | same encodings as `gemma-4` |
| 13 | 6 | `sarvam-30b` | Sarvam AI — Sarvam 2026 tokenizer 262k (Gemma-style SentencePiece BPE) | 262,144 | BPE (SentencePiece-style, byte fallback) | **4.610** | 4.640 | 2.577 | 1.599 | 1.807 | 48.7% | 491/491 | 0 | 4.652 |  |
| 14 | 7 | `falcon-h1` | TII — Falcon-H1 BPE 261k (34B) | 261,120 | byte-level BPE | **4.506** | 4.563 | 2.519 | 1.636 | 1.846 | 45.6% | 491/491 | 0 | 4.546 |  |
| 15 | 8 | `gpt-4o` | OpenAI — o200k_base (tiktoken) | 200,000 | byte-level BPE | **4.338** | 4.384 | 2.425 | 1.700 | 1.931 | 41.5% | 491/491 | 0 | 4.358 | same encodings as `gpt-oss`, `phi-4-mini` |
| 16 | 9 | `gpt-oss` | OpenAI — o200k_harmony | 200,019 | byte-level BPE | **4.338** | 4.384 | 2.425 | 1.700 | 1.931 | 41.5% | 491/491 | 0 | 4.358 | same encodings as `gpt-4o`, `phi-4-mini` |
| 17 | 10 | `phi-4-mini` | Microsoft — Phi-4-mini (o200k-derived) 200k | 200,029 | byte-level BPE | **4.338** | 4.384 | 2.425 | 1.700 | 1.931 | 41.5% | 491/491 | 0 | 4.358 | same encodings as `gpt-4o`, `gpt-oss` |
| 18 | – | `mbert` | Google — mBERT WordPiece 119k (cased) | 119,547 | WordPiece | **4.151** | 4.233 | 2.320 | 1.776 | 2.035 | 41.9% | 0/491 | 3,744 | 4.140 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 19 | 11 | `tiny-aya` | Cohere — Tiny Aya byte-level BPE 261k | 261,010 | byte-level BPE | **4.106** | 4.152 | 2.295 | 1.796 | 2.032 | 38.4% | 491/491 | 0 | 4.138 |  |
| 20 | – | `mt5` | Google — mT5 SentencePiece Unigram 250k | 250,100 | SentencePiece Unigram + byte fallback | **3.912** | 3.960 | 2.187 | 1.885 | 1.874 | 41.3% | 175/491 | 0 | 3.903 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 21 | 12 | `mistral-nemo` | Mistral AI — Tekken (tiktoken-style BPE, 131k) | 131,072 | byte-level BPE | **3.901** | 3.942 | 2.181 | 1.890 | 2.149 | 30.4% | 491/491 | 0 | 3.916 | same encodings as `mistral-small-4`, `apertus`, `sarvam-m`, `nemotron-3` |
| 22 | 13 | `mistral-small-4` | Mistral AI — Tekken (tekken.json v15, 131k) | 131,072 | byte-level BPE | **3.901** | 3.942 | 2.181 | 1.890 | 2.149 | 30.4% | 491/491 | 0 | 3.916 | same encodings as `mistral-nemo`, `apertus`, `sarvam-m`, `nemotron-3` |
| 23 | 14 | `apertus` | Swiss AI — Apertus BPE 131k | 131,072 | byte-level BPE | **3.901** | 3.942 | 2.181 | 1.890 | 2.149 | 30.4% | 491/491 | 0 | 3.916 | same encodings as `mistral-nemo`, `mistral-small-4`, `sarvam-m`, `nemotron-3` |
| 24 | 15 | `sarvam-m` | Sarvam AI — Sarvam-M (Mistral Small 3.1 Tekken) | 131,072 | byte-level BPE | **3.901** | 3.942 | 2.181 | 1.890 | 2.149 | 30.4% | 491/491 | 0 | 3.916 | same encodings as `mistral-nemo`, `mistral-small-4`, `apertus`, `nemotron-3` |
| 25 | 16 | `nemotron-3` | NVIDIA — Nemotron 3 BPE 131k | 131,072 | byte-level BPE | **3.901** | 3.942 | 2.181 | 1.890 | 2.149 | 30.4% | 491/491 | 0 | 3.916 | same encodings as `mistral-nemo`, `mistral-small-4`, `apertus`, `sarvam-m` |
| 26 | 17 | `gemma-2` | Google — Gemma 2 SentencePiece 256k | 256,000 | BPE (SentencePiece-style, byte fallback) | **3.787** | 3.834 | 2.117 | 1.947 | 2.206 | 30.4% | 491/491 | 0 | 3.813 |  |
| 27 | 18 | `llama-4` | Meta — Llama 4 BPE 202k | 201,135 | byte-level BPE | **3.727** | 3.759 | 2.083 | 1.978 | 2.251 | 27.5% | 491/491 | 0 | 3.740 |  |
| 28 | 19 | `command-a-plus` | Cohere — Command A+ (05-2026) byte-level BPE 255k | 255,032 | byte-level BPE | **3.685** | 3.729 | 2.060 | 2.001 | 2.274 | 26.1% | 491/491 | 0 | 3.696 |  |
| 29 | 20 | `qwen-3.5` | Alibaba Qwen — Qwen3.5 byte-level BPE 248k | 248,070 | byte-level BPE | **3.598** | 3.631 | 2.011 | 2.049 | 2.323 | 25.7% | 491/491 | 0 | 3.622 | same encodings as `qwen-3.8` |
| 30 | 21 | `qwen-3.8` | Alibaba Qwen — Qwen3.8 byte-level BPE 248k | 248,077 | byte-level BPE | **3.598** | 3.631 | 2.011 | 2.049 | 2.323 | 25.7% | 491/491 | 0 | 3.622 | same encodings as `qwen-3.5` |
| 31 | 22 | `kimi-k2` | Moonshot AI — Kimi tiktoken BPE 160k | 163,840 | byte-level BPE | **3.530** | 3.543 | 1.973 | 2.089 | 2.375 | 27.8% | 491/491 | 0 | 3.542 |  |
| 32 | 23 | `ernie-4.5` | Baidu — ERNIE 4.5 SentencePiece 103k | 101,304 | BPE (SentencePiece-style, byte fallback) | **3.528** | 3.540 | 1.972 | 2.090 | 2.369 | 27.2% | 491/491 | 0 | 3.550 |  |
| 33 | 24 | `pashto-lfm2.5` | community (nassimjp) — LFM2.5 + Pashto vocabulary extension | 125,039 | byte-level BPE | **3.509** | 3.531 | 1.962 | 2.101 | 2.386 | 22.5% | 491/491 | 0 | 3.519 |  |
| 34 | 25 | `deepseek-v3` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 128,815 | byte-level BPE | **3.372** | 3.418 | 1.885 | 2.187 | 2.476 | 23.2% | 491/491 | 0 | 3.390 | same encodings as `deepseek-r1`, `deepseek-v4`, `deepseek-v4.1` |
| 35 | 26 | `deepseek-r1` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 128,815 | byte-level BPE | **3.372** | 3.418 | 1.885 | 2.187 | 2.476 | 23.2% | 491/491 | 0 | 3.390 | same encodings as `deepseek-v3`, `deepseek-v4`, `deepseek-v4.1` |
| 36 | 27 | `deepseek-v4` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 129,280 | byte-level BPE | **3.372** | 3.418 | 1.885 | 2.187 | 2.476 | 23.2% | 491/491 | 0 | 3.390 | same encodings as `deepseek-v3`, `deepseek-r1`, `deepseek-v4.1` |
| 37 | 28 | `deepseek-v4.1` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 129,280 | byte-level BPE | **3.372** | 3.418 | 1.885 | 2.187 | 2.476 | 23.2% | 491/491 | 0 | 3.390 | same encodings as `deepseek-v3`, `deepseek-r1`, `deepseek-v4` |
| 38 | 29 | `glm-5` | Zhipu AI / Z.ai — GLM-5 byte-level BPE 155k | 154,856 | byte-level BPE | **3.025** | 3.084 | 1.691 | 2.437 | 2.763 | 13.7% | 491/491 | 0 | 3.040 |  |
| 39 | 30 | `command-r` | Cohere — Command-R / Aya Expanse byte-level BPE 256k | 255,029 | byte-level BPE | **2.645** | 2.747 | 1.479 | 2.788 | 3.107 | 21.7% | 491/491 | 0 | 2.656 | same encodings as `command-r7b` |
| 40 | 31 | `command-r7b` | Cohere — Command R7B byte-level BPE 256k | 255,033 | byte-level BPE | **2.645** | 2.747 | 1.479 | 2.788 | 3.107 | 21.7% | 491/491 | 0 | 2.656 | same encodings as `command-r` |
| 41 | 32 | `llama-3` | Meta — Llama 3 tiktoken-BPE 128k | 128,256 | byte-level BPE | **2.641** | 2.736 | 1.476 | 2.792 | 3.119 | 17.8% | 491/491 | 0 | 2.651 | same encodings as `urdu-llama3-almanach`, `urdu-llama3.2-custom`, `alif-1.0`, `qalb-1.0` |
| 42 | 33 | `urdu-llama3-almanach` | Inria ALMAnaCH — Llama-3-8B-mono-Urdu (Urdu-adapted vocab) | 128,256 | byte-level BPE | **2.641** | 2.736 | 1.476 | 2.792 | 3.119 | 17.8% | 491/491 | 0 | 2.651 | same encodings as `llama-3`, `urdu-llama3.2-custom`, `alif-1.0`, `qalb-1.0` |
| 43 | 34 | `urdu-llama3.2-custom` | community (SabahNawab) — Llama-3.2-3B + custom Urdu tokens | 145,613 | byte-level BPE | **2.641** | 2.736 | 1.476 | 2.792 | 3.119 | 17.8% | 491/491 | 0 | 2.651 | same encodings as `llama-3`, `urdu-llama3-almanach`, `alif-1.0`, `qalb-1.0` |
| 44 | 35 | `alif-1.0` | Traversaal.ai — Alif-1.0-8B (Llama 3.1 tokenizer) | 128,256 | byte-level BPE | **2.641** | 2.736 | 1.476 | 2.792 | 3.119 | 17.8% | 491/491 | 0 | 2.651 | same encodings as `llama-3`, `urdu-llama3-almanach`, `urdu-llama3.2-custom`, `qalb-1.0` |
| 45 | 36 | `qalb-1.0` | Qalb (enstazao) — Qalb-1.0-8B (Llama 3.1 tokenizer) | 128,256 | byte-level BPE | **2.641** | 2.736 | 1.476 | 2.792 | 3.119 | 17.8% | 491/491 | 0 | 2.651 | same encodings as `llama-3`, `urdu-llama3-almanach`, `urdu-llama3.2-custom`, `alif-1.0` |
| 46 | 37 | `qwen-3` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 151,669 | byte-level BPE | **2.633** | 2.684 | 1.472 | 2.801 | 3.082 | 3.0% | 491/491 | 0 | 2.642 | same encodings as `qwen-2.5` |
| 47 | 38 | `qwen-2.5` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 151,665 | byte-level BPE | **2.633** | 2.684 | 1.472 | 2.801 | 3.082 | 3.0% | 491/491 | 0 | 2.642 | same encodings as `qwen-3` |
| 48 | 39 | `glm-4.5` | Zhipu AI / Z.ai — GLM-4.5 BPE 151k | 151,365 | byte-level BPE | **2.482** | 2.564 | 1.388 | 2.970 | 3.373 | 13.6% | 491/491 | 0 | 2.491 |  |
| 49 | 40 | `urdu-llama-bilal` | community (BilalKhan1) — Llama + Urdu tokenizer | 129,435 | byte-level BPE | **2.372** | 2.382 | 1.326 | 3.108 | 2.834 | 26.1% | 491/491 | 0 | 2.379 |  |
| 50 | 41 | `minimax-m2` | MiniMax — MiniMax BPE 200k | 200,054 | byte-level BPE | **2.268** | 2.352 | 1.268 | 3.252 | 3.604 | 7.0% | 491/491 | 0 | 2.274 | same encodings as `minimax-m3` |
| 51 | 42 | `minimax-m3` | MiniMax — MiniMax-M3 byte-level BPE 200k | 200,061 | byte-level BPE | **2.268** | 2.352 | 1.268 | 3.252 | 3.604 | 7.0% | 491/491 | 0 | 2.274 | same encodings as `minimax-m2` |
| 52 | 43 | `eurollm` | UTTER (EU) — EuroLLM SentencePiece BPE 128k | 128,000 | BPE (SentencePiece-style, byte fallback) | **2.147** | 2.207 | 1.200 | 3.434 | 3.678 | 6.1% | 491/491 | 0 | 2.150 |  |
| 53 | 44 | `hunyuan` | Tencent — Hunyuan byte-level BPE 128k | 128,166 | byte-level BPE | **1.932** | 1.975 | 1.080 | 3.816 | 4.245 | 0.6% | 491/491 | 0 | 1.936 |  |
| 54 | 45 | `gpt-4` | OpenAI — cl100k_base (tiktoken) | 100,263 | byte-level BPE | **1.919** | 1.967 | 1.073 | 3.842 | 4.275 | 0.7% | 491/491 | 0 | 1.923 | same encodings as `phi-4`, `olmo-2` |
| 55 | 46 | `phi-4` | Microsoft — Phi-4 (cl100k-derived) 100k | 100,352 | byte-level BPE | **1.919** | 1.967 | 1.073 | 3.842 | 4.275 | 0.7% | 491/491 | 0 | 1.923 | same encodings as `gpt-4`, `olmo-2` |
| 56 | 47 | `olmo-2` | Ai2 — OLMo 2/3 (cl100k-derived) 100k | 100,278 | byte-level BPE | **1.919** | 1.967 | 1.073 | 3.842 | 4.275 | 0.7% | 491/491 | 0 | 1.923 | same encodings as `gpt-4`, `phi-4` |
| 57 | 48 | `granite-4.2` | IBM — Granite 4.x (cl100k-derived) 100k | 100,352 | byte-level BPE | **1.919** | 1.966 | 1.073 | 3.843 | 4.275 | 0.7% | 491/491 | 0 | 1.923 |  |
| 58 | 49 | `lfm2` | Liquid AI — LFM2/LFM2.5 byte-level BPE 65k | 64,402 | byte-level BPE | **1.900** | 1.961 | 1.062 | 3.880 | 4.181 | 3.3% | 491/491 | 0 | 1.903 |  |
| 59 | 50 | `falcon-3` | TII — Falcon3 BPE 131k | 131,072 | byte-level BPE | **1.813** | 1.844 | 1.013 | 4.067 | 4.504 | 1.1% | 491/491 | 0 | 1.817 |  |
| 60 | 51 | `llama-2` | Meta — Llama 2 SentencePiece BPE 32k | 32,000 | BPE (SentencePiece-style, byte fallback) | **1.714** | 1.722 | 0.958 | 4.302 | 3.937 | 0.6% | 491/491 | 0 | 1.711 | same encodings as `urdu-llama2-almanach` |
| 61 | 52 | `urdu-llama2-almanach` | Inria ALMAnaCH — Llama-2-7B-mono-Urdu (Urdu-adapted vocab) | 32,000 | BPE (SentencePiece-style, byte fallback) | **1.714** | 1.722 | 0.958 | 4.302 | 3.937 | 0.6% | 491/491 | 0 | 1.711 | same encodings as `llama-2` |
| 62 | 53 | `grok-1` | xAI — Grok-1 SentencePiece 131k | 131,072 | BPE (SentencePiece-style, byte fallback) | **1.618** | 1.646 | 0.905 | 4.557 | 4.822 | 0.6% | 491/491 | 0 | 1.619 |  |
| 63 | – | `claude-legacy` | Anthropic — Claude 1/2 BPE 65k (legacy, public) | 65,000 | byte-level BPE | **1.554** | 1.579 | 0.869 | 4.744 | 4.945 | 0.6% | 467/491 | 0 | 1.556 | **not lossless** |
| 64 | 54 | `gpt-2` | OpenAI — GPT-2 byte-level BPE | 50,257 | byte-level BPE | **1.385** | 1.406 | 0.774 | 5.325 | 5.749 | 0.5% | 491/491 | 0 | 1.386 |  |
| 65 | 55 | `sarvam-1` | Sarvam AI — Sarvam-1 SentencePiece 68k (Indic) | 68,096 | BPE (SentencePiece-style, byte fallback) | **1.005** | 1.002 | 0.562 | 7.333 | 7.407 | 0.1% | 491/491 | 0 | 1.003 |  |
| 66 | 56 | `byt5` | Google — ByT5 raw UTF-8 bytes | 384 | bytes (UTF-8, no merges) | **1.000** | 1.000 | 0.559 | 7.373 | 7.450 | 0.1% | 491/491 | 0 | 1.000 |  |

## Leaky reference (not a competitor)

| tokenizer | why it is not ranked | vocab | bytes/tok | fertility | STRR | G1 | lines b/tok |
|---|---|---:|---:|---:|---:|---:|---:|
| `hindko-probe-bpe32k` | **LEAKY**: trained on the whole corpus, including this test text | 32,000 | 7.578 | 1.115 | 91.1% | 175/491 | 7.559 |

## Per source (test_strict)

test_strict holds newspaper 289 docs / 530,487 bytes, book 185 docs / 885,697 bytes, web 17 docs / 34,842 bytes. Web is small, so its numbers are noisy.

| tokenizer | newspaper b/tok | newspaper fert. | newspaper × tok | book b/tok | book fert. | book × tok | web b/tok | web fert. | web × tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`R2-A10-MinGram-P1r3-D2-48k`** | 7.714 | 1.114 | 1.000 | 7.209 | 1.128 | 1.000 | 6.741 | 1.199 | 1.000 |
| `gpt-4o` | 4.542 | 1.895 | 1.698 | 4.229 | 1.948 | 1.705 | 4.197 | 1.994 | 1.606 |
| `gpt-4` | 2.006 | 4.224 | 3.846 | 1.871 | 4.304 | 3.854 | 1.924 | 4.257 | 3.503 |
| `gemma-4` | 4.851 | 1.772 | 1.590 | 4.484 | 1.825 | 1.608 | 4.431 | 1.856 | 1.521 |
| `gemma-2` | 3.963 | 2.173 | 1.946 | 3.690 | 2.224 | 1.954 | 3.733 | 2.212 | 1.806 |
| `llama-4` | 3.887 | 2.218 | 1.984 | 3.638 | 2.269 | 1.982 | 3.705 | 2.260 | 1.819 |
| `llama-3` | 2.871 | 2.963 | 2.687 | 2.520 | 3.209 | 2.861 | 2.650 | 3.096 | 2.543 |
| `llama-2` | 1.732 | 4.026 | 4.454 | 1.703 | 3.885 | 4.233 | 1.719 | 3.965 | 3.920 |
| `qwen-3.5` | 3.762 | 2.291 | 2.051 | 3.510 | 2.340 | 2.054 | 3.524 | 2.349 | 1.913 |
| `qwen-3` | 2.724 | 3.083 | 2.832 | 2.581 | 3.082 | 2.793 | 2.635 | 3.068 | 2.559 |
| `deepseek-v4.1` | 3.509 | 2.450 | 2.198 | 3.295 | 2.490 | 2.188 | 3.366 | 2.470 | 2.003 |
| `deepseek-v3` | 3.509 | 2.450 | 2.198 | 3.295 | 2.490 | 2.188 | 3.366 | 2.470 | 2.003 |
| `mistral-nemo` | 4.072 | 2.116 | 1.894 | 3.806 | 2.168 | 1.894 | 3.886 | 2.155 | 1.734 |
| `command-a-plus` | 3.839 | 2.243 | 2.009 | 3.599 | 2.291 | 2.003 | 3.680 | 2.274 | 1.832 |
| `phi-4` | 2.006 | 4.224 | 3.846 | 1.871 | 4.304 | 3.854 | 1.924 | 4.257 | 3.503 |
| `grok-1` | 1.666 | 4.839 | 4.631 | 1.590 | 4.814 | 4.534 | 1.634 | 4.789 | 4.126 |
| `kimi-k2` | 3.631 | 2.375 | 2.124 | 3.473 | 2.374 | 2.076 | 3.488 | 2.405 | 1.933 |
| `glm-5` | 3.174 | 2.712 | 2.430 | 2.943 | 2.792 | 2.450 | 3.028 | 2.752 | 2.226 |
| `falcon-h1` | 4.762 | 1.802 | 1.620 | 4.372 | 1.870 | 1.649 | 4.361 | 1.874 | 1.546 |
| `bloom` | 5.614 | 1.535 | 1.374 | 4.872 | 1.688 | 1.480 | 4.958 | 1.683 | 1.359 |
| `roberta-urdu` | 6.223 | 1.384 | 1.240 | 5.455 | 1.498 | 1.322 | 5.307 | 1.540 | 1.270 |
| `tiny-aya` | 4.239 | 2.031 | 1.820 | 4.035 | 2.032 | 1.787 | 3.999 | 2.061 | 1.686 |
| `claude-legacy` | 1.600 | 4.965 | 4.821 | 1.528 | 4.932 | 4.719 | 1.559 | 4.969 | 4.323 |

## Tokenizers that are not lossless on test_strict

| tokenizer | G1 fail docs | fail modulo whitespace | chars lost | chars added | UNK |
|---|---:|---:|---:|---:|---:|
| `urdu-bert-64k` | 491 | 490 | 23,415 | 14,904 | 2 |
| `urdu-gpt2-20k` | 325 | 325 | 319 | 18,924 | 3,633 |
| `muril` | 491 | 433 | 160 | 190 | 38 |
| `indicbert-v2` | 491 | 491 | 9 | 175,712 | 2 |
| `sindhi-xlmr` | 319 | 30 | 180 | 621 | 14 |
| `xlm-r` | 318 | 28 | 175 | 645 | 6 |
| `nllb-200` | 341 | 160 | 3,653 | 9,500 | 1,777 |
| `mbert` | 491 | 435 | 4,170 | 18,720 | 3,744 |
| `mt5` | 316 | 24 | 169 | 615 | 0 |
| `claude-legacy` | 24 | 24 | 169 | 615 | 0 |
| `hindko-probe-bpe32k` (leaky) | 316 | 24 | 169 | 615 | 0 |

The failure causes are the ones diagnosed on dev (`eval/BASELINES_DEV.md`, `eval/lossy_diagnosis.json`): NFKC rewriting, dropped line breaks, character folding, UNK tokens. They were not re-diagnosed on test.

## In-study LM candidates on test (from the final-test bundle encoding)

These four are the LM test candidates of `colab/FINAL_TEST.md`. Their bytes/token and G1 below come from the encoding already made for the final-test bundle (`colab/build_final_test/VERIFY_FINAL_TEST.json`), not from the harness. For the chosen tokenizer both routes must agree.

| tokenizer | role | test bytes/tok (bundle) | dev bytes/tok (dev bundle) | test tokens | G1 harness encoder | G1 native encoder |
|---|---|---:|---:|---:|---:|---:|
| `A1-P1r3-D2-16k` | baseline | 6.948 | 6.788 | 208,826 | 491/491 | 491/491 |
| `R2-A10-MinGram-P1r3-D2-48k` | chosen | 7.373 | 7.224 | 196,793 | 491/491 | 491/491 |
| `R2-A4-SPnat-D2-32k` | report-only | 7.277 | 7.076 | 199,390 | 491/491 | 491/491 |
| `R2-A10-MinGram-P1r3-D2-32k` | report-only | 7.275 | 7.126 | 199,455 | 491/491 | 491/491 |

Cross-check: chosen tokenizer bytes/token on test = 7.373362 (harness) vs 7.373362 (bundle) → identical.

## What this page does not cover

- **Bits-per-byte on test** comes from the one-shot LM run (`colab/FINAL_TEST.md`), for the four LM test candidates only; external tokenizers have no LM in this study.
- **Robustness and morphology** were computed by the same harness run and are in `eval/test_competitors.json` (robustness) and the per-tokenizer `summary.json` files; they are report-only, as on dev.
- **G2–G5, R1:** gates for the project's own candidates, run on dev (PLAN §4.3); not repeated here.

## Reproduce

```
set PYTHONIOENCODING=utf-8
python eval\run_test_competitors.py --shard 0 --of 3   (and --shard 1, --shard 2, in parallel)
python eval\report_test_competitors.py
```

Results: `eval/results/test_strict/<name>/`. Harness code sha256: adapters.py `bc377b786f…`, harness.py `31db30dd0e…`, morph_eval.py `73e32d88f9…`, perturb.py `f965f79f7a…`.
