# External tokenizers on Hindko dev (Stage 0 baselines)

Generated 2026-09-26T11:26:59Z by `eval/report_baselines.py` from the harness outputs in `eval/results/`. Every number is measured; nothing on this page touches the test split.

- **Text:** `data/dev_strict.jsonl`, the evaluation set: 836 documents, 1,445,513 UTF-8 bytes, 172,226 whitespace words, sha256 `b655395372cd…`. It is the strict validation split of manifest `76582d3a…`, in canonical form (`hp.normalize` 1.0.1).
- **Tokenizers:** 66 external tokenizers (all working entries of `baselines/manifest.json`), each loaded through `baselines/load_baselines.py` and used with its own native encoder. No BOS/EOS/CLS tokens are added.
  - 65 are ranked. **55 of the 65 round-trip every dev_strict document exactly** (G1).
  - **The raw ranking flatters lossy tokenizers.** 7 of the top 10 by bytes/token are not lossless (`urdu-bert-64k`, `urdu-gpt2-20k`, `muril`, `indicbert-v2`, `sindhi-xlmr`, `xlm-r`, `nllb-200`): they drop line breaks, strip or fold characters, or emit UNK, so they encode less text than they were given. Among lossless tokenizers the best are `roberta-urdu` 5.771 bytes/token, `bloom` 5.215 bytes/token, `gemma-4` 4.640 bytes/token (column *ll #*).
  - `hindko-probe-bpe32k` is **LEAKY**: it is this project's earlier probe, trained on the whole corpus including dev and test. It is reported in its own section and never ranked.
- **Ranking:** by bytes/token on dev_strict, descending. Bytes/token is a screening metric, never decisive on its own (PLAN §4.2); the decision metric is LM bits-per-byte (PLAN §5), which needs trained candidates.
- **NSL:** NSL against A1-P1-16k is pending, because that tokenizer is not trained yet (`harness.py nsl` adds it later). The column *NSL vs o200k* is a **provisional** substitute: tokens ÷ GPT-4o's tokens on the same documents.

## How to read the columns

- **bytes/tok, chars/tok:** Σ UTF-8 bytes (or characters) ÷ Σ tokens. Higher = fewer tokens.
- **fertility:** the mean number of tokens that overlap a whitespace word. It is measured on the in-context encoding of whole documents, through the tokenizers' character offsets. Tokens made only of whitespace (a lone `▁`, `\n`) belong to no word.
- **STRR:** the share of words that are a single token. **cont.:** the share split into ≥ 2 tokens (= 1 − STRR).
- **G1:** documents with `decode(encode(doc)) == doc`, out of 836. **UNK:** unknown-token count.
- **lines b/tok:** bytes/token when every line is encoded on its own, so no line-break tokens exist at all. It compares tokenizers that silently drop line breaks (turning them into spaces) with lossless ones on equal terms.

## Ranking on dev_strict

*#* = rank by bytes/token; *ll #* = rank among the tokenizers that round-trip all 836 documents.

| # | ll # | tokenizer | provider / family | vocab | algorithm | bytes/tok | chars/tok | NSL vs o200k | fertility | STRR | G1 | UNK | lines b/tok | note |
|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | – | `urdu-bert-64k` | community (farahadeeba) — Urdu BERT WordPiece 64k | 64,000 | WordPiece | **6.436** | 3.608 | 0.681 | 1.304 | 75.9% | 0/836 | 36 | 6.406 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 2 | 1 | `roberta-urdu` | UrduHack — RoBERTa-Urdu byte-level BPE 52k (monolingual Urdu) | 52,000 | byte-level BPE | **5.771** | 3.236 | 0.760 | 1.414 | 73.5% | 836/836 | 0 | 5.905 |  |
| 3 | – | `urdu-gpt2-20k` | community (aariciah) — Urdu GPT-2 BPE 20k (Urdu-only vocab, NFKC+lowercase) | 20,000 | BPE (character-level, no byte fallback) | **5.676** | 3.182 | 0.772 | 1.439 | 67.5% | 149/836 | 6,875 | 5.860 | **not lossless**; line breaks: line breaks become UNK tokens |
| 4 | – | `muril` | Google — MuRIL WordPiece 197k | 197,258 | WordPiece | **5.613** | 3.147 | 0.781 | 1.495 | 64.6% | 1/836 | 152 | 5.587 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 5 | – | `indicbert-v2` | AI4Bharat — IndicBERT v2 WordPiece 250k | 250,000 | WordPiece | **5.469** | 3.066 | 0.802 | 1.535 | 60.8% | 0/836 | 32 | 5.443 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 6 | – | `sindhi-xlmr` | community (Kashif786) — XLM-R + Sindhi vocabulary extension (Unigram 265k) | 264,635 | Unigram (SentencePiece) | **5.243** | 2.939 | 0.836 | 1.537 | 61.0% | 154/836 | 51 | 5.218 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 7 | 2 | `bloom` | BigScience — BLOOM byte-level BPE 250k | 250,680 | byte-level BPE | **5.215** | 2.924 | 0.841 | 1.579 | 58.9% | 836/836 | 0 | 5.285 |  |
| 8 | – | `xlm-r` | Meta — XLM-R SentencePiece Unigram 250k | 250,002 | Unigram (SentencePiece) | **4.947** | 2.774 | 0.886 | 1.618 | 58.6% | 154/836 | 48 | 4.924 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 9 | – | `nllb-200` | Meta — NLLB-200 SentencePiece BPE 256k | 256,204 | BPE (character-level, no byte fallback) | **4.737** | 2.656 | 0.926 | 1.769 | 48.6% | 146/836 | 690 | 4.714 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 10 | 3 | `gemma-4` | Google — Gemma 4 SentencePiece 262k | 262,144 | BPE (SentencePiece-style, byte fallback) | **4.640** | 2.601 | 0.945 | 1.765 | 49.6% | 836/836 | 0 | 4.720 | same encodings as `gemma-3` |
| 11 | 4 | `gemma-3` | Google — Gemma 3 SentencePiece 262k | 262,145 | BPE (SentencePiece-style, byte fallback) | **4.640** | 2.601 | 0.945 | 1.765 | 49.6% | 836/836 | 0 | 4.720 | same encodings as `gemma-4` |
| 12 | 5 | `sarvam-30b` | Sarvam AI — Sarvam 2026 tokenizer 262k (Gemma-style SentencePiece BPE) | 262,144 | BPE (SentencePiece-style, byte fallback) | **4.640** | 2.601 | 0.945 | 1.765 | 49.6% | 836/836 | 0 | 4.720 |  |
| 13 | 6 | `falcon-h1` | TII — Falcon-H1 BPE 261k (34B) | 261,120 | byte-level BPE | **4.563** | 2.558 | 0.961 | 1.792 | 47.2% | 836/836 | 0 | 4.640 |  |
| 14 | 7 | `gpt-4o` | OpenAI — o200k_base (tiktoken) | 200,000 | byte-level BPE | **4.384** | 2.458 | 1.000 | 1.883 | 42.7% | 836/836 | 0 | 4.428 | same encodings as `gpt-oss`, `phi-4-mini` |
| 15 | 8 | `gpt-oss` | OpenAI — o200k_harmony | 200,019 | byte-level BPE | **4.384** | 2.458 | 1.000 | 1.883 | 42.7% | 836/836 | 0 | 4.428 | same encodings as `gpt-4o`, `phi-4-mini` |
| 16 | 9 | `phi-4-mini` | Microsoft — Phi-4-mini (o200k-derived) 200k | 200,029 | byte-level BPE | **4.384** | 2.458 | 1.000 | 1.883 | 42.7% | 836/836 | 0 | 4.428 | same encodings as `gpt-4o`, `gpt-oss` |
| 17 | – | `mbert` | Google — mBERT WordPiece 119k (cased) | 119,547 | WordPiece | **4.233** | 2.373 | 1.036 | 1.983 | 43.2% | 1/836 | 2,639 | 4.213 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 18 | 10 | `tiny-aya` | Cohere — Tiny Aya byte-level BPE 261k | 261,010 | byte-level BPE | **4.152** | 2.328 | 1.056 | 1.977 | 39.4% | 836/836 | 0 | 4.215 |  |
| 19 | – | `mt5` | Google — mT5 SentencePiece Unigram 250k | 250,100 | SentencePiece Unigram + byte fallback | **3.960** | 2.220 | 1.107 | 1.831 | 42.1% | 157/836 | 0 | 3.941 | **not lossless**; line breaks: each run of line breaks becomes one space |
| 20 | 11 | `mistral-nemo` | Mistral AI — Tekken (tiktoken-style BPE, 131k) | 131,072 | byte-level BPE | **3.942** | 2.210 | 1.112 | 2.097 | 31.2% | 836/836 | 0 | 3.975 | same encodings as `mistral-small-4`, `apertus`, `sarvam-m`, `nemotron-3` |
| 21 | 12 | `mistral-small-4` | Mistral AI — Tekken (tekken.json v15, 131k) | 131,072 | byte-level BPE | **3.942** | 2.210 | 1.112 | 2.097 | 31.2% | 836/836 | 0 | 3.975 | same encodings as `mistral-nemo`, `apertus`, `sarvam-m`, `nemotron-3` |
| 22 | 13 | `apertus` | Swiss AI — Apertus BPE 131k | 131,072 | byte-level BPE | **3.942** | 2.210 | 1.112 | 2.097 | 31.2% | 836/836 | 0 | 3.975 | same encodings as `mistral-nemo`, `mistral-small-4`, `sarvam-m`, `nemotron-3` |
| 23 | 14 | `sarvam-m` | Sarvam AI — Sarvam-M (Mistral Small 3.1 Tekken) | 131,072 | byte-level BPE | **3.942** | 2.210 | 1.112 | 2.097 | 31.2% | 836/836 | 0 | 3.975 | same encodings as `mistral-nemo`, `mistral-small-4`, `apertus`, `nemotron-3` |
| 24 | 15 | `nemotron-3` | NVIDIA — Nemotron 3 BPE 131k | 131,072 | byte-level BPE | **3.942** | 2.210 | 1.112 | 2.097 | 31.2% | 836/836 | 0 | 3.975 | same encodings as `mistral-nemo`, `mistral-small-4`, `apertus`, `sarvam-m` |
| 25 | 16 | `gemma-2` | Google — Gemma 2 SentencePiece 256k | 256,000 | BPE (SentencePiece-style, byte fallback) | **3.834** | 2.149 | 1.144 | 2.145 | 32.0% | 836/836 | 0 | 3.885 |  |
| 26 | 17 | `llama-4` | Meta — Llama 4 BPE 202k | 201,135 | byte-level BPE | **3.759** | 2.108 | 1.166 | 2.201 | 28.5% | 836/836 | 0 | 3.788 |  |
| 27 | 18 | `command-a-plus` | Cohere — Command A+ (05-2026) byte-level BPE 255k | 255,032 | byte-level BPE | **3.729** | 2.091 | 1.176 | 2.217 | 27.1% | 836/836 | 0 | 3.757 |  |
| 28 | 19 | `qwen-3.5` | Alibaba Qwen — Qwen3.5 byte-level BPE 248k | 248,070 | byte-level BPE | **3.631** | 2.036 | 1.208 | 2.268 | 26.6% | 836/836 | 0 | 3.676 | same encodings as `qwen-3.8` |
| 29 | 20 | `qwen-3.8` | Alibaba Qwen — Qwen3.8 byte-level BPE 248k | 248,077 | byte-level BPE | **3.631** | 2.036 | 1.208 | 2.268 | 26.6% | 836/836 | 0 | 3.676 | same encodings as `qwen-3.5` |
| 30 | 21 | `kimi-k2` | Moonshot AI — Kimi tiktoken BPE 160k | 163,840 | byte-level BPE | **3.543** | 1.986 | 1.238 | 2.337 | 28.6% | 836/836 | 0 | 3.569 |  |
| 31 | 22 | `ernie-4.5` | Baidu — ERNIE 4.5 SentencePiece 103k | 101,304 | BPE (SentencePiece-style, byte fallback) | **3.540** | 1.984 | 1.239 | 2.326 | 27.9% | 836/836 | 0 | 3.583 |  |
| 32 | 23 | `pashto-lfm2.5` | community (nassimjp) — LFM2.5 + Pashto vocabulary extension | 125,039 | byte-level BPE | **3.531** | 1.980 | 1.242 | 2.340 | 22.8% | 836/836 | 0 | 3.555 |  |
| 33 | 24 | `deepseek-v3` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 128,815 | byte-level BPE | **3.418** | 1.916 | 1.283 | 2.411 | 24.5% | 836/836 | 0 | 3.453 | same encodings as `deepseek-r1`, `deepseek-v4`, `deepseek-v4.1` |
| 34 | 25 | `deepseek-r1` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 128,815 | byte-level BPE | **3.418** | 1.916 | 1.283 | 2.411 | 24.5% | 836/836 | 0 | 3.453 | same encodings as `deepseek-v3`, `deepseek-v4`, `deepseek-v4.1` |
| 35 | 26 | `deepseek-v4` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 129,280 | byte-level BPE | **3.418** | 1.916 | 1.283 | 2.411 | 24.5% | 836/836 | 0 | 3.453 | same encodings as `deepseek-v3`, `deepseek-r1`, `deepseek-v4.1` |
| 36 | 27 | `deepseek-v4.1` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 129,280 | byte-level BPE | **3.418** | 1.916 | 1.283 | 2.411 | 24.5% | 836/836 | 0 | 3.453 | same encodings as `deepseek-v3`, `deepseek-r1`, `deepseek-v4` |
| 37 | 28 | `glm-5` | Zhipu AI / Z.ai — GLM-5 byte-level BPE 155k | 154,856 | byte-level BPE | **3.084** | 1.729 | 1.422 | 2.676 | 14.3% | 836/836 | 0 | 3.112 |  |
| 38 | 29 | `command-r` | Cohere — Command-R / Aya Expanse byte-level BPE 256k | 255,029 | byte-level BPE | **2.747** | 1.540 | 1.596 | 2.961 | 23.2% | 836/836 | 0 | 2.770 | same encodings as `command-r7b` |
| 39 | 30 | `command-r7b` | Cohere — Command R7B byte-level BPE 256k | 255,033 | byte-level BPE | **2.747** | 1.540 | 1.596 | 2.961 | 23.2% | 836/836 | 0 | 2.770 | same encodings as `command-r` |
| 40 | 31 | `llama-3` | Meta — Llama 3 tiktoken-BPE 128k | 128,256 | byte-level BPE | **2.736** | 1.534 | 1.602 | 2.981 | 19.0% | 836/836 | 0 | 2.756 | same encodings as `urdu-llama3-almanach`, `urdu-llama3.2-custom`, `alif-1.0`, `qalb-1.0` |
| 41 | 32 | `urdu-llama3-almanach` | Inria ALMAnaCH — Llama-3-8B-mono-Urdu (Urdu-adapted vocab) | 128,256 | byte-level BPE | **2.736** | 1.534 | 1.602 | 2.981 | 19.0% | 836/836 | 0 | 2.756 | same encodings as `llama-3`, `urdu-llama3.2-custom`, `alif-1.0`, `qalb-1.0` |
| 42 | 33 | `urdu-llama3.2-custom` | community (SabahNawab) — Llama-3.2-3B + custom Urdu tokens | 145,613 | byte-level BPE | **2.736** | 1.534 | 1.602 | 2.981 | 19.0% | 836/836 | 0 | 2.756 | same encodings as `llama-3`, `urdu-llama3-almanach`, `alif-1.0`, `qalb-1.0` |
| 43 | 34 | `alif-1.0` | Traversaal.ai — Alif-1.0-8B (Llama 3.1 tokenizer) | 128,256 | byte-level BPE | **2.736** | 1.534 | 1.602 | 2.981 | 19.0% | 836/836 | 0 | 2.756 | same encodings as `llama-3`, `urdu-llama3-almanach`, `urdu-llama3.2-custom`, `qalb-1.0` |
| 44 | 35 | `qalb-1.0` | Qalb (enstazao) — Qalb-1.0-8B (Llama 3.1 tokenizer) | 128,256 | byte-level BPE | **2.736** | 1.534 | 1.602 | 2.981 | 19.0% | 836/836 | 0 | 2.756 | same encodings as `llama-3`, `urdu-llama3-almanach`, `urdu-llama3.2-custom`, `alif-1.0` |
| 45 | 36 | `qwen-3` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 151,669 | byte-level BPE | **2.684** | 1.505 | 1.634 | 2.998 | 3.1% | 836/836 | 0 | 2.703 | same encodings as `qwen-2.5` |
| 46 | 37 | `qwen-2.5` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 151,665 | byte-level BPE | **2.684** | 1.505 | 1.634 | 2.998 | 3.1% | 836/836 | 0 | 2.703 | same encodings as `qwen-3` |
| 47 | 38 | `glm-4.5` | Zhipu AI / Z.ai — GLM-4.5 BPE 151k | 151,365 | byte-level BPE | **2.564** | 1.437 | 1.710 | 3.228 | 14.3% | 836/836 | 0 | 2.581 |  |
| 48 | 39 | `urdu-llama-bilal` | community (BilalKhan1) — Llama + Urdu tokenizer | 129,435 | byte-level BPE | **2.382** | 1.335 | 1.841 | 2.792 | 25.9% | 836/836 | 0 | 2.395 |  |
| 49 | 40 | `minimax-m2` | MiniMax — MiniMax BPE 200k | 200,054 | byte-level BPE | **2.352** | 1.319 | 1.864 | 3.444 | 7.7% | 836/836 | 0 | 2.365 | same encodings as `minimax-m3` |
| 50 | 41 | `minimax-m3` | MiniMax — MiniMax-M3 byte-level BPE 200k | 200,061 | byte-level BPE | **2.352** | 1.319 | 1.864 | 3.444 | 7.7% | 836/836 | 0 | 2.365 | same encodings as `minimax-m2` |
| 51 | 42 | `eurollm` | UTTER (EU) — EuroLLM SentencePiece BPE 128k | 128,000 | BPE (SentencePiece-style, byte fallback) | **2.207** | 1.237 | 1.987 | 3.551 | 6.6% | 836/836 | 0 | 2.215 |  |
| 52 | 43 | `hunyuan` | Tencent — Hunyuan byte-level BPE 128k | 128,166 | byte-level BPE | **1.975** | 1.107 | 2.220 | 4.121 | 0.6% | 836/836 | 0 | 1.983 |  |
| 53 | 44 | `gpt-4` | OpenAI — cl100k_base (tiktoken) | 100,263 | byte-level BPE | **1.967** | 1.103 | 2.229 | 4.138 | 0.8% | 836/836 | 0 | 1.975 | same encodings as `phi-4`, `olmo-2` |
| 54 | 45 | `phi-4` | Microsoft — Phi-4 (cl100k-derived) 100k | 100,352 | byte-level BPE | **1.967** | 1.103 | 2.229 | 4.138 | 0.8% | 836/836 | 0 | 1.975 | same encodings as `gpt-4`, `olmo-2` |
| 55 | 46 | `olmo-2` | Ai2 — OLMo 2/3 (cl100k-derived) 100k | 100,278 | byte-level BPE | **1.967** | 1.103 | 2.229 | 4.138 | 0.8% | 836/836 | 0 | 1.975 | same encodings as `gpt-4`, `phi-4` |
| 56 | 47 | `granite-4.2` | IBM — Granite 4.x (cl100k-derived) 100k | 100,352 | byte-level BPE | **1.966** | 1.102 | 2.230 | 4.138 | 0.8% | 836/836 | 0 | 1.975 |  |
| 57 | 48 | `lfm2` | Liquid AI — LFM2/LFM2.5 byte-level BPE 65k | 64,402 | byte-level BPE | **1.961** | 1.100 | 2.235 | 4.020 | 3.2% | 836/836 | 0 | 1.969 |  |
| 58 | 49 | `falcon-3` | TII — Falcon3 BPE 131k | 131,072 | byte-level BPE | **1.844** | 1.034 | 2.377 | 4.392 | 0.7% | 836/836 | 0 | 1.852 |  |
| 59 | 50 | `llama-2` | Meta — Llama 2 SentencePiece BPE 32k | 32,000 | BPE (SentencePiece-style, byte fallback) | **1.722** | 0.966 | 2.546 | 3.883 | 0.6% | 836/836 | 0 | 1.715 | same encodings as `urdu-llama2-almanach` |
| 60 | 51 | `urdu-llama2-almanach` | Inria ALMAnaCH — Llama-2-7B-mono-Urdu (Urdu-adapted vocab) | 32,000 | BPE (SentencePiece-style, byte fallback) | **1.722** | 0.966 | 2.546 | 3.883 | 0.6% | 836/836 | 0 | 1.715 | same encodings as `llama-2` |
| 61 | 52 | `grok-1` | xAI — Grok-1 SentencePiece 131k | 131,072 | BPE (SentencePiece-style, byte fallback) | **1.646** | 0.923 | 2.664 | 4.706 | 0.5% | 836/836 | 0 | 1.646 |  |
| 62 | – | `claude-legacy` | Anthropic — Claude 1/2 BPE 65k (legacy, public) | 65,000 | byte-level BPE | **1.579** | 0.885 | 2.776 | 4.831 | 0.7% | 742/836 | 0 | 1.584 | **not lossless** |
| 63 | 53 | `gpt-2` | OpenAI — GPT-2 byte-level BPE | 50,257 | byte-level BPE | **1.406** | 0.788 | 3.118 | 5.626 | 0.5% | 836/836 | 0 | 1.409 |  |
| 64 | 54 | `sarvam-1` | Sarvam AI — Sarvam-1 SentencePiece 68k (Indic) | 68,096 | BPE (SentencePiece-style, byte fallback) | **1.002** | 0.562 | 4.375 | 7.381 | 0.1% | 836/836 | 0 | 0.998 |  |
| 65 | 55 | `byt5` | Google — ByT5 raw UTF-8 bytes | 384 | bytes (UTF-8, no merges) | **1.000** | 0.561 | 4.384 | 7.398 | 0.1% | 836/836 | 0 | 1.000 |  |

## Leaky reference (not a competitor)

| tokenizer | why it is not ranked | vocab | bytes/tok | fertility | STRR | G1 | UNK | lines b/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `hindko-probe-bpe32k` | **LEAKY**: trained on the whole corpus, including dev and test | 32,000 | 7.529 | 1.114 | 90.9% | 157/836 | 0 | 7.494 |

Its numbers show what an in-domain vocabulary can reach. They are optimistic, because it saw this very text.

## Tokenizers that are not lossless on dev_strict

The table counts, over all 836 documents: failed round trips; failures that remain after all whitespace is collapsed to single spaces (i.e. more than line-break or space handling is lost); non-whitespace characters lost and added (multiset difference); and UNK tokens. A tokenizer that drops line breaks spends no tokens on them, which flatters its bytes/token; compare *lines b/tok* instead.

| tokenizer | G1 fail docs | fail modulo whitespace | chars lost | chars added | UNK | line breaks (measured in failing docs: in → after decode) | most lost / added characters |
|---|---:|---:|---:|---:|---:|---|---|
| `urdu-bert-64k` | 836 | 835 | 18,644 | 10,917 | 36 | 6,826 → 0 | lost: Yeh With Hamza Above ×6,034, Damma ×4,065, Alef With Madda Above ×3,905; added: Yeh ×6,034, Alef ×3,908, Waw ×261 |
| `urdu-gpt2-20k` | 687 | 687 | 1,726 | 38,798 | 6,875 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Latin Capital T ×92, Latin Capital C ×82; added: Latin Small U ×6,955, Latin Small N ×6,900, Latin Small K ×6,881 |
| `muril` | 835 | 663 | 646 | 760 | 152 | 6,826 → 0 | lost: Sign Sallallahou Alayhe Wassallam ×107, Alef ×73, Farsi Yeh ×50; added: Left Square Bracket ×152, Latin Capital U ×152, Latin Capital N ×152 |
| `indicbert-v2` | 836 | 836 | 136 | 156,468 | 32 | 6,826 → 0 | lost: Heh Goal ×25, Farsi Yeh ×17, Teh With Small V ×16; added: Number Sign ×156,308, Left Square Bracket ×32, Latin Capital U ×32 |
| `sindhi-xlmr` | 682 | 116 | 1,230 | 3,975 | 51 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Ligature Sallallahou Alayhe Wasallam ×32, Teh With Small V ×16; added: Full Stop ×3,447, Lam ×160, Heh ×64 |
| `xlm-r` | 682 | 116 | 1,230 | 4,167 | 48 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Ligature Sallallahou Alayhe Wasallam ×32, Teh With Small V ×16; added: Full Stop ×3,447, Lam ×160, Heh ×64 |
| `nllb-200` | 690 | 232 | 2,357 | 7,377 | 690 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Right Single Quotation Mark ×548, Left Single Quotation Mark ×539; added: Full Stop ×3,447, Less-Than Sign ×690, Latin Small U ×690 |
| `mbert` | 835 | 740 | 4,039 | 13,195 | 2,639 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Right Single Quotation Mark ×548, Left Single Quotation Mark ×539; added: Left Square Bracket ×2,639, Latin Capital U ×2,639, Latin Capital N ×2,639 |
| `mt5` | 679 | 94 | 1,182 | 3,927 | 0 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Ligature Sallallahou Alayhe Wasallam ×32, Replacement Character ×1; added: Full Stop ×3,447, Lam ×160, Heh ×64 |
| `claude-legacy` | 94 | 94 | 1,181 | 3,927 | 0 | 1,895 → 1,895 | lost: Horizontal Ellipsis ×1,149, Ligature Sallallahou Alayhe Wasallam ×32; added: Full Stop ×3,447, Lam ×160, Heh ×64 |
| `hindko-probe-bpe32k` (leaky) | 679 | 94 | 1,182 | 3,927 | 0 | 6,826 → 0 | lost: Horizontal Ellipsis ×1,149, Ligature Sallallahou Alayhe Wasallam ×32, Replacement Character ×1; added: Full Stop ×3,447, Lam ×160, Heh ×64 |

Full character lists (top 8, with code points): `eval/lossy_diagnosis.json`. What the lists show:

- **NFKC normalizers** (`mt5`, `xlm-r`, `sindhi-xlmr`, `nllb-200`, `claude-legacy`, and the leaky probe) rewrite every `…` as three full stops (1,149 → 3,447) and expand `ﷺ` into Arabic words.
- **`urdu-bert-64k`** folds letters (`ئ` → Arabic `ي` ×6,034; `آ` → `ا` ×3,905) and strips harakat (damma ×4,065, kasra ×2,284). Its top bytes/token is therefore measured on text it partly discards.
- **`indicbert-v2`**'s tokenizer.json has no WordPiece decoder, so `decode()` leaves the `##` continuation markers in the text (156,308 `#` characters added).
- **UNK tokens** (`[UNK]`, `<unk>`) replace characters outside the vocabulary: `mbert` ×2,639, `urdu-gpt2-20k` ×6,875 (mostly line breaks), `nllb-200` ×690, `muril` ×152. `xlm-r` and `sindhi-xlmr` also lose Hindko tone letters (U+08BF ×16, U+08BE ×10 among their top losses).
- Every tokenizer above except `claude-legacy` loses all 6,826 line breaks of dev_strict: as spaces, or as UNK tokens in `urdu-gpt2-20k`. `claude-legacy` keeps them.

## Per source (dev_strict)

dev_strict holds newspaper 280 docs / 532,872 bytes, book 539 docs / 878,168 bytes, web 17 docs / 34,473 bytes. Web is small (17 documents), so its numbers are noisy.

| # | tokenizer | newspaper b/tok | newspaper fert. | book b/tok | book fert. | web b/tok | web fert. |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `urdu-bert-64k` | 6.717 | 1.294 | 6.299 | 1.305 | 5.899 | 1.423 |
| 2 | `roberta-urdu` | 6.281 | 1.373 | 5.527 | 1.432 | 5.103 | 1.561 |
| 3 | `urdu-gpt2-20k` | 6.236 | 1.383 | 5.405 | 1.466 | 5.118 | 1.556 |
| 4 | `muril` | 6.018 | 1.445 | 5.414 | 1.518 | 5.100 | 1.646 |
| 5 | `indicbert-v2` | 5.786 | 1.502 | 5.311 | 1.548 | 5.019 | 1.673 |
| 6 | `sindhi-xlmr` | 5.512 | 1.512 | 5.106 | 1.547 | 4.895 | 1.649 |
| 7 | `bloom` | 5.655 | 1.525 | 4.995 | 1.605 | 4.825 | 1.720 |
| 8 | `xlm-r` | 5.211 | 1.588 | 4.811 | 1.631 | 4.667 | 1.725 |
| 9 | `nllb-200` | 4.803 | 1.809 | 4.707 | 1.743 | 4.503 | 1.862 |
| 10 | `gemma-4` | 4.879 | 1.765 | 4.520 | 1.761 | 4.284 | 1.872 |
| 11 | `gemma-3` | 4.879 | 1.765 | 4.520 | 1.761 | 4.284 | 1.872 |
| 12 | `sarvam-30b` | 4.879 | 1.765 | 4.520 | 1.761 | 4.284 | 1.872 |
| 13 | `falcon-h1` | 4.774 | 1.800 | 4.459 | 1.782 | 4.173 | 1.922 |
| 14 | `gpt-4o` | 4.571 | 1.886 | 4.289 | 1.876 | 4.123 | 2.016 |
| 15 | `gpt-oss` | 4.571 | 1.886 | 4.289 | 1.876 | 4.123 | 2.016 |
| 16 | `phi-4-mini` | 4.571 | 1.886 | 4.289 | 1.876 | 4.123 | 2.016 |
| 17 | `mbert` | 4.295 | 2.024 | 4.204 | 1.955 | 4.054 | 2.071 |
| 18 | `tiny-aya` | 4.240 | 2.033 | 4.112 | 1.942 | 3.871 | 2.082 |
| 19 | `mt5` | 4.039 | 1.852 | 3.916 | 1.816 | 3.894 | 1.916 |
| 20 | `mistral-nemo` | 4.085 | 2.112 | 3.865 | 2.085 | 3.807 | 2.188 |
| 21 | `mistral-small-4` | 4.085 | 2.112 | 3.865 | 2.085 | 3.807 | 2.188 |
| 22 | `apertus` | 4.085 | 2.112 | 3.865 | 2.085 | 3.807 | 2.188 |
| 23 | `sarvam-m` | 4.085 | 2.112 | 3.865 | 2.085 | 3.807 | 2.188 |
| 24 | `nemotron-3` | 4.085 | 2.112 | 3.865 | 2.085 | 3.807 | 2.188 |
| 25 | `gemma-2` | 3.977 | 2.168 | 3.759 | 2.129 | 3.634 | 2.223 |
| 26 | `llama-4` | 3.902 | 2.212 | 3.683 | 2.191 | 3.629 | 2.296 |
| 27 | `command-a-plus` | 3.842 | 2.244 | 3.669 | 2.197 | 3.597 | 2.314 |
| 28 | `qwen-3.5` | 3.768 | 2.290 | 3.561 | 2.251 | 3.410 | 2.374 |
| 29 | `qwen-3.8` | 3.768 | 2.290 | 3.561 | 2.251 | 3.410 | 2.374 |
| 30 | `kimi-k2` | 3.631 | 2.379 | 3.497 | 2.308 | 3.388 | 2.456 |
| 31 | `ernie-4.5` | 3.643 | 2.367 | 3.487 | 2.299 | 3.362 | 2.410 |
| 32 | `pashto-lfm2.5` | 3.612 | 2.383 | 3.488 | 2.312 | 3.422 | 2.432 |
| 33 | `deepseek-v3` | 3.513 | 2.451 | 3.372 | 2.383 | 3.211 | 2.531 |
| 34 | `deepseek-r1` | 3.513 | 2.451 | 3.372 | 2.383 | 3.211 | 2.531 |
| 35 | `deepseek-v4` | 3.513 | 2.451 | 3.372 | 2.383 | 3.211 | 2.531 |
| 36 | `deepseek-v4.1` | 3.513 | 2.451 | 3.372 | 2.383 | 3.211 | 2.531 |
| 37 | `glm-5` | 3.179 | 2.711 | 3.037 | 2.650 | 2.875 | 2.831 |
| 38 | `command-r` | 2.911 | 2.920 | 2.668 | 2.975 | 2.476 | 3.221 |
| 39 | `command-r7b` | 2.911 | 2.920 | 2.668 | 2.975 | 2.476 | 3.221 |
| 40 | `llama-3` | 2.872 | 2.966 | 2.670 | 2.981 | 2.494 | 3.210 |
| 41 | `urdu-llama3-almanach` | 2.872 | 2.966 | 2.670 | 2.981 | 2.494 | 3.210 |
| 42 | `urdu-llama3.2-custom` | 2.872 | 2.966 | 2.670 | 2.981 | 2.494 | 3.210 |
| 43 | `alif-1.0` | 2.872 | 2.966 | 2.670 | 2.981 | 2.494 | 3.210 |
| 44 | `qalb-1.0` | 2.872 | 2.966 | 2.670 | 2.981 | 2.494 | 3.210 |
| 45 | `qwen-3` | 2.739 | 3.071 | 2.656 | 2.953 | 2.568 | 3.064 |
| 46 | `qwen-2.5` | 2.739 | 3.071 | 2.656 | 2.953 | 2.568 | 3.064 |
| 47 | `glm-4.5` | 2.676 | 3.225 | 2.509 | 3.220 | 2.350 | 3.483 |
| 48 | `urdu-llama-bilal` | 2.426 | 2.863 | 2.361 | 2.745 | 2.259 | 2.974 |
| 49 | `minimax-m2` | 2.437 | 3.466 | 2.311 | 3.422 | 2.163 | 3.684 |
| 50 | `minimax-m3` | 2.437 | 3.466 | 2.311 | 3.422 | 2.163 | 3.684 |
| 51 | `eurollm` | 2.280 | 3.585 | 2.171 | 3.524 | 2.059 | 3.741 |
| 52 | `hunyuan` | 2.017 | 4.207 | 1.954 | 4.065 | 1.866 | 4.295 |
| 53 | `gpt-4` | 2.010 | 4.223 | 1.946 | 4.082 | 1.861 | 4.306 |
| 54 | `phi-4` | 2.010 | 4.223 | 1.946 | 4.082 | 1.861 | 4.306 |
| 55 | `olmo-2` | 2.010 | 4.223 | 1.946 | 4.082 | 1.861 | 4.306 |
| 56 | `granite-4.2` | 2.009 | 4.223 | 1.944 | 4.082 | 1.860 | 4.306 |
| 57 | `lfm2` | 2.017 | 4.071 | 1.935 | 3.983 | 1.831 | 4.245 |
| 58 | `falcon-3` | 1.875 | 4.505 | 1.830 | 4.321 | 1.754 | 4.566 |
| 59 | `llama-2` | 1.731 | 4.033 | 1.719 | 3.792 | 1.684 | 3.995 |
| 60 | `urdu-llama2-almanach` | 1.731 | 4.033 | 1.719 | 3.792 | 1.684 | 3.995 |
| 61 | `grok-1` | 1.666 | 4.841 | 1.636 | 4.623 | 1.580 | 4.858 |
| 62 | `claude-legacy` | 1.600 | 4.975 | 1.569 | 4.745 | 1.533 | 4.944 |
| 63 | `gpt-2` | 1.413 | 5.839 | 1.403 | 5.502 | 1.375 | 5.697 |
| 64 | `sarvam-1` | 1.002 | 7.685 | 1.002 | 7.206 | 1.003 | 7.377 |
| 65 | `byt5` | 1.000 | 7.698 | 1.000 | 7.225 | 1.000 | 7.400 |
| – | `hindko-probe-bpe32k` (LEAKY) | 7.844 | 1.108 | 7.382 | 1.113 | 6.763 | 1.238 |

## Per language_variety

- dev_strict varieties: hindko 751 docs, mixed 83 docs, no_signal 2 docs.
- dev_permissive (reporting only; it adds the Urdu-variety and other non-strict documents): hindko 832 docs, urdu 380 docs, mixed 107 docs, no_signal 32 docs, english 7 docs.

Bytes/token per variety (varieties with ≥ 10 documents). *perm. all* is the whole dev_permissive set.

| # | tokenizer | strict hindko | strict mixed | perm. hindko | perm. urdu | perm. mixed | perm. no_signal | perm. all |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `urdu-bert-64k` | 6.439 | 6.293 | 6.413 | 6.723 | 6.556 | 7.385 | 6.532 |
| 2 | `roberta-urdu` | 5.783 | 5.398 | 5.731 | 4.487 | 5.525 | 1.925 | 5.017 |
| 3 | `urdu-gpt2-20k` | 5.693 | 5.188 | 5.650 | 5.469 | 5.270 | 3.359 | 5.483 |
| 4 | `muril` | 5.621 | 5.346 | 5.590 | 5.192 | 5.597 | 3.332 | 5.366 |
| 5 | `indicbert-v2` | 5.474 | 5.284 | 5.456 | 5.377 | 5.493 | 3.845 | 5.377 |
| 6 | `sindhi-xlmr` | 5.246 | 5.159 | 5.223 | 5.348 | 5.338 | 3.757 | 5.218 |
| 7 | `bloom` | 5.230 | 4.738 | 5.198 | 5.029 | 4.767 | 3.595 | 5.078 |
| 8 | `xlm-r` | 4.950 | 4.855 | 4.925 | 4.790 | 5.038 | 2.707 | 4.790 |
| 9 | `nllb-200` | 4.734 | 4.821 | 4.727 | 4.577 | 4.860 | 2.720 | 4.600 |
| 10 | `gemma-4` | 4.649 | 4.349 | 4.626 | 4.661 | 4.331 | 3.217 | 4.585 |
| 11 | `gemma-3` | 4.649 | 4.349 | 4.626 | 4.661 | 4.331 | 3.217 | 4.585 |
| 12 | `sarvam-30b` | 4.649 | 4.349 | 4.626 | 4.661 | 4.331 | 3.217 | 4.585 |
| 13 | `falcon-h1` | 4.569 | 4.350 | 4.554 | 4.557 | 4.306 | 3.204 | 4.506 |
| 14 | `gpt-4o` | 4.391 | 4.174 | 4.376 | 4.362 | 4.114 | 2.959 | 4.318 |
| 15 | `gpt-oss` | 4.391 | 4.174 | 4.376 | 4.362 | 4.114 | 2.959 | 4.318 |
| 16 | `phi-4-mini` | 4.391 | 4.174 | 4.376 | 4.362 | 4.114 | 2.959 | 4.318 |
| 17 | `mbert` | 4.236 | 4.145 | 4.222 | 3.990 | 4.161 | 2.748 | 4.090 |
| 18 | `tiny-aya` | 4.154 | 4.105 | 4.144 | 4.047 | 4.037 | 2.572 | 4.053 |
| 19 | `mt5` | 3.957 | 4.060 | 3.951 | 3.898 | 4.103 | 2.621 | 3.891 |
| 20 | `mistral-nemo` | 3.946 | 3.828 | 3.935 | 4.110 | 3.826 | 3.078 | 3.965 |
| 21 | `mistral-small-4` | 3.946 | 3.828 | 3.935 | 4.110 | 3.826 | 3.078 | 3.965 |
| 22 | `apertus` | 3.946 | 3.828 | 3.935 | 4.110 | 3.826 | 3.078 | 3.965 |
| 23 | `sarvam-m` | 3.946 | 3.828 | 3.935 | 4.110 | 3.826 | 3.078 | 3.965 |
| 24 | `nemotron-3` | 3.946 | 3.828 | 3.935 | 4.110 | 3.826 | 3.078 | 3.965 |
| 25 | `gemma-2` | 3.836 | 3.756 | 3.826 | 3.875 | 3.740 | 2.860 | 3.812 |
| 26 | `llama-4` | 3.763 | 3.672 | 3.755 | 3.947 | 3.654 | 2.979 | 3.793 |
| 27 | `command-a-plus` | 3.732 | 3.635 | 3.728 | 3.947 | 3.628 | 3.188 | 3.783 |
| 28 | `qwen-3.5` | 3.633 | 3.561 | 3.625 | 3.636 | 3.569 | 2.988 | 3.611 |
| 29 | `qwen-3.8` | 3.633 | 3.561 | 3.625 | 3.636 | 3.569 | 2.988 | 3.611 |
| 30 | `kimi-k2` | 3.547 | 3.417 | 3.539 | 3.461 | 3.377 | 2.550 | 3.479 |
| 31 | `ernie-4.5` | 3.543 | 3.439 | 3.534 | 3.473 | 3.396 | 2.726 | 3.488 |
| 32 | `pashto-lfm2.5` | 3.533 | 3.470 | 3.529 | 3.730 | 3.449 | 2.952 | 3.577 |
| 33 | `deepseek-v3` | 3.422 | 3.304 | 3.413 | 3.311 | 3.304 | 2.865 | 3.362 |
| 34 | `deepseek-r1` | 3.422 | 3.304 | 3.413 | 3.311 | 3.304 | 2.865 | 3.362 |
| 35 | `deepseek-v4` | 3.422 | 3.304 | 3.413 | 3.311 | 3.304 | 2.865 | 3.362 |
| 36 | `deepseek-v4.1` | 3.422 | 3.304 | 3.413 | 3.311 | 3.304 | 2.865 | 3.362 |
| 37 | `glm-5` | 3.088 | 2.960 | 3.076 | 2.897 | 3.000 | 2.499 | 2.997 |
| 38 | `command-r` | 2.750 | 2.648 | 2.740 | 2.616 | 2.720 | 2.368 | 2.688 |
| 39 | `command-r7b` | 2.750 | 2.648 | 2.740 | 2.616 | 2.720 | 2.368 | 2.688 |
| 40 | `llama-3` | 2.739 | 2.645 | 2.730 | 2.702 | 2.712 | 2.729 | 2.722 |
| 41 | `urdu-llama3-almanach` | 2.739 | 2.645 | 2.730 | 2.702 | 2.712 | 2.729 | 2.722 |
| 42 | `urdu-llama3.2-custom` | 2.739 | 2.645 | 2.730 | 2.702 | 2.712 | 2.729 | 2.722 |
| 43 | `alif-1.0` | 2.739 | 2.645 | 2.730 | 2.702 | 2.712 | 2.729 | 2.722 |
| 44 | `qalb-1.0` | 2.739 | 2.645 | 2.730 | 2.702 | 2.712 | 2.729 | 2.722 |
| 45 | `qwen-3` | 2.686 | 2.626 | 2.681 | 2.514 | 2.666 | 2.276 | 2.613 |
| 46 | `qwen-2.5` | 2.686 | 2.626 | 2.681 | 2.514 | 2.666 | 2.276 | 2.613 |
| 47 | `glm-4.5` | 2.567 | 2.473 | 2.557 | 2.479 | 2.532 | 2.331 | 2.525 |
| 48 | `urdu-llama-bilal` | 2.381 | 2.438 | 2.379 | 2.608 | 2.466 | 2.251 | 2.452 |
| 49 | `minimax-m2` | 2.354 | 2.285 | 2.348 | 2.283 | 2.352 | 2.474 | 2.330 |
| 50 | `minimax-m3` | 2.354 | 2.285 | 2.348 | 2.283 | 2.352 | 2.474 | 2.330 |
| 51 | `eurollm` | 2.209 | 2.149 | 2.204 | 2.130 | 2.194 | 2.344 | 2.183 |
| 52 | `hunyuan` | 1.975 | 1.952 | 1.972 | 1.929 | 1.983 | 1.978 | 1.959 |
| 53 | `gpt-4` | 1.968 | 1.942 | 1.965 | 1.923 | 1.975 | 1.975 | 1.953 |
| 54 | `phi-4` | 1.968 | 1.942 | 1.965 | 1.923 | 1.975 | 1.975 | 1.953 |
| 55 | `olmo-2` | 1.968 | 1.942 | 1.965 | 1.923 | 1.975 | 1.975 | 1.953 |
| 56 | `granite-4.2` | 1.966 | 1.941 | 1.964 | 1.923 | 1.972 | 1.969 | 1.952 |
| 57 | `lfm2` | 1.963 | 1.887 | 1.959 | 1.899 | 1.947 | 2.126 | 1.943 |
| 58 | `falcon-3` | 1.846 | 1.808 | 1.843 | 1.827 | 1.823 | 1.948 | 1.841 |
| 59 | `llama-2` | 1.723 | 1.706 | 1.721 | 1.730 | 1.723 | 1.830 | 1.728 |
| 60 | `urdu-llama2-almanach` | 1.723 | 1.706 | 1.721 | 1.730 | 1.723 | 1.830 | 1.728 |
| 61 | `grok-1` | 1.647 | 1.590 | 1.645 | 1.580 | 1.603 | 1.658 | 1.623 |
| 62 | `claude-legacy` | 1.580 | 1.563 | 1.578 | 1.544 | 1.592 | 1.435 | 1.564 |
| 63 | `gpt-2` | 1.406 | 1.391 | 1.406 | 1.405 | 1.400 | 1.546 | 1.410 |
| 64 | `sarvam-1` | 1.002 | 0.999 | 1.002 | 1.006 | 1.000 | 1.009 | 1.005 |
| 65 | `byt5` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| – | `hindko-probe-bpe32k` (LEAKY) | 7.540 | 7.151 | 7.493 | 6.723 | 7.325 | 5.549 | 7.135 |

## Robustness under the four perturbations (dev_strict)

For each perturbation: the relative change in the total token count, and the share of the **affected** words whose segmentation changed. A word's segmentation is compared on the characters it shares with its perturbed form: which of them fall in the same token, which are split inside a character, and whether a token crosses into a neighbouring word. The perturbations (`eval/perturb.py`):

- **harakat:** delete every Arabic mark U+064B–U+065F and U+0670 (7,413 marks in 688 documents; 7,244 words affected).
- **digits:** swap ASCII ↔ Extended Arabic-Indic digits (1,664 digits, 669 words).
- **punct_space:** toggle the space before `۔` and `،` (7,970 edits, 7,810 words). The canonical corpus never has a space there, so on this text the probe *inserts* one. The PLAN wording is "delete the space", which would change nothing here.
- **zwnj:** insert ZWNJ at the element boundary of 281 Perso-Urdu compound words (119 types, e.g. `روزنامہ`, `زمیندار`, `خوشحال`) found by a fixed list of 3 prefixes and 12 head elements. The corpus has no ZWNJ at all.

| # | tokenizer | harakat Δtok | harakat seg-chg | digits Δtok | digits seg-chg | punct_space Δtok | punct_space seg-chg | zwnj Δtok | zwnj seg-chg |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `urdu-bert-64k` | 0.00% | 0.0% | 0.11% | 35.9% | 0.00% | 0.0% | 0.00% | 0.0% |
| 2 | `roberta-urdu` | -8.12% | 87.7% | 0.09% | 35.1% | 0.03% | 1.1% | 0.20% | 77.6% |
| 3 | `urdu-gpt2-20k` | -1.69% | 47.7% | 0.13% | 48.9% | 0.95% | 58.7% | 0.19% | 73.0% |
| 4 | `muril` | -2.15% | 58.7% | 0.36% | 80.9% | 0.00% | 0.0% | 0.00% | 0.0% |
| 5 | `indicbert-v2` | -1.94% | 71.5% | 0.19% | 56.2% | 0.06% | 2.1% | 0.14% | 45.6% |
| 6 | `sindhi-xlmr` | -1.11% | 37.2% | 0.14% | 42.6% | 0.69% | 41.7% | 0.04% | 36.7% |
| 7 | `bloom` | -2.00% | 57.3% | 0.56% | 94.9% | 0.04% | 1.3% | 0.12% | 32.7% |
| 8 | `xlm-r` | -4.09% | 78.0% | 0.13% | 42.0% | 0.63% | 37.7% | 0.05% | 35.9% |
| 9 | `nllb-200` | -4.06% | 72.0% | 0.18% | 65.6% | 0.40% | 17.2% | 0.01% | 48.0% |
| 10 | `gemma-4` | -2.60% | 64.7% | -0.26% | 36.8% | 0.48% | 30.7% | 0.09% | 23.8% |
| 11 | `gemma-3` | -2.60% | 64.7% | -0.26% | 36.8% | 0.48% | 30.7% | 0.09% | 23.8% |
| 12 | `sarvam-30b` | -2.60% | 64.7% | -0.26% | 36.8% | 0.48% | 30.7% | 0.09% | 23.8% |
| 13 | `falcon-h1` | -1.97% | 59.5% | 0.00% | 0.0% | 2.54% | 1.1% | 0.02% | 51.2% |
| 14 | `gpt-4o` | -2.32% | 62.1% | 0.15% | 65.3% | 0.47% | 1.1% | 0.07% | 28.5% |
| 15 | `gpt-oss` | -2.32% | 62.1% | 0.15% | 65.3% | 0.47% | 1.1% | 0.07% | 28.5% |
| 16 | `phi-4-mini` | -2.32% | 62.1% | 0.15% | 65.3% | 0.47% | 1.1% | 0.07% | 28.5% |
| 17 | `mbert` | -3.15% | 52.9% | 0.10% | 48.7% | 0.00% | 0.0% | 0.00% | 0.0% |
| 18 | `tiny-aya` | -3.43% | 66.0% | 0.00% | 0.0% | 0.00% | 0.1% | 0.08% | 26.7% |
| 19 | `mt5` | -2.97% | 76.2% | 0.13% | 49.0% | 1.49% | 45.1% | 0.02% | 32.0% |
| 20 | `mistral-nemo` | -2.26% | 54.5% | 0.00% | 0.0% | 0.41% | 0.1% | 0.06% | 38.4% |
| 21 | `mistral-small-4` | -2.26% | 54.5% | 0.00% | 0.0% | 0.41% | 0.1% | 0.06% | 38.4% |
| 22 | `apertus` | -2.26% | 54.5% | 0.00% | 0.0% | 0.41% | 0.1% | 0.06% | 38.4% |
| 23 | `sarvam-m` | -2.26% | 54.5% | 0.00% | 0.0% | 0.41% | 0.1% | 0.06% | 38.4% |
| 24 | `nemotron-3` | -2.26% | 54.5% | 0.00% | 0.0% | 0.41% | 0.1% | 0.06% | 38.4% |
| 25 | `gemma-2` | -2.70% | 52.0% | -0.24% | 49.5% | 0.04% | 1.9% | 0.07% | 23.8% |
| 26 | `llama-4` | -2.48% | 56.4% | 0.10% | 52.5% | 1.40% | 0.1% | 0.05% | 38.8% |
| 27 | `command-a-plus` | -2.00% | 53.0% | 0.19% | 80.7% | 0.39% | 0.1% | 0.06% | 30.6% |
| 28 | `qwen-3.5` | -1.91% | 49.0% | 0.40% | 100.0% | 0.00% | 0.1% | 0.07% | 17.4% |
| 29 | `qwen-3.8` | -1.91% | 49.0% | 0.40% | 100.0% | 0.00% | 0.1% | 0.07% | 17.4% |
| 30 | `kimi-k2` | -1.93% | 39.5% | 0.15% | 75.0% | 0.37% | 0.1% | 0.06% | 23.1% |
| 31 | `ernie-4.5` | -2.38% | 43.8% | -0.18% | 26.0% | 0.26% | 13.7% | 0.07% | 6.4% |
| 32 | `pashto-lfm2.5` | -2.10% | 49.6% | 0.19% | 80.7% | 0.36% | 0.1% | 0.05% | 24.2% |
| 33 | `deepseek-v3` | -2.18% | 57.1% | 0.11% | 63.8% | 1.27% | 0.1% | 0.13% | 21.0% |
| 34 | `deepseek-r1` | -2.18% | 57.1% | 0.11% | 63.8% | 1.27% | 0.1% | 0.13% | 21.0% |
| 35 | `deepseek-v4` | -2.18% | 57.1% | 0.11% | 63.8% | 1.27% | 0.1% | 0.13% | 21.0% |
| 36 | `deepseek-v4.1` | -2.18% | 57.1% | 0.11% | 63.8% | 1.27% | 0.1% | 0.13% | 21.0% |
| 37 | `glm-5` | -2.23% | 50.8% | 0.15% | 80.7% | 1.15% | 67.7% | 0.06% | 27.0% |
| 38 | `command-r` | -2.23% | 67.7% | 0.00% | 0.0% | 1.02% | 0.1% | 0.05% | 23.5% |
| 39 | `command-r7b` | -2.23% | 67.7% | 0.00% | 0.0% | 1.02% | 0.1% | 0.05% | 23.5% |
| 40 | `llama-3` | -1.63% | 56.9% | 0.05% | 37.1% | 1.02% | 0.1% | 0.04% | 35.9% |
| 41 | `urdu-llama3-almanach` | -1.63% | 56.9% | 0.05% | 37.1% | 1.02% | 0.1% | 0.04% | 35.9% |
| 42 | `urdu-llama3.2-custom` | -1.63% | 56.9% | 0.05% | 37.1% | 1.02% | 0.1% | 0.04% | 35.9% |
| 43 | `alif-1.0` | -1.63% | 56.9% | 0.05% | 37.1% | 1.02% | 0.1% | 0.04% | 35.9% |
| 44 | `qalb-1.0` | -1.63% | 56.9% | 0.05% | 37.1% | 1.02% | 0.1% | 0.04% | 35.9% |
| 45 | `qwen-3` | -1.62% | 15.9% | 0.29% | 100.0% | 1.00% | 0.1% | 0.05% | 16.0% |
| 46 | `qwen-2.5` | -1.62% | 15.9% | 0.29% | 100.0% | 1.00% | 0.1% | 0.05% | 16.0% |
| 47 | `glm-4.5` | -1.89% | 50.8% | 0.12% | 80.7% | 0.00% | 0.1% | 0.05% | 27.0% |
| 48 | `urdu-llama-bilal` | -1.34% | 40.9% | 0.04% | 37.1% | 0.89% | 0.1% | 0.04% | 9.3% |
| 49 | `minimax-m2` | -1.57% | 50.7% | 0.38% | 100.0% | 0.88% | 0.1% | 0.04% | 18.1% |
| 50 | `minimax-m3` | -1.57% | 50.7% | 0.38% | 100.0% | 0.88% | 0.1% | 0.04% | 18.1% |
| 51 | `eurollm` | -1.37% | 40.8% | 0.24% | 100.0% | 0.82% | 0.1% | 0.05% | 12.8% |
| 52 | `hunyuan` | -1.09% | 3.2% | 0.22% | 100.0% | 1.09% | 32.5% | 0.04% | 14.9% |
| 53 | `gpt-4` | -1.09% | 3.2% | 0.32% | 100.0% | 1.08% | 32.5% | 0.04% | 14.9% |
| 54 | `phi-4` | -1.09% | 3.2% | 0.32% | 100.0% | 1.08% | 32.5% | 0.04% | 14.9% |
| 55 | `olmo-2` | -1.09% | 3.2% | 0.32% | 100.0% | 1.08% | 32.5% | 0.04% | 14.9% |
| 56 | `granite-4.2` | -1.08% | 3.2% | 0.32% | 100.0% | 1.08% | 32.5% | 0.04% | 14.9% |
| 57 | `lfm2` | -1.34% | 41.2% | 0.32% | 100.0% | 0.73% | 0.0% | 0.08% | 10.3% |
| 58 | `falcon-3` | -1.12% | 31.4% | 0.17% | 100.0% | 1.02% | 0.0% | 0.07% | 2.5% |
| 59 | `llama-2` | -0.92% | 0.2% | 0.19% | 100.0% | 0.95% | 0.0% | 0.03% | 0.0% |
| 60 | `urdu-llama2-almanach` | -0.92% | 0.2% | 0.19% | 100.0% | 0.95% | 0.0% | 0.03% | 0.0% |
| 61 | `grok-1` | -1.57% | 13.7% | 0.18% | 100.0% | 0.91% | 0.0% | 0.10% | 1.1% |
| 62 | `claude-legacy` | -1.72% | 12.3% | 0.33% | 100.0% | 0.59% | 0.0% | 0.06% | 0.0% |
| 63 | `gpt-2` | -1.18% | 17.0% | 0.29% | 100.0% | 0.52% | 0.0% | 0.05% | 1.8% |
| 64 | `sarvam-1` | -1.03% | 0.0% | 0.11% | 100.0% | 0.55% | 0.0% | 0.06% | 0.0% |
| 65 | `byt5` | -1.02% | 0.0% | 0.11% | 100.0% | 0.55% | 0.0% | 0.06% | 0.0% |
| – | `hindko-probe-bpe32k` (LEAKY) | -0.32% | 8.2% | 0.10% | 40.2% | 2.55% | 65.3% | 0.13% | 84.0% |

Byte-level and byte-fallback tokenizers show 100% *digits* seg-chg when an ASCII digit (1 byte) becomes an Urdu digit (2 bytes) that they split into 2 byte tokens. That is a real change of segmentation, not an error.

## Report-only metrics (dev_strict)

Rényi efficiency rated the worse GPT-2 regex higher in our pilot (PLAN §4.2); it is never used to decide. Morphology is on the SILVER sets (518 high-confidence and 236 low-confidence words, `morphology/morph_eval.py`). It is report-only, because MorphScore does not predict LM quality (Arnett et al. 2025).

| # | tokenizer | Rényi α=2.5 | Rényi α=2 | vocab used | utilisation | morph F1 (high) | MorphScore (high) | stem kept (high) | tok/word (high) | docs/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `urdu-bert-64k` | 0.443 | 0.475 | 11,403 | 17.82% | 0.399 | 0.440 | 0.313 | 1.85 | 629 |
| 2 | `roberta-urdu` | 0.442 | 0.474 | 12,733 | 24.49% | 0.461 | 0.570 | 0.363 | 1.83 | 448 |
| 3 | `urdu-gpt2-20k` | 0.508 | 0.544 | 10,115 | 50.58% | 0.492 | 0.593 | 0.409 | 2.05 | 901 |
| 4 | `muril` | 0.404 | 0.431 | 7,340 | 3.72% | 0.530 | 0.602 | 0.452 | 2.09 | 551 |
| 5 | `indicbert-v2` | 0.397 | 0.423 | 6,215 | 2.49% | 0.560 | 0.597 | 0.488 | 2.13 | 898 |
| 6 | `sindhi-xlmr` | 0.377 | 0.405 | 6,557 | 2.48% | 0.641 | 0.721 | 0.564 | 2.22 | 349 |
| 7 | `bloom` | 0.413 | 0.437 | 6,740 | 2.69% | 0.467 | 0.553 | 0.400 | 2.20 | 405 |
| 8 | `xlm-r` | 0.371 | 0.399 | 5,501 | 2.20% | 0.660 | 0.747 | 0.577 | 2.24 | 376 |
| 9 | `nllb-200` | 0.388 | 0.415 | 4,945 | 1.93% | 0.432 | 0.538 | 0.373 | 2.39 | 572 |
| 10 | `gemma-4` | 0.403 | 0.425 | 4,240 | 1.62% | 0.417 | 0.514 | 0.344 | 2.49 | 714 |
| 11 | `gemma-3` | 0.403 | 0.425 | 4,240 | 1.62% | 0.417 | 0.514 | 0.344 | 2.49 | 770 |
| 12 | `sarvam-30b` | 0.403 | 0.425 | 4,240 | 1.62% | 0.417 | 0.514 | 0.344 | 2.49 | 571 |
| 13 | `falcon-h1` | 0.370 | 0.398 | 5,208 | 1.99% | 0.448 | 0.538 | 0.376 | 2.52 | 296 |
| 14 | `gpt-4o` | 0.387 | 0.415 | 3,666 | 1.83% | 0.469 | 0.586 | 0.402 | 2.59 | 456 |
| 15 | `gpt-oss` | 0.387 | 0.415 | 3,666 | 1.83% | 0.469 | 0.586 | 0.402 | 2.59 | 443 |
| 16 | `phi-4-mini` | 0.387 | 0.415 | 3,666 | 1.83% | 0.469 | 0.586 | 0.402 | 2.59 | 362 |
| 17 | `mbert` | 0.423 | 0.441 | 2,487 | 2.08% | 0.455 | 0.595 | 0.295 | 2.74 | 506 |
| 18 | `tiny-aya` | 0.376 | 0.400 | 3,682 | 1.41% | 0.395 | 0.498 | 0.301 | 2.63 | 313 |
| 19 | `mt5` | 0.263 | 0.302 | 4,055 | 1.62% | 0.555 | 0.651 | 0.390 | 2.47 | 1340 |
| 20 | `mistral-nemo` | 0.346 | 0.382 | 3,202 | 2.44% | 0.446 | 0.606 | 0.319 | 2.90 | 450 |
| 21 | `mistral-small-4` | 0.346 | 0.382 | 3,202 | 2.44% | 0.446 | 0.606 | 0.319 | 2.90 | 454 |
| 22 | `apertus` | 0.346 | 0.382 | 3,202 | 2.44% | 0.446 | 0.606 | 0.319 | 2.90 | 362 |
| 23 | `sarvam-m` | 0.346 | 0.382 | 3,202 | 2.44% | 0.446 | 0.606 | 0.319 | 2.90 | 372 |
| 24 | `nemotron-3` | 0.346 | 0.382 | 3,202 | 2.44% | 0.446 | 0.606 | 0.319 | 2.90 | 284 |
| 25 | `gemma-2` | 0.344 | 0.371 | 2,997 | 1.17% | 0.368 | 0.514 | 0.226 | 3.02 | 753 |
| 26 | `llama-4` | 0.335 | 0.367 | 2,691 | 1.34% | 0.437 | 0.612 | 0.284 | 3.01 | 446 |
| 27 | `command-a-plus` | 0.330 | 0.360 | 2,676 | 1.05% | 0.427 | 0.610 | 0.270 | 3.06 | 393 |
| 28 | `qwen-3.5` | 0.322 | 0.350 | 2,882 | 1.16% | 0.367 | 0.567 | 0.212 | 3.28 | 381 |
| 29 | `qwen-3.8` | 0.322 | 0.350 | 2,882 | 1.16% | 0.367 | 0.567 | 0.212 | 3.28 | 381 |
| 30 | `kimi-k2` | 0.392 | 0.408 | 1,217 | 0.74% | 0.460 | 0.669 | 0.187 | 3.16 | 376 |
| 31 | `ernie-4.5` | 0.406 | 0.425 | 1,197 | 1.18% | 0.426 | 0.630 | 0.178 | 3.15 | 755 |
| 32 | `pashto-lfm2.5` | 0.354 | 0.383 | 2,290 | 1.83% | 0.407 | 0.603 | 0.243 | 3.22 | 757 |
| 33 | `deepseek-v3` | 0.350 | 0.374 | 2,030 | 1.58% | 0.403 | 0.648 | 0.154 | 3.43 | 389 |
| 34 | `deepseek-r1` | 0.350 | 0.374 | 2,030 | 1.58% | 0.403 | 0.648 | 0.154 | 3.43 | 386 |
| 35 | `deepseek-v4` | 0.350 | 0.374 | 2,030 | 1.57% | 0.403 | 0.648 | 0.154 | 3.43 | 381 |
| 36 | `deepseek-v4.1` | 0.350 | 0.374 | 2,030 | 1.57% | 0.403 | 0.648 | 0.154 | 3.43 | 356 |
| 37 | `glm-5` | 0.322 | 0.345 | 1,582 | 1.02% | 0.355 | 0.623 | 0.104 | 3.78 | 387 |
| 38 | `command-r` | 0.253 | 0.278 | 2,860 | 1.12% | 0.267 | 0.584 | 0.172 | 4.52 | 312 |
| 39 | `command-r7b` | 0.253 | 0.278 | 2,860 | 1.12% | 0.267 | 0.584 | 0.172 | 4.52 | 327 |
| 40 | `llama-3` | 0.270 | 0.297 | 2,424 | 1.89% | 0.265 | 0.568 | 0.160 | 4.55 | 448 |
| 41 | `urdu-llama3-almanach` | 0.270 | 0.297 | 2,424 | 1.89% | 0.265 | 0.568 | 0.160 | 4.55 | 433 |
| 42 | `urdu-llama3.2-custom` | 0.267 | 0.294 | 2,424 | 1.66% | 0.265 | 0.568 | 0.160 | 4.55 | 538 |
| 43 | `alif-1.0` | 0.270 | 0.297 | 2,424 | 1.89% | 0.265 | 0.568 | 0.160 | 4.55 | 658 |
| 44 | `qalb-1.0` | 0.270 | 0.297 | 2,424 | 1.89% | 0.265 | 0.568 | 0.160 | 4.55 | 657 |
| 45 | `qwen-3` | 0.288 | 0.306 | 1,376 | 0.91% | 0.369 | 0.691 | 0.029 | 4.15 | 398 |
| 46 | `qwen-2.5` | 0.288 | 0.306 | 1,376 | 0.91% | 0.369 | 0.691 | 0.029 | 4.15 | 420 |
| 47 | `glm-4.5` | 0.281 | 0.304 | 1,569 | 1.04% | 0.272 | 0.623 | 0.104 | 4.85 | 367 |
| 48 | `urdu-llama-bilal` | 0.222 | 0.250 | 1,663 | 1.28% | 0.383 | 0.776 | 0.332 | 4.24 | 379 |
| 49 | `minimax-m2` | 0.260 | 0.279 | 1,964 | 0.98% | 0.308 | 0.733 | 0.081 | 5.18 | 328 |
| 50 | `minimax-m3` | 0.260 | 0.279 | 1,964 | 0.98% | 0.308 | 0.733 | 0.081 | 5.18 | 320 |
| 51 | `eurollm` | 0.269 | 0.287 | 1,231 | 0.96% | 0.332 | 0.791 | 0.075 | 5.19 | 397 |
| 52 | `hunyuan` | 0.283 | 0.297 | 377 | 0.29% | 0.322 | 0.878 | 0.000 | 5.99 | 469 |
| 53 | `gpt-4` | 0.290 | 0.304 | 545 | 0.54% | 0.322 | 0.878 | 0.000 | 5.99 | 610 |
| 54 | `phi-4` | 0.290 | 0.304 | 545 | 0.54% | 0.322 | 0.878 | 0.000 | 5.99 | 360 |
| 55 | `olmo-2` | 0.290 | 0.304 | 545 | 0.54% | 0.322 | 0.878 | 0.000 | 5.99 | 403 |
| 56 | `granite-4.2` | 0.290 | 0.304 | 515 | 0.51% | 0.322 | 0.878 | 0.000 | 5.99 | 421 |
| 57 | `lfm2` | 0.280 | 0.296 | 884 | 1.37% | 0.309 | 0.845 | 0.035 | 5.95 | 601 |
| 58 | `falcon-3` | 0.287 | 0.299 | 342 | 0.26% | 0.312 | 0.917 | 0.000 | 6.41 | 265 |
| 59 | `llama-2` | 0.240 | 0.258 | 343 | 1.07% | 0.405 | 1.000 | 0.000 | 5.68 | 689 |
| 60 | `urdu-llama2-almanach` | 0.240 | 0.258 | 343 | 1.07% | 0.405 | 1.000 | 0.000 | 5.68 | 731 |
| 61 | `grok-1` | 0.270 | 0.281 | 329 | 0.25% | 0.279 | 0.826 | 0.000 | 6.56 | 434 |
| 62 | `claude-legacy` | 0.279 | 0.289 | 581 | 0.89% | 0.321 | 0.969 | 0.000 | 6.69 | 283 |
| 63 | `gpt-2` | 0.274 | 0.287 | 560 | 1.11% | 0.281 | 0.985 | 0.000 | 7.81 | 598 |
| 64 | `sarvam-1` | 0.211 | 0.223 | 312 | 0.46% | 0.217 | 1.000 | 0.000 | 10.17 | 312 |
| 65 | `byt5` | 0.395 | 0.418 | 139 | 36.20% | 0.217 | 1.000 | 0.000 | 10.17 | 5442 |
| – | `hindko-probe-bpe32k` (LEAKY) | 0.482 | 0.528 | 14,824 | 46.33% | 0.029 | 0.778 | 0.014 | 1.02 | 839 |

*stem kept* = `stem_boundary_respected`: the word's stem|suffix boundary is a token boundary and the stem is not split. *docs/s* is plain `encode()` on this shared laptop CPU, one thread; it is a sanity check only.

## Partial-UTF-8 tokens in the vocabularies (R2, byte-level tokenizers)

| tokenizer | partial-UTF-8 tokens in vocab | occurrences on dev_strict |
|---|---:|---:|
| `roberta-urdu` | 220 | 20 |
| `bloom` | 1,336 | 120 |
| `falcon-h1` | 3,967 | 111 |
| `gpt-4o` | 1,434 | 256 |
| `gpt-oss` | 1,434 | 256 |
| `phi-4-mini` | 1,434 | 256 |
| `tiny-aya` | 9,300 | 198 |
| `mistral-nemo` | 1,307 | 168 |
| `mistral-small-4` | 1,307 | 168 |
| `apertus` | 1,307 | 168 |
| `sarvam-m` | 1,307 | 168 |
| `nemotron-3` | 1,307 | 168 |
| `llama-4` | 1,700 | 3,702 |
| `command-a-plus` | 6,019 | 131 |
| `qwen-3.5` | 816 | 93 |
| `qwen-3.8` | 816 | 93 |
| `kimi-k2` | 1,044 | 161 |
| `pashto-lfm2.5` | 1,428 | 456 |
| `deepseek-v3` | 1,342 | 856 |
| `deepseek-r1` | 1,342 | 856 |
| `deepseek-v4` | 1,342 | 856 |
| `deepseek-v4.1` | 1,342 | 856 |
| `glm-5` | 967 | 8,218 |
| `command-r` | 2,956 | 460 |
| `command-r7b` | 2,956 | 460 |
| `llama-3` | 1,224 | 1,391 |
| `urdu-llama3-almanach` | 1,224 | 1,391 |
| `urdu-llama3.2-custom` | 1,224 | 1,391 |
| `alif-1.0` | 1,224 | 1,391 |
| `qalb-1.0` | 1,224 | 1,391 |
| `qwen-3` | 1,320 | 381 |
| `qwen-2.5` | 1,320 | 381 |
| `glm-4.5` | 949 | 8,218 |
| `urdu-llama-bilal` | 1,224 | 788 |
| `minimax-m2` | 931 | 128 |
| `minimax-m3` | 931 | 128 |
| `hunyuan` | 835 | 10,238 |
| `gpt-4` | 645 | 10,238 |
| `phi-4` | 645 | 10,238 |
| `olmo-2` | 645 | 10,238 |
| `granite-4.2` | 645 | 10,238 |
| `lfm2` | 663 | 456 |
| `falcon-3` | 981 | 44,616 |
| `claude-legacy` | 625 | 60,623 |
| `gpt-2` | 216 | 112,610 |

## What this page does not cover

- **No test-split number.** The test split is evaluated once, in Stage 5.
- **G2–G5 and R1 were not run on the baselines.** They are health gates for the project's own *candidates* (PLAN §4.3). The harness runs them with `--gates all`; for external tokenizers, trained on other data, they would say nothing about our choice.
- **NSL against A1-P1-16k** waits for that tokenizer (Stage 1).
- **Bits-per-byte** needs LMs (Stage 3). Every column here is intrinsic.
- The *transformers* baselines (`muril`, `roberta-urdu`) take their character offsets from the same tokenizer object's `return_offsets_mapping`. The harness checks that those ids equal the baseline's own `encode()`; mismatching documents fall back to string alignment (`roberta-urdu`: 0 mismatching docs, `muril`: 0 mismatching docs).

## Reproduce

```
set PYTHONIOENCODING=utf-8
python eval\run_baselines.py --shard 0 --of 3   (and --shard 1, --shard 2, in parallel; ~10 min each)
python eval\report_baselines.py
```

Harness code sha256 (harness.py / adapters.py / perturb.py / morph_eval.py) for every result on this page: adapters.py `bc377b786f…`, harness.py `31db30dd0e…`, morph_eval.py `73e32d88f9…`, perturb.py `f965f79f7a…`.
