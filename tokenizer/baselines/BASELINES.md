# Competitor tokenizer baselines for the Hindko tokenizer

Generated 2026-09-26T10:16:50Z by `scripts/build_manifest.py` (numbers in this file are copied from `manifest.json`).

- **66 working baselines** out of 66 registered. **45 are recommended for the benchmark.** The other 21 are redundant, flagged `recommended_for_benchmark: false`: 20 encode all 8 samples and all 2,619 round-trip battery texts token-for-token like an earlier entry, with a token set that differs from it by at most 1%, and `urdu-llama3.2-custom` is marked redundant by a documented override. See *Baselines whose encodings match*.
- **55 of 66 are lossless on Hindko**: `decode(encode(x)) == x` for all 8 samples *and* for every text of the round-trip battery (2,619 real corpus texts, 2,832,596 characters, 24,468 line breaks; see *Samples, battery and definitions*). The other 11 are listed in *Tokenizers that are not lossless on Hindko*.
- Providers covered (36): AI4Bharat, Ai2, Alibaba Qwen, Anthropic, Baidu, BigScience, Cohere, DeepSeek, Google, IBM, Inria ALMAnaCH, Liquid AI, Meta, Microsoft, MiniMax, Mistral AI, Moonshot AI, NVIDIA, OpenAI, Qalb (enstazao), Sarvam AI, Swiss AI, TII, Tencent, Traversaal.ai, UTTER (EU), UrduHack, Zhipu AI / Z.ai, community (BilalKhan1), community (Kashif786), community (SabahNawab), community (aariciah), community (farahadeeba), community (nassimjp), this project, xAI.
- Files: tokenizer files only (allow-list), downloaded anonymously, revision pinned to the commit sha recorded in the manifest. No model weights, no login, no remote code executed.
- `hindko-probe-bpe32k` is the project's earlier SentencePiece probe. It was trained on the whole corpus, including the test split, so it is a **leaky reference and not a fair competitor**.

**Revision (2026-09-26, after review).** The first version tested losslessness on 5 single-line paragraphs only. None of them contained a line break, an Arabic presentation form or U+2026, so it labelled `mt5`, `xlm-r`, `sindhi-xlmr`, `claude-legacy`, `urdu-gpt2-20k`, `hindko-probe-bpe32k` lossless. Measured now: `mt5` NOT lossless (815/2,619 battery texts exact); `xlm-r` NOT lossless (760/2,619 battery texts exact); `claude-legacy` NOT lossless (2,414/2,619 battery texts exact); `urdu-gpt2-20k` NOT lossless (658/2,619 battery texts exact); `sindhi-xlmr` NOT lossless (759/2,619 battery texts exact); `hindko-probe-bpe32k` NOT lossless (815/2,619 battery texts exact). Three stress samples and the corpus battery were added, the redundancy rule now also compares battery encodings, and every tokenizer's line-break handling is recorded.

## Usage

```python
import sys; sys.path.insert(0, r"F:\Hindko\_tokenizer\baselines")
from load_baselines import load_baseline, list_baselines
names = list_baselines(recommended_only=True)   # de-duplicated set; list_baselines() gives all working ones
lossless = list_baselines(lossless_only=True)   # exact round trip on the samples and the whole battery
tok = load_baseline("gemma-4")
ids = tok.encode(text)      # list[int], no BOS/EOS/CLS/SEP added
text2 = tok.decode(ids)     # str
tok.vocab_size
```

`load_baseline` works offline from `files/`. It reads `tokenizer.json` with the `tokenizers` library, not with `transformers.AutoTokenizer`, because transformers 5.3 does not reproduce some repos' tokenizers. See *Verification* below.

## Samples, battery and definitions

**8 samples** (`samples_hindko.json`) from `F:\Hindko\hindko_dataset.jsonl` (strict subset), chosen deterministically. The 5 *core* slots each take the paragraph (a line of 200–900 chars) with the smallest `sha256(uid#paragraph_index)` that satisfies the slot. The 3 *stress* slots each have their own required predicate: a whole multi-line record (600–3,000 chars, at least 4 line breaks including a blank line, smallest `sha256(uid#doc)`), a paragraph containing U+FDFA `ﷺ`, and a paragraph containing U+2026 `…`.

| group | slot | uid | source | chars | words | line breaks |
|---|---|---|---|---:|---:|---:|
| core | newspaper + ASCII digits | `161867f30ebdfd6f` | newspaper | 585 | 125 | 0 |
| core | newspaper + harakat | `210e67d9e320ecad` | newspaper | 391 | 81 | 0 |
| core | book + harakat | `8632f3e1efad8344` | book | 305 | 66 | 0 |
| core | book + Extended Arabic-Indic digits | `6e457c476fdcbadf` | book | 239 | 50 | 0 |
| core | web | `b441b7b6c6e92b87` | web | 414 | 97 | 0 |
| stress | multi-line document (whole record, >=4 line breaks incl. a blank line) | `34957d6d50d55fb2` | book | 1429 | 307 | 4 |
| stress | paragraph with presentation form U+FDFA | `cd25dd1aea340609` | book | 353 | 69 | 0 |
| stress | paragraph with U+2026 horizontal ellipsis | `09a12910b952b177` | book | 254 | 52 | 0 |

**Round-trip battery** (`manifest.json → roundtrip_battery`): a) every validation+test document of the evaluation split (splits/split_manifest.jsonl, permissive tier, which contains the strict tier), whole text with line breaks; b) character coverage: for every distinct code point of the permissive corpus except U+0020 and LF (lines are split on LF; both occur throughout part a), the up-to-3 lines (<= 2,000 chars) with the smallest sha256(uid#line_index) containing it, or a 600-char window of the smallest-key longer line if it occurs only in longer lines.

| part | texts | characters | line breaks |
|---|---:|---:|---:|
| char-coverage | 351 | 83,615 | 0 |
| eval:test | 910 | 1,337,808 | 12,093 |
| eval:validation | 1,358 | 1,411,173 | 12,375 |
| **total** | 2,619 | 2,832,596 | 24,468 |

The coverage lines contain 265 distinct code points: every code point of the permissive corpus except the space and LF, which occur throughout the eval documents. Occurrences in the battery of characters that some tokenizers lose: U+000A LINE FEED ×24,468, U+2026 HORIZONTAL ELLIPSIS ×3,607, U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM ×66, U+FDF2 ARABIC LIGATURE ALLAH ISOLATED FORM ×4, U+2018 LEFT SINGLE QUOTATION MARK ×3,664, U+2019 RIGHT SINGLE QUOTATION MARK ×4,271, U+0601 ARABIC SIGN SANAH ×32.

- **Round trip:** x = NFC(text); exact iff decode(encode(x)) == x, with no special tokens added and none skipped.
- **Lossless:** lossless = exact round trip on all 8 samples (5 core paragraphs + 3 stress samples) AND on every text of the round-trip battery.
- **UNK rate:** unk_rate: UNK tokens / all tokens over the 5 core samples; unk_rate_battery: the same over the round-trip battery. UNK ids = the tokenizer model's unk token (byte-level BPE and bytes have none).
- **Line breaks:** decode(encode(w1 + LF + w2)) and decode(encode(w1 + LF + LF + w2)) for two Hindko words; the only whitespace in the corpus is U+0020 and LF. roundtrip_battery.eval_docs_tokens.line_break_token_share = share of a tokenizer's tokens on the eval documents that disappears when LF is replaced by a space.
- **tok/word** below is total tokens ÷ whitespace words over the 5 core paragraphs (419 words, no line breaks). It is a quick sanity check, **not the benchmark**; ranking the competitors needs the held-out split and the full metric suite.

## Baselines

Lossless = exact round trip on all 8 samples and all 2,619 battery texts. Line breaks = what `decode(encode(x))` makes of LF. UNK rate is over the battery. "Same as" = redundant with the named, earlier baseline.

| name | provider — tokenizer | year | source repo | vocab | algorithm | byte fallback | licence (repo) | lossless | line breaks | UNK rate (battery) | tok/word (core) | same as |
|---|---|---:|---|---:|---|---|---|:---:|---|---:|---:|---|
| `indicbert-v2` | AI4Bharat — IndicBERT v2 WordPiece 250k | 2022 | `ai4bharat/IndicBERTv2-MLM-only` | 250,000 | WordPiece | no | mit | **no** | → 1 space per run | 0.00026 | 1.51 |  |
| `olmo-2` | Ai2 — OLMo 2/3 (cl100k-derived) 100k | 2024 | `allenai/OLMo-2-1124-7B` | 100,278 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 4.25 | gpt-4 |
| `qwen-3` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 2025 | `Qwen/Qwen3-0.6B` | 151,669 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 3.16 |  |
| `qwen-2.5` | Alibaba Qwen — Qwen2/3 byte-level BPE 151k | 2024 | `Qwen/Qwen2.5-0.5B` | 151,665 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 3.16 | qwen-3 |
| `ernie-4.5` | Baidu — ERNIE 4.5 SentencePiece 103k | 2025 | `baidu/ERNIE-4.5-0.3B-PT` | 101,304 | BPE (SentencePiece-style, byte fallback) | yes | apache-2.0 | yes | kept | 0.00000 | 2.26 |  |
| `bloom` | BigScience — BLOOM byte-level BPE 250k | 2022 | `bigscience/bloom-560m` | 250,680 | byte-level BPE | byte-level | bigscience-bloom-rail-1.0 | yes | kept | 0.00000 | 1.56 |  |
| `command-r` | Cohere — Command-R / Aya Expanse byte-level BPE 256k | 2024 | `adamo1139/aya-expanse-8b-ungated` — byte-identical mirror of `CohereLabs/aya-expanse-8b` | 255,029 | byte-level BPE | byte-level | cc-by-nc-4.0 | yes | kept | 0.00000 | 3.06 |  |
| `command-r7b` | Cohere — Command R7B byte-level BPE 256k | 2024 | `mlx-community/c4ai-command-r7b-12-2024-bf16` — byte-identical mirror of `CohereLabs/c4ai-command-r7b-12-2024` | 255,033 | byte-level BPE | byte-level | cc-by-nc-4.0 | yes | kept | 0.00000 | 3.06 | command-r |
| `deepseek-r1` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 2025 | `deepseek-ai/DeepSeek-R1` | 128,815 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 2.44 | deepseek-v3 |
| `deepseek-v3` | DeepSeek — DeepSeek-V3 byte-level BPE 128k | 2024 | `deepseek-ai/DeepSeek-V3` | 128,815 | byte-level BPE | byte-level | None | yes | kept | 0.00000 | 2.44 |  |
| `gemma-3` | Google — Gemma 3 SentencePiece 262k | 2025 | `unsloth/gemma-3-1b-it` — byte-identical mirror of `google/gemma-3-1b-it` | 262,145 | BPE (SentencePiece-style, byte fallback) | yes | gemma | yes | kept | 0.00000 | 1.74 | gemma-4 |
| `gemma-2` | Google — Gemma 2 SentencePiece 256k | 2024 | `unsloth/gemma-2-2b-it` — byte-identical mirror of `google/gemma-2-2b-it` | 256,000 | BPE (SentencePiece-style, byte fallback) | yes | gemma | yes | kept | 0.00000 | 2.13 |  |
| `byt5` | Google — ByT5 raw UTF-8 bytes | 2021 | `google/byt5-small` | 384 | bytes (UTF-8, no merges) | n/a: bytes | apache-2.0 | yes | kept | 0.00000 | 8.22 |  |
| `muril` | Google — MuRIL WordPiece 197k | 2021 | `google/muril-base-cased` | 197,258 | WordPiece | no | apache-2.0 | **no** | → 1 space per run | 0.00091 | 1.50 |  |
| `mt5` | Google — mT5 SentencePiece Unigram 250k | 2020 | `google/mt5-small` | 250,100 | SentencePiece Unigram + byte fallback | yes | apache-2.0 | **no** | → 1 space per run | 0.00000 | 2.12 |  |
| `mbert` | Google — mBERT WordPiece 119k (cased) | 2018 | `google-bert/bert-base-multilingual-cased` | 119,547 | WordPiece | no | apache-2.0 | **no** | → 1 space per run | 0.01100 | 1.94 |  |
| `lfm2` | Liquid AI — LFM2/LFM2.5 byte-level BPE 65k | 2025 | `LiquidAI/LFM2.5-1.2B-Instruct` | 64,402 | byte-level BPE | byte-level | other:lfm1.0 | yes | kept | 0.00000 | 4.30 |  |
| `llama-4` | Meta — Llama 4 BPE 202k | 2025 | `unsloth/Llama-4-Scout-17B-16E-Instruct` — byte-identical mirror of `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 201,135 | byte-level BPE | byte-level | other:llama4 | yes | kept | 0.00000 | 2.18 |  |
| `llama-3` | Meta — Llama 3 tiktoken-BPE 128k | 2024 | `NousResearch/Meta-Llama-3.1-8B-Instruct` — byte-identical mirror of `meta-llama/Llama-3.1-8B-Instruct` | 128,256 | byte-level BPE | byte-level | llama3.1 | yes | kept | 0.00000 | 3.06 |  |
| `nllb-200` | Meta — NLLB-200 SentencePiece BPE 256k | 2022 | `facebook/nllb-200-distilled-600M` | 256,204 | BPE (character-level, no byte fallback) | no | cc-by-nc-4.0 | **no** | → 1 space per run | 0.00485 | 1.75 |  |
| `xlm-r` | Meta — XLM-R SentencePiece Unigram 250k | 2019 | `FacebookAI/xlm-roberta-base` | 250,002 | Unigram (SentencePiece) | no | mit | **no** | → 1 space per run | 0.00054 | 1.69 |  |
| `phi-4-mini` | Microsoft — Phi-4-mini (o200k-derived) 200k | 2025 | `microsoft/Phi-4-mini-instruct` | 200,029 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 1.87 | gpt-4o |
| `phi-4` | Microsoft — Phi-4 (cl100k-derived) 100k | 2024 | `microsoft/phi-4` | 100,352 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 4.25 | gpt-4 |
| `minimax-m2` | MiniMax — MiniMax BPE 200k | 2025 | `MiniMaxAI/MiniMax-M2` | 200,054 | byte-level BPE | byte-level | other:modified-mit | yes | kept | 0.00000 | 3.60 |  |
| `mistral-nemo` | Mistral AI — Tekken (tiktoken-style BPE, 131k) | 2024 | `mistralai/Mistral-Nemo-Instruct-2407` | 131,072 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.12 |  |
| `kimi-k2` | Moonshot AI — Kimi tiktoken BPE 160k | 2025 | `moonshotai/Kimi-K2-Instruct` | 163,840 | byte-level BPE | byte-level | other:modified-mit | yes | kept | 0.00000 | 2.27 |  |
| `nemotron-3` | NVIDIA — Nemotron 3 BPE 131k | 2025 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 131,072 | byte-level BPE | byte-level | other:nvidia-nemotron-open-model-license | yes | kept | 0.00000 | 2.12 | mistral-nemo |
| `gpt-oss` | OpenAI — o200k_harmony | 2025 | `openai/gpt-oss-20b` | 200,019 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 1.87 | gpt-4o |
| `gpt-4o` | OpenAI — o200k_base (tiktoken) | 2024 | `Xenova/gpt-4o` — community conversion of `tiktoken o200k_base (openaipublic)` | 200,000 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 1.87 |  |
| `gpt-4` | OpenAI — cl100k_base (tiktoken) | 2023 | `Xenova/gpt-4` — community conversion of `tiktoken cl100k_base (openaipublic)` | 100,263 | byte-level BPE | byte-level | None | yes | kept | 0.00000 | 4.25 |  |
| `sarvam-m` | Sarvam AI — Sarvam-M (Mistral Small 3.1 Tekken) | 2025 | `sarvamai/sarvam-m` | 131,072 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.12 | mistral-nemo |
| `sarvam-1` | Sarvam AI — Sarvam-1 SentencePiece 68k (Indic) | 2024 | `sarvamai/sarvam-1` | 68,096 | BPE (SentencePiece-style, byte fallback) | yes | None | yes | kept | 0.00000 | 8.21 |  |
| `apertus` | Swiss AI — Apertus BPE 131k | 2025 | `swiss-ai/Apertus-8B-Instruct-2509` | 131,072 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.12 | mistral-nemo |
| `falcon-h1` | TII — Falcon-H1 BPE 261k (34B) | 2025 | `tiiuae/Falcon-H1-34B-Instruct` | 261,120 | byte-level BPE | byte-level | other:falcon-llm-license | yes | kept | 0.00000 | 1.84 |  |
| `falcon-3` | TII — Falcon3 BPE 131k | 2024 | `tiiuae/Falcon3-7B-Instruct` | 131,072 | byte-level BPE | byte-level | other:falcon-llm-license | yes | kept | 0.00000 | 4.50 |  |
| `hunyuan` | Tencent — Hunyuan byte-level BPE 128k | 2025 | `tencent/Hunyuan-7B-Instruct` | 128,166 | byte-level BPE | byte-level | None | yes | kept | 0.00000 | 4.22 |  |
| `eurollm` | UTTER (EU) — EuroLLM SentencePiece BPE 128k | 2024 | `utter-project/EuroLLM-9B-Instruct-2512` | 128,000 | BPE (SentencePiece-style, byte fallback) | yes | apache-2.0 | yes | kept | 0.00000 | 3.84 |  |
| `glm-4.5` | Zhipu AI / Z.ai — GLM-4.5 BPE 151k | 2025 | `zai-org/GLM-4.5` | 151,365 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 3.25 |  |
| `grok-1` | xAI — Grok-1 SentencePiece 131k | 2024 | `Xenova/grok-1-tokenizer` — community conversion of `xai-org/grok-1 (tokenizer.model)` | 131,072 | BPE (SentencePiece-style, byte fallback) | yes | apache-2.0 | yes | kept | 0.00000 | 5.03 |  |
| `qwen-3.5` | Alibaba Qwen — Qwen3.5 byte-level BPE 248k | 2026 | `Qwen/Qwen3.5-0.8B` | 248,070 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.29 |  |
| `qwen-3.8` | Alibaba Qwen — Qwen3.8 byte-level BPE 248k | 2026 | `Qwen/Qwen3.8-27B` | 248,077 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.29 | qwen-3.5 |
| `command-a-plus` | Cohere — Command A+ (05-2026) byte-level BPE 255k | 2026 | `CohereLabs/command-a-plus-05-2026-bf16` | 255,032 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.21 |  |
| `tiny-aya` | Cohere — Tiny Aya byte-level BPE 261k | 2026 | `mlx-community/tiny-aya-global-8bit-mlx` — mirror (bytes differ) of `CohereLabs/tiny-aya-global` | 261,010 | byte-level BPE | byte-level | cc-by-nc-4.0 | yes | kept | 0.00000 | 1.98 |  |
| `deepseek-v4` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 2026 | `deepseek-ai/DeepSeek-V4-Flash` | 129,280 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 2.44 | deepseek-v3 |
| `deepseek-v4.1` | DeepSeek — DeepSeek-V4 byte-level BPE 129k | 2026 | `deepseek-ai/DeepSeek-V4.1-Flash` | 129,280 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 2.44 | deepseek-v3 |
| `gemma-4` | Google — Gemma 4 SentencePiece 262k | 2026 | `google/gemma-4-E2B-it` | 262,144 | BPE (SentencePiece-style, byte fallback) | yes | apache-2.0 | yes | kept | 0.00000 | 1.74 |  |
| `granite-4.2` | IBM — Granite 4.x (cl100k-derived) 100k | 2026 | `ibm-granite/granite-4.2-8b` | 100,352 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 4.25 |  |
| `minimax-m3` | MiniMax — MiniMax-M3 byte-level BPE 200k | 2026 | `MiniMaxAI/MiniMax-M3` | 200,061 | byte-level BPE | byte-level | other:minimax-community | yes | kept | 0.00000 | 3.60 | minimax-m2 |
| `mistral-small-4` | Mistral AI — Tekken (tekken.json v15, 131k) | 2026 | `mistralai/Mistral-Small-4-119B-2603` | 131,072 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.12 | mistral-nemo |
| `sarvam-30b` | Sarvam AI — Sarvam 2026 tokenizer 262k (Gemma-style SentencePiece BPE) | 2026 | `sarvamai/sarvam-30b` | 262,144 | BPE (SentencePiece-style, byte fallback) | yes | apache-2.0 | yes | kept | 0.00000 | 1.74 |  |
| `glm-5` | Zhipu AI / Z.ai — GLM-5 byte-level BPE 155k | 2026 | `zai-org/GLM-5` | 154,856 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 2.69 |  |
| `urdu-llama2-almanach` | Inria ALMAnaCH — Llama-2-7B-mono-Urdu (Urdu-adapted vocab) | 2025 | `almanach/Llama-2-7B-mono-Urdu` | 32,000 | BPE (SentencePiece-style, byte fallback) | yes | None | yes | kept | 0.00000 | 4.84 | llama-2 |
| `urdu-llama3-almanach` | Inria ALMAnaCH — Llama-3-8B-mono-Urdu (Urdu-adapted vocab) | 2025 | `almanach/Llama-3-8B-mono-Urdu` | 128,256 | byte-level BPE | byte-level | None | yes | kept | 0.00000 | 3.06 | llama-3 |
| `qalb-1.0` | Qalb (enstazao) — Qalb-1.0-8B (Llama 3.1 tokenizer) | 2026 | `enstazao/Qalb-1.0-8B-Instruct` | 128,256 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 3.06 | llama-3 |
| `alif-1.0` | Traversaal.ai — Alif-1.0-8B (Llama 3.1 tokenizer) | 2025 | `large-traversaal/Alif-1.0-8B-Instruct` | 128,256 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 3.06 | llama-3 |
| `roberta-urdu` | UrduHack — RoBERTa-Urdu byte-level BPE 52k (monolingual Urdu) | 2020 | `urduhack/roberta-urdu-small` | 52,000 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 1.33 |  |
| `urdu-llama-bilal` | community (BilalKhan1) — Llama + Urdu tokenizer | 2024 | `BilalKhan1/llama-urdu-tokenizer` | 129,435 | byte-level BPE | byte-level | None | yes | kept | 0.00000 | 3.51 |  |
| `urdu-llama3.2-custom` | community (SabahNawab) — Llama-3.2-3B + custom Urdu tokens | 2025 | `SabahNawab/Meta_Llama_3.2_3B_Urdu_Custom_Tokenizer` | 145,613 | byte-level BPE | byte-level | None | yes | kept | 0.00000 | 3.06 | llama-3 |
| `urdu-gpt2-20k` | community (aariciah) — Urdu GPT-2 BPE 20k (Urdu-only vocab, NFKC+lowercase) | 2026 | `aariciah/gpt2-urdu-20k-lc` | 20,000 | BPE (character-level, no byte fallback) | no | None | **no** | → UNK | 0.02780 | 1.37 |  |
| `urdu-bert-64k` | community (farahadeeba) — Urdu BERT WordPiece 64k | 2026 | `farahadeeba/urdu-bert-64k` | 64,000 | WordPiece | no | apache-2.0 | **no** | → 1 space per run | 0.00042 | 1.31 |  |
| `sindhi-xlmr` | community (Kashif786) — XLM-R + Sindhi vocabulary extension (Unigram 265k) | 2026 | `Kashif786/xlm-roberta-base-sindhi-extended` | 264,635 | Unigram (SentencePiece) | no | None | **no** | → 1 space per run | 0.00067 | 1.63 |  |
| `pashto-lfm2.5` | community (nassimjp) — LFM2.5 + Pashto vocabulary extension | 2026 | `nassimjp/LFM2.5-2.6B-Pashto-Zi` | 125,039 | byte-level BPE | byte-level | apache-2.0 | yes | kept | 0.00000 | 2.33 |  |
| `claude-legacy` | Anthropic — Claude 1/2 BPE 65k (legacy, public) | 2023 | `Xenova/claude-tokenizer` — community conversion of `anthropic-tokenizer-typescript claude.json` | 65,000 | byte-level BPE | byte-level | mit | **no** | kept | 0.00000 | 5.28 |  |
| `llama-2` | Meta — Llama 2 SentencePiece BPE 32k | 2023 | `NousResearch/Llama-2-7b-hf` — mirror (not byte-verified) of `meta-llama/Llama-2-7b-hf` | 32,000 | BPE (SentencePiece-style, byte fallback) | yes | None | yes | kept | 0.00000 | 4.84 |  |
| `gpt-2` | OpenAI — GPT-2 byte-level BPE | 2019 | `openai-community/gpt2` | 50,257 | byte-level BPE | byte-level | mit | yes | kept | 0.00000 | 5.93 |  |
| `hindko-probe-bpe32k` | this project — SentencePiece BPE 32k trained on the whole Hindko corpus | 2026 | local file (project probe) | 32,000 | SentencePiece BPE + byte fallback | yes | project-internal (not released) | **no** | → 1 space per run | 0.00000 | 1.19 |  |

Tiers: rows are grouped as core (the requested list), frontier-2026 (newest releases found on the hub on 2026-09-26), urdu (Urdu-specific), regional (Perso-Arabic neighbours with vocabulary extensions: Pashto, Sindhi), and reference (legacy or leaky).

### Tokenizers that are not lossless on Hindko (8 samples + 2,619-text battery)

Tested on the 8 samples (5 core + 3 stress) and on every battery text (2,268 eval-split documents plus the character-coverage lines). Characters lost are summed over the failing battery texts (multiset difference input − output, so a character that is changed into another counts as lost). A tokenizer can fail in several ways at once.

| name | samples exact | battery: eval docs exact | battery: coverage lines exact | line breaks (in → out) | UNK tokens (battery) | characters lost (battery, top 5) | first difference (expected → got) |
|---|---:|---:|---:|---|---:|---|---|
| `mt5` | 5/8 | 493/2,268 | 322/351 | → 1 space per run (24,468 → 0) | 0 | U+000A LINE FEED ×24,468; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM ×66; U+FFFD REPLACEMENT CHARACTER ×5; U+FDF2 ARABIC LIGATURE ALLAH ISOLATED FORM ×4 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `mbert` | 0/8 | 11/2,268 | 64/351 | → 1 space per run (24,468 → 0) | 13,355 | U+000A LINE FEED ×24,468; U+2019 RIGHT SINGLE QUOTATION MARK ×4,271; U+2018 LEFT SINGLE QUOTATION MARK ×3,664; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+0627 ARABIC LETTER ALEF ×792 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `muril` | 0/8 | 11/2,268 | 68/351 | → 1 space per run (24,468 → 0) | 842 | U+000A LINE FEED ×24,468; U+0627 ARABIC LETTER ALEF ×340; U+06CC ARABIC LETTER FARSI YEH ×256; U+0648 ARABIC LETTER WAW ×213; U+0768 ARABIC LETTER NOON WITH SMALL TAH ×208 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `xlm-r` | 5/8 | 478/2,268 | 282/351 | → 1 space per run (24,468 → 0) | 556 | U+000A LINE FEED ×24,468; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+0768 ARABIC LETTER NOON WITH SMALL TAH ×208; U+0759 ARABIC LETTER DAL WITH TWO DOTS VERTICALLY BELOW AND SMALL TAH ×74; U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM ×66 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `nllb-200` | 4/8 | 412/2,268 | 250/351 | → 1 space per run (24,468 → 0) | 5,210 | U+000A LINE FEED ×24,468; U+2019 RIGHT SINGLE QUOTATION MARK ×4,271; U+2018 LEFT SINGLE QUOTATION MARK ×3,664; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+0768 ARABIC LETTER NOON WITH SMALL TAH ×208 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `indicbert-v2` | 0/8 | 0/2,268 | 12/351 | → 1 space per run (24,468 → 0) | 246 | U+000A LINE FEED ×24,468; U+0627 ARABIC LETTER ALEF ×120; U+06CC ARABIC LETTER FARSI YEH ×88; U+08BF ARABIC LETTER TEH WITH SMALL V ×65; U+0648 ARABIC LETTER WAW ×65 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گند ##ھار #` (U+000A → U+0020) |
| `claude-legacy` | 6/8 | 2,089/2,268 | 325/351 | kept (24,468 → 24,468) | 0 | U+2026 HORIZONTAL ELLIPSIS ×3,607; U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM ×66; U+FDF2 ARABIC LIGATURE ALLAH ISOLATED FORM ×4; U+0678 ARABIC LETTER HIGH HAMZA YEH ×3; U+2033 DOUBLE PRIME ×1 | `ور روضۂ نبویﷺ سے آپ کو ہ` → `ور روضۂ نبویصلى الله علي` (U+FDFA → U+0635) |
| `urdu-bert-64k` | 0/8 | 2/2,268 | 20/351 | → 1 space per run (24,468 → 0) | 326 | U+000A LINE FEED ×24,468; U+0626 ARABIC LETTER YEH WITH HAMZA ABOVE ×22,969; U+064E ARABIC FATHA ×18,086; U+064F ARABIC DAMMA ×17,360; U+0622 ARABIC LETTER ALEF WITH MADDA ABOVE ×15,997 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `urdu-gpt2-20k` | 5/8 | 459/2,268 | 199/351 | → UNK (24,468 → 0) | 25,055 | U+000A LINE FEED ×24,468; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+0768 ARABIC LETTER NOON WITH SMALL TAH ×208; U+0054 LATIN CAPITAL LETTER T ×151; U+0043 LATIN CAPITAL LETTER C ×147 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران<unk>گندھارا` (U+000A → U+003C) |
| `sindhi-xlmr` | 5/8 | 477/2,268 | 282/351 | → 1 space per run (24,468 → 0) | 641 | U+000A LINE FEED ×24,468; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+0768 ARABIC LETTER NOON WITH SMALL TAH ×208; U+0759 ARABIC LETTER DAL WITH TWO DOTS VERTICALLY BELOW AND SMALL TAH ×74; U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM ×66 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |
| `hindko-probe-bpe32k` | 5/8 | 493/2,268 | 322/351 | → 1 space per run (24,468 → 0) | 0 | U+000A LINE FEED ×24,468; U+2026 HORIZONTAL ELLIPSIS ×3,607; U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM ×66; U+FFFD REPLACEMENT CHARACTER ×5; U+FDF2 ARABIC LIGATURE ALLAH ISOLATED FORM ×4 | `محترم ممبران⏎گندھارا ہند` → `محترم ممبران گندھارا ہند` (U+000A → U+0020) |

Failure flags per tokenizer (number of battery texts): `mt5`: line breaks lost 1,770, non-whitespace characters changed 208; `mbert`: line breaks lost 1,770, non-whitespace characters changed 1,051, UNK tokens 1,039; `muril`: line breaks lost 1,770, non-whitespace characters changed 236, UNK tokens 213; `xlm-r`: line breaks lost 1,770, non-whitespace characters changed 301, UNK tokens 94; `nllb-200`: line breaks lost 1,770, non-whitespace characters changed 790, UNK tokens 665; `indicbert-v2`: non-whitespace characters changed 2,594, line breaks lost 1,770, UNK tokens 56; `claude-legacy`: non-whitespace characters changed 205; `urdu-bert-64k`: non-whitespace characters changed 2,543, line breaks lost 1,770, UNK tokens 73; `urdu-gpt2-20k`: non-whitespace characters changed 1,961, UNK tokens 1,826, line breaks lost 1,770; `sindhi-xlmr`: line breaks lost 1,770, non-whitespace characters changed 303, UNK tokens 111; `hindko-probe-bpe32k`: line breaks lost 1,770, non-whitespace characters changed 208.

### Line breaks and token counts

The corpus is multi-line: 24,468 line breaks in the battery. 10 tokenizers do not keep them: `mt5`, `mbert`, `muril`, `xlm-r`, `nllb-200`, `indicbert-v2`, `urdu-bert-64k`, `sindhi-xlmr`, `hindko-probe-bpe32k` (each run of line breaks becomes one space); `urdu-gpt2-20k` (line breaks become UNK tokens). The other 56 keep every line break in the probes and in the battery.

Dropping line breaks is a free saving in any token count, because a tokenizer that turns LF into a space effectively encodes the text with every LF replaced by a space. For the 56 tokenizers that keep line breaks, the original eval documents cost 0.00–2.97% more tokens than that space-joined text (median 1.32%; share = (tokens(text) − tokens(text with LF → space)) / tokens(text); `byt5` gets 0 because LF and a space are each one byte). The benchmark must therefore either compare token counts on single-line segments (split on LF, as the 5 core samples are), or count on the original text and report each tokenizer's `newline_handling` next to its numbers. Per-tokenizer values are in `manifest.json → baselines[].roundtrip_battery.eval_docs_tokens`.

### Baselines whose encodings match an earlier baseline

These encode all 8 samples and all 2,619 battery texts (2,832,596 characters) exactly like an earlier baseline. The vocabularies were also compared directly. Columns: tokens only in this entry / only in the other entry / shared tokens with the same id.

| name | duplicate of | only here | only in other | shared, same id / shared |
|---|---|---:|---:|---|
| `gpt-oss` | `gpt-4o` | 19 | 0 | 200,000 / 200,000 |
| `gemma-3` | `gemma-4` | 20 | 19 | 255,938 / 262,125 |
| `qwen-2.5` | `qwen-3` | 0 | 4 | 151,665 / 151,665 |
| `qwen-3.8` | `qwen-3.5` | 7 | 0 | 248,070 / 248,070 |
| `deepseek-r1` | `deepseek-v3` | 2 | 2 | 128,813 / 128,813 |
| `deepseek-v4` | `deepseek-v3` | 465 | 0 | 128,815 / 128,815 |
| `deepseek-v4.1` | `deepseek-v3` | 466 | 1 | 128,814 / 128,814 |
| `mistral-small-4` | `mistral-nemo` | 14 | 14 | 131,054 / 131,058 |
| `command-r7b` | `command-r` | 13 | 9 | 255,020 / 255,020 |
| `phi-4` | `gpt-4` | 89 | 0 | 100,263 / 100,263 |
| `phi-4-mini` | `gpt-4o` | 29 | 0 | 200,000 / 200,000 |
| `apertus` | `mistral-nemo` | 60 | 60 | 131,011 / 131,012 |
| `sarvam-m` | `mistral-nemo` | 6 | 6 | 131,062 / 131,066 |
| `minimax-m3` | `minimax-m2` | 9 | 2 | 200,052 / 200,052 |
| `nemotron-3` | `mistral-nemo` | 8 | 8 | 131,064 / 131,064 |
| `olmo-2` | `gpt-4` | 15 | 0 | 100,263 / 100,263 |
| `urdu-llama3-almanach` | `llama-3` | 3 | 3 | 128,007 / 128,253 |
| `urdu-llama2-almanach` | `llama-2` | 0 | 0 | 32,000 / 32,000 |
| `urdu-llama3.2-custom` | `llama-3` | 17,357 | 0 | 128,256 / 128,256 |
| `alif-1.0` | `llama-3` | 0 | 0 | 128,256 / 128,256 |
| `qalb-1.0` | `llama-3` | 0 | 0 | 128,256 / 128,256 |

Small differences are usually special or reserved tokens. Rule: an entry is marked redundant (`recommended_for_benchmark: false`) only if its sample and battery encodings match *and* its token set differs from the other entry's by at most 1% of its vocabulary. `urdu-llama3.2-custom` is marked redundant by hand: its extra tokens are added tokens written in the byte-level alphabet and can never match raw text, so it always encodes like llama-3. The benchmark can score one representative per redundant group, but should name every member, since each is a different model family.

## Pipelines (normalizer / pre-tokenizer)

| name | normalizer | pre-tokenizer (regexes truncated) | decoder |
|---|---|---|---|
| `alif-1.0` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `apertus` |  | Split(regex='[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `bloom` |  | Split(regex=' ?[^(\\s\|[.,!?…。，、।۔،])]+', behavior=Isolated) + ByteLevel(add_prefix_space=False, use_regex=False) | ByteLevel(add_prefix_space=True, use_regex=False) |
| `byt5` |  |  | bytes |
| `claude-legacy` | NFKC | ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `command-a-plus` |  | Split(regex='\\d{1,3}(?=(?:\\d{3})*\\b)', behavior=Isolated) + Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `command-r` | NFC | Digits(individual=True) + ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `command-r7b` | NFC | Digits(individual=True) + ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `deepseek-r1` |  | Split(regex='\\p{N}{1,3}', behavior=Isolated) + Split(regex='[一-龥\u3040-ゟ゠-ヿ]+', behavior=Isolated) + Split(regex='[!"#$%&\'()*+,\\-./:;<=>?@\\[\\\\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `deepseek-v3` |  | Split(regex='\\p{N}{1,3}', behavior=Isolated) + Split(regex='[一-龥\u3040-ゟ゠-ヿ]+', behavior=Isolated) + Split(regex='[!"#$%&\'()*+,\\-./:;<=>?@\\[\\\\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `deepseek-v4` |  | Split(regex='\\p{N}{1,3}', behavior=Isolated) + Split(regex='[一-龥\u3040-ゟ゠-ヿ]+', behavior=Isolated) + Split(regex='[!"#$%&\'()*+,\\-./:;<=>?@\\[\\\\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `deepseek-v4.1` |  | Split(regex='\\p{N}{1,3}', behavior=Isolated) + Split(regex='[一-龥\u3040-ゟ゠-ヿ]+', behavior=Isolated) + Split(regex='[!"#$%&\'()*+,\\-./:;<=>?@\\[\\\\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `ernie-4.5` | Replace(' '->'▁') |  | Replace('▁'->' ') + ByteFallback + Fuse |
| `eurollm` | Prepend('▁') + Replace(' '->'▁') |  | Replace('▁'->' ') + ByteFallback + Fuse + Strip(left=None, … |
| `falcon-3` |  | Punctuation + ByteLevel(add_prefix_space=False, use_regex=True) + Digits(individual=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `falcon-h1` |  | Punctuation + Split(regex='[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `gemma-2` | Replace(' '->'▁') | Split(regex=' ', behavior=MergedWithPrevious) | Replace('▁'->' ') + ByteFallback + Fuse |
| `gemma-3` | Replace(' '->'▁') | Split(regex=' ', behavior=MergedWithPrevious) | Replace('▁'->' ') + ByteFallback + Fuse |
| `gemma-4` | Replace(' '->'▁') | Split(regex=' ', behavior=MergedWithPrevious) | Replace('▁'->' ') + ByteFallback + Fuse |
| `glm-4.5` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `glm-5` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `gpt-2` |  | ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `gpt-4` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `gpt-4o` |  | Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `gpt-oss` |  | Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `granite-4.2` |  | ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `grok-1` | Prepend('▁') + Replace(' '->'▁') |  | Replace('▁'->' ') + ByteFallback + Fuse + Strip(left=None, … |
| `hindko-probe-bpe32k` | spm normalizer_spec.name='nmt_nfkc' add_dummy_prefix=True remove_extra_whitespaces=True | spm split_by_whitespace=True split_digits=False split_by_unicode_script=True max_sentencepiece_length=16 | sentencepiece |
| `hunyuan` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", behavi… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `indicbert-v2` |  | Whitespace |  |
| `kimi-k2` |  | Split(regex="[\\p{Han}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s\|'t… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `lfm2` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `llama-2` | Prepend('▁') + Replace(' '->'▁') |  | Replace('▁'->' ') + ByteFallback + Fuse + Strip(left=None, … |
| `llama-3` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `llama-4` |  | Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `mbert` | BertNormalizer(clean_text=True, chinese_chars=True, strip_accents=None, lowercase=False) | BertPreTokenizer | WordPiece |
| `minimax-m2` | NFC | Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `minimax-m3` | NFC | Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `mistral-nemo` |  | Split(regex='[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `mistral-small-4` |  | Split(regex='[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `mt5` | spm normalizer_spec.name='nmt_nfkc' add_dummy_prefix=True remove_extra_whitespaces=True | spm split_by_whitespace=True split_digits=False split_by_unicode_script=True max_sentencepiece_length=16 | sentencepiece |
| `muril` | BertNormalizer(clean_text=True, chinese_chars=True, strip_accents=False, lowercase=False) | BertPreTokenizer | WordPiece |
| `nemotron-3` |  | Split(regex='[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `nllb-200` | Precompiled(SentencePiece charsmap) + Replace(' {2,}'->' ') | Metaspace(replacement='▁', prepend_scheme=always, split=True) | Metaspace(replacement='▁', prepend_scheme=always, split=Tru… |
| `olmo-2` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `pashto-lfm2.5` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `phi-4` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `phi-4-mini` |  | Split(regex="[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `qalb-1.0` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `qwen-2.5` | NFC | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", behavi… | ByteLevel(add_prefix_space=False, use_regex=False) |
| `qwen-3` | NFC | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", behavi… | ByteLevel(add_prefix_space=False, use_regex=False) |
| `qwen-3.5` | NFC | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+\|\\p{N}\| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)… | ByteLevel(add_prefix_space=False, use_regex=False) |
| `qwen-3.8` | NFC | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+\|\\p{N}\| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)… | ByteLevel(add_prefix_space=False, use_regex=False) |
| `roberta-urdu` |  | ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `sarvam-1` |  | Metaspace(replacement='▁', prepend_scheme=first, split=False) | Replace('▁'->' ') + ByteFallback + Fuse + Strip(left=None, … |
| `sarvam-30b` | Replace(' '->'▁') | Split(regex=' ', behavior=MergedWithPrevious) | Replace('▁'->' ') + ByteFallback + Fuse |
| `sarvam-m` |  | Split(regex='[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `sindhi-xlmr` | Precompiled(SentencePiece charsmap) | WhitespaceSplit + Metaspace(replacement='▁', prepend_scheme=always, split=True) | Metaspace(replacement='▁', prepend_scheme=always, split=Tru… |
| `tiny-aya` | NFC | Digits(individual=True) + ByteLevel(add_prefix_space=False, use_regex=True) | ByteLevel(add_prefix_space=True, use_regex=True) |
| `urdu-bert-64k` | BertNormalizer(clean_text=True, chinese_chars=True, strip_accents=None, lowercase=True) | BertPreTokenizer | WordPiece |
| `urdu-gpt2-20k` | NFKC + Lowercase | Metaspace(replacement='▁', prepend_scheme=always, split=True) | Metaspace(replacement='▁', prepend_scheme=always, split=Tru… |
| `urdu-llama-bilal` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `urdu-llama2-almanach` | Prepend('▁') + Replace(' '->'▁') |  | Replace('▁'->' ') + ByteFallback + Fuse + Strip(left=None, … |
| `urdu-llama3-almanach` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `urdu-llama3.2-custom` |  | Split(regex="(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+", b… | ByteLevel(add_prefix_space=True, use_regex=True) |
| `xlm-r` | Precompiled(SentencePiece charsmap) | WhitespaceSplit + Metaspace(replacement='▁', prepend_scheme=always, split=True) | Metaspace(replacement='▁', prepend_scheme=always, split=Tru… |

Full pipeline strings, added-token counts, Arabic-script vocabulary counts, file sha256s and pinned revisions are in `manifest.json`.

### Arabic-script vocabulary

Count of vocabulary entries containing at least one Arabic-script code point (U+0600–06FF, 0750–077F, 08A0–08FF, FB50–FDFF, FE70–FEFF). For byte-level BPE, tokens are decoded from the byte alphabet first, and partial UTF-8 sequences are ignored.

| name | vocab | Arabic-script tokens | of which ≥2 Arabic letters |
|---|---:|---:|---:|
| `urdu-bert-64k` | 64,000 | 52,388 | 51,562 |
| `roberta-urdu` | 52,000 | 49,719 | 49,379 |
| `hindko-probe-bpe32k` | 32,000 | 30,852 | 30,644 |
| `sindhi-xlmr` | 264,635 | 29,177 | 28,861 |
| `falcon-h1` | 261,120 | 26,319 | 26,090 |
| `urdu-llama3.2-custom` | 145,613 | 21,158 | 20,897 |
| `bloom` | 250,680 | 20,854 | 20,569 |
| `urdu-gpt2-20k` | 20,000 | 17,981 | 17,512 |
| `muril` | 197,258 | 16,397 | 15,742 |
| `tiny-aya` | 261,010 | 15,584 | 15,392 |
| `xlm-r` | 250,002 | 14,644 | 14,377 |
| `nllb-200` | 256,204 | 14,175 | 13,870 |
| `indicbert-v2` | 250,000 | 9,952 | 8,925 |
| `mistral-nemo` | 131,072 | 9,447 | 9,259 |
| `mistral-small-4` | 131,072 | 9,447 | 9,259 |
| `apertus` | 131,072 | 9,447 | 9,259 |
| `sarvam-m` | 131,072 | 9,447 | 9,259 |
| `nemotron-3` | 131,072 | 9,447 | 9,259 |
| `qwen-3.5` | 248,070 | 8,826 | 8,686 |
| `qwen-3.8` | 248,077 | 8,826 | 8,686 |
| `gemma-4` | 262,144 | 8,460 | 8,044 |
| `gemma-3` | 262,145 | 8,460 | 8,044 |
| `sarvam-30b` | 262,144 | 8,460 | 8,044 |
| `pashto-lfm2.5` | 125,039 | 8,286 | 8,119 |
| `gpt-4o` | 200,000 | 8,055 | 7,831 |
| `gpt-oss` | 200,019 | 8,055 | 7,831 |
| `phi-4-mini` | 200,029 | 8,055 | 7,831 |
| `command-a-plus` | 255,032 | 7,613 | 7,460 |
| `mt5` | 250,100 | 7,459 | 7,116 |
| `command-r` | 255,029 | 6,627 | 6,508 |
| `command-r7b` | 255,033 | 6,627 | 6,508 |
| `gemma-2` | 256,000 | 6,274 | 5,866 |
| `minimax-m2` | 200,054 | 6,129 | 6,015 |
| `minimax-m3` | 200,061 | 6,129 | 6,015 |
| `urdu-llama-bilal` | 129,435 | 4,989 | 4,804 |
| `mbert` | 119,547 | 4,873 | 4,649 |
| `llama-4` | 201,135 | 4,490 | 4,289 |
| `qwen-3` | 151,669 | 4,018 | 3,449 |
| `qwen-2.5` | 151,665 | 4,018 | 3,449 |
| `llama-3` | 128,256 | 3,810 | 3,634 |
| `urdu-llama3-almanach` | 128,256 | 3,810 | 3,634 |
| `alif-1.0` | 128,256 | 3,810 | 3,634 |
| `qalb-1.0` | 128,256 | 3,810 | 3,634 |
| `deepseek-v3` | 128,815 | 3,193 | 3,055 |
| `deepseek-r1` | 128,815 | 3,193 | 3,055 |
| `deepseek-v4` | 129,280 | 3,193 | 3,055 |
| `deepseek-v4.1` | 129,280 | 3,193 | 3,055 |
| `eurollm` | 128,000 | 2,539 | 2,431 |
| `glm-5` | 154,856 | 2,195 | 1,880 |
| `glm-4.5` | 151,365 | 2,004 | 1,880 |
| `ernie-4.5` | 101,304 | 1,548 | 1,357 |
| `kimi-k2` | 163,840 | 1,091 | 943 |
| `lfm2` | 64,402 | 781 | 699 |
| `hunyuan` | 128,166 | 118 | 36 |
| `gpt-4` | 100,263 | 113 | 36 |
| `phi-4` | 100,352 | 113 | 36 |
| `olmo-2` | 100,278 | 113 | 36 |
| `granite-4.2` | 100,352 | 113 | 36 |
| `falcon-3` | 131,072 | 86 | 23 |
| `grok-1` | 131,072 | 69 | 33 |
| `llama-2` | 32,000 | 55 | 2 |
| `urdu-llama2-almanach` | 32,000 | 55 | 2 |
| `claude-legacy` | 65,000 | 34 | 3 |
| `gpt-2` | 50,257 | 22 | 2 |
| `sarvam-1` | 68,096 | 0 | 0 |

## Verification

**Mirrors.** For gated official repos, the hub still exposes file metadata (git blob id and LFS sha256) without login. The mirrors marked *byte-identical* have the same blob id or LFS sha256 as the official file:

- `llama-3`: `NousResearch/Meta-Llama-3.1-8B-Instruct` = `meta-llama/Llama-3.1-8B-Instruct` (official gated: manual).
- `llama-4`: `unsloth/Llama-4-Scout-17B-16E-Instruct` = `meta-llama/Llama-4-Scout-17B-16E-Instruct` (official gated: manual).
- `gemma-3`: `unsloth/gemma-3-1b-it` = `google/gemma-3-1b-it` (official gated: manual).
- `gemma-2`: `unsloth/gemma-2-2b-it` = `google/gemma-2-2b-it` (official gated: manual).
- `command-r`: `adamo1139/aya-expanse-8b-ungated` = `CohereLabs/aya-expanse-8b` (official gated: auto).
- `command-r7b`: `mlx-community/c4ai-command-r7b-12-2024-bf16` = `CohereLabs/c4ai-command-r7b-12-2024` (official gated: auto).
- `eurollm`: the ungated official re-release `utter-project/EuroLLM-9B-Instruct-2512` has the same `tokenizer.model` LFS sha256 as the gated 2024 `EuroLLM-9B-Instruct`.
- `tiny-aya`: the ungated mirror's `tokenizer.json` is **not** byte-identical to the gated official one (187 bytes smaller). Equivalence could not be checked without logging in, so treat this row as provisional.

**Conversions** (all numbers from `manifest.json → crosschecks`). The first checks ran on 2,000 single-line paragraphs, the 5 core samples and one ASCII stress string; the *(multi-line)* checks repeat them on 200 multi-line strict records (first 3,000 characters each, line breaks kept) plus the 3 stress samples:

- Kimi K2: the official `tiktoken.model` was converted to a `tokenizers` BPE here (`files/kimi-k2/tokenizer.converted.json`). Its split regex was read from `tokenization_kimi.py` as text with `ast.literal_eval`; the file was never imported or run. Checked against a pure-Python implementation of tiktoken's algorithm on 2,006 texts (121,545 tokens): 0 mismatches and 0 round-trip failures.
- mistral-nemo: HF tokenizer.json vs native tekken.json: 0 mismatching texts out of 2,006 (109,228 tokens; tekken v3, 1000 special ids).
- mistral-small-4: HF tokenizer.json vs native tekken.json: 0 mismatching texts out of 2,006 (109,228 tokens; tekken v15, 1000 special ids).
- gpt-4o vs gpt-oss: 0 mismatching texts out of 2,006; base vocab ids < 199,998 with a different token: 0.
- gpt-4 vs phi-4: 0 mismatching texts out of 2,006; base vocab ids < 100,256 with a different token: 0.
- Kimi K2.5, K2.6 and K3 publish the same `tiktoken.model` (identical LFS sha256), and their split regex matches K2's: Kimi-K2-Instruct=same, Kimi-K2.5=same, Kimi-K2.6=same, Kimi-K3=same.
- Kimi K2 (multi-line): 0 mismatches vs the reference and 0 round-trip failures on 203 texts (125,882 tokens, 3,058 line breaks).
- mistral-nemo: HF tokenizer.json vs native tekken.json (multi-line): 0 mismatching texts out of 203 (113,016 tokens; tekken v3, 3,058 line breaks).
- mistral-small-4: HF tokenizer.json vs native tekken.json (multi-line): 0 mismatching texts out of 203 (113,016 tokens; tekken v15, 3,058 line breaks).
- gpt-4o vs gpt-oss (multi-line): 0 mismatching texts out of 203 (3,058 line breaks).
- gpt-4 vs phi-4 (multi-line): 0 mismatching texts out of 203 (3,058 line breaks).

**transformers cross-check.** Each baseline was also loaded with transformers 5.3 (`AutoTokenizer`, or the explicit class where AutoTokenizer cannot resolve a bare vocab file), and its ids on the 8 samples were compared with ours:

- Agree on 8/8: 50 baselines.
- **gpt-4o**: DIFFERS on 5/8 (AutoTokenizer -> GPT2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **gpt-4**: DIFFERS on 1/8 (AutoTokenizer -> GPT2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **qwen-3.5**: DIFFERS on 1/8 (AutoTokenizer -> Qwen2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **qwen-3.8**: DIFFERS on 1/8 (AutoTokenizer -> Qwen2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **deepseek-v3**: DIFFERS on 8/8 (AutoTokenizer -> LlamaTokenizer); transformers' normalizer and pre_tokenizer differ from the published tokenizer.json.
- **deepseek-r1**: DIFFERS on 8/8 (AutoTokenizer -> LlamaTokenizer); transformers' normalizer and pre_tokenizer differ from the published tokenizer.json.
- **phi-4**: DIFFERS on 1/8 (AutoTokenizer -> GPT2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **phi-4-mini**: DIFFERS on 5/8 (AutoTokenizer -> GPT2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **sarvam-m**: DIFFERS on 8/8 (AutoTokenizer -> LlamaTokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- **minimax-m2**: DIFFERS on 1/8 (AutoTokenizer -> GPT2Tokenizer); transformers' normalizer and pre_tokenizer differ from the published tokenizer.json.
- **claude-legacy**: DIFFERS on 2/8 (AutoTokenizer -> GPT2Tokenizer); transformers' normalizer and pre_tokenizer differ from the published tokenizer.json.
- **olmo-2**: DIFFERS on 1/8 (AutoTokenizer -> GPT2Tokenizer); transformers' pre_tokenizer differs from the published tokenizer.json.
- muril: n/a: baseline itself is loaded with transformers BertTokenizer.
- kimi-k2: skipped: official tokenizer needs trust_remote_code (not executed).
- roberta-urdu: n/a: baseline itself is loaded with transformers RobertaTokenizer.
- hindko-probe-bpe32k: skipped: local sentencepiece model.

The multi-line and stress samples raised the number of disagreements from 5 (on the 5 core paragraphs) to 12. In 12 of the 12 cases, the pipeline transformers builds differs from the published `tokenizer.json` (`manifest.json → transformers_pipeline_diff` records both versions). In 5.3, classes such as `LlamaTokenizerFast` (named in the DeepSeek-V3/R1 and Sarvam-M configs), `GPT2Tokenizer` and `Qwen2Tokenizer` rebuild their class-default pre-tokenizer instead of loading the repo's, and drop a normalizer the file declares. Examples: for Qwen 3.5/3.8 the published regex keeps combining marks inside words (`[\p{L}\p{M}]+`) and transformers substitutes Qwen2's `\p{L}+`; for `claude-legacy` it drops the file's NFKC normalizer; for DeepSeek-V3 and Sarvam-M it builds a Metaspace pipeline, and the first 120 characters of a sample came out as the single token `()`, so all Arabic-script text was dropped. Our loader runs the published files as they are. DeepSeek-V4 loads through the generic backend, agrees with transformers, and encodes the samples identically to the V3 file. The Mistral files match `tekken.json`, and Xenova/gpt-4o matches OpenAI's gpt-oss, on single-line and multi-line text. **Benchmark code must use `load_baseline()`, not `AutoTokenizer`.**

## Urdu-specific tokenizers

The task asked for Urdu LLMs with an **extended** Urdu vocabulary. Every Urdu candidate below was measured, not assumed:

| name | repo | vocab | extended vs base? | new Arabic-script tokens | lossless | encodings (samples + battery) |
|---|---|---:|---|---:|:---:|---|
| `urdu-llama3-almanach` | `almanach/Llama-3-8B-mono-Urdu` | 128,256 | no (base llama-3, 3 new surface forms) | 0 | yes | identical to llama-3, urdu-llama3.2-custom |
| `urdu-llama2-almanach` | `almanach/Llama-2-7B-mono-Urdu` | 32,000 | no (base llama-2, 0 new surface forms) | 0 | yes | identical to llama-2 |
| `urdu-llama3.2-custom` | `SabahNawab/Meta_Llama_3.2_3B_Urdu_Custom_Tokenizer` | 145,613 | yes (base llama-3, 17,348 new surface forms) | 17,347 | yes | identical to llama-3, urdu-llama3-almanach |
| `urdu-llama-bilal` | `BilalKhan1/llama-urdu-tokenizer` | 129,435 | yes (base llama-3, 1,080 new surface forms) | 1,077 | yes | 1470 tokens on the 5 core samples |
| `roberta-urdu` | `urduhack/roberta-urdu-small` | 52,000 | n/a (own vocabulary) | – | yes | 557 tokens on the 5 core samples |
| `urdu-bert-64k` | `farahadeeba/urdu-bert-64k` | 64,000 | n/a (own vocabulary) | – | **no** | 550 tokens on the 5 core samples |
| `alif-1.0` | `large-traversaal/Alif-1.0-8B-Instruct` | 128,256 | no (base llama-3, 0 new surface forms) | 0 | yes | identical to llama-3, urdu-llama3-almanach |
| `qalb-1.0` | `enstazao/Qalb-1.0-8B-Instruct` | 128,256 | no (base llama-3, 0 new surface forms) | 0 | yes | identical to llama-3, urdu-llama3-almanach |
| `urdu-gpt2-20k` | `aariciah/gpt2-urdu-20k-lc` | 20,000 | n/a (own vocabulary) | – | **no** | 574 tokens on the 5 core samples |
| `pashto-lfm2.5` | `nassimjp/LFM2.5-2.6B-Pashto-Zi` | 125,039 | yes (base lfm2, 60,380 new surface forms) | 7,505 | yes | 978 tokens on the 5 core samples |
| `sindhi-xlmr` | `Kashif786/xlm-roberta-base-sindhi-extended` | 264,635 | yes (base xlm-r, 14,633 new surface forms) | 14,533 | **no** | 684 tokens on the 5 core samples |

Findings:

- `alif-1.0`, `qalb-1.0`, `urdu-llama3-almanach` have Llama 3's vocabulary (no new surface forms containing Arabic script) and encode all 8 samples and all 2,619 battery texts identically to `llama-3`. `urdu-llama2-almanach` likewise matches `llama-2`. None of these Urdu LLMs extends the vocabulary.
- `urdu-llama3.2-custom` has 17,357 more ids than Llama 3. The extra tokens are stored as *added tokens* written in the byte-level alphabet (for example `ĠÙ¾ÛĮØª`), which never match raw text, so it encodes everything identically to Llama 3. The extension does nothing.
- `urdu-llama-bilal` has 1,179 more ids than Llama 3, as raw-Urdu *added tokens*. Added tokens are matched before pre-tokenization, even inside words, so the token count goes up: 1470 vs 1284 for Llama 3 on the 5 core samples.
- There is no strong Urdu LLM with a working vocabulary extension on the hub. The Urdu-script competitors worth benchmarking are therefore the Urdu-native tokenizers of small Urdu models: `roberta-urdu` (byte-level BPE 52k; lossless), `urdu-gpt2-20k` (20k, NFKC + Lowercase; **not lossless**: 658/2,619 battery texts exact; line breaks → UNK; loses U+000A LINE FEED, U+2026 HORIZONTAL ELLIPSIS, U+0768 ARABIC LETTER NOON WITH SMALL TAH) and `urdu-bert-64k` (WordPiece; **not lossless**: 22/2,619 battery texts exact; line breaks → 1 space per run; loses U+000A LINE FEED, U+0626 ARABIC LETTER YEH WITH HAMZA ABOVE, U+064E ARABIC FATHA). Two Perso-Arabic neighbours with real extensions were added from the scan: `pashto-lfm2.5` (Pashto-extended LFM2.5 LLM; lossless) and `sindhi-xlmr` (Sindhi-extended XLM-R; **not lossless**: 759/2,619 battery texts exact; line breaks → 1 space per run; loses U+000A LINE FEED, U+2026 HORIZONTAL ELLIPSIS, U+0768 ARABIC LETTER NOON WITH SMALL TAH). Add to these the multilingual large-vocabulary models (Gemma 3/4, Sarvam-30B, BLOOM, XLM-R, NLLB, MuRIL, IndicBERT v2).
- A wider hub scan (`scripts/urdu_scan.py`; queries: urdu, shahmukhi, saraiki, punjabi, pashto, sindhi, kashmiri, balochi, pakistan, hindko, lahnda) looked at 305 ungated repos (speech, classification, GGUF and adapter repos excluded). It found 123 tokenizer files that differ from everything above, and downloaded and measured the 45 with the most downloads and likes. The rest were not examined. See `scripts/_urdu_scan.json` and the table below. The scan ran before the stress samples and the battery existed: its RT column covers the 5 single-line core paragraphs only, so `True` there does **not** mean lossless (the scan candidates were not run on the battery; the selected ones above were).

| repo | downloads | vocab | Arabic-script tokens | tokens on 5 core samples | RT (5 core paragraphs only) | UNK |
|---|---:|---:|---:|---:|:---:|---:|
| `aariciah/gpt2-urdu-20k-lc` | 1544 | 20,000 | 17,981 | 574 | 5/5 | 0 |
| `aariciah/gpt2-urdu-dutch-merge` | 33 | 38,622 | 17,982 | 574 | 5/5 | 0 |
| `aariciah/gpt2-urdu-configC-20k` | 30 | 20,000 | 17,980 | 574 | 5/5 | 0 |
| `sibt-rj/albert-large-urdu` | 25 | 30,004 | 29,004 | 595 | 0/5 | 4 |
| `zirak-ai/pashto-bert-v1` | 31 | 30,000 | 29,358 | 597 | 0/5 | 145 |
| `l3cube-pune/punjabi-bert` | 45 | 197,285 | 16,398 | 628 | 0/5 | 0 |
| `tasal9/pashto-base-bloom` | 36 | 250,680 | 20,854 | 655 | 5/5 | 0 |
| `aakashMeghwar01/SindhiLM` | 98 | 24,000 | 23,613 | 668 | 0/5 | 0 |
| `Kashif786/xlm-roberta-base-sindhi-extended` | 126 | 264,635 | 29,177 | 684 | 5/5 | 0 |
| `halimajaved592/urdu-english-code-switching-xlm-roberta` | 110 | 250,002 | 14,644 | 706 | 5/5 | 0 |
| `areebanaz/mbart-legal-urdu` | 36 | 250,054 | 14,644 | 706 | 5/5 | 0 |
| `abdullah693/gemma-3-4b-it-urdu-edu-reasoning` | 193 | 262,145 | 8,460 | 729 | 5/5 | 0 |
| `areebanaz/nllb-legal-urdu` | 57 | 256,204 | 14,175 | 735 | 4/5 | 4 |
| `syedaafatima145/nllb-shahmukhi-12k` | 3 | 256,204 | 14,175 | 735 | 4/5 | 4 |
| `ahmedfarazsyk/sindhi-nllb-1.3b` | 0 | 256,204 | 14,175 | 735 | 4/5 | 4 |
| `Touqeer19/kashmiri-english-nllb-merged` | 33 | 256,204 | 14,175 | 735 | 4/5 | 4 |
| `Kashif786/distilbert-base-multilingual-cased-sindhi-cpt` | 140 | 119,547 | 4,873 | 812 | 0/5 | 6 |
| `fahadqazi/Sindhi-BPE-Tokenizer` | 0 | 13,032 | 12,667 | 819 | 2/5 | 0 |
| `Kashif786/distilbert-base-multilingual-cased-sindhi-wordpiece` | 828 | 136,521 | 20,501 | 836 | 0/5 | 0 |
| `PakistanLegalAI/test_arslan1` | 8 | 32,100 | 0 | 849 | 0/5 | 420 |
| `themohal/saraiki-english-translation-mt5` | 44 | 250,100 | 7,459 | 886 | 5/5 | 0 |
| `Imran1/gpt2-urdu-news` | 221 | 100,000 | 4,886 | 908 | 5/5 | 0 |
| `nassimjp/LFM2.5-2.6B-Pashto-Zi` | 7252 | 125,039 | 8,286 | 978 | 5/5 | 0 |
| `nassimjp/LFM2.5-2.6B-Pashto-Zi-b` | 233 | 125,073 | 8,315 | 978 | 5/5 | 0 |
| `HaiderSultanArc/t5-small-english-to-urdu` | 40 | 32,100 | 11,904 | 981 | 4/5 | 2 |
| `tasal9/ZamAI-LIama3-Pashto` | 53 | 128,256 | 3,810 | 1284 | 5/5 | 0 |
| `junaid008/qehwa-pashto-llm` | 515 | 151,666 | 4,018 | 1323 | 5/5 | 0 |
| `shaikhsalman/ZabaanAI-Urdu-3B` | 1537 | 151,665 | 4,018 | 1323 | 5/5 | 0 |
| `Khurram123/Qwen-Urdu-Shaheen-7B-Instruct-v1` | 71 | 151,666 | 4,018 | 1323 | 5/5 | 0 |
| `embedingHF/bilingual-roman-urdu-embedder` | 37 | 30,522 | 88 | 1418 | 0/5 | 44 |
| `aakashMeghwar01/SindhiLM-Qwen-0.5B-v2` | 518 | 159,422 | 11,775 | 1424 | 5/5 | 0 |
| `nassimjp/LFM2.5-350M-Pashto-Expansion` | 3708 | 64,439 | 818 | 1477 | 5/5 | 0 |
| `nassimjp/LFM2.5-1.2B-Base-Pashto` | 137 | 64,442 | 823 | 1489 | 5/5 | 0 |
| `nassimjp/LFM2.5-1.2B-Instruct-Pashto` | 23 | 64,444 | 823 | 1489 | 5/5 | 0 |
| `BilalKhan1/llama-urdu-model` | 28 | 51,437 | 1,201 | 1721 | 5/5 | 0 |
| `nassimjp/MiniCPM5-2B-Pashto` | 2755 | 130,606 | 82 | 1845 | 5/5 | 0 |
| `nassimjp/MiniCPM5-2B-SFT-Pashto-Instruct` | 1894 | 130,606 | 82 | 1845 | 5/5 | 0 |
| `prthmgoyl/slm-indic-punjabi-extended-llama-pretrained` | 3 | 136,953 | 3,886 | 1927 | 5/5 | 0 |
| `aariciah/gpt2-urdu-dutch-routed` | 380 | 38,611 | 17,981 | 1932 | 5/5 | 0 |
| `tasal9/ZamAI-Mistral-7B-Pashto` | 35 | 32,768 | 71 | 1945 | 5/5 | 0 |
| `tasal9/ZamAI-Phi-3-Mini-Pashto` | 63 | 32,011 | 55 | 2030 | 5/5 | 0 |

## Not collected

- Cohere Command A (03-2025): official repo gated (auto). The only full ungated copy checked (unsloth/c4ai-command-a-03-2025) ships a tokenizer.json that is byte-identical to Command R7B's, not to official Command A's (20,124,922 vs 19,597,349 bytes), so it was not used. Command A+ (05-2026, official, ungated) is included instead.
- AI4Bharat IndicTrans2 (indic-en / en-indic 1B): gated (auto), and its tokenizer needs remote code (tokenization_indictrans.py). Skipped; IndicBERT v2 represents AI4Bharat.
- AI4Bharat IndicBERT v3 (2025-12): gated (auto). Skipped.
- Swiss AI Apertus v1.5 (2026-07): gated (auto). Apertus 2509 (ungated) is included.
- xAI Grok-2: the official tokenizer.tok.json was downloaded (files/grok-2/) but NOT converted. It lists byte tokens and ids but not the pre-tokenizer regex ('word_split': 'V1' refers to code outside the file), so any conversion would be unverifiable. Grok-1 (community conversion) is included.
- Not public at all: Gemini (per the Gemma 3 technical report, Gemma 3 uses the same tokenizer as Gemini 2.0; not verifiable here), Claude 3 and later (only the legacy Claude 1/2 tokenizer is public and is included as a reference), and any OpenAI encoding newer than o200k_base/o200k_harmony (none was found on the hub).
- Meta Llama 3/4 and Google Gemma 2/3 official repos: gated (manual). Byte-identical ungated mirrors were used; see each row.
- UrduLLaMA: no repo found on the hub (searched 'UrduLlama', 'urdu-llama'). Urdu-Mistral: the 'urdu-mistral' search returned only small fine-tunes and LoRA adapters (the-usan/*, Maan23/*, duaafatima/*); their tokenizers were NOT individually checked. Alif-1.0, Qalb-1.0 and almanach mono-Urdu were checked: their vocabularies are NOT extended.
- Urdu/Perso-Arabic hub scan: 123 distinct new tokenizer signatures were found, but only the 45 with the highest downloads+likes were downloaded and measured (scripts/_urdu_scan.json); the remaining low-download repos were not examined.

## Caveats

- Losslessness is measured on the released text (NFC, before the canonical normalisation in `_tokenizer/normalization`). That normalisation folds `ﷲ` but keeps `ﷺ`, so NFKC-type tokenizers stay lossy after it. The battery covers every code point of the corpus, but only the eval split in full; a tokenizer could still fail on a context that occurs only in train.
- The 5-paragraph token counts are a sanity check, not a ranking. They are single-line, so line-break handling does not affect them.
- `vocab` is the number of distinct token ids including added/special tokens (`len(get_vocab(with_added_tokens=True))`). The model's embedding matrix can be larger (padding), and `id_space` in the manifest gives max id + 1.
- Licences are copied from each repo's model card metadata. `None` means the card declares no licence tag (for example DeepSeek-V3 and Hunyuan, which link a custom licence file instead). Mirrors inherit the official licence, recorded as `official_license` where known. Several licences (cc-by-nc-4.0, Llama, Gemma, Falcon, BLOOM RAIL) carry use restrictions. `files/` is a local working copy: check each licence before redistributing it.
- Release years come from the model releases (the repo creation date is in the manifest). They are not audited beyond that.
- Several frontier tokenizers are not public (Gemini, Claude 3 and later, OpenAI models after o200k/o200k_harmony). Gemma 3 is the closest public proxy for Gemini 2.0: the Gemma 3 report says they share a tokenizer.

## Reproduce

```
cd F:\Hindko\_tokenizer\baselines\scripts
set HF_HOME=F:\Hindko\_tokenizer\hf_cache & set PYTHONIOENCODING=utf-8
python probe_hub.py candidates.txt _probe_hub.json     # hub metadata (also candidates2/3/4.txt -> _probe_hub2/3/4.json)
python download.py                                      # tokenizer files only, pinned revisions
python make_samples.py                                  # 8 samples + single-line and multi-line verification sets
python convert_kimi.py && python crosscheck.py          # conversions and cross-checks (crosscheck.py queries the hub for Kimi versions)
python crosscheck_multiline.py                          # the same cross-checks on multi-line text (offline)
python urdu_scan.py                                     # wider Urdu / Perso-Arabic scan
python build_manifest.py && python write_md.py          # needs ../../splits/split_manifest.jsonl for the battery
```
