# Refreshed competitor sweep: does any existing tokenizer beat the released Hindko tokenizer? (2026-09-27)

Generated 2026-09-27T12:49:42Z by `sota/competitors/scripts/report_refresh.py` + `write_md.py`. Text: `data/test_strict.jsonl` (491 documents, 1,451,026 UTF-8 bytes, 171,769 words, sha256 `a74c33ad008f…`). Metrics: `eval/harness.py` (code sha256 identical to the run behind `eval/TEST_COMPETITORS.md`; checked for all 552 harness rows), each tokenizer with its own native encoder, no BOS/EOS/CLS added. Reporting only: nothing here changes a decision.

## Verdict

**No SOTA threat found.** No external tokenizer found in this sweep beats the released tokenizer on bytes/token or on fertility while round-tripping every test document.

- Released tokenizer (Hindko tokenizer 1.0.0, SentencePiece Unigram, 32,768 ids): **7.277 bytes/token**, fertility **1.135**, STRR 89.1%, G1 491/491, 199,390 tokens. Rank **1 of 153** by bytes/token among the lossless tokenizers (551 ranked rows in total, lossy ones included; rank 13 overall).
- Best lossless external tokenizer by bytes/token: `ProximaAI__urnova-95m` (ProximaAI; ProximaAI/urnova-95m) at **6.061** bytes/token, fertility 1.373. The released tokenizer uses **16.7 % fewer tokens** on the same 491 documents.
- Best lossless external tokenizer by fertility (tokens per word): `ProximaAI__urnova-95m` at 1.373 (released: 1.135).
- **Lossy tokenizers with a higher raw bytes/token than the released one: 12.** None is a threat: each fails the round trip on (almost) every document, and each gets its number by emitting UNK for the words it cannot spell (one UNK can stand for a whole word or line), so it encodes less text than it was given:

  | tokenizer | repo | bytes/tok | G1 | UNK share of its tokens | non-space chars lost | vocab | algorithm |
  |---|---|---:|---:|---:|---:|---:|---|
  | `Ayesha-fatima__urdu-articulation-screener` | Ayesha-fatima/urdu-articulation-screener | 2955.246 | 0/491 | 100.0% | 639,781 | 392 | WordLevel |
  | `internlm__Intern-S1-Pro` | internlm/Intern-S1-Pro | 2118.286 | 0/491 | 85.0% | 639,684 | 1,024 | BPE |
  | `internlm__Intern-S1` | internlm/Intern-S1 | 2102.936 | 0/491 | 83.3% | 639,677 | 512 | BPE |
  | `Omarrran__Kashmiri_Tokenizers__kashmiri_word_tokenizer` | Omarrran/Kashmiri_Tokenizers/kashmiri_word_tokenizer | 7.801 | 0/491 | 39.2% | 287,590 | 110,995 | WordLevel |
  | `ijazulhaq__bert-base-pashto-c` | ijazulhaq/bert-base-pashto-c | 7.698 | 0/491 | 97.9% | 635,839 | 703 | WordPiece |
  | `Kashif786__sindhi-universal-ultimate` | Kashif786/sindhi-universal-ultimate | 7.698 | 0/491 | 100.0% | 639,807 | 5 | WordPiece |
  | `mirfan899__da-sentiment` | mirfan899/da-sentiment | 7.691 | 0/491 | 98.8% | 636,647 | 31,748 | WordPiece |
  | `pnbonillo__clasificador-muchocine` | pnbonillo/clasificador-muchocine | 7.687 | 0/491 | 98.7% | 636,699 | 31,002 | WordPiece |
  | `AnanthZeke__TaNER-1k-indic_glue` | AnanthZeke/TaNER-1k-indic_glue | 7.666 | 0/491 | 98.5% | 637,040 | 1,023 | WordPiece |
  | `tcepi__HelBERT-uncased-fs-lsg-13-indicios` | tcepi/HelBERT-uncased-fs-lsg-13-indicios | 7.661 | 0/491 | 97.5% | 634,798 | 33,013 | WordPiece |
  | `almanach__Biomed-Enriched-classifier` | almanach/Biomed-Enriched-classifier | 7.644 | 0/491 | 95.4% | 630,168 | 30,522 | WordPiece |
  | `Ibrahimbaroud__personal-information-detection-arabic` | Ibrahimbaroud/personal-information-detection-arabic | 7.379 | 0/491 | 74.5% | 523,175 | 64,000 | WordPiece |

- Claim this supports (PLAN §8 / Amendment 1 §5 / TEST_RESULTS §12 framework): *on the strict Hindko test split, the released tokenizer uses fewer tokens per byte than every external tokenizer measured here that round-trips the text* — a statement about token counts on this corpus, not about downstream model quality. A tokenizer is not a model: it cannot generate, translate or answer anything by itself.

## What was searched (as of 2026-09-27)

- **Hugging Face hub (anonymous API, metadata only):** 43 keyword searches on repo ids (Hindko: `hindko`, `hindku`, `hinko`, `hazarewal`, `hazara`, `potohari`/`pothwari`, `pahari`, `lahnda`; Saraiki, Shahmukhi/Punjabi, Urdu, Pashto, Sindhi, Kashmiri, Balochi, Brahui, Khowar, Shina, Gojri, `pakistan`, `peshawar`, `nastaliq`, `perso-arabic`, `indic`, `indo-aryan`, `south-asian`, `bharat`), 27 language-tag filters (ISO 639: hno, hnd, ur, urd, pnb, skr, lah, phr, ps, pus, pbt, pbu, pst, sd, snd, ks, kas, bal, bcc, bgp, khw, scl, gju, pa, pan, trw, bft; text pipelines only), the 2025-2026 repos of 91 model labs, and 216 named frontier / regional repos; then a second pass (34 keyword, 33 author, 10 named queries) for leads from web and GitHub search (UrduLM/ALIF, Markhor, PakMosaic, BrahmicTokenizer, Muse Glimmer, Nemotron 3.5, Atria Dawn, Dots3, Ornith, Hy-MT2, Nex-N2.5, Bonsai 2, Ling 3.0, MiMo V2.6, MiniCPM5, Grok 2/3, gpt-oss derivatives).
- **Result:** 14,856 repos found, 7,760 with tokenizer files (root or first-level folder). 302 gated repos were not logged into (see *Not collected*); 563 speech / image-generation repos were skipped. The tokenizer files of 6,895 repos were identified by git blob id / LFS sha256 (HEAD requests); 1,765 repos carry a file byte-identical to one already benchmarked, and 1,759 distinct new files remained.
- **Bandwidth triage.** The link to the hub ran at 0.25-0.75 MB/s, so the 901 new files larger than 5 MB (mostly re-saved Qwen / Llama / Gemma / Mistral tokenizers inside fine-tunes, ~13 GB) were triaged with HTTP range reads before any full download: 308 are byte-identical in their whole `model` section, pipeline and effective added tokens to a tokenizer already on disk (`scripts/range_triage.py`; behaviour provably identical on this text), and 84 have the same first 300 vocabulary entries, the same pipeline, no added token that occurs in the test text, and 100% of the vocabulary/merge strings sampled from the middle and end of the file inside the reference's vocabulary (`scripts/triage2.py`; a re-save of that reference, NOT measured). 95 files whose `model` section is byte-identical to a file on disk but whose pipeline or added tokens differ were rebuilt exactly from their own first bytes plus the reference's `model` section (`scripts/reconstruct.py`; length checked against the hub) and measured. 263 large files whose sampled vocabulary strings are < 3 % Perso-Arabic and whose vocabulary family (first 300 entries) is represented by a measured file were not downloaded (`_not_downloaded.json`): a vocabulary with almost no Perso-Arabic entries cannot compete on Hindko (the best such families measured here stay below 5 bytes/token). Every other file (all files <= 5 MB, and every remaining large file) was downloaded in full and measured.
- **Measured:** 880 tokenizers downloaded (tokenizer files only; pinned commit; sha256 in `_download_log.json`), 957 loaded and encoded all 491 test documents (`screen/`), in 502 distinct encoding behaviours; 17 of them encode the test text exactly like one of the 66 earlier baselines, and the full harness ran on one representative of each of the other 485 behaviours (`results/`). 13 downloaded tokenizers could not be loaded (listed in `competitors_all.json`).
- **GitHub and the web** (GitHub search API; web search): no Hindko tokenizer and no Hindko language model exists (GitHub hits for `hindko`: a learning game, an idiom project, a Bible translation, a spoken-digit classifier, a keyboard). Urdu tokenizer projects found on GitHub only (MusW02/Urdu-TwoStage-BPE, zainali93/Markhor, salmanmasih/UrduLegalTok, course projects) publish code rather than hub tokenizer files and were not run (no third-party code). The only hub repo with Hindko in its name that has a tokenizer, `bisma10/hindko-punjabi-urdu-chatbot` (a LoRA adapter), ships its base model's tokenizer — see its row.

## Updated ranking (summary)

*fewer tokens* = 1 − tokens(released) ÷ tokens(tokenizer) on the same 491 documents. *origin*: `baseline` = one of the 66 of 2026-09-26, `new` = found by this refresh. Rows are distinct encoding behaviours (members of a behaviour group are listed in `competitors_all.json`). ⚠ = not lossless.

| # | ll # | tokenizer | origin | provider / repo | vocab | bytes/tok | fertility | STRR | G1 | UNK share | released uses fewer tokens |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | – | `Ayesha-fatima__urdu-articulation-screener` | new | Ayesha-fatima — Ayesha-fatima/urdu-articulation-screener | 392 | 2955.246 | 1.000 | 100.0% | 0/491 ⚠ | 100.0% | -40509.0 % |
| 2 | – | `internlm__Intern-S1-Pro` | new | internlm — internlm/Intern-S1-Pro | 1,024 | 2118.286 | 1.001 | 100.0% | 0/491 ⚠ | 85.0% | -29008.0 % |
| 3 | – | `internlm__Intern-S1` | new | internlm — internlm/Intern-S1 | 512 | 2102.936 | 1.001 | 100.0% | 0/491 ⚠ | 83.3% | -28797.1 % |
| 4 | – | `Omarrran__Kashmiri_Tokenizers__kashmiri_word_tokenizer` | new | Omarrran — Omarrran/Kashmiri_Tokenizers/kashmiri_word_tokenizer | 110,995 | 7.801 | 1.083 | 93.1% | 0/491 ⚠ | 39.2% | -7.2 % |
| 5 | – | `ijazulhaq__bert-base-pashto-c` | new | ijazulhaq — ijazulhaq/bert-base-pashto-c | 703 | 7.698 | 1.097 | 92.8% | 0/491 ⚠ | 97.9% | -5.8 % |
| 6 | – | `Kashif786__sindhi-universal-ultimate` | new | Kashif786 — Kashif786/sindhi-universal-ultimate | 5 | 7.698 | 1.097 | 92.8% | 0/491 ⚠ | 100.0% | -5.8 % |
| 7 | – | `mirfan899__da-sentiment` | new | mirfan899 — mirfan899/da-sentiment | 31,748 | 7.691 | 1.098 | 92.7% | 0/491 ⚠ | 98.8% | -5.7 % |
| 8 | – | `pnbonillo__clasificador-muchocine` | new | pnbonillo — pnbonillo/clasificador-muchocine | 31,002 | 7.687 | 1.099 | 92.7% | 0/491 ⚠ | 98.7% | -5.6 % |
| 9 | – | `AnanthZeke__TaNER-1k-indic_glue` | new | AnanthZeke — AnanthZeke/TaNER-1k-indic_glue | 1,023 | 7.666 | 1.102 | 92.6% | 0/491 ⚠ | 98.5% | -5.3 % |
| 10 | – | `tcepi__HelBERT-uncased-fs-lsg-13-indicios` | new | tcepi — tcepi/HelBERT-uncased-fs-lsg-13-indicios | 33,013 | 7.661 | 1.103 | 92.5% | 0/491 ⚠ | 97.5% | -5.3 % |
| 11 | – | `almanach__Biomed-Enriched-classifier` | new | almanach — almanach/Biomed-Enriched-classifier | 30,522 | 7.644 | 1.105 | 92.2% | 0/491 ⚠ | 95.4% | -5.0 % |
| 12 | – | `Ibrahimbaroud__personal-information-detection-arabic` | new | Ibrahimbaroud — Ibrahimbaroud/personal-information-detection-arabic | 64,000 | 7.379 | 1.145 | 88.8% | 0/491 ⚠ | 74.5% | -1.4 % |
| 13 | 1 | **`hindko-tokenizer-1.0.0-released`** | released | this study | 32,768 | 7.277 | 1.135 | 89.1% | 491/491 | 0.0% | – |
| 14 | – | `eshaaftab900__urdu-bert-base-2` | new | eshaaftab900 — eshaaftab900/urdu-bert-base-2 | 50,000 | 6.692 | 1.262 | 79.7% | 0/491 ⚠ | 0.1% | 8.0 % |
| 15 | – | `openbmb__chattts_tokenizer` | new | openbmb — openbmb/chattts_tokenizer | 21,178 | 6.585 | 1.283 | 82.7% | 0/491 ⚠ | 75.9% | 9.5 % |
| 16 | – | `goldfish-models__pnb_arab_full` | new | goldfish-models — goldfish-models/pnb_arab_full | 51,200 | 6.450 | 1.308 | 79.0% | 166/491 ⚠ | 0.0% | 11.4 % |
| 17 | – | `mahwizzzz__avey-b-ur` | new | mahwizzzz — mahwizzzz/avey-b-ur | 32,000 | 6.300 | 1.341 | 74.7% | 0/491 ⚠ | 0.0% | 13.4 % |
| 18 | – | `themohal__trocr-small-saraiki` | new | themohal — themohal/trocr-small-saraiki | 64,202 | 6.296 | 1.310 | 76.6% | 169/491 ⚠ | 0.1% | 13.5 % |
| 19 | – | `eshaaftab900__urdu-bert-base` | new | eshaaftab900 — eshaaftab900/urdu-bert-base | 60,000 | 6.255 | 1.351 | 73.0% | 0/491 ⚠ | 3.4% | 14.0 % |
| 20 | – | `urdu-bert-64k` | baseline | community (farahadeeba) — farahadeeba/urdu-bert-64k | 64,000 | 6.221 | 1.358 | 73.6% | 0/491 ⚠ | 0.0% | 14.5 % |
| 21 | – | `hafeez007__balochi-tokenizers` | new | hafeez007 — hafeez007/balochi-tokenizers | 64,000 | 6.188 | 1.365 | 71.9% | 175/491 ⚠ | 0.0% | 15.0 % |
| 22 | – | `goldfish-models__pnb_arab_10mb` | new | goldfish-models — goldfish-models/pnb_arab_10mb | 51,200 | 6.099 | 1.376 | 72.6% | 170/491 ⚠ | 0.0% | 16.2 % |
| 23 | 2 | `ProximaAI__urnova-95m` | new | ProximaAI — ProximaAI/urnova-95m | 50,048 | 6.061 | 1.373 | 74.7% | 491/491 | 0.0% | 16.7 % |
| 24 | – | `goldfish-models__urd_arab_100mb` | new | goldfish-models — goldfish-models/urd_arab_100mb | 51,200 | 6.055 | 1.387 | 73.4% | 165/491 ⚠ | 0.0% | 16.8 % |
| 25 | – | `themohal__SaraikiRoBERTa-MTL-v1` | new | themohal — themohal/SaraikiRoBERTa-MTL-v1 | 103,345 | 5.998 | 1.403 | 81.7% | 41/491 ⚠ | 0.0% | 17.6 % |
| 26 | – | `ijazulhaq__bert-base-pashto-v1` | new | ijazulhaq — ijazulhaq/bert-base-pashto-v1 | 30,000 | 5.989 | 1.411 | 68.6% | 0/491 ⚠ | 26.5% | 17.7 % |
| 27 | 3 | `DunbaaBERT__DunbaaBERT_52k_base` | new | DunbaaBERT — DunbaaBERT/DunbaaBERT_52k_base | 52,010 | 5.964 | 1.395 | 73.2% | 491/491 | 0.0% | 18.0 % |
| 28 | – | `goldfish-models__pnb_arab_5mb` | new | goldfish-models — goldfish-models/pnb_arab_5mb | 40,960 | 5.950 | 1.412 | 69.5% | 169/491 ⚠ | 0.0% | 18.2 % |
| 29 | 4 | `orature__ALIF-Base-100M` | new | orature — orature/ALIF-Base-100M | 32,000 | 5.912 | 1.405 | 71.0% | 491/491 | 0.0% | 18.8 % |
| 30 | – | `Humair332__omnivoice-urdu__checkpoint-1000` | new | Humair332 — Humair332/omnivoice-urdu/checkpoint-1000 | 50,007 | 5.873 | 1.417 | 72.2% | 467/491 ⚠ | 0.0% | 19.3 % |
| 31 | – | `goldfish-models__urd_arab_10mb` | new | goldfish-models — goldfish-models/urd_arab_10mb | 51,200 | 5.858 | 1.434 | 68.3% | 165/491 ⚠ | 0.0% | 19.5 % |
| 32 | – | `shah-bakhsh__balochi-tokenizer` | new | shah-bakhsh — shah-bakhsh/balochi-tokenizer | 48,000 | 5.806 | 1.455 | 64.9% | 0/491 ⚠ | 1.4% | 20.2 % |
| 33 | – | `HPLT__hplt_bert_base_ur` | new | HPLT — HPLT/hplt_bert_base_ur | 32,768 | 5.755 | 1.447 | 68.1% | 467/491 ⚠ | 0.0% | 20.9 % |
| 34 | 5 | `DunbaaBERT__DunbaaBERT_32k_base` | new | DunbaaBERT — DunbaaBERT/DunbaaBERT_32k_base | 32,010 | 5.718 | 1.456 | 69.7% | 491/491 | 0.0% | 21.4 % |
| 35 | – | `mahwizzzz__UrduNER` | new | mahwizzzz — mahwizzzz/UrduNER | 52,000 | 5.710 | 1.458 | 72.1% | 0/491 ⚠ | 0.0% | 21.5 % |
| 36 | 6 | `roberta-urdu` | baseline | UrduHack — urduhack/roberta-urdu-small | 52,000 | 5.709 | 1.459 | 72.1% | 491/491 | 0.0% | 21.6 % |
| 37 | – | `goldfish-models__urd_arab_5mb` | new | goldfish-models — goldfish-models/urd_arab_5mb | 38,912 | 5.687 | 1.476 | 65.8% | 165/491 ⚠ | 0.0% | 21.8 % |
| 38 | 7 | `ibm-research__ia-multilingual-original-script-roberta` | new | ibm-research — ibm-research/ia-multilingual-original-script-roberta | 110,000 | 5.668 | 1.469 | 69.3% | 491/491 | 0.0% | 22.1 % |
| 39 | – | `urdu-gpt2-20k` | baseline | community (aariciah) — aariciah/gpt2-urdu-20k-lc | 20,000 | 5.633 | 1.478 | 65.9% | 166/491 ⚠ | 1.4% | 22.6 % |
| 40 | – | `aariciah__gpt2-urdu-configC-20k__gpt2-urdu-configC-20k-GPT2TokenizerFast` | new | aariciah — aariciah/gpt2-urdu-configC-20k/gpt2-urdu-configC-20k-GPT2TokenizerFas | 20,000 | 5.632 | 1.479 | 65.9% | 173/491 ⚠ | 1.4% | 22.6 % |
| 56 | 10 | `bloom` | baseline | BigScience — bigscience/bloom-560m | 250,680 | 5.122 | 1.633 | 57.2% | 491/491 | 0.0% | 29.6 % |
| 85 | 15 | `gemma-4` | baseline | Google — google/gemma-4-E2B-it | 262,144 | 4.610 | 1.807 | 48.7% | 491/491 | 0.0% | 36.6 % |
| 86 | 16 | `sarvam-30b` | baseline | Sarvam AI — sarvamai/sarvam-30b | 262,144 | 4.610 | 1.807 | 48.7% | 491/491 | 0.0% | 36.6 % |
| 97 | 19 | `falcon-h1` | baseline | TII — tiiuae/Falcon-H1-34B-Instruct | 261,120 | 4.506 | 1.846 | 45.6% | 491/491 | 0.0% | 38.1 % |
| 119 | 21 | `gpt-4o` | baseline | OpenAI — Xenova/gpt-4o | 200,000 | 4.338 | 1.931 | 41.5% | 491/491 | 0.0% | 40.4 % |
| 190 | 29 | `tiny-aya` | baseline | Cohere — mlx-community/tiny-aya-global-8bit-mlx | 261,010 | 4.106 | 2.032 | 38.4% | 491/491 | 0.0% | 43.6 % |
| 211 | 32 | `mistral-nemo` | baseline | Mistral AI — mistralai/Mistral-Nemo-Instruct-2407 | 131,072 | 3.901 | 2.149 | 30.4% | 491/491 | 0.0% | 46.4 % |
| 235 | 40 | `llama-4` | baseline | Meta — unsloth/Llama-4-Scout-17B-16E-Instruct | 201,135 | 3.727 | 2.251 | 27.5% | 491/491 | 0.0% | 48.8 % |
| 239 | 41 | `command-a-plus` | baseline | Cohere — CohereLabs/command-a-plus-05-2026-bf16 | 255,032 | 3.685 | 2.274 | 26.1% | 491/491 | 0.0% | 49.4 % |
| 251 | 43 | `qwen-3.5` | baseline | Alibaba Qwen — Qwen/Qwen3.5-0.8B | 248,070 | 3.598 | 2.323 | 25.7% | 491/491 | 0.0% | 50.6 % |
| 257 | 46 | `kimi-k2` | baseline | Moonshot AI — moonshotai/Kimi-K2-Instruct | 163,840 | 3.530 | 2.375 | 27.8% | 491/491 | 0.0% | 51.5 % |
| 270 | 54 | `deepseek-v4.1` | baseline | DeepSeek — deepseek-ai/DeepSeek-V4.1-Flash | 129,280 | 3.372 | 2.476 | 23.2% | 491/491 | 0.0% | 53.7 % |
| 287 | 60 | `glm-5` | baseline | Zhipu AI / Z.ai — zai-org/GLM-5 | 154,856 | 3.025 | 2.763 | 13.7% | 491/491 | 0.0% | 58.4 % |
| 354 | 96 | `minimax-m3` | baseline | MiniMax — MiniMaxAI/MiniMax-M3 | 200,061 | 2.268 | 3.604 | 7.0% | 491/491 | 0.0% | 68.8 % |
| 370 | 105 | `phi-4` | baseline | Microsoft — microsoft/phi-4 | 100,352 | 1.919 | 4.275 | 0.7% | 491/491 | 0.0% | 73.6 % |
| 371 | 106 | `granite-4.2` | baseline | IBM — ibm-granite/granite-4.2-8b | 100,352 | 1.919 | 4.275 | 0.7% | 491/491 | 0.0% | 73.6 % |
| 453 | 128 | `grok-1` | baseline | xAI — Xenova/grok-1-tokenizer | 131,072 | 1.618 | 4.822 | 0.6% | 491/491 | 0.0% | 77.8 % |
| 461 | – | `claude-legacy` | baseline | Anthropic — Xenova/claude-tokenizer | 65,000 | 1.554 | 4.945 | 0.6% | 467/491 ⚠ | 0.0% | 78.6 % |

The table shows the top 40 rows by bytes/token plus the headline model families; all 551 ranked rows are in `competitors_all.json`.

## New tokenizers found by this refresh (distinct behaviours, best first)

| tokenizer | repo | found by | created | vocab | algorithm | bytes/tok | fertility | G1 | same behaviour as |
|---|---|---|---|---:|---|---:|---:|---:|---|
| `Ayesha-fatima__urdu-articulation-screener` | Ayesha-fatima/urdu-articulation-screener | kw:urdu | 2026-07-16 | 392 | WordLevel | 2955.246 | 1.000 | 0/491 | – |
| `internlm__Intern-S1-Pro` | internlm/Intern-S1-Pro | author:internlm | 2026-02-02 | 1,024 | BPE | 2118.286 | 1.001 | 0/491 | – |
| `internlm__Intern-S1` | internlm/Intern-S1 | author:internlm | 2025-07-24 | 512 | BPE | 2102.936 | 1.001 | 0/491 | – |
| `Omarrran__Kashmiri_Tokenizers__kashmiri_word_tokenizer` | Omarrran/Kashmiri_Tokenizers/kashmiri_word_tokenizer | kw:kashmiri | 2026-04-16 | 110,995 | WordLevel | 7.801 | 1.083 | 0/491 | Omarrran__Kashmiri_Word_Tokenizer |
| `ijazulhaq__bert-base-pashto-c` | ijazulhaq/bert-base-pashto-c | kw:pashto | 2023-07-29 | 703 | WordPiece | 7.698 | 1.097 | 0/491 | – |
| `Kashif786__sindhi-universal-ultimate` | Kashif786/sindhi-universal-ultimate | kw:sindhi | 2026-02-26 | 5 | WordPiece | 7.698 | 1.097 | 0/491 | – |
| `mirfan899__da-sentiment` | mirfan899/da-sentiment | author2:mirfan899 | 2023-01-27 | 31,748 | WordPiece | 7.691 | 1.098 | 0/491 | – |
| `pnbonillo__clasificador-muchocine` | pnbonillo/clasificador-muchocine | kw:pnb | 2023-11-09 | 31,002 | WordPiece | 7.687 | 1.099 | 0/491 | – |
| `AnanthZeke__TaNER-1k-indic_glue` | AnanthZeke/TaNER-1k-indic_glue | kw:indic | 2023-05-17 | 1,023 | WordPiece | 7.666 | 1.102 | 0/491 | AnanthZeke__TaNER-2k-indic_glue, AnanthZeke__TaNER-4k-indic_glue, AnanthZeke__Ta |
| `tcepi__HelBERT-uncased-fs-lsg-13-indicios` | tcepi/HelBERT-uncased-fs-lsg-13-indicios | kw:indic | 2024-08-23 | 33,013 | WordPiece | 7.661 | 1.103 | 0/491 | – |
| `almanach__Biomed-Enriched-classifier` | almanach/Biomed-Enriched-classifier | author:almanach | 2025-01-07 | 30,522 | WordPiece | 7.644 | 1.105 | 0/491 | – |
| `Ibrahimbaroud__personal-information-detection-arabic` | Ibrahimbaroud/personal-information-detection-arabic | kw:perso-arabic | 2026-03-04 | 64,000 | WordPiece | 7.379 | 1.145 | 0/491 | – |
| `eshaaftab900__urdu-bert-base-2` | eshaaftab900/urdu-bert-base-2 | kw:urdu | 2025-12-21 | 50,000 | WordPiece | 6.692 | 1.262 | 0/491 | eshaaftab900__urdu-electra-base-generator |
| `openbmb__chattts_tokenizer` | openbmb/chattts_tokenizer | author:openbmb | 2025-01-14 | 21,178 | WordPiece | 6.585 | 1.283 | 0/491 | – |
| `goldfish-models__pnb_arab_full` | goldfish-models/pnb_arab_full | kw:pnb | 2024-08-11 | 51,200 | Unigram | 6.450 | 1.308 | 166/491 | – |
| `mahwizzzz__avey-b-ur` | mahwizzzz/avey-b-ur | lang:ur | 2026-09-10 | 32,000 | WordPiece | 6.300 | 1.341 | 0/491 | – |
| `themohal__trocr-small-saraiki` | themohal/trocr-small-saraiki | kw:saraiki | 2026-08-21 | 64,202 | Unigram | 6.296 | 1.310 | 169/491 | – |
| `eshaaftab900__urdu-bert-base` | eshaaftab900/urdu-bert-base | kw:urdu | 2025-08-27 | 60,000 | WordPiece | 6.255 | 1.351 | 0/491 | – |
| `hafeez007__balochi-tokenizers` | hafeez007/balochi-tokenizers | kw:balochi | 2026-06-07 | 64,000 | BPE | 6.188 | 1.365 | 175/491 | – |
| `goldfish-models__pnb_arab_10mb` | goldfish-models/pnb_arab_10mb | kw:pnb | 2024-08-13 | 51,200 | Unigram | 6.099 | 1.376 | 170/491 | – |
| `ProximaAI__urnova-95m` | ProximaAI/urnova-95m | author2:ProximaAI | 2026-07-31 | 50,048 | BPE | 6.061 | 1.373 | 491/491 | – |
| `goldfish-models__urd_arab_100mb` | goldfish-models/urd_arab_100mb | lang:urd | 2024-08-12 | 51,200 | Unigram | 6.055 | 1.387 | 165/491 | goldfish-models__urd_arab_1000mb |
| `themohal__SaraikiRoBERTa-MTL-v1` | themohal/SaraikiRoBERTa-MTL-v1 | kw:saraiki | 2026-08-15 | 103,345 | BPE | 5.998 | 1.403 | 41/491 | – |
| `ijazulhaq__bert-base-pashto-v1` | ijazulhaq/bert-base-pashto-v1 | kw:pashto | 2023-04-15 | 30,000 | WordPiece | 5.989 | 1.411 | 0/491 | – |
| `DunbaaBERT__DunbaaBERT_52k_base` | DunbaaBERT/DunbaaBERT_52k_base | lang:ur | 2026-04-28 | 52,010 | BPE | 5.964 | 1.395 | 491/491 | – |
| `goldfish-models__pnb_arab_5mb` | goldfish-models/pnb_arab_5mb | kw:pnb | 2024-08-15 | 40,960 | Unigram | 5.950 | 1.412 | 169/491 | – |
| `orature__ALIF-Base-100M` | orature/ALIF-Base-100M | author:orature | 2025-05-07 | 32,000 | BPE | 5.912 | 1.405 | 491/491 | – |
| `Humair332__omnivoice-urdu__checkpoint-1000` | Humair332/omnivoice-urdu/checkpoint-1000 | kw:urdu | 2026-04-20 | 50,007 | BPE | 5.873 | 1.417 | 467/491 | – |
| `goldfish-models__urd_arab_10mb` | goldfish-models/urd_arab_10mb | lang:urd | 2024-08-13 | 51,200 | Unigram | 5.858 | 1.434 | 165/491 | – |
| `shah-bakhsh__balochi-tokenizer` | shah-bakhsh/balochi-tokenizer | kw:balochi | 2026-08-04 | 48,000 | BPE | 5.806 | 1.455 | 0/491 | – |

485 new behaviours in total; the rest (lower bytes/token) are in `competitors_all.json`.

## Named frontier and regional models: what happened to each tokenizer

| model / family | repo | status in this sweep |
|---|---|---|
| OpenAI gpt-oss (o200k_harmony); newer GPT-5-family encodings are not public | `openai/gpt-oss-20b` | tokenizer.json: file byte-identical to gpt-oss/tokenizer.json |
|  | `openai/gpt-oss-120b` | tokenizer.json: file byte-identical to gpt-oss/tokenizer.json |
| xAI Grok 2 (the only public xAI tokenizer after Grok-1) | `xai-org/grok-2` | tokenizer.tok.json: file byte-identical to grok-2/tokenizer.tok.json |
|  | `alvarobartt/grok-2-tokenizer` | tokenizer.json: new file, not downloaded |
|  | `unsloth/grok-2` | not probed (non-text pipeline or filtered) |
| Z.ai GLM-4.5 / 4.6 / 4.7 / 5.x | `zai-org/GLM-4.5` | tokenizer.json: file byte-identical to glm-4.5/tokenizer.json |
|  | `zai-org/GLM-4.6` | tokenizer.json: file byte-identical to glm-4.5/tokenizer.json |
|  | `zai-org/GLM-4.7` | tokenizer.json: file byte-identical to glm-4.5/tokenizer.json |
|  | `zai-org/GLM-4.7-Flash` | tokenizer.json: file byte-identical to glm-5/tokenizer.json |
|  | `zai-org/GLM-5.2` | tokenizer.json: file byte-identical to glm-5/tokenizer.json |
|  | `zai-org/GLM-5.3` | tokenizer.json: file byte-identical to glm-5/tokenizer.json |
| MiniMax M2.x / M3 | `MiniMaxAI/MiniMax-M2.5` | tokenizer.json: file byte-identical to minimax-m2/tokenizer.json |
|  | `MiniMaxAI/MiniMax-M2.7` | tokenizer.json: file byte-identical to minimax-m2/tokenizer.json |
|  | `MiniMaxAI/MiniMax-M3` | tokenizer.json: file byte-identical to minimax-m3/tokenizer.json |
| Tencent Hunyuan / Hy-MT | `tencent/Hunyuan-A13B-Instruct` | tokenizer.json: file byte-identical to hunyuan/tokenizer.json |
|  | `tencent/Hunyuan-MT-7B` | tokenizer.json: file byte-identical to hunyuan/tokenizer.json |
|  | `tencent/Hy-MT2-1.8B` | tokenizer.json: new file, not downloaded |
|  | `tencent/Hy-MT2-30B-A3B` | tokenizer.json: new file, not downloaded |
| IBM Granite 4.x | `ibm-granite/granite-4.0-h-small` | tokenizer.json: new file, not downloaded |
|  | `ibm-granite/granite-4.0-micro` | tokenizer.json: new file, not downloaded |
|  | `ibm-granite/granite-4.1-8b` | tokenizer.json: new file, not downloaded |
|  | `ibm-granite/granite-4.2-8b` | tokenizer.json: file byte-identical to granite-4.2/tokenizer.json |
| Ai2 OLMo 3 | `allenai/Olmo-3-7B-Instruct` | tokenizer.json: new file, not downloaded |
|  | `allenai/Olmo-3-1125-32B` | tokenizer.json: file byte-identical to olmo-2/tokenizer.json |
| Hugging Face SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | tokenizer.json: model section byte-identical to baseline `alif-1.0` (range reads; identical encodings) |
| Microsoft Phi-5 | `microsoft/Phi-5` | not on the hub (RepositoryNotFoundError) |
|  | `microsoft/phi-5` | not on the hub (RepositoryNotFoundError) |
| Cohere Command A / Aya Vision | `CohereLabs/c4ai-command-a-03-2025` | gated (auto); not logged into, not measured |
|  | `CohereLabs/command-a-reasoning-08-2025` | gated (auto); not logged into, not measured |
|  | `CohereLabs/aya-vision-8b` | gated (auto); not logged into, not measured |
|  | `CohereLabs/command-a-plus-05-2026-bf16` | tokenizer.json: file byte-identical to command-a-plus/tokenizer.json |
| Krutrim | `krutrim-ai-labs/Krutrim-1-instruct` | tokenizer.json: measured 5.168 bytes/token, G1 0/491 |
|  | `krutrim-ai-labs/Krutrim-2-instruct` | tokenizer.json: new file, not downloaded |
| Sarvam | `sarvamai/sarvam-105b` | tokenizer.json: file byte-identical to sarvam-30b/tokenizer.json |
|  | `sarvamai/sarvam-translate` | tokenizer.json: file byte-identical to gemma-3/tokenizer.json |
|  | `sarvamai/sarvam-30b` | tokenizer.json: file byte-identical to sarvam-30b/tokenizer.json |
| AI4Bharat Airavata | `ai4bharat/Airavata` | gated (auto); not logged into, not measured |
| AI4Bharat IndicTrans2 | `ai4bharat/indictrans2-indic-en-1B` | gated (auto); not logged into, not measured |
|  | `ai4bharat/indictrans2-en-indic-dist-200M` | gated (auto); not logged into, not measured |
| AI4Bharat IndicBERT v3 | `ai4bharat/IndicBERT-v3-270M` | gated (auto); its tokenizer file is byte-identical to _urdu_candidates/abdullah693__gemma-3-4b-it-urdu-edu-reasoning/tokenizer.json |
| Meta Llama 4 variants | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | gated (manual); its tokenizer file is byte-identical to llama-4/tokenizer.json |
|  | `unsloth/Llama-4-Maverick-17B-128E-Instruct` | tokenizer.json: file byte-identical to llama-4/tokenizer.json |
| Google Gemma 3n | `google/gemma-3n-E2B-it` | gated (manual); not logged into, not measured |
|  | `unsloth/gemma-3n-E2B-it` | tokenizer.json: same vocabulary as baseline `gemma-3` (range-read sample; not measured) |
| Mistral Medium 3.5 | `mistralai/Mistral-Medium-3.5-128B` | tokenizer.json: file byte-identical to mistral-small-4/tokenizer.json |
|  | `unsloth/Mistral-Medium-3.5-128B` | tokenizer.json: file byte-identical to mistral-small-4/tokenizer.json |
| DeepSeek V4 (Pro / Flash-0731) | `deepseek-ai/DeepSeek-V4-Pro` | tokenizer.json: file byte-identical to deepseek-v4/tokenizer.json |
|  | `deepseek-ai/DeepSeek-V4-Flash-0731` | tokenizer.json: file byte-identical to deepseek-v4/tokenizer.json |
| Qwen3-Next / Qwen3.8-Flash-Next | `Qwen/Qwen3-Next-80B-A3B-Instruct` | tokenizer.json: file byte-identical to qwen-3/tokenizer.json |
|  | `Qwen/Qwen3.8-Flash-Next` | tokenizer.json: file byte-identical to qwen-3.8/tokenizer.json |
| Moonshot Kimi K2.x | `moonshotai/Kimi-K2.5` | tiktoken.model: file byte-identical to kimi-k2/tiktoken.model |
|  | `moonshotai/Kimi-K2.6` | tiktoken.model: file byte-identical to kimi-k2/tiktoken.model |
| TII Falcon-H1 (all sizes) | `tiiuae/Falcon-H1-0.5B-Instruct` | tokenizer.json: measured 1.146 bytes/token, G1 491/491, same encodings as `tiiuae__Falcon-H1-0.5B-Base` |
|  | `tiiuae/Falcon-H1-1.5B-Instruct` | tokenizer.json: same vocabulary as baseline `falcon-h1` (range-read sample; not measured) |
|  | `tiiuae/Falcon-H1-3B-Instruct` | tokenizer.json: same vocabulary as baseline `falcon-h1` (range-read sample; not measured) |
|  | `tiiuae/Falcon-H1-7B-Instruct` | tokenizer.json: new file, not downloaded |
|  | `tiiuae/Falcon-H1R-7B` | tokenizer.json: new file, not downloaded |
| NVIDIA Nemotron-H | `nvidia/Nemotron-H-8B-Base-8K` | tokenizer.json: new file, not downloaded |
|  | `nvidia/Nemotron-H-4B-Instruct-128K` | tokenizer.json: new file, not downloaded |
| Meta Muse Glimmer (2026-08) | `meta-models/Muse-Glimmer-30B` | tokenizer.json: model section byte-identical to baseline `llama-4` (range reads; identical encodings) |
| NVIDIA Nemotron 3.5 Lightning (2026-08) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | tokenizer.json: model section byte-identical to refresh file `nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16` (range reads; identical encodings) |
| Shanghai AI Lab Atria Dawn (2026-09) | `internlm/Atria-Dawn-Preview` | tokenizer.json: file byte-identical to glm-5/tokenizer.json |
| OpenBMB MiniCPM5 (2026-09) | `openbmb/MiniCPM5-2B` | tokenizer.json: model section byte-identical to 2026-09-26 Urdu-scan file of nassimjp/MiniCPM5-2B-Pashto (range reads; identical encodings) |
| inclusionAI Ling 3.0 (2026-08/09) | `inclusionAI/Ling-3.0-tiny` | tokenizer.json: new file, not downloaded |
|  | `inclusionAI/Ling-3.0-flash` | tokenizer.json: new file, not downloaded |
| Dots Studio dots3 (2026-08) | `dots-studio/dots3-note-prev` | tokenizer.json: model section byte-identical to baseline `qwen-3` (range reads; identical encodings) |
| DeepReinforce Ornith (2026-08) | `unsloth/Ornith-1.0-9B` | tokenizer.json: model section byte-identical to refresh file `unsloth__Qwen3.8-27B-NVFP4` (range reads; identical encodings) |
| Nex AGI Nex-N2.5 (2026-09) | `nex-agi/Nex-N2.5-mini` | tokenizer.json: measured 3.598 bytes/token, G1 491/491, same encodings as baseline `qwen-3.5`, `qwen-3.8` |
| Xiaomi MiMo V2.5 / V2.6 (2026-09) | `XiaomiMiMo/MiMo-V2.5` | tokenizer.json: model section byte-identical to baseline `qwen-2.5` (range reads; identical encodings) |
|  | `XiaomiMiMo/MiMo-V2.6-Flash-RL` | tokenizer.json: model section byte-identical to baseline `qwen-3` (range reads; identical encodings) |
| UrduLM / ALIF-Base-100M (Urdu, 2026-01) | `orature/ALIF-Base-100M` | spm: measured 5.912 bytes/token, G1 491/491 |
| ProximaAI urnova-95m / PakMosaic (Pakistan languages) | `ProximaAI/urnova-95m` | tokenizer.json: measured 6.061 bytes/token, G1 491/491 |
|  | `ProximaAI/PakMosaic` | no tokenizer file in the repo |
| BrahmicTokenizer-131K (Indic, 2026-05) | `theschoolofai/BrahmicTokenizer-131K` | tokenizer.json: new file, not downloaded |
| Hindko-named hub repo | `bisma10/hindko-punjabi-urdu-chatbot` | tokenizer.json: measured 1.791 bytes/token, G1 491/491, same encodings as `upstage__SOLAR-10.7B-Instruct-v1.0` |
| Saraiki (themohal) | `themohal/saraiki-qwen3-8b-cpt` | gated (manual); not logged into, not measured |
|  | `themohal/saraiki-roberta-base-small-finetuned3` | gated (manual); not logged into, not measured |
|  | `themohal/trocr-small-saraiki` | tokenizer.json: measured 6.296 bytes/token, G1 169/491 |

## Best tokenizers per neighbouring language (lossless only)

- Urdu: `ProximaAI__urnova-95m` (ProximaAI/urnova-95m) 6.061 bytes/token, fertility 1.373 — the released tokenizer uses 16.7 % fewer tokens.
- Punjabi / Shahmukhi: `hgnoi__PnBJ7Z4wxY29v1kZ` (hgnoi/PnBJ7Z4wxY29v1kZ) 1.918 bytes/token, fertility 4.278 — the released tokenizer uses 73.6 % fewer tokens.
- Saraiki: no lossless tokenizer with the language in its name.
- Pashto: `pashto-lfm2.5` (nassimjp/LFM2.5-2.6B-Pashto-Zi) 3.509 bytes/token, fertility 2.386 — the released tokenizer uses 51.8 % fewer tokens.
- Sindhi: `alphaedge-ai__gemma-3-270m-it-snd-32768` (alphaedge-ai/gemma-3-270m-it-snd-32768) 4.532 bytes/token, fertility 1.839 — the released tokenizer uses 37.7 % fewer tokens.
- Kashmiri: no lossless tokenizer with the language in its name.
- Balochi: `agemagician__mlong-t5-tglobal-base` (agemagician/mlong-t5-tglobal-base) 4.153 bytes/token, fertility 1.884 — the released tokenizer uses 42.9 % fewer tokens.

## Not collected

- **Gated repos (112 checked by file metadata, never logged into):** 47 carry a tokenizer file byte-identical to one measured here (e.g. `Kalana/multilingual-sft-qwen-3-14b-punjabi`, `RAFAY-484/Urdu-Punjabi-V2`, `yaggibook/test_23_PnBGi5`, `Alefiah/mt5-base-finetuned-urdu-finetuned-xsum`, `cxfajar197/urdu-ocr`, `Sibghat7/Urdu-Phish-Gauard`). 65 could not be matched and were not measured: `google/embeddinggemma-300m`, `google/medgemma-4b-it`, `google/gemma-3n-E2B-it`, `google/gemma-3-270m`, `google/translategemma-4b-it`, `CohereLabs/command-a-vision-07-2025`, `google/gemma-3n-E4B-it`, `google/vaultgemma-1b`, `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-base`, `inception42/Jais-2-8B-Chat`, `ai4bharat/Cadence`, `CohereLabs/aya-vision-8b`, `bharatgenai/patram-7b-instruct`, `google/t5gemma-2b-2b-ul2`, `CohereLabs/c4ai-command-a-03-2025`, `facebook/MobileLLM-R1-950M`, `ai4bharat/Airavata`, `CohereLabs/command-a-reasoning-08-2025`, `CohereLabs/tiny-aya-l2-thinker`, `CohereLabs/tiny-aya-fire`, `CohereLabs/tiny-aya-earth`, `CohereLabs/aya-vision-32b`, `CohereLabs/tiny-aya-water`, `CoRover/BharatGPT-Instruct`.
- **Not convertible without running repo code:** tiktoken rank files whose split regex lives only in Python code (`BAAI/Emu3.5`, `baseten/o200k-base-tiktoken`, `Salesforce/GTA1-32B`, `tencent/Hunyuan-A13B-Instruct-GPTQ-Int4`, `unsloth/inkling-NVFP4`), and xAI's `tokenizer.tok.json` (Grok 2; as in `baselines/manifest.json`).
- **Not public:** Gemini, Claude 3+ and any OpenAI encoding newer than o200k_base / o200k_harmony (unchanged since 2026-09-26). MiMo-V2.6 (not found on the hub under that name).
- **Large files classified without a full download:** 392 (see above); their encodings are identical (byte-identical model section) or expected identical (sampled vocabulary) to a measured tokenizer, so none can exceed the rows shown here.
- **Download errors:** 0.

## Caveats

- Bytes/token and fertility are screening metrics; LM bits-per-byte (PLAN §4.2) is the decision metric and was not re-run here. The released tokenizer's LM results are in `analysis/TEST_RESULTS.md`.
- Lossy tokenizers (UNK-emitting WordPiece/WordLevel vocabularies, NFKC/lower-casing pipelines, line-break dropping) can show a higher raw bytes/token than a lossless one; they are ranked but never counted as a threat.
- The hub search covers repo ids, language tags and the listed labs; a tokenizer published under an unrelated name, without a language tag, or outside the hub cannot be excluded.
