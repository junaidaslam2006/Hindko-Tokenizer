# Final competitor ranking: released Hindko tokenizer vs every external tokenizer measured (2026-09-27)

Generated 2026-09-27T13:07:05Z by `sota/competitors/scripts/finalize.py` + `build_final.py` + `write_final_md.py` from tokenizers **already measured** (no new search or download for this file). Text: `data/test_strict.jsonl` (491 documents, 1,451,026 UTF-8 bytes, 171,769 words, sha256 `a74c33ad008f…`). Metrics from `eval/harness.py` (same code as `eval/TEST_COMPETITORS.md`), each tokenizer with its own native encoder, no BOS/EOS/CLS. Machine-readable: `competitors_final.json`. Reporting only; nothing here changes a decision.

## Answer

**Does any lossless external tokenizer beat the released one on bytes/token or fertility? No.**

- Released tokenizer (SentencePiece Unigram, 32,768 ids, `tokenizer.json` sha256 `49f301c5…`): **7.277 bytes/token**, fertility **1.135**, STRR 89.1 %, G1 491/491, 199,390 tokens.
- Rank **1 of 131** distinct lossless encoding behaviours (130 lossless external behaviours + the released one) on both bytes/token and fertility.
- Best lossless external tokenizer: `ProximaAI__urnova-95m` (ProximaAI/urnova-95m) at **6.061 bytes/token**, fertility 1.373 — the released tokenizer uses **16.7 % fewer tokens** on the same 491 documents.
- It is also the best lossless external tokenizer by fertility (1.373 vs released 1.135).
- 12 lossy tokenizers show a higher *raw* bytes/token (and 57 a lower-or-equal fertility). None is a competitor: each fails the round trip on (almost) every document, mostly by emitting UNK for words it cannot spell, so it encodes less text than it was given (table below).
- Robustness check: the best tokenizer that fails G1 but emits no UNK and loses no non-space character (`mahwizzzz__UrduNER`, G1 0/491) reaches 5.710 bytes/token — also below the released tokenizer.

**Claim this supports** (PLAN §8 / Amendment 1 §5 / TEST_RESULTS §12 framing): *on the strict Hindko test split, the released tokenizer produces fewer tokens than every external tokenizer measured here that round-trips the text.* It is a statement about token counts on this corpus, for the tokenizers found and measured by 2026-09-27. A tokenizer is not a model: it cannot generate, translate or answer anything, and this ranking says nothing about downstream model quality. Closed tokenizers (Gemini, Claude 3+, OpenAI encodings newer than o200k) are not public and are not covered.

## What was compared

| count | value |
|---|---:|
| external harness rows (full metrics, test_strict) | 550 |
| &nbsp;&nbsp;of which new (hub refresh, `results/`) | 485 |
| &nbsp;&nbsp;of which study baselines (`eval/results/test_strict/`) | 65 |
| **distinct encoding behaviours** (identical per-document token counts + round-trip outcome merged) | **528** |
| &nbsp;&nbsp;lossless (G1 491/491) / lossy | 130 / 398 |
| **vocabulary families** (heuristic, see Method) | **466** |
| &nbsp;&nbsp;families with at least one lossless member | 102 |
| hub repos whose tokenizer is represented by a measured row (byte-identical file or identical behaviour) | 2,608 |
| tokenizer files loaded and screened in the refresh | 957 |

Excluded: `hindko-probe-bpe32k` (the study's in-corpus probe, trained on text that includes the test split; never ranked) and the released tokenizer itself.
The study's baseline notes missed that `sarvam-30b` encodes the test text exactly like `gemma-3`/`gemma-4` (identical per-document vectors); here they are merged.

## Top 15 lossless external tokenizers (distinct behaviours, by bytes/token)

*fewer tokens* = 1 − tokens(released) ÷ tokens(tokenizer) on the same 491 documents. *aliases* = other measured rows with identical encodings; *repos* = hub repos carrying this tokenizer (byte-identical file or identical behaviour).

| # | tokenizer | repo / provider | origin | vocab family | algorithm | vocab | bytes/tok | fertility | STRR | released uses fewer tokens | aliases | repos |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| – | **released (Hindko 1.0.0)** | this study | released | – | Unigram | 32,768 | **7.277** | **1.135** | 89.1 % | – | – | – |
| 1 | `ProximaAI__urnova-95m` | ProximaAI/urnova-95m | new | ProximaAI/urnova-95m (BPE 50,048) | BPE | 50,048 | 6.061 | 1.373 | 74.7 % | 16.7 % | – | 1 |
| 2 | `DunbaaBERT__DunbaaBERT_52k_base` | DunbaaBERT/DunbaaBERT_52k_base | new | DunbaaBERT/DunbaaBERT_52k_base (BPE 52,010) | BPE | 52,010 | 5.964 | 1.395 | 73.2 % | 18.0 % | – | 1 |
| 3 | `orature__ALIF-Base-100M` | orature/ALIF-Base-100M | new | orature/ALIF-Base-100M (BPE 32,000) | BPE | 32,000 | 5.912 | 1.405 | 71.0 % | 18.8 % | – | 1 |
| 4 | `DunbaaBERT__DunbaaBERT_32k_base` | DunbaaBERT/DunbaaBERT_32k_base | new | DunbaaBERT/DunbaaBERT_32k_base (BPE 32,010) | BPE | 32,010 | 5.718 | 1.456 | 69.7 % | 21.4 % | – | 1 |
| 5 | `roberta-urdu` | urduhack/roberta-urdu-small | baseline | roberta-urdu | BPE | 52,000 | 5.709 | 1.459 | 72.1 % | 21.6 % | – | 21 |
| 6 | `ibm-research__ia-multilingual-original-script-roberta` | ibm-research/ia-multilingual-original-script-robe… | new | ibm-research/ia-multilingual-original-script… | BPE | 110,000 | 5.668 | 1.469 | 69.3 % | 22.1 % | – | 1 |
| 7 | `mahwizzzz__urdu-dummy` | mahwizzzz/urdu-dummy | new | mahwizzzz/urdu-dummy (BPE 25,000) | BPE | 25,000 | 5.586 | 1.491 | 67.1 % | 23.2 % | – | 1 |
| 8 | `hadidev__robertaurdu` | hadidev/robertaurdu | new | hadidev/robertaurdu (BPE 59,458) | BPE | 59,458 | 5.523 | 1.501 | 77.5 % | 24.1 % | – | 1 |
| 9 | `bloom` | bigscience/bloom-560m | baseline | bloom | BPE | 250,680 | 5.122 | 1.633 | 57.2 % | 29.6 % | – | 1 |
| 10 | `santhosh__madlad400-3b-ct2` | santhosh/madlad400-3b-ct2 | new | santhosh/madlad400-3b-ct2 (Unigram 256,000) | Unigram | 256,000 | 5.080 | 1.599 | 56.2 % | 30.2 % | – | 8 |
| 11 | `mlx-community__K2-Horizon-MoVA-36B-A4B-oQ4e` | mlx-community/K2-Horizon-MoVA-36B-A4B-oQ4e | new | mlx-community/K2-Horizon-MoVA-36B-A4B-oQ4e (… | BPE | 250,624 | 5.064 | 1.651 | 56.8 % | 30.4 % | – | 3 |
| 12 | `MaLA-LM__mala-500-10b-v2` | MaLA-LM/mala-500-10b-v2 | new | llama-2 / urdu-llama2-almanach | BPE | 260,164 | 4.913 | 1.698 | 55.6 % | 32.5 % | – | 1 |
| 13 | `gemma-3` | unsloth/gemma-3-1b-it | baseline | gemma-3 | BPE | 262,145 | 4.610 | 1.807 | 48.7 % | 36.6 % | `gemma-4`, `sarvam-30b` | 3 |
| 14 | `alphaedge-ai__gemma-3-270m-it-snd-32768` | alphaedge-ai/gemma-3-270m-it-snd-32768 | new | gemma-3 | BPE | 32,768 | 4.532 | 1.839 | 47.0 % | 37.7 % | – | 3 |
| 15 | `Palash123__indic_200K_all_tokenizer` | Palash123/indic_200K_all_tokenizer | new | Palash123/indic_200K_all_tokenizer (BPE 200,… | BPE | 200,000 | 4.528 | 1.840 | 48.5 % | 37.8 % | – | 1 |

## Top 15 lossless vocabulary families (best member of each family)

| # | family (label) | best member | bytes/tok | fertility | released uses fewer tokens | measured rows in family |
|---:|---|---|---:|---:|---:|---:|
| 1 | ProximaAI/urnova-95m (BPE 50,048) | `ProximaAI__urnova-95m` | 6.061 | 1.373 | 16.7 % | 1 |
| 2 | DunbaaBERT/DunbaaBERT_52k_base (BPE 52,010) | `DunbaaBERT__DunbaaBERT_52k_base` | 5.964 | 1.395 | 18.0 % | 1 |
| 3 | orature/ALIF-Base-100M (BPE 32,000) | `orature__ALIF-Base-100M` | 5.912 | 1.405 | 18.8 % | 1 |
| 4 | DunbaaBERT/DunbaaBERT_32k_base (BPE 32,010) | `DunbaaBERT__DunbaaBERT_32k_base` | 5.718 | 1.456 | 21.4 % | 1 |
| 5 | roberta-urdu | `roberta-urdu` | 5.709 | 1.459 | 21.6 % | 2 |
| 6 | ibm-research/ia-multilingual-original-script-roberta (BPE 1… | `ibm-research__ia-multilingual-original-script-roberta` | 5.668 | 1.469 | 22.1 % | 1 |
| 7 | mahwizzzz/urdu-dummy (BPE 25,000) | `mahwizzzz__urdu-dummy` | 5.586 | 1.491 | 23.2 % | 1 |
| 8 | hadidev/robertaurdu (BPE 59,458) | `hadidev__robertaurdu` | 5.523 | 1.501 | 24.1 % | 1 |
| 9 | bloom | `bloom` | 5.122 | 1.633 | 29.6 % | 1 |
| 10 | santhosh/madlad400-3b-ct2 (Unigram 256,000) | `santhosh__madlad400-3b-ct2` | 5.080 | 1.599 | 30.2 % | 1 |
| 11 | mlx-community/K2-Horizon-MoVA-36B-A4B-oQ4e (BPE 250,624) | `mlx-community__K2-Horizon-MoVA-36B-A4B-oQ4e` | 5.064 | 1.651 | 30.4 % | 1 |
| 12 | llama-2 / urdu-llama2-almanach | `MaLA-LM__mala-500-10b-v2` | 4.913 | 1.698 | 32.5 % | 5 |
| 13 | gemma-3 | `gemma-3` | 4.610 | 1.807 | 36.6 % | 7 |
| 14 | Palash123/indic_200K_all_tokenizer (BPE 200,000) | `Palash123__indic_200K_all_tokenizer` | 4.528 | 1.840 | 37.8 % | 2 |
| 15 | falcon-h1 | `falcon-h1` | 4.506 | 1.846 | 38.1 % | 1 |

## Notable lossy tokenizers

Lossy = fails G1 on at least one document. These are ranked but never counted as competitors: UNK tokens and dropped characters inflate raw bytes/token.

| tokenizer | repo | algorithm | vocab | bytes/tok | fertility | G1 | UNK share of tokens | non-space chars lost | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Ayesha-fatima__urdu-articulation-screener` | Ayesha-fatima/urdu-articulation-screener | WordLevel | 392 | 2955.246 | 1.000 | 0/491 | 100.0 % | 639,781 | raw bytes/token above released |
| `internlm__Intern-S1-Pro` | internlm/Intern-S1-Pro | BPE | 1,024 | 2118.286 | 1.001 | 0/491 | 85.0 % | 639,684 | raw bytes/token above released |
| `internlm__Intern-S1` | internlm/Intern-S1 | BPE | 512 | 2102.936 | 1.001 | 0/491 | 83.3 % | 639,677 | raw bytes/token above released |
| `Omarrran__Kashmiri_Tokenizers__kashmiri_word_tokenizer` | Omarrran/Kashmiri_Tokenizers/kashmiri_word_tokeni… | WordLevel | 110,995 | 7.801 | 1.083 | 0/491 | 39.2 % | 287,590 | raw bytes/token above released |
| `ijazulhaq__bert-base-pashto-c` | ijazulhaq/bert-base-pashto-c | WordPiece | 703 | 7.698 | 1.097 | 0/491 | 97.9 % | 635,839 | raw bytes/token above released |
| `Kashif786__sindhi-universal-ultimate` | Kashif786/sindhi-universal-ultimate | WordPiece | 5 | 7.698 | 1.097 | 0/491 | 100.0 % | 639,807 | raw bytes/token above released |
| `mirfan899__da-sentiment` | mirfan899/da-sentiment | WordPiece | 31,748 | 7.691 | 1.098 | 0/491 | 98.8 % | 636,647 | raw bytes/token above released |
| `pnbonillo__clasificador-muchocine` | pnbonillo/clasificador-muchocine | WordPiece | 31,002 | 7.687 | 1.099 | 0/491 | 98.7 % | 636,699 | raw bytes/token above released |
| `AnanthZeke__TaNER-1k-indic_glue` | AnanthZeke/TaNER-1k-indic_glue | WordPiece | 1,023 | 7.666 | 1.102 | 0/491 | 98.5 % | 637,040 | raw bytes/token above released |
| `tcepi__HelBERT-uncased-fs-lsg-13-indicios` | tcepi/HelBERT-uncased-fs-lsg-13-indicios | WordPiece | 33,013 | 7.661 | 1.103 | 0/491 | 97.5 % | 634,798 | raw bytes/token above released |
| `almanach__Biomed-Enriched-classifier` | almanach/Biomed-Enriched-classifier | WordPiece | 30,522 | 7.644 | 1.105 | 0/491 | 95.4 % | 630,168 | raw bytes/token above released |
| `Ibrahimbaroud__personal-information-detection-arabic` | Ibrahimbaroud/personal-information-detection-arab… | WordPiece | 64,000 | 7.379 | 1.145 | 0/491 | 74.5 % | 523,175 | raw bytes/token above released |
| `eshaaftab900__urdu-bert-base-2` | eshaaftab900/urdu-bert-base-2 | WordPiece | 50,000 | 6.692 | 1.262 | 0/491 | 0.1 % | 538 | best lossy below released |
| `openbmb__chattts_tokenizer` | openbmb/chattts_tokenizer | WordPiece | 21,178 | 6.585 | 1.283 | 0/491 | 75.9 % | 585,903 | best lossy below released |
| `goldfish-models__pnb_arab_full` | goldfish-models/pnb_arab_full | Unigram | 51,200 | 6.450 | 1.308 | 166/491 | 0.0 % | 219 | best lossy below released |
| `mahwizzzz__avey-b-ur` | mahwizzzz/avey-b-ur | WordPiece | 32,000 | 6.300 | 1.341 | 0/491 | 0.0 % | 141 | best lossy below released |
| `themohal__trocr-small-saraiki` | themohal/trocr-small-saraiki | Unigram | 64,202 | 6.296 | 1.310 | 169/491 | 0.1 % | 448 | best lossy below released |
| `eshaaftab900__urdu-bert-base` | eshaaftab900/urdu-bert-base | WordPiece | 60,000 | 6.255 | 1.351 | 0/491 | 3.4 % | 9,246 | best lossy below released |
| `urdu-bert-64k` | farahadeeba/urdu-bert-64k | WordPiece | 64,000 | 6.221 | 1.358 | 0/491 | 0.0 % | 23,415 | best lossy below released |
| `hafeez007__balochi-tokenizers` | hafeez007/balochi-tokenizers | BPE | 64,000 | 6.188 | 1.365 | 175/491 | 0.0 % | 169 | best lossy below released |

## Study baselines (frontier and regional tokenizers), deduplicated

| tokenizer | aliases (identical encodings) | provider | bytes/tok | fertility | G1 | released uses fewer tokens |
|---|---|---|---:|---:|---:|---:|
| `urdu-bert-64k` | – | community (farahadeeba) | 6.221 | 1.358 | 0/491 ⚠ | 14.5 % |
| `roberta-urdu` | – | UrduHack | 5.709 | 1.459 | 491/491 | 21.6 % |
| `urdu-gpt2-20k` | – | community (aariciah) | 5.633 | 1.478 | 166/491 ⚠ | 22.6 % |
| `muril` | – | Google | 5.392 | 1.567 | 0/491 ⚠ | 25.9 % |
| `indicbert-v2` | – | AI4Bharat | 5.299 | 1.594 | 0/491 ⚠ | 27.2 % |
| `bloom` | – | BigScience | 5.122 | 1.633 | 491/491 | 29.6 % |
| `sindhi-xlmr` | – | community (Kashif786) | 5.069 | 1.599 | 172/491 ⚠ | 30.3 % |
| `xlm-r` | – | Meta | 4.792 | 1.680 | 173/491 ⚠ | 34.2 % |
| `nllb-200` | – | Meta | 4.617 | 1.824 | 150/491 ⚠ | 36.6 % |
| `gemma-3` | `gemma-4`, `sarvam-30b` | Google | 4.610 | 1.807 | 491/491 | 36.6 % |
| `falcon-h1` | – | TII | 4.506 | 1.846 | 491/491 | 38.1 % |
| `gpt-4o` | `gpt-oss`, `phi-4-mini` | OpenAI | 4.338 | 1.931 | 491/491 | 40.4 % |
| `mbert` | – | Google | 4.151 | 2.035 | 0/491 ⚠ | 43.0 % |
| `tiny-aya` | – | Cohere | 4.106 | 2.032 | 491/491 | 43.6 % |
| `mt5` | – | Google | 3.912 | 1.874 | 175/491 ⚠ | 46.2 % |
| `apertus` | `mistral-nemo`, `mistral-small-4`, `nemotron-3`, `sarvam-m` | Swiss AI | 3.901 | 2.149 | 491/491 | 46.4 % |
| `gemma-2` | – | Google | 3.787 | 2.206 | 491/491 | 48.0 % |
| `llama-4` | – | Meta | 3.727 | 2.251 | 491/491 | 48.8 % |
| `command-a-plus` | – | Cohere | 3.685 | 2.274 | 491/491 | 49.4 % |
| `qwen-3.5` | `qwen-3.8` | Alibaba Qwen | 3.598 | 2.323 | 491/491 | 50.6 % |
| `kimi-k2` | – | Moonshot AI | 3.530 | 2.375 | 491/491 | 51.5 % |
| `ernie-4.5` | – | Baidu | 3.528 | 2.369 | 491/491 | 51.5 % |
| `pashto-lfm2.5` | – | community (nassimjp) | 3.509 | 2.386 | 491/491 | 51.8 % |
| `deepseek-r1` | `deepseek-v3`, `deepseek-v4`, `deepseek-v4.1` | DeepSeek | 3.372 | 2.476 | 491/491 | 53.7 % |
| `glm-5` | – | Zhipu AI / Z.ai | 3.025 | 2.763 | 491/491 | 58.4 % |
| `command-r` | `command-r7b` | Cohere | 2.645 | 3.107 | 491/491 | 63.7 % |
| `alif-1.0` | `llama-3`, `qalb-1.0`, `urdu-llama3-almanach`, `urdu-llama3.2-custom` | Traversaal.ai | 2.641 | 3.119 | 491/491 | 63.7 % |
| `qwen-2.5` | `qwen-3` | Alibaba Qwen | 2.633 | 3.082 | 491/491 | 63.8 % |
| `glm-4.5` | – | Zhipu AI / Z.ai | 2.482 | 3.373 | 491/491 | 65.9 % |
| `urdu-llama-bilal` | – | community (BilalKhan1) | 2.372 | 2.834 | 491/491 | 67.4 % |
| `minimax-m2` | `minimax-m3` | MiniMax | 2.268 | 3.604 | 491/491 | 68.8 % |
| `eurollm` | – | UTTER (EU) | 2.147 | 3.678 | 491/491 | 70.5 % |
| `hunyuan` | – | Tencent | 1.932 | 4.245 | 491/491 | 73.4 % |
| `gpt-4` | `olmo-2`, `phi-4` | OpenAI | 1.919 | 4.275 | 491/491 | 73.6 % |
| `granite-4.2` | – | IBM | 1.919 | 4.275 | 491/491 | 73.6 % |
| `lfm2` | – | Liquid AI | 1.900 | 4.181 | 491/491 | 73.9 % |
| `falcon-3` | – | TII | 1.813 | 4.504 | 491/491 | 75.1 % |
| `llama-2` | `urdu-llama2-almanach` | Meta | 1.714 | 3.937 | 491/491 | 76.5 % |
| `grok-1` | – | xAI | 1.618 | 4.822 | 491/491 | 77.8 % |
| `claude-legacy` | – | Anthropic | 1.554 | 4.945 | 467/491 ⚠ | 78.6 % |
| `gpt-2` | – | OpenAI | 1.385 | 5.749 | 491/491 | 81.0 % |
| `sarvam-1` | – | Sarvam AI | 1.005 | 7.407 | 491/491 | 86.2 % |
| `byt5` | – | Google | 1.000 | 7.450 | 491/491 | 86.3 % |

## Search coverage — the search was cut short

The ranking uses the measurement snapshot of 2026-09-27T12:49:42Z (the refresh report `COMPETITORS_REFRESH.md`). The web/hub search was then **stopped for time**; anything found or downloaded after the snapshot is **not** in this ranking.

- **Pass 1** (`_search.json`, 2026-09-27T08:17:53Z): 43 repo-id keywords (Hindko: hindko, hindku, hinko, hazarewal, hazara, potohari/pothwari, pahari, lahnda; Saraiki, Shahmukhi/Punjabi, Urdu, Pashto, Sindhi, Kashmiri, Balochi, Brahui, Khowar, Shina, Gojri, regional/script terms), 27 ISO 639 language tags (hno, hnd, ur, urd, pnb, skr, lah, phr, ps, pus, pbt, pbu, pst, sd, snd, ks, kas, bal, bcc, bgp, khw, scl, gju, pa, pan, trw, bft), 91 model-lab authors, 216 named frontier/regional repos (15 not on the hub); 198 queries → 10,995 repos, 7,309 with tokenizer files.
- **Pass 2** (`_search2.json`, 2026-09-27T08:46:42Z): 34 keywords, 33 authors, 10 named repos from web/GitHub leads (UrduLM/ALIF, Markhor, PakMosaic, BrahmicTokenizer, 2026 lab releases); 67 queries → 3,861 repos, 634 kept after filtering.
- **Totals at the snapshot:** 14,856 repos found, 7,760 with tokenizer files, 6,895 probed by file hash, 880 new files downloaded + 95 reconstructed, 957 screened OK, 502 behaviour groups, 0 download errors; 302 gated repos and 563 speech/image repos skipped; 392 large files classified by range reads (identical or expected-identical to a measured tokenizer) instead of downloaded.
- **GitHub and web:** GitHub search API + web search (refresh, before the snapshot): no Hindko tokenizer or Hindko LM found; Urdu tokenizer projects on GitHub publish code only and were not run (no third-party code).
- **Pass 3 — stopped** (`_search3.json`, 2026-09-27T12:54:36Z): 32 author queries for the regional authors surfaced by passes 1–2 (DunbaaBERT, eshaaftab900, Humair332, hafeez007, shah-bakhsh, ijazulhaq, Hammad712, alphaedge-ai, Omarrran, HPLT, orature, ProximaAI, goldfish-models, zuhri025, SabahNawab, BilalKhan1, farahadeeba, urduhack, Imran1, Khurram123, shaikhsalman, mfayazkhan, large-traversaal, enstazao, nassimjp, tasal9, zirak-ai, APARSIN, Abdullah104, aakashMeghwar01, themohal, mahwizzzz) → 2,789 repos, 2,564 with tokenizer files, 508 kept after filtering. The last line of the probe log (`logs/probe_pass3.log`) reports 458 repos still to probe when the step was stopped; nothing from pass 3 was screened or measured (all `screen/` and `results/` files predate the snapshot).
- **Downloads after the snapshot:** `_download_log.json` holds 1026 entries, 47 of them logged after the snapshot (a low-priority download batch, `logs/download_final_lo.log`, was still writing when this file was generated); they are not screened or measured. Download errors in the log: MoritzLaurer/xlm-v-base-mnli-xnli (RemoteProtocolError: peer closed connection without sending ).
- **Not collected:** gated repos (not logged into): 302 skipped at search, 65 unmatched after metadata check; closed tokenizers: Gemini, Claude 3+, OpenAI encodings newer than o200k_base / o200k_harmony; tiktoken rank files whose split regex exists only in repo Python code; xAI tokenizer.tok.json (Grok 2); 392 large files classified by range reads as identical / expected-identical to a measured tokenizer (not measured themselves); 13 downloaded tokenizers that could not be loaded.

## Method and caveats

- **metrics:** bytes/token = UTF-8 bytes / tokens; fertility = tokens / whitespace word; each tokenizer with its own native encoder, no BOS/EOS/CLS (eval/harness.py)
- **lossless:** G1: decode(encode(doc)) == doc for all 491 test documents
- **dedup:** behaviour = sha256 of the per-document (token count, round-trip ok) vector over the 491 docs (same key as scripts/screen.py); identical vectors are treated as identical encodings (a strong proxy, not a token-by-token proof)
- **family:** vocabulary family = sha256 of vocabulary entries at ids 0-299 and 1000-1099 (heuristic lineage key; fine-tunes and appended vocab extensions stay in the base family)
- **metric_check:** all 550 rows re-read from their own summary.json: 0 mismatches with competitors_all.json
- Bytes/token and fertility are screening metrics; the study's decision metric is LM bits-per-byte (PLAN §4.2), which was not re-run for these competitors. The frontier LM comparison (`colab/build_frontier_test/`, `FRONTIER_LM.md`) is **planned**, not run (Colab GPU quota exhausted).
- The hub search covers repo ids, language tags and listed labs; a tokenizer published under an unrelated name, without a language tag, gated, or outside the hub cannot be excluded.
- A vocabulary family is a lineage heuristic (same vocabulary prefix); families whose first entries are generic (byte alphabets, `[unused]` slots) may be split or merged imperfectly. Deduplication itself uses the exact per-document fingerprint, not the family key.
