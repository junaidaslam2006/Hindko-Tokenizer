### TABLE confirm

| # | tokenizer | round | vocab | mean dev bpb (5 seeds) | seed s.d. | Δ vs baseline, 95 % CI | Δ vs best, 95 % CI | Holm p vs best | top set |
|---:|---|---|---:|---:|---:|---|---|---:|---|
| 1 | `R2-A4-SPnat-D2-32k` **(released)** | R2 (exploratory) | 32,768 | 1.22584 | 0.23 % | -1.62 % [-2.11, -1.24] | best | – | **yes** |
| 2 | `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | R2 (exploratory) | 49,152 | 1.22636 | 0.15 % | -1.58 % [-1.87, -1.36] | +0.04 % [-0.40, +0.57] | 0.8366 | **yes** |
| 3 | `R2-A10-MinGram-P1r3-D2-32k` (alternative) | R2 (exploratory) | 32,768 | 1.22844 | 0.21 % | -1.41 % [-1.66, -1.19] | +0.21 % [-0.17, +0.65] | 0.5800 | **yes** |
| 4 | `R2-A4-SPnat-D2-48k` | R2 (exploratory) | 49,152 | 1.23205 | 0.38 % | -1.13 % [-1.69, -0.64] | +0.51 % [+0.14, +0.87] | 0.0234 | no |
| 5 | `R2-A1-P1r3-D2-48k` | R2 (exploratory) | 49,152 | 1.23374 | 0.17 % | -0.99 % [-1.21, -0.78] | +0.64 % [+0.25, +1.12] | 0.0080 | no |
| 6 | `A1-P1r3-D2-32k` | R1 (pre-registered) | 32,768 | 1.23597 | 0.07 % | -0.81 % [-1.00, -0.66] | +0.83 % [+0.47, +1.26] | <1e-4 | no |
| 7 | `A4-SPnat-D1-16k` | R1 (pre-registered) | 16,384 | 1.23925 | 0.81 % | -0.55 % [-1.30, +0.23] | +1.09 % [+0.43, +1.83] | 0.0040 | no |
| 8 | `A10-MinGram-P1-D1-16k` | R1 (pre-registered) | 16,384 | 1.24180 | 0.20 % | -0.34 % [-0.57, -0.10] | +1.30 % [+0.91, +1.79] | <1e-4 | no |
| 9 | `A1-P1r3-D2-16k` (baseline) | R1 (pre-registered) | 16,384 | 1.24607 | 0.10 % | baseline | +1.65 % [+1.25, +2.15] | <1e-4 | no |
| 10 | `A1-P1-D1-16k` | R1 (added arm) | 16,384 | 1.25430 | 0.16 % | +0.66 % [+0.46, +0.86] | +2.32 % [+1.94, +2.80] | <1e-4 | no |
| 11 | `A7-PickyBPE-P1-D1-16k-tau0.9` | R1 (pre-registered) | 16,384 | 1.25446 | 0.10 % | +0.67 % [+0.51, +0.84] | +2.34 % [+1.97, +2.77] | <1e-4 | no |
| 12 | `A1-P1r3-D1-16k` | R1 (added arm) | 16,384 | 1.25532 | 0.18 % | +0.74 % [+0.51, +0.97] | +2.41 % [+1.98, +2.96] | <1e-4 | no |
| 13 | `A1-P1r3-D2-8k` | R1 (pre-registered) | 8,192 | 1.25672 | 0.18 % | +0.86 % [+0.37, +1.29] | +2.52 % [+1.88, +3.24] | <1e-4 | no |
| 14 | `A3-SPnat-D1-16k` | R1 (pre-registered) | 16,384 | 1.26651 | 1.46 % | +1.64 % [+0.43, +2.96] | +3.32 % [+2.15, +4.61] | <1e-4 | no |
| 15 | `R2-A6-SBPE-P1r3-D2-32k-t080` | R2 (exploratory) | 32,768 | 1.27375 | 0.12 % | +2.22 % [+1.82, +2.61] | +3.91 % [+3.53, +4.32] | <1e-4 | no |
| 16 | `A6-SBPE-P1-D1-16k-t090` | R1 (pre-registered) | 16,384 | 1.27485 | 0.20 % | +2.31 % [+2.02, +2.63] | +4.00 % [+3.62, +4.47] | <1e-4 | no |
| – | `A6-SBPE-P1-D1-32k-t080` (conditional rank 8, not ranked) | R1 (pre-registered) | 32,768 | 1.27960 | 0.13 % | +2.69 % [+2.25, +3.11] | +4.39 % [+4.04, +4.78] | – | no |
| 17 | `A6-SBPE-P1-D1-16k-t080` | R1 (pre-registered) | 16,384 | 1.28692 | 0.32 % | +3.28 % [+2.88, +3.71] | +4.98 % [+4.54, +5.52] | <1e-4 | no |

### TABLE large

| # | tokenizer | total params | mean dev bpb (2 seeds) | seeds 1 / 2 | Δ vs baseline, 95 % CI | Δ vs released, 95 % CI | confirm-scale bpb (rank) |
|---:|---|---:|---:|---|---|---|---|
| 1 | `R2-A4-SPnat-D2-32k` **(released)** | 23.31M | 1.13373 | 1.13319 / 1.13428 | -1.336 % [-1.78, -1.03] | – | 1.22584 (1) |
| 2 | `R2-A10-MinGram-P1r3-D2-32k` (alternative) | 23.31M | 1.14087 | 1.14016 / 1.14157 | -0.715 % [-0.91, -0.52] | +0.629 % [+0.31, +1.05] | 1.22844 (3) |
| 3 | `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 29.59M | 1.14407 | 1.14336 / 1.14478 | -0.436 % [-0.62, -0.29] | +0.911 % [+0.56, +1.37] | 1.22636 (2) |
| 4 | `A1-P1r3-D2-16k` (baseline) | 17.02M | 1.14908 | 1.14899 / 1.14917 | baseline | +1.354 % [+1.04, +1.81] | 1.24607 (9) |
| 5 | `A1-P1r3-D2-32k` | 23.31M | 1.14939 | 1.15108 / 1.14771 | +0.027 % [-0.30, +0.32] | +1.381 % [+1.02, +1.82] | 1.23597 (6) |

LR sweep of the large arbiter (baseline, seed 1): 0.0005 → 1.14899; 0.001 → 1.18164; 0.002 → 1.22207

Per source, Δ vs baseline (large arbiter, same bootstrap restricted to one source):

| source | `R2-A4-SPnat-D2-32k` | `R2-A10-MinGram-P1r3-D2-32k` | `R2-A10-MinGram-P1r3-D2-48k` | `A1-P1r3-D2-32k` |
|---|---|---|---|---|
| book | -1.40 % [-2.17, -0.97] | -0.50 % [-0.72, -0.26] | -0.24 % [-0.41, -0.03] | +0.26 % [-0.08, +0.57] |
| newspaper | -1.16 % [-1.43, -0.94] | -1.13 % [-1.39, -0.85] | -0.84 % [-1.18, -0.55] | -0.48 % [-0.88, -0.13] |
| web | -1.72 % [-4.12, +0.07] | -1.01 % [-2.55, +0.21] | -0.66 % [-1.91, +0.65] | +0.11 % [-1.70, +1.30] |

Top-set pairs (large arbiter): `R2-A4-SPnat-D2-32k - R2-A10-MinGram-P1r3-D2-48k` -0.90 % [-1.35, -0.56], p <1e-4; `R2-A4-SPnat-D2-32k - R2-A10-MinGram-P1r3-D2-32k` -0.62 % [-1.04, -0.31], p 0.0004; `R2-A10-MinGram-P1r3-D2-32k - R2-A10-MinGram-P1r3-D2-48k` -0.28 % [-0.46, -0.07], p 0.0110

### TABLE test intrinsic headline

| tokenizer | provider / family | vocab | test bytes/token | G1 lossless (test) | × tokens vs released | released uses fewer tokens |
|---|---|---:|---:|---:|---:|---:|
| **`R2-A4-SPnat-D2-32k`** (released) | this study: SentencePiece Unigram, D2 | 32,768 | **7.277** | 491/491 | 1.000 | – |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | this study: MinGram, P1r3, D2 | 49,152 | 7.373 | 491/491 | 0.987 | -1.3 % |
| `R2-A10-MinGram-P1r3-D2-32k` (alternative) | this study: MinGram, P1r3, D2 | 32,768 | 7.275 | 491/491 | 1.000 | +0.0 % |
| `A1-P1r3-D2-16k` (baseline) | this study: byte-level BPE, P1r3, D2 | 16,384 | 6.948 | 491/491 | 1.047 | 4.5 % |
| `gpt-4o` | OpenAI | 200,000 | 4.338 | 491/491 | 1.678 | 40.4 % |
| `gpt-4` | OpenAI | 100,263 | 1.919 | 491/491 | 3.792 | 73.6 % |
| `gemma-4` | Google | 262,144 | 4.610 | 491/491 | 1.578 | 36.6 % |
| `gemma-2` | Google | 256,000 | 3.787 | 491/491 | 1.922 | 48.0 % |
| `llama-4` | Meta | 201,135 | 3.727 | 491/491 | 1.953 | 48.8 % |
| `llama-3` | Meta | 128,256 | 2.641 | 491/491 | 2.755 | 63.7 % |
| `llama-2` | Meta | 32,000 | 1.714 | 491/491 | 4.246 | 76.5 % |
| `qwen-3.5` | Alibaba Qwen | 248,070 | 3.598 | 491/491 | 2.022 | 50.6 % |
| `qwen-3` | Alibaba Qwen | 151,669 | 2.633 | 491/491 | 2.764 | 63.8 % |
| `deepseek-v4.1` | DeepSeek | 129,280 | 3.372 | 491/491 | 2.158 | 53.7 % |
| `mistral-nemo` | Mistral AI | 131,072 | 3.901 | 491/491 | 1.865 | 46.4 % |
| `command-a-plus` | Cohere | 255,032 | 3.685 | 491/491 | 1.975 | 49.4 % |
| `grok-1` | xAI | 131,072 | 1.618 | 491/491 | 4.497 | 77.8 % |
| `kimi-k2` | Moonshot AI | 163,840 | 3.530 | 491/491 | 2.062 | 51.5 % |
| `glm-5` | Zhipu AI / Z.ai | 154,856 | 3.025 | 491/491 | 2.405 | 58.4 % |
| `falcon-h1` | TII | 261,120 | 4.506 | 491/491 | 1.615 | 38.1 % |
| `bloom` | BigScience | 250,680 | 5.122 | 491/491 | 1.421 | 29.6 % |
| `tiny-aya` | Cohere | 261,010 | 4.106 | 491/491 | 1.772 | 43.6 % |
| `roberta-urdu` | UrduHack | 52,000 | 5.709 | 491/491 | 1.275 | 21.6 % |
| `claude-legacy` | Anthropic | 65,000 | 1.554 | 467/491 ⚠ | 4.682 | 78.6 % |

### TABLE test intrinsic top10

| rank | tokenizer | provider / family | bytes/token (test) | lossless on test |
|---:|---|---|---:|---|
| 1 | **`R2-A4-SPnat-D2-32k`** (released) | this study | 7.277 | yes (491/491) |
| 2 | `urdu-bert-64k` | community (farahadeeba) | 6.221 | **no** (0/491) |
| 3 | `roberta-urdu` | UrduHack | 5.709 | yes |
| 4 | `urdu-gpt2-20k` | community (aariciah) | 5.633 | **no** (166/491) |
| 5 | `muril` | Google | 5.392 | **no** (0/491) |
| 6 | `indicbert-v2` | AI4Bharat | 5.299 | **no** (0/491) |
| 7 | `bloom` | BigScience | 5.122 | yes |
| 8 | `sindhi-xlmr` | community (Kashif786) | 5.069 | **no** (172/491) |
| 9 | `xlm-r` | Meta | 4.792 | **no** (173/491) |
| 10 | `nllb-200` | Meta | 4.617 | **no** (150/491) |
| 11 | `gemma-4` | Google | 4.610 | yes |

RELEASED rank 1 of 66; among lossless: 1 of 56; best lossless external roberta-urdu 5.709; fewer tokens 21.6 %
externals lossless on test: 55 of 65

### TABLE gates

| tokenizer | G1 lossless (dev_strict) | G2 unreachable whole-char tokens | G3 specials atomic, never in corpus | G4 deterministic retrain | G5 no sub-character tokens | partial-UTF-8 tokens (R2) |
|---|---|---|---|---|---|---|
| `R2-A4-SPnat-D2-32k` (released default) | 836/836 | pass (0 of 32,441 tested) | pass (0 occurrences) | pass | pass | 0 |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 836/836 | pass (0 of 48,831 tested) | pass (0 occurrences) | pass | pass | 0 |
| `R2-A10-MinGram-P1r3-D2-32k` (alternative) | 836/836 | pass (0 of 32,447 tested) | pass (0 occurrences) | pass | pass | 0 |
| `A1-P1r3-D2-16k` (baseline (standard recipe)) | 836/836 | pass (0 of 16,055 tested) | pass (0 occurrences) | pass | n/a (byte-level) | 9 |

### TABLE support

| tokenizer | learned tokens | train freq = 0 | train freq < 20 | train freq < 100 | median train freq | train_D1 tokens | dev bytes/token | dev fertility | dev STRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R2-A4-SPnat-D2-32k` (released default) | 32,441 | 2 | 13,182 (40.6 %) | 25,413 (78.3 %) | 27 | 7,305,029 | 7.076 | 1.141 | 88.6 % |
| `R2-A10-MinGram-P1r3-D2-48k` (pre-registered pick) | 48,831 | 1 | 27,246 (55.8 %) | 41,802 (85.6 %) | 16 | 7,158,278 | 7.224 | 1.122 | 89.8 % |
| `R2-A10-MinGram-P1r3-D2-32k` (alternative) | 32,447 | 0 | 11,329 (34.9 %) | 25,148 (77.5 %) | 32 | 7,316,797 | 7.126 | 1.138 | 88.7 % |
| `A1-P1r3-D2-16k` (baseline (standard recipe)) | 16,064 | 33 | 577 (3.6 %) | 8,491 (52.9 %) | 91 | 7,762,075 | 6.788 | 1.197 | 84.8 % |

### TABLE released dev by source

| slice | docs | bytes | bytes/token | chars/token | fertility | STRR | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all dev_strict | 836 | 1,445,513 | 7.076 | 3.967 | 1.141 | 88.6 % | 836/836 |
| source: book | 539 | 878,168 | 6.858 | 3.851 | 1.140 | 88.4 % | 539/539 |
| source: newspaper | 280 | 532,872 | 7.531 | 4.210 | 1.136 | 89.3 % | 280/280 |
| source: web | 17 | 34,473 | 6.310 | 3.553 | 1.244 | 82.2 % | 17/17 |
| variety: hindko | 751 | 1,405,500 | 7.099 | 3.980 | 1.141 | 88.7 % | 751/751 |
| variety: mixed | 83 | 39,031 | 6.391 | 3.595 | 1.144 | 87.2 % | 83/83 |
| variety: no_signal | 2 | 982 | 5.308 | 2.946 | 1.438 | 68.8 % | 2/2 |

Robustness (dev_strict, PLAN §4.2 perturbations): harakat: token count -0.37 %, 9.6 % of affected words re-segmented; digits: token count -0.65 %, 80.6 % of affected words re-segmented; punct_space: token count +6.07 %, 56.6 % of affected words re-segmented; zwnj: token count +0.57 %, 89.0 % of affected words re-segmented

NSL vs A1-P1-D1-16k: 0.9485

### TABLE trackb

| folder | base models | base licence | new tokens | len(tokenizer) | dev bytes/token base → extended | dev tokens | fertility | STRR | new tokens < 100 train occ. | English / code / other-script docs changed | Urdu dev token change (news / books) |
|---|---|---|---:|---:|---|---:|---|---|---:|---|---|
| `gemma3-hindko-cbpe2048/` | Gemma 3 (1B-27B) | gemma | 2,048 | 264,193 | 4.640 → **5.995** | −22.6 % | 1.765 → 1.357 | 49.6 % → 74.0 % | 51 (2.5 %) | 0 / 0 / 0 | −11.4 % / −7.9 % |
| `gemma4-hindko-cbpe2048/` | Gemma 4 (E2B-31B); same pieces and merges as Gemma 3, different control tokens | apache-2.0 | 2,048 | 264,192 | 4.640 → **5.995** | −22.6 % | 1.765 → 1.357 | 49.6 % → 74.0 % | 51 (2.5 %) | 0 / 0 / 0 | −11.4 % / −7.9 % |
| `llama3-hindko-cbpe2048/` | Llama 3/3.1/3.2/3.3 (also Alif-1.0, Qalb-1.0) | llama3.1 | 2,048 | 130,304 | 2.736 → **5.470** | −50.0 % | 2.981 → 1.504 | 19.0 % → 64.3 % | 40 (2.0 %) | 0 / 0 / 0 | −49.2 % / −41.3 % |
| `qwen3-hindko-cbpe2048/` | Qwen3 (0.6B-235B), Qwen2/2.5 share the encodings | apache-2.0 | 2,048 | 153,717 | 2.684 → **5.162** | −48.0 % | 2.998 → 1.595 | 3.1 % → 59.9 % | 34 (1.7 %) | 0 / 0 / 0 | −48.4 % / −42.0 % |
| `qwen3.5-hindko-cbpe2048/` | Qwen3.5 (0.8B-397B), Qwen3.8 shares the encodings | apache-2.0 | 2,048 | 250,125 | 3.631 → **5.601** | −35.2 % | 2.268 → 1.465 | 26.6 % → 67.0 % | 37 (1.8 %) | 0 / 0 / 0 | −26.5 % / −23.4 % |

