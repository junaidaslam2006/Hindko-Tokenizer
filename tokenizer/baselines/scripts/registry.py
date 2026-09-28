"""Registry of competitor tokenizers for the Hindko tokenizer benchmark.

Every entry is data only.  Fields:
  name          short slug used by load_baseline(name)
  provider      organisation that trained the tokenizer
  family        model family / tokenizer name
  models        which models use this exact tokenizer (as far as verified; see notes)
  year          release year of the tokenizer (first model that shipped it)
  repo          Hugging Face repo the files were downloaded from
  official      official source repo when `repo` is a mirror/conversion (None when repo is official)
  mirror_kind   None | "byte-identical mirror" | "community conversion" | "mirror (bytes differ)"
  loader        hf_json | spm | auto | bytes | tiktoken_ranks | local_spm
  tier          core | frontier-2026 | urdu | regional (Perso-Arabic neighbours: Pashto, Sindhi) | reference
  notes         free text
Files downloaded = allow-listed tokenizer files that exist at the repo root (see download.py).
"""

ENTRIES = [
    # ---------------- OpenAI ----------------
    dict(name="gpt-4o", provider="OpenAI", family="o200k_base (tiktoken)", models="GPT-4o, GPT-4o-mini, GPT-4.1, o-series reasoning models",
         year=2024, repo="Xenova/gpt-4o", official="tiktoken o200k_base (openaipublic)", mirror_kind="community conversion",
         loader="hf_json", tier="core", crosscheck="gpt-oss",
         notes="HF conversion of tiktoken o200k_base; cross-checked against OpenAI's own gpt-oss tokenizer.json (o200k_harmony = o200k_base + specials)."),
    dict(name="gpt-4", provider="OpenAI", family="cl100k_base (tiktoken)", models="GPT-4, GPT-4-turbo, GPT-3.5-turbo, text-embedding-3",
         year=2023, repo="Xenova/gpt-4", official="tiktoken cl100k_base (openaipublic)", mirror_kind="community conversion",
         loader="hf_json", tier="core", crosscheck="phi-4",
         notes="HF conversion of tiktoken cl100k_base; cross-checked against Microsoft phi-4 (cl100k-derived vocabulary)."),
    dict(name="gpt-oss", provider="OpenAI", family="o200k_harmony", models="gpt-oss-20b, gpt-oss-120b, gpt-oss-safeguard",
         year=2025, repo="openai/gpt-oss-20b", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="Official. tokenizer.json byte-identical in gpt-oss-20b and gpt-oss-120b."),
    dict(name="gpt-2", provider="OpenAI", family="GPT-2 byte-level BPE", models="GPT-2 (also GPT-3 r50k)",
         year=2019, repo="openai-community/gpt2", official=None, mirror_kind=None, loader="hf_json", tier="reference",
         notes="Historic byte-level BPE baseline (50k)."),
    # ---------------- Meta ----------------
    dict(name="llama-3", provider="Meta", family="Llama 3 tiktoken-BPE 128k", models="Llama 3, 3.1, 3.2, 3.3 (also SmolLM3, Alif-1.0, Qalb-1.0)",
         year=2024, repo="NousResearch/Meta-Llama-3.1-8B-Instruct", official="meta-llama/Llama-3.1-8B-Instruct",
         mirror_kind="byte-identical mirror", loader="hf_json", tier="core",
         notes="Official repo is gated (manual). Mirror tokenizer.json blob id == official Llama-3.1-8B-Instruct and Llama-3.2-1B-Instruct blob id (checked via hub metadata)."),
    dict(name="llama-4", provider="Meta", family="Llama 4 BPE 202k", models="Llama 4 Scout, Llama 4 Maverick",
         year=2025, repo="unsloth/Llama-4-Scout-17B-16E-Instruct", official="meta-llama/Llama-4-Scout-17B-16E-Instruct",
         mirror_kind="byte-identical mirror", loader="hf_json", tier="core",
         notes="Official repo gated (manual). tokenizer.json and tokenizer.model LFS sha256 identical to official (hub metadata)."),
    dict(name="llama-2", provider="Meta", family="Llama 2 SentencePiece BPE 32k", models="Llama 2 (base of many 2023 Urdu fine-tunes)",
         year=2023, repo="NousResearch/Llama-2-7b-hf", official="meta-llama/Llama-2-7b-hf", mirror_kind="mirror (not byte-verified)",
         loader="hf_json", tier="reference",
         notes="Kept as the base for the Urdu-extended Llama-2 tokenizer (almanach) comparison."),
    # ---------------- Google ----------------
    dict(name="gemma-4", provider="Google", family="Gemma 4 SentencePiece 262k", models="Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B)",
         year=2026, repo="google/gemma-4-E2B-it", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026",
         notes="Official, ungated (Apache-2.0). tokenizer.json byte-identical across E2B-it and 31B-it."),
    dict(name="gemma-3", provider="Google", family="Gemma 3 SentencePiece 262k", models="Gemma 3 (1B-27B); the Gemma 3 technical report states Gemini 2.0 uses the same tokenizer",
         year=2025, repo="unsloth/gemma-3-1b-it", official="google/gemma-3-1b-it", mirror_kind="byte-identical mirror",
         loader="hf_json", tier="core", notes="Official gated (manual). LFS sha256 of tokenizer.json and tokenizer.model identical to official."),
    dict(name="gemma-2", provider="Google", family="Gemma 2 SentencePiece 256k", models="Gemma 1/2",
         year=2024, repo="unsloth/gemma-2-2b-it", official="google/gemma-2-2b-it", mirror_kind="byte-identical mirror",
         loader="hf_json", tier="core", notes="Official gated (manual). LFS sha256 identical to official."),
    dict(name="mt5", provider="Google", family="mT5 SentencePiece Unigram 250k", models="mT5, mT0",
         year=2020, repo="google/mt5-small", official=None, mirror_kind=None, loader="spm", hf_class="T5Tokenizer", tier="core",
         notes="Native SentencePiece model (spiece.model)."),
    dict(name="byt5", provider="Google", family="ByT5 raw UTF-8 bytes", models="ByT5",
         year=2021, repo="google/byt5-small", official=None, mirror_kind=None, loader="bytes", hf_class="ByT5Tokenizer", tier="core",
         notes="Byte baseline: id = byte + 3 (pad/eos/unk = 0/1/2); vocab 384 incl. 125 sentinel ids."),
    dict(name="mbert", provider="Google", family="mBERT WordPiece 119k (cased)", models="bert-base-multilingual-cased",
         year=2018, repo="google-bert/bert-base-multilingual-cased", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="WordPiece; BERT pre-tokenization splits punctuation, so decode(encode(x)) is not lossless by design."),
    dict(name="muril", provider="Google", family="MuRIL WordPiece 197k", models="MuRIL (17 Indian languages incl. Urdu)",
         year=2021, repo="google/muril-base-cased", official=None, mirror_kind=None, loader="auto", hf_class="BertTokenizer", tier="core",
         notes="Only vocab.txt is published; loaded via transformers BertTokenizer (fast WordPiece; tokenizer_config: lowercase=False, strip_accents=False)."),
    # ---------------- Qwen ----------------
    dict(name="qwen-3", provider="Alibaba Qwen", family="Qwen2/3 byte-level BPE 151k", models="Qwen3 (and Qwen3-Next)",
         year=2025, repo="Qwen/Qwen3-0.6B", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="qwen-2.5", provider="Alibaba Qwen", family="Qwen2/3 byte-level BPE 151k", models="Qwen2, Qwen2.5 (also MiMo, ZabaanAI-Urdu-3B)",
         year=2024, repo="Qwen/Qwen2.5-0.5B", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="qwen-3.5", provider="Alibaba Qwen", family="Qwen3.5 byte-level BPE 248k", models="Qwen3.5 (0.8B-397B)",
         year=2026, repo="Qwen/Qwen3.5-0.8B", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026", notes="Official."),
    dict(name="qwen-3.8", provider="Alibaba Qwen", family="Qwen3.8 byte-level BPE 248k", models="Qwen3.8 (27B, Flash-Next, 2.4T-A95B)",
         year=2026, repo="Qwen/Qwen3.8-27B", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026",
         notes="Official; newest Qwen release on the hub at collection time (2026-08)."),
    # ---------------- DeepSeek ----------------
    dict(name="deepseek-v3", provider="DeepSeek", family="DeepSeek-V3 byte-level BPE 128k", models="DeepSeek-V3, V3.1, V3.2",
         year=2024, repo="deepseek-ai/DeepSeek-V3", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="deepseek-r1", provider="DeepSeek", family="DeepSeek-V3 byte-level BPE 128k", models="DeepSeek-R1",
         year=2025, repo="deepseek-ai/DeepSeek-R1", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="deepseek-v4", provider="DeepSeek", family="DeepSeek-V4 byte-level BPE 129k", models="DeepSeek-V4-Flash, V4-Pro (byte-identical file)",
         year=2026, repo="deepseek-ai/DeepSeek-V4-Flash", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026", notes="Official."),
    dict(name="deepseek-v4.1", provider="DeepSeek", family="DeepSeek-V4 byte-level BPE 129k", models="DeepSeek-V4.1-Flash",
         year=2026, repo="deepseek-ai/DeepSeek-V4.1-Flash", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026",
         notes="Official; newest DeepSeek release on the hub at collection time (2026-09)."),
    # ---------------- Mistral ----------------
    dict(name="mistral-nemo", provider="Mistral AI", family="Tekken (tiktoken-style BPE, 131k)", models="Mistral NeMo, Pixtral, Ministral 8B, Mistral Small 3.x",
         year=2024, repo="mistralai/Mistral-Nemo-Instruct-2407", official=None, mirror_kind=None, loader="hf_json", tier="core",
         crosscheck_tekken=True, notes="Official, ungated. HF tokenizer.json cross-checked against the native tekken.json with a reference BPE."),
    dict(name="mistral-small-4", provider="Mistral AI", family="Tekken (tekken.json v15, 131k)", models="Mistral Small 4, Mistral Medium 3.5 (byte-identical files)",
         year=2026, repo="mistralai/Mistral-Small-4-119B-2603", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026",
         crosscheck_tekken=True, notes="Official, ungated."),
    # ---------------- Cohere ----------------
    dict(name="command-r", provider="Cohere", family="Command-R / Aya Expanse byte-level BPE 256k", models="Command-R v01, Command-R+, Aya Expanse 8B/32B (identical tokenizer.json)",
         year=2024, repo="adamo1139/aya-expanse-8b-ungated", official="CohereLabs/aya-expanse-8b", mirror_kind="byte-identical mirror",
         loader="hf_json", tier="core",
         notes="Official repos gated (auto). Mirror tokenizer.json LFS sha256 == CohereLabs/aya-expanse-8b == CohereForAI/c4ai-command-r-v01."),
    dict(name="command-r7b", provider="Cohere", family="Command R7B byte-level BPE 256k", models="Command R7B (12-2024), Aya Vision",
         year=2024, repo="mlx-community/c4ai-command-r7b-12-2024-bf16", official="CohereLabs/c4ai-command-r7b-12-2024",
         mirror_kind="byte-identical mirror", loader="hf_json", tier="core", notes="Official gated (auto). LFS sha256 identical."),
    dict(name="command-a-plus", provider="Cohere", family="Command A+ (05-2026) byte-level BPE 255k", models="Command A+ 05-2026, North-Mini-Code-1.0 (byte-identical file)",
         year=2026, repo="CohereLabs/command-a-plus-05-2026-bf16", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026",
         notes="Official, ungated (Apache-2.0)."),
    dict(name="tiny-aya", provider="Cohere", family="Tiny Aya byte-level BPE 261k", models="Tiny Aya Global/Earth/Fire/Water (3.35B, 2026)",
         year=2026, repo="mlx-community/tiny-aya-global-8bit-mlx", official="CohereLabs/tiny-aya-global",
         mirror_kind="mirror (bytes differ)", loader="hf_json", tier="frontier-2026",
         notes="Official gated (auto). Mirror tokenizer.json is 187 bytes smaller than official; content equality NOT verifiable without login."),
    # ---------------- BigScience / Meta encoders ----------------
    dict(name="bloom", provider="BigScience", family="BLOOM byte-level BPE 250k", models="BLOOM, BLOOMZ",
         year=2022, repo="bigscience/bloom-560m", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="xlm-r", provider="Meta", family="XLM-R SentencePiece Unigram 250k", models="XLM-RoBERTa (base/large/XL)",
         year=2019, repo="FacebookAI/xlm-roberta-base", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="HF ids = fairseq ids (spm id + 1, specials remapped)."),
    dict(name="nllb-200", provider="Meta", family="NLLB-200 SentencePiece BPE 256k", models="NLLB-200 (incl. urd_Arab, pnb_Arab)",
         year=2022, repo="facebook/nllb-200-distilled-600M", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    # ---------------- Other frontier / open families ----------------
    dict(name="kimi-k2", provider="Moonshot AI", family="Kimi tiktoken BPE 160k", models="Kimi K2, K2.5, K2.6, K3, Kimi-Linear (identical tiktoken.model)",
         year=2025, repo="moonshotai/Kimi-K2-Instruct", official=None, mirror_kind=None, loader="tiktoken_ranks", tier="core",
         notes="Only tiktoken.model + remote-code tokenizer are published. Remote code is NOT executed: the rank file is converted to a "
               "tokenizers BPE here and verified token-for-token against a pure-Python tiktoken-algorithm reference."),
    dict(name="glm-4.5", provider="Zhipu AI / Z.ai", family="GLM-4.5 BPE 151k", models="GLM-4.5, 4.5-Air, 4.6, 4.7 (byte-identical files)",
         year=2025, repo="zai-org/GLM-4.5", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="glm-5", provider="Zhipu AI / Z.ai", family="GLM-5 byte-level BPE 155k", models="GLM-5, 5.1-5.3, GLM-4.7-Flash (same file size; 5 and 5.3 byte-identical)",
         year=2026, repo="zai-org/GLM-5", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026", notes="Official."),
    dict(name="phi-4", provider="Microsoft", family="Phi-4 (cl100k-derived) 100k", models="Phi-4, Phi-4-reasoning",
         year=2024, repo="microsoft/phi-4", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="phi-4-mini", provider="Microsoft", family="Phi-4-mini (o200k-derived) 200k", models="Phi-4-mini, Phi-4-multimodal",
         year=2025, repo="microsoft/Phi-4-mini-instruct", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="falcon-3", provider="TII", family="Falcon3 BPE 131k", models="Falcon3 1B-10B",
         year=2024, repo="tiiuae/Falcon3-7B-Instruct", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="falcon-h1", provider="TII", family="Falcon-H1 BPE 261k (34B)", models="Falcon-H1-34B",
         year=2025, repo="tiiuae/Falcon-H1-34B-Instruct", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="Falcon-H1 sizes use different vocab sizes; the 34B has the largest."),
    dict(name="eurollm", provider="UTTER (EU)", family="EuroLLM SentencePiece BPE 128k", models="EuroLLM-1.7B/9B/22B",
         year=2024, repo="utter-project/EuroLLM-9B-Instruct-2512", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="Official ungated 2512 re-release; its tokenizer.model LFS sha256 == the gated 2024 EuroLLM-9B-Instruct tokenizer.model."),
    dict(name="apertus", provider="Swiss AI", family="Apertus BPE 131k", models="Apertus 8B/70B (2509)",
         year=2025, repo="swiss-ai/Apertus-8B-Instruct-2509", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="sarvam-1", provider="Sarvam AI", family="Sarvam-1 SentencePiece 68k (Indic)", models="Sarvam-1 2B",
         year=2024, repo="sarvamai/sarvam-1", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official; Indic-focused 2B model."),
    dict(name="sarvam-m", provider="Sarvam AI", family="Sarvam-M (Mistral Small 3.1 Tekken)", models="Sarvam-M 24B",
         year=2025, repo="sarvamai/sarvam-m", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="sarvam-30b", provider="Sarvam AI", family="Sarvam 2026 tokenizer 262k (Gemma-style SentencePiece BPE)", models="Sarvam-30B, Sarvam-105B (byte-identical file)",
         year=2026, repo="sarvamai/sarvam-30b", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026", notes="Official."),
    dict(name="indicbert-v2", provider="AI4Bharat", family="IndicBERT v2 WordPiece 250k", models="IndicBERT v2 (24 Indic languages incl. Urdu)",
         year=2022, repo="ai4bharat/IndicBERTv2-MLM-only", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="IndicTrans2 repos are gated (auto) and need remote code -> skipped; IndicBERT v2 represents AI4Bharat."),
    dict(name="minimax-m2", provider="MiniMax", family="MiniMax BPE 200k", models="MiniMax-M1, M2, M2.5, M2.7",
         year=2025, repo="MiniMaxAI/MiniMax-M2", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="minimax-m3", provider="MiniMax", family="MiniMax-M3 byte-level BPE 200k", models="MiniMax-M3",
         year=2026, repo="MiniMaxAI/MiniMax-M3", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026", notes="Official."),
    dict(name="grok-1", provider="xAI", family="Grok-1 SentencePiece 131k", models="Grok-1",
         year=2024, repo="Xenova/grok-1-tokenizer", official="xai-org/grok-1 (tokenizer.model)", mirror_kind="community conversion",
         loader="hf_json", tier="core", notes="The official xai-org/grok-1 repo lists no tokenizer file at root; community conversion used."),
    dict(name="claude-legacy", provider="Anthropic", family="Claude 1/2 BPE 65k (legacy, public)", models="Claude 1.x/2.x only - NOT Claude 3+",
         year=2023, repo="Xenova/claude-tokenizer", official="anthropic-tokenizer-typescript claude.json", mirror_kind="community conversion",
         loader="hf_json", tier="reference", notes="Anthropic has not published tokenizers for Claude 3 or later; this legacy one is included only for completeness."),
    dict(name="nemotron-3", provider="NVIDIA", family="Nemotron 3 BPE 131k", models="Nemotron 3 Nano (and Nemotron-Nano-v2 family)",
         year=2025, repo="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="olmo-2", provider="Ai2", family="OLMo 2/3 (cl100k-derived) 100k", models="OLMo 2, Olmo 3",
         year=2024, repo="allenai/OLMo-2-1124-7B", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="hunyuan", provider="Tencent", family="Hunyuan byte-level BPE 128k", models="Hunyuan-7B/A13B (2025)",
         year=2025, repo="tencent/Hunyuan-7B-Instruct", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="ernie-4.5", provider="Baidu", family="ERNIE 4.5 SentencePiece 103k", models="ERNIE 4.5",
         year=2025, repo="baidu/ERNIE-4.5-0.3B-PT", official=None, mirror_kind=None, loader="hf_json", tier="core", notes="Official."),
    dict(name="granite-4.2", provider="IBM", family="Granite 4.x (cl100k-derived) 100k", models="Granite 4.x",
         year=2026, repo="ibm-granite/granite-4.2-8b", official=None, mirror_kind=None, loader="hf_json", tier="frontier-2026", notes="Official."),
    # ---------------- Urdu-specific (vocab-extended) ----------------
    dict(name="urdu-llama3-almanach", provider="Inria ALMAnaCH", family="Llama-3-8B-mono-Urdu (Urdu-adapted vocab)", models="almanach/Llama-3-8B-mono-Urdu",
         year=2025, repo="almanach/Llama-3-8B-mono-Urdu", official=None, mirror_kind=None, loader="hf_json", tier="urdu",
         notes="Candidate Urdu-adapted LLM tokenizer; vocab change vs Llama 3 is measured, not assumed."),
    dict(name="urdu-llama2-almanach", provider="Inria ALMAnaCH", family="Llama-2-7B-mono-Urdu (Urdu-adapted vocab)", models="almanach/Llama-2-7B-mono-Urdu",
         year=2025, repo="almanach/Llama-2-7B-mono-Urdu", official=None, mirror_kind=None, loader="hf_json", tier="urdu",
         notes="Candidate Urdu-adapted LLM tokenizer; vocab change vs Llama 2 is measured, not assumed."),
    dict(name="urdu-llama3.2-custom", provider="community (SabahNawab)", family="Llama-3.2-3B + custom Urdu tokens", models="SabahNawab/Meta_Llama_3.2_3B_Urdu_Custom_Tokenizer",
         year=2025, repo="SabahNawab/Meta_Llama_3.2_3B_Urdu_Custom_Tokenizer", official=None, mirror_kind=None, loader="hf_json", tier="urdu",
         redundant_override="its extra tokens are added tokens written in the byte-level alphabet and can never match raw text, "
                            "so it always encodes like llama-3",
         notes="Candidate; small community project."),
    dict(name="urdu-llama-bilal", provider="community (BilalKhan1)", family="Llama + Urdu tokenizer", models="BilalKhan1/llama-urdu-tokenizer",
         year=2024, repo="BilalKhan1/llama-urdu-tokenizer", official=None, mirror_kind=None, loader="hf_json", tier="urdu", notes="Candidate."),
    dict(name="roberta-urdu", provider="UrduHack", family="RoBERTa-Urdu byte-level BPE 52k (monolingual Urdu)", models="urduhack/roberta-urdu-small",
         year=2020, repo="urduhack/roberta-urdu-small", official=None, mirror_kind=None, loader="auto", hf_class="RobertaTokenizer", tier="urdu",
         notes="Monolingual Urdu tokenizer (not an LLM); reference for a script-matched vocabulary. Only vocab.json+merges.txt published; loaded via transformers RobertaTokenizer."),
    dict(name="urdu-bert-64k", provider="community (farahadeeba)", family="Urdu BERT WordPiece 64k", models="farahadeeba/urdu-bert-64k",
         year=2026, repo="farahadeeba/urdu-bert-64k", official=None, mirror_kind=None, loader="hf_json", tier="urdu", notes="Candidate."),
    dict(name="alif-1.0", provider="Traversaal.ai", family="Alif-1.0-8B (Llama 3.1 tokenizer)", models="large-traversaal/Alif-1.0-8B-Instruct",
         year=2025, repo="large-traversaal/Alif-1.0-8B-Instruct", official=None, mirror_kind=None, loader="hf_json", tier="urdu",
         notes="Strong Urdu LLM; checked whether its vocab is extended."),
    dict(name="qalb-1.0", provider="Qalb (enstazao)", family="Qalb-1.0-8B (Llama 3.1 tokenizer)", models="enstazao/Qalb-1.0-8B-Instruct",
         year=2026, repo="enstazao/Qalb-1.0-8B-Instruct", official=None, mirror_kind=None, loader="hf_json", tier="urdu",
         notes="Strong Urdu LLM; checked whether its vocab is extended."),
    dict(name="urdu-gpt2-20k", provider="community (aariciah)", family="Urdu GPT-2 BPE 20k (Urdu-only vocab, NFKC+lowercase)",
         models="aariciah/gpt2-urdu-20k-lc", year=2026, repo="aariciah/gpt2-urdu-20k-lc", official=None, mirror_kind=None,
         loader="hf_json", tier="urdu",
         notes="Found by scripts/urdu_scan.py: the most token-efficient Urdu-native tokenizer in the scan among those that round-tripped "
               "the 5 single-line core paragraphs exactly. Normalizer NFKC+Lowercase and no byte fallback; on real corpus text it is "
               "NOT lossless (see `lossless` and `roundtrip_battery`)."),
    dict(name="lfm2", provider="Liquid AI", family="LFM2/LFM2.5 byte-level BPE 65k", models="LFM2, LFM2.5",
         year=2025, repo="LiquidAI/LFM2.5-1.2B-Instruct", official=None, mirror_kind=None, loader="hf_json", tier="core",
         notes="Official. Included mainly as the base for the Pashto-extended LFM2.5 below."),
    dict(name="pashto-lfm2.5", provider="community (nassimjp)", family="LFM2.5 + Pashto vocabulary extension",
         models="nassimjp/LFM2.5-2.6B-Pashto-Zi", year=2026, repo="nassimjp/LFM2.5-2.6B-Pashto-Zi", official=None, mirror_kind=None,
         loader="hf_json", tier="regional",
         notes="Found by scripts/urdu_scan.py: the most-downloaded LLM in the scan with a real Perso-Arabic (Pashto) vocabulary extension."),
    dict(name="sindhi-xlmr", provider="community (Kashif786)", family="XLM-R + Sindhi vocabulary extension (Unigram 265k)",
         models="Kashif786/xlm-roberta-base-sindhi-extended", year=2026, repo="Kashif786/xlm-roberta-base-sindhi-extended", official=None,
         mirror_kind=None, loader="hf_json", tier="regional",
         notes="Found by scripts/urdu_scan.py: XLM-R extended for Sindhi (Perso-Arabic script). Exact on the 5 single-line core "
               "paragraphs only; on real corpus text it is NOT lossless, like XLM-R (see `lossless` and `roundtrip_battery`)."),
    # ---------------- Project reference ----------------
    dict(name="hindko-probe-bpe32k", provider="this project", family="SentencePiece BPE 32k trained on the whole Hindko corpus", models="-",
         year=2026, repo=r"local:F:\Hindko\_pipeline\_tokenizer_probe\hindko_bpe32k.model", official=None, mirror_kind=None,
         loader="local_spm", tier="reference",
         notes="LEAKY REFERENCE, NOT A FAIR COMPETITOR: trained on the whole corpus including any future test split."),
]

ALLOW = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
         "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "tiktoken.model", "tekken.json"]
ALLOW_IF_NO_JSON = ["vocab.txt", "vocab.json", "merges.txt"]
ALLOW_EXTRA = {"tiktoken_ranks": ["tokenization_kimi.py"]}  # read as TEXT to copy the split regex; never imported/executed

# Things that were looked for but are NOT baselines (reason recorded; nothing here was logged into).
NOT_COLLECTED = [
    "Cohere Command A (03-2025): official repo gated (auto). The only full ungated copy checked (unsloth/c4ai-command-a-03-2025) ships a "
    "tokenizer.json that is byte-identical to Command R7B's, not to official Command A's (20,124,922 vs 19,597,349 bytes), so it was not used. "
    "Command A+ (05-2026, official, ungated) is included instead.",
    "AI4Bharat IndicTrans2 (indic-en / en-indic 1B): gated (auto), and its tokenizer needs remote code (tokenization_indictrans.py). Skipped; "
    "IndicBERT v2 represents AI4Bharat.",
    "AI4Bharat IndicBERT v3 (2025-12): gated (auto). Skipped.",
    "Swiss AI Apertus v1.5 (2026-07): gated (auto). Apertus 2509 (ungated) is included.",
    "xAI Grok-2: the official tokenizer.tok.json was downloaded (files/grok-2/) but NOT converted. It lists byte tokens and ids but not the "
    "pre-tokenizer regex ('word_split': 'V1' refers to code outside the file), so any conversion would be unverifiable. "
    "Grok-1 (community conversion) is included.",
    "Not public at all: Gemini (per the Gemma 3 technical report, Gemma 3 uses the same tokenizer as Gemini 2.0; not verifiable here), Claude 3 and later (only the legacy "
    "Claude 1/2 tokenizer is public and is included as a reference), and any OpenAI encoding newer than o200k_base/o200k_harmony "
    "(none was found on the hub).",
    "Meta Llama 3/4 and Google Gemma 2/3 official repos: gated (manual). Byte-identical ungated mirrors were used; see each row.",
    "UrduLLaMA: no repo found on the hub (searched 'UrduLlama', 'urdu-llama'). Urdu-Mistral: the 'urdu-mistral' search returned only "
    "small fine-tunes and LoRA adapters (the-usan/*, Maan23/*, duaafatima/*); their tokenizers were NOT individually checked. "
    "Alif-1.0, Qalb-1.0 and almanach mono-Urdu were checked: their vocabularies are NOT extended.",
    "Urdu/Perso-Arabic hub scan: 123 distinct new tokenizer signatures were found, but only the 45 with the highest downloads+likes "
    "were downloaded and measured (scripts/_urdu_scan.json); the remaining low-download repos were not examined.",
]
