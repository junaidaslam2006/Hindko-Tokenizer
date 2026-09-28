# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 1: search the Hugging Face hub (anonymous, metadata only).

Three kinds of search, all recorded in ../_search.json:
  A. keyword search on repo ids (Hindko, Hindko-adjacent Lahnda varieties, Urdu, Shahmukhi, Saraiki, Pashto, ...)
  B. language-tag filters (ISO 639 codes: hno, hnd, ur, pnb, skr, lah, phr, ps, sd, ks, bal, ...), text pipelines only
  C. author listings: the newest repos of frontier / popular open-model labs (created 2025-01-01 or later)
  D. an explicit list of named frontier / regional repos (task list + web search leads)
For every repo, the root and first-level-subfolder tokenizer files are listed from the `siblings` metadata.
No file is downloaded here. No token is sent (token=False).
"""
import json
import os
import sys
import time

os.environ["HF_HOME"] = r"F:\Hindko\_tokenizer\hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import HfApi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "_search.json")
api = HfApi(token=False)

KEYWORDS = [
    # Hindko and its neighbours (Lahnda cluster)
    "hindko", "hindku", "hinko", "hazarewal", "hazara", "potohari", "pothwari", "pothohari", "pahari", "lahnda", "lahanda",
    "saraiki", "seraiki", "siraiki", "multani", "shahmukhi", "punjabi", "panjabi", "pnb",
    # Pakistan's other Perso-Arabic-script languages
    "urdu", "pashto", "pushto", "pukhto", "sindhi", "kashmiri", "balochi", "baluchi", "brahui", "khowar", "shina", "gojri",
    "pakistan", "peshawar", "nastaliq", "nastaleeq", "perso-arabic", "persoarabic",
    # broader
    "indic", "indo-aryan", "indoaryan", "south-asian", "southasian", "bharat",
]
LANG_TAGS = ["hno", "hnd", "ur", "urd", "pnb", "skr", "lah", "phr", "ps", "pus", "pbt", "pbu", "pst", "sd", "snd", "ks", "kas",
             "bal", "bcc", "bgp", "khw", "scl", "gju", "pa", "pan", "trw", "bft"]
AUTHORS = [
    "openai", "meta-llama", "facebook", "google", "Qwen", "deepseek-ai", "mistralai", "CohereLabs", "CohereForAI", "microsoft",
    "nvidia", "ibm-granite", "allenai", "HuggingFaceTB", "tiiuae", "zai-org", "THUDM", "moonshotai", "MiniMaxAI", "tencent",
    "baidu", "xai-org", "sarvamai", "ai4bharat", "krutrim-ai-labs", "LiquidAI", "stepfun-ai", "XiaomiMiMo", "inclusionAI",
    "ByteDance-Seed", "swiss-ai", "utter-project", "apple", "amazon", "Snowflake", "internlm", "openbmb", "rednote-hilab",
    "meituan-longcat", "inceptionai", "humain-ai", "QCRI", "SeaLLMs", "aisingapore", "sail", "EleutherAI", "Zyphra", "arcee-ai",
    "ServiceNow-AI", "BSC-LT", "PrimeIntellect", "NousResearch", "jinaai", "BAAI", "Alibaba-NLP", "intfloat", "LGAI-EXAONE",
    "upstage", "kakaocorp", "naver-hyperclovax", "SakanaAI", "tokyotech-llm", "sbintuitions", "stabilityai", "databricks",
    "Salesforce", "01-ai", "bigscience", "CohereLabsCommunity", "cognitivecomputations", "MBZUAI", "MBZUAI-Paris", "G42",
    "large-traversaal", "orature", "urduhack", "l3cube-pune", "HPLT", "Helsinki-NLP", "cis-lmu", "MaLA-LM", "theschoolofai",
    "BharatGPT", "CoRover", "soketlabs", "LingoIITGN", "nassimjp", "enstazao", "almanach", "unsloth", "mlx-community",
]
NAMED = [
    # task list, frontier / popular 2025-2026 (the ones already in baselines/manifest.json are re-checked by blob id)
    "openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b", "xai-org/grok-2", "alvarobartt/grok-2-tokenizer",
    "zai-org/GLM-4.5", "zai-org/GLM-4.6", "zai-org/GLM-4.7", "zai-org/GLM-4.5-Air", "zai-org/GLM-4.6V", "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M1-80k", "MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2.1", "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7",
    "MiniMaxAI/MiniMax-Text-01", "tencent/Hunyuan-A13B-Instruct", "tencent/Hunyuan-MT-7B", "tencent/Hunyuan-7B-Instruct",
    "tencent/Hunyuan-MT-Chimera-7B", "ibm-granite/granite-4.0-h-small", "ibm-granite/granite-4.0-micro", "ibm-granite/granite-4.0-1b",
    "ibm-granite/granite-4.0-350m", "ibm-granite/granite-3.3-8b-instruct", "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-32B-Think", "allenai/OLMo-2-1124-7B", "HuggingFaceTB/SmolLM3-3B", "HuggingFaceTB/SmolLM2-1.7B",
    "microsoft/phi-4", "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-flash-reasoning", "microsoft/Phi-5", "microsoft/phi-5",
    "microsoft/Phi-4-multimodal-instruct", "CohereLabs/c4ai-command-a-03-2025", "CohereLabs/command-a-reasoning-08-2025",
    "CohereLabs/command-a-translate-08-2025", "CohereLabs/command-a-vision-07-2025", "CohereLabs/aya-vision-8b",
    "CohereLabs/aya-vision-32b", "CohereLabs/aya-expanse-8b", "CohereLabs/tiny-aya-global", "CohereLabs/aya-101",
    "krutrim-ai-labs/Krutrim-1-instruct", "krutrim-ai-labs/Krutrim-2-instruct", "krutrim-ai-labs/Chitrarth",
    "sarvamai/sarvam-1", "sarvamai/sarvam-m", "sarvamai/sarvam-30b", "sarvamai/sarvam-105b", "sarvamai/sarvam-translate",
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base", "ai4bharat/Airavata", "ai4bharat/indictrans2-indic-en-1B", "ai4bharat/indictrans2-en-indic-1B",
    "ai4bharat/indictrans2-indic-indic-1B", "ai4bharat/indictrans2-indic-en-dist-200M", "ai4bharat/IndicBARTSS", "ai4bharat/IndicBART",
    "ai4bharat/indic-bert", "ai4bharat/IndicBERTv2-MLM-only", "ai4bharat/IndicBERT-v3-270M",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct", "unsloth/Llama-4-Maverick-17B-128E-Instruct",
    "google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it", "unsloth/gemma-3n-E2B-it", "google/gemma-3-270m", "google/embeddinggemma-300m",
    "google/t5gemma-2b-2b-ul2", "google/madlad400-3b-mt", "google/umt5-small", "google/mt5-small", "google/rembert", "google/canine-s",
    "google/gemma-4-E2B-it", "google/gemma-4-31B-it", "google/translategemma-4b-it", "google/vaultgemma-1b", "google/medgemma-4b-it",
    "mistralai/Mistral-Medium-3.5-128B", "mistralai/Magistral-Small-2509", "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Mistral-Large-3-675B-Instruct-2512", "mistralai/Devstral-Small-2507", "mistralai/Voxtral-Mini-3B-2507",
    "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-OCR", "Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-0905", "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K3",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct", "tiiuae/Falcon-H1-0.5B-Instruct", "tiiuae/Falcon-H1-1.5B-Instruct",
    "tiiuae/Falcon-H1-3B-Instruct", "tiiuae/Falcon-H1-7B-Instruct", "tiiuae/Falcon-H1-34B-Instruct", "tiiuae/Falcon-H1R-7B",
    "tiiuae/Falcon-Arabic-7B-Instruct", "tiiuae/falcon-mamba-7b", "tiiuae/Falcon-E-3B-Instruct", "tiiuae/Falcon3-7B-Instruct",
    "nvidia/Nemotron-H-8B-Base-8K", "nvidia/Nemotron-H-4B-Instruct-128K", "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nvidia/Nemotron-Mini-4B-Instruct",
    "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct", "nvidia/Mistral-NeMo-Minitron-8B-Base", "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
    "XiaomiMiMo/MiMo-V2.5", "XiaomiMiMo/MiMo-7B-RL", "stepfun-ai/Step-3.5-Flash", "stepfun-ai/step3", "baidu/ERNIE-4.5-0.3B-PT",
    "baidu/ERNIE-4.5-21B-A3B-PT", "baidu/ERNIE-5", "inclusionAI/Ling-lite-1.5", "inclusionAI/Ling-mini-2.0", "inclusionAI/Ring-mini-2.0",
    "ByteDance-Seed/Seed-OSS-36B-Instruct", "LiquidAI/LFM2-2.6B", "LiquidAI/LFM2.5-1.2B-Instruct", "LiquidAI/LFM2-8B-A1B",
    "swiss-ai/Apertus-8B-Instruct-2509", "swiss-ai/Apertus-70B-Instruct-2509", "utter-project/EuroLLM-9B-Instruct-2512",
    "utter-project/EuroLLM-22B-Instruct-2512", "BSC-LT/salamandra-7b-instruct", "BSC-LT/ALIA-40b", "openbmb/MiniCPM4-8B",
    "openbmb/MiniCPM4.1-8B", "rednote-hilab/dots.llm1.inst", "meituan-longcat/LongCat-Flash-Chat", "internlm/internlm3-8b-instruct",
    "inceptionai/jais-family-6p7b-chat", "inceptionai/Jais-2-8B-Chat", "humain-ai/ALLaM-7B-Instruct-preview", "QCRI/Fanar-1-9B-Instruct",
    "aisingapore/Gemma-SEA-LION-v4-27B-IT", "SeaLLMs/SeaLLMs-v3-7B-Chat", "sail/Sailor2-8B-Chat", "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "upstage/SOLAR-10.7B-Instruct-v1.0", "arcee-ai/AFM-4.5B", "ServiceNow-AI/Apriel-1.5-15b-Thinker", "Zyphra/ZAYA1-base",
    "EleutherAI/pythia-160m", "PrimeIntellect/INTELLECT-3", "NousResearch/Hermes-4-14B", "facebook/xglm-564M", "facebook/xlm-v-base",
    "facebook/mbart-large-50", "facebook/m2m100_418M", "facebook/nllb-200-distilled-600M", "facebook/seamless-m4t-v2-large",
    "facebook/MobileLLM-R1-950M", "cis-lmu/glot500-base", "MaLA-LM/mala-500-10b-v2", "MaLA-LM/emma-500-llama3.1-8b-mono",
    "microsoft/mdeberta-v3-base", "microsoft/Multilingual-MiniLM-L12-H384", "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-large", "BAAI/bge-m3", "Alibaba-NLP/gte-multilingual-base", "jinaai/jina-embeddings-v3",
    "Snowflake/snowflake-arctic-embed-l-v2.0", "openai/whisper-large-v3", "openai/whisper-large-v3-turbo",
    # Urdu / Punjabi / regional leads from web search
    "orature/ALIF-Base-100M", "theschoolofai/BrahmicTokenizer-131K", "large-traversaal/Alif-1.0-8B-Instruct",
    "large-traversaal/Alif-1.0-3B-Instruct", "enstazao/Qalb-1.0-8B-Instruct", "urduhack/roberta-urdu-small", "junaid008/qehwa-pashto-llm",
    "bisma10/hindko-punjabi-urdu-chatbot", "Helsinki-NLP/opus-mt-ur-en", "Helsinki-NLP/opus-mt-en-ur", "Helsinki-NLP/opus-mt-pa-en",
    "Helsinki-NLP/opus-mt-en-pa", "Helsinki-NLP/opus-mt-mul-en", "Helsinki-NLP/opus-mt-en-mul", "l3cube-pune/punjabi-bert",
    "l3cube-pune/hindi-bert-v2", "zirak-ai/pashto-bert-v1", "HPLT/hplt_bert_base_ur", "HPLT/hplt_bert_base_2_0_urd-Arab",
    "HPLT/hplt_bert_base_2_0_pbt-Arab", "HPLT/hplt_bert_base_2_0_snd-Arab", "HPLT/hplt_bert_base_2_0_pnb-Arab",
    "HPLT/hplt_gpt_bert_base_3_0_urd_Arab", "HPLT/hplt_gpt_bert_base_3_0_pbt_Arab", "HPLT/hplt_gpt_bert_base_3_0_snd_Arab",
    "HPLT/hplt_gpt_bert_base_3_0_pnb_Arab", "HPLT/hplt_gpt_bert_base_3_0_skr_Arab", "HPLT/hplt_gpt_bert_base_3_0_hno_Arab",
]

TEXT_PIPES = {None, "text-generation", "fill-mask", "text2text-generation", "translation", "feature-extraction", "sentence-similarity",
              "text-classification", "token-classification", "summarization", "question-answering", "image-text-to-text",
              "zero-shot-classification", "table-question-answering", "text-ranking", "any-to-any"}
TOK_NAMES = ("tokenizer.json", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "tiktoken.model", "tekken.json",
             "vocab.json", "merges.txt", "vocab.txt", "source.spm", "target.spm", "tokenizer.tok.json", "sentencepiece.model",
             "spm.model", "bpe.model")
EXPAND = ["gated", "downloads", "likes", "pipeline_tag", "createdAt", "siblings", "lastModified", "library_name", "tags"]


def tok_files(m):
    out = []
    for s in (m.siblings or []):
        f = s.rfilename
        parts = f.split("/")
        if len(parts) > 2:
            continue
        base = parts[-1]
        if base in TOK_NAMES or base.endswith(".tiktoken") or (base.endswith(".model") and "token" in base.lower()):
            out.append(f)
    return out


def row(m, how):
    return {"repo": m.id, "how": how, "downloads": m.downloads, "likes": m.likes, "pipe": m.pipeline_tag,
            "created": str(m.created_at)[:10] if m.created_at else None,
            "last_modified": str(m.last_modified)[:10] if getattr(m, "last_modified", None) else None,
            "gated": m.gated, "library": getattr(m, "library_name", None),
            "tok_files": tok_files(m), "n_files": len(m.siblings or [])}


def listing(kw, tries=4):
    for t in range(tries):
        try:
            return list(api.list_models(token=False, expand=EXPAND, **kw))
        except Exception as e:  # noqa: BLE001
            print("retry", kw, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(5 * (t + 1))
    return []


def main():
    rows, seen, log = [], {}, []

    def add(ms, how, text_only=False):
        n_new = 0
        for m in ms:
            if text_only and m.pipeline_tag not in TEXT_PIPES:
                continue
            if m.id in seen:
                seen[m.id]["how"] += "|" + how
                continue
            r = row(m, how)
            seen[m.id] = r
            rows.append(r)
            n_new += 1
        log.append({"query": how, "n_results": len(ms), "n_new": n_new})
        print(how, len(ms), "new", n_new, flush=True)

    for q in KEYWORDS:
        add(listing(dict(search=q, sort="downloads", limit=1500)), "kw:" + q)
    for lang in LANG_TAGS:
        add(listing(dict(filter=lang, sort="downloads", limit=1500)), "lang:" + lang, text_only=True)
    for a in AUTHORS:
        ms = listing(dict(author=a, sort="created_at", limit=300))
        ms = [m for m in ms if m.created_at is None or str(m.created_at)[:10] >= "2025-01-01"]
        add(ms, "author:" + a, text_only=True)
    named_missing = []
    for r in NAMED:
        if r in seen:
            seen[r]["how"] += "|named"
            continue
        try:
            m = api.model_info(r, expand=EXPAND, token=False)
            add([m], "named")
        except Exception as e:  # noqa: BLE001
            named_missing.append({"repo": r, "error": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])})
            print("named missing", r, type(e).__name__, flush=True)
    json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "keywords": KEYWORDS, "lang_tags": LANG_TAGS,
               "authors": AUTHORS, "named": NAMED, "named_missing": named_missing, "queries": log,
               "n_repos": len(rows), "n_with_tok_files": sum(1 for r in rows if r["tok_files"]), "repos": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("repos", len(rows), "with tokenizer files", sum(1 for r in rows if r["tok_files"]), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
