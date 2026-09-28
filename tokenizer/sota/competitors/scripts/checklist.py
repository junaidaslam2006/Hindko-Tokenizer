# -*- coding: utf-8 -*-
"""Per-repo status of the task's named frontier / regional models: what happened to each one's tokenizer in the sweep.
Output: ../_checklist.json (read by write_md.py)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ITEMS = [
    ("OpenAI gpt-oss (o200k_harmony); newer GPT-5-family encodings are not public", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]),
    ("xAI Grok 2 (the only public xAI tokenizer after Grok-1)", ["xai-org/grok-2", "alvarobartt/grok-2-tokenizer", "unsloth/grok-2"]),
    ("Z.ai GLM-4.5 / 4.6 / 4.7 / 5.x", ["zai-org/GLM-4.5", "zai-org/GLM-4.6", "zai-org/GLM-4.7", "zai-org/GLM-4.7-Flash", "zai-org/GLM-5.2",
                                        "zai-org/GLM-5.3"]),
    ("MiniMax M2.x / M3", ["MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7", "MiniMaxAI/MiniMax-M3"]),
    ("Tencent Hunyuan / Hy-MT", ["tencent/Hunyuan-A13B-Instruct", "tencent/Hunyuan-MT-7B", "tencent/Hy-MT2-1.8B", "tencent/Hy-MT2-30B-A3B"]),
    ("IBM Granite 4.x", ["ibm-granite/granite-4.0-h-small", "ibm-granite/granite-4.0-micro", "ibm-granite/granite-4.1-8b",
                         "ibm-granite/granite-4.2-8b"]),
    ("Ai2 OLMo 3", ["allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-1125-32B"]),
    ("Hugging Face SmolLM3", ["HuggingFaceTB/SmolLM3-3B"]),
    ("Microsoft Phi-5", ["microsoft/Phi-5", "microsoft/phi-5"]),
    ("Cohere Command A / Aya Vision", ["CohereLabs/c4ai-command-a-03-2025", "CohereLabs/command-a-reasoning-08-2025",
                                       "CohereLabs/aya-vision-8b", "CohereLabs/command-a-plus-05-2026-bf16"]),
    ("Krutrim", ["krutrim-ai-labs/Krutrim-1-instruct", "krutrim-ai-labs/Krutrim-2-instruct"]),
    ("Sarvam", ["sarvamai/sarvam-105b", "sarvamai/sarvam-translate", "sarvamai/sarvam-30b"]),
    ("AI4Bharat Airavata", ["ai4bharat/Airavata"]),
    ("AI4Bharat IndicTrans2", ["ai4bharat/indictrans2-indic-en-1B", "ai4bharat/indictrans2-en-indic-dist-200M"]),
    ("AI4Bharat IndicBERT v3", ["ai4bharat/IndicBERT-v3-270M"]),
    ("Meta Llama 4 variants", ["meta-llama/Llama-4-Maverick-17B-128E-Instruct", "unsloth/Llama-4-Maverick-17B-128E-Instruct"]),
    ("Google Gemma 3n", ["google/gemma-3n-E2B-it", "unsloth/gemma-3n-E2B-it"]),
    ("Mistral Medium 3.5", ["mistralai/Mistral-Medium-3.5-128B", "unsloth/Mistral-Medium-3.5-128B"]),
    ("DeepSeek V4 (Pro / Flash-0731)", ["deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash-0731"]),
    ("Qwen3-Next / Qwen3.8-Flash-Next", ["Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3.8-Flash-Next"]),
    ("Moonshot Kimi K2.x", ["moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6"]),
    ("TII Falcon-H1 (all sizes)", ["tiiuae/Falcon-H1-0.5B-Instruct", "tiiuae/Falcon-H1-1.5B-Instruct", "tiiuae/Falcon-H1-3B-Instruct",
                                   "tiiuae/Falcon-H1-7B-Instruct", "tiiuae/Falcon-H1R-7B"]),
    ("NVIDIA Nemotron-H", ["nvidia/Nemotron-H-8B-Base-8K", "nvidia/Nemotron-H-4B-Instruct-128K"]),
    ("Meta Muse Glimmer (2026-08)", ["meta-models/Muse-Glimmer-30B"]),
    ("NVIDIA Nemotron 3.5 Lightning (2026-08)", ["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"]),
    ("Shanghai AI Lab Atria Dawn (2026-09)", ["internlm/Atria-Dawn-Preview"]),
    ("OpenBMB MiniCPM5 (2026-09)", ["openbmb/MiniCPM5-2B"]),
    ("inclusionAI Ling 3.0 (2026-08/09)", ["inclusionAI/Ling-3.0-tiny", "inclusionAI/Ling-3.0-flash"]),
    ("Dots Studio dots3 (2026-08)", ["dots-studio/dots3-note-prev"]),
    ("DeepReinforce Ornith (2026-08)", ["unsloth/Ornith-1.0-9B"]),
    ("Nex AGI Nex-N2.5 (2026-09)", ["nex-agi/Nex-N2.5-mini"]),
    ("Xiaomi MiMo V2.5 / V2.6 (2026-09)", ["XiaomiMiMo/MiMo-V2.5", "XiaomiMiMo/MiMo-V2.6-Flash-RL"]),
    ("UrduLM / ALIF-Base-100M (Urdu, 2026-01)", ["orature/ALIF-Base-100M"]),
    ("ProximaAI urnova-95m / PakMosaic (Pakistan languages)", ["ProximaAI/urnova-95m", "ProximaAI/PakMosaic"]),
    ("BrahmicTokenizer-131K (Indic, 2026-05)", ["theschoolofai/BrahmicTokenizer-131K"]),
    ("Hindko-named hub repo", ["bisma10/hindko-punjabi-urdu-chatbot"]),
    ("Saraiki (themohal)", ["themohal/saraiki-qwen3-8b-cpt", "themohal/saraiki-roberta-base-small-finetuned3", "themohal/trocr-small-saraiki"]),
]


def jl(n):
    p = os.path.join(ROOT, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def base_name(path):
    p = path.replace("\\", "/")
    if p.startswith("baselines/files/_urdu_candidates/"):
        return "2026-09-26 Urdu-scan file of %s" % p.split("/")[3].replace("__", "/")
    if p.startswith("baselines/files/"):
        return "baseline `%s`" % p.split("/")[2]
    if p.startswith("sota/competitors/files/"):
        return "refresh file `%s`" % p.split("/")[3]
    if p.startswith("refresh:"):
        return "refresh file `%s`" % p.split(":", 1)[1].split("/")[0]
    return p


def main():
    per = {}
    for f in ("_probe.json", "_probe2.json"):
        d = jl(f)
        for r in (d or {}).get("per_repo", []):
            per.setdefault(r["repo"], r)
    srch = {r["repo"]: r for f in ("_search.json", "_search2.json") for r in (jl(f) or {}).get("repos", [])}
    missing = {m["repo"]: m for f in ("_search.json", "_search2.json") for m in (jl(f) or {}).get("named_missing", [])}
    dl = jl("_download_log.json") or {}
    screens = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(ROOT, "screen", "*.json"))}
    groups = jl("_groups.json") or {"groups": []}
    g_of = {m["name"]: g for g in groups["groups"] for m in g["members"]}
    tri = {r["sig"]: r for f in ("_triage.json", "_triage2.json", "_triage2b.json") for r in (jl(f) or {}).get("results", [])
           if r["status"] != "needs_download" or r["sig"] not in {}}
    tri_final = {}
    for f in ("_triage.json", "_triage2.json", "_triage2b.json"):
        for r in (jl(f) or {}).get("results", []):
            if r["status"] != "needs_download" or r["sig"] not in tri_final:
                tri_final[r["sig"]] = r
    gated = {g["repo"]: g for g in (jl("_gated.json") or {}).get("repos", [])}
    comp = jl("competitors_all.json")
    rows_by = {r["name"]: r for r in (comp or {}).get("tokenizers", [])}

    def one(repo):
        if repo in missing:
            return "not on the hub (%s)" % missing[repo]["error"].split(":")[0]
        s = srch.get(repo)
        if s is None:
            return "not found by the searches"
        if s.get("gated"):
            g = gated.get(repo)
            if g and g.get("all_identical_to_collected"):
                return "gated (%s); its tokenizer file is byte-identical to %s" % (s["gated"], base_name(g["files"][0]["identical_to"]))
            return "gated (%s); not logged into, not measured" % s["gated"]
        if not s["tok_files"]:
            return "no tokenizer file in the repo"
        p = per.get(repo)
        if p is None:
            return "not probed (non-text pipeline or filtered)"
        if "error" in p:
            return "probe error: %s" % p["error"][:80]
        outs = []
        for d in p["dirs"]:
            where = ("/" + d["dir"]) if d["dir"] else ""
            if d["known_as"]:
                outs.append("%s%s: file byte-identical to %s" % (d["kind"], where, base_name(d["known_as"])))
                continue
            rec = dl.get(d["sig"])
            if rec and "local_dir" in rec:
                name = rec["local_dir"].split("/", 1)[1]
                sc = screens.get(name)
                if sc and sc.get("status") == "ok":
                    g = g_of.get(name)
                    b = sc["screen"]["bytes_per_token"]
                    txt = "%s%s: measured %.3f bytes/token, G1 %d/491" % (d["kind"], where, b, sc["screen"]["g1_pass_docs"])
                    if g and g["baselines"]:
                        txt += ", same encodings as baseline %s" % ", ".join("`%s`" % x for x in g["baselines"])
                    elif g and g["representative"] and g["representative"] != name:
                        txt += ", same encodings as `%s`" % g["representative"]
                    outs.append(txt)
                else:
                    outs.append("%s%s: downloaded, not loadable (%s)" % (d["kind"], where, (sc or {}).get("status", "?")[:80]))
                continue
            t = tri_final.get(d["sig"])
            if t and t["status"] == "identical_behaviour":
                outs.append("%s%s: model section byte-identical to %s (range reads; identical encodings)" % (
                    d["kind"], where, base_name(t["reference"])))
            elif t and t["status"] == "same_vocabulary_sampled":
                outs.append("%s%s: same vocabulary as %s (range-read sample; not measured)" % (
                    d["kind"], where, base_name(t["family_reference"])))
            elif rec and "error" in rec:
                outs.append("%s%s: download error %s" % (d["kind"], where, rec["error"][:60]))
            else:
                outs.append("%s%s: new file, not downloaded" % (d["kind"], where))
        return "; ".join(outs) or "no tokenizer directory recognised"

    out = [{"label": lab, "repos": [{"repo": r, "status": one(r)} for r in repos]} for lab, repos in ITEMS]
    json.dump(out, open(os.path.join(ROOT, "_checklist.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for it in out:
        print("##", it["label"])
        for r in it["repos"]:
            print("   ", r["repo"], "->", r["status"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
