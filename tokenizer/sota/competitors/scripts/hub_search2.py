# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 1b: supplementary hub search for leads found by web / GitHub search after stage 1
(Urdu / Pakistan-language projects, and open-weight releases of Jul-Sep 2026 listed by release trackers).
Repos already in ../_search.json are not repeated.  Output: ../_search2.json (same format).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hub_search as S  # noqa: E402

OUT = os.path.join(S.ROOT, "_search2.json")
KEYWORDS = ["markhor", "pakmosaic", "urdulm", "alif-base", "qehwa", "zabaan", "shaheen", "baat", "urdu-gpt", "urdugpt",
            "muse-glimmer", "Nemotron-3.5", "Atria-Dawn", "Dots3", "Ornith", "Hy-MT2", "Nex-N2.5", "Bonsai-2", "Ling-3.0",
            "MiMo-V2.6", "MiniCPM5", "grok-2", "grok-3", "gpt-oss", "o200k", "BrahmicTokenizer", "tokenizer-urdu", "urdu-tokenizer",
            "urdu_tokenizer", "UrduTokenizer", "punjabi-tokenizer", "pashto-tokenizer", "sindhi-tokenizer", "saraiki-tokenizer"]
AUTHORS = ["ma1993", "mahwizzzz", "themohal", "ProximaAI", "orature", "adiled", "hadidev", "NeerjaK", "aariciah", "Kashif786",
           "aakashMeghwar01", "zirak-ai", "tasal9", "junaid008", "meta-models", "DeepReinforce", "nex-agi", "PrismML",
           "dots-studio", "rednote-hilab", "OpenGVLab", "theschoolofai", "bisma10", "Xhaheen", "mirfan899", "Mavkif",
           "traversaal-ai", "large-traversaal", "enstazao", "UrduHack", "salmanpopa", "SLPG", "abdulwaheed1"]
NAMED = ["ProximaAI/PakMosaic", "hadidev/gpt2-urdu-tokenizer-withgpt2", "NeerjaK/Urdu_Model", "meta-models/Muse-Glimmer-30B",
         "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "internlm/Atria-Dawn-Preview", "tencent/Hy-MT2-30B-A3B",
         "XiaomiMiMo/MiMo-V2.6-Flash", "openbmb/MiniCPM5-2B", "alvarobartt/grok-2-tokenizer"]


def main():
    prev = {r["repo"] for r in json.load(open(S.OUT, encoding="utf-8"))["repos"]}
    rows, seen, log = [], set(prev), []

    def add(ms, how, text_only=False):
        n = 0
        for m in ms:
            if (text_only and m.pipeline_tag not in S.TEXT_PIPES) or m.id in seen:
                continue
            seen.add(m.id)
            rows.append(S.row(m, how))
            n += 1
        log.append({"query": how, "n_results": len(ms), "n_new": n})
        print(how, len(ms), "new", n, flush=True)

    for q in KEYWORDS:
        add(S.listing(dict(search=q, sort="downloads", limit=500)), "kw2:" + q)
    for a in AUTHORS:
        add(S.listing(dict(author=a, sort="created_at", limit=300)), "author2:" + a)
    missing = []
    for r in NAMED:
        if r in seen:
            continue
        try:
            add([S.api.model_info(r, expand=S.EXPAND, token=False)], "named2")
        except Exception as e:  # noqa: BLE001
            missing.append({"repo": r, "error": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])})
    json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "keywords": KEYWORDS, "authors": AUTHORS,
               "named": NAMED, "named_missing": missing, "queries": log, "n_repos": len(rows),
               "n_with_tok_files": sum(1 for r in rows if r["tok_files"]), "repos": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("new repos", len(rows), "with tokenizer files", sum(1 for r in rows if r["tok_files"]), "missing", missing, flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
