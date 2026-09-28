"""Search the hub for ungated mirrors and Urdu-specific models (metadata only)."""
import json
import os
import sys

os.environ.setdefault("HF_HOME", r"F:\Hindko\_tokenizer\hf_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import HfApi

api = HfApi()
queries = sys.argv[1:]
res = {}
for q in queries:
    rows = []
    try:
        for m in api.list_models(search=q, sort="downloads", limit=40,
                                 expand=["gated", "downloads", "likes", "createdAt", "pipeline_tag", "tags"]):
            rows.append({"id": m.id, "gated": m.gated, "downloads": m.downloads, "likes": m.likes,
                         "created": str(m.created_at)[:10], "pipe": m.pipeline_tag})
    except Exception as e:  # noqa: BLE001
        rows.append({"error": str(e)[:200]})
    res[q] = rows
    print(f"== {q}")
    for r in rows:
        print("  ", r)
json.dump(res, open("_search_" + "_".join(x.replace('/', '-') for x in queries)[:60] + ".json", "w", encoding="utf-8"), indent=1)
