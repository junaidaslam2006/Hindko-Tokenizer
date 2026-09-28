"""List the newest repos of major labs (metadata only) to find 2025-2026 tokenizer releases."""
import os
import re
import sys

os.environ.setdefault("HF_HOME", r"F:\Hindko\_tokenizer\hf_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import HfApi

api = HfApi()
SKIP = re.compile(r"GGUF|gguf|FP8|fp8|NVFP4|AWQ|GPTQ|MLX|mlx|bnb|int4|int8|MXFP4|onnx|ONNX|eagle|Eagle|dflash|DFlash", re.I)
for a in sys.argv[1:]:
    print(f"== {a}")
    try:
        n = 0
        for m in api.list_models(author=a, sort="createdAt", limit=60, expand=["gated", "downloads", "createdAt", "pipeline_tag"]):
            if SKIP.search(m.id):
                continue
            print(f"   {str(m.created_at)[:10]} {m.id} gated={m.gated} dl={m.downloads} pipe={m.pipeline_tag}")
            n += 1
            if n >= 14:
                break
    except Exception as e:  # noqa: BLE001
        print("   ERR", str(e)[:200])
