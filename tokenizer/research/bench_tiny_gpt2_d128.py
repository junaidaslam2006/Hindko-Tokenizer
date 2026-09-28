# -*- coding: utf-8 -*-
"""Extra throughput points: d=128 models, 1 thread (companion to bench_tiny_gpt2.py)."""
import json, os, sys
sys.argv = [sys.argv[0], "1"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_tiny_gpt2 as b
res = []
for c in [(8192, 128, 4, 4, 256, 8), (16384, 128, 4, 4, 256, 8), (32768, 128, 4, 4, 256, 8)]:
    r = b.bench(*c); print(json.dumps(r), flush=True); res.append(r)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_tiny_gpt2_d128_t1.json"), "w") as f:
    json.dump({"note": "1 thread, FTZ on, machine shared; indicative", "results": res}, f, indent=1)
