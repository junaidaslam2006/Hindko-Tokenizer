# -*- coding: utf-8 -*-
"""Encode-speed sanity check (one thread, shared CPU; never used for any decision): PickyBPE event-ordered
encoder with a cold and a warm pretoken cache, vs the HF native A1 reference, on dev_strict and train_D1.

    python speed_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402
import pickybpe_factory as F  # noqa: E402


def timed(fn, texts):
    t0 = time.perf_counter()
    n = sum(len(fn(t)) for t in texts)
    return time.perf_counter() - t0, n


def main():
    from tokenizers import Tokenizer
    out = {}
    for view, split in (("dev_strict", "validation"), ("train_D1", "train")):
        docs, info = C.load_view(view, split)
        texts = [d["text"] for d in docs]
        mb = sum(len(t.encode("utf-8")) for t in texts) / 1e6
        tk = F.load_16k()
        cold, n1 = timed(tk.encode, texts)
        warm, n2 = timed(tk.encode, texts)
        hf = Tokenizer.from_file(os.path.join(HERE, "models", "a1_ref_P1_D1_16k.json"))
        hft, n3 = timed(lambda t: hf.encode(t, add_special_tokens=False).ids, texts)
        out[view] = {"docs": len(texts), "MB": round(mb, 3),
                     "pickybpe_cold_s": round(cold, 2), "pickybpe_cold_MB_per_s": round(mb / cold, 3),
                     "pickybpe_warm_s": round(warm, 2), "pickybpe_warm_MB_per_s": round(mb / warm, 3),
                     "pickybpe_cache_entries": len(tk._cache), "a1_hf_s": round(hft, 2),
                     "a1_hf_MB_per_s": round(mb / hft, 3), "tokens_pickybpe": n1, "tokens_a1": n3}
        assert n1 == n2
        C.log(view, json.dumps(out[view]))
    out["note"] = "one thread, shared laptop CPU, other agents running: sanity only"
    json.dump(out, open(os.path.join(HERE, "verify", "speed_check.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
