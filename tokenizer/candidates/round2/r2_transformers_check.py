# -*- coding: utf-8 -*-
"""Round 2: HF-native loading check (PLAN 7 tie-breaker a) for every round-2 tokenizer.json candidate.

    python r2_transformers_check.py        -> transformers_check.json

transformers.PreTrainedTokenizerFast(tokenizer_file=...) must give the same ids as tokenizers.Tokenizer on every
dev_strict document and decode them back to the text (as candidates/mingram/logs/transformers_check.log and
candidates/superbpe/verify.py did for Stage 2). dev_strict only; the test split is never read.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_common as RC  # noqa: E402

sys.path.insert(0, RC.EVAL)
import harness as H  # noqa: E402

FILES = {
    RC.IDS["A1-48k"]: os.path.join(RC.R2, "standard", "tok", RC.IDS["A1-48k"], "tokenizer.json"),
    RC.IDS["A10-32k"]: os.path.join(RC.R2, "mingram", "runs", "mingram_P1r3_D2_32768", "tokenizer.json"),
    RC.IDS["A10-48k"]: os.path.join(RC.R2, "mingram", "runs", "mingram_P1r3_D2_49152", "tokenizer.json"),
    RC.IDS["A6-32k"]: os.path.join(RC.R2, "superbpe", "sbpe_32k_t080_p1r3_d2", "tokenizer.json"),
}


def main():
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    from tokenizers import Tokenizer
    import transformers
    from transformers import PreTrainedTokenizerFast
    docs, info = H.load_docs("dev_strict")
    out = {"transformers": transformers.__version__, "tokenizers": __import__("tokenizers").__version__,
           "dataset": {k: info[k] for k in ("name", "sha256", "docs")}, "results": {}}
    for cid, p in FILES.items():
        if not os.path.exists(p):
            out["results"][cid] = {"status": "missing"}
            continue
        tk = Tokenizer.from_file(p)
        ptf = PreTrainedTokenizerFast(tokenizer_file=p)
        mism = rt = 0
        for d in docs:
            a = ptf(d["text"], add_special_tokens=False)["input_ids"]
            if a != tk.encode(d["text"], add_special_tokens=False).ids:
                mism += 1
            rt += ptf.decode(a, skip_special_tokens=False, clean_up_tokenization_spaces=False) == d["text"]
        out["results"][cid] = {"path": p, "sha256": RC.sha256_file(p), "len_tokenizer": len(ptf),
                               "dev_strict_mismatch_docs": mism, "dev_strict_roundtrip_docs": rt, "docs": len(docs)}
        RC.log(cid, out["results"][cid])
    out["code_sha256"] = RC.code_sha("r2_transformers_check.py", "r2_common.py")
    RC.dump(out, os.path.join(RC.R2, "transformers_check.json"))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
