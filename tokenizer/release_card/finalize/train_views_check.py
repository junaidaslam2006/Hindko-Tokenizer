# -*- coding: utf-8 -*-
"""Round trip and id equality of the final release folder on the train views (train_D2, train_D1), for the card's
Section 3.1 table: tokenizer.json, transformers and sp.model (newline convention). Writes train_views_check.json.

    set PYTHONIOENCODING=utf-8
    python release_card\\finalize\\train_views_check.py
"""
import datetime
import json
import os
import sys

REL = r"F:\Hindko\tokenizer"
DATA = r"F:\Hindko\_tokenizer\data"
OUT = r"F:\Hindko\_tokenizer\release_card\finalize\train_views_check.json"
sys.path.insert(0, os.path.join(REL, "examples"))


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from usage import HindkoSP
    tk = Tokenizer.from_file(os.path.join(REL, "tokenizer.json"))
    tf = AutoTokenizer.from_pretrained(REL)
    sp = HindkoSP(os.path.join(REL, "sp.model"))
    res = {"what": __doc__.split("\n")[0], "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "views": {}}
    for view in ("train_D2", "train_D1"):
        with open(os.path.join(DATA, view + ".jsonl"), encoding="utf-8") as f:
            texts = [json.loads(l)["text"] for l in f]
        hf = [e.ids for e in tk.encode_batch(texts)]
        tfi = tf(texts)["input_ids"]
        r = {"docs": len(texts), "hf_decode_exact": 0, "tf_ids_equal_hf": 0, "tf_decode_exact": 0,
             "sp_ids_equal_hf": 0, "sp_decode_exact": 0}
        for t, a, b in zip(texts, hf, tfi):
            r["hf_decode_exact"] += tk.decode(a, skip_special_tokens=False) == t
            r["tf_ids_equal_hf"] += list(b) == a
            r["tf_decode_exact"] += tf.decode(b, skip_special_tokens=False) == t
            s = sp.encode(t)
            r["sp_ids_equal_hf"] += s == a
            r["sp_decode_exact"] += sp.decode(s) == t
        res["views"][view] = r
        print(view, json.dumps(r), flush=True)
    json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
