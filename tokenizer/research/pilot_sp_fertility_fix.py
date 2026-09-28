# -*- coding: utf-8 -*-
"""Correct the SentencePiece fertility numbers of the pilot.

pilot_tokenizers.py encoded " "+word; with add_dummy_prefix SentencePiece then
emits an extra lone U+2581 token, inflating fertility by ~1. Here each word is
encoded bare (the dummy prefix supplies the word-initial marker), which matches
how a word inside running text is segmented."""
import hashlib, json, os, collections
import sentencepiece as spm
SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pilot")
te = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if int(hashlib.md5(r["uid"].encode()).hexdigest(), 16) % 20 == 0: te.append(r["text"])
wc = collections.Counter(w for t in te for w in t.split()); tot = sum(wc.values())
rows = []
for V in (16000, 32000):
    sp = spm.SentencePieceProcessor(model_file=os.path.join(OUT, "sp_unigram_%d.model" % V))
    fert = 0; single = 0
    for w, c in wc.items():
        k = len(sp.encode(w)); fert += k * c; single += c if k == 1 else 0
    row = {"tokenizer": "sp_unigram", "vocab": V, "fertility_tok_per_word": round(fert / tot, 3),
           "pct_words_single_token": round(100 * single / tot, 1)}
    print(json.dumps(row)); rows.append(row)
with open(os.path.join(OUT, "pilot_sp_fertility_fixed.json"), "w") as g: json.dump(rows, g, indent=1)
