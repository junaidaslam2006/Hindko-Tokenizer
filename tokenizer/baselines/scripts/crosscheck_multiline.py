"""Repeat the conversion cross-checks on MULTI-LINE text (offline; added 2026-09-26 after review).

The original checks (convert_kimi.py, crosscheck.py) ran on 2,000 single-line paragraphs + the 5 core samples + one
ASCII stress string, so line-break handling was barely exercised.  Here the same comparisons run on
scripts/_verify_multiline.json (200 strict records, first 3,000 chars, line breaks kept) + the 3 stress samples:
1. kimi-k2: our converted tokenizer.json vs the pure-Python tiktoken reference (tiktoken_tools.RefBPE) + round trip.
2. mistral-nemo, mistral-small-4: HF tokenizer.json vs the native tekken.json (reference algorithm).
3. gpt-4o (Xenova) vs gpt-oss (OpenAI); gpt-4 (Xenova) vs phi-4 (Microsoft).
No network access, nothing downloaded, no remote code executed.
Output: scripts/_crosscheck_multiline.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiktoken_tools as tt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

B = r"F:\Hindko\_tokenizer\baselines"
F = os.path.join(B, "files")
texts = [p["text"] for p in json.load(open(os.path.join(B, "scripts", "_verify_multiline.json"), encoding="utf-8"))]
texts += [s["text"] for s in json.load(open(os.path.join(B, "samples_hindko.json"), encoding="utf-8"))["samples"]
          if s["group"] == "stress"]
N_NL = sum(t.count("\n") for t in texts)


def hf(path):
    t = Tokenizer.from_file(path)
    t.no_truncation()
    t.no_padding()
    return t


def compare(enc_a, enc_b, dec_a=None):
    mism, first, ntok, rt_fail = 0, None, 0, 0
    for t in texts:
        a, b = enc_a(t), enc_b(t)
        ntok += len(b)
        if a != b:
            mism += 1
            if first is None:
                first = {"text": t[:120], "a": a[:30], "b": b[:30]}
        if dec_a is not None and dec_a(a) != t:
            rt_fail += 1
    r = {"n_texts": len(texts), "n_line_breaks": N_NL, "n_tokens_b": ntok, "n_texts_mismatch": mism, "first_mismatch": first}
    if dec_a is not None:
        r["n_roundtrip_failures_a"] = rt_fail
    return r


out = {}
# 1. Kimi
D = os.path.join(F, "kimi-k2")
ranks = tt.load_ranks(os.path.join(D, "tiktoken.model"))
ref = tt.RefBPE(ranks, tt.kimi_pattern(os.path.join(D, "tokenization_kimi.py")))
tk = hf(os.path.join(D, "tokenizer.converted.json"))
r = compare(lambda t: tk.encode(t, add_special_tokens=False).ids, ref.encode,
            lambda ids: tk.decode(ids, skip_special_tokens=False))
out["kimi-k2 conversion vs reference (multi-line)"] = r
print("kimi", r, flush=True)
del ref, tk, ranks

# 2. Mistral tekken
for name in ["mistral-nemo", "mistral-small-4"]:
    ranks, pattern, n_special, _, cfg = tt.load_tekken(os.path.join(F, name, "tekken.json"))
    ref = tt.RefBPE(ranks, pattern, offset=n_special)
    tk = hf(os.path.join(F, name, "tokenizer.json"))
    r = compare(lambda t: tk.encode(t, add_special_tokens=False).ids, ref.encode)
    r["tekken_version"] = cfg.get("version")
    out[f"{name}: HF tokenizer.json vs native tekken.json (multi-line)"] = r
    print(name, r, flush=True)
    del ref, tk, ranks

# 3. OpenAI encodings
for a_name, b_name in [("gpt-4o", "gpt-oss"), ("gpt-4", "phi-4")]:
    a, b = hf(os.path.join(F, a_name, "tokenizer.json")), hf(os.path.join(F, b_name, "tokenizer.json"))
    r = compare(lambda t: a.encode(t, add_special_tokens=False).ids, lambda t: b.encode(t, add_special_tokens=False).ids)
    out[f"{a_name} vs {b_name} (multi-line)"] = r
    print(a_name, b_name, r, flush=True)

json.dump(out, open(os.path.join(B, "scripts", "_crosscheck_multiline.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
