"""Independent cross-checks of community conversions / HF conversions against native or official files.

1. mistral-nemo, mistral-small-4: HF tokenizer.json vs the native tekken.json (reference tiktoken algorithm, ids offset by n_special).
2. gpt-4o (Xenova conversion) vs OpenAI's official gpt-oss tokenizer.json (o200k_harmony = o200k_base + specials):
   base vocab (ids < 199998) and encodings on the verification set.
3. gpt-4 (Xenova conversion of cl100k_base) vs microsoft/phi-4 (cl100k-derived): base vocab ids < 100256 and encodings.
4. kimi-k2: split regex in K2 vs K2.5/K2.6/K3 tokenization_kimi.py (text comparison only; nothing executed).
Output: scripts/_crosscheck.json
"""
import json
import os
import sys

os.environ["HF_HOME"] = r"F:\Hindko\_tokenizer\hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import tiktoken_tools as tt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

B = r"F:\Hindko\_tokenizer\baselines"
F = os.path.join(B, "files")
texts = [p["text"] for p in json.load(open(os.path.join(B, "scripts", "_verify_paragraphs.json"), encoding="utf-8"))]
# the 5 core samples (formerly samples_hindko5.json); multi-line text is checked in crosscheck_multiline.py
texts += [s["text"] for s in json.load(open(os.path.join(B, "samples_hindko.json"), encoding="utf-8"))["samples"] if s["group"] == "core"]
texts.append("Hello world! It's 2026-09-26, price $1,234.56 -- don't   stop\n\n  tabs\tand CAPS WORDS; 中文字符 تے ۱۲۳۴")


def hf(name):
    t = Tokenizer.from_file(os.path.join(F, name, "tokenizer.json"))
    t.no_truncation()
    t.no_padding()
    return t


def compare(enc_a, enc_b):
    mism, first, ntok = 0, None, 0
    for t in texts:
        a, b = enc_a(t), enc_b(t)
        ntok += len(b)
        if a != b:
            mism += 1
            if first is None:
                first = {"text": t[:120], "a": a[:30], "b": b[:30]}
    return {"n_texts": len(texts), "n_tokens_b": ntok, "n_texts_mismatch": mism, "first_mismatch": first}


out = {}
for name in ["mistral-nemo", "mistral-small-4"]:
    ranks, pattern, n_special, specials, cfg = tt.load_tekken(os.path.join(F, name, "tekken.json"))
    ref = tt.RefBPE(ranks, pattern, offset=n_special)
    tk = hf(name)
    r = compare(lambda t: tk.encode(t, add_special_tokens=False).ids, ref.encode)
    r.update(tekken_version=cfg.get("version"), n_special=n_special, n_ranks=len(ranks), pattern=pattern)
    out[f"{name}: HF tokenizer.json vs native tekken.json"] = r
    print(name, {k: v for k, v in r.items() if k != "pattern"}, flush=True)


def base_vocab_equal(a, b, n):
    va = {i: t for t, i in a.get_vocab(False).items() if i < n}
    vb = {i: t for t, i in b.get_vocab(False).items() if i < n}
    diff = [i for i in range(n) if va.get(i) != vb.get(i)]
    return {"n_checked": n, "n_ids_differ": len(diff), "first_diff_ids": diff[:10]}


for a_name, b_name, n in [("gpt-4o", "gpt-oss", 199998), ("gpt-4", "phi-4", 100256)]:
    a, b = hf(a_name), hf(b_name)
    r = compare(lambda t: a.encode(t, add_special_tokens=False).ids, lambda t: b.encode(t, add_special_tokens=False).ids)
    r["base_vocab"] = base_vocab_equal(a, b, n)
    out[f"{a_name} vs {b_name}"] = r
    print(a_name, b_name, r, flush=True)

# Kimi: compare split regex across versions (text only)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi()
pats = {"moonshotai/Kimi-K2-Instruct": tt.kimi_pattern(os.path.join(F, "kimi-k2", "tokenization_kimi.py"))}
for repo in ["moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6", "moonshotai/Kimi-K3"]:
    sha = api.model_info(repo).sha
    p = hf_hub_download(repo, "tokenization_kimi.py", revision=sha,
                        local_dir=os.path.join(F, "kimi-k2", "other_versions", repo.split("/")[1]))
    try:
        pats[repo] = tt.kimi_pattern(p)
    except Exception as e:  # noqa: BLE001
        pats[repo] = f"PARSE FAILED: {e}"
base = pats["moonshotai/Kimi-K2-Instruct"]
out["kimi split regex vs K2"] = {r: (p == base) for r, p in pats.items()}
print(out["kimi split regex vs K2"])
json.dump(out, open(os.path.join(B, "scripts", "_crosscheck.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
