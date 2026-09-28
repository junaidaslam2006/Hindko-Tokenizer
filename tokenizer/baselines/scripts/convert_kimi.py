"""Convert Kimi K2's official tiktoken.model into a tokenizers JSON and verify it token-for-token.

Verification: (1) against the pure-Python tiktoken-algorithm reference (tiktoken_tools.RefBPE) on the
2,000-paragraph Hindko verification set + the 5 samples + an ASCII/English/digits stress string;
(2) decode(encode(x)) == x on the same texts.
Output: files/kimi-k2/tokenizer.converted.json and scripts/_kimi_verify.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import tiktoken_tools as tt  # noqa: E402

D = r"F:\Hindko\_tokenizer\baselines\files\kimi-k2"
OUT = os.path.join(D, "tokenizer.converted.json")
VER = r"F:\Hindko\_tokenizer\baselines\scripts\_kimi_verify.json"


def main():
    ranks = tt.load_ranks(os.path.join(D, "tiktoken.model"))
    pattern = tt.kimi_pattern(os.path.join(D, "tokenization_kimi.py"))
    cfg = json.load(open(os.path.join(D, "tokenizer_config.json"), encoding="utf-8"))
    named = {int(k): v["content"] for k, v in cfg["added_tokens_decoder"].items()}
    n_base = len(ranks)
    assert sorted(ranks.values()) == list(range(n_base))
    # tokenization_kimi.py reserves 256 special ids after the base vocab; unnamed ones are <|reserved_token_i|>
    specials = [(i, named.get(i, f"<|reserved_token_{i}|>")) for i in range(n_base, n_base + 256)]
    tk = tt.to_tokenizers_json(ranks, pattern, specials)
    tk.save(OUT)

    ref = tt.RefBPE(ranks, pattern)
    texts = [p["text"] for p in json.load(open(r"F:\Hindko\_tokenizer\baselines\scripts\_verify_paragraphs.json", encoding="utf-8"))]
    # the 5 core samples (formerly samples_hindko5.json); multi-line text is checked in crosscheck_multiline.py
    texts += [s["text"] for s in json.load(open(r"F:\Hindko\_tokenizer\baselines\samples_hindko.json", encoding="utf-8"))["samples"]
              if s["group"] == "core"]
    texts.append("Hello world! It's 2026-09-26, price $1,234.56 -- don't   stop\n\n  tabs\tand CAPS WORDS; 中文字符 تے ۱۲۳۴")
    n_mismatch, n_rt_fail, n_tok, first = 0, 0, 0, None
    for t in texts:
        a = tk.encode(t, add_special_tokens=False).ids
        b = ref.encode(t)
        n_tok += len(b)
        if a != b:
            n_mismatch += 1
            if first is None:
                first = {"text": t[:200], "converted": a[:40], "reference": b[:40]}
        if tk.decode(a, skip_special_tokens=False) != t:
            n_rt_fail += 1
    rep = {"n_base_ranks": n_base, "n_special": len(specials), "vocab_size": tk.get_vocab_size(True),
           "pattern": pattern, "n_texts": len(texts), "n_reference_tokens": n_tok,
           "n_texts_mismatch_vs_reference": n_mismatch, "n_roundtrip_failures": n_rt_fail, "first_mismatch": first}
    json.dump(rep, open(VER, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "pattern"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
