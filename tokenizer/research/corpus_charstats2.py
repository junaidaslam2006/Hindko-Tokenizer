# -*- coding: utf-8 -*-
"""Second pass: list format chars / presentation forms / Arabic-Ext-A letters and
count how often a combining mark directly follows a non-letter (which a \\p{L}+ regex
would split off), plus the frequency profile of short function words. Read-only."""
import json, collections, unicodedata, os, re

SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_charstats2.json")

def main():
    cf = collections.Counter(); pres = collections.Counter(); exta = collections.Counter()
    words = collections.Counter()
    n_words_with_mark = 0; n_words = 0
    per_source_marks = collections.Counter(); per_source_chars = collections.Counter()
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); t = r["text"]; s = r.get("source")
            per_source_chars[s] += len(t)
            for ch in t:
                cp = ord(ch); cat = unicodedata.category(ch)
                if cat == "Cf": cf["U+%04X %s" % (cp, unicodedata.name(ch, "?"))] += 1
                if 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF:
                    pres["U+%04X %s" % (cp, unicodedata.name(ch, "?"))] += 1
                if 0x08A0 <= cp <= 0x08FF: exta["U+%04X %s" % (cp, unicodedata.name(ch, "?"))] += 1
                if cat.startswith("M"): per_source_marks[s] += 1
            for w in t.split():
                n_words += 1
                if any(unicodedata.category(c).startswith("M") for c in w): n_words_with_mark += 1
                words[w] += 1
    total = sum(words.values())
    top = words.most_common(60)
    cum = 0; cov = {}
    for i, (w, n) in enumerate(words.most_common(), 1):
        cum += n
        if i in (100, 1000, 8000, 16000, 32000, 64000, 128000):
            cov[i] = round(cum / total, 4)
    res = {
        "format_chars_Cf": dict(cf.most_common()),
        "presentation_forms": dict(pres.most_common(30)),
        "arabic_ext_a": dict(exta.most_common()),
        "marks_per_1000_chars_by_source": {k: round(1000 * per_source_marks[k] / per_source_chars[k], 2) for k in per_source_chars},
        "fraction_whitespace_words_containing_a_mark": round(n_words_with_mark / n_words, 4),
        "token_coverage_of_top_k_whitespace_words": cov,
        "top60_whitespace_words": top,
    }
    with open(OUT, "w", encoding="utf-8") as g: json.dump(res, g, ensure_ascii=False, indent=1)
    print("ok")

if __name__ == "__main__":
    main()
