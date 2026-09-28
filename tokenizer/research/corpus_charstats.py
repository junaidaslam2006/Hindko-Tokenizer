# -*- coding: utf-8 -*-
"""Character/word inventory of the Hindko corpus, to inform tokenizer design.

Read-only on the released dataset. Writes corpus_charstats.json next to this file.
Deterministic: no sampling, sorted outputs.
"""
import json, sys, unicodedata, collections, re, os, time

SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_charstats.json")

def cat_of(cp):
    ch = chr(cp)
    if 0x0660 <= cp <= 0x0669: return "digit_arabic_indic_0660"
    if 0x06F0 <= cp <= 0x06F9: return "digit_ext_arabic_indic_06F0"
    if 0x30 <= cp <= 0x39: return "digit_ascii"
    if cp == 0x200C: return "zwnj"
    if cp == 0x200D: return "zwj"
    if cp == 0x200B: return "zwsp"
    if cp == 0x200F or cp == 0x200E or 0x202A <= cp <= 0x202E or 0x2066 <= cp <= 0x2069: return "bidi_control"
    if cp == 0x0640: return "kashida"
    if 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF: return "arabic_presentation_form"
    if 0x08A0 <= cp <= 0x08FF: return "arabic_ext_a"
    if cp == 0xFFFD: return "replacement_char"
    if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A: return "latin_letter"
    c = unicodedata.category(ch)
    if c in ("Mn", "Mc", "Me"):
        return "combining_mark"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "arabic_block_" + c
    if c.startswith("Z") or ch in "\n\t\r": return "whitespace"
    if c.startswith("P"): return "punct_other"
    if c.startswith("S"): return "symbol_other"
    return "other_" + c

def main():
    t0 = time.time()
    per_cp = collections.Counter()
    per_cat = collections.Counter()
    per_source_chars = collections.Counter()
    per_source_docs = collections.Counter()
    per_tier = collections.Counter()
    words = collections.Counter()
    word_tokens = 0
    digit_runs = collections.Counter()
    n_docs = 0
    total_chars = 0
    total_bytes = 0
    nfc_changed = 0
    digit_run_re = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r["text"]
            n_docs += 1
            total_chars += len(t)
            total_bytes += len(t.encode("utf-8"))
            per_source_chars[r.get("source")] += len(t)
            per_source_docs[r.get("source")] += 1
            per_tier[r.get("quality_tier")] += 1
            if unicodedata.normalize("NFC", t) != t:
                nfc_changed += 1
            per_cp.update(t)
            for w in t.split():
                words[w] += 1
                word_tokens += 1
            for m in digit_run_re.finditer(t):
                digit_runs[min(len(m.group()), 9)] += 1
    for ch, n in per_cp.items():
        per_cat[cat_of(ord(ch))] += n
    freq_of_freq = collections.Counter()
    for w, n in words.items():
        freq_of_freq[min(n, 10)] += 1
    top_marks = sorted(((n, "U+%04X" % ord(c), unicodedata.name(c, "?")) for c, n in per_cp.items()
                        if unicodedata.category(c).startswith("M")), reverse=True)[:30]
    top_nonarabic = sorted(((n, "U+%04X" % ord(c), unicodedata.name(c, "?")) for c, n in per_cp.items()
                            if not (0x0600 <= ord(c) <= 0x06FF) and not c.isspace()), reverse=True)[:40]
    res = {
        "source_file": SRC,
        "n_docs": n_docs,
        "total_chars": total_chars,
        "total_utf8_bytes": total_bytes,
        "bytes_per_char": round(total_bytes / total_chars, 4),
        "docs_not_nfc": nfc_changed,
        "whitespace_word_tokens": word_tokens,
        "distinct_whitespace_words": len(words),
        "word_types_by_freq_(10=10+)": dict(sorted(freq_of_freq.items())),
        "distinct_codepoints": len(per_cp),
        "chars_by_category": dict(sorted(per_cat.items(), key=lambda x: -x[1])),
        "chars_by_source": dict(per_source_chars),
        "docs_by_source": dict(per_source_docs),
        "docs_by_tier": dict(per_tier),
        "digit_run_length_hist_(9=9+)": dict(sorted(digit_runs.items())),
        "top_combining_marks": top_marks,
        "top_non_arabic_block_chars": top_nonarabic,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as g:
        json.dump(res, g, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ["n_docs", "total_chars", "total_utf8_bytes", "whitespace_word_tokens", "distinct_whitespace_words", "runtime_s"]}))

if __name__ == "__main__":
    main()
