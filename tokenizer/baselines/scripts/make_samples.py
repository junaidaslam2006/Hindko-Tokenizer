"""Deterministically pick the Hindko test samples (+ verification sets) from the STRICT dataset. No randomness.

CORE (5 paragraphs, used for tokens/word sanity numbers and the 5-sample UNK rate):
  paragraphs = text.split('\n'), stripped, 200 <= len <= 900 chars.  Each paragraph gets
  key = sha256(f"{uid}#{paragraph_index}").  For each slot, the paragraph with the smallest key that satisfies
  the slot predicate (and whose uid was not used before) is chosen.
STRESS (3 samples, each with its OWN required predicate; added 2026-09-26 after review, because no core sample
  contains a line break, a presentation form or U+2026, and several tokenizers lose exactly these):
  - multi-line document: a WHOLE record (not split on '\n'), 600 <= len <= 3000, >= 4 line breaks and at least one
    blank line ('\n\n'); key = sha256(f"{uid}#doc").
  - a paragraph (as above) containing U+FDFA ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM (a presentation form that the
    canonical normalisation keeps; 2,605 occurrences in the strict file).
  - a paragraph (as above) containing U+2026 HORIZONTAL ELLIPSIS (12,509 occurrences in the strict file).
Verification sets (used only to verify our own tiktoken->tokenizers conversions and the HF-vs-native cross-checks):
  - _verify_paragraphs.json: the 2,000 paragraphs with the smallest keys overall (any source, 20 <= len <= 4000).
  - _verify_multiline.json: the 200 strict records with >= 2 line breaks and the smallest sha256(f"verify-ml#{uid}"),
    first 3,000 characters of each (line breaks kept).
Outputs: baselines/samples_hindko.json, baselines/scripts/_verify_paragraphs.json, baselines/scripts/_verify_multiline.json
"""
import hashlib
import json
import unicodedata

SRC = r"F:\Hindko\hindko_dataset.jsonl"
OUTS = r"F:\Hindko\_tokenizer\baselines\samples_hindko.json"
OUTV = r"F:\Hindko\_tokenizer\baselines\scripts\_verify_paragraphs.json"
OUTML = r"F:\Hindko\_tokenizer\baselines\scripts\_verify_multiline.json"

HARAKAT = {chr(c) for c in range(0x064B, 0x0653)}
EXT_DIGITS = {chr(c) for c in range(0x06F0, 0x06FA)}
FDFA = chr(0xFDFA)
ELLIPSIS = chr(0x2026)


def is_pres(ch):
    o = ord(ch)
    return 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


# (group, label, kind, predicate(source, text)); kind "para" draws from paragraphs, "doc" from whole records
SLOTS = [
    ("core", "newspaper + ASCII digits", "para", lambda s, t: s == "newspaper" and any("0" <= c <= "9" for c in t)),
    ("core", "newspaper + harakat", "para", lambda s, t: s == "newspaper" and any(c in HARAKAT for c in t)),
    ("core", "book + harakat", "para", lambda s, t: s == "book" and any(c in HARAKAT for c in t)),
    # Until 2026-09-26 this slot read "Extended Arabic-Indic digits OR presentation forms"; the paragraph it picked has
    # digits only.  Requiring digits alone selects the same paragraph, so the label now says what the sample contains.
    ("core", "book + Extended Arabic-Indic digits", "para", lambda s, t: s == "book" and any(c in EXT_DIGITS for c in t)),
    ("core", "web", "para", lambda s, t: s == "web"),
    ("stress", "multi-line document (whole record, >=4 line breaks incl. a blank line)", "doc",
     lambda s, t: 600 <= len(t) <= 3000 and t.count("\n") >= 4 and "\n\n" in t),
    ("stress", "paragraph with presentation form U+FDFA", "para", lambda s, t: FDFA in t),
    ("stress", "paragraph with U+2026 horizontal ellipsis", "para", lambda s, t: ELLIPSIS in t),
]


def main():
    paras, docs, allp, ml = [], [], [], []
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        t = r["text"]
        docs.append((hashlib.sha256(f"{r['uid']}#doc".encode()).hexdigest(), r["uid"], r["source"], None, t))
        if t.count("\n") >= 2:
            ml.append((hashlib.sha256(f"verify-ml#{r['uid']}".encode()).hexdigest(), r["uid"], r["source"], t[:3000]))
        for i, p in enumerate(t.split("\n")):
            p = p.strip()
            key = hashlib.sha256(f"{r['uid']}#{i}".encode()).hexdigest()
            if 20 <= len(p) <= 4000:
                allp.append((key, r["uid"], r["source"], i, p))
            if 200 <= len(p) <= 900:
                paras.append((key, r["uid"], r["source"], i, p))
    paras.sort()
    docs.sort()
    allp.sort()
    ml.sort()
    chosen, used = [], set()
    for group, label, kind, pred in SLOTS:
        pool = paras if kind == "para" else docs
        for key, uid, src, i, p in pool:
            if uid not in used and pred(src, p):
                assert unicodedata.normalize("NFC", p) == p
                chosen.append({"group": group, "slot": label, "kind": kind, "uid": uid, "source": src, "paragraph_index": i,
                               "key": key, "n_chars": len(p), "n_words": len(p.split()), "n_newlines": p.count("\n"),
                               "text": p})
                used.add(uid)
                break
        else:
            raise SystemExit(f"no sample for slot {label!r}")
    json.dump({"rule": __doc__.strip(), "source_file": SRC, "samples": chosen},
              open(OUTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ver = [{"uid": u, "source": s, "paragraph_index": i, "text": p} for _, u, s, i, p in allp[:2000]]
    json.dump(ver, open(OUTV, "w", encoding="utf-8"), ensure_ascii=False)
    verml = [{"uid": u, "source": s, "text": t} for _, u, s, t in ml[:200]]
    json.dump(verml, open(OUTML, "w", encoding="utf-8"), ensure_ascii=False)
    for c in chosen:
        print(c["group"], "|", c["slot"], c["uid"], c["source"], c["n_chars"], "nl=", c["n_newlines"], c["text"][:60].replace("\n", " / "))
    print("verify set:", len(ver), "paragraphs,", sum(len(v["text"]) for v in ver), "chars")
    print("multi-line verify set:", len(verml), "texts,", sum(len(v["text"]) for v in verml), "chars,",
          sum(v["text"].count("\n") for v in verml), "line breaks")


if __name__ == "__main__":
    main()
