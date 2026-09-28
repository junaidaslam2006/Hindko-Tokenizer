# -*- coding: utf-8 -*-
"""Synthetic stress set for the sp32k release equivalence test -> release_build/sp32k/stress/stress.jsonl
Deterministic (numpy PCG64(20260927)). Every code point is built with chr(). Categories:
  newlines, spaces, digits, latin, codepoints (every code point of every data view), zwnj, tone_letters,
  ties (runs X^n of every piece X whose X+X is a piece, alone and after real-text prefixes of growing length),
  pieces (every NORMAL piece string), fuzz_*, unknown_scripts, long_lines (beyond the float32 bound),
  u2581 (literal U+2581: lossy in BOTH encoders), specials (literal special-token strings, chat text).
"""
import json
import os
import collections

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
DATA = os.path.join(TOK, "data")
OUT = os.path.join(TOK, r"release_build\sp32k\stress")
MODEL = os.path.join(TOK, r"release_build\sp32k\sp.model")
NL = chr(10)
SP = chr(32)
os.makedirs(OUT, exist_ok=True)
rng = np.random.Generator(np.random.PCG64(20260927))
items = []


def add(cat, text):
    items.append({"id": "%s-%06d" % (cat, sum(1 for x in items if x["cat"] == cat)), "cat": cat, "text": text})


def load_texts(name):
    with open(os.path.join(DATA, name + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


views = ["train_D1", "train_D2", "train_D3", "dev_strict", "dev_permissive", "test_strict"]
cp_count = collections.Counter()
zw_words = collections.Counter()
for v in views:
    for t in load_texts(v):
        cp_count.update(t)
        if chr(0x200C) in t or chr(0x200D) in t:
            zw_words.update(w for w in t.split() if chr(0x200C) in w or chr(0x200D) in w)
dev = load_texts("dev_strict")
dev_lines = [l for t in dev for l in t.split(NL) if l.strip()]
long_line = max(dev_lines, key=len)
H = "".join(chr(c) for c in (0x06C1, 0x0646, 0x062F, 0x06A9, 0x0648))                 # a Hindko word
W2 = "".join(chr(c) for c in (0x067E, 0x0634, 0x0648, 0x0631))
SENT = " ".join([H, W2, "".join(chr(c) for c in (0x062F, 0x06D2)), "".join(chr(c) for c in (0x0627, 0x06CC, 0x06C1)),
                 "".join(chr(c) for c in (0x06A9, 0x062A, 0x0627, 0x0628)) + chr(0x06D4)])

# ---------------------------------------------------------------- newlines
base = ["", NL, NL * 2, NL * 3, NL * 10, H + NL, NL + H, H + NL * 2 + W2, NL + SP + NL, SP + NL + SP, NL + SP,
        SP + NL, H + NL + NL + NL + W2 + NL, NL * 4 + SENT + NL * 4, SENT + chr(13) + NL + SENT, chr(13) + NL,
        chr(13), SENT + chr(13), NL + NL + SP * 3 + NL + NL]
for t in base:
    add("newlines", t)
for _ in range(400):
    parts = []
    for k in range(int(rng.integers(1, 8))):
        parts.append(rng.choice(["", SP, SP * 2, H, W2, SENT, "abc", "12", chr(0x06F1) * 2]))
        parts.append(NL * int(rng.integers(0, 5)))
    add("newlines", "".join(parts))

# ---------------------------------------------------------------- spaces
spaces = [chr(c) for c in (0x20, 0x09, 0x0B, 0x0C, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
                           0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF)]
for n in range(1, 9):
    add("spaces", SP * n)
    add("spaces", SP * n + SENT)
    add("spaces", SENT + SP * n)
    add("spaces", SP * n + SENT + SP * n)
    add("spaces", H + SP * n + W2)
    add("spaces", SENT + NL + SP * n + SENT + SP * n + NL + SP * n)
for s in spaces:
    for t in (s, s * 3, s + H, H + s, H + s + W2, SP + s + SP, s + NL + s):
        add("spaces", t)

# ---------------------------------------------------------------- digits
digit_sets = {"ascii": [chr(c) for c in range(0x30, 0x3A)], "ext_arabic_indic": [chr(c) for c in range(0x06F0, 0x06FA)],
              "arabic_indic": [chr(c) for c in range(0x0660, 0x066A)]}
for name, ds in digit_sets.items():
    add("digits", "".join(ds))
    for d in ds:
        for n in (1, 2, 3, 4, 5, 7, 11, 16):
            add("digits", d * n)
            add("digits", SENT + SP + d * n + chr(0x06D4))
    for _ in range(150):
        L = int(rng.integers(1, 41))
        num = "".join(rng.choice(ds, L))
        ctx = int(rng.integers(0, 4))
        add("digits", [num, SP + num + SP, H + SP + num + SP + W2, "(" + num + ")"][ctx])
for _ in range(200):
    ds = digit_sets[rng.choice(list(digit_sets))]
    sep = rng.choice([",", ".", ":", "/", "-", chr(0x066B), chr(0x066C), chr(0x060C), SP])
    add("digits", sep.join("".join(rng.choice(ds, int(rng.integers(1, 5)))) for _ in range(int(rng.integers(2, 6)))))

# ---------------------------------------------------------------- latin
english = ["The quick brown fox jumps over the lazy dog.", "Newborns laugh in their sleep, say researchers.",
           "Peshawar is one of the oldest living cities in South Asia.", "HELLO World", "e-mail: someone@example.org",
           "https://example.com/path?q=1&r=two#frag", "def f(x):\n    return x ** 2\n", "C'est déjà l'été — naïve café.",
           "Straße, Ångström, Øre, ñandú, Ærø", "UPPER lower MiXeD 123 abc-def_ghi", "a  b   c    d", "www.hindko.org"]
for t in english:
    add("latin", t)
    add("latin", SENT + SP + t + SP + SENT)
add("latin", "".join(chr(c) for c in range(0x21, 0x7F)))
for c in range(0x20, 0x7F):
    add("latin", chr(c)); add("latin", H + chr(c) + W2); add("latin", chr(c) * 3)
for c in "abcdefghijklmnopqrstuvwxyzABCXYZ!?.-_=*#":
    for n in range(1, 13):
        add("latin", c * n)
for _ in range(300):
    L = int(rng.integers(1, 12))
    add("latin", SP.join("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
                                            int(rng.integers(1, 12)))) for _ in range(L)))

# ---------------------------------------------------------------- every code point of the corpus
cps = sorted(cp_count)
for c in cps:
    if c == NL:
        continue
    for t in (c, SP + c, c + SP, c * 2, c * 3, c * 7, H + c + W2, H + SP + c + SP + W2, c + SENT, SENT + c):
        add("codepoints", t)

# ---------------------------------------------------------------- ZWNJ / ZWJ
ZWNJ, ZWJ = chr(0x200C), chr(0x200D)
for z in (ZWNJ, ZWJ):
    for t in (z, z * 2, SP + z + SP, z + H, H + z, H[:2] + z + H[2:], H[:2] + z * 2 + H[2:], W2 + z + NL + z + W2,
              SENT.replace(SP, z), SENT.replace(SP, SP + z), SENT.replace(SP, z + SP)):
        add("zwnj", t)
for w, _ in zw_words.most_common(300):
    add("zwnj", w); add("zwnj", SENT + SP + w + SP + SENT)
# ZWNJ/ZWJ do not occur in the corpus: insert them at random positions inside real dev lines
for _ in range(600):
    l = dev_lines[int(rng.integers(0, len(dev_lines)))][:200]
    z = ZWNJ if rng.random() < 0.75 else ZWJ
    for _k in range(int(rng.integers(1, 4))):
        pos = int(rng.integers(0, len(l) + 1))
        l = l[:pos] + z * int(rng.integers(1, 3)) + l[pos:]
    add("zwnj", l)

# ---------------------------------------------------------------- Hindko tone letters (U+08BE..U+08C2) and U+0768
tones = [chr(c) for c in range(0x08BE, 0x08C3)]
bases_decomposed = [chr(c) for c in (0x067E, 0x062A, 0x0679, 0x0686, 0x06A9, 0x062C, 0x06AF)]
for t in tones + [chr(0x0768)]:
    for s in (t, t * 2, t * 3, SP + t, t + SP, H + t, t + H, H[:2] + t + H[2:], t + chr(0x064E), t + chr(0x0651),
              SENT + SP + t + H + SP + SENT, t + ZWNJ + H):
        add("tone_letters", s)
for b in bases_decomposed:
    add("tone_letters", b + chr(0x065A)); add("tone_letters", H + b + chr(0x065A) + W2)
tone_words = collections.Counter(w for t in dev + load_texts("train_D2") for w in t.split() if any(x in w for x in tones))
for w, _ in tone_words.most_common(400):
    add("tone_letters", w); add("tone_letters", SENT + SP + w + SP + SENT)

# ---------------------------------------------------------------- ties: runs of every piece X with X+X a piece
from sentencepiece import sentencepiece_model_pb2 as spb
m = spb.ModelProto(); m.ParseFromString(open(MODEL, "rb").read())
T = spb.ModelProto.SentencePiece
normal = [p.piece for p in m.pieces if p.type == T.NORMAL]
nset = set(normal)
fam = sorted(x for x in normal if chr(0x2581) not in x and x + x in nset)
prefixes = [long_line[:n] for n in (0, 40, 200, 800, 1600, 3200, 5600)]
for x in fam:
    for n in range(2, 9):
        run = x * n
        add("ties", run)
        for pf in prefixes[1:]:
            add("ties", pf + SP + run)
            add("ties", pf + run + SP + H)
with open(os.path.join(TOK, r"release_build\sp32k\stress\tie_families.json"), "w", encoding="utf-8") as f:
    json.dump(fam, f, ensure_ascii=False)

# ---------------------------------------------------------------- every NORMAL piece string
for x in normal:
    s = x.replace(chr(0x2581), SP)
    add("pieces", s)
    if s.startswith(SP):
        add("pieces", s[1:])

# ---------------------------------------------------------------- fuzz
cp_list = [c for c in cps if c != NL]
freq = np.array([cp_count[c] for c in cp_list], dtype=np.float64); freq /= freq.sum()
for _ in range(5000):
    L = int(rng.integers(1, 61))
    add("fuzz_uniform_cp", "".join(rng.choice(cp_list + [SP, SP, NL], L)))
for _ in range(5000):
    L = int(rng.integers(1, 121))
    add("fuzz_freq_cp", "".join(rng.choice(cp_list, L, p=freq)))
dec = [x.replace(chr(0x2581), SP) for x in normal]
for _ in range(5000):
    L = int(rng.integers(1, 40))
    add("fuzz_pieces", "".join(dec[i] if rng.random() > 0.05 else NL for i in rng.integers(0, len(dec), L)))
words = [w for l in dev_lines for w in l.split(SP)]
for _ in range(3000):
    L = int(rng.integers(1, 60))
    add("fuzz_words", SP.join(words[i] for i in rng.integers(0, len(words), L)))

# ---------------------------------------------------------------- characters outside the vocabulary
unk = ["".join(chr(c) for c in range(0x4E00, 0x4E20)), chr(0x1F600) + chr(0x1F44D) + chr(0x1F3FD),
       chr(0x1F468) + ZWJ + chr(0x1F469) + ZWJ + chr(0x1F467), "".join(chr(c) for c in range(0x0915, 0x0939)),
       "".join(chr(c) for c in range(0x0410, 0x0430)), "".join(chr(c) for c in range(0x0391, 0x03A9)),
       "".join(chr(c) for c in range(0x05D0, 0x05EA)), "".join(chr(c) for c in list(range(0x00, 0x0A)) + list(range(0x0B, 0x20))),
       chr(0x7F) + "".join(chr(c) for c in range(0x80, 0xA0)), chr(0xE000) + chr(0xF8FF) + chr(0xFFFF) + chr(0x10FFFF),
       chr(0x200E) + chr(0x200F) + chr(0x061C) + chr(0x202A) + chr(0x202C), chr(0xFFFD) * 3, "a" + chr(0x0301) + chr(0x0327),
       "|", "||", "a|b", "<|", "|>", "<unk>", "<0x41>", "<s>", "</s>", "[UNK]"]
for t in unk:
    add("unknown_scripts", t); add("unknown_scripts", SENT + SP + t + SP + SENT); add("unknown_scripts", t * 5)
for _ in range(500):
    L = int(rng.integers(1, 30))
    add("unknown_scripts", "".join(chr(int(c)) for c in rng.integers(0x2E80, 0x9FFF, L)))

# ---------------------------------------------------------------- lines longer than the float32 bound
cat_line = SP.join(dev_lines)
for n in (8000, 12000, 20000, 40000, 80000):
    base_l = cat_line[:n]
    add("long_lines", base_l)
    for x in fam[:12]:
        add("long_lines", base_l + SP + x * 3 + SP + H)

# ---------------------------------------------------------------- literal U+2581 (lossy in both encoders)
for t in (chr(0x2581), chr(0x2581) * 3, H + chr(0x2581) + W2, SP + chr(0x2581), chr(0x2581) + SP + H, NL + chr(0x2581)):
    add("u2581", t)

# ---------------------------------------------------------------- literal special-token strings / chat text
specials = [p.piece for p in m.pieces if p.type in (T.USER_DEFINED, T.CONTROL) and p.piece != NL]
for s in specials:
    for t in (s, s + H, H + s, H + SP + s + SP + W2, SP + s, s + SP, s + NL + H, s + s, NL + s + NL):
        add("specials", t)
for turns in range(1, 6):
    txt = ""
    for k in range(turns):
        role = ["system", "user", "assistant"][k % 3]
        body = SP.join(rng.choice(dev_lines, int(rng.integers(1, 3))).tolist())
        txt += "<|im_start|>" + role + NL + body + "<|im_end|>" + NL
    add("specials", "<|bos|>" + txt + "<|im_start|>assistant" + NL)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "stress.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
cats = collections.Counter(it["cat"] for it in items)
summary = {"items": len(items), "categories": dict(cats), "corpus_code_points": len(cps),
           "corpus_code_points_list": ["U+%04X" % ord(c) for c in cps], "tie_families": len(fam),
           "seed": "numpy PCG64(20260927)", "views_scanned_for_code_points": views}
with open(os.path.join(OUT, "stress_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "corpus_code_points_list"}, ensure_ascii=False, indent=1))
