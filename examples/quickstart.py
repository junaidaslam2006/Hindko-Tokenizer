"""Quickstart for the Hindko tokenizer (SentencePiece Unigram, 32,768 ids), release 1.0.0.

    pip install tokenizers transformers huggingface_hub
    export HF_TOKEN=hf_...            # the repository is private: a read token is required
    python examples/quickstart.py     # or: python examples/quickstart.py path/to/local/folder

What it shows
  1. loading with transformers (AutoTokenizer) and with tokenizers (the canonical encoder);
  2. encode / decode round trip (nothing is added automatically: no BOS/EOS);
  3. the ChatML chat template;
  4. normalize(): the corpus "data form" for messy input, and why it matters (token counts
     for raw vs normalized text, including Arabic-keyboard letters).

normalize() below is a self-contained adaptation of the corpus pipeline's hp.normalize 1.0.1
(rules R01-R14). It is written to produce the same output as that module; it does not import it.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

REPO_ID = "junaid008/hindko-tokenizer"

# ============================================================================ normalize()
# Canonical "data form" (hp.normalize 1.0.1). Deterministic, idempotent, NFC in and out.
# It folds only encoding noise and never merges letters that differ in Hindko/Urdu.
# Not reversible: after it, decode() returns normalize(x), not x. Do not use it where
# whitespace must be preserved exactly (code, tables, verbatim quotes).

NORMALIZATION_VERSION = "1.0.1"


def _cls(cps) -> str:
    cps = sorted(set(cps))
    out, i = [], 0
    while i < len(cps):
        j = i
        while j + 1 < len(cps) and cps[j + 1] == cps[j] + 1:
            j += 1
        a, b = cps[i], cps[j]
        out.append(re.escape(chr(a)) if a == b else re.escape(chr(a)) + "-" + re.escape(chr(b)))
        i = j + 1
    return "".join(out)


def _nfc(t: str) -> str:
    return t if unicodedata.is_normalized("NFC", t) else unicodedata.normalize("NFC", t)


ZWNJ, ZWJ, KASHIDA, SMALL_V = chr(0x200C), chr(0x200D), chr(0x0640), chr(0x065A)
ALLAH_LIGATURE = chr(0xFDF2)
ALLAH_WORD = "".join(map(chr, (0x0627, 0x0644, 0x0644, 0x06C1)))  # corpus spelling, HEH GOAL

ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
ARABIC_SCRIPT = [cp for lo, hi in ARABIC_BLOCKS for cp in range(lo, hi + 1)]
_WORD_CPS = [cp for lo, hi in ARABIC_BLOCKS[:4] for cp in range(lo, hi + 1)
             if unicodedata.category(chr(cp)) in ("Lo", "Lm", "Mn", "Mc")]
WORD_RE = re.compile("[" + _cls(_WORD_CPS + [0x200C]) + "]+")

# letters of Arabic orthography proper; letters only Urdu/Hindko use (evidence of Urdu orthography)
ARABIC_PROPER = frozenset(list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B)) + [0x0671, 0x066E, 0x066F])
URDU_EVIDENCE = frozenset([0x067E, 0x0679, 0x0686, 0x0688, 0x0691, 0x0698, 0x06A9, 0x06AF, 0x06BA, 0x06BE,
                           0x06C1, 0x06C2, 0x06C3, 0x06CC, 0x06D2, 0x06D3, 0x0768] + list(range(0x08BE, 0x08C3)))
URDU_EVIDENCE_MARKS = frozenset([0x065A])
HIGH_HAMZA_YEH = 0x0678
FOLD_ALLOWED = ARABIC_PROPER | URDU_EVIDENCE | {HIGH_HAMZA_YEH}
LETTER_FOLD = {0x064A: 0x06CC, 0x0649: 0x06CC, 0x0643: 0x06A9, 0x0629: 0x06C3, 0x0678: 0x0626}
_LETTER_FOLD_TABLE = {k: chr(v) for k, v in LETTER_FOLD.items()}
_FOLDABLE_RE = re.compile("[" + _cls(LETTER_FOLD) + "]")

TONE_BASE = {0x067E: 0x08BE, 0x062A: 0x08BF, 0x0679: 0x08C0, 0x0686: 0x08C1, 0x06A9: 0x08C2}
_TONE_MAP = {chr(k): chr(v) for k, v in TONE_BASE.items()}
TONE_RE = re.compile("([" + _cls(TONE_BASE) + "])([" + _cls(list(range(0x064B, 0x0653)) + [0x0670]) + "]*)"
                     + re.escape(SMALL_V))

NEWLINE_RE = re.compile("\r\n|[" + _cls([0x0D, 0x0B, 0x0C, 0x85, 0x2028, 0x2029]) + "]")
CONTROL_RE = re.compile("[" + _cls(list(range(0x00, 0x09)) + list(range(0x0B, 0x20)) + list(range(0x7F, 0xA0))) + "]")

PRESENTATION = [cp for lo, hi in ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF)) for cp in range(lo, hi + 1)]
PRESENTATION_KEEP = frozenset(list(range(0xFD3E, 0xFD50)) + [0xFDCF, 0xFDFA, 0xFDFB, 0xFDFC, 0xFDFD, 0xFDFE, 0xFDFF])
PRESENTATION_RE = re.compile("[" + _cls(PRESENTATION) + "]")
ALLAH_RE = re.compile("[" + _cls([0x0627, 0xFE8D, 0xFE8E]) + "]?" + re.escape(ALLAH_LIGATURE))


def _presentation_target(cp: int) -> str:
    c = chr(cp)
    if cp in PRESENTATION_KEEP or cp == 0xFDF2:
        return c
    d = unicodedata.normalize("NFKC", c)
    if d == c:
        return c
    core = d.lstrip(" " + KASHIDA)
    if core and all(unicodedata.category(x) == "Mn" for x in core):
        return core
    if " " in d:
        return c
    return unicodedata.normalize("NFC", d)


_PRESENTATION_MAP = {chr(cp): _presentation_target(cp) for cp in PRESENTATION}

INVISIBLE = ([0x00AD, 0x034F, 0x061C, 0x180E, 0x200B, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064,
              0xFEFF] + list(range(0x202A, 0x202F)) + list(range(0x2066, 0x2070)))
INVISIBLE_RE = re.compile("[" + _cls(INVISIBLE) + "]")
_ZWJ_NEIGH = _cls(ARABIC_SCRIPT + [0x200C, 0x200D]) + r"\s"
ZWJ_RE = re.compile("(?<![^" + _ZWJ_NEIGH + "])" + ZWJ + "|" + ZWJ + "(?![^" + _ZWJ_NEIGH + "])")
SPACES = [0x09] + [cp for cp in range(0x80, 0x3001) if unicodedata.category(chr(cp)) == "Zs"]
SPACES_RE = re.compile("[" + _cls(SPACES) + "]")
MULTISPACE_RE = re.compile(" {2,}")
LINE_EDGE_SPACE_RE = re.compile("^ +| +$", re.M)
MULTI_NL_RE = re.compile("\n{3,}")
ARABIC_INDIC_DIGITS = {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}
ARABIC_INDIC_RE = re.compile("[" + _cls(range(0x0660, 0x066A)) + "]")


def _presentation(t: str) -> str:
    if not PRESENTATION_RE.search(t):
        return t
    t = ALLAH_RE.sub(ALLAH_WORD, t)
    return PRESENTATION_RE.sub(lambda m: _PRESENTATION_MAP[m.group()], t)


def _zwnj(t: str) -> tuple[str, int]:
    if ZWNJ not in t:
        return t, 0
    out, removed, i, n = [], 0, 0, len(t)
    while i < n:
        c = t[i]
        if c != ZWNJ:
            out.append(c)
            i += 1
            continue
        j = i
        while j < n and t[j] == ZWNJ:
            j += 1
        prev_ok = bool(out) and unicodedata.category(out[-1])[0] in "LM"
        next_ok = j < n and unicodedata.category(t[j])[0] in "LM"
        if prev_ok and next_ok:
            out.append(ZWNJ)
            removed += (j - i) - 1
        else:
            removed += j - i
        i = j
    return "".join(out), removed


def _spaces(t: str) -> str:
    t = SPACES_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t)
    t = LINE_EDGE_SPACE_RE.sub("", t)
    return MULTI_NL_RE.sub("\n\n", t)


def _word_is_urdu_orthography(word: str) -> bool:
    evidence = False
    for c in word:
        o = ord(c)
        if unicodedata.category(c)[0] == "M":
            evidence = evidence or o in URDU_EVIDENCE_MARKS
            continue
        if o == 0x200C:
            continue
        if o not in FOLD_ALLOWED:
            return False
        if o in URDU_EVIDENCE or o == HIGH_HAMZA_YEH:
            evidence = True
    return evidence


def _arabic_letters(t: str) -> str:
    if not _FOLDABLE_RE.search(t):
        return t

    def f(m):
        w = m.group()
        if _FOLDABLE_RE.search(w) and _word_is_urdu_orthography(w):
            return w.translate(_LETTER_FOLD_TABLE)
        return w
    return WORD_RE.sub(f, t)


def normalize(text: str) -> str:
    """Return the canonical data form of `text` (the form the tokenizer was trained and evaluated on)."""
    t = _nfc(text)                                           # R01
    t = NEWLINE_RE.sub("\n", t)                              # R02
    t = CONTROL_RE.sub("", t)                                # R03
    t = _presentation(t)                                     # R04
    t = t.replace(KASHIDA, "")                               # R05
    t = INVISIBLE_RE.sub("", t)                              # R06
    t = _nfc(t)                                              # R10 before R07/R08
    for _ in range(t.count(ZWJ) + t.count(ZWNJ) + 1):        # R07 R08 R09 R10 until stable
        t, k1 = ZWJ_RE.subn("", t) if ZWJ in t else (t, 0)
        t, k2 = _zwnj(t)
        t = _nfc(_spaces(t))
        if not (k1 or k2):
            break
    t = _arabic_letters(t)                                   # R11 (only inside provably Urdu/Hindko words)
    if SMALL_V in t:                                         # R12 Hindko tone letters
        t = TONE_RE.sub(lambda m: _TONE_MAP[m.group(1)] + m.group(2), t)
    t = ARABIC_INDIC_RE.sub(lambda m: ARABIC_INDIC_DIGITS[m.group()], t)  # R13
    return _nfc(t)                                           # R14


def fold_arabic_yeh_kaf(text: str) -> str:
    """Optional, stronger fix for Arabic-keyboard input: fold EVERY ي->ی and ك->ک.

    Use only when the input is known to be Hindko or Urdu (it also rewrites Arabic quotations).
    ه (Arabic HEH) is deliberately not folded: Arabic-keyboard users type it for both ہ and ھ."""
    return text.replace("ي", "ی").replace("ك", "ک")


# ============================================================================ demo
def load(path_or_repo: str):
    """Load with transformers if available, else with tokenizers. Works for a local folder or the hub."""
    token = os.environ.get("HF_TOKEN")
    try:
        from transformers import AutoTokenizer
        return "transformers", AutoTokenizer.from_pretrained(path_or_repo, token=token)
    except ImportError:
        from tokenizers import Tokenizer
        if Path(path_or_repo).is_dir():
            return "tokenizers", Tokenizer.from_file(str(Path(path_or_repo) / "tokenizer.json"))
        return "tokenizers", Tokenizer.from_pretrained(path_or_repo, token=token)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else (
        str(Path(__file__).resolve().parents[1]) if (Path(__file__).resolve().parents[1] / "tokenizer.json").exists()
        else REPO_ID)
    kind, tok = load(src)
    enc = (lambda s: tok(s)["input_ids"]) if kind == "transformers" else (lambda s: tok.encode(s).ids)
    dec = ((lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
           if kind == "transformers" else (lambda ids: tok.decode(ids, skip_special_tokens=False)))
    pieces = (lambda ids: tok.convert_ids_to_tokens(ids)) if kind == "transformers" else (
        lambda ids: [tok.id_to_token(i) for i in ids])
    print(f"loaded from {src} with {kind}")

    # 1. encode / decode (a sentence from the held-out test split; ~ "Very few people know how and where drama was born.")
    text = "ایہہ بہت کہٹ لوک جانڑدین کہ ڈرامے دی پیدائش کسراں تے کتھے ہوئی آئی۔"
    ids = enc(text)
    print(f"\n{len(text.split())} words -> {len(ids)} tokens")
    print(" | ".join(pieces(ids)))
    assert dec(ids) == text, "round trip failed"
    print("round trip exact: True  (no BOS/EOS added; add <|bos|>=1 / <|endoftext|>=0 yourself)")

    # 2. chat template (transformers only)
    if kind == "transformers":
        msgs = [{"role": "user", "content": "سلام! تساں کیہہ حال اے؟"}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print("\nchat template:", repr(rendered))

    # 3. messy input: normalize() first
    messy = "ایہہ  بہت​ کہٹ لوک\r\nجانڑدین کہ ڈرامے دی پیدائش"
    print(f"\nmessy input: raw {len(enc(messy))} tokens, normalize() {len(enc(normalize(messy)))} tokens")

    # 4. Arabic-keyboard letters (ي ك ه typed for ی ک ہ): the tokenizer's clearest weak spot
    ak = text.replace("ی", "ي").replace("ک", "ك").replace("ہ", "ه")
    n_clean, n_ak = len(enc(text)), len(enc(ak))
    n_norm, n_fold = len(enc(normalize(ak))), len(enc(fold_arabic_yeh_kaf(normalize(ak))))
    print(f"Arabic-keyboard typing: clean {n_clean}, as typed {n_ak}, normalize() {n_norm}, "
          f"normalize()+YEH/KAF fold {n_fold} tokens")


if __name__ == "__main__":
    main()
