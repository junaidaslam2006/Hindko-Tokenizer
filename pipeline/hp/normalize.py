"""Canonical text form ("data form") for the Hindko tokenizer and models.

Spec, evidence and numbers: F:/Hindko/_tokenizer/normalization/NORMALIZATION.md
Tests:                      F:/Hindko/_tokenizer/normalization/test_normalize.py

Two levels are kept apart:
  (a) DATA form - normalize() below. Applied to corpus text before tokenizer
      training and before any evaluation. Conservative: it folds only
      variants that are encoding noise (presentation forms, invisible and
      control characters, odd spaces, Arabic-keyboard letters inside words
      that are provably written in Urdu/Hindko orthography, Arabic-Indic
      digits, decomposed Hindko tone letters). It never merges letters that
      are different in Urdu/Hindko (e.g. HEH GOAL vs HEH DOACHASHMEE), never
      touches Arabic HEH (ambiguous), punctuation, quotes, Latin text,
      ASCII vs Urdu digits, harakat, U+FFFD or the <PHONE>/<EMAIL>
      placeholders.
  (b) TOKENIZER - lossless byte-level BPE with no normalizer; normalize() is
      exported for users who want their input in the training form.

normalize(text) is deterministic, idempotent, and NFC before and after.
The character classes are built from Python's unicodedata (Python 3.11 ships
Unicode 14.0.0); NORMALIZATION_VERSION changes whenever a rule changes.

Rules (each rule's id is used by change_report):
  R01 nfc            NFC (input may be in any normalization form)
  R02 newlines       CRLF, CR, VT, FF, NEL, LS, PS -> LF
  R03 controls       C0 (except TAB, LF), DEL and C1 controls removed
  R04 presentation   Arabic presentation forms U+FB50-FDFF, U+FE70-FEFF ->
                     their NFKC letters; ALLAH ligature -> the corpus spelling
                     with HEH GOAL (a directly preceding alef, or alef
                     presentation form, is absorbed);
                     spacing/tatweel forms of harakat -> the bare mark;
                     symbols without a letter equivalent are kept (SAW
                     ligature, ornate parentheses, bismillah, rial sign, ...)
  R05 kashida        U+0640 removed (settled decision)
  R06 invisible      ZWSP, LRM, RLM, ALM, bidi embeddings/overrides/isolates,
                     WORD JOINER, invisible operators, BOM, SOFT HYPHEN, CGJ,
                     MVS, deprecated format characters removed
  R07 zwj            ZWJ removed unless both neighbours are non-Arabic,
                     non-space characters (keeps emoji ZWJ sequences)
  R08 zwnj           ZWNJ kept only between two letters/marks (settled:
                     preserved); runs collapse to one; elsewhere removed
  R09 spaces         every Unicode space separator and TAB -> SPACE; runs of
                     spaces collapse; spaces at line ends stripped; 3+ LF ->
                     2 LF
  R10 nfc2           NFC after the removals and conversions of R03-R09
  R11 arabic_letters inside a word proven to be in Urdu/Hindko orthography:
                     ARABIC YEH / ALEF MAKSURA -> FARSI YEH, ARABIC KAF ->
                     KEHEH, TEH MARBUTA -> TEH MARBUTA GOAL, HIGH HAMZA YEH
                     -> YEH WITH HAMZA ABOVE
  R12 tone_letters   PEH/TEH/TTEH/TCHEH/KEHEH (+harakat) + U+065A -> the
                     Hindko letters U+08BE..U+08C2 (as hp/inpage.py does)
  R13 digits         Arabic-Indic digits U+0660-0669 -> Extended
                     Arabic-Indic U+06F0-06F9 (ASCII digits are kept)
  R14 nfc_final      NFC guard (a no-op when the rules above are correct)

Execution order (since 1.0.1):
  1. R01 R02 R03 R04 R05 R06, once.
  2. R10 (NFC), so that R07 and R08 judge neighbours in canonical order.
     R04 can turn a presentation form into a haraka, and R03/R05/R06 removals
     can bring a base and its marks together; before NFC the neighbour of a
     ZWJ/ZWNJ is then not the one it has in the output.
  3. The context block R07 R08 R09 R10, repeated until R07 and R08 remove
     nothing. A ZWJ/ZWNJ removal lets NFC reorder or compose the marks on
     either side of it, which can give another ZWJ/ZWNJ a new neighbour
     (e.g. '=' + U+0338 composes to the symbol U+2260, so a ZWNJ after it is
     no longer between letters/marks; an Arabic mark reordered past a
     non-Arabic one becomes a ZWJ's neighbour). Every extra pass removes at
     least one ZWJ/ZWNJ, so there are at most (#ZWJ + #ZWNJ + 1) passes; real
     text needs one.
  4. R11 R12 R13 R14, once.
Why the result is a fixed point: after step 3 the text is NFC and no rule of
the block changes it. R11/R12/R13 only swap Arabic-script letters for
Arabic-script letters (R12 also drops U+065A after a tone base), which never
changes the category of a ZWJ/ZWNJ neighbour and never creates a canonical
composition or reordering; R01-R06 find nothing left to do on their output.
The test suite checks change_report(normalize(x)) == {} on fuzz strings and on
every corpus record.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

# 1.0.0  first version
# 1.0.1  idempotency fix: R07/R08 now run on NFC-ordered text and repeat with
#        NFC until stable; R04 also absorbs an alef presentation form before
#        the ALLAH ligature. Output on the released corpus is unchanged.
NORMALIZATION_VERSION = '1.0.1'
UNICODE_VERSION = unicodedata.unidata_version

RULES: List[Tuple[str, str]] = [
    ('R01_nfc', 'Unicode NFC'),
    ('R02_newlines', 'CRLF/CR/VT/FF/NEL/LS/PS -> LF'),
    ('R03_controls', 'C0 (except TAB, LF), DEL, C1 controls removed'),
    ('R04_presentation', 'Arabic presentation forms -> letters (NFKC), ALLAH ligature -> HEH GOAL spelling'),
    ('R05_kashida', 'tatweel U+0640 removed'),
    ('R06_invisible', 'ZWSP, bidi controls, BOM, WJ, SHY, CGJ, ... removed'),
    ('R07_zwj', 'ZWJ removed next to Arabic script, space or text edge'),
    ('R08_zwnj', 'ZWNJ kept only between letters/marks; runs collapsed'),
    ('R09_spaces', 'space separators/TAB -> SPACE, runs collapsed, line ends stripped, 3+ LF -> 2'),
    ('R10_nfc2', 'NFC before R07 and after every R07-R09 pass'),
    ('R11_arabic_letters', 'ي ى ك ة ٸ -> ی ی ک ۃ ئ inside proven Urdu/Hindko words'),
    ('R12_tone_letters', 'letter + U+065A -> Hindko tone letter U+08BE..U+08C2'),
    ('R13_digits', 'Arabic-Indic digits -> Extended Arabic-Indic digits'),
    ('R14_nfc_final', 'NFC guard'),
]
RULE_IDS = [r for r, _ in RULES]


def _cls(cps) -> str:
    """Regex character class body for an iterable of codepoints."""
    cps = sorted(set(cps))
    out, i = [], 0
    while i < len(cps):
        j = i
        while j + 1 < len(cps) and cps[j + 1] == cps[j] + 1:
            j += 1
        a, b = cps[i], cps[j]
        out.append(re.escape(chr(a)) if a == b else re.escape(chr(a)) + '-' + re.escape(chr(b)))
        i = j + 1
    return ''.join(out)


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


# ---------------------------------------------------------------------------
# codepoint sets (all built from integers so the file bytes stay ASCII)
# ---------------------------------------------------------------------------
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
KASHIDA = chr(0x0640)
SMALL_V = chr(0x065A)
ALEF = chr(0x0627)
ALLAH_LIGATURE = chr(0xFDF2)
# the corpus spelling of the ALLAH ligature (hp/inpage.py: U+06C1 HEH GOAL)
ALLAH_WORD = ''.join(map(chr, (0x0627, 0x0644, 0x0644, 0x06C1)))

ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
ARABIC_SCRIPT = [cp for lo, hi in ARABIC_BLOCKS for cp in range(lo, hi + 1)]
# letters and marks of the (non-presentation) Arabic blocks: what a "word" is made of
_WORD_CPS = [cp for lo, hi in ARABIC_BLOCKS[:4] for cp in range(lo, hi + 1)
             if unicodedata.category(chr(cp)) in ('Lo', 'Lm', 'Mn', 'Mc')]
WORD_RE = re.compile('[' + _cls(_WORD_CPS + [0x200C]) + ']+')

# Letters of Arabic orthography proper (hamza .. ghain, feh .. yeh, alef wasla,
# dotless beh/qaf). A word made only of these may be Arabic.
ARABIC_PROPER = frozenset(list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B)) + [0x0671, 0x066E, 0x066F])
# Letters of the Urdu/Hindko alphabet that Arabic orthography never uses:
# PEH TTEH TCHEH DDAL RREH JEH KEHEH GAF NOON-GHUNNA HEH-DOACHASHMEE HEH-GOAL
# HEH-GOAL+HAMZA TEH-MARBUTA-GOAL FARSI-YEH YEH-BARREE YEH-BARREE+HAMZA, the
# Hindko tone letters U+08BE..U+08C2 and NOON WITH SMALL TAH (retroflex nasal).
URDU_EVIDENCE = frozenset([0x067E, 0x0679, 0x0686, 0x0688, 0x0691, 0x0698, 0x06A9, 0x06AF, 0x06BA, 0x06BE,
                           0x06C1, 0x06C2, 0x06C3, 0x06CC, 0x06D2, 0x06D3, 0x0768]
                          + list(range(0x08BE, 0x08C3)))
# Marks that Arabic orthography never uses: U+065A SMALL V ABOVE (Hindko tone
# mark / dictionary jazm). Counted as evidence so that R11 and R12 commute.
URDU_EVIDENCE_MARKS = frozenset([0x065A])
HIGH_HAMZA_YEH = 0x0678
# A word may be folded only if all its letters are in this set: any other
# Arabic-script letter (Pashto, Sindhi, Saraiki implosives, Kashmiri, Kurdish,
# ...) blocks the rule, because e.g. Sindhi and Pashto use ARABIC YEH on purpose.
FOLD_ALLOWED = ARABIC_PROPER | URDU_EVIDENCE | {HIGH_HAMZA_YEH}
LETTER_FOLD = {0x064A: 0x06CC,   # ARABIC YEH       -> FARSI YEH
               0x0649: 0x06CC,   # ALEF MAKSURA     -> FARSI YEH
               0x0643: 0x06A9,   # ARABIC KAF       -> KEHEH
               0x0629: 0x06C3,   # TEH MARBUTA      -> TEH MARBUTA GOAL
               0x0678: 0x0626}   # HIGH HAMZA YEH   -> YEH WITH HAMZA ABOVE
_LETTER_FOLD_TABLE = {k: chr(v) for k, v in LETTER_FOLD.items()}
_FOLDABLE_RE = re.compile('[' + _cls(LETTER_FOLD) + ']')

# Hindko tone letters (Unicode 13): PEH/TEH/TTEH/TCHEH/KEHEH WITH SMALL V
TONE_BASE = {0x067E: 0x08BE, 0x062A: 0x08BF, 0x0679: 0x08C0, 0x0686: 0x08C1, 0x06A9: 0x08C2}
_TONE_MAP = {chr(k): chr(v) for k, v in TONE_BASE.items()}
TONE_RE = re.compile('([' + _cls(TONE_BASE) + '])([' + _cls(list(range(0x064B, 0x0653)) + [0x0670]) + ']*)'
                     + re.escape(SMALL_V))

# R02 / R03
NEWLINE_RE = re.compile('\r\n|[' + _cls([0x0D, 0x0B, 0x0C, 0x85, 0x2028, 0x2029]) + ']')
CONTROL_RE = re.compile('[' + _cls(list(range(0x00, 0x09)) + list(range(0x0B, 0x20)) + list(range(0x7F, 0xA0))) + ']')

# R04 presentation forms
PRESENTATION = [cp for lo, hi in ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF)) for cp in range(lo, hi + 1)]
# kept as they are: symbols with no letter equivalent in running Urdu text
PRESENTATION_KEEP = frozenset(list(range(0xFD3E, 0xFD50))                 # ornate parentheses, honorific ligatures (U14)
                              + [0xFDCF, 0xFDFA, 0xFDFB, 0xFDFC, 0xFDFD, 0xFDFE, 0xFDFF])
PRESENTATION_RE = re.compile('[' + _cls(PRESENTATION) + ']')
# ALEF, ALEF ISOLATED FORM, ALEF FINAL FORM
ALLAH_RE = re.compile('[' + _cls([0x0627, 0xFE8D, 0xFE8E]) + ']?' + re.escape(ALLAH_LIGATURE))


def _presentation_target(cp: int) -> str:
    c = chr(cp)
    if cp in PRESENTATION_KEEP or cp == 0xFDF2:
        return c
    d = unicodedata.normalize('NFKC', c)
    if d == c:
        return c                       # unassigned / no decomposition (e.g. BOM, which R06 removes)
    core = d.lstrip(' ' + KASHIDA)
    if core and all(unicodedata.category(x) == 'Mn' for x in core):
        return core                    # spacing / tatweel forms of harakat -> the bare mark
    if ' ' in d:
        return c                       # multi-word ligature: keep the symbol
    return _nfc(d)


_PRESENTATION_MAP = {chr(cp): _presentation_target(cp) for cp in PRESENTATION}

# R06 invisible format characters (ZWNJ and ZWJ have their own rules)
INVISIBLE = ([0x00AD, 0x034F, 0x061C, 0x180E, 0x200B, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064,
              0xFEFF] + list(range(0x202A, 0x202F)) + list(range(0x2066, 0x2070)))
INVISIBLE_RE = re.compile('[' + _cls(INVISIBLE) + ']')

# R07 ZWJ: removed when a neighbour is Arabic script, whitespace, ZWJ/ZWNJ or the text edge
_ZWJ_NEIGH = _cls(ARABIC_SCRIPT + [0x200C, 0x200D]) + r'\s'
ZWJ_RE = re.compile('(?<![^' + _ZWJ_NEIGH + '])' + ZWJ + '|' + ZWJ + '(?![^' + _ZWJ_NEIGH + '])')

# R09 spaces: all Zs except SPACE, plus TAB
SPACES = [0x09] + [cp for cp in range(0x80, 0x3001) if unicodedata.category(chr(cp)) == 'Zs']
SPACES_RE = re.compile('[' + _cls(SPACES) + ']')
MULTISPACE_RE = re.compile(' {2,}')
LINE_EDGE_SPACE_RE = re.compile('^ +| +$', re.M)
MULTI_NL_RE = re.compile('\n{3,}')

# R13
ARABIC_INDIC_DIGITS = {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}
ARABIC_INDIC_RE = re.compile('[' + _cls(range(0x0660, 0x066A)) + ']')


# ---------------------------------------------------------------------------
# rules: each returns (new_text, number of input characters changed)
# ---------------------------------------------------------------------------
def _count_diff(a: str, b: str) -> int:
    """Characters of `a` not kept unchanged in `b` (only used for NFC, rare)."""
    if a == b:
        return 0
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(i2 - i1 for tag, i1, i2, _, _ in sm.get_opcodes() if tag != 'equal') or 1


def r_nfc(t: str) -> Tuple[str, int]:
    if unicodedata.is_normalized('NFC', t):
        return t, 0
    n = _nfc(t)
    return n, _count_diff(t, n)


def r_newlines(t: str) -> Tuple[str, int]:
    n = 0

    def f(m):
        nonlocal n
        n += len(m.group())
        return '\n'
    return NEWLINE_RE.sub(f, t), n


def _remove(rx, t: str) -> Tuple[str, int]:
    out, n = rx.subn('', t)
    return out, n


def r_controls(t: str) -> Tuple[str, int]:
    return _remove(CONTROL_RE, t)


def r_presentation(t: str) -> Tuple[str, int]:
    if not PRESENTATION_RE.search(t):
        return t, 0
    n = 0

    def allah(m):
        nonlocal n
        n += len(m.group())
        return ALLAH_WORD
    t = ALLAH_RE.sub(allah, t)

    def f(m):
        nonlocal n
        c = m.group()
        r = _PRESENTATION_MAP[c]
        if r != c:
            n += 1
        return r
    return PRESENTATION_RE.sub(f, t), n


def r_kashida(t: str) -> Tuple[str, int]:
    k = t.count(KASHIDA)
    return (t.replace(KASHIDA, ''), k) if k else (t, 0)


def r_invisible(t: str) -> Tuple[str, int]:
    return _remove(INVISIBLE_RE, t)


def r_zwj(t: str) -> Tuple[str, int]:
    if ZWJ not in t:
        return t, 0
    return _remove(ZWJ_RE, t)


def _is_letter_or_mark(c: str) -> bool:
    return unicodedata.category(c)[0] in 'LM'


def r_zwnj(t: str) -> Tuple[str, int]:
    if ZWNJ not in t:
        return t, 0
    out, n, i, L = [], 0, 0, len(t)
    while i < L:
        c = t[i]
        if c != ZWNJ:
            out.append(c)
            i += 1
            continue
        j = i
        while j < L and t[j] == ZWNJ:
            j += 1
        run = j - i
        prev_ok = bool(out) and _is_letter_or_mark(out[-1])
        next_ok = j < L and _is_letter_or_mark(t[j])
        if prev_ok and next_ok:
            out.append(ZWNJ)
            n += run - 1
        else:
            n += run
        i = j
    return ''.join(out), n


def r_spaces(t: str) -> Tuple[str, int]:
    n = 0
    t, k = SPACES_RE.subn(' ', t)
    n += k

    def shrink(m):
        nonlocal n
        n += len(m.group()) - 1
        return ' '
    t = MULTISPACE_RE.sub(shrink, t)

    def strip(m):
        nonlocal n
        n += len(m.group())
        return ''
    t = LINE_EDGE_SPACE_RE.sub(strip, t)

    def nl(m):
        nonlocal n
        n += len(m.group()) - 2
        return '\n\n'
    t = MULTI_NL_RE.sub(nl, t)
    return t, n


def word_is_urdu_orthography(word: str) -> bool:
    """True if every letter of `word` belongs to Arabic + Urdu/Hindko
    orthography and at least one letter (or U+065A) never occurs in Arabic
    orthography. Marks and ZWNJ are otherwise ignored."""
    evidence = False
    for c in word:
        o = ord(c)
        cat = unicodedata.category(c)
        if cat[0] == 'M':
            if o in URDU_EVIDENCE_MARKS:
                evidence = True
            continue
        if o == 0x200C:
            continue
        if o not in FOLD_ALLOWED:
            return False
        if o in URDU_EVIDENCE or o == HIGH_HAMZA_YEH:
            evidence = True
    return evidence


def r_arabic_letters(t: str) -> Tuple[str, int]:
    if not _FOLDABLE_RE.search(t):
        return t, 0
    n = 0

    def f(m):
        nonlocal n
        w = m.group()
        if not _FOLDABLE_RE.search(w) or not word_is_urdu_orthography(w):
            return w
        k = len(_FOLDABLE_RE.findall(w))
        n += k
        return w.translate(_LETTER_FOLD_TABLE)
    return WORD_RE.sub(f, t), n


def r_tone_letters(t: str) -> Tuple[str, int]:
    if SMALL_V not in t:
        return t, 0
    n = 0

    def f(m):
        nonlocal n
        n += 2                         # base letter + small v become one letter
        return _TONE_MAP[m.group(1)] + m.group(2)
    return TONE_RE.sub(f, t), n


def r_digits(t: str) -> Tuple[str, int]:
    if not ARABIC_INDIC_RE.search(t):
        return t, 0
    n = 0

    def f(m):
        nonlocal n
        n += 1
        return ARABIC_INDIC_DIGITS[m.group()]
    return ARABIC_INDIC_RE.sub(f, t), n


# step 1: once
_PIPELINE_PRE = [
    ('R01_nfc', r_nfc),
    ('R02_newlines', r_newlines),
    ('R03_controls', r_controls),
    ('R04_presentation', r_presentation),
    ('R05_kashida', r_kashida),
    ('R06_invisible', r_invisible),
]
# step 2 runs R10 once; step 3 repeats this block until R07 and R08 remove nothing
_CONTEXT_BLOCK = [
    ('R07_zwj', r_zwj),
    ('R08_zwnj', r_zwnj),
    ('R09_spaces', r_spaces),
    ('R10_nfc2', r_nfc),
]
_CONTEXT_RULES = ('R07_zwj', 'R08_zwnj')
# step 4: once
_PIPELINE_POST = [
    ('R11_arabic_letters', r_arabic_letters),
    ('R12_tone_letters', r_tone_letters),
    ('R13_digits', r_digits),
    ('R14_nfc_final', r_nfc),
]
assert [r for r, _ in _PIPELINE_PRE + _CONTEXT_BLOCK + _PIPELINE_POST] == RULE_IDS


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def normalize_with_report(text: str) -> Tuple[str, Dict[str, int]]:
    """Return (canonical_text, {rule_id: input characters changed}); only
    rules that changed something appear in the dict. A rule that runs more
    than once (R10, and R07-R09 when the context block repeats) reports the
    sum over its runs."""
    report: Dict[str, int] = {}

    def run(rid, fn) -> int:
        nonlocal text
        text, k = fn(text)
        if k:
            report[rid] = report.get(rid, 0) + k
        return k

    for rid, fn in _PIPELINE_PRE:
        run(rid, fn)
    run('R10_nfc2', r_nfc)                           # canonical order before R07/R08 look at neighbours
    # every pass except the last removes >= 1 ZWJ/ZWNJ, and no rule adds one
    max_passes = text.count(ZWJ) + text.count(ZWNJ) + 1
    for _ in range(max_passes):
        removed = 0
        for rid, fn in _CONTEXT_BLOCK:
            k = run(rid, fn)
            if rid in _CONTEXT_RULES:
                removed += k
        if not removed:
            break
    else:                                            # unreachable; kept as a guard
        raise RuntimeError('normalize: ZWJ/ZWNJ block did not reach a fixed point')
    for rid, fn in _PIPELINE_POST:
        run(rid, fn)
    return text, report


def normalize(text: str) -> str:
    """The canonical data form of `text` (see module docstring)."""
    return normalize_with_report(text)[0]


def change_report(text: str) -> Dict[str, int]:
    """{rule_id: characters changed} for `text` (empty when already canonical)."""
    return normalize_with_report(text)[1]


def is_canonical(text: str) -> bool:
    return normalize(text) == text


__all__ = ['normalize', 'normalize_with_report', 'change_report', 'is_canonical', 'word_is_urdu_orthography',
           'NORMALIZATION_VERSION', 'UNICODE_VERSION', 'RULES', 'RULE_IDS']
