# -*- coding: utf-8 -*-
"""The four robustness perturbations of PLAN.md 4.2 (TokSuite-style), with exact character alignment.

Each perturbation maps an ORIGINAL (canonical) text to a PERTURBED text and returns a
`Perturbed` object:
    text       the perturbed string
    orig_idx   int array, one entry per perturbed character: the index of the original character
               it came from (a replaced character keeps its origin), or -1 for an inserted one
    edited     bool array over ORIGINAL characters: True where the perturbation touched the text
               (deleted or replaced characters, the character right after an insertion point, and
               for a deleted space the characters on both sides of it). A word is "affected" if
               any of its characters is edited.
The alignment lets the harness compare a tokenizer's segmentation of every original word before
and after the perturbation on the characters the two texts share (harness.robustness).

Definitions (all codepoints built with chr(), so this file stays ASCII-safe):
  harakat      delete every Arabic mark U+064B..U+065F and U+0670 (the harakat class of
               hp.lang.HARAKAT_RE: tanween, fatha/damma/kasra, shadda, sukun, maddah, hamza
               marks, subscript alef, small v (U+065A), ..., superscript alef). A word made only
               of marks vanishes; the space runs this leaves are collapsed and line-edge spaces
               removed, as normalize R09 would.
  digits       swap the digit script: ASCII 0-9 <-> Extended Arabic-Indic U+06F0..U+06F9
               (both directions at once, value-preserving).
  punct_space  toggle the SPACE (U+0020) immediately before U+06D4 ARABIC FULL STOP and
               U+060C ARABIC COMMA: delete it where present (the PLAN 4.2 wording), insert it
               where the mark follows a non-whitespace character. DEVIATION: the canonical
               corpus never has that space (0 occurrences in train D1 and dev), so the literal
               "delete" probe would touch nothing; on our data this probe inserts the space.
  zwnj         insert ZWNJ U+200C at the element boundary of Perso-Urdu compounds found by a
               fixed element list (COMPOUND_PREFIXES / COMPOUND_SUFFIXES below): at most one
               ZWNJ per word, prefix rule first, longest suffix first. The corpus contains no
               ZWNJ at all, so this probe measures how a tokenizer copes with the half-space
               that Urdu keyboards produce in such compounds. The list was chosen for precision
               from a manual look at dev word types (short, noisy elements such as ہم کم نا بد
               وار بان دان ستان were left out); it is small by design and some matches are
               still not compounds (e.g. اقتدار). The affected words are listed by
               harness.perturbation_census().
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ZWNJ = chr(0x200C)
SPACE = " "
FULL_STOP = chr(0x06D4)
ARABIC_COMMA = chr(0x060C)
HARAKAT = frozenset(list(range(0x064B, 0x0660)) + [0x0670])
_ASCII = [chr(0x30 + i) for i in range(10)]
_EXT = [chr(0x06F0 + i) for i in range(10)]
DIGIT_SWAP = {**{a: e for a, e in zip(_ASCII, _EXT)}, **{e: a for a, e in zip(_ASCII, _EXT)}}


def _s(*cps):
    return "".join(chr(c) for c in cps)


# Arabic-script letters and marks (the WORD_RE class of hp.normalize, without ZWNJ)
_ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF))
_WORD_CPS = sorted(cp for lo, hi in _ARABIC_BLOCKS for cp in range(lo, hi + 1)
                   if unicodedata.category(chr(cp)) in ("Lo", "Lm", "Mn", "Mc"))
ARABIC_WORD_RE = re.compile("[" + "".join(re.escape(chr(c)) for c in _WORD_CPS) + "]+")

# (element, minimum number of LETTERS in the rest of the word)
COMPOUND_PREFIXES: List[Tuple[str, int]] = [
    (_s(0x0628, 0x06D2), 3),                  # be-     'without'
    (_s(0x062E, 0x0648, 0x0634), 3),          # khush-  'good, happy'   (rest must not start with yeh: khushi)
    (_s(0x063A, 0x06CC, 0x0631), 3),          # ghair-  'non-'
]
PREFIX_REST_NOT_START = {_s(0x062E, 0x0648, 0x0634): {chr(0x06CC)}}
COMPOUND_SUFFIXES: List[Tuple[str, int]] = [
    (_s(0x062E, 0x0627, 0x0646, 0x06C1), 2), (_s(0x062E, 0x0627, 0x0646, 0x06D2), 2),              # khana/khane 'house'
    (_s(0x062E, 0x0627, 0x0646, 0x0648, 0x06BA), 2), (_s(0x062E, 0x0627, 0x0646, 0x0627, 0x06BA), 2),
    (_s(0x0646, 0x0627, 0x0645, 0x06C1), 2), (_s(0x0646, 0x0627, 0x0645, 0x06D2), 2),              # nama/name 'letter, book'
    (_s(0x0646, 0x0627, 0x0645, 0x0648, 0x06BA), 2), (_s(0x0646, 0x0627, 0x0645, 0x0627, 0x06BA), 2),
    (_s(0x06AF, 0x0627, 0x06C1), 2), (_s(0x06AF, 0x0627, 0x06C1, 0x06CC, 0x06BA), 2),              # gah 'place'
    (_s(0x06AF, 0x0627, 0x06C1, 0x0627, 0x06BA), 2), (_s(0x06AF, 0x0627, 0x06C1, 0x0648, 0x06BA), 2),
    (_s(0x062F, 0x0627, 0x0631), 3), (_s(0x062F, 0x0627, 0x0631, 0x06CC), 3),                      # dar 'holder'
    (_s(0x062F, 0x0627, 0x0631, 0x0627, 0x06BA), 3), (_s(0x062F, 0x0627, 0x0631, 0x0648, 0x06BA), 3),
    (_s(0x0645, 0x0646, 0x062F), 3), (_s(0x0645, 0x0646, 0x062F, 0x06CC), 3),                      # mand 'possessing'
    (_s(0x0646, 0x06AF, 0x0627, 0x0631), 2),                                                        # nigar 'writer'
    (_s(0x06AF, 0x0627, 0x0631), 3), (_s(0x06AF, 0x0627, 0x0631, 0x06CC), 3),                      # gar 'doer'
    (_s(0x0622, 0x0628, 0x0627, 0x062F), 2),                                                        # abad 'town'
    (_s(0x0632, 0x0627, 0x062F, 0x06C1), 2), (_s(0x0632, 0x0627, 0x062F, 0x06D2), 2),              # zada 'born of'
    (_s(0x0632, 0x0627, 0x062F, 0x06CC), 2), (_s(0x0632, 0x0627, 0x062F, 0x06CC, 0x0627, 0x06BA), 2),
    (_s(0x0628, 0x0627, 0x0632), 3),                                                                # baz 'player'
]
COMPOUND_SUFFIXES.sort(key=lambda x: -len(x[0]))   # longest first (nigar before gar)


def _letters(s: str) -> int:
    return sum(1 for c in s if unicodedata.category(c)[0] == "L")


def compound_split(word: str) -> Optional[int]:
    """Index at which a ZWNJ is inserted in `word` (a run of Arabic letters/marks), or None."""
    for pre, min_rest in COMPOUND_PREFIXES:
        if word.startswith(pre):
            rest = word[len(pre):]
            if (_letters(rest) >= min_rest and unicodedata.category(rest[0])[0] == "L"
                    and rest[0] not in PREFIX_REST_NOT_START.get(pre, ())):
                return len(pre)
    for suf, min_first in COMPOUND_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            first = word[:-len(suf)]
            if _letters(first) >= min_first:
                return len(first)
    return None


@dataclass
class Perturbed:
    name: str
    text: str
    orig_idx: np.ndarray      # int32, len(text)
    edited: np.ndarray        # bool, len(original)
    n_edits: int              # number of edit operations (chars deleted/replaced/inserted)


def _build(text: str, delete: Sequence[int] = (), replace: Optional[Dict[int, str]] = None,
           insert_before: Optional[Dict[int, str]] = None) -> Tuple[str, List[int]]:
    """Apply deletions, 1:1 replacements and insertions (all in ORIGINAL coordinates)."""
    dele = set(delete)
    replace = replace or {}
    insert_before = insert_before or {}
    out, idx = [], []
    for i, c in enumerate(text):
        ins = insert_before.get(i)
        if ins:
            out.extend(ins); idx.extend([-1] * len(ins))
        if i in dele:
            continue
        out.append(replace.get(i, c)); idx.append(i)
    ins = insert_before.get(len(text))
    if ins:
        out.extend(ins); idx.extend([-1] * len(ins))
    return "".join(out), idx


def _fix_spaces(text: str, idx: List[int]) -> Tuple[str, List[int], List[int]]:
    """Collapse space runs and strip spaces at line edges (normalize R09 behaviour), tracking origins.
    Returns (text, idx, original indices of the removed spaces)."""
    n = len(text)
    drop = set()
    for i, c in enumerate(text):
        if c != SPACE:
            continue
        prev = text[i - 1] if i > 0 else "\n"
        nxt = text[i + 1] if i + 1 < n else "\n"
        if prev == SPACE or prev == "\n" or nxt == "\n":
            drop.add(i)
    if not drop:
        return text, idx, []
    removed = [idx[i] for i in sorted(drop) if idx[i] >= 0]
    keep = [i for i in range(n) if i not in drop]
    return "".join(text[i] for i in keep), [idx[i] for i in keep], removed


def _finish(name: str, orig: str, text: str, idx: List[int], edited: np.ndarray, n_edits: int) -> Perturbed:
    return Perturbed(name, text, np.asarray(idx, dtype=np.int32), edited, n_edits)


def _mark_neighbours(orig: str, edited: np.ndarray, pos: int):
    """Mark the nearest non-space characters left and right of original position `pos`."""
    j = pos - 1
    while j >= 0 and orig[j] == SPACE:
        j -= 1
    if j >= 0 and not orig[j].isspace():
        edited[j] = True
    j = pos + 1
    while j < len(orig) and orig[j] == SPACE:
        j += 1
    if j < len(orig) and not orig[j].isspace():
        edited[j] = True


def perturb_harakat(orig: str) -> Perturbed:
    dele = [i for i, c in enumerate(orig) if ord(c) in HARAKAT]
    edited = np.zeros(len(orig), dtype=bool)
    if not dele:
        return _finish("harakat", orig, orig, list(range(len(orig))), edited, 0)
    edited[dele] = True
    text, idx = _build(orig, delete=dele)
    text, idx, removed = _fix_spaces(text, idx)
    for r in removed:
        edited[r] = True
        _mark_neighbours(orig, edited, r)
    return _finish("harakat", orig, text, idx, edited, len(dele) + len(removed))


def perturb_digits(orig: str) -> Perturbed:
    rep = {i: DIGIT_SWAP[c] for i, c in enumerate(orig) if c in DIGIT_SWAP}
    edited = np.zeros(len(orig), dtype=bool)
    if rep:
        edited[list(rep)] = True
    text, idx = _build(orig, replace=rep)
    return _finish("digits", orig, text, idx, edited, len(rep))


def perturb_punct_space(orig: str) -> Perturbed:
    """Toggle the space before U+06D4 / U+060C: delete it where present (PLAN wording), insert it where the
    mark directly follows a non-whitespace character. The canonical corpus has 0 spaces before these marks
    (train D1: 0 vs 513,754 attached), so on our data this probe inserts the space."""
    dele, ins = [], {}
    for i in range(1, len(orig)):
        if orig[i] in (FULL_STOP, ARABIC_COMMA):
            if orig[i - 1] == SPACE:
                dele.append(i - 1)
            elif not orig[i - 1].isspace():
                ins[i] = SPACE
    edited = np.zeros(len(orig), dtype=bool)
    for i in dele:
        edited[i] = True
        _mark_neighbours(orig, edited, i)
    for i in ins:
        edited[i] = True
        edited[i - 1] = True
    text, idx = _build(orig, delete=dele, insert_before=ins)
    return _finish("punct_space", orig, text, idx, edited, len(dele) + len(ins))


def perturb_zwnj(orig: str) -> Perturbed:
    ins = {}
    for m in ARABIC_WORD_RE.finditer(orig):
        j = compound_split(m.group())
        if j is not None:
            ins[m.start() + j] = ZWNJ
    edited = np.zeros(len(orig), dtype=bool)
    if ins:
        edited[list(ins)] = True
    text, idx = _build(orig, insert_before=ins)
    return _finish("zwnj", orig, text, idx, edited, len(ins))


PERTURBATIONS: Dict[str, Callable[[str], Perturbed]] = {
    "harakat": perturb_harakat,
    "digits": perturb_digits,
    "punct_space": perturb_punct_space,
    "zwnj": perturb_zwnj,
}


def check_alignment(orig: str, p: Perturbed) -> None:
    """Invariant check: every non-inserted perturbed char equals its origin, except 1:1 digit swaps."""
    for k, i in enumerate(p.orig_idx.tolist()):
        c = p.text[k]
        if i < 0:
            assert c in (ZWNJ, SPACE), (p.name, k)
        else:
            assert c == orig[i] or DIGIT_SWAP.get(orig[i]) == c, (p.name, k, i)
    kept = p.orig_idx[p.orig_idx >= 0]
    assert np.all(np.diff(kept) > 0), p.name
