# -*- coding: utf-8 -*-
"""Three extra perturbations for realistic user input, in the format of eval/perturb.py (same Perturbed objects,
so eval/harness.seg_compare measures them exactly like the four PLAN 4.2 probes). Codepoints built with chr().

  arabic_keyboard  a user typing on an Arabic keyboard layout: every FARSI YEH U+06CC -> ARABIC YEH U+064A,
                   KEHEH U+06A9 -> ARABIC KAF U+0643, HEH GOAL U+06C1 -> ARABIC HEH U+0647 (1:1 replacements).
                   These are the three letters the task names. hp.normalize folds YEH and KAF back only inside
                   words it can prove are Urdu/Hindko orthography and never folds HEH (NORMALIZATION.md), so the
                   probe is reported raw and, separately, after normalize() (token counts only).
  nr_nnu           the retroflex nasal written without the Gandhara Hindko Board digraph: NOON (+ optional SUKUN)
                   + RREH  ->  NOON WITH SMALL TAH U+0768 (the Punjabi/Saraiki convention; attested in the
                   corpus 436 times against 63,958 for the digraph, NORMALIZATION.md section 10).
  nr_n             the same digraph written as a plain NOON (Urdu-style spelling, e.g. 'water' with n instead of
                   n+rr; attested as a separate spelling in the corpus).
  Both nr probes rewrite EVERY NOON(+SUKUN)+RREH sequence. A few are an ordinary n + rr sequence, not the
  retroflex nasal (NORMALIZATION.md); the probe does not try to tell them apart.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, r"F:\Hindko\_tokenizer\eval")
import perturb as P  # noqa: E402

YEH, KAF, HEH_GOAL = chr(0x06CC), chr(0x06A9), chr(0x06C1)
AR_YEH, AR_KAF, AR_HEH = chr(0x064A), chr(0x0643), chr(0x0647)
AK = {YEH: AR_YEH, KAF: AR_KAF, HEH_GOAL: AR_HEH}
NOON, RREH, SUKUN, NOON_TAH = chr(0x0646), chr(0x0691), chr(0x0652), chr(0x0768)
NR_RE = re.compile(NOON + SUKUN + "?" + RREH)


def perturb_arabic_keyboard(orig: str) -> P.Perturbed:
    rep = {i: AK[c] for i, c in enumerate(orig) if c in AK}
    edited = np.zeros(len(orig), dtype=bool)
    if rep:
        edited[list(rep)] = True
    text, idx = P._build(orig, replace=rep)
    return P.Perturbed("arabic_keyboard", text, np.asarray(idx, dtype=np.int32), edited, len(rep))


def _nr(orig: str, mode: str) -> P.Perturbed:
    rep, dele = {}, []
    edited = np.zeros(len(orig), dtype=bool)
    n = 0
    for m in NR_RE.finditer(orig):
        a, b = m.start(), m.end()
        edited[a:b] = True
        if mode == "nnu":
            rep[a] = NOON_TAH
        dele.extend(range(a + 1, b))
        n += b - a - (0 if mode == "nnu" else 1)
    text, idx = P._build(orig, delete=dele, replace=rep)
    return P.Perturbed("nr_" + mode, text, np.asarray(idx, dtype=np.int32), edited, n)


def perturb_nr_nnu(orig: str) -> P.Perturbed:
    return _nr(orig, "nnu")


def perturb_nr_n(orig: str) -> P.Perturbed:
    return _nr(orig, "n")


EXTRA = {"arabic_keyboard": perturb_arabic_keyboard, "nr_nnu": perturb_nr_nnu, "nr_n": perturb_nr_n}
ALL = {**P.PERTURBATIONS, **EXTRA}
ORDER = ["harakat", "digits", "punct_space", "zwnj", "arabic_keyboard", "nr_nnu", "nr_n"]


def check(orig: str, p: P.Perturbed) -> None:
    """Alignment invariant: kept characters map to strictly increasing origins; each equals its origin or is the
    documented replacement of it."""
    allowed = {**{a: {b} for a, b in AK.items()}, NOON: {NOON_TAH}}
    for k, i in enumerate(p.orig_idx.tolist()):
        c = p.text[k]
        if i < 0:
            assert c in (P.ZWNJ, P.SPACE), (p.name, k)
        else:
            o = orig[i]
            assert c == o or P.DIGIT_SWAP.get(o) == c or c in allowed.get(o, ()), (p.name, k, i)
    kept = p.orig_idx[p.orig_idx >= 0]
    assert np.all(np.diff(kept) > 0), p.name
