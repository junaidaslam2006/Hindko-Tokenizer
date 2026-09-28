"""Book segmentation, revision resolution and book metadata.

The 'Hindko Books' collection (Gandhara Hindko Board/Academy book series,
~95 numbered Farma folders + working folders) is typeset in InPage like the
newspaper, but its layout is different, so the newspaper segmenter
(segment.segment_articles) must never be used on it: it treats every frame of
<= 90 chars as a headline, which would split every poem (verse frames have a
median of 27.5 chars).

Entry point: process_books(book_files, newspaper_texts, log) -> dict with
records / imprints / line_dedup / per_file / book_meta / report.

Pipeline (every rule measured by the probe_books_* scripts in _pipeline/;
the numbers are quoted where each rule is implemented):

  1. classify frames per file (layout rules L1-L6, L9, L12, L13)
  2. resolve revisions and remove copies at LINE level, before any record is
     built (rules R1-R7: W1 W2 B X F + revision clusters); newspaper overlap
     is flagged per record, never removed line by line (R8 correction)
  3. build units and records from the surviving lines (L7, L8, L11, L12)
  4. attach evidenced book metadata (imprint > folder name; L10) and
     language / script / genre signals (hp/lang.py)

Principles: frame order is reading order and is never changed; text is only
removed when it is a copy of text that is kept (every removal has a
line_dedup audit row) or when it is structural (length bytes, running heads,
table of contents / imprint -> side records, separators...); every metadata
value carries a *_source; nothing is guessed. The output is deterministic:
fixed processing order, no randomness, no dependence on set iteration order.

Output of process_books():
  records     one dict per record, in (file, passage_index) order. Besides the
              BOOKS_SPEC schema: frame_indices (the source frames; line
              indices into text_all.split('\\n')), n_chars, split_part,
              genre_label / language_label (in-text headings), book_form,
              first_edition_year (+ _source), publication_date ('YYYY' or
              'YYYY-MM'), author_role (the imprint's author label: مصنف,
              شاعر, مترجم, مرتب ...), imprint_field_sources (field -> the
              file whose imprint states it; the edition fields always come
              from ONE imprint), imprint_source_paths, language_source,
              hindko_score_v2 /
              language_variety_v2 (whole-token, Arabic/Persian lines left
              out), latin_share, unmapped_glyphs (U+FFFD kept and counted).
              role 'toc' records carry toc_entries / toc_block_lines.
  imprints    per file with an imprint window: every label/value pair as
              written (with line indices), every frame of the window as
              written, the parsed fields and notes.
  line_dedup  every removed frame: source_path, line_index, reason (W1
              W1_GAP W2 L4 B X F TOC_COPY LABEL_OF_REMOVED_LINE), duplicate_of
              ('path#line', or the file path for F), duplicate_source
              (same_file / book), coverage, n_chars, via (reason of the
              intermediate copy when a same-file pointer was resolved, or of
              the removed line a speaker / language label belonged to),
              label_of_line (LABEL_OF_REMOVED_LINE only), copy_lines
              (W1_GAP only: the first-copy lines whose union covers it).
  per_file    per file: lines/chars by layout category and by final
              disposition (record / dedup / toc_copy / toc_title /
              no_extractable_text / one of STRUCTURAL), removals by reason,
              flags, cluster role, processing order, running headers,
              title-page frames, record counts.
  book_meta   per top-level folder: series number, folder title/author,
              imprint values (+ source file) and conflicts, genre + evidence,
              book form, Hindko score, Pothohari evidence, running headers.
  report      aggregate numbers for processing_report.json.
  clusters    the revision clusters (members, canonical, containment).
"""
from __future__ import annotations

import difflib
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from itertools import compress, repeat
from operator import is_
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import clean, inpage, lang, segment

BOOKS_ROOT = 'Hindko Books/HINDKO BOOKS DATA/'

# ==========================================================================
# paths and file roles
# ==========================================================================


def book_rel(rel: str) -> str:
    """Path relative to HINDKO BOOKS DATA/, '/'-separated."""
    r = rel.replace('\\', '/')
    return r[len(BOOKS_ROOT):] if r.startswith(BOOKS_ROOT) else r


def folder_of(rel: str) -> str:
    return book_rel(rel).split('/')[0]


def farma_no(rel: str) -> int:
    m = re.match(r'(\d+)', book_rel(rel))
    return int(m.group(1)) if m else 10 ** 6


# Processing-order exception (BOOKS_SPEC_CORRECTIONS, Revisions): the Hindko
# Lughat parts in the working folder 'Dictionary with matlan adition' are the
# expanded, corrected successor of the 98th GHA dictionary (they cover 99.1% of
# it). Processed after 98th, 1,146,932 of their chars were removed as X copies
# of 98th's older text (keeping its misspellings الواداع / اندورنی / ضابط and
# dropping the Lughat's الوداع / اندرونی / ضابطہ). They sort at 97.5: after the
# 25th Pothohari dictionary (key 25, which keeps the entries it shares, 47.5%
# of its distinct lines, under its 'pothohari' label) and before 98th.
LUGHAT_FOLDER = 'dictionary with matlan adition'
LUGHAT_ORDER = 97.5


def order_no(rel: str) -> float:
    """Farma number used by the R1 processing order (with the Lughat exception)."""
    if folder_of(rel).strip().lower() == LUGHAT_FOLDER and re.search(
            r'lughat', os.path.basename(rel), re.I):
        return LUGHAT_ORDER
    return float(farma_no(rel))


def is_superseded_dir(rel: str) -> bool:
    """Under an Old/ or Del/ sub-folder (R7: always flagged)."""
    parts = book_rel(rel).split('/')
    return any(p.strip().upper() in ('OLD', 'DEL') for p in parts[1:-1])


def is_copy_dir(rel: str) -> bool:
    return folder_of(rel).strip().upper().endswith('- COPY')


def is_backup(rel: str) -> bool:
    return rel.lower().endswith('.b01')


# L11: FEHRIST / PRELIMINARY / intro / HARF E AWAL / WRITEUP / flap files are
# front matter (probe_books_layout_fehrist.FRONT_NAMES + the ' FEH' suffix of
# 'FARMA NO 26 FEH.INP' from probe_books_dedup.role_of). Across the 35 such
# files the chars are prose 78.1%, verse 9.5%, contents 7.8%, imprint 2.3%;
# dropping them as 'just contents' would lose ~440k chars of prefaces.
FRONT_NAME_RE = re.compile(
    r'fehrist|fehrest|\bfeh\b|preliminary|\bintro|harf e awal|write ?up|flap', re.I)
# L12: reference works by path. Measured: the 4 dictionary folders hold 10.76M
# chars (40% of all book chars) of headwords + mostly Urdu glosses; 131st is a
# parallel اُردو / ھندکو phrasebook; the Arabic/Persian line rules fail on
# their respellings (24 of 41 and 12 of 12 wrong). The words include the
# folder 'Dictionary with matlan adition'. 'Mattlan/Matlan' is NOT a lexicon
# word (BOOKS_SPEC_CORRECTIONS L12): '000 BOOK MATTLAN.INP' (both copies) is
# the 99th PROVERBS book - proverb + glosses + Urdu translation + Hindko
# explanation, running text, genre 'proverbs' (lang.PATH_GENRE). The role is
# decided PER FILE, so the copy inside the Dictionary folder is not a lexicon
# either.
LEXICON_PATH_RE = re.compile(r'dictionary|lughat|mutaradif|bol ?chal', re.I)
PROVERBS_FILE_RE = re.compile(r'matt?lan', re.I)


def file_role(rel: str) -> str:
    """'front_matter' | 'lexicon' | 'main' from the path alone (front matter
    wins: 25th/98th FEHRIST and flap files are prefaces, not word lists)."""
    br = book_rel(rel)
    base = os.path.basename(br)
    if FRONT_NAME_RE.search(base):
        return 'front_matter'
    if PROVERBS_FILE_RE.search(base):
        return 'main'
    if LEXICON_PATH_RE.search(br):
        return 'lexicon'
    return 'main'


_FOLDER_RE = re.compile(r'^(\d+)\s*(?:st|nd|rd|th)\b\s*(?:Farma\b)?\s*(.*)$', re.I)
_FOLDER_DATE_RE = re.compile(r'\s+\d{1,2}-\d{1,2}-\d{2,4}\s*$')


def parse_folder_name(folder: str) -> dict:
    """Series number, and the romanised title/author written in a numbered
    Farma folder name ('17th Farma Ishq Samandro Doonga by Zafar Naveed Jani').
    Working folders without a number yield nothing. Trailing dd-mm-yy dates
    are upload/work dates, not publication dates, and are not used."""
    m = _FOLDER_RE.match(folder.strip())
    if not m:
        return {'series': None, 'title': None, 'author': None}
    rest = _FOLDER_DATE_RE.sub('', m.group(2)).strip(' -')
    parts = re.split(r'\s+by\s+', rest, maxsplit=1, flags=re.I)
    title = parts[0].strip(' -') or None
    author = parts[1].strip(' -') if len(parts) > 1 and parts[1].strip(' -') else None
    return {'series': int(m.group(1)), 'title': title, 'author': author}


# ==========================================================================
# line keys (probe_books_dedup.py)
# ==========================================================================
_WS = re.compile(r'\s+')
_DIAC = re.compile('[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')
_NONLETTER = re.compile(r'[\W\d_]+', re.U)
_FOLD = str.maketrans({'ي': 'ی', 'ى': 'ی', 'ك': 'ک',
                       'ه': 'ہ', 'ە': 'ہ', 'ة': 'ۃ'})
_FOLD_PAIRS = tuple((chr(a), b) for a, b in sorted(_FOLD.items()))
NGRAM = 10


def _uniq(x: np.ndarray) -> np.ndarray:
    """Sorted unique values (sort-based; np.unique's hash path is slower here)."""
    if len(x) < 2:
        return np.array(x, copy=True)
    s = np.sort(x)
    keep = np.empty(len(s), dtype=bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


def loose_key(line: str) -> str:
    """Letters-only line key: NFC, diacritics/tatweel/punctuation/digits/
    whitespace removed, Arabic yeh/kaf/heh folded to the Urdu forms. Used only
    where letters alone matter (TOC / label statistics); copy removal uses
    dedup_key / dedup_hkey below."""
    s = clean.normalize_unicode(line)
    s = _DIAC.sub('', s)
    for a, b in _FOLD_PAIRS:          # = .translate(_FOLD); str.replace is faster on UCS-2
        if a in s:
            s = s.replace(a, b)
    return _NONLETTER.sub('', s)


# Copy identity (R1 X/B/W1/W2 and the F n-grams). Revisions re-space and
# re-punctuate lines, so spacing, punctuation, tatweel and the Arabic
# yeh/kaf/heh spellings are ignored - but NOT digits, and in lexicons NOT
# harakat (BOOKS_SPEC_CORRECTIONS, X). The letters-only loose key made 1,178
# removed lines (23,328 chars) 'copies' of lines with a different digit
# sequence (106th 'جلد نمبر2، دسمبر 1991، شمارہ نمبر 25' as a copy of
# 'جلد نمبر1، دسمبر 1989، شمارہ نمبر 1'; 'حرفی نمبر:- 239' of 'حرفی نمبر114';
# a name + phone number of the bare name) and 780 lexicon lines copies of
# other pronunciations (131st 'اَو' of 'اُو', 'کَدَّوْ' of 'کدو'; ~290 harakat
# variants such as تُلوا / تَلوا). Urdu/Arabic-Indic digits are folded to ASCII
# so '۱۹۹۱' and '1991' stay equal.
_DIGIT_FOLD = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
_HARAKAT_CLASS = 'ً-ٰٟ'
_DIAC_NOHARAKAT = re.compile('[ؐ-ؚۖ-ۭـ]')
_NONKEY = re.compile(r'[\W_]+', re.U)                       # keeps letters and digits
_NONKEY_H = re.compile('(?:[^\\w%s]|_)+' % _HARAKAT_CLASS, re.U)   # ... and harakat
_HAS_DIGIT = re.compile('[0-9۰-۹٠-٩]')


def dedup_key(line: str) -> str:
    """Copy identity of a line in running text: letters + digits (ASCII),
    harakat / tatweel / punctuation / whitespace removed, yeh/kaf/heh folded."""
    s = clean.normalize_unicode(line)
    s = _DIAC.sub('', s)
    for a, b in _FOLD_PAIRS:
        if a in s:
            s = s.replace(a, b)
    if _HAS_DIGIT.search(s):
        s = s.translate(_DIGIT_FOLD)
    return _NONKEY.sub('', s)


def dedup_hkey(line: str) -> str:
    """Copy identity in lexicon files: as dedup_key but harakat are kept (a
    dictionary's harakat are its pronunciations)."""
    s = clean.normalize_unicode(line)
    s = _DIAC_NOHARAKAT.sub('', s)
    for a, b in _FOLD_PAIRS:
        if a in s:
            s = s.replace(a, b)
    if _HAS_DIGIT.search(s):
        s = s.translate(_DIGIT_FOLD)
    return _NONKEY_H.sub('', s)


def ngram_hashes(s: str, n: int = NGRAM) -> np.ndarray:
    """Vectorised polynomial hash of all char n-grams of s (uint64)."""
    if not s:
        return np.zeros(0, dtype=np.uint64)
    a = np.frombuffer(s.encode('utf-32-le'), dtype='<u4').astype(np.uint64)
    if len(a) < n:
        n = len(a)
    m = len(a) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    p = np.uint64(1000003)
    with np.errstate(over='ignore'):
        for j in range(n):
            h = h * p + a[j:j + m]
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h


def ngram_hashes_many(keys: List[str], n: int = NGRAM) -> List[np.ndarray]:
    """ngram_hashes for many strings with one vectorised pass over their
    concatenation; n-grams crossing a boundary are simply not sliced out, so
    each result is identical to ngram_hashes(key)."""
    if not keys:
        return []
    a = np.frombuffer(''.join(keys).encode('utf-32-le'), dtype='<u4').astype(np.uint64)
    m = len(a) - n + 1
    h = np.zeros(max(m, 0), dtype=np.uint64)
    if m > 0:
        p = np.uint64(1000003)
        with np.errstate(over='ignore'):
            for j in range(n):
                h = h * p + a[j:j + m]
            h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
    out, pos = [], 0
    for k in keys:
        ln = len(k)
        out.append(h[pos:pos + ln - n + 1] if ln >= n else ngram_hashes(k, n))
        pos += ln
    return out


# ==========================================================================
# frame primitives (probe_books_layout_common.py)
# ==========================================================================
AR = segment.ARABIC
# Arabic-script letters, incl. the Hindko tone letters U+08BE..U+08C2 that the
# decoder composes since the 2026-09-25 adjudication (13,169 in the books,
# mostly dictionary respellings such as a lone tone letter + ڑ): without them
# a two-letter respelling frame would count 1 letter and become residue.
AR_LETTER = re.compile('[\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FF\u08BE-\u08C2]')
_DIGITS = '0-9\u06F0-\u06F9\u0660-\u0669'
RE_NUMERIC = re.compile(r'^[\s(\[]*[%s]{1,4}\s*[)\].۔\-–:]?\s*$' % _DIGITS)
RE_RANGE = re.compile(r'^\s*[%s]{1,4}\s*[-–]\s*[%s]{1,4}\s*$' % (_DIGITS, _DIGITS))
RE_ABJAD_FOLIO = re.compile(r'^\(?[\u0627-\u06D2]\)?$')     # (الف) (ب) folios


def is_numeric(t: str) -> bool:
    return bool(RE_NUMERIC.match(t) or RE_RANGE.match(t))


def is_residue_frame(t: str) -> bool:
    """No Arabic-script char and not a number: InPage table/style residue."""
    return not AR.search(t) and not is_numeric(t)


def single_char_category(s: str) -> Optional[str]:
    """L2 (BOOKS_SPEC_CORRECTIONS): a frame that is one character after
    stripping gets the NORMAL residue rule - no string-level length-byte
    stripping. The paragraph byte-length field (0x0D + u32) is removed at byte
    level by the decoder; on the current decodes only 10 of 255 lone ASCII
    digits between two text frames fit ord(char) ~ 2 x len(next frame): they
    are table serials, TOC serials and list numbers (53rd, 130th, 216th...).
    Most of the ~11.2M single-char frames ('@' 2.1M, '~' 1.0M, 'X', ',' and
    1.1M ASCII digits) sit in InPage's style/object residue, outside the text
    flow. So: a lone digit is a number (-> 'numbering' by L6, or a page number
    inside a TOC); anything else is 'residue' HERE - classify_file then keeps
    the lone ':' / '=' (joiners), '…' / '٭' (separators) and Arabic letters
    that sit directly between two text frames (review 2026-09-25: 3,600 ':'
    and 2,786 '=' frames in the text flow were dropped). U+FFFD is counted.
    Returns None for a number (decided later)."""
    return None if is_numeric(s) else 'residue'


_AR4 = re.compile('(?:' + AR.pattern + '.*?){%d}' % segment.MIN_ARABIC_CHARS, re.S)


def is_content_line_fast(t: str) -> bool:
    """Same result as segment.is_content_line (the R9 form/genre statistics
    were measured on those lines), without building the whitespace-free
    copy: str.split() splits on exactly the characters re's \\s matches."""
    if len(t.strip()) < segment.MIN_CONTENT_LEN or not _AR4.search(t):
        return False
    if len(t) > segment.MAX_ASCII_ONLY_LEN:
        compact = sum(map(len, t.split()))
        if compact > segment.MAX_ASCII_ONLY_LEN and len(AR.findall(t)) / compact < 0.30:
            return False
    return True


_STRIP = re.compile('[\u064B-\u065F\u0670\u0614\u0610-\u061A\u06D6-\u06ED\u0640'
                    '\u200C\u200D\u200E\u200F\\s،,۔.:;؛!؟?\'"()\\[\\]{}‘’“”\\-–—*]')


def norm(s: str) -> str:
    """Normalisation for TOC-title and imprint-label matching."""
    s = unicodedata.normalize('NFC', s)
    s = s.replace('ي', 'ی').replace('ك', 'ک').replace('ە', 'ہ')
    return _STRIP.sub('', s)


# ==========================================================================
# layout vocabulary (probe_books_layout_records.py)
# ==========================================================================
# L6(a) separators: ۔۔۔۔۔ x1,594, ۔۔۔ x1,373, ۔۔۔۔ x460, longer dot runs
# ~460; hard poem boundaries (never text). U+2026 '…' is in the class: the
# decoder adjudication maps \x04\xe0 to '…' (one per glyph; 95.5% of E0 runs
# of length >= 3 are whole separator lines, ~23,756 separator glyphs), so a
# frame made only of dots / ۔ / … is a separator. A single '…' inside a text
# frame is punctuation (dialogue pauses) and never a boundary: the class must
# fill the whole frame.
RE_SEP = re.compile(r'^[\s۔.\-–—_*٭★☆•●○◊~=+…]{2,}$')
# L6(d) numbering frames ('۱۔', '(۱۴)'): soft boundary. Rubai numbers
# (۱)-(۱۶۳) in 118th delimit units.
RE_NUMBERING = re.compile(r'^[\s(\[]*[0-9\u06F0-\u06F9]{1,4}\s*[)\].\u06D4\-:\u060E]*\s*$|^[\s(]*[ivxIVX]{1,5}[)\].\u06D4]\s*$')
# L6(b) genre headings: غزل x256 in 11 books, plus نعت, قطعہ, چار بیتہ,
# حمدیہ / نعتیہ / عشقیہ... Closed list (extend as books arrive).
GENRE_HEADINGS = frozenset({
    'غزل', 'نظم', 'نعت', 'حمد', 'قطعہ', 'قطعات', 'رباعی', 'رباعیات', 'ماہیا', 'ماہیے',
    'ماہیئے', 'گیت', 'چار بیتہ', 'چاربیتہ', 'حرفی', 'سی حرفی', 'دوہڑا', 'دوہڑے', 'بولیاں',
    'ٹپہ', 'ٹپے', 'منقبت', 'سلام', 'مرثیہ', 'نوحہ', 'قصیدہ', 'مثنوی', 'کافی', 'حمدیہ',
    'نعتیہ', 'عشقیہ', 'آزاد نظم', 'ہائیکو', 'غزلاں', 'نظماں', 'فردیات', 'متفرق اشعار',
    'لوک گیت', 'ہندکو لوک گیت', 'نمکین غزل'})
# L6(c) language labels in bilingual books: ہندکو / اُردو x84 each in 207th,
# اُردو ترجمہ x217 in 196th, 145/147 in 131st.
LANG_LABELS = frozenset({'ہندکو', 'ھندکو', 'ا\u064Fردو', 'اردو', 'فارسی', 'پشتو',
                         'انگریزی', 'پنجابی', 'ا\u064Fردو ترجمہ'})
END_PUNCT = ('۔', '؟', '!', '?')
MIN_CHARS = 200          # L7: merge units below this (49.2% of units are)
MAX_CHARS = 6000         # L7: split units above this
SPEAKER_MAX = 24         # L6(e) speaker label: <=24 chars ending ':' seen >=3x
SPEAKER_MIN_COUNT = 3
HEADER_RUN_MIN = 3       # L3
HEADER_RUN_INSIDE_MIN = 5
HEADER_MAX_LEN = 60
# L3: the main span (first..last frame of >= 25 chars) only decides whether a
# header run is 'inside' (needs 5+) or at an end (3+), and which frames get
# the informational 'outside_main_span' record flag. It is NOT a cut: as a
# cut it dropped real text at the ends of books (BOOKS_SPEC_CORRECTIONS L3:
# 112th's last ~46 mahiye, 167th's final ghazal, 35th's final poem, 165th's
# glossary tail, 47th's endnotes, 50th's bibliography, 109th's cast list -
# 1,192 frames / 17,283 chars of >= 8 letters after the span and 353 / 4,609
# before it; 236 + 59 of them existed in no record at all).
MAIN_SPAN_MIN_LEN = 25


def is_verse(t: str) -> bool:
    """Verse-shaped frame (one misra per frame; median 27.5 chars)."""
    return 8 <= len(t) <= 75 and not t.rstrip().endswith(END_PUNCT) and len(t.split()) <= 14


def label_key(t: str) -> str:
    return t.strip().rstrip(':۔ ').strip()


# ==========================================================================
# tables of contents (probe_books_layout_order.parse_tocs)
# ==========================================================================
# Contents are not 'title .... 23' lines: they are split into serial / title /
# page frames. Dot leaders are NOT a TOC signal (462 of 699 are syllable
# separators in the Pothohari dictionary). 80 blocks / 4,143 entries in 75
# files; 98.77% of uniquely matched entries lie on the longest in-order
# sequence of the body, and page numbers never go backwards in 3,700 of 3,771
# steps - which is also the evidence that frame order is reading order (L1).
TOC_MARKERS = frozenset({'فہرست', 'ترتیب', 'حسن\u0650 ترتیب', 'حسن ترتیب', 'فہرست مضامین',
                         'فہرست کلام', 'عنوانات', 'فہرست\u0650 مضامین', 'مندرجات'})
TOC_HEADER_CELLS = frozenset({'عنوان', 'عنوانات', 'صفحہ', 'صفحہ نمبر', 'نمبر شمار', 'نمبرشمار',
                              'صفحات', 'حرف', 'نام', 'شاعر', 'مصنف', 'نمبر'})


# TOC keys (review 2026-09-25): norm() keeps digits and the era sign, so the
# 53rd body heading 'راشد جاوید 1956' never matched its entry 'راشد جاوید 1956
# ء' and the previous title ('اجمل ملک1948 ء') ran on over 11 records of
# ~20 other journalists. The key drops ء / ؁ / U+0601 and folds digits.
_TOC_DROP = re.compile('[%s]' % ''.join(map(chr, (0x0621, 0x0601, 0x0600))))
_TOC_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')


def toc_key(s: str) -> str:
    """Key for matching TOC entries to body frames (norm + no ء / year sign,
    ASCII digits, space-insensitive)."""
    return _TOC_DROP.sub('', norm(s)).translate(_TOC_DIGITS)


# Heading / byline TOCs (review 2026-09-25): anthology and conference TOCs set
# each row as heading cell / byline cell / page, so the walk read the heading
# as a page-less entry and the byline as the paged entry - and the byline
# (unique in the body) became the unit title: 187 records / 730,924 chars
# titled with a contributor's name (33rd 128: 85 of 86 paged entries follow a
# page-less one; 46th 0.86, 45th 0.61, 19th 0.58, 31st 0.42, 206th 0.35). In a
# block where >= 30% of the paged entries follow a page-less entry, each such
# pair is one entry: the heading with the page, the byline kept as its
# 'byline' (never a title candidate). In other blocks the second cells of
# such pairs are 'byline candidates': build_units keeps a candidate that
# directly follows a title-only heading as a line of that unit.
TOC_BYLINE_SHARE = 0.3


def _pair_bylines(entries):
    paged = sum(1 for e in entries if e[2] is not None)
    alt = [k for k in range(len(entries) - 1) if entries[k][2] is None and entries[k + 1][2] is not None]
    if not alt:
        return entries, {}, []
    if not paged or len(alt) / paged < TOC_BYLINE_SHARE:
        return entries, {}, [entries[k + 1][1] for k in alt]
    alt_set = set(alt)
    out, by = [], {}
    k = 0
    while k < len(entries):
        e = entries[k]
        if k in alt_set:
            b = entries[k + 1]
            out.append((e[0], e[1], b[2]))
            by[e[0]] = (b[0], b[1])
            k += 2
            continue
        out.append(e)
        k += 1
    return out, by, []


def parse_tocs(lines: List[str], candidates: Optional[List[int]] = None) -> List[dict]:
    """TOC blocks: {'marker_idx', 'end_idx', 'entries': [(line_idx, title,
    page or None)], 'numeric', 'frames'}. Single characters are kept here
    (pages 1-9). 'frames' are exactly the frames the block consumed (marker,
    header cells, numbers, entries): only those become 'toc'.

    Same walk as probe_books_layout_order.parse_tocs over the non-residue
    frames (keep_single=True), started only at marker frames, so the frames
    between markers are never visited. The block ENDS AT ITS LAST PAGE
    NUMBER: the probe walk also took the up-to-3 unnumbered frames after it
    (the walk stops on the 4th) - two became page-less 'entries' and the
    third was marked 'toc' without being an entry. Those frames are the body:
    173rd's first misra 'میں ڈٹھی محبوب سیّاں ساری' became an entry and its
    second misra was lost; dedications ('انتساب' + 'جنہاں دیاں یاداں' in 120th,
    130th, 141st, 149th, 150th) moved into toc records; 166th's book heading
    'غزواتِ نبویﷺ' became the last entry and then matched the first body frame
    (s10_toc / toc_tail: 59 such block ends, 1,133 chars, 22 frames in no
    record). They are now classified like any other frame."""
    n = len(lines)
    tocs = []
    scan_from = 0
    idx = range(n) if candidates is None else candidates      # markers are multi-char frames
    for m in [i for i in idx if lines[i].strip() in TOC_MARKERS]:
        if m < scan_from:
            continue
        entries, numeric, since_num, pending = [], 0, 0, None
        frames = [m]
        last_num = None            # line of the last page number consumed
        i = m + 1
        while i < n:
            t = lines[i]
            if is_residue_frame(t):
                i += 1
                continue
            t = t.strip()
            if len(t) >= 120:
                break
            if t in TOC_HEADER_CELLS:
                frames.append(i)
                i += 1
                continue
            if is_numeric(t) or RE_ABJAD_FOLIO.match(t):
                numeric += 1
                since_num = 0
                if pending is not None:
                    entries.append(pending + (t,))
                    pending = None
                frames.append(i)
                last_num = i
                i += 1
                continue
            since_num += 1
            if since_num > 3:        # numbering stopped -> TOC over
                break
            if pending is not None:
                entries.append(pending + (None,))
            pending = (i, t)
            frames.append(i)
            i += 1
        if last_num is None:
            entries, frames = [], [m]
        else:
            entries = [e for e in entries if e[0] < last_num]
            frames = [f for f in frames if f <= last_num]
        # heading / byline rows are paired before the size test: a row is one
        # entry with one page number
        entries, bylines, cands = _pair_bylines(entries)
        if len(entries) >= 3 and numeric >= len(entries):
            tocs.append({'marker_idx': m, 'end_idx': last_num, 'entries': entries,
                         'numeric': numeric, 'frames': frames, 'bylines': bylines,
                         'byline_candidates': cands})
            scan_from = last_num + 1
        else:
            scan_from = m + 1
    return tocs


# ==========================================================================
# imprint (probe_books_layout_colophon.py)
# ==========================================================================
# The imprint is typeset as label / value frame pairs anchored on جملہ حقوق.
# Evidence per folder (of 95): year 87, publisher 87, ISBN 87, price 87,
# title 85, rights line 85, GHA reference 54, first+reprint editions both
# labelled 11. Labels are read ONLY inside the window: 242nd (reviews) holds
# 100 year and 100 price labels of other books outside its own imprint.
IMPRINT_LABELS = {
    'title': ['نام کتاب', 'کتاب دا ناں', 'کتاب کا نام', 'نام'],
    # credit labels (review 2026-09-25): every label naming a person's part
    # in the book. The bare 'ترتیب' is NOT one - it is the contents heading
    # and yielded 'نمبر شمار' in 7 windows (120th, 139th, 141st, 149th, 150th,
    # 67th); compound labels ('مترجم/ شاعر', 'تحقیق وتالیف', 'محقق، مرتبہ')
    # are recognised part by part (credit_roles). What each credit IS -
    # author / translator / compiler - is decided by credit_roles.
    'author': ['مصنف', 'مصنفہ', 'شاعر', 'شاعرہ', 'تحقیق و تصنیف', 'تحقیق وتصنیف', 'مؤلف', 'مولف',
               'مرتب', 'مرتبہ', 'تصنیف', 'ترتیب و تدوین', 'تحقیق و ترتیب', 'مترجم', 'کلام'],
    'subject': ['موضوع'],
    'language': ['زبان'],
    'year': ['سال اشاعت', 'سال\u0650 اشاعت', 'سن\u0650 اشاعت', 'سن اشاعت', 'اشاعت\u0650 اول', 'اشاعت اول',
             'اشاعت\u0650 ثانی', 'اشاعت ثانی', 'طبع اول', 'طبع\u0650 اول', 'اشاعت', 'پہلی واری', 'تاریخ اشاعت'],
    'publisher': ['ناشر', 'پبلشر', 'پبلشرز', 'مطبع', 'ادارہ', 'چھاپالنے والے'],
    'supervised_by': ['اہتمام\u0650 اشاعت', 'اہتمام اشاعت', 'زیر\u0650 اہتمام', 'زیر اہتمام'],
    'printer': ['پرنٹر', 'پرنٹنگ', 'پریس', 'طابع', 'چھاپہ خانہ'],
    'edition': ['وار', 'چ\u064Eھپن ط وار', 'ایڈیشن'],
    'price': ['قیمت', 'ہدیہ'],
    'isbn': ['ISBN No.', 'ISBN', 'ISBN No'],
    # 'جی ایچ اے اشاعت (ابتدائی)' (120th #4331 -> 'F.120/2017', 329th #6684 ->
    # 'F.132/2017', 139th; BOOKS_SPEC_CORRECTIONS L10): the parenthesised
    # qualifier is part of the label; norm() drops only the brackets
    'gha_ref': ['جی ایچ اے اشاعت حوالہ', 'جی ایچ اے اشاعت', 'اشاعت حوالہ',
                'جی ایچ اے اشاعت (ابتدائی)', 'جی ایچ اے اشاعت حوالہ (ابتدائی)'],
    'composer': ['کمپوزنگ', 'کمپوزکاری', 'کمپوزنگ/سرورق', 'پروفنگ', 'سرورق', 'ٹائٹل'],
}
_LABEL_OF = {}
for _f, _labs in IMPRINT_LABELS.items():
    for _lab in _labs:
        _LABEL_OF[norm(_lab)] = _f
# Credit words, by role. A credit label is one of these or a compound of
# them joined by '/', '،', ',' or 'و' (spaced or glued: 'تحقیق ،تحریروانڈیکس'
# 130th, 'تصنیف وتالیف' 131st, 'محقق ومؤلف' 165th, 'مؤلف/مترجم' 27th,
# 'مترجم/ شاعر' 209th, 'محقق، مرتبہ' 25th, 'تحقیق و تحریر' 57th / 59th-63rd,
# 'تحقیق وتالیف' MATTLAN, 'تلاش، ترتیب وتالیف' 19th). 'ترتیب' and 'تحقیق'
# alone are not credits ('ترتیب' is the contents heading, 'تحقیق' a subject
# value); inside a compound they are. Review 2026-09-25: translators and
# compilers were all 'author' and compound labels were not read at all.
CREDIT_WORDS = {
    'author': ('مصنف', 'مصنفہ', 'شاعر', 'شاعرہ', 'تصنیف', 'کلام', 'شاعری', 'تحریر', 'تخلیق'),
    'translator': ('مترجم', 'ترجمہ', 'منظوم ترجمہ'),
    'compiler': ('مرتب', 'مرتبہ', 'مرتبین', 'مؤلف', 'مولف', 'تالیف', 'ترتیب', 'تدوین', 'مدون',
                 'تحقیق', 'محقق', 'تلاش', 'انڈیکس'),
}
_CREDIT_ROLE = {norm(w): r for r, ws in CREDIT_WORDS.items() for w in ws}
_CREDIT_NOT_ALONE = frozenset(norm(w) for w in ('ترتیب', 'تحقیق', 'تلاش', 'انڈیکس', 'ترجمہ', 'شاعری',
                                                'تحریر', 'تخلیق', 'تالیف', 'تدوین'))
# lone words that are credit labels only where a frame is read as a label
# (85th 'شاعری' / 'احمد علی سائیںؒ', 3rd 'ترتیب' / 'محمد ضیاء الدین'); a
# contents cell is never their value ('ترتیب' / 'نمبر شمار' is the TOC
# heading: _is_contents_cell)
# 'ترتیب' is NOT a weak label either (confirmation 2026-09-25): it is the
# contents heading, and as a weak label it (a) made the TOC heading's next
# cell ('حصہ اول' ...) the compiler of 20th Farma and (b) counted as an imprint
# label in the end-word lookahead, so dedications ('انتساب') and TOC header
# cells were swallowed into imprint windows in 23 books. The one real use
# (3rd Farma 'ترتیب' / 'محمد ضیاء الدین') is lost: null is preferred to a
# wrong credit.
_CREDIT_WEAK_LABEL = frozenset(norm(w) for w in ('شاعری', 'تحقیق', 'تحریر', 'تالیف', 'تدوین', 'ترجمہ'))
# A dedication heading always ends an imprint window, whatever follows it.
_DEDICATION_N = norm('انتساب')
_CREDIT_SPLIT = re.compile(r'\s*[/،,؛]\s*|\s+و\s*|\s+و(?=\S)')


def _credit_part(p: str) -> Optional[List[str]]:
    """Roles of one part of a credit label ('تحریروانڈیکس' = تحریر + انڈیکس)."""
    n = norm(p)
    if not n:
        return []
    if n in _CREDIT_ROLE:
        return [_CREDIT_ROLE[n]]
    if n.startswith('و') and n[1:] in _CREDIT_ROLE:
        return [_CREDIT_ROLE[n[1:]]]
    for k, ch in enumerate(n):
        if ch == 'و' and 0 < k < len(n) - 1 and n[:k] in _CREDIT_ROLE:
            rest = _credit_part(n[k + 1:])
            if rest:
                return [_CREDIT_ROLE[n[:k]]] + rest
    return None


def credit_roles(label: str, weak: bool = False) -> Optional[List[str]]:
    """The roles a credit label names, in order, or None when it is not a
    credit label (every part must be a credit word; a lone 'ترتیب' / 'تحقیق'
    / 'شاعری' ... is not a label: they are headings and subject values)."""
    t = label.strip().rstrip(':۔-').strip()
    if len(t) > 40:
        return None
    parts = [x for x in _CREDIT_SPLIT.split(t) if x and x.strip()]
    if not parts:
        return None
    roles = []
    for p in parts:
        r = _credit_part(p)
        if r is None:
            return None
        roles += r
    if not roles or (len(roles) == 1 and norm(t) in _CREDIT_NOT_ALONE and not (
            weak and norm(t) in _CREDIT_WEAK_LABEL)):
        return None
    return roles


# Rights lines (the imprint anchor): 'جملہ حقوق ...', its typo 'جملہ حقوب'
# (118th #5457 'جملہ حقوب بحق گندھارا ...') and the Hindko form 'سارے حق ...
# محفوظ' (340th 'سارے حَق مصنف دے حَق بِچ محفوظ این', 358th #461 'سارے حق شاعر
# دے حق بِچ محفوظ ہین'). Without them those three windows were anchored 12
# frames before the first year label and cut off the title and author pairs
# (review 2026-09-25). The non-standard forms must be followed by >= 2 labels.
RIGHTS = re.compile('جملہ\s*حقو[قب]|سارے\s*حَ?ق\s.{0,40}محفوظ')
RIGHTS_STRICT = re.compile('جملہ\s*حقوق')
RIGHTS_MAX_LEN = 80
IMPRINT_WINDOW = 90
# End words, compared after norm() (an exact comparison missed 'اِنتساب' with
# zer in 206th / 307th / 346th). Review 2026-09-25: the window also swallowed
# the dedication 'ایصال ثواب' ... (8th FEHRIST #8877-8884), the section
# headings 'سرنانواں' / 'سرِنانواں' (343rd #9106, 340th #499), 'پیش لفظ' (3rd
# #8524), 'ہک ضروری گہل' (250th #7157, 255th #7158, 4th #595) and 165th's
# opening bismillah (#10332). An end word ends the window only when no
# imprint label follows within 6 frames: 'پیش لفظ' had been dropped from the
# set because a credit pair ('پیش لفظ' / name) ended windows early (L9 risk).
IMPRINT_END = frozenset({'انتساب', 'فہرست', 'حسنِ ترتیب', 'فہرست مضامین', 'سرنانواں',
                         'ایصال ثواب', 'پیش لفظ', 'ہک ضروری گہل', 'عنوانات'})
IMPRINT_END_N = frozenset(norm(x) for x in IMPRINT_END)
IMPRINT_END_LOOKAHEAD = 6
# After the last label / value pair the window runs on only over the address
# block ('ملنے دا پتہ' + addresses, phone numbers): of 1,029 non-pair frames
# in the windows most were credits and addresses; the 15 body-text frames
# (dedications, section headings, a bismillah) all FOLLOWED that block.
ADDRESS_WORDS = ('پتہ', 'پتا', 'روڈ', 'ٹاؤن', 'بازار', 'مارکیٹ', 'ڈپو', 'سنز', 'بکس', 'بک',
                 'پشاور', 'اسلام آباد', 'راولپنڈی', 'اکیڈمی', 'بورڈ', 'موبائل', 'فون', 'پاکستان',
                 'پلازہ', 'ایبٹ آباد', 'لاہور', 'کراچی', 'مانسہرہ', 'ہری پور', 'سیکریٹری', 'گلی',
                 'محلہ', 'بلاک', 'کالونی', 'ای میل', 'ویب', 'تعداد')
RE_PHONE_LIKE = re.compile('^[\s\d۰-۹٠-٩\-–+,،/().]+$')
RE_YEAR = re.compile(r'(1[89]\d\d|20[0-3]\d)')
RE_ISBN = re.compile(r'97[89][\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?[\dX]|97[89][\d\-\s]{7,16}')
RE_GHA_LIKE = re.compile(r'F\s*[.\-]\s*\d')
# 'سال اشاعت (دوم)' (173rd: 'اپریل1995ئ' then '(دوم) 2017ئ') - the edition
# word may be written in parentheses; the probe regex missed it, which is why
# the probe reported only 1995 for 173rd. The parenthesised form is the text's
# own label, so accepting it is not an inference.
# 'اشاعت سوئم' (the ئ spelling of the third edition, next to the accepted
# 'دوئم'; 255th #7138 'اشاعت سوئم' / '2019') was not a label, so 255th got its
# SECOND edition (2018) as publication year and edition (review 2026-09-25);
# 'چہارم' / 'چوتھا' (fourth) are added with it.
RE_YEAR_LABEL = re.compile(
    r'^(?:سالِ?|سنِ?|تاریخ)?\s*(?:اشاعتِ?|طبعِ?)\s*\(?\s*'
    r'(اول|اوّل|دوم|دوئم|ثانی|سوم|سوئم|ثالث|چہارم|چوتھا|چوتھی)?\s*\)?$')
_TO_ASCII = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
# 'مء' / 'جولاء': since the 2026-09-25 adjudication the decoder writes ء at
# word end (\x04\xa3), so years end in 'ء' ('2017ء') and these months lose
# their ئ; the old spellings 'مئ' / 'جولائ' / '2017ئ' are still accepted
URDU_MONTHS = {'جنوری': 1, 'فروری': 2, 'مارچ': 3, 'اپریل': 4, 'مئی': 5, 'مئ': 5, 'مء': 5,
               'جون': 6, 'جولائی': 7, 'جولائ': 7, 'جولاء': 7, 'اگست': 8, 'ستمبر': 9,
               'اکتوبر': 10, 'نومبر': 11, 'دسمبر': 12}
# '(دوسرا ایڈیشن)' written under the title (40th FEHRIST): an edition
# statement in the window without a label; read as an undated edition entry
RE_EDITION_STATEMENT = re.compile(r'^\(?\s*(\S+)\s+ایڈیشن\s*\)?$')


def imprint_label_field(t: str, weak: bool = False) -> Optional[str]:
    """Imprint field a label frame names, else None. weak=True also accepts
    a lone credit word that is otherwise a heading or a subject value
    ('شاعری', 'تحقیق' ...): used where a frame is read as a LABEL, never
    where it is tested as a value."""
    t2 = t.strip().rstrip(':۔-').strip()
    if len(t2) <= 24 and 'اہتمام' not in t2 and 'حوالہ' not in t2 and RE_YEAR_LABEL.match(t2):
        return 'year'
    n2 = norm(t2)
    if len(t2) <= 40 and 'چھپن' in n2 and ('تاریخ' in n2 or 'سنہ' in n2):
        return 'year'
    f = _LABEL_OF.get(n2)
    if f is None and credit_roles(t2, weak) is not None:
        f = 'author'            # a credit label (author / translator / compiler)
    return f


def edition_of(t: str) -> Optional[int]:
    m = RE_YEAR_LABEL.match(t.strip().rstrip(':۔-').strip())
    q = m.group(1) if m else None
    if q in ('اول', 'اوّل'):
        return 1
    if q in ('دوم', 'دوئم', 'ثانی'):
        return 2
    if q in ('سوم', 'سوئم', 'ثالث'):
        return 3
    if q in ('چہارم', 'چوتھا', 'چوتھی'):
        return 4
    return None


# Ordinals written as the value of an edition label ('وار:' -> 'پہلی', 'دوئی').
EDITION_WORDS = {
    1: ('پہلی', 'پہلا', 'پہلے', 'اول', 'پہلو'),
    2: ('دوئی', 'دوجی', 'دوجا', 'دوسری', 'دوسرا', 'دوم', 'دوئم', 'ثانی'),
    3: ('تیجی', 'تیجا', 'تریجی', 'تیسری', 'تیسرا', 'سوم', 'سوئم', 'ثالث'),
    4: ('چوتھی', 'چوتھا', 'چہارم'),
    5: ('پنجویں', 'پانچویں', 'پنجم'),
}
_EDITION_OF_WORD = {w: n for n, ws in EDITION_WORDS.items() for w in ws}


def edition_from_value(v: str) -> Optional[int]:
    """Edition number from an edition label's value (first ordinal token)."""
    for t in lang.tokens(v):
        if t in _EDITION_OF_WORD:
            return _EDITION_OF_WORD[t]
    return None


_DIGIT_RE = re.compile(r'\d')


def _imprint_frame(t: str) -> Optional[str]:
    """A frame usable in an imprint (Arabic text, numbers, ISBN / F.nn refs)."""
    t = t.strip()
    if len(t) >= 2 and (AR.search(t) or _DIGIT_RE.search(t) or 'ISBN' in t):
        return t
    return None


def imprint_frames_from(lines: List[str], start: int, limit: int) -> List[Tuple[int, str]]:
    out = []
    for i in range(max(0, start), len(lines)):
        t = _imprint_frame(lines[i])
        if t is not None:
            out.append((i, t))
            if len(out) >= limit:
                break
    return out


def _is_imprint_sep(t: str) -> bool:
    return bool(re.fullmatch(r'[\s۔.\-–—_:]+', t))


# A frame of >= 120 chars is a prose paragraph, never a label or value: over
# all 208 files the imprint values run to <= 65 chars (a list of composers in
# 98th), while the 90-frame window without an end word ran into prose in 8
# files - e.g. the 1,584-1,601-char publisher's note 'ہک ضروری گہل' in 250th,
# 255th and 4th (whose whole decodable text is that note), the story in
# 346th/Saray Hath, the Urdu sketches of 47th, the preface of 165th - 65
# frames of >= 200 chars that would have been removed as 'imprint'. No label
# was found after such a frame except two in prose (label_fallback windows).
IMPRINT_PROSE_MIN = 120


def _is_end_word(t: str) -> bool:
    nt = norm(t)
    return nt in IMPRINT_END_N or (nt.startswith('بسم') and 'رحیم' in nt)


def _address_like(t: str) -> bool:
    """A frame of the address block that closes an imprint (see ADDRESS_WORDS)."""
    s = t.strip()
    if RE_PHONE_LIKE.match(s) and len(re.findall('[0-9۰-۹٠-٩]', s)) >= 5:
        return True
    if 'ISBN' in s or '@' in s or 'www' in s.lower():
        return True
    return any(w in s for w in ADDRESS_WORDS)


def _value_end(fr, n, end):
    """Index of the value frame of the label at n (separators skipped), or None."""
    m = n + 1
    while m < end and _is_imprint_sep(fr[m][1]):
        m += 1
    return m if m < end else None


def _imprint_window(fr, anchor):
    """End (exclusive) of the window that starts at fr[anchor]: the first prose
    frame (>= 120 chars), an end word not followed by a label within 6 frames,
    or 90 frames - then cut back to the last label / value pair (or inline
    ISBN / edition statement) plus the address block after it."""
    end = min(len(fr), anchor + IMPRINT_WINDOW)
    for n in range(anchor, end):
        t = fr[n][1]
        if len(t) >= IMPRINT_PROSE_MIN:
            end = n
            break
        if n > anchor + 3 and norm(t.strip().rstrip(':۔-').strip()) == _DEDICATION_N:
            end = n
            break
        if n > anchor + 3 and _is_end_word(t) and not any(
                imprint_label_field(fr[m][1], weak=True)
                for m in range(n + 1, min(len(fr), n + 1 + IMPRINT_END_LOOKAHEAD))):
            end = n
            break
    last = anchor
    n = anchor
    while n < end:
        t = fr[n][1]
        if imprint_label_field(t, weak=True):
            last = n
            m = _value_end(fr, n, end)
            if m is not None and not imprint_label_field(fr[m][1]) and not _is_contents_cell(fr[m][1]):
                last = m
                n = m + 1
                continue
        elif ('ISBN' in t and RE_ISBN.search(t)) or RE_EDITION_STATEMENT.match(t.strip()):
            last = n
        n += 1
    m = last + 1
    while m < end and (_address_like(fr[m][1]) or _is_imprint_sep(fr[m][1])):
        m += 1
    return m


def _is_contents_cell(t: str) -> bool:
    s = t.strip()
    return s in TOC_HEADER_CELLS or s in TOC_MARKERS


def _read_pairs(fr, anchor, end):
    """Label / value pairs of a window. A value frame is consumed (it is never
    read as the next label): 'موضوع' / 'شاعری' is the subject pair even though
    'شاعری' also serves as a credit label ('شاعری' / 'احمد علی سائیںؒ', 85th).
    An unlabelled '(دوسرا ایڈیشن)' (40th) is an edition statement."""
    pairs, inline_isbn = [], []
    n_labels = 0
    n = anchor
    while n < end:
        i, t = fr[n]
        f = imprint_label_field(t, weak=True)
        if not f:
            if 'ISBN' in t and RE_ISBN.search(t):      # 'ISBN No. 978-...' in one frame
                inline_isbn.append((RE_ISBN.search(t).group(0), i))
            m_ed = RE_EDITION_STATEMENT.match(t.strip())
            if m_ed and m_ed.group(1) in _EDITION_OF_WORD:
                pairs.append({'field': 'edition', 'label': t, 'value': t,
                              'label_line': i, 'value_line': i, 'kind': 'statement'})
            n += 1
            continue
        n_labels += 1
        m = _value_end(fr, n, end)
        if m is not None:
            v = fr[m][1]
            vf = imprint_label_field(v)
            if (not vf or (f in ('subject', 'language') and vf == 'author')) and not _is_contents_cell(v):
                pairs.append({'field': f, 'label': t, 'value': v,
                              'label_line': i, 'value_line': fr[m][0]})
                n = m + 1
                continue
        n += 1
    return pairs, inline_isbn, n_labels


_FALLBACK_HINT = re.compile('اشاعت|طبع|چھپن|چَھپن|سنہ|واری|ISBN|حوالہ|قیمت|ہدیہ')
# title-page frames read before a rights-line anchor (credits only; the frames
# stay text): 19th #6968 'کلام' / 'استاد عبدالرشید تاج', 31st 'کلام' /
# 'عبداللطیف ساجنؔ', 206th #7457 'تصنیف' / 'پطرس بخاری', 212th 'تحقیق و تصنیف'
# / 'نذیر بھٹی', 234th 'شاعر' / 'تاج سعید' (review 2026-09-25)
TITLE_PAGE_FRAMES = 12


def _prev_imprint_frames(lines: List[str], line: int, limit: int) -> List[int]:
    out = []
    j = line - 1
    while j >= 0 and len(out) < limit:
        if _imprint_frame(lines[j]) is not None:
            out.append(j)
        j -= 1
    return out                   # nearest first


def _fallback_start(lines: List[str], i: int) -> int:
    """Start of a label-anchored window whose first year / ISBN / GHA / price
    label is at line i: back over the label / value frames before it (not a
    fixed 12 frames: 118th, 340th and 358th lost their title and author
    pairs to that), stopping at a prose frame or a frame that is neither."""
    prev = _prev_imprint_frames(lines, i, 40)
    start, k = i, 0
    while k < len(prev):
        t = lines[prev[k]].strip()
        if len(t) >= IMPRINT_PROSE_MIN:
            break
        if imprint_label_field(t, weak=True) or RIGHTS.search(t):
            start = prev[k]
            k += 1
            continue
        if _is_imprint_sep(t):
            k += 1
            continue
        if k + 1 < len(prev) and imprint_label_field(lines[prev[k + 1]].strip(), weak=True):
            start = prev[k + 1]          # a value and its label
            k += 2
            continue
        break
    return start


def title_page_pairs(lines: List[str], a_line: int) -> List[dict]:
    """Title / credit label-value pairs among the <= 12 frames before a rights
    line (metadata only: the frames stay text)."""
    prev = []
    for j in _prev_imprint_frames(lines, a_line, TITLE_PAGE_FRAMES):
        t = lines[j].strip()
        if len(t) >= IMPRINT_PROSE_MIN:
            break
        prev.append((j, t))
    prev.reverse()
    pairs = []
    n = 0
    while n < len(prev) - 1:
        i, t = prev[n]
        f = imprint_label_field(t, weak=True)
        if f in ('title', 'author'):
            i2, v = prev[n + 1]
            if not imprint_label_field(v) and not RIGHTS.search(v) and not _is_contents_cell(v):
                pairs.append({'field': f, 'label': t, 'value': v, 'label_line': i, 'value_line': i2})
                n += 2
                continue
        n += 1
    return pairs


def find_imprints(lines: List[str], candidates: Optional[List[int]] = None) -> List[dict]:
    """Imprint windows of a file. The first rights frame (جملہ حقوق, its typo
    جملہ حقوب, or the Hindko 'سارے حق ... محفوظ'; <= 80 chars) anchors the
    window (else the label / value run before the first year / ISBN / GHA /
    price label); it runs to the first prose frame (>= 120 chars), an end
    word, or 90 frames, and is then cut back to its last label / value pair
    plus the address block (_imprint_window). Later rights anchors are
    accepted only when their window holds >= 3 label frames: in the doubled
    'NEW FARMA SIZE' files the second copy repeats the imprint (102nd, 112th,
    173rd, 26th, 305th). A label-anchored (fallback) window also needs >= 3
    labels: real imprints carry 7-13, while a stray label in a conference
    paper (44th) carries 1; a non-standard rights line needs >= 2. Frames are
    only read near anchors. 'title_page_pairs': credits written on the title
    page just before the rights line (metadata only)."""
    idx = range(len(lines)) if candidates is None else candidates   # multi-char frames
    anchors = [i for i in idx if ('حقو' in lines[i] or 'محفوظ' in lines[i]) and RIGHTS.search(lines[i])
               and len(lines[i].strip()) <= RIGHTS_MAX_LEN and _imprint_frame(lines[i]) is not None]
    fallback = False
    if not anchors:
        for i in idx:
            l = lines[i]
            # labels normalise to <= 20 chars, so longer frames cannot match;
            # every year/ISBN/GHA/price label contains one of the hint words
            if len(l) <= 60 and _FALLBACK_HINT.search(l) and _imprint_frame(l) is not None and \
                    imprint_label_field(l.strip()) in ('year', 'isbn', 'gha_ref', 'price'):
                anchors = [_fallback_start(lines, i)]
                fallback = True
                break
    out = []
    last_end_line = -1
    for a_line in anchors:
        if a_line <= last_end_line:
            continue
        fr = imprint_frames_from(lines, a_line, IMPRINT_WINDOW)
        if not fr:
            continue
        end = _imprint_window(fr, 0)
        if end <= 0:
            continue
        pairs, inline_isbn, n_labels = _read_pairs(fr, 0, end)
        if out and n_labels < 3:
            continue
        if fallback and n_labels < 3:
            continue        # a stray year/price label in running text (44th: 1 label)
        if not fallback and not RIGHTS_STRICT.search(fr[0][1]) and n_labels < 2:
            continue        # 'سارے حق ... محفوظ' in running text
        out.append({'anchor_line': fr[0][0], 'end_line': fr[end - 1][0],
                    'frame_lines': [fr[n][0] for n in range(end)],
                    'rights_line': fr[0][1] if not fallback else None,
                    'anchor_kind': 'label_fallback' if fallback else 'rights_line',
                    'pairs': pairs, 'inline_isbn': inline_isbn, 'n_labels': n_labels,
                    'title_page_pairs': [] if fallback else title_page_pairs(lines, fr[0][0])})
        last_end_line = fr[end - 1][0]
    return out

def _years_months(value: str):
    v = value.translate(_TO_ASCII)
    years = [int(y) for y in RE_YEAR.findall(v)]
    months = sorted({URDU_MONTHS[t] for t in lang.tokens(value) if t in URDU_MONTHS})
    return years, months


def _isbn13(raw: str) -> Optional[str]:
    digits = re.sub(r'[^0-9X]', '', raw.upper())
    return digits if len(digits) == 13 and digits.isdigit() else None


def _isbn_check_ok(d: str) -> bool:
    s = sum(int(c) * (1 if k % 2 == 0 else 3) for k, c in enumerate(d[:12]))
    return (10 - s % 10) % 10 == int(d[12])


CREDIT_ROLES = ('author', 'translator', 'compiler')
# fields that belong to one edition's block of the imprint (see parse_imprint)
BLOCK_FIELDS = ('publisher', 'printer', 'price', 'supervised_by')


def parse_imprint(window: dict) -> dict:
    """Parsed fields of one imprint window (L10). Values are the frames as
    written. Years are digits only (the era sign after them - ء, or ئ on
    decodes before 2026-09-25 - is ignored);
    month precision only when an Urdu month name is present. THE IMPRINT
    YEAR IS THE YEAR OF THIS EDITION, not when the text was written (3rd:
    first edition 1980, this edition March 2016; its preface is dated 1980).
    ISBN kept only when it has exactly 13 digits (4 placeholders like
    '978-969-687- -' become None; 26th and 58th share one ISBN, so it is not
    a unique key). gha_ref is kept raw (173rd: imprint 1995 but F.173/17 -
    no year is inferred from it)."""
    first = {}
    for p in window['pairs']:
        first.setdefault(p['field'], p)
    out = {'book_title': None, 'book_title_from': None, 'publisher': None,
           'subject': None, 'language_statement': None, 'price': None, 'supervised_by': None,
           'printer': None, 'gha_ref': None, 'publication_year': None, 'publication_date': None,
           'publication_date_precision': None, 'edition': None, 'first_edition_year': None,
           'year_values': [], 'edition_values': [], 'isbn': None, 'isbn_raw': None,
           'isbn_check_digit_ok': None, 'notes': []}
    for role in CREDIT_ROLES:
        out[role] = out[role + '_label'] = out[role + '_from'] = None
    for f, key in (('title', 'book_title'), ('publisher', 'publisher'),
                   ('subject', 'subject'), ('language', 'language_statement'), ('price', 'price'),
                   ('supervised_by', 'supervised_by'), ('printer', 'printer'), ('gha_ref', 'gha_ref')):
        if f in first:
            out[key] = first[f]['value']
    if out['book_title'] is not None:
        out['book_title_from'] = 'imprint'
    # credits (review 2026-09-25): each credit label names one or more roles
    # (credit_roles); the first value per role wins, imprint pairs before the
    # title-page pairs written just above the rights line. A role the imprint
    # does not state is taken from the title page (19th: the imprint names
    # only the compiler 'مرتبہ' / 'نسیم سحرؔ', the title page the poet 'کلام'
    # / 'استاد عبدالرشید تاج').
    for src, prs in (('imprint', window['pairs']), ('title_page', window.get('title_page_pairs', ()))):
        for p in prs:
            if p['field'] == 'title' and src == 'title_page' and out['book_title'] is None:
                out['book_title'], out['book_title_from'] = p['value'], 'title_page'
            if p['field'] != 'author':
                continue
            for role in dict.fromkeys(credit_roles(p['label'], True) or ()):
                if out[role] is None:
                    out[role], out[role + '_label'], out[role + '_from'] = p['value'], p['label'], src
    # the GHA label is sometimes left empty and the next frame is a name
    # (216th: 'محمد ضیائ الدین'); only an 'F.nn' / 'F-nn' reference is kept (raw)
    if out['gha_ref'] is not None and not RE_GHA_LIKE.search(out['gha_ref']):
        out['notes'].append('GHA reference label without a reference value (%r): not used' % out['gha_ref'])
        out['gha_ref'] = None
    # years / edition (L10). Editions are modelled per imprint: an edition
    # can be stated by the year label ('سال اشاعت دوئم', 'اشاعتِ ثانی', 'سال
    # اشاعت (دوم)') or by an edition pair ('وار:' / 'چَھپن ط وار:' / 'ایڈیشن'
    # -> 'پہلی', 'دوئی ۔ اکتوبر ۲۰۱۸ئ'). The edition pairs were not read before,
    # so 358th (وار: پہلی / چھپنے دی تاریخ: ۲ دسمبر ۲۰۰۵ / وار: دوئی ۔ اکتوبر
    # ۲۰۱۸ئ, ISBN of the GHA 2018 edition) got publication_year 2005 and no
    # edition. publication_year is the year of the HIGHEST edition stated in
    # this window; when that edition carries no date, no year is chosen.
    entries = []
    for pos, p in enumerate(window['pairs']):
        if p['field'] == 'year':
            years, months = _years_months(p['value'])
            out['year_values'].append(p['value'])
            entries.append({'edition': edition_of(p['label']), 'years': years, 'months': months,
                            'value': p['value'], 'pos': pos, 'kind': 'year'})
        elif p['field'] == 'edition':
            out['edition_values'].append(p['value'])
            ed = edition_from_value(p['value'])
            if ed is not None:
                years, months = _years_months(p['value'])
                entries.append({'edition': ed, 'years': years, 'months': months,
                                'value': p['value'], 'pos': pos, 'kind': 'edition'})
    for k, e in enumerate(entries):
        # an undated edition statement opens a block of pairs that runs to the
        # next edition statement: the first unlabelled date inside that block
        # is the edition's (358th: 'وار: پہلی' / composer / printer /
        # 'چھپنے دی تاریخ: ۲ دسمبر ۲۰۰۵' / price, then 'وار: دوئی ۔ اکتوبر ۲۰۱۸')
        if e['kind'] != 'edition' or e['years']:
            continue
        for nxt in entries[k + 1:]:
            if nxt['kind'] == 'edition':
                break
            if nxt['edition'] is None and nxt['years'] and not nxt.get('used'):
                e['years'], e['months'], nxt['used'] = nxt['years'], nxt['months'], True
                out['notes'].append('edition %d dated by the date pair %r in its block'
                                    % (e['edition'], nxt['value']))
                break
    stated = [e for e in entries if e['edition'] is not None]
    unl = [e for e in entries if e['edition'] is None and e['years'] and not e.get('used')]
    # edition blocks (review 2026-09-25, 358th): when a window holds two or
    # more edition statements of its own ('وار:' pairs), each opens the block
    # of pairs that runs to the next one (the pairs before the first belong
    # to the first). The edition-specific credits - publisher, printer, price,
    # supervision - are read from the block of the edition that supplies the
    # publication year; a value stated only in another edition's block is
    # not used (358th: the Rawalpindi publisher of the 2005 first edition was
    # combined with the GHA 2018 second edition's year and ISBN).
    ed_stmts = sorted((e for e in entries if e['kind'] == 'edition'), key=lambda e: e['pos'])
    blocks = None
    if len({e['edition'] for e in ed_stmts}) >= 2:
        bounds = [0] + [e['pos'] for e in ed_stmts[1:]] + [len(window['pairs'])]
        blocks = [(bounds[k], bounds[k + 1], ed_stmts[k]['edition']) for k in range(len(ed_stmts))]
    undated = [e for e in stated if not e['years']]
    if len(stated) == 1 and undated and len(unl) == 1:
        # one edition statement and one unlabelled date in the window (340th:
        # 'چَھپنے ط دا مہینہ، سَنہ تہ جأ: اکتوبر، ۲۰۱۸' + 'چَھپن ط وار: پہلی')
        e = undated[0]
        e['years'], e['months'], unl[0]['used'] = unl[0]['years'], unl[0]['months'], True
        out['notes'].append('edition %d dated by the only date pair %r' % (e['edition'], unl[0]['value']))
        unl = []
    chosen = None
    if stated:
        top = max(e['edition'] for e in stated)
        out['edition'] = top
        dated_top = [e for e in stated if e['edition'] == top and e['years']]
        if dated_top:
            chosen = dated_top[0]
        else:
            out['notes'].append('edition %d stated without a date: no publication year' % top)
        ed1 = [e for e in stated if e['edition'] == 1 and len(set(e['years'])) == 1]
        if top > 1 and ed1:
            out['first_edition_year'] = ed1[0]['years'][0]
    elif unl and len({y for e in unl for y in e['years']}) == 1:
        chosen = unl[0]
    elif unl:
        out['notes'].append('several unlabelled imprint years %s: none chosen'
                            % sorted({y for e in unl for y in e['years']}))
    if chosen is not None:
        ys = sorted(set(chosen['years']))
        if len(ys) == 1:
            out['publication_year'] = ys[0]
            if len(chosen['months']) == 1:
                out['publication_date'] = '%04d-%02d' % (ys[0], chosen['months'][0])
                out['publication_date_precision'] = 'month'
            else:
                out['publication_date'] = '%04d' % ys[0]
                out['publication_date_precision'] = 'year'
        else:
            out['notes'].append('several years in one imprint value %r: none chosen' % chosen['value'])
    in_block = None
    if blocks is not None and chosen is not None:
        for a, b, ed in blocks:
            if a <= chosen['pos'] < b:
                in_block = (a, b, ed)
    if in_block is not None:
        a, b, ed = in_block
        for f in BLOCK_FIELDS:
            vals = [p['value'] for pos, p in enumerate(window['pairs']) if p['field'] == f and a <= pos < b]
            other = [p['value'] for pos, p in enumerate(window['pairs']) if p['field'] == f and not a <= pos < b]
            if vals:
                out[f] = vals[0]
            elif other:
                out[f] = None
                out['notes'].append('%s %r is stated only for another edition than edition %d: not used'
                                    % (f, other[0], ed))
    # ISBN (of the chosen edition's block when the window has edition blocks)
    cands = [(p['value'], p['value_line']) for pos, p in enumerate(window['pairs']) if p['field'] == 'isbn'
             and (in_block is None or in_block[0] <= pos < in_block[1])]
    cands += window['inline_isbn']
    for raw, _ in cands:
        m = RE_ISBN.search(raw)
        if not m:
            continue
        if out['isbn_raw'] is None:
            out['isbn_raw'] = m.group(0).strip()
        d = _isbn13(m.group(0))
        if d:
            out['isbn'], out['isbn_raw'] = d, m.group(0).strip()
            out['isbn_check_digit_ok'] = _isbn_check_ok(d)
            break
    if out['isbn_raw'] and not out['isbn']:
        out['notes'].append('ISBN placeholder/invalid %r: not used' % out['isbn_raw'])
    return out


# ==========================================================================
# frame classification per file
# ==========================================================================
# every character segment.ARABIC matches (single-char frames are tested by set lookup)
_AR_CHARS = frozenset(chr(c) for c in range(0x0600, 0xFF00) if AR.match(chr(c)))
# chars that can make a frame non-residue: Arabic script, digits, separator
# chars, whitespace (a two-char frame has < 3 Latin letters, so never English)
_SPECIAL_CHARS = _AR_CHARS | frozenset(
    '0123456789' + ''.join(map(chr, range(0x06F0, 0x06FA))) + ''.join(map(chr, range(0x0660, 0x066A)))
    + ' ' + chr(9) + chr(10) + chr(11) + chr(12) + chr(13) + chr(0xA0)
    + ''.join(map(chr, (0x06D4, 0x2E, 0x2D, 0x2013, 0x2014, 0x5F, 0x2A, 0x066D, 0x2605, 0x2606,
                        0x2022, 0x25CF, 0x25CB, 0x25CA, 0x7E, 0x3D, 0x2B, 0x2026))))
# Structural categories (principle 4: removed from the text, kept as
# metadata / side records / boundaries). 'lenbyte' and 'outside' no longer
# exist (L2 and L3 corrections).
STRUCTURAL = ('residue', 'header_run', 'toc', 'imprint', 'separator',
              'genre_heading', 'language_label', 'numbering', 'speaker', 'fragment', 'joiner')
# structural categories whose text is not kept anywhere else: no frame with
# >= 2 Arabic letters or >= 3 Latin letters may be in them (test C22), except
# a long frame of < 30% script (L5 blob) and the R8 Latin residue classes
# (paths, addresses, font strings, InPage style fragments outside the text
# flow). 'joiner' is a lone ':' / '=' not merged into a line (its two
# neighbours were removed as copies). genre_heading / language_label frames
# are now LINES of the unit they open (review 2026-09-25), so a frame keeps
# that category only where no record took it.
DROPPED_STRUCTURAL = ('residue', 'separator', 'numbering', 'joiner')
TEXTUAL = ('text', 'english_text')
LONG_BLOB = 200           # segment.MAX_ASCII_ONLY_LEN: long frame ...
LONG_BLOB_SCRIPT = 0.30   # ... with < 30% Arabic script = binary/style residue (L5)


_NUM_SEP_CHAR = re.compile('[0-9\u06F0-\u06F9\u0660-\u0669\s\u06D4.\-\u2013\u2014_*\u066D\u2605\u2606\u2022\u25CF\u25CB\u25CA~=+\u2026]')
AR_LETTER2 = re.compile(AR_LETTER.pattern + '.*?' + AR_LETTER.pattern, re.S)
_IS_ONE = (1).__eq__
_IS_NOT_ONE = (1).__ne__
_IS_TWO = (2).__eq__



_THREE_LATIN = re.compile('[A-Za-z][^A-Za-z]*[A-Za-z][^A-Za-z]*[A-Za-z]')


def _latin_candidates(lines: List[str]) -> set:
    """Distinct frames of a file with >= 3 Latin letters (R8 repetition count)."""
    return {t for t in compress(lines, map((3).__le__, map(len, lines))) if _THREE_LATIN.search(t)}


# L2 (BOOKS_SPEC_CORRECTIONS; review 2026-09-25): lone characters are NOT
# stripped at string level. A lone ':' or '=' frame directly between two text
# frames is the speaker / label separator of dramas and glossaries (305th
# 'وخت' / ':' / 'دن' x1,360, 307th 'نامسلم' / '=' / 'اجازت اے۔' x2,732, 196th
# 'وضاحت' / ':' / gloss x216; 1,674 ':' and 2,754 '=' frames inside records
# were dropped): it becomes a 'joiner' and build_units writes 'name: text' /
# 'name = text' as one line. A lone '…' / '٭' / '*' between two text frames
# is a separator (118th: '…' between each Persian quatrain and its Hindko
# rendering, 278 frames). A lone Arabic letter between two text frames is
# text when it is the letter heading of a si-harfi stanza (the next frame
# starts with it: 149th 'ٹ' / 'ٹ ٹُر گئے ...') or the letter being taught /
# introduced (the next frame speaks of a حرف: 131st 'ن' / 'یہ حروف تہجی میں
# ...', 106th 'ن' / 'حرفی'); the 'س' before the invocation verse of many
# books (an ornament glyph) is not. Everything else single ('@' 2.1M, '~'
# 1.0M ...) stays residue; U+FFFD is counted.
JOINER_CHARS = frozenset(':=')
LONE_SEP_CHARS = frozenset('…٭*•●★☆◊')
_LONE_NEIGHBOUR = ('text', 'speaker', 'genre_heading', 'language_label')
# L6 labels (review 2026-09-25): a genre / language word is a heading only
# between boundaries - not a word of a sentence or verse set in its own frame
# (112th 'جس دی زبان پشتو ائی برے میرے نال' / 'ہندکو' / 'اِسراں بول دی ...',
# 111th 'ہندکو' / 'بولنے والیآں دے ناں۔'), not a headword ('حمد:' + its
# definition in 135th's glossary of genres), not a greeting before a letter
# (135th 'سلام'), not a cell of a row of column heads (136th 'ہندکو' / 'پشتو'
# / 'اُردو' x20, 44th 'ہندکو' / 'پنجابی'). 122 such frames left the text.
LABEL_MID_WORDS = 5      # an unfinished line of >= 5 words before a bare language word
PROSE_NEXT_LEN = 75      # is_verse's upper bound: a longer next frame is prose
LANG_LABELS_NOSPACE = frozenset(re.sub(r'\s+', '', x) for x in LANG_LABELS)
# Isolated fragments (review 2026-09-25): 2,801 frames of 2-4 Arabic letters
# (<= 8 chars) with no text frame within 3 raw frames on either side sit in
# InPage's style / object residue ('یٹ' 111, 'اٹ' 106, 'نٹ' 52, 'رٹ' 50 ...;
# 2,485 were record lines, and B removed real words as 'copies' of them: 'یا'
# of 'ی ا'). They are the structural category 'fragment'; their text is kept
# in per_file['fragments'].
FRAGMENT_MAX_LETTERS = 4
FRAGMENT_MAX_LEN = 8
FRAGMENT_WINDOW = 3
_FRAGMENT_QUIET = ('residue', 'numbering', 'separator')
# R8 embedding: a Latin frame directly between two text frames (or between
# text and another English frame) is text of that flow
_EMBED_CATS = ('text', 'speaker', 'genre_heading', 'language_label', 'english_text')


def _textlike(t: str) -> bool:
    s = t.strip()
    return bool(AR_LETTER2.search(s)) and not RE_SEP.match(s) and not is_numeric(s)


def _is_prose_frame(t: str) -> bool:
    s = t.strip()
    return len(s) > PROSE_NEXT_LEN or s.endswith(END_PUNCT)


def _label_category(lines: List[str], i: int, t: str, kt: str, prev, nxt, cat) -> str:
    """genre_heading / language_label, or what the label word is instead:
    'speaker' (a colon-ended label before text: prefixed to it), 'text'."""
    t_b = nxt[1] if nxt is not None else None
    next_text = t_b is not None and _textlike(t_b)
    if t.rstrip().endswith((':', ':۔', ':-')) and next_text:
        return 'speaker'
    if i + 2 < len(lines) and lines[i + 1].strip() in JOINER_CHARS and _textlike(lines[i + 2]):
        return 'text'          # 'اُردو ترجمہ' / ':' / translation (196th): one joined line
    is_lang = kt in LANG_LABELS
    if is_lang and any(x is not None and re.sub(r'\s+', '', label_key(x[1])) in LANG_LABELS_NOSPACE
                       for x in (prev, nxt)):
        return 'text'          # a row of column heads
    if next_text and _is_prose_frame(t_b):
        return 'text'          # a word before running prose (a greeting, 'ہندکو بولنے ...')
    if is_lang and prev is not None and cat[prev[0]] == 'text':
        a = prev[1].strip()
        if len(a.split()) >= LABEL_MID_WORDS and not a.endswith(END_PUNCT + (':',)):
            return 'text'      # inside a sentence / verse
    return 'genre_heading' if kt in GENRE_HEADINGS else 'language_label'


def classify_file(lines: List[str], latin_counts: Counter, kind: Optional[str] = None) -> dict:
    """Assign one layout category to every frame (line) of a file.

    Order (probe_books_layout_records.classify_file, plus the L5 long-frame
    script test and the R8 English test):
      residue  - one char after stripping and not a number, and not one of
                 the lone characters kept by L2 below; KNOWN_NOISE; no Arabic
                 letter and not a number / English; <= 1 Arabic letter; or a
                 > 200-char frame under 30% Arabic script (L5)
      header_run - L3: >= 3 identical consecutive frames <= 60 chars at an
                 end of the text (before / after the main span), >= 5 inside
      toc / imprint - L9 side records (only the frames a TOC block consumed)
      separator / genre_heading / language_label / numbering / speaker - L6
                 (label words only where they stand between boundaries,
                 _label_category)
      joiner   - L2: a lone ':' / '=' between two text frames
      text     - everything else with >= 2 Arabic letters (NO length minimum:
                 is_content_line's 12-char floor dropped 27.6% of Arabic frames
                 in books - poem titles, speaker labels, short misras - while
                 letting through almost no residue, 0.22%), a lone Arabic
                 letter between text frames, and (R8 words) short English
                 frames inside the main span or directly between text frames
                 (see lang.is_english_word_frame)
      english_text - R8 English frames (every Latin frame of a .docx, which
                 has no InPage style residue: 22nd's Index.docx)
      fragment - an isolated 2-4-letter frame amid residue (text kept in
                 per_file['fragments'])
    L3: NOTHING is cut for lying before or after the main span: frames there
    are classified by the same rules (the span only sizes header runs and
    marks records 'outside_main_span').
    """
    n = len(lines)
    cat: List[Optional[str]] = [None] * n
    is_docx = kind == 'docx'
    # ~82% of all frames (11.2M of 13.7M) are single characters of InPage's
    # style/object residue ('@' 2.1M, '~' 1.0M ...): decided in bulk
    lens = list(map(len, lines))
    single_num = []                        # lone digits: numbering frames / TOC pages
    for i in compress(range(n), map(_IS_ONE, lens)):
        if single_char_category(lines[i]) is None:
            single_num.append(i)
        else:
            cat[i] = 'residue'
    multi = list(compress(range(n), map(_IS_NOT_ONE, lens)))
    # 1.9M more are two-char symbol pairs ('T"', '~W'): without an Arabic
    # char, digit, separator char or space they can only be residue
    for i in compress(range(n), map(_IS_TWO, lens)):
        t = lines[i]
        if (t[0] not in _SPECIAL_CHARS and t[1] not in _SPECIAL_CHARS
                and not t[0].isspace() and not t[1].isspace()):
            cat[i] = 'residue'
    latin_cands = []                       # decided once the text frames are known (R8)
    for i in multi:
        if cat[i] is not None:
            continue
        t = lines[i]
        has_ar = AR.search(t) is not None
        if not has_ar:
            # fast path for longer symbol frames: without an Arabic char, a
            # digit or a separator char it can only be residue or (with >= 3
            # Latin letters) English
            if not _NUM_SEP_CHAR.search(t) and not _THREE_LATIN.search(t):
                cat[i] = 'residue'
                continue
            s = t.strip()
            if len(s) <= 1:                # one char after stripping: the single-char rule
                if not s or single_char_category(s) is not None:
                    cat[i] = 'residue'
                    continue
            elif inpage.is_noise(t):
                cat[i] = 'residue'
                continue
        sep = bool(RE_SEP.match(t))
        num = is_numeric(t)
        two_letters = has_ar and AR_LETTER2.search(t) is not None     # >= 2 Arabic letters
        non_text = (not has_ar and not num and not sep) or (not two_letters and not sep and not num)
        if not non_text and two_letters and len(t) > LONG_BLOB:
            compact = re.sub(r'\s', '', t)
            if len(compact) > LONG_BLOB and segment.script_fraction(t) < LONG_BLOB_SCRIPT:
                non_text = True
        if non_text:
            cat[i] = 'residue'
            # a Latin frame (Arabic-script punctuation / digits allowed, no
            # Arabic letter): English or a word frame is decided below (R8)
            if _THREE_LATIN.search(t) and not AR_LETTER.search(t) and not inpage.is_noise(t):
                latin_cands.append(i)
    lf = [(i, lines[i]) for i in compress(range(n), map(is_, cat, repeat(None)))]

    texts = [t for _, t in lf]
    runs = []
    k = 0
    while k < len(texts):
        m = k
        while m + 1 < len(texts) and texts[m + 1] == texts[k]:
            m += 1
        if m - k + 1 >= HEADER_RUN_MIN and len(texts[k]) <= HEADER_MAX_LEN and not RE_SEP.match(texts[k]):
            runs.append((k, m))
        k = m + 1
    in_run = set()
    for a, b in runs:
        in_run.update(range(a, b + 1))
    longs = [k for k, t in enumerate(texts) if len(t) >= MAIN_SPAN_MIN_LEN and k not in in_run]
    headers = []
    span = None
    lo = hi = None
    if longs:
        lo, hi = longs[0], longs[-1]
        span = (lf[lo][0], lf[hi][0])
    for a, b in runs:
        inside = lo is not None and a >= lo and b <= hi
        if not inside or (b - a + 1) >= HEADER_RUN_INSIDE_MIN:
            for kk in range(a, b + 1):
                cat[lf[kk][0]] = 'header_run'
            headers.append({'text': texts[a], 'count': b - a + 1,
                            'position': ('inside' if inside else 'before' if lo is None or b < lo
                                         else 'after' if a > hi else 'across'),
                            'first_line': lf[a][0]})

    # L9: TOC blocks (only the frames a block consumed) and imprint windows
    tocs = parse_tocs(lines, multi)
    for tb in tocs:
        for i in tb['frames']:
            if cat[i] is None:
                cat[i] = 'toc'
    imprints = find_imprints(lines, multi)
    imp_lines = set()
    for w in imprints:
        imp_lines.update(w['frame_lines'])
        for i in w['frame_lines']:
            if cat[i] is None:
                cat[i] = 'imprint'

    # L6 structure (speaker labels need >= 3 occurrences in the file)
    cnt = Counter(texts)
    for k, (i, t) in enumerate(lf):
        if cat[i] is not None:
            continue
        kt = label_key(t)
        if RE_SEP.match(t):
            cat[i] = 'separator'
        elif kt in GENRE_HEADINGS or kt in LANG_LABELS:
            cat[i] = _label_category(lines, i, t, kt, lf[k - 1] if k else None,
                                     lf[k + 1] if k + 1 < len(lf) else None, cat)
        elif RE_NUMBERING.match(t) or is_numeric(t):
            cat[i] = 'numbering'
        elif (len(t) <= SPEAKER_MAX and t.rstrip().endswith((':', ':۔', ':-'))
              and cnt[t] >= SPEAKER_MIN_COUNT):
            cat[i] = 'speaker'
        elif AR_LETTER2.search(t):
            cat[i] = 'text'
        else:
            cat[i] = 'residue'

    # L2 lone characters between text frames (joiners, separators, letters)
    single_kept = []
    for i in multi:
        if cat[i] not in _LONE_NEIGHBOUR:
            continue
        j = i + 1
        if j + 1 >= n or cat[j] != 'residue':
            continue
        s = lines[j].strip()
        if len(s) != 1 or cat[j + 1] not in _LONE_NEIGHBOUR:
            continue
        if s in JOINER_CHARS and cat[j + 1] == 'text':
            cat[j] = 'joiner'
        elif s in LONE_SEP_CHARS:
            cat[j] = 'separator'
        elif (AR_LETTER.fullmatch(s) and cat[i] == 'text' and cat[j + 1] == 'text'
              and (lines[j + 1].lstrip().startswith(s) or 'حرف' in lines[j + 1])):
            cat[j] = 'text'
        else:
            continue
        if lens[j] == 1:
            single_kept.append(j)

    # R8: Latin frames. A .docx has no InPage style residue: every Latin
    # frame that is not a path / address is English text (22nd Index.docx,
    # 169 entries, was dropped). In an InPage file: English text anywhere
    # (R8, >= 3 words ...); short English word frames inside the main span or
    # directly between two text frames (then without the >= 5-file repeat
    # rule: 208th 'Time'); inside an imprint window -> imprint.
    cand_set = set(latin_cands)
    std = {}

    def std_ok(j):
        """A Latin neighbour that is English on its own (a run of English frames)."""
        if j not in std:
            t_, nf_ = lines[j], latin_counts.get(lines[j], 0)
            std[j] = lang.is_english_text_frame(t_, nf_) or lang.is_english_word_frame(t_, nf_)
        return std[j]

    def textlike(j):
        if not 0 <= j < n:
            return False
        if cat[j] in _EMBED_CATS:
            return True
        return j in cand_set and std_ok(j)
    for i in latin_cands:
        if cat[i] != 'residue':
            continue
        if i in imp_lines:
            cat[i] = 'imprint'          # an English publisher / 'ISBN No.' label
            continue
        t = lines[i]
        nf = latin_counts.get(t, 0)
        if is_docx:
            if lang.latin_frame_category(t, nf, True) not in ('path', 'email_url'):
                cat[i] = 'english_text'
            continue
        emb = textlike(i - 1) and textlike(i + 1)
        if lang.is_english_text_frame(t, nf, emb):
            cat[i] = 'english_text'
        elif lang.is_english_word_frame(t, nf, emb) and (
                emb or (span is not None and span[0] <= i <= span[1])):
            cat[i] = 'text'

    # isolated fragments amid residue
    fragments = []
    loud = [i for i in multi if cat[i] not in _FRAGMENT_QUIET]
    short = [i for i in loud if cat[i] == 'text' and len(lines[i].strip()) <= FRAGMENT_MAX_LEN
             and len(AR_LETTER.findall(lines[i])) <= FRAGMENT_MAX_LETTERS]
    if short:
        la = np.array(loud, dtype=np.int64)
        for i in short:
            k = int(np.searchsorted(la, i))
            left = la[k - 1] if k > 0 else None
            right = la[k + 1] if k + 1 < len(la) else None
            if (left is None or i - left > FRAGMENT_WINDOW) and (right is None or right - i > FRAGMENT_WINDOW):
                cat[i] = 'fragment'
                fragments.append([i, lines[i]])
    if single_kept:
        multi = sorted(multi + single_kept)
    # title-page evidence (frames before the main span; they stay text)
    title_page = [t for i, t in lf if span and i < span[0] and cat[i] == 'text'][:15]
    return {'cat': cat, 'tocs': tocs, 'imprints': imprints, 'headers': headers,
            'span': span, 'title_page': title_page, 'multi': multi, 'single_num': single_num,
            'fragments': fragments}


# L12 structural fallback: colon-headword share. Measured (verify_books_layout
# EXTRA2): frames <= 40 chars ending ':' / ':۔' are 48.5% of the 25th
# dictionary's frames, but also 31-41% in dramas (102nd, 109th, 129th, 308th),
# where they are repeated speaker labels (distinct share 1.9-28.6%);
# dictionary headwords are distinct (95.9-100%). BOOKS_SPEC_CORRECTIONS L12
# sets the structure test to colon-ended frames >= 8% of the frames AND >= 90%
# of them distinct. Re-measured on the 2026-09-25 decodes (the spurious
# \x04':' colons are gone; text frames of files not found by path words), the
# pair fires for 165th SANJHA BAGH (.INP / .B01: 39.2%, 100% distinct) and
# Porranian Galan (18.4%, 100%) - the glossary of obsolete Hindko words
# ('اوڈی :' / 'اوگنْڑ:' / 'اُڈیکوانْڑ:' + a paragraph explaining each), a
# reference work the old long-next guard kept as prose - and for 329th
# Ganjeena e Salook (11.3%, 94.3%): a treatise in sessions ('مجلس نمبر 1..')
# with topic headings ('توحید وجودی:', 'بیعت:'). What separates them is the
# order of the headwords: a glossary is alphabetical - consecutive headwords
# whose first letters do not go backwards in the Urdu alphabet: 165th
# 0.939-0.944 - while topic headings are not: 329th 0.559, 308th 0.588, 213th
# 0.572, 46th 0.539, 60th-63rd 0.52-0.57 (fix3/a08_alpha). So a third test
# (alphabetical order) is added; >= 200 text frames keep small FEHRIST files
# (11-25 frames) out. Review 2026-09-25: a 0.85 threshold was calibrated on
# 165th alone and rejects every real dictionary (25th 0.673, Lughat Par-1
# 0.707, Par-2 0.760, Part-3 0.797 - found by their path words, but the test
# must accept them): the threshold is 0.65, between the dictionaries
# (>= 0.67) and the topic-heading books (<= 0.59).
LEX_COLON_SHARE = 0.08
LEX_COLON_DISTINCT = 0.90
LEX_COLON_SORTED = 0.65
LEX_MIN_FRAMES = 200
# L12 per FILE (review 2026-09-25): a path word marks a lexicon candidate,
# but a file whose text is mostly prose paragraphs is not a word list - 98th
# '000 CONTENT.INP' (title page, dedication, contents, Urdu prefaces) has
# 88.1% of its text chars in frames of >= 200 chars; the dictionaries 0.0-
# 1.1%, 131st's phrasebook 11.5%, 208th 9.0%. Such a file is front matter
# (with a TOC or imprint) or main.
LEX_PROSE_FRAME = 200
LEX_PROSE_SHARE = 0.5
URDU_ALPHABET = 'آابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھءیے'
_ALPHA_RANK = {c: k for k, c in enumerate(URDU_ALPHABET)}
_ALPHA_RANK.update({'ي': _ALPHA_RANK['ی'], 'ى': _ALPHA_RANK['ی'], 'ك': _ALPHA_RANK['ک'],
                    'ه': _ALPHA_RANK['ہ'], 'ۃ': _ALPHA_RANK['ہ'], 'ة': _ALPHA_RANK['ہ'],
                    'أ': _ALPHA_RANK['ا'], 'إ': _ALPHA_RANK['ا'], 'ٱ': _ALPHA_RANK['ا'],
                    'ۓ': _ALPHA_RANK['ے'], 'ئ': _ALPHA_RANK['ء'], 'ؤ': _ALPHA_RANK['و']})


def _first_letter_rank(t: str) -> Optional[int]:
    for ch in t:
        k = _ALPHA_RANK.get(ch)
        if k is not None:
            return k
    return None


def colon_headword_stats(texts: List[str]) -> Tuple[float, float, float]:
    """(share of colon-ended frames <= 40 chars, share of them distinct,
    share of consecutive ones whose first letters are in alphabetical order)."""
    col = [t for t in texts if len(t) <= 40 and t.endswith((':', ':۔'))]
    if not texts or not col:
        return 0.0, 0.0, 0.0
    share = len(col) / len(texts)
    distinct = len(set(col)) / len(col)
    rk = [x for x in map(_first_letter_rank, col) if x is not None]
    pairs = list(zip(rk, rk[1:]))
    ordered = sum(1 for x, y in pairs if y >= x) / len(pairs) if pairs else 0.0
    return share, distinct, ordered


# ==========================================================================
# line-level dedup (R1-R8; probe_books_dedup_final.py)
# ==========================================================================
SUBST = 40         # loose-key chars of a 'substantive' line
RUN_MIN = 5        # W1 ordered run length
RUN_GAP = 100      # W1: first copy >= 100 lines earlier ...
RUN_CHARS = 1000   # ... or run >= 1,000 chars ...
RUN_LONGLINE = 200  # ... or any line >= 200 chars
STACK_MIN = 3      # W2
XRUN_MIN = 3       # X: run of >= 3 already-seen lines
FUZZ = 0.8         # F: 10-gram coverage by ONE earlier file
W1_GAP_MAX = 3     # W1: bridge interruptions of <= 3 fuzzy-copy lines ...
W1_GAP_WINDOW = 8  # ... of the first-copy lines between the two runs' copies
W1_GAP_LONG_MAX = 30  # ... and longer ones when EVERY line is a fuzzy copy
DOUBLED_SHARE = 0.4  # whole-book doubling: W1 removes >= 40% of the file's chars
BOILER_FRACTION = 0.25
BOILER_MIN_DOCS = 12
BOILER_MAX_LEN = 60


def within_file_masks(ks: List[str], lens: List[int]):
    """W1 and W2 (probe_books_dedup_final.masks_within), returning for each
    dropped position the position of the copy that is kept.

    W1: 2nd copy inside an ordered run of >= 5 lines (>= 2 distinct keys)
    repeating an earlier stretch, when the first copy is >= 100 lines earlier
    or the run holds >= 1,000 chars or a line >= 200 chars. Measured: 14 files
    hold their whole book twice (9 'NEW FARMA SIZE', 60th-63rd 'FINAL BOOK',
    2 drafts in 80th/Del; 1,331,636 chars, second copy equal to the first on
    >= 99.9% of lines), plus duplicated prose chapters in 9th (8-12k chars);
    none of it is refrain or radif. Short-gap runs of short lines are kept as
    chorus candidates (3,342 chars, e.g. the 32nd Lok Geet refrain 6 lines on).
    W2: stacks of >= 3 identical consecutive lines -> keep the first (running
    heads such as شیراز اُللغات x593; 3,884 extra copies, 51,983 chars).
    Isolated within-file repeats (refrains, radif, speaker tags) are NEVER
    removed."""
    n = len(ks)
    w1: List[Optional[int]] = [None] * n
    w2: List[Optional[int]] = [None] * n
    chorus = [False] * n
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and ks[j] == ks[i]:
            j += 1
        if j - i >= STACK_MIN:
            for t in range(i + 1, j):
                w2[t] = i
        i = j
    first = {}
    for i, k in enumerate(ks):
        first.setdefault(k, i)
    i = 0
    while i < n:
        j = first[ks[i]]
        if j < i:
            L = 0
            while i + L < n and j + L < i and ks[i + L] == ks[j + L]:
                L += 1
            L = max(L, 1)
            rc = sum(lens[i:i + L])
            big = rc >= RUN_CHARS or max(lens[i:i + L]) >= RUN_LONGLINE
            if L >= RUN_MIN and len(set(ks[i:i + L])) >= 2:
                if i - j >= RUN_GAP or big:
                    for t in range(L):
                        w1[i + t] = j + t
                    runs.append((i, j, L))
                else:
                    for t in range(L):
                        chorus[i + t] = True
            i += L
        else:
            i += 1
    # W1 gap bridging (BOOKS_SPEC_CORRECTIONS W1): a doubled region is
    # extended across interruptions of <= 3 lines that are fuzzy copies
    # (>= 80% of distinct 10-grams) of the ALIGNED first-copy lines - revision
    # debris between two aligned runs (10th: 194 lines / 7,268 chars; 9th 11 /
    # 2,673; 85th 22 / 1,240; s21_w1_orphans: 35 kept lines, 1,978 chars).
    # The aligned first-copy lines are those from the copy of run 1's last
    # line to the copy of run 2's first line (at most W1_GAP_WINDOW lines on):
    # revisions split and merge lines there, so the offset may change and a
    # gap line is tested against the union of that window (requiring one
    # shared offset and one aligned line found only 8 of these lines).
    # Review 2026-09-25: most orphans sit in interruptions of 4-25 lines (9th:
    # seven kept lines between W1 9912 and 9920, e.g. 9914 fully covered; 10th
    # 4,798 chars, 9th 1,795, 85th 396 still >= 0.8 covered). An interruption
    # of up to 30 lines whose first-copy window is not much longer (<= 3 x the
    # gap + 8) is bridged when EVERY line in it is a >= 0.8 fuzzy copy of the
    # window; otherwise nothing is removed and its covered lines are reported
    # as w1_orphan (record flag), since a revision may have added lines there.
    gap, orphan = {}, {}
    for (i1, j1, l1), (i2, j2, _) in zip(runs, runs[1:]):
        g0 = i1 + l1
        a = j1 + l1 - 1
        glen, wlen = i2 - g0, j2 - a
        if glen < 1 or wlen <= 0 or j2 >= g0:
            continue
        short = glen <= W1_GAP_MAX and wlen <= W1_GAP_WINDOW
        if not short and not (glen <= W1_GAP_LONG_MAX and wlen <= 3 * glen + W1_GAP_WINDOW):
            continue
        win = list(range(a, j2 + 1))
        wg = [_uniq(ngram_hashes(ks[x])) for x in win]
        wh = _uniq(np.concatenate(wg))
        if not len(wh):
            continue
        covs = {}
        for g in range(g0, i2):
            if w1[g] is not None or w2[g] is not None:
                continue
            h = _uniq(ngram_hashes(ks[g]))
            if not len(h):
                continue
            pos = np.searchsorted(wh, h)
            pos[pos >= len(wh)] = 0
            covs[g] = (float((wh[pos] == h).mean()), h)
        bridge = short or (covs and all(c_ >= FUZZ for c_, _ in covs.values()))
        for g, (cov, h) in covs.items():
            if cov < FUZZ:
                continue
            # the window line sharing most 10-grams is named as the copy
            best = max(win, key=lambda x, h=h: (int(np.isin(h, wg[x - a]).sum()), -x))
            if bridge:
                gap[g] = (best, round(cov, 4), win)
            else:
                orphan[g] = (best, round(cov, 4))
    return w1, w2, chorus, runs, gap, orphan


class _GramIndex:
    """10-gram hashes of the lines kept so far, as sorted (hash, file) levels
    merged like a binary counter (log-structured). A (hash, file) pair occurs
    once, so per-file coverage = number of a line's distinct 10-grams found
    with that file id / all its distinct 10-grams."""

    def __init__(self):
        self.levels: List[Tuple[np.ndarray, np.ndarray]] = []

    def add(self, hashes: np.ndarray, fid: int) -> None:
        h = _uniq(hashes)
        if not len(h):
            return
        self.levels.append((h, np.full(len(h), fid, dtype=np.int32)))
        while len(self.levels) >= 2 and len(self.levels[-2][0]) <= 2 * len(self.levels[-1][0]):
            (h1, f1), (h2, f2) = self.levels[-2], self.levels[-1]
            hh = np.concatenate([h1, h2])
            ff = np.concatenate([f1, f2])
            # levels hold increasing file ids, so a stable sort on the hash
            # keeps (hash, file) order - same result as lexsort((ff, hh))
            o = np.argsort(hh, kind='stable')
            self.levels[-2:] = [(hh[o], ff[o])]

    def any_hit(self, H: np.ndarray) -> np.ndarray:
        order = np.argsort(H, kind='stable')     # sorted queries: cache-friendly lookups
        Hs = H[order]
        hs_hit = np.zeros(len(H), dtype=bool)
        for h, _ in self.levels:
            pos = np.searchsorted(h, Hs)
            pos[pos >= len(h)] = 0
            hs_hit |= h[pos] == Hs
        hit = np.empty(len(H), dtype=bool)
        hit[order] = hs_hit
        return hit

    def pairs(self, H: np.ndarray, owner: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All (owner, file, query row) matches of the hashes H."""
        own_out, fid_out, row_out = [], [], []
        for h, f in self.levels:
            left = np.searchsorted(h, H, 'left')
            right = np.searchsorted(h, H, 'right')
            cnt = right - left
            if not cnt.any():
                continue
            rows = np.repeat(np.arange(len(H)), cnt)
            starts = np.repeat(left, cnt)
            offs = np.arange(len(rows)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            own_out.append(owner[rows])
            fid_out.append(f[starts + offs])
            row_out.append(rows)
        if not own_out:
            return np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int64)
        return np.concatenate(own_out), np.concatenate(fid_out), np.concatenate(row_out)


def _title_tokens(folder: str) -> List[str]:
    t = re.sub(r'^\s*\d+\w*\s+(Farma\s*)?', '', folder, flags=re.I)
    t = re.split(r'\s+by\s+|\s+By\s+', t)[0]
    return [w for w in re.split(r'[^A-Za-z]+', t.upper()) if len(w) >= 3 and w not in ('FARMA',)]


def _title_match(rel: str) -> float:
    tt = _title_tokens(folder_of(rel))
    nt = [w for w in re.split(r'[^A-Za-z]+', os.path.splitext(os.path.basename(book_rel(rel)))[0].upper())
          if len(w) >= 3]
    if not tt:
        return 0.0
    hit = sum(1 for w in tt if any(difflib.SequenceMatcher(None, w, x).ratio() >= 0.75 for x in nt))
    return hit / len(tt)


def _name_bonus(rel: str) -> int:
    b = os.path.basename(rel).upper()
    return sum(1 for t in ('FINAL', 'NEW', 'CORRECT', 'DONE', 'LATEST') if t in b)


def detect_clusters(files: List[str], seqs: Dict[str, List[tuple]],
                    twins: frozenset = frozenset()) -> Tuple[Dict[str, str], List[dict], Dict[str, str]]:
    """R7 revision clusters (probe_books_dedup_final.detect_clusters) and the
    canonical choice as written in R7:
      edge: shared distinct-line chars cover >= 50% of a file with >= 2,000
            distinct chars, OR same-folder pair with >= 2,000 shared chars and
            10-gram containment >= 50% either way;
      canonical: (1) not under Old/ Del/, not a '- Copy' folder, not .B01;
            (2) the file with most distinct chars, unless others are mutual
            >= 95% copies of it - then numbered Farma folder first, file name
            matching the folder title, FINAL/NEW/CORRECT/DONE/LATEST in the
            name, then most distinct chars.
    Measured: 23 clusters, 42 superseded files (8,261,977 raw chars); only
    40,246 chars in them exist in no retained file, and those are KEPT
    (flagged). The probe CODE ranked by distinct chars only; R7 as written
    differs for 2 clusters (adversarial check): MATTLAN (a 3-char tie,
    numbered 99th folder vs working folder) and 340th, where the code picked
    'PRELIMINARY PAGES.INP' (a superset: prelims + book) over the book file
    'Ruttan Diyan Mattan.inp'. The written rule is used: it keeps the book in
    its main file (role book_passage) instead of attributing it to front
    matter. Only source_path attribution changes, not the retained text.
    On this module's text lines (TOC / imprint frames are structural, not
    text) the rule finds 22 clusters / 38 superseded files: the FEH files of
    26th (Old), 85th and 118th (Old, +.B01) no longer cluster because their
    remaining text is prefaces, not the book - their copied lines are still
    removed by X/F, only the processing order and flag differ."""
    disp: Dict[str, Dict[str, int]] = {}
    dch: Dict[str, int] = {}
    for r in files:
        d: Dict[str, int] = {}
        for _, t, k in seqs[r]:
            d.setdefault(k, len(t))
        disp[r] = d
        dch[r] = sum(d.values())
    inv: Dict[str, List[str]] = defaultdict(list)
    for r in files:
        for k in disp[r]:
            inv[k].append(r)
    shared: Counter = Counter()
    for k, fs in inv.items():
        # the 40-file cap counts files without their .B01 twins, so the edges
        # do not depend on --include-all-backups
        if 1 < len(fs) and sum(1 for f in fs if f not in twins) <= 40:
            for a_i in range(len(fs)):
                for b_i in range(a_i + 1, len(fs)):
                    a, b = fs[a_i], fs[b_i]
                    shared[(a, b)] += min(disp[a][k], disp[b][k])
    grams: Dict[str, np.ndarray] = {}

    def G(r):
        if r not in grams:
            arr = ngram_hashes_many(list(disp[r]))
            grams[r] = _uniq(np.concatenate(arr)) if arr else np.zeros(0, np.uint64)
        return grams[r]

    def ngc(a, b):
        ga, gb = G(a), G(b)
        if not len(ga) or not len(gb):
            return 0.0
        pos = np.searchsorted(gb, ga)
        pos[pos >= len(gb)] = 0
        return float((gb[pos] == ga).mean())

    parent = {r: r for r in files}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), s in sorted(shared.items()):
        ca = s / dch[a] if dch[a] else 0.0
        cb = s / dch[b] if dch[b] else 0.0
        ok = (ca >= 0.5 and dch[a] >= 2000) or (cb >= 0.5 and dch[b] >= 2000)
        if not ok and folder_of(a) == folder_of(b) and s >= 2000:
            if max(ngc(a, b), ngc(b, a)) >= 0.5:
                ok = True
        if ok:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[str, List[str]] = defaultdict(list)
    for r in files:
        groups[find(r)].append(r)

    def line_cont(a, b):
        key = (a, b) if (a, b) in shared else (b, a)
        return shared.get(key, 0) / max(1, dch[a])

    def bad(r):
        return is_superseded_dir(r) or is_copy_dir(r) or is_backup(r)

    sup: Dict[str, str] = {}
    label: Dict[str, str] = {}
    info = []
    for root in sorted(groups):
        g = sorted(groups[root])
        if len(g) < 2:
            continue
        good = [r for r in g if not bad(r)] or list(g)
        good.sort(key=lambda r: (-dch[r], r))
        best = good[0]
        tie = [best] + [r for r in good[1:] if line_cont(r, best) >= 0.95 and line_cont(best, r) >= 0.95]
        if len(tie) > 1:
            # co-equal files (mutual >= 95%): the numbered Farma folder first,
            # then the most distinct chars (BOOKS_SPEC_CORRECTIONS R7: 340th's
            # PRELIMINARY PAGES.INP, prelims + the whole book, 205,877 distinct
            # chars, is canonical over 'Ruttan Diyan Mattan.inp', 203,526),
            # then the file name matching the folder title / FINAL...
            tie.sort(key=lambda r: (-(farma_no(r) < 10 ** 6), -dch[r], -_title_match(r), -_name_bonus(r), r))
        canon = tie[0]
        for r in g:
            if r != canon:
                sup[r] = canon
                # co-equal: in the tie group, or a mutual >= 95% copy of the
                # canonical outside it (the 98th '- Copy' folder file is
                # byte-identical but was never in `tie`, which is built from
                # the preferred files only - review 2026-09-25)
                co_eq = r in tie or (line_cont(r, canon) >= 0.95 and line_cont(canon, r) >= 0.95)
                label[r] = superseded_label(r, canon, co_eq, dch[r], dch[canon])
        info.append({'canonical': canon, 'members': g, 'tie_group': tie if len(tie) > 1 else [],
                     'distinct_chars': {r: dch[r] for r in g},
                     'line_in_canonical': {r: round(line_cont(r, canon), 4) for r in g if r != canon},
                     'superseded_labels': {r: label[r] for r in g if r != canon}})
    return sup, info, label


# R7 labels of superseded files (BOOKS_SPEC_CORRECTIONS R7): only true older
# drafts get 'from_superseded_revision'. Files under Old/ / Del/ and .B01
# backups are older saves by location; FEHRIST / preliminary files are front
# matter (their records already carry role front_matter); co-equal twins
# (mutual >= 95% containment: the 98th '- Copy' folder, the MATTLAN copy,
# 340th's book file) and part files (<= 80% of the
# canonical's distinct chars: the 10th surah files, 103rd's conference file,
# 346th Saray Hath, 66th zaheerudin) are 'part_of_book'. A superseded file of
# similar size that is not a mutual copy (165th SANJHA BAGH: 96.3% / 94.8%,
# 66th FARMA NO 66: 84% of the canonical) is an older draft.
PART_MAX_SHARE = 0.80


def superseded_label(r: str, canon: str, co_equal: bool, dch_r: int, dch_c: int) -> str:
    if is_superseded_dir(r) or is_backup(r):
        return 'from_superseded_revision'
    if FRONT_NAME_RE.search(os.path.basename(book_rel(r))):
        return 'front_matter'
    if co_equal or (dch_c and dch_r <= PART_MAX_SHARE * dch_c):
        return 'part_of_book'
    return 'from_superseded_revision'


# ==========================================================================
# units and records (L7 / L8; probe_books_layout_records.py)
# ==========================================================================


SKIP_CATS = ('residue', 'header_run', 'toc', 'imprint', 'fragment')
TITLE_MAX_LEN = 120
# stale titles (review 2026-09-25): a body frame whose TOC key equals another
# entry's key once digits are dropped ('۱۔ حمد تے نعت' against the entry 'حمد
# تے نعت' in 3rd) is a heading the exact match missed; it ends the running
# title (3rd: 'ہندکو چار بیتہ' ran on over the numbered sections).
_TOC_NODIGIT = re.compile('[0-9]+')
TOC_LOOSE_MIN = 4
# a speaker / label waits for its text frame at most this many raw frames;
# 25th's last headword 'کرنا:' (#73858) was prefixed to a font fragment
# 54,700 frames on ('کرنا: Nark'; review 2026-09-25)
PENDING_MAX_GAP = 8


def toc_title_hits(lines: List[str], cat: List[str], removed: Dict[int, dict],
                   toc_titles, order) -> Counter:
    """Number of body frames of a file (text, not removed, <= 120 chars) per
    TOC key (L8 ambiguity test). process_books adds these up over all
    (non-twin) files of the folder, since the TOC entries are pooled per
    folder (review 2026-09-25: 133 titles matched frames in two files -
    118th canonical and Old 110, 40th FEHRIST and body 10, 24th 'پرچھاواں')."""
    tset = toc_titles if isinstance(toc_titles, (set, frozenset)) else {
        e for e in toc_titles if len(e) >= 3}
    hits: Counter = Counter()
    if not tset:
        return hits
    for i in order:
        if cat[i] == 'text' and i not in removed and len(lines[i]) <= TITLE_MAX_LEN:
            nt = toc_key(lines[i])
            if nt in tset:
                hits[nt] += 1
    return hits


def _starts_with_label(t: str) -> bool:
    """The frame opens with a language-label word ('اُردو /معنی'): a column
    head, so a preceding label is not prefixed to it."""
    w = t.split()
    return bool(w) and w[0].strip('،,/:;-') in LANG_LABELS


def build_units(lines: List[str], cat: List[str], removed: Dict[int, dict], is_lexicon: bool,
                toc_titles, order: Optional[List[int]] = None, hits: Optional[Counter] = None,
                bylines=frozenset(), label_ok=None):
    """Units = surviving text frames between structural boundaries, in frame
    order (probe_books_layout_records.build_units).

    L6: separators and numbering are boundaries; genre headings and language
    labels are boundaries AND the first line of the unit they open (review
    2026-09-25: their text was carried only as genre_label / language_label;
    a heading line keeps its unit across a following boundary). A genre label
    holds until the next genre heading; a language label only for the block it
    opens - separators, numbering, genre headings and titles reset it (one
    'اُردو ترجمہ:' frame labelled 662,723 chars of 99th, one 'پنجابی' column
    head 229,466 chars of 45th). A speaker label is prefixed to the next text
    frame ('name: line') when that frame is at most 8 raw frames on. A joiner
    (lone ':' / '=' between two text frames, L2) writes 'a: b' / 'a = b' as
    one line: the extra frames are returned in `joined` (anchor frame ->
    frames). In lexicon files there are no passage units; a language label is
    prefixed like a speaker (131st parallel اُردو / ھندکو cells keep their
    labels) and genre words are text (process_books: dictionary headwords
    'ماہیا:' / 'دوہڑا:' / 'گیت:'). A label that is followed by another label or
    by a frame that opens with a language word (208th's column heads 'ہندکو'
    / 'انگریزی' / 'اُردو /معنی') or by nothing is kept as its own line; a
    label whose text line was removed as a copy is returned in label_drops
    (process_books writes its audit row) when `label_ok(label, line)` says the
    kept copy carries the same label, else kept as its own line.

    L8 (BOOKS_SPEC_CORRECTIONS): a frame (<= 120 chars) matching a TOC entry
    of the book folder (toc_key) opens a new section and becomes its `title` -
    unless that entry matches more than one kept frame of the FOLDER (`hits`;
    the title would be a guess). An entry that matches several frames ends the
    current title instead of letting it run on (53rd). Matches are accepted in
    increasing BODY position; there is no TOC-order pointer (166th: 25 chapter
    headings were rejected by it; two-column TOCs interleave rows, 343rd). A
    TOC-matched ghazal first misra (verse, >= 20 chars) is also the unit's
    first line (17th: 73 of 95 contents entries are first misras); a prose
    heading is consumed as the title only (title_only) and the title carries
    over a following boundary to the next non-empty unit (154th). A heading
    left without text (two titles in a row, or at the end) is kept as a line;
    a byline candidate (`bylines`) right after a title-only heading is a line
    of that unit, not a new title. English frames (R8) form their own units
    (records with language 'en')."""
    idx = range(len(lines)) if order is None else order
    tset = {e for e in toc_titles if len(e) >= 3} if not is_lexicon else set()
    tloose = {x for x in (_TOC_NODIGIT.sub('', e) for e in tset) if len(x) >= TOC_LOOSE_MIN}
    if hits is None:
        hits = toc_title_hits(lines, cat, removed, tset, idx) if tset else Counter()
    units: List[dict] = []
    eng_units: List[dict] = []
    title_only: List[int] = []
    label_drops: List[Tuple[int, int]] = []
    joined: Dict[int, List[int]] = {}
    st = {'section': 0, 'lang': None, 'genre': None}

    def new_unit():
        return {'lines': [], 'idx': [], 'title': None, 'title_i': None, 'lang': st['lang'],
                'genre': st['genre'], 'section': st['section'], 'n_head': 0}
    cur = new_unit()
    eng = None
    pending = None                      # (label text, frame index)
    pjoin = None                        # (joiner char, frame index, previous frame is the last line)
    last_i = -10

    def flush_pending():
        """A label with no text line of its own is kept as a line."""
        nonlocal pending, last_i
        if pending is not None:
            cur['lines'].append(lines[pending[1]].strip())
            cur['idx'].append(pending[1])
            last_i = pending[1]
            pending = None

    def flush_join():
        """A joiner whose next frame does not join: a suffix of the previous line."""
        nonlocal pjoin
        if pjoin is not None:
            ch, ji, ok = pjoin
            if ok and cur['lines']:
                cur['lines'][-1] = cur['lines'][-1] + (' =' if ch == '=' else ch)
                joined.setdefault(cur['idx'][-1], []).append(ji)
            pjoin = None

    def untitle(u):
        """An empty titled unit about to be replaced: its heading becomes a line."""
        if u['title_i'] is not None and not u['lines']:
            title_only.remove(u['title_i'])
            u['lines'].append(u['title'])
            u['idx'].append(u['title_i'])

    def heading_only(u):
        return bool(u['lines']) and u['n_head'] == len(u['lines'])

    for i in idx:
        t = lines[i]
        k = cat[i]
        if k in SKIP_CATS:
            continue
        if k == 'english_text':
            if i in removed:
                continue
            if eng is None:
                eng = {'lines': [], 'idx': [], 'title': None, 'lang': None, 'genre': None, 'section': 0}
            eng['lines'].append(t)
            eng['idx'].append(i)
            continue
        if eng is not None:
            eng_units.append(eng)
            eng = None
        if k == 'joiner':
            flush_join()
            pjoin = (t.strip(), i, bool(cur['lines']) and last_i == i - 1)
            continue
        if k in ('separator', 'numbering', 'genre_heading', 'language_label') and not is_lexicon:
            flush_join()
            flush_pending()
            if k in ('separator', 'numbering'):
                if cur['lines'] and not heading_only(cur):
                    units.append(cur)
                    st['lang'] = None
                    cur = new_unit()
                continue                # a heading / TOC title carries over the boundary
            if k == 'genre_heading':
                st['genre'] = label_key(t)
                if not heading_only(cur):
                    st['lang'] = None
            else:
                st['lang'] = label_key(t)
            if cur['lines'] and not heading_only(cur):
                units.append(cur)
                cur = new_unit()
            cur['lang'], cur['genre'] = st['lang'], st['genre']
            cur['lines'].append(t.strip())
            cur['idx'].append(i)
            cur['n_head'] += 1
            last_i = i
            continue
        if k in ('separator', 'numbering', 'genre_heading'):
            continue                    # lexicon (genre words were turned into text)
        if k in ('speaker', 'language_label'):
            flush_join()
            flush_pending()
            pending = (t.rstrip(':۔- ').strip() if k == 'speaker' else label_key(t), i)
            continue
        # text
        if pending is not None and i - pending[1] > PENDING_MAX_GAP:
            flush_pending()
        if i in removed:
            if pjoin is not None:
                ch, ji, ok = pjoin
                pjoin = None
                if ok and cur['lines']:
                    cur['lines'][-1] = cur['lines'][-1] + (' =' if ch == '=' else ch)
                    joined.setdefault(cur['idx'][-1], []).append(ji)
            if pending is not None:
                if label_ok is None or label_ok(pending[1], i):
                    label_drops.append((pending[1], i))
                    pending = None
                else:
                    flush_pending()     # the kept copy lacks the label: keep it
            continue
        if tset and len(t) <= TITLE_MAX_LEN:
            nt = toc_key(t)
            if nt in tset:
                nh = hits.get(nt, 0)
                if nh == 1 and not (nt in bylines and cur['title_i'] is not None and not cur['lines']):
                    flush_join()
                    flush_pending()
                    carry_l, carry_i = [], []
                    if cur['lines']:
                        if heading_only(cur) and cur['title_i'] is None:
                            carry_l, carry_i = cur['lines'], cur['idx']     # headings go with the title
                        else:
                            units.append(cur)
                        cur = new_unit()
                    prev = cur
                    st['section'] += 1
                    st['lang'] = None
                    cur = new_unit()
                    if prev['title_i'] is not None:        # two headings in a row
                        untitle(prev)
                        cur['lines'].extend(prev['lines'])
                        cur['idx'].extend(prev['idx'])
                    cur['lines'].extend(carry_l)
                    cur['idx'].extend(carry_i)
                    cur['n_head'] = len(cur['lines'])
                    cur['title'] = t
                    if is_verse(t) and len(t) >= 20:
                        cur['lines'].append(t)
                        cur['idx'].append(i)
                    else:
                        cur['title_i'] = i
                        title_only.append(i)
                    last_i = i
                    continue
                if nh > 1 and cur['title'] is not None:
                    # an ambiguous TOC heading: the current title ends here
                    flush_join()
                    flush_pending()
                    untitle(cur)
                    if cur['lines']:
                        units.append(cur)
                    st['section'] += 1
                    st['lang'] = None
                    cur = new_unit()
            elif cur['title'] is not None and tloose:
                lk = _TOC_NODIGIT.sub('', nt)
                if lk in tloose and lk != _TOC_NODIGIT.sub('', toc_key(cur['title'])):
                    # a heading that matches another TOC entry only without its
                    # numbering ('۱۔ حمد تے نعت' / 'حمد تے نعت'): not a title,
                    # but the current title ends here (stale titles)
                    flush_join()
                    flush_pending()
                    untitle(cur)
                    if cur['lines']:
                        units.append(cur)
                    st['section'] += 1
                    st['lang'] = None
                    cur = new_unit()
        if pending is not None:
            if _starts_with_label(t):
                flush_pending()         # a row of column heads (208th: ہندکو / انگریزی / اُردو /معنی)
            else:
                t = '%s: %s' % (pending[0], t)
                pending = None
        if pjoin is not None:
            ch, ji, ok = pjoin
            pjoin = None
            if ok and cur['lines']:
                cur['lines'][-1] = cur['lines'][-1] + (': ' if ch == ':' else ' = ') + t
                joined.setdefault(cur['idx'][-1], []).extend((ji, i))
                last_i = i
                continue
            t = ch + ' ' + t
            joined.setdefault(i, []).append(ji)
        cur['lines'].append(t)
        cur['idx'].append(i)
        last_i = i
    flush_join()
    flush_pending()
    untitle(cur)
    if cur['lines']:
        units.append(cur)
    if eng is not None:
        eng_units.append(eng)
    return units, eng_units, title_only, label_drops, joined


def unit_kind(lines: List[str]) -> str:
    ch = sum(len(t) for t in lines)
    v = sum(len(t) for t in lines if is_verse(t))
    return 'verse' if ch and v / ch >= 0.6 else 'prose'


def split_long_frame(t: str, max_chars: int = MAX_CHARS) -> List[str]:
    """L7(4): a single frame > 6,000 chars is split at sentence ends (۔).
    A piece with no ۔ inside stays whole (the 'single-frame' exception)."""
    if len(t) <= max_chars:
        return [t]
    parts = re.split(r'(?<=۔)', t)
    pieces, cur = [], ''
    for p in parts:
        if cur and len(cur) + len(p) > max_chars:
            pieces.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        pieces.append(cur)
    return [p.strip() for p in pieces if p.strip()]


def _new_rec(items, kind, u, split):
    return {'lines': [t for t, _ in items], 'idx': [ix for _, ix in items], 'kind': kind,
            'title': u['title'], 'n_units': 1, 'section': u['section'], 'lang': u['lang'],
            'genre': u['genre'], 'split': split, 'unit_breaks': []}


def _rec_len(r) -> int:
    return sum(len(t) for t in r['lines']) + max(0, len(r['lines']) - 1) + 2 * len(r['unit_breaks'])


# L7 verse splitting (BOOKS_SPEC_CORRECTIONS; review 2026-09-25): couplet
# parity is counted from the first verse line after a title / numbering /
# heading line and restarts after every stanza heading ('حرفی نمبر:- 25',
# 'موضوع: ...', a line ending ':'), and a unit over 6,000 chars is cut at a
# stanza boundary within the tolerance before parity is used: 106th VIRSA's
# harfi was cut inside a couplet because the count ran from 'ختم شد' / the
# author heading / a bio paragraph at the unit start (6 of 20 split verse
# units were misaligned that way).
STANZA_HEAD_RE = re.compile(r'(?:نمبر|نمبرشمار)\s*[:\-–۔]*\s*[0-9۰-۹]+\s*$'
                            r'|^\s*(?:موضوع|عنوان)\s*:|:\s*$')
SPLIT_TOLERANCE = 1.2
SPLIT_MIN_SHARE = 0.5


def _is_stanza_head(t: str) -> bool:
    return not is_verse(t) or bool(STANZA_HEAD_RE.search(t))


def _verse_cuts(items: List[Tuple[str, int]]) -> List[List[Tuple[str, int]]]:
    """Chunks of a verse unit: cut before a stanza head (preferred, the nearest
    to 6,000 chars within [0.5, 1.2] x 6,000), else after an even number of
    verse lines since the last head, else (past 1.2 x 6,000) anywhere."""
    n = len(items)
    head = [_is_stanza_head(t) for t, _ in items]
    # even[k]: a cut before item k keeps couplets whole
    even, cnt = [False] * (n + 1), 0
    for k in range(n):
        # never directly after a head (it would end the previous chunk)
        even[k] = cnt % 2 == 0 and not (k > 0 and head[k - 1])
        cnt = 0 if head[k] else cnt + 1
    even[n] = True
    out, start = [], 0
    while start < n:
        clen, k = 0, start
        while k < n and (k == start or clen + len(items[k][0]) + 1 <= MAX_CHARS):
            clen += len(items[k][0]) + 1
            k += 1
        if k >= n:
            out.append(items[start:])
            break
        # candidates: stanza heads in (start, stop] with the chunk length in range
        lo_len, hi_len = SPLIT_MIN_SHARE * MAX_CHARS, SPLIT_TOLERANCE * MAX_CHARS
        pref = [0]
        for x in range(start, n):
            pref.append(pref[-1] + len(items[x][0]) + 1)
            if pref[-1] > hi_len:
                break
        best = None
        for c in range(start + 1, start + len(pref) - 1):
            L_ = pref[c - start]
            # before the FIRST of a run of heads ('حرفی نمبر:- 24' / 'موضوع:
            # فریاد' stay together with their stanza)
            if head[c] and not head[c - 1] and lo_len <= L_ <= hi_len:
                d = abs(L_ - MAX_CHARS)
                if best is None or d < best[0]:
                    best = (d, c)
        if best is None:
            for c in range(k, start + len(pref) - 1):          # forward: first even cut within tolerance
                if even[c] and pref[c - start] <= hi_len:
                    best = (0, c)
                    break
        if best is None:
            for c in range(k - 1, start, -1):                  # backward: last even cut
                if even[c] and pref[c - start] >= lo_len:
                    best = (0, c)
                    break
        cut = best[1] if best is not None else k
        out.append(items[start:cut])
        start = cut
    return out


def make_records(units: List[dict]) -> List[dict]:
    """L7: split units > 6,000 chars at frame boundaries (verse: at a stanza
    boundary or after a complete couplet, _verse_cuts); merge units < 200
    chars forward into following units of the same kind / section / language
    label / genre (never across a TOC-titled unit), then append remaining
    short records to the previous one under the same constraints. Merged
    units are separated by a blank line. Simulation: 10,392 units (49.2% <
    200 chars) -> 8,616 records, 12.8% of them < 200 chars = 0.51% of chars;
    56 records > 6,000 because one frame is (34th, median frame 848 chars) -
    hence split_long_frame."""
    recs = []
    buf = None
    for u in units:
        kind = unit_kind(u['lines'])
        items = []
        for t, ix in zip(u['lines'], u['idx']):
            for piece in split_long_frame(t):
                items.append((piece, ix))
        total = sum(len(t) for t, _ in items) + len(items) - 1
        if total > MAX_CHARS:
            # verse cuts also for a 'prose' unit whose LINES are mostly verse
            # (a poet's bio paragraph before 110 chaarbita lines, 3rd: the
            # char share said prose and the plain cut split couplets)
            if kind == 'verse' or 2 * sum(1 for t, _ in items if is_verse(t)) >= len(items):
                chunks = _verse_cuts(items)
            else:
                chunks, chunk, clen = [], [], 0
                for t, ix in items:
                    if chunk and clen + len(t) > MAX_CHARS:
                        chunks.append(chunk)
                        chunk, clen = [], 0
                    chunk.append((t, ix))
                    clen += len(t) + 1
                if chunk:
                    chunks.append(chunk)
            for chunk in chunks:
                recs.append(_new_rec(chunk, kind, u, True))
            buf = None
            continue
        if buf is not None:
            same = (buf['kind'] == kind and buf['section'] == u['section'] and buf['lang'] == u['lang']
                    and buf['genre'] == u['genre'] and u['title'] is None)
            if same and _rec_len(buf) < MIN_CHARS:
                buf['unit_breaks'].append(len(buf['lines']))
                buf['lines'].extend(t for t, _ in items)
                buf['idx'].extend(ix for _, ix in items)
                buf['n_units'] += 1
                continue
        buf = _new_rec(items, kind, u, False)
        recs.append(buf)
    out = []
    for r in recs:
        if (out and _rec_len(r) < MIN_CHARS and not r['split'] and r['title'] is None
                and not out[-1]['split'] and out[-1]['kind'] == r['kind']
                and out[-1]['section'] == r['section'] and out[-1]['lang'] == r['lang']
                and out[-1]['genre'] == r['genre']):
            p = out[-1]
            base = len(p['lines'])
            p['unit_breaks'].append(base)
            p['unit_breaks'].extend(base + b for b in r['unit_breaks'])
            p['lines'].extend(r['lines'])
            p['idx'].extend(r['idx'])
            p['n_units'] += r['n_units']
            continue
        out.append(r)
    return out


HEADWORD_MAX = 40


def _is_headword(t: str) -> bool:
    return len(t.strip()) <= HEADWORD_MAX and t.rstrip().endswith((':', ':۔'))


def lexicon_chunks(items: List[Tuple[str, int]]) -> List[dict]:
    """L12: reference works are chunked into 2-6k-char records at frame
    boundaries, in order, never merged with prose (entries are short, so
    passage units are meaningless there). A chunk never ends on a headword
    (<= 40 chars ending ':'): 33 of 604 boundaries in the Hindko Lughat parts
    separated a headword from its gloss ('کڈھ:' | '(کَڈھ) کڈھنا(نکالنا...')."""
    out, cur, clen = [], [], 0
    for t0, ix in items:
        for t in split_long_frame(t0):
            add = len(t) + (1 if cur else 0)
            if cur and clen + add > MAX_CHARS:
                carry = [cur.pop()] if len(cur) > 1 and _is_headword(cur[-1][0]) else []
                out.append(cur)
                cur = carry
                clen = sum(len(x) for x, _ in carry)
                add = len(t) + (1 if cur else 0)
            cur.append((t, ix))
            clen += add
    if cur:
        out.append(cur)
    return [{'lines': [t for t, _ in c], 'idx': [ix for _, ix in c], 'kind': 'lexicon', 'title': None,
             'n_units': 1, 'section': 0, 'lang': None, 'genre': None, 'split': len(out) > 1,
             'unit_breaks': []} for c in out]


def _rec_text_lines(r) -> List[str]:
    breaks = set(r['unit_breaks'])
    out = []
    for k, t in enumerate(r['lines']):
        if k in breaks and k > 0:
            out.append('')
        out.append(t)
    return out


# ==========================================================================
# main entry point
# ==========================================================================
# Imprint fields travel as TWO bundles (L10, BOOKS_SPEC_CORRECTIONS 'model
# editions PER FILE'): the work bundle (title and the credits - author,
# translator, compiler, each with its label and whether the imprint or the
# title page states it) and the edition bundle (edition, year, date,
# first-edition year, ISBN + checksum, GHA reference, and - review
# 2026-09-25 - the publisher, which belongs to an edition: 358th). A file
# with its own imprint uses only that imprint for a bundle it states.
# Otherwise the edition bundle is taken WHOLE from the one canonical imprint
# of the file's WORK whose bundle contains every other canonical imprint's
# statements (none when they disagree -> imprint_conflict), and each work
# field from the canonical imprints of the work when all that state it
# agree. Field-by-field fallback of edition fields across files made records
# that no imprint states: 4th's 'PATTHAR DA JIGAR FEHRIST NEW SIZE' (March
# 2016, ISBN ...546, no edition) got edition 2 and first edition 2015 from
# the other file's 2019 second-edition imprint. Every record names the file
# of each value (imprint_field_sources).
IMPRINT_WORK_FIELDS = ('book_title', 'author', 'translator', 'compiler')
IMPRINT_EDITION_FIELDS = ('publication_year', 'publication_date', 'publication_date_precision',
                          'edition', 'first_edition_year', 'isbn', 'gha_ref', 'publisher')
IMPRINT_FIELDS = IMPRINT_WORK_FIELDS + IMPRINT_EDITION_FIELDS
_BUNDLE_EXTRA = {'book_title': ('book_title_from',),
                 'author': ('author_label', 'author_from'),
                 'translator': ('translator_label', 'translator_from'),
                 'compiler': ('compiler_label', 'compiler_from'),
                 'isbn': ('isbn_check_digit_ok',)}
CONFLICT_FIELDS = ('book_title', 'author', 'translator', 'compiler', 'publisher', 'publication_year',
                   'edition', 'isbn', 'gha_ref')
# Works (review 2026-09-25, blocker): folder consensus copied the imprint of
# '000 FEHRIST MATTLAN.INP' - the front matter of the 99th PROVERBS book
# 'ہندکو مٚتلاں' (ISBN 978-969-687-083-8) - onto the three Hindko Lughat parts
# in the same working folder (812 lexicon records, 24% of all book chars),
# a different work whose own title page reads 'گندھارا ہندکو لُغت', while the
# 99th proverbs book itself got nothing. A file now inherits imprint values
# only from files of the same WORK: all files of a numbered Farma folder
# (one Farma number = one book), the members of one revision cluster (the
# 98th '- Copy' folder file, the MATTLAN copy in the working folder), and in
# a working folder the files whose names share a distinctive word
# ('000 BOOK MATTLAN' / '000 FEHRIST MATTLAN'; 'Hindko Lughat Par-1' /
# 'Par-2' / 'Part-3').
GENERIC_FILE_WORDS = frozenset({
    'FEHRIST', 'FEHREST', 'BOOK', 'NEW', 'SIZE', 'FINAL', 'FARMA', 'PRELIMINARY', 'PAGES', 'COPY',
    'OLD', 'DEL', 'CORRECTED', 'CORRECTION', 'FILE', 'DONE', 'LATEST', 'INP', 'HINDKO', 'PART',
    'PAR', 'THE', 'AND', 'WRITEUP', 'WRITE', 'INTRO', 'CONTENT', 'CONTENTS', 'FLAP', 'FLAPPER',
    'RECEIVED', 'FROM', 'SIZE', 'ARRANGE'})
URDU_SECTIONS_MIN = 0.10
# Folder-name fallback values (review 2026-09-25): working-copy and label
# tokens are not part of a title or name ('Elahi Bakhsh Akhtar Awan - Copy',
# 'EXCLUSSIVE Hindko Mattlan', '(SCANNED)', '(PP-112)', '(2nd Edition)',
# '... 4th Int Conference Mushaira', '... 2nd Int Hindko Conference 2012').
# The values are romanised by whoever named the folder: source
# 'folder_name_romanised'.
_FOLDER_JUNK = re.compile(r'\s*-\s*Copy\b|\bEXCLUS+IVE\b|\(\s*SCANNED\s*\)|\(\s*(?:FINAL-?)?PP-?\d+\s*\)'
                          r'|\(?\s*\d+(?:st|nd|rd|th)\s+Edition\s*\)?|\(\s*FINAL[^)]*\)', re.I)
_FOLDER_CONF_TAIL = re.compile(r'\s+\d+(?:st|nd|rd|th)\s+Int(?:ernational)?\b.*$', re.I)
FOLDER_NAME_SOURCE = 'folder_name_romanised'
# language statements (review 2026-09-25): the imprint's 'زبان:' pair (340th
# 'زبان:' / 'پہاڑی') or a title-page phrase '<language> زبان' (196th
# 'پوٹھوہاری زبان کی ضرب الامثال'). 'pothohari' stays reserved for the 25th
# folder (BOOKS_SPEC_CORRECTIONS, Language): elsewhere a Pothohari statement
# is a flag.
LANGUAGE_NAMES = {'پہاڑی': 'pahari', 'ہندکو': 'hindko', 'ھندکو': 'hindko', 'پوٹھوہاری': 'pothohari',
                  'اردو': 'urdu', 'پنجابی': 'punjabi', 'گوجری': 'gojri', 'پشتو': 'pashto'}
RE_LANG_STATEMENT = re.compile('(%s)\\s+زبان' % '|'.join(LANGUAGE_NAMES))
# record language (review 2026-09-25): record evidence first - a running-text
# record whose own marker score says Urdu is 'urdu', one with >= 50% of its
# chars in Arabic (Quranic) lines 'arabic' (26th) - then the book's own
# language statement, then the collection default ('hindko', source
# 'collection_default:Hindko Books' - a default, not evidence). Lexicon
# records keep the collection default (their Urdu glosses are flagged
# urdu_glosses).
ARABIC_RECORD_SHARE = 0.5
# F (review 2026-09-25): a line >= 80% covered by ONE earlier file is still
# kept when the rest is significant - a contiguous uncovered stretch of >= F_KEEP_RUN
# key chars or > F_KEEP_SHARE of its key (first 15 / 5%): 838 of 1,218 F rows held text that exists in
# no kept line (7,802 key chars; 306th's story opening, 213th's own book
# title in a shared preface template, 80th/Del draft wording). It is kept
# with the flag fuzzy_copy_of(<file>). A removed F line records its
# uncovered key chars and spans. Tightened (fix5): with 15 / 5% the removed
# lines still held 3,047 uncovered key chars in stretches of 6-14 key chars -
# whole words and names ('اتفاق نال', 'منشی گلان', 'چادراں جوریا' in the 80th
# drafts); only a stretch of <= 5 key chars (a spelling variant or a short
# particle) and <= 2% of the line is now 'trivial' enough to remove. A line whose digit runs are not in the
# matched kept lines (+-3) of that file is kept too (99 lines, e.g. 78th
# '1976ء کلیاں ہندکو اردو ...' against 45th's line without the year).
F_KEEP_RUN = 6
F_KEEP_SHARE = 0.02
F_DIGIT_WINDOW = 3
_DIGIT_RUN = re.compile(r'[0-9۰-۹٠-٩]+')
# B (review 2026-09-25): boilerplate needs >= 2 tokens and >= 4 letters - the
# one-word frame 'یا' ('or', 44 rows) was removed from sentences and verses as
# a 'copy' of the garbage fragment 'ی ا' - and a frame sitting between two
# text frames of >= 8 chars (that are not boilerplate themselves) is running
# text, never boilerplate. The kept first occurrence is chosen per
# whitespace-normalised raw text, not per key alone.
BOILER_MIN_TOKENS = 2
BOILER_MIN_LETTERS = 4
BOILER_SANDWICH_LEN = 8
# L13 (BOOKS_SPEC_CORRECTIONS): 'no_extractable_text' only when the decoded
# text is negligible against the file's payload - 289th SCANNED: 20 text
# chars in an 82 MB file (79.5 MB of picture streams). Short digital files
# are real text and are never flagged by frame count (53rd HARF E AWAL
# 6,566 chars, 66th zaheerudin 6,298, 346th Saray Hath, 325th, 57th WRITEUP
# ...), so the old 'little_decodable_text' flag is gone. When the caller
# passes no file size, < 200 text chars alone decides (the strict gate's
# minimum record length, clean.STRICT['min_chars']).
L13_MIN_TEXT_CHARS = clean.STRICT['min_chars']     # 200
L13_TEXT_PER_BYTE = 1e-5
# 250th and 255th are front matter only: the Drive source has no body file
# for them (only the one front-matter file + its TXT). Detected as a folder
# whose files hold an imprint or a contents table but < 3,000 text chars in
# all (250th 1,778, 255th 1,824; the next smallest book folder, 325th, holds
# 5,840): records are kept, flag 'body_missing_from_source'.
BODY_MISSING_MAX_CHARS = 3000
# X runs (R1): a short line (< 40 key chars) is removed only as part of a
# copied PASSAGE - a chain of >= 3 consecutive lines whose kept copies lie in
# ONE earlier file at increasing positions at most 3 lines apart. The old
# test only asked that each line's key was kept somewhere: 307th's 165 scene
# headers ('سین نمبر N' / 'وخت' / 'دن' / 'مقام' / 'آؤٹ ڈور') were removed
# against 4 unrelated books per header (25th's dictionary 'وخت', 10th's 'دن',
# 59th's 'مقام'), and 4,332 short lines (40,761 chars) outside lexicons had
# copies whose neighbours did not match.
XRUN_GAP = 3
XRUN_CAND_CAP = 256       # kept copies examined per line (earliest first)
_LOC_SHIFT = 24           # location = source id << 24 | position
# ... and, outside lexicons, a chain copied from ANOTHER book folder must
# carry at least one substantive line's worth of text (40 key chars). A
# one-source chain of shorter lines is a shared template, not a copied
# passage: after the one-source rule, 307th Ishq Pecha still lost 102 of its
# 165 scene headers ('Cut' / 'سین نمبر N' / 'وخت' / 'دن' / 'مقام', 17-25 key
# chars) to the same template in the 305th Qadam dramas, 242nd's review
# headers ('<book> / مترجم / <name>') matched other books' title pages, 131st
# weekday lists matched 44th (fix3/a06b_chains: 750 lines / 3,562 chars
# outside lexicons; every chain inspected was a template or a list). Chains
# within one folder (revisions, part files: 132 lines) and in lexicons (the
# corrections keep entries shared with 25th in 25th) are copies.
XRUN_MIN_CHARS = 40


def _unmapped_in(lines: List[str], cat: List[str], multi: List[int], category: str) -> int:
    """U+FFFD count in the multi-char frames of one category (located with
    one scan of the joined text instead of a count() per frame)."""
    joined = '\n'.join(lines)
    if inpage.UNMAPPED not in joined:
        return 0
    lens = np.fromiter(map(len, lines), dtype=np.int64, count=len(lines))
    starts = np.concatenate(([0], np.cumsum(lens + 1)))
    pos = np.array([m.start() for m in re.finditer(inpage.UNMAPPED, joined)], dtype=np.int64)
    line_of = np.searchsorted(starts, pos, side='right') - 1
    return sum(1 for i in line_of.tolist() if len(lines[i]) > 1 and cat[i] == category)


def _loc(rel: str, idx: int) -> str:
    return '%s#%d' % (rel, idx)


def _resolve_same_file_targets(rel: str, rem: Dict[int, dict]) -> None:
    """W1/W2/L4 point at the earlier copy in the same file. When that copy was
    itself removed (e.g. both halves of a doubled superseded draft are also in
    the canonical file), follow the chain so every audit row names a copy that
    is actually kept (or the newspaper record / file that holds it)."""
    for li in sorted(rem):
        d = rem[li]
        if d['duplicate_source'] != 'same_file':
            continue
        cur, seen = d, {li}
        while cur['duplicate_source'] == 'same_file':
            ix = int(cur['duplicate_of'].rpartition('#')[2])
            if ix in rem and ix not in seen:
                seen.add(ix)
                cur = rem[ix]
            else:
                break
        if cur is not d:
            d['duplicate_of'] = cur['duplicate_of']
            d['duplicate_source'] = cur['duplicate_source']
            d['coverage'] = cur['coverage']
            d['via'] = cur['reason']
            if 'copy_lines' in cur:         # via a W1_GAP line: a fuzzy copy of these lines
                d['copy_lines'] = cur['copy_lines']


def _chain_locs(cands: List[List[int]]) -> List[Optional[int]]:
    """For consecutive lines with the locations (source << 24 | position) of
    their kept copies, the location through which each line belongs to a
    chain of >= XRUN_MIN lines whose copies lie in ONE source at increasing
    positions <= XRUN_GAP apart (None: in no such chain). Ties -> the
    smallest location (earliest source)."""
    m = len(cands)
    fwd: List[Dict[int, int]] = []
    prev: Optional[Dict[int, int]] = None
    for t in range(m):
        d = {}
        for loc in cands[t]:
            best = 0
            if prev:
                for g in range(1, XRUN_GAP + 1):
                    v = prev.get(loc - g)
                    if v is not None and v > best:
                        best = v
            d[loc] = best + 1
        fwd.append(d)
        prev = d
    out: List[Optional[int]] = [None] * m
    nxt: Optional[Dict[int, int]] = None
    for t in range(m - 1, -1, -1):
        d = {}
        bestloc, bestlen = None, 0
        for loc in cands[t]:
            best = 0
            if nxt:
                for g in range(1, XRUN_GAP + 1):
                    v = nxt.get(loc + g)
                    if v is not None and v > best:
                        best = v
            d[loc] = best + 1
            tot = fwd[t][loc] + best
            if tot > bestlen or (tot == bestlen and loc < bestloc):
                bestlen, bestloc = tot, loc
        if bestlen >= XRUN_MIN:
            out[t] = bestloc
        nxt = d
    return out


def run_chain_members(cands: List[List[int]]) -> List[Optional[int]]:
    """_chain_locs over each maximal run of lines that have kept copies."""
    out: List[Optional[int]] = [None] * len(cands)
    i, n = 0, len(cands)
    while i < n:
        if not cands[i]:
            i += 1
            continue
        j = i
        while j < n and cands[j]:
            j += 1
        if j - i >= XRUN_MIN:
            out[i:j] = _chain_locs(cands[i:j])
        i = j
    return out


def drop_template_chains(chain: List[Optional[int]], klens: List[int], same_folder) -> List[Optional[int]]:
    """X runs: un-chain one-source segments whose lines hold < XRUN_MIN_CHARS
    key chars in all when the source is in another book folder
    (same_folder(fid) False). A segment = consecutive lines whose chain
    locations share the source and increase."""
    out = list(chain)
    t, n = 0, len(chain)
    while t < n:
        if chain[t] is None:
            t += 1
            continue
        fid = chain[t] >> _LOC_SHIFT
        u = t
        while (u + 1 < n and chain[u + 1] is not None and chain[u + 1] >> _LOC_SHIFT == fid
               and chain[u + 1] > chain[u]):
            u += 1
        if sum(klens[t:u + 1]) < XRUN_MIN_CHARS and not same_folder(fid):
            out[t:u + 1] = [None] * (u + 1 - t)
        t = u + 1
    return out


def _cands(d1, d2, k1, k2) -> List[int]:
    a = d1.get(k1)
    b = d2.get(k2)
    if a and b:
        return a[:XRUN_CAND_CAP] + b[:XRUN_CAND_CAP]
    if a:
        return a[:XRUN_CAND_CAP]
    if b:
        return b[:XRUN_CAND_CAP]
    return []


LEX_STYLE_WORD_RE = re.compile(r'^[A-Za-z]{3,12}$')


def _lexicon_categories(c: dict, lines: Optional[List[str]] = None) -> None:
    """Lexicon files (L12): genre words are dictionary headwords ('ماہیا:',
    'دوہڑا:', 'گیت:', 'ٹپہ:', 'نوحہ۔' in 25th and the Lughat parts - 11 were
    skipped as genre headings, leaving their glosses headless; a colon-ended
    genre word classified 'speaker' is one too), and English frames are the
    dictionary's own column (208th): all are text cells. Latin frames outside
    the text flow stay residue: turning every out-of-span 'word' frame of a
    lexicon into text put 101 InPage font-table fragments into lexicon
    records ('Sulu' x17, 'Macromed', 'Wingdings', 'apher 4.1 Na'; review
    2026-09-25) - classify_file's embedding test decides instead."""
    cat = c['cat']
    n = len(cat)
    for i in c['multi']:
        if cat[i] in ('genre_heading', 'english_text'):
            cat[i] = 'text'
        elif cat[i] == 'speaker' and lines is not None and label_key(lines[i]) in GENRE_HEADINGS:
            cat[i] = 'text'
        elif (cat[i] == 'residue' and lines is not None and 0 < i < n - 1
              and cat[i - 1] == 'text' and cat[i + 1] == 'text'
              and LEX_STYLE_WORD_RE.match(lines[i].strip())
              and lang.latin_frame_category(lines[i], 0, True) == 'font_style'):
            # one style-name word between two text cells is an English
            # column entry (208th 'Light', 'Medium', 'New'), not a font table
            cat[i] = 'text'


def clean_folder_value(v: Optional[str]) -> Optional[str]:
    """A folder-name title / author without working-copy and label tokens."""
    if not v:
        return None
    v = _FOLDER_JUNK.sub(' ', v)
    v = _FOLDER_CONF_TAIL.sub('', v)
    v = re.sub(r'\s+', ' ', v).strip(' -')
    return v or None


def work_tokens(rel: str) -> set:
    """Distinctive words of a file name (generic words such as FEHRIST / BOOK /
    NEW SIZE / FINAL removed): files of a working folder that share one are
    one work."""
    stem = os.path.splitext(os.path.basename(rel))[0]
    return {w for w in re.split(r'[^A-Za-z]+', stem.upper()) if len(w) >= 3} - GENERIC_FILE_WORDS


def _wsn(s: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s)).strip()


def _uncovered(klen: int, hit: np.ndarray):
    """Key positions not covered by any matched n-gram: (spans, n chars,
    longest run). hit[j]: the j-th n-gram (in order) was found."""
    if klen == 0:
        return [], 0, 0
    width = min(NGRAM, klen)
    idx = np.nonzero(hit)[0]
    diff = np.zeros(klen + 1, dtype=np.int64)
    np.add.at(diff, idx, 1)
    np.add.at(diff, np.minimum(idx + width, klen), -1)
    cov = np.cumsum(diff)[:klen] > 0
    spans, run, best, start = [], 0, 0, None
    for p_, c_ in enumerate(cov.tolist()):
        if not c_:
            if start is None:
                start = p_
            run += 1
            best = max(best, run)
        else:
            if start is not None:
                spans.append([start, p_])
            start, run = None, 0
    if start is not None:
        spans.append([start, klen])
    return spans, int((~cov).sum()), best


def _digit_runs(s: str) -> set:
    return {m.translate(_DIGIT_FOLD) for m in _DIGIT_RUN.findall(s)}


def _variant(a: str, b: str) -> Optional[str]:
    """How a removed line differs from the kept copy it names: None (same
    text), 'whitespace', 'punct' (punctuation / tatweel / yeh-kaf-heh
    spellings), 'harakat' (vowel marks the kept copy lacks or differs in),
    'digits', 'other'."""
    if a == b or _wsn(a) == _wsn(b):
        return None
    if re.sub(r'\s+', '', a) == re.sub(r'\s+', '', b):
        return 'whitespace'
    if dedup_hkey(a) == dedup_hkey(b):
        return 'punct'
    if dedup_key(a) == dedup_key(b):
        return 'harakat'
    if loose_key(a) == loose_key(b):
        return 'digits'
    return 'other'


def process_books(book_files: List[dict], newspaper_texts: List[tuple], log=print) -> dict:
    """See the module docstring. `book_files`: [{'rel', 'kind', 'text_all'}]
    (+ optional 'size' in bytes, used by the L13 payload test);
    `newspaper_texts`: [(newspaper_record_key, text)] of the newspaper records
    already kept in this build. R8 (BOOKS_SPEC_CORRECTIONS): book lines are NOT
    removed because they appear in the newspaper - book records stay whole and
    carry 'also_in_newspaper(share=x.xx)'; the build's record-level MinHash
    removes records that are near-identical to an article."""
    t0, c0 = time.time(), time.process_time()
    _log = log

    def log(msg):
        _log('%s  (%.1fs wall, %.1fs cpu)' % (msg, time.time() - t0, time.process_time() - c0))
    files = sorted({bf['rel'].replace('\\', '/'): bf for bf in book_files}.items())
    rels = [r for r, _ in files]
    kinds = {r: bf.get('kind') for r, bf in files}
    sizes = {r: bf.get('size') for r, bf in files}
    log('books: %d files' % len(rels))

    # .B01 twins (same dir + stem .inp present) are only passed with
    # --include-all-backups. They are excluded from every cross-file STATISTIC
    # (English repeat counts, boilerplate, cluster caps, TOC entries and TOC
    # title counts, imprint consensus, book form / genre / score) and
    # processed LAST, so the records of every other file do not depend on
    # that option (test C15).
    inp_stems = {(os.path.dirname(r), os.path.splitext(os.path.basename(r))[0].lower().strip())
                 for r in rels if r.lower().endswith('.inp')}
    twins = frozenset(r for r in rels if is_backup(r) and (
        os.path.dirname(r), os.path.splitext(os.path.basename(r))[0].lower().strip()) in inp_stems)

    # ---- R8 English: identical Latin frames across files ------------------
    split = {r: (bf['text_all'].split('\n') if bf['text_all'] else []) for r, bf in files}
    latin_counts: Counter = Counter()
    for r, _ in files:
        if r not in twins:
            latin_counts.update(_latin_candidates(split[r]))
    log('latin frame counts')

    # ---- 1. classification --------------------------------------------------
    L: Dict[str, dict] = {}
    for r, bf in files:
        lines = split.pop(r)
        c = classify_file(lines, latin_counts, kinds.get(r))
        c['n_lines'] = len(lines)
        c['role'] = file_role(r)
        cat = c['cat']
        texts = [lines[i] for i in c['multi'] if cat[i] in TEXTUAL]
        tch = sum(map(len, texts))
        c['prose_share'] = round(sum(len(t) for t in texts if len(t) >= LEX_PROSE_FRAME) / tch, 4) if tch else 0.0
        if c['role'] == 'lexicon':
            ev = LEXICON_PATH_RE.search(book_rel(r)).group(0)
            if c['prose_share'] >= LEX_PROSE_SHARE:
                # L12 per file: a path-word file of prose paragraphs (98th
                # '000 CONTENT.INP') is not a word list
                c['role'] = 'front_matter' if (c['tocs'] or c['imprints']) else 'main'
                c['role_evidence'] = ('path:%s but %.0f%% of its text chars in frames of >= %d chars'
                                      % (ev, 100 * c['prose_share'], LEX_PROSE_FRAME))
            else:
                c['lexicon_evidence'] = 'path:%s' % ev
        elif c['role'] == 'main':
            # L12 structural fallback (colon-headword test, see LEX_COLON_*)
            share, distinct, ordered = colon_headword_stats(texts)
            c['colon_stats'] = (round(share, 4), round(distinct, 4), round(ordered, 4))
            if (len(texts) >= LEX_MIN_FRAMES and share >= LEX_COLON_SHARE
                    and distinct >= LEX_COLON_DISTINCT and ordered >= LEX_COLON_SORTED):
                c['role'] = 'lexicon'
                c['lexicon_evidence'] = ('structure:colon_headwords(share=%.2f,distinct=%.2f,alphabetical=%.2f)'
                                         % (share, distinct, ordered))
        if c['role'] == 'lexicon':
            _lexicon_categories(c, lines)
        keyf = dedup_hkey if c['role'] == 'lexicon' else dedup_key
        seq = []
        for i in c['multi']:
            if cat[i] in TEXTUAL:
                t = lines[i]
                k = keyf(t)
                if k:
                    seq.append((i, t, k))
        c['seq'] = seq
        c['n_text_frames'] = len(seq)
        c['n_text_chars'] = sum(len(t) for _, t, _ in seq)
        if (c['role'] == 'main' and c['imprints'] and c['n_text_chars'] < BODY_MISSING_MAX_CHARS
                and not FRONT_NAME_RE.search(os.path.basename(r))):
            # a file that holds only an imprint and a title page / publisher's
            # note (250th TABEERAN, 255th SOO-E-HARMAIN, 4th 'Pathar da
            # Jigar.INP' - the 2nd-edition prelims) is front matter
            c['role'] = 'front_matter'
            c['role_evidence'] = 'imprint and < %d text chars' % BODY_MISSING_MAX_CHARS
        c['cat_lines'] = Counter(cat)
        cc = Counter()
        for i in c['multi']:
            cc[cat[i]] += len(lines[i])
        for k_, v_ in c['cat_lines'].items():     # single-char frames: 1 char each
            cc[k_] += v_
        for i in c['multi']:
            cc[cat[i]] -= 1
        c['cat_chars'] = +cc
        c['fffd_single'] = lines.count(inpage.UNMAPPED)
        c['fffd_residue'] = _unmapped_in(lines, cat, c['multi'], 'residue')
        # content-line stats for book form / genre (R9 used is_content_line lines)
        cl = [lines[i] for i in c['multi']
              if len(lines[i]) >= segment.MIN_CONTENT_LEN and is_content_line_fast(lines[i])]
        c['cl_n'] = len(cl)
        c['cl_short'] = sum(1 for t in cl if len(t) <= lang.SHORT_LINE_MAX)
        c['cl_chars'] = sum(len(t) for t in cl)
        c['front_text'] = '\n'.join(cl[:lang.FRONT_LINES])
        c['lines'] = lines
        L[r] = c
    log('classified frames')

    # ---- L13: files without extractable text; folders without their body ----
    no_text = set()
    for r in rels:
        c = L[r]
        sz = sizes.get(r)
        if c['n_text_chars'] < L13_MIN_TEXT_CHARS and (
                sz is None or c['n_text_chars'] < L13_TEXT_PER_BYTE * sz):
            no_text.add(r)
    by_folder: Dict[str, List[str]] = defaultdict(list)
    for r in rels:
        by_folder[folder_of(r)].append(r)
    folders = sorted(by_folder)
    body_missing = set()
    for fo in folders:
        mem = [r for r in by_folder[fo] if r not in twins and r not in no_text]
        if mem and sum(L[r]['n_text_chars'] for r in mem) < BODY_MISSING_MAX_CHARS and any(
                L[r]['imprints'] or L[r]['tocs'] for r in mem):
            body_missing.add(fo)

    # ---- 2. dedup ------------------------------------------------------------
    active = [r for r in rels if r not in no_text]
    seqs = {r: L[r]['seq'] for r in active}
    sup, clusters, sup_label = detect_clusters(active, seqs, twins)
    # a front-matter-named file that is the canonical copy of a cluster with a
    # main file holds the book (340th PRELIMINARY PAGES.INP: prelims + book):
    # its records are book passages
    line_in_canon: Dict[str, float] = {}
    for cl_ in clusters:
        cn = cl_['canonical']
        line_in_canon.update(cl_['line_in_canonical'])
        if L[cn]['role'] == 'front_matter' and any(
                L[m]['role'] == 'main' for m in cl_['members'] if m != cn):
            L[cn]['role'] = 'main'
            L[cn]['role_evidence'] = 'canonical copy of a cluster with a main file'
    order = sorted(active, key=lambda r: (r in twins, r in sup, order_no(r), r))
    order_pos = {r: k for k, r in enumerate(order)}
    log('revision clusters: %d, superseded files: %d' % (len(clusters), len(sup)))

    # B: boilerplate lines (<= 60 chars, >= 2 tokens and >= 4 letters, in
    # >= 25% of book folders)
    fcount: Counter = Counter()
    first_len: Dict[str, int] = {}
    first_text: Dict[str, str] = {}
    by_folder_keys: Dict[str, set] = defaultdict(set)
    for r in active:
        if r in twins:
            continue
        fkeys = by_folder_keys[folder_of(r)]
        for _, t, k in seqs[r]:
            fkeys.add(k)
            if k not in first_len:
                first_len[k], first_text[k] = len(t), t
    for fo in sorted(by_folder_keys):
        fcount.update(by_folder_keys[fo])
    thresh = max(BOILER_MIN_DOCS, BOILER_FRACTION * len(folders))
    boiler = {k for k, c in fcount.items() if c >= thresh and first_len[k] <= BOILER_MAX_LEN
              and len(first_text[k].split()) >= BOILER_MIN_TOKENS
              and len(AR_LETTER.findall(first_text[k])) >= BOILER_MIN_LETTERS}
    del by_folder_keys, first_text

    # R8: the newspaper is NOT part of the line-level passes; its keys are
    # only used for the per-record also_in_newspaper share (section 5)
    news_K: Dict[str, List[int]] = defaultdict(list)
    news_H: Dict[str, List[int]] = defaultdict(list)
    for ri, (nkey, text) in enumerate(newspaper_texts):
        for pos, ln in enumerate(text.split('\n')):
            k = dedup_key(ln)
            if k:
                news_K[k].append((ri << _LOC_SHIFT) | pos)
                news_H[dedup_hkey(ln)].append((ri << _LOC_SHIFT) | pos)
    log('newspaper: %d records, %d keys (for the R8 overlap share only)' % (
        len(newspaper_texts), len(news_K)))

    # kept book lines: key -> locations (fid << 24 | seq position), by the
    # role of the file holding them. Copy identity: dedup_hkey equal, or
    # dedup_key equal when neither line is in a lexicon.
    K_nonlex: Dict[str, List[int]] = defaultdict(list)
    H_nonlex: Dict[str, List[int]] = defaultdict(list)
    H_lex: Dict[str, List[int]] = defaultdict(list)
    index_k = _GramIndex()          # 10-grams of dedup_key of kept lines (all roles)
    index_h = _GramIndex()          # 10-grams of dedup_hkey (queried by lexicon lines)
    fid_path: Dict[int, str] = {}
    fid_of: Dict[str, int] = {}
    seq_lines: Dict[int, List[int]] = {}
    line_grams_cache: Dict[Tuple[int, bool], tuple] = {}

    def where(loc: int) -> str:
        fid, pos = loc >> _LOC_SHIFT, loc & ((1 << _LOC_SHIFT) - 1)
        return _loc(fid_path[fid], seq_lines[fid][pos])

    def line_grams(fid: int, lex_key: bool):
        """(sorted 10-gram hashes, their kept-line seq positions) of the kept
        lines of an already processed file (F digit test)."""
        kk = (fid, lex_key)
        if kk not in line_grams_cache:
            rr = fid_path[fid]
            rem_ = removed[rr]
            kf = dedup_hkey if lex_key else dedup_key
            hs, ps = [], []
            for pos, (li, t, _) in enumerate(L[rr]['seq']):
                if li in rem_:
                    continue
                g = _uniq(ngram_hashes(kf(t)))
                hs.append(g)
                ps.append(np.full(len(g), pos, dtype=np.int64))
            if hs:
                H_ = np.concatenate(hs)
                P_ = np.concatenate(ps)
                o = np.argsort(H_, kind='stable')
                line_grams_cache[kk] = (H_[o], P_[o])
            else:
                line_grams_cache[kk] = (np.zeros(0, np.uint64), np.zeros(0, np.int64))
        return line_grams_cache[kk]

    removed: Dict[str, Dict[int, dict]] = {r: {} for r in rels}
    boiler_first: Dict[Tuple[str, str], str] = {}
    doubled = set()
    w1_chars: Dict[str, int] = {}
    for fpos, r in enumerate(order, 1):
        fid_path[fpos] = r
        fid_of[r] = fpos
        c = L[r]
        seq = c['seq']
        cat = c['cat']
        lines = c['lines']
        seq_lines[fpos] = [li for li, _, _ in seq]
        is_lex = c['role'] == 'lexicon'
        ks = [k for _, _, k in seq]
        if is_lex:
            hks = ks
            kks = [dedup_key(t) for _, t, _ in seq]
        else:
            kks = ks
            hks = [dedup_hkey(t) for _, t, _ in seq]
        lens = [len(t) for _, t, _ in seq]
        w1, w2, chorus, runs, w1gap, w1orph = within_file_masks(ks, lens)
        c['w1_orphan'] = {seq[g][0]: (seq[b_][0], cv) for g, (b_, cv) in w1orph.items()}
        rem = removed[r]
        # whole-book doubling (L4 via W1): flag, and drop second-copy lines
        # whose twin in the first copy was structural (title page, imprint...)
        w1c = sum(lens[t] for t in range(len(seq)) if w1[t] is not None)
        w1_chars[r] = w1c
        total_c = sum(lens)
        l4 = {}
        if total_c and w1c >= DOUBLED_SHARE * total_c:
            doubled.add(r)
            long_runs = [i for i, j, n_ in runs if i - j >= len(seq) / 4]
            if long_runs:
                s_line = seq[min(long_runs)][0]
                keyf = dedup_hkey if is_lex else dedup_key
                first_keys: Dict[str, int] = {}
                for i in range(s_line):
                    if cat[i] in ('residue', 'fragment') or len(lines[i]) < 2:
                        continue
                    k = keyf(lines[i])
                    if k:
                        first_keys.setdefault(k, i)
                for t, (i, txt, k) in enumerate(seq):
                    if i >= s_line and w1[t] is None and w2[t] is None and k in first_keys:
                        l4[t] = first_keys[k]
        # X: kept copies in earlier files
        if is_lex:
            cands = [_cands(H_nonlex, H_lex, hks[t], hks[t]) for t in range(len(seq))]
        else:
            cands = [_cands(K_nonlex, H_lex, kks[t], hks[t]) for t in range(len(seq))]
        chain = run_chain_members(cands)
        if not is_lex:
            fo_r = folder_of(r)
            chain = drop_template_chains(chain, [len(k) for k in ks],
                                         lambda fid: folder_of(fid_path[fid]) == fo_r)
        # X in superseded files (review 2026-09-25, R7): a superseded file's
        # line that is in a run of >= 3 consecutive already-seen lines (copies
        # in ANY earlier kept set) is removed, and in a near-identical draft
        # (>= 90% of its lines in the canonical) so is every line kept in its
        # own canonical: the byte-identical 98th '- Copy' kept 2,321 lines
        # (106,880 chars) that its canonical keeps, because the canonical's
        # neighbouring lines had gone to the Lughat and no one-source chain
        # formed; 130,534 chars of such copies in all superseded files.
        sup_rule = [None] * len(seq)
        canon_fid = fid_of.get(sup.get(r)) if r in sup else None
        if r in sup:
            near = line_in_canon.get(r, 0.0) >= 0.9
            t = 0
            while t < len(seq):
                if not cands[t]:
                    t += 1
                    continue
                u = t
                while u < len(seq) and cands[u]:
                    u += 1
                for x in range(t, u):
                    in_canon = canon_fid is not None and any(l >> _LOC_SHIFT == canon_fid for l in cands[x])
                    if near and in_canon:
                        sup_rule[x] = 'superseded_canonical'
                    elif u - t >= XRUN_MIN:
                        sup_rule[x] = 'superseded_run'
                t = u
        subst = [t for t in range(len(seq)) if len(ks[t]) >= SUBST]
        grams_of = dict(zip(subst, ngram_hashes_many([ks[t] for t in subst])))
        index = index_h if is_lex else index_k
        fuzzy_cand = []
        kept_t = []
        c['fuzzy_kept'] = {}
        for t, (li, txt, k) in enumerate(seq):
            n_c = len(txt)
            if w1[t] is not None:
                rem[li] = {'reason': 'W1', 'duplicate_of': _loc(r, seq[w1[t]][0]), 'coverage': 1.0,
                           'n_chars': n_c, 'duplicate_source': 'same_file'}
                continue
            if t in w1gap:
                tw, cov, win = w1gap[t]
                rem[li] = {'reason': 'W1_GAP', 'duplicate_of': _loc(r, seq[tw][0]), 'coverage': cov,
                           'copy_lines': [seq[x][0] for x in win],
                           'n_chars': n_c, 'duplicate_source': 'same_file'}
                continue
            if w2[t] is not None:
                rem[li] = {'reason': 'W2', 'duplicate_of': _loc(r, seq[w2[t]][0]), 'coverage': 1.0,
                           'n_chars': n_c, 'duplicate_source': 'same_file'}
                continue
            if t in l4:
                rem[li] = {'reason': 'L4', 'duplicate_of': _loc(r, l4[t]), 'coverage': 1.0,
                           'n_chars': n_c, 'duplicate_source': 'same_file'}
                continue
            if k in boiler:
                bk = (k, _wsn(txt))
                sandwiched = all(
                    0 <= x < len(lines) and cat[x] == 'text' and len(lines[x].strip()) >= BOILER_SANDWICH_LEN
                    and (dedup_hkey if is_lex else dedup_key)(lines[x]) not in boiler
                    for x in (li - 1, li + 1))
                if bk in boiler_first and not sandwiched:
                    rem[li] = {'reason': 'B', 'duplicate_of': boiler_first[bk], 'coverage': 1.0,
                               'n_chars': n_c, 'duplicate_source': 'book'}
                    continue
            if cands[t] and (len(k) >= SUBST or chain[t] is not None or sup_rule[t] is not None):
                if chain[t] is not None:
                    loc, rule = chain[t], 'chain'
                elif sup_rule[t] is not None:
                    inc = [l for l in cands[t] if canon_fid is not None and l >> _LOC_SHIFT == canon_fid]
                    loc, rule = (min(inc) if inc else min(cands[t])), sup_rule[t]
                else:
                    loc, rule = min(cands[t]), 'substantive'
                rem[li] = {'reason': 'X', 'duplicate_of': where(loc), 'coverage': 1.0,
                           'n_chars': n_c, 'duplicate_source': 'book',
                           'x_run': chain[t] is not None, 'x_rule': rule}
                continue
            if not cands[t] and len(k) >= SUBST and index.levels:
                fuzzy_cand.append(t)
            kept_t.append(t)
        # F: fuzzy copies (>= 80% of distinct 10-grams in ONE earlier file)
        f_drop = {}
        if fuzzy_cand:
            hs = [_uniq(grams_of[t]) for t in fuzzy_cand]
            H = np.concatenate(hs)
            owner = np.repeat(np.arange(len(fuzzy_cand)), [len(h) for h in hs])
            nper = np.array([len(h) for h in hs], dtype=np.float64)
            hit = index.any_hit(H)
            ucov = np.bincount(owner, weights=hit, minlength=len(fuzzy_cand)) / nper
            sel = np.nonzero(ucov >= FUZZ)[0]
            if len(sel):
                m = np.isin(owner, sel)
                Hm = H[m]
                own, fids, rows = index.pairs(Hm, owner[m])
                if len(own):
                    key = own.astype(np.int64) * (len(order) + 1) + fids
                    uk, cnts = np.unique(key, return_counts=True)
                    best: Dict[int, Tuple[float, int]] = {}
                    for kk, cc in zip(uk.tolist(), cnts.tolist()):
                        o_, f_ = divmod(kk, len(order) + 1)
                        cov = cc / nper[o_]
                        b = best.get(o_)
                        if b is None or cov > b[0] or (cov == b[0] and f_ < b[1]):
                            best[o_] = (cov, f_)
                    for o_ in sorted(best):
                        cov, f_ = best[o_]
                        if cov < FUZZ:
                            continue
                        t = fuzzy_cand[o_]
                        li, txt, k = seq[t]
                        matched = np.unique(Hm[rows[(own == o_) & (fids == f_)]])
                        g_all = grams_of[t]
                        spans, unc, run_ = _uncovered(len(k), np.isin(g_all, matched))
                        why = None
                        if run_ >= F_KEEP_RUN or unc > F_KEEP_SHARE * len(k):
                            why = 'uncovered_text'
                        else:
                            dr = _digit_runs(txt)
                            if dr:
                                gh, gp = line_grams(f_, is_lex)
                                lo = np.searchsorted(gh, matched, 'left')
                                hi = np.searchsorted(gh, matched, 'right')
                                poss = set()
                                for a_, b_ in zip(lo.tolist(), hi.tolist()):
                                    poss.update(gp[a_:b_].tolist())
                                tseq = L[fid_path[f_]]['seq']
                                tl = L[fid_path[f_]]['lines']
                                win = set()
                                for pp in poss:
                                    for q in range(max(0, pp - F_DIGIT_WINDOW), min(len(tseq), pp + F_DIGIT_WINDOW + 1)):
                                        win |= _digit_runs(tl[tseq[q][0]])
                                if not dr <= win:
                                    why = 'digits'
                        if why is not None:
                            c['fuzzy_kept'][li] = (fid_path[f_], round(float(cov), 4), why)
                            continue
                        f_drop[t] = {'reason': 'F', 'duplicate_of': fid_path[f_],
                                     'coverage': round(float(cov), 4),
                                     'n_chars': len(txt), 'duplicate_source': 'book',
                                     'uncovered_chars': unc, 'uncovered_key_spans': spans,
                                     'uncovered_key_text': [k[a_:b_] for a_, b_ in spans]}
        kept_k_hashes, kept_h_hashes = [], []
        for t in kept_t:
            li, txt, k = seq[t]
            if t in f_drop:
                rem[li] = f_drop[t]
                continue
            loc = (fpos << _LOC_SHIFT) | t
            if is_lex:
                H_lex[hks[t]].append(loc)
            else:
                K_nonlex[kks[t]].append(loc)
                H_nonlex[hks[t]].append(loc)
            if k in boiler:
                bk = (k, _wsn(txt))
                if bk not in boiler_first:
                    boiler_first[bk] = _loc(r, li)     # B keeps the first KEPT occurrence of that text
            if len(k) >= SUBST:
                g = grams_of[t]
                if is_lex:
                    kept_h_hashes.append(g)
                    kept_k_hashes.append(g if kks[t] == hks[t] else ngram_hashes(kks[t]))
                else:
                    kept_k_hashes.append(g)
                    kept_h_hashes.append(g if kks[t] == hks[t] else ngram_hashes(hks[t]))
        if kept_k_hashes:
            index_k.add(np.concatenate(kept_k_hashes), fpos)
            index_h.add(np.concatenate(kept_h_hashes), fpos)
        _resolve_same_file_targets(r, rem)
        c['chorus_chars'] = sum(lens[t] for t in range(len(seq)) if chorus[t] and w1[t] is None)
    log('line dedup done')
    del index_k, index_h, K_nonlex, H_nonlex, H_lex
    line_grams_cache.clear()

    # ---- 3. folder-level evidence: TOCs, imprints, works, genre, form ---------
    # TOC entries per folder (a set of toc_key: L8 matches in body order, no
    # pointer); byline candidates (second cells of page-less / paged pairs in
    # blocks that are not heading / byline TOCs)
    toc_entries: Dict[str, set] = {}
    toc_bylines: Dict[str, set] = {}
    for fo in folders:
        es, bs = set(), set()
        for r in by_folder[fo]:
            if r in twins:
                continue
            for tb in L[r]['tocs']:
                es.update(toc_key(t) for _, t, _ in tb['entries'])
                bs.update(toc_key(t) for t in tb.get('byline_candidates', ()))
        toc_entries[fo] = es
        toc_bylines[fo] = bs
    # L8: TOC title matches are counted over the kept frames of ALL files of
    # the folder (the entries are pooled per folder)
    folder_hits: Dict[str, Counter] = {}
    for fo in folders:
        h = Counter()
        tset = {e for e in toc_entries[fo] if len(e) >= 3}
        if tset:
            for r in by_folder[fo]:
                if r in twins or r in no_text or L[r]['role'] == 'lexicon':
                    continue
                c = L[r]
                h.update(toc_title_hits(c['lines'], c['cat'], removed[r], tset,
                                        sorted(c['multi'] + c['single_num'])))
        folder_hits[fo] = h
    # TOC blocks are deduplicated in PROCESSING order (a .inp before its .B01)
    toc_first: Dict[str, str] = {}
    toc_copy_of: Dict[Tuple[str, int], str] = {}
    for r in order:
        for tb in L[r]['tocs']:
            sig = '\n'.join(toc_key(t) for _, t, _ in tb['entries'])
            if sig in toc_first:
                toc_copy_of[(r, tb['marker_idx'])] = toc_first[sig]
            else:
                toc_first[sig] = _loc(r, tb['marker_idx'])

    imprints_out = []
    file_imprint: Dict[str, dict] = {}
    isbn_folders: Dict[str, set] = defaultdict(set)
    for r in rels:
        ws = L[r]['imprints']
        if not ws:
            continue
        parsed = [parse_imprint(w) for w in ws]
        p0 = parsed[0]
        conflicts = []
        for p_ in parsed[1:]:
            for f in CONFLICT_FIELDS:
                if p_[f] is not None and p0[f] is not None and p_[f] != p0[f]:
                    conflicts.append(f)
        file_imprint[r] = p0
        if r not in twins:
            for p_ in parsed:
                if p_['isbn']:
                    isbn_folders[p_['isbn']].add(folder_of(r))
        lines = L[r]['lines']
        imprints_out.append({
            'source_path': r, 'book_folder': folder_of(r), 'n_windows': len(ws),
            'windows': [{'anchor_line': w['anchor_line'], 'end_line': w['end_line'],
                         'anchor_kind': w['anchor_kind'], 'rights_line': w['rights_line'],
                         'pairs': [{'field': p_['field'], 'label': p_['label'], 'value': p_['value'],
                                    'label_line': p_['label_line'], 'value_line': p_['value_line']}
                                   for p_ in w['pairs']],
                         'title_page_pairs': [{'field': p_['field'], 'label': p_['label'], 'value': p_['value'],
                                               'label_line': p_['label_line'], 'value_line': p_['value_line']}
                                              for p_ in w.get('title_page_pairs', ())],
                         'inline_isbn': [x for x, _ in w['inline_isbn']],
                         # every frame of the window as written (credits, addresses)
                         'frames': [[i, lines[i]] for i in w['frame_lines']]} for w in ws],
            'parsed': p0, 'parsed_other_windows': parsed[1:],
            'windows_disagree_on': sorted(set(conflicts)),
        })
    isbn_shared = {isbn: sorted(fos) for isbn, fos in isbn_folders.items() if len(fos) > 1}

    def bundle_of(p_: dict, fields) -> Optional[tuple]:
        vals = tuple(p_[f] for f in fields)
        return vals if any(v is not None for v in vals) else None

    # works (see GENERIC_FILE_WORDS): union-find over the files
    wpar = {r: r for r in rels}

    def wfind(x):
        while wpar[x] != x:
            wpar[x] = wpar[wpar[x]]
            x = wpar[x]
        return x

    def wunion(a, b):
        ra, rb = wfind(a), wfind(b)
        if ra != rb:
            wpar[max(ra, rb)] = min(ra, rb)
    for fo in folders:
        mem = sorted(by_folder[fo])
        if parse_folder_name(fo)['series'] is not None:
            for r in mem[1:]:
                wunion(mem[0], r)
        else:
            tok = {r: work_tokens(r) for r in mem}
            for a_i, a in enumerate(mem):
                for b in mem[a_i + 1:]:
                    if tok[a] & tok[b]:
                        wunion(a, b)
    for cl_ in clusters:
        for m_ in cl_['members']:
            wunion(cl_['canonical'], m_)
    work_members: Dict[str, List[str]] = defaultdict(list)
    for r in rels:
        work_members[wfind(r)].append(r)

    def work_consensus(members: List[str]) -> dict:
        imp_files = [r for r in sorted(members, key=lambda x: (x in twins, x in sup or is_superseded_dir(x), x))
                     if r in file_imprint]
        prim = ([r for r in imp_files if r not in sup and not is_superseded_dir(r) and r not in twins]
                or [r for r in imp_files if r not in twins] or imp_files)
        values, value_src, conflicts = {}, {}, {}
        for f in IMPRINT_WORK_FIELDS:
            vals = []
            for r in prim:
                v = file_imprint[r][f]
                if v is not None and v not in vals:
                    vals.append(v)
            if len(vals) == 1:
                src = [r for r in prim if file_imprint[r][f] == vals[0]][0]
                values[f], value_src[f] = vals[0], src
                for x in _BUNDLE_EXTRA.get(f, ()):
                    values[x] = file_imprint[src][x]
            elif len(vals) > 1:
                conflicts[f] = vals
        bundles = []
        for r in prim:
            bd = bundle_of(file_imprint[r], IMPRINT_EDITION_FIELDS)
            if bd is not None and bd not in [x for x, _ in bundles]:
                bundles.append((bd, r))
        # one edition bundle for the work: the bundle that contains every
        # other canonical imprint's statements (identical, or a superset of
        # them); imprints that disagree -> no bundle, imprint_conflict
        dom = [(bd, src) for bd, src in bundles
               if all(all(x is None or x == y for x, y in zip(ob, bd)) for ob, _ in bundles)]
        for k_, f in enumerate(IMPRINT_EDITION_FIELDS):
            vs = []
            for bd, _ in bundles:
                if bd[k_] is not None and bd[k_] not in vs:
                    vs.append(bd[k_])
            if len(vs) > 1:
                conflicts[f] = vs
        if dom:
            bd, src = dom[0]
            for f, v in zip(IMPRINT_EDITION_FIELDS, bd):
                if v is not None:
                    values[f], value_src[f] = v, src
            values['isbn_check_digit_ok'] = file_imprint[src]['isbn_check_digit_ok']
        subj = [file_imprint[r]['subject'] for r in prim if file_imprint[r]['subject']]
        lang_st = [file_imprint[r]['language_statement'] for r in prim if file_imprint[r]['language_statement']]
        return {'imprint_files': imp_files, 'values': values, 'value_src': value_src,
                'conflicts': {f: v for f, v in conflicts.items() if f in CONFLICT_FIELDS},
                'subject': subj[0] if subj else None,
                'language_statement': lang_st[0] if lang_st else None}
    work_meta = {w: work_consensus(m) for w, m in work_members.items()}
    work_of = {r: wfind(r) for r in rels}

    book_meta: Dict[str, dict] = {}
    for fo in folders:
        members = sorted(by_folder[fo])
        stat_files = [r for r in members if not is_superseded_dir(r) and not is_copy_dir(r)
                      and r not in twins]
        n_cl = sum(L[r]['cl_n'] for r in stat_files)
        n_short = sum(L[r]['cl_short'] for r in stat_files)
        n_ch = sum(L[r]['cl_chars'] for r in stat_files)
        form = lang.book_form(n_cl, n_short, n_ch)
        front = '\n'.join(L[r]['front_text'] for r in stat_files)
        fnames = [book_rel(r)[len(fo) + 1:] for r in stat_files]
        # the folder's main work: the one holding most of its text
        wch = Counter()
        for r in members:
            wch[work_of[r]] += L[r]['n_text_chars']
        main_work = sorted(wch.items(), key=lambda z: (-z[1], z[0]))[0][0]
        wm = work_meta[main_work]
        genre, gev = lang.book_genre(fo, fnames, front, form, wm['subject'], wm['values'].get('book_title'))
        gnote = lang.genre_conflict(fo, fnames, front, form, wm['subject'], wm['values'].get('book_title'))
        pf = parse_folder_name(fo)
        poth = lang.pothohari_evidence(fo, (t for r in members for t in L[r]['lines']
                                            if 'پوٹھوہار' in t))
        stmts = []
        for r in members:
            if r in twins:
                continue
            cands_ = list(L[r]['title_page'])
            for w in L[r]['imprints']:
                a_ = w['anchor_line']
                cands_ += [L[r]['lines'][x] for x in range(max(0, a_ - 30), a_)
                           if L[r]['cat'][x] == 'text']
            for t in cands_:
                for m_ in RE_LANG_STATEMENT.finditer(t):
                    if (m_.group(1), t) not in stmts:
                        stmts.append((m_.group(1), t))
        book_meta[fo] = {
            'book_folder': fo, 'files': members,
            'book_series_number': pf['series'],
            'book_series_number_source': 'folder_name' if pf['series'] is not None else None,
            'folder_title': clean_folder_value(pf['title']), 'folder_author': clean_folder_value(pf['author']),
            'form': form, 'form_basis': {'content_lines': n_cl, 'short_lines': n_short, 'chars': n_ch},
            'genre': genre, 'genre_evidence': gev, 'genre_note': gnote,
            'imprint_subject': wm['subject'],
            'imprint_files': wm['imprint_files'],
            'imprint_values': {f: wm['values'][f] for f in IMPRINT_FIELDS if f in wm['values']},
            'imprint_value_source_path': wm['value_src'],
            'imprint_bundle_extras': {x: wm['values'].get(x) for fs in _BUNDLE_EXTRA.values() for x in fs},
            'imprint_conflict': wm['conflicts'],
            'works': [{'files': sorted(work_members[w]), 'imprint_files': work_meta[w]['imprint_files'],
                       'imprint_values': {f: work_meta[w]['values'][f] for f in IMPRINT_FIELDS
                                          if f in work_meta[w]['values']},
                       'imprint_conflict': work_meta[w]['conflicts']}
                      for w in sorted({work_of[r] for r in members})],
            'language_statement': wm['language_statement'],
            'title_page_language_statements': [{'language': a, 'text': b} for a, b in stmts],
            'pothohari_evidence': poth,
            'running_headers': sorted({h['text'] for r in members for h in L[r]['headers']}),
            'body_missing_from_source': fo in body_missing,
        }

    # ---- 4. units / records ---------------------------------------------------
    def label_ok(r, label_i, line_i):
        """The kept copy of the removed line `line_i` carries the same label."""
        d = removed[r].get(line_i)
        if d is None or '#' not in d['duplicate_of']:
            return False
        q, _, j = d['duplicate_of'].rpartition('#')
        if q not in L or 'lines' not in L[q]:
            return False
        j = int(j)
        ql, qc = L[q]['lines'], L[q]['cat']
        lab = L[r]['lines'][label_i].strip().rstrip(':۔- ').strip()
        for x in range(j - 1, max(-1, j - 1 - PENDING_MAX_GAP), -1):
            if qc[x] in ('residue', 'fragment'):
                continue
            return qc[x] in ('speaker', 'language_label') and ql[x].strip().rstrip(':۔- ').strip() == lab
        return False

    records = []
    per_file = []
    for r in rels:
        c = L[r]
        fo = folder_of(r)
        lines = c['lines']
        cat = c['cat']
        rem = removed[r]
        # final disposition per line: structural category, or for textual
        # lines dedup / record / toc_title / no_extractable_text
        disp: List[Optional[str]] = list(cat)
        textual = [i for i in c['multi'] if cat[i] in TEXTUAL]
        for i in textual:
            disp[i] = ('no_extractable_text' if r in no_text
                       else 'dedup' if i in rem else 'unassigned')
        file_recs = []
        tocs_out = []
        title_only = []
        joined = {}
        if r not in no_text:
            role = c['role']
            for tb in c['tocs']:
                tocs_out.append({'marker_idx': tb['marker_idx'], 'end_idx': tb['end_idx'],
                                 'entries': list(tb['entries']), 'frames': list(tb['frames']),
                                 'bylines': dict(tb.get('bylines', {}))})
            is_lex = role == 'lexicon'
            units, eng_units, title_only, label_drops, joined = build_units(
                lines, cat, rem, is_lex, toc_entries.get(fo, set()),
                sorted(c['multi'] + c['single_num']), hits=folder_hits.get(fo),
                bylines=toc_bylines.get(fo, frozenset()),
                label_ok=lambda li, ti, r=r: label_ok(r, li, ti))
            for i in title_only:
                disp[i] = 'toc_title'
            # a speaker / language label whose text line was removed as a copy
            # whose kept copy carries the same label: the label goes with its line
            for li, ti in label_drops:
                d = rem[ti]
                rem[li] = {'reason': 'LABEL_OF_REMOVED_LINE', 'duplicate_of': d['duplicate_of'],
                           'coverage': d['coverage'], 'n_chars': len(lines[li]),
                           'duplicate_source': d['duplicate_source'], 'via': d['reason'],
                           'label_of_line': ti}
                disp[li] = 'dedup'
            if is_lex:
                items = [(t, ix) for u in units for t, ix in zip(u['lines'], u['idx'])]
                recs = lexicon_chunks(items)
                role_of_rec = 'lexicon'
            else:
                recs = make_records(units)
                role_of_rec = 'front_matter' if role == 'front_matter' else 'book_passage'
            for rr in recs:
                rr['role'] = role_of_rec
            erecs = make_records(eng_units)
            for rr in erecs:
                rr['role'] = 'english_text'
            file_recs = recs + erecs
            for rr in file_recs:
                rr['frames'] = sorted(set(rr['idx']) | {x for ix in rr['idx'] for x in joined.get(ix, ())})
                for ix in rr['frames']:
                    disp[ix] = 'record'
            # record frames that are not dedup text lines (heading lines,
            # labels kept as a line, joiners): the C16 / C17 replays skip them
            c['label_lines'] = sorted({ix for rr in file_recs for ix in rr['frames']
                                       if cat[ix] not in TEXTUAL})
        c['disp'] = disp
        c['file_recs'] = file_recs
        c['tocs_out'] = tocs_out
        c['title_only'] = title_only
        c['joined'] = joined

    log('units and records built')

    # ---- 5. record metadata + flags ---------------------------------------
    # folder-level whole-token Hindko score (R3/R5), on kept running text of
    # non-superseded files, Arabic/Persian-flagged lines excluded
    line_cache: Dict[str, tuple] = {}

    def line_info(t):
        """(arabic, persian, hindko hits, urdu hits, latin letters, latin +
        Arabic-script letters) of one record line; computed once per text."""
        v = line_cache.get(t)
        if v is None:
            ar = lang.is_arabic_line(t)
            fa = (not ar) and lang.is_persian_line(t)
            h, u = lang.marker_counts(t)
            lat = len(lang.LATIN_RE.findall(t))
            both = lat + len(lang.AR_LETTER_RE.findall(t))
            v = (ar, fa, h, u, lat, both)
            line_cache[t] = v
        return v

    # R3 book flags: urdu_dominant_book if the book score <= 0.30;
    # urdu_sections(x%) if >= 10% of the chars are in Urdu-labelled records
    # (records play the role of the probe's ~2k-char windows).
    folder_hu: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for r in rels:
        c = L[r]
        if r in sup or is_superseded_dir(r) or is_copy_dir(r) or r in twins:
            continue
        for rr in c['file_recs']:
            if rr['role'] not in ('book_passage', 'front_matter'):
                continue
            hh = uu = nch = 0
            for t in rr['lines']:
                ar, fa, h, u, _, _ = line_info(t)
                if ar or fa:
                    continue
                hh += h
                uu += u
                nch += len(t)
            acc = folder_hu[folder_of(r)]
            acc[0] += hh
            acc[1] += uu
            acc[2] += nch
            if hh + uu and hh / (hh + uu) <= lang.URDU_MAX:
                acc[3] += nch
    for fo in folders:
        hh, uu, nch, uch = folder_hu.get(fo, [0, 0, 0, 0])
        score = hh / (hh + uu) if hh + uu else None
        book_meta[fo]['hindko_score_v2'] = round(score, 4) if score is not None else None
        book_meta[fo]['urdu_record_char_share'] = round(uch / nch, 4) if nch else None
        book_meta[fo]['urdu_dominant_book'] = score is not None and score <= lang.URDU_MAX
        book_meta[fo]['urdu_sections'] = (not book_meta[fo]['urdu_dominant_book'] and bool(nch)
                                          and uch / nch >= URDU_SECTIONS_MIN)

    news_overlap: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    line_dedup = []
    for r in rels:
        c = L[r]
        fo = folder_of(r)
        bm = book_meta[fo]
        wm = work_meta[work_of[r]]
        own = file_imprint.get(r)
        pf_title, pf_author = bm['folder_title'], bm['folder_author']
        conflict = bool(wm['conflicts'])
        lines = c['lines']
        is_lex_file = c['role'] == 'lexicon'
        keyf = dedup_hkey if is_lex_file else dedup_key
        news_idx = news_H if is_lex_file else news_K

        # imprint bundles: own imprint first, else the consensus of the work
        md = {f: None for f in IMPRINT_FIELDS}
        md_src: Dict[str, str] = {}
        extras = {x: None for fs in _BUNDLE_EXTRA.values() for x in fs}
        for fields in (IMPRINT_WORK_FIELDS, IMPRINT_EDITION_FIELDS):
            if own is not None and bundle_of(own, fields) is not None:
                for f in fields:
                    if own[f] is not None:
                        md[f], md_src[f] = own[f], r
                        for x in _BUNDLE_EXTRA.get(f, ()):
                            extras[x] = own[x]
            else:
                for f in fields:
                    if f in wm['values'] and f not in wm['conflicts']:
                        md[f] = wm['values'][f]
                        md_src[f] = wm['value_src'][f]
                        for x in _BUNDLE_EXTRA.get(f, ()):
                            extras[x] = wm['values'].get(x)
        title_src = None
        if md['book_title'] is not None:
            title_src = extras['book_title_from'] or 'imprint'
        elif pf_title:
            md['book_title'], title_src = pf_title, FOLDER_NAME_SOURCE
        credit_src = {}
        for role_ in CREDIT_ROLES:
            if md[role_] is not None:
                credit_src[role_] = extras[role_ + '_from'] or 'imprint'
        # the folder name's author only when neither the imprint nor the title
        # page credits anyone (review 2026-09-25: a stated translator /
        # compiler is not replaced by a romanised folder name)
        if md['author'] is None and not credit_src and pf_author:
            md['author'], credit_src['author'] = pf_author, FOLDER_NAME_SOURCE
        author_role = (extras['author_label'] or '').strip().rstrip(':۔- ').strip() or None \
            if credit_src.get('author') in ('imprint', 'title_page') else None
        poth = bm['pothohari_evidence']
        # genre is per book folder (R9), except the proverbs book: '000 BOOK
        # MATTLAN.INP' and its FEHRIST are the 99th PROVERBS book also where
        # a copy sits in the 'Dictionary with matlan adition' folder
        # (BOOKS_SPEC_CORRECTIONS L12); 'proverbs' is a genre that file names
        # may set (lang.FILE_GENRE_OK). A lexicon file in another genre's
        # folder is a dictionary by its path.
        genre, genre_ev = bm['genre'], bm['genre_evidence']
        m_prov = PROVERBS_FILE_RE.search(os.path.basename(r))
        if m_prov and genre != 'proverbs':
            genre, genre_ev = 'proverbs', 'file:%s' % m_prov.group(0)
        superseded = r in sup
        file_flags = []
        if superseded:
            file_flags.append(sup_label.get(r, 'from_superseded_revision'))
        if is_superseded_dir(r):
            file_flags.append('from_superseded_dir')
        if r in doubled:
            file_flags.append('inpage_duplicate_story_removed')
        if conflict:
            file_flags.append('imprint_conflict')
        if bm['body_missing_from_source']:
            file_flags.append('body_missing_from_source')
        if md['isbn'] is not None and extras['isbn_check_digit_ok'] is False:
            file_flags.append('isbn_invalid_checksum')
        if md['isbn'] is not None and md['isbn'] in isbn_shared:
            file_flags.append('isbn_shared_with(%s)' % '; '.join(
                x for x in isbn_shared[md['isbn']] if x != fo))
        for st_ in bm['title_page_language_statements']:
            if LANGUAGE_NAMES.get(st_['language']) not in ('hindko', None):
                fl = 'title_page_language(%s)' % st_['language']
                if fl not in file_flags:
                    file_flags.append(fl)
        stated_lang = LANGUAGE_NAMES.get(re.sub(r'[\s:۔]+', '', wm['language_statement'] or ''))
        span = c['span']
        joined = c['joined']
        fuzzy_kept = c.get('fuzzy_kept', {})
        w1_orphan = c.get('w1_orphan', {})

        recs_sorted = sorted(c['file_recs'], key=lambda x: (min(x['frames']), x['role']))
        toc_recs = []
        for tb in c['tocs_out']:
            blk = [i for i in tb['frames'] if c['cat'][i] == 'toc']
            copy_of = toc_copy_of.get((r, tb['marker_idx']))
            if copy_of is not None:
                for i in blk:
                    line_dedup.append({'source_path': r, 'line_index': i, 'reason': 'TOC_COPY',
                                       'duplicate_of': copy_of, 'coverage': 1.0,
                                       'n_chars': len(lines[i]), 'duplicate_source': 'book',
                                       'via': None})
                    c['disp'][i] = 'toc_copy'
                continue
            tl_ = []
            ents = []
            for ix, t, p_ in tb['entries']:
                tl_.append(t)
                e = {'line_index': ix, 'title': t, 'page': p_}
                if ix in tb['bylines']:
                    bl, bt = tb['bylines'][ix]
                    e['byline'], e['byline_line_index'] = bt, bl
                    tl_.append(bt)
                ents.append(e)
            toc_recs.append({'lines': tl_, 'idx': [], 'frames': [], 'kind': 'toc', 'title': None, 'n_units': 1,
                             'role': 'toc', 'split': False, 'unit_breaks': [], 'lang': None, 'genre': None,
                             'toc_entries': ents,
                             'toc_block_lines': [tb['marker_idx'], tb['end_idx']], '_first': tb['marker_idx']})
        allrecs = [(min(x['frames']), 1, x) for x in recs_sorted] + [(x['_first'], 0, x) for x in toc_recs]
        allrecs.sort(key=lambda z: (z[0], z[1]))
        for n_, (_, _, rr) in enumerate(allrecs, 1):
            text_lines = _rec_text_lines(rr)
            text = '\n'.join(text_lines)
            role = rr['role']
            flags = list(file_flags)
            ar_ch = fa_ch = hh = uu = lat = both = 0
            lat_dom = False
            is_running = role in ('book_passage', 'front_matter')
            for t in rr['lines']:
                if role == 'toc':
                    break
                ar, fa, h, u, la, bo = line_info(t)
                lat += la
                both += bo
                if la and la >= 0.5 * bo:
                    lat_dom = True
                if is_running and ar:          # rule v4 / v2 not applied to lexicons (R6)
                    ar_ch += len(t)
                elif is_running and fa:
                    fa_ch += len(t)
                else:
                    hh += h
                    uu += u
            tot = sum(len(t) for t in rr['lines']) or 1
            ar_share = round(ar_ch / tot, 4)
            fa_share = round(fa_ch / tot, 4)
            if ar_share >= 0.30:
                flags.append('arabic_dominant')
            elif ar_ch:
                flags.append('contains_arabic(%.2f)' % ar_share)
            if fa_ch:
                flags.append('persian_passage')
            hs = hh / (hh + uu) if hh + uu else 0.5
            if hh + uu == 0:
                variety = 'no_signal'
            else:
                variety = ('hindko' if hs >= lang.HINDKO_MIN else 'urdu' if hs <= lang.URDU_MAX else 'mixed')
            # record labels override the book label (BOOKS_SPEC_CORRECTIONS,
            # Language; review 2026-09-25): the book-level Urdu flags go only
            # on records without their own contrary evidence - never on
            # Hindko or mixed records (urdu_dominant_book) / Hindko records
            # (urdu_sections), nor on Arabic-dominant records (26th)
            if is_running and ar_share < 0.30:
                if bm['urdu_dominant_book'] and variety in ('urdu', 'no_signal'):
                    flags.append('urdu_dominant_book')
                elif bm['urdu_sections'] and variety in ('urdu', 'mixed', 'no_signal'):
                    flags.append('urdu_sections(%.0f%%)' % (100 * bm['urdu_record_char_share']))
            if role == 'lexicon' and hh + uu and hs <= lang.URDU_MAX:
                flags.append('urdu_glosses')
            if lat_dom and role != 'english_text':
                flags.append('english_dominant_line')
            if len(text) < MIN_CHARS and role != 'toc':
                flags.append('short_unit')
            frames = rr['frames']
            if frames and (span is None or frames[0] < span[0] or frames[-1] > span[1]):
                flags.append('outside_main_span')
            fz = sorted({fuzzy_kept[i][0] for i in frames if i in fuzzy_kept})
            for p_ in fz:
                flags.append('fuzzy_copy_of(%s)' % p_)
            n_orph = sum(1 for i in frames if i in w1_orphan)
            if n_orph:
                flags.append('w1_orphan_lines(%d)' % n_orph)
            # R8: share of the record's chars in lines that are exactly in
            # kept newspaper text (substantive lines, or a >= 3-line passage
            # copied from one newspaper record)
            if frames and newspaper_texts:
                fk = [keyf(lines[i]) for i in frames]
                nc = [news_idx.get(k_, [])[:XRUN_CAND_CAP] if k_ else [] for k_ in fk]
                nch_ = run_chain_members(nc)
                tot_c = sum(len(lines[i]) for i in frames)
                hit_c = sum(len(lines[i]) for q, i in enumerate(frames)
                            if nc[q] and (len(fk[q]) >= SUBST or nch_[q] is not None))
                if tot_c and hit_c:
                    flags.append('also_in_newspaper(share=%.2f)' % (hit_c / tot_c))
                if role != 'toc':
                    acc = news_overlap[fo]
                    acc[0] += hit_c
                    acc[1] += tot_c
            if role == 'english_text':
                language, lsrc = 'en', 'r8_english_frame'
            elif poth:
                language, lsrc = 'pothohari', poth
            elif is_running and ar_share >= ARABIC_RECORD_SHARE:
                language, lsrc = 'arabic', 'record:arabic_share=%.2f' % ar_share
            elif is_running and variety == 'urdu':
                language, lsrc = 'urdu', 'record:hindko_score_v2=%.2f' % hs
            elif stated_lang and stated_lang not in ('pothohari',):
                language, lsrc = stated_lang, 'imprint:زبان=%s' % wm['language_statement'].strip(' :۔')
            else:
                # no record / book evidence: the collection's default, marked
                # as a default (not as evidence about this record)
                language, lsrc = 'hindko', 'collection_default:Hindko Books'
            form = {'verse': 'verse', 'prose': 'prose', 'lexicon': 'lexicon', 'toc': 'mixed'}[rr['kind']]
            if form in ('verse', 'prose') and _is_list(rr['lines']):
                form = 'list'
            elif form == 'prose' and bm['form'] == 'verse' and _verse_like_long(rr['lines']):
                form = 'verse'
            rec = {
                'source_path': r, 'source_file': os.path.basename(r), 'passage_index': n_,
                'text': text, 'lines': text_lines, 'frame_indices': frames,
                'n_chars': len(text),
                'role': role, 'form': form, 'n_units': rr['n_units'], 'split_part': rr['split'],
                'title': rr['title'], 'title_source': 'toc_entry' if rr['title'] else None,
                'heading_raw': None,
                'genre_label': rr.get('genre'), 'language_label': rr.get('lang'),
                'book_folder': fo,
                'book_series_number': bm['book_series_number'],
                'book_series_number_source': bm['book_series_number_source'],
                'book_title': md['book_title'], 'book_title_source': title_src,
                'author': md['author'], 'author_source': credit_src.get('author'), 'author_role': author_role,
                'translator': md['translator'], 'translator_source': credit_src.get('translator'),
                'translator_label': extras['translator_label'] if md['translator'] is not None else None,
                'compiler': md['compiler'], 'compiler_source': credit_src.get('compiler'),
                'compiler_label': extras['compiler_label'] if md['compiler'] is not None else None,
                'publisher': md['publisher'], 'publisher_source': 'imprint' if md['publisher'] is not None else None,
                'publication_year': md['publication_year'],
                'publication_date': md['publication_date'],
                'publication_date_precision': md['publication_date_precision'],
                'publication_year_source': 'imprint' if md['publication_year'] is not None else None,
                'edition': md['edition'], 'edition_source': 'imprint' if md['edition'] is not None else None,
                'first_edition_year': md['first_edition_year'],
                'first_edition_year_source': 'imprint' if md['first_edition_year'] is not None else None,
                'isbn': md['isbn'], 'isbn_source': 'imprint' if md['isbn'] is not None else None,
                'gha_ref': md['gha_ref'], 'gha_ref_source': 'imprint' if md['gha_ref'] is not None else None,
                'imprint_field_sources': {f: md_src[f] for f in IMPRINT_FIELDS if f in md_src},
                'imprint_source_paths': sorted(set(md_src.values())),
                'imprint_conflict': conflict,
                'genre': genre, 'genre_evidence': genre_ev, 'book_form': bm['form'],
                'language': language, 'language_source': lsrc,
                'content_flags': flags,
                'arabic_share': ar_share, 'persian_share': fa_share,
                'superseded': superseded,
                'hindko_score_v2': round(hs, 4) if role not in ('english_text', 'toc') else None,
                'language_variety_v2': variety if role not in ('english_text', 'toc') else None,
                'latin_share': round(lat / both, 4) if both else 0.0,
                'unmapped_glyphs': text.count(inpage.UNMAPPED),
            }
            if fz:
                rec['fuzzy_copy_lines'] = [[i, fuzzy_kept[i][0], fuzzy_kept[i][1], fuzzy_kept[i][2]]
                                           for i in frames if i in fuzzy_kept]
            if role == 'toc':
                rec['toc_entries'] = rr['toc_entries']
                rec['toc_block_lines'] = rr['toc_block_lines']
            records.append(rec)
        # line_dedup rows for this file
        for i in sorted(removed[r]):
            d = removed[r][i]
            row = {'source_path': r, 'line_index': i, 'reason': d['reason'],
                   'duplicate_of': d['duplicate_of'], 'coverage': d['coverage'],
                   'n_chars': d['n_chars'], 'duplicate_source': d['duplicate_source'],
                   'via': d.get('via')}
            for x in ('label_of_line', 'copy_lines', 'x_rule', 'uncovered_chars', 'uncovered_key_spans',
                      'uncovered_key_text'):
                if x in d:
                    row[x] = d[x]
            if d['reason'] in ('W1', 'W2', 'X', 'B', 'L4') and '#' in d['duplicate_of']:
                q, _, j = d['duplicate_of'].rpartition('#')
                if q in L:
                    v = _variant(lines[i], L[q]['lines'][int(j)])
                    if v is not None:
                        row['variant'] = v
            line_dedup.append(row)
        # per-file stats
        disp_lines = Counter(c['disp'])
        dc = Counter()
        for i in c['multi']:
            dc[c['disp'][i]] += len(lines[i]) - 1
        disp_chars = +(dc + disp_lines)            # + 1 char per frame
        rem_by = defaultdict(lambda: {'lines': 0, 'chars': 0})
        for d in removed[r].values():
            rem_by[d['reason']]['lines'] += 1
            rem_by[d['reason']]['chars'] += d['n_chars']
        flags = list(file_flags)
        if r in no_text:
            flags.append('no_extractable_text')
        per_file.append({
            'source_path': r, 'kind': kinds.get(r), 'book_folder': fo, 'file_role': c['role'],
            'role_evidence': c.get('role_evidence'),
            'lexicon_evidence': c.get('lexicon_evidence'), 'colon_stats': c.get('colon_stats'),
            'prose_frame_share': c.get('prose_share'),
            'n_lines': c['n_lines'], 'size_bytes': sizes.get(r),
            'lines_by_category': dict(sorted(c['cat_lines'].items())),
            'chars_by_category': dict(sorted(c['cat_chars'].items())),
            'text_frames': c['n_text_frames'], 'text_chars': c['n_text_chars'],
            'disposition_lines': dict(sorted(disp_lines.items())),
            'disposition_chars': dict(sorted(disp_chars.items())),
            'removed': {k: dict(v) for k, v in sorted(rem_by.items())},
            'w1_chars': w1_chars.get(r, 0), 'chorus_candidate_chars_kept': c.get('chorus_chars', 0),
            'flags': flags,
            'superseded_by': sup.get(r), 'superseded_label': sup_label.get(r),
            'processing_order': order_pos.get(r), 'backup_twin': r in twins,
            'work_files': sorted(work_members[work_of[r]]),
            'main_span_lines': list(c['span']) if c['span'] else None,
            'running_headers': c['headers'], 'title_page_frames': c['title_page'],
            'toc_blocks': len(c['tocs']), 'imprint_windows': len(c['imprints']),
            'toc_title_lines': sorted(c.get('title_only', [])),
            'label_lines_in_records': c.get('label_lines', []),
            # isolated 2-4-letter fragments amid InPage residue: their text
            'fragments': c['fragments'],
            'joined_lines': {str(k): v for k, v in sorted(c.get('joined', {}).items())},
            'fuzzy_copies_kept': [[i, v[0], v[1], v[2]] for i, v in sorted(c.get('fuzzy_kept', {}).items())],
            'w1_orphan_lines': [[i, v[0], v[1]] for i, v in sorted(c.get('w1_orphan', {}).items())],
            'unmapped_in_single_char_frames': c['fffd_single'], 'unmapped_in_residue': c['fffd_residue'],
        })
    for r in rels:
        L[r].pop('lines', None)
    # fill record counts per file
    nrec = Counter(x['source_path'] for x in records)
    rch = Counter()
    for x in records:
        rch[x['source_path']] += x['n_chars']
    for pf in per_file:
        pf['records'] = nrec.get(pf['source_path'], 0)
        pf['record_chars'] = rch.get(pf['source_path'], 0)
        pf['conservation_ok'] = pf['disposition_lines'].get('unassigned', 0) == 0
    for fo in folders:
        hit_c, tot_c = news_overlap.get(fo, [0, 0])
        book_meta[fo]['newspaper_overlap_share'] = round(hit_c / tot_c, 4) if tot_c else None

    report = build_report(records, line_dedup, per_file, book_meta, imprints_out, clusters, sup,
                          boiler, newspaper_texts, doubled, no_text, body_missing)
    report['isbn_shared_between_folders'] = isbn_shared
    log('books: %d records, %d chars; %d removed lines' % (
        len(records), sum(x['n_chars'] for x in records), len(line_dedup)))
    return {'records': records, 'imprints': imprints_out, 'line_dedup': line_dedup,
            'per_file': per_file, 'book_meta': book_meta, 'report': report,
            'clusters': clusters}


# record form (review 2026-09-25): lists of names or single words (53rd's
# journalists, word tables) were 'verse' because their lines are short; a
# record whose lines are mostly <= 3 tokens is 'list'. In a verse book, a
# record of long unpunctuated lines (10th's Quran verse translation, misras
# of 76-110 chars) is 'verse', not 'prose'.
LIST_MAX_TOKENS = 3
LIST_MIN_LINES = 5
LIST_SHARE = 0.7


def _is_list(lines: List[str]) -> bool:
    ls = [t for t in lines if t.strip()]
    if len(ls) < LIST_MIN_LINES:
        return False
    return sum(1 for t in ls if len(t.split()) <= LIST_MAX_TOKENS) >= LIST_SHARE * len(ls)


def _verse_like_long(lines: List[str]) -> bool:
    ls = [t for t in lines if t.strip()]
    if not ls:
        return False
    ok = sum(1 for t in ls if len(t) <= 110 and not t.rstrip().endswith(END_PUNCT))
    return ok >= 0.8 * len(ls)

def build_report(records, line_dedup, per_file, book_meta, imprints, clusters, sup, boiler,
                 newspaper_texts, doubled, no_text, body_missing) -> dict:
    def by(key):
        return {str(k): v for k, v in sorted(Counter(key(x) for x in records).items(), key=lambda z: str(z[0]))}

    def chars_by(key):
        return {str(k): v for k, v in sorted(_sum_by(records, key).items(), key=lambda z: str(z[0]))}
    raw_chars = sum(sum(pf['chars_by_category'].values()) for pf in per_file)
    text_chars = sum(pf['text_chars'] for pf in per_file)
    rem = defaultdict(lambda: {'lines': 0, 'chars': 0})
    for d in line_dedup:
        rem[d['reason']]['lines'] += 1
        rem[d['reason']]['chars'] += d['n_chars']
    struct = Counter()
    for pf in per_file:
        for k, v in pf['disposition_chars'].items():
            struct[k] += v
    folders = sorted(book_meta)
    with_imprint = [fo for fo in folders if book_meta[fo]['imprint_files']]
    fields = Counter()
    for fo in with_imprint:
        for f, v in book_meta[fo]['imprint_values'].items():
            if v is not None:
                fields[f] += 1
    overlap = {fo: book_meta[fo]['newspaper_overlap_share'] for fo in folders
               if book_meta[fo].get('newspaper_overlap_share')}
    return {
        'files': len(per_file), 'folders': len(folders),
        'records': len(records), 'record_chars': sum(x['n_chars'] for x in records),
        'records_by_role': by(lambda x: x['role']), 'chars_by_role': chars_by(lambda x: x['role']),
        'records_by_form': by(lambda x: x['form']), 'chars_by_form': chars_by(lambda x: x['form']),
        'records_by_genre': by(lambda x: x['genre']), 'chars_by_genre': chars_by(lambda x: x['genre']),
        'records_by_language': by(lambda x: x['language']),
        'raw_frame_chars': raw_chars, 'text_frame_chars': text_chars,
        'removed_by_reason': {k: dict(v) for k, v in sorted(rem.items())},
        'chars_by_disposition': dict(sorted(struct.items())),
        'revision_clusters': len(clusters), 'superseded_files': len(sup),
        'boilerplate_keys': len(boiler),
        'newspaper_records_seen': len(newspaper_texts),
        'newspaper_overlap_share_by_folder': dict(sorted(overlap.items(), key=lambda z: -z[1])),
        'records_also_in_newspaper': sum(1 for x in records if any(
            f.startswith('also_in_newspaper') for f in x['content_flags'])),
        'doubled_files': sorted(doubled),
        'no_extractable_text_files': sorted(no_text),
        'body_missing_from_source_folders': sorted(body_missing),
        'imprint_folders': len(with_imprint),
        'imprint_field_folders': dict(sorted(fields.items())),
        'imprint_conflict_folders': sorted(fo for fo in folders if book_meta[fo]['imprint_conflict']),
        'urdu_dominant_book_folders': sorted(fo for fo in folders if book_meta[fo].get('urdu_dominant_book')),
        'pothohari_folders': sorted(fo for fo in folders if book_meta[fo]['pothohari_evidence']),
        'short_records': sum(1 for x in records if x['n_chars'] < MIN_CHARS and x['role'] != 'toc'),
        'titled_records': sum(1 for x in records if x['title']),
    }


def _sum_by(records, key):
    out = Counter()
    for x in records:
        out[key(x)] += x['n_chars']
    return out
