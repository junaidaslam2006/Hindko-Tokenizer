"""Language, script and genre signals for the Gandhara Hindko book collection.

Everything here was measured on the 207 decoded book files (probe scripts and
their *_out.txt files in _pipeline/). Summaries of the findings are in
_books_cache/wf/probes_rest.txt. The rules are deliberately conservative:
they FLAG text (for a lower tier / an informational field); nothing in this
module ever deletes text.

Contents
  * whole-token Hindko/Urdu scorer ............. hindko_score_v2, language_variety_v2
  * Arabic (Quranic) line flag, rule v4 ......... is_arabic_line
  * Persian line flag, rule v2 .................. is_persian_line
  * Latin / English ............................. latin_share, is_english_text_frame
  * book-level form and genre (R9) .............. book_form, book_genre
  * Pothohari evidence (25th folder only) ....... pothohari_evidence

Why a new scorer instead of segment.hindko_score: the newspaper markers are
matched as raw SUBSTRINGS, and in the books most hits fall inside longer
words (probe_books_markers_out.txt section A): 'اب ' 93.1% inside کتاب,
'وتا' 100% inside Urdu ہوتا (1,183 of 1,849 hits), 'ؤں' 98.0% (گاؤں, پاؤں),
'سی ' 59.2% (اسی, کسی), 'کُن' 86.9%, 'سے ' 47.7% (کسے, ایسے), 'تھی' 60.3%
(تھیں, ساتھی), 'ہیں' 40.1% (نہیں 4,101), 'رہی' 72.7% (رہیا). On 192 gold
Hindko/Urdu lines labelled by reading, the substring scorer gets 121 right and
15 in the wrong direction; the whole-token scorer below gets 142 right and 5
wrong. Over 6,813 ~2k-char book windows 'mixed' drops from 2,175 to 269.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from . import segment

# --------------------------------------------------------------------------
# tokenisation (probe_books_langvar_lib.py)
# --------------------------------------------------------------------------
HARAKAT_RE = re.compile('[\u064B-\u065F\u0670]')
# honorific signs ؐ ؑ ؒ ؓ ؔ are not harakat; stripped before tokenising
HONORIFIC_RE = re.compile('[\u0610-\u061A]')
LATIN_RE = re.compile('[A-Za-z]')
# letters that never occur in Arabic *or* Persian orthography: ٹ ڈ ڑ ں ے ۓ,
# and the Hindko tone letters U+08BE..U+08C2 (ࢾ ࢿ ࣀ ࣁ ࣂ: پ ت ٹ چ ک with
# small v), which the decoder composes since the 2026-09-25 adjudication
# (\x04\x3b; 13,169 in the book decodes, mostly dictionary respellings).
# They are Hindko orthography that never occurs in Arabic or Persian text,
# so a line holding one is not an Arabic / Persian line. (On the decode the
# rules were calibrated on, the mark was a separate ';' and the base letter
# counted as an ordinary letter; the tone letters occur almost only in
# lexicon files, where rules v4 / v2 are not applied.)
HINDKO_TONE_LETTERS = frozenset('ࢾࢿࣀࣁࣂ')
URDU_ONLY_LETTERS = frozenset('ٹڈڑںےۓ') | HINDKO_TONE_LETTERS
# letters absent from Arabic but shared by Persian and Urdu: پ چ ژ گ
PERSO_URDU_LETTERS = frozenset('پچژگ')
TOKEN_SPLIT_RE = re.compile(
    r'[\s،؛؟۔.,:;!?()\[\]"\'‘’“”/\-_*'
    r'\u20260-9\u06F0-\u06F9\u0660-\u0669\uFFFD]+')


def _is_ar_letter(ch: str) -> bool:
    o = ord(ch)
    if not (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or ch in HINDKO_TONE_LETTERS
            or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF):
        return False
    return unicodedata.category(ch) == 'Lo'


def _char_class(pred) -> str:
    cps = [cp for lo, hi in ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08BE, 0x08C2), (0xFB50, 0xFDFF),
                             (0xFE70, 0xFEFF))
           for cp in range(lo, hi + 1) if pred(chr(cp))]
    return '[' + ''.join(re.escape(chr(cp)) for cp in cps) + ']'


# Same sets as the per-character loop of probe_books_langvar_lib.features,
# precompiled so a line is counted with a few regex scans.
AR_LETTER_RE = re.compile(_char_class(_is_ar_letter))
HARAKAT_COUNT_RE = re.compile('[\u064B-\u065F\u0670]')
URDU_ONLY_RE = re.compile('[' + ''.join(sorted(URDU_ONLY_LETTERS)) + ']')
PERSO_URDU_RE = re.compile('[' + ''.join(sorted(PERSO_URDU_LETTERS)) + ']')


def strip_marks(s: str) -> str:
    return HARAKAT_RE.sub('', HONORIFIC_RE.sub('', s)).replace('\u0656', '')


def tokens(s: str) -> List[str]:
    """Whole tokens: harakat/honorifics stripped, split on space, punctuation,
    digits and U+FFFD (probe_books_langvar_lib.tokens)."""
    return [t for t in TOKEN_SPLIT_RE.split(strip_marks(s)) if t]


# --------------------------------------------------------------------------
# whole-token Hindko / Urdu markers
# --------------------------------------------------------------------------
# (1) The existing newspaper word markers (segment.HINDKO_MARKERS /
#     URDU_MARKERS) matched as WHOLE tokens, minus the ones whose hits are
#     substring artefacts (کُن ہونڑ وتا ؤں رُل گھل on the Hindko side; اب on
#     the Urdu side) and minus میں, which is the Hindko word for 'I'
#     (7.7% of 'Urdu' hits inside high-confidence Hindko windows).
#     Same sets as H_TOK / U_TOK in probe_books_langvar.py.
HINDKO_EXISTING = ('دا', 'دی', 'دے', 'نال', 'اچ', 'سی', 'کیہڑی', 'اوہ',
                   'پئے', 'ہوسی', 'جانڑیں', 'برانڑیں')
URDU_EXISTING = ('کا', 'کی', 'کے', 'سے', 'ہے', 'جس', 'لئے', 'گیا', 'تھا',
                 'تھی', 'ہیں', 'رہا', 'رہی', 'کیا', 'ہوئی')
# (2) Validated new whole-token markers = probe_books_markers_kept.json
#     (probe_books_markers.py section D). A word was kept only if it passed in
#     >=3 of 4 discovery/validation folder splits (precision >=0.95 on the
#     discovery half, >=0.90 on the held-out folders, >=0.80 on the newspaper
#     set) and appeared in >=10 folders. 41 Hindko, 43 Urdu. Two are topical
#     (پشور, صیب) - kept as validated, noted as a risk in R3.
#     تھیں ('from') and گھر ('house') are deliberately NOT in the Urdu list:
#     both are Hazara-Hindko forms as well (تھیں: 340th 428x, 343th 57x,
#     3rd 54x, 358th 46x) - R10.
HINDKO_VALIDATED = (
    'ہک', 'ہوندے', 'جاوے', 'ائی', 'جدو', 'اناں', 'طراں', 'تے', 'کیتا', 'ائے',
    'انہاں', 'صیب', 'کردے', 'آپڑے', 'آں', 'ہوندی', 'ہووے', 'وخت', 'آپڑی',
    'ایا', 'جاندے', 'وے', 'پشور', 'بچ', 'دیاں', 'وسے', 'کیتی', 'ہوندا',
    'وسطے', 'ون', 'نوں', 'ہونڑے', 'اپڑے', 'اپڑی', 'جاندی', 'توں', 'ای',
    'جاندا', 'ساڈے', 'جیہڑا', 'لوکاں')
URDU_VALIDATED = (
    'ہمارے', 'ہوتے', 'جب', 'تھے', 'اپنی', 'وہ', 'کیلئے', 'جائے', 'ایک',
    'لوگوں', 'یہ', 'ساتھ', 'اپنے', 'آج', 'ہونے', 'کہا', 'مجھے', 'یہاں', 'بات',
    'اور', 'کرتا', 'جیسے', 'اسے', 'جاتا', 'انہوں', 'کبھی', 'سامنے', 'جاتے',
    'کرتے', 'انہیں', 'گا', 'جاتی', 'کام', 'اپنا', 'ہوتا', 'نہیں', 'ہوں',
    'ہوتی', 'کئے', 'زبانوں', 'بنا', 'وہاں', 'دیا')
HINDKO_TOKENS = frozenset(HINDKO_EXISTING + HINDKO_VALIDATED)
URDU_TOKENS = frozenset(URDU_EXISTING + URDU_VALIDATED)
assert not (HINDKO_TOKENS & URDU_TOKENS)
assert len(HINDKO_VALIDATED) == 41 and len(URDU_VALIDATED) == 43

HINDKO_MIN = 0.70
URDU_MAX = 0.30


def marker_counts(text: str) -> Tuple[int, int]:
    """(Hindko hits, Urdu hits) with whole-token matching."""
    h = u = 0
    for t in tokens(text):
        if t in HINDKO_TOKENS:
            h += 1
        elif t in URDU_TOKENS:
            u += 1
    return h, u


def hindko_score_v2(text: str) -> float:
    """0.0 = only Urdu markers, 1.0 = only Hindko markers, 0.5 = no signal."""
    h, u = marker_counts(text)
    if h + u == 0:
        return 0.5
    return h / (h + u)


def language_variety_v2(text: str) -> str:
    """'hindko' (score >= 0.70), 'urdu' (<= 0.30), 'mixed', or 'no_signal'
    when no marker occurs at all (score reported as 0.5). Kept distinct from
    'mixed' because the probe measured it as a separate class (11 of 6,813
    windows) and it must not be read as evidence of code-mixing."""
    h, u = marker_counts(text)
    if h + u == 0:
        return 'no_signal'
    s = h / (h + u)
    if s >= HINDKO_MIN:
        return 'hindko'
    if s <= URDU_MAX:
        return 'urdu'
    return 'mixed'


# --------------------------------------------------------------------------
# per-line features (probe_books_langvar_lib.features)
# --------------------------------------------------------------------------
# Arabic function words, unvocalised as InPage emits them (Urdu ی/ک).
AR_FUNC = frozenset({
    'من', 'فی', 'علی', 'علی\u0670', 'علے', 'الی', 'الی\u0670', 'ان', 'ان\u0651', 'لا', 'ما',
    'ولا', 'وما', 'قد', 'لم', 'لن', 'عن', 'ثم', 'کان', 'ھو', 'ھم', 'الذی',
    'الذین', 'ذلک', 'ھذا', 'انا', 'انہ', 'لہ', 'بہ', 'فیہ', 'منہ', 'علیہ',
    'علیھم', 'ربنا', 'یا', 'اذا', 'الا', 'لقد', 'وھو', 'ومن', 'فان', 'واللہ',
    'انک', 'کل', 'قال', 'قل'})
# Persian-only subset used as 'strong Persian' evidence by rule v4 (faf)
FA_STRONG = frozenset({
    'است', 'را', 'از', 'ز', 'چہ', 'چو', 'ایں', 'اندر', 'ندارد', 'گفت', 'بود',
    'شد', 'خویش', 'ترا', 'مرا', 'دریں', 'دارد', 'زاں', 'کند', 'کنی', 'گوید',
    'نمی', 'بیا'})


def features(text: str) -> Dict[str, float]:
    """Character/token features of one line. `h`/`u` are the EXISTING
    substring marker counts (segment.HINDKO_MARKERS/URDU_MARKERS): rule v4
    and Persian rule v2 were calibrated with exactly those, so they are kept
    here unchanged."""
    letters = len(AR_LETTER_RE.findall(text))
    harakat = len(HARAKAT_COUNT_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    urdu_only = len(URDU_ONLY_RE.findall(text))
    perso = len(PERSO_URDU_RE.findall(text))
    toks = tokens(text)
    faf = sum(1 for t in toks if t in FA_STRONG)
    h = sum(text.count(m) for m in segment.HINDKO_MARKERS)
    u = sum(text.count(m) for m in segment.URDU_MARKERS)
    return {
        'letters': letters, 'harakat': harakat, 'latin': latin,
        'urdu_only': urdu_only, 'perso': perso, 'faf': faf, 'h': h, 'u': u,
        'harakat_density': harakat / letters if letters else 0.0,
        'latin_share': latin / (latin + letters) if (latin + letters) else 0.0,
        'urdu_only_share': urdu_only / letters if letters else 0.0,
    }


# --------------------------------------------------------------------------
# Arabic (Quranic) line flag - rule v4 (probe_books_arabic.rule_arabic)
# --------------------------------------------------------------------------
# Measured: gold set of 280 manually labelled lines -> precision 1.00, recall
# 0.957 (44 of 46; the misses are unvocalised Qasida Burda lines). Absence of
# Urdu-only and Perso-Urdu letters holds for 46/46 Arabic lines vs 6/146
# Hindko and 2/46 Urdu lines. Harakat alone is worse (>=0.25: P 0.947 R 0.783).
# Frozen 200-line corpus audit: running text outside the Arabic-heavy folders
# 51 correct, 6 titles/citations, 2 wrong of 59; inside 26th/115th/50th 37
# correct + 3 citation lists of 40; DICTIONARY lines 24 wrong of 41 (the
# pronunciation respellings) - so the rule must NOT be applied to lexicons.
# The 'Arabic-only letters' idea cannot work: ي ك ة ى occur 0 times, InPage
# writes Urdu ی ک ۃ even inside the Quran.
NAME_AL = frozenset({'الدین', 'الرحمن', 'الرحمان', 'اللہ', 'الله', 'الحق',
                     'الاسلام', 'الحسن', 'الحاج', 'الہی', 'ال\u0670ہی', 'الحسین',
                     'الرشید', 'القادر', 'الملک', 'العزیز'})
NON_ARTICLE_AL = frozenset({'والا', 'والی', 'والے', 'والیاں', 'والیو', 'والیا',
                            'بالا', 'بالکل', 'بالکہ', 'الزام', 'الگ', 'الٹا',
                            'الٹی', 'الٹے', 'الماری', 'الفاظ', 'الف', 'البتہ',
                            'الغرض', 'الیکشن', 'الجھن', 'الو', 'الاؤ', 'فالتو'})
AMBIG_ARF_V3 = frozenset({'علی', 'یا', 'من', 'ما', 'انا', 'کل', 'لہ', 'ان',
                          'قل', 'لم', 'فی'})
DROP_ARF_V3 = frozenset({'منہ', 'بہ', 'انہ', 'ولا'})
FALLBACK_EXCLUDE = frozenset({'یا', 'علی'})
AMBIG_ARF_V4 = AMBIG_ARF_V3 | {'لا', 'قد', 'کان', 'ھو', 'علیہ', 'ھم', 'عن'}
TANWEEN_SUKUN_RE = re.compile('[\u064B-\u064D\u0652]')


def _arabic_signals(text: str):
    toks = tokens(text)
    al_nn = sum(1 for t in toks
                if len(t) >= 4 and t not in NAME_AL and t not in NON_ARTICLE_AL
                and (t.startswith('ال') or t.startswith('وال')
                     or t.startswith('بال') or t.startswith('فال')))
    arf = [t for t in toks
           if (t in AR_FUNC or t == 'بسم') and t not in DROP_ARF_V3]
    arf_fb = {t for t in arf if t not in FALLBACK_EXCLUDE}
    return len(toks), al_nn, arf, arf_fb


def is_arabic_line(line: str, f: Optional[dict] = None) -> bool:
    """Rule v4, unchanged. Base: >=8 Arabic-script letters, zero Urdu-only
    and zero Perso-Urdu letters. Then (a) >=12 letters and harakat/letter
    >= 0.45, or >= 0.30 with tanween/sukun or an article/function-word cue;
    or (b) no Hindko/Urdu marker, no strong Persian word, >=3 tokens and one
    of: >=2 article tokens making >=15% of tokens; >=1 unambiguous Arabic
    function word; an article token plus a function word or harakat >=0.15;
    >=2 distinct function words other than یا/علی."""
    if f is None:
        if URDU_ONLY_RE.search(line) or PERSO_URDU_RE.search(line):
            return False            # fast path, same outcome as the base test
        f = features(line)
    if f['letters'] < 8 or f['urdu_only'] > 0 or f['perso'] > 0:
        return False
    ntok, al_nn, arf, arf_fb = _arabic_signals(line)
    arf_strong = sum(1 for t in arf if t not in AMBIG_ARF_V4)
    if f['letters'] >= 12:
        if f['harakat_density'] >= 0.45:
            return True
        if f['harakat_density'] >= 0.30 and (
                TANWEEN_SUKUN_RE.search(line) or al_nn or arf):
            return True
    if f['h'] > 0 or f['u'] > 0 or f['faf'] > 0:
        return False
    if ntok < 3:
        return False
    if al_nn >= 2 and al_nn / ntok >= 0.15:
        return True
    if arf_strong >= 1:
        return True
    if al_nn >= 1 and (len(arf) >= 1 or f['harakat_density'] >= 0.15):
        return True
    return len(arf_fb) >= 2


# --------------------------------------------------------------------------
# Persian line flag - rule v2 (probe_books_arabic.rule_persian)
# --------------------------------------------------------------------------
# Precision-oriented. Gold: precision 4/4, recall 4/12. Audit of 60 flagged
# lines: running text 45 correct, 1 mixed, 2 unidentified, 0 wrong of 48;
# dictionary lines 0 correct of 12 -> not applied to lexicons. v1 flagged
# 5,411 lines, driven by respelling fragments (ز 2,970 hits, را 992) and the
# Hindko words ایں (2sg copula), اندر, چو ('from'); those are not used. مرا
# (Hindko 'my') removed; خویش ('خویش قبیلہ') and نمی demoted to the wide list.
FA_STRONG_V2 = frozenset({'است', 'از', 'چہ', 'ترا', 'بود', 'شد', 'خویش',
                          'گفت', 'دارد', 'ندارد', 'کند', 'کنی', 'نمی', 'زاں',
                          'دریں', 'گوید'})
FA_DECISIVE = frozenset({'است', 'ندارد', 'دریں', 'بود', 'گفت', 'دارد', 'گوید'})


def is_persian_line(line: str, f: Optional[dict] = None) -> bool:
    """Rule v2, unchanged: >=8 letters, no Hindko marker, Urdu-only-letter
    share <= 0.11, <30% single-letter tokens, and either one decisive Persian
    word or >=2 words of the wider list outnumbering the Urdu-marker hits."""
    if f is None:
        if any(m in line for m in segment.HINDKO_MARKERS):
            return False            # fast path: h > 0
        if not any(t in FA_STRONG_V2 for t in tokens(line)):
            return False            # fast path: no Persian word at all
        f = features(line)
    if f['letters'] < 8 or f['h'] > 0 or f['urdu_only_share'] > 0.11:
        return False
    toks = tokens(line)
    if not toks or sum(1 for t in toks if len(t) == 1) / len(toks) >= 0.30:
        return False
    fa = [t for t in toks if t in FA_STRONG_V2]
    if not fa:
        return False
    if any(t in FA_DECISIVE for t in fa):
        # Urdu substring markers fire inside Persian words ('سے ' in 'کسے'),
        # so a decisive Persian word is not outvoted by them
        return True
    return len(fa) >= 2 and len(fa) > f['u']


# --------------------------------------------------------------------------
# Latin / English (R8, probe_books_langvar.py section 2, probe_books_latin.py)
# --------------------------------------------------------------------------
# Measured: the content filter silently dropped 181 pure-English frames
# (16,394 chars) in 25 folders (drama dialogue in 216th, English quotes and
# bibliographies in the conference volumes, proverbs in 38th). The rest of
# what it drops is residue: 4,361 frames repeated across files, 3,078 other
# fragments, 282 emails/URLs, 119 file paths, 97 font names. Of the 161
# frames repeating in >=5 files only 3 would pass the English test alone.
# Font / style words (review 2026-09-25, R8): the old single list matched
# colour and style words by PREFIX (FONT_RE.match), so real English such as
# 208th's 'Black Mailing', 'Red Cross' and 'Not - No - None' (the English
# column of the Hindko-English synonym dictionary) was residue. The list is
# now split: FONT_FAMILY words (a font family name, or one of the truncated
# family fragments InPage leaves in its font table: 'imes New Roman', 'rad
# Arabic Bold', 'ombay Black', 'aben', 'atool', 'alat', 'Talat' ...) make a
# frame font residue on their own; STYLE words (bold, black, red, none ...)
# only when EVERY word of the frame is a style / family word. Whole-word
# matches only (fullmatch), never prefixes.
FONT_FAMILY_RE = re.compile(
    r'(arial|times|roman|nastaliq|nastaleeq|naskh|arabic|unicode|ms|sindhi|zohar|noori|jameel|'
    r'aswad|sulus|rouqa|bombay|ombay|zakharif|akharif|macromedia|fonto|syllable|trad|zoha|nask|'
    r'regu|omed|aben|alat|talat|atool|batool|seer|asaar|ahori|lahori|kram|akram|tahoma|verdana|'
    r'calibri|garamond|helvetica|wingdings|webdings|symbol|courier|baltic|nafees|alvi|faiz)', re.I)
STYLE_WORD_RE = re.compile(
    r'(bold|italic|regular|normal|narrow|black|white|red|blue|green|yellow|gray|grey|cyan|magenta|'
    r'shadow|character|simplified|none|old|new|light|medium|heavy|condensed|outline)', re.I)
# kept for callers of the old name: a word that is a family or style word
FONT_RE = re.compile('(?:%s|%s)' % (FONT_FAMILY_RE.pattern, STYLE_WORD_RE.pattern), re.I)
PATH_RE = re.compile(r'([A-Za-z]:\\|\.(jpe?g|png|tiff?|bmp|gif|inp|pdf|cdr|psd)\b)', re.I)
# whole addresses only: the old '.com\b' alternative matched the degree
# 'B.Com' (208th) - a domain needs >= 2 name chars before the dot
URL_RE = re.compile(r'(@[A-Za-z0-9]|www\.|https?:|[A-Za-z0-9\-]{2,}\.(?:com|org|net|pk|gov|edu)\b)', re.I)
EN_WORD_RE = re.compile(r'\b[A-Za-z]{2,}\b')
# R8: InPage style-table strings that pass the word test when they occur in
# fewer than 5 files. Matched on the frame text with whitespace collapsed.
ENGLISH_NOISE = frozenset({'InPage Arabic Document', '3 POINT CENTRE JUS',
                           'imes New Roman', 'imes New Roman Baltic'})
ENGLISH_REPEAT_FILES = 5     # identical frame in >= 5 files -> residue
# Arabic-script characters that are not letters (۔ ، ؛ ؟ ؎ Urdu digits ...):
# a Latin frame that carries one (Lughat '(Statistics)۔', 3rd '۲۰ Linguistic
# Survey of India ...', 44th '۳؎: G A Grierson ...') is still an English frame;
# the R8 tests are applied with them blanked (review 2026-09-25: 73 such
# frames were residue because any Arabic-script char disabled the tests).
_AR_NONLETTER_RE = re.compile(
    '[؀-ؠػ-ـً-ٰ۔-ۭ۰-۹٠-٭﴾﴿]')


def blank_arabic_nonletters(s: str) -> str:
    """s with Arabic-script punctuation / digits / signs replaced by spaces."""
    return _AR_NONLETTER_RE.sub(' ', s)


# font-table strings are short ('Jameel Noori Nastaleeq', 'rad Arabic Bold',
# 'Arial Unicode MS'); a longer frame naming a font family ('The Times of
# India reported ...') is judged by the English test instead
FONT_STRING_MAX_WORDS = 4


def is_font_word(w: str) -> bool:
    return bool(FONT_FAMILY_RE.fullmatch(w) or STYLE_WORD_RE.fullmatch(w))


def is_font_string(words: List[str]) -> bool:
    """A font / style-table string: a family word, or only style words."""
    if not words:
        return False
    if any(FONT_FAMILY_RE.fullmatch(w) for w in words):
        return True
    return all(STYLE_WORD_RE.fullmatch(w) for w in words)


def latin_share(text: str) -> float:
    """Latin letters / (Latin + Arabic-script letters)."""
    latin = len(LATIN_RE.findall(text))
    if not latin:
        return 0.0
    letters = len(AR_LETTER_RE.findall(text))
    return latin / (latin + letters)


def english_words(s: str) -> List[str]:
    return [w for w in EN_WORD_RE.findall(s) if re.search('[aeiouyAEIOUY]', w)]


def latin_frame_category(frame: str, n_files_with_same_frame: int, embedded: bool = False) -> str:
    """R8 category of a frame that is not Hindko/Urdu text and has >= 3 Latin
    letters: path / email_url / repeated_residue / english_text / font_style
    / other_residue (probe_books_langvar.classify_latin_frame + ENGLISH_NOISE).
    Arabic-script punctuation / digits in the frame are blanked first.
    `embedded`: the frame sits directly between two text frames (books.
    classify_file); the >= 5-file repeat rule does not apply there - 208th's
    'Time' repeats as residue in 5 files but is a cell of the dictionary's
    English column (review 2026-09-25)."""
    s = blank_arabic_nonletters(frame).strip()
    if len(LATIN_RE.findall(s)) < 3:
        return 'not_latin'
    if re.sub(r'\s+', ' ', s) in ENGLISH_NOISE:
        return 'font_style'
    if PATH_RE.search(s):
        return 'path'
    if URL_RE.search(s):
        return 'email_url'
    if n_files_with_same_frame >= ENGLISH_REPEAT_FILES and not embedded:
        return 'repeated_residue'
    words = english_words(s)
    latin = len(LATIN_RE.findall(s))
    compact = len(re.sub(r'\s', '', s))
    allw = EN_WORD_RE.findall(s)
    if len(allw) <= FONT_STRING_MAX_WORDS and (is_font_string(words) or is_font_string(allw)):
        return 'font_style'
    font_words = sum(1 for w in words if is_font_word(w))
    if (len(words) >= 3 and latin >= 12 and latin / max(1, compact) >= 0.6
            and font_words < len(words) / 2):
        return 'english_text'
    return 'other_residue'


def is_english_text_frame(frame: str, n_files_with_same_frame: int, embedded: bool = False) -> bool:
    """R8: >=3 English-like words (>=2 letters with a vowel), >=12 Latin
    letters, Latin >= 60% of non-space chars, fewer than half the words font
    names, not a font string / path / email / URL, and the identical frame in
    < 5 files (unless embedded between text frames)."""
    return latin_frame_category(frame, n_files_with_same_frame, embedded) == 'english_text'


# Short English frames that fail the 3-word R8 test but are ordinary words:
# only Latin letters, digits, spaces and plain punctuation (a leading bracket,
# quote or digit allowed: 25th's glosses '(Guitar)', '(Hook)', 307th
# '(Dissolve)', 8th '(Men Prayers)'; a stray ')' before the gloss: 25th /
# Lughat ') (Ladyfinger'), at least one English-like word, not a
# font / style string, not a path / email / URL, and the identical frame in
# < 5 files unless embedded. Measured (eng208b / eng_all): 1,579 such frames
# (8,869 chars) in 21 files were dropped as residue inside main spans - 1,108
# of them the English column of the 208th Hindko-English synonym dictionary
# ('Theater', 'Thermometer', 'Ticket'), English dialogue in 216th ('OK- Take
# care.', 'Why not.') and screenplay directions (CUT / FADE OUT: 305th 182,
# 307th 165, 109th 78). books.classify_file keeps them as text inside the
# main span, anywhere they sit directly between text frames, and everywhere
# in a .docx; elsewhere (InPage style tables) they stay residue.
EN_WORD_FRAME_RE = re.compile(r"^[()\[‘'\"0-9 ]*[A-Za-z][A-Za-z0-9 .,'’‘()\[\]/:;!?&\"\-]*$")


def is_english_word_frame(frame: str, n_files_with_same_frame: int, embedded: bool = False) -> bool:
    """A short English frame (see above). Embedded between text frames an
    abbreviation without a vowel word also counts (208th's English column
    'L.L.B', 'L.L.D', 'B.Sc')."""
    s = blank_arabic_nonletters(frame).strip()
    if latin_frame_category(s, n_files_with_same_frame, embedded) != 'other_residue':
        return False
    if not EN_WORD_FRAME_RE.match(s):
        return False
    words = english_words(s)
    if not words and not embedded:
        return False
    return not is_font_string(words or EN_WORD_RE.findall(s))


# --------------------------------------------------------------------------
# book-level form and genre (R9, probe_books_langvar.py sections 3-4)
# --------------------------------------------------------------------------
# Form: share of content LINES (segment.is_content_line) of <= 60 chars.
# Calibrated on books whose path names the genre: poetry minimum 0.82 (13 of
# 14 at >= 0.91); prose median 0.47, maximum 0.87.
VERSE_LINE_SHORT = 0.90
PROSE_LINE_SHORT = 0.60
FORM_MIN_CHARS = 3000
SHORT_LINE_MAX = 60


def book_form(n_lines: int, n_short_lines: int, n_chars: int) -> str:
    if n_chars < FORM_MIN_CHARS:
        return 'insufficient_text'
    share = n_short_lines / n_lines if n_lines else 0.0
    if share >= VERSE_LINE_SHORT:
        return 'verse'
    if share <= PROSE_LINE_SHORT:
        return 'prose'
    return 'mixed'


# (regex on the FOLDER name, genre) - first match wins
PATH_GENRE = [
    (r'dictionary|lughat|mutaradif', 'dictionary'),
    (r'matt?lan|akhan|proverb', 'proverbs'),
    (r'bol ?chal', 'language_learning'),
    (r'mushaira', 'poetry'),
    (r'conference', 'conference_proceedings'),
    (r'lok ?geet', 'folk_songs'),
    (r'lok ?kahan', 'folk_tales'),
    (r'quran|sorah|surah|hadees', 'scripture_translation'),
    (r'burda|mairaj|naat', 'religious_poetry'),
    (r'mahiye|chaarbitay|chaarbaitey|char ?bait|ghazal|gahzal|\bkalam\b|poetry|deewan|harfi', 'poetry'),
    (r'afsan', 'short_stories'),
    (r'nawal|novel', 'novel'),
    (r'filim|film', 'film_script'),
    (r'drama', 'drama'),
    (r'tabsar|naqd', 'criticism_reviews'),
    (r'mazameen', 'essays'),
    (r'mazah', 'humour'),
    (r'harmain|hajj|umra|safar', 'pilgrimage_travelogue'),
    (r'tazkira|namwar|sapoot', 'biography'),   # before 'qadeem': 'Tazkira Qadeem o Jadeed ... Shorra'
    (r'tareekh|aasar|qadeem', 'history'),
    (r'tandrusti', 'health'),
    (r'ghazwat|islam|salook|marifat|molvi', 'religious_prose'),
]
# Genres that may also be read from FILE / sub-folder names. File names also
# carry composer names ('OLARAY ... (COMPOSED BY ZIA-UL-ISLAM' made 170th
# 'religious' in the first run), so religious/history rules are folder-only
# (BOOKS_SPEC: 'file names NOT used for religious/history keywords';
# probes_rest R9). scripture_translation / religious_poetry are therefore NOT
# file genres: 'SORAH BAKRA ... .inp' made the 10th Mushq-e-Gulab folder
# 'scripture_translation' (114 records, 203,278 chars) by file name alone.
FILE_GENRE_OK = frozenset({
    'dictionary', 'proverbs', 'conference_proceedings', 'folk_songs',
    'poetry', 'short_stories', 'novel', 'film_script', 'drama',
    'criticism_reviews', 'essays'})
RELIGIOUS_HISTORY_GENRES = frozenset({'scripture_translation', 'religious_poetry',
                                      'religious_prose', 'history'})
assert not (FILE_GENRE_OK & RELIGIOUS_HISTORY_GENRES)
# front-matter phrases (whole-token sequences) -> genre; used only when no
# path rule fires
TEXT_GENRE = [
    ('dictionary', ['لغت', 'ڈکشنری']),
    ('proverbs', ['اکھان', 'ضرب الامثال']),
    ('conference_proceedings', ['کانفرنس']),
    ('sketches', ['خاکے', 'خاکہ']),
    ('short_stories', ['افسانے', 'افسانیاں', 'افسانوی']),
    ('novel', ['ناول']),
    ('drama', ['ڈرامہ', 'ڈرامے', 'ڈرامیاں']),
    ('film_script', ['فلم']),
    ('poetry', ['شعری مجموعہ', 'مجموعہ کلام', 'مجموعۂ کلام', 'غزلاں', 'حرفیاں',
                'ماہیے', 'چاربیتے', 'نظماں']),
]
TEXT_GENRE_MIN = 3
# 'کانفرنس' x3 (37th columns) and 'فلم' x4 (212th Urdu prose) were too weak
TEXT_GENRE_MIN_SPECIAL = {'conference_proceedings': 20, 'film_script': 10}
PROSE_GENRES = frozenset({'short_stories', 'novel', 'drama', 'film_script',
                          'sketches', 'conference_proceedings'})
FRONT_LINES = 150            # first N content lines of each file = front matter


# A genre word that names the FIELD a book studies is not the book's genre
# (review 2026-09-25): 'خاکہ نگاری' / 'مزاح نگاری' (sketch / humour writing as
# a subject: 47th 'صوبہ سرحد میں خاکہ نگاری', 152nd), and a romanised folder
# keyword followed by a genitive postposition ('Dramay Day Fani Taqazay' =
# 'the technical requirements OF drama', 306th, a treatise) or by 'Nigari'.
FIELD_SUFFIX = 'نگاری'
PATH_TOPIC_NEXT = frozenset({'day', 'dey', 'di', 'da', 'de', 'dy', 'ki', 'ka', 'ke', 'nigari'})
# front text: a runner-up genre with >= half the top genre's hits makes the
# front-text evidence ambiguous (306th's treatise discusses ڈرامہ, ناول and
# افسانہ); no text genre is chosen then
FRONT_AMBIGUOUS = 0.5


def front_genre(front_text: str) -> Optional[Tuple[str, int, str]]:
    """(genre, hits, first phrase) with the most whole-token phrase hits above
    the genre's threshold, else None. A phrase followed by 'نگاری' (the field
    of study) does not count; an ambiguous result (runner-up >= half the top)
    is None."""
    joined = ' ' + ' '.join(tokens(front_text)) + ' '
    cands = []
    for genre, phrases in TEXT_GENRE:
        n = sum(joined.count(' ' + p + ' ') - joined.count(' ' + p + ' ' + FIELD_SUFFIX + ' ')
                for p in phrases)
        if n >= TEXT_GENRE_MIN_SPECIAL.get(genre, TEXT_GENRE_MIN):
            cands.append((n, genre, phrases[0]))
    if not cands:
        return None
    cands.sort(key=lambda z: (-z[0], z[1]))
    if len(cands) > 1 and cands[1][0] >= FRONT_AMBIGUOUS * cands[0][0]:
        return None
    n, genre, ph = cands[0]
    return genre, n, ph


# The imprint's 'موضوع' (subject) value is the book's own statement of what
# it is and is used FIRST (review 2026-09-25: topic keywords made 545 records
# / 1.78M chars of treatises, studies and memoirs 'drama', 'sketches',
# 'humour', 'film_script', 'scripture_translation', 'religious_prose').
# Whole-token phrases of the subject value -> genre.
SUBJECT_GENRE = [
    ('dictionary', ['لغت', 'ڈکشنری']),
    ('proverbs', ['اکھان', 'ضرب الامثال', 'متلاں']),
    ('religious_poetry', ['نعت', 'نعتیہ', 'نعتاں', 'حمد']),
    ('religious_prose', ['تصوف']),
    ('folk_songs', ['لوک گیت']),
    ('folk_tales', ['لوک کہانی', 'لوک کہانیاں']),
    ('poetry', ['شاعری', 'حرفیاں', 'حرفی', 'نظماں', 'غزلاں', 'گونڑ', 'شعری', 'ماہیے']),
    ('short_stories', ['افسانے', 'افسانہ']),
    ('novel', ['ناول']),
    ('drama', ['ڈرامہ', 'ڈرامے']),
    ('travelogue', ['سفرنامہ', 'سفر نامہ']),
    ('pilgrimage_travelogue', ['حج', 'عمرہ']),
    ('biography', ['سوانح', 'سرگزشت', 'تذکرہ']),
    ('essays', ['کالمی مجموعہ', 'مضامین']),
    ('research_criticism', ['تحقیق', 'تحقیقی', 'تنقید', 'تنقیدی', 'مقالات', 'مقالہ', 'تقاریر',
                            FIELD_SUFFIX]),
]
SUBJECT_PROSE_MARK = 'نثر'
# genre -> genres that are a more specific kind of it (the keyword genre is
# then kept, with both pieces of evidence)
GENRE_REFINES = {
    'poetry': frozenset({'religious_poetry', 'folk_songs'}),
    'travelogue': frozenset({'pilgrimage_travelogue'}),
    'research_criticism': frozenset({'conference_proceedings', 'criticism_reviews', 'history',
                                     'biography', 'dictionary', 'language_learning', 'proverbs',
                                     'religious_prose', 'health', 'essays'}),
}
# subject genres that are prose: against a measured verse form they conflict
SUBJECT_PROSE_GENRES = frozenset({'research_criticism', 'essays', 'novel', 'short_stories', 'drama',
                                  'biography', 'religious_prose', 'travelogue',
                                  'pilgrimage_travelogue', 'folk_tales'})


def subject_genres(subject: Optional[str]) -> Tuple[set, bool]:
    """(genres named by an imprint subject value, prose marker present).
    'قرآن' + 'ترجمہ' is a scripture translation; 'قرآن' otherwise (26th
    'قرآن پاک کے بارے میں' - ABOUT the Quran) religious prose. Research is a
    method: with another genre it is dropped ('سوانح عمری/تحقیق' = biography,
    'لغت/تحقیق' = dictionary); a genre and its refinement keep the refinement
    ('ہندکو نعتیہ شاعری' = religious_poetry)."""
    if not subject:
        return set(), False
    toks = tokens(subject)
    joined = ' ' + ' '.join(toks) + ' '
    out = set()
    for g, phrases in SUBJECT_GENRE:
        if any(' ' + p + ' ' in joined for p in phrases):
            out.add(g)
    if 'قرآن' in toks:
        out.add('scripture_translation' if ('ترجمہ' in toks or 'ترجمے' in toks) else 'religious_prose')
    if len(out) > 1:
        out.discard('research_criticism')
    for g in list(out):
        if GENRE_REFINES.get(g, frozenset()) & out:
            out.discard(g)
    return out, SUBJECT_PROSE_MARK in toks


def _path_genre(names: str, allowed=None):
    for rx, g in PATH_GENRE:
        for m in re.finditer(rx, names, re.I):
            if allowed is not None and g not in allowed:
                break
            # the rest of the romanised word, then the next word
            tail = names[m.end():]
            nxt = re.match(r'[A-Za-z]*[\s\-]+([A-Za-z]+)', tail)
            if nxt and nxt.group(1).lower() in PATH_TOPIC_NEXT:
                continue            # 'Dramay Day ...': the keyword is a topic
            return g, m.group(0)
    return None


def keyword_genre(folder: str, file_names: Iterable[str], front_text: str,
                  form: str) -> Tuple[Optional[str], Optional[str]]:
    """The keyword evidence: folder-name keyword -> file-name keyword
    (non-religious/history genres only) -> front-matter phrases (dropped when
    they contradict the measured form or are ambiguous) -> form=verse -> None.
    Evidence strings: 'path:<kw>', 'file:<kw>', 'text:<phrase>(xN)',
    'form:verse'."""
    pg = _path_genre(folder)
    if pg:
        return pg[0], 'path:%s' % pg[1]
    pg = _path_genre(' '.join(file_names), FILE_GENRE_OK)
    if pg:
        return pg[0], 'file:%s' % pg[1]
    tg = front_genre(front_text)
    if tg and ((tg[0] == 'poetry' and form == 'prose')
               or (tg[0] in PROSE_GENRES and form == 'verse')):
        tg = None       # 20th, 22nd, 52nd, 165th: front-matter genre contradicts the form
    if tg:
        return tg[0], 'text:%s(x%d)' % (tg[2], tg[1])
    if form == 'verse':
        return 'poetry', 'form:verse'
    return None, None


# The imprint's book TITLE is used when there is no subject, for one thing
# only: a title that names a genre as its topic - 'X نگاری' (47th 'صوبہ سرحد
# میں خاکہ نگاری') or a genre word followed by a genitive / locative
# postposition in a prose book (306th 'ڈرامہ دے فنی تقاضے', 'the technical
# requirements OF drama') - is research / criticism, not that genre.
TITLE_GENRE_WORDS = frozenset({'ڈرامہ', 'ڈرامے', 'افسانہ', 'افسانے', 'ناول', 'شاعری', 'غزل', 'خاکہ',
                               'خاکے', 'فلم', 'مزاح', 'نظم', 'ماہیا', 'حرفی', 'چاربیتہ', 'کالم'})
TITLE_TOPIC_NEXT = frozenset({'دے', 'دا', 'دی', 'دیاں', 'کے', 'کا', 'کی', 'میں', 'اچ', 'بچ', 'وچ'})


def title_topic(title: Optional[str], form: str) -> bool:
    if not title:
        return False
    toks = tokens(title)
    if FIELD_SUFFIX in toks:
        return True
    if form == 'verse':
        return False
    return any(a in TITLE_GENRE_WORDS and b in TITLE_TOPIC_NEXT for a, b in zip(toks, toks[1:]))


def book_genre(folder: str, file_names: Iterable[str], front_text: str,
               form: str, subject: Optional[str] = None,
               title: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """(genre, genre_evidence). The imprint subject first: one subject genre
    wins over keyword evidence ('imprint_subject:<value>'), except that a
    keyword genre refining it is kept ('imprint_subject:<value>; path:<kw>':
    conference proceedings are research papers, a pilgrimage travelogue is a
    travelogue); a prose subject against a measured verse form, a subject
    naming two genres, or poetry with the prose mark ('ہندکو شاعری/نثر') is a
    conflict -> (None, None) (see genre_conflict). No subject genre: the
    keyword evidence (keyword_genre)."""
    g0, ev0 = keyword_genre(folder, file_names, front_text, form)
    S, prose_mark = subject_genres(subject)
    what = 'imprint_subject:%s' % subject
    if not S and title_topic(title, form):
        S, prose_mark, what = {'research_criticism'}, False, 'imprint_title:%s' % title
    if not S:
        return g0, ev0
    if len(S) > 1 or (prose_mark and S == {'poetry'}):
        return None, None
    G = next(iter(S))
    if g0 is not None and (g0 == G or g0 in GENRE_REFINES.get(G, frozenset())):
        return g0, '%s; %s' % (what, ev0)
    if ev0 == 'form:verse' and G in SUBJECT_PROSE_GENRES:
        return None, None
    return G, what


def genre_conflict(folder: str, file_names: Iterable[str], front_text: str,
                   form: str, subject: Optional[str] = None, title: Optional[str] = None) -> Optional[str]:
    """Why book_genre returned no genre although evidence exists (a note for
    book_meta), else None."""
    g0, ev0 = keyword_genre(folder, file_names, front_text, form)
    S, prose_mark = subject_genres(subject)
    if not S and title_topic(title, form):
        S, prose_mark, subject = {'research_criticism'}, False, title
    if not S:
        return None
    if len(S) > 1:
        return 'imprint subject %r names %s' % (subject, sorted(S))
    if prose_mark and S == {'poetry'}:
        return 'imprint subject %r names poetry and prose' % subject
    G = next(iter(S))
    if ev0 == 'form:verse' and G in SUBJECT_PROSE_GENRES and not (
            g0 == G or g0 in GENRE_REFINES.get(G, frozenset())):
        return 'imprint subject %r (%s) against the measured verse form' % (subject, G)
    return None


# --------------------------------------------------------------------------
# Pothohari (R7, probe_books_pothohari.py)
# --------------------------------------------------------------------------
# Only the 25th folder, and only with BOTH path and title-page evidence.
# Measured: the title page says پوٹھوہاری لغت; 43 lines mention پوٹھوہار; the
# invocation formula uses Pothohari 'رب ناں' where 94 of 95 folders use
# Hindko 'رب دا'; na-series genitive share 0.824 vs 0.010 in other books;
# نانہہ 385 vs 0. These whole-book rates are diagnostics, not a line test.
POTHOHARI_PATH_RE = re.compile(r'pothohari', re.I)
POTHOHARI_TITLE = 'پوٹھوہاری لغت'


def pothohari_evidence(folder: str, texts: Iterable[str]) -> Optional[str]:
    """Evidence string when the folder is path- AND text-evidenced as the
    Pothohari dictionary, else None."""
    m = POTHOHARI_PATH_RE.search(folder)
    if not m:
        return None
    for t in texts:
        if POTHOHARI_TITLE in re.sub(r'\s+', ' ', t):
            return 'path:%s; text:%s' % (m.group(0), POTHOHARI_TITLE)
    return None
