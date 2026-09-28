"""Article segmentation and reliable metadata derivation.

Weekly Hindkowan layout files are highly regular. Each InPage text frame
decodes to one line, and an article is:

    headline   (short,  ~44-70 chars)
    subhead    (medium, ~73-182 chars, optional)
    body       (long,   ~340-720 chars, opening with a dateline such as
                        "پشور(ہندکووان نیوز)" or "لاہور(ویب ڈیسک)")

Photo captions appear as short frames with no following body. Boundaries are
therefore detected structurally (frame length + dateline), not by guessing.

Metadata is only emitted where it is actually evidenced by the file path or
the document text. Anything uncertain stays null.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

# Arabic script blocks incl. presentation forms
ARABIC = re.compile(
    '[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
    '\uFB50-\uFDFF\uFE70-\uFEFF]'
)
DIGITS = re.compile('[0-9\u06F0-\u06F9]')

HEADLINE_MAX = 90     # frames at or below this read as headline/subhead
BODY_MIN = 200        # frames at or above this read as article body
CAPTION_ONLY_MAX = 120

# city/agency tokens that open a dateline
DATELINE_CITIES = [
    'پشور', 'پشاو ر', 'پشاور', 'لاہور', 'کراچی', 'اسلام آباد', 'ہزارہ',
    'مانسہرہ', 'ایبٹ آباد', 'ہری پور', 'کوہاٹ', 'بنوں', 'ٹانک', 'صوابی',
    'مردان', 'سوات', 'چترال', 'بٹگرام', 'شانگلہ', 'نوشہرہ', 'چارسدہ',
    'خیبر', 'مالاکند', 'دیر', 'کابل', 'لندن', 'دہلی', 'ممبئی', 'واشنگٹن',
    'انقرہ', 'تہران', 'جدہ', 'دبئی', 'کرا چی', 'کوہا ٹ',
]
DATELINE_SOURCES = [
    'ہندکووان نیوز', 'ہندکووا ن نیوز', 'ویب ڈیسک', 'نمائندہ ہندکووان',
    'سٹاف رپورٹر', 'خصوصی رپورٹ', 'بیورو رپورٹ', 'اے پی پی', 'رائٹرز',
    'پی پی آئی', 'یو این آئی', 'نامہ نگار',
]

_RE_CITY_PAREN = re.compile(
    r'^(%s)\s*\(' % '|'.join(re.escape(c) for c in DATELINE_CITIES))
_RE_ANY_PAREN = re.compile(r'^([^\s()،,]{2,24})\(([^)]{1,48})\)')
_RE_CITY_COMMA = re.compile(
    r'^(%s)\s*[،,]' % '|'.join(re.escape(c) for c in DATELINE_CITIES))


def script_fraction(line: str) -> float:
    """Fraction of non-space characters that are Arabic-script."""
    compact = re.sub(r'\s', '', line)
    if not compact:
        return 0.0
    return len(ARABIC.findall(compact)) / len(compact)


# A decoded frame must carry real language signal to be treated as content.
# InPage streams interleave formatting tables, so decoding also yields short
# symbol runs ('T"', '~W', 'q', '{') and long binary blobs. Filtering here
# keeps them out of segmentation, where they would otherwise be mistaken for
# headlines (and even for titles).
MIN_ARABIC_CHARS = 4
MIN_ALNUM_CHARS = 5
MAX_ASCII_ONLY_LEN = 200
# Observed real headlines run 44-70 chars. Shorter frames are editing
# residue (e.g. "ت ترک ک", "کے لئے جلدی") that must not become titles.
MIN_CONTENT_LEN = 12
MIN_TITLE_LEN = 20


def is_content_line(line: str) -> bool:
    """True when a decoded frame carries genuine Hindko/Urdu language signal.

    Requires Arabic-script characters. InPage streams interleave font and
    style tables, which decode to pure-ASCII residue ('Tahoma', 'Arial
    Baltic', 'head-12r'); those are dropped here. Legitimate English, names,
    numbers and technical terms are unaffected because they occur *inside*
    frames that also contain Urdu script.
    """
    compact = re.sub(r'\s', '', line)
    if not compact:
        return False
    if len(ARABIC.findall(compact)) < MIN_ARABIC_CHARS:
        return False
    if len(line.strip()) < MIN_CONTENT_LEN:
        return False
    # A long frame with almost no script is binary/style residue.
    if len(compact) > MAX_ASCII_ONLY_LEN and script_fraction(line) < 0.30:
        return False
    return True


# A headline must look like a headline before it is emitted as `title`.
# Prevents font-table residue (e.g. 'head-12r') and binary fragments from
# becoming metadata.
_TITLE_FORBIDDEN = re.compile(r'[A-Za-z_|@~`{}\[\]<>=+*#$%^&\\]')


def is_valid_title(line: str) -> bool:
    if not line or not (MIN_TITLE_LEN <= len(line) <= 160):
        return False
    if _TITLE_FORBIDDEN.search(line):
        return False
    return script_fraction(line) >= 0.80


# --------------------------------------------------------------------------
# Hindko vs Urdu
# --------------------------------------------------------------------------
# Weekly Hindkowan is a Hindko paper, but it also carries Urdu columns and
# poetry (e.g. nusrat naseem.inp), and some stories carry a Hindko subhead
# paired with an Urdu one. The genitive/postposition sets diverge sharply and
# give a reliable discriminator: measured ratios are 0.76-0.87 for Hindko copy
# and 0.04-0.09 for Urdu copy, with a wide gap between.
HINDKO_MARKERS = (
    'دا ', 'دی ', 'دے ', 'نال', 'اچ ', 'کُن', 'ہونڑ', 'سی ', 'کیہڑی',
    'اوہ', 'پئے', 'وتا', 'ؤں', 'برانڑیں', 'رُل', 'گھل', 'ہوسی', 'جانڑیں',
)
URDU_MARKERS = (
    'کا ', 'کی ', 'کے ', 'میں', 'سے ', 'ہے ', 'اب ', 'جس ', 'لئے',
    'گیا', 'تھا', 'تھی', 'ہیں', 'رہا', 'رہی', 'کیا', 'ہوئی',
)


def hindko_score(text: str) -> float:
    """0.0 = purely Urdu markers, 1.0 = purely Hindko markers, 0.5 = no signal."""
    h = sum(text.count(m) for m in HINDKO_MARKERS)
    u = sum(text.count(m) for m in URDU_MARKERS)
    if h + u == 0:
        return 0.5
    return h / (h + u)


def language_variety(text: str) -> str:
    s = hindko_score(text)
    if s >= 0.70:
        return 'hindko'
    if s <= 0.30:
        return 'urdu'
    return 'mixed'


def is_dateline(line: str) -> bool:
    """True when a line opens with a place/source dateline."""
    if _RE_CITY_PAREN.match(line) or _RE_CITY_COMMA.match(line):
        return True
    m = _RE_ANY_PAREN.match(line)
    if m:
        inner = m.group(2)
        return any(s in inner for s in DATELINE_SOURCES)
    return False


def classify(line: str) -> str:
    n = len(line)
    if n >= BODY_MIN:
        return 'body'
    if n <= HEADLINE_MAX:
        return 'headline'
    return 'subhead'


def segment_articles(lines: List[str]) -> List[dict]:
    """Group decoded text frames into articles.

    Two distinct layouts occur in this archive:

    *2023 issues* - frames are in reading order: headline, optional subhead,
    then body. Merging them yields whole articles with a real title.

    *2024-2026 issues* - InPage stores frames in object order, NOT reading
    order. Headlines appear *after* their bodies (e.g. in 7.8.P2.INP the
    headline "بنگلہ دیش کل تے اَج" sits two frames past the body it belongs
    to), and one article's paragraphs are interleaved with another's. Article
    boundaries are therefore not recoverable from frame sequence at all.

    So a new article starts whenever the current one already has a body frame
    and the next frame is a headline-length line, a body-length line, or opens
    with a dateline. The middle case is what stops 2024-26 pages collapsing
    into a single 24k-character blob; it means a long article split across
    several frames becomes several coherent records rather than being fused
    with unrelated neighbours. Only short "subhead" frames continue an article.
    """
    articles: List[dict] = []
    cur: List[str] = []
    has_body = False

    def flush():
        if cur:
            articles.append(build_article(cur))

    for line in lines:
        starts_new = False
        if cur and has_body:
            starts_new = (len(line) <= HEADLINE_MAX
                          or len(line) >= BODY_MIN
                          or is_dateline(line))
        if starts_new:
            flush()
            cur, has_body = [], False
        cur.append(line)
        if len(line) >= BODY_MIN:
            has_body = True
    flush()
    return [a for a in articles if a['lines']]


def build_article(lines: List[str]) -> dict:
    kinds = [classify(ln) for ln in lines]
    body_idx = [i for i, k in enumerate(kinds) if k == 'body']

    title: Optional[str] = None
    title_source: Optional[str] = None
    if len(lines) >= 2 and kinds[0] == 'headline' and body_idx:
        if is_valid_title(lines[0]):
            title = lines[0]
            title_source = 'first_frame_headline'

    has_body = bool(body_idx)
    if has_body:
        role = 'article'
    elif all(k == 'headline' for k in kinds):
        role = 'caption_or_heading' if sum(map(len, lines)) <= CAPTION_ONLY_MAX * len(lines) else 'fragment'
    else:
        role = 'fragment'

    return {
        'lines': lines,
        'kinds': kinds,
        'title': title,
        'title_source': title_source,
        'role': role,
        'has_body': has_body,
        'starts_with_dateline': is_dateline(lines[0]) if lines else False,
        'text': '\n'.join(lines),
        'n_chars': sum(len(ln) for ln in lines),
        'n_frames': len(lines),
    }


# --------------------------------------------------------------------------
# metadata from the source path
# --------------------------------------------------------------------------
MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11,
    'december': 12, 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

_RE_ISSUE_NO = re.compile(r'Issue\s*No\.?\s*(\d{2,4})', re.I)
_RE_DMY = re.compile(r'\b(\d{1,2})[.\-/](\d{1,2})[.\-/](20\d{2})\b')
_RE_DMY_REV = re.compile(r'\b(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b')
_RE_MONTH_YEAR = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|'
    r'November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
    r'[- ](20\d{2})\b', re.I)


def derive_issue(rel_path: str) -> Optional[str]:
    """Issue number, from an explicit 'Issue No. N' or a bare numeric folder."""
    m = _RE_ISSUE_NO.search(rel_path)
    if m:
        return m.group(1)
    parts = rel_path.replace('\\', '/').split('/')
    for part in reversed(parts[:-1]):          # ignore the filename itself
        if part.isdigit() and 100 <= int(part) <= 9999:
            return part
    return None


def derive_date(rel_path: str) -> tuple[Optional[str], Optional[str]]:
    """(date, precision) using only dates evidenced in the path.

    precision is 'day', 'month', 'year' or None. Never guessed.
    """
    name = os.path.basename(rel_path)
    stem = os.path.splitext(name)[0]

    for source in (stem, rel_path):
        m = _RE_DMY.search(source)
        if m:
            d, mo, y = (int(x) for x in m.groups())
            if 1 <= d <= 31 and 1 <= mo <= 12:
                return '%04d-%02d-%02d' % (y, mo, d), 'day'
        m = _RE_DMY_REV.search(source)
        if m:
            y, mo, d = (int(x) for x in m.groups())
            if 1 <= d <= 31 and 1 <= mo <= 12:
                return '%04d-%02d-%02d' % (y, mo, d), 'day'

    m = _RE_MONTH_YEAR.search(rel_path)
    if m:
        mon = MONTHS.get(m.group(1).lower()[:3]) or MONTHS.get(m.group(1).lower())
        if mon:
            return '%s-%02d' % (m.group(2), mon), 'month'

    # Year only when the *immediate* parent folder is unambiguously a year
    # (e.g. ".../2024 Newspapers/"). Matching a year anywhere in the path
    # would pick up unrelated segments such as "2024 to 2026".
    parent = os.path.basename(os.path.dirname(rel_path.replace('\\', '/')))
    m = re.fullmatch(r'(20\d{2})(?:\s*Newspapers?)?', parent, re.I)
    if m:
        return m.group(1), 'year'
    return None, None


def derive_section(rel_path: str) -> Optional[str]:
    """Coarse role of the source file, from its filename only."""
    stem = os.path.splitext(os.path.basename(rel_path))[0].lower()
    if 'idari' in stem or 'idary' in stem:
        return 'editorial'
    if 'front page' in stem or stem.startswith('fp'):
        return 'front_page'
    if 'back page' in stem:
        return 'back_page'
    if re.match(r'^p\s*\d', stem):
        return 'inner_page'
    return None
