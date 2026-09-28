"""Cleaning, deduplication and dual-tier quality gating."""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .segment import ARABIC, DIGITS, script_fraction

# --------------------------------------------------------------------------
# normalisation / cleaning
# --------------------------------------------------------------------------
CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
REPLACEMENT = '\ufffd'

# zero-width / bidi marks we normalise rather than delete: ZWNJ is meaningful
# in Urdu (نیم فاصلہ), so it is preserved in a canonical form.
ZW_RE = re.compile('[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]')

WS_RE = re.compile(r'[ \t\u00a0\u2007\u202f]+')
MULTI_NL_RE = re.compile(r'\n{3,}')

# punctuation repair: InPage legacy output often doubles/triples these
PUNCT_REPEAT_RE = re.compile(r'([،۔؛:!؟?.\-—–])\1{2,}')
# Horizontal whitespace only: the earlier \s+ also swallowed a newline before a
# line that starts with punctuation and so joined two lines (QA 2026-09-25).
# '…' (the InPage E0 ellipsis glyph) is punctuation like '۔'.
SPACE_BEFORE_PUNCT_RE = re.compile(r'[ \t]+([،۔؛:!؟?.\)\]…])')
SPACE_AFTER_PAREN_RE = re.compile(r'(\()\s+')

# Characters that never occur in legitimate Urdu/Hindko newspaper prose but
# do appear as InPage frame/binary residue (e.g. the truncated frame
# "ل رحمان نے ک_"). Stripped individually so surrounding real text survives.
RESIDUE_CHARS_RE = re.compile('[_|~`{}<>\\\\]')

# Arabic tatweel / kashida. Stripped by the InPage decoder already; this is a
# safety net for text arriving from other paths (e.g. OCR).
TATWEEL_RE = re.compile('\u0640+')

# standalone page numbers / folios
PAGE_NO_RE = re.compile(r'^\s*[0-9\u06F0-\u06F9]{1,4}\s*$')

# advertisement / contact signals
#
# Deliberately narrow. An earlier broader list (including فروخت 'sale',
# امپورٹڈ 'imported', پلاٹ 'plot', ڈیلر 'dealer', آفر 'offer') rejected real
# news copy - e.g. a story headlined "ایل پی جی مافیا دی کمر ٹُٹ گئی، قیمت 210
# روپے کلو مقرر" was discarded purely because it discussed LPG being sold.
# Commerce is a normal newspaper topic, so an ad must show *commercial
# intent*: a contact number, an order/discount call-to-action, or several
# advertising-specific phrases together.
PHONE_RE = re.compile(r'(?:\+?92|0)?3[0-9]{2}[\s\-]?[0-9]{7}')

# --------------------------------------------------------------------------
# contact identifiers (PII)
# --------------------------------------------------------------------------
# The book collection brought the first private contact identifiers into the
# training text (QA 2026-09-25: 3 Pakistani mobiles, a Kuwaiti mobile, a
# private office landline, a personal e-mail and an old-style NIC number in 6
# records; the newspaper text had none besides institutional numbers). Every
# contact identifier - private or institutional, which a pattern cannot tell
# apart - is replaced by a typed placeholder; the surrounding text is kept and
# the record is flagged. Digits are matched in Latin, Urdu and Arabic-Indic
# forms, and in the reversed group order right-to-left typesetting produces.
_DIGIT_FOLD = str.maketrans({**{chr(0x06F0 + i): str(i) for i in range(10)},
                             **{chr(0x0660 + i): str(i) for i in range(10)}})
_SEP = r'[\s\-.]?'
PII_PATTERNS = (
    ('<EMAIL>', re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')),
    ('<ID_NUMBER>', re.compile(r'(?<![0-9])[0-9]{5}-[0-9]{7}-[0-9](?![0-9])')),          # CNIC
    ('<ID_NUMBER>', re.compile(r'(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{6}(?![0-9])')),        # old NIC
    # keyword-anchored ID numbers ('شناختی کارڈ نمبر ...', 'CNIC ...')
    ('<ID_NUMBER>', re.compile(r'(?:شناختی\s*کارڈ|CNIC|N\.?I\.?C)[^\n0-9]{0,25}?'
                               r'([A-Za-z]?[+\-]?[0-9][0-9\-]{4,})')),
    ('<PHONE>', re.compile(r'(?<![0-9])(?:\+|00)[0-9]{2,3}' + _SEP + r'[0-9]{6,10}(?![0-9])')),  # international
    ('<PHONE>', re.compile(r'(?<![0-9])(?:92|0)?3[0-9]{2}' + _SEP + r'[0-9]{7}(?![0-9])')),     # PK mobile
    ('<PHONE>', re.compile(r'(?<![0-9])[0-9]{7}' + _SEP + r'0?3[0-9]{2}(?![0-9])')),            # mobile, RTL-reversed
    ('<PHONE>', re.compile(r'(?<![0-9])0[1-9][0-9]{1,3}[\s\-][0-9]{5,8}(?![0-9])')),           # PK landline
    ('<PHONE>', re.compile(r'(?<![0-9])[0-9]{5,8}[\s\-]0[1-9][0-9]{1,3}(?![0-9])')),           # landline, RTL-reversed
)


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Replace contact identifiers with typed placeholders.

    Returns (masked_text, {placeholder: count}). Matching runs on a copy with
    Urdu/Arabic-Indic digits folded to ASCII (same length, so spans map 1:1).
    """
    folded = text.translate(_DIGIT_FOLD)
    spans = []
    for tag, rx in PII_PATTERNS:
        for m in rx.finditer(folded):
            s, e = (m.span(1) if m.lastindex else m.span())
            if not any(s < e2 and s2 < e for s2, e2, _ in spans):
                spans.append((s, e, tag))
    if not spans:
        return text, {}
    counts: Dict[str, int] = {}
    out, pos = [], 0
    for s, e, tag in sorted(spans):
        out.append(text[pos:s])
        out.append(tag)
        pos = e
        counts[tag] = counts.get(tag, 0) + 1
    out.append(text[pos:])
    return ''.join(out), counts
URL_RE = re.compile(r'(?:https?://|www\.)[^\s]+', re.I)
AD_KEYWORDS = (
    'اشتہار', 'تشہیر', 'فری ہوم ڈلیوری', 'آرڈر کریں', 'ابھی آرڈر',
    'خصوصی رعایت', 'رعایت', 'مکان فروخت', 'کرایہ پر دستیاب', 'درکار',
    'call now', 'order now', 'free home delivery', 'discount',
)

# masthead / running-header fragments common to newspaper pages
MASTHEAD_HINTS = ('ہفتہ وار', 'ہندکووان', 'سال', 'جلد', 'شمارہ', 'صفحہ', 'قیمت')

# The running masthead is typed into the InPage page itself, e.g.
#   "ہفت روزہ ہندکو وان (بدھ7 مارچ 2024ئ)"
# Its embedded date differs every issue, so the cross-document boilerplate
# detector (which matches on exact repeated lines) can never catch it. It is
# matched structurally instead. Length-capped so that an article genuinely
# discussing the newspaper is not caught.
MASTHEAD_RE = re.compile(
    r'^\s*(?:ہفت\s*روزہ|ہفتہ\s*وار|روزنامہ)\s*ہندکو\s*وان?', re.U)
MASTHEAD_MAX_LEN = 80


def is_masthead(line: str) -> bool:
    return len(line) <= MASTHEAD_MAX_LEN and bool(MASTHEAD_RE.match(line))


def normalize_unicode(text: str) -> str:
    """NFC-normalise and strip invisible/control noise."""
    text = unicodedata.normalize('NFC', text)
    text = CONTROL_RE.sub('', text)
    # keep ZWNJ (U+200C) which Urdu legitimately uses; drop the rest
    text = re.sub('[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]', '', text)
    text = text.replace('\u200d', '')
    return text


def clean_text(text: str, keep_blank_lines: bool = False) -> str:
    """Normalise one document's text.

    keep_blank_lines=True keeps a single blank line wherever the input had one
    or more (runs collapse to one). Book records use a blank line to separate
    the units (poems, sections) they are merged from; newspaper records never
    contain blank lines, so their output is unchanged by this option.
    """
    if keep_blank_lines:
        blocks = re.split(r'\n\s*\n', text.replace('\r\n', '\n').replace('\r', '\n'))
        cleaned = [clean_text(b) for b in blocks]
        return '\n\n'.join(b for b in cleaned if b)
    text = normalize_unicode(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = RESIDUE_CHARS_RE.sub('', text)
    text = TATWEEL_RE.sub('', text)
    text = WS_RE.sub(' ', text)
    text = PUNCT_REPEAT_RE.sub(r'\1\1', text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r'\1', text)
    text = SPACE_AFTER_PAREN_RE.sub(r'\1', text)
    text = MULTI_NL_RE.sub('\n\n', text)
    lines = [ln.strip() for ln in text.split('\n')]
    lines = [ln for ln in lines if ln and not PAGE_NO_RE.match(ln)]
    return '\n'.join(lines).strip()


def looks_like_ad(text: str) -> bool:
    """Conservative advertisement test.

    A dateline opening ("پشور(ہندکووان نیوز)...") marks wire/reporter copy, so
    datelined text is never treated as an ad however many commerce words it
    contains. Otherwise commercial *intent* is required, not merely a
    commercial topic.
    """
    from .segment import is_dateline

    first = text.lstrip().split('\n', 1)[0]
    if is_dateline(first):
        return False

    low = text.lower()
    hits = sum(1 for k in AD_KEYWORDS if k.lower() in low)
    has_phone = bool(PHONE_RE.search(text))
    has_url = bool(URL_RE.search(text))

    if has_phone and hits >= 1:
        return True
    if hits >= 3:
        return True
    if has_url and hits >= 2:
        return True
    return False


def _is_language_char(c: str) -> bool:
    # Letters, combining marks (harakat), numbers and ALL punctuation count as
    # language; symbols (= + × - Unicode Sm/So), U+FFFD, private-use and control
    # characters do not. The Arabic signs U+0600-U+060F (year sign ؁, verse
    # sign ؎, date separator ؍, footnote marker ؂ ...) are orthographic, not
    # residue, although Unicode files some of them as Cf/So.
    return unicodedata.category(c)[0] in 'LMNP' or '\u0600' <= c <= '\u060f'


def non_language_ratio(text: str) -> float:
    """Share of non-space characters that are not letters, marks, digits or
    punctuation (by Unicode general category).

    Until 2026-09-24 Arabic letters were counted twice (once as ARABIC, once
    again by isalpha()), so the ratio could go negative ('ابپ=' scored -0.5)
    and the gate could not see symbol or U+FFFD density in Arabic-script text
    (verify_books_glyphs.py, G10). The same double count also hid that the
    old fixed punctuation list lacked the Urdu quotation marks ‘‘ ’’, which
    would have pushed 20 strict dialogue-heavy newspaper articles over the
    0.06 gate once the count was corrected - hence category-based now.
    """
    compact = re.sub(r'\s', '', text)
    if not compact:
        return 1.0
    good = sum(1 for c in compact if _is_language_char(c))
    return 1.0 - (good / len(compact))


def max_line_repeat_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) < 2:
        return 0.0
    most = Counter(lines).most_common(1)[0][1]
    return most / len(lines)


def word_count(text: str) -> int:
    return len([w for w in re.split(r'\s+', text) if w])


# --------------------------------------------------------------------------
# boilerplate: repeated headers/footers detected across the corpus
# --------------------------------------------------------------------------
class BoilerplateDetector:
    """Flags lines that recur across many documents (mastheads, folios,
    footers, navigation). Fitted on the whole corpus before cleaning.

    Deliberately conservative. QA over this archive found that the most
    frequently repeated lines are *real syndicated news copy* (the same
    academy-meeting report carried by several issues), not running heads -
    the masthead lives in the CorelDRAW artwork, not in InPage. So the
    detector requires a line to be both very common AND short: mastheads,
    folios and footers are short, article sentences are not. Anything long
    is left for the deduplicator to handle instead.
    """

    def __init__(self, doc_fraction: float = 0.25, min_docs: int = 12,
                 max_len: int = 60):
        self.doc_fraction = doc_fraction
        self.min_docs = min_docs
        self.max_len = max_len
        self.boiler: Dict[str, int] = {}
        self.n_docs = 0

    @staticmethod
    def _key(line: str) -> str:
        return re.sub(r'\s+', ' ', line).strip()

    def fit(self, docs: Iterable[List[str]]) -> 'BoilerplateDetector':
        counts: Counter = Counter()
        self.n_docs = 0
        for lines in docs:
            self.n_docs += 1
            for ln in set(self._key(x) for x in lines):
                counts[ln] += 1
        if self.n_docs:
            thresh = max(self.min_docs, self.doc_fraction * self.n_docs)
            self.boiler = {ln: c for ln, c in counts.items()
                           if c >= thresh and len(ln) <= self.max_len}
        return self

    def is_boilerplate(self, line: str) -> bool:
        return self._key(line) in self.boiler

    def report(self, top: int = 40) -> List[dict]:
        return [{'line': ln, 'docs': c}
                for ln, c in sorted(self.boiler.items(),
                                    key=lambda x: -x[1])[:top]]


def strip_boilerplate(text: str, det: BoilerplateDetector) -> Tuple[str, int]:
    kept, removed = [], 0
    for ln in text.split('\n'):
        if det.is_boilerplate(ln):
            removed += 1
            continue
        kept.append(ln)
    return '\n'.join(kept).strip(), removed


# --------------------------------------------------------------------------
# deduplication: exact + near-duplicate (MinHash LSH)
# --------------------------------------------------------------------------
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _shingles(text: str, k: int = 5) -> set:
    compact = re.sub(r'\s+', '', text)
    if len(compact) < k:
        return {compact} if compact else set()
    return {compact[i:i + k] for i in range(len(compact) - k + 1)}


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(),
                          'big') & _MAX_HASH


class MinHasher:
    """Small dependency-free MinHash + banded LSH for near-duplicate finding."""

    def __init__(self, num_perm: int = 128, bands: int = 16, seed: int = 1):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        rnd = _Lcg(seed)
        self.a = [rnd.next() % _MERSENNE_PRIME for _ in range(num_perm)]
        self.b = [rnd.next() % _MERSENNE_PRIME for _ in range(num_perm)]

    def signature(self, text: str) -> List[int]:
        sh = _shingles(text)
        if not sh:
            return [_MAX_HASH] * self.num_perm
        hashes = [_hash64(s) for s in sh]
        sig = []
        for i in range(self.num_perm):
            ai, bi = self.a[i], self.b[i]
            sig.append(min(((ai * h + bi) % _MERSENNE_PRIME) & _MAX_HASH
                           for h in hashes))
        return sig

    def band_keys(self, sig: List[int]) -> List[str]:
        return [hashlib.blake2b(
            ('%d|%s' % (b, sig[b * self.rows:(b + 1) * self.rows])).encode(),
            digest_size=8).hexdigest() for b in range(self.bands)]


class _Lcg:
    """Deterministic PRNG so signatures are reproducible across runs."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) \
            & 0xFFFFFFFFFFFFFFFF
        return self.state >> 33


def jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    eq = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return eq / len(sig_a)


def _dedup_norm(text: str) -> str:
    return re.sub(r'\s+', ' ', clean_text(text))


# Worker-side state for Deduper.precompute(). Signatures are a pure function
# of (text, num_perm, seed), so computing them in a process pool gives
# byte-identical results to the sequential path - only faster. With the book
# collection the corpus is ~4x larger and signatures dominate the build time.
_POOL_MH: Optional['MinHasher'] = None


def _pool_init(num_perm: int) -> None:
    global _POOL_MH
    _POOL_MH = MinHasher(num_perm=num_perm)


def _pool_signature(norm: str) -> List[int]:
    return _POOL_MH.signature(norm)


class Deduper:
    VERIFY_MIN_EST = 0.70
    VERIFY_TOP = 5

    def __init__(self, near_threshold: float = 0.85, num_perm: int = 128):
        self.near_threshold = near_threshold
        self.num_perm = num_perm
        self.mh = MinHasher(num_perm=num_perm)
        self.seen_exact: Dict[str, str] = {}
        self.buckets: Dict[str, List[int]] = defaultdict(list)
        self.sigs: List[List[int]] = []
        self.ids: List[str] = []
        self.norms: List[str] = []
        self._sig_cache: Dict[str, List[int]] = {}

    def precompute(self, texts: Iterable[str], processes: Optional[int] = None) -> int:
        """Compute MinHash signatures for `texts` in parallel ahead of check().

        Purely a speed-up: check() uses a cached signature when present and
        computes it itself otherwise, so results do not depend on this call.
        Returns the number of signatures computed.
        """
        import multiprocessing as mp
        todo: Dict[str, str] = {}
        for text in texts:
            norm = _dedup_norm(text)
            digest = hashlib.sha256(norm.encode('utf-8')).hexdigest()
            if digest not in self._sig_cache:
                todo[digest] = norm
        if not todo:
            return 0
        digests = list(todo)
        with mp.Pool(processes=processes, initializer=_pool_init,
                     initargs=(self.num_perm,)) as pool:
            sigs = pool.map(_pool_signature, [todo[d] for d in digests],
                            chunksize=16)
        self._sig_cache.update(zip(digests, sigs))
        return len(digests)

    def check(self, doc_id: str, text: str) -> Tuple[str, Optional[str], float]:
        """Returns (status, duplicate_of, similarity).

        status is one of: 'unique', 'exact_duplicate', 'near_duplicate'.
        """
        norm = _dedup_norm(text)
        digest = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        if digest in self.seen_exact:
            return 'exact_duplicate', self.seen_exact[digest], 1.0

        sig = self._sig_cache.pop(digest, None) or self.mh.signature(norm)
        idx = len(self.sigs)
        keys = self.mh.band_keys(sig)
        # LSH candidates are ranked by their MinHash estimate, then VERIFIED
        # with the exact Jaccard of the 5-shingle sets: the estimate alone let
        # 24 of 392 newspaper near-duplicates below 0.85 be deleted and 13
        # pairs above it survive, and made decisions flip whenever a decoder
        # fix touched a few characters (QA 2026-09-25). An estimate below
        # VERIFY_MIN_EST is > 4 standard deviations from 0.85 at 128 perms.
        est = {}
        for key in keys:
            for other in self.buckets.get(key, ()):
                if other not in est:
                    est[other] = jaccard(sig, self.sigs[other])
        best_sim, best_id = 0.0, None
        if est:
            sh = _shingles(norm)
            for other, e in sorted(est.items(), key=lambda x: -x[1])[:self.VERIFY_TOP]:
                if e < self.VERIFY_MIN_EST:
                    break
                osh = _shingles(self.norms[other])
                union = len(sh | osh)
                sim = (len(sh & osh) / union) if union else 0.0
                if sim > best_sim:
                    best_sim, best_id = sim, self.ids[other]

        self.seen_exact[digest] = doc_id
        self.sigs.append(sig)
        self.ids.append(doc_id)
        self.norms.append(norm)
        for key in keys:
            self.buckets[key].append(idx)

        if best_sim >= self.near_threshold:
            return 'near_duplicate', best_id, round(best_sim, 4)
        return 'unique', None, round(best_sim, 4)


# --------------------------------------------------------------------------
# quality gates
# --------------------------------------------------------------------------
STRICT = dict(
    min_chars=200,
    min_words=25,
    min_script_fraction=0.70,
    max_non_language=0.06,
    max_line_repeat=0.60,
    max_unmapped_ratio=0.005,
    require_body=True,
)
PERMISSIVE = dict(
    min_chars=60,
    min_words=8,
    min_script_fraction=0.45,
    max_non_language=0.25,
    max_line_repeat=0.90,
    max_unmapped_ratio=0.05,
    require_body=False,
)


# Roles whose records are built from body text by construction. 'article' is
# the newspaper case (a record containing a body-length frame); the book roles
# come from hp/books.py, which builds records only from classified text units,
# so the newspaper "has a body frame" test does not apply to them. Book roles
# that are not running text (lexicon, toc, english_text) are demoted by
# build.gate_record with their own, more specific reason.
BODY_ROLES = frozenset({'article', 'book_passage', 'front_matter', 'lexicon',
                        'toc', 'english_text',
                        # web records (hp/web.py) are page main text / dataset rows
                        'web_page', 'transcript', 'sentence_set'})


def evaluate(text: str, *, role: str = 'article', unmapped: int = 0,
             is_ad: bool = False) -> Tuple[List[str], dict]:
    """Return (reasons_failed, metrics) for a candidate document."""
    n_chars = len(text)
    n_words = word_count(text)
    sf = script_fraction(text)
    nl = non_language_ratio(text)
    rep = max_line_repeat_ratio(text)
    unmapped_ratio = (unmapped / n_chars) if n_chars else 1.0

    metrics = {
        'n_chars': n_chars,
        'n_words': n_words,
        'script_fraction': round(sf, 4),
        'non_language_ratio': round(nl, 4),
        'max_line_repeat': round(rep, 4),
        'unmapped_glyphs': unmapped,
        'unmapped_ratio': round(unmapped_ratio, 6),
        'is_ad': is_ad,
        'role': role,
    }

    reasons = []
    if n_chars == 0:
        reasons.append('empty')
    if n_chars < PERMISSIVE['min_chars']:
        reasons.append('too_short(%d<%d)' % (n_chars, PERMISSIVE['min_chars']))
    elif n_chars < STRICT['min_chars']:
        reasons.append('below_strict_length(%d<%d)' % (n_chars, STRICT['min_chars']))
    if n_words < PERMISSIVE['min_words']:
        reasons.append('too_few_words(%d)' % n_words)
    elif n_words < STRICT['min_words']:
        reasons.append('below_strict_words(%d)' % n_words)
    if sf < PERMISSIVE['min_script_fraction']:
        reasons.append('low_script_fraction(%.2f)' % sf)
    elif sf < STRICT['min_script_fraction']:
        reasons.append('below_strict_script_fraction(%.2f)' % sf)
    if nl > PERMISSIVE['max_non_language']:
        reasons.append('excess_non_language(%.2f)' % nl)
    elif nl > STRICT['max_non_language']:
        reasons.append('below_strict_non_language(%.2f)' % nl)
    if rep > PERMISSIVE['max_line_repeat']:
        reasons.append('excessive_repetition(%.2f)' % rep)
    elif rep > STRICT['max_line_repeat']:
        reasons.append('below_strict_repetition(%.2f)' % rep)
    if unmapped_ratio > PERMISSIVE['max_unmapped_ratio']:
        reasons.append('corrupt_glyphs(%.4f)' % unmapped_ratio)
    elif unmapped_ratio > STRICT['max_unmapped_ratio']:
        reasons.append('below_strict_glyph_purity(%.4f)' % unmapped_ratio)
    if is_ad:
        reasons.append('advertisement')
    if role not in BODY_ROLES and STRICT['require_body']:
        reasons.append('no_body_frame(role=%s)' % role)

    return reasons, metrics


def tier(reasons: List[str]) -> str:
    """Classify gate outcome.

    'strict'      -> passes the research/pretraining-grade gate
    'permissive'  -> usable but flagged; fails only soft strict thresholds
    'reject'      -> fails a hard permissive threshold
    """
    if not reasons:
        return 'strict'
    hard = ('empty', 'too_short', 'too_few_words', 'low_script_fraction',
            'excess_non_language', 'excessive_repetition', 'corrupt_glyphs',
            'advertisement')
    for r in reasons:
        base = r.split('(')[0]
        if base in hard:
            return 'reject'
    return 'permissive'
