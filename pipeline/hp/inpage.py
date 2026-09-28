"""InPage (.inp / .B01) text extraction.

Weekly Hindkowan was typeset in InPage, whose .inp files are OLE compound
documents. The payload stream ("InPage100") stores text as \\x04-prefixed
byte pairs in a legacy glyph encoding (Noori Nastaliq / ZoharSindhi), with
Latin digits and letters stored inline as bare ASCII bytes, and structural
control bytes interleaved between text frames.

Decoding therefore has three parts:
  1. read the OLE payload stream
  2. re-encode to the \\x04 pair form the mapping table expects, preserving
     bare ASCII inline (this is what keeps prices, years and dates intact)
  3. run the vendored glyph map, which also applies InPage's contextual
     rules (medial bari-yee, hamza combinations, diacritics)

Glyph mapping table (GPL-3.0, fetched separately) from
https://github.com/salmanasmat/InPageToUnicode (mapping/glyph_map.py),
which itself ports ltrc/inPageToUnicode and umer0586/unicode-inpage-converter.
See pipeline/README.md.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import List, Optional

import olefile

from . import glyph_map

# Glyphs observed in the Hindkowan archive that the vendored table lacks.
# Each was determined from evidence in the source documents, not guessed;
# the supporting context is recorded beside it.
EXTRA_GLYPHS = {
    # Confirmed by "آخر" and "آ گئی" in 262/Idariya.inp.
    '\x04\xc8': '\u0622',                       # آ
    # Verified across 27 occurrences in 94 documents: every single one is
    # immediately preceded by \x04\x81 (ا), and the surrounding text is a
    # fixed formula or name that only reads correctly as ا + للہ = اللہ:
    #   بیت ا?   -> بیت اللہ        رسول ا?  -> رسول اللہ
    #   انشائ ا? -> انشاء اللہ       ا? تعالیٰ -> اللہ تعالیٰ
    #   سمیع ا? خان -> سمیع اللہ خان   ارباب شفیع ا? -> ارباب شفیع اللہ
    # Mapped to the three decomposed letters (not the U+FDF2 ligature) so the
    # corpus stays in the form Urdu NLP tooling expects - with Urdu heh goal
    # U+06C1, as this comment always said. Until 2026-09-25 the code emitted
    # Arabic heh U+0647 instead (291 newspaper / 1,832 book occurrences, the
    # only U+0647 in the whole corpus), so the ligature's اللہ never matched
    # the letter-by-letter typed اللہ (2,024 in the newspaper). See also the
    # space restoration after CB in decode_payload.
    '\x04\xcb': '\u0644\u0644\u06c1',           # للہ
    # (\x04\xe0 was mapped to the full stop here on 2026-09-20; it is an
    #  ellipsis - see its entry further down.)
    # WAW WITH HAMZA ABOVE. Confirmed by loanwords that only read one way:
    # "سلائٹر ہا?س" -> ہاؤس (slaughter house), "بیک گرا?نڈ" -> گراؤنڈ
    # (background). Preceded by \x04\x81 (ا) in 13/13 in-content cases,
    # matching the اؤ sequence. Upstream reaches ؤ only via the two-byte
    # \x04\xa2\x04\xbf sequence; this is the single-byte form.
    '\x04\xb6': '\u0624',                       # ؤ
    # Kashida / tatweel (ـ), used for line justification in Nastaliq. It is in
    # the mapper's MANUAL_KEYS, so it must be recognised here or the decoder
    # turns it into U+FFFD and the mapper's own kashida handling never runs.
    # The value below is not used directly - see remove_kashida in
    # decode_payload.
    '\x04\xa9': '\u0640',                       # ـ

    # ---- resolved 2026-09-24 from the book collection (probe_books_glyphs*.py,
    # ---- verify_books_glyphs.py). Each was left unmapped (U+FFFD) before.
    # Every reading below was reached from the documents' own internal
    # evidence; the upstream ltrc/inPageToUnicode table and the Drive TXT
    # exports agree with all of them, which was treated as corroboration only.
    #
    # EQUALS. 9,726 in-content occurrences. 27 distinct arithmetic chains in
    # the Quran-19 book verify only with '=' (e.g. 2521+1892+1249=5662), and
    # 65 distinct Names of Allah are followed by their exact abjad value
    # (الغفار=۱۲۸۱, المجید=۵۷, القدوس=۱۷۰). Contradicting contexts: 0.
    '\x04\xec': '=',
    # ORNATE PARENTHESES. 46 of 46 in-content pairs match with DC first on the
    # same line, enclosing Quranic items: ﴿سورۃ آلِ عمران﴾, آیت نمبر ﴿۳۰﴾.
    # U+FD3F is the opening and U+FD3E the closing bracket (Unicode 14).
    '\x04\xdc': '\ufd3f',                       # ORNATE LEFT PARENTHESIS
    '\x04\xdb': '\ufd3e',                       # ORNATE RIGHT PARENTHESIS
    # HYPHEN/DASH. 680 in-content occurrences: 406 in the Urdu ':-' convention
    # (حرفی نمبر:- 1), year ranges (323-356), phone numbers such as 03xx-xxxxxxx,
    # 'والیو-لا-شی (Valu-la-Shi)', سائینو-تبتن (Sino-Tibetan). In 306th the
    # same typist uses it 214x between clauses but the real ۔ 1,198x, so it is
    # not a third encoding of the full stop. Contradicting contexts: 0.
    # (F5 is handled after the mapper - see POST_MAPPER_GLYPHS - so that
    #  Urdu-digit runs joined by it keep their whole-run order.)
    # SLASH. Must go THROUGH the mapper (like \x04\xf1) so its digit-run
    # reversal orders fractions correctly: 39 of 39 Quranic inheritance shares
    # glossed in words ('ایک بٹہ دو (۱/۲)') come out numerator-first, and the
    # sequence matches Quran 4:11-12/4:176 for 12 of 13 (the 13th disagrees in
    # the author's words too). Also پروڈیوسر/ڈائریکٹر, newspaper ٹی اے/ڈی اے.
    '\x04\xdf': '/',
    # ARABIC FIVE-POINTED STAR, a bullet/ornament: 0 of 134 book and 0 of 11
    # newspaper occurrences are inside a word; 84 line-initial bullets,
    # '٭ کٹ ٭' scene cuts, footnote pairs, '٭٭٭' section breaks. This replaces
    # the earlier 'stretch glyph in poetry' note: that sample came from a probe
    # that still showed unmapped kashida as '?'. Decoded now, the newspaper
    # heading reads '٭٭بیادِ رفتگاں٭٭' with its words intact.
    '\x04\xe8': '\u066d',                       # ARABIC FIVE POINTED STAR

    # ---- decided 2026-09-24/25 by two independent adjudicators each (orthography
    # ---- lens + glyph-byte lens), after the Drive TXT exports exposed systematic
    # ---- differences. Reports: _books_cache/wf/adjudication_all.txt, scripts
    # ---- probe_adj_*.py. Each change was diffed over every newspaper + book file.
    #
    # HAMZA. The vendored table maps \xa3 to ئ (U+0626), although its own reverse
    # table sends ء to \xa3 and its contextual rules ('ء + letter -> ئ', 'ءء',
    # the year-sign reorder 'ء؁ -> ؁ء') only work if \xa3 first becomes ء.
    # Upstream ltrc maps \xa3 to ء. With this value the mapper's own rules give
    # ئ before letters and ء elsewhere: 'ضیاء', 'ماء', '2017ء', 'انشاء اللہ',
    # 'شعراء'. Before, the corpus contained no U+0621 at all and 17,786 book /
    # 5,263 newspaper word-final or isolated ئ. Words that genuinely end in a
    # seated hamza (البارئ, یستهزئ) are typed with \xa4\xbf, never \xa3.
    # See HAMZA_SEAT_RE for hamza followed by more than one mark.
    '\x04\xa3': '\u0621',
    # YEH + HAMZA MARK. \xa4 is ی and \xbf the hamza-above modifier: the pair is
    # ONE yeh carrying hamza, i.e. ئ (one tooth). The vendored 'یئ' (two teeth)
    # agreed with the TXT export in only 4.9% of 1,158 book tokens. Genuinely
    # two-tooth spellings are typed \xa4\xa3 and are unaffected.
    '\x04\xa4\x04\xbf': '\u0626',
    # ELLIPSIS - overturns the 2026-09-20 reading as a second full stop (۔).
    # E0 and the full stop F3 are different marks used side by side in the same
    # paragraphs: F3 ends 25.1% of book paragraphs it occurs in, E0 1.8%
    # (newspaper 19.1% vs 0.15%, lower for E0 in 21 of 21 files); the next word
    # after E0 is تے/مگر/کہ/کیونکہ 10-41x more often; E0 sits between a noun and
    # its case marker ('دادا جان… نے', 'قاضی محمد اسلم وکیل… نوں') where a full
    # stop is impossible; '……۔' (E0 then F3) closes 44 paragraphs; 95.5% of
    # runs of 3+ E0 are dotted separator lines. One U+2026 per glyph keeps the
    # mapping reversible and run lengths intact (upstream ltrc/TXT write two).
    '\x04\xe0': '\u2026',
    # SQUARE BRACKETS - the vendored table has the pair mirrored. With fb='['
    # and fa=']' 18,727 of 19,216 book lines nest properly (0 with the vendored
    # values), fb comes first on a line 19,015 times vs 201, and the contents
    # become the dictionaries' closed label set: [مثل] 2,383, [ہ۔ا۔مذکر] 2,128,
    # [ع۔ا۔مذکر] 1,758 ... Upstream ltrc has fa ']' and fb '['.
    '\x04\xfa': ']',
    '\x04\xfb': '[',
}

# Glyphs substituted AFTER the vendored mapper. Its digit-run reversal treats
# '+' and '×' as part of a number, so mapping these before it runs would
# reverse operand order in 186 digit-adjacent places and break every abjad
# alignment (و+ا+ح+د = ۴+۸+۱+۶ instead of ۶+۱+۸+۴). The walk emits a
# private-use placeholder, which survives the mapper unchanged (verified in
# 139 of 139 book files), and it is translated afterwards.
#   PLUS  - 13 of 15 distinct chains verify only under '+' (98+54+57=209);
#           the other 2 are arithmetic slips in the book itself.
#   TIMES - 19 of 25 chains verify only under '×' (19×298=5662); in the other
#           6 the product is exact and the typo is elsewhere in the chain.
POST_MAPPER_GLYPHS = {
    0xE4: ('\ue0e4', '+'),
    # HYPHEN/DASH (evidence in the EXTRA_GLYPHS comment above). Post-mapper so
    # that DASH_DIGIT_RUN_RE can restore whole-run order in Urdu-digit runs.
    0xF5: ('\ue0f5', '-'),
    0xEB: ('\ue0eb', '\u00d7'),                 # MULTIPLICATION SIGN
}
_POST_MAPPER_TRANSLATE = str.maketrans(
    {ph: ch for ph, ch in POST_MAPPER_GLYPHS.values()})

# The mapper reverses each Urdu-digit group on its own, because '-' is not in
# its digit-run class, while a run joined by '/' (DF, F1) is reversed as a
# whole. So a dash-joined run came out with its groups in reverse order:
# 6 of 6 such runs in the books read xxxxxxx-03xx (in Urdu digits) for 03xx-xxxxxxx and
# (13-2012) for 2012-13 (verify_books_glyphs.py, G5). Reversing the group order
# restores whole-run order. ASCII digits are never reversed by the mapper, so
# runs such as (460-520) are already right and are not touched.
_DASH_PH = POST_MAPPER_GLYPHS[0xF5][0]
DASH_DIGIT_RUN_RE = re.compile('[\u06f0-\u06f9]+(?:' + _DASH_PH +
                               '[\u06f0-\u06f9]+)+')


def _restore_dash_digit_runs(text: str) -> str:
    return DASH_DIGIT_RUN_RE.sub(
        lambda m: _DASH_PH.join(reversed(m.group(0).split(_DASH_PH))), text)


# HAMZA SEAT completion. The mapper turns ء into ئ before a letter and before
# ONE mark from its short list, but not before sukun, maddah or two or more
# marks. With \xa3 now decoding to ء, that left 12 medial Quranic seats as ء
# (جِئْتُمْ, ما شِئْتَ, شَعَآئِرِ, الصّٰٓئِمٰتِ, اُولٰٓئِکَ, الْمُطْمَئِنَّةُ). At least
# one mark is required: with zero marks it would turn glued years such as
# '۱۹۴۱؁ءوچ' into ئ. With this rule the output equals, over all 306,029 \xa3
# tokens, the byte rule "ئ when the next glyph (skipping marks) is a letter".
_MAPPER_ALPHABET = ('ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکكگلمنوئیےؤهۂةأـآيھإہۃں')
HAMZA_SEAT_RE = re.compile(
    '\u0621(?=[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]+[' + _MAPPER_ALPHABET + '])')

# HINDKO TONE LETTERS. \x04\x3b sits on the voiceless stops پ ت ٹ چ ک in 60.6%
# (dictionaries) / 53.9% (other books) of its uses, word-initially in 84.9%,
# marking the Hindko low tone of old voiced aspirates: the Lughat respells 227
# such headwords with the aspirate (ت;اگا = دَھاگا, ٹ;ینا = ڈَھے۔نا) and glosses
# پ;ابی as بھابھی. Unicode 13 encodes exactly these letters for Hindko
# (U+08BE..U+08C2, "... WITH SMALL V"). The walk emits U+065A SMALL V ABOVE
# and it is composed here; NFC does not do it (no canonical decomposition).
# On other letters (where the dictionaries use the same glyph as a jazm) the
# combining U+065A stays, which is what the page shows. Dropping the mark
# instead, as the TXT export does, would merge distinct Hindko words.
SMALL_V = '\u065a'
SMALL_V_LETTERS = {'\u067e': '\u08be', '\u062a': '\u08bf', '\u0679': '\u08c0',
                   '\u0686': '\u08c1', '\u06a9': '\u08c2'}
SMALL_V_RE = re.compile('([\u067e\u062a\u0679\u0686\u06a9])([\u064b-\u0652\u0670]*)\u065a')


def _compose_small_v(text: str) -> str:
    return SMALL_V_RE.sub(lambda m: SMALL_V_LETTERS[m.group(1)] + m.group(2), text)


# InPage starts every paragraph record with 0x0D and a 32-bit little-endian
# byte count (a second form, 0x0D 00000000 [00|01] <u32>, occurs in ~12k
# paragraphs of 39th, 131st, 10th ...). When a byte of that count is
# printable ASCII the walk emitted it as a lone character 'frame' - 302,688
# such frames in the 207 book files, 23,640 of them digits that look like
# verse or page numbers. Neutralising the field at byte level changes nothing
# longer than 2 characters across all 207 book files (verify_books_layout_
# nolen.py), whereas stripping lone characters at string level would also
# delete 3,610 real ':' frames, 2,794 '=' frames and 1,118 counted U+FFFD
# frames. Only the two count bytes that can be printable are zeroed.
PARA_FIELD_RE = re.compile(rb'(?<!\x04)\x0d(.)(.)\x00\x00', re.S)
PARA_FIELD2_RE = re.compile(
    rb'(?<!\x04)\x0d\x00\x00\x00\x00[\x00\x01](.)(.)\x00\x00', re.S)


def neutralise_paragraph_length_fields(raw: bytes) -> tuple:
    """Zero the printable count bytes of every paragraph header.

    Returns (bytes, n_fields, n_bytes_zeroed).
    """
    buf = bytearray(raw)
    n_fields = n_zeroed = 0
    for rx in (PARA_FIELD_RE, PARA_FIELD2_RE):
        for m in rx.finditer(raw):
            n_fields += 1
            for pos in (m.start(1), m.start(2)):
                if 0x20 <= buf[pos] <= 0x7E:
                    buf[pos] = 0
                    n_zeroed += 1
    return bytes(buf), n_fields, n_zeroed

# \x04 + one of these is a structural marker (text-frame / paragraph
# boundary), not a glyph. The whole C0 control range qualifies: QA over the
# archive showed \x04\x06, \x04\x0e, \x04\x10, \x04\x09 etc. accounting for
# ~82% of what initially looked like "unmapped glyphs".
CONTROL_BYTES = frozenset(range(0x00, 0x20)) | frozenset({0x7C, 0x7E, 0x7F})

# Byte pairs seen in real text that are deliberately LEFT UNMAPPED, so they
# surface as U+FFFD and are counted rather than guessed. The five newspaper
# entries of 2026-09-20 (E8 DF F5 DC DB) were resolved on 2026-09-24 with the
# far larger book evidence - see EXTRA_GLYPHS above. What remains:
UNRESOLVED_GLYPHS = {
    0xB7: {'in_content': 19,
           'note': 'one author only (19th Farma Pohl Parotey, BOOK + FEHRIST). '
                   'Every occurrence is word-final after a letter and the '
                   'contexts fit ئ, ی and ے equally (the same author writes '
                   'مرثی with ی and ایوئی with ئ). Upstream ltrc says ئ; that '
                   'is not enough to meet the project bar, and a wrong guess '
                   'changes words. Revisit with a print/page image.',
           'samples': ['واقع?', 'مرث?', 'کیونکہ شافع? روزِ جزا آیا',
                       'بولدی ا?۔', 'پونچدی ر? اے']},
}

# Strings InPage writes into the document itself; they decode as text but are
# application metadata, not article content.
KNOWN_NOISE = frozenset({
    'InPage Arabic Document', 'Normal', 'Noori Nastaliq', 'Naskh',
    'Noori Character', 'Arial', 'Simplified Arabic', 'ZoharSindhi',
    'None', 'White', 'Black', 'Gray', 'Grey', 'Red', 'Yellow', 'Green',
    'Cyan', 'Blue', 'Magenta',
})

UNMAPPED = '\ufffd'

_patched = False


def _patch_glyph_map() -> List[str]:
    """Merge EXTRA_GLYPHS into the vendored table and rebuild its matcher."""
    global _patched
    added = [k for k in EXTRA_GLYPHS if k not in glyph_map.ITU_MAP]
    glyph_map.ITU_MAP.update(EXTRA_GLYPHS)
    if added:
        keys = sorted((k for k in glyph_map.ITU_MAP
                       if k not in glyph_map.MANUAL_KEYS),
                      key=len, reverse=True)
        glyph_map.ITU_REGEX = re.compile('|'.join(re.escape(k) for k in keys))
    _patched = True
    return added


_patch_glyph_map()


def _valid_second_bytes() -> frozenset:
    """Every byte that may follow \\x04 in the mapping table.

    Built from whole keys rather than single pairs, so the second half of a
    multi-byte mapping (e.g. \\x04\\x81\\x04\\x08) is not mistaken for an
    unmapped glyph.
    """
    out = set()
    for key in glyph_map.ITU_MAP:
        for i in range(0, len(key) - 1, 2):
            if key[i] == '\x04':
                out.add(ord(key[i + 1]))
    return frozenset(out)


VALID_SECOND_BYTES = _valid_second_bytes()


def _single_char_value(key: str):
    v = glyph_map.ITU_MAP.get(key)
    return v if v and len(v) == 1 else None


# Glyph bytes whose table value is one letter / one combining mark. Used by the
# walk to decide whether a mark-like pair is attached to a base letter.
LETTER_BYTES = frozenset(
    b for b in range(0x80, 0x100)
    if (_single_char_value('\x04' + chr(b)) or ' ').isalpha())
MARK_BYTES = frozenset(
    b for b in range(0x80, 0x100)
    if unicodedata.category(_single_char_value('\x04' + chr(b)) or ' ')[0] == 'M'
) | {0xB4, 0xBF}


def payload_stream(ole: olefile.OleFileIO) -> Optional[str]:
    """Name of the stream holding the document text."""
    names = ['/'.join(entry) for entry in ole.listdir()]
    for name in names:
        if name.startswith('InPage'):
            return name
    for name in names:
        if 'Document' in name:
            return name
    return names[0] if names else None


SUKUN = '\u0652'


def _attached(out: List[str]) -> bool:
    """True when the last emitted piece is a letter or a mark on one."""
    if not out:
        return False
    last = out[-1]
    if last in (SUKUN, SMALL_V):
        return True
    return (len(last) == 2 and last[0] == '\x04'
            and (ord(last[1]) in LETTER_BYTES or ord(last[1]) in MARK_BYTES))


def decode_payload(raw: bytes) -> tuple:
    """Turn raw payload bytes into Unicode text.

    Control bytes become newlines so that each surviving line corresponds to
    one InPage text frame (in practice: one headline, subhead or body block).
    Glyph pairs the mapping table does not know become U+FFFD and are counted,
    so corruption is surfaced rather than silently passed through as a stray
    latin-1 character.

    Returns (text, unmapped_count).
    """
    out: List[str] = []
    unmapped = 0
    space_before_letter = False
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == 0x04 and i + 1 < n:
            b = raw[i + 1]
            if b in CONTROL_BYTES:
                space_before_letter = False
                out.append('\n')
                i += 2
                continue
            if b in POST_MAPPER_GLYPHS:
                out.append(POST_MAPPER_GLYPHS[b][0])
                space_before_letter = False
                i += 2
                continue
            if b in (0x3A, 0x3B):
                # JAZM (3A) and HINDKO TONE MARK (3B), not ':' / ';' - see
                # SMALL_V_RE and the note above POST_MAPPER_GLYPHS. Kept only
                # when attached to a base letter; orphans (no letter before
                # them: 668 + 80 in the books) are dropped.
                if _attached(out):
                    out.append(SUKUN if b == 0x3A else SMALL_V)
                i += 2
                continue
            if b == 0xB4:
                # ROUND SUKUN glyph. On noon (428 book tokens, 0 newspaper) it
                # marks the retroflex nasal in pronunciation keys and verse
                # ('کہانْڑیاں') and becomes the table's sukun \x04\xb1. Every
                # other use is ornamental - ayah circles, degree-like signs,
                # filler runs, stray marks on alif - and stays dropped, as the
                # vendored table ('') always did.
                if out and out[-1] == '\x04\xa0':
                    out.append('\x04\xb1')
                i += 2
                continue
            if b in VALID_SECOND_BYTES:
                if space_before_letter and b in LETTER_BYTES and b != 0xF6:
                    # the word after the ligature was glued to it
                    out.append('\x04 ')
                if b not in MARK_BYTES:
                    space_before_letter = False
                out.append('\x04' + chr(b))
                if b == 0xCB:
                    # ALLAH LIGATURE glued to the next word ('اللهنے',
                    # 'اللهتعالیٰ': 84 book + 3 newspaper sites): the ligature
                    # is non-joining, so a letter right after it (marks
                    # skipped) means the typist's space was lost. Restored at
                    # byte level; typed اللہ + letter (اللہب, ولی اللہی,
                    # اَللہُمَّ) is never touched.
                    space_before_letter = True
                i += 2
                continue
            if 0x20 <= b <= 0x7E:
                # A structural \x04 sitting immediately before literal ASCII.
                # Dropping only the marker keeps the character; treating the
                # pair as a glyph would silently eat real text (observed for
                # \x04'0', \x04'N', ...). \x04':' and \x04';' are handled
                # above: they are diacritics, not punctuation.
                i += 1
                continue
            out.append(UNMAPPED)
            unmapped += 1
            space_before_letter = False
            i += 2
            continue
        space_before_letter = False
        if 0x20 <= c <= 0x7E:
            out.append(chr(c))     # inline ASCII: digits, latin, punctuation
            i += 1
            continue
        out.append('\n')           # any other byte is structural
        i += 1

    options = dict(glyph_map.DEFAULT_OPTIONS)
    # Kashida (tatweel) is Nastaliq line-justification, not language: it occurs
    # in long self-repeating runs (\x04\xa9 preceded by \x04\xa9 391 times) to
    # stretch verse to the column width. It carries no linguistic information
    # and badly fragments tokens, so it is dropped for an NLP corpus.
    options['remove_kashida'] = True
    text = glyph_map.inpage_to_unicode(''.join(out), options)
    # U+FFFD survives the mapper: it is not \x04-prefixed in the table, so the
    # \x04 the mapper's preprocessing adds is stripped again afterwards. The
    # POST_MAPPER_GLYPHS placeholders survive the same way.
    text = HAMZA_SEAT_RE.sub('\u0626', text)
    text = _compose_small_v(text)
    text = _restore_dash_digit_runs(text)
    text = text.translate(_POST_MAPPER_TRANSLATE)
    return text, unmapped


def split_lines(text: str) -> List[str]:
    """Normalise decoded text into candidate content lines."""
    lines = []
    for line in text.split('\n'):
        line = line.replace('\x00', '')
        line = re.sub(r'[ \t\r\f\v]+', ' ', line).strip()
        if line:
            lines.append(line)
    return lines


def is_noise(line: str) -> bool:
    if line in KNOWN_NOISE:
        return True
    stripped = re.sub(r'[^A-Za-z]', '', line)
    return bool(stripped) and stripped in KNOWN_NOISE


class ExtractionError(Exception):
    pass


def extract_file(path: str) -> dict:
    """Extract an InPage document. Returns a dict with text + diagnostics."""
    if not olefile.isOleFile(path):
        raise ExtractionError('not an OLE compound document')

    ole = olefile.OleFileIO(path)
    try:
        stream = payload_stream(ole)
        if stream is None:
            raise ExtractionError('no streams in OLE container')
        raw = ole.openstream(stream.split('/')).read()
        streams = ['/'.join(e) for e in ole.listdir()]
    finally:
        ole.close()

    if not raw:
        raise ExtractionError('payload stream %r is empty' % stream)

    raw, n_para_fields, n_field_bytes = neutralise_paragraph_length_fields(raw)
    text, unmapped = decode_payload(raw)
    lines = split_lines(text)
    content = [ln for ln in lines if not is_noise(ln)]

    return {
        'stream': stream,
        'streams': streams,
        'payload_bytes': len(raw),
        'text': '\n'.join(content),
        'text_all': '\n'.join(lines),
        'n_lines_all': len(lines),
        'n_lines_content': len(content),
        'n_noise_lines': len(lines) - len(content),
        'unmapped_glyphs': unmapped,
        'unmapped_in_text': text.count(UNMAPPED),
        'paragraph_fields': n_para_fields,
        'paragraph_field_bytes_zeroed': n_field_bytes,
    }


def file_kind(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return ext if ext in ('inp', 'b01') else None
