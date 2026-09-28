"""Unit tests for F:/Hindko/_pipeline/hp/normalize.py.

Run:  PYTHONIOENCODING=utf-8 python test_normalize.py            (unit tests + 3 fuzz sets)
      PYTHONIOENCODING=utf-8 python test_normalize.py --corpus   (also every corpus record)

Every codepoint is written as chr(0x....) through the helper u() so that the
file stays pure ASCII and cannot be altered by an editor.
Deterministic: the fuzz sets use fixed seeds (20260926, 20260927, 20260928).

The fixed-point check is stronger than normalize(normalize(x)) == normalize(x):
normalize_with_report(normalize(x)) must return the same text AND an empty
report, i.e. no rule changes anything in canonical text.

Fuzz sets:
  basic    60,000 strings from a 104-item pool of corpus-relevant characters
           (the v1.0.0 set; it missed the v1.0.0 idempotency bug)
  focused 400,000 strings of ZWJ/ZWNJ, bases that compose with marks
           (Latin, '=' '<' '>' and arrows for U+0338, Arabic letters that take
           hamza/madda, Devanagari), non-Arabic combining marks of many
           combining classes (all of U+0300-036F, U+093C, U+094D, Hebrew
           points, ...), Arabic marks and every presentation-form haraka
           (FE70-FE7F, FC5E-FC63, FCF2-FCF4)
  broad   300,000 strings over every assigned codepoint of the Arabic blocks
           (0600-06FF, 0750-077F, 0870-089F, 08A0-08FF, FB50-FDFF, FE70-FEFF)
           plus the focused pool
"""
import json
import random
import sys
import unicodedata

sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import normalize as N  # noqa: E402


def u(*cps):
    return ''.join(chr(c) for c in cps)


# letters
ALEF, BEH, TEH, DAL, REH, SEEN, AIN, LAM, MEEM, NOON, WAW = (u(0x0627), u(0x0628), u(0x062A), u(0x062F), u(0x0631),
                                                           u(0x0633), u(0x0639), u(0x0644), u(0x0645), u(0x0646),
                                                           u(0x0648))
AR_YEH, ALEF_MAKSURA, AR_KAF, AR_HEH, TEH_MARBUTA = u(0x064A), u(0x0649), u(0x0643), u(0x0647), u(0x0629)
FA_YEH, KEHEH, HEH_GOAL, HEH_DO, TEH_MARBUTA_GOAL = u(0x06CC), u(0x06A9), u(0x06C1), u(0x06BE), u(0x06C3)
YEH_BARREE, NOON_GHUNNA, PEH, TTEH, TCHEH, GAF, RREH = (u(0x06D2), u(0x06BA), u(0x067E), u(0x0679), u(0x0686),
                                                        u(0x06AF), u(0x0691))
YEH_HAMZA, HIGH_HAMZA_YEH, HAMZA, HAMZA_ABOVE = u(0x0626), u(0x0678), u(0x0621), u(0x0654)
NOON_SMALL_TAH = u(0x0768)
PASHTO_SHEEN = u(0x069A)     # seen with dot below and dot above (Pashto)
SINDHI_KAF = u(0x06AA)
SARAIKI_DDAL = u(0x0759)
# marks
FATHA, DAMMA, KASRA, SHADDA, SUKUN, SMALL_V = u(0x064E), u(0x064F), u(0x0650), u(0x0651), u(0x0652), u(0x065A)
# others
ZWNJ, ZWJ, ZWSP, NBSP, LRM, RLM, ALM, BOM, SHY, CGJ, WJ = (u(0x200C), u(0x200D), u(0x200B), u(0x00A0), u(0x200E),
                                                           u(0x200F), u(0x061C), u(0xFEFF), u(0x00AD), u(0x034F),
                                                           u(0x2060))
KASHIDA = u(0x0640)
FULL_STOP, COMMA = u(0x06D4), u(0x060C)
SAW, ALLAH_LIG, ORNATE_L, ORNATE_R, BISMILLAH = u(0xFDFA), u(0xFDF2), u(0xFD3E), u(0xFD3F), u(0xFDFD)
URDU_DIGITS = u(*range(0x06F0, 0x06FA))
ARABIC_DIGITS = u(*range(0x0660, 0x066A))
TONE = {PEH: u(0x08BE), TEH: u(0x08BF), TTEH: u(0x08C0), TCHEH: u(0x08C1), KEHEH: u(0x08C2)}

ALLAH = ALEF + LAM + LAM + HEH_GOAL

FAILS = []
NCHECK = 0


def check(name, got, want):
    global NCHECK
    NCHECK += 1
    if got != want:
        FAILS.append(name)
        print('FAIL %-55s got=%r want=%r' % (name, got, want))


def n(s):
    return N.normalize(s)


def rep(s):
    return N.change_report(s)


# ---------------------------------------------------------------------------
def test_r01_nfc():
    # alef + maddah above composes to ALEF WITH MADDA ABOVE
    check('R01 alef+maddah -> U+0622', n(ALEF + u(0x0653)), u(0x0622))
    # heh goal + hamza above -> U+06C2 ; yeh barree + hamza above -> U+06D3
    check('R01 heh goal+hamza -> U+06C2', n(HEH_GOAL + HAMZA_ABOVE), u(0x06C2))
    check('R01 yeh barree+hamza -> U+06D3', n(YEH_BARREE + HAMZA_ABOVE), u(0x06D3))
    # FARSI YEH + hamza above has no precomposed form and stays two codepoints
    check('R01 farsi yeh+hamza stays', n(FA_YEH + HAMZA_ABOVE), FA_YEH + HAMZA_ABOVE)
    # mark order: shadda typed before fatha is reordered (fatha ccc 30 < shadda 33)
    check('R01 shadda+fatha reordered', n(BEH + SHADDA + FATHA), BEH + FATHA + SHADDA)
    check('R01 report', rep(BEH + SHADDA + FATHA).get('R01_nfc', 0) > 0, True)
    # Latin
    check('R01 latin e+acute', n('e' + u(0x0301)), u(0x00E9))


def test_r02_newlines():
    check('R02 CRLF', n('a\r\nb'), 'a\nb')
    check('R02 CR', n('a\rb'), 'a\nb')
    check('R02 LS/PS/NEL/VT/FF', n('a' + u(0x2028) + 'b' + u(0x2029) + 'c' + u(0x85) + 'd' + u(0x0B) + 'e' + u(0x0C) + 'f'),
          'a\nb\nc\nd\ne\nf')


def test_r03_controls():
    check('R03 C0 removed', n('a' + u(0x01) + 'b' + u(0x1F)), 'ab')
    check('R03 DEL removed', n('a' + u(0x7F) + 'b'), 'ab')
    # the corpus case: U+0095 (a mis-decoded cp1252 bullet) always followed by a space
    check('R03 C1 U+0095 removed', n(BEH + ALEF + u(0x95) + ' ' + GAF), BEH + ALEF + ' ' + GAF)
    check('R03 TAB not a control (becomes space in R09)', n('a\tb'), 'a b')
    check('R03 LF kept', n('a\nb'), 'a\nb')


def test_r04_presentation():
    # isolated/final/initial/medial letter forms -> base letters
    check('R04 FARSI YEH isolated', n(u(0xFBFC)), FA_YEH)
    check('R04 HEH GOAL forms', n(u(0xFBA6) + u(0xFBA7) + u(0xFBA8) + u(0xFBA9)), HEH_GOAL * 4)
    check('R04 HEH DOACHASHMEE forms stay do-chashmi', n(u(0xFBAA) + u(0xFBAD)), HEH_DO * 2)
    check('R04 KEHEH forms', n(u(0xFB8E)), KEHEH)
    check('R04 lam-alef ligature', n(u(0xFEFB)), LAM + ALEF)
    check('R04 lam-alef-madda ligature', n(u(0xFEF5)), LAM + u(0x0622))
    # a word typed entirely in presentation forms: meem-initial, alef-final, noon-isolated? -> letters
    check('R04 word in forms', n(u(0xFEE3) + u(0xFE8E) + u(0xFEE5)), MEEM + ALEF + NOON)
    # Arabic YEH presentation forms become ARABIC YEH, then R11 folds them only in a proven word
    check('R04+R11 yeh form in Urdu word', n(u(0xFEF3) + NOON_GHUNNA), FA_YEH + NOON_GHUNNA)
    check('R04 yeh form alone stays Arabic yeh', n(u(0xFEF1)), AR_YEH)
    # Arabic HEH forms become ARABIC HEH (never HEH GOAL or DOACHASHMEE)
    check('R04 Arabic heh form -> U+0647', n(u(0xFEE9)), AR_HEH)
    # spacing / tatweel forms of harakat -> bare mark (no space inserted inside a word)
    check('R04 fatha isolated form', n(BEH + u(0xFE76)), BEH + FATHA)
    check('R04 fatha medial form', n(BEH + u(0xFE77)), BEH + FATHA)
    check('R04 shadda+dammatan ligature', n(BEH + u(0xFC5E)), BEH + u(0x064C) + SHADDA)
    # ALLAH ligature -> the corpus spelling with HEH GOAL; a preceding alef is absorbed
    check('R04 ALLAH ligature', n(ALLAH_LIG), ALLAH)
    check('R04 alef+ALLAH ligature (corpus: 14 of 15)', n(ALEF + ALLAH_LIG), ALLAH)
    check('R04 name +ullah', n(AIN + BEH + DAL + ALEF + ALLAH_LIG), AIN + BEH + DAL + ALLAH)
    # 1.0.1: an alef presentation form (isolated / final) before the ligature is absorbed too
    check('R04 alef isolated form+ALLAH ligature', n(u(0xFE8D) + ALLAH_LIG), ALLAH)
    check('R04 alef final form+ALLAH ligature', n(BEH + u(0xFE8E) + ALLAH_LIG), BEH + ALLAH)
    # kept symbols
    check('R04 SAW ligature kept', n(SAW), SAW)
    check('R04 ornate parentheses kept', n(ORNATE_L + BEH + ORNATE_R), ORNATE_L + BEH + ORNATE_R)
    check('R04 bismillah kept', n(BISMILLAH), BISMILLAH)
    check('R04 rial sign kept', n(u(0xFDFC)), u(0xFDFC))
    # other word ligatures -> NFKC letters (Arabic spelling kept: no Urdu evidence)
    check('R04 MOHAMMAD ligature', n(u(0xFDF4)), MEEM + u(0x062D) + MEEM + DAL)
    check('R04 report counts', rep(ALEF + ALLAH_LIG), {'R04_presentation': 2})


def test_r05_kashida():
    check('R05 kashida removed', n(BEH + KASHIDA + KASHIDA + ALEF), BEH + ALEF)
    check('R05 kashida between mark and letter', n(BEH + FATHA + KASHIDA + NOON), BEH + FATHA + NOON)
    # kashida separating a base from its mark: the mark reattaches, NFC recomposes
    check('R05 kashida then maddah recomposes', n(ALEF + KASHIDA + u(0x0653)), u(0x0622))
    check('R05 report', rep(BEH + KASHIDA + ALEF), {'R05_kashida': 1})


def test_r06_invisible():
    for name, c in (('ZWSP', ZWSP), ('LRM', LRM), ('RLM', RLM), ('ALM', ALM), ('BOM', BOM), ('SHY', SHY),
                    ('CGJ', CGJ), ('WJ', WJ), ('LRE', u(0x202A)), ('RLE', u(0x202B)), ('PDF', u(0x202C)),
                    ('LRO', u(0x202D)), ('RLO', u(0x202E)), ('LRI', u(0x2066)), ('RLI', u(0x2067)),
                    ('FSI', u(0x2068)), ('PDI', u(0x2069)), ('MVS', u(0x180E)), ('INV TIMES', u(0x2062))):
        check('R06 %s removed' % name, n(BEH + c + ALEF), BEH + ALEF)
    check('R06 leading BOM', n(BOM + 'abc'), 'abc')
    check('R06 bidi around a number', n(RLM + '2024' + LRM), '2024')


def test_r07_zwj():
    check('R07 ZWJ in Arabic word removed', n(BEH + ZWJ + ALEF), BEH + ALEF)
    check('R07 ZWJ after Arabic letter at word end removed', n(HEH_GOAL + ZWJ + ' ' + BEH), HEH_GOAL + ' ' + BEH)
    check('R07 ZWJ at text edges removed', n(ZWJ + 'a' + ZWJ), 'a')
    emoji = u(0x1F468) + ZWJ + u(0x1F469) + ZWJ + u(0x1F467)
    check('R07 emoji ZWJ sequence kept', n(emoji), emoji)
    deva = u(0x0915) + u(0x094D) + ZWJ + u(0x0937)
    check('R07 Devanagari ZWJ kept', n(deva), deva)


def test_r08_zwnj():
    w = BEH + ZWNJ + TEH                  # settled decision: preserved between letters
    check('R08 ZWNJ between letters kept', n(w), w)
    check('R08 ZWNJ after a mark kept', n(BEH + FATHA + ZWNJ + TEH), BEH + FATHA + ZWNJ + TEH)
    check('R08 ZWNJ run collapsed', n(BEH + ZWNJ + ZWNJ + ZWNJ + TEH), w)
    check('R08 ZWNJ before space removed', n(BEH + ZWNJ + ' ' + TEH), BEH + ' ' + TEH)
    check('R08 ZWNJ after space removed', n(BEH + ' ' + ZWNJ + TEH), BEH + ' ' + TEH)
    check('R08 ZWNJ at edges removed', n(ZWNJ + BEH + ZWNJ), BEH)
    check('R08 ZWNJ next to punctuation removed', n(BEH + ZWNJ + FULL_STOP), BEH + FULL_STOP)
    check('R08 ZWNJ next to digit removed', n(BEH + ZWNJ + URDU_DIGITS[1]), BEH + URDU_DIGITS[1])
    check('R08 ZWNJ exposed after ZWSP removal', n(BEH + ZWSP + ZWNJ + TEH), w)
    check('R08 report', rep(BEH + ZWNJ + ZWNJ + TEH), {'R08_zwnj': 1})


def test_r09_spaces():
    check('R09 NBSP', n(BEH + NBSP + TEH), BEH + ' ' + TEH)
    for cp in (0x2000, 0x2002, 0x2003, 0x2007, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000, 0x1680):
        check('R09 U+%04X -> space' % cp, n('a' + u(cp) + 'b'), 'a b')
    check('R09 runs collapse', n('a   ' + NBSP + ' b'), 'a b')
    check('R09 line-end spaces stripped', n('  a  \n  b  '), 'a\nb')
    check('R09 3+ newlines -> 2', n('a\n\n\n\nb'), 'a\n\nb')
    check('R09 blank line (book unit separator) kept', n('a\n\nb'), 'a\n\nb')
    check('R09 ZWSP is not a space', n('a' + ZWSP + 'b'), 'ab')


def test_r10_nfc2():
    # a removal (R06) brings alef and maddah together; R10 recomposes them
    check('R10 recompose after ZWSP removal', n(ALEF + ZWSP + u(0x0653)), u(0x0622))
    # a presentation-form mark (R04) lands next to another mark in the wrong order
    check('R10 reorder after R04', n(BEH + u(0xFE7C) + FATHA), BEH + FATHA + SHADDA)


def test_r07_r08_fixed_point():
    """Regressions for the v1.0.0 idempotency bug. R07/R08 judged a ZWJ/ZWNJ by
    its neighbours before NFC had reordered/composed the marks around it, and a
    removal could expose another ZWJ/ZWNJ. Each case: exact output, then the
    output is a fixed point (same text, empty report)."""
    man = u(0x1F468)
    cases = [
        # reviewer reproducer 1: R04 turns U+FE7E into SUKUN; NFC puts it next to the ZWJ
        ('reviewer 1: a ZWJ U+0301 U+FE7E', u(0x61) + ZWJ + u(0x0301) + u(0xFE7E), u(0x00E1) + SUKUN),
        # reviewer reproducer 2: emoji + ZWJ + non-Arabic mark + FATHA isolated form
        ('reviewer 2: emoji ZWJ U+0301 U+FE76', man + ZWJ + u(0x0301) + u(0xFE76), man + FATHA + u(0x0301)),
        # R08: '=' + U+0338 composes to the symbol U+2260 only after R06 removes the ZWSP
        ('R08: = ZWSP U+0338 ZWNJ a', '=' + ZWSP + u(0x0338) + ZWNJ + 'a', u(0x2260) + 'a'),
        # R08: the first ZWNJ's removal lets '=' + U+0338 compose; the second ZWNJ then follows a symbol
        ('R08: = ZWNJ U+0338 ZWNJ a', '=' + ZWNJ + u(0x0338) + ZWNJ + 'a', u(0x2260) + 'a'),
        # R07 removes both ZWJs (each has a ZWJ neighbour); '=' + U+0338 composes; R08 must see that
        ('R07+R08: = ZWJ ZWJ U+0338 ZWNJ a', '=' + ZWJ + ZWJ + u(0x0338) + ZWNJ + 'a', u(0x2260) + 'a'),
        # R07 chain: removing ZWJ 1 lets NFC move HAMZA ABOVE (ccc 230) past U+0334 (ccc 1), next to ZWJ 2
        ('R07 chain of 2', BEH + HAMZA_ABOVE + ZWJ + u(0x0334) + ZWJ + 'b', BEH + u(0x0334) + HAMZA_ABOVE + 'b'),
    ]
    # a chain of 50 needs 50 removal passes: a fixed cap of 3 passes would not be enough
    chain = BEH + HAMZA_ABOVE + (ZWJ + u(0x0334)) * 50 + ZWJ + 'b'
    cases.append(('R07 chain of 50', chain, BEH + u(0x0334) * 50 + HAMZA_ABOVE + 'b'))
    for name, s, want in cases:
        a = n(s)
        check('FIX %s: output' % name, a, want)
        b, r = N.normalize_with_report(a)
        check('FIX %s: fixed point' % name, (b, r), (a, {}))
    # ZWJs that stay: every neighbour non-Arabic and non-space, in NFC order
    kept = u(0x61) + ZWJ + u(0x0334) + u(0x0301)
    check('FIX ZWJ before non-Arabic marks kept', n(kept), kept)
    # canonical order decides: U+FE7E (SUKUN, ccc 34) typed before U+0334 (ccc 1) ends up after it
    check('FIX ZWJ judged in NFC order', n('a' + ZWJ + u(0xFE7E) + u(0x0334)), 'a' + ZWJ + u(0x0334) + SUKUN)
    check('FIX report counts the removal', 'R07_zwj' in rep(u(0x61) + ZWJ + u(0x0301) + u(0xFE7E)), True)


def test_r11_arabic_letters():
    # proven words (contain a letter Arabic never uses)
    check('R11 Arabic yeh in word with NOON GHUNNA', n(MEEM + AR_YEH + NOON_GHUNNA), MEEM + FA_YEH + NOON_GHUNNA)
    check('R11 Arabic kaf in word with YEH BARREE', n(AR_KAF + YEH_BARREE), KEHEH + YEH_BARREE)
    check('R11 kaf+yeh with RREH', n(SEEN + AR_YEH + AR_KAF + RREH + WAW + NOON), SEEN + FA_YEH + KEHEH + RREH + WAW + NOON)
    check('R11 alef maksura with FARSI YEH', n(AIN + FA_YEH + SEEN + ALEF_MAKSURA + u(0x0670)),
          AIN + FA_YEH + SEEN + FA_YEH + u(0x0670))
    check('R11 teh marbuta with KEHEH', n(u(0x0632) + KEHEH + ALEF + TEH_MARBUTA), u(0x0632) + KEHEH + ALEF + TEH_MARBUTA_GOAL)
    check('R11 high hamza yeh', n(NOON + HIGH_HAMZA_YEH + FA_YEH + NOON_GHUNNA), NOON + YEH_HAMZA + FA_YEH + NOON_GHUNNA)
    check('R11 high hamza yeh alone is its own evidence', n(u(0x0642) + ALEF + HIGH_HAMZA_YEH + MEEM),
          u(0x0642) + ALEF + YEH_HAMZA + MEEM)
    check('R11 SMALL V counts as evidence', n(TEH + SMALL_V + AR_YEH), TONE[TEH] + FA_YEH)
    # NOT proven: every letter is also an Arabic letter -> untouched
    for name, w in (('Arabic kaf word', AR_KAF + WAW), ('Arabic yeh word', AR_YEH + ALEF),
                    ('Ali', AIN + LAM + AR_YEH), ('alef maksura word', u(0x062D) + TEH + SHADDA + ALEF_MAKSURA),
                    ('teh marbuta word', u(0x0641) + ALEF + u(0x0637) + MEEM + TEH_MARBUTA)):
        check('R11 unproven %s untouched' % name, n(w), w)
    # a proven NEIGHBOUR word is not proof: the rule is per word
    check('R11 per-word', n(AR_KAF + WAW + ' ' + FA_YEH), AR_KAF + WAW + ' ' + FA_YEH)
    # blockers: letters of other orthographies that use ARABIC YEH on purpose
    for name, b in (('Pashto', PASHTO_SHEEN), ('Sindhi', SINDHI_KAF), ('Saraiki implosive', SARAIKI_DDAL)):
        w = b + FA_YEH + AR_YEH
        check('R11 blocked by %s letter' % name, n(w), w)
    # Arabic HEH is NEVER folded, even in a proven word (HEH GOAL vs DOACHASHMEE is undecidable)
    w = AIN + LAM + FA_YEH + AR_HEH
    check('R11 Arabic heh untouched', n(w), w)
    w = TEH + AR_HEH + ALEF
    check('R11 Arabic heh (tha) untouched', n(w), w)
    # HEH GOAL and HEH DOACHASHMEE are never merged
    check('R11 heh goal vs do-chashmi kept', n(KEHEH + HEH_GOAL + ' ' + KEHEH + HEH_DO), KEHEH + HEH_GOAL + ' ' + KEHEH + HEH_DO)
    # ZWNJ inside the word does not split it
    check('R11 word with ZWNJ', n(AR_KAF + ZWNJ + GAF), KEHEH + ZWNJ + GAF)
    check('R11 report', rep(MEEM + AR_YEH + NOON_GHUNNA), {'R11_arabic_letters': 1})
    check('R11 word_is_urdu_orthography', N.word_is_urdu_orthography(AR_KAF + WAW), False)


def test_r12_tone_letters():
    for base, tone in TONE.items():
        check('R12 compose U+%04X' % ord(base), n(base + SMALL_V + ALEF), tone + ALEF)
        check('R12 compose with haraka U+%04X' % ord(base), n(base + FATHA + SMALL_V), tone + FATHA)
    # small v typed before the haraka: NFC puts the haraka first, then R12 composes
    check('R12 small v before haraka', n(PEH + SMALL_V + FATHA), TONE[PEH] + FATHA)
    # on other letters U+065A stays a combining mark (dictionary jazm, hp/inpage.py)
    check('R12 small v on NOON stays', n(NOON + SMALL_V), NOON + SMALL_V)
    check('R12 small v on BEH stays', n(BEH + SMALL_V), BEH + SMALL_V)
    # Arabic kaf + small v: R11 folds (small v is evidence), R12 composes
    check('R12 Arabic kaf + small v', n(AR_KAF + SMALL_V), TONE[KEHEH])
    # precomposed tone letters are untouched
    for tone in TONE.values():
        check('R12 precomposed U+%04X kept' % ord(tone), n(tone + ALEF), tone + ALEF)
    check('R12 report', rep(PEH + SMALL_V), {'R12_tone_letters': 2})


def test_r13_digits():
    check('R13 Arabic-Indic -> Extended', n(ARABIC_DIGITS), URDU_DIGITS)
    check('R13 Extended kept', n(URDU_DIGITS), URDU_DIGITS)
    check('R13 ASCII kept', n('0123456789'), '0123456789')
    check('R13 verse number in parentheses', n('(' + u(0x0667, 0x0664) + ')'), '(' + u(0x06F7, 0x06F4) + ')')
    check('R13 mixed run left as written', n(u(0x06F1) + '1'), u(0x06F1) + '1')
    check('R13 report', rep(u(0x0661, 0x0662)), {'R13_digits': 2})


def test_things_never_touched():
    keep = [
        ('retroflex nasal NOON WITH SMALL TAH', NOON_SMALL_TAH + ALEF),
        ('retroflex nasal digraph', NOON + RREH),
        ('noon + sukun', NOON + SUKUN),
        ('izafat kasra after space', BEH + ' ' + KASRA + ' ' + TEH),
        ('doubled haraka', BEH + FATHA + FATHA),
        ('hamza forms', HAMZA + ' ' + YEH_HAMZA + ' ' + FA_YEH + HAMZA + ' ' + u(0x0623) + ' ' + u(0x0624)),
        ('Urdu punctuation', FULL_STOP + COMMA + u(0x061F) + u(0x061B) + u(0x066A)),
        ('ASCII punctuation', '. , ? ; % : ! " \''),
        ('quotes', u(0x2018) + u(0x2018) + BEH + u(0x2019) + u(0x2019) + u(0x201C) + u(0x201D) + u(0xAB) + u(0xBB)),
        ('ellipsis', u(0x2026)),
        ('Latin case', 'COVID-19 Hindko'),
        ('PII placeholders', '<PHONE> <EMAIL> <ID_NUMBER>'),
        ('replacement char', BEH + u(0xFFFD) + ALEF),
        ('SAW', SAW), ('honorific combining sign', MEEM + u(0x0610)),
        ('year sign', URDU_DIGITS[1] + URDU_DIGITS[9] + u(0x0601) + HAMZA),
        ('superscript alef', AIN + LAM + FA_YEH + u(0x0670)),
        ('arabic decimal separator', u(0x066B)),
        ('Arabic heh alone', AR_HEH),
        ('yeh barree vs farsi yeh', u(0x062F) + FA_YEH + ' ' + u(0x062F) + YEH_BARREE),
        ('noon ghunna vs noon', NOON_GHUNNA + ' ' + NOON),
    ]
    for name, s in keep:
        check('KEEP %s' % name, n(s), s)


def test_idempotent_and_nfc(samples, label):
    """normalize(x) is NFC, and a fixed point: normalizing it again returns the
    same text with an empty change report (no rule fires)."""
    bad = 0
    for s in samples:
        a = n(s)
        b, r = N.normalize_with_report(a)
        if a != b or r or not unicodedata.is_normalized('NFC', a):
            bad += 1
            if bad <= 5:
                print('  not a fixed point / non-NFC:', [hex(ord(c)) for c in s], '->', [hex(ord(c)) for c in a],
                      '->', [hex(ord(c)) for c in b], r)
    check('idempotent+NFC+empty report on %d %s strings' % (len(samples), label), bad, 0)
    return bad


# pools for the widened fuzz sets (1.0.1)
NON_ARABIC_MARKS = ([cp for cp in range(0x0300, 0x0370)]                    # all combining diacriticals, ccc 1..240
                    + [0x093C, 0x094D, 0x0DCA, 0x0E38, 0x0F71, 0x0F72, 0x0F74]  # nukta 7, viramas 9, Thai 103, Tibetan 129-132
                    + list(range(0x05B0, 0x05BE))                          # Hebrew points, ccc 10-22
                    + [0x20D0, 0x20D2, 0x20DB, 0xFE0F])                     # combining symbols, VS16 (ccc 0)
ARABIC_MARKS = ([cp for cp in list(range(0x064B, 0x0660)) + [0x0670] + list(range(0x0610, 0x061B))
                 + list(range(0x06D6, 0x06EE)) + list(range(0x08D3, 0x0900))
                 if unicodedata.category(chr(cp)) == 'Mn'])
PRESENTATION_HARAKAT = ([cp for cp in range(0xFE70, 0xFE80) if unicodedata.category(chr(cp)) != 'Cn']
                        + list(range(0xFC5E, 0xFC64)) + [0xFCF2, 0xFCF3, 0xFCF4])
COMPOSING_BASES = ([0x61, 0x65, 0x6F, 0x41, 0x3D, 0x3C, 0x3E, 0x2190, 0x2192, 0x2194, 0x2203, 0x2208, 0x2223,
                    0x223C, 0x2243]                                        # Latin; symbols that compose with U+0338
                   + [0x0627, 0x0648, 0x064A, 0x06C1, 0x06D2, 0x06D5, 0x0628, 0x0646, 0x067E, 0x06A9]  # Arabic
                   + [0x0915, 0x0937, 0x0928, 0x1F468, 0x1F469, 0x2764])   # Devanagari, emoji


def focused_pool():
    p = [ZWJ] * 6 + [ZWNJ] * 6
    p += [u(c) for c in NON_ARABIC_MARKS + ARABIC_MARKS + PRESENTATION_HARAKAT + COMPOSING_BASES]
    p += [u(c) for c in (0xFE8D, 0xFE8E, 0xFDF2, 0xFBA6, 0xFEF1, 0xFED9)]   # letter forms, ALLAH ligature
    p += [' ', ' ', '\n', NBSP, ZWSP, KASHIDA, LRM, BOM, u(0x95), '1', u(0x0661), '.', FULL_STOP]
    return p


def fuzz_from(pool, k, seed, maxlen=14):
    rng = random.Random(seed)
    return [''.join(rng.choice(pool) for _ in range(rng.randint(1, maxlen))) for _ in range(k)]


def fuzz_focused(k=400000, seed=20260927):
    return fuzz_from(focused_pool(), k, seed)


def fuzz_broad(k=300000, seed=20260928):
    blocks = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF),
              (0xFE70, 0xFEFF))
    arabic = [u(cp) for lo, hi in blocks for cp in range(lo, hi + 1) if unicodedata.category(chr(cp)) != 'Cn']
    return fuzz_from(arabic + focused_pool(), k, seed)


def fuzz_strings(k=60000, seed=20260926):
    rng = random.Random(seed)
    pool = [ALEF, BEH, TEH, DAL, REH, SEEN, AIN, LAM, MEEM, NOON, WAW, AR_YEH, ALEF_MAKSURA, AR_KAF, AR_HEH,
            TEH_MARBUTA, FA_YEH, KEHEH, HEH_GOAL, HEH_DO, TEH_MARBUTA_GOAL, YEH_BARREE, NOON_GHUNNA, PEH, TTEH,
            TCHEH, GAF, RREH, YEH_HAMZA, HIGH_HAMZA_YEH, HAMZA, HAMZA_ABOVE, NOON_SMALL_TAH, PASHTO_SHEEN,
            SINDHI_KAF, SARAIKI_DDAL, FATHA, DAMMA, KASRA, SHADDA, SUKUN, SMALL_V, u(0x0653), u(0x0670),
            ZWNJ, ZWJ, ZWSP, NBSP, LRM, RLM, ALM, BOM, SHY, CGJ, WJ, KASHIDA, FULL_STOP, SAW, ALLAH_LIG, ORNATE_L,
            ' ', ' ', ' ', '\n', '\n', '\r', '\t', u(0x85), u(0x95), u(0x01), 'a', 'Z', '1', '.', u(0x0301),
            u(0x1F468), u(0x2028)]
    pool += list(TONE.values()) + list(ARABIC_DIGITS[:3]) + list(URDU_DIGITS[:3])
    pool += [u(cp) for cp in (0xFBFC, 0xFEF1, 0xFEF3, 0xFED9, 0xFEE9, 0xFBA6, 0xFBAA, 0xFEFB, 0xFE76, 0xFE77,
                              0xFC5E, 0xFE7C, 0xFDF4, 0xFDFC, 0xFE70, 0xFE71, 0xFE73)]
    return [''.join(rng.choice(pool) for _ in range(rng.randint(1, 14))) for _ in range(k)]


def corpus_records(path=r'F:\Hindko\hindko_dataset_permissive.jsonl'):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)['text']


def main():
    print('hp.normalize NORMALIZATION_VERSION %s, Unicode %s' % (N.NORMALIZATION_VERSION, N.UNICODE_VERSION))
    for name, fn in sorted(globals().items()):
        if name.startswith('test_r') or name == 'test_things_never_touched':
            fn()
    print('unit checks: %d, failed %d' % (NCHECK, len(FAILS)))
    for label, gen in (('basic', fuzz_strings), ('focused', fuzz_focused), ('broad', fuzz_broad)):
        bad = test_idempotent_and_nfc(gen(), label)
        print('fuzz %-8s failures: %d' % (label, bad))
    if '--corpus' in sys.argv:
        bad = test_idempotent_and_nfc(list(corpus_records()), 'corpus')
        print('corpus records failures: %d' % bad)
    print('%d checks, %d failed' % (NCHECK, len(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
