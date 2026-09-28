"""Checks that morph_eval.py behaves sensibly. Run: python tests/test_morph_eval.py"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import morph_eval as M  # noqa: E402

HIGH = os.path.join(ROOT, 'morph_silver_high.tsv')
LOW = os.path.join(ROOT, 'morph_silver_low.tsv')


def close(a, b, eps=1e-9):
    return abs(a - b) < eps


def test_gold_files_load_and_validate():
    high, low = M.load_gold(HIGH), M.load_gold(LOW)
    assert 400 <= len(high) + len(low) <= 800, (len(high), len(low))
    assert len({g.word for g in high} & {g.word for g in low}) == 0
    assert all(min(g.req) >= 2 for g in high + low), 'stems have >= 2 letters'
    assert all(not g.alts for g in high), 'high-confidence words have no alternative segmentation'


def test_char_level():
    gold = M.load_gold(HIGH)
    r = M.evaluate(M.char_tokenizer, gold)
    assert r['n_unaligned'] == 0
    assert close(r['boundary_recall'], 1.0)               # every boundary is predicted
    assert r['boundary_precision'] < 0.5                  # ... together with many wrong ones
    assert close(r['morphscore'], 1.0)                    # MorphScore alone rewards over-splitting
    assert close(r['stem_intact'], 0.0)                   # every stem (>= 2 letters) is split
    assert close(r['stem_boundary_respected'], 0.0)


def test_whole_word():
    gold = M.load_gold(HIGH)
    r = M.evaluate(M.whole_word_tokenizer, gold)
    assert close(r['boundary_recall'], 0.0)
    assert math.isnan(r['boundary_precision'])            # no predicted boundaries at all
    assert math.isnan(r['morphscore'])                    # no multi-token words
    assert close(r['morphscore_all'], 0.0)
    assert close(r['stem_intact'], 1.0)
    assert close(r['stem_boundary_respected'], 0.0)
    assert close(r['single_token_rate'], 1.0)


def test_oracle():
    for path in (HIGH, LOW):
        gold = M.load_gold(path)
        r = M.evaluate(M.make_oracle(gold), gold)
        for k in ('boundary_precision', 'boundary_recall', 'boundary_f1', 'morphscore', 'stem_intact',
                  'stem_boundary_respected', 'exact_match'):
            assert close(r[k], 1.0), (path, k, r[k])


def test_ordering_random_between_trivial():
    gold = M.load_gold(HIGH)
    rnd = M.evaluate(M.make_random(0.3, 13), gold)
    ch = M.evaluate(M.char_tokenizer, gold)
    ww = M.evaluate(M.whole_word_tokenizer, gold)
    assert ww['boundary_f1'] < rnd['boundary_f1'] < ch['boundary_f1'] or rnd['boundary_f1'] < ch['boundary_f1']
    assert ch['stem_boundary_respected'] < rnd['stem_boundary_respected'] < 1.0


def test_optional_and_alternative_boundaries():
    # کردیاں: required کر|دیاں (2), optional 3 and 4 (inside the suffix)
    g = M.GoldWord('کردیاں', [2], [3, 4], [], 'V.IPFV', 1, 'T1', 'high', 'کر|دیاں')
    d = M.score_word(g, [M.char_to_byte(g.word)[2], M.char_to_byte(g.word)[3]])   # کر|د|یاں
    assert d['tp'] == 1 and d['fp'] == 0 and d['respected']
    d = M.score_word(g, [M.char_to_byte(g.word)[1]])                               # ک|ردیاں: stem split
    assert d['tp'] == 0 and d['fp'] == 1 and not d['stem_intact']
    # بچیاں: primary بچ|یاں (2), alternative بچی|اں (3)
    g = M.GoldWord('بچیاں', [2], [3], [[3]], 'N.OBL.PL', 1, 'T2', 'low', 'بچ|یاں')
    d = M.score_word(g, [M.char_to_byte(g.word)[3]])
    assert d['respected'] and d['alt']


def _rows(path):
    import csv
    return list(csv.DictReader(open(path, encoding='utf-8'), delimiter='\t'))


def _review():
    import csv
    out = {}
    for r in csv.DictReader(open(os.path.join(ROOT, 'review', 'review_decisions.tsv'), encoding='utf-8'),
                            delimiter='\t'):
        if not r['word'].startswith('#'):
            out.setdefault(r['word'], []).append(r)
    return out


def test_reported_homographs_are_gone():
    # round-2 review: words whose own tokens mostly belong to another lexeme (KWIC in review_decisions.tsv)
    gold = {r['word'] for r in _rows(HIGH) + _rows(LOW)}
    for w in ('روے', 'پاسو', 'سونڑے', 'جلدی', 'ملاں', 'آنڑیں', 'آنڑاں', 'آنڑے', 'حویلیاں', 'پچھاں', 'بنڑاں'):
        assert w not in gold, w


def test_no_word_explained_by_a_one_letter_verb_stem():
    # آ 'come' (also written ا) + an inventory verb suffix: out of scope, must not be re-parsed with a longer stem
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from morph_inventory import VERB_SUFFIXES
    sufs = {s['suffix'] for s in VERB_SUFFIXES if s['target']}
    for r in _rows(HIGH) + _rows(LOW):
        for stem in ('آ', 'ا'):
            assert not (r['word'].startswith(stem) and r['word'][len(stem):] in sufs), r['word']


def test_alternatives_are_reviewed_readings():
    # an alternative is scored as correct, so each one must be a reading the review confirmed by KWIC
    rev = _review()
    rejected = {('بیماریاں', '5'), ('قربانیاں', '5'), ('تیاریاں', '4'), ('خوشیاں', '3'), ('شہریاں', '3'),
                ('پشوریاں', '4'), ('یہودیاں', '4'), ('قیدیاں', '3'), ('سبزیاں', '3'), ('دوائیاں', '4'),
                ('مصرعیاں', '5'), ('پیسیاں', '4'), ('اسدیاں', '4')}
    for r in _rows(LOW):
        for alt in [a for a in r['alternatives'].split(';') if a]:
            assert (r['word'], alt) not in rejected, (r['word'], alt)
            assert any(x['action'] == 'alt_ok' and x['alt'].strip() == alt for x in rev.get(r['word'], [])), \
                (r['word'], alt, 'alternative without a review alt_ok row')
    assert not any(r['alternatives'] for r in _rows(HIGH))


def test_high_entries_pass_the_context_check():
    # verb entries of the high set: own-context metric within the thresholds of induce.PARAMS
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from induce import PARAMS
    for r in _rows(HIGH):
        assert 'context check failed' not in r['notes'], r['word']
        if not r['category'].startswith('V.'):
            continue
        m = dict(kv.split('=') for kv in r['context'].split())
        if 'obl' in m:
            assert float(m['obl']) >= PARAMS['CTX_INF_OBL_MIN'] or float(m['aux']) >= PARAMS['CTX_INF_OBL_AUX_EXEMPT'], r
        else:
            assert float(m.get('nom', m.get('detq'))) <= PARAMS['CTX_NOM_MAX'], r


def test_offsets_and_byte_level_and_markers():
    w = 'کردیاں'
    c2b = M.char_to_byte(w)
    P, n = M.boundaries_from_offsets(w, [(0, 2), (2, 6)])
    assert P == [c2b[2]] and n == 2
    # SentencePiece-style pieces with a word-boundary marker, and a lone marker
    P, n = M.boundaries_from_tokens(w, ['▁', '▁کر', 'دیاں'])
    assert P == [c2b[2]] and n == 2
    # WordPiece continuation marker
    P, n = M.boundaries_from_tokens(w, ['کر', '##دیاں'])
    assert P == [c2b[2]]
    # GPT-2 byte-level strings, including a split in the middle of a 2-byte letter
    enc = M._bytes_to_unicode()
    b = w.encode('utf-8')
    toks = [''.join(enc[x] for x in b[:3]), ''.join(enc[x] for x in b[3:])]    # cut after byte 3 = mid-letter
    P, n = M.boundaries_from_tokens(w, toks, byte_level=True)
    assert P == [3] and 3 not in c2b                     # a mid-character boundary can never be correct
    # byte-fallback pieces
    P, n = M.boundaries_from_tokens(w, ['کر'] + [f'<0x{x:02X}>' for x in 'دیاں'.encode('utf-8')])
    assert P[0] == c2b[2] and n == 1 + len('دیاں'.encode('utf-8'))
    # a leading space (prefix-space encoders) is ignored
    P, n = M.boundaries_from_tokens(w, [' کر', 'دیاں'])
    assert P == [c2b[2]]
    # tokens that do not spell the word are reported as unaligned
    P, n = M.boundaries_from_tokens(w, ['[UNK]'])
    assert P is None


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
            print('ok', name)
    print(f'{n} tests passed')
