# -*- coding: utf-8 -*-
"""Acceptance tests + report CLI for hp/books.py and hp/lang.py.

Runs the book pipeline end-to-end over every book file on disk
(raw_files/Hindko Books/HINDKO BOOKS DATA/**/*.inp|.B01|.docx), decoded with
the CURRENT hp.inpage decoder, and checks the acceptance criteria of
_books_cache/BOOKS_SPEC.md as corrected by BOOKS_SPEC_CORRECTIONS.md, plus the
regression checks for the adversarial review findings (C0 unit cases, C3,
C3b, C8, C10, C12c, C14-C21). Decodes are cached under
_pipeline/_scratch/books_impl/ keyed by a fingerprint of the decoder source
(hp/inpage.py, hp/glyph_map.py, hp/docx.py), so a decoder change invalidates
the cache automatically.

Usage (PYTHONIOENCODING=utf-8):
    python test_books.py                 # all book files incl. .B01 twins (207 + docx)
    python test_books.py --mode default  # as build.py default: .B01 with an .inp twin skipped
    python test_books.py --no-determinism --no-mode-compare
Writes _scratch/books_impl/test_report_<mode>.json and exits 1 if a check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RAW_FILES = r'F:\Hindko\raw_files'
BOOKS_DIR = os.path.join(RAW_FILES, 'Hindko Books', 'HINDKO BOOKS DATA')
NEWSPAPER_JSONL = r'F:\Hindko\hindko_dataset_permissive.jsonl'
SCRATCH = os.path.join(HERE, '_scratch', 'books_impl')
DOUBLED_LIST = os.path.join(HERE, 'probe_books_dedup_doubled_exact_out.txt')


# --------------------------------------------------------------------------
# discovery + decode cache
# --------------------------------------------------------------------------
def discover_book_files(mode: str = 'all') -> list:
    """(rel, kind, abs_path) for every book text source, rel relative to
    raw_files/ with '/'. mode='default' skips .B01 files with a same-stem
    .inp in the same directory (build.py's default)."""
    out = []
    for dp, dn, fn in os.walk(BOOKS_DIR):
        dn.sort()
        rel_dir = os.path.relpath(dp, RAW_FILES).replace(os.sep, '/')
        if rel_dir.split('/')[2:3] == ['TXT']:
            continue                      # derived exports, never sources
        for f in sorted(fn):
            ext = os.path.splitext(f)[1].lower().lstrip('.')
            if ext not in ('inp', 'b01', 'docx'):
                continue
            if f.startswith('~$'):
                continue                  # Word owner/lock file
            out.append((rel_dir + '/' + f, ext, os.path.join(dp, f)))
    if mode == 'default':
        inp_keys = {(os.path.dirname(r), os.path.splitext(os.path.basename(r))[0].lower().strip())
                    for r, k, _ in out if k == 'inp'}
        out = [(r, k, p) for r, k, p in out
               if not (k == 'b01' and (os.path.dirname(r), os.path.splitext(
                   os.path.basename(r))[0].lower().strip()) in inp_keys)]
    out.sort()
    return out


def decoder_fingerprint() -> str:
    h = hashlib.sha256()
    for name in ('inpage.py', 'glyph_map.py', 'docx.py'):
        with open(os.path.join(HERE, 'hp', name), 'rb') as fh:
            h.update(name.encode() + b'\0' + fh.read())
    return h.hexdigest()[:16]


def _decode_one(args):
    rel, kind, path = args
    from hp import inpage, docx
    t = time.time()
    try:
        if kind == 'docx':
            text = docx.extract_text(path)
            text_all = '\n'.join(inpage.split_lines(text))
            unmapped = 0
        else:
            info = inpage.extract_file(path)
            text_all = info['text_all']
            unmapped = info['unmapped_glyphs']
        err = None
    except Exception as e:                      # surfaced, not hidden
        text_all, unmapped, err = '', 0, '%s: %s' % (type(e).__name__, e)
    return rel, {'kind': kind, 'text_all': text_all, 'unmapped': unmapped,
                 'error': err, 'size': os.path.getsize(path),
                 'mtime': int(os.path.getmtime(path)), 'secs': round(time.time() - t, 2)}


def load_decoded(files: list, log=print) -> dict:
    """{rel: decode record}; cached per decoder fingerprint under SCRATCH."""
    os.makedirs(SCRATCH, exist_ok=True)
    fp = decoder_fingerprint()
    cache_path = os.path.join(SCRATCH, 'decoded_%s.pkl' % fp)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            cache = pickle.load(fh)
    todo = []
    for rel, kind, path in files:
        c = cache.get(rel)
        if not c or c['size'] != os.path.getsize(path) or c['mtime'] != int(os.path.getmtime(path)):
            todo.append((rel, kind, path))
    if todo:
        log('decoding %d book files with decoder %s (cache: %s)' % (len(todo), fp, cache_path))
        t = time.time()
        with ProcessPoolExecutor(max_workers=6) as ex:
            for rel, rec in ex.map(_decode_one, todo, chunksize=1):
                cache[rel] = rec
        tmp = cache_path + '.tmp%d' % os.getpid()
        with open(tmp, 'wb') as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        log('decoded in %.1fs' % (time.time() - t))
    return {rel: cache[rel] for rel, _, _ in files}


def load_newspaper(path: str = NEWSPAPER_JSONL) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                # Once the combined build is released the file also holds book
                # records; only newspaper records are the "earlier source".
                if r.get('source', 'newspaper') != 'newspaper':
                    continue
                out.append((r['id'], r['text']))
    return out


def book_inputs(files: list, decoded: dict) -> list:
    # 'size' (bytes on disk) feeds the L13 payload test, as build.stage_books passes it
    return [{'rel': rel, 'kind': kind, 'text_all': decoded[rel]['text_all'], 'size': decoded[rel]['size']}
            for rel, kind, _ in files if not decoded[rel]['error']]


# --------------------------------------------------------------------------
# helpers for the checks
# --------------------------------------------------------------------------
def canonical_digest(out: dict) -> str:
    s = json.dumps(out, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def file_digests(out: dict) -> dict:
    """{source_path: digest of that file's records} (C15 mode comparison)."""
    by = defaultdict(list)
    for r in out['records']:
        by[r['source_path']].append(r)
    for pf in out['per_file']:
        by.setdefault(pf['source_path'], [])
    return {p: hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
                              .encode('utf-8')).hexdigest()[:20] for p, v in sorted(by.items())}


def doubled_expected() -> list:
    """(prefix, suffix) patterns of the 14 whole-book-doubling files listed
    (abbreviated with '...') in probe_books_dedup_doubled_exact_out.txt."""
    pats = []
    with open(DOUBLED_LIST, encoding='utf-8') as fh:
        for line in fh.read().split('\n')[1:]:
            m = re.match(r'^(.*?\.(?:INP|inp|B01|b01))\s+\d+', line)
            if not m:
                continue
            p = m.group(1).replace('\\', '/')
            a, b = p.split('...', 1) if '...' in p else (p, p)
            pats.append((a, b))
    return pats


def run_pipeline(files, decoded, news, log=print):
    from hp import books
    inputs = book_inputs(files, decoded)
    t0, c0 = time.time(), time.process_time()
    out = books.process_books(inputs, news, log=log)
    return out, time.time() - t0, time.process_time() - c0


# --------------------------------------------------------------------------
# acceptance checks
# --------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail, soft=False):
        self.rows.append({'check': name, 'status': ('PASS' if ok else ('WARN' if soft else 'FAIL')),
                          'detail': detail})
        print('  [%s] %-50s %s' % (self.rows[-1]['status'], name, detail), flush=True)

    @property
    def failed(self):
        return [r for r in self.rows if r['status'] == 'FAIL']


def _parse_pairs(pairs):
    from hp import books
    w = {'pairs': [{'field': f, 'label': l, 'value': v, 'label_line': k, 'value_line': k + 1}
                   for k, (f, l, v) in enumerate(pairs)], 'inline_isbn': []}
    return books.parse_imprint(w)


# invented Hindko-like lines for the synthetic layout cases
_P1 = ['ساون دی رت آئی تے بدل وسیا', 'باغاں دے وچ پھل کھڑے سارے', 'ٹھنڈی ہوا نے دل نوں موہیا',
       'یاداں دے دیوے بلے سارے', 'راتاں لمیاں تے نیندر تھوڑی', 'اکھیاں دے وچ تارے سارے']
_P2 = ['پہاڑاں اتے برف پئی اے', 'ندیاں دا پانی ٹھنڈا ہویا', 'چولہے دے کول بیٹھے سارے',
       'قصے کہانیاں چھڑیاں گلاں', 'بال نیانڑے سون لگے', 'چن وی بدلاں اولے ہویا']
_PROSE = ['اس شہر دی پرانی گلی وچ ہک نکا جیہا گھر سی جتھے ہک بزرگ رہندا سی تے لوک اس کول آندے سن۔',
          'او ہر روز سویرے اٹھ کے باغ وچ جاندا تے بوٹیاں نوں پانی دیندا تے فیر کتاب پڑھدا سی۔',
          'ہک دن ہک مسافر آیا تے اس نے پچھیا کہ ایہہ رستہ کدھر جاندا اے تے بزرگ نے دسیا۔',
          'اس توں بعد او مسافر وی اسے شہر وچ رہن لگ پیا تے دوہاں دی یاری پکی ہو گئی۔']


def unit_checks(ck: Checks):
    """C0: lang.py cases + synthetic books.py cases (one per review finding
    that a small input can reproduce)."""
    from hp import lang, books
    en = 'Handi and hindko both of which mean the language'
    R = books.BOOKS_ROOT
    cases = [
        ('Hindko line -> hindko', lang.language_variety_v2(
            'اس دے نال اوہ وی ہک واری آیا تے آکھن لگا کہ میں تے جاندا ہوندا واں') == 'hindko'),
        ('Urdu line -> urdu', lang.language_variety_v2(
            'قدیم پشاور میں بدمعاشی کا بھی بہت چرچا رہا ہے اور لوگ اس کے ساتھ تھے') == 'urdu'),
        ('whole token: کتاب has no Urdu marker', lang.marker_counts('کتاب') == (0, 0)),
        ('whole token: ہوتا is Urdu', lang.marker_counts('ہوتا') == (0, 1)),
        ('no marker -> 0.5 / no_signal', lang.hindko_score_v2('پشاور شہر') == 0.5
         and lang.language_variety_v2('پشاور شہر') == 'no_signal'),
        ('Arabic rule v4 flags a Quranic line', lang.is_arabic_line('بِسمِ اللہِ الرَّحمٰنِ الرَّحِیمِ')),
        ('Arabic rule v4 rejects Hindko', not lang.is_arabic_line('اس دے نال اوہ وی ہک واری آیا تے آکھن لگا')),
        ('Persian rule v2 flags a Persian line', lang.is_persian_line('دلِ من روشن از سوزِ درون است')),
        ('English frame (R8)', lang.is_english_text_frame(en, 1)),
        ('English repeated in >=5 files = residue', not lang.is_english_text_frame(en, 5)),
        ('font names are not English text', not lang.is_english_text_frame('Times New Roman Bold Italic', 1)),
        ('latin_share', abs(lang.latin_share('abc ابج') - 0.5) < 1e-9),
        ('Pothohari only with path + text', bool(lang.pothohari_evidence(
            '25th Farma Pothohari Dictionary', ['پوٹھوہاری لغت']))
         and not lang.pothohari_evidence('24th Farma X', ['پوٹھوہاری لغت'])),
        ('validated markers 41 H / 43 U', len(lang.HINDKO_VALIDATED) == 41 and len(lang.URDU_VALIDATED) == 43),
        ('تھیں گھر میں are not Urdu markers', not ({'تھیں', 'گھر', 'میں'} & lang.URDU_TOKENS)),
    ]
    # review: FILE_GENRE_OK - a religious keyword in a FILE name never sets a genre
    cases += [
        ('genre: NAAT file name sets no genre', lang.book_genre(
            '400th Farma Some Book', ['NAAT SHAREEF.INP', 'BOOK.INP'], '', 'prose') == (None, None)),
        ('genre: SORAH file name is not scripture_translation', lang.book_genre(
            '400th Farma Some Book', ['SORAH YASEEN TARJUMA.INP'], '', 'verse') == ('poetry', 'form:verse')),
        ('genre: non-religious file keyword still used', lang.book_genre(
            '400th Farma Some Book', ['DRAMA PART 1.INP'], '', 'prose') == ('drama', 'file:DRAMA')),
        ('genre: no religious/history genre is a file genre',
         not (lang.FILE_GENRE_OK & lang.RELIGIOUS_HISTORY_GENRES)),
        ('R8 words: short English word frame kept', lang.is_english_word_frame('Thermometer', 1)
         and not lang.is_english_text_frame('Thermometer', 1)),
        ('R8 words: font / path / repeated frames are not words',
         not lang.is_english_word_frame('Times New Roman', 1)
         and not lang.is_english_word_frame('C:\\book\\a.inp', 1)
         and not lang.is_english_word_frame('Ticket', 5)),
        ('Hindko tone letters are never Arabic / Persian',
         not lang.is_arabic_line('\u08bfَنْ فِیْ الْاَرْضِ وَالسَّمَاءِ وَمَا بَیْنَہُمَا')),
    ]

    def classify(lines, kind=None):
        return books.classify_file(lines, Counter(), kind)

    def units(lines, lex=False, toc=(), removed=None, kind=None):
        c = classify(lines, kind)
        if lex:
            books._lexicon_categories(c, lines)
        us, eus, to, ld, _ = books.build_units(lines, c['cat'], removed or {}, lex,
                                               {books.toc_key(x) for x in toc})
        return c, us, to, ld

    # review: RE_SEP / U+2026 - a frame of dots, ۔ or … is a poem boundary;
    # a single … inside text is punctuation
    c1, u1, _, _ = units(_P1 + ['……………'] + _P2)
    c2, u2, _, _ = units(_P1 + ['۔۔۔۔۔'] + _P2)
    c3 = classify(['اوہ آیا… تے فیر ٹر گیا', 'اوہ آیا تے فیر ٹر گیا'])
    cases += [
        ('separator: …………… splits two poems', c1['cat'][6] == 'separator' and len(u1) == 2),
        ('separator: ۔۔۔۔۔ splits two poems', c2['cat'][6] == 'separator' and len(u2) == 2),
        ('separator: a single … inside text stays text', c3['cat'][0] == 'text'),
    ]
    # review: L3 main-span cut - short frames before/after the >= 25-char span
    # are text; only >= 3 identical <= 60-char frames at an end are a header run
    long_ = 'ایہہ ہک لمبی سطر اے جیہڑی پنجی حرفاں توں ودھ اے'
    c4 = classify(['پہلی نکی سطر', long_, 'چھوٹی سطر ہک اے', 'چھوٹی سطر دو اے'] + ['کتاب دا ناں'] * 3)
    cases += [
        ('L3: frames before / after the main span are text',
         c4['cat'][:4] == ['text'] * 4 and 'outside' not in books.STRUCTURAL),
        ('L3: 3 identical short frames at the end = header run', c4['cat'][4:] == ['header_run'] * 3),
    ]
    # review: parse_tocs - the block ends at its last page number; the frames
    # after it (dedication, first misra) are body text, not entries / 'toc'
    toc_lines = ['فہرست', 'پہلا باب', '5', 'دوجا باب', '9', 'تیجا باب', '13', 'انتساب',
                 'اپنی ماں دے ناں', 'جیہڑی ہمیشہ یاد رہسی', long_]
    tb = books.parse_tocs(toc_lines)
    c5 = classify(toc_lines)
    cases += [
        ('TOC: block ends at its last page number', len(tb) == 1 and tb[0]['end_idx'] == 6
         and [e[2] for e in tb[0]['entries']] == ['5', '9', '13'] and max(tb[0]['frames']) == 6),
        ('TOC: frames after the block are text', c5['cat'][7:] == ['text'] * 4),
    ]
    # review: L8 - titles in increasing BODY position, no TOC-order pointer;
    # an entry matching > 1 frame is never a title; a title carries over a
    # following boundary
    toc = ['پہلا باب', 'دوجا باب', 'تیجا باب', 'کتاب دا ناں']
    body = ['کتاب دا ناں', 'پہلا باب', _PROSE[0], 'دوجا باب', _PROSE[1], 'تیجا باب', _PROSE[2]]
    _, u6, _, _ = units(body, toc=toc)
    _, u7, _, _ = units(body + ['دوجا باب', _PROSE[3]], toc=toc)
    _, u8, to8, _ = units(['پہلا باب', '۔۔۔۔۔', _PROSE[0]], toc=toc)
    t6 = {u['title'] for u in u6}
    t7 = {u['title'] for u in u7}
    cases += [
        ('L8: a late entry matched first does not block later titles',
         {'پہلا باب', 'دوجا باب', 'تیجا باب'} <= t6),
        ('L8: an entry matching 2 frames is not a title', 'دوجا باب' not in t7 and 'تیجا باب' in t7),
        ('L8: a title-only heading carries over a boundary',
         len(u8) == 1 and u8[0]['title'] == 'پہلا باب' and u8[0]['lines'] == [_PROSE[0]] and to8 == [0]),
    ]
    # review: speaker label of a removed line -> audit pair (label_drops)
    sp = [_PROSE[0], 'باجی گل:', _PROSE[1], 'باجی گل:', _PROSE[2], 'باجی گل:', _PROSE[3]]
    c9, u9, _, ld9 = units(sp, removed={4: {'reason': 'X'}})
    cases += [
        ('speaker label of a removed line is reported', c9['cat'][3] == 'speaker' and ld9 == [(3, 4)]
         and u9[0]['lines'][1] == 'باجی گل: ' + _PROSE[1]),
    ]
    # review: lexicon headwords that are genre words stay text
    lex = ['ماہیا:', '(ما۔ہی۔یا) ہک قسم دا لوک گیت', 'گیت:', '(گی۔ت) گاون دی چیز']
    c10 = classify(lex)
    before = list(c10['cat'])
    books._lexicon_categories(c10, lex)
    _, u10, _, _ = units(lex, lex=True)
    cases += [
        ('lexicon: genre-word headwords are text', before[0] in ('genre_heading', 'speaker')
         and c10['cat'][0] == 'text' and sum(len(u['lines']) for u in u10) == 4),
    ]
    _, u11, _, _ = units(['ہندکو', 'انگریزی', 'اُردو /معنی', 'آفس', 'Office', 'دفتر'], lex=True)
    _, u12, _, _ = units(['اُردو', 'آپ کیسے ہیں', 'ھندکو', 'تساں کیویں او'], lex=True)
    cases += [
        ('lexicon: column heads stay separate lines',
         [t for u in u11 for t in u['lines']] == ['ہندکو', 'انگریزی', 'اُردو /معنی', 'آفس', 'Office', 'دفتر']),
        ('lexicon: a language label prefixes its cell',
         [t for u in u12 for t in u['lines']] == ['اُردو: آپ کیسے ہیں', 'ھندکو: تساں کیویں او']),
    ]
    items = []
    for k in range(400):
        items.append(('سرناواں:' if k % 2 == 0 else 'ب' * (60 + (k * 7) % 50), k))
    ch = books.lexicon_chunks(items)
    cases += [
        ('lexicon chunks never end on a headword', len(ch) > 3 and all(
            not books._is_headword(x['lines'][-1]) for x in ch[:-1])
         and [i for x in ch for i in x['idx']] == list(range(400))),
    ]
    # review: loose_key identity - digits and (in lexicons) harakat matter
    cases += [
        ('dedup_key keeps digits', books.dedup_key('(جلد نمبر1، دسمبر 1989)') != books.dedup_key(
            '(جلد نمبر2، دسمبر 1991)') and books.dedup_key('سن ۱۹۹۱') == books.dedup_key('سن 1991')),
        ('dedup_key ignores punctuation / spacing / harakat', books.dedup_key('کتاب، دا ناں!')
         == books.dedup_key('کتاب دا  ناں') and books.dedup_key('اَو') == books.dedup_key('اُو')),
        ('dedup_hkey keeps harakat (lexicons)', books.dedup_hkey('اَو') != books.dedup_hkey('اُو')
         and books.dedup_hkey('کَدَّوْ') != books.dedup_hkey('کدو')),
    ]
    # review: X run membership - a short line is removed only inside a chain
    # of >= 3 lines copied from ONE source at increasing positions

    def loc(f, p):
        return (f << books._LOC_SHIFT) | p
    cases += [
        ('X: 3 short lines copied from 3 sources are no run',
         books.run_chain_members([[loc(1, 5)], [loc(2, 7)], [loc(3, 9)]]) == [None, None, None]),
        ('X: 3 short lines copied in order from one source are a run',
         None not in books.run_chain_members([[loc(1, 5)], [loc(1, 6)], [loc(1, 8)]])),
        ('X: a short template chain from another folder is not a copy',
         books.drop_template_chains([loc(1, 5), loc(1, 6), loc(1, 8)], [8, 3, 2], lambda f: False)
         == [None] * 3
         and None not in books.drop_template_chains([loc(1, 5), loc(1, 6), loc(1, 8)], [8, 3, 2],
                                                    lambda f: True)
         and None not in books.drop_template_chains([loc(1, 5), loc(1, 6), loc(1, 8)], [20, 18, 22],
                                                    lambda f: False)),
    ]
    # review / corrections L12: structural lexicon test (colon headwords >= 8%,
    # >= 90% distinct) + alphabetical order (a glossary vs topic headings)
    expl = 'اس لفظ دا مطلب اے پرانے ویلے دی ہک شے جیہڑی ہن نئیں ورتی جاندی'
    gl = [x for w in ['آری', 'اوڈی', 'اوگن', 'بار', 'پانی', 'تارا', 'جال', 'چاک', 'دار', 'ڈور', 'رات', 'سال']
          for x in (w + ':', expl)]
    tp = [x for w in ['توحید', 'بیعت', 'ادب', 'مجلس', 'ذکر', 'اخلاص', 'صبر', 'توبہ', 'شکر', 'امید']
          for x in (w + ':', expl)]
    sg, st = books.colon_headword_stats(gl), books.colon_headword_stats(tp)
    cases += [
        ('L12: alphabetical colon headwords = glossary structure', sg[0] >= books.LEX_COLON_SHARE
         and sg[1] >= books.LEX_COLON_DISTINCT and sg[2] >= books.LEX_COLON_SORTED),
        ('L12: topic headings are not in alphabetical order', st[0] >= books.LEX_COLON_SHARE
         and st[2] < books.LEX_COLON_SORTED),
    ]
    # review / corrections W1: a doubled region bridges <= 3 lines that are
    # fuzzy copies of the first-copy lines between the two aligned runs (a
    # revision merged two lines, so the offset changes)
    words = ['الف', 'بے', 'پے', 'تے', 'ٹے', 'ثے', 'جیم', 'چے', 'حے', 'خے', 'دال', 'ڈال', 'ذال']
    base = [books.dedup_key('ہک لمبی سطر جیہڑی %s والی اے تے اس وچ کافی حرف نیں' % w) for w in words]
    filler = [books.dedup_key('وچکار دی سطر %s %s' % (a, b)) for a in words for b in words[:3]]
    ks_ = base + filler + base[:5] + [base[5] + base[6]] + base[7:]
    w1_, _, _, runs_, gap_, _ = books.within_file_masks(ks_, [250] * len(ks_))
    g_at = len(base) + len(filler) + 5
    cases += [
        ('W1: a merged line between two doubled runs is bridged (W1_GAP)',
         len(runs_) == 2 and set(gap_) == {g_at} and gap_[g_at][0] in (5, 6) and w1_[g_at] is None),
    ]
    # review: roles / order / imprint parsing
    cases += [
        ('MATTLAN is the proverbs book, not a lexicon (both copies)',
         books.file_role(R + '99th Farma EXCLUSSIVE Hindko Mattlan/000 BOOK MATTLAN.INP') == 'main'
         and books.file_role(R + 'Dictionary with matlan adition/000 BOOK MATTLAN.INP') == 'main'
         and books.file_role(R + 'Dictionary with matlan adition/Hindko Lughat Par-1.inp') == 'lexicon'),
        ('Lughat parts are ordered at 97.5 (after 25th, before 98th)',
         books.order_no(R + '25th Farma Pothohari Dictionary by Tahir Shiraz/x.INP') == 25
         < books.order_no(R + 'Dictionary with matlan adition/Hindko Lughat Part-3.inp') == 97.5
         < books.order_no(R + '98th Farma EXCLUSSIVE Gandhara Hindko Dictionary/000 CONTENT.INP')),
        ('imprint: GHA label with (ابتدائی) is gha_ref',
         books.imprint_label_field('جی ایچ اے اشاعت (ابتدائی)') == 'gha_ref'),
        ('imprint: end word compared after norm (اِنتساب)', books.norm('اِنتساب') in books.IMPRINT_END_N),
        ('ISBN-13 check digit', books._isbn_check_ok('9780306406157')
         and not books._isbn_check_ok('9789696870374')),
    ]
    p358 = _parse_pairs([('edition', 'وار:', 'پہلی'), ('year', 'چھپنے دی تاریخ:', '۲ دسمبر ۲۰۰۵،'),
                         ('edition', 'وار:', 'دوئی ۔ اکتوبر ۲۰۱۸ء')])
    p173 = _parse_pairs([('year', 'سال اشاعت', 'اپریل1995ء'), ('year', 'سال اشاعت (دوم)', '2017ء')])
    pmay = _parse_pairs([('year', 'سال اشاعت', 'مء 2017ء')])
    pjul = _parse_pairs([('year', 'سال اشاعت', 'جولاء2016ئ')])
    cases += [
        ('imprint: edition pairs model the edition (358th)',
         (p358['publication_year'], p358['publication_date'], p358['edition'], p358['first_edition_year'])
         == (2018, '2018-10', 2, 2005)),
        ('imprint: labelled second edition (173rd)',
         (p173['publication_year'], p173['edition']) == (2017, 2)),
        ('imprint: years ending in ء / months مء and جولاء',
         pmay['publication_date'] == '2017-05' and pjul['publication_date'] == '2016-07'),
    ]
    cases += review3_unit_cases(books, lang, units, classify)
    bad = [n for n, ok in cases if not ok]
    ck.add('C0 lang.py + books.py unit checks', not bad, '%d/%d ok%s' % (
        len(cases) - len(bad), len(cases), (' FAILED: %s' % bad) if bad else ''))


def review3_unit_cases(books, lang, units, classify):
    """Synthetic cases for the third review round (2026-09-25): one per
    finding a small input can reproduce."""
    import numpy as np
    V = ['ساون دی رت آئی تے بدل وسیا', 'باغاں دے وچ پھل کھڑے سارے', 'ٹھنڈی ہوا نے دل نوں موہیا',
         'یاداں دے دیوے بلے سارے']
    out = []

    def flat(us):
        return [t for u in us for t in u['lines']]
    # labels: a colon headword before its definition is prefixed, no genre label
    c, us, _, _ = units(['بلحاظِ موضوع', 'حمد:', 'اوہ نظم جس بچ اللہ دی تعریف کیتی جلے۔ چاہے اوہ کسی وی ہیئت بچ ہووے۔',
                         'نعت:', 'اوہ نظم جس دا موضوع حضور دی تعریف تے مدح اے۔ ایہہ کسی وی ہیئت بچ لکھی جلدی اے۔'])
    out.append(('labels: colon headword + definition is one line, no genre label',
                'حمد: اوہ نظم جس بچ اللہ دی تعریف کیتی جلے۔ چاہے اوہ کسی وی ہیئت بچ ہووے۔' in flat(us)
                and all(u['genre'] is None for u in us)))
    # a language word inside a verse sentence stays text
    c, us, _, _ = units(['جس دی زبان پشتو ائی برے میرے نال', 'ہندکو', 'اِسراں بول دی ائی جیکرو اُس دی زبان',
                         'ہندکو', 'ہووے'])
    out.append(('labels: a language word inside a sentence is text, no language label',
                c['cat'][:5] == ['text'] * 5 and len(us) == 1 and us[0]['lang'] is None))
    # a word before running prose ('ہندکو' / 'بولنے والیآں دے ناں۔'; a greeting)
    c, us, _, _ = units(['انتساب', 'ہندکو', 'بولنے والیآں دے ناں۔', 'میر ے سرتاج', 'سلام',
                         'آپڑاں خیال رخیو، میری فکر نہ کریو، بے بے جی نوں میرا سلام آخیو تے بچیاں نوں پیار دیو۔'])
    out.append(('labels: a label word before prose is text', c['cat'][1] == 'text' and c['cat'][4] == 'text'
                and 'ہندکو' in flat(us) and 'سلام' in flat(us)))
    # a row of column heads
    c, us, _, _ = units(['نذیر بھٹی', 'ہندکو', 'پشتو', 'اُردو', 'آپ۔آپی'])
    out.append(('labels: a row of language column heads is text', c['cat'][1:4] == ['text'] * 3
                and all(u['lang'] is None for u in us)))
    # a genuine heading opens its unit and is its first line; a language label
    # holds for its block only
    c, us, _, _ = units(['۔۔۔۔۔', 'غزل'] + V + ['۔۔۔۔۔', 'اُردو'] + V[:2] + ['۔۔۔۔۔'] + V[2:])
    out.append(('labels: a heading is the first line of its unit; language label scoped to its block',
                c['cat'][1] == 'genre_heading' and us[0]['lines'][0] == 'غزل' and us[0]['genre'] == 'غزل'
                and us[1]['lines'][0] == 'اُردو' and us[1]['lang'] == 'اُردو' and us[2]['lang'] is None))
    # L2: lone ':' / '=' between text frames join the line; lone '…' is a boundary
    c, us, _, _ = units(['سین نمبر 1', 'وخت', ':', 'دن', 'نامسلم', '=', 'اجازت اے۔'])
    out.append(('L2: lone : and = between text frames are joiners', c['cat'][2] == 'joiner'
                and c['cat'][5] == 'joiner' and flat(us) == ['سین نمبر 1', 'وخت: دن', 'نامسلم = اجازت اے۔']))
    c, us, _, _ = units(V[:2] + ['…'] + V[2:])
    out.append(('L2: a lone … between text frames is a separator', c['cat'][2] == 'separator' and len(us) == 2))
    c = classify(['ملکؔ پیار کر کے کے کَھٹیا اے اُلٹا سرے اُتے الزام آیا', 'ٹ',
                  'ٹ ٹُر گئے وانگ مسافراں دے تیرے درتوں کہن کے چہول خالی', 'مٹھے بول', 'س',
                  'لے کے ناں میں سوہنیط رب دا کراں کلام بیان'])
    out.append(('L2: a si-harfi letter heading is text, an ornament letter is not',
                c['cat'][1] == 'text' and c['cat'][4] == 'residue'))
    # R8: English between text frames, abbreviations, bracketed glosses, docx
    c = classify(['بی کام', 'B.Com', 'بی کام (ڈگری)/ تجارت کی ڈگری', 'ایل ایل بی', 'L.L.B', 'قانون دی ڈگری',
                  'گٹار', '(Guitar)', 'ہک ساز', 'انِ دین_', 'atool', '^'])
    out.append(('R8: English words / abbreviations / glosses between text frames are text',
                c['cat'][1] == 'text' and c['cat'][4] == 'text' and c['cat'][7] == 'text' and c['cat'][10] == 'residue'))
    c = classify(['INDEX', 'Abdul sittar 44', 'KPK 29 58'], 'docx')
    out.append(('R8: every Latin frame of a docx is English text', c['cat'] == ['english_text'] * 3))
    out.append(('R8: font strings vs real words', lang.latin_frame_category('rad Arabic Bold', 1) == 'font_style'
                and lang.is_english_word_frame('Black Mailing', 1) and lang.is_english_word_frame('Red Cross', 1)
                and lang.latin_frame_category('B.Com', 1) != 'email_url'
                and lang.is_english_text_frame('(Ultra Violet Rays)۔ Heat and light of the sun', 1)))
    # isolated fragments amid residue
    c = classify(['@', '~', '@', 'x', '@', 'اٹ', '@', '~', '@', '@', '~'] + V)
    out.append(('fragment: an isolated 2-letter frame amid residue', c['cat'][5] == 'fragment'
                and c['fragments'] == [[5, 'اٹ']]))
    # credits and imprint labels
    out.append(('credits: compound labels by role', books.credit_roles('مترجم/ شاعر') == ['translator', 'author']
                and books.credit_roles('تحقیق ،تحریروانڈیکس') == ['compiler', 'author', 'compiler']
                and books.credit_roles('محقق، مرتبہ') == ['compiler', 'compiler']
                and books.credit_roles('ترتیب') is None and books.imprint_label_field('ترتیب') is None))
    out.append(('imprint: اشاعت سوئم is the third edition', books.imprint_label_field('اشاعت سوئم') == 'year'
                and books.edition_of('اشاعت سوئم') == 3))
    out.append(('imprint: Hindko rights line and the typo anchor',
                bool(books.RIGHTS.search('سارے حق شاعر دے حق بِچ محفوظ ہین'))
                and bool(books.RIGHTS.search('جملہ حقوب بحق گندھارا ہندکو اکیڈمی محفوظ'))))
    imp = ['جملہ حقوق بحق گندھارا ہندکو اکیڈمی محفوظ نیں', 'نام کتاب', 'کتاب دا ناں', 'مصنف', 'مصنف دا ناں',
           'سال اشاعت', '2017ء', 'قیمت', '400روپے', 'ملنے دا پتہ', 'گندھارا ہندکو اکیڈمی پشاور',
           '091-0000000,0000001', 'ایصال ثواب', 'اپنڑے مرحوم جگر گوشے', 'فلاں فلاں', 'دے ناں']
    w = books.find_imprints(imp)
    out.append(('imprint: the window ends after the address block (dedication stays text)',
                len(w) == 1 and w[0]['frame_lines'] == list(range(12))))
    blk = _parse_pairs([('publisher', 'پبلشرز:', 'میاں سمیع اللہ قریشی'), ('edition', 'وار:', 'پہلی'),
                        ('year', 'چھپنے دی تاریخ:', '۲ دسمبر ۲۰۰۵،'), ('edition', 'وار:', 'دوئی ۔ اکتوبر ۲۰۱۸ء'),
                        ('printer', 'چھاپہ خانہ:', 'جی ایچ اے لیزر پرنٹنگ پشاور')])
    out.append(('imprint: publisher of another edition block is not used (358th)',
                blk['publication_year'] == 2018 and blk['publisher'] is None
                and blk['printer'] == 'جی ایچ اے لیزر پرنٹنگ پشاور'))
    cr = _parse_pairs([('author', 'مرتبہ', 'نسیم سحرؔ'), ('author', 'مترجم/ شاعر', 'سید سعید گیلانی')])
    out.append(('imprint: credits by role', cr['compiler'] == 'نسیم سحرؔ' and cr['author'] == 'سید سعید گیلانی'
                and cr['translator'] == 'سید سعید گیلانی' and cr['author_label'] == 'مترجم/ شاعر'))
    # TOC: key, byline pairing
    out.append(('TOC key ignores the era sign and spacing', books.toc_key('راشد جاوید 1956 ء') == books.toc_key('راشد جاوید 1956')))
    tb = books.parse_tocs(['فہرست', '1', 'حروفِ میزان', 'سید محمد نورالحسنین', '15', '2', 'سخن ہائے گفتنی',
                           'منظور الٰہی قادری', '18', '3', 'علم دا مرکز', 'ڈاکٹر اُمّ سلمیٰ', '17'])
    out.append(('TOC: heading / byline rows pair up', len(tb) == 1 and [e[1] for e in tb[0]['entries']]
                == ['حروفِ میزان', 'سخن ہائے گفتنی', 'علم دا مرکز'] and tb[0]['bylines'][2][1] == 'سید محمد نورالحسنین'))
    # L8: an ambiguous heading ends the running title
    toc = ['پہلا باب', 'دوجا باب']
    _, u9, _, _ = units(['پہلا باب', _PROSE[0], 'دوجا باب', _PROSE[1], 'دوجا باب', _PROSE[2]], toc=toc)
    out.append(('L8: a title does not run on over an ambiguous heading',
                [u['title'] for u in u9] == ['پہلا باب', None]))
    # L7 verse cuts at stanza heads
    items = []
    for k in range(12):
        items.append(('حرفی نمبر:- %d' % k, len(items)))
        for m in range(19):
            items.append((V[m % 4] + ' ' + 'ا' * (m % 3), len(items)))
    ch = books._verse_cuts(items)
    out.append(('L7: verse units are cut before stanza headings', len(ch) > 1 and all(
        c_[0][0].startswith('حرفی نمبر') for c_ in ch[1:]) and sum(len(c_) for c_ in ch) == len(items)))
    # F uncovered stretch
    spans, unc, run = books._uncovered(30, np.array([True] * 5 + [False] * 16))
    out.append(('F: uncovered key chars and spans', unc == 16 and run == 16 and spans == [[14, 30]]))
    # genre / folder names
    out.append(('genre: imprint subject first', lang.book_genre('111th Farma Wafa', [], ' '.join(['فلم'] * 50), 'prose',
                                                                 'سرگزشت/تحقیق') == ('biography', 'imprint_subject:سرگزشت/تحقیق')
                and lang.book_genre('306th Farma Dramay Day Fani Taqazay', [], '', 'prose', None,
                                    'ڈرامہ دے فنی تقاضے')[0] == 'research_criticism'
                and lang.book_genre('135th Farma X', [], '', 'mixed', 'ہندکو شاعری/نثر') == (None, None)))
    # fix5: a numbered heading that matches a TOC entry only without its
    # number ends the running title (3rd Chaarbita / 53rd stale titles)
    _, u10, _, _ = units(['پہلا باب', _PROSE[0], '۱۔ دوجا باب', _PROSE[1]], toc=['پہلا باب', 'دوجا باب'])
    out.append(('L8: a title ends at a heading matching another entry without its number',
                [u['title'] for u in u10] == ['پہلا باب', None] and u10[1]['lines'][0] == '۱۔ دوجا باب'))
    # fix5: L7 never cuts between two stanza heads or right after a head
    items = []
    for k in range(12):
        items.append(('حرفی نمبر:- %d' % k, len(items)))
        items.append(('موضوع: فریاد %d' % k, len(items)))
        for m in range(19):
            items.append((V[m % 4] + ' ' + 'ا' * (m % 3), len(items)))
    ch = books._verse_cuts(items)
    out.append(('L7: a verse chunk never ends on a stanza head', len(ch) > 1 and all(
        not books._is_stanza_head(c_[-1][0]) for c_ in ch[:-1]) and all(
        c_[0][0].startswith('حرفی نمبر') for c_ in ch[1:])))
    # fix5: a unit whose char share says prose but whose lines are mostly
    # verse (a bio paragraph before the poem, 3rd Chaarbita) is cut by couplet
    bio = ' '.join(_PROSE) * 3
    vl = [V[m % 4] + ' ' + 'ا' * (m % 3) for m in range(300)]
    ul = [bio[:790]] * 8 + vl
    mr = books.make_records([{'lines': ul, 'idx': list(range(len(ul))), 'title': None, 'lang': None,
                              'genre': None, 'section': 0}])
    cnt, odd = 0, 0
    for k_, r_ in enumerate(mr):
        for ln in r_['lines']:
            cnt = 0 if books._is_stanza_head(ln) else cnt + 1
        if k_ < len(mr) - 1 and not books._is_stanza_head(mr[k_ + 1]['lines'][0]) and cnt % 2:
            odd += 1
    out.append(('L7: mostly-verse lines are cut by couplet even when the chars say prose',
                books.unit_kind(ul) == 'prose' and len(mr) > 1 and odd == 0))
    # fix5: a style-name word between two cells of a lexicon is an English entry
    lx = ['ہلکا', 'Light', 'ہولا', 'درمیانہ', 'Medium', 'وچکارلا']
    c = classify(lx)
    books._lexicon_categories(c, lx)
    out.append(('lexicon: a style-name word between text cells is text (208th Light / Medium)',
                c['cat'][1] == 'text' and c['cat'][4] == 'text'))
    out.append(('R8: a stray ) before a bracketed gloss', lang.is_english_word_frame(') (Ladyfinger', 1, True)))
    out.append(('F: only trivial uncovered stretches are removed', books.F_KEEP_RUN <= 6 and books.F_KEEP_SHARE <= 0.02))
    out.append(('folder names lose working-copy tokens',
                books.clean_folder_value('Elahi Bakhsh Akhtar Awan - Copy') == 'Elahi Bakhsh Akhtar Awan'
                and books.clean_folder_value('EXCLUSSIVE Hindko Mattlan') == 'Hindko Mattlan'
                and books.clean_folder_value('Mehkiyan Gallan Amn Mushaira 4th Int Conference Mushaira')
                == 'Mehkiyan Gallan Amn Mushaira'))
    return out


def run_checks(out, files, decoded, news, secs, cpu, ck: Checks):
    from hp import books, lang, segment
    recs = out['records']
    rep = out['report']
    lines_of = {rel: (decoded[rel]['text_all'].split('\n') if decoded[rel]['text_all'] else [])
                for rel, _, _ in files}
    non_toc = [r for r in recs if r['role'] != 'toc']
    pfm = {pf['source_path']: pf for pf in out['per_file']}
    role_of = {p: pf['file_role'] for p, pf in pfm.items()}
    folder_pf = defaultdict(list)
    for pf in out['per_file']:
        folder_pf[pf['book_folder']].append(pf)

    def by_folder(prefix):
        return [pf for fo, pfs in sorted(folder_pf.items()) if fo.startswith(prefix) for pf in pfs]

    kd, kh = {}, {}

    def dkey(t):
        """books.dedup_key, memoised (one key computation per distinct text)."""
        v = kd.get(t)
        if v is None:
            v = kd[t] = books.dedup_key(t)
        return v

    def hkey(t):
        v = kh.get(t)
        if v is None:
            v = kh[t] = books.dedup_hkey(t)
        return v

    def own_key(p, t):
        return hkey(t) if role_of.get(p) == 'lexicon' else dkey(t)

    # C1 counts and char totals by role / form / genre
    for k in ('records_by_role', 'chars_by_role', 'records_by_form', 'chars_by_form',
              'records_by_genre', 'chars_by_genre', 'records_by_language', 'removed_by_reason',
              'chars_by_disposition'):
        print('  %-20s %s' % (k, rep[k]))
    ck.add('C1 records produced', len(recs) > 0, '%d records, %d chars' % (len(recs), rep['record_chars']))

    # C2 kept vs raw. The probe's 13-20M was measured with book lines that
    # repeat newspaper text removed; since R8 (BOOKS_SPEC_CORRECTIONS) they
    # stay in the book records (flag also_in_newspaper(share=x.xx)), so the
    # range is tested on the kept chars minus that overlap.
    kept = sum(r['n_chars'] for r in non_toc)
    raw, text = rep['raw_frame_chars'], rep['text_frame_chars']
    news_ch = 0.0
    for r in non_toc:
        for f in r['content_flags']:
            if f.startswith('also_in_newspaper(share='):
                L = lines_of[r['source_path']]
                news_ch += float(f[len('also_in_newspaper(share='):-1]) * sum(len(L[i]) for i in r['frame_indices'])
    ck.add('C2 kept chars in the probe range 13-20M', 13_000_000 <= kept - news_ch <= 20_000_000,
           'kept %d of %d text-frame chars (%.1f%%), %d without the newspaper overlap kept by R8 (~%d); '
           'raw frames %d' % (kept, text, 100.0 * kept / max(1, text), kept - news_ch, news_ch, raw), soft=True)

    # C3 no substantive exact duplicate lines across files. Copy identity as
    # in books (review: loose keys merged digit / harakat variants):
    # dedup_hkey for any pair involving a lexicon line, dedup_key otherwise.
    h_files, k_files = defaultdict(set), defaultdict(set)
    rec_frames = defaultdict(set)
    for r in recs:
        rec_frames[r['source_path']].update(r['frame_indices'])
    for r in non_toc:
        p = r['source_path']
        L = lines_of[p]
        for i in r['frame_indices']:
            if len(own_key(p, L[i])) < books.SUBST:
                continue
            h_files[hkey(L[i])].add(p)
            if role_of[p] != 'lexicon':
                k_files[dkey(L[i])].add(p)
    cross = sorted(k for k, s in h_files.items() if len(s) > 1) + sorted(
        k for k, s in k_files.items() if len(s) > 1)
    ck.add('C3 0 substantive exact dups across files', not cross,
           '%d keys in >1 file (of %d substantive keys)' % (len(cross), len(h_files)))

    # C3b R8 (BOOKS_SPEC_CORRECTIONS): no book line is removed because it is in
    # the newspaper; records holding kept newspaper lines are flagged
    # also_in_newspaper(share=x.xx) instead, with the share recomputed here
    news_k, news_h = set(), set()
    for _, t in news:
        for ln in t.split('\n'):
            k = dkey(ln)
            if len(k) >= books.SUBST:
                news_k.add(k)
                news_h.add(hkey(ln))
    by_news = [d for d in out['line_dedup'] if d['duplicate_source'] == 'newspaper']
    unflagged, low_share, n_in_news = [], [], 0
    for r in non_toc:
        p = r['source_path']
        L = lines_of[p]
        lex = role_of[p] == 'lexicon'
        hit = sum(len(L[i]) for i in r['frame_indices']
                  if len(own_key(p, L[i])) >= books.SUBST
                  and ((hkey(L[i]) in news_h) if lex else (dkey(L[i]) in news_k)))
        if not hit:
            continue
        n_in_news += 1
        fl = [f for f in r['content_flags'] if f.startswith('also_in_newspaper(share=')]
        if not fl:
            unflagged.append((books.book_rel(p)[:40], r['passage_index']))
            continue
        share = float(fl[0][len('also_in_newspaper(share='):-1])
        tot = sum(len(L[i]) for i in r['frame_indices'])
        if share + 0.006 < hit / tot:       # the flag also counts short lines of copied passages
            low_share.append((books.book_rel(p)[:40], r['passage_index'], share, round(hit / tot, 3)))
    ov = rep.get('newspaper_overlap_share_by_folder', {})
    ck.add('C3b R8: newspaper overlap flagged, not removed', not by_news and not unflagged and not low_share,
           '%d rows removed for the newspaper; %d records hold kept newspaper lines, %d unflagged %s, '
           '%d with a low share %s; %d records flagged; top book overlap %s' % (
               len(by_news), n_in_news, len(unflagged), unflagged[:3], len(low_share), low_share[:3],
               rep['records_also_in_newspaper'], list(ov.items())[:5]))

    # C4 W1 doubled files == probe list
    pats = doubled_expected()
    got_rel = sorted(books.book_rel(pf['source_path']) for pf in out['per_file']
                     if 'inpage_duplicate_story_removed' in pf['flags'])
    matched, unmatched = set(), []
    for a, b in pats:
        hit = [p for p in got_rel if p.startswith(a) and p.endswith(b)]
        if len(hit) == 1:
            matched.add(hit[0])
        else:
            unmatched.append(a + '...' + b)
    extra = [p for p in got_rel if p not in matched]
    ck.add('C4 W1 doubled files == probe list (14)', len(pats) == 14 and not unmatched and not extra,
           'found %d, expected %d; missing %s; extra %s' % (len(got_rel), len(pats), unmatched, extra))

    # C5 record size
    big = [r for r in non_toc if r['n_chars'] > 12000 and len(r['frame_indices']) > 1]
    big1 = [r for r in non_toc if r['n_chars'] > 12000 and len(r['frame_indices']) == 1]
    ck.add('C5 no record >12k except single frames', not big,
           '%d multi-frame >12k; %d single-frame >12k; max %d' % (
               len(big), len(big1), max((r['n_chars'] for r in non_toc), default=0)))

    # C6 short records
    short = [r for r in non_toc if r['n_chars'] < 200]
    unflagged_s = [r for r in short if 'short_unit' not in r['content_flags']]
    share = len(short) / max(1, len(non_toc))
    ck.add('C6 records <200 chars <= ~15%, all flagged', share <= 0.15 and not unflagged_s,
           '%d of %d (%.1f%%), %d chars (%.2f%% of kept); unflagged %d' % (
               len(short), len(non_toc), 100 * share, sum(r['n_chars'] for r in short),
               100.0 * sum(r['n_chars'] for r in short) / max(1, kept), len(unflagged_s)))

    # C7 titles
    badt = [r for r in recs if (r['title'] is not None) != (r['title_source'] == 'toc_entry')]
    ck.add('C7 title only with title_source=toc_entry', not badt,
           '%d titled records; %d inconsistent' % (sum(1 for r in recs if r['title']), len(badt)))

    # C8 every metadata value has a source (and every source has a value);
    # review: first_edition_year_source, per-field imprint source paths,
    # author_role, the edition bundle from ONE imprint, isbn_invalid_checksum
    pairs = [('book_series_number', 'book_series_number_source'), ('book_title', 'book_title_source'),
             ('author', 'author_source'), ('translator', 'translator_source'), ('compiler', 'compiler_source'),
             ('publisher', 'publisher_source'),
             ('publication_year', 'publication_year_source'), ('edition', 'edition_source'),
             ('first_edition_year', 'first_edition_year_source'),
             ('isbn', 'isbn_source'), ('gha_ref', 'gha_ref_source'), ('genre', 'genre_evidence'),
             ('language', 'language_source')]
    miss = Counter()
    ed_fields = ('publication_year', 'publication_date', 'publication_date_precision', 'edition',
                 'first_edition_year', 'isbn', 'gha_ref')
    imprint_fields = ('book_title', 'author', 'publisher') + ed_fields
    imp_parsed = {}
    for imp in out['imprints']:
        imp_parsed[imp['source_path']] = [imp['parsed']] + list(imp['parsed_other_windows'])
    for r in recs:
        for v, s in pairs:
            if (r.get(v) is not None) != (r.get(s) is not None):
                miss['%s/%s' % (v, s)] += 1
        if (r.get('publication_date') is not None) != (r.get('publication_year') is not None) or \
                (r.get('publication_date_precision') is not None) != (r.get('publication_year') is not None):
            miss['publication_date/precision'] += 1
        src = r.get('imprint_field_sources') or {}
        for f in imprint_fields:
            # review 2026-09-25: a creator read from the title page before the
            # rights line is part of that file's imprint window too
            from_imprint = r.get(f) is not None and (f not in ('book_title', 'author')
                                                      or r.get(f + '_source') in ('imprint', 'title_page'))
            if from_imprint != (f in src):
                miss['imprint_field_sources:%s' % f] += 1
            elif f in src and all(p_.get(f) != r[f] for p_ in imp_parsed.get(src[f], [])):
                miss['value_not_in_named_imprint:%s' % f] += 1
        if len({src[f] for f in ed_fields if f in src}) > 1:
            miss['edition_bundle_from_2_files'] += 1
        if (r.get('author_source') in ('imprint', 'title_page')) != (r.get('author_role') is not None):
            miss['author_role'] += 1
        want_flag = r['isbn'] is not None and not books._isbn_check_ok(r['isbn'])
        if want_flag != ('isbn_invalid_checksum' in r['content_flags']):
            miss['isbn_invalid_checksum'] += 1
    isbn_bad = sum(1 for r in recs if r['isbn'] is not None and not re.fullmatch(r'\d{13}', r['isbn']))
    year_bad = sum(1 for r in recs if r['publication_year'] is not None and not isinstance(r['publication_year'], int))
    ck.add('C8 every metadata value has a source', not miss and not isbn_bad and not year_bad,
           'violations %s; bad ISBN %d; non-int years %d; isbn_invalid_checksum records %d' % (
               dict(miss), isbn_bad, year_bad,
               sum(1 for r in recs if 'isbn_invalid_checksum' in r['content_flags'])))

    # C9 imprint coverage (report)
    bm = out['book_meta']
    numbered = [fo for fo in bm if bm[fo]['book_series_number'] is not None]
    with_imp = [fo for fo in bm if bm[fo]['imprint_files']]

    def has(f):
        return [fo for fo in bm if bm[fo]['imprint_values'].get(f) is not None or bm[fo]['imprint_conflict'].get(f)]
    ck.add('C9 imprint coverage (~87 of 95+ folders)', len(has('publication_year')) >= 80,
           '%d folders (%d numbered); imprint block %d; year %d, publisher %d, ISBN %d, title %d, '
           'edition %d, gha_ref %d; conflict folders %s' % (
               len(bm), len(numbered), len(with_imp), len(has('publication_year')), len(has('publisher')),
               len(has('isbn')), len(has('book_title')), len(has('edition')), len(has('gha_ref')),
               rep['imprint_conflict_folders']), soft=True)

    # C10 L13 (corrections): 289th -> no records; 250th / 255th front matter
    # only -> body_missing_from_source with records; no frame-count flag
    r289 = by_folder('289th')
    ok289 = bool(r289) and all(pf['records'] == 0 and 'no_extractable_text' in pf['flags'] for pf in r289)
    bmiss = by_folder('250th') + by_folder('255th')
    okbm = len(bmiss) >= 2 and all('body_missing_from_source' in pf['flags'] and pf['records'] > 0
                                   for pf in bmiss)
    nte = [books.book_rel(pf['source_path']) for pf in out['per_file'] if 'no_extractable_text' in pf['flags']]
    little = sum(1 for pf in out['per_file'] if 'little_decodable_text' in pf['flags'])
    ck.add('C10 L13: 289th no records; 250th/255th body missing', ok289 and okbm and not little
           and len(nte) == len(r289),
           '289th %s; 250th/255th %s; no_extractable_text files %s; little_decodable_text %d' % (
               [(pf['records'], pf['text_chars']) for pf in r289],
               [(pf['records'], pf['flags']) for pf in bmiss], nte, little))
    small = [(books.book_rel(pf['source_path'])[:70], pf['text_frames'], pf['text_chars'], pf['records'])
             for pf in out['per_file'] if pf['text_chars'] < 2000]
    print('  files with < 2,000 text chars (path, frames, chars, records):')
    for x in small:
        print('     %s' % (x,))

    # C12 conservation per file
    dd_by_file = defaultdict(set)
    dd_row = {}
    for d in out['line_dedup']:
        dd_by_file[d['source_path']].add(d['line_index'])
        dd_row[(d['source_path'], d['line_index'])] = d
    structural = set(books.STRUCTURAL) | {'toc_title', 'no_extractable_text'}
    bad_files = []
    for pf in out['per_file']:
        p = pf['source_path']
        n = len(lines_of[p])
        disp = pf['disposition_lines']
        n_struct = sum(v for k, v in disp.items() if k in structural)
        a, b = rec_frames[p], dd_by_file[p]
        ok = (pf['n_lines'] == n and not (a & b) and len(a) + len(b) + n_struct == n
              and disp.get('record', 0) == len(a) and disp.get('dedup', 0) + disp.get('toc_copy', 0) == len(b)
              and not (set(disp) - structural - {'record', 'dedup', 'toc_copy'}))
        if not ok:
            bad_files.append((p, n, len(a), len(b), n_struct, dict(disp)))
    ck.add('C12 conservation in = records + dedup + structural', not bad_files,
           '%d files, %d violations %s' % (len(out['per_file']), len(bad_files), bad_files[:2]))

    # C12b every removed line points at a kept copy; C12c exact removals really
    # are copies: equal copy keys (digits always; harakat when a lexicon is involved)
    news_ids = {k for k, _ in news}
    kept_loc = {'%s#%d' % (r['source_path'], i) for r in recs for i in r['frame_indices']}
    title_set = {p: set(pf['toc_title_lines']) for p, pf in pfm.items()}
    files_with_records = {r['source_path'] for r in recs}
    bad_audit, mism, mism_ex = Counter(), Counter(), []
    for d in out['line_dedup']:
        tgt = d['duplicate_of']
        if d['duplicate_source'] == 'newspaper':
            ok = tgt in news_ids
        elif '#' not in tgt:          # F: covered by one earlier file's kept lines
            ok = tgt in files_with_records
        elif d['reason'] in ('L4', 'TOC_COPY') and d.get('via') is None:
            ok = True                 # copy of a structural frame (title page / imprint / TOC)
        else:
            ok = tgt in kept_loc
            if not ok:                # kept copy consumed as a TOC title (metadata)
                p, _, ix = tgt.rpartition('#')
                ok = p in lines_of and int(ix) in title_set[p]
        if not ok:
            bad_audit[d['reason']] += 1
        if d['reason'] in ('W1', 'W2', 'X', 'B', 'L4') and '#' in tgt and d.get('via') == 'W1_GAP':
            # an exact copy of a line that was itself a W1_GAP fuzzy copy: the
            # union of the first-copy lines it names must cover it (>= 0.80)
            p = d['source_path']
            a_ = set(books.ngram_hashes(own_key(p, lines_of[p][d['line_index']])).tolist())
            b_ = set(h for x in d['copy_lines'] for h in books.ngram_hashes(own_key(p, lines_of[p][x])).tolist())
            if not a_ or len(a_ & b_) / len(a_) < books.FUZZ - 1e-9:
                mism[d['reason'] + '/W1_GAP'] += 1
        elif d['reason'] in ('W1', 'W2', 'X', 'B', 'L4') and '#' in tgt:
            p, _, ix = tgt.rpartition('#')
            src_line = lines_of[d['source_path']][d['line_index']]
            kept_line = lines_of[p][int(ix)]
            lex = role_of[p] == 'lexicon' or role_of[d['source_path']] == 'lexicon'
            if d.get('via') and d['via'] != d['reason']:
                lex = role_of[p] == 'lexicon' and role_of[d['source_path']] == 'lexicon'
            same = (hkey(src_line) == hkey(kept_line) if lex
                    else dkey(src_line) == dkey(kept_line))
            if not same:
                mism[d['reason']] += 1
                if len(mism_ex) < 4:
                    mism_ex.append((d['reason'], d.get('via'), src_line[:40], kept_line[:40]))
    ck.add('C12b every removed line points at a kept copy', not bad_audit,
           '%d removal rows; unresolved %s' % (len(out['line_dedup']), dict(bad_audit)))
    ck.add('C12c exact removals equal their kept copy (digits, harakat)', not mism,
           'mismatches %s %s' % (dict(mism), mism_ex))

    # C13 runtime
    ck.add('C13 runtime < 10 min', secs < 600, 'process_books wall %.1fs, cpu %.1fs' % (secs, cpu), soft=True)

    # C14 no silent loss (review: L3 'outside' cut, TOC tails, imprint end
    # words, lexicon headwords): every frame with >= 8 Arabic letters that is
    # in no record, no line_dedup row and no TOC title is structural text that
    # is KEPT elsewhere: a running header (per_file running_headers), a TOC
    # block of a toc record, an imprint window (imprints[].windows[].frames),
    # an in-text genre / language label, or a speaker label prefixed to a
    # record line; the only exception is a > 200-char frame under 30% script
    # (L5 binary/style blob).
    toc_ranges = defaultdict(list)
    rec_lines_by_file = defaultdict(list)
    for r in recs:
        if r['role'] == 'toc':
            toc_ranges[r['source_path']].append(tuple(r['toc_block_lines']))
        else:
            rec_lines_by_file[r['source_path']].extend(r['lines'])
    imp_frames = defaultdict(set)
    for imp in out['imprints']:
        for w in imp['windows']:
            imp_frames[imp['source_path']].update(i for i, _ in w['frames'])
    AL = books.AR_LETTER
    expl, unexpl = Counter(), []
    for pf in out['per_file']:
        p = pf['source_path']
        if 'no_extractable_text' in pf['flags']:
            continue
        L = lines_of[p]
        heads = {h['text'] for h in pf['running_headers']}
        tl = set(pf['toc_title_lines'])
        lex = pf['file_role'] == 'lexicon'
        prefixes = None
        for i, t in enumerate(L):
            if i in rec_frames[p] or i in dd_by_file[p] or i in tl:
                continue
            # review 2026-09-25: genre / language words ('ہندکو' 5 letters,
            # 'حمد:' 3) were removed from text as 'labels' below the 8-letter
            # floor and counted as explained; a label is now a record line
            # (or a speaker-style prefix), so every label frame is checked
            kt = books.label_key(t) if len(t) <= 40 else ''
            is_label = kt in books.GENRE_HEADINGS or kt in books.LANG_LABELS
            if not is_label and (len(t) < 8 or len(AL.findall(t)) < 8):
                continue
            if t in heads:
                expl['header_run'] += 1
            elif any(a <= i <= b for a, b in toc_ranges[p]):
                expl['toc'] += 1
            elif i in imp_frames[p]:
                expl['imprint'] += 1
            elif len(t) > 200 and segment.script_fraction(t) < 0.30:
                expl['low_script_blob'] += 1
            else:
                if prefixes is None:
                    prefixes = {x.split(': ', 1)[0] for x in rec_lines_by_file[p] if ': ' in x}
                if len(t) <= books.SPEAKER_MAX and (t.rstrip(':۔- ').strip() in prefixes or kt in prefixes):
                    expl['speaker'] += 1
                else:
                    unexpl.append((books.book_rel(p)[:45], i, t[:50]))
    ck.add('C14 no frame with >= 8 letters dropped silently', not unexpl,
           'kept as structure %s; unexplained %d %s' % (dict(expl), len(unexpl), unexpl[:4]))

    # C16 F (review: newspaper union / named source): recompute every F row's
    # 10-gram coverage from the kept lines of the ONE file it names
    import numpy as np
    src_cache = {}

    def src_grams(p, keyf):
        k = (p, keyf.__name__)
        if k not in src_cache:
            fr = sorted((rec_frames[p] - set(pfm[p]['label_lines_in_records'])) | set(pfm[p]['toc_title_lines']))
            # substantive by the source file's OWN key (as the index was built)
            ks = [keyf(lines_of[p][i]) for i in fr if len(own_key(p, lines_of[p][i])) >= books.SUBST]
            src_cache[k] = (np.unique(np.concatenate(books.ngram_hashes_many(ks))) if ks
                            else np.zeros(0, np.uint64))
        return src_cache[k]
    fbad, fmis, nf = [], 0, 0
    for d in out['line_dedup']:
        if d['reason'] != 'F':
            continue
        nf += 1
        keyf = hkey if role_of[d['source_path']] == 'lexicon' else dkey
        h = np.unique(books.ngram_hashes(keyf(lines_of[d['source_path']][d['line_index']])))
        g = src_grams(d['duplicate_of'], keyf)            # sorted, unique
        cov = 0.0
        if len(h) and len(g):
            pos = np.searchsorted(g, h)
            pos[pos >= len(g)] = 0
            cov = float((g[pos] == h).mean())
        if cov < books.FUZZ - 1e-9:
            fbad.append((books.book_rel(d['source_path'])[:40], d['line_index'], round(cov, 3), d['coverage']))
        if abs(cov - d['coverage']) > 1e-3:
            fmis += 1
    ck.add('C16 F coverage >= 0.80 by the ONE named file', not fbad and not fmis,
           '%d F rows; below 0.80 %d %s; reported coverage differs %d' % (nf, len(fbad), fbad[:3], fmis))

    # C17 X runs (review: 307th scene headers). The X pass is replayed from
    # the output: files in processing order, each file's text-line sequence
    # (record frames + TOC titles + removed lines), and the kept lines of the
    # files processed before it as the candidate copies. Every short X row
    # (< 40 key chars) must lie in a chain of >= 3 consecutive lines whose
    # kept copies are in ONE earlier file at increasing positions <= 3 apart
    # (and, outside lexicons, a chain from another book folder must hold >= 40
    # key chars: templates) - and must name exactly that chain's copy.
    rows_by_file = defaultdict(dict)
    for d in out['line_dedup']:
        rows_by_file[d['source_path']][d['line_index']] = d
    order = sorted((pf['processing_order'], p) for p, pf in pfm.items() if pf['processing_order'] is not None)
    K_n, H_n, H_l = defaultdict(list), defaultdict(list), defaultdict(list)
    fid_path, seq_of = {}, {}
    xbad, nx_short, nx_canon = [], 0, 0
    for fid, (_, p) in enumerate(order, 1):
        rows = rows_by_file[p]
        tl = set(pfm[p]['toc_title_lines'])
        seq = sorted((rec_frames[p] - set(pfm[p]['label_lines_in_records'])) | tl
                     | {i for i, d in rows.items() if d['reason'] not in ('LABEL_OF_REMOVED_LINE', 'TOC_COPY')})
        fid_path[fid], seq_of[fid] = p, seq
        L = lines_of[p]
        lex = role_of[p] == 'lexicon'
        dk = [dkey(L[i]) for i in seq]
        hk = [hkey(L[i]) for i in seq]
        own = hk if lex else dk
        cands = [books._cands(H_n, H_l, hk[t], hk[t]) if lex else books._cands(K_n, H_l, dk[t], hk[t])
                 for t in range(len(seq))]
        chain = books.run_chain_members(cands)
        if not lex:
            fo_p = books.folder_of(p)
            chain = books.drop_template_chains(chain, [len(x) for x in own],
                                               lambda f: books.folder_of(fid_path[f]) == fo_p)
        for t, i in enumerate(seq):
            d = rows.get(i)
            if d is not None and d['reason'] == 'X' and d.get('x_rule') in ('superseded_canonical',
                                                                           'superseded_run'):
                # review 2026-09-25 (R7/X): in a superseded file a line kept in
                # its own canonical, or inside a run of >= 3 consecutive
                # already-seen lines, is a copy whatever the one-source chain
                # (the 98th '- Copy' twin kept 2,321 such lines); C12c checks
                # the key
                nx_canon += 1
                if pfm[p]['superseded_by'] is None or (
                        d['x_rule'] == 'superseded_canonical'
                        and pfm[p]['superseded_by'] != d['duplicate_of'].rpartition('#')[0]):
                    xbad.append((books.book_rel(p)[:40], i, L[i][:30], 'canonical?'))
                elif d['x_rule'] == 'superseded_run':
                    a_ = t
                    while a_ > 0 and cands[a_ - 1]:
                        a_ -= 1
                    b_ = t
                    while b_ + 1 < len(cands) and cands[b_ + 1]:
                        b_ += 1
                    if not cands[t] or b_ - a_ + 1 < books.XRUN_MIN:
                        xbad.append((books.book_rel(p)[:40], i, L[i][:30], 'run?'))
            elif d is not None and d['reason'] == 'X' and len(own[t]) < books.SUBST:
                nx_short += 1
                c = chain[t]
                named = None if c is None else '%s#%d' % (
                    fid_path[c >> books._LOC_SHIFT], seq_of[c >> books._LOC_SHIFT][c & ((1 << books._LOC_SHIFT) - 1)])
                if named is None or named != d['duplicate_of']:
                    xbad.append((books.book_rel(p)[:40], i, L[i][:30], named is not None))
        for t, i in enumerate(seq):
            if i in rec_frames[p] or i in tl:
                loc = (fid << books._LOC_SHIFT) | t
                if lex:
                    H_l[hk[t]].append(loc)
                else:
                    K_n[dk[t]].append(loc)
                    H_n[hk[t]].append(loc)
    # 307th: every scene header is in a record or a within-file copy (W1/W2)
    scene = Counter()
    for pf in by_folder('307th'):
        p = pf['source_path']
        for i, t in enumerate(lines_of[p]):
            if t.strip().startswith('سین نمبر'):
                d = dd_row.get((p, i))
                scene['record' if i in rec_frames[p] else d['reason'] if d else 'other'] += 1
    scene_lost = sum(v for k, v in scene.items() if k not in ('record', 'W1', 'W2'))
    ck.add('C17 short X removals are copied passages', not xbad and not scene_lost,
           '%d short-line X rows replayed, %d superseded-canonical X rows, %d without their one-source chain / '
           'canonical %s; 307th scene headers %s' % (nx_short, nx_canon, len(xbad), xbad[:3], dict(scene)))

    # C18 TOC blocks (review: parse_tocs tail): every toc record's block ends
    # on a page number and all its entries lie inside the block
    tbad = []
    for r in recs:
        if r['role'] != 'toc':
            continue
        L = lines_of[r['source_path']]
        a, b = r['toc_block_lines']
        endt = L[b].strip()
        if not (books.is_numeric(endt) or books.RE_ABJAD_FOLIO.match(endt)) or any(
                not (a < e['line_index'] < b) for e in r['toc_entries']) or r['toc_entries'][-1]['page'] is None:
            tbad.append((books.book_rel(r['source_path'])[:40], a, b, endt[:20]))
    ck.add('C18 TOC blocks end at their last page number', not tbad,
           '%d toc records; bad %d %s' % (sum(1 for r in recs if r['role'] == 'toc'), len(tbad), tbad[:3]))

    # C19 titles (review: L8 pointer): every title is a TOC entry of its
    # folder and matches exactly ONE kept body frame of the FOLDER (review
    # 2026-09-25: TOC entries are pooled per folder, so the ambiguity count is
    # too - .B01 twins excluded - and matched by books.toc_key: no era sign,
    # ASCII digits, space-insensitive)
    entries = defaultdict(set)
    for r in recs:
        if r['role'] == 'toc':
            entries[r['book_folder']].update(books.toc_key(e['title']) for e in r['toc_entries'])
    key_count = defaultdict(Counter)
    for p, pf in pfm.items():
        if pf['backup_twin']:
            continue
        for i in (rec_frames[p] - set(pf['label_lines_in_records'])) | set(pf['toc_title_lines']):
            if len(lines_of[p][i]) <= books.TITLE_MAX_LEN:
                key_count[pf['book_folder']][books.toc_key(lines_of[p][i])] += 1
    ibad = []
    for r in recs:
        if not r['title']:
            continue
        p = r['source_path']
        nt = books.toc_key(r['title'])
        n_ = key_count[r['book_folder']][nt]
        if nt not in entries[r['book_folder']] or (n_ != 1 and not pfm[p]['backup_twin']):
            ibad.append((books.book_rel(p)[:40], r['title'][:30], n_))
    t166 = sum(1 for r in recs if r['title'] and r['book_folder'].startswith('166th'))
    ck.add('C19 titles: unique body match of a TOC entry', not ibad and t166 >= 20,
           '%d titled records; bad %d %s; 166th titled records %d' % (
               sum(1 for r in recs if r['title']), len(ibad), ibad[:3], t166))

    # C20 BOOKS_SPEC_CORRECTIONS specifics (review findings)
    probs = []
    lug = [pf for pf in by_folder('Dictionary with matlan adition') if 'lughat' in pf['source_path'].lower()]
    lug_vs98 = sum(d['n_chars'] for pf in lug for d in rows_by_file[pf['source_path']].values()
                   if books.folder_of(d['duplicate_of']).startswith('98th') and not pf['backup_twin'])
    if not lug or lug_vs98:
        probs.append('Lughat chars removed as copies of 98th: %d' % lug_vs98)
    for pf in by_folder('99th') + [x for x in by_folder('Dictionary with matlan adition')
                                   if 'MATTLAN' in x['source_path'] and 'FEHRIST' not in x['source_path']]:
        if pf['file_role'] != 'main':
            probs.append('MATTLAN role %s' % pf['file_role'])
    for pf in by_folder('165th'):
        if 'Old' not in pf['source_path'] and (pf['file_role'] != 'lexicon' or not (
                pf['lexicon_evidence'] or '').startswith('structure:')):
            probs.append('165th glossary role %s' % pf['file_role'])
    for pf in by_folder('329th'):
        if pf['file_role'] != 'main':
            probs.append('329th role %s' % pf['file_role'])
    mg = {r['genre'] for r in recs if 'MATTLAN' in os.path.basename(r['source_path'])}
    if mg != {'proverbs'}:
        probs.append('MATTLAN record genres %s' % sorted(mg, key=str))
    g99 = bm.get([fo for fo in bm if fo.startswith('99th')][0], {}).get('genre') if any(
        fo.startswith('99th') for fo in bm) else None
    if g99 != 'proverbs':
        probs.append('99th genre %s' % g99)
    for pf in by_folder('340th'):
        base = os.path.basename(pf['source_path'])
        if base.startswith('PRELIMINARY') and (pf['superseded_by'] or pf['file_role'] != 'main'):
            probs.append('340th PRELIMINARY not canonical/main')
        if base.startswith('Ruttan') and pf['superseded_label'] not in (None, 'part_of_book'):
            probs.append('340th book file label %s' % pf['superseded_label'])
    lex_gh = [books.book_rel(pf['source_path'])[:40] for pf in out['per_file']
              if pf['file_role'] == 'lexicon' and pf['disposition_lines'].get('genre_heading')]
    if lex_gh:
        probs.append('lexicon files with genre headings %s' % lex_gh[:3])
    udb = sum(1 for r in recs if r['language_variety_v2'] == 'hindko' and 'urdu_dominant_book' in r['content_flags'])
    if udb:
        probs.append('%d hindko records flagged urdu_dominant_book' % udb)
    for d in out['line_dedup']:
        if d['reason'] == 'LABEL_OF_REMOVED_LINE' and d['label_of_line'] not in dd_by_file[d['source_path']]:
            probs.append('label row without removed line')
            break
    # 10th: never a genre from a FILE name (its imprint subject 'قرآن پاک دا
    # منظوم ہندکو ترجمہ' may state it: review 2026-09-25, genre from evidence)
    g10 = [(bm[fo]['genre'], bm[fo]['genre_evidence']) for fo in bm if fo.startswith('10th')]
    if any(g == 'scripture_translation' and not (ev or '').startswith('imprint_') for g, ev in g10):
        probs.append('10th genre from a file name: %s' % g10)
    # imprints: 358th edition bundle; 4th FEHRIST own imprint; GHA label variant; 53rd checksum
    for fo_pre, want in (('358th', {'publication_year': 2018, 'edition': 2}),):
        rs = [r for r in recs if r['book_folder'].startswith(fo_pre) and r['role'] != 'toc']
        got = {(r['publication_year'], r['edition']) for r in rs}
        if got != {(want['publication_year'], want['edition'])}:
            probs.append('%s year/edition %s' % (fo_pre, sorted(got, key=str)))
    r4 = [r for r in recs if r['book_folder'].startswith('4th Farma') and 'FEHRIST' in r['source_path']
          and not r['source_path'].endswith('.B01')]
    if any(r['edition'] is not None or r['first_edition_year'] is not None for r in r4):
        probs.append('4th FEHRIST records carry another file\'s edition')
    for fo_pre in ('120th', '329th', '139th'):
        if not any(r['gha_ref'] for r in recs if r['book_folder'].startswith(fo_pre)):
            probs.append('%s gha_ref missing' % fo_pre)
    if not any('isbn_invalid_checksum' in r['content_flags'] for r in recs if r['book_folder'].startswith('53rd')):
        probs.append('53rd isbn_invalid_checksum missing')
    w1g = [d for d in out['line_dedup'] if d['reason'] == 'W1_GAP']
    for d in w1g:            # coverage by the union of the first-copy window lines it names
        p = d['source_path']
        a = np.unique(books.ngram_hashes(own_key(p, lines_of[p][d['line_index']])))
        b = np.unique(np.concatenate([books.ngram_hashes(own_key(p, lines_of[p][x])) for x in d['copy_lines']]))
        if not len(a) or np.isin(a, b).mean() < books.FUZZ - 1e-9 or max(d['copy_lines']) >= d['line_index']:
            probs.append('W1_GAP row below 0.80')
            break
    ck.add('C20 corrections: order / roles / flags / imprints', not probs,
           '%s; Lughat removed vs 98th %d chars; W1_GAP rows %d (%d chars)' % (
               probs or 'ok', lug_vs98, len(w1g), sum(d['n_chars'] for d in w1g)))

    # C21 R8 words (review: English column of 208th, dialogue, directions)
    lat_all = lat_kept = 0
    for pf in by_folder('208th'):
        p = pf['source_path']
        for i, t in enumerate(lines_of[p]):
            if len(t) >= 2 and not books.AR.search(t) and lang.is_english_word_frame(t, 1):
                lat_all += 1
                lat_kept += i in rec_frames[p] or i in dd_by_file[p]
    ck.add('C21 208th English column kept', lat_all == 0 or lat_kept >= 0.9 * lat_all,
           '%d of %d English word frames in records / line_dedup' % (lat_kept, lat_all))

    c22 = review2_checks(out, lines_of, recs, pfm, bm, rec_frames, dd_row, by_folder, ck)

    fffd = sum(r['unmapped_glyphs'] for r in recs)
    print('  U+FFFD in records: %d; in single-char frames: %d; in residue frames: %d' % (
        fffd, sum(pf['unmapped_in_single_char_frames'] for pf in out['per_file']),
        sum(pf['unmapped_in_residue'] for pf in out['per_file'])))
    return {'kept_chars': kept, 'text_chars': text, 'raw_chars': raw, 'short_records': len(short),
            'doubled_found': got_rel, 'cross_dup_keys': cross[:20], 'c14_explained': dict(expl),
            'c14_unexplained': unexpl[:50]}


def review2_checks(out, lines_of, recs, pfm, bm, rec_frames, dd_row, by_folder, ck: Checks) -> dict:
    """C22: regression checks for the second adversarial review round
    (_books_cache/wf/review2_findings.json), one group per reproduced finding.
    Hard failures are the reviewers' reproduced cases and the invariants the
    fixes establish; the broad measures (stale titles, F uncovered text) are
    reported."""
    from hp import books, lang
    fails = defaultdict(list)
    info = {}
    R = books.book_rel
    non_toc = [r for r in recs if r['role'] != 'toc']
    rec_of = {}
    for r in recs:
        for i in r['frame_indices']:
            rec_of[(r['source_path'], i)] = (r['source_path'], r['passage_index'])

    # (1) BLOCKER: no work inherits another work's imprint (Hindko Lughat got
    # the 99th proverbs book's 'ہندکو مٚتلاں' / ISBN ...083-8 by folder consensus)
    matt_isbn = '9789696870838'
    for r in recs:
        p = r['source_path']
        if 'lughat' in os.path.basename(p).lower() and (
                r['isbn'] == matt_isbn or r['book_title'] == 'ہندکو مٚتلاں'
                or r['publisher'] is not None or r['publication_year'] is not None):
            fails['lughat_imprint'].append((R(p)[-30:], r['passage_index']))
        for f, src in (r.get('imprint_field_sources') or {}).items():
            if src == p or books.parse_folder_name(books.folder_of(p))['series'] is not None:
                continue
            same_cluster = pfm[p]['superseded_by'] == src or pfm[src]['superseded_by'] == p or (
                pfm[p]['superseded_by'] is not None and pfm[p]['superseded_by'] == pfm[src]['superseded_by'])
            if not (books.work_tokens(p) & books.work_tokens(src)) and not same_cluster:
                fails['imprint_from_other_work'].append((R(p)[-30:], f, R(src)[-30:]))
                break
    if not any(r['isbn'] == matt_isbn for r in recs if r['book_folder'].startswith('99th')):
        fails['99th_lacks_its_imprint'].append(True)

    # (2) titles are never a TOC byline cell
    byl = defaultdict(set)
    ttl = defaultdict(set)
    for r in recs:
        if r['role'] == 'toc':
            for e in r['toc_entries']:
                ttl[r['book_folder']].add(books.toc_key(e['title']))
                if e.get('byline'):
                    byl[r['book_folder']].add(books.toc_key(e['byline']))
    for r in recs:
        if r['title'] and books.toc_key(r['title']) in byl[r['book_folder']] - ttl[r['book_folder']]:
            fails['title_is_byline'].append((r['book_folder'][:20], r['passage_index'], r['title'][:30]))
    # (3) stale titles: the 53rd case, and the reviewers' broad measure (a
    # titled record holding another unique TOC heading, loose key) reported
    n_ajmal = sum(1 for r in recs if r['book_folder'].startswith('53rd') and (r['title'] or '').startswith('اجمل ملک'))
    if n_ajmal > 2:
        fails['53rd_stale_title'].append(n_ajmal)
    tk = defaultdict(Counter)
    for r in recs:
        if r['role'] == 'toc':
            for e in r['toc_entries']:
                k = books.loose_key(e['title'])
                if len(k) >= 6:
                    tk[r['book_folder']][k] += 1
    stale = [r for r in non_toc if r['title'] and any(
        len(ln) <= 120 and tk[r['book_folder']].get(books.loose_key(ln)) == 1
        and books.loose_key(ln) != books.loose_key(r['title']) for ln in r['lines'] if ln)]
    info['stale_titles'] = (len(stale), sum(r['n_chars'] for r in stale))

    # (4) credits: the creator the title page / imprint names is the author;
    # a translator / compiler is never labelled author; no folder-name author
    # when a credit is stated
    for r in non_toc:
        if (r['author_source'] or '').startswith('folder_name') and (r['translator'] or r['compiler']):
            fails['folder_author_despite_credit'].append((r['book_folder'][:20], r['passage_index']))
        if r['author_source'] in ('imprint', 'title_page') and 'author' not in (
                books.credit_roles(r['author_role'] or '', weak=True) or []):
            fails['author_from_non_creator_label'].append((r['book_folder'][:20], r['author_role']))
        if 'نمبر شمار' in (r['author'], r['compiler'], r['translator']):
            fails['toc_heading_as_credit'].append(r['book_folder'][:20])
    want = {'19th': ('استاد عبدالرشید تاج', 'compiler', 'نسیم سحرؔ'),
            '206th': ('پطرس بخاری', 'translator', 'علی اویس خیالؔ'),
            '85th': ('احمد علی سائیںؒ', 'compiler', 'محمد ضیاء الدین')}
    for pre, (a, role, who) in want.items():
        got = {(r['author'], r[role]) for r in non_toc if r['book_folder'].split(' ')[0] == pre}
        if got != {(a, who)}:
            fails['credits_%s' % pre].append(sorted(got, key=str)[:3])

    # (5) genre is not a topic keyword
    topic = {'306th': 'drama', '47th': 'sketches', '152nd': 'humour', '111th': 'film_script',
             '26th': 'scripture_translation', '33rd': 'religious_prose', '135th': 'short_stories'}
    for fo, m in bm.items():
        pre = fo.split(' ')[0]
        if topic.get(pre) is not None and m['genre'] == topic[pre]:
            fails['topic_genre'].append((pre, m['genre'], m['genre_evidence']))

    # (6) language from evidence; the collection value is marked a default
    for r in recs:
        src = r['language_source'] or ''
        if src.startswith('collection:'):
            fails['language_source_not_default'].append(r['book_folder'][:20])
            break
        if (r['role'] in ('book_passage', 'front_matter') and r['language_variety_v2'] == 'urdu'
                and r['language'] == 'hindko'):
            fails['urdu_record_labelled_hindko'].append((r['book_folder'][:20], r['passage_index']))
    if {r['language'] for r in recs if r['book_folder'].startswith('340th') and r['role'] != 'english_text'
            and r['language_variety_v2'] != 'urdu'} != {'pahari'}:
        fails['340th_pahari'].append(True)
    for r in recs:
        if r['book_folder'].startswith('196th') and r['language'] == 'hindko' and not any(
                f.startswith('title_page_language(') for f in r['content_flags']):
            fails['196th_pothohari_statement_unflagged'].append(r['passage_index'])

    # (7) imprint windows: a label-fallback window starts at its own first
    # label (no title / author label within 10 frames before it); the rights
    # line typo 'جملہ حقوب' and the Hindko rights line anchor; 'اشاعت سوئم'
    for imp in out['imprints']:
        L = lines_of[imp['source_path']]
        for w in imp['windows']:
            if w['anchor_kind'] != 'label_fallback':
                continue
            a = w['anchor_line']
            for j in range(max(0, a - 10), a):
                t = L[j].strip()
                if len(t) <= 40 and books.imprint_label_field(t, weak=True) in ('title', 'author'):
                    fails['fallback_window_cuts_labels'].append((R(imp['source_path'])[:30], j, t))
    for pre in ('118th', '340th', '358th'):
        srcs = {r['book_title_source'] for r in non_toc if r['book_folder'].startswith(pre)
                and 'Old/' not in r['source_path']}
        if srcs != {'imprint'}:
            fails['imprint_title_%s' % pre].append(sorted(srcs, key=str))
    got255 = {(r['publication_year'], r['edition']) for r in non_toc if r['book_folder'].startswith('255th')}
    if got255 != {(2019, 3)}:
        fails['255th_third_edition'].append(sorted(got255, key=str))

    # (8) R7: the byte-identical 98th '- Copy' twin is co-equal (part_of_book)
    for pf in by_folder('98th'):
        if ' - Copy/' in pf['source_path'] and pf['superseded_label'] != 'part_of_book':
            fails['98th_copy_label'].append(pf['superseded_label'])

    # (9) L7: a split verse chunk never ends on a stanza heading
    prev = None
    for r in recs:
        if (prev is not None and r['split_part'] and prev['split_part'] and prev['form'] == 'verse'
                and prev['source_path'] == r['source_path'] and prev['passage_index'] + 1 == r['passage_index']
                and prev['title'] == r['title'] and prev['lines']
                and books.STANZA_HEAD_RE.search(prev['lines'][-1]) and books.is_verse(r['lines'][0])):
            fails['verse_chunk_ends_on_heading'].append((prev['book_folder'][:20], prev['passage_index']))
        prev = r

    # (10) L2: a lone ':' / '=' between two text frames joins them; lone
    # single-character frames are not stripped at string level
    nj = 0
    lines_by_rec = defaultdict(list)
    for r in recs:
        lines_by_rec[(r['source_path'], r['passage_index'])] = r['lines']
    for pf in out['per_file']:
        p = pf['source_path']
        L = lines_of[p]
        for i in range(1, len(L) - 1):
            ch = L[i].strip()
            if ch not in (':', '=') or len(L[i]) > 2:
                continue
            ra, rb = rec_of.get((p, i - 1)), rec_of.get((p, i + 1))
            if ra is None or ra != rb:
                continue
            nj += 1
            a_, b_ = L[i - 1].strip(), L[i + 1].strip()
            joined = (a_ + ': ' + b_) if ch == ':' else (a_ + ' = ' + b_)
            if not any(joined in ln for ln in lines_by_rec[ra]):
                fails['lone_joiner_not_joined'].append((R(p)[:25], i, a_[-15:], ch, b_[:15]))
    info['lone_joiners_in_records'] = nj

    # (11) B: boilerplate is >= 2 tokens / >= 4 letters and its kept copy is
    # the same text (whitespace-normalised), never a garbage fragment ('ی ا')
    for d in out['line_dedup']:
        if d['reason'] != 'B':
            continue
        t = lines_of[d['source_path']][d['line_index']]
        p2, _, ix = d['duplicate_of'].rpartition('#')
        if len(t.split()) < books.BOILER_MIN_TOKENS or len(books.AR_LETTER.findall(t)) < books.BOILER_MIN_LETTERS:
            fails['b_short_frame'].append(t[:20])
        elif p2 in lines_of and books._wsn(lines_of[p2][int(ix)]) != books._wsn(t):
            fails['b_kept_copy_differs'].append((t[:20], lines_of[p2][int(ix)][:20]))

    # (12) F: only lines whose uncovered remainder is trivial are removed; the
    # reviewers' cases (306th #6905, 46th #7995, 213th #9190) are kept
    unc_total = 0
    for d in out['line_dedup']:
        if d['reason'] != 'F':
            continue
        unc_total += d['uncovered_chars']
        mx = max([b - a for a, b in d['uncovered_key_spans']] or [0])
        klen = len(books.dedup_hkey(lines_of[d['source_path']][d['line_index']])
                   if pfm[d['source_path']]['file_role'] == 'lexicon'
                   else books.dedup_key(lines_of[d['source_path']][d['line_index']]))
        if mx >= books.F_KEEP_RUN or d['uncovered_chars'] > books.F_KEEP_SHARE * klen:
            fails['f_nontrivial_uncovered'].append((R(d['source_path'])[:25], d['line_index'], mx))
    info['f_uncovered_key_chars'] = unc_total
    for pre, ix in (('306th', 6905), ('46th', 7995), ('213th', 9190)):
        if not any(ix in rec_frames[pf['source_path']] for pf in by_folder(pre)):
            fails['f_case_%s' % pre].append(ix)

    # (13) R8: an English frame between two frames of one record is text
    # (glosses, footnotes, abbreviations), not residue; the 22nd index (docx)
    lat3 = re.compile('[A-Za-z].*[A-Za-z].*[A-Za-z]')
    for pf in out['per_file']:
        p = pf['source_path']
        if 'no_extractable_text' in pf['flags']:
            continue
        L = lines_of[p]
        for i in range(1, len(L) - 1):
            t = L[i]
            if not lat3.search(t) or books.AR_LETTER.search(t) or i in rec_frames[p] or (p, i) in dd_row:
                continue
            ra = rec_of.get((p, i - 1))
            if ra is not None and ra == rec_of.get((p, i + 1)):
                cat_ = lang.latin_frame_category(t, 1, True)
                if cat_ not in ('path', 'email_url', 'font_style'):
                    fails['english_between_record_frames'].append((R(p)[:25], i, t[:30]))
    idx22 = [pf for pf in by_folder('22nd') if pf['source_path'].lower().endswith('.docx')]
    if idx22 and not any(r['source_path'] == idx22[0]['source_path'] and 'INDEX' in r['text'] for r in recs):
        fails['22nd_index_docx'].append(True)
    for w in ('B.Com', 'Time', 'Red Cross', 'L.L.B', 'Black Mailing'):
        if not any(w in r['lines'] for r in recs if r['book_folder'].startswith('208th')):
            fails['208th_english_column'].append(w)

    # (14) labels: genre / language words that are headwords, verse words or
    # column heads stay text; a language label does not run on
    if not any(ln.startswith('حمد: اوہ نظم') for r in recs if r['book_folder'].startswith('135th')
               for ln in r['lines']):
        fails['135th_glossary_headword'].append(True)
    for pf in by_folder('112th'):
        if 'NEW FARMA' in pf['source_path']:
            for ix in (23110, 23112):
                if ix < len(lines_of[pf['source_path']]) and lines_of[pf['source_path']][ix].strip() == 'ہندکو' \
                        and ix not in rec_frames[pf['source_path']] and (pf['source_path'], ix) not in dd_row:
                    fails['112th_verse_word'].append(ix)
    r196 = [r for r in recs if r['book_folder'].startswith('196th') and r['role'] != 'toc']
    if r196 and sum(1 for r in r196 if r['language_label'] == 'اُردو ترجمہ') > 0.5 * len(r196):
        fails['196th_sticky_language_label'].append(True)

    bad = {k: v[:3] for k, v in fails.items()}
    ck.add('C22 review-2 regressions (blocker + majors)', not fails,
           '%s; %s' % (('FAILED %s' % bad) if fails else 'ok', info))
    return {'fails': {k: len(v) for k, v in fails.items()}, 'info': info}


def mode_compare(out, files_mode, ck: Checks, d2: dict):
    """C15 (review: .B01 order / TOC dedup): the records of every file that
    both modes process are identical in 'all' (with .B01 twins) and 'default'."""
    d1 = file_digests(out)
    common = sorted(set(d1) & set(d2))
    diff = [p for p in common if d1[p] != d2[p]]
    ck.add('C15 records identical in all / default mode', bool(common) and not diff,
           '%d common files; %d differ %s' % (len(common), len(diff), diff[:3]))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('all', 'default'), default='all')
    ap.add_argument('--decode-cache', default=None,
                    help='use this decode pickle instead of decoding with the current decoder')
    ap.add_argument('--no-determinism', action='store_true')
    ap.add_argument('--no-mode-compare', action='store_true')
    ap.add_argument('--digest-only', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--file-digests', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--dump', default=None, help='pickle the full process_books output here')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    t_start = time.time()
    files = discover_book_files(args.mode)
    quiet = args.digest_only or args.file_digests
    if args.decode_cache:
        with open(args.decode_cache, 'rb') as fh:
            decoded = pickle.load(fh)
        missing = [r for r, _, _ in files if r not in decoded]
        if missing:
            raise SystemExit('decode cache lacks %d files, e.g. %s' % (len(missing), missing[:3]))
        decoder = 'cache:' + os.path.basename(args.decode_cache)
    else:
        decoded = load_decoded(files, log=(lambda m: None) if quiet else print)
        decoder = decoder_fingerprint()
    news = load_newspaper()
    if quiet:
        out, _, _ = run_pipeline(files, decoded, news, log=lambda m: None)
        if args.digest_only:
            print('DIGEST', canonical_digest(out))
        else:
            print('FILE_DIGESTS ' + json.dumps(file_digests(out), ensure_ascii=False))
        return
    print('=' * 100)
    print('BOOKS ACCEPTANCE TEST  mode=%s  files=%d  decoder=%s  newspaper records=%d' % (
        args.mode, len(files), decoder, len(news)))
    print('=' * 100)
    errs = [r for r, _, _ in files if decoded[r]['error']]
    print('decode errors: %d %s' % (len(errs), errs[:5]))
    t_load = time.time() - t_start
    out, secs, cpu = run_pipeline(files, decoded, news,
                                  log=lambda m: print('   [%7.1fs] %s' % (time.time() - t_start, m), flush=True))
    if args.dump:
        with open(args.dump, 'wb') as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    ck = Checks()
    print('\nCHECKS')
    unit_checks(ck)
    extra = run_checks(out, files, decoded, news, secs, cpu, ck)
    digest = canonical_digest(out)
    env = dict(os.environ, PYTHONHASHSEED='4242', PYTHONIOENCODING='utf-8')
    base = [sys.executable, os.path.abspath(__file__)]
    if args.decode_cache:
        base_cache = ['--decode-cache', args.decode_cache]
    else:
        base_cache = []
    # the two extra runs (C11, C15) run concurrently (peak RSS of one
    # process_books run: 1.3 GB)
    procs = {}
    if not args.no_determinism:
        procs['C11'] = subprocess.Popen(base + ['--digest-only', '--mode', args.mode] + base_cache, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        encoding='utf-8')
    if not args.no_mode_compare:
        other = 'default' if args.mode == 'all' else 'all'
        procs['C15'] = subprocess.Popen(base + ['--file-digests', '--mode', other] + base_cache, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        encoding='utf-8')
    res = {k: p.communicate() for k, p in procs.items()}
    if 'C11' in res:
        so, se = res['C11']
        m = re.search(r'DIGEST (\w+)', so or '')
        d2 = m.group(1) if m else None
        ck.add('C11 determinism (2nd run, other hash seed)', d2 == digest,
               'run1 %s run2 %s%s' % (digest[:16], (d2 or 'ERROR')[:16],
                                      '' if d2 else ' ' + (se or '')[-300:]))
    if 'C15' in res:
        so, se = res['C15']
        m = re.search(r'^FILE_DIGESTS (.*)$', so or '', re.M)
        if m:
            mode_compare(out, files, ck, json.loads(m.group(1)))
        else:
            ck.add('C15 records identical in all / default mode', False, 'ERROR ' + (se or '')[-300:])
    total = time.time() - t_start
    print('\nruntime: load/decode %.1fs, process_books %.1fs wall / %.1fs cpu, total %.1fs' % (
        t_load, secs, cpu, total))
    status = 'FAIL' if ck.failed else 'PASS'
    print('\nRESULT: %s  (%d checks, %d failed, %d warnings)' % (
        status, len(ck.rows), len(ck.failed), sum(1 for r in ck.rows if r['status'] == 'WARN')))
    os.makedirs(SCRATCH, exist_ok=True)
    rp = os.path.join(SCRATCH, 'test_report_%s.json' % args.mode)
    with open(rp, 'w', encoding='utf-8') as fh:
        json.dump({'mode': args.mode, 'decoder': decoder, 'status': status, 'checks': ck.rows,
                   'runtime': {'load_s': round(t_load, 1), 'process_wall_s': round(secs, 1),
                               'process_cpu_s': round(cpu, 1), 'total_s': round(total, 1)},
                   'digest': digest, 'report': out['report'], 'extra': extra},
                  fh, ensure_ascii=False, indent=1, default=str)
    print('report -> %s' % rp)
    sys.exit(1 if ck.failed else 0)


if __name__ == '__main__':
    main()
