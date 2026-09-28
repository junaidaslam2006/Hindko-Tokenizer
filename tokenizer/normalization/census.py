"""Character census of the released Hindko corpus, per source.

Reads (read-only) F:/Hindko/hindko_dataset_permissive.jsonl and
F:/Hindko/hindko_dataset.jsonl. Writes census/census.json and
census/census_out.txt under this folder.

For every codepoint: occurrences and number of records containing it, per
source (newspaper / book / web) and for the strict subset. Also: NFC
stability of each record, focus groups named in the task, word contexts for
the ambiguous letters, digit runs, retroflex-nasal spellings.

Deterministic: sorted iteration everywhere; no randomness.
Run: PYTHONIOENCODING=utf-8 python census.py
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

ROOT = r'F:\Hindko'
PERM = os.path.join(ROOT, 'hindko_dataset_permissive.jsonl')
STRICT = os.path.join(ROOT, 'hindko_dataset.jsonl')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'census')
os.makedirs(OUT, exist_ok=True)

SOURCES = ('newspaper', 'book', 'web')

# Word = maximal run of Arabic-script letters/marks (+ZWNJ, tone letters).
AR = '\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u200C'
WORD_RE = re.compile('[' + AR + ']+')
LATIN_WORD_RE = re.compile('[A-Za-z]+')

# Focus codepoints (task list). Built with chr() so the file bytes are safe.
FOCUS = {
    'yeh': [0x064A, 0x0649, 0x06CC, 0x06D2, 0x06D3, 0x0626, 0x06D0],
    'kaf': [0x0643, 0x06A9, 0x06AA, 0x06AF],
    'heh': [0x0647, 0x06C1, 0x06BE, 0x06C3, 0x0629, 0x06D5, 0x06C0, 0x06C2, 0x06FF],
    'hamza': [0x0621, 0x0626, 0x06C0, 0x06C2, 0x0654, 0x0655, 0x0674, 0x0623, 0x0624, 0x0625, 0x0622, 0x0671,
              0x0672, 0x0673, 0x0675, 0x0676, 0x0677, 0x0678],
    'alef': [0x0627, 0x0622, 0x0623, 0x0625, 0x0671, 0x0670, 0x0653],
    'digits_ascii': list(range(0x30, 0x3A)),
    'digits_ext_arabic_indic': list(range(0x06F0, 0x06FA)),
    'digits_arabic_indic': list(range(0x0660, 0x066A)),
    'invisible': [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
                  0x2066, 0x2067, 0x2068, 0x2069, 0x061C, 0xFEFF, 0x00AD, 0x2060, 0x034F],
    'spaces': [0x0020, 0x00A0, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
               0x2009, 0x200A, 0x202F, 0x205F, 0x3000, 0x0009, 0x000A, 0x000D, 0x000B, 0x000C, 0x0085,
               0x2028, 0x2029],
    'kashida': [0x0640],
    'tone_letters': [0x08BE, 0x08BF, 0x08C0, 0x08C1, 0x08C2, 0x065A],
    'retroflex_nasal': [0x0768, 0x06BB, 0x06B9, 0x06BC, 0x0769, 0x06BA, 0x0646],
    'urdu_punct': [0x06D4, 0x060C, 0x061B, 0x061F, 0x066A, 0x066B, 0x066C, 0x066D, 0x2026, 0x060D, 0x0600,
                   0x0601, 0x0602, 0x0603, 0x060E, 0x060F],
    'ascii_punct': [ord(c) for c in '.,;:?!%\'"()[]-/*+=#&@$'],
    'quotes': [0x2018, 0x2019, 0x201C, 0x201D, 0x00AB, 0x00BB, 0x2039, 0x203A, 0x201A, 0x201E, 0x0022,
               0x0027, 0x060E],
    'dashes': [0x002D, 0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0x06DD, 0x06DE],
}


def cat_group(cp):
    if 0xFB50 <= cp <= 0xFDFF:
        return 'presentation_A'
    if 0xFE70 <= cp <= 0xFEFF:
        return 'presentation_B'
    return None


def name(cp):
    return unicodedata.name(chr(cp), 'U+%04X' % cp)


def ctx(text, i, w=18):
    s = text[max(0, i - w):i + w + 1].replace('\n', ' / ')
    return s


def main():
    occ = {s: Counter() for s in SOURCES}          # source -> cp -> occurrences
    recs = {s: Counter() for s in SOURCES}         # source -> cp -> records containing
    strict_occ = Counter()
    strict_recs = Counter()
    nrec = Counter()
    nchar = Counter()
    not_nfc = Counter()
    not_nfc_examples = defaultdict(list)
    nfc_delta = Counter()                          # chars changed by NFC per source
    # word contexts for ambiguous letters
    focus_words = defaultdict(lambda: defaultdict(Counter))   # cp -> source -> word -> n
    examples = defaultdict(list)                   # cp -> [(source, uid, ctx)] first 12 by file order
    digit_runs = {s: Counter() for s in SOURCES}   # run script class -> count
    mixed_digit_runs = defaultdict(list)
    retro = {s: Counter() for s in SOURCES}
    retro_words = defaultdict(Counter)             # form -> word -> n
    web_sites = defaultdict(Counter)               # cp -> site -> occ (for web only)
    book_folders = defaultdict(Counter)            # cp -> book_folder -> occ
    news_issues = defaultdict(Counter)
    uid_tier = {}

    interesting = set()
    for v in FOCUS.values():
        interesting.update(v)
    word_ctx_cps = {0x0647, 0x0643, 0x064A, 0x0649, 0x0629, 0x06C3, 0x06C0, 0x06C2, 0x0654, 0x0626,
                    0x0623, 0x0624, 0x0625, 0x0671, 0x0768, 0x06BB, 0x0640, 0x200C, 0x200D, 0x06D5,
                    0x065A, 0x08BE, 0x08BF, 0x08C0, 0x08C1, 0x08C2, 0x0655, 0x0674, 0x06FF, 0x06D0}

    with open(PERM, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            s = r['source']
            t = r['text']
            uid = r['uid']
            uid_tier[uid] = r['quality_tier']
            nrec[s] += 1
            nchar[s] += len(t)
            if not unicodedata.is_normalized('NFC', t):
                not_nfc[s] += 1
                if len(not_nfc_examples[s]) < 10:
                    n = unicodedata.normalize('NFC', t)
                    # locate first difference
                    i = next((k for k in range(min(len(t), len(n))) if t[k] != n[k]), min(len(t), len(n)))
                    not_nfc_examples[s].append({'uid': uid, 'raw': [('U+%04X' % ord(c)) for c in t[i:i + 4]],
                                                'nfc': [('U+%04X' % ord(c)) for c in n[i:i + 4]]})
                nfc_delta[s] += abs(len(t) - len(unicodedata.normalize('NFC', t)))
            c = Counter(map(ord, t))
            occ[s].update(c)
            recs[s].update(c.keys())
            if r['quality_tier'] == 'strict':
                tn = unicodedata.normalize('NFC', t)
                cn = Counter(map(ord, tn))
                strict_occ.update(cn)
                strict_recs.update(cn.keys())
            hits = [cp for cp in c if cp in word_ctx_cps or cat_group(cp) or cp >= 0x10000
                    or (cp > 0x7F and cp not in interesting and not (0x0600 <= cp <= 0x06FF))]
            if hits:
                for m in WORD_RE.finditer(t):
                    w = m.group()
                    for ch in set(w):
                        cp = ord(ch)
                        if cp in word_ctx_cps or cat_group(cp):
                            focus_words[cp][s][w] += 1
                for i, ch in enumerate(t):
                    cp = ord(ch)
                    if cp in hits and len(examples[cp]) < 12:
                        examples[cp].append((s, uid, ctx(t, i)))
                for cp in hits:
                    if s == 'web':
                        web_sites[cp][r.get('site') or '?'] += c[cp]
                    elif s == 'book':
                        book_folders[cp][r.get('book_folder') or '?'] += c[cp]
                    else:
                        news_issues[cp][str(r.get('issue'))] += c[cp]
            # digit runs
            for m in re.finditer('[0-9\u06F0-\u06F9\u0660-\u0669]+', t):
                run = m.group()
                kinds = set()
                for ch in run:
                    o = ord(ch)
                    kinds.add('ascii' if o < 0x80 else ('ext' if o >= 0x06F0 else 'arabic_indic'))
                key = '+'.join(sorted(kinds))
                digit_runs[s][key] += 1
                if len(kinds) > 1 and len(mixed_digit_runs[s]) < 10:
                    mixed_digit_runs[s].append((uid, ctx(t, m.start())))
            # retroflex nasal spellings
            retro[s]['U+0768'] += t.count('\u0768')
            retro[s]['U+06BB'] += t.count('\u06BB')
            retro[s]['U+06BC'] += t.count('\u06BC')
            retro[s]['noon+reh_retroflex'] += t.count('\u0646\u0691')
            retro[s]['noon+sukun'] += t.count('\u0646\u0652')
            for m in WORD_RE.finditer(t):
                w = m.group()
                if '\u0768' in w:
                    retro_words['U+0768'][w] += 1
                if '\u0646\u0691' in w:
                    retro_words['noon+reh'][w] += 1
                if '\u0646\u0652' in w:
                    retro_words['noon+sukun'][w] += 1
                if '\u06BB' in w:
                    retro_words['U+06BB'][w] += 1

    # characters present in permissive (any source) but never in NFC strict
    allcps = set()
    for s in SOURCES:
        allcps.update(occ[s])
    unseen = sorted(cp for cp in allcps if cp not in strict_occ)

    def row(cp):
        d = {'cp': 'U+%04X' % cp, 'char': chr(cp) if unicodedata.category(chr(cp))[0] not in 'CZ' else '',
             'name': name(cp), 'cat': unicodedata.category(chr(cp))}
        for s in SOURCES:
            d[s] = [occ[s][cp], recs[s][cp]]
        d['strict_nfc'] = [strict_occ[cp], strict_recs[cp]]
        return d

    result = {
        'inputs': {'permissive': PERM, 'strict': STRICT},
        'records': dict(nrec), 'chars': dict(nchar),
        'not_nfc_records': dict(not_nfc), 'not_nfc_examples': not_nfc_examples,
        'nfc_length_delta': dict(nfc_delta),
        'all_codepoints': [row(cp) for cp in sorted(allcps)],
        'focus': {g: [row(cp) for cp in cps] for g, cps in FOCUS.items()},
        'presentation_forms': [row(cp) for cp in sorted(allcps) if cat_group(cp)],
        'unseen_in_strict_nfc': [row(cp) for cp in unseen],
        'focus_words_top': {('U+%04X' % cp): {s: focus_words[cp][s].most_common(40) for s in SOURCES}
                            for cp in sorted(focus_words)},
        'focus_words_types': {('U+%04X' % cp): {s: len(focus_words[cp][s]) for s in SOURCES}
                              for cp in sorted(focus_words)},
        'examples': {('U+%04X' % cp): v for cp, v in sorted(examples.items())},
        'web_sites': {('U+%04X' % cp): v.most_common(15) for cp, v in sorted(web_sites.items())},
        'book_folders': {('U+%04X' % cp): v.most_common(15) for cp, v in sorted(book_folders.items())},
        'news_issues': {('U+%04X' % cp): v.most_common(10) for cp, v in sorted(news_issues.items())},
        'digit_runs': {s: dict(v) for s, v in digit_runs.items()},
        'mixed_digit_runs': mixed_digit_runs,
        'retroflex_nasal': {s: dict(v) for s, v in retro.items()},
        'retroflex_words_top': {k: v.most_common(60) for k, v in retro_words.items()},
        'retroflex_word_types': {k: len(v) for k, v in retro_words.items()},
    }
    with open(os.path.join(OUT, 'census.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # readable summary
    L = []
    L.append('records %s' % dict(nrec))
    L.append('chars   %s' % dict(nchar))
    L.append('not-NFC records %s ; length delta %s' % (dict(not_nfc), dict(nfc_delta)))
    L.append('')
    hdr = '%-8s %-3s %-44s %-3s %16s %16s %16s %16s' % ('cp', 'ch', 'name', 'cat', 'newspaper', 'book', 'web',
                                                        'strict_nfc')
    for g, cps in FOCUS.items():
        L.append('== %s' % g)
        L.append(hdr)
        for cp in cps:
            d = row(cp)
            if sum(d[s][0] for s in SOURCES) == 0 and d['strict_nfc'][0] == 0:
                continue
            L.append('%-8s %-3s %-44s %-3s %16s %16s %16s %16s' % (
                d['cp'], d['char'], d['name'][:44], d['cat'],
                '%d/%d' % tuple(d['newspaper']), '%d/%d' % tuple(d['book']), '%d/%d' % tuple(d['web']),
                '%d/%d' % tuple(d['strict_nfc'])))
        L.append('')
    L.append('== presentation forms (occ/records)')
    for d in result['presentation_forms']:
        L.append('%-8s %-3s %-44s %16s %16s %16s %16s' % (d['cp'], d['char'], d['name'][:44],
                 '%d/%d' % tuple(d['newspaper']), '%d/%d' % tuple(d['book']), '%d/%d' % tuple(d['web']),
                 '%d/%d' % tuple(d['strict_nfc'])))
    L.append('')
    L.append('== codepoints in permissive never seen in NFC strict (%d)' % len(unseen))
    for d in result['unseen_in_strict_nfc']:
        L.append('%-8s %-3s %-44s %-3s %16s %16s %16s' % (d['cp'], d['char'], d['name'][:44], d['cat'],
                 '%d/%d' % tuple(d['newspaper']), '%d/%d' % tuple(d['book']), '%d/%d' % tuple(d['web'])))
    L.append('')
    L.append('== all non-ASCII codepoints outside U+0600-06FF (occ/records)')
    for cp in sorted(allcps):
        if cp < 0x80 or 0x0600 <= cp <= 0x06FF:
            continue
        d = row(cp)
        L.append('%-8s %-3s %-44s %-3s %16s %16s %16s %16s' % (d['cp'], d['char'], d['name'][:44], d['cat'],
                 '%d/%d' % tuple(d['newspaper']), '%d/%d' % tuple(d['book']), '%d/%d' % tuple(d['web']),
                 '%d/%d' % tuple(d['strict_nfc'])))
    L.append('')
    L.append('== digit runs %s' % json.dumps(result['digit_runs']))
    L.append('== mixed digit runs %s' % json.dumps(mixed_digit_runs, ensure_ascii=False))
    L.append('== retroflex nasal %s' % json.dumps(result['retroflex_nasal']))
    L.append('== retroflex word types %s' % json.dumps(result['retroflex_word_types']))
    with open(os.path.join(OUT, 'census_out.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L[:400]))


if __name__ == '__main__':
    main()
