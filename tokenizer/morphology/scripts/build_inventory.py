"""Step 3: the Hindko inflectional-morphology inventory with corpus evidence.

Joins the curated inventory (morph_inventory.py: spellings, glosses, sources,
dialect statements from the literature) with corpus evidence from the strict
release:
  * bound suffixes: every analysed candidate word (data/candidates_all.tsv,
    confidence high or low; dropped words excluded) whose winning parse ends
    in that suffix -> type/token counts, counts per dialect group, examples;
  * free items (postpositions, pronouns, auxiliaries): word counts per group;
  * the spelling of retroflex n.
Writes inventory/hindko_inflection_inventory.tsv, .json and inventory_table.md.

Corpus dialect label (distributional, NOT a grammar claim; register differs
between groups: the Hazara group is mostly transcribed speech):
  Hazara-enriched  : HAZ >= 5 and rate_HAZ >= 3 x rate_PESH
  absent in Hazara : expected HAZ >= 5 (at the corpus-wide rate) but HAZ <= expected/5, and PESH >= 10
  shared           : HAZ >= 3 and PESH >= 3
  insufficient     : otherwise
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from morph_inventory import (VERB_SUFFIXES, CAUS_FORMATIVES, NOUN_SUFFIXES, AGR_SUFFIXES,  # noqa: E402
                             FREE_ITEMS, BOUND_ITEMS)

GROUPS = ('PESH', 'BOOK', 'HAZ', 'OTHER')
# what a SOURCE says about an individual suffix (corpus evidence goes in its own column)
SUFFIX_LIT = {
    ('VERB', 'دا'): ('shared', 'RRS2011 paR-daa, tur-Daa (Muzaffarabad); BANO کردا وے (Peshawar)'),
    ('VERB', 'دی'): ('shared', 'RRS2011 likh-Dii (3b); BANO کردی اے'),
    ('VERB', 'دے'): ('shared', 'RRS2011 likh-De (4a); BANO کردے'),
    ('VERB', 'ندا'): ('shared', 'BANO ہوندا وے (Peshawar)'),
    ('VERB', 'یا'): ('shared', 'RRS2011 paR-yaa, tur-yaa, lang-yaa (Muzaffarabad)'),
    ('VERB', 'ا'): ('shared (Urdu-like)', 'BANO پڑھا ایا (Peshawar)'),
    ('VERB', 'سی'): ('shared', 'SH2010 (-s- future of Lahnda); RRS2011 paR-sii; BANO جاسی، ہوسی'),
    ('VERB', 'ساں'): ('Peshawari attested', 'BANO دیساں (1SG), دیساں = دونگی'),
    ('VERB', 'سیں'): ('Peshawari attested', 'BANO آسیں (2SG), کرسیں'),
    ('VERB', 'سن'): ('Peshawari attested', 'BANO جاسُن (3PL); written apart: پڑھ سُن'),
    ('VERB', 'سو'): ('-', 'CORPUS only'),
    ('VERB', 'نا'): ('shared (also Urdu INF)', 'BANO دینا واں, کرنا واں = PRS.1SG'),
    ('VERB', 'نڑیں'): ('Peshawari attested', 'BANO ہونیط ولی, دینیط دا (decode writes the n-with-small-tah glyph as ن+ط)'),
    ('VERB', 'نڑاں'): ('-', 'CORPUS only'),
    ('VERB', 'نڑا'): ('-', 'CORPUS only'),
    ('VERB', 'نڑے'): ('-', 'CORPUS only'),
    ('VERB', 'نی'): ('Peshawari attested', 'BANO پریکٹس کرنی وے (obligation, F)'),
    ('VERB', 'وے'): ('Peshawari attested', 'BANO کردیوے، ہووے'),
    ('VERB', 'کے'): ('shared', 'BANO جاکے فاتحہ'),
    ('NOUN_C', 'اں'): ('shared', 'RRS2011 chowkan, zatan, gallan; SH1980 obl.pl -a~ (Kohat); BANO شاگرداں، کتاباں'),
    ('NOUN_A', 'ے'): ('shared', 'SH1980 pUttUr -> obl pUtre (Kohat)'),
    ('NOUN_A', 'یاں'): ('-', 'CORPUS; BANO writes the ending apart after -e: بچے اں، جملے آں'),
    ('AGR', 'ا'): ('shared', 'BANO میرا، ساڈا، سُواڈا'),
}
SLOT_TO_ITEM = {'IPFV': 'V.IPFV', 'PFV': 'V.PFV', 'FUT': 'V.FUT', 'INF': 'V.INF', 'SBJV': 'V.SBJV',
                'CONJ': 'V.CONJ'}


def corpus_dialect(c, meta):
    gt = meta['group_tokens']
    P, H = c['PESH'], c['HAZ']
    tot = sum(c[g] for g in GROUPS)
    rp, rh = P / gt['PESH'], H / gt['HAZ']
    exp_h = tot * gt['HAZ'] / meta['tokens']
    if H >= 5 and rh >= 3 * rp:
        return 'Hazara-enriched'
    if exp_h >= 5 and H <= exp_h / 5 and P >= 10:
        return f'absent/rare in Hazara sources (HAZ {H}, expected {exp_h:.0f})'
    if H >= 3 and P >= 3:
        return 'shared'
    return 'insufficient data'


def rates(c, meta):
    gt = meta['group_tokens']
    return ' '.join(f'{g}:{c[g]} ({1e4 * c[g] / gt[g]:.1f}/10k)' for g in ('PESH', 'BOOK', 'HAZ'))


def main():
    d = json.load(open(os.path.join(ROOT, 'data', 'wordfreq_strict.json'), encoding='utf-8'))
    meta = d['meta']
    F = {r['form']: r for r in d['rows']}
    cands = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'candidates_all.tsv'), encoding='utf-8'),
                                delimiter='\t'))
    cands = [r for r in cands if r['confidence'] in ('high', 'low')]
    by_suffix = collections.defaultdict(list)
    for r in cands:
        sufs = r['suffixes'].split('+')
        fam = r['family']
        key = (('VERB' if fam in ('VERB', 'CAUS') else fam), sufs[-1])
        by_suffix[key].append(r)
        if fam == 'CAUS':
            by_suffix[('CAUS', sufs[0])].append(r)
    bound_meta = {b['id']: b for b in BOUND_ITEMS}
    rows = []

    def add_bound(row_id, section, family_key, suffix, gloss, attaches, item_id, extra_note=''):
        lst = by_suffix.get((family_key, suffix), [])
        c = {g: sum(int(r[g]) for r in lst if g in r) for g in ('PESH', 'BOOK', 'HAZ')}
        c['OTHER'] = sum(int(r['freq']) - int(r['PESH']) - int(r['BOOK']) - int(r['HAZ']) for r in lst)
        ex = sorted(lst, key=lambda r: (-int(r['freq']), r['word']))[:6]
        b = bound_meta.get(item_id, {})
        lit = SUFFIX_LIT.get((family_key, suffix), (b.get('dialect_lit', ''), b.get('sources', '')))
        rows.append(dict(
            id=row_id, section=section, form='-' + suffix, gloss=gloss, attaches_to=attaches,
            dialect_literature=lit[0], sources=lit[1],
            n_types=len(lst), n_tokens=sum(int(r['freq']) for r in lst),
            corpus_counts=rates(c, meta) if lst else '',
            dialect_corpus=corpus_dialect(c, meta) if lst else 'no analysed words',
            examples=' '.join(f"{r['segmentation']}:{r['freq']}" for r in ex), note=extra_note))

    for s in VERB_SUFFIXES:
        att = {'C': 'consonant-final verb stem', 'V': 'vowel-final verb stem', 'any': 'verb stem'}[s['stem']]
        note = '' if s['target'] else 'evidence only (not a gold target)'
        add_bound(f"V.{s['slot']}.{s['suffix']}", 'verb', 'VERB', s['suffix'], s['gloss'], att,
                  SLOT_TO_ITEM[s['slot']], note)
    for c in CAUS_FORMATIVES:
        add_bound(f"V.CAUS.{c['suffix']}", 'verb', 'CAUS', c['suffix'], c['gloss'] + ' formative (+ vowel-stem endings)',
                  'consonant-final verb stem', 'V.CAUS')
    for n in NOUN_SUFFIXES:
        fams = {'اں': ['NOUN_C', 'NOUN_I'], 'ے': ['NOUN_A'], 'یاں': ['NOUN_A']}[n['suffix']]
        for fam in fams:
            item = {'NOUN_C': 'N.PL', 'NOUN_I': 'N.F.PL', 'NOUN_A': 'N.OBL' if n['suffix'] == 'ے' else 'N.OBL.PL'}[fam]
            gl = n['gloss'] if fam != 'NOUN_I' else 'PL of F -ii nouns (کڑی -> کڑیاں)'
            att = {'NOUN_C': 'noun (consonant-final)', 'NOUN_I': 'F noun in -ی', 'NOUN_A': 'M noun in -ا / -ہ'}[fam]
            add_bound(f"N.{fam}.{n['suffix']}", 'noun', fam, n['suffix'], gl, att, item)
    for a in AGR_SUFFIXES:
        add_bound(f"AGR.{a['suffix']}", 'adjective/possessive', 'AGR', a['suffix'], a['gloss'],
                  '-aa adjective, possessive pronoun, suppletive perfective stem', 'ADJ.AGR')
    for suf in ('دا', 'دی', 'دے', 'دیاں'):
        add_bound(f"PRON.GEN.{suf}", 'pronoun + clitic', 'PRON_GEN', suf, 'GEN written joined to a pronoun (اسدا)',
                  'اس / جس / کس', 'P.GEN')
    for b in BOUND_ITEMS:
        if b['id'] in ('V.COP.CLITIC', 'ORTH.RETRO_N'):
            words = {'V.COP.CLITIC': ['کردین', 'سکدین', 'گئین', 'آگئین', 'ہوگئین', 'کہندین', 'پئین'],
                     'ORTH.RETRO_N': []}[b['id']]
            c = {g: sum(F.get(w, {}).get(g, 0) for w in words) for g in GROUPS}
            rows.append(dict(id=b['id'], section=b['section'], form=b['suffix'], gloss=b['gloss'], attaches_to='',
                             dialect_literature=b['dialect_lit'], sources=b['sources'],
                             n_types=sum(1 for w in words if w in F), n_tokens=sum(c.values()),
                             corpus_counts=rates(c, meta) if words else '',
                             dialect_corpus=corpus_dialect(c, meta) if words else 'see note',
                             examples=' '.join(f'{w}:{F[w]["freq"]}' for w in words if w in F),
                             note=b.get('not_in_gold', '')))
    for it in FREE_ITEMS:
        for w in it['words']:
            r = F.get(w)
            c = {g: (r[g] if r else 0) for g in GROUPS}
            rows.append(dict(id=it['id'], section=it['section'], form=w, gloss=it['gloss'],
                             attaches_to='free word', dialect_literature=it['dialect_lit'], sources=it['sources'],
                             n_types=1 if r else 0, n_tokens=r['freq'] if r else 0,
                             corpus_counts=rates(c, meta), dialect_corpus=corpus_dialect(c, meta),
                             examples='', note='not segmented (closed class)'))
    # retroflex n spelling (character counts over the strict corpus)
    ortho = collections.Counter()
    for line in open('F:/Hindko/hindko_dataset.jsonl', encoding='utf-8'):
        t = json.loads(line)['text']
        ortho['ن+ڑ'] += t.count('نڑ')
        ortho['U+0768'] += t.count(chr(0x0768))
        ortho['U+06BB'] += t.count(chr(0x06BB))
        ortho['ن+ط'] += t.count('نط')
    for x in rows:
        if x['id'] == 'ORTH.RETRO_N':
            x['examples'] = ('strict corpus character counts: ' + ', '.join(f'{k} {v}' for k, v in ortho.items())
                             + '; the book-decode artefact ن+ط comes from an InPage glyph for ن with small ط')
    os.makedirs(os.path.join(ROOT, 'inventory'), exist_ok=True)
    cols = ['id', 'section', 'form', 'gloss', 'attaches_to', 'dialect_literature', 'dialect_corpus', 'sources',
            'n_types', 'n_tokens', 'corpus_counts', 'examples', 'note']
    with open(os.path.join(ROOT, 'inventory', 'hindko_inflection_inventory.tsv'), 'w', encoding='utf-8',
              newline='\n') as f:
        f.write('\t'.join(cols) + '\n')
        for x in rows:
            f.write('\t'.join(str(x[c]).replace('\t', ' ') for c in cols) + '\n')
    with open(os.path.join(ROOT, 'inventory', 'hindko_inflection_inventory.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': {'corpus': meta, 'retroflex_n_counts': dict(ortho)}, 'rows': rows}, f,
                  ensure_ascii=False, indent=1)
    # compact markdown table (bound morphology only; free words summarised per item)
    md = ['| id | form | gloss | dialect (literature) | dialect (corpus) | types / tokens | corpus examples |',
          '|---|---|---|---|---|---|---|']
    for x in rows:
        if x['section'] in ('verb', 'noun', 'adjective/possessive', 'pronoun + clitic', 'orthography'):
            ex = x['examples'] if x['id'] == 'ORTH.RETRO_N' else ' '.join(x['examples'].split(' ')[:5])
            cells = [x['id'], x['form'], x['gloss'], (x['dialect_literature'] or '-') + ' [' + x['sources'] + ']',
                     x['dialect_corpus'], f"{x['n_types']} / {x['n_tokens']}", ex]
            md.append('| ' + ' | '.join(str(c).replace('|', '\|') for c in cells) + ' |')
    md.append('')
    md.append('| item | words (strict-corpus count, PESH / BOOK / HAZ) | gloss | dialect (literature) | dialect (corpus, per word) |')
    md.append('|---|---|---|---|---|')
    for it in FREE_ITEMS:
        ws = []
        dl = []
        for w in it['words']:
            r = F.get(w)
            if r:
                ws.append(f"{w} {r['freq']} ({r['PESH']}/{r['BOOK']}/{r['HAZ']})")
                dl.append(f"{w}: {corpus_dialect(r, meta).split(' (')[0]}")
            else:
                ws.append(f'{w} 0')
        cells = [it['id'], '; '.join(ws), it['gloss'], it['dialect_lit'] + ' [' + it['sources'] + ']', '; '.join(dl)]
        md.append('| ' + ' | '.join(str(c).replace('|', '\|') for c in cells) + ' |')
    open(os.path.join(ROOT, 'inventory', 'inventory_table.md'), 'w', encoding='utf-8', newline='\n').write(
        '\n'.join(md) + '\n')
    print(len(rows), 'inventory rows;', dict(ortho))


if __name__ == '__main__':
    main()
