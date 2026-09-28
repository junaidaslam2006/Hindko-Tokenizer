"""Step 1 of the Hindko morphology silver set: corpus word counts.

Reads the STRICT corpus (read-only) and writes
  data/wordfreq_strict.tsv   one row per word type (harakat stripped)
  data/wordfreq_strict.json  same, plus surface variants, for later steps
  data/urdu_vs_hindko_lines.json  word counts in Urdu-dominant vs
                             Hindko-dominant LINES of the permissive superset
                             (used only to flag Urdu-only words).

Tokenisation = hp.lang: NFC text as released, split on whitespace,
punctuation and digits (hp.lang.TOKEN_SPLIT_RE); the count key is the form
with harakat / honorific signs removed (hp.lang.strip_marks). The most
frequent raw surface spelling is kept for reference.

Dialect groups are assigned per source from the release documentation
(README.md web table; hp/web.py SOURCES dialect notes):
  PESH  newspaper (Weekly Hindkowan, Peshawar), web_gandharahindko, web_tvshia_hn
  BOOK  Gandhara Hindko Academy book series (Peshawar publisher; authors from
        Peshawar and Hazara, so the group is NOT dialect-pure)
  HAZ   omnilingual_asr_hno, common_voice_hno, web_aaprihindko,
        web_hindko_org, web_hindkomaza, web_hazarewall (Hazara / Northern)
  OTHER web_botanix_hnd (Southern Hindko), fineweb2_hindko, web_hindko_blogs
Deterministic: no randomness; output sorted.
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, 'F:/Hindko/_pipeline')
from hp.lang import TOKEN_SPLIT_RE, strip_marks, marker_counts  # noqa: E402

STRICT = 'F:/Hindko/hindko_dataset.jsonl'
PERMISSIVE = 'F:/Hindko/hindko_dataset_permissive.jsonl'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

GROUP = {
    'newspaper': 'PESH', 'web_gandharahindko': 'PESH', 'web_tvshia_hn': 'PESH',
    'book': 'BOOK',
    'omnilingual_asr_hno': 'HAZ', 'common_voice_hno': 'HAZ', 'web_aaprihindko': 'HAZ',
    'web_hindko_org': 'HAZ', 'web_hindkomaza': 'HAZ', 'web_hazarewall': 'HAZ',
    'web_botanix_hnd': 'OTHER', 'fineweb2_hindko': 'OTHER', 'web_hindko_blogs': 'OTHER',
}
GROUPS = ('PESH', 'BOOK', 'HAZ', 'OTHER')


def src_key(r):
    return r['source'] if r['source'] != 'web' else r['site']


def unit_key(r):
    s = r['source']
    if s == 'newspaper':
        return 'issue:' + str(r.get('issue') or r.get('source_path', '').split('/')[0:3])
    if s == 'book':
        return 'book:' + str(r.get('book_folder'))
    return 'web:' + str(r.get('site'))


def raw_tokens(text):
    return [t for t in TOKEN_SPLIT_RE.split(text) if t]


def main():
    os.makedirs(OUT, exist_ok=True)
    total = collections.Counter()
    by_group = collections.defaultdict(collections.Counter)
    n_docs = collections.Counter()
    units = collections.defaultdict(set)
    surface = collections.defaultdict(collections.Counter)
    group_tokens = collections.Counter()
    n_records = 0
    for line in open(STRICT, encoding='utf-8'):
        r = json.loads(line)
        n_records += 1
        g = GROUP.get(src_key(r), 'OTHER')
        u = unit_key(r)
        seen = set()
        for raw in raw_tokens(r['text']):
            w = strip_marks(raw)
            if not w:
                continue
            total[w] += 1
            by_group[g][w] += 1
            group_tokens[g] += 1
            surface[w][raw] += 1
            if w not in seen:
                seen.add(w)
                n_docs[w] += 1
                units[w].add(u)
    rows = []
    for w in sorted(total, key=lambda x: (-total[x], x)):
        rows.append({
            'form': w, 'freq': total[w],
            **{g: by_group[g][w] for g in GROUPS},
            'n_docs': n_docs[w], 'n_units': len(units[w]),
            'top_surface': surface[w].most_common(1)[0][0],
            'surfaces': dict(surface[w].most_common(5)),
        })
    with open(os.path.join(OUT, 'wordfreq_strict.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('form\tfreq\t' + '\t'.join(GROUPS) + '\tn_docs\tn_units\ttop_surface\n')
        for x in rows:
            f.write('\t'.join([x['form'], str(x['freq'])] + [str(x[g]) for g in GROUPS]
                              + [str(x['n_docs']), str(x['n_units']), x['top_surface']]) + '\n')
    meta = {'records': n_records, 'tokens': sum(total.values()), 'types': len(total),
            'group_tokens': dict(group_tokens), 'group_map': GROUP}
    with open(os.path.join(OUT, 'wordfreq_strict.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'rows': rows}, f, ensure_ascii=False)

    # Urdu-vs-Hindko LINE counts on the permissive superset (flagging only)
    ur = collections.Counter(); hi = collections.Counter(); nl = collections.Counter()
    for line in open(PERMISSIVE, encoding='utf-8'):
        r = json.loads(line)
        for ln in r['text'].split('\n'):
            h, u = marker_counts(ln)
            if h + u < 3:
                continue
            s = h / (h + u)
            toks = [strip_marks(t) for t in raw_tokens(ln)]
            toks = [t for t in toks if t]
            if s <= 0.2:
                ur.update(toks); nl['urdu_lines'] += 1
            elif s >= 0.8:
                hi.update(toks); nl['hindko_lines'] += 1
    with open(os.path.join(OUT, 'urdu_vs_hindko_lines.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': {**nl, 'urdu_tokens': sum(ur.values()), 'hindko_tokens': sum(hi.values()),
                            'rule': 'permissive lines with >=3 whole-token markers; urdu: hindko_score<=0.2, '
                                    'hindko: >=0.8 (hp.lang.marker_counts)'},
                   'urdu': dict(ur), 'hindko': dict(hi)}, f, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False))
    print(dict(nl), sum(ur.values()), sum(hi.values()))


if __name__ == '__main__':
    main()
