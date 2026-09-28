#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_load_split.py - checks for load_split.py and the split invariants.
Run: python test_load_split.py   (exit code 0 = all passed)"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import load_split as ls  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(detail)) if detail else ''))
    if not cond:
        fails.append(name)


report = json.load(open(os.path.join(HERE, 'splits_report.json'), encoding='utf-8'))
v = ls.verify()
check('verify(): manifest covers every permissive record', v['records'] == 18283, v['records'])
check('verify(): strict file matches strict records', v['strict_records'] == 12393, v['strict_records'])

# 1. every permissive record is in exactly one split; totals match the report
seen = collections.Counter()
words = collections.Counter()
for sp in ls.SPLITS:
    for uid, text, rec in ls.iter_split(sp):
        seen[uid] += 1
        words[sp] += rec['n_words']
        assert isinstance(text, str) and text == rec['text']
check('each uid yielded once across splits', len(seen) == 18283 and set(seen.values()) == {1})
for sp in ls.SPLITS:
    check('words(%s) == report' % sp, words[sp] == report['shares']['ALL'][sp]['words'],
          (words[sp], report['shares']['ALL'][sp]['words']))

# 2. strict tier: subset, inherits split, counts match the report
st = collections.Counter()
for sp in ls.SPLITS:
    for uid, _, rec in ls.iter_split(sp, tier='strict'):
        st[sp] += 1
        assert rec['quality_tier'] == 'strict'
for sp in ls.SPLITS:
    check('strict records(%s) == report' % sp, st[sp] == report['shares']['ALL'][sp]['strict_records'],
          (st[sp], report['shares']['ALL'][sp]['strict_records']))

# 3. filters
news_test = list(ls.iter_split('test', sources='newspaper'))
check('sources filter', news_test and all(r['source'] == 'newspaper' for _, _, r in news_test), len(news_test))
hk = list(ls.iter_split('val', varieties={'hindko'}))
check('alias val + varieties filter', hk and all(r['language_variety'] == 'hindko' for _, _, r in hk), len(hk))
tr_all = sum(1 for _ in ls.iter_split('train', sources='web'))
tr_drop = list(ls.iter_split('train', sources='web', drop_upstream_heldout=True))
dropped = tr_all - len(tr_drop)
check('drop_upstream_heldout removes only Omnilingual dev/test', dropped > 0 and all(
    not (r.get('site') == 'omnilingual_asr_hno' and r['web_meta']['split'] in ('dev', 'test')) for _, _, r in tr_drop),
    dropped)
try:
    list(ls.iter_split('holdout'))
    check('bad split name raises', False)
except ValueError:
    check('bad split name raises', True)
try:
    list(ls.iter_split('test', sources='radio'))
    check('bad source raises', False)
except ValueError:
    check('bad source raises', True)

# 4. group integrity: a group never has records in both validation and test,
#    and any train record of an eval group is a leak move
m = [json.loads(l) for l in open(ls.MANIFEST_PATH, encoding='utf-8')]
g_splits = collections.defaultdict(set)
for row in m:
    g_splits[row['group']].add(row['split'])
check('no group spans validation and test', not any({'validation', 'test'} <= s for s in g_splits.values()))
bad = [row for row in m if row['split'] == 'train' and len(g_splits[row['group']] - {'train'})
       and row['assignment'] != 'leak_moved_to_train']
check('train records of eval groups are all leak moves', not bad, len(bad))
check('assignment values', {row['assignment'] for row in m} <= {'group', 'leak_moved_to_train'})

# 5. report invariants
check('tolerance met', report['tolerance_met'])
for sp in ('validation', 'test'):
    check('zero leaks after resolution (%s)' % sp, report['leakage']['after_resolution'][sp]['leaked_records'] == 0)
    for s in ('newspaper', 'book', 'web'):
        share = report['shares'][s][sp]['share_pct']
        check('%s %s share within 5 +/- 1.5' % (s, sp), abs(share - 5) <= 1.5, share)
fd = report['leakage']['final_distribution']
check('max containment < 0.30', fd['containment_in_train']['max'] < 0.30, fd['containment_in_train']['max'])
check('max Jaccard < 0.50', fd['best_exact_jaccard_in_train']['max'] < 0.50, fd['best_exact_jaccard_in_train']['max'])
for k in ('crosscheck_deduper_normalisation', 'crosscheck_raw_whitespace_tokens',
          'crosscheck_ngram_candidates_exact_jaccard', 'worst_case_vs_entire_corpus_outside_own_group'):
    check('%s: 0 over threshold' % k, fd[k]['over_threshold'] == 0)
for k, x in report['leakage']['validation_vs_test'].items():
    check('%s: 0 over threshold' % k, x['over_threshold'] == 0)

print('\n%d failed' % len(fails))
sys.exit(1 if fails else 0)
