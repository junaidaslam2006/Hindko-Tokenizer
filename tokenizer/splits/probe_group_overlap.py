#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_group_overlap.py - measure how much text each pair of groups (same
source) shares, to decide whether some groups are really ONE edition / ONE
book stored twice. Diagnostic run on the RULE groups (before the near-duplicate merge it motivated). Read-only diagnostic; prints a summary and writes
probe_group_overlap_out.json next to this file."""
import collections
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_splits as ms  # noqa: E402

cfg = ms.CONFIG
recs, _ = ms.load_records()
keys, audit = ms.assign_groups(recs, cfg)
gnames = sorted(set(keys))
gi = {g: i for i, g in enumerate(gnames)}
gid = np.array([gi[k] for k in keys], np.int32)
src_of = {k: r['source'] for k, r in zip(keys, recs)}
tok_ids, grams, shingles, short, _ = ms.build_features(recs, cfg)
H, R, OWN, _ = ms.gram_occurrences(tok_ids, grams, short)
H, R = H[OWN], R[OWN]   # own 8-grams only
G = gid[R]

# distinct (gram, group) pairs
order = np.lexsort((G, H))
Hs, Gs = H[order], G[order]
keep = np.ones(len(Hs), bool)
keep[1:] = (Hs[1:] != Hs[:-1]) | (Gs[1:] != Gs[:-1])
Hs, Gs = Hs[keep], Gs[keep]
grams_per_group = np.bincount(Gs, minlength=len(gnames))
new_h = np.ones(len(Hs), bool)
new_h[1:] = Hs[1:] != Hs[:-1]
starts = np.nonzero(new_h)[0]
ends = np.append(starts[1:], len(Hs))
ng = ends - starts
print('grams by #groups:', dict(sorted(collections.Counter(np.minimum(ng, 10).tolist()).items())))

S = collections.Counter()
MAXG = 6          # ignore formulaic grams shared by > 6 groups
for s, e in zip(starts[ng >= 2].tolist(), ends[ng >= 2].tolist()):
    if e - s > MAXG:
        continue
    gs = Gs[s:e].tolist()
    for a, b in itertools.combinations(gs, 2):
        S[(a, b)] += 1

rows = []
for (a, b), c in S.items():
    ga, gb = gnames[a], gnames[b]
    ov = c / max(1, min(grams_per_group[a], grams_per_group[b]))
    rows.append(dict(a=ga, b=gb, shared=c, grams_a=int(grams_per_group[a]),
                     grams_b=int(grams_per_group[b]), overlap=round(ov, 4),
                     same_source=src_of[ga] == src_of[gb]))
rows.sort(key=lambda x: -x['overlap'])
for scope in ('newspaper', 'book', 'web'):
    ov = np.array([r['overlap'] for r in rows if r['same_source'] and src_of[r['a']] == scope])
    if len(ov):
        hist = np.histogram(ov, bins=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01])[0]
        print(scope, 'pairs', len(ov), 'hist', hist.tolist())
cross = np.array([r['overlap'] for r in rows if not r['same_source']])
print('cross-source pairs', len(cross), np.histogram(cross, bins=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01])[0].tolist())
print('\nTOP same-source pairs:')
for r in [r for r in rows if r['same_source']][:70]:
    print('%.3f %6d %6d %6d | %s || %s' % (r['overlap'], r['shared'], r['grams_a'], r['grams_b'],
                                           r['a'].replace('news:dir:Hindkowan Newspaper Data/Hindko wan data 2024 to 2026/', 'D:'),
                                           r['b'].replace('news:dir:Hindkowan Newspaper Data/Hindko wan data 2024 to 2026/', 'D:')))
print('\nTOP cross-source pairs:')
for r in [r for r in rows if not r['same_source']][:25]:
    print('%.3f %6d %6d %6d | %s || %s' % (r['overlap'], r['shared'], r['grams_a'], r['grams_b'], r['a'][:70], r['b'][:70]))
with open(os.path.join(ms.HERE, 'probe_group_overlap_out.json'), 'w', encoding='utf-8') as f:
    json.dump(rows[:3000], f, ensure_ascii=False, indent=0)
