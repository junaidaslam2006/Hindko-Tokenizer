#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_overlap_merge.py - dry run of the 'near-duplicate groups' merge rule:
same-source groups sharing >= THR of the smaller group's distinct 8-grams
(counting only 8-grams found in <= MAXG groups) would be merged. Prints the
resulting components per source for a few thresholds. Diagnostic run on the RULE groups (before the near-duplicate merge it motivated). Read-only."""
import collections
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_splits as ms  # noqa: E402

cfg = ms.CONFIG
recs, _ = ms.load_records()
keys, _ = ms.assign_groups(recs, cfg)
gnames = sorted(set(keys))
gi = {g: i for i, g in enumerate(gnames)}
gid = np.array([gi[k] for k in keys], np.int32)
src_of = {k: r['source'] for k, r in zip(keys, recs)}
words = collections.Counter()
for k, r in zip(keys, recs):
    words[k] += r['n_words']
tok_ids, grams, shingles, short, _ = ms.build_features(recs, cfg)
H, R, OWN, _ = ms.gram_occurrences(tok_ids, grams, short)
H, R = H[OWN], R[OWN]
G = gid[R]
order = np.lexsort((G, H))
Hs, Gs = H[order], G[order]
keep = np.ones(len(Hs), bool)
keep[1:] = (Hs[1:] != Hs[:-1]) | (Gs[1:] != Gs[:-1])
Hs, Gs = Hs[keep], Gs[keep]
gpg = np.bincount(Gs, minlength=len(gnames))
new_h = np.ones(len(Hs), bool)
new_h[1:] = Hs[1:] != Hs[:-1]
st = np.nonzero(new_h)[0]
en = np.append(st[1:], len(Hs))
ng = en - st
short_name = lambda g: g.replace('news:dir:Hindkowan Newspaper Data/Hindko wan data 2024 to 2026/', 'D:')
for MAXG in (10,):
    S = collections.Counter()
    sel = (ng >= 2) & (ng <= MAXG)
    for s, e in zip(st[sel].tolist(), en[sel].tolist()):
        for a, b in itertools.combinations(Gs[s:e].tolist(), 2):
            S[(a, b)] += 1
    for THR in (0.2, 0.25, 0.3):
        uf = ms.UnionFind()
        edges = []
        for (a, b), c in S.items():
            ga, gb = gnames[a], gnames[b]
            if src_of[ga] != src_of[gb]:
                continue
            ov = c / max(1, min(gpg[a], gpg[b]))
            if ov >= THR:
                uf.union(ga, gb)
                edges.append((round(ov, 3), short_name(ga), short_name(gb)))
        comps = collections.defaultdict(list)
        for g in gnames:
            comps[uf.find(g)].append(g)
        multi = [v for v in comps.values() if len(v) > 1]
        print('\nMAXG=%d THR=%.2f: %d edges, %d merged components' % (MAXG, THR, len(edges), len(multi)))
        by_src = collections.Counter(src_of[v[0]] for v in multi)
        print('  components by source:', dict(by_src))
        for v in sorted(multi, key=lambda v: -sum(words[g] for g in v)):
            if src_of[v[0]] == 'web' and all(':record:' in g for g in v) and THR != 0.25:
                continue
            print('  %6d words | %s' % (sum(words[g] for g in v), ' + '.join(short_name(g) for g in sorted(v))[:400]))
        if THR == 0.25:
            for e in sorted(edges, reverse=True):
                if not e[1].startswith('web:'):
                    print('    edge', e)
