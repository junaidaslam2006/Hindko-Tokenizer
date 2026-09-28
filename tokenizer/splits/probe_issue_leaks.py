#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_issue_leaks.py - for groups that are ineligible because >= 50 % of
their words are leak-prone, find WHERE the duplicated text lives (which
groups / sources contain it). Diagnostic run on the RULE groups (before the near-duplicate merge it motivated). Read-only diagnostic."""
import collections
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
N = len(recs)
words = np.array([r['n_words'] for r in recs], float)
tok_ids, grams, shingles, short, _ = ms.build_features(recs, cfg)
H, R, OWN, _ = ms.gram_occurrences(tok_ids, grams, short)
sigs, empty = ms.minhash_signatures(shingles, cfg)
PI, PJ, _ = ms.minhash_candidates(sigs, empty, cfg)
PJAC = ms.exact_jaccard(shingles, PI, PJ)
idx = ms.LeakIndex(H, R, OWN, gid, N, PI, PJ, PJAC)
inP, cont, jac, rounds = ms.leak_prone_fixpoint(idx, cfg, np.zeros(N, bool))

g_words = np.bincount(gid, weights=words, minlength=len(gnames))
g_p = np.bincount(gid, weights=words * inP, minlength=len(gnames))
bad = [g for g in gnames if g_p[gi[g]] / g_words[gi[g]] >= 0.5]
print('ineligible-by-leak groups:', len(bad), collections.Counter(g.split(':')[0] + ':' + g.split(':')[1] for g in bad))

short_name = lambda g: g.replace('news:dir:Hindkowan Newspaper Data/Hindko wan data 2024 to 2026/', 'D:')
out = {}
src_of_partner = collections.Counter()
for g in bad:
    members = np.nonzero((gid == gi[g]) & inP)[0]
    pc = collections.Counter()
    pw = collections.Counter()
    for r in members.tolist():
        pool = gid != gid[r]
        pool |= inP & (np.arange(N) != r)
        q, qc = idx.top_container(r, pool)
        tgt = keys[q] if q is not None else '(jaccard-only)'
        pc[tgt] += 1
        pw[tgt] += words[r]
        src_of_partner[(recs[r]['source'], recs[q]['source'] if q is not None else None)] += words[r]
    out[g] = dict(words=int(g_words[gi[g]]), leak_prone_words=int(g_p[gi[g]]),
                  partners=[(short_name(k), int(pw[k]), pc[k]) for k, _ in pw.most_common(6)])
for g in bad[:60]:
    o = out[g]
    print(short_name(g), o['words'], o['leak_prone_words'])
    for p in o['partners'][:4]:
        print('      ', p)
print('leak-prone words in ineligible groups by (source, partner source):', dict(src_of_partner))

# example: issue record vs its top partner, first 200 chars each
ex = [g for g in bad if g.startswith('news:issue:')][:3]
for g in ex:
    r = int(np.nonzero((gid == gi[g]) & inP)[0][0])
    pool = gid != gid[r]
    q, qc = idx.top_container(r, pool)
    print('\n==', g, recs[r]['uid'], recs[r]['source_path'], 'share', round(qc, 3), 'cont', round(cont[r], 3))
    print('   ', recs[r]['text'][:200].replace('\n', ' / '))
    print('  ->', keys[q], recs[q]['uid'], recs[q].get('source_path'))
    print('   ', recs[q]['text'][:200].replace('\n', ' / '))
json.dump(out, open(os.path.join(ms.HERE, 'probe_issue_leaks_out.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=0)
