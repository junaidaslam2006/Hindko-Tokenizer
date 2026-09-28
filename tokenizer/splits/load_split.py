#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""load_split.py - stream one split of the Hindko corpus, text untouched.

    import sys; sys.path.insert(0, r'F:\\Hindko\\_tokenizer\\splits')
    from load_split import iter_split

    for uid, text, rec in iter_split('test', tier='strict', sources=['newspaper']):
        ...

* Split membership comes from split_manifest.jsonl (written by make_splits.py).
* Text is yielded exactly as released (NFC, ZWNJ kept, kashida removed).
  Apply the canonical normalisation downstream, then materialise; the split
  was verified leak-free under an aggressive matching normalisation, so it
  stays leak-free whatever digit / presentation-form policy is chosen.
* Records come out in dataset file order (deterministic).
* Every yielded record is checked against the manifest (uid, source, tier,
  n_words); a mismatch means the dataset changed after the split was built
  and raises instead of silently mixing splits.

CLI:  python load_split.py            -> verify manifest vs datasets + summary table
"""
from __future__ import annotations

import collections
import json
import os
import sys
from typing import Dict, Iterable, Iterator, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
MANIFEST_PATH = os.path.join(HERE, 'split_manifest.jsonl')
DATA = {
    'permissive': os.path.join(ROOT, 'hindko_dataset_permissive.jsonl'),
    'strict': os.path.join(ROOT, 'hindko_dataset.jsonl'),
}
SPLITS = ('train', 'validation', 'test')
ALIASES = {'val': 'validation', 'valid': 'validation', 'dev': 'validation'}
SOURCES = ('newspaper', 'book', 'web')

_MANIFEST_CACHE: Dict[str, Dict[str, dict]] = {}


def load_manifest(path: str = MANIFEST_PATH) -> Dict[str, dict]:
    """uid -> manifest row (cached per path)."""
    path = os.path.abspath(path)
    if path not in _MANIFEST_CACHE:
        rows = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if row['uid'] in rows:
                    raise RuntimeError('duplicate uid %s in %s' % (row['uid'], path))
                rows[row['uid']] = row
        _MANIFEST_CACHE[path] = rows
    return _MANIFEST_CACHE[path]


def _as_set(x, allowed, name) -> Optional[set]:
    if x is None:
        return None
    s = {x} if isinstance(x, str) else set(x)
    if allowed is not None and not s <= set(allowed):
        raise ValueError('%s must be drawn from %s, got %s' % (name, allowed, sorted(s)))
    return s


def _check(rec: dict, row: Optional[dict], tier: str) -> dict:
    if row is None:
        raise RuntimeError('uid %s is not in the split manifest: the dataset changed after the split '
                           'was built - re-run make_splits.py' % rec['uid'])
    if (row['source'] != rec['source'] or row['n_words'] != rec['n_words']
            or row['quality_tier'] != rec['quality_tier']
            or (tier == 'strict' and row['quality_tier'] != 'strict')):
        raise RuntimeError('manifest row for uid %s does not match the dataset record - re-run '
                           'make_splits.py' % rec['uid'])
    return row


def iter_split(split: str, tier: str = 'permissive', sources: Optional[Iterable[str]] = None,
               varieties: Optional[Iterable[str]] = None, drop_upstream_heldout: bool = False,
               manifest_path: str = MANIFEST_PATH) -> Iterator[Tuple[str, str, dict]]:
    """Yield (uid, text, record) for every record of `split`.

    split       'train' | 'validation' | 'test'  ('val', 'valid', 'dev' accepted)
    tier        'permissive' (all 18,283 records) | 'strict' (the 12,393-record subset;
                strict records inherit their permissive record's split)
    sources     None, or any of 'newspaper', 'book', 'web'
    varieties   None, or language_variety values to keep, e.g. {'hindko'} for a
                Hindko-only evaluation, {'hindko', 'mixed'} ...
    drop_upstream_heldout
                drop Omnilingual-ASR records whose UPSTREAM split is dev/test
                (speakers spk03 spk04 spk07 spk08 spk12 spk13). Use it for training
                data whenever a model will also be scored on the Omnilingual ASR
                benchmark; see splits_report.json -> omnilingual_upstream_split.
    """
    split = ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError('split must be one of %s, got %r' % (SPLITS, split))
    if tier not in DATA:
        raise ValueError('tier must be one of %s, got %r' % (tuple(DATA), tier))
    src_f = _as_set(sources, SOURCES, 'sources')
    var_f = _as_set(varieties, None, 'varieties')
    manifest = load_manifest(manifest_path)
    with open(DATA[tier], encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            row = _check(rec, manifest.get(rec['uid']), tier)
            if row['split'] != split:
                continue
            if src_f is not None and rec['source'] not in src_f:
                continue
            if var_f is not None and rec['language_variety'] not in var_f:
                continue
            if drop_upstream_heldout and row.get('upstream_split') in ('dev', 'test'):
                continue
            yield rec['uid'], rec['text'], rec


def verify(manifest_path: str = MANIFEST_PATH) -> dict:
    """Full consistency check of the manifest against both dataset files.
    Returns summary counts; raises on any mismatch."""
    manifest = load_manifest(manifest_path)
    seen = set()
    table = collections.Counter()
    with open(DATA['permissive'], encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            row = _check(rec, manifest.get(rec['uid']), 'permissive')
            seen.add(rec['uid'])
            for tier in ('permissive',) + (('strict',) if rec['quality_tier'] == 'strict' else ()):
                table[(tier, row['split'], rec['source'], 'records')] += 1
                table[(tier, row['split'], rec['source'], 'words')] += rec['n_words']
    missing = set(manifest) - seen
    if missing:
        raise RuntimeError('%d manifest uids are not in the permissive dataset' % len(missing))
    n_strict = 0
    with open(DATA['strict'], encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            _check(rec, manifest.get(rec['uid']), 'strict')
            n_strict += 1
    if n_strict != sum(v for k, v in table.items() if k[0] == 'strict' and k[3] == 'records'):
        raise RuntimeError('strict dataset does not match the strict records of the permissive dataset')
    return dict(records=len(seen), strict_records=n_strict, table=table)


def _main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    v = verify()
    print('manifest OK: %d permissive records, %d strict records' % (v['records'], v['strict_records']))
    t = v['table']
    for tier in ('permissive', 'strict'):
        print('\n%s' % tier)
        print('  %-10s %12s %12s %12s' % ('source', *SPLITS))
        for s in SOURCES + ('ALL',):
            srcs = SOURCES if s == 'ALL' else (s,)
            w = {sp: sum(t[(tier, sp, x, 'words')] for x in srcs) for sp in SPLITS}
            tot = sum(w.values())
            print('  %-10s ' % s + ' '.join('%12s' % ('%d (%.2f%%)' % (w[sp], 100 * w[sp] / tot) if tot else '-')
                                           for sp in SPLITS))


if __name__ == '__main__':
    _main()
