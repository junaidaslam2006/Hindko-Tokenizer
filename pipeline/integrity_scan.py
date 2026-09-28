"""Scan raw_files/ for integrity failures and quarantine them.

Uses hp.filecheck (container-signature validation, not just size). Bad files
are moved to raw_files/_quarantine/<same relative path> and appended to
_pipeline/corrupt_at_source.jsonl so the downloader will not keep re-fetching
them and the pipeline will never read them.

    python integrity_scan.py            # scan + quarantine
    python integrity_scan.py --report   # scan only, change nothing
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hp import filecheck

ROOT = r'F:\Hindko'
RAW = os.path.join(ROOT, 'raw_files')
QUARANTINE = os.path.join(RAW, '_quarantine')
CORRUPTLOG = os.path.join(ROOT, '_pipeline', 'corrupt_at_source.jsonl')


def iter_files():
    for dirpath, dirnames, filenames in os.walk(RAW):
        dirnames[:] = [d for d in dirnames if d != '_quarantine']
        for f in filenames:
            if f.startswith('_'):
                continue
            yield os.path.join(dirpath, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true',
                    help='only report; do not move or log anything')
    args = ap.parse_args()

    stats = collections.defaultdict(lambda: collections.Counter())
    bad = []
    parts = []

    for p in iter_files():
        if p.endswith('.part'):
            parts.append(p)
            continue
        rel = os.path.relpath(p, RAW).replace(os.sep, '/')
        ext = os.path.splitext(p)[1].lower().lstrip('.')
        ok, why = filecheck.validate(p)
        stats[ext]['total'] += 1
        if ok:
            stats[ext]['ok'] += 1
        else:
            stats[ext][why] += 1
            bad.append((rel, why, os.path.getsize(p)))

    print('=' * 74)
    print('INTEGRITY SCAN of raw_files/')
    print('=' * 74)
    print('%-6s %7s %7s   %s' % ('ext', 'total', 'ok', 'failures'))
    for ext in sorted(stats, key=lambda e: -stats[e]['total']):
        c = stats[ext]
        fails = {k: v for k, v in c.items() if k not in ('total', 'ok')}
        print('%-6s %7d %7d   %s' % (ext, c['total'], c['ok'],
                                     fails if fails else '-'))

    print('\nleftover .part files: %d' % len(parts))
    print('integrity failures  : %d' % len(bad))
    for rel, why, sz in bad[:25]:
        print('   %-28s %10d  %s' % (why, sz, rel[-72:]))
    if len(bad) > 25:
        print('   ... %d more' % (len(bad) - 25))

    if args.report:
        print('\n--report: nothing moved.')
        return 0

    if not bad:
        print('\nnothing to quarantine.')
        return 0

    moved = 0
    with open(CORRUPTLOG, 'a', encoding='utf-8', newline='\n') as log:
        for rel, why, sz in bad:
            src = os.path.join(RAW, rel.replace('/', os.sep))
            dst = os.path.join(QUARANTINE, rel.replace('/', os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                print('   could not move %s: %s' % (rel[-50:], e))
                continue
            log.write(json.dumps({
                'path': rel, 'reason': why, 'size': sz,
                'quarantined_to': os.path.relpath(dst, ROOT).replace(os.sep, '/'),
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }, ensure_ascii=False) + '\n')

    # remove empty leftover .part files
    removed = 0
    for p in parts:
        try:
            if os.path.getsize(p) == 0:
                os.remove(p)
                removed += 1
        except OSError:
            pass

    print('\nquarantined %d file(s) -> %s' % (moved, QUARANTINE))
    print('removed %d empty .part file(s)' % removed)
    print('logged -> %s' % CORRUPTLOG)
    return 0


if __name__ == '__main__':
    sys.exit(main())
