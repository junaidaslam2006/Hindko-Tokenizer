"""Verify that raw_files/ contains the Drive mirror, and report what is missing.

Run this after downloading the folder manually. It is read-only.

    python F:\\Hindko\\_pipeline\\verify_mirror.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

ROOT = r'F:\Hindko'
LISTING = os.path.join(ROOT, '_pipeline', 'drive_listing.json')
RAW_FILES = os.path.join(ROOT, 'raw_files')

TEXT_EXTS = {'inp', 'b01'}
OCR_EXTS = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}
NO_TEXT_EXTS = {'cdr', 'eps', 'tmp', 'zip'}


def rel_paths_under(root):
    """All files under root, as POSIX-style paths relative to root."""
    out = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            if os.path.basename(rel).startswith('_'):
                continue            # our own manifest / index files
            if rel.endswith('.part'):
                continue
            out.add(rel)
    return out


def main():
    entries = json.load(open(LISTING, encoding='utf-8'))
    wanted = {e['path'] for e in entries}
    have = rel_paths_under(RAW_FILES)

    # Common zip-extraction mistake: an extra nesting level.
    top_dirs = {p.split('/')[0] for p in have}
    if wanted and not (wanted & have):
        print('!! No expected paths found under raw_files/.')
        print('   Found top-level entries: %s' % sorted(top_dirs)[:10])
        print('   Expected top-level     : %s'
              % sorted({p.split('/')[0] for p in wanted}))
        print('   If you extracted a zip, make sure "Hindkowan Newspaper Data"')
        print('   sits DIRECTLY inside raw_files\\ (not doubly nested).')

    present = wanted & have
    missing = wanted - have
    extra = have - wanted

    by_ext_present = collections.Counter(
        os.path.splitext(p)[1].lower().lstrip('.') for p in present)
    by_ext_missing = collections.Counter(
        os.path.splitext(p)[1].lower().lstrip('.') for p in missing)

    total_bytes = 0
    for p in present:
        try:
            total_bytes += os.path.getsize(os.path.join(RAW_FILES, p.replace('/', os.sep)))
        except OSError:
            pass

    print('=' * 72)
    print('MIRROR VERIFICATION')
    print('=' * 72)
    print('Drive listing entries : %d' % len(wanted))
    print('Present on disk       : %d' % len(present))
    print('Missing               : %d' % len(missing))
    print('Unexpected extra files: %d' % len(extra))
    print('Size on disk          : %.2f GB' % (total_bytes / 1e9))

    print('\n%-8s %9s %9s' % ('ext', 'present', 'missing'))
    for ext in sorted(set(by_ext_present) | set(by_ext_missing),
                      key=lambda e: -(by_ext_present[e] + by_ext_missing[e])):
        print('.%-7s %9d %9d' % (ext, by_ext_present[ext], by_ext_missing[ext]))

    def group_status(exts, label):
        want = {p for p in wanted
                if os.path.splitext(p)[1].lower().lstrip('.') in exts}
        got = want & have
        print('\n%s: %d / %d present (%.1f%%)'
              % (label, len(got), len(want), 100.0 * len(got) / max(len(want), 1)))
        miss = sorted(want - have)
        if miss:
            print('  missing examples:')
            for m in miss[:10]:
                print('    -', m)
            if len(miss) > 10:
                print('    ... %d more' % (len(miss) - 10))
        return len(got), len(want)

    t_got, t_want = group_status(TEXT_EXTS, 'TEXT SOURCES (.inp/.B01)')
    o_got, o_want = group_status(OCR_EXTS, 'OCR CANDIDATES (images)')
    group_status(NO_TEXT_EXTS, 'NO-TEXT ARCHIVAL (.cdr/.eps/.tmp/.zip)')

    if extra:
        print('\nunexpected extra files (first 10):')
        for e in sorted(extra)[:10]:
            print('    +', e)

    print('\n' + '=' * 72)
    if t_got == t_want and t_want:
        print('READY: all text sources present. The text dataset can be built now.')
        if o_got == o_want:
            print('       All OCR candidates present too - full pipeline possible.')
        else:
            print('       OCR stage still needs %d more image(s).' % (o_want - o_got))
    elif t_got:
        print('PARTIAL: %d/%d text sources present. You can build a partial '
              'dataset now, or wait for the rest.' % (t_got, t_want))
    else:
        print('NOT READY: no text sources found under raw_files/ yet.')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
