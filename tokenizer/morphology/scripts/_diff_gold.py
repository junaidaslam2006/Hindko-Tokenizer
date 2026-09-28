"""Review helper: compare the current gold / candidate files with an earlier copy.

    python scripts/_diff_gold.py OLD_DIR

OLD_DIR holds morph_silver_high.tsv, morph_silver_low.tsv, candidates_all.tsv of the earlier run.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rd(path):
    return {r['word']: r for r in csv.DictReader(open(path, encoding='utf-8'), delimiter='\t')}


def main():
    old = sys.argv[1]
    oh, ol = rd(os.path.join(old, 'morph_silver_high.tsv')), rd(os.path.join(old, 'morph_silver_low.tsv'))
    nh, nl = rd(os.path.join(ROOT, 'morph_silver_high.tsv')), rd(os.path.join(ROOT, 'morph_silver_low.tsv'))
    oc = rd(os.path.join(old, 'candidates_all.tsv'))
    nc = rd(os.path.join(ROOT, 'data', 'candidates_all.tsv'))
    for name, a, b in (('HIGH', oh, nh), ('LOW', ol, nl)):
        gone = sorted(set(a) - set(b), key=lambda w: (a[w]['category'], -int(a[w]['freq'])))
        new = sorted(set(b) - set(a), key=lambda w: (b[w]['category'], -int(b[w]['freq'])))
        print(f'=== {name}: {len(a)} -> {len(b)}; removed {len(gone)}, added {len(new)}')
        for w in gone:
            now = nc.get(w)
            where = 'HIGH' if w in nh else 'LOW' if w in nl else (now['confidence'] if now else 'no parse')
            print(f"  - {a[w]['id']} {a[w]['segmentation']} {a[w]['category']} f={a[w]['freq']} -> {where}"
                  + (f" | {now['notes'][:160]}" if now else ''))
        for w in new:
            print(f"  + {b[w]['id']} {b[w]['segmentation']} {b[w]['category']} f={b[w]['freq']} ctx={b[w].get('context', '')}"
                  f" alt={b[w]['alternatives']} | {b[w]['notes'][:140]}")
        changed = [w for w in set(a) & set(b) if (a[w]['boundaries'], a[w]['alternatives']) !=
                   (b[w]['boundaries'], b[w]['alternatives'])]
        for w in sorted(changed):
            print(f"  ~ {w}: {a[w]['segmentation']} alt={a[w]['alternatives']} -> {b[w]['segmentation']} alt={b[w]['alternatives']}")
    lost = sorted(set(oc) - set(nc))
    print('=== candidates that lost every parse:', ' '.join(f"{w}({oc[w]['confidence']})" for w in lost))


if __name__ == '__main__':
    main()
