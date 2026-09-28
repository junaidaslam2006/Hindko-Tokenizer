"""Print gold / candidate rows for given words or ids (review helper)."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = ['morph_silver_high.tsv', 'morph_silver_low.tsv', 'data/candidates_all.tsv']
keys = set(sys.argv[1:])
for fn in files:
    for r in csv.DictReader(open(os.path.join(ROOT, fn), encoding='utf-8'), delimiter='\t'):
        if r.get('id') in keys or r['word'] in keys:
            print(fn, '::', ' | '.join(f'{k}={v}' for k, v in r.items() if v not in ('', None)))
            print()
