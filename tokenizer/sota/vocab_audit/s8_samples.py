"""Vocabulary audit, step 8: the 100 most frequent pieces on test_strict and 100 random
learned pieces (seed 20260927), with their support. The linguistic notes are added by
the auditor in s8b_notes.py.
Output: top100_test.tsv, random100.tsv
"""
import json, os, random

OUT = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(OUT, 'piece_stats.json'), encoding='utf-8'))
S = json.load(open(os.path.join(OUT, 'scan.json'), encoding='utf-8'))['per_piece']
tot_test = sum(r['test_count'] for r in P)
learned = [r for r in P if r['type'] == 'NORMAL']
top = sorted([r for r in P if r['test_count'] > 0], key=lambda r: -r['test_count'])[:100]
rng = random.Random(20260927)
rand = sorted(rng.sample(learned, 100), key=lambda r: r['id'])

def dump(rows, fn):
    cum = 0
    with open(os.path.join(OUT, fn), 'w', encoding='utf-8') as f:
        f.write('rank\tid\tpiece\ttype\tclass\ttest_count\ttest_pct\tcum_pct\td2_count\td2_docs\td2_units\twhole_word_share\n')
        for k, r in enumerate(rows, 1):
            cum += r['test_count']
            ww = S[r['id']]['whole_word_share']
            f.write('%d\t%d\t%s\t%s\t%s\t%d\t%.3f\t%.2f\t%d\t%d\t%d\t%s\n' % (
                k, r['id'], r['piece'].replace('\n', '\\n'), r['type'], r['class'], r['test_count'], 100 * r['test_count'] / tot_test,
                100 * cum / tot_test, r['d2_count'], r['d2_docs'], r['d2_units'], '%.2f' % ww if ww is not None else ''))
dump(top, 'top100_test.tsv')
dump(rand, 'random100.tsv')
print('top100 cover %.2f%% of test tokens' % (100 * sum(r['test_count'] for r in top) / tot_test))
for fn in ('top100_test.tsv', 'random100.tsv'):
    print('==', fn)
    for l in open(os.path.join(OUT, fn), encoding='utf-8').read().splitlines()[1:]:
        x = l.split('\t'); print(x[0], x[1], x[2], x[4], x[5], x[8], x[9], x[10], x[11])
