"""Review helper: merge a batch of new review rows into review/review_decisions.tsv.

    python scripts/_merge_review.py NEW_ROWS.tsv

NEW_ROWS.tsv has the columns word, action, reason, alt (no header). A row whose
word already has a 'drop' / 'low' row with a DIFFERENT action replaces that row
(the old reason is kept and the new one appended); every other row is appended.
Exact duplicates are skipped, so the script can be re-run safely.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'review', 'review_decisions.tsv')
COLS = ['word', 'action', 'reason', 'alt']


def main():
    rows = list(csv.DictReader(open(PATH, encoding='utf-8'), delimiter='\t'))
    for r in rows:
        r['alt'] = (r.get('alt') or '').strip()
    new = []
    for ln in open(sys.argv[1], encoding='utf-8'):
        ln = ln.rstrip('\n')
        if not ln.strip():
            continue
        parts = ln.split('\t') + ['', '', '']
        new.append(dict(word=parts[0], action=parts[1], reason=parts[2], alt=parts[3].strip()))
    replaced = appended = skipped = 0
    for n in new:
        if any(r == n for r in rows):
            skipped += 1
            continue
        conf_rows = [r for r in rows if r['word'] == n['word'] and r['action'] in ('drop', 'low')]
        if n['action'] in ('drop', 'low') and conf_rows:
            r = conf_rows[0]
            if r['action'] == n['action'] and n['reason'] in r['reason']:
                skipped += 1
                continue
            r['reason'] = f"{r['reason']} || round 2: {n['reason']}"
            r['action'] = n['action']
            replaced += 1
        else:
            rows.append(n)
            appended += 1
    with open(PATH, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(COLS) + '\n')
        for r in rows:
            f.write('\t'.join(r[c] for c in COLS) + '\n')
    print(f'replaced {replaced}, appended {appended}, skipped {skipped}; total rows {len(rows)}')


if __name__ == '__main__':
    main()
