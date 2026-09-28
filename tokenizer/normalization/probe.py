"""Print contexts of codepoints / sequences in the permissive corpus.

usage: python probe.py [--n N] [--width W] [--source S] PATTERN [PATTERN ...]
PATTERN is a regex; write codepoints as \\uXXXX (python re understands them).
Output: per pattern, total matches, records, per-source counts, then N
contexts in file order (deterministic).
"""
import argparse
import json
import re
from collections import Counter

PERM = r'F:\Hindko\hindko_dataset_permissive.jsonl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='+')
    ap.add_argument('--n', type=int, default=15)
    ap.add_argument('--width', type=int, default=25)
    ap.add_argument('--source', default=None)
    ap.add_argument('--words', action='store_true', help='also print top words containing the match')
    a = ap.parse_args()
    rxs = [re.compile(p) for p in a.patterns]
    tot = [Counter() for _ in rxs]
    recs = [Counter() for _ in rxs]
    ex = [[] for _ in rxs]
    words = [Counter() for _ in rxs]
    wre = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u200CA-Za-z]+')
    with open(PERM, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if a.source and r['source'] != a.source:
                continue
            t = r['text']
            for k, rx in enumerate(rxs):
                ms = list(rx.finditer(t))
                if not ms:
                    continue
                tot[k][r['source']] += len(ms)
                recs[k][r['source']] += 1
                for m in ms:
                    if len(ex[k]) < a.n:
                        s = t[max(0, m.start() - a.width):m.end() + a.width].replace('\n', ' / ')
                        tag = r.get('site') or r.get('book_folder') or r.get('issue')
                        ex[k].append('%s|%s|%s| %s' % (r['source'], r['uid'], str(tag)[:30], s))
                    if a.words:
                        # word around match
                        s0 = m.start()
                        b = s0
                        while b > 0 and wre.match(t[b - 1]):
                            b -= 1
                        e = m.end()
                        while e < len(t) and wre.match(t[e]):
                            e += 1
                        words[k][t[b:e]] += 1
    for k, p in enumerate(a.patterns):
        print('=== %r  occ=%s  recs=%s' % (p, dict(tot[k]), dict(recs[k])))
        for e in ex[k]:
            print('   ', e)
        if a.words:
            print('   words(%d types): %s' % (len(words[k]), words[k].most_common(60)))


if __name__ == '__main__':
    main()
