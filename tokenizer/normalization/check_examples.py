"""Verify the 20 examples in NORMALIZATION.md: 'before' occurs in the corpus and normalize(before) == after."""
import json, re, sys
sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp.normalize import normalize
md = open('NORMALIZATION.md', encoding='utf-8').read()
sec = md.split('## 8.')[1].split('## 9.')[0]
rows = [l for l in sec.splitlines() if re.match(r'\| \d+ \|', l)]
corpus = [json.loads(l)['text'] for l in open(r'F:\Hindko\hindko_dataset_permissive.jsonl', encoding='utf-8')]
big = '\n'.join(corpus)
ok = 0
for l in rows:
    cells = [c.strip() for c in l.strip('|').split('|')]
    num, before, after = cells[0], cells[3], cells[4]
    b = re.findall(r'`([^`]*)`', before)
    a = re.findall(r'`([^`]*)`', after)[0]
    b = (b[0] + chr(0x95) + b[1]) if len(b) == 2 else b[0]
    found = b in big
    norm = normalize(b)
    good = found and norm == a
    ok += good
    print(num, 'found' if found else 'NOT FOUND', 'norm==after' if norm == a else 'NORM DIFFERS %r' % norm)
print('%d/%d examples verified' % (ok, len(rows)))
