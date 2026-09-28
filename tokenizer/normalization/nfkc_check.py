"""What NFKC (SentencePiece's default nmt_nfkc core) would change in the canonical corpus."""
import json, sys, unicodedata
from collections import Counter
sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import normalize as N
chg = Counter(); recs = Counter(); growth = 0
with open(r'F:\Hindko\hindko_dataset_permissive.jsonl', encoding='utf-8') as f:
    for line in f:
        t = N.normalize(json.loads(line)['text'])
        seen = set()
        for ch in set(t):
            k = unicodedata.normalize('NFKC', ch)
            if k != ch:
                n = t.count(ch); chg[ch] += n; seen.add(ch)
                growth += n * (len(k) - 1)
        for ch in seen: recs[ch] += 1
out = {('U+%04X %s' % (ord(c), unicodedata.name(c, '?'))): [n, recs[c], unicodedata.normalize('NFKC', c)] for c, n in chg.most_common()}
json.dump({'changed': out, 'length_growth_chars': growth}, open('nfkc_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=0)); print('growth', growth)
