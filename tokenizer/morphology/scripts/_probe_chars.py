import json, re, collections, sys
sys.path.insert(0, 'F:/Hindko/_pipeline')
from hp.lang import strip_marks
pats = {'nR (ن+ڑ)': 'نڑ', 'U+0768 ݨ': chr(0x0768), 'U+06BB ڻ': chr(0x06BB), 'n+tah نط': 'نط', 'U+0769': chr(0x0769), 'U+06B9': chr(0x06B9), 'ZWNJ': chr(0x200C), 'tone letters': None}
c = collections.Counter(); bysrc = collections.defaultdict(collections.Counter)
for l in open('F:/Hindko/hindko_dataset.jsonl', encoding='utf-8'):
    r = json.loads(l); t = r['text']
    for k, p in pats.items():
        if p is None:
            n = sum(1 for ch in t if 0x08BE <= ord(ch) <= 0x08C2)
        else:
            n = t.count(p)
        c[k] += n; bysrc[r['source'] if r['source'] != 'web' else r['site']][k] += n
print(c)
for s, cc in bysrc.items(): print(s, dict(cc))
