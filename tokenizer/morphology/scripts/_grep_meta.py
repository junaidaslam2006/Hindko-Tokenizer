import json, re, sys, collections
pat = re.compile(sys.argv[1])
fn = sys.argv[2] if len(sys.argv) > 2 else 'F:/Hindko/hindko_dataset_permissive.jsonl'
lim = int(sys.argv[3]) if len(sys.argv) > 3 else 40
n = 0
per = collections.Counter()
for l in open(fn, encoding='utf-8'):
    r = json.loads(l)
    for m in pat.finditer(r['text']):
        per[(r['source'], r.get('book_title') or r.get('site') or r.get('issue'))] += 1
        if n < lim:
            s = r['text'][max(0, m.start()-150): m.end()+150].replace('\n', ' / ')
            print(f"[{r['uid']} {r['source']} {r.get('book_title') or r.get('site')}] ...{s}...\n")
        n += 1
print('TOTAL', n)
for k, v in per.most_common(15): print(v, k)
