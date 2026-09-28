import json, sys, collections
sys.path.insert(0, 'F:/Hindko/_pipeline')
from hp.lang import TOKEN_SPLIT_RE, strip_marks
AUX = set('وے اے ہے ہیں ہن ون ان ایا ائی ائے ایاں ایہا ایہی ایہے ایہیاں آسا آسی آسے آسیاں اسا سا پیا پئی پئے پئیاں واں آں ایں ویں ہاں رہیا رئے رئی'.split())
words = sys.argv[1:]
tot = collections.Counter(); aux = collections.Counter()
for line in open('F:/Hindko/hindko_dataset.jsonl', encoding='utf-8'):
    r = json.loads(line)
    for ln in r['text'].split('\n'):
        toks = [strip_marks(t) for t in TOKEN_SPLIT_RE.split(ln) if t]
        toks = [t for t in toks if t]
        for i, t in enumerate(toks):
            if t in words:
                tot[t] += 1
                if i + 1 < len(toks) and toks[i+1] in AUX: aux[t] += 1
for w in words:
    print(w, tot[w], round(aux[w]/tot[w], 3) if tot[w] else None)
