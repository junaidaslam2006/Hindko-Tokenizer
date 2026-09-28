"""Alphabet of the canonical data form: distinct codepoints, coverage curve,
and codepoints of the canonical permissive corpus never seen in the canonical
strict corpus. Writes alphabet/alphabet.json. Deterministic.
"""
import json
import os
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import normalize as N  # noqa: E402

PERM = r'F:\Hindko\hindko_dataset_permissive.jsonl'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'alphabet')
os.makedirs(OUT, exist_ok=True)


def main():
    perm = Counter()
    strict = Counter()
    perm_src = {}
    with open(PERM, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            t = N.normalize(r['text'])
            c = Counter(t)
            perm.update(c)
            if r['quality_tier'] == 'strict':
                strict.update(c)
            for ch in c:
                perm_src.setdefault(ch, Counter())[r['source'] + ('/' + (r.get('site') or '') if r['source'] == 'web'
                                                                  else '')] += c[ch]
    total = sum(perm.values())
    ranked = sorted(perm.items(), key=lambda x: (-x[1], x[0]))
    cum, curve = 0, {}
    marks = [0.99, 0.999, 0.9995, 0.9999, 0.99999, 1.0]
    for i, (ch, k) in enumerate(ranked, 1):
        cum += k
        for m in marks:
            if m not in curve and cum / total >= m - 1e-12:
                curve[m] = i
    unseen = [{'cp': 'U+%04X' % ord(ch), 'name': unicodedata.name(ch, '?'), 'count': k,
               'sources': dict(perm_src[ch].most_common())}
              for ch, k in sorted(perm.items(), key=lambda x: ord(x[0])) if ch not in strict]
    res = {'normalization_version': N.NORMALIZATION_VERSION,
           'chars_permissive': total, 'chars_strict': sum(strict.values()),
           'distinct_codepoints_permissive': len(perm), 'distinct_codepoints_strict': len(strict),
           'codepoints_for_coverage': {str(k): v for k, v in curve.items()},
           'unseen_in_strict_after_normalization': unseen,
           'n_unseen_in_strict_after_normalization': len(unseen)}
    with open(os.path.join(OUT, 'alphabet.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'unseen_in_strict_after_normalization'}, indent=1))
    for u in unseen:
        print(u['cp'], u['name'][:40], u['count'], u['sources'])


if __name__ == '__main__':
    main()
