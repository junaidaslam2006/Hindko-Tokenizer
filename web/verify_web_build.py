"""Verify a staged build that adds web sources against the current release.

1. Every newspaper/book record of the release must be in the staged build with
   the same text and the same tier (web intake must not change released records),
   and the staged build must add no newspaper/book records.
2. uid uniqueness; every web record has url, site, license field, retrieval.
3. Web tally per site x tier (docs, words) + language varieties + flags.
4. Writes samples of strict web records per site for reading.

    python verify_web_build.py F:/Hindko/_staging_web
"""
import collections
import hashlib
import json
import os
import random
import sys

REL = r'F:\Hindko'


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def key(r):
    return r['uid']


def h(t):
    return hashlib.sha1(t.encode('utf-8')).hexdigest()


def main(stage):
    ok = True
    rel_s = load(os.path.join(REL, 'hindko_dataset.jsonl'))
    rel_p = load(os.path.join(REL, 'hindko_dataset_permissive.jsonl'))
    st_s = load(os.path.join(stage, 'hindko_dataset.jsonl'))
    st_p = load(os.path.join(stage, 'hindko_dataset_permissive.jsonl'))

    # 1. released records unchanged
    for name, rel, st in (('strict', rel_s, st_s), ('permissive', rel_p, st_p)):
        R = {key(r): r for r in rel}
        S = {key(r): r for r in st if r['source'] != 'web'}
        missing = [u for u in R if u not in S]
        added = [u for u in S if u not in R]
        text_changed = [u for u in R if u in S and h(R[u]['text']) != h(S[u]['text'])]
        tier_changed = [u for u in R if u in S and R[u].get('quality_tier') != S[u].get('quality_tier')]
        flags_changed = [u for u in R if u in S and R[u].get('quality_flags') != S[u].get('quality_flags')]
        print('%-10s released %d | staged non-web %d | missing %d added %d text_changed %d tier_changed %d flags_changed %d'
              % (name, len(R), len(S), len(missing), len(added), len(text_changed), len(tier_changed), len(flags_changed)))
        for u in (missing + added + text_changed + tier_changed + flags_changed)[:10]:
            print('   e.g.', u, (R.get(u) or S.get(u))['source_path'])
        if missing or added or text_changed or tier_changed:
            ok = False

    # 2. schema
    uids = [r['uid'] for r in st_p]
    dup = [u for u, n in collections.Counter(uids).items() if n > 1]
    print('uids unique:', not dup, '(dups %d)' % len(dup))
    ok &= not dup
    web = [r for r in st_p if r['source'] == 'web']
    bad = [r['uid'] for r in web if not (r.get('url') and r.get('site') and 'license' in r and r.get('retrieval'))]
    print('web records missing url/site/license/retrieval:', len(bad))
    ok &= not bad
    strict_uids = {r['uid'] for r in st_s}
    wrong = [r['uid'] for r in st_s if r['source'] == 'web' and r.get('language_variety') != 'hindko']
    print('strict web records not hindko:', len(wrong))
    ok &= not wrong

    # 3. tally
    tally = collections.defaultdict(lambda: collections.Counter())
    for r in web:
        t = 'strict' if r['uid'] in strict_uids else 'permissive'
        tally[r['site']][t + '_docs'] += 1
        tally[r['site']][t + '_words'] += r.get('n_words') or len(r['text'].split())
    print('\n%-26s %8s %9s %8s %9s' % ('site', 'strict', 'words', 'perm', 'words'))
    tot = collections.Counter()
    for site in sorted(tally):
        c = tally[site]
        tot.update(c)
        print('%-26s %8d %9d %8d %9d' % (site, c['strict_docs'], c['strict_words'], c['permissive_docs'],
                                         c['permissive_words']))
    print('%-26s %8d %9d %8d %9d' % ('TOTAL', tot['strict_docs'], tot['strict_words'], tot['permissive_docs'],
                                     tot['permissive_words']))
    flags = collections.Counter(f.split('(')[0] for r in web for f in r.get('quality_flags') or [])
    print('\nweb quality flags:', dict(flags.most_common()))
    print('web pii masked records:', sum(1 for r in web if r.get('pii_masked')))

    # 4. samples
    random.seed(7)
    out = os.path.join(stage, 'web_samples.txt')
    with open(out, 'w', encoding='utf-8') as f:
        by = collections.defaultdict(list)
        for r in st_s:
            if r['source'] == 'web':
                by[r['site']].append(r)
        for site, rs in sorted(by.items()):
            for r in random.sample(rs, min(3, len(rs))):
                f.write('=== %s | %s | %s\n%s\n\n' % (site, r['uid'], r['url'], r['text'][:700]))
    print('samples ->', out)
    print('\nVERDICT:', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main(sys.argv[1])
