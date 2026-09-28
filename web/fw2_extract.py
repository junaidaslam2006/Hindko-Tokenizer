"""Turn FineWeb-2 scan hits (fw2/hits_*.jsonl) into a web source.

Every hit is re-scored with the current pipeline classifier (hp/langid.py).
A hit is kept when
  * P(hindko) >= 0.9 over the whole document and its domain is not one where
    most hits fall below that (see bad_domains), or
  * its URL is on a known Hindko site (hp.web.SITE_PREFIXES), whatever the score -
    hp/web.py then applies that site's policy (prior, dialect note, demotion).
Kept documents go to raw_files/web/fineweb2_hindko/docs.jsonl; the rest are
tallied by domain in fw2/extract_report.json so the decision can be audited.

    python fw2_extract.py
"""
import collections
import glob
import json
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import langid, web  # noqa: E402

HITS = r'F:\Hindko\_web\fw2'
OUT = r'F:\Hindko\raw_files\web\fineweb2_hindko'
MIN_P = web.FINEWEB_MIN_P
DOMAIN_MIN_HITS = 5


def main():
    m = langid.model()
    hits, seen = [], set()
    for f in sorted(glob.glob(os.path.join(HITS, 'hits_*.jsonl'))):
        for l in open(f, encoding='utf-8'):
            h = json.loads(l)
            if h['id'] in seen:
                continue
            seen.add(h['id'])
            hits.append(h)
    B = 500
    for i in range(0, len(hits), B):
        batch = hits[i:i + B]
        P = m.proba([h['text'][:6000] for h in batch])
        for h, p in zip(batch, P):
            h['p_v1'] = {k: round(v, 4) for k, v in p.items()}
    # Domain rule: a domain with >= DOMAIN_MIN_HITS hits of which fewer than half
    # reach MIN_P is a Punjabi/Saraiki/Urdu site whose best pages merely look
    # like Hindko (pakistanpoint.com, folkpunjab.org, pnb.wikipedia.org).
    dom_of = lambda h: urlparse(h['url']).netloc.replace('www.', '')
    per = collections.defaultdict(list)
    for h in hits:
        per[dom_of(h)].append(h['p_v1']['hindko'] >= MIN_P)
    bad_domains = {d for d, v in per.items() if len(v) >= DOMAIN_MIN_HITS and sum(v) < 0.5 * len(v)}
    kept, dropped = [], collections.Counter()
    for h in hits:
        site = web.site_for_url(h['url'])
        h['known_site'] = site
        if site or (h['p_v1']['hindko'] >= MIN_P and dom_of(h) not in bad_domains):
            kept.append(h)
        else:
            dropped[dom_of(h)] += 1
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'docs.jsonl'), 'w', encoding='utf-8') as out:
        for h in kept:
            out.write(json.dumps({
                'doc_id': 'fw2_' + h['id'].split(':')[-1].replace('-', '')[:24],
                'source_id': 'fineweb2_hindko',
                'url': h['url'],
                'title': None,
                'text': h['text'],
                'date': (h.get('date') or '')[:10] or None,
                'license': 'ODC-By-1.0 (FineWeb-2 dataset); page text (c) its publisher',
                'attribution': 'HuggingFaceFW/fineweb-2 (%s), Common Crawl %s' % (h['fw2_config'], h.get('dump')),
                'register': 'web',
                'meta': {'fineweb2_id': h['id'], 'fineweb2_config': h['fw2_config'], 'dump': h.get('dump'),
                         'glotlid_language': h.get('language'), 'glotlid_score': h.get('language_score'),
                         'scan_p': h.get('p'), 'p_v1': h['p_v1'], 'known_site': h['known_site']},
            }, ensure_ascii=False) + '\n')
    dom = collections.Counter(urlparse(h['url']).netloc.replace('www.', '') for h in kept)
    rep = {'hits_scanned': len(hits), 'kept': len(kept), 'min_p': MIN_P,
           'bad_domains': sorted(bad_domains),
           'kept_words': sum(len(h['text'].split()) for h in kept),
           'kept_by_domain': dom.most_common(), 'dropped_by_domain': dropped.most_common(200)}
    json.dump(rep, open(os.path.join(HITS, 'extract_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('hits', len(hits), 'kept', len(kept), 'words', rep['kept_words'])
    for d, n in dom.most_common(40):
        print('  %5d %s' % (n, d))


if __name__ == '__main__':
    main()
