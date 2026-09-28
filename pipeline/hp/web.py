"""Web sources (stage 3c): Hindko text collected from the open web.

Inputs are files already on disk; nothing here fetches:
  raw_files/web/<source_id>/docs.jsonl    API / feed / Wayback / dataset pulls
                                          (_web/ingest_hf.py, fetch_small_sites.py,
                                          wayback_site.py, fw2_extract.py)
  raw_files/web/<source_id>/pages.jsonl   polite site crawls (_web/crawl.py)

Per source, lines repeated across many pages of the same site (menus, footers,
share buttons, "posted by") are removed first and counted. Each document is then
cleaned, PII-masked, and language-classified block by block with hp.langid (Hindko
vs Punjabi / Urdu / Saraiki / Kashmiri / Persian / Arabic / Pashto / Sindhi / other), because web pages mix an Urdu intro with
a Hindko poem, or a Hindko column with Urdu comments.

Language decision:
  * sources with a site-level Hindko prior (the site publishes in Hindko and the
    survey read its pages): a document is Hindko unless Urdu (or another non-Lahnda
    language) blocks dominate. Where the classifier prefers Punjabi/Saraiki (formal
    minutes, Hazara speech transcripts) the document stays 'hindko' but carries
    'langid_disagrees(...)', which keeps it out of the strict tier.
  * sources without a prior (pages mined from FineWeb-2): Hindko only when the
    classifier says so with P(hindko) >= FINEWEB_MIN_P.
Nothing is dropped for language: non-Hindko documents go to the permissive tier
flagged (build.gate_record), exactly like Urdu newspaper columns and Urdu books.
"""
import collections
import hashlib
import json
import os
import re
from typing import Dict, List

from . import clean as cl
from . import lang
from . import langid

WEB_ROOT_REL = 'web'
# Without a site prior the classifier alone decides, and it confuses Lahore
# Punjabi with Hindko at ~2% per block (held-out test, _web/langid_train.log):
# the first FineWeb-2 extraction at 0.5 kept pakistanpoint.com / folkpunjab.org /
# pnb.wikipedia Punjabi pages. 0.9 keeps the Hindko pages that were read.
FINEWEB_MIN_P = 0.9
BLOCK_MIN_CHARS = 250          # classify blocks of at least this many characters
BOILER_MIN_PAGES = 3           # a line on >= this many pages of one site ...
BOILER_MIN_SHARE = 0.10        # ... and on >= this share of them is site furniture
BOILER_MAX_LEN = 200           # longer lines are content even if repeated (dedup handles copies)
URDU_DOMINANT = 0.5            # char-weighted Urdu share above which a doc is Urdu
MIXED_URDU_SHARE = 0.25        # Hindko docs with this much Urdu get a flag and leave strict
# classifier classes that are never Hindko, even on a Hindko site (Punjabi and
# Saraiki calls on Hindko sites are formal/Hazara/Attock Hindko - see decide_language)
OTHER_LANGUAGES = ('urdu', 'kashmiri', 'persian', 'arabic', 'pashto', 'sindhi', 'other')

# Site-level knowledge from the survey (_web/survey/*.md). 'prior' = the site
# publishes Hindko; 'dialect' is what the survey could establish; 'demote' puts
# the whole source in the permissive tier with that flag until resolved.
# 'section_only': Hindko is one section of an otherwise Urdu/other site (or the
# page was mined from a general crawl), so a page that is NOT Hindko was never
# part of a Hindko publication: build.gate_record rejects it (kept in rejected/)
# instead of demoting it, unlike Urdu posts on an all-Hindko site.
SOURCES: Dict[str, dict] = {
    'common_voice_hno': dict(prior='hindko', role='sentence_set', dialect='Northern Hindko (Mansehra/Balakot/Kaghan)',
                             retrieval='hf_parquet_text_columns'),
    'omnilingual_asr_hno': dict(prior='hindko', role='transcript', dialect='Northern Hindko (Hazara), spontaneous speech',
                                retrieval='hf_datasets_server_rows'),
    'web_gandharahindko': dict(prior='hindko', role='web_page', dialect='Peshawari Hindko', retrieval='wordpress_rest_api'),
    'web_aaprihindko': dict(prior='hindko', role='web_page', dialect='Hazara Hindko (Abbottabad)', retrieval='wayback'),
    'web_hindko_org': dict(prior='hindko', role='web_page', dialect='Hazara Hindko (Abbottabad/Mansehra)',
                           retrieval='sitemap_crawl'),
    'web_hindkomaza': dict(prior='hindko', role='web_page', dialect='Hazara Hindko', retrieval='sitemap_crawl'),
    'web_hindko_blogs': dict(prior='hindko', role='web_page', dialect='Hindko (blogs; some Urdu posts)',
                             retrieval='blogger_feed'),
    'web_noukeqalam': dict(prior='hindko', role='web_page', retrieval='wayback', section_only=True,
                           dialect='Hindko/Pahari border (genitive na/ni/ne); needs native-speaker review',
                           demote='dialect_unverified(hindko_or_pahari)'),
    'web_iattock': dict(prior='hindko', role='web_page', retrieval='wayback', section_only=True,
                        dialect='Chhachhi/Campbellpuri (Attock); the site calls it an accent of Punjabi, '
                                'linguists usually group Chhachhi with Hindko; needs native-speaker review',
                        demote='dialect_unverified(chhachhi_hindko_or_punjabi)'),
    'web_tvshia_hn': dict(prior='hindko', role='web_page', dialect='Peshawari Hindko', retrieval='wayback',
                          section_only=True),   # multilingual Shia TV site; /hn/ also serves Urdu news
    # found by the FineWeb-2 scan (_web/fw2_extract.py), then fetched whole
    'web_hazarewall': dict(prior='hindko', role='web_page', dialect='Hazara Hindko (stories, poetry)',
                           retrieval='wayback'),
    'web_botanix_hnd': dict(prior='hindko', role='web_page', retrieval='sitemap_crawl',
                            dialect='Southern Hindko (hnd section of a multilingual botany site)'),
    'web_galyattimes': dict(prior='hindko', role='web_page', retrieval='wayback', section_only=True,
                            dialect='Galyat (Murree hills) Hindko/Pahari border (genitive ni/ne, bich); '
                                    'needs native-speaker review',
                            demote='dialect_unverified(hindko_or_pahari)'),
    'web_mansehra_com': dict(prior=None, role='web_page', section_only=True, dialect='Hazara (Mansehra) local news site; Hindko '
                             'columns among Urdu news', retrieval='wordpress_rest_api'),
    'fineweb2_hindko': dict(prior=None, role='web_page', dialect=None, retrieval='fineweb2_scan', section_only=True),
}
DEFAULT_SOURCE = dict(prior=None, role='web_page', dialect=None, retrieval='unknown')
# Known Hindko sites by URL prefix (host without www + path). A page from one of
# these that reached us another way (FineWeb-2) gets that site's policy.
SITE_PREFIXES = {
    'gandharahindko.com/': 'web_gandharahindko',
    'aaprihindko.com/': 'web_aaprihindko',
    'hindko.org/hno': 'web_hindko_org',
    'hindkomaza.home.blog/': 'web_hindkomaza',
    'hindkopoint.blogspot.com/': 'web_hindko_blogs',
    'noukeqalam.com/islam/': 'web_noukeqalam',
    'noukeqalam.com/archives/islam/': 'web_noukeqalam',
    'iattock.com/': 'web_iattock',
    'tvshia.com/hn/': 'web_tvshia_hn',
    'hazar-e-wall.com/': 'web_hazarewall',
    'botanix.kpr.eu/hnd/': 'web_botanix_hnd',
    'botanix.kpr-eshop.eu/hnd/': 'web_botanix_hnd',
    'galyattimes.com/': 'web_galyattimes',
}


def site_for_url(url: str):
    u = re.sub(r'^https?://(www\.)?', '', url or '').lower()
    u = re.sub(r'^([^/:]+):\d+', r'\1', u)
    for pre, sid in SITE_PREFIXES.items():
        if u.startswith(pre):
            return sid
    return None


def policy(sid: str, url: str = None) -> dict:
    pol = SOURCES.get(sid, DEFAULT_SOURCE)
    if pol['prior'] is None and url:
        site = site_for_url(url)
        if site:
            pol = dict(SOURCES[site], retrieval=pol['retrieval'], via_site=site)
    return pol


# When the same page is held twice (site API/Wayback and FineWeb-2), the copy
# processed first wins in the deduplicator: direct site copies before FineWeb.
SOURCE_ORDER = {s: i for i, s in enumerate(SOURCES)}

_NORM_LINE_RE = re.compile(r'\s+')
# page furniture that is not text: media file names and file sizes (video pages)
FILE_LINE_RE = re.compile(r'^\s*(\S+\.(mp4|mp3|flv|avi|mov|wav|m4a|jpe?g|png|gif|pdf|zip|rar)|'
                          r'[0-9.,]+\s*(KB|MB|GB|kb|mb|gb))\s*$')
# index pages (category / tag / pagination / video lists): lists of titles, not text
LISTING_URL_RE = re.compile(r'/(category|tag|author)(/|$)|/page/\d+|[?&](page|paged|cat)=\d|'
                            r'/hn/(video|photo|category)(\?|/|$)|/sitemap/?$|'
                            r'^https?://[^/]+/?$', re.I)      # a site's home page lists its posts


def _line_key(line: str) -> str:
    return _NORM_LINE_RE.sub(' ', line).strip()


def load_docs(web_dir: str) -> List[dict]:
    docs = []
    if not os.path.isdir(web_dir):
        return docs
    for sid in sorted(os.listdir(web_dir)):
        d = os.path.join(web_dir, sid)
        p = os.path.join(d, 'docs.jsonl')
        if os.path.exists(p):
            for i, l in enumerate(open(p, encoding='utf-8')):
                x = json.loads(l)
                x.setdefault('source_id', sid)
                x['_file'] = 'docs.jsonl'
                x['_line'] = i
                docs.append(x)
        p = os.path.join(d, 'pages.jsonl')
        if os.path.exists(p):
            seen = set()
            for i, l in enumerate(open(p, encoding='utf-8')):
                x = json.loads(l)
                if x.get('status') != 200 or not x.get('text') or x['url'] in seen:
                    continue
                seen.add(x['url'])
                docs.append({'doc_id': '%s_%s' % (sid, hashlib.sha1(x['url'].encode()).hexdigest()[:12]),
                             'source_id': sid, 'url': x['url'], 'title': x.get('title'), 'text': x['text'],
                             'date': x.get('date'), 'license': None, 'attribution': None, 'register': 'web',
                             'meta': {'fetched_at': x.get('fetched_at'), 'final_url': x.get('final_url')},
                             '_file': 'pages.jsonl', '_line': i})
    return docs


def site_boilerplate(docs: List[dict]) -> Dict[str, set]:
    """Per source: short lines that recur on many of its pages."""
    by_src = collections.defaultdict(list)
    for x in docs:
        by_src[x['source_id']].append(x)
    out = {}
    for sid, xs in by_src.items():
        if len(xs) < 5 or SOURCES.get(sid, DEFAULT_SOURCE)['role'] != 'web_page':
            out[sid] = set()
            continue
        # Count over content pages only: a listing page repeats the opening
        # lines of the posts it lists, and those must not become "furniture"
        # to be cut out of the posts themselves.
        content = [x for x in xs if not LISTING_URL_RE.search(x.get('url') or '')] or xs
        c = collections.Counter()
        for x in content:
            c.update({_line_key(l) for l in (x.get('text') or '').split('\n')
                      if l.strip() and len(l) <= BOILER_MAX_LEN})
        need = max(BOILER_MIN_PAGES, BOILER_MIN_SHARE * len(content))
        out[sid] = {k for k, n in c.items() if n >= need}
    return out


def blocks(text: str) -> List[str]:
    """Blank-line blocks, merged forward until >= BLOCK_MIN_CHARS."""
    out, buf = [], ''
    for b in re.split(r'\n\s*\n|\n', text):
        b = b.strip()
        if not b:
            continue
        buf = (buf + '\n' + b) if buf else b
        if len(buf) >= BLOCK_MIN_CHARS:
            out.append(buf)
            buf = ''
    if buf:
        if out and len(buf) < BLOCK_MIN_CHARS:
            out[-1] += '\n' + buf
        else:
            out.append(buf)
    return out


def classify(text: str) -> dict:
    """Char-weighted class shares over blocks + whole-doc probabilities."""
    bs = [b for b in blocks(text) if len(langid.norm(b)) >= 40]
    if not bs:
        return {'shares': {}, 'p': {}, 'n_blocks': 0}
    m = langid.model()
    P = m.proba(bs)
    w = [len(langid.norm(b)) for b in bs]
    tot = float(sum(w))
    shares = collections.Counter()
    p_avg = collections.Counter()
    for p, wt in zip(P, w):
        shares[max(p, key=p.get)] += wt / tot
        for k, v in p.items():
            p_avg[k] += v * wt / tot
    return {'shares': {k: round(v, 4) for k, v in shares.items()},
            'p': {k: round(v, 4) for k, v in p_avg.items()}, 'n_blocks': len(bs)}


def decide_language(pol: dict, cls: dict) -> tuple:
    """(language_variety, flags) for one web document under its site policy."""
    sh, p = cls['shares'], cls['p']
    flags = []
    if not sh:
        return 'no_signal', flags
    top = max(p, key=p.get)
    urdu = sh.get('urdu', 0.0)
    if pol['prior'] == 'hindko':
        for other in OTHER_LANGUAGES:
            if sh.get(other, 0.0) >= URDU_DOMINANT:
                return other, flags
        if top != 'hindko':
            flags.append('langid_disagrees(top=%s,p=%.2f)' % (top, p[top]))
        if urdu >= MIXED_URDU_SHARE:
            flags.append('urdu_blocks(share=%.2f)' % urdu)
        return 'hindko', flags
    # no site prior: the classifier decides
    if p.get('hindko', 0.0) >= FINEWEB_MIN_P:
        if urdu >= MIXED_URDU_SHARE:
            flags.append('urdu_blocks(share=%.2f)' % urdu)
        return 'hindko', flags
    return top, flags


def process_web(web_dir: str, log) -> dict:
    docs = load_docs(web_dir)
    boiler = site_boilerplate(docs)
    recs = []
    stats = collections.Counter()
    per_source = collections.defaultdict(collections.Counter)
    for x in docs:
        sid = x['source_id']
        pol = policy(sid, x.get('url'))
        raw = x.get('text') or ''
        bl = boiler.get(sid, set())
        kept_lines, n_boiler = [], 0
        for l in raw.split('\n'):
            if l.strip() and (_line_key(l) in bl or FILE_LINE_RE.match(l)):
                n_boiler += 1
                continue
            kept_lines.append(l)
        text, pii = cl.mask_pii(cl.clean_text('\n'.join(kept_lines), keep_blank_lines=True))
        cls = classify(text)
        variety, lflags = decide_language(pol, cls)
        flags = list(lflags)
        if pol.get('demote'):
            flags.append(pol['demote'])
        if LISTING_URL_RE.search(x.get('url') or ''):
            flags.append('listing_page')
        h_score = lang.hindko_score_v2(text)
        rel = '%s/%s/%s#%d' % (WEB_ROOT_REL, sid, x['_file'], x['_line'])
        meta = dict(x.get('meta') or {})
        a = {
            'source': 'web',
            'source_path': rel,
            'source_file': sid,
            'article_index': 0,
            'web_order': (SOURCE_ORDER.get(sid, len(SOURCE_ORDER)), sid, x['_file'], x['_line']),
            'doc_id': x.get('doc_id'),
            'url': x.get('url'),
            'site': pol.get('via_site') or sid,
            'title': (x.get('title') or None),
            'title_source': 'web_page_title' if x.get('title') else None,
            'date': (x.get('date') or None),
            'date_precision': 'day' if x.get('date') and len(str(x['date'])) >= 10 else None,
            'date_source': ('wayback_capture' if pol['retrieval'] == 'wayback'
                            else 'publisher_metadata' if x.get('date') else None),
            'license': x.get('license'),
            'attribution': x.get('attribution'),
            'register': x.get('register'),
            'retrieval': pol['retrieval'],
            'dialect_note': pol.get('dialect'),
            'role': pol['role'],
            'web_section_only': bool(pol.get('section_only')),
            'extraction_method': {'wayback': 'wayback_html_extract', 'sitemap_crawl': 'html_extract',
                                  'wordpress_rest_api': 'wordpress_api_html_extract',
                                  'blogger_feed': 'blogger_feed_html_extract',
                                  'fineweb2_scan': 'fineweb2_trafilatura',
                                  'hf_parquet_text_columns': 'dataset_text_column',
                                  'hf_datasets_server_rows': 'dataset_text_column'}.get(pol['retrieval'], 'html_extract'),
            'text': text,
            'pii_masked': pii,
            'unmapped_glyphs': text.count(cl.REPLACEMENT),
            'is_ad': False,
            'hindko_score': round(h_score, 4),
            'language_variety': variety,
            'langid_shares': cls['shares'],
            'langid_p': cls['p'],
            'content_flags': flags,
            'boilerplate_lines_removed': n_boiler,
            'web_meta': meta,
        }
        stats['docs'] += 1
        stats['boilerplate_lines_removed'] += n_boiler
        per_source[sid][variety] += 1
        recs.append(a)
    recs.sort(key=lambda a: a['web_order'])
    log('web: %d documents from %d sources; %d site-furniture lines removed'
        % (len(recs), len(per_source), stats['boilerplate_lines_removed']))
    for sid, c in sorted(per_source.items()):
        log('  web %-24s %s' % (sid, dict(c)))
    return {'records': recs, 'stats': dict(stats),
            'per_source': {k: dict(v) for k, v in per_source.items()},
            'boilerplate_lines': {k: len(v) for k, v in boiler.items()}}
