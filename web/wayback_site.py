"""Recover a dead Hindko website from the Internet Archive Wayback Machine.

For every distinct URL (CDX collapse=urlkey) under the prefix, fetch the
latest successful HTML capture (raw 'id_' form, no Wayback toolbar), extract
the main text and write one document per page. One request at a time.

    python wayback_site.py web_noukeqalam noukeqalam.com '/(archives/)?islam/|/archives/\d+/islam'
    python wayback_site.py web_iattock iattock.com
    python wayback_site.py web_tvshia_hn tvshia.com/hn/

Output: F:/Hindko/raw_files/web/<source_id>/docs.jsonl (+ raw/)
"""
import re
import sys
from urllib.parse import unquote

from bs4 import BeautifulSoup

from fetch_small_sites import ROOT, doc, get, html_text, write  # noqa: F401

SKIP = re.compile(r'/(wp-json|feed|wp-content|wp-includes|xmlrpc|tag|author|comments|wp-login)[/.]|replytocom|'
                  r'\.(css|js|png|jpe?g|gif|xml|ico|pdf|mp3|mp4|woff2?)(\?|$)', re.I)


def cdx(prefix, raw):
    rows = []
    for www in ('', 'www.'):
        r = get('https://web.archive.org/cdx/search/cdx', raw,
                params={'url': www + prefix + ('*' if prefix.endswith('/') else '/*'), 'output': 'json',
                        'filter': ['statuscode:200', 'mimetype:text/html'], 'fl': 'urlkey,timestamp,original,digest'})
        if r is not None and r.status_code == 200 and r.text.strip():
            rows += r.json()[1:]
    latest = {}
    for urlkey, ts, orig, digest in rows:
        if SKIP.search(orig):
            continue
        k = re.sub(r'^https?://(www\.)?', '', orig)
        k = re.sub(r'^([^/:]+):(?:80|443)(?=/|$)', lambda m: m.group(1), k).rstrip('/').lower()  # :80 = no port
        if k not in latest or ts > latest[k][0]:
            latest[k] = (ts, orig, digest)
    return latest


def main_content(soup):
    """The page's main text container: Drupal's system-main block, WordPress'
    entry/post content, then <article>, <main>, <body>. Sidebars (lists of other
    posts/videos) otherwise get extracted as if they were the page's text."""
    return (soup.find(id='block-system-main')
            or soup.find(class_=re.compile(r'^(entry-content|post-content|td-post-content|post-entry)$'))
            or soup.find('article') or soup.find('main') or soup.body)


def main(source_id, prefix, include=None, max_pages=5000, keys=None):
    raw = '%s\\%s\\raw' % (ROOT, source_id)
    latest = cdx(prefix, raw)
    if include:
        inc = re.compile(include)
        latest = {k: v for k, v in latest.items() if inc.search(unquote(v[1]))}
    if keys is not None:
        latest = {k: v for k, v in latest.items() if k in keys}
    print(source_id, 'distinct URLs with captures', len(latest))
    docs, seen = [], set()
    for k, (ts, orig, digest) in sorted(latest.items()):
        if digest in seen or len(docs) >= max_pages:
            continue
        seen.add(digest)
        wb = 'https://web.archive.org/web/%sid_/%s' % (ts, orig)
        p = get(wb, raw)
        if p is None or p.status_code != 200:
            continue
        soup = BeautifulSoup(p.text, 'lxml')
        art = main_content(soup)
        text = html_text(str(art)) if art else ''
        title = soup.title.get_text().strip() if soup.title else None
        docs.append(doc(source_id, orig, title, text, date=ts[:4] + '-' + ts[4:6] + '-' + ts[6:8],
                        license='unknown (site defunct; content (c) its authors)',
                        attribution='%s (via Internet Archive Wayback Machine)' % prefix.split('/')[0],
                        wayback=wb, capture=ts))
        if len(docs) % 25 == 0:
            print(source_id, len(docs), orig[:100], flush=True)
    write(source_id, docs)


def via_index(source_id, prefix, index_regex):
    """Fetch the archived index pages matching index_regex (e.g. a dialect
    category and its /page/N/), collect the article links on them, and recover
    only those articles (their latest captures) plus the index pages."""
    raw = '%s\\%s\\raw' % (ROOT, source_id)
    latest = cdx(prefix, raw)
    idx = re.compile(index_regex)
    host = prefix.split('/')[0]
    wanted = set()
    for k, (ts, orig, digest) in sorted(latest.items()):
        if not idx.search(unquote(orig)):
            continue
        p = get('https://web.archive.org/web/%sid_/%s' % (ts, orig), raw)
        if p is None or p.status_code != 200:
            continue
        for a in BeautifulSoup(p.text, 'lxml').find_all('a', href=True):
            href = a['href']
            if host in href and not SKIP.search(href) and '/category/' not in href and '/page/' not in href:
                wanted.add(re.sub(r'^https?://(www\.)?', '', href).rstrip('/').lower())
    keep = {k for k in latest if k in wanted}
    print(source_id, 'index pages linked', len(wanted), 'with captures', len(keep))
    main(source_id, prefix, keys=keep)


if __name__ == '__main__':
    if len(sys.argv) > 4 and sys.argv[3] == '--via-index':
        via_index(sys.argv[1], sys.argv[2], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
