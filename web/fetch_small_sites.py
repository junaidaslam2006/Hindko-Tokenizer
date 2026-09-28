"""Fetch the small Unicode-Hindko web sources found by the survey
(F:/Hindko/_web/survey/websites.md). Uses each site's API or feed; one
request at a time with a delay; generic User-Agent (no personal data).

  ghb          gandharahindko.com WordPress REST API: all posts. The API only;
               the survey saw signs of cloaked SEO spam on the site's HTML pages.
  aaprihindko  aaprihindko.com (dead domain), 2020-22 Wayback captures via CDX
  hlcs         hindko.org (Hindko Language & Culture Society), /hno pages via sitemap
  hindkomaza   hindkomaza.home.blog through the public WordPress.com API
  blogspot     hindkopoint.blogspot.com, hindko-pk.blogspot.com through Blogger JSON feeds
  incubator    Wikimedia Incubator Wp/hno, Wp/hnd, Wt/hnd through the MediaWiki API

    python fetch_small_sites.py ghb aaprihindko hlcs hindkomaza blogspot incubator

Output: F:/Hindko/raw_files/web/<source_id>/docs.jsonl (+ raw/ responses)
"""
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

ROOT = r'F:\Hindko\raw_files\web'
UA = 'HindkoCorpusBot/1.0 (research corpus of the Hindko language; polite crawler)'
DELAY = 1.5
S = requests.Session()
S.headers['User-Agent'] = UA
_robots = {}


def robots_ok(url):
    p = urlparse(url)
    host = p.scheme + '://' + p.netloc
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = S.get(host + '/robots.txt', timeout=30)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except Exception:
            rp.parse([])
        _robots[host] = rp
    return _robots[host].can_fetch(UA, url)


def fix_encoding(text):
    """Undo UTF-8 read as ISO-8859-1 (requests' default for text/* without a
    charset). Lossless when the text came straight from response.text; a
    genuine Latin-1 page does not decode as UTF-8 and is returned unchanged."""
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def set_encoding(r):
    """Decode as UTF-8 when the server names no charset and the bytes are UTF-8."""
    if 'charset' not in r.headers.get('content-type', '').lower():
        try:
            r.content.decode('utf-8')
            r.encoding = 'utf-8'
        except UnicodeDecodeError:
            r.encoding = r.apparent_encoding
    return r


class Cached:
    """A response replayed from raw/ (status 200 only was cached)."""
    def __init__(self, url, text):
        self.url, self.text, self.status_code = url, fix_encoding(text), 200
        self.headers = {'content-type': 'text/html; charset=utf-8'}

    def json(self):
        return json.loads(self.text)


def cache_path(raw_dir, url, kw):
    h = hashlib.sha1((url + json.dumps(kw.get('params') or {}, sort_keys=True)).encode()).hexdigest()
    return os.path.join(raw_dir, h + '.gz')


def get(url, raw_dir=None, **kw):
    if raw_dir:
        cp = cache_path(raw_dir, url, kw)
        if os.path.exists(cp):
            with gzip.open(cp, 'rt', encoding='utf-8') as g:
                body = g.read().split('\n', 2)
            if len(body) == 3:
                return Cached(url, body[2])
    if not robots_ok(url):
        print('robots.txt disallows', url)
        return None
    for attempt in range(4):
        try:
            r = S.get(url, timeout=90, **kw)
        except Exception as e:
            print('ERR', url, e)
            time.sleep(DELAY * 4)
            continue
        time.sleep(DELAY)
        set_encoding(r)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue
        if raw_dir and r.status_code == 200:
            os.makedirs(raw_dir, exist_ok=True)
            with gzip.open(cache_path(raw_dir, url, kw), 'wt', encoding='utf-8') as g:
                g.write(url + '\n' + json.dumps(kw.get('params') or {}) + '\n' + r.text)
        return r
    return None


def html_text(html):
    """Block-level text from HTML: one line per paragraph / verse line."""
    soup = BeautifulSoup(html or '', 'lxml')
    for t in soup(['script', 'style', 'noscript', 'iframe', 'form', 'nav', 'footer', 'header', 'aside']):
        t.decompose()
    for br in soup.find_all('br'):
        br.replace_with('\n')
    lines = []
    blocks = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'td', 'pre', 'div'])
    if not blocks:
        return soup.get_text('\n').strip()
    for b in blocks:
        if b.find(['p', 'div', 'li', 'td', 'blockquote', 'h1', 'h2', 'h3', 'h4']):
            continue            # keep leaves only, avoid duplicated nested text
        t = b.get_text('')
        for ln in t.split('\n'):
            ln = re.sub(r'[ \t\u00a0]+', ' ', ln).strip()
            if ln:
                lines.append(ln)
    return '\n'.join(lines)


def write(source_id, docs):
    d = os.path.join(ROOT, source_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'docs.jsonl'), 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(source_id, len(docs), 'docs', sum(len(x['text'].split()) for x in docs), 'words')


def doc(source_id, url, title, text, date=None, license=None, attribution=None, **meta):
    return {'doc_id': source_id + '_' + hashlib.sha1(url.encode()).hexdigest()[:12], 'source_id': source_id,
            'url': url, 'title': title, 'text': text, 'date': date, 'license': license,
            'attribution': attribution, 'register': 'web', 'meta': meta}


# ---------------------------------------------------------------- sources

def ghb():
    sid, raw = 'web_gandharahindko', os.path.join(ROOT, 'web_gandharahindko', 'raw')
    base = 'https://gandharahindko.com/wp-json/wp/v2/'
    cats = {}
    page = 1
    while True:
        r = get(base + 'categories', raw, params={'per_page': 100, 'page': page})
        if r is None or r.status_code != 200 or not r.json():
            break
        cats.update({c['id']: c['name'] for c in r.json()})
        page += 1
    docs, page = [], 1
    while True:
        r = get(base + 'posts', raw, params={'per_page': 100, 'page': page})
        if r is None or r.status_code != 200:
            break
        posts = r.json()
        if not posts:
            break
        for p in posts:
            text = html_text(p['content']['rendered'])
            title = BeautifulSoup(p['title']['rendered'], 'lxml').get_text().strip()
            docs.append(doc(sid, p['link'], title, text, date=p.get('date', '')[:10],
                            license='all rights reserved (Gandhara Hindko Board)',
                            attribution='Gandhara Hindko Board / Academy, gandharahindko.com',
                            wp_id=p['id'], categories=[cats.get(c, c) for c in p.get('categories', [])]))
        total_pages = int(r.headers.get('X-WP-TotalPages', page))
        print('ghb posts page', page, '/', total_pages)
        if page >= total_pages:
            break
        page += 1
    write(sid, docs)


def wp_site(sid, base, attribution, license='unknown (site content (c) its publisher)'):
    """All posts of a live WordPress site through its REST API."""
    raw = os.path.join(ROOT, sid, 'raw')
    docs, page = [], 1
    while True:
        r = get(base.rstrip('/') + '/wp-json/wp/v2/posts', raw, params={'per_page': 100, 'page': page})
        if r is None or r.status_code != 200:
            break
        posts = r.json()
        if not posts:
            break
        for p in posts:
            docs.append(doc(sid, p['link'], BeautifulSoup(p['title']['rendered'], 'lxml').get_text().strip(),
                            html_text(p['content']['rendered']), date=p.get('date', '')[:10],
                            license=license, attribution=attribution, wp_id=p['id']))
        total_pages = int(r.headers.get('X-WP-TotalPages', page))
        print(sid, 'posts page', page, '/', total_pages, flush=True)
        if page >= total_pages:
            break
        page += 1
    write(sid, docs)


def mansehra():
    wp_site('web_mansehra_com', 'https://mansehra.com', 'mansehra.com')


def aaprihindko():
    sid, raw = 'web_aaprihindko', os.path.join(ROOT, 'web_aaprihindko', 'raw')
    r = get('https://web.archive.org/cdx/search/cdx', raw,
            params={'url': 'aaprihindko.com/*', 'from': '2020', 'to': '2022', 'output': 'json',
                    'filter': ['statuscode:200', 'mimetype:text/html'], 'collapse': 'urlkey'})
    rows = r.json()[1:] if r is not None and r.status_code == 200 and r.text.strip() else []
    # keep the LATEST 2020-22 capture of each URL (collapse=urlkey keeps the first; re-query per key is costly,
    # so take the first capture, which predates the 2024 domain-squatter spam either way)
    skip = re.compile(r'/(wp-json|feed|wp-content|wp-includes|xmlrpc|tag|author|page/\d+|comments)/|\?|\.(css|js|png|jpe?g|xml)$', re.I)
    docs, seen = [], set()
    for urlkey, ts, orig, mime, status, digest, length in rows:
        if skip.search(orig) or digest in seen:
            continue
        seen.add(digest)
        wb = 'https://web.archive.org/web/%sid_/%s' % (ts, orig)
        p = get(wb, raw)
        if p is None or p.status_code != 200:
            continue
        soup = BeautifulSoup(p.text, 'lxml')
        art = soup.find('article') or soup.find(class_=re.compile('entry-content|post-content')) or soup.body
        text = html_text(str(art))
        title = (soup.title.get_text().strip() if soup.title else None)
        docs.append(doc(sid, orig, title, text, date=ts[:4] + '-' + ts[4:6] + '-' + ts[6:8],
                        license='unknown (site defunct; content (c) its authors)',
                        attribution='aaprihindko.com (via Internet Archive Wayback Machine)',
                        wayback=wb, capture=ts))
        print('aapri', len(docs), orig[:90])
    write(sid, docs)


def hlcs():
    sid, raw = 'web_hindko_org', os.path.join(ROOT, 'web_hindko_org', 'raw')
    r = get('https://hindko.org/sitemap.xml', raw)
    locs = re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', r.text) if r is not None else []
    locs = [u for u in locs if '/hno' in u] or ['https://hindko.org/hno']
    docs = []
    for u in sorted(set(locs)):
        p = get(u, raw)
        if p is None or p.status_code != 200 or 'html' not in p.headers.get('content-type', ''):
            continue
        soup = BeautifulSoup(p.text, 'lxml')
        main = soup.find('main') or soup.find(id='content') or soup.find(class_=re.compile('region-content|node__content')) or soup.body
        docs.append(doc(sid, u, soup.title.get_text().strip() if soup.title else None, html_text(str(main)),
                        license='unknown (sponsored by Hindko Language & Culture Society; no licence stated)',
                        attribution='Hindko Language & Culture Society (HLCS), Abbottabad, hindko.org'))
    write(sid, docs)


def hindkomaza():
    sid, raw = 'web_hindkomaza', os.path.join(ROOT, 'web_hindkomaza', 'raw')
    docs = []
    for kind in ('posts', 'pages'):
        r = get('https://public-api.wordpress.com/wp/v2/sites/hindkomaza.home.blog/' + kind, raw,
                params={'per_page': 100})
        for p in (r.json() if r is not None and r.status_code == 200 else []):
            docs.append(doc(sid, p['link'], BeautifulSoup(p['title']['rendered'], 'lxml').get_text().strip(),
                            html_text(p['content']['rendered']), date=p.get('date', '')[:10],
                            license='unknown (blog; (c) author)', attribution='hindkomaza.home.blog', kind=kind))
    write(sid, docs)


def blogspot():
    sid, raw = 'web_hindko_blogs', os.path.join(ROOT, 'web_hindko_blogs', 'raw')
    docs = []
    for blog in ('hindkopoint.blogspot.com', 'hindko-pk.blogspot.com'):
        for kind in ('posts', 'pages'):
            r = get('https://%s/feeds/%s/default' % (blog, kind), raw, params={'alt': 'json', 'max-results': 500})
            if r is None or r.status_code != 200:
                continue
            for e in r.json().get('feed', {}).get('entry', []):
                link = next((l['href'] for l in e.get('link', []) if l.get('rel') == 'alternate'), None) or e['id']['$t']
                docs.append(doc(sid, link, e.get('title', {}).get('$t'), html_text(e.get('content', {}).get('$t', '')),
                                date=e.get('published', {}).get('$t', '')[:10],
                                license='unknown (blog; (c) author)', attribution=blog, blog=blog, kind=kind))
    write(sid, docs)


def incubator():
    sid, raw = 'web_wikimedia_incubator', os.path.join(ROOT, 'web_wikimedia_incubator', 'raw')
    api = 'https://incubator.wikimedia.org/w/api.php'
    docs = []
    for prefix in ('Wp/hno', 'Wp/hnd', 'Wt/hnd', 'Wt/hno'):
        r = get(api, raw, params={'action': 'query', 'list': 'allpages', 'apprefix': prefix, 'aplimit': 500,
                                  'apfilterredir': 'nonredirects', 'format': 'json'})
        for pg in (r.json().get('query', {}).get('allpages', []) if r is not None else []):
            p = get(api, raw, params={'action': 'parse', 'pageid': pg['pageid'], 'prop': 'text',
                                      'format': 'json', 'formatversion': 2})
            if p is None or 'parse' not in p.json():
                continue
            html = p.json()['parse']['text']
            soup = BeautifulSoup(html, 'lxml')
            for t in soup.select('.mw-editsection, .reference, .reflist, .navbox, .toc, table.metadata'):
                t.decompose()
            docs.append(doc(sid, 'https://incubator.wikimedia.org/wiki/' + pg['title'].replace(' ', '_'),
                            pg['title'], html_text(str(soup)), license='CC-BY-SA-4.0',
                            attribution='Wikimedia Incubator contributors', prefix=prefix))
    write(sid, docs)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[name]()
