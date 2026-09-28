"""Polite site crawler for Hindko web sources.

- obeys robots.txt (urllib.robotparser) and its Crawl-delay; default 1.5 s/request
- stays on the seed's host (plus explicitly allowed hosts)
- URL frontier from seeds + sitemaps + in-page links, filtered by include/exclude regexes
- stores raw HTML (gzip) and trafilatura-extracted main text, one JSONL line per page
- resumable: already-fetched URLs (in pages.jsonl) are skipped

    python crawl.py SITE_ID --seed https://example.pk/ [--sitemap URL] [--include REGEX]
                    [--exclude REGEX] [--max-pages N] [--delay S]

Output: F:/Hindko/raw_files/web/<SITE_ID>/pages.jsonl and html/<sha1>.html.gz
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import time
import urllib.robotparser
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

ROOT = r'F:\Hindko\raw_files\web'
UA = 'HindkoCorpusBot/1.0 (research corpus of the Hindko language; polite crawler)'
SKIP_EXT = re.compile(r'\.(jpe?g|png|gif|webp|svg|ico|css|js|mp3|mp4|avi|mov|wav|zip|rar|7z|exe|apk|woff2?|ttf)(\?|$)', re.I)
AR_RE = re.compile(r'[\u0600-\u06FF]')


def sha1(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def canon(url):
    url, _ = urldefrag(url)
    return url.rstrip('/') if urlparse(url).path not in ('', '/') else url


class Crawler:
    def __init__(self, site, seeds, sitemaps=(), include=None, exclude=None, max_pages=100000,
                 delay=1.5, hosts=(), keep_pdf=False):
        self.site = site
        self.dir = os.path.join(ROOT, site)
        os.makedirs(os.path.join(self.dir, 'html'), exist_ok=True)
        self.pages_path = os.path.join(self.dir, 'pages.jsonl')
        self.hosts = {urlparse(s).netloc for s in seeds} | set(hosts)
        self.include = re.compile(include) if include else None
        self.exclude = re.compile(exclude) if exclude else None
        self.max_pages = max_pages
        self.delay = delay
        self.keep_pdf = keep_pdf
        self.sess = requests.Session()
        self.sess.headers['User-Agent'] = UA
        self.robots = {}
        self.seen = set()
        self.done = set()
        if os.path.exists(self.pages_path):
            for l in open(self.pages_path, encoding='utf-8'):
                try:
                    self.done.add(json.loads(l)['url'])
                except Exception:
                    pass
        self.queue = deque()
        for s in seeds:
            self.push(s)
        for sm in sitemaps:
            self.load_sitemap(sm)

    def robot(self, url):
        host = urlparse(url).scheme + '://' + urlparse(url).netloc
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.sess.get(host + '/robots.txt', timeout=30)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            self.robots[host] = rp
            cd = rp.crawl_delay(UA) or rp.crawl_delay('*')
            if cd:
                self.delay = max(self.delay, float(cd))
        return self.robots[host]

    def allowed(self, url):
        p = urlparse(url)
        if p.scheme not in ('http', 'https') or p.netloc not in self.hosts:
            return False
        if SKIP_EXT.search(p.path) or (p.path.lower().endswith('.pdf') and not self.keep_pdf):
            return False
        if self.exclude and self.exclude.search(url):
            return False
        return self.robot(url).can_fetch(UA, url)

    def push(self, url, front=False):
        url = canon(url)
        if url in self.seen:
            return
        self.seen.add(url)
        if url in self.done or not self.allowed(url):
            return
        (self.queue.appendleft if front else self.queue.append)(url)

    def load_sitemap(self, url, depth=0):
        try:
            r = self.sess.get(url, timeout=60)
            time.sleep(self.delay)
        except Exception as e:
            print('sitemap error', url, e)
            return
        locs = re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', r.text)
        if '<sitemapindex' in r.text and depth < 3:
            for loc in locs:
                self.load_sitemap(loc, depth + 1)
        else:
            for loc in locs:
                if not self.include or self.include.search(loc):
                    self.push(loc)
        print('sitemap', url, len(locs), 'queue', len(self.queue))

    def run(self):
        n = 0
        with open(self.pages_path, 'a', encoding='utf-8') as out:
            while self.queue and n < self.max_pages:
                url = self.queue.popleft()
                t0 = time.time()
                try:
                    r = self.sess.get(url, timeout=60)
                except Exception as e:
                    print('ERR', url, e)
                    time.sleep(self.delay)
                    continue
                n += 1
                ctype = r.headers.get('content-type', '')
                if 'charset' not in ctype.lower():
                    # no declared charset: requests would decode as ISO-8859-1
                    try:
                        r.content.decode('utf-8')
                        r.encoding = 'utf-8'
                    except UnicodeDecodeError:
                        r.encoding = r.apparent_encoding
                rec = {'url': url, 'final_url': r.url, 'status': r.status_code,
                       'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       'content_type': ctype, 'site': self.site}
                if r.status_code == 200 and 'pdf' in ctype and self.keep_pdf:
                    pth = os.path.join(self.dir, 'pdf', sha1(url) + '.pdf')
                    os.makedirs(os.path.dirname(pth), exist_ok=True)
                    open(pth, 'wb').write(r.content)
                    rec['pdf'] = os.path.relpath(pth, self.dir)
                elif r.status_code == 200 and 'html' in ctype:
                    html = r.text
                    h = sha1(url)
                    with gzip.open(os.path.join(self.dir, 'html', h + '.html.gz'), 'wt', encoding='utf-8') as g:
                        g.write(html)
                    rec['html'] = 'html/%s.html.gz' % h
                    meta = trafilatura.extract_metadata(html)
                    rec['title'] = meta.title if meta else None
                    rec['date'] = meta.date if meta else None
                    rec['text'] = trafilatura.extract(html, include_comments=False, include_tables=True,
                                                      favor_recall=True, url=url) or ''
                    rec['n_ar_chars'] = len(AR_RE.findall(rec['text']))
                    soup = BeautifulSoup(html, 'lxml')
                    for a in soup.find_all('a', href=True):
                        link = urljoin(r.url, a['href'])
                        if not self.include or self.include.search(link):
                            self.push(link)
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                if n % 25 == 0:
                    print(self.site, 'fetched', n, 'queue', len(self.queue), 'last', url[:100])
                time.sleep(max(0.0, self.delay - (time.time() - t0)))
        print(self.site, 'DONE fetched', n, 'queue left', len(self.queue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('site')
    ap.add_argument('--seed', action='append', required=True)
    ap.add_argument('--sitemap', action='append', default=[])
    ap.add_argument('--host', action='append', default=[])
    ap.add_argument('--include')
    ap.add_argument('--exclude')
    ap.add_argument('--max-pages', type=int, default=100000)
    ap.add_argument('--delay', type=float, default=1.5)
    ap.add_argument('--keep-pdf', action='store_true')
    a = ap.parse_args()
    Crawler(a.site, a.seed, a.sitemap, a.include, a.exclude, a.max_pages, a.delay, a.host, a.keep_pdf).run()


if __name__ == '__main__':
    main()
