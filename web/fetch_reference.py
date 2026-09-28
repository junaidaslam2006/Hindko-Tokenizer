"""Fetch reference text for neighbouring languages (not Hindko) from
FineWeb-2 via the HF datasets-server rows API. Used only to learn a
Hindko-vs-Punjabi/Saraiki/Pahari discriminator; never enters the corpus.

    python fetch_reference.py pnb_Arab 3000
"""
import json
import os
import sys
import time

import requests

API = 'https://datasets-server.huggingface.co/rows'
OUT = r'F:\Hindko\_web\ref'


def fetch(config, n, split='train'):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, '%s.jsonl' % config)
    size = requests.get('https://datasets-server.huggingface.co/size',
                        params={'dataset': 'HuggingFaceFW/fineweb-2'}, timeout=60)
    total = None
    for c in size.json().get('size', {}).get('splits', []):
        if c['config'] == config and c['split'] == split:
            total = c['num_rows']
    if total is None:
        sys.exit('no such config/split: %s/%s' % (config, split))
    pages = max(1, n // 100)
    step = max(100, total // pages)
    got = 0
    with open(path, 'w', encoding='utf-8') as f:
        for off in range(0, total, step):
            for attempt in range(5):
                r = requests.get(API, params={'dataset': 'HuggingFaceFW/fineweb-2', 'config': config,
                                              'split': split, 'offset': off, 'length': 100}, timeout=120)
                if r.status_code == 200:
                    break
                time.sleep(5 * (attempt + 1))
            else:
                print('giving up at offset', off, r.status_code)
                continue
            for row in r.json()['rows']:
                d = row['row']
                f.write(json.dumps({'text': d['text'], 'url': d.get('url'),
                                    'lang_score': d.get('language_score')}, ensure_ascii=False) + '\n')
                got += 1
            if got >= n:
                break
    print(config, 'total rows', total, 'fetched', got, '->', path)


if __name__ == '__main__':
    fetch(sys.argv[1], int(sys.argv[2]), *(sys.argv[3:4]))
