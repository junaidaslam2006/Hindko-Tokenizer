"""Scan FineWeb-2 (96 Common Crawl snapshots, GlotLID-labelled) for Hindko.

GlotLID has no Hindko label, so Hindko pages were filed under the nearest
labels: mostly pnb_Arab (Western Punjabi), some under urd_Arab, and the
low-confidence ones under *_removed. We stream those parquet files (text
columns only, nothing is written to disk but the hits), apply a cheap marker
prefilter, then the Hindko-vs-neighbours classifier (hp.langid.Model).

    python fw2_scan.py pnb        # pnb_Arab train/test + pnb_Arab_removed, skr, kas (everything classified)
    python fw2_scan.py urd        # urd_Arab train/test + urd_Arab_removed (prefiltered)

Output: F:/Hindko/_web/fw2/hits_<config>_<file>.jsonl (P(hindko) >= 0.3), and
        F:/Hindko/_web/fw2/done.json (resumable, per file)
"""
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault('HF_HOME', r'F:\Hindko\_web\hf_home')

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

sys.path.insert(0, r'F:\Hindko\_pipeline')
sys.path.insert(0, r'F:\Hindko\_web')

OUT = r'F:\Hindko\_web\fw2'
REPO = 'datasets/HuggingFaceFW/fineweb-2/data'
SETS = {
    'pnb': [('pnb_Arab', 'train'), ('pnb_Arab', 'test'), ('pnb_Arab_removed', 'train'),
            ('skr_Arab', 'train'), ('skr_Arab', 'test'), ('kas_Arab', 'train'), ('kas_Arab', 'test')],
    'urd': [('urd_Arab', 'test'), ('urd_Arab', 'train'), ('urd_Arab_removed', 'train')],
}
COLS = ['text', 'id', 'dump', 'url', 'date', 'language', 'language_score']
KEEP_P = 0.3

_model = None


def _init():
    global _model, lang
    from hp import lang as _lang
    from hp.langid import Model
    lang = _lang
    _model = Model()


def work(args):
    rows, prefilter = args
    cand = []
    n_pre = 0
    for r in rows:
        t = r['text'] or ''
        if prefilter:
            h, u = lang.marker_counts(t[:20000])
            if h < 3 or h < 0.5 * u:
                continue
        n_pre += 1
        cand.append(r)
    hits = []
    if cand:
        P = _model.proba([r['text'][:6000] for r in cand])
        for r, p in zip(cand, P):
            if p['hindko'] >= KEEP_P:
                r = dict(r)
                r['p'] = {k: round(v, 4) for k, v in p.items()}
                hits.append(r)
    return len(rows), n_pre, hits


def batches(fs, path, bs=2000):
    with fs.open(path, 'rb', block_size=16 * 2 ** 20) as f:
        pf = pq.ParquetFile(f)
        cols = [c for c in COLS if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=bs, columns=cols):
            yield b.to_pylist()


def main(which):
    os.makedirs(OUT, exist_ok=True)
    done_path = os.path.join(OUT, 'done.json')
    done = json.load(open(done_path)) if os.path.exists(done_path) else {}
    fs = HfFileSystem()
    prefilter = which == 'urd'
    with mp.Pool(7, initializer=_init) as pool:
        for cfg, split in SETS[which]:
            try:
                files = sorted(x['name'] for x in fs.ls('%s/%s/%s' % (REPO, cfg, split), detail=True)
                               if x['name'].endswith('.parquet'))
            except FileNotFoundError:
                print('missing', cfg, split)
                continue
            for path in files:
                key = path.split('/data/')[1]
                if key in done:
                    continue
                t0 = time.time()
                out_path = os.path.join(OUT, 'hits_%s.jsonl' % key.replace('/', '__').replace('.parquet', ''))
                n = n_pre = n_hit = 0
                with open(out_path, 'w', encoding='utf-8') as out:
                    for n_rows, np_, hits in pool.imap(work, ((rows, prefilter) for rows in batches(fs, path))):
                        n += n_rows
                        n_pre += np_
                        n_hit += len(hits)
                        for h in hits:
                            h['fw2_config'] = cfg
                            h['fw2_file'] = key
                            out.write(json.dumps(h, ensure_ascii=False) + '\n')
                        if n % 100000 < 2000:
                            print('%s rows %d prefilter %d hits %d  %.0fs' % (key, n, n_pre, n_hit, time.time() - t0),
                                  flush=True)
                done[key] = {'rows': n, 'prefiltered': n_pre, 'hits': n_hit, 'seconds': round(time.time() - t0)}
                json.dump(done, open(done_path, 'w'), indent=1)
                print('DONE', key, done[key], flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
