"""Mirror the full Drive folder into raw_files/, preserving paths.

Design notes:
  * atomic: download to <dest>.part then os.replace, so an interrupted run
    never leaves a truncated file that looks complete
  * manifest: raw_files/_manifest.jsonl records every successful download;
    a file is skipped only if it is in the manifest AND still present with
    the recorded size (so partials/stale files get retried)
  * resumable + retrying: safe to kill and re-run
"""
import os, sys, json, time, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown

ROOT = r'F:\Hindko'
LISTING = os.path.join(ROOT, '_pipeline', 'drive_listing.json')
DEST_ROOT = os.path.join(ROOT, 'raw_files')
MANIFEST = os.path.join(DEST_ROOT, '_manifest.jsonl')
PROGRESS = os.path.join(ROOT, '_pipeline', 'download_progress.json')
FAILLOG = os.path.join(ROOT, '_pipeline', 'download_failures.jsonl')

WORKERS = 4
RETRIES = 4

_lock = threading.Lock()
_done = {}          # relpath -> size
_counter = {'ok': 0, 'skip': 0, 'fail': 0, 'bytes': 0}


def load_manifest():
    if not os.path.exists(MANIFEST):
        return
    with open(MANIFEST, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            _done[r['path']] = r['size']


def record(rec):
    with _lock:
        with open(MANIFEST, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def record_fail(rec):
    with _lock:
        with open(FAILLOG, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def progress():
    with _lock:
        snap = dict(_counter)
    snap['total'] = TOTAL
    snap['pct'] = round(100.0 * (snap['ok'] + snap['skip'] + snap['fail']) / max(TOTAL, 1), 2)
    snap['mb'] = round(snap['bytes'] / 1e6, 1)
    snap['ts'] = time.strftime('%H:%M:%S')
    tmp = PROGRESS + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(snap, fh)
    os.replace(tmp, PROGRESS)
    return snap


def work(entry):
    rel = entry['path']
    fid = entry['url'].split('id=')[-1]
    dest = os.path.join(DEST_ROOT, rel.replace('/', os.sep))
    part = dest + '.part'

    if rel in _done and os.path.exists(dest) and os.path.getsize(dest) == _done[rel]:
        with _lock:
            _counter['skip'] += 1
            _counter['bytes'] += _done[rel]
        return ('skip', rel, None)

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            gdown.download(id=fid, output=part, quiet=True, resume=True,
                           timeout=(30, 300), retries=0)
            if not os.path.exists(part):
                raise IOError('gdown produced no file')
            sz = os.path.getsize(part)
            if sz == 0:
                raise IOError('zero-byte download')
            os.replace(part, dest)
            record({'path': rel, 'id': fid, 'size': sz,
                    'ts': time.strftime('%Y-%m-%dT%H:%M:%S')})
            with _lock:
                _counter['ok'] += 1
                _counter['bytes'] += sz
            return ('ok', rel, sz)
        except Exception as e:                     # noqa: BLE001 - log & retry
            last = '%s: %s' % (type(e).__name__, e)
            try:
                if os.path.exists(part) and os.path.getsize(part) == 0:
                    os.remove(part)
            except OSError:
                pass
            time.sleep(min(2 ** attempt, 20) + (attempt * 0.5))

    with _lock:
        _counter['fail'] += 1
    record_fail({'path': rel, 'id': fid, 'error': last,
                 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')})
    return ('fail', rel, last)


entries = json.load(open(LISTING, encoding='utf-8'))
TOTAL = len(entries)

if __name__ == '__main__':
    os.makedirs(DEST_ROOT, exist_ok=True)
    load_manifest()
    print('entries=%d already-done=%d workers=%d' % (TOTAL, len(_done), WORKERS), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, e) for e in entries]
        n = 0
        for f in as_completed(futs):
            n += 1
            try:
                f.result()
            except Exception:
                traceback.print_exc()
            if n % 25 == 0 or n == TOTAL:
                s = progress()
                print('[%6.1fs] %d/%d ok=%d skip=%d fail=%d %.1fMB'
                      % (time.time() - t0, n, TOTAL, s['ok'], s['skip'], s['fail'], s['mb']),
                      flush=True)

    s = progress()
    print('FINAL', json.dumps(s), flush=True)
