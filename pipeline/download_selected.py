"""Download a selected subset of the Drive mirror into raw_files/.

Prioritises text-bearing files (.inp/.B01) so the corpus can be built long
before the large image/artwork transfers finish.

  --exts inp,b01,jpg,jpeg     which extensions to fetch (default: text only)
  --workers N                 concurrency (kept low: Google throttles hard)
  --limit N                   stop after N files (debug)

Shares raw_files/_manifest.jsonl with download_all.py, is atomic
(<dest>.part -> rename) and fully resumable.

Rate-limit handling: gdown raises FileURLRetrievalError when Google blocks
anonymous access for the IP. When that happens we back off hard (and grow the
pause on repeat offences) instead of hammering, because hammering is what
extends the block.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hp import filecheck

ROOT = r'F:\Hindko'
LISTING = os.path.join(ROOT, '_pipeline', 'drive_listing.json')
DEST_ROOT = os.path.join(ROOT, 'raw_files')
MANIFEST = os.path.join(DEST_ROOT, '_manifest.jsonl')
PROGRESS = os.path.join(ROOT, '_pipeline', 'download_progress.json')
FAILLOG = os.path.join(ROOT, '_pipeline', 'download_failures.jsonl')
# Files that downloaded "successfully" but fail container-integrity checks.
# Several issue-262 scans are zero-filled at the source, so retrying them only
# wastes rate-limit quota; they are recorded here and skipped on later runs.
CORRUPTLOG = os.path.join(ROOT, '_pipeline', 'corrupt_at_source.jsonl')
QUARANTINE = os.path.join(DEST_ROOT, '_quarantine')
# Optional Netscape-format cookies for drive.google.com. Anonymous downloads
# are quota-capped per IP (we observed ~50-70 files then a 300s block);
# authenticated requests are not, so this is the single biggest speed lever.
# The file is only ever handed to gdown - its contents are never read or
# printed by this pipeline.
COOKIES_FILE = os.path.join(ROOT, '_pipeline', 'cookies.txt')

# lower number = fetched earlier.
# Text-bearing sources first (.inp then .B01), then the two unknown small files
# (.zip may contain text; .tmp is a CorelDRAW scratch file worth checking),
# then the page scans needed for OCR, then the ~9.8 GB of .cdr/.eps artwork
# which is archival only - proven to contain no extractable text.
EXT_PRIORITY = {'inp': 0, 'b01': 1, 'zip': 1, 'tmp': 1,
                'jpg': 2, 'jpeg': 2, 'png': 2, 'tif': 2, 'tiff': 2,
                'cdr': 3, 'eps': 3}

RATE_LIMIT_MARKER = 'FileURLRetrievalError'

_lock = threading.Lock()
_done: dict = {}
_corrupt: set = set()
_state = {'ok': 0, 'skip': 0, 'fail': 0, 'bytes': 0, 'throttled': 0,
          'corrupt': 0, 'integrity_fail': 0}
_pause_until = [0.0]        # shared rate-limit cooldown
_pause_len = [15.0]         # grows on repeat throttling, decays on success
_streak = [0]               # consecutive successes since the last throttle
PAUSE_FLOOR = 15.0
PAUSE_CEIL = 300.0
DECAY_AFTER = 25            # successes needed to halve the cooldown


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


def load_corrupt():
    if not os.path.exists(CORRUPTLOG):
        return
    with open(CORRUPTLOG, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                _corrupt.add(json.loads(line)['path'])
            except (json.JSONDecodeError, KeyError):
                pass


def record_corrupt(rec):
    with _lock:
        _corrupt.add(rec['path'])
        with open(CORRUPTLOG, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def note_throttle():
    """Push the shared cooldown out and make the next one longer.

    The cooldown is a single shared deadline, not a per-worker penalty: when
    several threads hit the block at once they must not each add their own
    pause on top, or the wait compounds past the ceiling (observed as 599s
    with a 300s cap).
    """
    with _lock:
        _state['throttled'] += 1
        _streak[0] = 0
        now = time.time()
        _pause_len[0] = min(_pause_len[0] * 2, PAUSE_CEIL)
        _pause_until[0] = max(_pause_until[0], now + _pause_len[0])
        until = _pause_until[0]
    return until


def note_success():
    """Decay the cooldown once a run of downloads has gone cleanly.

    Without this the pause ratchets to the ceiling on the first few throttles
    and then never recovers, so a long job ends up paying 5 minutes per
    hiccup even when Google has long since forgiven us.
    """
    with _lock:
        _streak[0] += 1
        if _streak[0] >= DECAY_AFTER and _pause_len[0] > PAUSE_FLOOR:
            _pause_len[0] = max(PAUSE_FLOOR, _pause_len[0] / 2.0)
            _streak[0] = 0


def wait_if_throttled():
    while True:
        with _lock:
            remain = _pause_until[0] - time.time()
        if remain <= 0:
            return
        time.sleep(min(remain, 5.0))


def write_progress(total):
    with _lock:
        snap = dict(_state)
        snap['pause_len'] = round(_pause_len[0], 1)
        snap['cooldown_remaining'] = round(max(0.0, _pause_until[0] - time.time()), 1)
    snap['total'] = total
    finished = snap['ok'] + snap['skip'] + snap['fail']
    snap['pct'] = round(100.0 * finished / max(total, 1), 2)
    snap['mb'] = round(snap['bytes'] / 1e6, 1)
    snap['ts'] = time.strftime('%H:%M:%S')
    tmp = PROGRESS + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(snap, fh)
    os.replace(tmp, PROGRESS)
    return snap


def quarantine(src, rel, reason):
    """Move an untrustworthy file out of raw_files/ so it cannot enter the
    pipeline, while keeping it as evidence."""
    dest = os.path.join(QUARANTINE, rel.replace('/', os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.replace(src, dest)
    except OSError:
        try:
            os.remove(src)
        except OSError:
            pass
    record_corrupt({'path': rel, 'reason': reason,
                    'quarantined_to': os.path.relpath(dest, ROOT).replace(os.sep, '/'),
                    'ts': time.strftime('%Y-%m-%dT%H:%M:%S')})
    with _lock:
        _state['corrupt'] += 1


# Integrity failures that mean "the Drive object itself is bad" rather than
# "the transfer broke". Retrying these only burns rate-limit quota.
SOURCE_CORRUPT_REASONS = {'all_zero_content', 'zero_bytes'}


def work(entry, retries=5, retry_corrupt=False):
    rel = entry['path']
    fid = entry['url'].split('id=')[-1]
    dest = os.path.join(DEST_ROOT, rel.replace('/', os.sep))
    part = dest + '.part'

    if rel in _corrupt and not retry_corrupt:
        with _lock:
            _state['skip'] += 1
        return 'corrupt_known'

    if rel in _done and os.path.exists(dest) and os.path.getsize(dest) == _done[rel]:
        # Trust but verify: some files already in the manifest are zero-filled.
        ok, why = filecheck.validate(dest)
        if ok:
            with _lock:
                _state['skip'] += 1
                _state['bytes'] += _done[rel]
            return 'skip'
        print('  quarantining previously-accepted %s (%s)' % (rel[-56:], why),
              flush=True)
        quarantine(dest, rel, why)
        return 'corrupt'

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        wait_if_throttled()
        try:
            kw = {}
            if os.path.exists(COOKIES_FILE):
                kw['cookies_file'] = COOKIES_FILE
            gdown.download(id=fid, output=part, quiet=True, resume=True,
                           timeout=(30, 600), retries=0, **kw)
            if not os.path.exists(part):
                raise IOError('gdown produced no file')
            sz = os.path.getsize(part)
            if sz == 0:
                raise IOError('zero-byte download')

            ok, why = filecheck.validate(part, ext=os.path.splitext(dest)[1])
            if not ok:
                if why in SOURCE_CORRUPT_REASONS:
                    # The object is bad at the source; retrying is pointless.
                    print('  CORRUPT AT SOURCE %s (%s, %d bytes) -> quarantine'
                          % (rel[-56:], why, sz), flush=True)
                    quarantine(part, rel, why)
                    return 'corrupt'
                with _lock:
                    _state['integrity_fail'] += 1
                try:
                    os.remove(part)
                except OSError:
                    pass
                raise IOError('integrity check failed: %s' % why)

            os.replace(part, dest)
            record({'path': rel, 'id': fid, 'size': sz,
                    'ts': time.strftime('%Y-%m-%dT%H:%M:%S')})
            with _lock:
                _state['ok'] += 1
                _state['bytes'] += sz
            note_success()
            return 'ok'
        except Exception as e:                     # noqa: BLE001
            last = '%s: %s' % (type(e).__name__, e)
            try:
                if os.path.exists(part) and os.path.getsize(part) == 0:
                    os.remove(part)
            except OSError:
                pass
            if RATE_LIMIT_MARKER in last:
                until = note_throttle()
                print('  throttled; cooling down %.0fs (until %s)'
                      % (max(0, until - time.time()),
                         time.strftime('%H:%M:%S', time.localtime(until))),
                      flush=True)
            else:
                time.sleep(min(2 ** attempt, 20))

    with _lock:
        _state['fail'] += 1
    record_fail({'path': rel, 'id': fid, 'error': last,
                 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')})
    return 'fail'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exts', default='inp,b01')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--listing', default=LISTING,
                    help='Drive listing JSON to draw entries from (default: '
                         'the original newspaper listing)')
    ap.add_argument('--retry-corrupt', action='store_true',
                    help='re-attempt files previously recorded as corrupt at '
                         'source (useful once, to confirm they are still bad)')
    args = ap.parse_args()

    want = {e.strip().lower().lstrip('.') for e in args.exts.split(',') if e.strip()}
    entries = json.load(open(args.listing, encoding='utf-8-sig'))
    sel = [e for e in entries
           if os.path.splitext(e['path'])[1].lower().lstrip('.') in want]
    sel.sort(key=lambda e: (EXT_PRIORITY.get(
        os.path.splitext(e['path'])[1].lower().lstrip('.'), 9), e['path']))
    if args.limit:
        sel = sel[:args.limit]

    os.makedirs(DEST_ROOT, exist_ok=True)
    load_manifest()
    load_corrupt()
    total = len(sel)
    authed = os.path.exists(COOKIES_FILE)
    print('selected=%d exts=%s workers=%d already-done=%d known-corrupt=%d auth=%s'
          % (total, ','.join(sorted(want)), args.workers, len(_done),
             len(_corrupt), 'cookies' if authed else 'anonymous (quota-capped)'),
          flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, e, 5, args.retry_corrupt) for e in sel]
        n = 0
        for f in as_completed(futs):
            n += 1
            try:
                f.result()
            except Exception:
                traceback.print_exc()
            if n % 20 == 0 or n == total:
                s = write_progress(total)
                print('[%6.1fs] %d/%d ok=%d skip=%d fail=%d throttle=%d %.1fMB'
                      % (time.time() - t0, n, total, s['ok'], s['skip'],
                         s['fail'], s['throttled'], s['mb']), flush=True)

    s = write_progress(total)
    print('FINAL %s' % json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
