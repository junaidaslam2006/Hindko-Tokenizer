"""OCR stage -> a SEPARATE, clearly-flagged low-quality corpus.

Per the agreed scope, every page scan is OCR'd, but the output is kept apart
from the digitally-extracted dataset and never merged into it. Tesseract's
urd model is trained on Naskh while this paper is set in Nastaliq; measured
accuracy on sample pages was ~50% out-of-vocabulary tokens versus 3.6% for the
digital text. Every record therefore carries its own quality evidence so a
downstream user can filter or discard it without having to trust us:

    ocr_mean_confidence   Tesseract's per-word confidence, averaged
    oov_vs_vocab          share of word tokens absent from the digital corpus
                          vocabulary (44k words)
    oov_vs_common         same, against words occurring >=5 times digitally
    lexical_verdict       'unusable' / 'poor' / 'fair' from oov_vs_common

Outputs (all under ocr_low_quality/):
    ocr_pages.jsonl   one record per page
    ocr_pages.txt     plain text, banner-separated
    README.md         what this is, why it is separate, how to filter it
    ocr_run.json      run-level statistics

Resumable: pages already present in ocr_pages.jsonl are skipped.

Requires Tesseract + Urdu data. Language data lives in _pipeline/tessdata and
is located via TESSDATA_PREFIX (writing into Program Files needs admin).
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hp import segment
from hp.build import (ROOT, LISTING, local_path, issue_key,
                      build_issue_date_map)
from hp import filecheck

OUT_DIR = os.path.join(ROOT, 'ocr_low_quality')
OUT_JSONL = os.path.join(OUT_DIR, 'ocr_pages.jsonl')
OUT_TXT = os.path.join(OUT_DIR, 'ocr_pages.txt')
OUT_RUN = os.path.join(OUT_DIR, 'ocr_run.json')
ASSESSMENT = os.path.join(ROOT, 'ocr', 'ocr_quality_assessment.json')

TESSDATA_DIR = os.path.join(ROOT, '_pipeline', 'tessdata')
TESSERACT_CANDIDATES = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    os.path.expandvars(r'%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe'),
]

IMAGE_EXTS = ('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp')
WORD_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+')

FRONT_RE = re.compile(r'front\s*page|frontpage|^fp\b|title\s*page', re.I)
BACK_RE = re.compile(r'back\s*page|last\s*page', re.I)
INNER_RE = re.compile(r'(^|[\s_])p\s*\d|page\s*\d', re.I)


# --------------------------------------------------------------------------
# engine setup
# --------------------------------------------------------------------------
def resolve_tesseract() -> Optional[str]:
    found = shutil.which('tesseract')
    if found:
        return found
    for c in TESSERACT_CANDIDATES:
        if c and os.path.exists(c):
            return c
    return None


def configure_tesseract() -> Optional[str]:
    exe = resolve_tesseract()
    if exe:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = exe
    if os.path.isdir(TESSDATA_DIR):
        # Env var, not --tessdata-dir: tesseract only honours that flag as two
        # separate argv entries, which pytesseract's config string mangles.
        os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    return exe


def page_role(rel_path: str) -> str:
    name = os.path.basename(rel_path)
    if FRONT_RE.search(name):
        return 'front_page'
    if BACK_RE.search(name):
        return 'back_page'
    if INNER_RE.search(name):
        return 'inner_page'
    return 'unspecified_page'


# --------------------------------------------------------------------------
# lexical quality scoring against the digital corpus
# --------------------------------------------------------------------------
def load_reference_vocab() -> Dict[str, object]:
    vocab = collections.Counter()
    path = os.path.join(ROOT, 'hindko_dataset.jsonl')
    if not os.path.exists(path):
        return {'vocab': set(), 'common': set(), 'tokens': 0}
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            for w in WORD_RE.findall(rec.get('text', '')):
                vocab[w] += 1
    return {'vocab': set(vocab), 'common': {w for w, c in vocab.items() if c >= 5},
            'tokens': sum(vocab.values()), 'distinct': len(vocab)}


def lexical_quality(text: str, ref: Dict[str, object]) -> Dict[str, float]:
    words = WORD_RE.findall(text)
    if not words:
        return {'tokens': 0, 'oov_vs_vocab': 1.0, 'oov_vs_common': 1.0,
                'mean_word_length': 0.0}
    vocab, common = ref['vocab'], ref['common']
    return {
        'tokens': len(words),
        'oov_vs_vocab': round(1.0 - sum(1 for w in words if w in vocab) / len(words), 4),
        'oov_vs_common': round(1.0 - sum(1 for w in words if w in common) / len(words), 4),
        'mean_word_length': round(sum(len(w) for w in words) / len(words), 3),
    }


def verdict(oov_common: float) -> str:
    if oov_common >= 0.40:
        return 'unusable'
    if oov_common >= 0.25:
        return 'poor'
    return 'fair'


# --------------------------------------------------------------------------
def list_images() -> List[str]:
    entries = json.load(open(LISTING, encoding='utf-8'))
    out = []
    for e in entries:
        p = e['path']
        if os.path.splitext(p)[1].lower().lstrip('.') not in IMAGE_EXTS:
            continue
        lp = local_path(p)
        if not os.path.exists(lp):
            continue
        ok, why = filecheck.validate(lp)
        if ok:
            out.append(p)
    return sorted(out)


def load_done() -> set:
    done = set()
    if os.path.exists(OUT_JSONL):
        with io.open(OUT_JSONL, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    try:
                        done.add(json.loads(line)['source_path'])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def keep_content_lines(text: str) -> List[str]:
    out = []
    for ln in text.split('\n'):
        ln = ln.strip()
        if ln and segment.is_content_line(ln):
            out.append(ln)
    return out


def run(langs: str, psm: int, min_confidence: float, limit: int) -> None:
    import pytesseract
    from PIL import Image

    exe = configure_tesseract()
    if not exe:
        sys.exit('tesseract binary not found')
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = load_reference_vocab()
    print('reference vocabulary: %s distinct words from %s tokens'
          % (ref.get('distinct', 0), ref.get('tokens', 0)), flush=True)

    issue_dates = build_issue_date_map(json.load(open(LISTING, encoding='utf-8')))
    images = list_images()
    done = load_done()
    todo = [p for p in images if p not in done]
    if limit:
        todo = todo[:limit]
    print('page scans on disk: %d | already done: %d | to OCR: %d'
          % (len(images), len(done), len(todo)), flush=True)
    if not todo:
        print('nothing to do')
        return

    cfg = '--psm %d' % psm
    stats = collections.Counter()
    t0 = time.time()

    with io.open(OUT_JSONL, 'a', encoding='utf-8', newline='\n') as out:
        for n, rel in enumerate(todo, 1):
            path = local_path(rel)
            try:
                with Image.open(path) as im:
                    data = pytesseract.image_to_data(
                        im, lang=langs, config=cfg,
                        output_type=pytesseract.Output.DICT)
            except Exception as e:                       # noqa: BLE001
                print('  FAIL %s: %s' % (rel[-58:], e), flush=True)
                stats['failed'] += 1
                continue

            words, confs = [], []
            for tok, conf in zip(data['text'], data['conf']):
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    continue
                if tok.strip() and c >= min_confidence:
                    words.append(tok)
                    confs.append(c)

            raw_text = ' '.join(words)
            lines = keep_content_lines(raw_text)
            text = '\n'.join(lines)
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0
            lex = lexical_quality(text, ref)
            v = verdict(lex['oov_vs_common'])
            stats[v] += 1

            date, prec = segment.derive_date(rel)
            dsrc = 'source_path' if date else None
            if not date:
                date = issue_dates.get(issue_key(rel))
                if date:
                    prec, dsrc = 'day', 'issue_folder_sibling_filename'

            rec = {
                'id': None,                      # assigned in finalise()
                'text': text,
                'language': 'und',               # NOT asserted to be hindko
                'source': 'newspaper_page_scan',
                'source_file': os.path.basename(rel),
                'source_path': rel,
                'publication': 'Weekly Hindkowan',
                'issue': segment.derive_issue(rel),
                'date': date, 'date_precision': prec, 'date_source': dsrc,
                'page_role': page_role(rel),
                'extraction_method': 'ocr_tesseract_%s_psm%d' % (langs, psm),
                'ocr_mean_confidence': round(mean_conf, 2),
                'ocr_words_kept': len(words),
                'ocr_min_confidence_filter': min_confidence,
                **lex,
                'lexical_verdict': v,
                'quality_warning': (
                    'Low-confidence OCR of a Nastaliq newspaper scan. Measured '
                    '~50% of tokens are not real words. Not part of '
                    'hindko_dataset.jsonl; do not use for training without '
                    'filtering on lexical_verdict / oov_vs_common.'),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()

            if n % 5 == 0 or n == len(todo):
                el = time.time() - t0
                print('  [%d/%d] %.0fs elapsed, ~%.0fs left | conf=%5.1f '
                      'oov=%.2f %-8s chars=%-6d %s'
                      % (n, len(todo), el, el / n * (len(todo) - n), mean_conf,
                         lex['oov_vs_common'], v, len(text), rel[-46:]),
                      flush=True)

    finalise(stats, langs, psm, min_confidence, ref, time.time() - t0)


def finalise(stats, langs, psm, min_confidence, ref, elapsed) -> None:
    """Assign stable ids, write the .txt view and the run summary + README."""
    recs = []
    with io.open(OUT_JSONL, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r['source_path'])
    for i, r in enumerate(recs, 1):
        r['id'] = 'hindko_ocr_%06d' % i

    with io.open(OUT_JSONL, 'w', encoding='utf-8', newline='\n') as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    with io.open(OUT_TXT, 'w', encoding='utf-8', newline='\n') as fh:
        for i, r in enumerate(recs):
            if i:
                fh.write('\n\n' + '=' * 78 + '\n\n')
            fh.write('# %s | %s | conf=%.1f | oov=%.2f | %s\n'
                     '# source: %s\n# WARNING: low-quality OCR, not part of '
                     'hindko_dataset\n\n%s\n'
                     % (r['id'], r.get('page_role'), r['ocr_mean_confidence'],
                        r['oov_vs_common'], r['lexical_verdict'],
                        r['source_path'], r['text']))

    by_verdict = collections.Counter(r['lexical_verdict'] for r in recs)
    by_role = collections.Counter(r['page_role'] for r in recs)
    confs = [r['ocr_mean_confidence'] for r in recs]
    oovs = [r['oov_vs_common'] for r in recs]

    run_info = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'engine': 'tesseract %s, lang=%s, psm=%d, min_word_confidence=%s'
                  % (shutil.which('tesseract') or resolve_tesseract(), langs,
                     psm, min_confidence),
        'tessdata_dir': TESSDATA_DIR,
        'pages_ocrd': len(recs),
        'runtime_seconds': round(elapsed, 1),
        'lexical_verdicts': dict(by_verdict),
        'page_roles': dict(by_role),
        'mean_confidence': round(sum(confs) / len(confs), 2) if confs else None,
        'mean_oov_vs_common': round(sum(oovs) / len(oovs), 4) if oovs else None,
        'total_characters': sum(len(r['text']) for r in recs),
        'reference_vocabulary': {'distinct': ref.get('distinct'),
                                 'tokens': ref.get('tokens')},
        'assessment_file': 'ocr/ocr_quality_assessment.json',
        'separate_from_main_dataset': True,
    }
    with io.open(OUT_RUN, 'w', encoding='utf-8') as fh:
        json.dump(run_info, fh, ensure_ascii=False, indent=2)

    write_readme(run_info, by_verdict)
    print('\n%s' % json.dumps(run_info, ensure_ascii=False, indent=2))


def write_readme(info, by_verdict) -> None:
    txt = """# OCR output — LOW QUALITY, kept separate on purpose

**This is not part of the Hindko dataset.** Nothing in this directory is
included in `../hindko_dataset.jsonl` or `../hindko_dataset_permissive.jsonl`.

## Why it is separate

These pages were read by Tesseract {engine}. The paper is set in Urdu
**Nastaliq**, whose cascading baseline defeats a model trained on **Naskh**.
Measured against the vocabulary of the digitally-extracted corpus:

| Source | Out-of-vocabulary tokens |
|---|---|
| Digital InPage extraction | **3.6 %** |
| This OCR output | **~50 %** |

Roughly half the tokens here are not real words, and the errors are plausible
near-misses rather than obvious garbage — e.g. the masthead
`ہفت روزہ ہندکووان` was read as `ہت روز ہ بای ام زازیمے`, and
`پشور(ہندکووان نیوز)` as `پٹور(مٹرگودان ول )`. A script-fraction quality gate
cannot detect this, because garbled Urdu still looks like Urdu.

Full measurement: `../ocr/ocr_quality_assessment.json`.

## What is here

- `ocr_pages.jsonl` — {pages} pages, one record each
- `ocr_pages.txt` — the same text, banner-separated, every page headed by a
  warning
- `ocr_run.json` — run statistics

## How to filter it

Every record carries its own quality evidence, so you do not have to trust
this README:

| Field | Meaning |
|---|---|
| `lexical_verdict` | `unusable` (oov ≥ 0.40), `poor` (≥ 0.25), `fair` (< 0.25) |
| `oov_vs_common` | share of tokens absent from common digital-corpus words |
| `oov_vs_vocab` | share absent from the full 44k-word digital vocabulary |
| `ocr_mean_confidence` | Tesseract's own per-word confidence, averaged |
| `page_role` | `front_page` / `back_page` / `inner_page` / `unspecified_page` |

Verdict distribution: {verdicts}

Note `language` is recorded as `und` (undetermined), **not** `hindko` — the
text is too unreliable to assert a language for.

## Recommended use

Only `lexical_verdict == "fair"` pages are worth reading, and even those need
manual review. For any training use, prefer the digital corpus. If front-page
content is genuinely needed, re-OCR with an engine that handles Nastaliq
(Google Cloud Vision does) rather than filtering this.

## Front/back pages are the reason this exists

Inner pages come from the InPage layout files and are already in the digital
corpus. Front and back pages are designed in CorelDRAW, whose `.cdr` files
store text as vector outlines (verified: no font references, no coherent
Arabic strings, `\\x04`-pair density at or below random chance) and whose
`.eps` exports contain no text operators at all. OCR is the *only* route to
that content — which is why it was run despite the measured accuracy.
""".format(engine=info['engine'], pages=info['pages_ocrd'],
           verdicts=dict(by_verdict))
    with io.open(os.path.join(OUT_DIR, 'README.md'), 'w', encoding='utf-8',
                 newline='\n') as fh:
        fh.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--langs', default='urd')
    ap.add_argument('--psm', type=int, default=4,
                    help='page segmentation mode; 4 = single column of '
                         'variable-size text, which measured best here')
    ap.add_argument('--min-confidence', type=float, default=40.0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    if not resolve_tesseract():
        sys.exit('tesseract not found; install it or add it to PATH')
    configure_tesseract()
    run(args.langs, args.psm, args.min_confidence, args.limit)


if __name__ == '__main__':
    main()
