"""End-to-end build: extract -> segment -> clean -> dedup -> gate -> emit.

Reads the Drive inventory (_pipeline/drive_listing.json), processes every
file present under raw_files/, and writes:

    raw_extracted/            full decoded text per source file + index
    cleaned/articles.jsonl    every segmented article with metrics
    rejected/                 rejected text + rejected.jsonl with reasons
    hindko_dataset.jsonl/.txt              strict, research-grade
    hindko_dataset_permissive.jsonl/.txt   strict + flagged-usable
    processing_report.json

Idempotent: safe to re-run; outputs are rebuilt from scratch each time.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hp import inpage, segment, clean as cl, docx, lang, books, web

ROOT = r'F:\Hindko'
# Listing of the whole Drive root. Since 2026-09-24 the root holds two trees:
# 'Hindkowan Newspaper Data/' (the original 1,865 files, unchanged) and
# 'Hindko Books/' (404 files: the Gandhara Hindko Academy 'Farma' series). The
# original newspaper-only listing is kept as drive_listing.json for provenance.
LISTING = os.path.join(ROOT, '_pipeline', 'drive_listing_full.json')
if not os.path.exists(LISTING):
    LISTING = os.path.join(ROOT, '_pipeline', 'drive_listing.json')
DRIVE_ROOT_URL = os.environ.get('HINDKO_DRIVE_ROOT_URL', '')  # source-material Drive folder (not published)
BOOKS_PREFIX = 'Hindko Books/'
# Plain-text exports supplied alongside the book .inp files (one per .inp).
BOOKS_TXT_PREFIX = 'Hindko Books/HINDKO BOOKS DATA/TXT/'


def source_kind(rel_path: str) -> str:
    return 'book' if rel_path.startswith(BOOKS_PREFIX) else 'newspaper'


# Book sources go through hp/books.py (book segmentation, revision resolution,
# line-level dedup, imprint metadata) - never through the newspaper segmenter,
# which would mangle them. False parks every book file (the build then
# reproduces the newspaper-only corpus), which is how the 2026-09-24 guard
# kept releases safe while books.py was being written and reviewed.
BOOKS_INTEGRATED = True
# Web sources (raw_files/web/<source_id>/, collected by F:/Hindko/_web/*.py;
# survey and provenance in _web/survey/). Processed by hp/web.py.
WEB_INTEGRATED = True
RAW_FILES = os.path.join(ROOT, 'raw_files')
OCR_JSONL = os.path.join(ROOT, 'ocr', 'ocr_text.jsonl')
# Outputs. --out redirects them to a staging directory so a new build can be
# checked before it replaces the released files (inputs always come from ROOT).
OUT_ROOT = ROOT
RAW_EXTRACTED = os.path.join(OUT_ROOT, 'raw_extracted')
CLEANED = os.path.join(OUT_ROOT, 'cleaned')
REJECTED = os.path.join(OUT_ROOT, 'rejected')


def set_out_root(path: str) -> None:
    global OUT_ROOT, RAW_EXTRACTED, CLEANED, REJECTED
    OUT_ROOT = os.path.abspath(path)
    RAW_EXTRACTED = os.path.join(OUT_ROOT, 'raw_extracted')
    CLEANED = os.path.join(OUT_ROOT, 'cleaned')
    REJECTED = os.path.join(OUT_ROOT, 'rejected')

PUBLICATION = 'Weekly Hindkowan'
LANGUAGE = 'hindko'
SOURCE = 'newspaper'

INVALID_FS = re.compile(r'[<>:"|?*\x00-\x1f]')


def safe_name(rel_path: str) -> str:
    return INVALID_FS.sub('_', rel_path.replace('/', '__').replace('\\', '__'))


def local_path(rel_path: str) -> str:
    return os.path.join(RAW_FILES, rel_path.replace('/', os.sep))


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def discover(include_all_backups: bool = False) -> dict:
    entries = json.load(open(LISTING, encoding='utf-8-sig'))
    by_ext = collections.Counter()
    by_kind_ext = collections.Counter()
    text_sources: List[str] = []
    skipped_backup: List[str] = []
    orphan_backup: List[str] = []
    redundant_backup: List[str] = []
    ocr_candidates: List[str] = []
    no_text_value: List[str] = []
    txt_exports: List[str] = []
    office_owner_files: List[str] = []

    inp_keys = set()
    for e in entries:
        p = e['path']
        ext = os.path.splitext(p)[1].lower().lstrip('.')
        by_ext[ext] += 1
        if ext == 'inp':
            key = os.path.dirname(p) + '|' + os.path.splitext(os.path.basename(p))[0].lower().strip()
            inp_keys.add(key)

    books_pending: List[str] = []
    for e in entries:
        p = e['path']
        ext = os.path.splitext(p)[1].lower().lstrip('.')
        by_kind_ext[(source_kind(p), ext)] += 1
        if not BOOKS_INTEGRATED and source_kind(p) == 'book':
            books_pending.append(p)
            continue
        if ext == 'txt' and p.startswith(BOOKS_TXT_PREFIX):
            # Derived exports of the .inp files, not independent sources. Used
            # to cross-validate our decode (see books.compare_txt_exports),
            # never ingested as a second copy of the same text.
            txt_exports.append(p)
        elif ext == 'docx':
            if os.path.basename(p).startswith('~$'):
                office_owner_files.append(p)
            else:
                text_sources.append(p)
        elif ext == 'inp':
            text_sources.append(p)
        elif ext == 'b01':
            key = os.path.dirname(p) + '|' + os.path.splitext(os.path.basename(p))[0].lower().strip()
            if key in inp_keys:
                if include_all_backups:
                    # older revision of a file we already have: convert it
                    # anyway and let the deduplicator absorb the overlap
                    redundant_backup.append(p)
                    text_sources.append(p)
                else:
                    skipped_backup.append(p)
            else:
                orphan_backup.append(p)
                text_sources.append(p)
        elif ext in ('jpg', 'jpeg', 'png', 'tif', 'tiff'):
            ocr_candidates.append(p)
        else:
            no_text_value.append(p)

    text_sources.sort()
    return {
        'entries': entries,
        'total': len(entries),
        'by_ext': by_ext,
        'by_kind_ext': by_kind_ext,
        'txt_exports': txt_exports,
        'office_owner_files': office_owner_files,
        'books_pending_integration': books_pending,
        'text_sources': text_sources,
        'skipped_backup': skipped_backup,
        'orphan_backup': orphan_backup,
        'redundant_backup': redundant_backup,
        'ocr_candidates': ocr_candidates,
        'no_text_value': no_text_value,
    }


def resolve_backup_duplicates(inv: dict) -> Dict[str, str]:
    """For .B01 files that have a same-stem .inp, compare content hashes.

    Identical  -> skip (true duplicate)
    Different  -> process it too (it is not a redundant copy)
    Returns {b01_path: 'identical'|'different'|'missing'}.
    """
    # case-insensitive stem -> .inp Drive path, per directory
    inp_by_dir: Dict[str, str] = {}
    for e in inv['entries']:
        p = e['path']
        if p.lower().endswith('.inp'):
            d = os.path.dirname(p)
            stem = os.path.splitext(os.path.basename(p))[0].strip().lower()
            inp_by_dir[(d, stem)] = p

    verdicts: Dict[str, str] = {}
    for b01 in inv['skipped_backup']:
        d = os.path.dirname(b01)
        stem = os.path.splitext(os.path.basename(b01))[0].strip().lower()
        inp_path = inp_by_dir.get((d, stem))
        if not inp_path:
            verdicts[b01] = 'missing'
            continue
        lp_b, lp_i = local_path(b01), local_path(inp_path)
        if not (os.path.exists(lp_b) and os.path.exists(lp_i)):
            verdicts[b01] = 'missing'
            continue
        verdicts[b01] = ('identical' if sha256_file(lp_b) == sha256_file(lp_i)
                         else 'different')
    return verdicts


# --------------------------------------------------------------------------
# stage 1: extraction
# --------------------------------------------------------------------------
def stage_extract(inv: dict, backup_verdicts: Dict[str, str], log) -> dict:
    os.makedirs(RAW_EXTRACTED, exist_ok=True)
    index_path = os.path.join(RAW_EXTRACTED, '_extraction_index.jsonl')

    # .B01 proven to be byte-identical to their .inp are not re-extracted
    sources = list(inv['text_sources'])
    results, failures, missing = [], [], []

    with open(index_path, 'w', encoding='utf-8') as idx:
        for rel in sources:
            lp = local_path(rel)
            kind = os.path.splitext(rel)[1].lower().lstrip('.')
            rec = {
                'source_path': rel,
                'source_file': os.path.basename(rel),
                'source_kind': source_kind(rel),
                'kind': kind,
                'extraction_method': ('docx_xml' if kind == 'docx'
                                      else 'inpage_ole_glyph_decode'),
            }
            if not os.path.exists(lp):
                rec['status'] = 'missing_on_disk'
                missing.append(rel)
                idx.write(json.dumps(rec, ensure_ascii=False) + '\n')
                continue
            try:
                if kind == 'docx':
                    info = extract_docx(lp)
                else:
                    info = inpage.extract_file(lp)
            except Exception as e:                      # noqa: BLE001
                rec['status'] = 'extraction_failed'
                rec['error'] = '%s: %s' % (type(e).__name__, e)
                rec['size_bytes'] = os.path.getsize(lp)
                failures.append(rec)
                idx.write(json.dumps(rec, ensure_ascii=False) + '\n')
                continue

            out_txt = os.path.join(RAW_EXTRACTED, safe_name(rel) + '.txt')
            with open(out_txt, 'w', encoding='utf-8') as fh:
                fh.write(info['text_all'])

            rec.update({
                'status': 'ok',
                'size_bytes': os.path.getsize(lp),
                'raw_text_file': os.path.relpath(out_txt, OUT_ROOT).replace(os.sep, '/'),
                'stream': info['stream'],
                'payload_bytes': info['payload_bytes'],
                'n_lines_all': info['n_lines_all'],
                'n_lines_content': info['n_lines_content'],
                'n_noise_lines': info['n_noise_lines'],
                'unmapped_glyphs': info['unmapped_glyphs'],
                'sha256': sha256_file(lp),
            })
            results.append((rel, info, rec))
            idx.write(json.dumps(rec, ensure_ascii=False) + '\n')

    log('extracted %d files, %d failed, %d missing on disk'
        % (len(results), len(failures), len(missing)))
    return {'results': results, 'failures': failures, 'missing': missing,
            'backup_verdicts': backup_verdicts}


def extract_docx(path: str) -> dict:
    """Same result shape as inpage.extract_file(), for .docx sources."""
    text = docx.extract_text(path)
    lines = inpage.split_lines(text)
    return {
        'stream': 'word/document.xml', 'streams': ['word/document.xml'],
        'payload_bytes': os.path.getsize(path),
        'text': '\n'.join(lines), 'text_all': '\n'.join(lines),
        'n_lines_all': len(lines), 'n_lines_content': len(lines),
        'n_noise_lines': 0, 'unmapped_glyphs': 0,
        'unmapped_in_text': text.count(inpage.UNMAPPED),
    }


def issue_key(rel_path: str) -> str:
    """Issue-level grouping key (first three path components)."""
    return '/'.join(rel_path.replace('\\', '/').split('/')[:3])


def build_issue_date_map(entries: List[dict]) -> Dict[str, str]:
    """Publication date per issue folder, from unambiguous full dates in
    sibling filenames (e.g. 'P2 final 18.1.jpg' -> 2023-01-18).

    Only assigned when exactly ONE distinct full date appears anywhere in that
    issue folder. Anything ambiguous stays null - dates are never guessed.
    """
    groups: Dict[str, set] = collections.defaultdict(set)
    for e in entries:
        d, prec = segment.derive_date(e['path'])
        if prec == 'day':
            groups[issue_key(e['path'])].add(d)
    return {k: next(iter(v)) for k, v in groups.items() if len(v) == 1}


# --------------------------------------------------------------------------
# stage 2-3: segment, fit boilerplate, clean
# --------------------------------------------------------------------------
def score_language(text: str) -> tuple:
    """(hindko_score, language_variety) with the whole-token scorer.

    The original substring scorer (segment.hindko_score) counted markers
    inside longer words - e.g. Urdu 'اب ' inside کتاب (93% of its hits), Hindko
    'وتا' inside Urdu ہوتا (100%) - which left 32% of book windows 'mixed'.
    lang.hindko_score_v2 matches whole tokens with markers validated on
    held-out folders; on 192 hand-labelled lines it makes 5 wrong-direction
    calls vs 15 for the old scorer (probe_books_markers.py). Used for BOTH
    sources so the corpus has one definition of language_variety.
    """
    return (round(lang.hindko_score_v2(text), 4), lang.language_variety_v2(text))


def stage_segment_and_clean(results: list, issue_dates: Dict[str, str], log) -> dict:
    """Newspaper sources only (books go through hp/books.py)."""
    docs_lines: List[List[str]] = []
    prelim: List[dict] = []
    dropped_frames = 0
    mastheads_removed = 0

    for rel, info, rec in results:
        raw_lines = [ln for ln in inpage.split_lines(info['text_all'])
                     if not inpage.is_noise(ln)]
        lines = [ln for ln in raw_lines
                 if segment.is_content_line(ln) and not cl.is_masthead(ln)]
        mastheads_removed += sum(1 for ln in raw_lines
                                 if cl.is_masthead(ln)
                                 and segment.is_content_line(ln))
        dropped_frames += len(raw_lines) - len(lines)
        docs_lines.append(lines)
        arts = segment.segment_articles(lines)
        issue = segment.derive_issue(rel)
        date, prec = segment.derive_date(rel)
        date_source = 'source_path' if date else None
        if not date:
            date = issue_dates.get(issue_key(rel))
            if date:
                prec, date_source = 'day', 'issue_folder_sibling_filename'
        sec = segment.derive_section(rel)
        for i, a in enumerate(arts):
            prelim.append({
                'source_path': rel,
                'source_file': os.path.basename(rel),
                'issue': issue, 'date': date, 'date_precision': prec,
                'date_source': date_source,
                'section': sec, 'article_index': i + 1,
                'extraction_method': rec['extraction_method'],
                'raw_text': a['text'], 'title': a['title'],
                'title_source': a['title_source'], 'role': a['role'],
                'has_body': a['has_body'], 'n_frames': a['n_frames'],
                'lines': a['lines'],
            })

    log('segmented %d articles from %d newspaper documents '
        '(%d non-content frames dropped, %d mastheads removed)'
        % (len(prelim), len(results), dropped_frames, mastheads_removed))

    det = cl.BoilerplateDetector().fit(docs_lines)
    log('boilerplate lines detected: %d' % len(det.boiler))

    for a in prelim:
        joined = '\n'.join(a['lines'])
        stripped, n_removed = cl.strip_boilerplate(joined, det)
        a['boilerplate_lines_removed'] = n_removed
        a['text'], a['pii_masked'] = cl.mask_pii(cl.clean_text(stripped))
        a['unmapped_glyphs'] = a['text'].count(cl.REPLACEMENT)
        # A file named 'ads' is the paper's advertisement page (issue 345's
        # ads.B01 left a notice table of 4 private persons' names, fathers'
        # names and localities once its ID/amount columns were dropped).
        a['is_ad'] = (cl.looks_like_ad(a['text']) or
                      os.path.splitext(a['source_file'])[0].strip().lower() == 'ads')
        a['hindko_score'], a['language_variety'] = score_language(a['text'])
        a['source'] = 'newspaper'

    return {'articles': prelim, 'boilerplate': det,
            'dropped_frames': dropped_frames,
            'mastheads_removed': mastheads_removed}


# --------------------------------------------------------------------------
# stage 3b: books (hp/books.py)
# --------------------------------------------------------------------------
def stage_books(results: list, news_kept: List[dict], log) -> dict:
    """Segment, revision-resolve and line-dedup the book sources.

    Runs after the newspaper has been gated, so newspaper IDs are unaffected.
    Book text that also appears in kept newspaper records is NOT cut line by
    line (that left books such as 214th/129th as fragments); book records stay
    whole and carry 'also_in_newspaper(share=x.xx)', and the shared
    record-level MinHash below removes a book record that is a near-copy of an
    article. Book records then go through the same cleaning, language
    scoring, record-level dedup and gates as the paper.
    """
    # 'size' feeds the L13 test: a file is "no extractable text" only when its
    # decoded text is negligible against its payload + picture bytes (289th).
    files = [{'rel': rel, 'kind': rec['kind'], 'text_all': info['text_all'],
              'size': rec.get('size_bytes') or os.path.getsize(local_path(rel))}
             for rel, info, rec in results]
    method = {rel: rec['extraction_method'] for rel, info, rec in results}
    news_texts = [(record_key(a), a['text']) for a in news_kept]
    out = books.process_books(files, news_texts, log=log)

    recs = []
    for r in out['records']:
        a = dict(r)
        a['source'] = 'book'
        a['article_index'] = r['passage_index']
        a['extraction_method'] = method.get(r['source_path'], 'inpage_ole_glyph_decode')
        a['text'], a['pii_masked'] = cl.mask_pii(
            cl.clean_text(r.get('text') or '', keep_blank_lines=True))
        a['unmapped_glyphs'] = a['text'].count(cl.REPLACEMENT)
        # The newspaper advertisement test is not a gate for books (no ads
        # were observed in the book files); a positive result is only noted.
        a['is_ad'] = False
        flags = list(a.get('content_flags') or [])
        if a['text'] and cl.looks_like_ad(a['text']):
            flags.append('ad_like_phrasing')
        a['content_flags'] = flags
        a['hindko_score'], a['language_variety'] = score_language(a['text'])
        if a.get('language') == 'pothohari':
            a['language_variety'] = 'pothohari'
        elif a.get('language') == 'en':
            a['language_variety'] = 'english'
        # `date` for a book is the publication year of that edition as printed
        # in its imprint (label/value pair), never inferred from file dates.
        year = a.get('publication_year')
        a['date'] = (a.get('publication_date') or str(year)) if year else None
        a['date_precision'] = (a.get('publication_date_precision') or 'year') if year else None
        a['date_source'] = 'book_imprint' if year else None
        recs.append(a)
    recs.sort(key=candidate_order)

    os.makedirs(CLEANED, exist_ok=True)
    write_jsonl(os.path.join(CLEANED, 'book_line_dedup.jsonl'), out['line_dedup'])
    write_jsonl(os.path.join(CLEANED, 'book_imprints.jsonl'), out['imprints'])
    write_jsonl(os.path.join(CLEANED, 'book_files.jsonl'), out['per_file'])
    with open(os.path.join(CLEANED, 'book_meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(out['book_meta'], fh, ensure_ascii=False, indent=1)
    log('books: %d records from %d files (%d lines removed as copies)'
        % (len(recs), len(files), len(out['line_dedup'])))
    return {'records': recs, 'out': out}


# --------------------------------------------------------------------------
# stage 4-5: dedup + gate
# --------------------------------------------------------------------------
def candidate_order(a: dict) -> tuple:
    """Deterministic processing order for dedup (first copy wins).

    An .inp is the authoritative document and its same-stem .B01 an older
    auto-backup, so on a tie the .inp must come first; plain path order put
    'X.B01' before 'X.inp' and kept the backup (89 released newspaper records
    moved to their .B01 twin in the first staged build - QA 2026-09-25).
    """
    p = a['source_path']
    d, f = os.path.split(p)
    stem, ext = os.path.splitext(f)
    return (d, stem.lower().strip(), 1 if ext.lower() == '.b01' else 0, f,
            a.get('article_index', 0))


def record_key(a: dict) -> str:
    return '%s#%s' % (a['source_path'], a.get('article_index', 0))


# Book roles that are not running Hindko prose/verse. They are kept (demote,
# don't delete) but never enter the strict tier.
BOOK_ROLE_DEMOTIONS = {
    'lexicon': 'reference_work(lexicon)',
    'toc': 'table_of_contents',
    'english_text': 'english_text',
}
# Hard gate reasons a book passage may have only because it is short (a
# 3-line mahiya, a chaar-baita). The book segmenter already merges short
# units and filters residue structurally, so shortness alone demotes rather
# than rejects: "we never throw text away because it was merely inconvenient".
BOOK_SHORTNESS_REASONS = ('too_short', 'too_few_words')
# Line-flag shares above which a book record is not Hindko running text.
ARABIC_DOMINANT_SHARE = 0.5
PERSIAN_DOMINANT_SHARE = 0.5


def gate_record(a: dict, text: str) -> tuple:
    """Quality gate for one unique record. Returns (tier, reasons)."""
    role = a.get('role', 'article')
    reasons, metrics = cl.evaluate(
        text, role=role,
        unmapped=a.get('unmapped_glyphs', text.count(cl.REPLACEMENT)),
        is_ad=a.get('is_ad', False))
    a.update(metrics)
    t = cl.tier(reasons)
    is_book = a.get('source') == 'book'

    if is_book:
        if role == 'english_text':
            # English frames are script-less by definition; judge them on
            # everything except the script-fraction gate.
            reasons = [r for r in reasons
                       if r.split('(')[0] not in ('low_script_fraction',
                                                  'below_strict_script_fraction')]
            t = cl.tier(reasons)
        hard = [r for r in reasons if cl.tier([r]) == 'reject']
        if (t == 'reject' and hard and role in ('book_passage', 'front_matter')
                and all(r.split('(')[0] in BOOK_SHORTNESS_REASONS for r in hard)
                and text.strip()):
            t = 'permissive'
            reasons = reasons + ['short_unit']
        demotions = []
        if role in BOOK_ROLE_DEMOTIONS:
            demotions.append(BOOK_ROLE_DEMOTIONS[role])
        if (a.get('arabic_share') or 0) >= ARABIC_DOMINANT_SHARE:
            demotions.append('arabic_dominant_content(share=%.2f)' % a['arabic_share'])
        if (a.get('persian_share') or 0) >= PERSIAN_DOMINANT_SHARE:
            demotions.append('persian_dominant_content(share=%.2f)' % a['persian_share'])
        if a.get('superseded'):
            demotions.append('superseded_revision_unique_text')
        # The strict tier is Hindko. A record whose evidenced language is
        # another one (the 25th Farma is Pothohari; English frames are 'en') is
        # kept, flagged, in the permissive tier.
        if a.get('language') not in (None, 'hindko') and role != 'english_text':
            demotions.append('non_hindko_language(%s)' % a['language'])
        if demotions and t != 'reject':
            t = 'permissive'
            reasons = reasons + demotions

    if a.get('source') == 'web' and t != 'reject':
        # Web text: the strict tier is Hindko the classifier agrees with.
        # Everything else stays, flagged, in the permissive tier (hp/web.py).
        demotions = []
        variety = a.get('language_variety')
        if variety != 'hindko' and a.get('web_section_only'):
            # a non-Hindko page from a site where Hindko is only a section
            t = 'reject'
            reasons = reasons + ['non_hindko_page(%s)' % variety]
        elif variety not in ('hindko', 'urdu'):
            demotions.append('non_hindko_language(%s)' % variety)
        for f in a.get('content_flags') or []:
            if f.startswith(('langid_disagrees', 'urdu_blocks', 'dialect_unverified', 'listing_page')):
                demotions.append(f)
        if demotions and t != 'reject':
            t = 'permissive'
            reasons = reasons + demotions

    # Weekly Hindkowan also carries Urdu columns and poetry, and many "Hindko
    # books" are Urdu (conference papers, Urdu verse, dictionaries' glosses).
    # Demote Urdu-dominant items out of the strict Hindko corpus rather than
    # discarding them: they stay available, flagged, in the permissive dataset.
    if t == 'strict' and a.get('language_variety') == 'urdu':
        t = 'permissive'
        reasons = reasons + ['urdu_dominant_content(score=%.2f)'
                             % a.get('hindko_score', 0.0)]
    return t, reasons


def stage_dedup_and_gate(candidates: List[dict], deduper: 'cl.Deduper',
                         log, label: str) -> dict:
    """Exact + near-duplicate removal and dual-tier gating.

    `candidates` must already be in their final deterministic order. The same
    Deduper instance is shared across sources (newspaper first, then books),
    so a book passage that repeats a newspaper article is caught here too.
    """
    kept: List[dict] = []
    dupes: List[dict] = []
    rejected: List[dict] = []
    tier_counts = collections.Counter()

    n = deduper.precompute(a.get('text') or '' for a in candidates)
    log('%s: %d MinHash signatures precomputed in parallel' % (label, n))

    for a in candidates:
        text = a.get('text') or ''
        if not text.strip():
            rejected.append({**a, 'reject_reasons': ['empty_after_cleaning'],
                             'quality_tier': 'reject'})
            continue

        status, dup_of, sim = deduper.check(record_key(a), text)
        a['dedup_status'] = status
        a['duplicate_of'] = dup_of
        a['near_dup_similarity'] = sim

        if status != 'unique':
            dupes.append(a)
            continue

        t, reasons = gate_record(a, text)
        a['quality_flags'] = reasons
        a['quality_tier'] = t
        tier_counts[t] += 1
        if t == 'reject':
            a['reject_reasons'] = reasons
            rejected.append(a)
        else:
            kept.append(a)

    log('%s dedup: %d duplicates removed' % (label, len(dupes)))
    log('%s tiers: %s' % (label, dict(tier_counts)))
    return {'kept': kept, 'dupes': dupes, 'rejected': rejected,
            'tier_counts': tier_counts}


CONTAIN_N = 8                # word n-gram size
CONTAIN_THRESHOLD = 0.80     # share of a record's n-grams found in ONE other record
CONTAIN_MIN_GRAMS = 5        # shorter records carry too little evidence
CONTAIN_MAX_DF = 50          # n-grams in more records are boilerplate, not copies


def stage_containment(kept: List[dict], log) -> dict:
    """Flag and demote records that are mostly contained in ONE larger record.

    The record-level deduplicator removes exact copies and near-duplicates
    (Jaccard >= 0.85), so a record survives when the record containing it is
    larger - QA 2026-09-25 found ~900 strict records >= 80% contained in
    another (reprinted columns, book passages printed in the paper, revision
    fragments). Word 8-gram containment in a single other record at >= 0.80
    demotes the contained record (never the container) from strict to
    permissive with quality flag 'contained_in(<uid>, c=0.xx)'. Nothing is
    deleted. Ties between equally large records keep the earlier one.
    Exception (web intake 2026-09-26): when the container is a web page and the
    contained record is a newspaper/book record, the web page is the copy, so
    it is the one flagged 'contains_record(<uid>, c=0.xx)' and demoted.
    """
    import numpy as np
    grams = []
    for a in kept:
        toks = (a.get('text') or '').split()
        if len(toks) < CONTAIN_N:
            grams.append(np.zeros(0, dtype=np.int64))
            continue
        hs = {hash(' '.join(toks[i:i + CONTAIN_N]))
              for i in range(len(toks) - CONTAIN_N + 1)}
        grams.append(np.fromiter(hs, dtype=np.int64, count=len(hs)))
    sizes = np.array([len(g) for g in grams], dtype=np.int64)
    H = np.concatenate(grams) if grams else np.zeros(0, dtype=np.int64)
    R = np.repeat(np.arange(len(grams), dtype=np.int64), sizes)
    order = np.argsort(H, kind='stable')
    H, R = H[order], R[order]
    pair_counts = collections.Counter()
    if len(H):
        starts = np.flatnonzero(np.r_[True, H[1:] != H[:-1]])
        ends = np.r_[starts[1:], len(H)]
        df = ends - starts
        # df == 2: vectorised
        two = starts[df == 2]
        if len(two):
            a_, b_ = R[two], R[two + 1]
            lo, hi = np.minimum(a_, b_), np.maximum(a_, b_)
            codes, cnt = np.unique(lo * len(grams) + hi, return_counts=True)
            for c, k in zip(codes.tolist(), cnt.tolist()):
                pair_counts[(c // len(grams), c % len(grams))] += k
        for st, en in zip(starts[(df > 2) & (df <= CONTAIN_MAX_DF)].tolist(),
                          ends[(df > 2) & (df <= CONTAIN_MAX_DF)].tolist()):
            rs = sorted(set(R[st:en].tolist()))
            for x in range(len(rs)):
                for y in range(x + 1, len(rs)):
                    pair_counts[(rs[x], rs[y])] += 1
    best = {}                    # contained idx -> (container idx, containment)
    for (i, j), c in pair_counts.items():
        for small, big in ((i, j), (j, i)):
            if sizes[small] < CONTAIN_MIN_GRAMS:
                continue
            if sizes[big] < sizes[small] or (sizes[big] == sizes[small] and big > small):
                continue
            cont = c / sizes[small]
            if cont >= CONTAIN_THRESHOLD and cont > best.get(small, (None, 0.0))[1]:
                best[small] = (big, cont)
    demoted = collections.Counter()
    flagged = collections.Counter()
    for small, (big, cont) in best.items():
        a, b = kept[small], kept[big]
        if b.get('source') == 'web' and a.get('source') != 'web':
            # A web page that reprints a newspaper/book record (plus page text)
            # must not demote the original: the web copy is flagged instead.
            b['quality_flags'] = list(b.get('quality_flags') or []) + [
                'contains_record(%s, c=%.2f)' % (doc_uid(a), cont)]
            key = 'web>%s' % a.get('source')
            flagged[key] += 1
            if b.get('quality_tier') == 'strict':
                b['quality_tier'] = 'permissive'
                demoted[key] += 1
            continue
        flag = 'contained_in(%s, c=%.2f)' % (doc_uid(b), cont)
        a['quality_flags'] = list(a.get('quality_flags') or []) + [flag]
        key = '%s<-%s' % (a.get('source'), b.get('source'))
        flagged[key] += 1
        if a.get('quality_tier') == 'strict':
            a['quality_tier'] = 'permissive'
            demoted[key] += 1
    log('containment: %d records >= %.2f contained in one larger record; '
        '%d demoted from strict (%s)' % (len(best), CONTAIN_THRESHOLD,
                                         sum(demoted.values()), dict(demoted)))
    return {'flagged': dict(flagged), 'demoted_from_strict': dict(demoted),
            'n_gram': CONTAIN_N, 'threshold': CONTAIN_THRESHOLD,
            'min_grams': CONTAIN_MIN_GRAMS, 'max_df': CONTAIN_MAX_DF}


def merge_gated(*parts: dict) -> dict:
    out = {'kept': [], 'dupes': [], 'rejected': [],
           'tier_counts': collections.Counter()}
    for p in parts:
        for k in ('kept', 'dupes', 'rejected'):
            out[k].extend(p[k])
        out['tier_counts'].update(p['tier_counts'])
    return out


def doc_uid(a: dict) -> str:
    """Content-position identifier that survives rebuilds.

    `id` is a sequential index over the emitted records, so it shifts when any
    earlier record appears or disappears (e.g. after a decoder fix changes a
    gate outcome). `uid` depends only on where the record comes from - source
    path and position in that file - so downstream users can track records
    across releases.
    """
    key = '%s#%s' % (a['source_path'], a.get('article_index', 0))
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]


def make_record(a: dict, doc_id: str) -> dict:
    """Final JSONL schema: the user's required fields first, then extras.

    Newspaper records keep exactly the schema of the 2026-09-20 release (plus
    `uid`). Book records share the required fields and the quality fields and
    add their own evidence-backed metadata; a field is None when not evidenced.
    """
    source = a.get('source', 'newspaper')
    rec = {
        'id': doc_id,
        'text': a['text'],
        'language': a.get('language') or LANGUAGE,
        'source': source,
        'source_file': a['source_file'],
        'title': a.get('title'),
        'date': a.get('date'),
    }
    if source == 'web':
        rec.update({
            # --- additional metadata (web) ---
            'uid': doc_uid(a),
            'source_path': a['source_path'],
            'url': a.get('url'),
            'site': a.get('site'),
            'doc_id': a.get('doc_id'),
            'retrieval': a.get('retrieval'),
            'extraction_method': a.get('extraction_method'),
            'license': a.get('license'),
            'attribution': a.get('attribution'),
            'register': a.get('register'),
            'dialect_note': a.get('dialect_note'),
            'date_precision': a.get('date_precision'),
            'date_source': a.get('date_source'),
            'title_source': a.get('title_source'),
            'role': a.get('role'),
            'n_chars': a.get('n_chars', len(a['text'])),
            'n_words': a.get('n_words'),
            'script_fraction': a.get('script_fraction'),
            'language_variety': a.get('language_variety'),
            'hindko_score': a.get('hindko_score'),
            'langid_p': a.get('langid_p'),
            'langid_shares': a.get('langid_shares'),
            'quality_tier': a.get('quality_tier'),
            'quality_flags': a.get('quality_flags', []),
            'content_flags': a.get('content_flags', []),
            'boilerplate_lines_removed': a.get('boilerplate_lines_removed'),
            'web_meta': a.get('web_meta') or None,
            'pii_masked': a.get('pii_masked') or None,
        })
        return rec
    if source == 'book':
        rec.update({
            # --- additional, evidence-backed metadata (books) ---
            'uid': doc_uid(a),
            'source_path': a['source_path'],
            'date_precision': a.get('date_precision'),
            'date_source': a.get('date_source'),
            'book_folder': a.get('book_folder'),
            'book_title': a.get('book_title'),
            'book_title_source': a.get('book_title_source'),
            'book_series_number': a.get('book_series_number'),
            'book_series_number_source': a.get('book_series_number_source'),
            'author': a.get('author'),
            'author_role': a.get('author_role'),
            'author_source': a.get('author_source'),
            'publisher': a.get('publisher'),
            'publisher_source': a.get('publisher_source'),
            'publication_year': a.get('publication_year'),
            'publication_year_source': a.get('publication_year_source'),
            'edition': a.get('edition'),
            'edition_source': a.get('edition_source'),
            'first_edition_year': a.get('first_edition_year'),
            'first_edition_year_source': a.get('first_edition_year_source'),
            'isbn': a.get('isbn'),
            'isbn_source': a.get('isbn_source'),
            'gha_ref': a.get('gha_ref'),
            'gha_ref_source': a.get('gha_ref_source'),
            'imprint_conflict': a.get('imprint_conflict'),
            'language_source': a.get('language_source'),
            'genre': a.get('genre'),
            'genre_evidence': a.get('genre_evidence'),
            'form': a.get('form'),
            'extraction_method': a.get('extraction_method', 'inpage_ole_glyph_decode'),
            'passage_index': a.get('article_index'),
            'title_source': a.get('title_source'),
            'role': a.get('role'),
            'n_units': a.get('n_units'),
            'n_chars': a.get('n_chars', len(a['text'])),
            'n_words': a.get('n_words'),
            'script_fraction': a.get('script_fraction'),
            'language_variety': a.get('language_variety'),
            'hindko_score': a.get('hindko_score'),
            'arabic_share': a.get('arabic_share'),
            'quality_tier': a.get('quality_tier'),
            'quality_flags': a.get('quality_flags', []),
            'content_flags': a.get('content_flags', []),
            'pii_masked': a.get('pii_masked') or None,
        })
        return rec
    rec.update({
        # --- additional, evidence-backed metadata (newspaper) ---
        'uid': doc_uid(a),
        'publication': PUBLICATION,
        'source_path': a['source_path'],
        'issue': a.get('issue'),
        'date_precision': a.get('date_precision'),
        'date_source': a.get('date_source'),
        'section': a.get('section'),
        'extraction_method': a.get('extraction_method', 'inpage_ole_glyph_decode'),
        'article_index': a.get('article_index'),
        'title_source': a.get('title_source'),
        'role': a.get('role'),
        'n_chars': a.get('n_chars', len(a['text'])),
        'n_words': a.get('n_words'),
        'script_fraction': a.get('script_fraction'),
        'language_variety': a.get('language_variety'),
        'hindko_score': a.get('hindko_score'),
        'quality_tier': a.get('quality_tier'),
        'quality_flags': a.get('quality_flags', []),
        'pii_masked': a.get('pii_masked') or None,
    })
    return rec


# --------------------------------------------------------------------------
# stage 6: emit
# --------------------------------------------------------------------------
def write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_txt(path: str, records: List[dict]) -> None:
    sep = '\n\n' + ('=' * 78) + '\n\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for i, r in enumerate(records):
            if i:
                fh.write(sep)
            header = '# %s' % r['id']
            bits = [b for b in [r.get('issue') and 'issue %s' % r['issue'],
                                r.get('book_title'), r.get('date'),
                                r.get('title')] if b]
            if bits:
                header += ' | ' + ' | '.join(str(b) for b in bits)
            fh.write(header + '\n\n' + r['text'] + '\n')


def stage_emit(gated: dict, log) -> dict:
    kept = gated['kept']
    strict = [a for a in kept if a['quality_tier'] == 'strict']
    permissive_all = kept            # strict + flagged-usable

    def assign(docs):
        out = []
        for i, a in enumerate(docs, 1):
            out.append(make_record(a, 'hindko_%06d' % i))
        return out

    strict_recs = assign(strict)
    perm_recs = assign(permissive_all)

    write_jsonl(os.path.join(OUT_ROOT, 'hindko_dataset.jsonl'), strict_recs)
    write_txt(os.path.join(OUT_ROOT, 'hindko_dataset.txt'), strict_recs)
    write_jsonl(os.path.join(OUT_ROOT, 'hindko_dataset_permissive.jsonl'), perm_recs)
    write_txt(os.path.join(OUT_ROOT, 'hindko_dataset_permissive.txt'), perm_recs)

    os.makedirs(CLEANED, exist_ok=True)
    write_jsonl(os.path.join(CLEANED, 'articles.jsonl'),
                [{k: v for k, v in a.items() if k != 'lines'} for a in kept])

    # rejected: reasons index + text bodies kept separately
    os.makedirs(REJECTED, exist_ok=True)
    rej_index = []
    for i, a in enumerate(gated['rejected'], 1):
        rid = 'rejected_%06d' % i
        body = os.path.join(REJECTED, rid + '.txt')
        with open(body, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('# %s\n# source: %s\n# reasons: %s\n\n%s\n' % (
                rid, a['source_path'],
                '; '.join(a.get('reject_reasons') or ['unspecified']),
                a.get('text', '')))
        rej_index.append({
            'id': rid, 'source_path': a['source_path'],
            'source_file': a.get('source_file'),
            'reasons': a.get('reject_reasons') or ['unspecified'],
            'role': a.get('role'), 'n_chars': a.get('n_chars', len(a.get('text', ''))),
            'text_file': 'rejected/%s.txt' % rid,
        })
    write_jsonl(os.path.join(REJECTED, 'rejected.jsonl'), rej_index)

    # duplicates kept for audit
    dup_index = [{'source_path': a['source_path'],
                  'article_index': a.get('article_index'),
                  'dedup_status': a['dedup_status'],
                  'duplicate_of': a.get('duplicate_of'),
                  'similarity': a.get('near_dup_similarity'),
                  'n_chars': len(a.get('text', ''))}
                 for a in gated['dupes']]
    write_jsonl(os.path.join(CLEANED, 'duplicates.jsonl'), dup_index)

    log('wrote strict=%d permissive=%d rejected=%d duplicates=%d'
        % (len(strict_recs), len(perm_recs), len(rej_index), len(dup_index)))
    return {'strict': strict_recs, 'permissive': perm_recs,
            'rejected': rej_index, 'duplicates': dup_index}


# --------------------------------------------------------------------------
def load_ocr_records(enabled: bool = False) -> List[dict]:
    """OCR ingestion is OFF by default.

    Measured quality on this archive's Nastaliq scans was ~50% OOV against the
    digital corpus vocabulary (vs 3.6% for the digital text itself) - see
    ocr/ocr_quality_assessment.json. Such text is excluded from both datasets
    by decision. Pass --include-ocr to ingest ocr/ocr_text.jsonl anyway; those
    records carry role='ocr_page' and land in the permissive tier.
    """
    if not enabled or not os.path.exists(OCR_JSONL):
        return []
    out = []
    with open(OCR_JSONL, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r.setdefault('extraction_method', 'ocr')
            r.setdefault('role', 'ocr_page')
            r['unmapped_glyphs'] = r.get('text', '').count(cl.REPLACEMENT)
            out.append(r)
    return out


_FRONT_BACK_RE = re.compile(r'front\s*page|frontpage|back\s*page|last\s*page', re.I)


def _ocr_section(ocr_records: List[dict], inv: dict) -> dict:
    """Record the OCR decision, its measured basis, and the separate output.

    hp/ocr.py imports this module, so page-role detection is duplicated here
    rather than imported, to avoid a circular dependency.
    """
    images = inv['ocr_candidates']
    front_back = [p for p in images if _FRONT_BACK_RE.search(os.path.basename(p))]

    assess_path = os.path.join(ROOT, 'ocr', 'ocr_quality_assessment.json')
    assessment = None
    if os.path.exists(assess_path):
        try:
            with open(assess_path, encoding='utf-8') as fh:
                assessment = json.load(fh)
        except (OSError, json.JSONDecodeError):
            assessment = None

    run_path = os.path.join(ROOT, 'ocr_low_quality', 'ocr_run.json')
    run = None
    if os.path.exists(run_path):
        try:
            with open(run_path, encoding='utf-8') as fh:
                run = json.load(fh)
        except (OSError, json.JSONDecodeError):
            run = None

    section = {
        'status': ('ingested_into_main_dataset' if ocr_records
                   else ('produced_separately' if run else 'not_run')),
        'records_ingested_into_main_dataset': len(ocr_records),
        'page_images_in_archive': len(images),
        'of_which_front_or_back_page': len(front_back),
        'separate_deliverable': {
            'directory': 'ocr_low_quality/',
            'files': ['ocr_pages.jsonl', 'ocr_pages.txt', 'ocr_run.json',
                      'README.md'],
            'merged_into_hindko_dataset': False,
            'reason': ('Kept apart on purpose. Every record carries its own '
                       'ocr_mean_confidence, oov_vs_vocab, oov_vs_common and '
                       'lexical_verdict so a user can filter without trusting '
                       'our judgement. `language` is recorded as "und", not '
                       '"hindko", because the text is too unreliable to assert '
                       'a language for.'),
        },
        'coverage_gap': (
            'Front and back pages are designed in CorelDRAW (.cdr), which '
            'yields no extractable text, and .eps exports contain no text '
            'operators. Content that exists ONLY on those pages reaches the '
            'project only through this low-quality OCR output.'),
        'decision': (
            'OCR is produced but EXCLUDED from both main datasets. Tesseract '
            '5.4.0 with tessdata_best urd was measured against the vocabulary '
            'of the digitally-extracted corpus and produced ~50% '
            'out-of-vocabulary tokens on sample pages (3.6% for the digital '
            'text itself); across the full run no page scored better than '
            '"poor". Nastaliq script defeats the Naskh-trained model, and the '
            'errors are plausible near-misses that a script-fraction quality '
            'gate cannot detect - so merging them would silently corrupt the '
            'corpus rather than extend it.'),
        'evidence_file': 'ocr/ocr_quality_assessment.json',
    }
    if assessment:
        section['measured_sample'] = {
            'engine': assessment.get('engine'),
            'pages_measured': assessment.get('pages_measured'),
            'digital_oov_vs_common': assessment.get('digital_baseline', {}).get('oov_vs_common'),
            'ocr_oov_vs_full_vocab': assessment.get('ocr', {}).get('oov_vs_full_vocab'),
            'ocr_oov_vs_common': assessment.get('ocr', {}).get('oov_vs_common'),
            'usable_for_research_corpus': assessment.get('usable_for_research_corpus'),
        }
    if run:
        section['full_run'] = {
            'pages_ocrd': run.get('pages_ocrd'),
            'lexical_verdicts': run.get('lexical_verdicts'),
            'page_roles': run.get('page_roles'),
            'mean_confidence': run.get('mean_confidence'),
            'mean_oov_vs_common': run.get('mean_oov_vs_common'),
            'total_characters': run.get('total_characters'),
            'engine': run.get('engine'),
        }
    return section


def build_report(inv, ex, seg, gated, emitted, ocr_records, started, log,
                 news_gated=None, book_gated=None, book_stage=None) -> dict:
    det = seg['boilerplate']
    strict, perm = emitted['strict'], emitted['permissive']

    def totals(recs):
        chars = sum(r['n_chars'] or 0 for r in recs)
        words = sum(r['n_words'] or 0 for r in recs)
        return {'documents': len(recs), 'characters': chars, 'words': words}

    present = sum(1 for p in (e['path'] for e in inv['entries'])
                  if os.path.exists(local_path(p)))
    ocr_needed = [p for p in inv['ocr_candidates'] if os.path.exists(local_path(p))]

    reason_counts = collections.Counter()
    for a in gated['rejected']:
        for r in (a.get('reject_reasons') or ['unspecified']):
            reason_counts[r.split('(')[0]] += 1

    report = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'runtime_seconds': round(time.time() - started, 1),
        'source': {
            'drive_folder': DRIVE_ROOT_URL,
            'language': LANGUAGE,
            'trees': {
                'Hindkowan Newspaper Data/': {
                    'source': 'newspaper', 'publication': PUBLICATION,
                    'files': sum(v for (k, _), v in inv['by_kind_ext'].items()
                                 if k == 'newspaper')},
                'Hindko Books/': {
                    'source': 'book',
                    'publisher_note': ('publisher is taken per book from its own '
                                       'imprint page (see book_imprints.jsonl); '
                                       'the series is the Gandhara Hindko Academy '
                                       '"Farma" series'),
                    'files': sum(v for (k, _), v in inv['by_kind_ext'].items()
                                 if k == 'book'),
                    'subfolder_link': os.environ.get('HINDKO_DRIVE_BOOKS_URL', '')},
            },
            'file_counts_by_source_and_ext': {
                '%s/%s' % k: v for k, v in sorted(inv['by_kind_ext'].items())},
        },
        'file_type_disposition': {
            'inp': {
                'count': inv['by_ext'].get('inp', 0),
                'format': 'InPage typesetting document (OLE compound file)',
                'text': True,
                'handling': 'converted - primary corpus source',
            },
            'b01': {
                'count': inv['by_ext'].get('b01', 0),
                'format': 'InPage auto-backup (same OLE format as .inp)',
                'text': True,
                'handling': ('orphans (no same-stem .inp) always converted; with '
                             '--include-all-backups the %d redundant revisions are '
                             'converted too and absorbed by the deduplicators '
                             '(newspaper 2026-09-20: +29 records, +0.4%%)'
                             % len(inv.get('redundant_backup', []))),
            },
            'jpg': {
                'count': inv['by_ext'].get('jpg', 0) + inv['by_ext'].get('jpeg', 0),
                'format': 'rendered page scan',
                'text': False,
                'handling': 'OCR only -> ocr_low_quality/ (separate, flagged)',
            },
            'cdr': {
                'count': inv['by_ext'].get('cdr', 0),
                'format': 'CorelDRAW 9 (RIFF, "CDR9vrsn")',
                'text': False,
                'handling': ('archival mirror only. PROVEN to contain no '
                             'extractable text: no font references, no coherent '
                             'Arabic strings, and \\x04-pair density at or below '
                             'random chance (ratios 0.09-0.96 vs baseline 1.0). '
                             'Text is stored as vector outlines.'),
            },
            'eps': {
                'count': inv['by_ext'].get('eps', 0),
                'format': 'CorelDRAW EPS export (PostScript)',
                'text': False,
                'handling': ('archival mirror only. PROVEN no text: zero text '
                             'operators (the only show/curveto/lineto counts are '
                             'the prolog definitions), zero string literals, '
                             'zero font references.'),
            },
            'zip': {
                'count': inv['by_ext'].get('zip', 0),
                'format': 'ZIP archive (issue 282)',
                'text': False,
                'handling': ('REDUNDANT - its 4 members are the same issue-282 '
                             'page scans that exist as standalone Drive files at '
                             'identical paths. Extracted opportunistically with '
                             'a zip-slip guard and integrity validation; no '
                             'unique content.'),
            },
            'tmp': {
                'count': inv['by_ext'].get('tmp', 0),
                'format': '@@@CDRW.TMP - CorelDRAW 9 workspace scratch file',
                'text': False,
                'handling': ('no text. Same RIFF/CDR9vrsn container as .cdr, so '
                             'the same conclusion applies (verified again for the '
                             'one in Hindko Books/.../80th .../Del/).'),
            },
            'txt': {
                'count': len(inv.get('txt_exports', [])),
                'format': ('UTF-8 plain-text export of each book .inp, made by an '
                           'unknown third-party converter '
                           '(Hindko Books/HINDKO BOOKS DATA/TXT/)'),
                'text': True,
                'handling': ('NOT ingested - it is a second conversion of the same '
                             '.inp files, not new content (27.17M vs 27.40M '
                             'alignable chars, 99.997% of it contained in our '
                             'decode). Used as an independent witness to audit our '
                             'decoder: the aligned diff found our systematic errors '
                             'and also its own (glued paragraph-header bytes on '
                             '13,004 lines, raw hex for 1,141 first glyphs, dropped '
                             'ISBN digits). See probe_conv_*_out.txt.'),
            },
            'docx': {
                'count': inv['by_ext'].get('docx', 0),
                'format': 'Office Open XML document',
                'text': True,
                'handling': ('Index.docx (22nd Farma) extracted with hp/docx.py: a '
                             'romanised back-of-book index, 0 Arabic-script chars, '
                             'so it is gated like any other record. '
                             '~$glish content.docx is a Word owner/lock file (162 B, '
                             'not a ZIP) - no text; quarantined as zip_bad_magic.'),
                'owner_files': inv.get('office_owner_files', []),
            },
        },
        'files': {
            'total_discovered_in_drive': inv['total'],
            'present_on_disk': present,
            'not_downloaded': inv['total'] - present,
            'file_types_found': dict(sorted(inv['by_ext'].items(), key=lambda x: -x[1])),
            'text_bearing_sources': len(inv['text_sources']),
            'processed_successfully': len(ex['results']),
            'extraction_failed': len(ex['failures']),
            'missing_on_disk': len(ex['missing']),
            'b01_backups_skipped_as_duplicate': len(inv['skipped_backup']),
            'b01_backups_processed_as_orphan': len(inv['orphan_backup']),
            'b01_backups_processed_as_redundant_revision':
                len(inv.get('redundant_backup', [])),
            # Counted from what was actually extracted (the 2026-09-20 report
            # counted classified rather than processed backups - doc drift 5.2#5).
            'processed_by_source_and_kind': {
                '%s/%s' % k: v for k, v in sorted(collections.Counter(
                    (rec['source_kind'], rec['kind'])
                    for _, _, rec in ex['results']).items())},
            'files_requiring_ocr': len(ocr_needed),
            'files_with_no_extractable_text': len(inv['no_text_value']),
            'book_txt_exports_not_ingested': len(inv.get('txt_exports', [])),
        },
        'backup_hash_verdicts': dict(collections.Counter(ex['backup_verdicts'].values())),
        'extraction': {
            'method': 'InPage OLE (InPage100 stream) + legacy glyph->Unicode decode',
            'glyph_map_source': 'github.com/salmanasmat/InPageToUnicode (fetched separately)',
            'extra_glyphs_added': {k.encode('unicode_escape').decode():
                                   v for k, v in inpage.EXTRA_GLYPHS.items()},
            'inline_ascii_digits_preserved': True,
            # stream-level counts include style/format-table residue that never
            # reaches the corpus; corpus-level is the number that matters.
            'unmapped_glyphs_in_stream': sum(
                rec.get('unmapped_glyphs', 0) for _, _, rec in ex['results']),
            'unmapped_glyphs_in_final_dataset': sum(
                r['text'].count(cl.REPLACEMENT)
                for r in emitted['permissive']),
            'unresolved_glyph_bytes': {
                ('\\x04\\x%02x' % b): v for b, v in
                sorted(inpage.UNRESOLVED_GLYPHS.items(),
                       key=lambda x: -x[1]['in_content'])},
            'failures': ex['failures'][:50],
            'missing_examples': ex['missing'][:25],
        },
        'segmentation': {
            'articles_segmented': len(seg['articles']),
            'non_content_frames_dropped': seg['dropped_frames'],
            'masthead_headers_removed': seg['mastheads_removed'],
            'method': ('InPage text-frame grouping. 2023 issues store frames in '
                       'reading order (headline -> subhead -> body) so records '
                       'are whole articles with a real title. 2024-2026 issues '
                       'store frames in object order, not reading order, so '
                       'headlines can follow their bodies and article '
                       'boundaries are not recoverable from sequence; there each '
                       'frame becomes its own coherent record.'),
            'known_limitation': (
                'For 2024-2026 sources, records are frame-sized text blocks '
                'rather than guaranteed whole articles, and `title` is usually '
                'null because the matching headline frame cannot be reliably '
                'associated. No text is lost or reordered - only the grouping '
                'granularity differs.'),
            'boilerplate_lines_detected': len(det.boiler),
            'boilerplate_examples': det.report(30),
        },
        'books': (book_stage['out']['report'] if book_stage else
                  {'status': 'no book sources processed'}),
        'deduplication': {
            'exact_and_near_duplicates_removed': len(gated['dupes']),
            'by_source': {
                'newspaper': len(news_gated['dupes']) if news_gated else None,
                'book_records': len(book_gated['dupes']) if book_gated else None,
                'book_lines_removed_before_records': (
                    len(book_stage['out']['line_dedup']) if book_stage else 0),
                'book_line_dedup_audit': 'cleaned/book_line_dedup.jsonl',
            },
            'exact': sum(1 for a in gated['dupes'] if a['dedup_status'] == 'exact_duplicate'),
            'near': sum(1 for a in gated['dupes'] if a['dedup_status'] == 'near_duplicate'),
            'near_duplicate_threshold': 0.85,
            'algorithm': 'SHA-256 exact + 128-perm MinHash banded LSH',
        },
        'quality': {
            'tier_counts': dict(gated['tier_counts']),
            'tier_counts_by_source': {
                'newspaper': dict(news_gated['tier_counts']) if news_gated else None,
                'book': dict(book_gated['tier_counts']) if book_gated else None},
            'rejected_documents': len(gated['rejected']),
            'rejection_reasons': dict(reason_counts.most_common()),
            'strict_thresholds': cl.STRICT,
            'permissive_thresholds': cl.PERMISSIVE,
            'language_variety': {
                'strict_dataset': dict(collections.Counter(
                    r['language_variety'] for r in strict)),
                'permissive_dataset': dict(collections.Counter(
                    r['language_variety'] for r in perm)),
                'by_source': {
                    src: {'strict': dict(collections.Counter(
                              r['language_variety'] for r in strict if r['source'] == src)),
                          'permissive': dict(collections.Counter(
                              r['language_variety'] for r in perm if r['source'] == src))}
                    for src in ('newspaper', 'book', 'web')},
                'scorer': ('lang.hindko_score_v2: whole-token Hindko vs Urdu '
                           'marker ratio (validated markers, probe_books_markers.py)'),
                'note': ('Weekly Hindkowan also carries Urdu columns/poetry and '
                         'many "Hindko books" are Urdu (conference papers, Urdu '
                         'verse, dictionary glosses). Urdu-dominant items are '
                         'demoted from strict to permissive, never deleted.'),
            },
        },
        'ocr': _ocr_section(ocr_records, inv),
        'final_dataset': {
            'by_source': {
                src: {'strict': totals([r for r in strict if r['source'] == src]),
                      'permissive_including_strict':
                          totals([r for r in perm if r['source'] == src])}
                for src in ('newspaper', 'book', 'web')},
            'strict': {**totals(strict),
                       'files': ['hindko_dataset.jsonl', 'hindko_dataset.txt']},
            'permissive_including_strict': {
                **totals(perm),
                'files': ['hindko_dataset_permissive.jsonl',
                          'hindko_dataset_permissive.txt']},
            'bytes_on_disk': {
                'hindko_dataset.jsonl': os.path.getsize(os.path.join(OUT_ROOT, 'hindko_dataset.jsonl')),
                'hindko_dataset.txt': os.path.getsize(os.path.join(OUT_ROOT, 'hindko_dataset.txt')),
            },
        },
        'issues_covered': sorted({r['issue'] for r in strict if r.get('issue')},
                                 key=lambda x: int(x)),
        'problems': [],
    }

    if ex['failures']:
        report['problems'].append(
            '%d file(s) failed extraction - see extraction.failures' % len(ex['failures']))
    if ex['missing']:
        report['problems'].append(
            '%d text-bearing file(s) are not present under raw_files/ yet'
            % len(ex['missing']))
    if inv['total'] - present:
        report['problems'].append(
            '%d of %d Drive files not downloaded yet' % (inv['total'] - present, inv['total']))
    if report['extraction']['unmapped_glyphs_in_final_dataset']:
        report['problems'].append(
            '%d unmapped glyph(s) reached the final dataset; they are surfaced '
            'as U+FFFD and gated, never silently guessed. A further %d occur '
            'in style/format tables and are discarded before segmentation.'
            % (report['extraction']['unmapped_glyphs_in_final_dataset'],
               report['extraction']['unmapped_glyphs_in_stream']))
    elif report['extraction']['unmapped_glyphs_in_stream']:
        report['problems'].append(
            '%d unmapped glyph byte(s) occur in the source streams but all fall '
            'in style/format tables; none reached the dataset.'
            % report['extraction']['unmapped_glyphs_in_stream'])
    if not report['problems']:
        report['problems'].append('none')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='process only the first N source files (debug)')
    ap.add_argument('--include-differing-backups', action='store_true',
                    help='also process .B01 backups whose bytes differ from '
                         'their .inp twin. Off by default: .B01 files are '
                         'older InPage revisions of the same document, so the '
                         '.inp is authoritative and the backups mainly add '
                         'near-duplicate noise.')
    ap.add_argument('--include-all-backups', action='store_true',
                    help='convert every .B01 backup, including the 164 that are '
                         'older revisions of an .inp already in the corpus. The '
                         'deduplicator absorbs the overlap. Supersedes '
                         '--include-differing-backups.')
    ap.add_argument('--include-ocr', action='store_true',
                    help='ingest ocr/ocr_text.jsonl. Off by default: OCR of '
                         'these Nastaliq scans measured ~50%% OOV and is '
                         'excluded from the dataset by decision.')
    ap.add_argument('--no-web', action='store_true',
                    help='leave out the web sources in raw_files/web')
    ap.add_argument('--out', default=None,
                    help='write every output under this directory instead of '
                         'F:/Hindko (staging); inputs are always read from F:/Hindko')
    ap.add_argument('--release', action='store_true',
                    help='allow writing into F:/Hindko itself, replacing the '
                         'released dataset; without it a build must use --out')
    args = ap.parse_args()
    if args.out:
        set_out_root(args.out)
    elif not args.release:
        # Guard (2026-09-25): a plain run must not overwrite the release before
        # a staged build has passed QA.
        sys.exit('refusing to overwrite the released dataset in F:/Hindko: '
                 'use --out F:/Hindko/_staging for a trial build, or --release '
                 'once the staged build has passed QA')

    started = time.time()

    def log(msg):
        print('[%6.1fs] %s' % (time.time() - started, msg), flush=True)

    for d in (RAW_EXTRACTED, CLEANED, REJECTED):
        os.makedirs(d, exist_ok=True)

    log('discovering files')
    inv = discover(args.include_all_backups)
    log('drive entries=%d text sources=%d ocr candidates=%d' % (
        inv['total'], len(inv['text_sources']), len(inv['ocr_candidates'])))
    if args.include_all_backups:
        log('converting ALL .B01 backups: %d orphan + %d redundant-revision'
            % (len(inv['orphan_backup']), len(inv['redundant_backup'])))

    if args.limit:
        present = [p for p in inv['text_sources'] if os.path.exists(local_path(p))]
        keep = set(present[:args.limit])
        inv['text_sources'] = [p for p in inv['text_sources'] if p in keep]
        log('LIMIT active: %d files' % len(inv['text_sources']))

    log('hashing .B01 backups against their .inp twins')
    verdicts = resolve_backup_duplicates(inv)
    different = {k for k, v in verdicts.items() if v == 'different'}
    if different and args.include_differing_backups:
        log('%d .B01 differ from their .inp -> adding to sources' % len(different))
        inv['text_sources'] = sorted(set(inv['text_sources']) | different)
    elif different:
        log('%d .B01 differ from their .inp but are older revisions -> skipped '
            '(pass --include-differing-backups to include)' % len(different))

    log('stage 1: extraction')
    ex = stage_extract(inv, verdicts, log)

    news_results = [r for r in ex['results'] if r[2]['source_kind'] == 'newspaper']
    book_results = [r for r in ex['results'] if r[2]['source_kind'] == 'book']

    log('stage 2-3: newspaper segmentation + boilerplate + cleaning')
    issue_dates = build_issue_date_map(inv['entries'])
    log('issue folders with an unambiguous publication date: %d' % len(issue_dates))
    seg = stage_segment_and_clean(news_results, issue_dates, log)

    ocr_records = load_ocr_records(args.include_ocr)
    if ocr_records:
        log('ingesting %d OCR records' % len(ocr_records))
    else:
        log('OCR ingestion disabled (see ocr/ocr_quality_assessment.json)')

    log('stage 4-5: dedup + quality gates (newspaper)')
    deduper = cl.Deduper(near_threshold=0.85)
    # deterministic order so IDs are stable across runs
    news_candidates = sorted(seg['articles'], key=candidate_order)
    news_gated = stage_dedup_and_gate(news_candidates + ocr_records, deduper,
                                      log, 'newspaper')

    book_stage = None
    book_gated = {'kept': [], 'dupes': [], 'rejected': [],
                  'tier_counts': collections.Counter()}
    if book_results:
        log('stage 3b: books (%d files)' % len(book_results))
        book_stage = stage_books(book_results, news_gated['kept'], log)
        log('stage 4-5: dedup + quality gates (books)')
        book_gated = stage_dedup_and_gate(book_stage['records'], deduper, log, 'books')
    web_stage = None
    web_gated = {'kept': [], 'dupes': [], 'rejected': [],
                 'tier_counts': collections.Counter()}
    if WEB_INTEGRATED and not args.no_web:
        log('stage 3c: web sources')
        web_stage = web.process_web(os.path.join(RAW_FILES, 'web'), log)
        log('stage 4-5: dedup + quality gates (web)')
        web_gated = stage_dedup_and_gate(web_stage['records'], deduper, log, 'web')
    gated = merge_gated(news_gated, book_gated, web_gated)
    log('stage 5b: containment (word %d-grams)' % CONTAIN_N)
    containment = stage_containment(gated['kept'], log)
    for part in (news_gated, book_gated, web_gated, gated):
        part['tier_counts'] = collections.Counter(a['quality_tier'] for a in part['kept'])
        part['tier_counts']['reject'] = len(part['rejected'])

    log('stage 6: emit')
    emitted = stage_emit(gated, log)

    report = build_report(inv, ex, seg, gated, emitted, ocr_records, started, log,
                          news_gated=news_gated, book_gated=book_gated,
                          book_stage=book_stage)
    report['deduplication']['containment'] = containment
    report['deduplication']['by_source']['web_records'] = len(web_gated['dupes'])
    report['quality']['tier_counts_by_source']['web'] = dict(web_gated['tier_counts'])
    if web_stage:
        report['web'] = {
            'stats': web_stage['stats'],
            'language_by_source': web_stage['per_source'],
            'site_furniture_lines_by_source': web_stage['boilerplate_lines'],
            'tier_counts_by_site': {
                sid: dict(collections.Counter(a['quality_tier'] for a in web_gated['kept']
                                              if a['site'] == sid))
                for sid in sorted({a['site'] for a in web_stage['records']})},
            'classifier': 'hp/langid.py (models/langid_hno_v1.pkl), trained by _web/langid_train.py',
            'survey': '_web/survey/corpora.md, websites.md, religious_academic.md',
        }
    report['pii'] = {
        'policy': ('contact identifiers (phone numbers incl. RTL-reversed and '
                   'Urdu-digit forms, e-mails, CNIC/NIC numbers) are replaced '
                   'by typed placeholders <PHONE> <EMAIL> <ID_NUMBER>, private '
                   'and institutional alike; each record lists its counts in '
                   'pii_masked'),
        'records_masked': sum(1 for r in emitted['permissive'] if r.get('pii_masked')),
        'placeholders': dict(sum((collections.Counter(r['pii_masked'])
                                  for r in emitted['permissive'] if r.get('pii_masked')),
                                 collections.Counter())),
    }
    report['code_sha256'] = {
        f: hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f),
                               'rb').read()).hexdigest()[:16]
        for f in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__))))
        if f.endswith('.py')}
    for name in ('hindko_dataset_permissive.jsonl', 'hindko_dataset_permissive.txt'):
        report['final_dataset']['bytes_on_disk'][name] = os.path.getsize(
            os.path.join(OUT_ROOT, name))
    rp = os.path.join(OUT_ROOT, 'processing_report.json')
    with open(rp, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log('report -> %s' % rp)

    print('\n===== SUMMARY =====')
    print('strict documents      :', report['final_dataset']['strict']['documents'])
    print('strict characters     :', report['final_dataset']['strict']['characters'])
    print('strict words          :', report['final_dataset']['strict']['words'])
    print('permissive documents  :',
          report['final_dataset']['permissive_including_strict']['documents'])
    print('duplicates removed    :',
          report['deduplication']['exact_and_near_duplicates_removed'])
    print('rejected              :', report['quality']['rejected_documents'])
    print('problems              :', report['problems'])


if __name__ == '__main__':
    main()
