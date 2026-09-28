# Corpus pipeline

Builds the Hindko text corpus the tokenizer was trained on: decode, extract, segment, clean, deduplicate, gate and emit.

| File | Purpose |
|---|---|
| `hp/build.py` | End-to-end build (idempotent): extract, segment, clean, dedup, quality gates, strict and permissive tiers, `processing_report.json` |
| `hp/inpage.py` | InPage (legacy Urdu desktop-publishing) decoder |
| `hp/ocr.py` | OCR for scanned sources (Tesseract, resumable) |
| `hp/books.py`, `hp/docx.py`, `hp/segment.py`, `hp/clean.py` | Source-specific extraction, article segmentation and cleaning |
| `hp/normalize.py` | Canonical Unicode normalization used everywhere downstream |
| `hp/lang.py`, `hp/langid.py` | Script and language checks; the Hindko language-ID classifier |
| `hp/web.py` | Web-source policies and site-furniture removal |
| `download_all.py`, `integrity_scan.py`, `verify_mirror.py` | Mirror the source material and check it |
| `qa_dataset.py`, `final_numbers.py`, `verify_report.py`, `promote_release.py` | QA, statistics and release promotion |

## Rebuild

```bash
python pipeline/hp/build.py                 # full rebuild
python pipeline/qa_dataset.py               # QA report
python pipeline/final_numbers.py            # headline statistics
```

`hp/build.py` reads the source folders from environment variables (`HINDKO_DRIVE_ROOT_URL`, `HINDKO_DRIVE_BOOKS_URL`); the source material itself is not public.

## InPage glyph table (not included)

`hp/inpage.py` imports `hp/glyph_map.py`, the InPage-to-Unicode mapping table from [salmanasmat/InPageToUnicode](https://github.com/salmanasmat/InPageToUnicode) (file `mapping/glyph_map.py`, commit `17ac26d2b77bf11e053d1c5d2a855f385243089d`). It is licensed GPL-3.0 and is not redistributed here. Download it into `pipeline/hp/glyph_map.py` before running the decoder:

```bash
curl -L -o pipeline/hp/glyph_map.py https://raw.githubusercontent.com/salmanasmat/InPageToUnicode/17ac26d2b77bf11e053d1c5d2a855f385243089d/mapping/glyph_map.py
```
