import json, io, os

R = r'F:\Hindko'
rep = json.load(io.open(os.path.join(R, 'processing_report.json'), encoding='utf-8'))

print('=== final_dataset ===')
print(json.dumps(rep['final_dataset'], ensure_ascii=False, indent=2))
print('\n=== deduplication ===')
print(json.dumps(rep['deduplication'], ensure_ascii=False, indent=2))
print('\n=== files ===')
for k in ('total_discovered_in_drive', 'present_on_disk', 'not_downloaded',
          'text_bearing_sources', 'processed_successfully', 'extraction_failed',
          'missing_on_disk', 'b01_backups_skipped_as_duplicate',
          'b01_backups_processed_as_orphan',
          'b01_backups_processed_as_redundant_revision',
          'files_requiring_ocr', 'files_with_no_extractable_text'):
    print('  %-44s %s' % (k, rep['files'].get(k)))
print('\n=== segmentation ===')
for k in ('articles_segmented', 'non_content_frames_dropped',
          'masthead_headers_removed', 'boilerplate_lines_detected'):
    print('  %-32s %s' % (k, rep['segmentation'].get(k)))
print('\n=== extraction glyph detail ===')
e = rep['extraction']
for k in ('unmapped_glyphs_in_stream', 'unmapped_glyphs_in_final_dataset'):
    print('  %-34s %s' % (k, e.get(k)))
print('  extra_glyphs_added:', json.dumps(e.get('extra_glyphs_added'), ensure_ascii=False))
print('  unresolved:', list((e.get('unresolved_glyph_bytes') or {}).keys()))
print('\n=== quality ===')
print('  tiers:', rep['quality']['tier_counts'])
print('  rejection_reasons:', rep['quality']['rejection_reasons'])
print('  variety strict:', rep['quality']['language_variety']['strict_dataset'])
print('  variety perm  :', rep['quality']['language_variety']['permissive_dataset'])
print('\n=== ocr ===')
print(json.dumps(rep['ocr'], ensure_ascii=False, indent=2)[:900])
print('\n=== issues covered ===', len(rep.get('issues_covered') or []))
print('=== problems ===')
for p in rep['problems']:
    print('  -', p)

print('\n=== output file sizes ===')
for f in ['hindko_dataset.jsonl', 'hindko_dataset.txt',
          'hindko_dataset_permissive.jsonl', 'hindko_dataset_permissive.txt',
          'processing_report.json', 'README.md']:
    p = os.path.join(R, f)
    if os.path.exists(p):
        print('  %-34s %10.2f MB' % (f, os.path.getsize(p) / 1e6))
print('  %-34s %10.2f MB' % ('ocr_low_quality/ocr_pages.jsonl',
      os.path.getsize(os.path.join(R, 'ocr_low_quality', 'ocr_pages.jsonl')) / 1e6))

run = os.path.join(R, 'ocr_low_quality', 'ocr_run.json')
if os.path.exists(run):
    print('\n=== ocr_run.json ===')
    print(json.dumps(json.load(io.open(run, encoding='utf-8')), ensure_ascii=False, indent=2))
