import json, io, os

rep = json.load(io.open(r'F:\Hindko\processing_report.json', encoding='utf-8'))

print('top-level keys:', list(rep.keys()))

print('\n=== file_type_disposition ===')
ftd = rep.get('file_type_disposition')
if not ftd:
    print('  MISSING!')
else:
    tot = 0
    for k, v in ftd.items():
        tot += v['count']
        print('  %-5s n=%-4d text=%-5s %s' % (k, v['count'], v['text'],
                                              v['handling'][:64]))
    print('  accounted for: %d of %d Drive files' % (tot, rep['files']['total_discovered_in_drive']))

print('\n=== ocr section ===')
o = rep.get('ocr', {})
print('  status:', o.get('status'))
print('  records_ingested_into_main_dataset:', o.get('records_ingested_into_main_dataset'))
print('  page_images_in_archive:', o.get('page_images_in_archive'))
print('  of_which_front_or_back_page:', o.get('of_which_front_or_back_page'))
print('  separate_deliverable:', json.dumps(o.get('separate_deliverable'), ensure_ascii=False)[:220])
print('  full_run:', json.dumps(o.get('full_run'), ensure_ascii=False))
print('  measured_sample:', json.dumps(o.get('measured_sample'), ensure_ascii=False))

print('\n=== final dataset (should be unchanged) ===')
print(json.dumps(rep['final_dataset']['strict'], ensure_ascii=False))
print(json.dumps(rep['final_dataset']['permissive_including_strict'], ensure_ascii=False))

print('\n=== consistency check vs README claims ===')
checks = [
    ('strict documents', rep['final_dataset']['strict']['documents'], 5853),
    ('strict characters', rep['final_dataset']['strict']['characters'], 6324538),
    ('strict words', rep['final_dataset']['strict']['words'], 1306592),
    ('permissive documents', rep['final_dataset']['permissive_including_strict']['documents'], 6085),
    ('duplicates', rep['deduplication']['exact_and_near_duplicates_removed'], 2132),
    ('exact', rep['deduplication']['exact'], 1742),
    ('near', rep['deduplication']['near'], 390),
    ('mastheads', rep['segmentation']['masthead_headers_removed'], 217),
    ('articles segmented', rep['segmentation']['articles_segmented'], 8274),
    ('frames dropped', rep['segmentation']['non_content_frames_dropped'], 3337320),
    ('rejected', rep['quality']['rejected_documents'], 57),
    ('processed ok', rep['files']['processed_successfully'], 648),
    ('missing', rep['files']['missing_on_disk'], 9),
    ('extraction failed', rep['files']['extraction_failed'], 0),
    ('U+FFFD in dataset', rep['extraction']['unmapped_glyphs_in_final_dataset'], 57),
]
bad = 0
for name, actual, expected in checks:
    ok = actual == expected
    if not ok:
        bad += 1
    print('  %-22s %-12s expected %-12s %s' % (name, actual, expected, 'OK' if ok else 'MISMATCH'))
print('\nmismatches:', bad)

print('\n=== advertisement rejections (must be 0) ===')
rr = rep['quality']['rejection_reasons']
print('  ', rr)
print('   advertisement present:', 'advertisement' in rr)

print('\n=== language variety ===')
print('  strict:', rep['quality']['language_variety']['strict_dataset'])
print('  urdu in strict:', rep['quality']['language_variety']['strict_dataset'].get('urdu', 0))
