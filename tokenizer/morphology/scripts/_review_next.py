import csv, sys, collections
sys.path.insert(0, 'F:/Hindko/_tokenizer/morphology/scripts')
from induce import PARAMS
seen = set(open('F:/Hindko/_tokenizer/morphology/review/_reviewed_words.txt', encoding='utf-8').read().split())
rows = list(csv.DictReader(open('F:/Hindko/_tokenizer/morphology/data/candidates_all.tsv', encoding='utf-8'), delimiter='\t'))
conf = sys.argv[1]; margin = int(sys.argv[2])
quota = PARAMS['QUOTA_HIGH'] if conf == 'high' else PARAMS['QUOTA_LOW']
by = collections.defaultdict(list)
for r in rows:
    if r['confidence'] == conf: by[r['category']].append(r)
new = []
for cat in sorted(quota):
    lst = sorted(by[cat], key=lambda r: (-int(r['freq']), r['word']))[:quota[cat] + margin]
    for r in lst:
        if r['word'] not in seen:
            new.append(r)
            print(cat, r['segmentation'], r['freq'], '|', r['evidence'][:60], '|u', r['urdu_ratio'], '|pp', r['next_is_postposition'], '|', r['notes'][:80])
if len(sys.argv) > 3:
    with open('F:/Hindko/_tokenizer/morphology/review/_reviewed_words.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(r['word'] for r in new) + '\n')
print('NEW', len(new))
