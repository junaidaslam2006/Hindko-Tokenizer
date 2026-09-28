"""Measure OCR quality objectively using the digitally-extracted corpus as a
vocabulary reference.

Garbled Urdu still looks like Urdu to a script-fraction gate, so we need a
lexical test instead: what share of the OCR output's words are real words that
also occur in the clean InPage-extracted corpus? A same-page comparison isn't
always possible, but vocabulary overlap is a strong proxy - correct OCR of
Hindko newspaper prose must reuse the paper's own vocabulary.
"""
import io, json, os, re, sys, collections
sys.path.insert(0, r'F:\Hindko\_pipeline')

from hp import ocr as O, segment
from hp import filecheck

R = r'F:\Hindko'
WORD = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+')

# ---- 1. vocabulary from the clean digital corpus -------------------------
vocab = collections.Counter()
with io.open(os.path.join(R, 'hindko_dataset.jsonl'), encoding='utf-8') as fh:
    for line in fh:
        if not line.strip():
            continue
        rec = json.loads(line)
        for w in WORD.findall(rec['text']):
            vocab[w] += 1
print('digital corpus vocabulary: %d distinct words from %d tokens'
      % (len(vocab), sum(vocab.values())))

# frequent-word subset: a real OCR of this paper should hit common words hard
common = {w for w, c in vocab.items() if c >= 5}
print('words occurring >=5 times:', len(common))


def lexical_scores(text, label):
    words = WORD.findall(text)
    if not words:
        print('%-34s no words' % label)
        return None
    uniq = set(words)
    oov = 1.0 - (sum(1 for w in words if w in vocab) / len(words))
    oov_common = 1.0 - (sum(1 for w in words if w in common) / len(words))
    # mean word length: OCR garbage tends to fragment into short pieces
    ml = sum(len(w) for w in words) / len(words)
    print('%-34s tokens=%-5d uniq=%-5d OOV=%.1f%%  OOV(common)=%.1f%%  meanlen=%.2f'
          % (label, len(words), len(uniq), 100*oov, 100*oov_common, ml))
    return {'tokens': len(words), 'oov': oov, 'oov_common': oov_common,
            'mean_len': ml}


print('\n=== BASELINE: the digital corpus itself ===')
sample = []
with io.open(os.path.join(R, 'hindko_dataset.jsonl'), encoding='utf-8') as fh:
    for i, line in enumerate(fh):
        if i >= 40:
            break
        sample.append(json.loads(line)['text'])
base = lexical_scores('\n'.join(sample), 'digital (40 records)')

print('\n=== OCR OUTPUT ===')
O.configure_tesseract()
import pytesseract
from PIL import Image

imgs = []
for root, _, files in os.walk(os.path.join(R, 'raw_files')):
    if '_quarantine' in root:
        continue
    for f in files:
        if os.path.splitext(f)[1].lower().lstrip('.') in ('jpg', 'jpeg'):
            p = os.path.join(root, f)
            if filecheck.validate(p)[0]:
                imgs.append(p)
print('valid images:', len(imgs))

results = []
for p in imgs[:3]:
    name = os.path.basename(p)
    with Image.open(p) as im:
        txt = pytesseract.image_to_string(im, lang='urd',
                                          config=O.tess_config() + ' --psm 4')
    keep = [l for l in txt.split('\n') if segment.is_content_line(l.strip())]
    r = lexical_scores('\n'.join(keep), 'OCR psm4: %s' % name[:22])
    if r:
        results.append(r)

print('\n=== VERDICT ===')
if base and results:
    avg = sum(r['oov'] for r in results) / len(results)
    avg_common = sum(r['oov_common'] for r in results) / len(results)
    print('digital OOV : %.1f%%' % (100 * base['oov']))
    print('OCR     OOV : %.1f%%' % (100 * avg))
    print('ratio       : %.1fx worse' % (avg / max(base['oov'], 1e-9)))
    usable = avg < 0.35
    print('OCR usable for a research corpus:', usable)

    import time
    verdict = {
        'measured_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'engine': 'tesseract 5.4.0 + tessdata_best urd, --psm 4',
        'pages_measured': len(results),
        'method': ('share of OCR word tokens absent from the vocabulary of the '
                   'digitally-extracted corpus. The 0.0% digital OOV baseline is '
                   'circular by construction (the vocabulary is built from that '
                   'same corpus), so OOV-vs-common-words is the meaningful '
                   'comparison.'),
        'digital_baseline': {
            'oov_vs_common': round(base['oov_common'], 4),
            'mean_word_length': round(base['mean_len'], 3),
        },
        'ocr': {
            'oov_vs_full_vocab': round(avg, 4),
            'oov_vs_common': round(avg_common, 4),
            'mean_word_length': round(
                sum(r['mean_len'] for r in results) / len(results), 3),
        },
        'per_page': results,
        'usable_for_research_corpus': bool(usable),
        'representative_errors': [
            {'ocr': 'ہت روز ہ بای ام زازیمے', 'actual': 'ہفت روزہ ہندکووان'},
            {'ocr': 'پٹور(مٹرگودان ول )گن رحارامٹرکو',
             'actual': 'پشور(ہندکووان نیوز)گندھارا'},
        ],
        'decision': ('OCR EXCLUDED from both datasets by user decision. Nastaliq '
                     'defeats the Naskh-trained urd model: ~half of OCR tokens '
                     'are not real words, and the errors are plausible '
                     'near-misses that a script-fraction gate cannot detect. '
                     'Front/back-page content existing only as CorelDRAW '
                     'artwork is therefore absent from the corpus.'),
    }
    os.makedirs(os.path.join(R, 'ocr'), exist_ok=True)
    outp = os.path.join(R, 'ocr', 'ocr_quality_assessment.json')
    with io.open(outp, 'w', encoding='utf-8') as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)
    print('\nwrote', outp)
