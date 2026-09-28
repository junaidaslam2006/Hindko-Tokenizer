"""Verify the OCR engine and measure real quality on one newspaper page."""
import os, sys, time, json
sys.path.insert(0, r'F:\Hindko\_pipeline')

from hp import ocr as O

eng = O.check_engine()
print('=== ENGINE ===')
print(json.dumps(eng, indent=2, ensure_ascii=False))
if not eng['tesseract_binary']:
    sys.exit('tesseract not found')
if not eng['urdu_available']:
    sys.exit('urd not available; languages=%s' % eng['languages'])

# find a page image to test on (skip files that fail integrity validation)
from hp import filecheck
RAW = r'F:\Hindko\raw_files'
SAMP = r'F:\Hindko\_pipeline\samples'
cands = []
for base in (RAW, SAMP):
    for root, _, files in os.walk(base):
        if '_quarantine' in root:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower().lstrip('.') in ('jpg', 'jpeg'):
                p = os.path.join(root, f)
                if filecheck.validate(p)[0]:
                    cands.append(p)
print('\nvalid candidate images:', len(cands))
if not cands:
    sys.exit('no valid images to test')
front = [c for c in cands if 'front' in os.path.basename(c).lower()]
target = front[0] if front else cands[0]
print('testing on:', target)

from PIL import Image
import pytesseract

img = Image.open(target)
print('image size: %dx%d mode=%s  file=%.1f MB'
      % (img.width, img.height, img.mode, os.path.getsize(target) / 1e6))

for psm in (3, 4, 6):
    t0 = time.time()
    try:
        txt = pytesseract.image_to_string(
            img, lang='urd', config=O.tess_config() + ' --psm %d' % psm)
    except Exception as e:
        print('psm %d FAILED: %s' % (psm, e))
        continue
    dt = time.time() - t0
    import re
    from hp import segment
    lines = [l.strip() for l in txt.split('\n') if l.strip()]
    content = [l for l in lines if segment.is_content_line(l)]
    chars = sum(len(l) for l in content)
    print('\n--- psm %d : %.1fs, %d lines, %d content lines, %d chars ---'
          % (psm, dt, len(lines), len(content), chars))
    for l in content[:12]:
        print('   ', l[:110])
