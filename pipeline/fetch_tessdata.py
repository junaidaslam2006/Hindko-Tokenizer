"""Fetch Urdu traineddata for Tesseract into a local tessdata dir.

We keep it under _pipeline/tessdata and point Tesseract at it via
TESSDATA_PREFIX, so no admin rights are needed to add a language.

tessdata_best stores .traineddata via Git LFS, so raw.githubusercontent.com
returns an LFS *pointer* rather than the model. Try the LFS media host first
and validate that what we got is really a model file, not a pointer.
"""
import os
import sys

import requests

DEST = r'F:\Hindko\_pipeline\tessdata'
NAME = 'urd.traineddata'

CANDIDATES = [
    ('tessdata_best (LFS media)',
     'https://media.githubusercontent.com/media/tesseract-ocr/tessdata_best/main/urd.traineddata'),
    ('tessdata (standard, LFS media)',
     'https://media.githubusercontent.com/media/tesseract-ocr/tessdata/main/urd.traineddata'),
    ('tessdata_fast (LFS media)',
     'https://media.githubusercontent.com/media/tesseract-ocr/tessdata_fast/main/urd.traineddata'),
    ('tessdata_best (raw)',
     'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/urd.traineddata'),
]

MIN_VALID_BYTES = 200_000   # real models are megabytes; LFS pointers are ~130 bytes


def looks_like_pointer(head: bytes) -> bool:
    return head.lstrip().startswith(b'version https://git-lfs')


def main():
    os.makedirs(DEST, exist_ok=True)
    out = os.path.join(DEST, NAME)

    for label, url in CANDIDATES:
        print('trying %-34s %s' % (label, url), flush=True)
        try:
            r = requests.get(url, stream=True, timeout=180)
        except requests.RequestException as e:
            print('   error: %s' % e, flush=True)
            continue
        if r.status_code != 200:
            print('   HTTP %d' % r.status_code, flush=True)
            continue

        tmp = out + '.part'
        size = 0
        head = b''
        with open(tmp, 'wb') as fh:
            for chunk in r.iter_content(1 << 20):
                if not head:
                    head = chunk[:64]
                fh.write(chunk)
                size += len(chunk)

        if size < MIN_VALID_BYTES or looks_like_pointer(head):
            print('   rejected: %d bytes (LFS pointer or truncated)' % size, flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            continue

        os.replace(tmp, out)
        print('   OK: %s (%.2f MB)' % (out, size / 1e6), flush=True)
        print('\nUse with:  set TESSDATA_PREFIX=%s' % DEST)
        return 0

    print('\nFAILED: could not obtain a valid %s' % NAME, file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
