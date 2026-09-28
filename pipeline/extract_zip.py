"""Extract the issue-282 zip members, but only where the standalone Drive copy
is not already present. The zip is redundant (its 4 members duplicate files
that exist separately in Drive), so this is purely an opportunistic seed -
never overwrite a file the downloader may be writing.
"""
import os, sys, zipfile
sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import filecheck

RAW = r'F:\Hindko\raw_files'
Z = os.path.join(RAW, r'Hindkowan Newspaper Data\2023-2024-2025\282'
                      r'\Weekly Hindkowan Issue No. 282.zip')
BASE = os.path.dirname(Z)          # member paths already include the subfolder

extracted, skipped, bad = [], [], []
with zipfile.ZipFile(Z) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        dest = os.path.join(BASE, info.filename.replace('/', os.sep))
        if os.path.exists(dest):
            skipped.append((info.filename, os.path.getsize(dest), 'already present'))
            continue
        # guard against zip-slip: resolved dest must stay under BASE
        if not os.path.realpath(dest).startswith(os.path.realpath(BASE)):
            bad.append((info.filename, 'path escapes target dir'))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + '.zipextract'
        with zf.open(info) as src, open(tmp, 'wb') as out:
            out.write(src.read())
        ok, why = filecheck.validate(tmp, ext=os.path.splitext(dest)[1])
        if not ok:
            bad.append((info.filename, 'failed integrity: %s' % why))
            os.remove(tmp)
            continue
        os.replace(tmp, dest)
        extracted.append((info.filename, os.path.getsize(dest)))

print('zip members extracted : %d' % len(extracted))
for n, s in extracted:
    print('   + %10d  %s' % (s, n))
print('skipped (already there): %d' % len(skipped))
for n, s, r in skipped:
    print('   = %10d  %s (%s)' % (s, n, r))
print('rejected               : %d' % len(bad))
for n, r in bad:
    print('   ! %s -> %s' % (n, r))
