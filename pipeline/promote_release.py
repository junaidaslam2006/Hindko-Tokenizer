"""Promote a verified staging build to the released dataset at F:\\Hindko.

The current release is MOVED (not deleted) to a dated backup directory first,
then the staging outputs are moved into place. Usage:

    python promote_release.py F:\\Hindko\\_staging2 2026-09-20
"""
import os
import shutil
import sys

ROOT = r'F:\Hindko'
ITEMS = ['hindko_dataset.jsonl', 'hindko_dataset.txt',
         'hindko_dataset_permissive.jsonl', 'hindko_dataset_permissive.txt',
         'processing_report.json', 'cleaned', 'rejected', 'raw_extracted']


def main():
    stage = os.path.abspath(sys.argv[1])
    old_label = sys.argv[2]
    for it in ITEMS:
        if not os.path.exists(os.path.join(stage, it)):
            sys.exit('staging is incomplete, missing: %s' % it)
    backup = os.path.join(ROOT, '_previous_release_%s' % old_label)
    if os.path.exists(backup):
        sys.exit('backup directory already exists: %s' % backup)
    os.makedirs(backup)
    for it in ITEMS:
        src = os.path.join(ROOT, it)
        if os.path.exists(src):
            shutil.move(src, os.path.join(backup, it))
            print('backed up  ', it)
    for it in ITEMS:
        shutil.move(os.path.join(stage, it), os.path.join(ROOT, it))
        print('promoted   ', it)
    print('done. previous release in', backup)


if __name__ == '__main__':
    main()
