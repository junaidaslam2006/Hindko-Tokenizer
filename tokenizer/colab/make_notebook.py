# -*- coding: utf-8 -*-
"""make_notebook.py - write hindko_lm_arbiter.ipynb (Colab) next to this file.

Cells (pure Python; Colab already has torch and numpy - nothing is installed):
  1. GPU check          2. optional Google Drive mount (commented; off by default)
  3. locate the parts    4. CPU<->GPU parity check (fast)    5. full run (live output, resumable)
  6. zip the results
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

MD_INTRO = """# Hindko tokenizer LM arbiter (PLAN.md section 5) on a Colab GPU

1. **Runtime > Change runtime type > GPU** (T4 is enough; L4/A100 are faster).
2. Upload **every** `hk_bundle.tar.xz.part###`, `PARTS.json` and `run_all.py` into **one** folder:
   `/content/upload` (preferred; create it in the Files pane), or `/content`, or a Drive folder
   `MyDrive/hindko_lm_upload` (then set `USE_DRIVE = True` in cell 2).
3. Run the cells top to bottom. Cell 5 is resumable: if the runtime disconnects, re-run cells 1-5 and
   finished runs are skipped (with Drive mounted, results survive a disconnect).
4. Cell 6 zips the results. Download the zip (Files pane, or `files.download`) and hand it back.

The test split is not in the bundle. Nothing is installed or downloaded by this notebook."""

CELL_GPU = """# 1. GPU check
import subprocess, torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU: %s | compute capability %d.%d | %.1f GB' % (p.name, p.major, p.minor, p.total_memory / 1e9))
    print('training precision run_all.py will use:', 'bf16 autocast' if p.major >= 8 else 'fp16 autocast + GradScaler',
          '| evaluation: fp32')
else:
    print('NO GPU - Runtime > Change runtime type > GPU, then re-run this cell (CPU would take many hours).')
try:
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30).stdout)
except Exception as e:
    print('nvidia-smi not available:', e)"""

CELL_DRIVE = """# 2. OPTIONAL: Google Drive.
# With Drive mounted, results are written to MyDrive/hindko_lm_out/<bundle id>/ after EVERY run,
# so a disconnected runtime loses nothing and cell 5 resumes where it stopped.
# Mounting opens a Google sign-in / consent dialog that the ACCOUNT OWNER must approve.
# Leave USE_DRIVE = False to keep everything under /content (then download the zip from cell 6
# before the runtime is recycled).
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')"""

CELL_LOCATE = """# 3. Locate the uploaded parts (PARTS.json + hk_bundle.tar.xz.part### + run_all.py in one folder)
import os, json
SEARCH = ['/content/upload', '/content', '/content/drive/MyDrive/hindko_lm_upload', '/content/drive/MyDrive']
PARTS_DIR = next((d for d in SEARCH if os.path.isfile(os.path.join(d, 'PARTS.json'))), None)
assert PARTS_DIR, 'PARTS.json not found in %s - upload the files first' % SEARCH
P = json.load(open(os.path.join(PARTS_DIR, 'PARTS.json')))
missing = [p['name'] for p in P['parts'] if not os.path.isfile(os.path.join(PARTS_DIR, p['name']))]
wrong = [p['name'] for p in P['parts'] if os.path.isfile(os.path.join(PARTS_DIR, p['name']))
         and os.path.getsize(os.path.join(PARTS_DIR, p['name'])) != p['bytes']]
RUN_ALL = os.path.join(PARTS_DIR, 'run_all.py')
print('parts folder:', PARTS_DIR)
print('bundle', P['bundle_manifest_sha256'][:12], '|', len(P['parts']), 'parts,',
      round(sum(p['bytes'] for p in P['parts']) / 1e6, 1), 'MB | candidates:', ', '.join(P['candidates']))
print('baseline:', P['baseline_id'])
print('missing parts:', missing or 'none', '| wrong size:', wrong or 'none',
      '| run_all.py:', 'found' if os.path.isfile(RUN_ALL) else 'MISSING')
assert not missing and not wrong and os.path.isfile(RUN_ALL), 'upload the missing / broken files and re-run this cell'
OUT = '/content/drive/MyDrive/hindko_lm_out' if USE_DRIVE else '/content/out'
print('results will go to', OUT)"""

CELL_RUN_HELPER_AND_PARITY = """# 4. Verify + extract the bundle, then the CPU<->GPU parity check (same fixed tiny config as
#    PARITY.json: 100 steps, seed 1, baseline). Prints the GPU bpb (AMP and fp32) next to the CPU
#    reference; they should agree within ~1 %. Takes about a minute.
import subprocess, sys
def run_all(*extra):
    cmd = [sys.executable, '-u', RUN_ALL, '--parts', PARTS_DIR, '--work', '/content/work', '--out', OUT, *extra]
    print('$', ' '.join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
    rc = proc.wait()
    print('exit code', rc)
    return rc
run_all('--stages', 'parity', '--no-zip')"""

CELL_FULL = """# 5. The full arbiter: LR sweep -> Stage 3 (all candidates x 3 seeds) -> Stage 4 confirm (all
#    candidates x 3 seeds, +2 if the power check asks) -> 'large' (report-only addition).
#    Resumable: finished runs are skipped. Optional extra flags, e.g. '--max-hours', '10'
#    (stop cleanly before a session limit) or '--stages', 'lr,screen' (only some stages).
#    Print the expected runtime first with: run_all('--estimate')
rc = run_all()"""

CELL_ZIP = """# 6. Zip the results (run_all.py also writes a zip when it finishes)
import glob, shutil, os
roots = sorted(d for d in glob.glob(os.path.join(OUT, '*')) if os.path.isdir(d))
for d in roots:
    z = shutil.make_archive('/content/hindko_lm_results_' + os.path.basename(d), 'zip', d)
    print(z, round(os.path.getsize(z) / 1e6, 2), 'MB')
    if os.path.isfile(os.path.join(d, 'summary.txt')):
        print(open(os.path.join(d, 'summary.txt'), encoding='utf-8').read())
# To download through the browser:
# from google.colab import files; files.download(z)"""


def cell(src, kind="code"):
    lines = src.split("\n")
    body = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": body}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": [cell(MD_INTRO, "markdown"), cell(CELL_GPU), cell(CELL_DRIVE), cell(CELL_LOCATE),
                    cell(CELL_RUN_HELPER_AND_PARITY), cell(CELL_FULL), cell(CELL_ZIP)]}
    for i, c in enumerate(nb["cells"]):
        c["id"] = "c%02d" % i
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")   # syntax check
    p = os.path.join(HERE, "hindko_lm_arbiter.ipynb")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", p)


if __name__ == "__main__":
    main()
