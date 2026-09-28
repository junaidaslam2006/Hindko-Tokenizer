# -*- coding: utf-8 -*-
"""Write the FRONTIER (report-only test) variant of the Colab notebook into colab/build_frontier_test/upload/.

The generic hindko_lm_arbiter.ipynb (which build_bundle.py copies into every upload folder) says "the test
split is not in the bundle" and its cell 5 runs every stage; for this bundle that text is wrong and that cell is
refused by run_all.py. This script writes hindko_lm_arbiter_FRONTIER_TEST.ipynb with:
  * a warning in the first cell (this bundle's 'dev' set IS the strict test split; report-only; FRONTIER_LM.md);
  * cell 5 = PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, then
             run_all('--stages', 'parity,confirm', '--lr', '1e-3', '--no-zip')
and removes the generic copy from the upload folder, so only the right notebook is there. Cells 1-4 and 6 are
unchanged. It also copies colab/FRONTIER_LM.md into the upload folder.
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "hindko_lm_arbiter.ipynb")
UP = os.path.join(HERE, "build_frontier_test", "upload")
DST = os.path.join(UP, "hindko_lm_arbiter_FRONTIER_TEST.ipynb")
CMD = "rc = run_all('--stages', 'parity,confirm', '--lr', '1e-3', '--no-zip')"


def main():
    P = json.load(open(os.path.join(UP, "PARTS.json"), encoding="utf-8"))
    if P.get("kind") != "final_test_report":
        raise SystemExit("the upload folder is not a report-only test bundle")
    est = json.load(open(os.path.join(HERE, "build_frontier_test", "FRONTIER_ESTIMATE.json"), encoding="utf-8"))
    if est["bundle_manifest_sha256"] != P["bundle_manifest_sha256"]:
        raise SystemExit("FRONTIER_ESTIMATE.json belongs to another bundle; re-run frontier_estimate.py")
    hours = "%.0f-%.0f" % (est["total_hours"], est["total_hours_pessimistic_x1.5"])
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    md = "".join(cells[0]["source"])
    old = "The test split is not in the bundle. Nothing is installed or downloaded by this notebook."
    assert old in md, "notebook text changed; update this script"
    md = md.replace("# Hindko tokenizer LM arbiter (PLAN.md section 5) on a Colab GPU",
                    "# Hindko tokenizer vs frontier tokenizers - REPORT-ONLY LM comparison on the test split (Colab GPU)")
    md = md.replace(old, (
        "**REPORT-ONLY TEST bundle %s** (%d tokenizers: %s). Its 'dev' set IS the strict TEST split: every "
        "'dev bpb' this notebook prints is a TEST number. No decision depends on it; the released tokenizer is fixed. "
        "Run cells 1-6 in order on a T4; cell 5 runs only the Stage 4 'confirm' recipe (same small LM for every "
        "tokenizer, same bytes), LR 1e-3, seeds 1-3, no LR sweep. Expect about %s hours on a T4 (an estimate; "
        "resumable; mount Drive). Read `FRONTIER_LM.md` first. Nothing is installed or downloaded by this notebook."
        % (P["bundle_manifest_sha256"][:12], len(P["candidates"]), ", ".join(P["candidates"]), hours)))
    cells[0]["source"] = [ln + "\n" for ln in md.split("\n")[:-1]] + [md.split("\n")[-1]]
    c5 = cells[5]
    src = "".join(c5["source"])
    assert src.rstrip().endswith("rc = run_all()"), "notebook cell 5 changed; update this script"
    c5["source"] = [
        "# 5. REPORT-ONLY frontier comparison: Stage 4 'confirm' only (d=192, full train, 1 epoch), LR 1e-3 fixed (no\n",
        "#    sweep), seeds 1-3, %d tokenizers = %d runs (see FRONTIER_LM.md for the per-tokenizer time). Resumable:\n"
        % (len(P["candidates"]), 3 * len(P["candidates"])),
        "#    finished runs are skipped. Candidates run in priority order (llama-4, roberta-urdu, bloom last).\n",
        "#    Optional: add '--max-hours', '11' to stop cleanly before a session limit, or '--only', 'gpt-4o,gemma-3'\n",
        "#    for a subset (the baseline hindko-1.0.0 is always kept). Every 'dev' number printed here is a TEST number.\n",
        "import os\n",
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   # 200k+ vocabularies: less fragmentation\n",
        CMD]
    c5["outputs"] = []
    c5["execution_count"] = None
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    generic = os.path.join(UP, os.path.basename(SRC))
    if os.path.isfile(generic):
        os.remove(generic)
    doc = os.path.join(HERE, "FRONTIER_LM.md")
    if os.path.isfile(doc):
        shutil.copy2(doc, os.path.join(UP, "FRONTIER_LM.md"))
    print("wrote", DST)


if __name__ == "__main__":
    main()
