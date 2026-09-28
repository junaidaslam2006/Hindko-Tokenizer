# -*- coding: utf-8 -*-
"""Write the FINAL-TEST variant of the Colab notebook into colab/build_final_test/upload/.

The generic hindko_lm_arbiter.ipynb (which build_bundle.py copies into every upload folder) says "the test
split is not in the bundle" and its cell 5 runs every stage. For the final-test bundle that text is wrong and
that cell is refused by run_all.py. This script writes hindko_lm_arbiter_FINAL_TEST.ipynb with:
  * a warning in the first cell (this bundle's 'dev' set IS the strict test split; see FINAL_TEST.md);
  * cell 5 = run_all('--stages', 'parity,confirm', '--lr', '1e-3', '--no-zip', '--force-extra-seeds');
and removes the generic copy from the final-test upload folder, so only the right notebook is there.
Cells 1-4 and 6 are unchanged.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "hindko_lm_arbiter.ipynb")
UP = os.path.join(HERE, "build_final_test", "upload")
DST = os.path.join(UP, "hindko_lm_arbiter_FINAL_TEST.ipynb")
CMD = "rc = run_all('--stages', 'parity,confirm', '--lr', '1e-3', '--no-zip', '--force-extra-seeds')"


def main():
    P = json.load(open(os.path.join(UP, "PARTS.json"), encoding="utf-8"))
    if P.get("kind") != "final_test":
        raise SystemExit("the upload folder is not a final-test bundle")
    nb = json.load(open(SRC, encoding="utf-8"))
    cells = nb["cells"]
    md = "".join(cells[0]["source"])
    old = "The test split is not in the bundle. Nothing is installed or downloaded by this notebook."
    assert old in md, "notebook text changed; update this script"
    md = md.replace("# Hindko tokenizer LM arbiter (PLAN.md section 5) on a Colab GPU",
                    "# Hindko tokenizer LM arbiter - ONE-SHOT FINAL TEST (PLAN.md 7.6) on a Colab GPU")
    md = md.replace(old, (
        "**FINAL TEST bundle %s.** Its 'dev' set IS the strict TEST split: every 'dev bpb' this notebook prints is a "
        "TEST number. Run it once, on a T4 (as dev), cells 1-6 in order; cell 5 runs only the Stage 4 'confirm' "
        "recipe, LR 1e-3, seeds 1-5, no LR sweep. Read `colab/FINAL_TEST.md` first. Nothing is installed or "
        "downloaded by this notebook." % P["bundle_manifest_sha256"][:12]))
    cells[0]["source"] = [ln + "\n" for ln in md.split("\n")[:-1]] + [md.split("\n")[-1]]
    c5 = cells[5]
    src = "".join(c5["source"])
    assert src.rstrip().endswith("rc = run_all()"), "notebook cell 5 changed; update this script"
    c5["source"] = [
        "# 5. FINAL TEST: Stage 4 'confirm' only (d=192, full train, 1 epoch), LR 1e-3 fixed (no sweep), seeds 1-5\n",
        "#    (--force-extra-seeds), 4 candidates = 20 runs, ~35-40 min on a T4. Resumable: finished runs are skipped.\n",
        "#    run_all.py refuses any other stage on this bundle. Every 'dev' number printed here is a TEST number.\n",
        CMD]
    c5["outputs"] = []
    c5["execution_count"] = None
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    generic = os.path.join(UP, os.path.basename(SRC))
    if os.path.isfile(generic):
        os.remove(generic)
    print("wrote", DST)


if __name__ == "__main__":
    main()
