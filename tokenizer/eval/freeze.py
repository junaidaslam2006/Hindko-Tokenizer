# -*- coding: utf-8 -*-
"""PLAN Stage 0, step 1: freeze the split manifest and the canonical-form code.

Verifies (and refuses to write otherwise):
  * splits/split_manifest.jsonl sha256 == 76582d3a... (the hash PLAN.md 1.2 names);
  * both released dataset files hash to the values recorded in splits/splits_report.json
    (the inputs the manifest was built from);
  * _pipeline/hp/normalize.py reports NORMALIZATION_VERSION 1.0.1.
Writes F:/Hindko/_tokenizer/FROZEN.json. Every later script (data/materialize.py,
eval/harness.py) re-checks its inputs against this file and refuses to run on a mismatch.

split_facts.py is deliberately NOT re-run here (PLAN 1.2 asks for it): it computes sizes and
cluster counts on the test split, and this run must not compute anything on test. Its saved
output research/split_facts.json already records manifest sha256 76582d3a..., i.e. this hash.
"""
import datetime
import hashlib
import json
import os
import sys

ROOT = r"F:\Hindko"
TOK = os.path.join(ROOT, "_tokenizer")
EXPECTED_MANIFEST_SHA = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"
EXPECTED_NORM_VERSION = "1.0.1"
PATHS = {
    "split_manifest": os.path.join(TOK, "splits", "split_manifest.jsonl"),
    "load_split_py": os.path.join(TOK, "splits", "load_split.py"),
    "splits_report": os.path.join(TOK, "splits", "splits_report.json"),
    "normalize_py": os.path.join(ROOT, "_pipeline", "hp", "normalize.py"),
    "normalization_spec": os.path.join(TOK, "normalization", "NORMALIZATION.md"),
    "dataset_permissive": os.path.join(ROOT, "hindko_dataset_permissive.jsonl"),
    "dataset_strict": os.path.join(ROOT, "hindko_dataset.jsonl"),
    "split_facts_json": os.path.join(TOK, "research", "split_facts.json"),
    "plan": os.path.join(TOK, "research", "PLAN.md"),
}
OUT = os.path.join(TOK, "FROZEN.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = {k: {"path": p, "sha256": sha256(p), "bytes": os.path.getsize(p)} for k, p in PATHS.items()}
    errors = []
    if files["split_manifest"]["sha256"] != EXPECTED_MANIFEST_SHA:
        errors.append("split manifest sha256 %s != expected %s" % (files["split_manifest"]["sha256"], EXPECTED_MANIFEST_SHA))
    rep = json.load(open(PATHS["splits_report"], encoding="utf-8"))
    for key, name in (("dataset_permissive", "hindko_dataset_permissive.jsonl"), ("dataset_strict", "hindko_dataset.jsonl")):
        want = rep["inputs"][name]["sha256"]
        if files[key]["sha256"] != want:
            errors.append("%s sha256 %s != splits_report input %s" % (name, files[key]["sha256"], want))
        files[key]["matches_splits_report_input"] = files[key]["sha256"] == want
    sys.path.insert(0, os.path.join(ROOT, "_pipeline"))
    from hp import normalize as N
    if N.NORMALIZATION_VERSION != EXPECTED_NORM_VERSION:
        errors.append("normalize.py version %s != %s" % (N.NORMALIZATION_VERSION, EXPECTED_NORM_VERSION))
    facts = json.load(open(PATHS["split_facts_json"], encoding="utf-8"))
    n_rows = sum(1 for _ in open(PATHS["split_manifest"], encoding="utf-8"))
    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)
    frozen = {
        "what": "Frozen inputs of the Hindko tokenizer study (PLAN.md Stage 0). Later stages must verify these hashes.",
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_manifest": {"path": PATHS["split_manifest"], "sha256": files["split_manifest"]["sha256"],
                           "expected_sha256": EXPECTED_MANIFEST_SHA, "verified": True, "rows": n_rows,
                           "splits_report_seed": rep.get("seed"),
                           "make_splits_sha256": rep["code"]["make_splits_sha256"]},
        "normalize": {"path": PATHS["normalize_py"], "module": "hp.normalize", "version": N.NORMALIZATION_VERSION,
                      "sha256": files["normalize_py"]["sha256"], "unicode_version": N.UNICODE_VERSION,
                      "spec": PATHS["normalization_spec"], "spec_sha256": files["normalization_spec"]["sha256"]},
        "datasets": {"permissive": {k: files["dataset_permissive"][k] for k in ("path", "sha256", "bytes", "matches_splits_report_input")},
                     "strict": {k: files["dataset_strict"][k] for k in ("path", "sha256", "bytes", "matches_splits_report_input")},
                     "access": "read-only"},
        "other_files": {k: {"path": files[k]["path"], "sha256": files[k]["sha256"]}
                        for k in ("load_split_py", "splits_report", "plan", "split_facts_json")},
        "split_facts": {"manifest_sha256_recorded_in_split_facts_json": facts.get("manifest_sha256"),
                        "matches_frozen_manifest": facts.get("manifest_sha256") == files["split_manifest"]["sha256"],
                        "re_run_in_stage_0": False,
                        "reason": "split_facts.py computes sizes and cluster counts on the test split; this run computes "
                                  "nothing on test. The saved split_facts.json was produced on this manifest hash."},
        "test_split_policy": "Stage 0 wrote no test-split file and computed no metric on test. The one-shot test "
                             "evaluation is PLAN Stage 5.",
        "python": sys.version.split()[0],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=1)
    print(json.dumps(frozen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
