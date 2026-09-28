# -*- coding: utf-8 -*-
"""MANIFEST.json: sha256 and size of every file in candidates/pickybpe (except __pycache__ and the manifest),
plus a check that the code hashes recorded inside the shipped model equal the code on disk.

    python make_manifest.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402

SHIPPED = os.path.join(HERE, "models", "pickybpe_P1_D1_16k_tau0.9.json")


def main():
    files = {}
    for root, dirs, fs in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, HERE).replace(os.sep, "/")
            if rel == "MANIFEST.json":
                continue
            files[rel] = {"sha256": C.sha256_file(p), "bytes": os.path.getsize(p)}
    m = json.load(open(SHIPPED, encoding="utf-8"))
    rec = m["meta"]["code_sha256"]
    code_ok = {f: rec[f] == files[f]["sha256"] for f in rec}
    out = {"what": "PLAN.md A7 PickyBPE candidate (Stage 2); every file with its sha256",
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "shipped": {"custom_encoder_model": "models/pickybpe_P1_D1_16k_tau0.9.json",
                       "sha256": files["models/pickybpe_P1_D1_16k_tau0.9.json"]["sha256"],
                       "factory": "pickybpe_factory:load_16k", "encoder_code": "pickybpe.py",
                       "hf_tokenizer_json": None,
                       "hf_tokenizer_json_reason": "not shipped: exact equivalence on 100% of dev is impossible with "
                                                   "this vocabulary (verify/*.hf_equivalence.json)"},
           "frozen": C.verify_frozen(),
           "code_hashes_recorded_in_model_match_disk": code_ok,
           "files": files}
    json.dump(out, open(os.path.join(HERE, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, ensure_ascii=False, indent=1))
    print(len(files), "files")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
