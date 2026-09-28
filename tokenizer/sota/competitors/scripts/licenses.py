# -*- coding: utf-8 -*-
"""Licence of each NEW tokenizer row worth citing (lossless with bytes/token >= 4, or any row >= 5), read from the model
card's YAML front matter at the pinned revision (README.md fetched as text into a temp dir and deleted; anonymous).
Output: ../_licenses.json {repo: licence or null}; report_refresh.py merges it."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import download_refresh as D  # noqa: E402

ROOT = D.ROOT


def main():
    out_p = os.path.join(ROOT, "_licenses.json")
    lic = json.load(open(out_p, encoding="utf-8")) if os.path.exists(out_p) else {}
    want = {}
    for p in glob.glob(os.path.join(ROOT, "screen", "*.json")):
        s = json.load(open(p, encoding="utf-8"))
        sc = s.get("screen")
        if not sc:
            continue
        if (sc["g1_pass_docs"] == 491 and sc["bytes_per_token"] >= 4.0) or sc["bytes_per_token"] >= 5.0:
            want[s["repo"]] = s.get("revision")
    todo = [r for r in want if r not in lic]
    print("licences to read", len(todo), flush=True)
    for r in todo:
        lic[r] = D.license_from_card(r, want[r] or "main")
        print(r, lic[r], flush=True)
    json.dump(lic, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
