# -*- coding: utf-8 -*-
"""Write lm/WAVES_final_test.json: the FOUR test candidates fixed by analysis/decision.json
(test_candidates_fixed, frozen before anything touched test), in that order, each entry copied verbatim
from the round-2 waves file (the 3 round-2 tokenizers and the round-1 reference A1-P1r3-D2-16k are all
in lm/WAVES_round2.json) plus its test role. Read by colab/build_bundle.py --final-test.
Deterministic; no timestamps. Refuses if the id list does not hash to the recorded sha256."""
import hashlib
import json
import os

LM = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.dirname(LM)
DEC = os.path.join(TOK, "analysis", "decision.json")
SRC = os.path.join(LM, "WAVES_round2.json")
OUT = os.path.join(LM, "WAVES_final_test.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    d = json.load(open(DEC, encoding="utf-8"))
    fx = d["test_candidates_fixed"]
    ids = fx["ids"]
    if hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest() != fx["sha256_of_ids"]:
        raise SystemExit("test candidate ids do not match their recorded sha256")
    roles = {x["id"]: x["role"] for x in d["test_candidates"]}
    w = json.load(open(SRC, encoding="utf-8"))
    by_id = {e["id"]: e for e in w["waves"]}
    waves = []
    for i in ids:
        e = dict(by_id[i])
        if sha(e["tokenizer_path"]) != e["tokenizer_sha256"]:
            raise SystemExit("tokenizer file of %s changed since WAVES_round2.json" % i)
        role = roles[i]
        e["final_test_role"] = ("chosen" if role.startswith("chosen") else
                                "baseline" if role.startswith("baseline") else "report-only")
        e["final_test_role_detail"] = role
        waves.append(e)
    out = {"what": "The one-shot FINAL TEST candidates (PLAN 7.6), fixed by analysis/decision.json before the test "
                   "split was opened. Read by colab/build_bundle.py --final-test. Entries are copied verbatim from "
                   "lm/WAVES_round2.json, plus final_test_role; report-only rows never re-select.",
           "written_by": "lm/make_waves_final_test.py (deterministic; no timestamps)",
           "baseline": "A1-P1r3-D2-16k",
           "sources": {"decision_json": {"file": DEC, "sha256": sha(DEC)},
                       "waves_round2": {"file": SRC, "sha256": sha(SRC)}},
           "test_candidates_fixed": {"ids": ids, "sha256_of_ids": fx["sha256_of_ids"], "fixed_utc": fx["utc"]},
           "protocol": "Stage 4 'confirm' recipe only (d=192, full permissive train, 1 epoch), LR 1e-3 (the dev choice), "
                       "seeds 1-5, no LR sweep, GPU fp16 as on dev; see colab/FINAL_TEST.md",
           "waves": waves}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", OUT, [(e["id"], e["final_test_role"]) for e in waves])


if __name__ == "__main__":
    main()
