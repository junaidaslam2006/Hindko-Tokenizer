# -*- coding: utf-8 -*-
"""Describe the REMOVE events of a PickyBPE model (training data only; no dev or test text is read).

    python removal_report.py --model models/X.json --out verify/X.removals.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe as PB  # noqa: E402


def txt(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "<partial-utf8 %s>" % b.hex(" ")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path = a.model if os.path.isabs(a.model) else os.path.join(HERE, a.model)
    m = json.load(open(path, encoding="utf-8"))
    tb = [None if h is None else bytes.fromhex(h) for h in m["tokens"]]
    final = set(m["final_ids"])
    rows = []
    last_merge = None
    for k, e in enumerate(m["events"]):
        if e[0] == 0:
            last_merge = e
            continue
        x, split, occ, fx, fp = e[1], e[2], e[3], e[4], e[5]
        a_, b_, c_ = last_merge[1], last_merge[2], last_merge[3]
        rows.append({"event": k, "token": txt(tb[x]), "partial_utf8": not _valid(tb[x]),
                     "f_t_before": fx, "f_p": fp, "ios": round(fp / fx, 4), "left_after": occ,
                     "merged_into": txt(tb[c_]), "role": "left" if x == a_ else "right",
                     "split": [txt(tb[s]) for s in split], "present_at_end": x in final,
                     "merge_index": sum(1 for q in m["events"][:k] if q[0] == 0)})
    ios = [r["ios"] for r in rows]
    hist = collections.Counter(("1.0" if v >= 1 else "%.2f-%.2f" % (int(v * 50) / 50, int(v * 50) / 50 + 0.02))
                               for v in ios)
    readded = [r for r in rows if r["present_at_end"]]
    rep = {"model": path, "tau": m["tau"], "removal_events": len(rows),
           "distinct_tokens_removed": len({r["token"] for r in rows}),
           "removed_at_end": m["summary"]["removed_at_end"], "readded_and_present_at_end": len({r["token"] for r in readded}),
           "partial_utf8_removed": sum(1 for r in rows if r["partial_utf8"]),
           "no_occurrence_left (IoS = 1)": sum(1 for r in rows if r["left_after"] == 0),
           "occurrences_split_total": sum(r["left_after"] for r in rows),
           "role": dict(collections.Counter(r["role"] for r in rows)),
           "split_length": dict(collections.Counter(len(r["split"]) for r in rows)),
           "ios_histogram": dict(sorted(hist.items())),
           "f_t_before_quartiles": _quart([r["f_t_before"] for r in rows]),
           "left_after_quartiles": _quart([r["left_after"] for r in rows]),
           "removal_merge_index_quartiles": _quart([r["merge_index"] for r in rows]),
           "readded_examples": readded[:12],
           "largest_f_t": sorted(rows, key=lambda r: -r["f_t_before"])[:15],
           "first_20": rows[:20], "all": rows}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("readded_examples", "largest_f_t", "first_20", "all")},
                     ensure_ascii=False, indent=1))
    for r in rep["largest_f_t"][:10]:
        print(json.dumps({k: r[k] for k in ("token", "f_t_before", "f_p", "ios", "left_after", "merged_into", "split")},
                         ensure_ascii=False))
    for r in rep["readded_examples"][:6]:
        print("readded:", json.dumps({k: r[k] for k in ("token", "ios", "merged_into", "merge_index")}, ensure_ascii=False))


def _valid(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _quart(v):
    if not v:
        return None
    s = sorted(v)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]
    return {"min": s[0], "q1": q(.25), "median": q(.5), "q3": q(.75), "max": s[-1]}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
