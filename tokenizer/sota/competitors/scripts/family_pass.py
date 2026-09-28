# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 3d: choose which still-unresolved large tokenizer.json files to download.

Input: the files that stages 3b/3c left as needs_download.  For each one this reads the head of the file (up to the
start of its vocabulary) and hashes its first 300 vocabulary entries = its 'family'.  Downloaded in full:
  - one representative (most downloads + 50 x likes) of every family, and
  - every file whose sampled vocabulary strings are >= 3 % Arabic-script (triage2 'arabic_share_sampled'), i.e. every
    plausible Perso-Arabic-specialised vocabulary, whatever its family.
After these downloads, triage2.py --rerun re-checks the other members against the new references.
Output: ../_family.json and ../logs/_only_family.json (signature list for download_refresh.py --only)
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import range_triage as T  # noqa: E402
import triage2 as T2  # noqa: E402

ROOT = T.ROOT


def unresolved():
    t1 = {r["sig"]: r for r in json.load(open(os.path.join(ROOT, "_triage.json"), encoding="utf-8"))["results"]}
    final = dict(t1)
    for f in ("_triage2.json", "_triage2b.json", "_triage2c.json"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            for r in json.load(open(p, encoding="utf-8"))["results"]:
                final[r["sig"]] = dict(final.get(r["sig"], {}), **r)
    return [r for r in final.values() if r["status"] == "needs_download"], t1


def head_family(args):
    r, t1 = args
    url = T.url_of(r["repo"], t1[r["sig"]]["file"], t1[r["sig"]].get("commit"))
    out = {"sig": r["sig"], "repo": r["repo"], "size": r["size"], "arabic_share_sampled": r.get("arabic_share_sampled")}
    try:
        head, off = T.head_until_model(url, r["sig"])
        if off is None:
            out["family"] = None
            return out
        fv = T2.first_vocab(head[off:].decode("utf-8", errors="ignore"))
        out["family"] = hashlib.sha256(json.dumps(fv).encode()).hexdigest()[:16] if fv else None
        out["first_tokens"] = list(fv[:8]) if fv else None
    except Exception as e:  # noqa: BLE001
        out["family"] = None
        out["error"] = "%s: %s" % (type(e).__name__, str(e)[:100])
    return out


def main():
    todo, t1 = unresolved()
    probes = {}
    for f in ("_probe.json", "_probe2.json"):
        for g in json.load(open(os.path.join(ROOT, f), encoding="utf-8"))["groups"]:
            probes.setdefault(g["sig"], g)
    with ThreadPoolExecutor(10) as ex:
        res = list(ex.map(head_family, [(r, t1) for r in todo]))
    fams = {}
    for x in res:
        g = probes[x["sig"]]
        x["score"] = max((m.get("downloads") or 0) + 50 * (m.get("likes") or 0) for m in g["members"])
        fams.setdefault(x["family"] or ("single:" + x["sig"]), []).append(x)
    pick = set()
    for k, xs in fams.items():
        pick.add(max(xs, key=lambda x: x["score"])["sig"])
    ar = {x["sig"] for x in res if (x.get("arabic_share_sampled") or 0) >= 0.03}
    pick |= ar
    json.dump({"n_unresolved": len(res), "n_families": len(fams), "n_arabic_heavy": len(ar), "n_to_download": len(pick),
               "mb_to_download": round(sum(x["size"] for x in res if x["sig"] in pick) / 1e6, 1),
               "families": {k: [x["repo"] for x in v] for k, v in fams.items()}, "results": res},
              open(os.path.join(ROOT, "_family.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(sorted(pick), open(os.path.join(ROOT, "logs", "_only_family.json"), "w"))
    print("unresolved", len(res), "families", len(fams), "arabic-heavy", len(ar), "to download", len(pick),
          "MB", round(sum(x["size"] for x in res if x["sig"] in pick) / 1e6, 1), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
