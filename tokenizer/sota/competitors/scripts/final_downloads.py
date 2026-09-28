# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 3f: the last download decision for large tokenizer.json files still unresolved
after stages 3b-3e.

Downloaded in full (-> logs/_only_final.json):
  A. every file whose sampled vocabulary strings are >= 3 % Perso-Arabic script (a plausible Hindko competitor);
  B. every file whose Perso-Arabic share could not be sampled (range read failed or no "model" section in 2.5 MB);
  C. one representative per vocabulary family (same first 300 entries) whose sampled Perso-Arabic share reaches 0.5 %
     somewhere in the family and which has no measured reference on disk.
Not downloaded (-> ../_not_downloaded.json, with the evidence): the rest, i.e. files whose sampled vocabulary is < 3 %
Perso-Arabic AND whose family is either already represented by a measured tokenizer (first 300 entries identical) or
shows < 0.5 % Perso-Arabic strings in every sampled member.  Such a vocabulary cannot approach the released tokenizer
on Hindko (every measured tokenizer with < 3 % Perso-Arabic vocabulary stays far below 5 bytes/token here).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import family_pass as F  # noqa: E402

ROOT = F.ROOT


def main():
    todo, t1 = F.unresolved()
    dl = json.load(open(os.path.join(ROOT, "_download_log.json"), encoding="utf-8"))
    todo = [r for r in todo if r["sig"] not in dl]
    fam = {x["sig"]: x for x in json.load(open(os.path.join(ROOT, "_family.json"), encoding="utf-8"))["results"]}
    probes = {}
    for f in ("_probe.json", "_probe2.json"):
        for g in json.load(open(os.path.join(ROOT, f), encoding="utf-8"))["groups"]:
            probes.setdefault(g["sig"], g)
    pick, reasons, fams = set(), {}, {}
    import range_triage as T
    import triage2 as T2
    for r in todo:
        if r.get("arabic_share_sampled") is None:
            # no "model" section within 2.5 MB (huge added-token lists of multimodal models): the last 32 KB of the
            # file are always inside the model section, so sample the Perso-Arabic share there
            try:
                t = t1[r["sig"]]
                tail = T.cached_rng(T.url_of(r["repo"], t["file"], t.get("commit")), r["sig"], -32 * 1024)
                samp = T2.strings(tail)[:-1]
                if len(samp) >= 200:
                    r["arabic_share_sampled"] = T2.arabic_share(samp)
                    r["arabic_share_from"] = "last 32 KB only"
            except Exception as e:  # noqa: BLE001
                r["tail_error"] = "%s: %s" % (type(e).__name__, str(e)[:80])
        ar = r.get("arabic_share_sampled")
        f = (fam.get(r["sig"]) or {}).get("family") or ("single:" + r["sig"])
        fams.setdefault(f, []).append(r)
        if ar is None:
            pick.add(r["sig"])
            reasons[r["sig"]] = "B: Perso-Arabic share not sampled (%s)" % (r.get("reason") or "")[:80]
        elif ar >= 0.03:
            pick.add(r["sig"])
            reasons[r["sig"]] = "A: sampled vocabulary %.1f %% Perso-Arabic" % (100 * ar)
    not_dl = []
    for f, rs in fams.items():
        has_ref = any(x.get("family_reference") for x in rs)
        mx = max((x.get("arabic_share_sampled") or 0) for x in rs)
        rest = [x for x in rs if x["sig"] not in pick]
        if not rest:
            continue
        if not has_ref and mx >= 0.005 and not any(x["sig"] in pick for x in rs):
            rep = max(rest, key=lambda x: max((m.get("downloads") or 0) + 50 * (m.get("likes") or 0)
                                               for m in probes[x["sig"]]["members"]))
            pick.add(rep["sig"])
            reasons[rep["sig"]] = "C: representative of family %s (max sampled Perso-Arabic %.1f %%)" % (f[:16], 100 * mx)
            rest = [x for x in rest if x["sig"] != rep["sig"]]
        for x in rest:
            not_dl.append({"sig": x["sig"], "repo": x["repo"], "size": x["size"],
                           "arabic_share_sampled": x.get("arabic_share_sampled"), "family": f,
                           "family_reference": x.get("family_reference"), "family_max_arabic_share": mx,
                           "why_unresolved": x.get("reason"),
                           "n_repos_with_this_file": len(probes[x["sig"]]["members"])})
    json.dump(sorted(pick), open(os.path.join(ROOT, "logs", "_only_final.json"), "w"))
    json.dump({"rule": __doc__, "picked": {s: reasons[s] for s in sorted(pick)}}, open(os.path.join(ROOT, "_final_picks.json"), "w",
                                                                                      encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(not_dl, open(os.path.join(ROOT, "_not_downloaded.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("unresolved", len(todo), "download", len(pick), "MB", round(sum(r["size"] for r in todo if r["sig"] in pick) / 1e6, 1),
          "not downloaded", len(not_dl), "max Perso-Arabic share among them",
          max([x["arabic_share_sampled"] or 0 for x in not_dl] or [0]), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
