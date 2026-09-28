# -*- coding: utf-8 -*-
"""Refreshed competitor sweep: gated repos (not logged into, never downloaded).  The hub still exposes file metadata
(git blob id / LFS sha256) of a gated repo without login; this compares each gated repo's defining tokenizer file with
every file already on disk (baselines/files and sota/competitors/files).  A match means the tokenizer WAS evaluated
through an ungated byte-identical copy; no match means it is 'not collected: gated'.
Output: ../_gated.json
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe as P  # noqa: E402

ROOT = P.ROOT
REGIONAL = ("kw:hindko", "kw:hazara", "kw:pahari", "kw:saraiki", "kw:siraiki", "kw:multani", "kw:shahmukhi", "kw:punjabi",
            "kw:panjabi", "kw:pnb", "kw:urdu", "kw:pashto", "kw:pushto", "kw:sindhi", "kw:kashmiri", "kw:balochi", "kw:baluchi",
            "kw:khowar", "kw:shina", "kw:pakistan", "kw:peshawar", "kw:nastaliq", "kw:perso-arabic", "kw:indic", "kw:bharat",
            "lang:hno", "lang:hnd", "lang:ur", "lang:urd", "lang:pnb", "lang:skr", "lang:ps", "lang:pus", "lang:pbt", "lang:pst",
            "lang:sd", "lang:snd", "lang:ks", "lang:kas", "lang:bal", "lang:bgp", "lang:scl", "named", "named2", "kw2:", "author2:")


def local_hashes():
    kb, ks = P.known_hashes()
    for dp, _, fns in os.walk(os.path.join(ROOT, "files")):
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.getsize(p) > 80 * 1024 * 1024 or ".cache" in p:
                continue
            data = open(p, "rb").read()
            rel = "refresh:" + os.path.relpath(p, os.path.join(ROOT, "files")).replace("\\", "/")
            kb.setdefault(P.git_blob_sha1(data), rel)
            ks.setdefault(hashlib.sha256(data).hexdigest(), rel)
    return kb, ks


def main():
    rows = []
    for f in ("_search.json", "_search2.json"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            rows += json.load(open(p, encoding="utf-8"))["repos"]
    gated = [r for r in rows if r.get("gated") and r["tok_files"] and r.get("pipe") not in P.NONTEXT
             and any(h.startswith(REGIONAL) for h in r["how"].split("|"))]
    kb, ks = local_hashes()
    out = []
    for r in gated:
        rec = {"repo": r["repo"], "gated": r["gated"], "downloads": r["downloads"], "created": r["created"], "how": r["how"]}
        for t in range(4):
            try:
                info = P.api.model_info(r["repo"], files_metadata=True, token=False)
                sib = {s.rfilename: s for s in (info.siblings or [])}
                files = P.defining_files(r["tok_files"])
                res = []
                for fn in files:
                    s = sib.get(fn)
                    if s is None:
                        continue
                    blob, lfs = s.blob_id, (s.lfs.sha256 if s.lfs else None)
                    res.append({"file": fn, "size": s.size, "blob_id": blob, "lfs_sha256": lfs,
                                "identical_to": kb.get(blob) or (ks.get(lfs) if lfs else None)})
                rec["files"] = res
                rec["all_identical_to_collected"] = bool(res) and all(x["identical_to"] for x in res)
                rec["license"] = (info.card_data or {}).get("license") if info.card_data else None
                break
            except Exception as e:  # noqa: BLE001
                rec["error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
                if "429" in rec["error"]:
                    time.sleep(60 * (t + 1))
                else:
                    break
        out.append(rec)
        print(rec["repo"], rec.get("all_identical_to_collected"), [x.get("identical_to") for x in rec.get("files", [])],
              rec.get("error", ""), flush=True)
    json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_gated": len(out), "repos": out},
              open(os.path.join(ROOT, "_gated.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
