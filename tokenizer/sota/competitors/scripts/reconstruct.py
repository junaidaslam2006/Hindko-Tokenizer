# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 3e: rebuild (instead of download) large tokenizer.json files whose "model" section
stage 3b found identical to a tokenizer on disk but whose pipeline or added tokens differ.

tokenizers serialises the top-level keys in a fixed order and "model" is the last one, so such a file is exactly
    <its own first bytes, up to the start of "model">  +  <the reference file from the start of its "model" to the end>
The first part comes from the range read of stage 3b/3d (cached in logs/heads/), the second from the reference on
disk.  The rebuilt file is accepted only if its length equals the hub's size for the file and the whole thing parses and
loads with `tokenizers`; it is written to files/<owner>__<repo>/tokenizer.json and entered into _download_log.json with
"reconstructed_from" so the report can say so.  It is then screened and measured like any downloaded file.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import range_triage as T  # noqa: E402
import download_refresh as D  # noqa: E402

ROOT = T.ROOT


def main():
    tr = json.load(open(os.path.join(ROOT, "_triage.json"), encoding="utf-8"))["results"]
    log = json.load(open(D.LOG, encoding="utf-8"))
    probes = {}
    for f in ("_probe.json", "_probe2.json"):
        for g in json.load(open(os.path.join(ROOT, f), encoding="utf-8"))["groups"]:
            probes.setdefault(g["sig"], g)
    todo = [r for r in tr if r["status"] == "needs_download" and r["sig"] not in log and
            (r.get("reason", "").startswith("same model section as") or r.get("reason", "").startswith("added tokens occurring"))]
    print("to rebuild", len(todo), flush=True)
    done = 0
    for r in todo:
        reason = r["reason"]
        if reason.startswith("same model section as"):
            ref = reason[len("same model section as "):].split(" but ")[0]
        else:
            continue            # 'added tokens occurring' rows carry no reference path in stage 3b; they are downloaded
        refp = os.path.join(T.TOK, ref)
        try:
            head, off = T.head_until_model(T.url_of(r["repo"], r["file"], r.get("commit")), r["sig"])
            if off is None:
                raise RuntimeError("model section not found in the head")
            rb = open(refp, "rb").read()
            roff = T.model_offset(rb[:4 * 1024 * 1024])
            new = head[:off] + rb[roff:]
            if len(new) != r["size"]:
                raise RuntimeError("rebuilt length %d != hub size %d" % (len(new), r["size"]))
            from tokenizers import Tokenizer
            Tokenizer.from_str(new.decode("utf-8"))
            g = probes[r["sig"]]
            rep = max(g["members"], key=D.score)
            out = os.path.join(D.FILES, D.slug(rep["repo"], rep["dir"]))
            os.makedirs(os.path.join(out, rep["dir"]) if rep["dir"] else out, exist_ok=True)
            p = os.path.join(out, rep["files"][0])
            open(p, "wb").write(new)
            log[r["sig"]] = {"sig": r["sig"], "kind": "tokenizer.json", "repo": rep["repo"], "dir": rep["dir"],
                             "downloads": rep.get("downloads"), "likes": rep.get("likes"), "created": rep.get("created"),
                             "pipe": rep.get("pipe"), "how": rep.get("how"), "n_members": len(g["members"]),
                             "members": [m["repo"] + ("/" + m["dir"] if m["dir"] else "") for m in g["members"]][:50],
                             "revision": r.get("commit"), "files": {rep["files"][0]: {"bytes": len(new), "sha256": D.sha256(p)}},
                             "local_dir": os.path.relpath(out, ROOT).replace("\\", "/"),
                             "reconstructed_from": ref, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            done += 1
            print("rebuilt", rep["repo"], "from", ref, flush=True)
        except Exception as e:  # noqa: BLE001
            print("FAILED", r["repo"], type(e).__name__, str(e)[:150], flush=True)
    json.dump(log, open(D.LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("rebuilt", done, "of", len(todo), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
