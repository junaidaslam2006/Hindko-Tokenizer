# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 3: download TOKENIZER FILES ONLY for every new tokenizer signature of stage 2.

Same conventions as baselines/scripts/download.py: anonymous (token=False), no weights, revision pinned to the commit sha
the hub reports for the file, files <= 60 MB, allow-listed names only, sha256 recorded.  One representative repo per
signature (the member with the most downloads + 50 x likes); the other members are recorded as sharing the file.
Output: ../files/<owner>__<repo>[__<dir>]/..., ../_download_log.json (resumable).
Usage: python download_refresh.py [probe json ...]    (default ../_probe.json)
"""
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HOME"] = r"F:\Hindko\_tokenizer\hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"   # plain HTTPS via the CDN redirect: no xet-read-token call on the rate-limited api quota
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = os.path.join(ROOT, "files")
LOG = os.path.join(ROOT, "_download_log.json")
MAX_BYTES = 60 * 1024 * 1024
AUX = ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]   # same allow-list as baselines/scripts/registry.py
_lock = threading.Lock()
COMMITS = {}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slug(repo, d):
    s = repo.replace("/", "__")
    return s + ("__" + d.replace("/", "_") if d else "")


def score(m):
    return (m.get("downloads") or 0) + 50 * (m.get("likes") or 0)


def license_from_card(repo, rev):
    """License from the model card's YAML front matter (README.md read as text into a temp dir, then deleted)."""
    import re
    import tempfile
    try:
        with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "logs")) as td:
            p = hf_hub_download(repo, "README.md", revision=rev, local_dir=td, token=False)
            txt = open(p, encoding="utf-8", errors="replace").read(20000)
        m = re.match(r"^---\s*\n(.*?)\n---", txt, re.S)
        if not m:
            return None
        fm = m.group(1)
        lic = re.search(r"^license:\s*(.+)$", fm, re.M)
        name = re.search(r"^license_name:\s*(.+)$", fm, re.M)
        v = lic.group(1).strip().strip("'\"") if lic else None
        if v in (None, "other") and name:
            v = "other:" + name.group(1).strip().strip("'\"")
        return v
    except Exception:  # noqa: BLE001
        return None


def fetch(g):
    rep = max(g["members"], key=score)
    repo, d, files = rep["repo"], rep["dir"], rep["files"]
    rec = {"sig": g["sig"], "kind": g["kind"], "repo": repo, "dir": d, "downloads": rep.get("downloads"), "likes": rep.get("likes"),
           "created": rep.get("created"), "pipe": rep.get("pipe"), "how": rep.get("how"),
           "n_members": len(g["members"]), "members": [m["repo"] + ("/" + m["dir"] if m["dir"] else "") for m in g["members"]][:50],
           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if any((s or 0) > MAX_BYTES for s in g["sizes"]):
        rec["error"] = "tokenizer file larger than 60 MB; refused"
        return rec
    out = os.path.join(FILES, slug(repo, d))
    try:
        rev = COMMITS.get((repo, files[0]))       # the commit the probe's HEAD reported for this very file
        if rev is None:
            meta = None
            for t in range(5):
                try:
                    meta = get_hf_file_metadata(hf_hub_url(repo, files[0]), token=False)
                    break
                except Exception as e:  # noqa: BLE001
                    if "429" in str(e):
                        time.sleep(30 * (t + 1))
                    else:
                        raise
            rev = meta.commit_hash
        rec["revision"] = rev
        got = {}
        for f in files:
            p = hf_hub_download(repo, f, revision=rev, local_dir=out, token=False)
            got[f] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        for a in ([] if g["kind"] == "tokenizer.json" else AUX):   # tokenizer.json is self-contained
            f = (d + "/" + a) if d else a
            try:
                p = hf_hub_download(repo, f, revision=rev, local_dir=out, token=False)
                if os.path.getsize(p) > 5 * 1024 * 1024:
                    os.remove(p)
                    continue
                got[f] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
            except Exception:  # noqa: BLE001
                pass
        rec["files"] = got
        rec["local_dir"] = os.path.relpath(out, ROOT).replace("\\", "/")
    except Exception as e:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200])
    return rec


def main():
    args = sys.argv[1:]
    max_mb = float(args[args.index("--max-mb") + 1]) if "--max-mb" in args else None      # size tier
    only = set(json.load(open(args[args.index("--only") + 1], encoding="utf-8"))) if "--only" in args else None  # sig list
    srcs = [a for a in args if a.endswith(".json") and (not only or a != args[args.index("--only") + 1])] or         [os.path.join(ROOT, "_probe.json")]
    for line in open(os.path.join(ROOT, "_probe_cache.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        for f, m in (x.get("meta") or {}).items():
            if m.get("commit"):
                COMMITS[(x["repo"], f)] = m["commit"]
    log = json.load(open(LOG, encoding="utf-8")) if os.path.exists(LOG) else {}
    groups = []
    for s in srcs:
        groups += json.load(open(s, encoding="utf-8"))["groups"]
    todo = [g for g in groups if g["sig"] not in log or "error" in log[g["sig"]] and "60 MB" not in log[g["sig"]]["error"]]
    if max_mb is not None:
        todo = [g for g in todo if sum(x or 0 for x in g["sizes"]) <= max_mb * 1e6]
    if only is not None:
        todo = [g for g in todo if g["sig"] in only]
    # regional (Pakistan-language keyword / language-tag) signatures first, then the rest by popularity
    todo.sort(key=lambda g: (not any(h.split(":")[0] in ("kw", "lang", "kw2", "named", "named2") and
                                     not h.startswith(("kw:indic", "kw:bharat", "kw2:muse", "kw2:Nemotron"))
                                     for m in g["members"] for h in (m.get("how") or "").split("|")),))
    print("signatures", len(groups), "to download", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        for i, rec in enumerate(ex.map(fetch, todo)):
            with _lock:
                log[rec["sig"]] = rec
                if i % 25 == 0:
                    json.dump(log, open(LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
                    print(i, "%.0fs" % (time.time() - t0), rec["repo"], rec.get("error", ""), flush=True)
    json.dump(log, open(LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("done", len(log), "errors", sum(1 for v in log.values() if "error" in v), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
