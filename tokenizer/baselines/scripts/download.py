"""Download TOKENIZER FILES ONLY for every registry entry (anonymous; never logs in; never downloads weights).

- Files are chosen from an allow-list (registry.ALLOW ...) intersected with what exists at the repo root.
- Revision is pinned to the commit sha returned by the hub at download time.
- Output: baselines/files/<name>/<file>, and scripts/_download_log.json with sha256 of every file.
Usage: python download.py [name ...]
"""
import hashlib
import json
import os
import sys
import time

os.environ["HF_HOME"] = r"F:\Hindko\_tokenizer\hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import registry  # noqa: E402

BASE = r"F:\Hindko\_tokenizer\baselines"
FILES = os.path.join(BASE, "files")
LOG = os.path.join(BASE, "scripts", "_download_log.json")
MAX_BYTES = 60 * 1024 * 1024  # guard: no tokenizer file is this large; anything bigger is refused

api = HfApi()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def license_of(info):
    lic = None
    if info.card_data is not None:
        lic = info.card_data.get("license")
        if lic in ("other", None) and info.card_data.get("license_name"):
            lic = f"other:{info.card_data.get('license_name')}"
    if lic is None:
        lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), None)
    return lic


def main():
    want = set(sys.argv[1:])
    log = json.load(open(LOG, encoding="utf-8")) if os.path.exists(LOG) else {}
    for e in registry.ENTRIES:
        name = e["name"]
        if want and name not in want:
            continue
        if e["repo"].startswith("local:"):
            continue
        rec = {"repo": e["repo"], "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            info = api.model_info(e["repo"], files_metadata=True)
        except Exception as ex:  # noqa: BLE001
            rec["error"] = f"model_info: {type(ex).__name__}: {str(ex).splitlines()[0][:200]}"
            log[name] = rec
            print(name, "ERROR", rec["error"], flush=True)
            continue
        rec.update(revision=info.sha, gated=info.gated, license=license_of(info),
                   repo_created=str(getattr(info, "created_at", None))[:10], repo_last_modified=str(info.last_modified)[:10])
        root = {s.rfilename: s for s in (info.siblings or []) if "/" not in s.rfilename}
        names = [f for f in registry.ALLOW if f in root]
        if "tokenizer.json" not in root:
            names += [f for f in registry.ALLOW_IF_NO_JSON if f in root]
        names += [f for f in registry.ALLOW_EXTRA.get(e["loader"], []) if f in root]
        if info.gated:
            rec["error"] = f"repo is gated ({info.gated}); not logging in"
            log[name] = rec
            print(name, "SKIP gated", flush=True)
            continue
        got = {}
        for fn in names:
            size = root[fn].size
            if size is not None and size > MAX_BYTES:
                got[fn] = {"skipped": f"size {size} > guard"}
                continue
            p = hf_hub_download(e["repo"], fn, revision=info.sha, local_dir=os.path.join(FILES, name))
            got[fn] = {"bytes": os.path.getsize(p), "sha256": sha256(p),
                       "hub_blob_id": root[fn].blob_id, "hub_lfs_sha256": root[fn].lfs.sha256 if root[fn].lfs else None}
        rec["files"] = got
        log[name] = rec
        print(name, info.sha[:10], {k: v.get("bytes") for k, v in got.items()}, flush=True)
        json.dump(log, open(LOG, "w", encoding="utf-8"), indent=1, ensure_ascii=False, sort_keys=True)
    json.dump(log, open(LOG, "w", encoding="utf-8"), indent=1, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
