# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 2: file metadata (git blob id / LFS sha256 / size) of the tokenizer files of every
repo found in stage 1, and a content signature per tokenizer directory. Anonymous (token=False), metadata only.

A signature is the blob id of the file that defines the tokenizer in that directory, in this order of preference:
tokenizer.json > tekken.json > tiktoken.model / *.tiktoken > SentencePiece model > vocab.json+merges.txt > vocab.txt >
Marian source.spm > xAI tokenizer.tok.json.  A signature is KNOWN when that file's content is byte-identical to a file
already on disk under baselines/files/ (the 66 baselines, the 45 files of the 2026-09-26 Urdu scan, grok-2).
Output: ../_probe.json (resumable; per-repo records are cached in ../_probe_cache.jsonl).
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
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOK = r"F:\Hindko\_tokenizer"
BASEFILES = os.path.join(TOK, "baselines", "files")
SEARCH = os.path.join(ROOT, "_search.json")
CACHE = os.path.join(ROOT, "_probe_cache.jsonl")
OUT = os.path.join(ROOT, "_probe.json")
api = HfApi(token=False)
NONTEXT = {"automatic-speech-recognition", "text-to-speech", "text-to-audio", "audio-to-audio", "audio-classification",
           "text-to-image", "image-to-image", "text-to-video", "image-classification", "object-detection", "image-segmentation",
           "voice-activity-detection", "image-to-3d", "text-to-3d", "depth-estimation", "unconditional-image-generation"}
PREF = ["tokenizer.json", "tekken.json", "tiktoken.model", "*.tiktoken", "spm", "vocab.json+merges.txt", "vocab.txt", "source.spm",
        "tokenizer.tok.json"]
SPM_NAMES = ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "sentencepiece.model", "spm.model", "bpe.model")


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def known_hashes():
    blobs, shas = {}, {}
    for dp, _, fns in os.walk(BASEFILES):
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.getsize(p) > 80 * 1024 * 1024:
                continue
            data = open(p, "rb").read()
            rel = os.path.relpath(p, BASEFILES).replace("\\", "/")
            blobs.setdefault(git_blob_sha1(data), rel)
            shas.setdefault(hashlib.sha256(data).hexdigest(), rel)
    return blobs, shas


def pick(files_in_dir):
    names = {os.path.basename(f): f for f in files_in_dir}
    if "tokenizer.json" in names:
        return "tokenizer.json", [names["tokenizer.json"]]
    if "tekken.json" in names:
        return "tekken.json", [names["tekken.json"]]
    if "tiktoken.model" in names:
        return "tiktoken.model", [names["tiktoken.model"]]
    tk = [f for n, f in names.items() if n.endswith(".tiktoken")]
    if tk:
        return "*.tiktoken", tk[:1]
    sp = [names[n] for n in SPM_NAMES if n in names] + [f for n, f in names.items() if n.endswith(".model") and "token" in n.lower()
                                                         and n not in SPM_NAMES]
    if sp:
        return "spm", sp[:1]
    if "vocab.json" in names and "merges.txt" in names:
        return "vocab.json+merges.txt", [names["vocab.json"], names["merges.txt"]]
    if "vocab.txt" in names:
        return "vocab.txt", [names["vocab.txt"]]
    if "source.spm" in names:
        return "source.spm", [names["source.spm"]]
    if "tokenizer.tok.json" in names:
        return "tokenizer.tok.json", [names["tokenizer.tok.json"]]
    return None, []


_lock = threading.Lock()


def defining_files(tok_files):
    dirs = {}
    for f in tok_files:
        dirs.setdefault(os.path.dirname(f), []).append(f)
    out = []
    for d in dirs:
        out += pick(dirs[d])[1]
    return out


def probe_one(r):
    """HEAD on the resolve URL of each defining tokenizer file (the hub's 'resolvers' quota, 3000/5 min, not the
    'api' quota): ETag = git blob sha1 (regular file) or X-Linked-ETag = sha256 (LFS/xet file), size, commit."""
    meta = {}
    for f in defining_files(r["tok_files"]):
        for t in range(6):
            try:
                m = get_hf_file_metadata(hf_hub_url(r["repo"], f), token=False)
                et = (m.etag or "").strip('"')
                if et.startswith("W/"):
                    et = et[2:].strip('"')
                lfs = len(et) == 64
                meta[f] = {"blob_id": None if lfs else et, "lfs_sha256": et if lfs else None, "size": m.size,
                           "commit": m.commit_hash}
                break
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                msg = str(e).splitlines()[0][:160] if str(e) else name
                if name in ("GatedRepoError", "RepositoryNotFoundError", "DisabledRepoError", "EntryNotFoundError",
                            "RemoteEntryNotFoundError"):
                    return {"repo": r["repo"], "error": "%s: %s" % (name, msg)}
                if "429" in msg or "Too Many" in msg:
                    time.sleep(30 * (t + 1))
                else:
                    time.sleep(3 * (t + 1))
        else:
            return {"repo": r["repo"], "error": "gave up after retries"}
    return {"repo": r["repo"], "meta": meta, "via": "resolve HEAD"}


def main():
    global SEARCH, OUT
    if "--search" in sys.argv:              # e.g. --search ../_search2.json --out ../_probe2.json (cache is shared)
        SEARCH = os.path.abspath(sys.argv[sys.argv.index("--search") + 1])
        OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    srch = json.load(open(SEARCH, encoding="utf-8"))
    repos = [r for r in srch["repos"] if r["tok_files"]]
    skipped = {"gated": [], "non_text_pipeline": []}
    todo = []
    for r in repos:
        if r.get("gated"):
            skipped["gated"].append(r["repo"])
        elif r.get("pipe") in NONTEXT:
            skipped["non_text_pipeline"].append(r["repo"])
        else:
            todo.append(r)
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            x = json.loads(line)
            if "error" not in x or x["error"].startswith(("GatedRepoError", "RepositoryNotFoundError")):
                done[x["repo"]] = x
    rest = [r for r in todo if r["repo"] not in done]
    print("repos with tokenizer files", len(repos), "to probe", len(todo), "cached", len(done), "remaining", len(rest), flush=True)
    t0 = time.time()
    out_path = OUT
    if "--from-cache" in sys.argv:          # interim grouping of what is cached so far; no network
        rest, out_path = [], OUT.replace(".json", "_interim.json")
        todo = [r for r in todo if r["repo"] in done]
    with open(CACHE, "a", encoding="utf-8") as fc, ThreadPoolExecutor(3) as ex:
        for i, x in enumerate(ex.map(probe_one, rest)):
            with _lock:
                fc.write(json.dumps(x, ensure_ascii=False) + "\n")
                fc.flush()
            done[x["repo"]] = x
            if i % 250 == 0:
                print(i, "%.0fs" % (time.time() - t0), flush=True)
    kb, ks = known_hashes()
    byrepo = {r["repo"]: r for r in repos}
    sigs, per_repo = {}, []
    for r in todo:
        x = done.get(r["repo"])
        if not x or "error" in x:
            per_repo.append({"repo": r["repo"], "error": (x or {}).get("error", "not probed")})
            continue
        dirs = {}
        for f in x["meta"]:
            dirs.setdefault(os.path.dirname(f), []).append(f)
        recs = []
        for d in sorted(dirs, key=lambda s: (s != "", s)):
            kind, files = pick(dirs[d])
            if kind is None:
                continue
            m = [x["meta"][f] for f in files]
            sig = "+".join(q["lfs_sha256"] or q["blob_id"] for q in m)
            known = [kb.get(q["blob_id"]) or (ks.get(q["lfs_sha256"]) if q["lfs_sha256"] else None) for q in m]
            rec = {"dir": d, "kind": kind, "files": files, "sizes": [q["size"] for q in m], "sig": sig,
                   "known_as": known[0] if all(known) else None}
            recs.append(rec)
            if rec["known_as"] is None:
                g = sigs.setdefault(sig, {"sig": sig, "kind": kind, "sizes": rec["sizes"], "members": []})
                g["members"].append({"repo": r["repo"], "dir": d, "files": files, "downloads": r.get("downloads"),
                                     "likes": r.get("likes"), "created": r.get("created"), "pipe": r.get("pipe"), "how": r.get("how")})
        per_repo.append({"repo": r["repo"], "dirs": recs})
    groups = sorted(sigs.values(), key=lambda g: -max((m["downloads"] or 0) + 50 * (m["likes"] or 0) for m in g["members"]))
    n_known = sum(1 for p in per_repo if p.get("dirs") and all(d["known_as"] for d in p["dirs"]))
    json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_repos_with_tok_files": len(repos),
               "n_probed": len(todo), "skipped": {k: len(v) for k, v in skipped.items()}, "skipped_repos": skipped,
               "n_errors": sum(1 for p in per_repo if "error" in p), "n_repos_all_known": n_known,
               "n_new_signatures": len(groups), "groups": groups, "per_repo": per_repo},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("probed", len(todo), "errors", sum(1 for p in per_repo if "error" in p), "all-known repos", n_known,
          "new signatures", len(groups), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
