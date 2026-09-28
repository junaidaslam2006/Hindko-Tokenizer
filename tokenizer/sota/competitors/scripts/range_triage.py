# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 3b: bandwidth-saving triage of LARGE tokenizer.json files (> 5 MB).

The link to the hub measured ~0.25-0.75 MB/s, so the ~13.5 GB of large tokenizer.json files (mostly re-saved copies of
Qwen / Llama / Gemma / Mistral tokenizers inside fine-tunes) cannot all be downloaded. For each large file this reads
three byte ranges over HTTPS (anonymous; resolvers quota): the first 128 KB, 16 KB from the middle of the "model"
section, and the last 32 KB.  A file is declared IDENTICAL IN BEHAVIOUR to a reference tokenizer.json already on disk
(baselines/files or sota/competitors/files) when ALL of these hold:
  1. its "model" section has the same byte length as the reference's, and its middle 16 KB and last 32 KB are
     byte-identical to the reference's (so vocabulary / merges / scores are, to overwhelming probability, identical);
  2. normalizer, pre_tokenizer and decoder parse to the same JSON as the reference's;
  3. no added token that the reference lacks occurs anywhere in the test text (an added token is matched in raw text
     before the model, so one that never occurs cannot change an encoding).
Everything else (no reference match, a different pipeline, an added token that occurs in the text, a model section
not reachable in the first 1 MB) is marked needs_download and is downloaded in full.
Output: ../_triage.json
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOK = r"F:\Hindko\_tokenizer"
UA = {"User-Agent": "Mozilla/5.0 (compatible; tokenizer-benchmark-script)"}
HEAD_N, MID_N, TAIL_N = 128 * 1024, 16 * 1024, 32 * 1024
MODEL_RE = [re.compile(rb'\n  "model": \{'), re.compile(rb',"model":\{'), re.compile(rb'\n "model": \{'),
            re.compile(rb'\n\t"model": \{'), re.compile(rb',\s*"model"\s*:\s*\{')]
client = httpx.Client(follow_redirects=True, headers=UA, timeout=120)


def model_offset(head):
    best = None
    for rx in MODEL_RE:
        m = rx.search(head)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def prefix_json(head, off):
    """Parse the top-level keys that precede "model" (version, truncation, padding, added_tokens, normalizer, ...)."""
    txt = head[:off].decode("utf-8", errors="strict").rstrip().rstrip(",")
    return json.loads(txt + "}")


def ref_index():
    idx = {}
    paths = glob.glob(os.path.join(TOK, "baselines", "files", "**", "*.json"), recursive=True) + \
        glob.glob(os.path.join(ROOT, "files", "**", "tokenizer.json"), recursive=True)
    for p in paths:
        if ".cache" in p or not os.path.basename(p).startswith("tokenizer"):
            continue
        data = open(p, "rb").read()
        off = model_offset(data[:4 * 1024 * 1024])
        if off is None:
            continue
        L = len(data) - off
        mid = off + L // 2
        key = (L, hashlib.sha256(data[-TAIL_N:]).hexdigest(), hashlib.sha256(data[mid:mid + MID_N]).hexdigest())
        try:
            pj = prefix_json(data, off)
        except Exception:  # noqa: BLE001
            continue
        idx.setdefault(key, []).append({"path": os.path.relpath(p, TOK).replace("\\", "/"), "prefix": pj})
    return idx


def rng(url, a, b=None):
    h = {"Range": "bytes=%d-%s" % (a, "" if b is None else b)} if a >= 0 else {"Range": "bytes=%d" % a}
    for t in range(6):
        r = client.get(url, headers=h)
        if r.status_code == 429:
            time.sleep(30 * (t + 1))
            continue
        if r.status_code not in (200, 206):
            raise RuntimeError("HTTP %d" % r.status_code)
        return r.content if r.status_code == 206 else (r.content[a:b + 1 if b is not None else None] if a >= 0 else r.content[a:])
    raise RuntimeError("429 retries exhausted")


HEAD_CACHE = os.path.join(ROOT, "logs", "heads")


def head_until_model(url, sig, steps=(192 * 1024, 768 * 1024, 2560 * 1024)):
    """First bytes of the file up to (and past) the start of its "model" section; cached on disk per signature."""
    os.makedirs(HEAD_CACHE, exist_ok=True)
    cp = os.path.join(HEAD_CACHE, sig + ".bin")
    if os.path.exists(cp):
        head = open(cp, "rb").read()
        off = model_offset(head)
        if off is not None:
            return head, off
    head, off = b"", None
    for n in steps:
        head = rng(url, 0, n - 1)
        off = model_offset(head)
        if off is not None:
            break
    if off is not None:
        open(cp, "wb").write(head[:off + 192 * 1024])
    return head, off


def cached_rng(url, key, a, b=None):
    os.makedirs(HEAD_CACHE, exist_ok=True)
    cp = os.path.join(HEAD_CACHE, "%s.%s.bin" % (key, "tail" if a < 0 else "r%d" % a))
    if os.path.exists(cp):
        return open(cp, "rb").read()
    data = rng(url, a, b)
    open(cp, "wb").write(data)
    return data


def url_of(repo, f, commit):
    return "https://huggingface.co/%s/resolve/%s/%s" % (repo, commit or "main", f)


def triage(args):
    g, rep, commit, idx, text = args
    f = rep["files"][0]
    size = g["sizes"][0]
    url = url_of(rep["repo"], f, commit)
    out = {"sig": g["sig"], "repo": rep["repo"], "file": f, "size": size, "commit": commit}
    try:
        head = rng(url, 0, HEAD_N - 1)
        off = model_offset(head)
        if off is None:
            head = rng(url, 0, 1024 * 1024 - 1)
            off = model_offset(head)
        if off is None:
            out.update(status="needs_download", reason="model section not in the first 1 MB")
            return out
        L = size - off
        mid = off + L // 2
        tail = rng(url, -TAIL_N)
        midb = rng(url, mid, mid + MID_N - 1)
        key = (L, hashlib.sha256(tail).hexdigest(), hashlib.sha256(midb).hexdigest())
        refs = idx.get(key)
        if not refs:
            out.update(status="needs_download", reason="model section differs from every tokenizer.json on disk")
            return out
        pj = prefix_json(head, off)
        for ref in refs:
            rp = ref["prefix"]
            same_pipe = all(pj.get(k) == rp.get(k) for k in ("normalizer", "pre_tokenizer", "decoder"))
            if not same_pipe:
                continue
            ref_added = {a["content"] for a in rp.get("added_tokens", [])}
            extra = [a for a in pj.get("added_tokens", []) if a["content"] not in ref_added]
            occurring = [a["content"] for a in extra if a["content"] and a["content"] in text]
            if occurring:
                out.update(status="needs_download", reason="added tokens occurring in the test text: %r" % occurring[:5])
                return out
            out.update(status="identical_behaviour", reference=ref["path"], n_extra_added_tokens=len(extra),
                       extra_added_examples=[a["content"] for a in extra[:5]])
            return out
        out.update(status="needs_download", reason="same model section as %s but a different normalizer / pre_tokenizer / "
                                                   "decoder" % refs[0]["path"])
        return out
    except Exception as e:  # noqa: BLE001
        out.update(status="needs_download", reason="range read failed: %s: %s" % (type(e).__name__, str(e)[:120]))
        return out


def main():
    groups = []
    for f in ("_probe.json", "_probe2.json"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            groups += json.load(open(p, encoding="utf-8"))["groups"]
    seen, big = set(), []
    for g in groups:
        if g["sig"] in seen:
            continue
        seen.add(g["sig"])
        if g["kind"] == "tokenizer.json" and sum(x or 0 for x in g["sizes"]) > 5e6:
            big.append(g)
    commits = {}
    for line in open(os.path.join(ROOT, "_probe_cache.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        for f, m in (x.get("meta") or {}).items():
            commits[(x["repo"], f)] = m.get("commit")
    text = "\n".join(json.loads(line)["text"] for line in open(os.path.join(TOK, "data", "test_strict.jsonl"), encoding="utf-8"))
    idx = ref_index()
    print("references", sum(len(v) for v in idx.values()), "large tokenizer.json signatures", len(big), flush=True)
    jobs = []
    for g in big:
        rep = max(g["members"], key=lambda m: (m.get("downloads") or 0) + 50 * (m.get("likes") or 0))
        jobs.append((g, rep, commits.get((rep["repo"], rep["files"][0])), idx, text))
    res = []
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(triage, jobs)):
            res.append(r)
            if i % 50 == 0:
                print(i, r["repo"], r["status"], r.get("reference") or r.get("reason"), flush=True)
    json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": __doc__, "n": len(res),
               "n_identical_behaviour": sum(1 for r in res if r["status"] == "identical_behaviour"),
               "n_needs_download": sum(1 for r in res if r["status"] == "needs_download"), "results": res},
              open(os.path.join(ROOT, "_triage.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("identical", sum(1 for r in res if r["status"] == "identical_behaviour"), "needs download",
          sum(1 for r in res if r["status"] == "needs_download"), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
