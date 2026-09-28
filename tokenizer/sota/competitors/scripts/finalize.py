# -*- coding: utf-8 -*-
"""Final competitor ranking (2026-09-27), from tokenizers ALREADY measured.  No search, no download.

Inputs : competitors_all.json (merged refresh table: 485 new behaviours + 66 study baselines + released),
         results/<name>/docs.jsonl and eval/results/test_strict/<name>/docs.jsonl (per-document rows, test_strict),
         the local tokenizer files (for the vocabulary-family hash), _search*.json, _download_log.json.
Outputs: ../COMPETITORS_FINAL.md is written by write_final_md.py; this script writes ../competitors_final.json.

Dedup   : 'behaviour' = sha256 of the per-document (tokens, round-trip) vector (same definition as screen.py).
          A second, looser key merges tokenizers whose per-document token counts are identical (round-trip may differ).
Family  : 'vocabulary family' = sha256 of vocabulary entries at ids 0-299 and 1000-1099 (heuristic lineage key:
          fine-tunes / vocab extensions that keep the base vocabulary prefix fall in the base family).
Lossless: G1 = every one of the 491 test documents decodes back to exactly its input.
"""
import collections
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = r"F:\Hindko\_tokenizer"
C = os.path.join(ROOT, "sota", "competitors")
RELEASED = "hindko-tokenizer-1.0.0-released"
LEAKY = "hindko-probe-bpe32k"
WIN = list(range(0, 300)) + list(range(1000, 1100))


def docs_path(name, origin):
    if origin.startswith("baseline"):
        return os.path.join(ROOT, "eval", "results", "test_strict", name, "docs.jsonl")
    return os.path.join(C, "results", name, "docs.jsonl")


def summary_path(name, origin):
    return os.path.join(os.path.dirname(docs_path(name, origin)), "summary.json")


def fingerprints(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["uid"])
    exact = hashlib.sha256(json.dumps([[r["tokens"], r["rt"]] for r in rows]).encode()).hexdigest()
    toks = hashlib.sha256(json.dumps([r["tokens"] for r in rows]).encode()).hexdigest()
    return exact, toks, len(rows)


def _vocab_from_json(path):
    try:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(path)
        n = t.get_vocab_size(with_added_tokens=False)
        return [t.id_to_token(i) for i in WIN if i < n]
    except Exception:
        pass
    d = json.load(open(path, encoding="utf-8"))
    v = (d.get("model") or {}).get("vocab") if isinstance(d, dict) else None
    if v is None and isinstance(d, dict):
        v = d.get("vocab")
    if isinstance(v, list):
        v = [x[0] if isinstance(x, list) else (x.get("token_str") or x.get("token_bytes") if isinstance(x, dict) else x) for x in v]
        return [v[i] for i in WIN if i < len(v)]
    if isinstance(v, dict):
        inv = sorted(v.items(), key=lambda kv: kv[1])
        return [inv[i][0] for i in WIN if i < len(inv)]
    return None


def _vocab_from_spm(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=path)
    n = sp.get_piece_size()
    return [sp.id_to_piece(i) for i in WIN if i < n]


def _vocab_from_dir(d):
    if not d or not os.path.isdir(d):
        return None, None
    for fn in ("tokenizer.json", "tokenizer.converted.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return _vocab_from_json(p), fn
    for p in glob.glob(os.path.join(d, "*.model")) + glob.glob(os.path.join(d, "*.spm")):
        try:
            return _vocab_from_spm(p), os.path.basename(p)
        except Exception:
            pass
    p = os.path.join(d, "vocab.txt")
    if os.path.exists(p):
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        return [lines[i] for i in WIN if i < len(lines)], "vocab.txt"
    p = os.path.join(d, "vocab.json")
    if os.path.exists(p):
        v = json.load(open(p, encoding="utf-8"))
        inv = sorted(v.items(), key=lambda kv: kv[1])
        return [inv[i][0] for i in WIN if i < len(inv)], "vocab.json"
    return None, None


def vocab_family(args):
    name, origin, path, kind = args
    try:
        if kind and "bytes" in kind:
            return name, "bytes", "raw UTF-8 bytes"
        toks, src = None, None
        if path and os.path.exists(path):
            if path.endswith((".model", ".spm")):
                toks, src = _vocab_from_spm(path), os.path.basename(path)
            elif path.endswith(".json"):
                toks, src = _vocab_from_json(path), os.path.basename(path)
        if toks is None:
            base = os.path.join(ROOT, "baselines", "files", name) if origin.startswith("baseline") else os.path.join(C, "files", name)
            toks, src = _vocab_from_dir(base)
        if not toks:
            return name, None, "vocab not readable"
        return name, hashlib.sha256(json.dumps(toks, ensure_ascii=False).encode()).hexdigest()[:16], src
    except Exception as e:  # noqa: BLE001
        return name, None, "%s: %s" % (type(e).__name__, str(e)[:80])


def main():
    A = json.load(open(os.path.join(C, "competitors_all.json"), encoding="utf-8"))
    T = A["tokenizers"]
    rel = next(t for t in T if t["name"] == RELEASED)
    ext = [t for t in T if t["name"] not in (RELEASED, LEAKY)]
    # --- metrics check against the per-tokenizer summary.json + fingerprints
    jobs, mism = [], []
    for t in ext:
        sp = summary_path(t["name"], t["origin"])
        s = json.load(open(sp, encoding="utf-8"))
        o = s["metrics"]["overall"]
        if abs(o["bytes_per_token"] - t["test_strict"]["overall"]["bytes_per_token"]) > 1e-9 or o["tokens"] != t["test_strict"]["overall"]["tokens"]:
            mism.append(t["name"])
        t["_fp_exact"], t["_fp_tokens"], nd = fingerprints(docs_path(t["name"], t["origin"]))
        assert nd == 491, (t["name"], nd)
        jobs.append((t["name"], t["origin"], s["tokenizer"].get("path") or "", s["tokenizer"].get("kind") or ""))
    with ProcessPoolExecutor(max_workers=3) as ex:
        fam = {n: (h, src) for n, h, src in ex.map(vocab_family, jobs, chunksize=4)}
    json.dump({"metric_mismatch": mism, "family": fam}, open(os.path.join(C, "logs", "_final_family.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("metric mismatches:", len(mism), "family unreadable:", sum(1 for v in fam.values() if v[0] is None))


if __name__ == "__main__":
    sys.exit(main())
