# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 3c: second-level triage of the large tokenizer.json files that stage 3b could not
match byte-for-byte (typically the same vocabulary re-saved by another `tokenizers` / `transformers` version, which
changes the bytes of the "model" section but not its content).

For each such file (reading 192 KB from the start, 16 KB from the middle and 32 KB from the end, over HTTPS):
  1. FAMILY: the first 300 vocabulary entries of its "model" section (in file order) are compared with those of every
     reference tokenizer.json on disk; a reference with the identical first 300 entries is the family candidate.
  2. SAMPLED MEMBERSHIP: every JSON string in the middle and end samples (vocabulary pieces, or the halves of merges)
     is looked up in the family reference's vocabulary + merges + added tokens.  100% membership (with >= 200 strings
     sampled) and the same model type means no sampled entry is new; together with an unchanged family head this is
     classified 'same_vocabulary_sampled' (a re-save of the reference; its encodings are expected to equal the
     reference's, but they are NOT measured, and the report says so).
  3. The share of Arabic-script strings in the samples is recorded as a second signal (a Perso-Arabic vocabulary
     extension or a Perso-Arabic-specialised retrain shows a high share).
Everything else -> needs_download (downloaded and measured in full).
Output: ../_triage2.json
"""
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import range_triage as T  # noqa: E402

ROOT = T.ROOT
TEXT = "\n".join(json.loads(line)["text"] for line in open(os.path.join(T.TOK, "data", "test_strict.jsonl"), encoding="utf-8"))
# a JSON string that starts right after '[', '{' or ',' is a vocabulary key, a merge or a list element; anchoring on that
# context keeps the parse aligned even when a byte-range chunk starts in the middle of a string (an unescaped quote can
# never occur inside a JSON string, so these delimiters cannot match inside one)
STR_RE = re.compile(r'[\[{,]\s*"((?:[^"\\]|\\.)*)"')
BL_CHARS = set("ĠĊĉÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß")
B2U = None


def is_ar(c):
    o = ord(c)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def bl_decode(s):
    global B2U
    if B2U is None:
        sys.path.insert(0, os.path.join(T.TOK, "baselines", "scripts"))
        from tiktoken_tools import bytes_to_unicode
        B2U = {v: k for k, v in bytes_to_unicode().items()}
    try:
        return bytes(B2U[c] for c in s).decode("utf-8", errors="ignore")
    except KeyError:
        return s


def strings(chunk):
    out = []
    for m in STR_RE.finditer(chunk.decode("utf-8", errors="ignore")):
        try:
            out.append(json.loads('"%s"' % m.group(1)))
        except Exception:  # noqa: BLE001
            pass
    return out


def is_arabic_entry(x):
    """A vocabulary piece or merge ("a b" or one half) that contains an Arabic-script character once byte-level
    (GPT-2 alphabet) pieces are decoded; the halves of a merge are decoded separately."""
    for part in (x.split(" ") if " " in x else [x]):
        d = bl_decode(part) if any(c in BL_CHARS for c in part) else part
        if any(is_ar(c) for c in d):
            return True
    return False


def arabic_share(samp):
    return (sum(1 for x in samp if is_arabic_entry(x)) / len(samp)) if samp else None


def first_vocab(section_text, n=300):
    i = section_text.find('"vocab"')
    if i < 0:
        return None
    s = strings(section_text[i + 7:].encode("utf-8"))
    s = [x for x in s if x not in ("type", "vocab", "merges")]
    return tuple(s[:n]) if len(s) >= n else None


def ref_heads():
    heads = {}
    paths = glob.glob(os.path.join(T.TOK, "baselines", "files", "**", "tokenizer*.json"), recursive=True) + \
        glob.glob(os.path.join(ROOT, "files", "**", "tokenizer.json"), recursive=True)
    for p in paths:
        if ".cache" in p:
            continue
        data = open(p, "rb").read()
        off = T.model_offset(data[:4 * 1024 * 1024])
        if off is None:
            continue
        fv = first_vocab(data[off:off + 256 * 1024].decode("utf-8", errors="ignore"))
        if fv:
            heads.setdefault(fv, []).append(p)
    return heads


_REFSETS = {}
_REFPRE = {}


def ref_prefix(p):
    if p not in _REFPRE:
        j = json.load(open(p, encoding="utf-8"))
        _REFPRE[p] = {k: j.get(k) for k in ("normalizer", "pre_tokenizer", "decoder", "added_tokens")}
    return _REFPRE[p]


def ref_set(p):
    if p not in _REFSETS:
        j = json.load(open(p, encoding="utf-8"))
        m = j["model"]
        s = set()
        v = m.get("vocab")
        if isinstance(v, dict):
            s.update(v.keys())
        elif isinstance(v, list):
            s.update(x[0] for x in v)
        for x in m.get("merges") or []:
            if isinstance(x, str):
                s.add(x)
                s.update(x.split(" "))
            else:
                s.update(x)
        s.update(a["content"] for a in j.get("added_tokens", []))
        _REFSETS[p] = (s, m.get("type"))
        if len(_REFSETS) > 12:
            _REFSETS.pop(next(iter(_REFSETS)))
    return _REFSETS[p]


def analyse(args):
    r, commit_url, heads = args
    out = {"sig": r["sig"], "repo": r["repo"], "size": r["size"]}
    try:
        head, off = T.head_until_model(commit_url, r["sig"])   # 192 KB, else 768 KB, else 2.5 MB (cached)
        if off is None:
            out.update(status="needs_download", reason="model section not in the first %d KB" % (HEAD_BYTES // 1024))
            return out
        fv = first_vocab(head[off:].decode("utf-8", errors="ignore"))
        L = r["size"] - off
        mid = T.cached_rng(commit_url, r["sig"], off + L // 2, off + L // 2 + 16 * 1024 - 1)
        tail = T.cached_rng(commit_url, r["sig"], -32 * 1024)
        samp = strings(mid)[:-1] + strings(tail)[:-1]         # drop the possibly cut last string of each chunk
        samp = [x for x in samp if x not in ("type", "vocab", "merges", "unk_token", "BPE", "Unigram", "WordPiece")]
        out["n_sampled"] = len(samp)
        out["arabic_share_sampled"] = arabic_share(samp)
        refs = heads.get(fv) if fv else None
        if not refs:
            out.update(status="needs_download", reason="first 300 vocabulary entries match no tokenizer on disk")
            return out
        best = None
        mtype = re.search(rb'"type"\s*:\s*"(\w+)"', head[off:off + 2000])
        mtype = mtype.group(1).decode() if mtype else None
        for p in refs:
            s, t = ref_set(p)
            parts_ok = sum(1 for x in samp if x in s or all(y in s for y in x.split(" ")))
            frac = parts_ok / len(samp) if samp else 0
            if best is None or frac > best[1]:
                best = (p, frac, t)
        out.update(family_reference=os.path.relpath(best[0], T.TOK).replace("\\", "/"), sampled_membership=best[1],
                   model_type=mtype)
        # pipeline and added tokens (both sit before "model", i.e. inside the 192 KB head)
        pj = T.prefix_json(head, off)
        rj = ref_prefix(best[0])
        same_pipe = all(pj.get(k) == rj.get(k) for k in ("normalizer", "pre_tokenizer", "decoder"))
        ref_added = {a["content"] for a in rj.get("added_tokens", [])}
        occurring = [a["content"] for a in pj.get("added_tokens", []) if a["content"] not in ref_added and a["content"]
                     and a["content"] in TEXT]
        out.update(same_pipeline_as_reference=same_pipe, added_tokens_occurring_in_test=occurring[:5])
        if not same_pipe:
            out.update(status="needs_download", reason="normalizer / pre_tokenizer / decoder differ from the family reference")
        elif occurring:
            out.update(status="needs_download", reason="added tokens that occur in the test text")
        elif len(samp) >= 200 and best[1] == 1.0 and (mtype is None or mtype == best[2]):
            out["status"] = "same_vocabulary_sampled"
        else:
            out.update(status="needs_download", reason="sampled entries not all in the family reference (%.4f)" % best[1])
        return out
    except Exception as e:  # noqa: BLE001
        out.update(status="needs_download", reason="range read failed: %s: %s" % (type(e).__name__, str(e)[:120]))
        return out


HEAD_BYTES = 192 * 1024


def main():
    global HEAD_BYTES
    tr = json.load(open(os.path.join(ROOT, "_triage.json"), encoding="utf-8"))
    todo = [r for r in tr["results"] if r["status"] == "needs_download"]
    out_name = "_triage2.json"
    dl0 = json.load(open(os.path.join(ROOT, "_download_log.json"), encoding="utf-8"))
    todo = [r for r in todo if r["sig"] not in dl0]      # downloaded / rebuilt files are measured directly
    if "--rerun" in sys.argv:             # after family representatives were downloaded: re-check what is still unresolved
        import family_pass
        todo, _ = family_pass.unresolved()
        t1 = {r["sig"]: r for r in tr["results"]}
        todo = [dict(t1[r["sig"]]) for r in todo]
        dl = json.load(open(os.path.join(ROOT, "_download_log.json"), encoding="utf-8"))
        todo = [r for r in todo if r["sig"] not in dl]      # the downloaded ones are measured directly
        out_name = "_triage2c.json"
    if "--bighead" in sys.argv:           # second pass: files whose "model" section starts after 192 KB (big added-token lists)
        HEAD_BYTES = 2560 * 1024
        prev = {x["sig"]: x for x in json.load(open(os.path.join(ROOT, "_triage2.json"), encoding="utf-8"))["results"]}
        todo = [r for r in todo if "not in the first" in (prev.get(r["sig"], {}).get("reason") or "")]
        out_name = "_triage2b.json"
    heads = ref_heads()
    print("second-level triage of", len(todo), "files; reference families", len(heads), flush=True)
    jobs = [(r, T.url_of(r["repo"], r["file"], r.get("commit")), heads) for r in todo]
    res = []
    with ThreadPoolExecutor(10) as ex:   # I/O-bound range reads, one process
        for i, x in enumerate(ex.map(analyse, jobs)):
            res.append(x)
            if i % 50 == 0:
                print(i, x["repo"], x["status"], x.get("family_reference"), x.get("sampled_membership"),
                      x.get("arabic_share_sampled"), flush=True)
    json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": __doc__, "n": len(res),
               "n_same_vocabulary_sampled": sum(1 for x in res if x["status"] == "same_vocabulary_sampled"),
               "n_needs_download": sum(1 for x in res if x["status"] == "needs_download"), "results": res},
              open(os.path.join(ROOT, out_name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("same vocabulary (sampled)", sum(1 for x in res if x["status"] == "same_vocabulary_sampled"), "needs download",
          sum(1 for x in res if x["status"] == "needs_download"), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
