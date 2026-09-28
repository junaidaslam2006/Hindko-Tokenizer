# -*- coding: utf-8 -*-
"""build_bundle.py - encode train (D1) and dev_strict with every finalist's exact shipped encoder and
pack everything the Colab LM arbiter needs into parts of at most 9.5 MB.

    python build_bundle.py                                   # finalists from F:/Hindko/_tokenizer/lm/WAVES.json
    python build_bundle.py --candidates P1 hf P2 sentencepiece --ids a b --baseline a   # testing

What it does
  1. candidates: WAVES.json (id, tokenizer_path, encoder_kind hf|sentencepiece|custom per finalist)
     or --candidates path kind [path kind ...].
  2. text: data/train_D1.jsonl (permissive train, all documents) and data/dev_strict.jsonl, both
     sorted by uid (one fixed document order). Every uid is checked against the split manifest:
     train rows must be split=train, dev rows split=validation and tier=strict. No file whose name
     contains "test" is ever opened; the TEST split is never read, encoded or bundled.
  3. encoders: eval/harness.py loaders (the exact encoder of the intrinsic sweep) when available;
     --encoder-source builtin uses the native HF / SentencePiece encoders (SentencePiece with the
     PLAN 1.1 line wrapper when the model has a "\\n" piece). Custom encoders need the harness.
  4. verifies decode(encode(doc)) == doc for every dev document, that no special id occurs inside a
     document, and that every id < n_vocab.
  5. stores uint16 (uint32 if an id > 65535) token arrays + int64 document offsets per candidate;
     shared once: per-document UTF-8 bytes, train uids, dev uids/group/cluster/source, the fixed
     0.5 MB dev subset and the parity slice.
  6. manifest.json: sha256 of every file, candidate metadata (tokenizer sha256, specials, n_vocab,
     train bytes/token, ctx_tokens, round-trip result, encoder identity).
  7. runs the fixed PARITY config on this CPU (100 steps, seed 1, baseline) -> PARITY.json.
  8. tar + xz, split into parts <= 9.5 MB, PARTS.json (names, sizes, sha256), plus run_all.py.

Machine limits: at most 2 worker processes, 1 thread each; writes only under --out (default
F:/Hindko/_tokenizer/colab/build).

--final-test (PLAN 7.6, the ONE-SHOT final test; the only mode that ever opens the test split):
  encodes data/test_strict.jsonl (written by data/materialize_test.py and checked against
  data/test_manifest.json) IN PLACE OF dev_strict. The arrays, records and numbers keep their 'dev'
  names, so every 'dev' number computed from such a bundle is a TEST number. The manifest (and
  PARTS.json) say kind='final_test'; run_all.py then allows only the parity and confirm stages.
  Every such build appends a record (time, candidates + tokenizer sha256, test-file sha256, code
  sha256) to F:/Hindko/_tokenizer/FINAL_TEST_LOG.json before encoding and completes it at the end.
  Default --out: colab/build_final_test. Without the flag nothing changes: no file whose name
  contains "test" is opened and any test uid aborts the build, exactly as before.

--final-test --report-only "PURPOSE" (added for the frontier-tokenizer comparison, colab/FRONTIER_LM.md): the
  same test build with every --final-test guard, but manifest/PARTS kind='final_test_report'. It marks a
  RE-USE of the test split for a report-only comparison on which no decision depends; the log entry records
  PURPOSE. run_all.py then allows only parity + confirm, needs --lr, and runs seeds 1-3 (the power check is
  reported, never acted on; --force-extra-seeds adds 4, 5). Default --out: colab/build_final_test_report.

encoder_kind 'baseline' (same addition): an external tokenizer of baselines/manifest.json by its name
  (--candidates gpt-4o baseline, or a WAVES entry with "encoder_kind": "baseline", "baseline_name": ...),
  loaded through eval/harness.py -> adapters.from_baseline -> baselines/load_baselines.py (the objects of
  the intrinsic tables; --encoder-source builtin uses load_baselines directly). tokenizer_sha256 is the
  sha256 of its load_path file (a directory: sha256 of the sorted "name sha256" lines of its files).
  Token arrays are uint32 when n_vocab > 65536 (unchanged rule).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import importlib
import inspect
import io
import json
import lzma
import os
import re
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(TOK, "data")
EVAL_DIR = os.path.join(TOK, "eval")
WAVES_DEFAULT = os.path.join(TOK, "lm", "WAVES.json")
SPLIT_MANIFEST = os.path.join(TOK, "splits", "split_manifest.jsonl")
PART_BYTES_DEFAULT = 9_500_000          # < 10 MB under any definition of MB
ARCHIVE = "hk_bundle.tar.xz"
SUBSET_SEED = 12345                     # fixed, candidate- and run-independent
SUBSET_BYTES = 500_000                  # PLAN 5: fixed 0.5 MB dev subset for the learning curve
PARITY_EVAL_BYTES = 100_000             # parity dev slice: first subset docs up to 100 KB
FINAL_TEST_LOG = os.path.join(TOK, "FINAL_TEST_LOG.json")    # --final-test only
BASELINES_DIR = os.path.join(TOK, "baselines")
# looked up in this order; the names after the first three were appended for external tokenizers (Llama 3/4,
# DeepSeek-V3), so every earlier candidate resolves exactly as before
_DS = chr(0xFF5C), chr(0x2581)                                # DeepSeek's fullwidth bar and U+2581
SPECIAL_EOT = ("<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>",
               "<%send%sof%ssentence%s>" % (_DS[0], _DS[1], _DS[1], _DS[0]))
SPECIAL_BOS = ("<|bos|>", "<s>", "<bos>", "<|begin_of_text|>",
               "<%sbegin%sof%ssentence%s>" % (_DS[0], _DS[1], _DS[1], _DS[0]))

sys.path.insert(0, HERE)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


# ============================================================ candidates
def _collect_entries(obj, out):
    if isinstance(obj, dict):
        path = obj.get("tokenizer_path") or obj.get("tokenizer") or obj.get("path")
        kind = obj.get("encoder_kind") or obj.get("kind")
        if isinstance(path, str) and isinstance(kind, str):
            out.append(obj)
            return
        for v in obj.values():
            _collect_entries(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_entries(v, out)


def norm_kind(k):
    k = k.strip().lower()
    return {"sp": "sentencepiece", "spm": "sentencepiece", "huggingface": "hf", "tokenizers": "hf"}.get(k, k)


def load_waves(path):
    with open(path, encoding="utf-8") as f:
        w = json.load(f)
    ents = []
    _collect_entries(w, ents)
    base_dir = os.path.dirname(os.path.abspath(path))
    cands, seen = [], set()
    for rank, e in enumerate(ents, 1):
        p = e.get("tokenizer_path") or e.get("tokenizer") or e.get("path")
        if not os.path.isabs(p):
            p2 = os.path.join(base_dir, p)
            p = p2 if os.path.exists(p2) else os.path.join(TOK, p)
        cid = str(e.get("id") or os.path.splitext(os.path.basename(p))[0])
        if cid in seen:
            continue
        seen.add(cid)
        extra = {k: v for k, v in e.items() if k not in ("tokenizer_path", "tokenizer", "path", "encoder_kind", "kind", "id")
                 and isinstance(v, (str, int, float, bool, type(None)))}
        kind = norm_kind(e.get("encoder_kind") or e.get("kind"))
        cands.append({"id": cid, "path": os.path.normpath(p), "kind": kind,
                      "rank": rank, "waves_fields": extra,
                      "baseline_name": (e.get("baseline_name") or cid) if kind == "baseline" else None,
                      "custom_spec": e.get("custom") or e.get("factory") or e.get("custom_factory"),
                      "no_newline_wrapper": e.get("newline_wrapper") is False or bool(e.get("no_newline_wrapper")),
                      "is_baseline_flag": bool(e.get("baseline")) or str(e.get("role", "")).lower() == "baseline"})
    top_baseline = w.get("baseline") if isinstance(w, dict) and isinstance(w.get("baseline"), str) else None
    return cands, top_baseline, sha256_file(path)


def pick_baseline(cands, explicit=None, top=None):
    ids = [c["id"] for c in cands]
    for b in (explicit, top):
        if b:
            if b not in ids:
                raise SystemExit("baseline %r is not among the candidates %s" % (b, ids))
            return b, "explicit"
    flagged = [c["id"] for c in cands if c.get("is_baseline_flag")]
    if flagged:
        return flagged[0], "flag in WAVES.json"
    for c in cands:
        if re.search(r"a1.*p1.*16", c["id"], re.I) or re.search(r"bpe.*p1.*16", c["id"], re.I):
            return c["id"], "id pattern A1/BPE-P1-16k"
    log("WARNING: no baseline marked; using the first candidate %s" % ids[0])
    return ids[0], "first candidate (fallback)"


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


# ============================================================ text
def load_split_manifest():
    if not os.path.isfile(SPLIT_MANIFEST):
        return None, None
    rows = {}
    with open(SPLIT_MANIFEST, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["uid"]] = (r["split"], r.get("quality_tier"))
    return rows, sha256_file(SPLIT_MANIFEST)


def read_jsonl(path, want_split, man, require_strict=False, final_test=False):
    """final_test=True (build_bundle.py --final-test only): `path` IS the strict test split; the two
    test guards are lifted for this one file, and every uid must then be split=test."""
    base = os.path.basename(path).lower()
    if final_test:
        if want_split != "test" or man is None:
            raise SystemExit("--final-test reads only the test split, and only with the split manifest present")
    elif "test" in base:
        raise SystemExit("refusing to open %s: the test split is never read" % path)
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            r = json.loads(line)
            uid = r.get("uid") or r.get("id")
            if uid is None or not isinstance(r.get("text"), str):
                raise SystemExit("%s:%d has no uid/text" % (path, ln))
            if man is not None:
                if uid not in man:
                    raise SystemExit("%s: uid %s is not in the split manifest" % (path, uid))
                sp, tier = man[uid]
                if sp == "test" and not final_test:
                    raise SystemExit("%s: uid %s belongs to the TEST split - aborting" % (path, uid))
                if sp != want_split:
                    raise SystemExit("%s: uid %s is split %s, expected %s" % (path, uid, sp, want_split))
                if require_strict and tier != "strict":
                    raise SystemExit("%s: uid %s is tier %s, dev must be strict" % (path, uid, tier))
            if require_strict and r.get("tier", "strict") != "strict":
                raise SystemExit("%s: uid %s is not strict" % (path, uid))
            recs.append({"uid": uid, "text": r["text"], "group": r.get("group"), "cluster": r.get("cluster"),
                         "source": r.get("source"), "variety": r.get("variety")})
    recs.sort(key=lambda r: r["uid"])
    if len({r["uid"] for r in recs}) != len(recs):
        raise SystemExit("%s has duplicate uids" % path)
    return recs


# ============================================================ encoders
class Enc:
    """Uniform encoder view: encode(str)->list[int], decode(list[int])->str, n_vocab, token_to_id(str)."""

    def __init__(self, encode, decode, n_vocab, token_to_id, source, detail, encode_batch=None, identity=None):
        self.encode, self.decode, self.n_vocab, self.token_to_id = encode, decode, n_vocab, token_to_id
        self.source, self.detail, self.identity = source, detail, identity
        self.encode_batch = encode_batch or (lambda ts: [encode(t) for t in ts])


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def _wrap_harness_adapter(ad, h, hsha):
    """eval/adapters.py adapter (the objects harness.run() evaluates): encode, encode_batch, decode,
    special_tokens {str: id}, id_upper (exclusive id bound), identity()."""
    spec = getattr(ad, "special_tokens", None) or {}
    ident = None
    if callable(getattr(ad, "identity", None)):
        try:
            ident = _jsonable(ad.identity())
        except Exception as e:  # pragma: no cover
            ident = {"identity_error": repr(e)}
    try:
        code = h.code_hashes() if callable(getattr(h, "code_hashes", None)) else None
    except Exception:  # pragma: no cover
        code = None
    n_vocab = int(getattr(ad, "id_upper", 0) or getattr(ad, "vocab_size"))
    return Enc(lambda t: [int(i) for i in ad.encode(t)],
               lambda ids: ad.decode([int(i) for i in ids]),
               n_vocab, lambda s: (int(spec[s]) if s in spec and spec[s] is not None else None),
               "eval/harness.py:build_adapter -> %s" % type(ad).__name__,
               "harness sha256 %s" % hsha,
               encode_batch=lambda ts: [[int(i) for i in x] for x in ad.encode_batch(list(ts))],
               identity={"adapter_identity": ident, "code_hashes": code, "harness_sha256": hsha})


def _hf_builtin(path):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(path)
    vocab = tk.get_vocab(with_added_tokens=True)
    return Enc(lambda t: tk.encode(t, add_special_tokens=False).ids,
               lambda ids: tk.decode(list(ids), skip_special_tokens=False),
               max(vocab.values()) + 1, tk.token_to_id, "builtin:hf", "tokenizers.Tokenizer.from_file; "
               "encode(add_special_tokens=False); decode(skip_special_tokens=False)")


def _sp_builtin(path, wrapper=True):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=path)

    def t2i(s):
        i = sp.piece_to_id(s)
        return i if sp.id_to_piece(i) == s and not sp.is_unknown(i) else None

    nl = t2i("\n")
    if wrapper and nl is None:     # same rule as eval/adapters.py SPAdapter
        raise SystemExit("newline wrapper needs a '\\n' piece in %s (use --no-newline-wrapper for pilot models)" % path)
    if not wrapper:
        return Enc(lambda t: sp.encode(t), lambda ids: sp.decode(list(ids)), sp.get_piece_size(), t2i,
                   "builtin:sentencepiece", "whole-document sp.encode (no newline wrapper)")

    def enc(t):
        out = []
        for k, line in enumerate(t.split("\n")):
            if k:
                out.append(nl)
            if line:
                out.extend(sp.encode(line))
        return out

    def dec(ids):
        segs, cur = [], []
        for i in ids:
            if i == nl:
                segs.append(cur)
                cur = []
            else:
                cur.append(i)
        segs.append(cur)
        return "\n".join(sp.decode(s) if s else "" for s in segs)

    return Enc(enc, dec, sp.get_piece_size(), t2i, "builtin:sentencepiece",
               "PLAN 1.1 wrapper: split at \\n, sp.encode per line, join with the newline piece id %d" % nl)


def _import_harness():
    p = os.path.join(EVAL_DIR, "harness.py")
    if not os.path.isfile(p):
        return None, None
    if EVAL_DIR not in sys.path:
        sys.path.insert(0, EVAL_DIR)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # read-only use of eval/: never write __pycache__ there
    try:
        return importlib.import_module("harness"), sha256_file(p)
    finally:
        sys.dont_write_bytecode = old


def _wrap_harness_obj(obj, fn_name, hsha):
    """Adapt whatever the harness loader returns to Enc."""
    try:
        from tokenizers import Tokenizer as _HFT
    except Exception:  # pragma: no cover
        _HFT = ()
    if isinstance(obj, tuple) and len(obj) >= 2 and callable(obj[0]) and callable(obj[1]):
        e, d = obj[0], obj[1]
        target = obj[2] if len(obj) > 2 else None
    else:
        target = obj
        if _HFT and isinstance(obj, _HFT):
            e = lambda t: obj.encode(t, add_special_tokens=False)
            d = lambda ids: obj.decode(list(ids), skip_special_tokens=False)
        else:
            e = getattr(obj, "encode")
            d = getattr(obj, "decode")

    def enc(t):
        r = e(t)
        r = getattr(r, "ids", r)
        return [int(i) for i in r]

    def dec(ids):
        return d(list(int(i) for i in ids))

    n_vocab = None
    for attr in ("n_vocab", "vocab_size", "get_vocab_size", "get_piece_size", "__len__"):
        v = getattr(target, attr, None) if target is not None else None
        if v is None:
            continue
        try:
            v = v(True) if attr == "get_vocab_size" else (v() if callable(v) else v)
        except TypeError:
            v = v()
        if isinstance(v, int) and v > 0:
            n_vocab = v
            break
    get_vocab = getattr(target, "get_vocab", None) if target is not None else None
    if callable(get_vocab):
        try:
            vv = get_vocab(True) if _HFT and isinstance(target, _HFT) else get_vocab()
            if isinstance(vv, dict) and vv:
                n_vocab = max(n_vocab or 0, max(vv.values()) + 1)
        except Exception:
            pass

    def t2i(s):
        for attr in ("token_to_id", "piece_to_id"):
            f = getattr(target, attr, None) if target is not None else None
            if callable(f):
                i = f(s)
                if i is None:
                    return None
                back = getattr(target, "id_to_token", None) or getattr(target, "id_to_piece", None)
                if callable(back) and back(int(i)) != s:
                    return None
                return int(i)
        sp = getattr(target, "special_tokens", None) if target is not None else None
        if isinstance(sp, dict) and s in sp:
            return int(sp[s])
        return None

    if n_vocab is None:
        raise RuntimeError("harness encoder from %s exposes no vocabulary size" % fn_name)
    return Enc(enc, dec, n_vocab, t2i, "eval/harness.py:%s" % fn_name, "harness sha256 %s" % hsha)


def _load_baselines_module():
    if BASELINES_DIR not in sys.path:
        sys.path.insert(0, BASELINES_DIR)
    os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # read-only use of baselines/
    try:
        return importlib.import_module("load_baselines")
    finally:
        sys.dont_write_bytecode = old


def baseline_spec(name):
    """The baselines/manifest.json entry of an external tokenizer (encoder_kind 'baseline')."""
    for e in _load_baselines_module().manifest()["baselines"]:
        if e["name"] == name:
            if not e.get("working") or (not e.get("load_path") and e.get("loader") != "bytes"):
                raise SystemExit("baseline %r is not a working baseline in baselines/manifest.json" % name)
            return e
    raise SystemExit("unknown baseline %r (see baselines/manifest.json)" % name)


def baseline_path_sha(spec):
    """(absolute load_path, sha256): a file's sha256, or for a directory the sha256 of its sorted
    'relative/path sha256' lines (every file in it)."""
    lp = spec.get("load_path")
    p = os.path.normpath(lp if os.path.isabs(lp) else os.path.join(BASELINES_DIR, lp))
    if os.path.isfile(p):
        return p, sha256_file(p)
    if os.path.isdir(p):
        lines = []
        for root, _, fs in os.walk(p):
            for fn in fs:
                full = os.path.join(root, fn)
                lines.append("%s %s" % (os.path.relpath(full, p).replace(os.sep, "/"), sha256_file(full)))
        return p, hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    raise SystemExit("baseline %s: load_path %s not found" % (spec["name"], p))


def _baseline_builtin(name):
    b = _load_baselines_module().load_baseline(name)
    raw = b.raw

    def t2i(s):
        f = getattr(raw, "token_to_id", None)          # tokenizers.Tokenizer
        if callable(f):
            return f(s)
        f = getattr(raw, "convert_tokens_to_ids", None)  # transformers
        if callable(f) and s in (getattr(raw, "all_special_tokens", None) or []):
            return int(f(s))
        f = getattr(raw, "piece_to_id", None)          # sentencepiece
        if callable(f):
            i = f(s)
            return i if raw.id_to_piece(i) == s and not raw.is_unknown(i) else None
        return None
    n = b.vocab_size
    gv = getattr(raw, "get_vocab", None)
    if callable(gv):
        try:
            v = gv(with_added_tokens=True)
        except TypeError:
            v = gv()
        n = max(n, max(v.values()) + 1)
    return Enc(b.encode, b.decode, n, t2i, "builtin:load_baselines", "baselines/load_baselines.py load_baseline(%r) "
               "(loader %s)" % (name, b.loader))


def load_encoder(path, kind, source, cid=None, custom_spec=None, no_newline_wrapper=False, baseline_name=None):
    """source: auto | harness | builtin. With the harness, the encoder is exactly the adapter that
    `harness.py run` evaluates (HFAdapter / SPAdapter with the PLAN 1.1 newline wrapper / CustomAdapter;
    kind 'baseline': adapters.from_baseline(baseline_name), the objects of baselines/load_baselines.py).
    no_newline_wrapper mirrors `harness.py run --spm X --no-newline-wrapper` (only for SentencePiece
    models trained without a '\\n' piece, e.g. the pilot models)."""
    if kind == "baseline" and not baseline_name:
        raise SystemExit("encoder kind 'baseline' needs a baseline name (%s)" % cid)
    if source in ("auto", "harness"):
        h, hsha = _import_harness()
        if h is not None and callable(getattr(h, "build_adapter", None)):
            if kind == "custom" and not custom_spec:
                raise SystemExit("custom encoder %s needs a 'custom' field MODULE:FACTORY in WAVES.json" % cid)
            ns = argparse.Namespace(tokenizer_json=path if kind == "hf" else None,
                                    spm=path if kind == "sentencepiece" else None,
                                    no_newline_wrapper=bool(no_newline_wrapper),
                                    baseline=baseline_name if kind == "baseline" else None,
                                    custom=custom_spec if kind == "custom" else None,
                                    name=cid or os.path.splitext(os.path.basename(path))[0])
            return _wrap_harness_adapter(h.build_adapter(ns), h, hsha)
        if h is not None:
            for name in ("load_encoder", "load_tokenizer", "get_encoder", "make_encoder", "load"):
                fn = getattr(h, name, None)
                if not callable(fn):
                    continue
                try:
                    params = inspect.signature(fn).parameters
                except (TypeError, ValueError):
                    params = {}
                if "kind" in params or "encoder_kind" in params or len(params) >= 2:
                    kw = "kind" if "kind" in params else ("encoder_kind" if "encoder_kind" in params else None)
                    obj = fn(path, **{kw: kind}) if kw else fn(path, kind)
                else:
                    obj = fn(path)
                return _wrap_harness_obj(obj, name, hsha)
            if source == "harness":
                raise SystemExit("eval/harness.py has no known loader (load_encoder/load_tokenizer/...)")
        elif source == "harness":
            raise SystemExit("eval/harness.py not found and --encoder-source harness was requested")
    if kind == "hf":
        return _hf_builtin(path)
    if kind == "sentencepiece":
        return _sp_builtin(path, wrapper=not no_newline_wrapper)
    if kind == "baseline":
        return _baseline_builtin(baseline_name)
    raise SystemExit("encoder kind %r needs eval/harness.py (custom encoders have no builtin loader)" % kind)


# ============================================================ worker
_TRAIN = None
_DEV = None


def _init_worker(train_texts, dev_texts):
    global _TRAIN, _DEV
    _TRAIN, _DEV = train_texts, dev_texts
    os.environ["RAYON_RS_NUM_CPUS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"


def _encode_all(enc, texts, batch=256):
    import numpy as np
    arrs, offs = [], [0]
    for j in range(0, len(texts), batch):
        for ids in enc.encode_batch(texts[j:j + batch]):
            a = np.asarray(ids, dtype=np.int64)
            arrs.append(a)
            offs.append(offs[-1] + len(a))
    assert len(arrs) == len(texts)
    toks = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.int64)
    return toks, np.asarray(offs, dtype=np.int64)


def encode_candidate(cand, stage_dir, enc_source, allow_rt_fail):
    import numpy as np
    t0 = time.time()
    enc = load_encoder(cand["path"], cand["kind"], enc_source, cid=cand["id"], custom_spec=cand.get("custom_spec"),
                       no_newline_wrapper=cand.get("no_newline_wrapper", False),
                       baseline_name=cand.get("baseline_name"))
    n_vocab = int(enc.n_vocab)
    appended = []
    ids = {}
    for role, names in (("eot", SPECIAL_EOT), ("bos", SPECIAL_BOS)):
        got = None
        for nm in names:
            i = enc.token_to_id(nm)
            if i is not None:
                got = (int(i), nm)
                break
        if got is None:
            got = (n_vocab, "<appended:%s>" % role)
            appended.append({"role": role, "id": n_vocab})
            n_vocab += 1
        ids[role] = got
    eot, bos = ids["eot"][0], ids["bos"][0]
    tr_toks, tr_offs = _encode_all(enc, _TRAIN)
    t_enc_train = time.time() - t0
    dv_toks, dv_offs = _encode_all(enc, _DEV)
    # G1 on dev: decode(encode(doc)) == doc
    rt_fail = []
    for k, t in enumerate(_DEV):
        back = enc.decode(dv_toks[dv_offs[k]:dv_offs[k + 1]].tolist())
        if back != t:
            pos = next((j for j, (a, b) in enumerate(zip(back, t)) if a != b), min(len(back), len(t)))
            rt_fail.append({"doc": k, "first_diff_char": pos})
    problems = []
    if rt_fail and not allow_rt_fail:
        problems.append("round trip fails on %d dev docs (first: %s)" % (len(rt_fail), rt_fail[:3]))
    for name, tt in (("train", tr_toks), ("dev", dv_toks)):
        if len(tt) and (tt.min() < 0 or tt.max() >= n_vocab):
            problems.append("%s ids out of range [0, %d)" % (name, n_vocab))
        nspec = int(np.isin(tt, [eot, bos]).sum())
        if nspec:
            problems.append("%s: %d special ids (EOT/BOS) inside documents (PLAN G3)" % (name, nspec))
    if problems:
        raise RuntimeError("candidate %s: %s" % (cand["id"], "; ".join(problems)))
    dtype = np.uint16 if n_vocab <= 65536 else np.uint32
    cdir = os.path.join(stage_dir, "cand", slug(cand["id"]))
    os.makedirs(cdir, exist_ok=True)
    files = {}
    for name, arr in (("train_tokens", tr_toks.astype(dtype)), ("train_offsets", tr_offs),
                      ("dev_tokens", dv_toks.astype(dtype)), ("dev_offsets", dv_offs)):
        rel = "cand/%s/%s.npy" % (slug(cand["id"]), name)
        np.save(os.path.join(stage_dir, rel), arr, allow_pickle=False)
        files["%s/%s" % (cand["id"], name)] = rel
    return {"id": cand["id"], "kind": cand["kind"], "n_vocab": n_vocab, "eot_id": eot, "bos_id": bos,
            "eot_token": ids["eot"][1], "bos_token": ids["bos"][1], "specials_appended": appended,
            "dtype": np.dtype(dtype).name, "train_tokens": int(len(tr_toks)), "dev_tokens": int(len(dv_toks)),
            "encoder_source": enc.source, "encoder_detail": enc.detail, "encoder_identity": enc.identity,
            "roundtrip_dev_failures": len(rt_fail), "roundtrip_dev_examples": rt_fail[:5],
            "encode_seconds": {"train": round(t_enc_train, 1), "total": round(time.time() - t0, 1)},
            "files": files}


# ============================================================ packing
def _tar_filter(ti):
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def pack(stage_dir, upload_dir, part_bytes, preset):
    os.makedirs(upload_dir, exist_ok=True)
    for f in os.listdir(upload_dir):
        if f.startswith(ARCHIVE + ".part") or f == "PARTS.json":
            os.remove(os.path.join(upload_dir, f))
    names = []
    for root, _, fs in os.walk(stage_dir):
        for fn in fs:
            full = os.path.join(root, fn)
            names.append(os.path.relpath(full, stage_dir).replace(os.sep, "/"))
    names.sort()
    tar_path = os.path.join(upload_dir, "_bundle.tar")
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        for n in names:
            tf.add(os.path.join(stage_dir, n), arcname="bundle/" + n, filter=_tar_filter)
    # xz; lc0/lp1/pb1 suit the 2-byte token arrays (measured on a uint16 token array: ratio 0.581 vs
    # 0.589 with the default lc3/lp0/pb2 - small, but free)
    filters = [{"id": lzma.FILTER_LZMA2, "preset": preset, "lc": 0, "lp": 1, "pb": 1}]
    xz_path = os.path.join(upload_dir, "_" + ARCHIVE)
    with open(tar_path, "rb") as fi, lzma.open(xz_path, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                                                  filters=filters) as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    tar_bytes = os.path.getsize(tar_path)
    os.remove(tar_path)
    arch_sha = sha256_file(xz_path)
    arch_bytes = os.path.getsize(xz_path)
    parts = []
    with open(xz_path, "rb") as f:
        k = 0
        while True:
            blk = f.read(part_bytes)
            if not blk:
                break
            name = "%s.part%03d" % (ARCHIVE, k)
            with open(os.path.join(upload_dir, name), "wb") as g:
                g.write(blk)
            parts.append({"name": name, "bytes": len(blk), "sha256": hashlib.sha256(blk).hexdigest()})
            k += 1
    os.remove(xz_path)
    return {"archive": ARCHIVE, "archive_sha256": arch_sha, "archive_bytes": arch_bytes, "tar_bytes": tar_bytes,
            "compression": {"format": "xz", "filters": "LZMA2 preset %d lc0 lp1 pb1" % preset},
            "part_max_bytes": part_bytes, "parts": parts}


def final_test_log(entry=None, update=None):
    """--final-test audit trail: append `entry` to FINAL_TEST_LOG.json and return its index, or merge
    update=(index, fields) into an existing entry. An entry left at status 'started' is an attempt
    that opened the test file but did not finish."""
    doc = {"what": "Every build of a one-shot FINAL TEST bundle (colab/build_bundle.py --final-test; PLAN 7.6). "
                   "The strict test split is opened only by these builds.", "entries": []}
    if os.path.isfile(FINAL_TEST_LOG):
        with open(FINAL_TEST_LOG, encoding="utf-8") as f:
            doc = json.load(f)
    if entry is not None:
        doc["entries"].append(entry)
        idx = len(doc["entries"]) - 1
    else:
        idx = update[0]
        doc["entries"][idx].update(update[1])
    tmp = FINAL_TEST_LOG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, FINAL_TEST_LOG)
    return idx


# ============================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Colab LM-arbiter bundle.")
    ap.add_argument("--waves", default=WAVES_DEFAULT)
    ap.add_argument("--candidates", nargs="+", default=None, metavar="PATH KIND",
                    help="testing: path1 kind1 path2 kind2 ... (kind: hf | sentencepiece | custom)")
    ap.add_argument("--ids", nargs="+", default=None, help="ids for --candidates (default: file stems)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--train-jsonl", default=os.path.join(DATA_DIR, "train_D1.jsonl"))
    ap.add_argument("--dev-jsonl", default=None,
                    help="default data/dev_strict.jsonl (data/test_strict.jsonl with --final-test)")
    ap.add_argument("--final-test", action="store_true",
                    help="ONE-SHOT final test (PLAN 7.6): encode data/test_strict.jsonl in place of dev_strict, "
                         "manifest kind='final_test', record in FINAL_TEST_LOG.json")
    ap.add_argument("--report-only", default=None, metavar="PURPOSE",
                    help="with --final-test: a REPORT-ONLY re-use of the test split (manifest kind "
                         "'final_test_report'; no decision depends on it); PURPOSE is recorded in the log entry")
    ap.add_argument("--max-train-docs", type=int, default=None, help="testing: first N train docs by uid")
    ap.add_argument("--max-dev-docs", type=int, default=None, help="testing: first N dev docs by uid")
    ap.add_argument("--out", default=None, help="default colab/build (colab/build_final_test with --final-test)")
    ap.add_argument("--part-mb", type=float, default=PART_BYTES_DEFAULT / 1e6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--encoder-source", default="auto", choices=["auto", "harness", "builtin"],
                    help="auto = eval/harness.py when present (the sweep's exact encoder), else builtin")
    ap.add_argument("--no-newline-wrapper", action="store_true",
                    help="testing with --candidates: SentencePiece models without a '\\n' piece (pilot models), "
                         "as `harness.py run --no-newline-wrapper`")
    ap.add_argument("--allow-roundtrip-failures", action="store_true", help="testing only; recorded in the manifest")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--parity-threads", type=int, default=2)
    ap.add_argument("--xz-preset", type=int, default=9)
    ap.add_argument("--no-copy-parity", action="store_true", help="do not copy PARITY.json to colab/PARITY.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.report_only is not None and (not a.final_test or not a.report_only.strip()):
        raise SystemExit("--report-only PURPOSE needs --final-test and a non-empty PURPOSE")
    kind_label = "final_test_report" if a.report_only else "final_test"
    if a.dev_jsonl is None:
        a.dev_jsonl = os.path.join(DATA_DIR, "test_strict.jsonl" if a.final_test else "dev_strict.jsonl")
    if a.out is None:
        a.out = os.path.join(HERE, ("build_final_test_report" if a.report_only else "build_final_test")
                             if a.final_test else "build")
    if a.final_test and (a.max_train_docs or a.max_dev_docs or a.allow_roundtrip_failures):
        raise SystemExit("--final-test encodes the whole train and test files and needs G1 on every test document "
                         "(no --max-train-docs / --max-dev-docs / --allow-roundtrip-failures)")
    os.environ.setdefault("RAYON_RS_NUM_CPUS", "1")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ---- candidates
    waves_sha, top_base = None, None
    if a.candidates:
        if len(a.candidates) % 2:
            raise SystemExit("--candidates takes pairs: path kind")
        pairs = list(zip(a.candidates[0::2], a.candidates[1::2]))
        ids = a.ids or [os.path.splitext(os.path.basename(p))[0] for p, _ in pairs]
        if len(ids) != len(pairs):
            raise SystemExit("--ids must match the number of candidates")
        cands = [{"id": i, "path": os.path.abspath(p), "kind": norm_kind(k), "rank": r, "waves_fields": {},
                  "no_newline_wrapper": bool(a.no_newline_wrapper) and norm_kind(k) == "sentencepiece",
                  "baseline_name": p if norm_kind(k) == "baseline" else None}      # --candidates NAME baseline
                 for r, (i, (p, k)) in enumerate(zip(ids, pairs), 1)]
        cand_source = "command line"
    else:
        if not os.path.isfile(a.waves):
            raise SystemExit("WAVES.json not found at %s (use --candidates for testing)" % a.waves)
        cands, top_base, waves_sha = load_waves(a.waves)
        cand_source = a.waves
    for c in cands:
        if c["kind"] not in ("hf", "sentencepiece", "custom", "baseline"):
            raise SystemExit("unknown encoder kind %r for %s" % (c["kind"], c["id"]))
        if c["kind"] == "custom" and not c.get("custom_spec") and ":" in os.path.basename(c["path"]) \
                and not os.path.isfile(c["path"]):
            c["custom_spec"] = os.path.basename(c["path"])        # --candidates MODULE:FACTORY custom
        if c["kind"] == "baseline":
            spec = baseline_spec(c["baseline_name"])
            bpath, bsha = baseline_path_sha(spec)
            if a.candidates is None and os.path.normcase(c["path"]) != os.path.normcase(bpath):
                raise SystemExit("%s: tokenizer_path %s is not the load_path %s of baseline %s"
                                 % (c["id"], c["path"], bpath, c["baseline_name"]))
            c["path"], c["tokenizer_sha256"] = bpath, bsha
            c["baseline_spec"] = {k: spec.get(k) for k in (
                "name", "tier", "provider", "family", "repo", "revision", "official", "mirror_kind", "loader",
                "load_path", "algorithm", "byte_fallback", "vocab_size", "vocab_sha256", "lossless",
                "newline_handling", "same_encodings_as", "identical_to", "license", "year")}
            continue
        if os.path.isfile(c["path"]):
            c["tokenizer_sha256"] = sha256_file(c["path"])
        elif c["kind"] == "custom" and c.get("custom_spec"):
            c["tokenizer_sha256"] = None                          # identity recorded by the harness adapter
        else:
            raise SystemExit("tokenizer file missing for %s: %s" % (c["id"], c["path"]))
    baseline, base_rule = pick_baseline(cands, a.baseline, top_base)
    ids_in = [c["id"] for c in cands]                             # input order (final-test log)
    cands.sort(key=lambda c: (c["id"] != baseline, c["rank"]))    # baseline first, then WAVES order
    log("candidates (%s): %s | baseline %s (%s)" % (cand_source, [c["id"] for c in cands], baseline, base_rule))

    # ---- text (train + dev only)
    man, man_sha = load_split_manifest()
    if man is None:
        log("WARNING: split manifest not found; uid/split guard disabled")
    train = read_jsonl(a.train_jsonl, "train", man)
    dev = read_jsonl(a.dev_jsonl, "test" if a.final_test else "validation", man, require_strict=True,
                     final_test=a.final_test)
    n_train_all, n_dev_all = len(train), len(dev)
    if a.max_train_docs:
        train = train[:a.max_train_docs]
    if a.max_dev_docs:
        dev = dev[:a.max_dev_docs]
    empty_tr = sum(1 for r in train if not r["text"])
    empty_dv = sum(1 for r in dev if not r["text"])
    train = [r for r in train if r["text"]]
    dev = [r for r in dev if r["text"]]
    import numpy as np
    tr_bytes = np.array([len(r["text"].encode("utf-8")) for r in train], dtype=np.int64)
    dv_bytes = np.array([len(r["text"].encode("utf-8")) for r in dev], dtype=np.int64)
    log("text: train %d docs %.3f MB | %s %d docs %.3f MB | %.1fs" %
        (len(train), tr_bytes.sum() / 1e6, "TEST_strict (stored as 'dev')" if a.final_test else "dev_strict",
         len(dev), dv_bytes.sum() / 1e6, time.time() - t0))
    data_man_p = os.path.join(os.path.dirname(os.path.abspath(a.train_jsonl)), "data_manifest.json")
    data_man = None
    if os.path.isfile(data_man_p):
        with open(data_man_p, encoding="utf-8") as f:
            data_man = json.load(f)
    tr_sha, dv_sha = sha256_file(a.train_jsonl), sha256_file(a.dev_jsonl)
    if data_man:
        for p, s in ((a.train_jsonl, tr_sha), (a.dev_jsonl, dv_sha)):
            v = data_man.get("views", {}).get(os.path.splitext(os.path.basename(p))[0])
            if v and v.get("sha256") and v["sha256"] != s:
                raise SystemExit("%s sha256 differs from data_manifest.json" % p)
    log_idx, test_man_p = None, None
    if a.final_test:
        test_man_p = os.path.join(os.path.dirname(os.path.abspath(a.dev_jsonl)), "test_manifest.json")
        if not os.path.isfile(test_man_p):
            raise SystemExit("--final-test needs %s (written by data/materialize_test.py)" % test_man_p)
        with open(test_man_p, encoding="utf-8") as f:
            tm = json.load(f)
        tv = tm.get("views", {}).get(os.path.splitext(os.path.basename(a.dev_jsonl))[0]) or {}
        if tv.get("sha256") != dv_sha or tm.get("split_manifest_sha256") != man_sha:
            raise SystemExit("%s does not match %s (file sha256 / split manifest sha256)" % (a.dev_jsonl, test_man_p))
        h, _ = _import_harness()
        prev = 0
        if os.path.isfile(FINAL_TEST_LOG):
            with open(FINAL_TEST_LOG, encoding="utf-8") as f:
                prev = len(json.load(f).get("entries", []))
        if prev:
            log("WARNING: FINAL_TEST_LOG.json already holds %d final-test build(s); this one is recorded too" % prev)
        entry = {
            "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "started", "previous_entries": prev, "argv": list(argv) if argv is not None else sys.argv[1:],
            "candidates": [{"id": c["id"], "role": c.get("waves_fields", {}).get("final_test_role"),
                            "encoder_kind": c["kind"], "tokenizer_path": c["path"],
                            "tokenizer_sha256": c["tokenizer_sha256"],
                            **({"baseline_name": c["baseline_name"]} if c["kind"] == "baseline" else {})}
                           for c in cands],
            "candidate_ids_input_order": ids_in,
            "candidate_ids_sha256": hashlib.sha256("\n".join(ids_in).encode("utf-8")).hexdigest(),
            "baseline_id": baseline, "candidates_source": cand_source, "waves_sha256": waves_sha,
            "test_file": {"path": os.path.abspath(a.dev_jsonl), "sha256": dv_sha, "docs": len(dev),
                          "bytes": int(dv_bytes.sum()), "test_manifest_sha256": sha256_file(test_man_p)},
            "train_file": {"path": os.path.abspath(a.train_jsonl), "sha256": tr_sha},
            "split_manifest_sha256": man_sha,
            "code": {"build_bundle_sha256": sha256_file(os.path.abspath(__file__)),
                     "hk_lm_sha256": sha256_file(os.path.join(HERE, "hk_lm.py")),
                     "run_all_sha256": sha256_file(os.path.join(HERE, "run_all.py")),
                     "harness_code_sha256": h.code_hashes() if h is not None else None},
            "out": os.path.abspath(a.out)}
        if a.report_only:
            entry = {"kind": kind_label, "report_only": True,
                     "test_reuse": "The strict test split is used AGAIN here (after the one-shot final test of "
                                   "entries 0-1), for a REPORT-ONLY comparison. No decision depends on it: the "
                                   "released tokenizer is fixed (analysis/AMENDMENT_1.md) and nothing is re-selected "
                                   "or tuned on these numbers.",
                     "purpose": a.report_only.strip(), **entry}
        log_idx = final_test_log(entry=entry)
        log("FINAL TEST%s: record #%d appended to %s" % (" (REPORT-ONLY re-use)" if a.report_only else "",
                                                         log_idx, FINAL_TEST_LOG))

    # ---- staging
    stage = os.path.join(a.out, "staging")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(os.path.join(stage, "shared"))
    files = {}

    def save_np(key, rel, arr):
        np.save(os.path.join(stage, rel), arr, allow_pickle=False)
        files[key] = rel

    def save_text(key, rel, text):
        with open(os.path.join(stage, rel), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        files[key] = rel

    save_np("train_bytes", "shared/train_bytes.npy", tr_bytes)
    save_np("dev_bytes", "shared/dev_bytes.npy", dv_bytes)
    save_text("train_uids", "shared/train_uids.txt", "\n".join(r["uid"] for r in train) + "\n")
    save_text("dev_docs", "shared/dev_docs.json", json.dumps(
        [{"uid": r["uid"], "group": r["group"], "cluster": r["cluster"], "source": r["source"],
          "variety": r["variety"], "bytes": int(b)} for r, b in zip(dev, dv_bytes)], ensure_ascii=False))
    perm = np.random.Generator(np.random.PCG64(SUBSET_SEED)).permutation(len(dev))
    cum = np.cumsum(dv_bytes[perm])
    k = min(len(dev), int(np.searchsorted(cum, SUBSET_BYTES, side="left")) + 1)
    subset_order = perm[:k]
    kp = max(1, int(np.searchsorted(cum, PARITY_EVAL_BYTES, side="right")))
    parity_idx = np.sort(perm[:min(kp, k)]).astype(np.int64)
    subset_idx = np.sort(subset_order).astype(np.int64)
    save_np("dev_subset_idx", "shared/dev_subset_idx.npy", subset_idx)
    save_np("dev_parity_idx", "shared/dev_parity_idx.npy", parity_idx)

    # ---- encode (<= 2 processes)
    train_texts = [r["text"] for r in train]
    dev_texts = [r["text"] for r in dev]
    metas = {}
    nw = max(1, min(a.workers, 2, len(cands)))
    if nw == 1:
        _init_worker(train_texts, dev_texts)
        for c in cands:
            log("encoding %s (%s) ..." % (c["id"], c["kind"]))
            metas[c["id"]] = encode_candidate(c, stage, a.encoder_source, a.allow_roundtrip_failures)
    else:
        with cf.ProcessPoolExecutor(max_workers=nw, initializer=_init_worker,
                                    initargs=(train_texts, dev_texts)) as ex:
            futs = {ex.submit(encode_candidate, c, stage, a.encoder_source, a.allow_roundtrip_failures): c["id"]
                    for c in cands}
            for fu in cf.as_completed(futs):
                metas[futs[fu]] = fu.result()
                m = metas[futs[fu]]
                log("encoded %s: %d train / %d dev tokens, %s, %.0fs, rt failures %d" %
                    (m["id"], m["train_tokens"], m["dev_tokens"], m["encoder_source"],
                     m["encode_seconds"]["total"], m["roundtrip_dev_failures"]))
    import hk_lm
    cand_meta = []
    for c in cands:
        m = metas[c["id"]]
        bpt_tr = float(tr_bytes.sum()) / m["train_tokens"]
        bpt_dv = float(dv_bytes.sum()) / m["dev_tokens"]
        files.update(m.pop("files"))
        arrays = {n: "%s/%s" % (c["id"], n) for n in ("train_tokens", "train_offsets", "dev_tokens", "dev_offsets")}
        if c["kind"] == "baseline":
            m = dict(m, baseline_name=c["baseline_name"], baseline_spec=c["baseline_spec"])
        cand_meta.append(dict(m, rank=c["rank"], tokenizer_path=c["path"], tokenizer_file=os.path.basename(c["path"]),
                              tokenizer_sha256=c["tokenizer_sha256"], custom_spec=c.get("custom_spec"),
                              sp_newline_wrapper=(not c.get("no_newline_wrapper")) if c["kind"] == "sentencepiece" else None,
                              waves_fields=c.get("waves_fields", {}),
                              train_bytes_per_token=round(bpt_tr, 6), dev_bytes_per_token=round(bpt_dv, 6),
                              ctx_tokens=hk_lm.ctx_tokens(round(bpt_tr, 6)), is_baseline=c["id"] == baseline,
                              arrays=arrays))
        log("  %-28s n_vocab %6d  train b/t %.4f  ctx %d  dev b/t %.4f  eot %d bos %d%s" %
            (c["id"], m["n_vocab"], bpt_tr, hk_lm.ctx_tokens(round(bpt_tr, 6)), bpt_dv, m["eot_id"], m["bos_id"],
             "  (appended %s)" % [x["role"] for x in m["specials_appended"]] if m["specials_appended"] else ""))

    # ---- code files + manifest
    shutil.copy2(os.path.join(HERE, "hk_lm.py"), os.path.join(stage, "hk_lm.py"))
    shutil.copy2(os.path.join(HERE, "run_all.py"), os.path.join(stage, "run_all.py"))
    files["hk_lm.py"] = "hk_lm.py"
    files["run_all.py"] = "run_all.py"
    file_meta = {}
    for key, rel in sorted(files.items()):
        p = os.path.join(stage, rel)
        file_meta[key] = {"path": rel, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    manifest = {
        "format": "hk_bundle/1",
        "protocol": hk_lm.PROTOCOL,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": {"build_bundle_sha256": sha256_file(os.path.abspath(__file__)),
                    "hk_lm_sha256": file_meta["hk_lm.py"]["sha256"], "run_all_sha256": file_meta["run_all.py"]["sha256"],
                    "python": sys.version.split()[0]},
        "candidates_source": cand_source, "waves_sha256": waves_sha,
        "baseline_id": baseline, "baseline_rule": base_rule,
        "data": {"train_jsonl": os.path.abspath(a.train_jsonl), "train_sha256": tr_sha,
                 "dev_jsonl": os.path.abspath(a.dev_jsonl), "dev_sha256": dv_sha,
                 "data_manifest_sha256": sha256_file(data_man_p) if data_man else None,
                 "normalize_version": (data_man or {}).get("normalize_version"),
                 "split_manifest_sha256": man_sha,
                 "train_view": "D1 permissive train, all documents, sorted by uid",
                 "dev_view": "dev_strict (validation, quality_tier strict), sorted by uid",
                 "n_train_docs": len(train), "n_dev_docs": len(dev),
                 "n_train_docs_in_file": n_train_all, "n_dev_docs_in_file": n_dev_all,
                 "train_bytes": int(tr_bytes.sum()), "dev_bytes": int(dv_bytes.sum()),
                 "empty_docs_dropped": {"train": empty_tr, "dev": empty_dv},
                 "slice_for_testing": ({"max_train_docs": a.max_train_docs, "max_dev_docs": a.max_dev_docs}
                                       if (a.max_train_docs or a.max_dev_docs) else None),
                 "bytes_definition": "UTF-8 bytes of the canonical (normalised) document text, no separators"},
        "dev_subset": {"seed": SUBSET_SEED, "target_bytes": SUBSET_BYTES, "n_docs": int(len(subset_idx)),
                       "bytes": int(dv_bytes[subset_idx].sum()),
                       "rule": "PCG64(12345).permutation(n_dev); take docs until >= 0.5 MB; indices sorted"},
        "dev_parity": {"n_docs": int(len(parity_idx)), "bytes": int(dv_bytes[parity_idx].sum()),
                       "rule": "leading docs of the same permutation up to 100 KB (at least 1)"},
        "allow_roundtrip_failures": bool(a.allow_roundtrip_failures),
        "candidates": cand_meta,
        "files": file_meta,
        "test_split": "never read, encoded or bundled",
    }
    if a.final_test:
        manifest = {"format": manifest["format"], "kind": kind_label, **manifest}
        manifest["data"]["dev_view"] = ("test_strict = the strict TEST split (split test, quality_tier strict), "
                                        "sorted by uid, stored under the 'dev' names")
        manifest["test_split"] = ("ENCODED: this is the one-shot FINAL TEST bundle (PLAN 7.6). The 'dev' arrays, "
                                  "dev_docs, dev_subset, dev_parity and every 'dev' number computed from this "
                                  "bundle are the strict TEST split")
        manifest["final_test"] = {"test_manifest": test_man_p, "test_manifest_sha256": sha256_file(test_man_p),
                                  "log": FINAL_TEST_LOG, "log_entry": log_idx,
                                  "candidate_ids_sha256": hashlib.sha256("\n".join(ids_in).encode("utf-8")).hexdigest(),
                                  "run_all": "only the parity and confirm stages run on this bundle (run_all.py "
                                             "refuses the others); see colab/FINAL_TEST.md"}
        if a.report_only:
            manifest["test_split"] = ("ENCODED: REPORT-ONLY re-use of the strict TEST split (kind "
                                      "'final_test_report'), after the one-shot final test. No decision depends on "
                                      "it. The 'dev' arrays, dev_docs, dev_subset, dev_parity and every 'dev' number "
                                      "computed from this bundle are the strict TEST split")
            manifest["final_test"].update({
                "report_only": True, "purpose": a.report_only.strip(),
                "run_all": "only the parity and confirm stages run on this bundle, with a fixed --lr; confirm runs "
                           "seeds 1-3 (the PLAN 6 power check is written but never adds seeds; --force-extra-seeds "
                           "adds 4 and 5); no LR sweep or re-sweep (run_all.py refuses the rest)"})
    with open(os.path.join(stage, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    msha = sha256_file(os.path.join(stage, "manifest.json"))
    log("manifest sha256 %s (%d files)" % (msha, len(file_meta)))

    # ---- CPU parity reference
    parity = None
    if not a.no_parity:
        b = hk_lm.Bundle(stage, verify=True)
        rt = hk_lm.Runtime("cpu", threads=a.parity_threads)
        log("parity: %d steps, seed 1, lr %g, baseline %s, CPU %d threads ..." %
            (hk_lm.PARITY_STEPS, hk_lm.PARITY_LR, baseline, rt.threads))
        rec = hk_lm.run_one(b, baseline, "parity", 1, hk_lm.PARITY_LR, rt, label="parity")
        parity = {"what": "CPU reference of the fixed tiny config; run_all.py re-runs it on the GPU first",
                  "bundle_manifest_sha256": msha, "candidate": baseline, "stage_recipe": "parity",
                  "config": {"model": "screen", "steps": rec["data"]["steps"], "seed": 1, "lr": hk_lm.PARITY_LR,
                             "ctx_tokens": rec["data"]["ctx_tokens"], "budget_bytes": rec["data"]["budget_bytes"],
                             "eval_docs": rec["dev"]["n_docs"], "eval_bytes": rec["dev"]["sum_bytes"]},
                  "tolerance_rel_bpb": 0.01,
                  "cpu": {"bpb": rec["bpb"], "sum_bits": rec["dev"]["sum_bits"], "sum_bytes": rec["dev"]["sum_bytes"],
                          "final_loss": rec["train_loss"][-1], "first_loss": rec["train_loss"][0],
                          "train_loss": rec["train_loss"], "bits": rec["dev"]["bits"],
                          "runtime": rec["runtime"], "wall_time_s": rec["wall_time_s"]},
                  "hk_lm_sha256": rec["hk_lm_sha256"]}
        with open(os.path.join(stage, "PARITY.json"), "w", encoding="utf-8") as f:
            json.dump(parity, f, ensure_ascii=False, indent=1)
        if not a.no_copy_parity and not a.final_test:
            shutil.copy2(os.path.join(stage, "PARITY.json"), os.path.join(HERE, "PARITY.json"))
        log("parity CPU bpb %.6f (%.0fs)" % (rec["bpb"], rec["wall_time_s"]["total"]))

    # ---- pack + split
    upload = os.path.join(a.out, "upload")
    info = pack(stage, upload, int(a.part_mb * 1e6), a.xz_preset)
    shutil.copy2(os.path.join(HERE, "run_all.py"), os.path.join(upload, "run_all.py"))
    nb = os.path.join(HERE, "hindko_lm_arbiter.ipynb")
    if os.path.isfile(nb):
        shutil.copy2(nb, os.path.join(upload, os.path.basename(nb)))
    parts_json = dict(info, bundle_manifest_sha256=msha, created_utc=manifest["created_utc"],
                      candidates=[c["id"] for c in cand_meta], baseline_id=baseline,
                      loose_files={"run_all.py": sha256_file(os.path.join(upload, "run_all.py"))},
                      parity_included=parity is not None,
                      upload_note="upload every *.part### file, PARTS.json and run_all.py into ONE folder")
    if a.final_test:
        parts_json["kind"] = kind_label
    with open(os.path.join(upload, "PARTS.json"), "w", encoding="utf-8") as f:
        json.dump(parts_json, f, indent=1)
    tot = sum(p["bytes"] for p in info["parts"])
    log("packed: tar %.1f MB -> xz %.1f MB in %d parts (max %.2f MB) | %s" %
        (info["tar_bytes"] / 1e6, tot / 1e6, len(info["parts"]), max(p["bytes"] for p in info["parts"]) / 1e6, upload))
    if a.final_test:
        final_test_log(update=(log_idx, {
            "status": "completed",
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bundle_manifest_sha256": msha, "archive_sha256": info["archive_sha256"], "n_parts": len(info["parts"]),
            "g1_roundtrip_test_failures": {c["id"]: c["roundtrip_dev_failures"] for c in cand_meta},
            "test_tokens": {c["id"]: c["dev_tokens"] for c in cand_meta},
            "parity_cpu_bpb_on_test_slice": parity["cpu"]["bpb"] if parity else None}))
        log("FINAL TEST: record #%d completed (bundle %s)" % (log_idx, msha[:12]))
    log("done in %.0fs" % (time.time() - t0))
    return {"stage": stage, "upload": upload, "manifest_sha256": msha, "parts": info, "parity": parity}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
