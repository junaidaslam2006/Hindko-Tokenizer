# -*- coding: utf-8 -*-
"""Hindko tokenizer evaluation harness (PLAN.md Stage 0, sections 1.3, 4.2, 4.3).

One entry point for every tokenizer the study compares:
    python harness.py run --tokenizer-json PATH [--name N]          HF tokenizers file
    python harness.py run --spm PATH.model [--no-newline-wrapper]   SentencePiece (PLAN 1.1 newline wrapper on)
    python harness.py run --baseline NAME                           external baseline (baselines/load_baselines.py)
    python harness.py run --custom MODULE:FACTORY                   custom Python encoder (adapters.CustomAdapter)
      common options: --data dev_strict|dev_permissive|PATH   --gates g1|all   --train-data train_D1
                      --g4-retrain PATH   --nsl-ref RESULTS_DIR   --out DIR   --threads N   --force
    python harness.py nsl --ref RESULTS_DIR TARGET_DIR...           add NSL to finished runs (post hoc)
    python harness.py census [--data dev_strict]                    tokenizer-independent facts: perturbation
                                                                    coverage, G3 corpus scan
From Python:  run(adapter, data='dev_strict', ...) with an adapter from adapters.py.

Outputs (default eval/results/<dataset>/<name>/):
    docs.jsonl     one row per document (fields in DOC_FIELDS below), for the cluster bootstrap of PLAN 6
    summary.json   aggregates overall / by source / by language_variety, gates, properties, provenance

Metric definitions (all per document, aggregated as ratios of sums):
    bytes_per_token      sum UTF-8 bytes / sum tokens (canonical text, no BOS/EOS)
    chars_per_token      sum characters / sum tokens
    fertility            sum over whitespace words of the number of tokens overlapping the word, from the
                         in-context encoding of the whole document and the tokens' character offsets
                         (PLAN 4.2). Tokens whose text is empty or whitespace (a lone '▁', '\\n') belong to no
                         word; a token with an empty span belongs to the word containing its position.
                         A superword token counts once for each word it touches.
    tokens_per_word      sum tokens / sum words (plain ratio; counts whitespace tokens too)
    continued_word_rate  share of words overlapped by >= 2 tokens;  strr = share overlapped by exactly 1
    renyi_eff_a2.5/a2.0  H_alpha(p) / log2 |V| of the pooled token distribution (Zouhar et al.; TokEval). REPORT ONLY
    vocab_utilisation    distinct ids seen / |V|
    nsl                  sum tokens(candidate) / sum tokens(reference) on the same documents (reference =
                         A1-P1-16k once it exists; `nsl` subcommand)
    robustness[p]        for each perturbation p of perturb.py (harakat, digits, punct_space, zwnj):
                         rel_token_change = (tokens(perturbed) - tokens) / tokens over all documents;
                         extra_tokens_per_affected_word; seg_change_rate = share of compared words whose
                         segmentation changed; seg_change_rate_affected = the same among words the
                         perturbation touched. A word's segmentation is compared on the characters it shares
                         with its perturbed image: the partition of those characters into tokens, the set of
                         characters split inside by a token boundary (sub-character tokens), and whether a token
                         crosses into a neighbouring word, must all be equal.
    g1                   decode(encode(doc)) == doc; failures are also classified (equal modulo whitespace?
                         non-whitespace characters lost / added) and UNK tokens counted
    bytes_per_token_lines  the same text encoded line by line (no line-break tokens at all); lets tokenizers
                         that drop line breaks be compared on equal terms (reported, not ranked)
    encode speed         documents/s and MB/s of plain encode() on this CPU, one thread (sanity only)
    morphology           morphology/morph_eval.py on the SILVER sets (report only; PLAN 4.2)
Gates (PLAN 4.3): G1 lossless; G2 self-tokenization of every learned whole-character token; G3 special block
atomic and absent from corpus text; G4 determinism (needs --g4-retrain); G5 no sub-character tokens in
character-level vocabularies. Properties: R1 train support profile; R2 partial-UTF-8 tokens.

TEST DISCIPLINE: the harness refuses any dataset containing a test-split uid unless --allow-test is given
(reserved for PLAN Stage 5).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.dirname(HERE)
DATA_DIR = os.path.join(TOK, "data")
RESULTS = os.path.join(HERE, "results")
MORPH_DIR = os.path.join(TOK, "morphology")
FROZEN_PATH = os.path.join(TOK, "FROZEN.json")
HARNESS_VERSION = "1.0.0"
PERT_NAMES = ("harakat", "digits", "punct_space", "zwnj")
DATASETS = ("dev_strict", "dev_permissive", "train_D1", "train_D2", "train_D3")
SPECIAL_TOKENS = (["<|endoftext|>", "<|bos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]
                  + ["<|reserved_%d|>" % i for i in range(59)])      # PLAN 2.1: 64 in total
assert len(SPECIAL_TOKENS) == 64
FULL_STOP = chr(0x06D4)
DOC_FIELDS = ["uid", "source", "variety", "cluster", "group", "bytes", "chars", "words", "tokens", "pieces", "cont",
              "single", "nocover", "rt", "rt_ws", "lost", "added", "unk", "lines_tokens", "lines_bytes", "enc_s"] + \
             ["%s_%s" % (p, f) for p in PERT_NAMES for f in ("tok", "edits", "cmp", "chg", "aff", "achg", "van")]


def _setup_threads(n: int):
    os.environ.setdefault("RAYON_NUM_THREADS", str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if n > 1 else "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import perturb as P  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def clean(x):
    """Strict JSON: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def code_hashes():
    files = {"harness.py": os.path.join(HERE, "harness.py"), "adapters.py": os.path.join(HERE, "adapters.py"),
             "perturb.py": os.path.join(HERE, "perturb.py"), "morph_eval.py": os.path.join(MORPH_DIR, "morph_eval.py")}
    return {k: sha256_file(v) for k, v in files.items()}


# ------------------------------------------------------------------------------------------- data
_MANIFEST_SPLIT = None


def _manifest_split():
    global _MANIFEST_SPLIT
    if _MANIFEST_SPLIT is None:
        with open(FROZEN_PATH, encoding="utf-8") as f:
            fr = json.load(f)
        path = fr["split_manifest"]["path"]
        if sha256_file(path) != fr["split_manifest"]["sha256"]:
            raise SystemExit("split manifest differs from FROZEN.json")
        _MANIFEST_SPLIT = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _MANIFEST_SPLIT[r["uid"]] = r["split"]
    return _MANIFEST_SPLIT


def resolve_data(data: str) -> str:
    return os.path.join(DATA_DIR, data + ".jsonl") if data in DATASETS else data


def load_docs(data: str, allow_test: bool = False) -> (List[dict], dict):
    path = resolve_data(data)
    with open(path, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f]
    split = _manifest_split()
    n_test = sum(1 for d in docs if split.get(d["uid"]) == "test")
    if n_test and not allow_test:
        raise SystemExit("REFUSED: %s contains %d test-split documents (PLAN: test is used once, Stage 5; "
                         "pass --allow-test only there)" % (path, n_test))
    info = {"name": os.path.splitext(os.path.basename(path))[0], "path": path, "sha256": sha256_file(path),
            "docs": len(docs), "test_docs": n_test}
    dm = os.path.join(DATA_DIR, "data_manifest.json")
    if os.path.exists(dm) and info["name"] in DATASETS:
        rec = json.load(open(dm, encoding="utf-8"))["views"].get(info["name"])
        if rec and rec["sha256"] != info["sha256"]:
            raise SystemExit("%s does not match data/data_manifest.json" % path)
        info["matches_data_manifest"] = bool(rec)
    return docs, info


# ---------------------------------------------------------------------------------- word geometry
WORD_RE = re.compile(r"\S+")


def word_spans(text: str):
    sp = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    if not sp:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = np.asarray(sp, dtype=np.int64)
    return a[:, 0], a[:, 1]


def word_piece_counts(ws, we, st, en, notext_tok):
    """Number of tokens overlapping each word (see module docstring)."""
    nw = len(ws)
    out = np.zeros(nw, np.int64)
    if nw == 0 or len(st) == 0:
        return out
    diff = np.zeros(nw + 1, np.int64)
    valid = ~notext_tok
    ne = valid & (en > st)
    s, e = st[ne], en[ne]
    first = np.searchsorted(we, s, side="right")
    last = np.searchsorted(ws, e, side="left") - 1
    ok = first <= last
    np.add.at(diff, first[ok], 1)
    np.add.at(diff, last[ok] + 1, -1)
    em = valid & (en == st)
    if em.any():
        p = st[em]
        w = np.searchsorted(ws, p, side="right") - 1
        okm = w >= 0
        okm[okm] = p[okm] < we[w[okm]]
        np.add.at(diff, w[okm], 1)
        np.add.at(diff, w[okm] + 1, -1)
    return np.cumsum(diff)[:nw]


def char_maps(L, st, en, keep):
    """For the kept tokens: starts, ends, first/last token covering each char (-1: none), coverage count."""
    idx = np.nonzero(keep & (en > st))[0]
    s, e = st[idx], en[idx]
    n = len(s)
    cov = np.zeros(L + 1, np.int64)
    np.add.at(cov, np.clip(s, 0, L), 1)
    np.add.at(cov, np.clip(e, 0, L), -1)
    cov = np.cumsum(cov)[:L]
    ft = np.full(L, -1, np.int64)
    lt = np.full(L, -1, np.int64)
    if n:
        if np.all(np.diff(s) >= 0) and np.all(np.diff(e) >= 0):
            c = np.arange(L)
            f = np.searchsorted(e, c, side="right")
            fv = f < n
            fv[fv] = s[f[fv]] <= c[fv]
            ft[fv] = f[fv]
            l = np.searchsorted(s, c, side="right") - 1
            lv = l >= 0
            lv[lv] = e[l[lv]] > c[lv]
            lt[lv] = l[lv]
        else:
            for k in range(n - 1, -1, -1):
                ft[s[k]:e[k]] = k
            for k in range(n):
                lt[s[k]:e[k]] = k
    return s, e, ft, lt, cov


def nonws_prefix(text):
    a = np.fromiter((not c.isspace() for c in text), dtype=np.int64, count=len(text))
    out = np.zeros(len(text) + 1, np.int64)
    np.cumsum(a, out=out[1:])
    return out


def seg_compare(pert: P.Perturbed, ws, we, ow, omaps, onw, pmaps, pnw):
    """Per-document robustness counts for one perturbation (see module docstring)."""
    nw = len(ws)
    os_, oe, oft, olt, ocov = omaps
    ps, pe, pft, plt, pcov = pmaps
    oi = pert.orig_idx
    pk = np.nonzero(oi >= 0)[0]
    oc = oi[pk]
    w = ow[oc]
    m = w >= 0
    oc, pk, w = oc[m], pk[m], w[m]
    compared = np.zeros(nw, bool)
    changed = np.zeros(nw, bool)
    if len(w):
        compared[w] = True
        d = (ocov[oc] > 1) != (pcov[pk] > 1)
        changed[w[d]] = True
        same = w[1:] == w[:-1]
        bo = olt[oc[:-1]] != oft[oc[1:]]
        bp = plt[pk[:-1]] != pft[pk[1:]]
        dd = same & (bo != bp)
        changed[w[1:][dd]] = True
        words_c, kf = np.unique(w, return_index=True)
        kl = len(w) - 1 - np.unique(w[::-1], return_index=True)[1]
        # original edges: the word's own first/last character
        a0, b0 = ws[words_c], we[words_c]
        t = oft[a0]
        lo = (t >= 0)
        lo[lo] = (os_[t[lo]] < a0[lo]) & (onw[a0[lo]] - onw[os_[t[lo]]] > 0)
        t = olt[b0 - 1]
        ro = (t >= 0)
        ro[ro] = (oe[t[ro]] > b0[ro]) & (onw[np.minimum(oe[t[ro]], len(onw) - 1)] - onw[b0[ro]] > 0)
        # perturbed edges: first/last shared character of the word's image
        a1, b1 = pk[kf], pk[kl] + 1
        t = pft[a1]
        lp = (t >= 0)
        lp[lp] = (ps[t[lp]] < a1[lp]) & (pnw[a1[lp]] - pnw[ps[t[lp]]] > 0)
        t = plt[b1 - 1]
        rp = (t >= 0)
        rp[rp] = (pe[t[rp]] > b1[rp]) & (pnw[np.minimum(pe[t[rp]], len(pnw) - 1)] - pnw[b1[rp]] > 0)
        changed[words_c[(lo != lp) | (ro != rp)]] = True
    E = np.zeros(len(pert.edited) + 1, np.int64)
    np.cumsum(pert.edited.astype(np.int64), out=E[1:])
    aff = (E[we] - E[ws]) > 0 if nw else np.zeros(0, bool)
    return {"cmp": int(compared.sum()), "chg": int((changed & compared).sum()), "aff": int((aff & compared).sum()),
            "achg": int((aff & changed & compared).sum()), "van": int((~compared).sum())}


# ------------------------------------------------------------------------------------ per document
def _prepare_docs(docs):
    """Tokenizer-independent per-document data, computed once per dataset and reused across tokenizers."""
    prep = []
    for d in docs:
        t = d["text"]
        ws, we = word_spans(t)
        L = len(t)
        c = np.arange(L)
        ow = np.searchsorted(ws, c, side="right") - 1 if len(ws) else np.full(L, -1)
        ok = ow >= 0
        ok[ok] = c[ok] < we[ow[ok]]
        ow = np.where(ok, ow, -1)
        perts = {}
        for p in PERT_NAMES:
            pp = P.PERTURBATIONS[p](t)
            perts[p] = (pp, nonws_prefix(pp.text) if pp.n_edits else None)
        lines = [ln for ln in t.split("\n") if ln]
        prep.append({"ws": ws, "we": we, "ow": ow, "nw": nonws_prefix(t), "perts": perts, "lines": lines,
                     "bytes": len(t.encode("utf-8")), "lines_bytes": sum(len(ln.encode("utf-8")) for ln in lines)})
    return prep


def _ws_norm(s):
    return " ".join(s.split())


def doc_eval(adapter, doc, prep, notext):
    t = doc["text"]
    ids, st, en = adapter.encode_offsets(t)
    ids_a = np.asarray(ids, dtype=np.int64)
    st = np.asarray(st, dtype=np.int64)
    en = np.asarray(en, dtype=np.int64)
    nt = notext[ids_a] if len(ids_a) else np.zeros(0, bool)
    pieces = word_piece_counts(prep["ws"], prep["we"], st, en, nt)
    dec = adapter.decode(ids)
    rt = dec == t
    row = {"uid": doc["uid"], "source": doc["source"], "variety": doc["variety"], "cluster": doc.get("cluster"),
           "group": doc.get("group"), "bytes": prep["bytes"], "chars": len(t), "words": len(prep["ws"]),
           "tokens": len(ids), "pieces": int(pieces.sum()), "cont": int((pieces >= 2).sum()),
           "single": int((pieces == 1).sum()), "nocover": int((pieces == 0).sum()), "rt": int(rt),
           "rt_ws": int(rt or _ws_norm(dec) == _ws_norm(t)), "lost": 0, "added": 0,
           "unk": int(sum(1 for i in ids if i in adapter.unk_ids))}
    if not rt:
        a = collections.Counter(c for c in t if not c.isspace())
        b = collections.Counter(c for c in dec if not c.isspace())
        row["lost"] = int(sum((a - b).values()))
        row["added"] = int(sum((b - a).values()))
    lt = adapter.encode_batch(prep["lines"]) if prep["lines"] else []
    row["lines_tokens"] = int(sum(len(x) for x in lt))
    row["lines_bytes"] = prep["lines_bytes"]
    t0 = time.perf_counter()
    adapter.encode(t)
    row["enc_s"] = time.perf_counter() - t0
    keep = ~nt
    omaps = char_maps(len(t), st, en, keep)
    for p in PERT_NAMES:
        pp, pnw = prep["perts"][p]
        if pp.n_edits == 0:
            r = {"tok": len(ids), "edits": 0, "cmp": int(len(prep["ws"])), "chg": 0, "aff": 0, "achg": 0, "van": 0}
        else:
            pids, pst, pen = adapter.encode_offsets(pp.text)
            pids_a = np.asarray(pids, dtype=np.int64)
            pst = np.asarray(pst, dtype=np.int64)
            pen = np.asarray(pen, dtype=np.int64)
            pkeep = ~notext[pids_a] if len(pids_a) else np.zeros(0, bool)
            pmaps = char_maps(len(pp.text), pst, pen, pkeep)
            r = seg_compare(pp, prep["ws"], prep["we"], prep["ow"], omaps, prep["nw"], pmaps, pnw)
            r["tok"] = len(pids)
            r["edits"] = pp.n_edits
        for k, v in r.items():
            row["%s_%s" % (p, k)] = v
    return row, ids_a


# ------------------------------------------------------------------------------------- aggregation
def renyi_eff(counts: np.ndarray, V: int, alpha: float) -> Optional[float]:
    tot = counts.sum()
    if tot == 0 or V <= 1:
        return None
    p = counts[counts > 0].astype(np.float64) / tot
    h = math.log2(float(np.sum(p ** alpha))) / (1 - alpha)
    return h / math.log2(V)


def aggregate(rows: List[dict], counts: Optional[np.ndarray], V: int) -> dict:
    S = lambda k: sum(r[k] for r in rows)
    n_tok, n_words = S("tokens"), S("words")
    out = {"docs": len(rows), "bytes": S("bytes"), "chars": S("chars"), "words": n_words, "tokens": n_tok}
    out["bytes_per_token"] = out["bytes"] / n_tok if n_tok else None
    out["chars_per_token"] = out["chars"] / n_tok if n_tok else None
    out["fertility"] = S("pieces") / n_words if n_words else None
    out["tokens_per_word"] = n_tok / n_words if n_words else None
    out["continued_word_rate"] = S("cont") / n_words if n_words else None
    out["strr"] = S("single") / n_words if n_words else None
    out["words_without_token"] = S("nocover")
    out["g1_pass_docs"] = S("rt")
    out["g1_fail_docs"] = len(rows) - S("rt")
    out["g1_fail_docs_modulo_whitespace"] = len(rows) - S("rt_ws")
    out["nonws_chars_lost"] = S("lost")
    out["nonws_chars_added"] = S("added")
    out["unk_tokens"] = S("unk")
    out["unk_rate"] = S("unk") / n_tok if n_tok else None
    lt = S("lines_tokens")
    out["bytes_per_token_lines"] = S("lines_bytes") / lt if lt else None
    out["encode_seconds"] = S("enc_s")
    if counts is not None:
        out["renyi_eff_a2.5"] = renyi_eff(counts, V, 2.5)
        out["renyi_eff_a2.0"] = renyi_eff(counts, V, 2.0)
        out["vocab_used"] = int(np.count_nonzero(counts))
        out["vocab_utilisation"] = out["vocab_used"] / V if V else None
    rob = {}
    for p in PERT_NAMES:
        tok = S(p + "_tok")
        aff = S(p + "_aff")
        cmp_ = S(p + "_cmp")
        rob[p] = {"edits": S(p + "_edits"), "docs_affected": sum(1 for r in rows if r[p + "_edits"]),
                  "tokens": tok, "rel_token_change": (tok - n_tok) / n_tok if n_tok else None,
                  "words_compared": cmp_, "words_affected": aff, "words_vanished": S(p + "_van"),
                  "extra_tokens_per_affected_word": (tok - n_tok) / aff if aff else None,
                  "seg_change_rate": S(p + "_chg") / cmp_ if cmp_ else None,
                  "seg_change_rate_affected": S(p + "_achg") / aff if aff else None,
                  "seg_changed_unaffected_words": S(p + "_chg") - S(p + "_achg")}
    out["robustness"] = rob
    return out


# ---------------------------------------------------------------------------------------- morphology
def morph_encoder(adapter):
    from adapters import B2U, char_byte_table
    ctx = FULL_STOP + " "
    cb = len(ctx.encode("utf-8"))
    notext = adapter.notext

    def enc(word):
        text = ctx + word
        wb = text.encode("utf-8")
        ids, st, en = adapter.encode_offsets(text)
        spans = adapter.byte_spans(text, ids)
        if spans is None:
            c2b = char_byte_table(text)
            spans = [(int(c2b[s]), int(c2b[e])) for i, s, e in zip(ids, st, en)
                     if e > s and not (i < len(notext) and notext[i])]
        pieces = []
        for s, e in spans:
            a, b = max(int(s), cb), min(int(e), len(wb))
            if b > a:
                pieces.append(wb[a:b])
        return ["".join(B2U[x] for x in p) for p in pieces]
    enc.byte_level = True
    return enc


def morphology(adapter) -> dict:
    sys.path.insert(0, MORPH_DIR)
    import morph_eval as ME
    enc = morph_encoder(adapter)
    out = {"note": "SILVER sets, report only (PLAN 4.2); each word encoded in context after '%s ' and cut to its "
                   "own bytes; morph_eval.evaluate(..., byte_level=True)" % FULL_STOP}
    for key, fn in (("silver_high", "morph_silver_high.tsv"), ("silver_low", "morph_silver_low.tsv")):
        gold = ME.load_gold(os.path.join(MORPH_DIR, fn))
        res = ME.evaluate(enc, gold, byte_level=True, weight="type")
        out[key] = {k: res[k] for k in ("n_gold", "n_unaligned", "boundary_precision", "boundary_recall", "boundary_f1",
                                        "boundary_f1_macro", "morphscore", "morphscore_all", "stem_intact",
                                        "stem_boundary_respected", "exact_match", "single_token_rate",
                                        "tokens_per_word")}
        out[key]["unaligned_examples"] = res["unaligned_examples"][:5]
    return out


# ---------------------------------------------------------------------------------------------- gates
def learned_ids(adapter) -> List[int]:
    """R1's 'learned tokens': not special/added, not a base byte/<0xNN> piece, not UNK, with known non-empty
    bytes; for character-level BPE also not a single character (the alphabet)."""
    spec = adapter.special_ids | frozenset(getattr(adapter, "added", {}).values())
    out = []
    for i in range(adapter.id_upper):
        if i in spec or i in adapter.base_ids or i in adapter.unk_ids:
            continue
        b = adapter.token_bytes(i)
        if not b:
            continue
        if adapter.model_type == "BPE" and not adapter.byte_level:
            try:
                if len(b.decode("utf-8")) == 1:
                    continue
            except UnicodeDecodeError:
                pass
        out.append(i)
    return out


def token_text(adapter, i) -> Optional[str]:
    b = adapter.token_bytes(i)
    if b is None:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def gate_g2(adapter) -> dict:
    """Self-tokenization: encode_isolated(text(id)) == [id] for every learned id whose bytes are valid UTF-8.
    Exempt: base bytes / <0xNN>, partial-UTF-8 tokens (R2), specials, UNK. Tokens with whitespace inside
    (superwords) need a context test (PLAN 4.3: shortest train chunk) and are listed, not tested here."""
    fails, superword, partial, tested = [], [], 0, 0
    for i in learned_ids(adapter):
        txt = token_text(adapter, i)
        if txt is None:
            partial += 1
            continue
        if any(c.isspace() for c in txt[1:]) and not txt.isspace():
            superword.append(i)
            continue
        tested += 1
        ids = adapter.encode_isolated(txt)
        if list(ids) != [i]:
            fails.append({"id": i, "text": txt, "encodes_to": list(ids)[:12]})
    return {"pass": len(fails) == 0, "tested": tested, "failures": len(fails), "failure_list": fails[:50],
            "exempt_partial_utf8": partial, "superword_tokens_untested": len(superword),
            "superword_examples": [token_text(adapter, i) for i in superword[:10]]}


def g3_corpus_scan(datasets=("train_D1", "train_D2", "train_D3", "dev_strict", "dev_permissive")) -> dict:
    """Occurrences of the special-token marker strings (and of '<|' at all) in the materialised train/dev text."""
    cache = os.path.join(HERE, "g3_corpus_scan.json")
    if os.path.exists(cache):
        c = json.load(open(cache, encoding="utf-8"))
        if c.get("datasets_sha256") == {d: sha256_file(resolve_data(d)) for d in datasets}:
            return c
    occ = collections.Counter()
    angle = 0
    for d in datasets:
        for line in open(resolve_data(d), encoding="utf-8"):
            t = json.loads(line)["text"]
            if "<|" in t:
                angle += t.count("<|")
                for s in SPECIAL_TOKENS:
                    occ[s] += t.count(s)
    out = {"datasets": list(datasets), "datasets_sha256": {d: sha256_file(resolve_data(d)) for d in datasets},
           "marker_occurrences": sum(occ.values()), "per_marker_nonzero": {k: v for k, v in occ.items() if v},
           "substring_<|_occurrences": angle, "test_split_scanned": False}
    json.dump(out, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def gate_g3(adapter) -> dict:
    corpus = g3_corpus_scan()
    present = {s: adapter.special_tokens[s] for s in SPECIAL_TOKENS if s in adapter.special_tokens}
    missing = [s for s in SPECIAL_TOKENS if s not in adapter.special_tokens]
    bad = []
    notext = adapter.notext
    for s, i in present.items():
        # atomic = the special's id occurs once; any other token of the isolated encoding carries no text
        # (a SentencePiece dummy-prefix '▁' before the special is allowed)
        alone = list(adapter.encode(s))
        ctx = "اب " + s + " پت"
        ids = list(adapter.encode(ctx))
        ok = (alone.count(i) == 1 and all(x == i or (x < len(notext) and notext[x]) for x in alone)
              and ids.count(i) == 1 and adapter.decode(ids) == ctx)
        if not ok:
            bad.append({"special": s, "id": i, "alone": alone[:8], "in_context_count": ids.count(i)})
    corpus_ok = corpus["marker_occurrences"] == 0
    return {"pass": corpus_ok and not bad and not missing, "corpus_marker_occurrences": corpus["marker_occurrences"],
            "corpus_scanned": corpus["datasets"], "block_size_present": len(present), "block_missing": missing[:10],
            "block_missing_n": len(missing), "not_atomic": bad[:20], "not_atomic_n": len(bad)}


def gate_g4(path_a: Optional[str], path_b: Optional[str]) -> dict:
    if not path_a or not path_b:
        return {"pass": None, "status": "not run: pass --g4-retrain PATH (a second training with identical inputs)"}
    ha, hb = sha256_file(path_a), sha256_file(path_b)
    same = ha == hb
    json_same = None
    if not same and path_a.endswith(".json") and path_b.endswith(".json"):
        json_same = json.load(open(path_a, encoding="utf-8")) == json.load(open(path_b, encoding="utf-8"))
    return {"pass": bool(same or json_same), "sha256_a": ha, "sha256_b": hb, "bytes_identical": same,
            "json_identical": json_same, "file_b": path_b}


def gate_g5(adapter) -> dict:
    if adapter.byte_level:
        return {"pass": None, "status": "not applicable: byte-level vocabulary (partial-UTF-8 tokens are R2)"}
    bad = []
    for i in range(adapter.id_upper):
        if i in adapter.base_ids or i in adapter.special_ids:
            continue
        b = adapter.token_bytes(i)
        if b is not None and b and not _valid(b):
            bad.append({"id": i, "hex": b.hex(" ")})
    return {"pass": len(bad) == 0, "sub_character_tokens": len(bad), "examples": bad[:20]}


def _valid(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def train_counts(adapter, train: str, batch=500) -> (np.ndarray, int, dict):
    with open(resolve_data(train), encoding="utf-8") as f:
        docs = [json.loads(l)["text"] for l in f]
    split = _manifest_split()
    cnt = np.zeros(adapter.id_upper, np.int64)
    n = 0
    t0 = time.time()
    for i in range(0, len(docs), batch):
        for ids in adapter.encode_batch(docs[i:i + batch]):
            a = np.asarray(ids, dtype=np.int64)
            n += a.size
            if a.size:
                cnt += np.bincount(a, minlength=adapter.id_upper)[:adapter.id_upper]
    return cnt, n, {"train": train, "path": resolve_data(train), "sha256": sha256_file(resolve_data(train)),
                    "docs": len(docs), "tokens": n, "seconds": round(time.time() - t0, 1)}


def _profile(freq: np.ndarray, leaf_mask: Optional[np.ndarray] = None) -> dict:
    n = len(freq)
    if n == 0:
        return {"learned_tokens": 0}
    out = {"learned_tokens": n, "train_freq_eq0": int((freq == 0).sum()), "train_freq_lt20": int((freq < 20).sum()),
           "train_freq_lt100": int((freq < 100).sum()), "pct_lt20": 100 * float((freq < 20).mean()),
           "pct_lt100": 100 * float((freq < 100).mean()), "median_train_freq": float(np.median(freq))}
    if leaf_mask is not None:
        out["learned_leaves"] = int(leaf_mask.sum())
        out["lt20_leaves"] = int(((freq < 20) & leaf_mask).sum())
        out["lt20_intermediate"] = int(((freq < 20) & ~leaf_mask).sum())
        out["eq0_intermediate"] = int(((freq == 0) & ~leaf_mask).sum())
    return out


def prop_r1(adapter, counts: np.ndarray, info: dict) -> dict:
    lid = np.asarray(learned_ids(adapter), dtype=np.int64)
    freq = counts[lid]
    leaf = None
    if adapter.merges:
        comps = set()
        for a, b, _ in adapter.merges:
            comps.add(a); comps.add(b)
        leaf = np.asarray([i not in comps for i in lid.tolist()], dtype=bool)
    multi = []
    for i in lid.tolist():
        txt = token_text(adapter, i)
        core = txt.strip() if txt is not None else None
        multi.append(core is None or len(core) > 1)
    multi = np.asarray(multi, dtype=bool)
    rare = [{"id": int(i), "text": token_text(adapter, int(i)), "train_freq": int(counts[i])}
            for i in lid[np.argsort(freq, kind="stable")][:25].tolist()]
    return {"train": info, "all_learned": _profile(freq, leaf),
            "multichar_learned": _profile(freq[multi], leaf[multi] if leaf is not None else None),
            "rarest_learned": rare,
            "definition": "learned = not special/added/base-byte/<0xNN>/UNK; char-level BPE also excludes single "
                          "characters; multichar = token text minus surrounding spaces longer than 1 char or "
                          "not valid UTF-8; leaves = learned tokens that are no merge component (BPE)"}


def prop_r2(adapter, train_cnt: Optional[np.ndarray], dev_cnt: np.ndarray) -> dict:
    items = []
    for i in range(adapter.id_upper):
        if i in adapter.base_ids or i in adapter.special_ids or i in adapter.unk_ids:
            continue
        b = adapter.token_bytes(i)
        if b and not _valid(b):
            items.append({"id": i, "hex": b.hex(" "), "dev_freq": int(dev_cnt[i]) if i < len(dev_cnt) else 0,
                          "train_freq": int(train_cnt[i]) if train_cnt is not None else None})
    return {"partial_utf8_tokens": len(items),
            "with_train_freq_0": (sum(1 for x in items if x["train_freq"] == 0) if train_cnt is not None else None),
            "dev_occurrences": sum(x["dev_freq"] for x in items), "list": items[:60]}


# ------------------------------------------------------------------------------------------------ NSL
def nsl_from_docs(cand_docs: str, ref_docs: str) -> dict:
    ref = {}
    for l in open(ref_docs, encoding="utf-8"):
        r = json.loads(l)
        ref[r["uid"]] = r["tokens"]
    tot = collections.defaultdict(lambda: [0, 0])
    for l in open(cand_docs, encoding="utf-8"):
        r = json.loads(l)
        if r["uid"] not in ref:
            continue
        for key in ("overall", "source:" + r["source"], "variety:" + r["variety"]):
            tot[key][0] += r["tokens"]; tot[key][1] += ref[r["uid"]]
    return {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(tot.items())} | \
           {"n_docs_matched": sum(1 for l in open(cand_docs, encoding="utf-8") if json.loads(l)["uid"] in ref),
            "reference_docs": ref_docs}


# ------------------------------------------------------------------------------------------------ run
_PREP_CACHE = {}


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s)


def run(adapter, data: str = "dev_strict", gates: str = "g1", train: str = "train_D1", g4_retrain: Optional[str] = None,
        nsl_ref: Optional[str] = None, out_dir: Optional[str] = None, allow_test: bool = False, force: bool = False,
        do_morph: bool = True, extra: Optional[dict] = None) -> dict:
    t_start = time.time()
    docs, dinfo = load_docs(data, allow_test)
    out_dir = out_dir or os.path.join(RESULTS, dinfo["name"], safe_name(adapter.name))
    os.makedirs(out_dir, exist_ok=True)
    ident = adapter.identity()
    if hasattr(adapter, "baseline"):
        ident["baseline"] = adapter.baseline
    summ_path = os.path.join(out_dir, "summary.json")
    hashes = code_hashes()
    if not force and os.path.exists(summ_path):
        old = json.load(open(summ_path, encoding="utf-8"))
        if (old.get("tokenizer", {}).get("sha256") == ident.get("sha256") and old.get("harness", {}).get("code") == hashes
                and old.get("dataset", {}).get("sha256") == dinfo["sha256"] and old.get("options", {}).get("gates") == gates):
            log(adapter.name, "up to date, skipped")
            return old
    key = dinfo["sha256"]
    if key not in _PREP_CACHE:
        t0 = time.time()
        if len(_PREP_CACHE) >= 2:
            _PREP_CACHE.pop(next(iter(_PREP_CACHE)))
        _PREP_CACHE[key] = _prepare_docs(docs)
        log("prepared %d docs (perturbations, word spans) in %.1fs" % (len(docs), time.time() - t0))
    prep = _PREP_CACHE[key]
    notext = adapter.notext
    V = adapter.vocab_size
    counts = np.zeros(adapter.id_upper, np.int64)
    by_src = collections.defaultdict(lambda: np.zeros(adapter.id_upper, np.int64))
    by_var = collections.defaultdict(lambda: np.zeros(adapter.id_upper, np.int64))
    rows = []
    t0 = time.time()
    for d, pr in zip(docs, prep):
        row, ids = doc_eval(adapter, d, pr, notext)
        rows.append(row)
        if ids.size:
            bc = np.bincount(ids, minlength=adapter.id_upper)[:adapter.id_upper]
            counts += bc; by_src[d["source"]] += bc; by_var[d["variety"]] += bc
    t_docs = time.time() - t0
    with open(os.path.join(out_dir, "docs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(clean({k: r[k] for k in DOC_FIELDS}), ensure_ascii=False) + "\n")
    metrics = {"overall": aggregate(rows, counts, V),
               "by_source": {s: aggregate([r for r in rows if r["source"] == s], by_src[s], V) for s in sorted(by_src)},
               "by_variety": {v: aggregate([r for r in rows if r["variety"] == v], by_var[v], V) for v in sorted(by_var)}}
    ov = metrics["overall"]
    summary = {"harness": {"version": HARNESS_VERSION, "code": hashes, "run_utc": datetime.datetime.now(
                   datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "python": sys.version.split()[0],
                   "threads": os.environ.get("RAYON_NUM_THREADS")},
               "frozen": _frozen_short(), "dataset": dinfo, "tokenizer": ident,
               "options": {"gates": gates, "train": train if gates == "all" else None, "g4_retrain": g4_retrain},
               "metrics": metrics,
               "speed": {"docs_per_s": len(rows) / ov["encode_seconds"] if ov["encode_seconds"] else None,
                         "MB_per_s": ov["bytes"] / 1e6 / ov["encode_seconds"] if ov["encode_seconds"] else None,
                         "note": "plain encode() per document, sequential; sanity only (shared CPU)"}}
    g1 = {"pass": ov["g1_fail_docs"] == 0, "docs": ov["docs"], "fail_docs": ov["g1_fail_docs"],
          "fail_docs_modulo_whitespace": ov["g1_fail_docs_modulo_whitespace"], "unk_tokens": ov["unk_tokens"],
          "nonws_chars_lost": ov["nonws_chars_lost"], "nonws_chars_added": ov["nonws_chars_added"],
          "dataset": dinfo["name"], "fail_uids": [r["uid"] for r in rows if not r["rt"]][:20]}
    summary["gates"] = {"G1": g1}
    summary["properties"] = {}
    train_cnt = None
    if gates == "all":
        t1 = time.time()
        summary["gates"]["G2"] = gate_g2(adapter)
        summary["gates"]["G3"] = gate_g3(adapter)
        summary["gates"]["G4"] = gate_g4(getattr(adapter, "path", None), g4_retrain)
        summary["gates"]["G5"] = gate_g5(adapter)
        train_cnt, _, tinfo = train_counts(adapter, train)
        summary["properties"]["R1"] = prop_r1(adapter, train_cnt, tinfo)
        summary["gates_seconds"] = round(time.time() - t1, 1)
    else:
        for g in ("G2", "G3", "G4", "G5"):
            summary["gates"][g] = {"pass": None, "status": "not run (gates=%s)" % gates}
        summary["properties"]["R1"] = {"status": "not run (needs --gates all and a train view)"}
    summary["properties"]["R2"] = prop_r2(adapter, train_cnt, counts)
    if do_morph:
        try:
            summary["morphology"] = morphology(adapter)
        except Exception as e:  # noqa: BLE001
            summary["morphology"] = {"error": repr(e)}
    summary["nsl"] = (nsl_from_docs(os.path.join(out_dir, "docs.jsonl"), os.path.join(nsl_ref, "docs.jsonl"))
                      if nsl_ref else {"status": "pending: reference A1-P1-16k does not exist yet (harness.py nsl)"})
    if hasattr(adapter, "offset_id_mismatch"):
        summary["tokenizer"]["offset_id_mismatch_docs"] = adapter.offset_id_mismatch
    if extra:
        summary["extra"] = extra
    summary["timing"] = {"docs_pass_s": round(t_docs, 1), "total_s": round(time.time() - t_start, 1)}
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(clean(summary), f, ensure_ascii=False, indent=1)
    log("%s: %d docs, %.3f bytes/token, fertility %.3f, G1 %d/%d, %.0fs" % (
        adapter.name, len(rows), ov["bytes_per_token"] or 0, ov["fertility"] or 0, ov["g1_pass_docs"], len(rows),
        time.time() - t_start))
    return summary


def _frozen_short():
    with open(FROZEN_PATH, encoding="utf-8") as f:
        fr = json.load(f)
    return {"split_manifest_sha256": fr["split_manifest"]["sha256"], "normalize_version": fr["normalize"]["version"],
            "normalize_sha256": fr["normalize"]["sha256"]}


# ----------------------------------------------------------------------------- pre-tokenizer differential
PRETOKENIZERS = {
    # Oniguruma (HF tokenizers) syntax, as in PLAN 2.1; the Python `regex` twin is derived by onig_to_py()
    "P1": r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+",
    "P1r3": r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+",
    "P0": r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
}


def onig_to_py(p: str) -> str:
    """Oniguruma \\x{HHHH} escapes -> Python `regex` \\uHHHH escapes (SOTA_TOKENIZATION notes)."""
    return re.sub(r"\\x\{([0-9A-Fa-f]{4})\}", lambda m: "\\u" + m.group(1), p)


def py_pretokens(rx, text):
    """Python twin of HF Split(behavior='isolated'): matches and the gaps between them, in order."""
    out, pos = [], 0
    for m in rx.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        if m.end() > m.start():
            out.append(m.group())
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


def pretokenizer_differential(names=("P1", "P1r3", "P0"), datasets=("dev_strict", "dev_permissive")) -> dict:
    """PLAN 1.3: the Python-regex and Oniguruma versions of each pre-tokenizer regex must give identical
    pretokens on dev."""
    import regex as pyre
    from tokenizers import pre_tokenizers, Regex
    out = {"regex_version": pyre.__version__, "tokenizers_version": __import__("tokenizers").__version__, "results": {}}
    for name in names:
        onig = PRETOKENIZERS[name]
        py = onig_to_py(onig)
        hf = pre_tokenizers.Split(Regex(onig), behavior="isolated")
        rx = pyre.compile(py)
        res = {"oniguruma": onig, "python": py}
        # dev has no ZWNJ at all; the ZWNJ-perturbed dev_strict text exercises the \x{200C} / ‌ escape
        for d in list(datasets) + ["dev_strict+zwnj"]:
            docs, _ = load_docs(d.split("+")[0])
            n_diff, n_tok, ex = 0, 0, []
            for doc in docs:
                t = doc["text"] if "+" not in d else P.perturb_zwnj(doc["text"]).text
                a = [s for s, _ in hf.pre_tokenize_str(t)]
                b = py_pretokens(rx, t)
                n_tok += len(a)
                if a != b:
                    n_diff += 1
                    if len(ex) < 3:
                        k = next(i for i, (x, y) in enumerate(zip(a + [None], b + [None])) if x != y)
                        ex.append({"uid": doc["uid"], "first_diff_index": k, "onig": a[k:k + 3], "python": b[k:k + 3]})
            res[d] = {"docs": len(docs), "docs_differing": n_diff, "pretokens_onig": n_tok, "examples": ex}
        res["identical"] = all(res[d]["docs_differing"] == 0 for d in list(datasets) + ["dev_strict+zwnj"])
        out["results"][name] = res
    json.dump(clean(out), open(os.path.join(HERE, "pretok_differential.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return out


# ---------------------------------------------------------------------------------------------- census
def census(data="dev_strict") -> dict:
    docs, dinfo = load_docs(data)
    out = {"dataset": dinfo, "perturbations": {}}
    for p in PERT_NAMES:
        n_docs = n_edits = n_words = 0
        types = collections.Counter()
        for d in docs:
            t = d["text"]
            pp = P.PERTURBATIONS[p](t)
            P.check_alignment(t, pp)
            if pp.n_edits:
                n_docs += 1; n_edits += pp.n_edits
                ws, we = word_spans(t)
                for a, b in zip(ws.tolist(), we.tolist()):
                    if pp.edited[a:b].any():
                        n_words += 1
                        if p == "zwnj":
                            types[t[a:b]] += 1
        out["perturbations"][p] = {"docs_affected": n_docs, "edits": n_edits, "words_affected": n_words}
        if p == "zwnj":
            out["perturbations"][p]["compound_word_types"] = len(types)
            out["perturbations"][p]["compound_words"] = [{"word": w, "count": c} for w, c in types.most_common()]
    out["g3_corpus_scan"] = g3_corpus_scan()
    path = os.path.join(HERE, "census_%s.json" % dinfo["name"])
    json.dump(clean(out), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


# ------------------------------------------------------------------------------------------------- CLI
def build_adapter(a) -> "object":
    import adapters as A
    if a.tokenizer_json:
        return A.HFAdapter(path=a.tokenizer_json, name=a.name)
    if a.spm:
        return A.SPAdapter(path=a.spm, name=a.name, newline_wrapper=not a.no_newline_wrapper)
    if a.baseline:
        return A.from_baseline(a.baseline)
    if a.custom:
        mod, fn = a.custom.split(":")
        obj = getattr(importlib.import_module(mod), fn)()
        return A.CustomAdapter(obj, name=a.name)
    raise SystemExit("give one of --tokenizer-json --spm --baseline --custom")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hindko tokenizer evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tokenizer-json")
    r.add_argument("--spm")
    r.add_argument("--no-newline-wrapper", action="store_true")
    r.add_argument("--baseline")
    r.add_argument("--custom")
    r.add_argument("--name")
    r.add_argument("--data", default="dev_strict")
    r.add_argument("--gates", default="g1", choices=["g1", "all"])
    r.add_argument("--train-data", default="train_D1")
    r.add_argument("--g4-retrain")
    r.add_argument("--nsl-ref")
    r.add_argument("--out")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--no-morph", action="store_true")
    r.add_argument("--allow-test", action="store_true")
    r.add_argument("--force", action="store_true")
    n = sub.add_parser("nsl")
    n.add_argument("--ref", required=True)
    n.add_argument("targets", nargs="+")
    c = sub.add_parser("census")
    c.add_argument("--data", default="dev_strict")
    sub.add_parser("pretok-diff")
    a = ap.parse_args(argv)
    if hasattr(a, "threads"):
        _setup_threads(a.threads)
    else:
        _setup_threads(1)
    if a.cmd == "run":
        ad = build_adapter(a)
        run(ad, data=a.data, gates=a.gates, train=a.train_data, g4_retrain=a.g4_retrain, nsl_ref=a.nsl_ref,
            out_dir=a.out, allow_test=a.allow_test, force=a.force, do_morph=not a.no_morph)
    elif a.cmd == "nsl":
        for t in a.targets:
            sp = os.path.join(t, "summary.json")
            s = json.load(open(sp, encoding="utf-8"))
            s["nsl"] = nsl_from_docs(os.path.join(t, "docs.jsonl"), os.path.join(a.ref, "docs.jsonl"))
            json.dump(clean(s), open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(t, s["nsl"].get("overall"))
    elif a.cmd == "census":
        out = census(a.data)
        print(json.dumps({k: v for k, v in out["perturbations"].items()}, ensure_ascii=False)[:3000])
    elif a.cmd == "pretok-diff":
        out = pretokenizer_differential()
        for k, v in out["results"].items():
            print(k, "identical" if v["identical"] else "DIFFERENT",
                  {d: v[d]["docs_differing"] for d in ("dev_strict", "dev_permissive", "dev_strict+zwnj")})


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
