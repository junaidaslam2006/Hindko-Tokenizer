# -*- coding: utf-8 -*-
"""Continued-BPE vocabulary extension of an existing tokenizer (PLAN.md 9.2).

Method: "continued BPE training" of Purason, Chizhov, Yamshchikov & Fishel, "Teaching Old Tokenizers New Words"
(Findings of EACL 2026, arXiv:2512.03989), REIMPLEMENTED FROM THE PAPER (their toolkit was not downloaded or run).
Merge learning is resumed from the state the base tokenizer leaves the training text in: every training unit is
encoded with the base tokenizer, and new merges are learned greedily (most frequent adjacent pair first) on the
resulting sequences of base tokens. The new merges are appended after all base merges, so at encoding time the
base merges run to completion first (= the unchanged base encoding) and the new merges then apply in the order
they were learned. Text in which no new merge fires is encoded exactly as by the base.

Implementation: the "code-point trick" (SOTA_TOKENIZATION.md 4, route a). Each base token id that occurs in a
training unit is mapped to one Supplementary Private Use Area code point (U+F0000..), each unit becomes a string
of such code points, and the stock Rust `tokenizers.trainers.BpeTrainer` learns the merges on these strings. The
PUA merges are translated back to base-token strings and appended to the base tokenizer.json.

Training units (where merges may form) follow the base's own segmentation rules:
  bytelevel  (Qwen, Llama): the base pre-tokenizer's pretokens (its normalizer + Split regex + ByteLevel).
  spbpe      (Gemma, SentencePiece BPE through its HF tokenizer.json, which has no effective pre-tokenizer):
             the base encoding of each whole document, cut into word units the way Gemma's own SentencePiece
             trainer_spec constrains pieces: a unit starts at every token beginning with U+2581 (the vocabulary
             has no piece with an inner U+2581 except '>▁</'); newline/whitespace-run, byte-fallback (<0xNN>),
             added/special and ASCII-digit tokens (split_digits=true) are never merged; a unit is also cut where
             the Unicode script changes (split_by_unicode_script=true; Inherited marks take the preceding script);
             a merge whose piece would exceed max_sentencepiece_length = 16 characters is dropped, together with
             every later merge that needs it.
Scope (all bases), two levels:
  unit level   only units containing at least one Arabic-script character (Script_Extensions=Arabic) are used for
               training, so no merge is learned from Latin, digit-only, code-like or whitespace units;
  merge level  (method 1.1.0) a learned merge is kept only if the new token can occur only in text that contains an
               Arabic-script character: its bytes contain a complete Arabic-script character, or they end in an
               incomplete UTF-8 sequence whose every assigned completion is Arabic-script (e.g. the lead bytes of
               U+08C0..U+08FF). Rejected merges are dropped together with every later merge that needs their
               result, like the Gemma length limit. Reason (measured with 1.0.0, archive_v1.0/): inside Arabic
               units, byte-level bases also learned merges such as '’’' (Qwen3 rank 460), ')\n' (Qwen3.5 rank
               4,639) and continuation-byte pairs such as 'a2 bf', which then fired in English, Python code and in
               single CJK/Braille/Mongolian characters.
The effect is verified by evaluate.py (check set) and audit.py (every Unicode scalar value), not assumed.

    python continued_bpe.py build --base qwen-3 [--tag rerun]      learn K_MAX new tokens (resumable per base)
    python continued_bpe.py materialize --base qwen-3 --k 4096     write sweep/<base>/k4096/tokenizer.json
    python continued_bpe.py all --base qwen-3                      build + materialize every k of the grid
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402

C.env_threads(2)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402

PUA_START = 0xF0000          # plane 15 (65,534 code points) then plane 16
MIN_FREQUENCY = 2            # as every BpeTrainer run of this project (pilot_tokenizers.py, gate_audit.py)
MARGIN = 1.25                # PUA merges trained = MARGIN * K_MAX (collisions / dropped merges are skipped)
SP_MAX_PIECE_CHARS = 16      # SentencePiece default max_sentencepiece_length (absent from Gemma's trainer_spec)
SP_MARK = "▁"
METHOD_VERSION = "1.1.0"


# ----------------------------------------------------------------------------------------- byte-level
def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = _b2u()
U2B = {u: b for b, u in B2U.items()}


def bl_bytes(s: str) -> bytes:
    return bytes(U2B[c] for c in s)


def token_text_bytes(kind: str, s: str) -> bytes:
    """UTF-8 bytes a model-vocab token string stands for."""
    if kind == "bytelevel":
        return bl_bytes(s)
    return s.replace(SP_MARK, " ").encode("utf-8")


_UNASSIGNED = None
_T_CACHE = {}


def _forces_arabic(t: bytes) -> bool:
    """t is an incomplete UTF-8 sequence. True iff it starts with a lead byte, the rest are continuation bytes, and
    every assigned code point whose UTF-8 encoding starts with t is Arabic-script (Script_Extensions)."""
    global _UNASSIGNED
    if t in _T_CACHE:
        return _T_CACHE[t]
    import itertools
    import regex
    if _UNASSIGNED is None:
        _UNASSIGNED = regex.compile(r"\p{Cn}")
    b0 = t[0] if t else 0
    n = 2 if 0xC2 <= b0 <= 0xDF else 3 if 0xE0 <= b0 <= 0xEF else 4 if 0xF0 <= b0 <= 0xF4 else 0
    ok = False
    if n and len(t) < n and all(0x80 <= x <= 0xBF for x in t[1:]):
        ok, seen = True, 0
        for tail in itertools.product(range(0x80, 0xC0), repeat=n - len(t)):
            try:
                c = (t + bytes(tail)).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if _UNASSIGNED.match(c):
                continue
            seen += 1
            if not C.has_arabic(c):
                ok = False
                break
        ok = ok and seen > 0
    _T_CACHE[t] = ok
    return ok


def in_scope(kind: str, s: str) -> bool:
    """Merge-level scope (method 1.1.0): the token can only occur in text containing an Arabic-script character."""
    b = token_text_bytes(kind, s)
    i = 0
    while i < len(b) and 0x80 <= b[i] <= 0xBF:      # orphan continuation bytes of a preceding character
        i += 1
    for cut in range(0, 4):
        end = len(b) - cut
        if end < i:
            break
        try:
            mid = b[i:end].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if C.has_arabic(mid):
            return True
        return cut > 0 and _forces_arabic(b[end:])
    return False


def units_bytelevel(base: str, docs, verify_docs: int = 300):
    tk = Tokenizer.from_file(C.base_tokenizer_path(base))
    norm, pre, model = tk.normalizer, tk.pre_tokenizer, tk.model
    pcount = collections.Counter()
    n_pre = 0
    for d in docs:
        t = d["text"]
        if norm is not None:
            t = norm.normalize_str(t)
        pts = pre.pre_tokenize_str(t)
        n_pre += len(pts)
        pcount.update(p for p, _ in pts)
    units = collections.Counter()
    stats = collections.Counter()
    seg = {}
    for p, c in pcount.items():
        ids = tuple(t.id for t in model.tokenize(p))
        seg[p] = ids
        txt = bl_bytes(p).decode("utf-8", errors="replace")
        ar = C.has_arabic(txt)
        stats["pretoken_occ"] += c
        stats["pretoken_types"] += 1
        stats["base_tokens"] += c * len(ids)
        if ar:
            stats["arabic_occ"] += c
            stats["arabic_types"] += 1
            stats["arabic_base_tokens"] += c * len(ids)
            if len(ids) >= 2:
                units[ids] += c
                stats["unit_occ"] += c
                stats["unit_types"] += 1
                stats["unit_base_tokens"] += c * len(ids)
    # verification: the per-pretoken model encoding concatenated == the full pipeline encoding
    rng = random.Random(20260926)
    sample = rng.sample(range(len(docs)), min(verify_docs, len(docs)))
    bad = 0
    for i in sample:
        t = docs[i]["text"]
        full = tk.encode(t, add_special_tokens=False).ids
        tt = norm.normalize_str(t) if norm is not None else t
        cat = [x for p, _ in pre.pre_tokenize_str(tt) for x in seg.get(p) or [y.id for y in model.tokenize(p)]]
        bad += int(full != cat)
    stats["verify_docs"] = len(sample)
    stats["verify_mismatch_docs"] = bad
    return units, dict(stats)


# ---------------------------------------------------------------------------------------- SentencePiece
def _piece_scripts(piece: str):
    """(first definite script, last definite script) of a piece's non-U+2581 characters; Inherited characters take
    the preceding script (SentencePiece IsValidSentencePiece); None where no definite script exists."""
    first = last = None
    for ch in piece:
        if ch == SP_MARK:
            continue
        s = C.char_script(ch)
        if s == "Inherited":
            continue
        if first is None:
            first = s
        last = s
    return first, last


def units_spbpe(base: str, docs, batch: int = 256):
    tk = Tokenizer.from_file(C.base_tokenizer_path(base))
    j = C.load_base_json(base)
    vocab = j["model"]["vocab"]
    inv = {i: s for s, i in vocab.items()}
    added = {a["id"] for a in j.get("added_tokens", [])}
    import re
    byte_rx = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    cls_cache = {}

    def tclass(i):
        c = cls_cache.get(i)
        if c is None:
            p = inv.get(i)
            if p is None or i in added or byte_rx.match(p):
                c = ("iso", None, None)
            elif p == SP_MARK:
                c = ("ws1", None, None)
            elif all(ch == SP_MARK or ch.isspace() for ch in p):
                c = ("iso", None, None)                     # newline runs, space runs, tabs
            elif any("0" <= ch <= "9" for ch in p):
                c = ("iso", None, None)                     # split_digits: ASCII digits are never merged
            else:
                f, l = _piece_scripts(p)
                c = ("norm", f, l)
            cls_cache[i] = c
        return c

    units = collections.Counter()
    stats = collections.Counter()

    def flush(cur):
        if not cur:
            return
        stats["unit_candidates"] += 1
        txt = "".join(inv[i] for i in cur)
        if not C.has_arabic(txt):
            stats["nonarabic_units"] += 1
            stats["nonarabic_unit_tokens"] += len(cur)
            return
        stats["arabic_units"] += 1
        stats["arabic_unit_tokens"] += len(cur)
        if len(cur) >= 2:
            units[tuple(cur)] += 1
            stats["unit_occ"] += 1
            stats["unit_base_tokens"] += len(cur)

    for s in range(0, len(docs), batch):
        encs = tk.encode_batch([d["text"] for d in docs[s:s + batch]], add_special_tokens=False)
        for e in encs:
            ids = e.ids
            stats["base_tokens"] += len(ids)
            cur, cur_script = [], None
            for i in ids:
                kind, f, l = tclass(i)
                if kind == "iso":
                    flush(cur); cur, cur_script = [], None
                    stats["iso_tokens"] += 1
                    continue
                starts_ws = kind == "ws1" or inv[i].startswith(SP_MARK)
                if starts_ws:
                    flush(cur); cur, cur_script = [i], None
                elif cur and cur_script is not None and f is not None and f != cur_script:
                    flush(cur); cur, cur_script = [i], None
                    stats["script_cuts"] += 1
                else:
                    cur.append(i)
                if l is not None:
                    cur_script = l
            flush(cur)
    stats["unit_types"] = len(units)
    return units, dict(stats)


# -------------------------------------------------------------------------------------------- training
def train_pua(units: collections.Counter, n_merges: int):
    alphabet = sorted({i for u in units for i in u})
    if len(alphabet) > 2 * 65534:
        raise SystemExit("alphabet too large for the PUA planes: %d" % len(alphabet))

    def cp(n):
        return PUA_START + n if n < 65534 else 0x100000 + (n - 65534)
    to_pua = {bid: chr(cp(n)) for n, bid in enumerate(alphabet)}
    from_pua = {v: k for k, v in to_pua.items()}
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(vocab_size=len(alphabet) + n_merges, min_frequency=MIN_FREQUENCY, show_progress=False,
                                  special_tokens=[], initial_alphabet=[])
    items = sorted(units.items())            # deterministic order

    def it():
        for u, c in items:
            w = "".join(to_pua[i] for i in u)
            while c > 0:
                r = min(c, 4096)
                yield " ".join([w] * r)
                c -= r
    t0 = time.time()
    tok.train_from_iterator(it(), trainer=trainer)
    dt = time.time() - t0
    mj = json.loads(tok.to_str())["model"]
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in mj["merges"]]
    return merges, from_pua, {"alphabet": len(alphabet), "pua_merges": len(merges), "pua_vocab": len(mj["vocab"]),
                              "train_seconds": round(dt, 1), "min_frequency": MIN_FREQUENCY,
                              "requested_vocab_size": len(alphabet) + n_merges}


# ------------------------------------------------------------------------------------------ translation
def translate(base: str, merges, from_pua, k_max: int):
    kind = C.BASES[base]["kind"]
    j = C.load_base_json(base)
    vocab = j["model"]["vocab"]
    inv = {i: s for s, i in vocab.items()}
    added_contents = {a["content"] for a in j.get("added_tokens", [])}
    base_pairs = set()
    for m in j["model"]["merges"]:
        base_pairs.add(tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)))
    first_new = C.first_new_id(base)             # after model vocab, tokenizer.json AND tokenizer_config added tokens
    tok2id = dict(vocab)
    next_id = first_new
    avail = set()
    strof_cache = {}

    def strof(ps):
        s = strof_cache.get(ps)
        if s is None:
            s = "".join(inv[from_pua[ch]] for ch in ps)
            strof_cache[ps] = s
        return s
    for ch in from_pua:
        avail.add(ch)
    out, seen_pairs = [], set()
    st = collections.Counter()
    dropped_examples = []
    n_new = 0
    for rank, (a, b) in enumerate(merges):
        if a not in avail or b not in avail:
            st["dropped_missing_component"] += 1
            continue
        ls, rs = strof(a), strof(b)
        res = ls + rs
        if kind == "spbpe" and len(res) > SP_MAX_PIECE_CHARS:
            st["dropped_too_long"] += 1
            continue
        if res in added_contents:
            st["dropped_added_token_string"] += 1
            continue
        if not in_scope(kind, res):
            st["dropped_out_of_scope"] += 1
            if len(dropped_examples) < 30:
                dropped_examples.append({"pua_rank": rank, "left": ls, "right": rs,
                                         "hex": token_text_bytes(kind, res).hex(" ")})
            continue
        if (ls, rs) in base_pairs:
            st["dropped_base_pair"] += 1          # cannot happen after the base merges; counted as a check
            continue
        if (ls, rs) in seen_pairs:
            st["duplicate_pair_skipped"] += 1
            avail.add(a + b)
            continue
        seen_pairs.add((ls, rs))
        avail.add(a + b)
        if res in tok2id:
            rid = tok2id[res]
            k = "existing_base" if rid < first_new else "existing_new"
            st["merge_to_" + k] += 1
        else:
            rid = next_id; next_id += 1
            tok2id[res] = rid
            n_new += 1
            k = "new"
        bb = token_text_bytes(kind, res)
        try:
            bb.decode("utf-8"); partial = False
        except UnicodeDecodeError:
            partial = True
        out.append({"pua_rank": rank, "left": ls, "right": rs, "result": res, "id": rid, "kind": k, "n_new": n_new,
                    "partial_utf8": partial, "base_seq_len": len(a) + len(b)})
        if n_new >= k_max:
            break
    st["new_tokens"] = n_new
    st["merges_kept"] = len(out)
    st["pua_merges_consumed"] = (out[-1]["pua_rank"] + 1) if out else 0
    st = dict(st)
    st["out_of_scope_examples"] = dropped_examples
    return out, st, first_new


# ---------------------------------------------------------------------------------------------- build
def work_dir(base, tag=None):
    return os.path.join(C.WORK, base + ("" if not tag else "__" + tag))


def build(base: str, tag: str = None, force: bool = False):
    wd = work_dir(base, tag)
    ext_path = os.path.join(wd, "extension.json")
    if os.path.exists(ext_path) and not force:
        C.log(base, "extension exists:", ext_path)
        return C.load_json(ext_path)
    fr = C.verify_frozen()
    t0 = time.time()
    docs = C.load_view("train_D1")
    C.log(base, "train_D1 loaded: %d docs (%.0fs)" % (len(docs), time.time() - t0))
    kind = C.BASES[base]["kind"]
    t1 = time.time()
    units, ustats = (units_bytelevel if kind == "bytelevel" else units_spbpe)(base, docs)
    ustats["seconds"] = round(time.time() - t1, 1)
    C.log(base, "units:", json.dumps(ustats))
    n_train = int(C.K_MAX * MARGIN)
    for attempt in range(4):
        merges, from_pua, tstats = train_pua(units, n_train)
        C.log(base, "PUA training:", json.dumps(tstats))
        out, xstats, first_new = translate(base, merges, from_pua, C.K_MAX)
        C.log(base, "translation:", json.dumps(xstats))
        if xstats["new_tokens"] >= C.K_MAX:
            break
        n_train = int(n_train * 1.3)
    else:
        raise SystemExit("could not reach %d new tokens" % C.K_MAX)
    res = {"what": "continued-BPE extension (reimplemented from Purason et al. 2026) of %s on Hindko train_D1" % base,
           "method_version": METHOD_VERSION, "base": base, "base_kind": kind,
           "base_tokenizer_json": C.base_tokenizer_path(base),
           "base_tokenizer_sha256": C.sha256_file(C.base_tokenizer_path(base)),
           "code_sha256": {"continued_bpe.py": C.sha256_file(os.path.abspath(__file__)),
                           "tb_common.py": C.sha256_file(os.path.join(HERE, "tb_common.py"))},
           "frozen": fr, "train_view": "train_D1", "k_max": C.K_MAX, "min_frequency": MIN_FREQUENCY,
           "first_new_id": first_new, "unit_stats": ustats, "train_stats": tstats, "translation_stats": xstats,
           "tokenizers_version": __import__("tokenizers").__version__,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "merges": out}
    C.dump_json(res, ext_path, indent=None)
    # the unit counts (base-id tuples) are kept for the support replay and the equivalence test
    with open(os.path.join(wd, "units.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for u, c in sorted(units.items()):
            f.write("%d\t%s\n" % (c, " ".join(map(str, u))))
    C.log(base, "build done in %.0fs ->" % (time.time() - t0), ext_path)
    return res


# ------------------------------------------------------------------------------------------ materialize
def select(ext: dict, k: int):
    """The merges of the k-token extension: every merge up to and including the one creating the k-th new token
    (continued BPE is greedy, so the k-extension is a prefix of the K_MAX one)."""
    out = []
    for m in ext["merges"]:
        if m["kind"] == "new" and m["n_new"] > k:
            break
        out.append(m)
        if m["kind"] == "new" and m["n_new"] == k:
            break
    assert sum(1 for m in out if m["kind"] == "new") == k, k
    return out


def extended_json(base: str, ext: dict, k: int) -> dict:
    j = C.load_base_json(base)
    j = copy.deepcopy(j)
    sel = select(ext, k)
    v = j["model"]["vocab"]
    # HF tokenizers re-derives the ids of added tokens that are absent from the model vocabulary as
    # len(model vocab), len + 1, ... when it loads a tokenizer.json. Appending new model tokens after the added
    # block would therefore shift every special token onto a new token's id (measured: Qwen3 '<tool_call>'
    # became id 152681). The added tokens are therefore first written into the model vocabulary at their own ids,
    # which makes the model vocabulary contiguous (the layout Gemma's own file already has); they are still
    # matched by the added-token pass before BPE, and no merge produces them.
    # Added tokens that only the base's tokenizer_config.json defines (Qwen3.5: 7 audio/TTS tokens) are added to the
    # tokenizer.json first, at their config ids, so that tokenizers and transformers agree on every id.
    j["added_tokens"] = sorted(j.get("added_tokens", []) + C.config_only_added_tokens(base), key=lambda a: a["id"])
    for a in j.get("added_tokens", []):
        if a["content"] in v:
            assert v[a["content"]] == a["id"], a
        else:
            assert a["id"] >= len(v) or a["id"] not in set(v.values()), a
            v[a["content"]] = a["id"]
    assert sorted(v.values()) == list(range(len(v))), "model vocabulary not contiguous after filling added tokens"
    assert len(v) == ext["first_new_id"], (len(v), ext["first_new_id"])
    str_fmt = isinstance(j["model"]["merges"][0], str)
    for m in sel:
        if m["kind"] == "new":
            assert m["result"] not in v
            v[m["result"]] = m["id"]
        else:
            assert v[m["result"]] == m["id"]
        if str_fmt:
            assert " " not in m["left"] and " " not in m["right"]
            j["model"]["merges"].append(m["left"] + " " + m["right"])
        else:
            j["model"]["merges"].append([m["left"], m["right"]])
    return j


def materialize(base: str, k: int, out_dir: str = None, ext: dict = None, tag: str = None) -> str:
    ext = ext or C.load_json(os.path.join(work_dir(base, tag), "extension.json"))
    out_dir = out_dir or os.path.join(C.SWEEP + ("" if not tag else "__" + tag), base, "k%d" % k)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    if k == 0:
        j = C.load_base_json(base)
    else:
        j = extended_json(base, ext, k)
    s = json.dumps(j, ensure_ascii=False, indent=2)
    tk = Tokenizer.from_str(s)                                  # validation: loads in stock tokenizers
    bj = C.load_base_json(base)
    n_total = C.next_free_id(bj) if k == 0 else C.first_new_id(base)
    assert tk.get_vocab_size(with_added_tokens=True) == n_total + k, (tk.get_vocab_size(True), n_total, k)
    for a in bj.get("added_tokens", []) + (C.config_only_added_tokens(base) if k else []):   # special ids kept
        assert tk.token_to_id(a["content"]) == a["id"], a
        assert tk.id_to_token(a["id"]) == a["content"], a
    if k:
        for m in select(ext, k)[-50:]:
            assert tk.token_to_id(m["result"]) == m["id"] and tk.id_to_token(m["id"]) == m["result"], m
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "materialize", "all"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--k", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        ext = build(a.base, tag=a.tag, force=a.force)
    if a.cmd == "materialize":
        C.log(materialize(a.base, a.k, tag=a.tag))
    if a.cmd == "all":
        man = {}
        for k in [0] + C.K_GRID:
            p = materialize(a.base, k, ext=ext, tag=a.tag)
            man["k%d" % k] = {"path": p, "sha256": C.sha256_file(p), "bytes": os.path.getsize(p)}
            C.log(a.base, "k=%d" % k, man["k%d" % k]["sha256"][:16], p)
        C.dump_json(man, os.path.join(C.SWEEP + ("" if not a.tag else "__" + a.tag), a.base, "sweep_manifest.json"))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
