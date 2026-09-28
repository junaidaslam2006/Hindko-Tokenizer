# -*- coding: utf-8 -*-
"""Reference encoder for the SuperBPE candidate: pure Python, independent of HF `tokenizers` at encode time.

It encodes exactly the way the stage-2 training assumed:
  1. added (special) tokens are cut out first (longest marker first), as HF does;
  2. the rest is split with the Python `regex` twin of S2 (common.onig_to_py);
  3. phase 1: inside each chunk, the stage-1 merges are applied by rank to the chunk's UTF-8 bytes
     (lowest rank first, leftmost position among equal ranks; a queued pair is applied only if that exact
     pair is still adjacent) -> stage-1 ids;
  4. phase 2: the stage-2 merges are applied the same way in the PUA space of the trainer (stage-1 id
     sequences; every stage-2 token is identified by its stage-1 id sequence, not by its bytes);
  5. every PUA-space token is mapped to the id of its byte string in the final tokenizer.json.
The HF-native tokenizer.json applies ALL merges by rank in byte space in one pass. The two agree when no
stage-2 token duplicates the bytes of another token (checked in build_info.json) and the Oniguruma and
Python regexes split identically (verify.py).

Also usable as a harness custom encoder:  harness.py run --custom ref_encoder:factory_<name>
(set SBPE_DIR), exposing encode/decode/token_bytes/vocab_size/special_tokens/base_ids/merges.
"""
from __future__ import annotations

import heapq
import json
import os
import sys

import regex

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def bpe_by_rank(ids, ranks):
    """HF Word::merge_all semantics (no dropout). ranks: {(a, b): (rank, new_id)}."""
    n = len(ids)
    if n < 2:
        return list(ids)
    sym = list(ids)
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    prv = list(range(-1, n - 1))
    alive = [True] * n
    heap = []
    for i in range(n - 1):
        m = ranks.get((sym[i], sym[i + 1]))
        if m is not None:
            heap.append((m[0], i, m[1]))
    heapq.heapify(heap)
    while heap:
        r, i, new = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1:
            continue
        m = ranks.get((sym[i], sym[j]))
        if m is None or m[0] != r:
            continue                                   # expired entry
        sym[i] = new
        alive[j] = False
        k = nxt[j]
        nxt[i] = k
        if k != -1:
            prv[k] = i
        p = prv[i]
        if p != -1:
            m = ranks.get((sym[p], sym[i]))
            if m is not None:
                heapq.heappush(heap, (m[0], p, m[1]))
        if k != -1:
            m = ranks.get((sym[i], sym[k]))
            if m is not None:
                heapq.heappush(heap, (m[0], i, m[1]))
    return [sym[i] for i in range(n) if alive[i]]


def py_pretokens(rx, text):
    """Python twin of HF Split(behavior='isolated')."""
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


class SuperBPERef:
    kind = "custom"
    byte_level = True
    char_level = False
    model_type = "SuperBPE-reference"

    def __init__(self, out_dir: str, name: str = None):
        self.out_dir = out_dir
        self.name = name or ("ref:" + os.path.basename(out_dir.rstrip("/\\")))
        pua = json.load(open(os.path.join(out_dir, "stage2_pua.json"), encoding="utf-8"))
        if C.sha256_file(pua["stage1"]) != pua["stage1_sha256"]:
            raise SystemExit("stage-1 file changed since the build")
        s1 = json.load(open(pua["stage1"], encoding="utf-8"))
        fin = json.load(open(os.path.join(out_dir, "tokenizer.json"), encoding="utf-8"))
        v1 = s1["model"]["vocab"]
        self.t = pua["t"]
        m1 = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in s1["model"]["merges"]]
        self.rank1 = {}
        for r, (a, b) in enumerate(m1):
            key = (v1[a], v1[b])
            if key not in self.rank1:                   # first occurrence (no duplicates expected)
                self.rank1[key] = (r, v1[a + b])
        self.byte_id = [v1[C.B2U[b]] for b in range(256)]
        id2tok1 = {i: s for s, i in v1.items()}
        # PUA space: alphabet = stage-1 ids; new tokens t, t+1, ... in creation order, keyed by id sequence
        seq_id = {(i,): i for i in range(C.N_SPECIAL, self.t)}
        id_seq = {i: s for s, i in seq_id.items()}
        self.rank2 = {}
        nid = self.t
        for r, (sa, sb) in enumerate(pua["merges"]):
            sa, sb = tuple(sa), tuple(sb)
            ns = sa + sb
            if ns not in seq_id:
                seq_id[ns] = nid
                id_seq[nid] = ns
                nid += 1
            key = (seq_id[sa], seq_id[sb])
            if key not in self.rank2:
                self.rank2[key] = (r, seq_id[ns])
        fv = fin["model"]["vocab"]
        self.vocab = fv
        self.pua_to_final = {}
        for i, s in id_seq.items():
            self.pua_to_final[i] = fv["".join(id2tok1[x] for x in s)]
        self.special_tokens = {a["content"]: a["id"] for a in fin["added_tokens"]}
        self.vocab_size = len(fv) + sum(1 for a in fin["added_tokens"] if a["content"] not in fv)
        self.id_upper = max(max(fv.values()), max(self.special_tokens.values())) + 1
        self.base_ids = frozenset(fv[C.B2U[b]] for b in range(256))
        self.unk_ids = frozenset()
        self._tb = {i: C.bl_to_bytes(s) for s, i in fv.items() if s not in self.special_tokens}
        for s, i in self.special_tokens.items():
            self._tb[i] = s.encode("utf-8")
        self.merges = [(fv[a], fv[b], fv[a + b]) for a, b in
                       (tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in fin["model"]["merges"])]
        self.rx = regex.compile(C.onig_to_py(C.S2))
        specs = sorted(self.special_tokens, key=len, reverse=True)
        self.spec_rx = regex.compile("|".join(regex.escape(s) for s in specs)) if specs else None
        self._cache = {}
        self.identity = {"encoder": "SuperBPE reference encoder (pure Python, two-phase: stage-1 merges then "
                                    "PUA-space stage-2 merges, per S2 chunk)",
                         "path": os.path.join(out_dir, "tokenizer.json"),
                         "sha256": C.sha256_file(os.path.join(out_dir, "tokenizer.json"))}

    # --- phases
    def phase1(self, chunk: str):
        return bpe_by_rank([self.byte_id[b] for b in chunk.encode("utf-8")], self.rank1)

    def encode_chunk(self, chunk: str):
        r = self._cache.get(chunk)
        if r is None:
            ids1 = self.phase1(chunk)
            ids2 = bpe_by_rank(ids1, self.rank2)
            r = [self.pua_to_final[i] for i in ids2]
            if len(self._cache) < 2_000_000:
                self._cache[chunk] = r
        return r

    def _pieces(self, text):
        if self.spec_rx is None:
            yield text, None
            return
        pos = 0
        for m in self.spec_rx.finditer(text):
            if m.start() > pos:
                yield text[pos:m.start()], None
            yield m.group(), self.special_tokens[m.group()]
            pos = m.end()
        if pos < len(text):
            yield text[pos:], None

    def encode(self, text: str):
        out = []
        for piece, sid in self._pieces(text):
            if sid is not None:
                out.append(sid)
                continue
            for ch in py_pretokens(self.rx, piece):
                out.extend(self.encode_chunk(ch))
        return out

    def encode_phase1(self, text: str):
        out = []
        for piece, sid in self._pieces(text):
            if sid is not None:
                out.append(sid)
                continue
            for ch in py_pretokens(self.rx, piece):
                out.extend(self.phase1(ch))
        return out

    def token_bytes(self, i):
        return self._tb.get(i)

    def decode(self, ids):
        return b"".join(self._tb[i] for i in ids).decode("utf-8", errors="replace")

    def encode_isolated(self, text):
        return self.encode(text)


def factory():
    return SuperBPERef(os.environ["SBPE_DIR"])
