# -*- coding: utf-8 -*-
"""PickyBPE, reimplemented from Chizhov, Arnett, Korotkova, Yamshchikov (2024), "BPE Gets Picky: Efficient
Vocabulary Refinement During Tokenizer Training", EMNLP 2024, arXiv:2409.04599. No code from the authors'
repository was used or run; this is a from-the-paper reimplementation.

Algorithm (byte-level, on pretoken types with counts; PLAN.md A7):
  * vocabulary = 64 special tokens + 256 byte tokens + learned tokens; bytes and specials are never removed.
  * each step: take the most frequent adjacent pair (a, b) (ties: smallest (id_a, id_b), as HF BpeTrainer),
    create or re-activate c = a+b, merge every occurrence left to right   -> MERGE event.
  * then, for x in (a, b) (once if a == b), x learned and still present:
        IoS(x | a, b) = f_p(a, b) / f_t(x)          (both frequencies taken before the merge;
                                                       f_p = number of merges made at this step)
    if IoS >= tau: remove x from the vocabulary and split its remaining occurrences   -> REMOVE event.
    The split of x is walk(x): x's creating pair (a', b'), each part expanded recursively until it is a
    token that is present at that moment (bytes always are). The split is stored in the event.
  * stop when the number of PRESENT tokens (specials + bytes + learned) reaches vocab_size.
  * a removed token can come back if its pair is chosen again later (it is then re-activated with the same
    internal id); a merge whose product already exists (another pair spelling the same bytes) re-uses it.
  * tau=None disables removals: the trainer is then plain BPE and must reproduce HF BpeTrainer exactly
    (verified in verify_pickybpe.py against the A1 reference).

Encoder ("event-ordered", as in the paper): for each pretoken, replay the training events in their original
order: at a MERGE event (a, b) -> c every adjacent (a, b) is merged left to right; at a REMOVE event every x is
replaced by its stored split. Events that cannot apply to the word are skipped by jumping to the next
applicable event index. This reproduces exactly the state the trainer reached for every training word.

Frequencies are counts over the training corpus (pretoken type counts x occurrences in the type).
"""
from __future__ import annotations

import bisect
import heapq
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import regex

from pickybpe_common import B2U, P1_ONIG, P1_PY, SPECIAL_TOKENS, sha256_bytes

FORMAT = "hindko-pickybpe/1"
M20 = (1 << 20) - 1
M40 = (1 << 40) - 1
N_SPECIAL = len(SPECIAL_TOKENS)
BYTE_ORDER = sorted(range(256), key=lambda b: B2U[b])     # HF ByteLevel alphabet order (sorted by char)
BYTE_ID = {b: N_SPECIAL + k for k, b in enumerate(BYTE_ORDER)}
N_BASE = N_SPECIAL + 256


# ================================================================================================= trainer
class Trainer:
    def __init__(self, word_items: Sequence[Tuple[bytes, int]], vocab_size: int, tau: Optional[float],
                 min_frequency: int = 2, log=print, log_every: int = 1000):
        self.V = vocab_size
        self.tau = tau
        self.min_frequency = min_frequency
        self.log = log
        self.log_every = log_every
        self.tok_bytes: List[Optional[bytes]] = [None] * N_SPECIAL + [bytes([b]) for b in BYTE_ORDER]
        self.bytes2id: Dict[bytes, int] = {self.tok_bytes[i]: i for i in range(N_SPECIAL, N_BASE)}
        self.parents: List[Optional[Tuple[int, int]]] = [None] * N_BASE
        self.present: List[bool] = [True] * N_BASE
        self.n_present = N_BASE
        self.words: List[List[int]] = [[BYTE_ID[x] for x in w] for w, _ in word_items]
        self.counts: List[int] = [int(c) for _, c in word_items]
        self.tfreq: List[int] = [0] * N_BASE
        self.pc: Dict[int, int] = defaultdict(int)          # pair key (a<<20|b) -> count (overlapping, as HF)
        self.pwhere: Dict[int, set] = defaultdict(set)      # pair key -> word ids (lazy: may hold stale ids)
        self.twhere: Dict[int, set] = defaultdict(set)      # learned token -> word ids (lazy)
        self.events: List[list] = []
        self.stats = {"merges": 0, "merges_new_token": 0, "merges_existing_present": 0, "merges_readd": 0,
                      "removals": 0, "removal_occurrences": 0, "split_len_hist": defaultdict(int)}
        for wi, (seq, n) in enumerate(zip(self.words, self.counts)):
            for t in seq:
                self.tfreq[t] += n
            for j in range(len(seq) - 1):
                p = (seq[j] << 20) | seq[j + 1]
                self.pc[p] += n
                self.pwhere[p].add(wi)
        self.heap = [((-c) << 40) + p for p, c in self.pc.items() if c > 0]
        heapq.heapify(self.heap)

    # ------------------------------------------------------------------------------------------ helpers
    def _new_id(self, b: bytes) -> int:
        i = len(self.tok_bytes)
        self.tok_bytes.append(b)
        self.bytes2id[b] = i
        self.parents.append(None)
        self.present.append(False)
        self.tfreq.append(0)
        return i

    def _apply_deltas(self, delta: Dict[int, int]):
        pc, heap, pwhere = self.pc, self.heap, self.pwhere
        for q, d in delta.items():
            if d:
                v = pc[q] + d
                if v:
                    pc[q] = v
                else:
                    del pc[q]
                    pwhere.pop(q, None)           # no word holds the pair any more
                if d > 0:
                    heapq.heappush(heap, ((-v) << 40) + q)

    def _rewrite(self, wids, fn, index_tokens) -> int:
        """Rewrite every word in `wids` with fn(seq) -> (new_seq, k); update pair counts and the pair index,
        and add the word to the token index of every learned token in `index_tokens` (the tokens the rewrite
        can introduce). Returns the weighted number of rewrites."""
        words, counts, pwhere, twhere = self.words, self.counts, self.pwhere, self.twhere
        delta: Dict[int, int] = defaultdict(int)
        total = 0
        for wi in sorted(wids):
            seq = words[wi]
            new, k = fn(seq)
            if not k:
                continue
            n = counts[wi]
            for j in range(len(seq) - 1):
                delta[(seq[j] << 20) | seq[j + 1]] -= n
            for j in range(len(new) - 1):
                q = (new[j] << 20) | new[j + 1]
                delta[q] += n
                pwhere[q].add(wi)
            for t in index_tokens:
                twhere[t].add(wi)
            words[wi] = new
            total += k * n
        self._apply_deltas(delta)
        return total

    def walk(self, t: int) -> List[int]:
        if t < N_BASE or self.present[t]:
            return [t]
        a, b = self.parents[t]
        return self.walk(a) + self.walk(b)

    # ------------------------------------------------------------------------------------------ operations
    def merge(self, a: int, b: int, c: int) -> int:
        tf = self.tfreq

        def fn(seq):
            L = len(seq)
            new = []
            i = k = 0
            while i < L:
                if i + 1 < L and seq[i] == a and seq[i + 1] == b:
                    new.append(c)
                    i += 2
                    k += 1
                else:
                    new.append(seq[i])
                    i += 1
            return new, k

        wids = self.pwhere.pop((a << 20) | b, ())
        m = self._rewrite(wids, fn, (c,))
        tf[a] -= m
        tf[b] -= m
        tf[c] += m
        for x in (a, b):
            if tf[x] == 0:
                self.twhere.pop(x, None)          # fully consumed: its index can go (re-created if it returns)
        return m

    def remove(self, x: int) -> (List[int], int):
        self.present[x] = False
        self.n_present -= 1
        split = self.walk(x)
        assert len(split) >= 2 and all(self.present[s] for s in split)
        assert b"".join(self.tok_bytes[s] for s in split) == self.tok_bytes[x]
        sp = list(split)

        def fn(seq):
            if x not in seq:
                return seq, 0
            new = []
            k = 0
            for t in seq:
                if t == x:
                    new.extend(sp)
                    k += 1
                else:
                    new.append(t)
            return new, k

        wids = self.twhere.pop(x, ())
        occ = self._rewrite(wids, fn, tuple(sorted({s for s in sp if s >= N_BASE})))
        self.tfreq[x] -= occ
        for s in split:
            self.tfreq[s] += occ
        assert self.tfreq[x] == 0, (x, self.tfreq[x])
        return split, occ

    def pop_best(self) -> Optional[Tuple[int, int]]:
        heap, pc = self.heap, self.pc
        while heap:
            key = heapq.heappop(heap)
            p = key & M40
            cnt = -(key >> 40)
            cur = pc.get(p, 0)
            if cur == cnt:
                return p, cnt
            if cur > 0:
                heapq.heappush(heap, ((-cur) << 40) + p)
        return None

    # ------------------------------------------------------------------------------------------ main loop
    def train(self) -> "Trainer":
        t0 = time.time()
        st = self.stats
        while self.n_present < self.V:
            best = self.pop_best()
            if best is None:
                self.log("no pairs left")
                break
            p, cnt = best
            if cnt < self.min_frequency:
                self.log("stopping: best pair count %d < min_frequency %d" % (cnt, self.min_frequency))
                break
            a, b = p >> 20, p & M20
            nb = self.tok_bytes[a] + self.tok_bytes[b]
            c = self.bytes2id.get(nb)
            if c is None:
                c = self._new_id(nb)
                st["merges_new_token"] += 1
            elif self.present[c]:
                st["merges_existing_present"] += 1
            else:
                st["merges_readd"] += 1
            if not self.present[c]:
                self.present[c] = True
                self.n_present += 1
            self.parents[c] = (a, b)
            fa, fb = self.tfreq[a], self.tfreq[b]
            m = self.merge(a, b, c)
            if a != b:
                assert m == cnt, (a, b, m, cnt)
            st["merges"] += 1
            ev = [0, a, b, c, m, fa, fb]
            self.events.append(ev)
            if self.tau is not None:
                for x, fx in ((a, fa),) if a == b else ((a, fa), (b, fb)):
                    if x >= N_BASE and self.present[x] and fx > 0 and m / fx >= self.tau:
                        split, occ = self.remove(x)
                        self.events.append([1, x, split, occ, fx, m])
                        st["removals"] += 1
                        st["removal_occurrences"] += occ
                        st["split_len_hist"][len(split)] += 1
            if st["merges"] % self.log_every == 0:
                self.log("merge %d: present %d / %d, removals %d, pair count %d, %.1fs" % (
                    st["merges"], self.n_present, self.V, st["removals"], cnt, time.time() - t0))
        st["seconds"] = round(time.time() - t0, 1)
        st["split_len_hist"] = dict(sorted(st["split_len_hist"].items()))
        return self

    # ------------------------------------------------------------------------------------------ export
    def final_state_counts(self) -> List[int]:
        """Token frequencies of the final training state (for the trainer/encoder consistency check)."""
        return list(self.tfreq)

    def to_model(self, meta: dict) -> dict:
        n = len(self.tok_bytes)
        final = list(range(N_SPECIAL)) + list(range(N_SPECIAL, N_BASE)) + \
            [i for i in range(N_BASE, n) if self.present[i]]
        created = [i for i in range(N_BASE, n)]
        removed_final = [i for i in created if not self.present[i]]
        ever_removed = sorted({e[1] for e in self.events if e[0] == 1})
        return {
            "format": FORMAT,
            "algorithm": "PickyBPE, reimplemented from Chizhov et al. 2024 (EMNLP 2024, arXiv:2409.04599); "
                         "byte-level; IoS removals during training; event-ordered encoder",
            "tau": self.tau,
            "vocab_size": len(final),
            "target_vocab_size": self.V,
            "min_frequency": self.min_frequency,
            "byte_level": True,
            "pretokenizer": {"name": "P1", "oniguruma": P1_ONIG, "python_regex": P1_PY,
                             "behavior": "isolated split (matches and gaps, in order), then UTF-8 bytes"},
            "special_tokens": SPECIAL_TOKENS,
            "n_special": N_SPECIAL,
            "byte_order": "internal ids %d..%d are the 256 bytes sorted by their GPT-2 byte-level character "
                          "(HF ByteLevel alphabet order)" % (N_SPECIAL, N_BASE - 1),
            "tokens": [None if b is None else b.hex() for b in self.tok_bytes],
            "final_ids": final,
            "events": self.events,
            "event_fields": {"0": "[0, a, b, c, merges_made, f_t(a) before, f_t(b) before]",
                             "1": "[1, x, split_ids, occurrences_split, f_t(x) before the merge, f_p(pair)]"},
            "summary": {"internal_tokens": n, "learned_created": len(created),
                        "removed_at_end": len(removed_final), "ever_removed": len(ever_removed),
                        "readded": len(set(ever_removed) - set(removed_final)),
                        "events": len(self.events),
                        "stats": {k: v for k, v in self.stats.items() if k != "seconds"}},
            "meta": meta,
        }


def save_model(model: dict, path: str) -> str:
    s = json.dumps(model, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    return sha256_bytes(s.encode("utf-8"))


# ================================================================================================= encoder
class PickyBPETokenizer:
    """Event-ordered PickyBPE encoder with the interface eval/adapters.CustomAdapter expects."""
    byte_level = True
    char_level = False
    model_type = "PickyBPE"

    def __init__(self, path: str, name: Optional[str] = None, cache_size: int = 1 << 20):
        from pickybpe_common import sha256_file
        self.path = path
        self.sha256 = sha256_file(path)
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("format") != FORMAT:
            raise ValueError("not a %s file: %s" % (FORMAT, path))
        self.m = m
        self.name = name or "pickybpe"
        self.tau = m["tau"]
        self.itok = [None if h is None else bytes.fromhex(h) for h in m["tokens"]]
        self.final_ids = m["final_ids"]
        self.i2f = {i: k for k, i in enumerate(self.final_ids)}
        self.vocab_size = len(self.final_ids)
        self.id_upper = self.vocab_size
        self.special_tokens = {s: k for k, s in enumerate(m["special_tokens"])}
        self.unk_ids = frozenset()
        self.base_ids = frozenset(self.i2f[i] for i in range(N_SPECIAL, N_BASE))
        self.fbytes = [self.itok[i] if i >= N_SPECIAL else m["special_tokens"][i].encode("utf-8")
                       for i in self.final_ids]
        self.rx = regex.compile(P1_PY)
        sp = sorted(m["special_tokens"], key=len, reverse=True)
        self.special_rx = regex.compile("|".join(regex.escape(s) for s in sp))
        # event tables
        mev = defaultdict(list)       # pair key -> sorted event indices
        rev = defaultdict(list)       # token -> sorted removal event indices
        self.ev_kind, self.ev_a, self.ev_b, self.ev_c, self.ev_split = [], [], [], [], []
        for k, e in enumerate(m["events"]):
            self.ev_kind.append(e[0])
            if e[0] == 0:
                mev[(e[1] << 20) | e[2]].append(k)
                self.ev_a.append(e[1]); self.ev_b.append(e[2]); self.ev_c.append(e[3]); self.ev_split.append(None)
            else:
                rev[e[1]].append(k)
                self.ev_a.append(e[1]); self.ev_b.append(None); self.ev_c.append(None)
                self.ev_split.append(tuple(e[2]))
        self.mev = dict(mev)
        self.rev = dict(rev)
        self.removed_at_end = frozenset(i for i in range(N_BASE, len(self.itok)) if i not in self.i2f)
        # merges for the harness R1 leaf/intermediate split: the last creating merge of every present learned
        # token, when both parts are present at the end too (final ids)
        last = {}
        for e in m["events"]:
            if e[0] == 0:
                last[e[3]] = (e[1], e[2])
        self.merges = [(self.i2f[a], self.i2f[b], self.i2f[c]) for c, (a, b) in last.items()
                       if c in self.i2f and a in self.i2f and b in self.i2f]
        self.merge_components_internal = {x for e in m["events"] if e[0] == 0 and e[3] in self.i2f
                                          for x in (e[1], e[2])}
        self._cache: Dict[bytes, Tuple[int, ...]] = {}
        self._cache_size = cache_size
        self.identity = {"path": path, "sha256": self.sha256, "format": FORMAT, "tau": self.tau,
                         "encoder": "custom Python event-ordered PickyBPE encoder (candidates/pickybpe/pickybpe.py, "
                                    "reimplemented from Chizhov et al. 2024)",
                         "pretokenizer": "P1 (Python regex twin)", "normalizer": None,
                         "removed_tokens_at_end": len(self.removed_at_end)}

    # ------------------------------------------------------------------------------------------ core
    def encode_word_internal(self, wb: bytes) -> Tuple[int, ...]:
        """Event-ordered encoding of one pretoken (UTF-8 bytes) into INTERNAL ids."""
        r = self._cache.get(wb)
        if r is not None:
            return r
        seq = [BYTE_ID[x] for x in wb]
        mev, rev = self.mev, self.rev
        kinds, eva, evb, evc, evs = self.ev_kind, self.ev_a, self.ev_b, self.ev_c, self.ev_split
        e = -1
        INF = 1 << 62
        while True:
            best = INF
            for j in range(len(seq) - 1):
                lst = mev.get((seq[j] << 20) | seq[j + 1])
                if lst:
                    if lst[-1] > e:
                        k = lst[0] if lst[0] > e else lst[bisect.bisect_right(lst, e)]
                        if k < best:
                            best = k
            for t in seq:
                if t >= N_BASE:
                    lst = rev.get(t)
                    if lst and lst[-1] > e:
                        k = lst[0] if lst[0] > e else lst[bisect.bisect_right(lst, e)]
                        if k < best:
                            best = k
            if best == INF:
                break
            if kinds[best] == 0:
                a, b, c = eva[best], evb[best], evc[best]
                new = []
                i, L = 0, len(seq)
                while i < L:
                    if i + 1 < L and seq[i] == a and seq[i + 1] == b:
                        new.append(c)
                        i += 2
                    else:
                        new.append(seq[i])
                        i += 1
                seq = new
            else:
                x, sp = eva[best], evs[best]
                new = []
                for t in seq:
                    if t == x:
                        new.extend(sp)
                    else:
                        new.append(t)
                seq = new
            e = best
        r = tuple(seq)
        if len(self._cache) < self._cache_size:
            self._cache[wb] = r
        return r

    def pretokens(self, text: str) -> List[str]:
        out, pos = [], 0
        for mm in self.rx.finditer(text):
            if mm.start() > pos:
                out.append(text[pos:mm.start()])
            if mm.end() > mm.start():
                out.append(mm.group())
            pos = mm.end()
        if pos < len(text):
            out.append(text[pos:])
        return out

    def _segments(self, text: str):
        """(is_special, piece) segments: special-token strings are matched first, as HF added tokens are."""
        pos = 0
        for mm in self.special_rx.finditer(text):
            if mm.start() > pos:
                yield False, text[pos:mm.start()]
            yield True, mm.group()
            pos = mm.end()
        if pos < len(text):
            yield False, text[pos:]

    def encode_internal(self, text: str) -> List[int]:
        out = []
        for sp, piece in self._segments(text):
            if sp:
                out.append(-1 - self.special_tokens[piece])      # marker, mapped in encode()
                continue
            for pt in self.pretokens(piece):
                out.extend(self.encode_word_internal(pt.encode("utf-8")))
        return out

    # ------------------------------------------------------------------------------------------ public API
    def encode(self, text: str) -> List[int]:
        i2f = self.i2f
        return [(-1 - i) if i < 0 else i2f[i] for i in self.encode_internal(text)]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def encode_isolated(self, text: str) -> List[int]:
        return self.encode(text)

    def decode_bytes(self, ids) -> bytes:
        fb = self.fbytes
        return b"".join(fb[i] for i in ids)

    def decode(self, ids) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> Optional[bytes]:
        if 0 <= i < self.vocab_size:
            return self.fbytes[i]
        return None

    def token_to_id(self, s: str) -> Optional[int]:
        return self.special_tokens.get(s)

    def id_to_token(self, i: int) -> Optional[str]:
        """Byte-level string of a final id (GPT-2 alphabet), specials as themselves."""
        if i < N_SPECIAL:
            return self.m["special_tokens"][i]
        b = self.token_bytes(i)
        return None if b is None else "".join(B2U[x] for x in b)

    def get_vocab(self) -> Dict[str, int]:
        return {self.id_to_token(i): i for i in range(self.vocab_size)}
