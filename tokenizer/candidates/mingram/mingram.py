# -*- coding: utf-8 -*-
"""MinGram as reimplemented from the paper.

Paper: S. Land (2026), "MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive
Morphological Alignment", arXiv 2606.27019 (v2, read in the arXiv HTML on 2026-09-26). The author's code
(sanderland/script_tok) was NOT downloaded, read or run (task rule: no third-party research code). Every
choice the paper leaves open is recorded in MINGRAM.md, section "Decisions the paper leaves open".

Algorithm 1 of the paper, as implemented here (train_mingram.py drives it):
  1  train BPE on the corpus to ceil(f * n) learned tokens (f = 1.15)                   -> seed vocabulary
  2  log p(t) from each seed token's frequency in the BPE-encoded corpus
  3  N_em = 2 hard-EM iterations:
       E: encode every pretoken with the minimum-token path, Unigram score as tie-break (lexicographic:
          token count first, then the summed log p), and count the tokens on that one path
       M: log p(t) = log(count(t) / sum of counts)
  4  one flat prune: drop the lowest-log-p learned (non-atomic) tokens until n remain; atomic tokens
     (every single character of the training alphabet, the 256 <0xNN> byte pieces) are never pruned

Hindko specifics (PLAN.md A10): P1 pretokens, byte_fallback, the 64-token special block, 16,384 in total.

Exact integer scores. log p is quantised to integer multiples of 2^-Q_BITS nats (score_q). The reference
Viterbi works on the pair (token count, sum of score_q) with integer arithmetic, so its comparisons are
exact. The HF export gives each piece the float score  -C_SHIFT + score_q / 2^Q_BITS. With these values every
path score is an exactly representable double, so the stock HF Unigram Viterbi (which maximises the summed
score) makes exactly the reference's lexicographic decision whenever the minimum path has fewer than
C_SHIFT / (|min log p| + 10) tokens (about 19,000 for this vocabulary; see MINGRAM.md), and it breaks exact
ties the same way (the first path found in a left-to-right scan is kept). equiv_test.py verifies this.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TOK = r"F:\Hindko\_tokenizer"
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(TOK, "eval"))
import harness as H  # noqa: E402  (load_docs with test refusal, P1 regex, py_pretokens, special block)

LABEL = "MinGram as reimplemented from the paper (Land 2026, arXiv 2606.27019)"
MINGRAM_VERSION = "1.0.0"
P1 = H.PRETOKENIZERS["P1"]
SPECIALS = list(H.SPECIAL_TOKENS)
UNK = "<unk>"
BYTE_PIECES = ["<0x%02X>" % b for b in range(256)]
Q_BITS = 9                      # score quantum 2^-9 nats (see below: JSON float parsing in tokenizers)
C_SHIFT = 2 ** 19               # the "-C" of the HF export (a power of two keeps every sum exact)
# Why 2^19 and 2^-9: tokenizers parses tokenizer.json with serde_json's default float parser, which is exact only
# when the decimal significand is below 2^53 (measured: with a 2^-20 quantum and 17-digit decimals, 353 of 16,384
# scores came back 1 ulp off). -2^19 + m/2^9 has at most 6 integer and 9 decimal digits (15 significant digits),
# so every score survives the file round trip bit-exactly (checked by check_hf_scores).
HF_UNK_PENALTY = 10.0           # tokenizers Unigram: unk score = min vocab score - 10.0
DEFAULT_MODEL = os.path.join(HERE, "runs", "mingram_P1_D1_16384", "mingram_model.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------------------------------------ pretokens
_RX = None


def pretokenize(text: str) -> List[str]:
    """P1 pretokens, Python `regex` twin of the Oniguruma pattern (identical on dev: eval/pretok_differential.json)."""
    global _RX
    if _RX is None:
        import regex
        _RX = regex.compile(H.onig_to_py(P1))
    return H.py_pretokens(_RX, text)


def type_counts(texts: Iterable[str]) -> collections.Counter:
    c = collections.Counter()
    for t in texts:
        c.update(pretokenize(t))
    return c


# ------------------------------------------------------------------------------------------- scores
def quantise(lp: float) -> int:
    return int(round(lp * (1 << Q_BITS)))


def mstep(counts: Dict[str, int], pieces: Sequence[str]) -> Tuple[Dict[str, int], dict, Dict[str, float]]:
    """log p(t) = log(count / total), quantised (and the unquantised value). A token with count 0 gets the floor
    log(0.5 / total) (the paper leaves zero counts unspecified; see MINGRAM.md)."""
    N = sum(counts.get(p, 0) for p in pieces)
    floor = math.log(0.5 / N)
    out, lpf = {}, {}
    for p in pieces:
        c = counts.get(p, 0)
        lpf[p] = math.log(c / N) if c > 0 else floor
        out[p] = quantise(lpf[p])
    return out, {"total_count": N, "floor_logp": floor, "floor_q": quantise(floor),
                 "zero_count_tokens": sum(1 for p in pieces if counts.get(p, 0) == 0)}, lpf


# ------------------------------------------------------------------------------------------ lattice
class Lattice:
    """Minimum-token Viterbi over one pretoken with integer tie-break scores.

    table: piece -> score_q for EVERY piece the HF model's trie holds (the HF Unigram trie contains the whole
    vocabulary; specials and <0xNN> pieces can never match inside a P1 pretoken, but they are kept for exactness).
    unk_q: score of an unknown-character node, used only where no single-character piece matches (as in HF).
    """

    def __init__(self, table: Dict[str, int], unk_q: int):
        self.table = table
        self.unk_q = unk_q
        self.prefixes = set()
        for p in table:
            for k in range(1, len(p) + 1):
                self.prefixes.add(p[:k])
        self.maxlen = max(len(p) for p in table)

    def best_hf(self, s: str) -> List[Tuple[int, int, Optional[str]]]:
        """Left-to-right; on an exact tie the first path found is kept (tokenizers' encode_optimized rule, which
        prefers the path whose last token starts earliest). Returns [(start, end, piece or None=unknown char)]."""
        L = len(s)
        INF = 1 << 62
        bc = [INF] * (L + 1)
        bs = [0] * (L + 1)
        bk = [0] * (L + 1)
        bp: List[Optional[str]] = [None] * (L + 1)
        bc[0] = 0
        table, pre, maxlen, unk = self.table, self.prefixes, self.maxlen, self.unk_q
        for i in range(L):
            c1 = bc[i] + 1
            s0 = bs[i]
            single = False
            end = L if L < i + maxlen else i + maxlen
            for j in range(i + 1, end + 1):
                sub = s[i:j]
                if sub not in pre:
                    break
                sc = table.get(sub)
                if sc is None:
                    continue
                if j == i + 1:
                    single = True
                sc += s0
                cj = bc[j]
                if c1 < cj or (c1 == cj and sc > bs[j]):
                    bc[j] = c1; bs[j] = sc; bk[j] = i; bp[j] = sub
            if not single:
                j = i + 1
                sc = s0 + unk
                cj = bc[j]
                if c1 < cj or (c1 == cj and sc > bs[j]):
                    bc[j] = c1; bs[j] = sc; bk[j] = i; bp[j] = None
        out = []
        j = L
        while j > 0:
            i = bk[j]
            out.append((i, j, bp[j]))
            j = i
        out.reverse()
        return out

    def best_paper(self, s: str) -> List[Tuple[int, int, Optional[str]]]:
        """The same objective with the paper's exact-tie rule ("prefers the segmentation with longest leading
        tokens"): right-to-left DP, candidates tried longest first, strict improvement only."""
        L = len(s)
        INF = 1 << 62
        fc = [INF] * (L + 1)
        fs = [0] * (L + 1)
        nx = [0] * (L + 1)
        fp: List[Optional[str]] = [None] * (L + 1)
        fc[L] = 0
        table, pre, maxlen, unk = self.table, self.prefixes, self.maxlen, self.unk_q
        for i in range(L - 1, -1, -1):
            cands = []
            single = False
            end = L if L < i + maxlen else i + maxlen
            for j in range(i + 1, end + 1):
                sub = s[i:j]
                if sub not in pre:
                    break
                sc = table.get(sub)
                if sc is None:
                    continue
                if j == i + 1:
                    single = True
                cands.append((j, sub, sc))
            if not single:
                cands.append((i + 1, None, unk))
            for j, sub, sc in sorted(cands, key=lambda x: -x[0]):
                c = fc[j] + 1
                t = fs[j] + sc
                if c < fc[i] or (c == fc[i] and t > fs[i]):
                    fc[i] = c; fs[i] = t; nx[i] = j; fp[i] = sub
        out = []
        i = 0
        while i < L:
            out.append((i, nx[i], fp[i]))
            i = nx[i]
        return out

    def min_count(self, s: str) -> int:
        """Minimum number of lattice nodes (no scores): an independent check of the count objective."""
        L = len(s)
        INF = 1 << 62
        bc = [INF] * (L + 1)
        bc[0] = 0
        table, pre, maxlen = self.table, self.prefixes, self.maxlen
        for i in range(L):
            c1 = bc[i] + 1
            single = False
            end = L if L < i + maxlen else i + maxlen
            for j in range(i + 1, end + 1):
                sub = s[i:j]
                if sub not in pre:
                    break
                if sub in table:
                    if j == i + 1:
                        single = True
                    if c1 < bc[j]:
                        bc[j] = c1
            if not single and c1 < bc[i + 1]:
                bc[i + 1] = c1
        return bc[L]


def estep(types: Dict[str, int], lat: Lattice, tie: str = "hf") -> Tuple[Dict[str, int], int]:
    """Hard E-step: counts along the single minimum-token path of every pretoken type, weighted by type frequency."""
    best = lat.best_hf if tie == "hf" else lat.best_paper
    cnt: Dict[str, int] = collections.defaultdict(int)
    ntok = 0
    unk = 0
    for s, f in types.items():
        path = best(s)
        ntok += f * len(path)
        for _, _, p in path:
            if p is None:
                unk += f
            else:
                cnt[p] += f
    if unk:
        raise RuntimeError("unknown characters in training pretokens: %d" % unk)
    return dict(cnt), ntok


# ------------------------------------------------------------------------------------------ model file
def model_float_score(q: int) -> float:
    return -C_SHIFT + q / (1 << Q_BITS)


def build_model(atomic_chars: Sequence[str], learned: Sequence[str], score_q: Dict[str, int],
                extra: dict) -> dict:
    """Id layout: 0-63 special block, 64 <unk>, 65-320 <0x00>..<0xFF>, then the alphabet (by code point), then the
    learned tokens by descending score (ties: by string). Specials, <unk> and byte pieces get the minimum score of
    the learned/atomic pieces, so HF's min_score (and hence its unknown-character score) is that minimum."""
    atomic_chars = sorted(atomic_chars)
    learned = sorted(learned, key=lambda p: (-score_q[p], p))
    minq = min(score_q[p] for p in list(atomic_chars) + list(learned))
    pieces = []
    for s in SPECIALS:
        pieces.append({"piece": s, "kind": "special", "score_q": minq})
    pieces.append({"piece": UNK, "kind": "unk", "score_q": minq})
    for b in BYTE_PIECES:
        pieces.append({"piece": b, "kind": "byte", "score_q": minq})
    for c in atomic_chars:
        pieces.append({"piece": c, "kind": "char", "score_q": score_q[c]})
    for p in learned:
        pieces.append({"piece": p, "kind": "learned", "score_q": score_q[p]})
    for i, p in enumerate(pieces):
        p["id"] = i
    m = {"label": LABEL, "mingram_version": MINGRAM_VERSION, "q_bits": Q_BITS, "c_shift": C_SHIFT,
         "hf_unk_penalty": HF_UNK_PENALTY, "pretokenizer": {"id": "P1", "oniguruma": P1},
         "special_tokens": SPECIALS, "unk_id": len(SPECIALS), "byte_fallback": True,
         "min_score_q": minq, "unk_score_q": minq - int(HF_UNK_PENALTY * (1 << Q_BITS)),
         "vocab_size": len(pieces), "pieces": pieces}
    m.update(extra)
    return m


def export_hf(model: dict, path: str) -> str:
    """HF-native candidate: a stock tokenizers Unigram whose piece scores are -C + log p (fewest tokens first,
    log p as tie-break), P1 Split pre-tokenizer, byte_fallback, ByteFallback+Fuse decoder, no normalizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex, AddedToken
    vocab = [(p["piece"], model_float_score(p["score_q"])) for p in model["pieces"]]
    tk = Tokenizer(models.Unigram(vocab, unk_id=model["unk_id"], byte_fallback=True))
    tk.normalizer = None
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(P1), behavior="isolated", invert=False)
    tk.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in SPECIALS])
    tk.save(path, pretty=True)
    return path


def check_hf_scores(model: dict, path: str) -> dict:
    """The scores the saved tokenizer.json holds after tokenizers parses and re-serialises it must be bit-identical
    to the intended doubles (JSON float round trip), ids must match, and the special ids must be 0-63."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(path)
    j = json.loads(tk.to_str())
    v = j["model"]["vocab"]
    bad = 0
    for p, (s, sc) in zip(model["pieces"], v):
        if s != p["piece"] or float(sc).hex() != model_float_score(p["score_q"]).hex():
            bad += 1
    spec_ok = all(tk.token_to_id(s) == i for i, s in enumerate(SPECIALS))
    raw = json.load(open(path, encoding="utf-8"))["model"]["vocab"]
    digits = max(len(re.sub(r"[^0-9]", "", repr(float(sc))).lstrip("0")) for _, sc in raw)
    return {"pieces": len(v), "score_or_piece_mismatch": bad, "len_match": len(v) == len(model["pieces"]),
            "max_significant_digits_in_file": digits,
            "special_ids_0_63": spec_ok, "unk_id": j["model"].get("unk_id"),
            "byte_fallback": j["model"].get("byte_fallback"), "vocab_size_with_added": tk.get_vocab_size(True)}


# ------------------------------------------------------------------------------------ reference encoder
class MinGramRef:
    """Pure-Python reference encoder of a trained MinGram model (mingram_model.json).

    tie='hf'    exact ties broken as tokenizers' left-to-right Viterbi does (the HF export must match this)
    tie='paper' exact ties broken by "longest leading tokens" (the paper's implementation note)
    Special-block strings in the text are emitted as their ids first (leftmost-longest), as HF's added
    vocabulary does; the rest is split into P1 pretokens and each pretoken is segmented independently. An unknown
    character becomes its UTF-8 bytes as <0xNN> pieces (byte_fallback). Exposes the CustomAdapter interface of
    eval/adapters.py.
    """
    model_type = "Unigram"
    byte_level = False
    char_level = True

    def __init__(self, model_path: str = DEFAULT_MODEL, tie: str = "hf"):
        self.path = model_path
        with open(model_path, encoding="utf-8") as f:
            m = json.load(f)
        self.m = m
        self.tie = tie
        self.pieces = [p["piece"] for p in m["pieces"]]
        self.kind = [p["kind"] for p in m["pieces"]]
        self.id_of = {p: i for i, p in enumerate(self.pieces)}
        table = {p["piece"]: p["score_q"] for p in m["pieces"]}
        self.lat = Lattice(table, m["unk_score_q"])
        self.vocab_size = self.id_upper = len(self.pieces)
        self.special_tokens = {s: self.id_of[s] for s in m["special_tokens"]}
        self.unk_ids = {m["unk_id"]}
        self.byte_id = [self.id_of[b] for b in BYTE_PIECES]
        self.base_ids = set(self.byte_id)
        self._cache: Dict[str, List[int]] = {}
        sp = sorted(self.special_tokens, key=len, reverse=True)
        self._sp_rx = re.compile("|".join(re.escape(s) for s in sp))
        self.name = "mingram-ref-" + tie
        self.identity = {"encoder": "MinGramRef pure-Python reference Viterbi (tie=%s), %s" % (tie, LABEL),
                         "path": model_path, "sha256": sha256_file(model_path)}

    def _encode_pretoken(self, s: str) -> List[int]:
        ids = self._cache.get(s)
        if ids is None:
            path = self.lat.best_hf(s) if self.tie == "hf" else self.lat.best_paper(s)
            ids = []
            for i, j, p in path:
                if p is None:
                    ids.extend(self.byte_id[b] for b in s[i:j].encode("utf-8"))
                else:
                    ids.append(self.id_of[p])
            if len(self._cache) < 2_000_000:
                self._cache[s] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        pos = 0
        for mm in self._sp_rx.finditer(text):
            for pt in pretokenize(text[pos:mm.start()]):
                out.extend(self._encode_pretoken(pt))
            out.append(self.special_tokens[mm.group()])
            pos = mm.end()
        for pt in pretokenize(text[pos:]):
            out.extend(self._encode_pretoken(pt))
        return out

    def encode_isolated(self, text: str) -> List[int]:
        return self.encode(text)

    def token_bytes(self, i: int) -> Optional[bytes]:
        if i < 0 or i >= len(self.pieces):
            return None
        k = self.kind[i]
        if k == "byte":
            return bytes([int(self.pieces[i][3:5], 16)])
        if k == "unk":
            return None
        return self.pieces[i].encode("utf-8")

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        buf = bytearray()
        for i in ids:
            if self.kind[i] == "byte":
                buf.append(int(self.pieces[i][3:5], 16))
                continue
            if buf:
                out.append(bytes(buf).decode("utf-8", errors="replace")); buf = bytearray()
            out.append(self.pieces[i] if self.kind[i] != "unk" else "")
        if buf:
            out.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(out)


def reference():
    """harness.py --custom mingram:reference"""
    return MinGramRef(DEFAULT_MODEL, "hf")
