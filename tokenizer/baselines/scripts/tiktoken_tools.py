"""tiktoken-format helpers WITHOUT tiktoken and WITHOUT executing any downloaded code.

- load_ranks(path)                       : parse a tiktoken rank file ("<base64 token> <rank>" per line)
- load_tekken(path)                      : parse Mistral tekken.json -> (ranks, pattern, n_special, special list)
- RefBPE(ranks, pattern, offset)         : pure-Python reference of tiktoken's algorithm
                                           (regex split with the `regex` module in V1 mode, then lowest-rank pair merging)
- to_tokenizers_json(ranks, pattern, specials) : build an equivalent HF `tokenizers` byte-level BPE
- kimi_pattern(path)                     : read the split regex out of tokenization_kimi.py by parsing its TEXT with ast
                                           (ast.literal_eval on the list literal; the file is never imported/executed)
"""
import ast
import base64
import json
import re

import regex


def load_ranks(path):
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    return ranks


def load_tekken(path):
    d = json.load(open(path, encoding="utf-8"))
    cfg = d["config"]
    n_special = cfg.get("default_num_special_tokens", len(d.get("special_tokens", [])))
    n_vocab = cfg.get("default_vocab_size", cfg.get("num_vocab_tokens")) - n_special
    ranks = {}
    for v in d["vocab"][:n_vocab]:
        ranks[base64.b64decode(v["token_bytes"])] = v["rank"]
    return ranks, cfg["pattern"], n_special, d.get("special_tokens", []), cfg


def kimi_pattern(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r'pat_str\s*=\s*"\|"\.join\(\s*(\[.*?\])\s*\)', src, re.S)
    parts = ast.literal_eval(m.group(1))  # literal list of raw strings only
    return "|".join(parts)


class RefBPE:
    """Reference tiktoken encoder (ordinary text, no special-token handling)."""

    def __init__(self, ranks, pattern, offset=0):
        self.ranks = ranks
        self.pat = regex.compile(pattern, flags=regex.V1)
        self.offset = offset
        self.cache = {}

    def _bpe(self, piece):
        r = self.ranks.get(piece)
        if r is not None:
            return [r]
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            best, idx = None, None
            for i in range(len(parts) - 1):
                rr = self.ranks.get(parts[i] + parts[i + 1])
                if rr is not None and (best is None or rr < best):
                    best, idx = rr, i
            if idx is None:
                break
            parts[idx:idx + 2] = [parts[idx] + parts[idx + 1]]
        return [self.ranks[p] for p in parts]

    def encode(self, text):
        out = []
        for m in self.pat.finditer(text):
            b = m.group(0).encode("utf-8")
            ids = self.cache.get(b)
            if ids is None:
                ids = self._bpe(b)
                self.cache[b] = ids
            out.extend(ids)
        return [i + self.offset for i in out]


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def to_tokenizers_json(ranks, pattern, specials):
    """specials: list of (id, content). Returns a tokenizers.Tokenizer."""
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers import AddedToken

    enc = bytes_to_unicode()

    def s(b):
        return "".join(enc[x] for x in b)

    vocab = {s(t): r for t, r in ranks.items()}
    merges = []
    for tok, rank in ranks.items():
        if len(tok) == 1:
            continue
        local = []
        for i in range(1, len(tok)):
            a, b = tok[:i], tok[i:]
            if a in ranks and b in ranks:
                local.append((a, b, rank))
        local.sort(key=lambda x: (ranks[x[0]], ranks[x[1]]))
        merges.extend(local)
    merges.sort(key=lambda x: x[2])
    merges = [(s(a), s(b)) for a, b, _ in merges]
    tk = Tokenizer(BPE(vocab=vocab, merges=merges, ignore_merges=True, byte_fallback=False, fuse_unk=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    specials = sorted(specials)
    # added tokens get ids in insertion order after the base vocab; ids must be contiguous for this to be exact
    expected = len(vocab)
    for i, content in specials:
        assert i == expected, f"special id {i} not contiguous (expected {expected})"
        expected += 1
    tk.add_special_tokens([AddedToken(c, special=True, normalized=False) for _, c in specials])
    return tk
