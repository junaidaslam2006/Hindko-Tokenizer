# -*- coding: utf-8 -*-
"""Uniform tokenizer adapters for the Hindko evaluation harness (eval/harness.py).

Every adapter exposes
    name, kind, vocab_size, id_upper (max id + 1)
    encode(text) -> list[int]                   native encoder, no BOS/EOS/CLS added
    decode(ids) -> str                          native decoder, special tokens not skipped
    encode_offsets(text) -> (ids, starts, ends) character offsets into `text` (end exclusive)
    encode_batch(texts) -> list[list[int]]
    token_bytes(id) -> bytes | None             the bytes the token contributes to the text
                                                (None when the adapter cannot tell)
    notext (np.bool_ array over ids)            token contributes no non-whitespace text
                                                (lone '▁', whitespace, control pieces)
    special_tokens {str: id}, special_ids, unk_ids, base_ids (256 byte tokens / <0xNN> pieces),
    byte_level, char_level, model_type, merges (BPE: list of (a_id, b_id, parent_id)) or None
    encode_isolated(text)                       encoding of a bare string with no dummy prefix (G2)
    identity() -> dict                          what the summary records about the tokenizer

Adapters:
    HFAdapter           a `tokenizers` tokenizer.json (or Tokenizer object)
    SPAdapter           a SentencePiece .model; newline_wrapper=True (default, PLAN.md 1.1) encodes
                        each line separately and joins lines with the dedicated '\\n' piece
    BytesAdapter        raw UTF-8 bytes (ByT5 id convention)
    TransformersAdapter a transformers tokenizer (baselines with loader 'auto')
    CustomAdapter       any Python object with encode/decode (see CustomAdapter docstring)
    from_baseline(name) any external baseline of baselines/manifest.json, loaded through
                        baselines/load_baselines.py (the same encode/decode objects)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
BASELINES_DIR = os.path.join(TOK, "baselines")
SP_MARK = "\u2581"
BYTE_PIECE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
LEAKY_BASELINES = {"hindko-probe-bpe32k"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte <-> printable-unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}


def byte_level_str_to_bytes(s: str) -> Optional[bytes]:
    try:
        return bytes(U2B[c] for c in s)
    except KeyError:
        return None


def is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def char_byte_table(text: str) -> np.ndarray:
    """cum[i] = UTF-8 byte offset of character i (len(text)+1 entries)."""
    lens = np.fromiter((len(c.encode("utf-8")) for c in text), dtype=np.int64, count=len(text))
    out = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum(lens, out=out[1:])
    return out


def byte_to_char_table(text: str) -> np.ndarray:
    """b2c[b] = index of the character containing byte b (len(bytes)+1 entries; last = len(text))."""
    cum = char_byte_table(text)
    nb = int(cum[-1])
    b2c = np.empty(nb + 1, dtype=np.int64)
    if len(text):
        b2c[:nb] = np.repeat(np.arange(len(text), dtype=np.int64), np.diff(cum))
    b2c[nb] = len(text)
    return b2c


class Adapter:
    kind = "abstract"
    byte_level = False
    char_level = True
    model_type = "?"
    merges = None
    notes: List[str]

    def __init__(self, name: str):
        self.name = name
        self.notes = []
        self._notext = None
        self._tb_cache: Dict[int, Optional[bytes]] = {}

    # --- required
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def encode_offsets(self, text: str):
        raise NotImplementedError

    # --- defaults
    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def encode_isolated(self, text: str) -> List[int]:
        return self.encode(text)

    def _token_bytes(self, i: int) -> Optional[bytes]:
        return None

    def token_bytes(self, i: int) -> Optional[bytes]:
        if i not in self._tb_cache:
            self._tb_cache[i] = self._token_bytes(i)
        return self._tb_cache[i]

    special_tokens: Dict[str, int] = {}
    unk_ids: frozenset = frozenset()
    base_ids: frozenset = frozenset()
    vocab_size = 0
    id_upper = 0

    @property
    def special_ids(self):
        return frozenset(self.special_tokens.values())

    @property
    def notext(self) -> np.ndarray:
        """True for ids whose bytes are empty or whitespace only (they never belong to a word)."""
        if self._notext is None:
            a = np.zeros(self.id_upper, dtype=bool)
            for i in range(self.id_upper):
                b = self.token_bytes(i)
                if b is not None and (len(b) == 0 or b.isspace()):
                    a[i] = True
            self._notext = a
        return self._notext

    def byte_spans(self, text: str, ids: Sequence[int]) -> Optional[np.ndarray]:
        """(n, 2) byte spans of the tokens in `text` from token_bytes, if the token bytes spell the text exactly,
        allowing a dummy-prefix space (SentencePiece '▁') that is absent from the text at the start of the text
        or of a line. None otherwise (lossy tokenizer, unknown token bytes)."""
        bs = [self.token_bytes(i) for i in ids]
        if any(b is None for b in bs):
            return None
        tb = text.encode("utf-8")
        if b"".join(bs) == tb:
            lens = np.fromiter((len(b) for b in bs), dtype=np.int64, count=len(bs))
            ends = np.cumsum(lens)
            return np.stack([ends - lens, ends], axis=1)
        out = np.zeros((len(bs), 2), dtype=np.int64)
        pos = 0
        n = len(tb)
        for k, b in enumerate(bs):
            L = len(b)
            if tb.startswith(b, pos):
                out[k] = (pos, pos + L); pos += L
            elif b[:1] == b" " and (pos == 0 or tb[pos - 1] == 10) and tb.startswith(b[1:], pos):
                out[k] = (pos, pos + L - 1); pos += L - 1
            else:
                return None
        return out if pos == n else None

    def identity(self) -> dict:
        return {"name": self.name, "kind": self.kind, "model_type": self.model_type, "vocab_size": self.vocab_size,
                "byte_level": self.byte_level, "char_level": self.char_level, "notes": self.notes}


# ----------------------------------------------------------------------------------------------- HF
class HFAdapter(Adapter):
    kind = "hf_tokenizer_json"

    def __init__(self, path: Optional[str] = None, tokenizer=None, name: Optional[str] = None):
        from tokenizers import Tokenizer
        super().__init__(name or (os.path.basename(path) if path else "hf"))
        self.path = path
        self.tk = tokenizer if tokenizer is not None else Tokenizer.from_file(path)
        self.tk.no_truncation()
        self.tk.no_padding()
        self.j = json.loads(self.tk.to_str())
        m = self.j["model"]
        self.model_type = m.get("type", "?")
        vocab = self.tk.get_vocab(with_added_tokens=True)
        self.vocab_size = len(vocab)
        self.id_upper = max(vocab.values()) + 1
        self.id_to_tok = {i: s for s, i in vocab.items()}
        pt = json.dumps(self.j.get("pre_tokenizer"), ensure_ascii=False)
        dec = json.dumps(self.j.get("decoder"), ensure_ascii=False)
        nm = json.dumps(self.j.get("normalizer"), ensure_ascii=False)
        self.byte_level = '"ByteLevel"' in pt or '"ByteLevel"' in dec
        self.char_level = not self.byte_level
        self.byte_fallback = bool(m.get("byte_fallback"))
        self.sp_style = SP_MARK in dec or SP_MARK in pt or SP_MARK in nm or '"Metaspace"' in pt
        self.added = {a["content"]: a["id"] for a in self.j.get("added_tokens", [])}
        self.special_tokens = {a["content"]: a["id"] for a in self.j.get("added_tokens", []) if a.get("special")}
        unk = set()
        if self.model_type == "Unigram" and m.get("unk_id") is not None:
            unk.add(int(m["unk_id"]))
        elif m.get("unk_token") is not None and self.tk.token_to_id(m["unk_token"]) is not None:
            unk.add(self.tk.token_to_id(m["unk_token"]))
        self.unk_ids = frozenset(unk)
        base = set()
        if self.byte_level:
            for b, u in B2U.items():
                if u in vocab:
                    base.add(vocab[u])
        if self.byte_fallback:
            for s, i in vocab.items():
                if BYTE_PIECE_RE.match(s):
                    base.add(i)
        self.base_ids = frozenset(base)
        if self.model_type == "BPE":
            mg = []
            for x in m.get("merges", []):
                a, b = (x.split(" ", 1) if isinstance(x, str) else x)
                p = vocab.get(a + b)
                if a in vocab and b in vocab and p is not None:
                    mg.append((vocab[a], vocab[b], p))
            self.merges = mg
        self._iso = None

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tk.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids):
        return self.tk.decode(list(ids), skip_special_tokens=False)

    def encode_offsets(self, text):
        e = self.tk.encode(text, add_special_tokens=False)
        off = np.asarray(e.offsets, dtype=np.int64).reshape(-1, 2)
        return e.ids, off[:, 0], off[:, 1]

    def _token_bytes(self, i):
        s = self.id_to_tok.get(i)
        if s is None:
            return None
        if s in self.added:
            return s.encode("utf-8")
        if self.byte_fallback or self.sp_style:
            mm = BYTE_PIECE_RE.match(s)
            if mm and (self.byte_fallback or i in self.base_ids):
                return bytes([int(mm.group(1), 16)])
        if self.byte_level:
            b = byte_level_str_to_bytes(s)
            return b if b is not None else s.encode("utf-8")
        if self.model_type == "WordPiece":
            return (s[2:] if s.startswith("##") else s).encode("utf-8")
        if self.sp_style:
            return s.replace(SP_MARK, " ").encode("utf-8")
        return s.encode("utf-8")

    def encode_isolated(self, text):
        """Encoding of a bare string: a Metaspace/Prepend dummy prefix is switched off."""
        if self._iso is None:
            j = json.loads(self.tk.to_str())
            changed = False

            def fix(node):
                nonlocal changed
                if isinstance(node, dict):
                    if node.get("type") == "Metaspace" and node.get("prepend_scheme", "always") != "never":
                        node["prepend_scheme"] = "never"; changed = True
                    if node.get("type") == "Sequence":
                        for key in ("normalizers", "pretokenizers"):
                            if key in node:
                                keep = [x for x in node[key] if not (isinstance(x, dict) and x.get("type") == "Prepend")]
                                if len(keep) != len(node[key]):
                                    node[key] = keep; changed = True
                    for v in node.values():
                        fix(v)
                elif isinstance(node, list):
                    for v in node:
                        fix(v)
            for k in ("normalizer", "pre_tokenizer"):
                fix(j.get(k))
            if j.get("normalizer", {}) and isinstance(j["normalizer"], dict) and j["normalizer"].get("type") == "Prepend":
                j["normalizer"] = None; changed = True
            from tokenizers import Tokenizer
            self._iso = Tokenizer.from_str(json.dumps(j)) if changed else self.tk
        return self._iso.encode(text, add_special_tokens=False).ids

    def identity(self):
        d = super().identity()
        d.update({"path": self.path, "sha256": sha256_file(self.path) if self.path and os.path.exists(self.path) else
                  hashlib.sha256(self.tk.to_str().encode()).hexdigest(),
                  "byte_fallback": self.byte_fallback, "normalizer": self.j.get("normalizer") is not None,
                  "n_added_tokens": len(self.added), "n_special_tokens": len(self.special_tokens),
                  "encoder": "tokenizers %s native (Tokenizer.encode, add_special_tokens=False)" % _ver("tokenizers")})
        return d


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return "?"


# ------------------------------------------------------------------------------------ SentencePiece
class SPAdapter(Adapter):
    kind = "sentencepiece"

    def __init__(self, path: Optional[str] = None, sp=None, name: Optional[str] = None, newline_wrapper: bool = True,
                 newline_piece: str = "\n"):
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as spb
        super().__init__(name or (os.path.basename(path) if path else "spm"))
        self.path = path
        self.sp = sp if sp is not None else spm.SentencePieceProcessor(model_file=path)
        self.proto = spb.ModelProto()
        self.proto.ParseFromString(self.sp.serialized_model_proto())
        self.model_type = {1: "Unigram", 2: "BPE", 3: "Word", 4: "Char"}.get(self.proto.trainer_spec.model_type, "?")
        self.vocab_size = self.id_upper = self.sp.get_piece_size()
        self.newline_wrapper = newline_wrapper
        self.nl_id = None
        if newline_wrapper:
            nid = self.sp.piece_to_id(newline_piece)
            if nid == self.sp.unk_id() or self.sp.id_to_piece(nid) != newline_piece:
                raise ValueError("newline_wrapper needs a '\\n' user-defined piece in %s" % self.name)
            self.nl_id = nid
        P = spb.ModelProto.SentencePiece
        self.user_defined = {p.piece: i for i, p in enumerate(self.proto.pieces) if p.type == P.USER_DEFINED}
        self.control = {p.piece: i for i, p in enumerate(self.proto.pieces) if p.type == P.CONTROL}
        self.special_tokens = {s: i for s, i in {**self.control, **self.user_defined}.items() if s != newline_piece}
        self.added = dict(self.user_defined)          # user-defined pieces (incl. '\n') are not 'learned' (R1)
        self.unk_ids = frozenset({self.sp.unk_id()}) if self.sp.unk_id() >= 0 else frozenset()
        self.base_ids = frozenset(i for i in range(self.vocab_size) if self.sp.is_byte(i))
        self.byte_level = False
        self.char_level = True
        self.normalizer_name = self.proto.normalizer_spec.name
        self.add_dummy_prefix = self.proto.normalizer_spec.add_dummy_prefix
        iso = spb.ModelProto()
        iso.ParseFromString(self.sp.serialized_model_proto())
        iso.normalizer_spec.add_dummy_prefix = False
        self.sp_iso = spm.SentencePieceProcessor(model_proto=iso.SerializeToString())

    def _lines(self, text):
        return text.split("\n")

    def encode(self, text):
        if not self.newline_wrapper:
            return self.sp.encode(text)
        out = []
        for k, ids in enumerate(self.sp.encode(self._lines(text))):
            if k:
                out.append(self.nl_id)
            out.extend(ids)
        return out

    def encode_batch(self, texts):
        if not self.newline_wrapper:
            return self.sp.encode(list(texts))
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        ids = list(ids)
        if not self.newline_wrapper:
            return self.sp.decode(ids)
        segs, cur = [], []
        for i in ids:
            if i == self.nl_id:
                segs.append(cur); cur = []
            else:
                cur.append(i)
        segs.append(cur)
        return "\n".join(self.sp.decode(s) if s else "" for s in segs)

    def _offsets_one(self, text, base):
        # sentencepiece's Python wrapper reports begin/end as CHARACTER offsets of the input str
        # (verified 2026-09-26 on sentencepiece 0.2.1: text[begin:end] == surface for identity models)
        p = self.sp.encode(text, out_type="immutable_proto")
        ids = [x.id for x in p.pieces]
        st = [base + x.begin for x in p.pieces]
        en = [base + x.end for x in p.pieces]
        return ids, st, en

    def encode_offsets(self, text):
        if not self.newline_wrapper:
            ids, st, en = self._offsets_one(text, 0)
        else:
            ids, st, en = [], [], []
            pos = 0
            for k, line in enumerate(self._lines(text)):
                if k:
                    ids.append(self.nl_id); st.append(pos - 1); en.append(pos)
                a, b, c = self._offsets_one(line, pos)
                ids += a; st += b; en += c
                pos += len(line) + 1
        return ids, np.asarray(st, dtype=np.int64), np.asarray(en, dtype=np.int64)

    def _token_bytes(self, i):
        if i < 0 or i >= self.vocab_size:
            return None
        piece = self.sp.id_to_piece(i)
        if self.sp.is_byte(i):
            return bytes([int(piece[3:5], 16)])
        if self.sp.is_control(i):
            return b""
        if self.sp.is_unknown(i):
            return None
        if piece in self.user_defined:
            return piece.encode("utf-8")
        return piece.replace(SP_MARK, " ").encode("utf-8")

    def encode_isolated(self, text):
        return self.sp_iso.encode(text)

    def identity(self):
        d = super().identity()
        d.update({"path": self.path, "sha256": sha256_file(self.path) if self.path and os.path.exists(self.path) else
                  hashlib.sha256(self.sp.serialized_model_proto()).hexdigest(),
                  "newline_wrapper": self.newline_wrapper, "normalization_rule_name": self.normalizer_name,
                  "add_dummy_prefix": self.add_dummy_prefix,
                  "remove_extra_whitespaces": self.proto.normalizer_spec.remove_extra_whitespaces,
                  "byte_fallback": self.proto.trainer_spec.byte_fallback,
                  "n_user_defined": len(self.user_defined), "n_control": len(self.control),
                  "encoder": "sentencepiece %s native%s" % (_ver("sentencepiece"),
                                                            " + newline wrapper (PLAN 1.1: lines encoded separately, "
                                                            "joined by the '\\n' piece)" if self.newline_wrapper else "")})
        return d


# ------------------------------------------------------------------------------------------- bytes
class BytesAdapter(Adapter):
    kind = "bytes"
    model_type = "bytes"
    byte_level = True
    char_level = False

    def __init__(self, name="bytes", offset=3, vocab_size=384):
        super().__init__(name)
        self.offset = offset
        self.vocab_size = self.id_upper = vocab_size
        self.base_ids = frozenset(range(offset, offset + 256))

    def encode(self, text):
        return [b + self.offset for b in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(i - self.offset for i in ids if self.offset <= i < self.offset + 256).decode("utf-8", errors="replace")

    def encode_offsets(self, text):
        cum = char_byte_table(text)
        lens = np.diff(cum)
        st = np.repeat(np.arange(len(text), dtype=np.int64), lens)
        return self.encode(text), st, st + 1

    def _token_bytes(self, i):
        return bytes([i - self.offset]) if self.offset <= i < self.offset + 256 else b""


# ------------------------------------------------------------------------------------ transformers
class TransformersAdapter(Adapter):
    """A transformers tokenizer object (baselines with loader 'auto'). encode/decode are the baseline's own
    functions; offsets come from the same object's return_offsets_mapping (fast tokenizers). encode_offsets
    checks that its ids equal encode(text) and counts mismatches in self.offset_id_mismatch."""
    kind = "transformers"

    def __init__(self, baseline):
        super().__init__(baseline.name)
        self.b = baseline
        self.tok = baseline.raw
        self.vocab_size = baseline.vocab_size
        vocab = self.tok.get_vocab()
        self.id_upper = max(max(vocab.values()) + 1, self.vocab_size)
        self.id_to_tok = {i: s for s, i in vocab.items()}
        self.unk_ids = frozenset(baseline.unk_ids)
        self.special_tokens = {s: self.tok.convert_tokens_to_ids(s) for s in getattr(self.tok, "all_special_tokens", [])}
        cls = type(self.tok).__name__
        self.byte_level = "Roberta" in cls or "GPT2" in cls
        self.char_level = not self.byte_level
        self.model_type = "WordPiece" if "Bert" in cls else ("BPE" if self.byte_level else "?")
        self.is_fast = bool(getattr(self.tok, "is_fast", False))
        self.offset_id_mismatch = 0
        if self.byte_level:
            self.base_ids = frozenset(vocab[u] for u in B2U.values() if u in vocab)

    def encode(self, text):
        return self.b.encode(text)

    def decode(self, ids):
        return self.b.decode(ids)

    def encode_offsets(self, text):
        ids = self.b.encode(text)
        if self.is_fast:
            out = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            if list(out["input_ids"]) == list(ids):
                off = np.asarray(out["offset_mapping"], dtype=np.int64).reshape(-1, 2)
                return ids, off[:, 0], off[:, 1]
            self.offset_id_mismatch += 1
        st, en = align_token_strings(text, [self.token_bytes(i) for i in ids], self.unk_ids, ids)
        return ids, st, en

    def _token_bytes(self, i):
        s = self.id_to_tok.get(i)
        if s is None:
            return None
        if s in self.special_tokens:
            return s.encode("utf-8")
        if self.byte_level:
            b = byte_level_str_to_bytes(s)
            return b if b is not None else s.encode("utf-8")
        if self.model_type == "WordPiece":
            return (s[2:] if s.startswith("##") else s).encode("utf-8")
        return s.replace(SP_MARK, " ").encode("utf-8")

    def identity(self):
        d = super().identity()
        d.update({"class": type(self.tok).__name__, "is_fast": self.is_fast,
                  "offset_id_mismatch_docs": self.offset_id_mismatch,
                  "encoder": "transformers %s %s (load_baselines 'auto')" % (_ver("transformers"), type(self.tok).__name__)})
        return d


def align_token_strings(text, tbytes, unk_ids, ids):
    """Fallback character offsets: greedily locate each token's text in `text` (skipping whitespace).
    Used only when an adapter has no native offsets."""
    st, en = [], []
    pos = 0
    L = len(text)
    for b, i in zip(tbytes, ids):
        s = (b or b"").decode("utf-8", errors="ignore").strip()
        while pos < L and text[pos].isspace():
            pos += 1
        if i in unk_ids or not s:
            j = pos
            if i in unk_ids:
                while j < L and not text[j].isspace():
                    j += 1
            st.append(pos); en.append(j); pos = j
            continue
        k = text.find(s, pos, pos + len(s) + 8)
        if k < 0:
            st.append(pos); en.append(pos)
            continue
        st.append(k); en.append(k + len(s)); pos = k + len(s)
    return np.asarray(st, dtype=np.int64), np.asarray(en, dtype=np.int64)


# -------------------------------------------------------------------------------------------- custom
class CustomAdapter(Adapter):
    """Wrap any Python encoder object.

    Required on `obj`:  encode(text)->list[int], decode(ids)->str, vocab_size (int)
    Recommended:        token_bytes(id)->bytes   (bytes the token contributes; gives offsets, fertility,
                                                  robustness, morphology, G2, G5 and R2)
    Optional:           encode_offsets(text)->(ids, starts, ends)   character offsets (overrides token_bytes)
                        special_tokens {str: id}, unk_ids, base_ids (byte tokens exempt from G2 / not
                        'learned' in R1), merges [(a_id, b_id, parent_id)], encode_isolated(text),
                        byte_level (bool), char_level (bool), model_type (str), identity (dict)
    """
    kind = "custom"

    def __init__(self, obj, name: Optional[str] = None):
        super().__init__(name or getattr(obj, "name", type(obj).__name__))
        self.obj = obj
        self.vocab_size = int(obj.vocab_size)
        self.id_upper = int(getattr(obj, "id_upper", self.vocab_size))
        self.special_tokens = dict(getattr(obj, "special_tokens", {}) or {})
        self.unk_ids = frozenset(getattr(obj, "unk_ids", ()) or ())
        self.base_ids = frozenset(getattr(obj, "base_ids", ()) or ())
        self.merges = getattr(obj, "merges", None)
        self.byte_level = bool(getattr(obj, "byte_level", False))
        self.char_level = bool(getattr(obj, "char_level", not self.byte_level))
        self.model_type = getattr(obj, "model_type", "custom")

    def encode(self, text):
        return list(self.obj.encode(text))

    def decode(self, ids):
        return self.obj.decode(list(ids))

    def encode_isolated(self, text):
        f = getattr(self.obj, "encode_isolated", None)
        return list(f(text)) if f else self.encode(text)

    def _token_bytes(self, i):
        f = getattr(self.obj, "token_bytes", None)
        return f(i) if f else None

    def encode_offsets(self, text):
        f = getattr(self.obj, "encode_offsets", None)
        if f:
            ids, st, en = f(text)
            return list(ids), np.asarray(st, dtype=np.int64), np.asarray(en, dtype=np.int64)
        ids = self.encode(text)
        spans = self.byte_spans(text, ids)
        if spans is None:
            raise ValueError("%s: custom encoder needs token_bytes() spelling the text, or encode_offsets()" % self.name)
        b2c = byte_to_char_table(text)
        nb = len(b2c) - 1
        st = b2c[np.clip(spans[:, 0], 0, nb)]
        # end char = the char containing the last byte, + 1 (a partial-char token still covers its char)
        last = np.clip(spans[:, 1] - 1, 0, max(nb - 1, 0))
        en = np.where(spans[:, 1] > spans[:, 0], b2c[last] + 1, st)
        return ids, st, en

    def identity(self):
        d = super().identity()
        d.update(getattr(self.obj, "identity", {}) or {})
        d["encoder"] = d.get("encoder", "custom Python encoder %s" % type(self.obj).__name__)
        return d


# ------------------------------------------------------------------------------------------ baselines
def baseline_spec(name: str) -> dict:
    sys.path.insert(0, BASELINES_DIR)
    from load_baselines import manifest
    for e in manifest()["baselines"]:
        if e["name"] == name:
            return e
    raise KeyError(name)


def from_baseline(name: str) -> Adapter:
    """An adapter around the SAME encode/decode objects that baselines/load_baselines.py builds."""
    sys.path.insert(0, BASELINES_DIR)
    os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from load_baselines import load_baseline
    spec = baseline_spec(name)
    b = load_baseline(name)
    lp = spec.get("load_path")
    path = os.path.join(BASELINES_DIR, lp) if lp and not os.path.isabs(lp) else lp
    if spec["loader"] in ("hf_json", "tiktoken_ranks"):
        a = HFAdapter(path=path, tokenizer=b.raw, name=name)
    elif spec["loader"] in ("spm", "local_spm"):
        a = SPAdapter(path=path, sp=b.raw, name=name, newline_wrapper=False)
    elif spec["loader"] == "auto":
        a = TransformersAdapter(b)
        a.path = path
    elif spec["loader"] == "bytes":
        a = BytesAdapter(name=name)
        a.path = None
    else:
        raise ValueError(spec["loader"])
    a.kind = "baseline:" + spec["loader"]
    # the baseline's own unk ids (load_baselines) are authoritative for UNK counting
    a.unk_ids = frozenset(b.unk_ids) | a.unk_ids
    a.baseline = {k: spec.get(k) for k in ("name", "tier", "provider", "family", "repo", "revision", "official",
                                           "mirror_kind", "loader", "algorithm", "byte_fallback", "vocab_size",
                                           "vocab_sha256", "lossless", "newline_handling", "recommended_for_benchmark",
                                           "identical_to", "same_encodings_as", "license", "year", "models")}
    a.baseline["leaky"] = name in LEAKY_BASELINES
    a.baseline["load_baselines_vocab_size"] = b.vocab_size
    a._baseline_obj = b
    return a
