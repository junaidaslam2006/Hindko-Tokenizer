# -*- coding: utf-8 -*-
"""Build the tie-safe SentencePiece model and its exact HF tokenizer.json.

Score rule (documented in EQUIVALENCE.md): every NORMAL piece score s is replaced by
    s' = round_half_even(s * 2**K) / 2**K
i.e. moved to the nearest multiple of 2**-K. Pieces, ids, types and every other field of the model are unchanged.
Why: SentencePiece's Viterbi (unigram_model.cc, EncodeOptimized) accumulates path scores in float32; HF tokenizers
(models/unigram, encode_optimized) accumulates in float64. Both keep the FIRST candidate (smallest start) at a
lattice node unless a later one is strictly greater. With scores on the 2**-K grid every partial path score is a
multiple of 2**-K and is represented EXACTLY in float32 while |score| < 2**(24-K), and in float64 while
|score| < 2**(53-K). Both encoders then compare the same exact numbers and apply the same tie rule.
"""
import json
import os
from common import *

LOW_SCORE = -1048576.0      # -2**20: HF score of pieces SentencePiece keeps out of its lattice (on every grid)


def load_gauge(path, K):
    """{char: w} with every w rounded (half-even) to the 2**-K grid, so that the gauge term is exact."""
    g = json.load(open(path, encoding="utf-8"))
    q = float(2 ** K)
    return {chr(c): round(w * q) / q for c, w in zip(g["chars"], g["w"])}


def rounded_proto(K, gauge=None):
    """s' = round_half_even(s * 2**K) / 2**K  [+ sum_{c in piece} w_K(c) if a gauge is given] for NORMAL pieces."""
    m, T = load_proto(OLD_MODEL)
    q = float(2 ** K)
    changes = []
    for i, p in enumerate(m.pieces):
        if p.type == T.NORMAL:
            s = float(p.score)
            s2 = round(s * q) / q
            if s2 != s:
                changes.append(abs(s2 - s))
            if gauge is not None:
                s2 = s2 + sum(gauge[c] for c in p.piece)
            if abs(s2) >= 2.0 ** (24 - K) or float(__import__("numpy").float32(s2)) != s2:
                raise RuntimeError("score not exact in float32: %r" % s2)
            p.score = s2
    return m, T, changes


def write_model(K, out_dir, gauge=None):
    m, T, changes = rounded_proto(K, gauge)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sp.model")
    with open(path, "wb") as f:
        f.write(m.SerializeToString())
    # vocab listing like spm_train's .vocab (piece \t score)
    with open(os.path.join(out_dir, "sp.vocab"), "w", encoding="utf-8", newline="\n") as f:
        for p in m.pieces:
            f.write("%s\t%s\n" % (p.piece, repr(float(p.score)) if p.score != int(p.score) else str(int(p.score))))
    return path, {"n_changed": len(changes), "max_abs_change": max(changes) if changes else 0.0,
                  "mean_abs_change": sum(changes) / len(changes) if changes else 0.0}


def hf_json_dict(model_path):
    """tokenizer.json content for a SentencePiece Unigram model used through the PLAN 1.1 newline convention."""
    m, T = load_proto(model_path)
    pieces = list(m.pieces)
    if m.trainer_spec.model_type != 1:
        raise ValueError("Unigram only")
    added = []
    for i, p in enumerate(pieces):
        if p.type in (T.USER_DEFINED, T.CONTROL):
            added.append({"id": i, "content": p.piece, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": p.piece != NL})
    unk = [i for i, p in enumerate(pieces) if p.type == T.UNKNOWN]
    vocab = [[p.piece, float(p.score) if p.type == T.NORMAL else LOW_SCORE] for p in pieces]
    j = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        # each line (the text between '\n' added tokens) gets SentencePiece's whitespace escaping:
        # ' ' -> U+2581 and ONE U+2581 dummy prefix (Prepend skips empty lines, as SentencePiece encodes '' to []).
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"String": " "}, "content": SP_MARK},
            {"type": "Prepend", "prepend": SP_MARK}]},
        # no pre-tokenizer: the Viterbi runs over the whole line, as SentencePiece's does
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Replace", "pattern": {"String": NL + SP_MARK}, "content": NL},
            {"type": "Strip", "content": SP_MARK, "start": 1, "stop": 0},
            {"type": "Replace", "pattern": {"String": SP_MARK}, "content": " "}]},
        "model": {"type": "Unigram", "unk_id": unk[0], "vocab": vocab, "byte_fallback": True},
    }
    return j


def write_hf_json(model_path, out_path):
    from tokenizers import Tokenizer
    j = hf_json_dict(model_path)
    tok = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    s = tok.to_str(pretty=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    # every score must survive the JSON round trip bit-exactly
    back = json.loads(s)["model"]["vocab"]
    for (p, sc), (p2, sc2) in zip(j["model"]["vocab"], back):
        if p != p2 or sc != sc2:
            raise RuntimeError("score/piece changed in JSON round trip: %r %r %r %r" % (p, sc, p2, sc2))
    return out_path
