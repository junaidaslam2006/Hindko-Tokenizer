# -*- coding: utf-8 -*-
"""Build the release files of R2-A4-SPnat-D2-32k into F:\\Hindko\\_tokenizer\\release_build\\sp32k\\ :

  sp.model / sp.vocab       the canonical SentencePiece model with tie-safe scores (same pieces, ids, types, specs)
  gauge.json                the exact per-character score offset (rule, fitted values, provenance)
  tokenizer.json            HF tokenizers 0.22.2 export, equal to sp.model + the PLAN 1.1 newline convention
  tokenizer_config.json     transformers config (special tokens, ChatML chat_template, no automatic BOS/EOS)
  special_tokens_map.json
  build_manifest.json       hashes of every input and output

Score rule (EQUIVALENCE.md section 2), for every NORMAL piece p with old score s(p):
    s'(p) = round_half_even(s(p) * 2**14) / 2**14  +  sum_{c in p} w(c)
where w(c) (gauge.json) is itself a multiple of 2**-14. Non-NORMAL pieces keep their scores (SentencePiece keeps
them out of its lattice). Everything else in the model proto is byte-identical in content.

Run: PYTHONIOENCODING=utf-8 python build_release.py
"""
import hashlib
import json
import os
import sys

import numpy as np

TOK = r"F:\Hindko\_tokenizer"
CAND = os.path.join(TOK, r"candidates\round2\standard\tok\R2-A4-SPnat-D2-32k")
OLD_MODEL = os.path.join(CAND, "sp.model")
OUT = os.path.join(TOK, r"release_build\sp32k")
DATA = os.path.join(TOK, "data")
K = 14                               # score grid 2**-K
CHUNK = 8                            # gauge fit: least squares on sums of 8 consecutive best-path pieces
LOW_SCORE = -1048576.0               # -2**20: HF score of the pieces SentencePiece keeps out of its lattice
SP_MARK = chr(0x2581)
NL = chr(10)
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_proto(path):
    from sentencepiece import sentencepiece_model_pb2 as spb
    m = spb.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m, spb.ModelProto.SentencePiece


def fit_gauge():
    """w(c) for every character of a NORMAL piece: least squares over train_D2 (the tokenizer's own training view),
    minimising the sum over chunks of CHUNK consecutive pieces of the old SentencePiece best paths of
    (sum of piece scores + sum of w over the chunk's characters)^2. Only the magnitude of running Viterbi scores
    depends on w; no encoding does (EQUIVALENCE.md 2.2)."""
    import sentencepiece as spm
    m, T = load_proto(OLD_MODEL)
    pieces = list(m.pieces)
    normal = [i for i, p in enumerate(pieces) if p.type == T.NORMAL]
    chars = sorted({c for i in normal for c in pieces[i].piece})
    cidx = {c: k for k, c in enumerate(chars)}
    F = np.zeros((len(pieces), len(chars)))
    for i in normal:
        for c in pieces[i].piece:
            F[i, cidx[c]] += 1
    score = np.array([p.score for p in pieces], dtype=np.float64)
    is_normal = np.zeros(len(pieces), bool); is_normal[normal] = True
    sp = spm.SentencePieceProcessor(model_file=OLD_MODEL)
    path = os.path.join(DATA, "train_D2.jsonl")
    lines = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            lines += [x for x in json.loads(l)["text"].split(NL) if x]
    XtX = np.zeros((len(chars), len(chars))); Xty = np.zeros(len(chars))
    for e in sp.encode(lines):
        e = [i for i in e if is_normal[i]]
        for k in range(0, len(e), CHUNK):
            ch = e[k:k + CHUNK]
            fv = F[ch].sum(0)
            XtX += np.outer(fv, fv); Xty += fv * (-score[ch].sum())
    w = np.linalg.solve(XtX + 1e-6 * np.eye(len(chars)), Xty)
    q = float(2 ** K)
    wq = [round(float(x) * q) / q for x in w]
    return {"rule": "s'(p) = round_half_even(s(p)*2^%d)/2^%d + sum_{c in p} w(c), NORMAL pieces only" % (K, K),
            "grid": "2^-%d" % K,
            "fit": "least squares, train_D2 (%s), old SentencePiece best paths, chunks of %d consecutive NORMAL pieces, "
                   "objective sum_chunks (sum s + sum w)^2, ridge 1e-6; then each w rounded half-even to the grid"
                   % (sha256_file(path)[:12], CHUNK),
            "note": "w is an exact gauge: every candidate at one Viterbi lattice node covers the same characters, so "
                    "adding w changes every candidate by the same exact amount and no comparison changes. It only "
                    "keeps the running float32 path score of SentencePiece small. Un-shifted rounded log-prob of a "
                    "piece = score - sum_{c in piece} w(c).",
            "w": {"U+%04X" % ord(c): v for c, v in zip(chars, wq)}}


def build_model(gauge):
    m, T = load_proto(OLD_MODEL)
    q = float(2 ** K)
    w = {chr(int(k[2:], 16)): v for k, v in gauge["w"].items()}
    rounding = []
    for p in m.pieces:
        if p.type != T.NORMAL:
            continue
        s = float(p.score)
        r = round(s * q) / q
        rounding.append(r - s)
        s2 = r + sum(w[c] for c in p.piece)
        if abs(s2) >= 2.0 ** (24 - K) or float(np.float32(s2)) != s2 or (s2 * q) != int(s2 * q):
            raise RuntimeError("score not exact on the grid in float32: %r" % s2)
        p.score = s2
    path = os.path.join(OUT, "sp.model")
    with open(path, "wb") as f:
        f.write(m.SerializeToString())
    with open(os.path.join(OUT, "sp.vocab"), "w", encoding="utf-8", newline="\n") as f:
        for p in m.pieces:
            f.write("%s\t%s\n" % (p.piece, repr(float(p.score))))
    ra = np.abs(np.array(rounding))
    return path, {"normal_pieces": len(rounding), "rounded_changed": int((ra > 0).sum()),
                  "max_abs_rounding": float(ra.max()), "mean_abs_rounding": float(ra.mean())}


def specials_of(m, T):
    return [(i, p.piece) for i, p in enumerate(m.pieces) if p.type in (T.USER_DEFINED, T.CONTROL) and p.piece != NL]


def build_tokenizer_json(model_path):
    from tokenizers import Tokenizer
    m, T = load_proto(model_path)
    pieces = list(m.pieces)
    added = []
    for i, p in enumerate(pieces):
        if p.type in (T.USER_DEFINED, T.CONTROL):
            added.append({"id": i, "content": p.piece, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": p.piece != NL})
    unk = [i for i, p in enumerate(pieces) if p.type == T.UNKNOWN]
    vocab = [[p.piece, float(p.score) if p.type == T.NORMAL else LOW_SCORE] for p in pieces]
    decoders = [{"type": "ByteFallback"}, {"type": "Fuse"},
                {"type": "Replace", "pattern": {"String": NL + SP_MARK}, "content": NL}]
    # a text segment that follows a special token got its own dummy prefix (it was split off before the
    # normalizer); remove exactly that prefix again
    for _, s in specials_of(m, T):
        decoders.append({"type": "Replace", "pattern": {"String": s + SP_MARK}, "content": s})
    decoders += [{"type": "Strip", "content": SP_MARK, "start": 1, "stop": 0},
                 {"type": "Replace", "pattern": {"String": SP_MARK}, "content": " "}]
    j = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
         "normalizer": {"type": "Sequence", "normalizers": [
             {"type": "Replace", "pattern": {"String": " "}, "content": SP_MARK},
             {"type": "Prepend", "prepend": SP_MARK}]},
         "pre_tokenizer": None, "post_processor": None,
         "decoder": {"type": "Sequence", "decoders": decoders},
         "model": {"type": "Unigram", "unk_id": unk[0], "vocab": vocab, "byte_fallback": True}}
    tok = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    s = tok.to_str(pretty=True)
    back = json.loads(s)
    if back["model"]["vocab"] != vocab:
        raise RuntimeError("vocab/scores changed in the tokenizers JSON round trip")
    if back["model"].get("byte_fallback") is not True or back["model"]["unk_id"] != unk[0]:
        raise RuntimeError("model fields changed in the round trip")
    path = os.path.join(OUT, "tokenizer.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    return path, added


def build_configs(added):
    by_id = {a["id"]: a for a in added}
    named = {"bos_token": "<|bos|>", "eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}
    extra = [a["content"] for a in added if a["special"] and a["content"] not in named.values()]
    atd = {str(a["id"]): {"content": a["content"], "lstrip": False, "normalized": False, "rstrip": False,
                          "single_word": False, "special": a["special"]} for a in added}
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast",
           "added_tokens_decoder": atd,
           **named,
           "unk_token": None,
           "additional_special_tokens": extra,
           "add_bos_token": False, "add_eos_token": False,
           "clean_up_tokenization_spaces": False, "split_special_tokens": False,
           "model_max_length": 1000000000000000019884624838656,
           "padding_side": "right",
           "chat_template": CHAT_TEMPLATE}
    stm = {**named, "additional_special_tokens": extra}
    for name, obj in (("tokenizer_config.json", cfg), ("special_tokens_map.json", stm)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return by_id


def main():
    import sentencepiece, tokenizers
    os.makedirs(OUT, exist_ok=True)
    gauge = fit_gauge()
    with open(os.path.join(OUT, "gauge.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(gauge, f, ensure_ascii=False, indent=1)
    mpath, rinfo = build_model(gauge)
    # the new model must keep every piece string, type and id and every non-score field
    mo, T = load_proto(OLD_MODEL); mn, _ = load_proto(mpath)
    assert len(mo.pieces) == len(mn.pieces) == 32768
    for a, b in zip(mo.pieces, mn.pieces):
        assert a.piece == b.piece and a.type == b.type
        if a.type != T.NORMAL:
            assert a.score == b.score
    for fld in ("trainer_spec", "normalizer_spec", "denormalizer_spec"):
        assert getattr(mo, fld).SerializeToString() == getattr(mn, fld).SerializeToString(), fld
    jpath, added = build_tokenizer_json(mpath)
    build_configs(added)
    sc = np.array([p.score for p in mn.pieces if p.type == T.NORMAL])
    man = {"candidate": "R2-A4-SPnat-D2-32k", "K": K, "grid": 2.0 ** -K,
           "float32_exact_bound": 2.0 ** (24 - K), "low_score_hf": LOW_SCORE,
           "rounding": rinfo, "new_normal_score_range": [float(sc.min()), float(sc.max())],
           "sentencepiece_unk_score": float(sc.min()) - 10.0,
           "versions": {"python": sys.version.split()[0], "sentencepiece": sentencepiece.__version__,
                        "tokenizers": tokenizers.__version__, "numpy": np.__version__},
           "inputs": {"old sp.model": {"path": OLD_MODEL, "sha256": sha256_file(OLD_MODEL)},
                      "old tokenizer.json (export with the tie mismatch)": {
                          "path": os.path.join(CAND, "tokenizer.json"),
                          "sha256": sha256_file(os.path.join(CAND, "tokenizer.json"))},
                      "train_D2.jsonl (gauge fit)": sha256_file(os.path.join(DATA, "train_D2.jsonl"))},
           "outputs": {n: sha256_file(os.path.join(OUT, n)) for n in
                       ("sp.model", "sp.vocab", "gauge.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json")},
           "script": {"build_release.py": sha256_file(os.path.abspath(__file__))}}
    with open(os.path.join(OUT, "build_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
