# -*- coding: utf-8 -*-
"""PLAN.md Stage 1 (intrinsic sweep): trainers, SentencePiece->HF export, gate extensions, G2 remedy.

Candidates (PLAN 2.1, 3 Stage 1), every one with the 64-token special block of PLAN 2.1 at ids 0..63,
no normalizer (HF normalizer=None; SentencePiece normalization_rule_name=identity), total vocabulary
= the nominal size (8k = 8,192 ... 48k = 49,152, all multiples of 64):
  A1  byte-level BPE    HF BpeTrainer; pre-tokenizer Split(P, isolated) + ByteLevel(use_regex=False);
                        P in {P1, P0, P1r3, Pm}; data D1 (P1 also D2, D3)
  A2  char-level BPE    HF BpeTrainer, BPE(byte_fallback=True); Split(P1); the 256 <0xNN> pieces are
                        inserted after the special block (gate_audit.py part_a2 construction)
  A3  SentencePiece BPE      model_type=bpe, byte_fallback, SPnat (split_by_whitespace, split_digits)
  A4  SentencePiece Unigram  model_type=unigram, byte_fallback, SPnat
  A5  HF Unigram        HF UnigramTrainer after Split(P1); '<unk>' (id 64) and the 256 <0xNN> pieces
                        (ids 65..320) inserted after the specials, byte_fallback=True (UNK is never emitted)
  A8  SentencePiece Unigram, split_by_whitespace=false ("superword"; PLAN 2.1)
SentencePiece models are trained on data/train_D1.lines.txt (non-empty lines) and encoded through the
harness newline wrapper (PLAN 1.1): '\\n' is a user-defined piece (id 64), UNK is id 65, the 256 byte
pieces are ids 66..321. Their HF tokenizer.json is an EXPORT (sp_to_hf_json), checked for encoder
equivalence against the native SentencePiece wrapper on dev; the native encoder is the candidate's identity.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import sys
import time

TOK = r"F:\Hindko\_tokenizer"
HERE = os.path.join(TOK, "candidates", "standard")
EVAL = os.path.join(TOK, "eval")
DATA = os.path.join(TOK, "data")
TOKDIR = os.path.join(HERE, "tok")
RESDIR = os.path.join(HERE, "results")
THREADS = int(os.environ.get("STD_THREADS", "2"))
SP_THREADS = 2          # fixed: SentencePiece records num_threads in the model file (G4 compares files)
os.environ.setdefault("RAYON_NUM_THREADS", str(THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if THREADS > 1 else "false")
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, EVAL)

import harness as H  # noqa: E402
import adapters as A  # noqa: E402

NL = chr(10)
SP_MARK = chr(0x2581)
SPECIALS = list(H.SPECIAL_TOKENS)
assert len(SPECIALS) == 64
SIZES = {"8k": 8192, "12k": 12288, "16k": 16384, "24k": 24576, "32k": 32768, "48k": 49152}
PRETOK = dict(H.PRETOKENIZERS)
# PLAN 2.1 Pm: marks as separate pretokens; the elided part ('...') is P1's remainder
PRETOK["Pm"] = r" ?[\p{L}\x{200C}\x{200D}]+|\p{M}+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
BPE_MIN_FREQUENCY = 2      # pilot value (PLAN 4.3: min_frequency 2 vs 20 gives identical merges)
SP_UNK_ID = 65
SP_NL_ID = 64
LOW_SCORE = -1.0e9         # HF export: pieces SentencePiece never puts into its lattice

ALGOS = {
    "A1": {"desc": "byte-level BPE", "impl": "HF tokenizers BpeTrainer, Split(P)+ByteLevel(use_regex=False)",
           "encoder": "HF native", "family": "BPE", "level": "byte"},
    "A2": {"desc": "char-level BPE + byte_fallback", "impl": "HF tokenizers BpeTrainer, BPE(byte_fallback=True), Split(P1)",
           "encoder": "HF native", "family": "BPE", "level": "char"},
    "A3": {"desc": "SentencePiece BPE", "impl": "sentencepiece model_type=bpe, byte_fallback, SPnat",
           "encoder": "SentencePiece native + newline wrapper", "family": "BPE", "level": "char"},
    "A4": {"desc": "SentencePiece Unigram", "impl": "sentencepiece model_type=unigram, byte_fallback, SPnat",
           "encoder": "SentencePiece native + newline wrapper", "family": "Unigram", "level": "char"},
    "A5": {"desc": "HF Unigram, P1", "impl": "HF tokenizers UnigramTrainer after Split(P1), byte_fallback",
           "encoder": "HF native", "family": "Unigram", "level": "char"},
    "A8": {"desc": "SentencePiece Unigram, superword", "impl": "sentencepiece model_type=unigram, split_by_whitespace=false",
           "encoder": "SentencePiece native + newline wrapper", "family": "Unigram", "level": "char"},
}
SP_ALGOS = ("A3", "A4", "A8")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha256_file(p):
    return H.sha256_file(p)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def versions():
    import tokenizers
    import sentencepiece
    import numpy
    return {"python": sys.version.split()[0], "tokenizers": tokenizers.__version__,
            "sentencepiece": sentencepiece.__version__, "numpy": numpy.__version__}


# ------------------------------------------------------------------------------------------ configs
def cfg(algo, pretok, data, size):
    return {"id": "%s-%s-%s-%s" % (algo, pretok, data, size), "algo": algo, "pretok": pretok, "data": data,
            "size": size, "vocab_total": SIZES[size]}


def all_configs():
    out = []
    for pt in ("P1", "P0", "P1r3", "Pm"):
        for s in SIZES:
            out.append(cfg("A1", pt, "D1", s))
    for d in ("D2", "D3"):
        for s in ("16k", "32k"):
            out.append(cfg("A1", "P1", d, s))
    for algo, pt in (("A2", "P1"), ("A3", "SPnat"), ("A4", "SPnat"), ("A5", "P1"), ("A8", "SPsuper")):
        for s in SIZES:
            out.append(cfg(algo, pt, "D1", s))
    return out


def handover_configs():
    """Built AFTER the Stage 1 rules, from their outcome (pre-tokenizer P1r3 x data mix D2), at the LM-stage sizes
    of ranks 1 and 6. Not part of the pre-registered grid and never used by a selection rule."""
    return [cfg("A1", "P1r3", "D2", s) for s in ("8k", "16k", "32k")]


def config_by_id(i):
    for c in all_configs() + handover_configs():
        if c["id"] == i:
            return c
    raise KeyError(i)


def tok_dir(c):
    return os.path.join(TOKDIR, c["id"])


def res_dir(c, suffix=""):
    return os.path.join(RESDIR, "dev_strict", c["id"] + suffix)


# --------------------------------------------------------------------------------------------- data
_DM = None
_VERIFIED = set()


def data_manifest():
    global _DM
    if _DM is None:
        _DM = json.load(open(os.path.join(DATA, "data_manifest.json"), encoding="utf-8"))
    return _DM


def verify_frozen():
    fr = json.load(open(H.FROZEN_PATH, encoding="utf-8"))
    ok = {"split_manifest": sha256_file(fr["split_manifest"]["path"]) == fr["split_manifest"]["sha256"]
          == "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2",
          "normalize_py": sha256_file(fr["normalize"]["path"]) == fr["normalize"]["sha256"],
          "data_manifest_split_hash": data_manifest()["split_manifest_sha256"] == fr["split_manifest"]["sha256"],
          "data_manifest_normalize_hash": data_manifest()["normalize_sha256"] == fr["normalize"]["sha256"]}
    if not all(ok.values()):
        raise SystemExit("FROZEN inputs changed: %s" % ok)
    return ok


def train_view(data):
    return "train_" + data


def verify_view(view, lines=False):
    key = (view, lines)
    if key in _VERIFIED:
        return
    rec = data_manifest()["views"][view]
    if lines:
        p = os.path.join(DATA, rec["lines_txt"]["file"])
        exp = rec["lines_txt"]["sha256"]
    else:
        p = os.path.join(DATA, rec["file"])
        exp = rec["sha256"]
    if sha256_file(p) != exp:
        raise SystemExit("%s does not match data_manifest.json" % p)
    _VERIFIED.add(key)


_TEXTS = {}


def read_texts(view):
    """Whole documents of a train view (normalised text, document order)."""
    if view not in _TEXTS:
        verify_view(view)
        _TEXTS.clear()
        _TEXTS[view] = [json.loads(l)["text"] for l in open(os.path.join(DATA, view + ".jsonl"), encoding="utf-8")]
    return _TEXTS[view]


def lines_file(view):
    verify_view(view, lines=True)
    return os.path.join(DATA, data_manifest()["views"][view]["lines_txt"]["file"])


# ------------------------------------------------------------------------------------------ trainers
def _hf():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
    return Tokenizer, models, trainers, pre_tokenizers, decoders, Regex


def _canonical_hf(j_or_tok, path):
    """Write through tokenizers' own serialiser (pretty) so that every HF file has one canonical form."""
    Tokenizer = _hf()[0]
    tok = Tokenizer.from_str(json.dumps(j_or_tok, ensure_ascii=False)) if isinstance(j_or_tok, dict) else j_or_tok
    tok.save(path, pretty=True)
    return tok


def train_a1(texts, V, pat, path):
    Tokenizer, models, trainers, pre_tokenizers, decoders, Regex = _hf()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pat), behavior="isolated"),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    params = {"vocab_size": V, "min_frequency": BPE_MIN_FREQUENCY, "initial_alphabet": "ByteLevel.alphabet() (256)",
              "special_tokens": "PLAN 2.1 block (64)", "limit_alphabet": None, "max_token_length": None,
              "continuing_subword_prefix": None, "end_of_word_suffix": None}
    trn = trainers.BpeTrainer(vocab_size=V, min_frequency=BPE_MIN_FREQUENCY, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIALS)
    tok.train_from_iterator(texts, trainer=trn, length=len(texts))
    _check_specials(tok, V)
    tok = _canonical_hf(tok, path)
    return tok, params


def _check_specials(tok, V=None):
    if V is not None and tok.get_vocab_size(with_added_tokens=True) != V:
        raise RuntimeError("vocabulary size %d != %d" % (tok.get_vocab_size(with_added_tokens=True), V))
    for k, s in enumerate(SPECIALS):
        if tok.token_to_id(s) != k:
            raise RuntimeError("special %s not at id %d" % (s, k))


def _insert_bytes_hf_bpe(j):
    items = sorted(j["model"]["vocab"].items(), key=lambda kv: kv[1])
    if [s for s, _ in items[:64]] != SPECIALS:
        raise RuntimeError("special block not at 0..63")
    new = {}
    for s, _ in items[:64]:
        new[s] = len(new)
    for b in range(256):
        new["<0x%02X>" % b] = len(new)
    for s, _ in items[64:]:
        if s in new:
            raise RuntimeError("collision with a byte piece: %r" % s)
        new[s] = len(new)
    j["model"]["vocab"] = new
    j["model"]["byte_fallback"] = True
    for at in j["added_tokens"]:
        at["id"] = new[at["content"]]
    j["decoder"] = {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
    return j


def train_a2(texts, V, pat, path):
    Tokenizer, models, trainers, pre_tokenizers, decoders, Regex = _hf()
    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(pat), behavior="isolated")
    params = {"vocab_size_trainer": V - 256, "vocab_size_total": V, "min_frequency": BPE_MIN_FREQUENCY,
              "limit_alphabet": None, "special_tokens": "PLAN 2.1 block (64)",
              "post": "256 <0xNN> byte-fallback pieces inserted at ids 64..319; decoder ByteFallback+Fuse"}
    trn = trainers.BpeTrainer(vocab_size=V - 256, min_frequency=BPE_MIN_FREQUENCY, show_progress=False,
                              special_tokens=SPECIALS)
    tok.train_from_iterator(texts, trainer=trn, length=len(texts))
    j = _insert_bytes_hf_bpe(json.loads(tok.to_str()))
    tok = _canonical_hf(j, path)
    _check_specials(tok, V)
    return tok, params


def train_a5(texts, V, pat, path):
    Tokenizer, models, trainers, pre_tokenizers, decoders, Regex = _hf()
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(pat), behavior="isolated")
    params = {"vocab_size_trainer": V - 257, "vocab_size_total": V, "shrinking_factor": 0.75, "max_piece_length": 16,
              "n_sub_iterations": 2, "unk_token": None, "special_tokens": "PLAN 2.1 block (64)",
              "post": "'<unk>' piece inserted at id 64 and the 256 <0xNN> pieces at ids 65..320, all with score "
                      "(min learned score - 10); unk_id=64; byte_fallback=True (the Unigram lattice needs an UNK node "
                      "for an unseen character, which tokenize() then turns into <0xNN> pieces); "
                      "decoder ByteFallback+Fuse"}
    trn = trainers.UnigramTrainer(vocab_size=V - 257, show_progress=False, special_tokens=SPECIALS,
                                  shrinking_factor=0.75, max_piece_length=16, n_sub_iterations=2)
    tok.train_from_iterator(texts, trainer=trn, length=len(texts))
    j = json.loads(tok.to_str())
    voc = j["model"]["vocab"]
    if [p for p, _ in voc[:64]] != SPECIALS:
        raise RuntimeError("special block not at 0..63")
    low = min(s for _, s in voc[64:]) - 10.0
    seen = {p for p, _ in voc}
    extra = [["<unk>", low]] + [["<0x%02X>" % b, low] for b in range(256)]
    if any(p in seen for p, _ in extra):
        raise RuntimeError("collision with an inserted piece")
    j["model"]["vocab"] = voc[:64] + extra + voc[64:]
    j["model"]["byte_fallback"] = True
    j["model"]["unk_id"] = 64
    j["decoder"] = {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
    tok = _canonical_hf(j, path)
    _check_specials(tok, V)
    return tok, params


def sp_params(inp, prefix, V, model_type, split_by_whitespace, threads=SP_THREADS):
    return dict(input=inp, model_prefix=prefix, model_type=model_type, vocab_size=V, character_coverage=1.0,
                byte_fallback=True, split_digits=True, split_by_whitespace=split_by_whitespace,
                split_by_unicode_script=True, split_by_number=True, max_sentencepiece_length=16,
                normalization_rule_name="identity", remove_extra_whitespaces=False, add_dummy_prefix=True,
                max_sentence_length=65536, user_defined_symbols=SPECIALS + [NL], unk_id=SP_UNK_ID,
                bos_id=-1, eos_id=-1, pad_id=-1, num_threads=threads, input_sentence_size=0,
                shuffle_input_sentence=False, minloglevel=2)


def train_sp(inp, prefix, V, model_type, split_by_whitespace):
    import sentencepiece as spm
    p = sp_params(inp, prefix, V, model_type, split_by_whitespace)
    spm.SentencePieceTrainer.train(**p)
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    if sp.get_piece_size() != V:
        raise RuntimeError("SentencePiece size %d != %d" % (sp.get_piece_size(), V))
    for k, s in enumerate(SPECIALS):
        if sp.id_to_piece(k) != s:
            raise RuntimeError("special %s not at id %d" % (s, k))
    if sp.id_to_piece(SP_NL_ID) != NL or sp.unk_id() != SP_UNK_ID:
        raise RuntimeError("newline/unk ids differ")
    rec = {k: v for k, v in p.items() if k not in ("user_defined_symbols",)}
    rec["user_defined_symbols"] = "PLAN 2.1 block (64, ids 0..63) + ['\\n'] (id 64)"
    return rec


# ------------------------------------------------------------------------------ SentencePiece -> HF
def _sp_proto(path):
    from sentencepiece import sentencepiece_model_pb2 as spb
    m = spb.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m, spb.ModelProto.SentencePiece


def sp_to_hf_json(model_path):
    """HF tokenizer.json EXPORT of a SentencePiece model that is used through the PLAN 1.1 newline wrapper.
    '\\n', the special block and CONTROL placeholders become added tokens (extracted before the model, as the
    wrapper splits lines); each line gets SentencePiece's own whitespace escaping (' '->U+2581 and one
    U+2581 dummy prefix, a Replace+Prepend normalizer: no Unicode normalisation); the model is HF Unigram
    (SentencePiece scores; non-NORMAL pieces get LOW_SCORE, since SentencePiece keeps them out of its lattice)
    or HF BPE (merges generated from the pieces, ranked by the merged piece's score, as SentencePiece BPE
    merges the highest-scoring adjacent pair). The decoder undoes the escaping per line."""
    Tokenizer = _hf()[0]
    m, T = _sp_proto(model_path)
    pieces = list(m.pieces)
    added = []
    for i, p in enumerate(pieces):
        if p.type in (T.USER_DEFINED, T.CONTROL):
            added.append({"id": i, "content": p.piece, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": p.piece != NL})
    unk = [i for i, p in enumerate(pieces) if p.type == T.UNKNOWN]
    unk_id = unk[0] if unk else None
    mt = m.trainer_spec.model_type
    if mt == 1:
        vocab = [[p.piece, float(p.score) if p.type == T.NORMAL else LOW_SCORE] for p in pieces]
        model = {"type": "Unigram", "unk_id": unk_id, "vocab": vocab, "byte_fallback": True}
    elif mt == 2:
        vocab = {p.piece: i for i, p in enumerate(pieces)}
        normal = {p.piece for p in pieces if p.type == T.NORMAL}
        cands = []
        for i, p in enumerate(pieces):
            if p.type != T.NORMAL or len(p.piece) < 2:
                continue
            for k in range(1, len(p.piece)):
                a, b = p.piece[:k], p.piece[k:]
                if a in normal and b in normal:
                    cands.append((-float(p.score), i, vocab[a], vocab[b], a, b))
        cands.sort()
        model = {"type": "BPE", "dropout": None, "unk_token": pieces[unk_id].piece if unk_id is not None else None,
                 "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False,
                 "byte_fallback": True, "ignore_merges": False, "vocab": vocab,
                 "merges": [[c[4], c[5]] for c in cands]}
    else:
        raise ValueError("model_type %d" % mt)
    j = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
         "normalizer": {"type": "Sequence", "normalizers": [
             {"type": "Replace", "pattern": {"String": " "}, "content": SP_MARK},
             {"type": "Prepend", "prepend": SP_MARK}]},
         "pre_tokenizer": None, "post_processor": None,
         "decoder": {"type": "Sequence", "decoders": [
             {"type": "ByteFallback"}, {"type": "Fuse"},
             {"type": "Replace", "pattern": {"String": NL + SP_MARK}, "content": NL},
             {"type": "Strip", "content": SP_MARK, "start": 1, "stop": 0},
             {"type": "Replace", "pattern": {"String": SP_MARK}, "content": " "}]},
         "model": model}
    tok = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    return tok.to_str(pretty=True)


def sp_hf_equivalence(model_path, json_path, texts_by_view):
    """Per document: HF-export ids == native SentencePiece+wrapper ids, and HF decode == text."""
    Tokenizer = _hf()[0]
    sp = A.SPAdapter(path=model_path, name="native", newline_wrapper=True)
    hf = Tokenizer.from_file(json_path)
    out = {}
    for view, docs in texts_by_view.items():
        n_ids = n_dec = n_len = 0
        ex = []
        for uid, t in docs:
            a = sp.encode(t)
            e = hf.encode(t, add_special_tokens=False).ids
            if a != e:
                n_ids += 1
                n_len += len(a) != len(e)
                if len(ex) < 3:
                    k = next((i for i, (x, y) in enumerate(zip(a + [None], e + [None])) if x != y), None)
                    ex.append({"uid": uid, "first_diff": k, "native": a[k:k + 4] if k is not None else None,
                               "hf": e[k:k + 4] if k is not None else None})
            if hf.decode(e, skip_special_tokens=False) != t:
                n_dec += 1
        out[view] = {"docs": len(docs), "ids_differ_docs": n_ids, "token_count_differs_docs": n_len,
                     "hf_decode_differs_docs": n_dec, "examples": ex}
    vals = [v for v in out.values() if isinstance(v, dict)]
    out["equivalent"] = all(v["ids_differ_docs"] == 0 and v["hf_decode_differs_docs"] == 0 for v in vals)
    out["equivalent_token_counts"] = all(v["token_count_differs_docs"] == 0 and v["hf_decode_differs_docs"] == 0
                                         for v in vals)
    return out


def g2_all_failures(adapter, test_superwords):
    """Every G2 failure (harness.gate_g2 truncates its list at 50), plus superword pieces when requested."""
    fails = []
    for i in H.learned_ids(adapter):
        txt = H.token_text(adapter, i)
        if txt is None:
            continue
        sw = any(c.isspace() for c in txt[1:]) and not txt.isspace()
        if sw and not test_superwords:
            continue
        if list(adapter.encode_isolated(txt)) != [i]:
            fails.append(i)
    return fails


def hf_unigram_functional_g4(path_a, path_b, texts_by_view):
    """Diagnostic for HF UnigramTrainer runs whose files differ: same piece set? score noise? identical dev
    encodings once ids are mapped through the piece strings?"""
    Tokenizer = _hf()[0]
    ja = json.load(open(path_a, encoding="utf-8"))
    jb = json.load(open(path_b, encoding="utf-8"))
    va, vb = ja["model"]["vocab"], jb["model"]["vocab"]
    sa, sb = {p: s for p, s in va}, {p: s for p, s in vb}
    same_set = set(sa) == set(sb)
    out = {"same_piece_set": same_set, "pieces_only_in_a": len(set(sa) - set(sb)),
           "pieces_only_in_b": len(set(sb) - set(sa)),
           "same_piece_order": [p for p, _ in va] == [p for p, _ in vb],
           "max_abs_score_diff": max((abs(sa[p] - sb[p]) for p in set(sa) & set(sb)), default=None)}
    ta, tb = Tokenizer.from_file(path_a), Tokenizer.from_file(path_b)
    idb = {p: i for i, (p, _) in enumerate(vb)}
    id_map = {i: idb.get(p, -1) for i, (p, _) in enumerate(va)}
    for view, docs in texts_by_view.items():
        n = 0
        for _, t in docs:
            a = [id_map[x] for x in ta.encode(t, add_special_tokens=False).ids]
            if a != tb.encode(t, add_special_tokens=False).ids:
                n += 1
        out[view] = {"docs": len(docs), "docs_differing_after_id_mapping": n}
    return out


# ------------------------------------------------------------------------------------- gate helpers
def g2_superword(adapter):
    """PLAN 4.3 G2 for learned tokens with whitespace inside (A8 superword pieces), which harness.gate_g2 lists
    but does not test. For a Unigram (Viterbi) encoder the isolated self-tokenization of the piece's own string
    is the exact reachability test: a piece that loses on its own string never wins inside a longer string
    (PLAN 4.3 remedy argument), and one that wins is produced whenever that string is a whole line."""
    fails, tested = [], 0
    for i in H.learned_ids(adapter):
        txt = H.token_text(adapter, i)
        if txt is None or txt.isspace() or not any(c.isspace() for c in txt[1:]):
            continue
        tested += 1
        ids = list(adapter.encode_isolated(txt))
        if ids != [i]:
            fails.append({"id": i, "text": txt, "encodes_to": ids[:12]})
    return {"tested": tested, "failures": len(fails), "failure_list": fails[:50],
            "method": "isolated self-tokenization (exact for Unigram/Viterbi encoders)"}


def g1_light(adapter, view):
    docs, info = H.load_docs(view)
    fails = [d["uid"] for d in docs if adapter.decode(adapter.encode(d["text"])) != d["text"]]
    return {"pass": not fails, "dataset": view, "docs": len(docs), "fail_docs": len(fails), "fail_uids": fails[:20],
            "sha256": info["sha256"]}


def dev_texts(views=("dev_strict", "dev_permissive")):
    out = {}
    for v in views:
        docs, _ = H.load_docs(v)
        out[v] = [(d["uid"], d["text"]) for d in docs]
    return out


def reencode_check(adapter_a, adapter_b, texts_by_view, id_map=None):
    """Dev encodings identical (after mapping ids of a through id_map, identity if None)."""
    out = {}
    for view, docs in texts_by_view.items():
        n = 0
        for _, t in docs:
            a = adapter_a.encode(t)
            if id_map is not None:
                a = [id_map.get(x, -1) for x in a]
            if a != adapter_b.encode(t):
                n += 1
        out[view] = {"docs": len(docs), "docs_changed": n}
    out["identical"] = all(v["docs_changed"] == 0 for k, v in out.items() if isinstance(v, dict))
    return out


# ------------------------------------------------------------------------------------- G2 remedy
def remedy_sp(model_path, fail_ids, out_path):
    """Delete failing pieces: the slot becomes a CONTROL placeholder '<|unused_k|>' (never emitted by the
    encoder, never matched in text), so ids stay stable and the total stays a multiple of 64."""
    m, T = _sp_proto(model_path)
    names = {}
    for k, i in enumerate(sorted(fail_ids)):
        names[i] = {"old": m.pieces[i].piece, "new": "<|unused_%d|>" % k}
        m.pieces[i].piece = "<|unused_%d|>" % k
        m.pieces[i].type = T.CONTROL
    with open(out_path, "wb") as f:
        f.write(m.SerializeToString())
    return names


def write_sp_vocab(model_path, out_path):
    """The .vocab listing SentencePiece writes next to a model (piece TAB score), regenerated from a model file."""
    m, T = _sp_proto(model_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for p in m.pieces:
            f.write("%s\t%s\n" % (p.piece, ("%g" % p.score) if p.score != int(p.score) else int(p.score)))


def remedy_hf_unigram(json_path, fail_ids, out_path):
    j = json.load(open(json_path, encoding="utf-8"))
    voc = j["model"]["vocab"]
    low = min(s for _, s in voc) - 10.0
    names = {}
    for k, i in enumerate(sorted(fail_ids)):
        new = "<|unused_%d|>" % k
        names[i] = {"old": voc[i][0], "new": new}
        voc[i] = [new, low]
        j["added_tokens"].append({"id": i, "content": new, "single_word": False, "lstrip": False, "rstrip": False,
                                  "normalized": False, "special": True})
    j["added_tokens"].sort(key=lambda a: a["id"])
    _canonical_hf(j, out_path)
    return names


# ------------------------------------------------------------------------------------- adapters
def make_adapter(c, which="final"):
    d = tok_dir(c)
    if c["algo"] in SP_ALGOS:
        p = os.path.join(d, "sp.model") if which == "final" else os.path.join(d, which, "sp.model")
        return A.SPAdapter(path=p, name=c["id"], newline_wrapper=True)
    p = os.path.join(d, "tokenizer.json") if which == "final" else os.path.join(d, which, "tokenizer.json")
    return A.HFAdapter(path=p, name=c["id"])


# ------------------------------------------------------------------------------------- Pm differential
def pretok_differential(names=("Pm",)):
    """PLAN 1.3 differential (Python `regex` vs Oniguruma) for pre-tokenizers not covered in Stage 0."""
    import regex as pyre
    from tokenizers import pre_tokenizers, Regex
    import perturb as P
    out = {"regex_version": pyre.__version__, "tokenizers_version": versions()["tokenizers"], "results": {}}
    for name in names:
        onig = PRETOK[name]
        py = H.onig_to_py(onig)
        hf = pre_tokenizers.Split(Regex(onig), behavior="isolated")
        rx = pyre.compile(py)
        res = {"oniguruma": onig, "python": py}
        for d in ("dev_strict", "dev_permissive", "dev_strict+zwnj", "dev_strict+harakat"):
            docs, _ = H.load_docs(d.split("+")[0])
            n_diff = n_tok = 0
            ex = []
            for doc in docs:
                t = doc["text"]
                if d.endswith("+zwnj"):
                    t = P.perturb_zwnj(t).text
                elif d.endswith("+harakat"):
                    t = P.PERTURBATIONS["harakat"](t).text
                a = [s for s, _ in hf.pre_tokenize_str(t)]
                b = H.py_pretokens(rx, t)
                n_tok += len(a)
                if a != b:
                    n_diff += 1
                    if len(ex) < 3:
                        ex.append({"uid": doc["uid"]})
            res[d] = {"docs": len(docs), "docs_differing": n_diff, "pretokens_onig": n_tok, "examples": ex}
        res["identical"] = all(res[k]["docs_differing"] == 0 for k in res if k.startswith("dev_"))
        out["results"][name] = res
    return out
