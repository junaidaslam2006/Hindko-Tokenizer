# -*- coding: utf-8 -*-
"""Inspect R2-A4-SPnat-D2-32k sp.model: piece types, scores, G2 remedy, and the known HF/SP mismatch."""
import json, os, sys, struct, collections
sys.path.insert(0, r"F:\Hindko\_tokenizer\eval")
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as spb
import adapters as A

D = r"F:\Hindko\_tokenizer\candidates\round2\standard\tok\R2-A4-SPnat-D2-32k"
m = spb.ModelProto(); m.ParseFromString(open(os.path.join(D, "sp.model"), "rb").read())
m0 = spb.ModelProto(); m0.ParseFromString(open(os.path.join(D, "pre_remedy", "sp.model"), "rb").read())
T = spb.ModelProto.SentencePiece
print("pieces", len(m.pieces), "types", collections.Counter(T.Type.Name(p.type) for p in m.pieces))
print("pre-remedy types", collections.Counter(T.Type.Name(p.type) for p in m0.pieces))
diff = [(i, p0.piece, T.Type.Name(p0.type), p0.score, p.piece, T.Type.Name(p.type), p.score)
        for i, (p0, p) in enumerate(zip(m0.pieces, m.pieces)) if (p0.piece, p0.type, p0.score) != (p.piece, p.type, p.score)]
print("remedy changed pieces:", diff)
print("normalizer_spec:", m.normalizer_spec)
print("denormalizer_spec:", m.denormalizer_spec)
ts = m.trainer_spec
print("trainer: model_type", ts.model_type, "byte_fallback", ts.byte_fallback, "unk_id", ts.unk_id,
      "split_digits", ts.split_digits, "treat_ws_suffix", ts.treat_whitespace_as_suffix,
      "allow_ws_only", ts.allow_whitespace_only_pieces, "unk_piece", ts.unk_piece, "unk_surface", repr(ts.unk_surface))
normal = [(i, p.piece, p.score) for i, p in enumerate(m.pieces) if p.type == T.NORMAL]
sc = [s for _, _, s in normal]
print("NORMAL:", len(normal), "min", min(sc), "max", max(sc))
print("first 70 pieces:", [(i, p.piece, T.Type.Name(p.type), p.score) for i, p in enumerate(m.pieces[:70])][60:])
print("byte pieces first/last:", [(i, p.piece, p.score) for i, p in enumerate(m.pieces) if p.type == T.BYTE][:2],
      [(i, p.piece, p.score) for i, p in enumerate(m.pieces) if p.type == T.BYTE][-1])
# pieces with interior U+2581
mark = "\u2581"
inner = [(i, p.piece) for i, p in enumerate(m.pieces) if p.type == T.NORMAL and mark in p.piece[1:]]
print("NORMAL pieces with interior U+2581:", len(inner), inner[:20])
ws_only = [(i, p.piece, p.score) for i, p in enumerate(m.pieces) if p.type == T.NORMAL and set(p.piece) <= {mark}]
print("whitespace-only NORMAL pieces:", ws_only)
# score float32 bit structure
tz = collections.Counter()
for _, _, s in normal:
    b = struct.unpack("<I", struct.pack("<f", s))[0]
    mant = b & 0x7FFFFF
    t = 0
    while t < 23 and not (mant >> t) & 1:
        t += 1
    tz[t] += 1
print("trailing zero mantissa bits histogram (NORMAL):", sorted(tz.items())[:10])
# duplicate scores
cnt = collections.Counter(sc)
dups = sum(v for v in cnt.values() if v > 1)
print("NORMAL pieces sharing an exact score with another piece:", dups, "distinct scores", len(cnt))
print("smallest |score| NORMAL:", sorted(normal, key=lambda x: abs(x[2]))[:5])
# the known mismatch
for i in (13249, 4904, 343, 2753):
    print(i, repr(m.pieces[i].piece), m.pieces[i].score)
docs = [json.loads(l) for l in open(r"F:\Hindko\_tokenizer\data\dev_permissive.jsonl", encoding="utf-8")]
doc = [d for d in docs if d["uid"] == "18ae9b755d88d873"][0]
sp = A.SPAdapter(path=os.path.join(D, "sp.model"))
ids = sp.encode(doc["text"])
from tokenizers import Tokenizer
hf = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
e = hf.encode(doc["text"], add_special_tokens=False).ids
k = next(i for i, (x, y) in enumerate(zip(ids, e)) if x != y)
print("first diff", k, "native", [(i, sp.sp.id_to_piece(i)) for i in ids[k-3:k+4]])
print("hf", [(i, sp.sp.id_to_piece(i)) for i in e[k-3:k+4]])
# which line
lines = doc["text"].split("\n")
for li, line in enumerate(lines):
    a = sp.sp.encode(line); b = hf.encode(line, add_special_tokens=False).ids
    if a != b:
        print("line", li, "chars", len(line), "tokens", len(a))
        pos = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        print("  line-pos", pos, "ctx", repr("".join(sp.sp.id_to_piece(i) for i in a[max(0,pos-3):pos+4])))
        print("  isolated word encodes:", [sp.sp.id_to_piece(i) for i in sp.sp.encode("".join(sp.sp.id_to_piece(i) for i in a[pos-1:pos+3]).replace(mark, " ").strip())])
        s = 0.0
        import numpy as np
        f32 = np.float32(0)
        for i in a[:pos]:
            f32 = np.float32(f32 + np.float32(m.pieces[i].score))
            s += m.pieces[i].score
        print("  prefix score f64", s, "f32", float(f32))
