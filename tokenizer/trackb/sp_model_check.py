# -*- coding: utf-8 -*-
"""Can the Gemma extension also be expressed as a SentencePiece .model? (measurement, PLAN 9.2 for SP bases)

The Gemma tokenizer.json is an HF BPE model with an explicit merge list, and the extension appends pair-specific
merges to it. SentencePiece's own BPE encoder has no merge list: it repeatedly merges the adjacent pair whose
CONCATENATION is a piece with the highest score, whatever the split. The analogous SentencePiece extension appends
the k new pieces (NORMAL, scores below every base piece, in learning order; the HF-only '<image_soft_token>'
of Gemma 3 is inserted first as a USER_DEFINED piece so that ids line up). This script builds that .model for
every k and measures how often it encodes dev_strict exactly like the extended tokenizer.json.

    python sp_model_check.py --base gemma-3    -> results/<base>/sp_model_check.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402
import continued_bpe as CB  # noqa: E402

C.env_threads(2)
import sentencepiece as spm  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as spb  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def build_proto(base, ext, k):
    m = spb.ModelProto()
    m.ParseFromString(open(os.path.join(C.BASES[base]["dir"], "tokenizer.model"), "rb").read())
    j = C.load_base_json(base)
    P = spb.ModelProto.SentencePiece
    n0 = len(m.pieces)
    for a in sorted(j.get("added_tokens", []), key=lambda a: a["id"]):
        if a["id"] >= n0:
            assert a["id"] == len(m.pieces)
            p = m.pieces.add(); p.piece = a["content"]; p.score = 0.0; p.type = P.USER_DEFINED
    assert len(m.pieces) == ext["first_new_id"]
    smin = min(p.score for p in m.pieces if p.type == P.NORMAL)
    for i, mm in enumerate(x for x in CB.select(ext, k) if x["kind"] == "new"):
        p = m.pieces.add(); p.piece = mm["result"]; p.score = smin - 1.0 - i; p.type = P.NORMAL
    m.trainer_spec.vocab_size = len(m.pieces)
    return m.SerializeToString()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma-3")
    a = ap.parse_args()
    ext = C.load_json(os.path.join(C.WORK, a.base, "extension.json"))
    dev = C.load_view("dev_strict")
    texts = [d["text"] for d in dev]
    out = {"base": a.base, "what": __doc__.split("\n\n")[0], "by_k": {}}
    for k in [0] + C.K_GRID:
        proto = build_proto(a.base, ext, k) if k else open(os.path.join(C.BASES[a.base]["dir"], "tokenizer.model"), "rb").read()
        sp = spm.SentencePieceProcessor(model_proto=proto)
        hf = Tokenizer.from_file(os.path.join(C.SWEEP, a.base, "k%d" % k, "tokenizer.json"))
        same = tok_sp = tok_hf = rt = 0
        ex = []
        for t in texts:
            s = sp.encode(t)
            h = hf.encode(t, add_special_tokens=False).ids
            same += int(s == h)
            tok_sp += len(s); tok_hf += len(h)
            rt += int(sp.decode(s) == t)
            if s != h and len(ex) < 3:
                i = next(i for i, (x, y) in enumerate(zip(s + [None], h + [None])) if x != y)
                ex.append({"sp": [sp.id_to_piece(x) for x in s[i:i + 4]], "hf": [hf.id_to_token(x) for x in h[i:i + 4]]})
        out["by_k"][k] = {"docs": len(texts), "identical_docs": same, "tokens_sp": tok_sp, "tokens_hf": tok_hf,
                          "sp_roundtrip_docs": rt, "examples": ex}
        C.log(a.base, k, json.dumps(out["by_k"][k], ensure_ascii=False)[:400])
    C.dump_json(out, os.path.join(C.RESULTS, a.base, "sp_model_check.json"))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
