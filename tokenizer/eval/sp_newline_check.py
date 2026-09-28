# -*- coding: utf-8 -*-
"""PLAN 1.1 verification (Stage 0): SentencePiece with a dedicated '\\n' user-defined piece under
normalization_rule_name=identity, encoded through the harness's newline wrapper.

Trains two small VERIFICATION models (NOT candidates; vocab 8,000, on every 5th line of train_D1.lines.txt):
SentencePiece Unigram and SentencePiece BPE, both with byte_fallback, split_digits, identity normalisation,
remove_extra_whitespaces=False, max_sentence_length=65536 and user_defined_symbols = ['\\n'] + the 64-token
special block of PLAN 2.1. Then:
  1. does training accept a newline user symbol under identity? (it raises otherwise)
  2. harness run on dev_strict with --gates all (G1-G5, R1 on train_D1) through the newline wrapper
  3. the same model WITHOUT the wrapper (whole documents, '\\n' matched as a user symbol inside the text):
     round trip, token count, and how many line-initial words lose the '▁' word-start marker
Writes eval/sp_check/*.model and eval/sp_newline_check.json. Uses 3 threads, one process.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402

H._setup_threads(3)
import adapters as A  # noqa: E402
import sentencepiece as spm  # noqa: E402

OUT = os.path.join(HERE, "sp_check")
os.makedirs(OUT, exist_ok=True)
VOCAB = 8000
STRIDE = 5


def main():
    t0 = time.time()
    src = os.path.join(H.DATA_DIR, "train_D1.lines.txt")
    sub = os.path.join(OUT, "train_D1_every%dth_line.txt" % STRIDE)
    n = 0
    with open(src, encoding="utf-8") as f, open(sub, "w", encoding="utf-8", newline="\n") as g:
        for k, line in enumerate(f):
            if k % STRIDE == 0:
                g.write(line); n += 1
    report = {"what": __doc__.split("\n")[0], "train_lines": n, "train_file_sha256": H.sha256_file(sub),
              "vocab_size": VOCAB, "models": {}}
    docs, _ = H.load_docs("dev_strict")
    for mtype in ("unigram", "bpe"):
        prefix = os.path.join(OUT, "sp_%s_%d_nlwrap" % (mtype, VOCAB))
        params = dict(input=sub, model_prefix=prefix, model_type=mtype, vocab_size=VOCAB, character_coverage=1.0,
                      byte_fallback=True, split_digits=True, normalization_rule_name="identity",
                      remove_extra_whitespaces=False, add_dummy_prefix=True, max_sentence_length=65536,
                      user_defined_symbols=["\n"] + H.SPECIAL_TOKENS, num_threads=3, input_sentence_size=0,
                      shuffle_input_sentence=False, minloglevel=2)
        rec = {"params": {k: v for k, v in params.items() if k != "user_defined_symbols"},
               "user_defined_symbols": "['\\n'] + 64 special tokens"}
        t1 = time.time()
        try:
            spm.SentencePieceTrainer.train(**params)
            rec["training_accepted_newline_user_symbol"] = True
        except Exception as e:  # noqa: BLE001
            rec["training_accepted_newline_user_symbol"] = False
            rec["error"] = repr(e)
            report["models"][mtype] = rec
            continue
        rec["train_s"] = round(time.time() - t1, 1)
        rec["model_sha256"] = H.sha256_file(prefix + ".model")
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        nid = sp.piece_to_id("\n")
        rec["newline_piece_id"] = nid
        # 2. harness through the wrapper
        ad = A.SPAdapter(path=prefix + ".model", name="spcheck_%s_%d" % (mtype, VOCAB), newline_wrapper=True)
        rec["newline_piece_is_user_defined"] = ad.user_defined.get(chr(10)) == nid
        s = H.run(ad, data="dev_strict", gates="all", train="train_D1",
                  out_dir=os.path.join(OUT, "results", ad.name), force=True)
        o = s["metrics"]["overall"]
        rec["wrapper"] = {"G1": s["gates"]["G1"], "G2": {k: s["gates"]["G2"][k] for k in ("pass", "tested", "failures")},
                          "G2_failure_examples": s["gates"]["G2"]["failure_list"][:5],
                          "G3": {k: s["gates"]["G3"][k] for k in ("pass", "block_size_present", "not_atomic_n")},
                          "G5": s["gates"]["G5"], "tokens": o["tokens"], "bytes_per_token": o["bytes_per_token"],
                          "fertility": o["fertility"],
                          "R1_multichar": s["properties"]["R1"]["multichar_learned"],
                          "byte_fallback_pieces_used_on_dev": None}
        # 3. without the wrapper: whole documents, '\n' as a user symbol inside the text
        raw_rt = raw_tok = nl_pieces = bytepieces = no_marker = 0
        wrap_bytepieces = 0
        for d in docs:
            t = d["text"]
            ids = sp.encode(t)
            raw_tok += len(ids)
            raw_rt += sp.decode(ids) == t
            nl_pieces += sum(1 for i in ids if i == nid)
            bytepieces += sum(1 for i in ids if sp.is_byte(i))
            wids = ad.encode(t)
            wrap_bytepieces += sum(1 for i in wids if sp.is_byte(i))
            # line-initial words: in the raw encoding the piece right after '\n' should start with '▁'
            for k, i in enumerate(ids[:-1]):
                if i == nid and not sp.id_to_piece(ids[k + 1]).startswith("▁") and ids[k + 1] != nid:
                    no_marker += 1
        rec["wrapper"]["byte_fallback_pieces_used_on_dev"] = wrap_bytepieces
        rec["no_wrapper_whole_docs"] = {"roundtrip_docs": raw_rt, "docs": len(docs), "tokens": raw_tok,
                                        "tokens_vs_wrapper": raw_tok - o["tokens"], "newline_pieces": nl_pieces,
                                        "byte_fallback_pieces": bytepieces,
                                        "line_initial_pieces_without_word_marker": no_marker}
        report["models"][mtype] = rec
        H.log(mtype, json.dumps(H.clean({"G1": rec["wrapper"]["G1"]["pass"], "raw": rec["no_wrapper_whole_docs"]})))
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "sp_newline_check.json"), "w", encoding="utf-8") as f:
        json.dump(H.clean(report), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
