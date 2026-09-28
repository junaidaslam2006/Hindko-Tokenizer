# -*- coding: utf-8 -*-
"""Equivalence verification of the sp32k release -> release_build/sp32k/equivalence.json

Encoders compared, per document (sets) and per item (stress):
  canonical   eval/adapters.py SPAdapter(release sp.model, newline_wrapper=True)   (PLAN 1.1 convention, unchanged code)
  hf          tokenizers.Tokenizer.from_file(release tokenizer.json).encode(text).ids   (DEFAULT arguments)
  tf          transformers.AutoTokenizer.from_pretrained(release folder)(text, add_special_tokens=False).input_ids,
              and tf.encode(text) with its defaults
  old         SPAdapter(old candidate sp.model)                  -> which documents the score change touched
  old_exact   the old candidate tokenizer.json (float64 Viterbi on the old scores) -> old float32 tie decisions
For every line whose canonical encoding changed old -> new, the exact (rational) old score of the old and of the
new segmentation is computed: a change is admissible only if the two are EXACTLY equal (a Viterbi tie).
"""
import collections
import json
import os
import re
import sys
import time
from fractions import Fraction

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
TOK = r"F:\Hindko\_tokenizer"
sys.path.insert(0, os.path.join(TOK, "eval"))
import adapters as A  # noqa: E402

REL = os.path.join(TOK, r"release_build\sp32k")
CAND = os.path.join(TOK, r"candidates\round2\standard\tok\R2-A4-SPnat-D2-32k")
DATA = os.path.join(TOK, "data")
SETS = ["dev_strict", "dev_permissive", "test_strict", "train_D2"]
NL = chr(10)


def load_proto(path):
    from sentencepiece import sentencepiece_model_pb2 as spb
    m = spb.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m, spb.ModelProto.SentencePiece


def main():
    from tokenizers import Tokenizer
    import transformers
    from transformers import AutoTokenizer
    t0 = time.time()
    can = A.SPAdapter(path=os.path.join(REL, "sp.model"), name="release-canonical")
    old = A.SPAdapter(path=os.path.join(CAND, "sp.model"), name="old-canonical")
    hf = Tokenizer.from_file(os.path.join(REL, "tokenizer.json"))
    old_exact = Tokenizer.from_file(os.path.join(CAND, "tokenizer.json"))
    tf = AutoTokenizer.from_pretrained(REL)
    mo, T = load_proto(os.path.join(CAND, "sp.model"))
    mn, _ = load_proto(os.path.join(REL, "sp.model"))
    so = [Fraction(p.score) if p.type == T.NORMAL else Fraction(0) for p in mo.pieces]
    sn = [Fraction(p.score) if p.type == T.NORMAL else Fraction(0) for p in mn.pieces]
    specials = [p.piece for p in mn.pieces if p.type in (T.USER_DEFINED, T.CONTROL) and p.piece != NL]
    sp_id = {p.piece: i for i, p in enumerate(mn.pieces)}
    spec_re = re.compile("(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")")

    def split_canonical(text):
        """specials cut out first (as HF added tokens are), each remaining segment encoded by the canonical encoder"""
        out = []
        for k, seg in enumerate(spec_re.split(text)):
            if k % 2:
                out.append(sp_id[seg])
            elif seg:
                out.extend(can.encode(seg))
        return out

    def line_ties(text):
        """for every line whose encoding changed old->new: exact old and new score gaps"""
        rows = []
        for line in text.split(NL):
            a = old.sp.encode(line); b = can.sp.encode(line)
            if a == b:
                continue
            ga = sum(so[x] for x in a) - sum(so[x] for x in b)
            gn = sum(sn[x] for x in b) - sum(sn[x] for x in a)
            rows.append({"old_gap_exact": str(ga), "new_gap_exact": str(gn),
                         "same_piece_multiset": collections.Counter(a) == collections.Counter(b),
                         "old": [old.sp.id_to_piece(x) for x in a if x not in b or a.count(x) != b.count(x)][:6],
                         "new": [old.sp.id_to_piece(x) for x in b if x not in a or a.count(x) != b.count(x)][:6]})
        return rows

    def run(items, label, compare_specials=False):
        texts = [t for _, t in items]
        ids_can = [can.encode(t) for t in texts]
        ids_old = [old.encode(t) for t in texts]
        ids_hf = [e.ids for e in hf.encode_batch(texts)]                      # default add_special_tokens=True
        ids_oe = [e.ids for e in old_exact.encode_batch(texts, add_special_tokens=False)]
        ids_tf = tf(texts, add_special_tokens=False)["input_ids"]
        r = collections.Counter()
        ex = collections.defaultdict(list)
        changed_rows = []
        for k, (uid, t) in enumerate(items):
            c = ids_can[k]
            ref = split_canonical(t) if compare_specials else c
            r["items"] += 1
            r["tokens_canonical"] += len(c)
            ok_hf = ids_hf[k] == ref
            ok_tf = list(ids_tf[k]) == ref
            r["hf_ids_equal"] += ok_hf
            r["tf_ids_equal"] += ok_tf
            if compare_specials:
                r["hf_ids_equal_to_plain_canonical"] += ids_hf[k] == c
            dec_hf = hf.decode(ids_hf[k], skip_special_tokens=False)
            dec_tf = tf.decode(ids_tf[k], skip_special_tokens=False)
            r["hf_decode_exact"] += dec_hf == t
            r["tf_decode_exact"] += dec_tf == t
            r["canonical_decode_exact"] += can.decode(c) == t
            if not ok_hf and len(ex["hf_mismatch"]) < 5:
                ex["hf_mismatch"].append({"id": uid, "text": t[:120], "canonical": ref[:40], "hf": ids_hf[k][:40]})
            if dec_hf != t and len(ex["hf_decode"]) < 5:
                ex["hf_decode"].append({"id": uid, "text": t[:120], "decoded": dec_hf[:120]})
            if not ok_tf and len(ex["tf_mismatch"]) < 5:
                ex["tf_mismatch"].append({"id": uid, "text": t[:120]})
            old_tie = ids_old[k] != ids_oe[k]            # old float32 SentencePiece != old exact arithmetic
            r["old_float32_tie_items"] += old_tie
            if ids_old[k] != c:
                r["changed_vs_old"] += 1
                r["changed_vs_old_in_old_tie_items"] += old_tie
                rows = line_ties(t)
                exact = all(Fraction(x["old_gap_exact"]) == 0 for x in rows) and bool(rows)
                r["changed_vs_old_all_lines_exact_ties"] += exact
                if not exact:
                    r["changed_vs_old_NOT_exact_tie"] += 1
                changed_rows.append({"id": uid, "old_float32_tie": old_tie, "lines": rows})
        out = dict(r)
        out["examples"] = dict(ex)
        out["changed_items"] = changed_rows[:40]
        print(label, json.dumps({k: v for k, v in out.items() if k not in ("examples", "changed_items")}), flush=True)
        return out

    res = {"what": "equivalence of the sp32k release encoders (see verify.py docstring)",
           "versions": {"python": sys.version.split()[0], "sentencepiece": __import__("sentencepiece").__version__,
                        "tokenizers": __import__("tokenizers").__version__, "transformers": transformers.__version__},
           "files": {n: A.sha256_file(os.path.join(REL, n)) for n in
                     ("sp.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")},
           "old_files": {"sp.model": A.sha256_file(os.path.join(CAND, "sp.model")),
                         "tokenizer.json": A.sha256_file(os.path.join(CAND, "tokenizer.json"))},
           "canonical_identity": can.identity(), "sets": {}, "stress": {}}
    for name in SETS:
        p = os.path.join(DATA, name + ".jsonl")
        with open(p, encoding="utf-8") as f:
            items = [(json.loads(l)["uid"], json.loads(l)["text"]) for l in f]
        out = run(items, name)
        out["sha256"] = A.sha256_file(p)
        res["sets"][name] = out
    stress_path = os.path.join(REL, r"stress\stress.jsonl")
    by_cat = collections.defaultdict(list)
    with open(stress_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            by_cat[r["cat"]].append((r["id"], r["text"]))
    for cat, items in by_cat.items():
        res["stress"][cat] = run(items, "stress/" + cat, compare_specials=(cat == "specials"))
    res["stress_sha256"] = A.sha256_file(stress_path)

    # transformers object checks
    msgs = [{"role": "system", "content": "تساں اک مددگار او۔"}, {"role": "user", "content": "سلام! تساں کیویں او؟"},
            {"role": "assistant", "content": "میں ٹھیک آں، شکریہ۔"}]
    chat_txt = tf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat_ids = tf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(chat_ids, "keys"):
        chat_ids = chat_ids["input_ids"]
    res["transformers"] = {
        "class": type(tf).__name__, "len": len(tf), "vocab_size": tf.vocab_size,
        "bos": [tf.bos_token, tf.bos_token_id], "eos": [tf.eos_token, tf.eos_token_id],
        "pad": [tf.pad_token, tf.pad_token_id], "unk": [tf.unk_token, tf.unk_token_id],
        "newline_id": tf.convert_tokens_to_ids(NL), "id_64": tf.convert_ids_to_tokens(64),
        "n_special_ids": len(tf.all_special_ids),
        "default_call_adds_nothing": tf("سلام دنیا")["input_ids"] == can.encode("سلام دنیا"),
        "encode_default_equals_canonical": tf.encode("سلام دنیا\n\n x ") == can.encode("سلام دنیا\n\n x "),
        "chat_text": chat_txt,
        "chat_ids_equal_hf_json": list(chat_ids) == hf.encode(chat_txt).ids,
        "chat_ids_equal_split_canonical": list(chat_ids) == split_canonical(chat_txt),
        "chat_decode_exact": tf.decode(chat_ids, skip_special_tokens=False) == chat_txt,
        "chat_decode_skip_specials": tf.decode(chat_ids, skip_special_tokens=True),
    }
    print(json.dumps(res["transformers"], ensure_ascii=False, indent=1))
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(REL, "equivalence_full.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
