# -*- coding: utf-8 -*-
"""SuperBPE-specific G2 work that eval/harness.py does not do (PLAN 4.3).

    python gates_sbpe.py OUT_DIR            -> OUT_DIR/g2_multiword.json

PLAN 4.3 G2: "self-tokenization, encode(decode([id])) == [id], for every learned non-special id whose bytes are
valid UTF-8 ... SuperBPE multi-word tokens are tested on the shortest train chunk that contains them".
harness.gate_g2 lists multi-word tokens (whitespace after the first character) as untested. For each of them
this script records three witnesses of reachability:
  (i)   isolated: encode(token text) == [id]. The token text is itself a valid stage-2 chunk (a run of
        words, or a word tail followed by words), so this is the ordinary G2 test applied to superwords;
  (ii)  chunk_raw: the shortest distinct train_D1 S2 chunk (characters; ties: first occurrence) that
        contains the token TEXT as a raw substring; PASS iff the id occurs in that chunk's encoding.
        This is the PLAN clause read literally. It also "finds" the text where its last word is only the
        prefix of a longer word (' تے اس' inside ' تے اسی'), which no tokenizer could pass;
  (iii) chunk_aligned: the shortest distinct train_D1 chunk whose STAGE-1 segmentation contains the token's
        stage-1 id sequence (so the token is a candidate there); PASS iff the id occurs in its encoding.
        A failure here means only that a competing superword wins in that context
        (' خاص طور' + ' تے' instead of ' خاص' + ' طور تے');
  plus the token's train frequency (whole train_D1 documents, as an LM sees them).
Verdict used for the gate (documented deviation, SUPERBPE.md): a multi-word token is REACHABLE iff (i) or
(iii) passes or its train frequency is > 0; G2 passes iff every multi-word token is reachable. The literal
reading (ii) is reported beside it.
"""
import bisect
import collections
import json
import os
import pickle
import sys
import time

import numpy as np
import regex

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import ref_encoder as R  # noqa: E402


def multiword_ids(fv, specials):
    out = []
    for s, i in fv.items():
        if s in specials:
            continue
        b = C.bl_to_bytes(s)
        try:
            txt = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(c.isspace() for c in txt[1:]) and not txt.isspace():
            out.append((i, txt))
    return sorted(out)


def shortest_index(items):
    """items: list of (key_string, sort_key). Returns big string, starts, ordered keys."""
    order = sorted(range(len(items)), key=lambda k: items[k][1])
    keys = [items[k][0] for k in order]
    starts, pos = [], 0
    for s in keys:
        starts.append(pos)
        pos += len(s) + 1
    return "\x00".join(keys), starts, order


def main():
    out_dir = sys.argv[1]
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(C.HERE, out_dir)
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    from tokenizers import Tokenizer
    C.verify_frozen()
    t0 = time.time()
    path = os.path.join(out_dir, "tokenizer.json")
    tk = Tokenizer.from_file(path)
    fv = tk.get_vocab(with_added_tokens=True)
    specials = set(C.SPECIAL_TOKENS)
    mw = multiword_ids(fv, specials)
    pua = json.load(open(os.path.join(out_dir, "stage2_pua.json"), encoding="utf-8"))
    s1 = json.load(open(pua["stage1"], encoding="utf-8"))
    id2tok1 = {i: s for s, i in s1["model"]["vocab"].items()}
    tb1 = lambda i: C.bl_to_bytes(id2tok1[i])
    # stage-1 id sequence of every final token (stage-1 tokens: itself; stage-2 tokens: from the PUA merges)
    seq_of_text = {}
    for sa, sb in pua["merges"]:
        seq = tuple(sa) + tuple(sb)
        seq_of_text.setdefault(b"".join(tb1(i) for i in seq).decode("utf-8", "replace"), seq)
    docs, info = C.load_view("train_D1", check_normalized=False)
    texts = [d["text"] for d in docs]
    # (ii) raw substring index over distinct multi-word chunk texts
    rx = regex.compile(C.onig_to_py(C.S2))
    first = {}
    for t in texts:
        for ch in R.py_pretokens(rx, t):
            if " " in ch.strip() and ch not in first:
                first[ch] = len(first)
    raw_items = [(c, (len(c), k)) for c, k in first.items()]
    big_raw, starts_raw, order_raw = shortest_index(raw_items)
    # (iii) stage-1-aligned index over distinct train chunks with >= 2 stage-1 tokens (chunk cache of the build)
    sha1 = C.sha256_file(pua["stage1"])
    cache = os.path.join(C.HERE, "work", "chunks_%s.pkl" % sha1[:16])
    if not os.path.exists(cache):
        raise SystemExit("chunk cache %s missing: run build_stage2.py first" % cache)
    with open(cache, "rb") as f:
        cc = pickle.load(f)
    assert cc["stage1_sha256"] == sha1 and cc["S2"] == C.S2
    chunk_seqs = list(cc["counts"])                       # insertion order = first occurrence in train_D1
    chunk_text = [b"".join(tb1(i) for i in s).decode("utf-8") for s in chunk_seqs]
    al_items = [("".join(chr(C.PUA_BASE + i) for i in s), (len(chunk_text[k]), k)) for k, s in enumerate(chunk_seqs)]
    big_al, starts_al, order_al = shortest_index(al_items)
    C.log("%d multi-word tokens; %d distinct multi-word train chunks (raw), %d chunk types with >= 2 stage-1 tokens"
          % (len(mw), len(first), len(chunk_seqs)))
    cnt = np.zeros(max(fv.values()) + 1, np.int64)
    for k in range(0, len(texts), 500):
        for e in tk.encode_batch(texts[k:k + 500], add_special_tokens=False):
            cnt += np.bincount(np.asarray(e.ids, dtype=np.int64), minlength=len(cnt))[:len(cnt)]
    raw_keys = [raw_items[k][0] for k in order_raw]
    rows = []
    for i, txt in mw:
        iso = tk.encode(txt, add_special_tokens=False).ids
        r = {"id": i, "text": txt, "words": C.n_words(txt.encode("utf-8")), "train_freq": int(cnt[i]),
             "isolated_ok": iso == [i]}
        p = big_raw.find(txt)
        if p < 0:
            r["chunk_raw_ok"], r["chunk_raw"] = False, None
        else:
            ch = raw_keys[bisect.bisect_right(starts_raw, p) - 1]
            r["chunk_raw"] = ch
            r["chunk_raw_ok"] = i in tk.encode(ch, add_special_tokens=False).ids
        seq = seq_of_text.get(txt)
        if seq is None:
            r["chunk_aligned_ok"], r["chunk_aligned"] = False, None
        else:
            p = big_al.find("".join(chr(C.PUA_BASE + x) for x in seq))
            if p < 0:
                r["chunk_aligned_ok"], r["chunk_aligned"] = False, None
            else:
                k = order_al[bisect.bisect_right(starts_al, p) - 1]
                r["chunk_aligned"] = chunk_text[k]
                enc = tk.encode(chunk_text[k], add_special_tokens=False).ids
                r["chunk_aligned_ok"] = i in enc
                if not r["chunk_aligned_ok"]:
                    r["chunk_aligned_encoding"] = [tk.id_to_token(x) and C.bl_to_bytes(tk.id_to_token(x)).decode("utf-8", "replace") for x in enc[:12]]
        r["reachable"] = r["isolated_ok"] or r["chunk_aligned_ok"] or r["train_freq"] > 0
        rows.append(r)
    n = len(rows)
    fail_raw = [r for r in rows if not r["chunk_raw_ok"]]
    fail_al = [r for r in rows if not r["chunk_aligned_ok"]]
    unreach = [r for r in rows if not r["reachable"]]
    res = {"tokenizer": path, "tokenizer_sha256": C.sha256_file(path), "train_view": info,
           "verdict_rule": "reachable iff isolated self-tokenization passes, or the id occurs in the encoding of the "
                           "shortest train chunk whose stage-1 segmentation contains the token (chunk_aligned), or "
                           "train frequency > 0; G2 (multi-word part) passes iff all are reachable",
           "multiword_tokens": n, "pass": not unreach, "unreachable": len(unreach),
           "isolated_failures": sum(1 for r in rows if not r["isolated_ok"]),
           "chunk_aligned_failures": len(fail_al), "chunk_raw_failures_literal_PLAN_clause": len(fail_raw),
           "chunk_raw_failures_that_pass_isolated": sum(1 for r in fail_raw if r["isolated_ok"]),
           "chunk_raw_failures_train_freq_min_median_max": ([int(np.min([r["train_freq"] for r in fail_raw])),
                                                             float(np.median([r["train_freq"] for r in fail_raw])),
                                                             int(np.max([r["train_freq"] for r in fail_raw]))]
                                                            if fail_raw else None),
           "chunk_aligned_failures_train_freq_min_median_max": ([int(np.min([r["train_freq"] for r in fail_al])),
                                                                 float(np.median([r["train_freq"] for r in fail_al])),
                                                                 int(np.max([r["train_freq"] for r in fail_al]))]
                                                                if fail_al else None),
           "train_freq_eq0": sum(1 for r in rows if r["train_freq"] == 0),
           "train_freq_lt20": sum(1 for r in rows if r["train_freq"] < 20),
           "train_freq_lt100": sum(1 for r in rows if r["train_freq"] < 100),
           "words_hist": dict(sorted(collections.Counter(r["words"] for r in rows).items())),
           "distinct_multiword_train_chunks_raw": len(first), "chunk_types_ge2_stage1_tokens": len(chunk_seqs),
           "unreachable_list": unreach[:50], "chunk_raw_failure_examples": fail_raw[:40],
           "chunk_aligned_failure_examples": fail_al[:40],
           "rarest": sorted(rows, key=lambda r: r["train_freq"])[:25],
           "seconds": round(time.time() - t0, 1), "code_sha256": C.sha256_file(os.path.abspath(__file__))}
    json.dump(res, open(os.path.join(out_dir, "g2_multiword.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("G2 multi-word %s: %d tokens; unreachable %d; isolated fail %d; chunk_aligned fail %d; chunk_raw (literal) "
          "fail %d; freq<20 %d (%.0fs)" % (out_dir, n, len(unreach), res["isolated_failures"], len(fail_al),
                                            len(fail_raw), res["train_freq_lt20"], res["seconds"]))


if __name__ == "__main__":
    main()
