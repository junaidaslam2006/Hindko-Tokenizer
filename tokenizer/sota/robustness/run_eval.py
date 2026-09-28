# -*- coding: utf-8 -*-
"""Evaluate the 11 tokenizers of rcommon.TOKENIZERS on every set of build_sets.py and on the perturbation suite.

    set PYTHONIOENCODING=utf-8
    python run_eval.py --shard K --of 3          (K = 0, 1, 2 in parallel; one thread each)
    python run_eval.py --only released           (one tokenizer)

Per tokenizer -> results/<key>.json:
  sets[set][form]   form = norm (hp.normalize applied) / raw (as found): docs, bytes, chars, words, tokens,
                    bytes_per_token, chars_per_token, fertility (harness definition: tokens overlapping each
                    whitespace word, from in-context offsets), lossless_docs (decode(encode(x)) == x), unk,
                    byte_fallback_tokens (<0xNN> pieces; None for byte-level BPE, where bytes are ordinary tokens)
  test_strict       in-distribution reference on the same 491 documents
  perturb[p]        p in perturb_extra.ORDER: harness robustness quantities (rel_token_change, words_affected,
                    extra_tokens_per_affected_word, seg_change_rate, seg_change_rate_affected, ...), computed with
                    eval/harness.char_maps + seg_compare exactly as harness.doc_eval does
  perturb_norm      arabic_keyboard followed by hp.normalize: token count only (normalize is not alignable)
  codemixed         tokens of the 100 mixed lines and of their pure-Hindko base sentences
  examples          token strings of the 3 card sentences
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcommon as C  # noqa: E402
import numpy as np  # noqa: E402
import harness as H  # noqa: E402
import perturb_extra as PX  # noqa: E402

SHARDS = [["released", "roberta-urdu", "gemma-3", "bloom"],
          ["mingram48k", "mt5", "gpt-4o", "llama-4"],
          ["llama-3", "qwen-3.5", "deepseek-v3"]]
SET_FILES = ["ooc_gotquestions_southern_hindko", "ooc_translatewiki_hno", "ooc_chup_di_kahani_pdf",
             "ooc_finepdfs_hindko_org", "lang_urdu", "lang_punjabi_shahmukhi", "lang_saraiki", "lang_pashto",
             "lang_english", "lang_python_code", "mixed_natural_test"]


def byte_fb_ids(ad):
    if getattr(ad, "byte_level", False):
        return None
    return ad.base_ids


def eval_texts(ad, notext, texts):
    bfb = byte_fb_ids(ad)
    a = {"docs": 0, "bytes": 0, "chars": 0, "words": 0, "tokens": 0, "pieces": 0, "lossless_docs": 0, "unk": 0,
         "byte_fallback_tokens": 0 if bfb is not None else None}
    for t in texts:
        ids, st, en = ad.encode_offsets(t)
        ids_a = np.asarray(ids, dtype=np.int64)
        st = np.asarray(st, dtype=np.int64)
        en = np.asarray(en, dtype=np.int64)
        nt = notext[ids_a] if len(ids_a) else np.zeros(0, bool)
        ws, we = H.word_spans(t)
        pieces = H.word_piece_counts(ws, we, st, en, nt)
        a["docs"] += 1
        a["bytes"] += len(t.encode("utf-8"))
        a["chars"] += len(t)
        a["words"] += len(ws)
        a["tokens"] += len(ids)
        a["pieces"] += int(pieces.sum())
        a["lossless_docs"] += int(ad.decode(ids) == t)
        a["unk"] += sum(1 for i in ids if i in ad.unk_ids)
        if bfb is not None:
            a["byte_fallback_tokens"] += sum(1 for i in ids if i in bfb)
    a["bytes_per_token"] = a["bytes"] / a["tokens"] if a["tokens"] else None
    a["chars_per_token"] = a["chars"] / a["tokens"] if a["tokens"] else None
    a["fertility"] = a["pieces"] / a["words"] if a["words"] else None
    return a


def prep_test(test):
    prep = []
    for d in test:
        t = d["text"]
        ws, we = H.word_spans(t)
        L = len(t)
        c = np.arange(L)
        ow = np.searchsorted(ws, c, side="right") - 1 if len(ws) else np.full(L, -1)
        ok = ow >= 0
        ok[ok] = c[ok] < we[ow[ok]]
        ow = np.where(ok, ow, -1)
        perts = {}
        for p in PX.ORDER:
            pp = PX.ALL[p](t)
            if p in PX.EXTRA:
                PX.check(t, pp)
            perts[p] = (pp, H.nonws_prefix(pp.text) if pp.n_edits else None)
        ak = perts["arabic_keyboard"][0]
        prep.append({"ws": ws, "we": we, "ow": ow, "nw": H.nonws_prefix(t), "perts": perts,
                     "ak_norm": C.normalize(ak.text) if ak.n_edits else t, "bytes": len(t.encode("utf-8"))})
    return prep


def eval_perturb(ad, notext, test, prep):
    base = {"docs": 0, "bytes": 0, "words": 0, "tokens": 0, "pieces": 0, "lossless_docs": 0}
    acc = {p: {k: 0 for k in ("tok", "edits", "cmp", "chg", "aff", "achg", "van", "docs_affected", "pbytes")}
           for p in PX.ORDER}
    ak_norm_tok = 0
    for d, pr in zip(test, prep):
        t = d["text"]
        ids, st, en = ad.encode_offsets(t)
        ids_a = np.asarray(ids, dtype=np.int64)
        st = np.asarray(st, dtype=np.int64)
        en = np.asarray(en, dtype=np.int64)
        nt = notext[ids_a] if len(ids_a) else np.zeros(0, bool)
        pieces = H.word_piece_counts(pr["ws"], pr["we"], st, en, nt)
        base["docs"] += 1
        base["bytes"] += pr["bytes"]
        base["words"] += len(pr["ws"])
        base["tokens"] += len(ids)
        base["pieces"] += int(pieces.sum())
        base["lossless_docs"] += int(ad.decode(ids) == t)
        omaps = H.char_maps(len(t), st, en, ~nt)
        for p in PX.ORDER:
            pp, pnw = pr["perts"][p]
            a = acc[p]
            if pp.n_edits == 0:
                r = {"tok": len(ids), "edits": 0, "cmp": len(pr["ws"]), "chg": 0, "aff": 0, "achg": 0, "van": 0}
                a["pbytes"] += pr["bytes"]
            else:
                pids, pst, pen = ad.encode_offsets(pp.text)
                pids_a = np.asarray(pids, dtype=np.int64)
                pst = np.asarray(pst, dtype=np.int64)
                pen = np.asarray(pen, dtype=np.int64)
                pkeep = ~notext[pids_a] if len(pids_a) else np.zeros(0, bool)
                pmaps = H.char_maps(len(pp.text), pst, pen, pkeep)
                r = H.seg_compare(pp, pr["ws"], pr["we"], pr["ow"], omaps, pr["nw"], pmaps, pnw)
                r["tok"] = len(pids)
                r["edits"] = pp.n_edits
                a["docs_affected"] += 1
                a["pbytes"] += len(pp.text.encode("utf-8"))
            for k, v in r.items():
                a[k] += v
        ak_norm_tok += len(ad.encode(pr["ak_norm"])) if pr["perts"]["arabic_keyboard"][0].n_edits else len(ids)
    n = base["tokens"]
    base["bytes_per_token"] = base["bytes"] / n
    base["fertility"] = base["pieces"] / base["words"]
    out = {}
    for p, a in acc.items():
        out[p] = {"edits": a["edits"], "docs_affected": a["docs_affected"], "tokens": a["tok"],
                  "rel_token_change": (a["tok"] - n) / n, "words_compared": a["cmp"], "words_affected": a["aff"],
                  "words_vanished": a["van"],
                  "extra_tokens_per_affected_word": (a["tok"] - n) / a["aff"] if a["aff"] else None,
                  "seg_change_rate": a["chg"] / a["cmp"] if a["cmp"] else None,
                  "seg_change_rate_affected": a["achg"] / a["aff"] if a["aff"] else None,
                  "seg_changed_unaffected_words": a["chg"] - a["achg"],
                  "perturbed_bytes": a["pbytes"], "perturbed_bytes_per_token": a["pbytes"] / a["tok"]}
    norm = {"arabic_keyboard_then_normalize": {"tokens": ak_norm_tok, "rel_token_change": (ak_norm_tok - n) / n}}
    return base, out, norm


FRAG = chr(0x25AF)      # WHITE VERTICAL RECTANGLE: one byte of an incomplete UTF-8 character (bidi-neutral)


def render(ad, text):
    """Token strings for display: a space the token carries is shown as U+2581; each byte that is only part of a
    character (byte-level BPE splitting a letter) is shown as U+25AF; the raw bytes are kept in 'hex'."""
    ids = ad.encode(text)
    toks, hexes = [], []
    for i in ids:
        b = ad.token_bytes(i)
        if b is None:
            s = "<id%d>" % i
            hexes.append(None)
        else:
            s = b.decode("utf-8", errors="backslashreplace")
            s = re.sub(r"\\x[0-9a-f]{2}", FRAG, s)
            hexes.append(b.hex())
        toks.append(s.replace(" ", chr(0x2581)).replace("\n", "\\n"))
    return {"ids": list(map(int, ids)), "tokens": toks, "hex": hexes, "n": len(ids), "lossless": ad.decode(ids) == text}


def examples_only(keys):
    ex = C.read_jsonl(os.path.join(C.SETS, "examples.jsonl"))
    for k in keys:
        p = os.path.join(C.RESULTS, k + ".json")
        res = json.load(open(p, encoding="utf-8"))
        ad = C.load_tokenizer(k)
        res["examples"] = [{"id": x["id"], **render(ad, x["text_raw"])} for x in ex]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(H.clean(res), f, ensure_ascii=False, indent=1)
        C_log(k, "examples re-rendered")


def run_one(key, test, prep):
    t0 = time.time()
    ad = C.load_tokenizer(key)
    notext = ad.notext
    res = {"key": key, "display": C.DISPLAY[key], "identity": H.clean(ad.identity()), "sets": {}}
    for s in SET_FILES:
        rows = C.read_jsonl(os.path.join(C.SETS, s + ".jsonl"))
        r = {"norm": eval_texts(ad, notext, [x["text_norm"] for x in rows])}
        if any(x["text_raw"] != x["text_norm"] for x in rows):
            r["raw"] = eval_texts(ad, notext, [x["text_raw"] for x in rows])
        else:
            r["raw"] = r["norm"]
        res["sets"][s] = r
        C_log(key, s, "%.2f b/tok" % r["norm"]["bytes_per_token"], "%.0fs" % (time.time() - t0))
    base, pert, pnorm = eval_perturb(ad, notext, test, prep)
    res["test_strict"] = base
    res["perturb"] = pert
    res["perturb_norm"] = pnorm
    C_log(key, "perturbations done", "%.0fs" % (time.time() - t0))
    cm = C.read_jsonl(os.path.join(C.SETS, "codemixed.jsonl"))
    mixed = eval_texts(ad, notext, [x["text_raw"] for x in cm])
    basecm = eval_texts(ad, notext, [x["base"] for x in cm])
    res["codemixed"] = {"mixed": mixed, "base": basecm,
                        "added_bytes": mixed["bytes"] - basecm["bytes"], "added_tokens": mixed["tokens"] - basecm["tokens"],
                        "per_line_tokens": [len(ad.encode(x["text_raw"])) for x in cm]}
    ex = C.read_jsonl(os.path.join(C.SETS, "examples.jsonl"))
    res["examples"] = [{"id": x["id"], **render(ad, x["text_raw"])} for x in ex]
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(C.RESULTS, key + ".json"), "w", encoding="utf-8") as f:
        json.dump(H.clean(res), f, ensure_ascii=False, indent=1)
    C_log(key, "DONE", "%.0fs" % (time.time() - t0))


def C_log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--of", type=int, default=3)
    ap.add_argument("--only")
    ap.add_argument("--examples-only", action="store_true", help="re-render the examples of existing results")
    a = ap.parse_args()
    if a.examples_only:
        examples_only([a.only] if a.only else C.TOK_KEYS)
        return
    keys = [a.only] if a.only else SHARDS[a.shard]
    os.makedirs(C.RESULTS, exist_ok=True)
    test = C.load_test()
    prep = prep_test(test)
    C_log("prepared", len(test), "test docs;", "tokenizers:", keys)
    for k in keys:
        try:
            run_one(k, test, prep)
        except Exception as e:  # noqa: BLE001
            import traceback
            C_log("ERROR", k, repr(e))
            traceback.print_exc()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
