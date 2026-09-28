# -*- coding: utf-8 -*-
"""Evaluate every (base, k) of the Track B continued-BPE sweep. Nothing here touches the test split.

Per (base, k in {0} + K_GRID), with the materialised sweep/<base>/k<k>/tokenizer.json:
  dev        eval/harness.py run() on dev_strict (gates g1): bytes/token, chars/token, fertility, STRR,
             continued-word rate, robustness, silver morphology, NSL vs the unextended base (k = 0), per source
  G2         self-tokenization of every NEW whole-character token (encode(decode([id])) == [id]); partial-UTF-8
             byte tokens are exempt and counted (R2). The harness G2 over the whole vocabulary at k = 0 and K_MAX.
  R1         support profile of the new tokens: train_D1 frequency under the extended tokenizer itself
             (= 0, < 20, < 100), leaves vs intermediate nodes of the new merge graph, dev usage
  equiv      HF-native encoding == reference continued-BPE encoding (base encoding, then the merge table applied
             by a pure-Python heap BPE with HF's (rank, position) order) on every dev_strict document
  checks     non-interference on the check set (checkset/checkset.json): ids identical to the base? For changed
             documents: every extended token is a concatenation of consecutive base tokens (coarsening) and the
             round trip is exact. Bulk code/English files (checkset/bulk_manifest.json) at k = K_MAX only: no new
             merge fires at K_MAX => none fires at any smaller k (the k-extension is a prefix of the K_MAX one).

    python evaluate.py --bases qwen-3 llama-3 [--force]     -> results/<base>/k<k>/{dev_strict/, trackb_eval.json}
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402

C.env_threads(2)
sys.path.insert(0, C.EVAL_DIR)
import harness as H  # noqa: E402
import adapters as A  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

EVAL_VERSION = "1.0.0"


# ------------------------------------------------------------------------------------------ reference
def rank_table(j: dict):
    vocab = j["model"]["vocab"]
    rank, to = {}, {}
    for r, m in enumerate(j["model"]["merges"]):
        a, b = (m if isinstance(m, list) else m.split(" ", 1))
        ia, ib, ip = vocab.get(a), vocab.get(b), vocab.get(a + b)
        if ia is None or ib is None or ip is None:
            continue
        if (ia, ib) not in rank:
            rank[(ia, ib)] = r
            to[(ia, ib)] = ip
    return rank, to


def merge_all(ids, rank, to):
    """HF tokenizers' Word::merge_all order: repeatedly the valid pair with the lowest (rank, left position)."""
    n = len(ids)
    if n < 2:
        return list(ids)
    sym = list(ids)
    nxt = list(range(1, n + 1)); nxt[-1] = -1
    prv = list(range(-1, n - 1))
    alive = [True] * n
    heap = []
    for i in range(n - 1):
        r = rank.get((sym[i], sym[i + 1]))
        if r is not None:
            heap.append((r, i))
    heapq.heapify(heap)
    while heap:
        r, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1:
            continue
        pair = (sym[i], sym[j])
        if rank.get(pair) != r:
            continue
        sym[i] = to[pair]
        alive[j] = False
        nxt[i] = nxt[j]
        if nxt[j] != -1:
            prv[nxt[j]] = i
        p = prv[i]
        if p != -1:
            rp = rank.get((sym[p], sym[i]))
            if rp is not None:
                heapq.heappush(heap, (rp, p))
        q = nxt[i]
        if q != -1:
            rq = rank.get((sym[i], sym[q]))
            if rq is not None:
                heapq.heappush(heap, (rq, i))
    out, i = [], 0
    while i != -1:
        out.append(sym[i]); i = nxt[i]
    return out


class Reference:
    """Continued-BPE reference encoder: the BASE tokenizer's encoding, then the extended merge table."""

    def __init__(self, base: str, ext_json: dict):
        self.kind = C.BASES[base]["kind"]
        self.base = Tokenizer.from_file(C.base_tokenizer_path(base))
        self.rank, self.to = rank_table(ext_json)
        self.ext_vocab = ext_json["model"]["vocab"]
        self.ignore_merges = bool(ext_json["model"].get("ignore_merges"))

    def encode(self, text: str):
        if self.kind == "spbpe":
            return merge_all(self.base.encode(text, add_special_tokens=False).ids, self.rank, self.to)
        norm, pre, model = self.base.normalizer, self.base.pre_tokenizer, self.base.model
        t = norm.normalize_str(text) if norm is not None else text
        out = []
        for p, _ in pre.pre_tokenize_str(t):
            if self.ignore_merges and p in self.ext_vocab:
                out.append(self.ext_vocab[p]); continue
            out.extend(merge_all([x.id for x in model.tokenize(p)], self.rank, self.to))
        return out


# ------------------------------------------------------------------------------------------- helpers
def tok_bytes_fn(ad):
    def f(i):
        b = ad.token_bytes(i)
        return b if b is not None else b""
    return f


def boundaries(ids, tb):
    out, pos = set(), 0
    for i in ids:
        pos += len(tb(i))
        out.add(pos)
    return out


def check_docs(docs, base_tk, ext_tk, ext_ad, base_ad):
    """Per-document comparison of the base and extended encodings."""
    tb_ext, tb_base = tok_bytes_fn(ext_ad), tok_bytes_fn(base_ad)
    res = collections.Counter()
    changed_ids = []
    texts = [d["text"] for d in docs]
    eb = [e.ids for e in base_tk.encode_batch(texts, add_special_tokens=False)]
    ee = [e.ids for e in ext_tk.encode_batch(texts, add_special_tokens=False)]
    for d, b, e in zip(docs, eb, ee):
        res["docs"] += 1
        res["bytes"] += len(d["text"].encode("utf-8"))
        res["tokens_base"] += len(b)
        res["tokens_ext"] += len(e)
        res["docs_with_arabic"] += int(C.has_arabic(d["text"]))
        if b == e:
            res["identical"] += 1
            continue
        res["changed"] += 1
        res["changed_with_arabic"] += int(C.has_arabic(d["text"]))
        if len(changed_ids) < 20:
            changed_ids.append(d.get("id") or d.get("path"))
        bb, be = boundaries(b, tb_base), boundaries(e, tb_ext)
        res["coarsening_ok"] += int(be <= bb)
        res["roundtrip_ok"] += int(ext_tk.decode(e, skip_special_tokens=False) == d["text"])
    out = dict(res)
    out["changed_examples"] = changed_ids
    out["token_reduction"] = (1 - res["tokens_ext"] / res["tokens_base"]) if res["tokens_base"] else None
    return out


def load_checkset():
    cs = C.load_json(os.path.join(HERE, "checkset", "checkset.json"))
    return cs["docs"]


def load_bulk(norm):
    bm = C.load_json(os.path.join(HERE, "checkset", "bulk_manifest.json"))
    out = {}
    for key in ("bulk_code", "bulk_english"):
        docs = []
        for x in bm[key]:
            if C.sha256_file(x["path"]) != x["sha256"]:
                raise SystemExit("bulk file changed: " + x["path"])
            raw = open(x["path"], encoding="utf-8", errors="strict").read()
            docs.append({"path": x["path"], "text": norm(raw)})
        out[key] = docs
    return out


# ------------------------------------------------------------------------------------------- one run
def eval_one(base: str, k: int, ext: dict, dev_docs, train_texts_n, checkset, bulk, force=False):
    import continued_bpe as CB
    out_dir = os.path.join(C.RESULTS, base, "k%d" % k)
    res_path = os.path.join(out_dir, "trackb_eval.json")
    path = os.path.join(C.SWEEP, base, "k%d" % k, "tokenizer.json")
    sha = C.sha256_file(path)
    man = C.load_json(os.path.join(C.SWEEP, base, "sweep_manifest.json"))
    assert man["k%d" % k]["sha256"] == sha, "sweep file differs from sweep_manifest"
    if os.path.exists(res_path) and not force:
        old = C.load_json(res_path)
        if old.get("tokenizer_sha256") == sha and old.get("eval_version") == EVAL_VERSION:
            C.log(base, k, "up to date")
            return old
    t0 = time.time()
    name = "%s+cbpe%d" % (base, k)
    ad = A.HFAdapter(path=path, name=name)
    base_path = os.path.join(C.SWEEP, base, "k0", "tokenizer.json")
    base_ad = ad if k == 0 else A.HFAdapter(path=base_path, name=base)
    first_new = ext["first_new_id"]
    new_ids = list(range(first_new, first_new + k))
    sel = CB.select(ext, k) if k else []
    assert [m["id"] for m in sel if m["kind"] == "new"] == new_ids
    R = {"base": base, "k": k, "tokenizer_path": path, "tokenizer_sha256": sha, "eval_version": EVAL_VERSION,
         "code_sha256": {f: C.sha256_file(os.path.join(HERE, f)) for f in ("evaluate.py", "continued_bpe.py", "tb_common.py")},
         "frozen": C.verify_frozen(), "vocab_size_total": ad.vocab_size, "id_upper": ad.id_upper, "first_new_id": first_new}
    # ---- dev metrics through the shared harness
    nsl_ref = os.path.join(C.RESULTS, base, "k0", "dev_strict") if k else None
    summ = H.run(ad, data="dev_strict", gates="g1", out_dir=os.path.join(out_dir, "dev_strict"), nsl_ref=nsl_ref,
                 force=force, do_morph=True)
    ov = summ["metrics"]["overall"]
    R["dev"] = {key: ov.get(key) for key in ("docs", "bytes", "chars", "words", "tokens", "bytes_per_token",
                                            "chars_per_token", "fertility", "tokens_per_word", "continued_word_rate",
                                            "strr", "g1_pass_docs", "g1_fail_docs", "unk_tokens", "bytes_per_token_lines",
                                            "vocab_used", "renyi_eff_a2.5")}
    R["dev"]["by_source"] = {s: {key: v.get(key) for key in ("docs", "bytes", "tokens", "bytes_per_token", "fertility", "strr")}
                             for s, v in summ["metrics"]["by_source"].items()}
    R["dev"]["by_variety"] = {s: {key: v.get(key) for key in ("docs", "bytes", "tokens", "bytes_per_token", "fertility", "strr")}
                              for s, v in summ["metrics"]["by_variety"].items()}
    R["dev"]["robustness"] = {p: {key: v.get(key) for key in ("rel_token_change", "seg_change_rate_affected")}
                              for p, v in ov["robustness"].items()}
    R["dev"]["nsl_vs_base"] = summ.get("nsl", {}).get("overall") if k else 1.0
    R["dev"]["morphology"] = {s: {key: summ.get("morphology", {}).get(s, {}).get(key)
                                  for key in ("boundary_f1", "morphscore", "single_token_rate", "tokens_per_word")}
                              for s in ("silver_high", "silver_low")}
    R["gates"] = {"G1": summ["gates"]["G1"]}
    C.log(name, "dev: %.4f bytes/token, fertility %.4f, STRR %.4f, G1 %d/%d" % (
        ov["bytes_per_token"], ov["fertility"], ov["strr"], ov["g1_pass_docs"], ov["docs"]))
    # ---- G2 on the new tokens (and the harness G2 over the whole vocabulary at k = 0 and K_MAX)
    fails, partial, tested = [], [], 0
    for i in new_ids:
        txt = H.token_text(ad, i)
        if txt is None:
            partial.append({"id": i, "hex": ad.token_bytes(i).hex(" ")})
            continue
        tested += 1
        enc = ad.encode_isolated(txt)
        if list(enc) != [i]:
            fails.append({"id": i, "text": txt, "encodes_to": list(enc)[:12]})
    R["gates"]["G2_new"] = {"pass": not fails, "tested": tested, "failures": len(fails), "failure_list": fails[:100],
                            "exempt_partial_utf8": len(partial), "partial_utf8_list": partial}
    if k in (0, C.K_MAX):
        g2 = H.gate_g2(ad)
        R["gates"]["G2_full_vocab"] = {x: g2[x] for x in ("pass", "tested", "failures", "exempt_partial_utf8",
                                                          "superword_tokens_untested")}
        R["gates"]["G2_full_vocab"]["failure_examples"] = g2["failure_list"][:10]
    # ---- train support (R1) and CPT data volume
    cnt, ntok, tinfo = H.train_counts(ad, "train_D1")
    R["train"] = {"tokens": int(ntok), "bytes": int(train_texts_n["bytes"]), "bytes_per_token": train_texts_n["bytes"] / ntok,
                  "seconds": tinfo["seconds"], "sha256": tinfo["sha256"]}
    if k:
        freq = cnt[np.asarray(new_ids)]
        comps = set()
        for m in sel:
            comps.add(ad.tk.token_to_id(m["left"])); comps.add(ad.tk.token_to_id(m["right"]))
        leaf = np.asarray([i not in comps for i in new_ids])
        partial_set = {p["id"] for p in partial}
        part = np.asarray([i in partial_set for i in new_ids])
        prof = H._profile(freq, leaf)
        prof["share_train_tokens_that_are_new"] = float(freq.sum() / ntok)
        prof["partial_utf8_new"] = int(part.sum())
        prof["partial_utf8_new_freq_lt20"] = int(((freq < 20) & part).sum())
        prof["lt100_intermediate"] = int(((freq < 100) & ~leaf).sum())
        prof["lt100_leaves"] = int(((freq < 100) & leaf).sum())
        prof["deciles_train_freq"] = [float(x) for x in np.percentile(freq, [10, 25, 50, 75, 90])]
        order = np.argsort(freq, kind="stable")
        prof["rarest"] = [{"id": int(new_ids[j]), "text": H.token_text(ad, new_ids[j]), "train_freq": int(freq[j]),
                           "leaf": bool(leaf[j])} for j in order[:20]]
        R["R1_new"] = prof
        # dev usage of the new tokens
        dcnt = np.zeros(ad.id_upper, np.int64)
        for e in ad.tk.encode_batch([d["text"] for d in dev_docs], add_special_tokens=False):
            a = np.asarray(e.ids, dtype=np.int64)
            if a.size:
                dcnt += np.bincount(a, minlength=ad.id_upper)[:ad.id_upper]
        dfreq = dcnt[np.asarray(new_ids)]
        R["dev_new_usage"] = {"new_tokens_seen_on_dev": int((dfreq > 0).sum()),
                              "share_new_seen_on_dev": float((dfreq > 0).mean()),
                              "share_dev_tokens_that_are_new": float(dfreq.sum() / dcnt.sum())}
        R["merge_kinds"] = dict(collections.Counter(m["kind"] for m in sel))
    # R1 over all learned tokens (base + new) as the harness defines it
    r1 = H.prop_r1(ad, cnt, tinfo)
    R["R1_all_learned"] = r1["all_learned"]
    C.log(name, "train: %d tokens (%.4f bytes/token)" % (ntok, R["train"]["bytes_per_token"]),
          ("new<100: %d (%.1f%%)" % (R["R1_new"]["train_freq_lt100"], R["R1_new"]["pct_lt100"])) if k else "")
    # ---- equivalence with the reference continued-BPE encoder, all dev docs
    if k:
        ref = Reference(base, json.loads(open(path, encoding="utf-8").read()))
        mism = []
        t1 = time.time()
        for d in dev_docs:
            hf = ad.encode(d["text"])
            rf = ref.encode(d["text"])
            if hf != rf:
                mism.append(d["uid"])
        R["equivalence"] = {"docs": len(dev_docs), "mismatch_docs": len(mism), "mismatch_uids": mism[:20],
                            "pass": not mism, "seconds": round(time.time() - t1, 1),
                            "reference": "base tokenizer encoding, then the extended merge table applied by a pure-Python "
                                         "heap BPE ((rank, left position) order, as tokenizers' Word::merge_all)"
                                         + ("; ignore_merges shortcut for whole pretokens in the vocabulary"
                                            if ref.ignore_merges else "")}
        del ref
        C.log(name, "equivalence: %d/%d docs differ" % (len(mism), len(dev_docs)))
    # ---- non-interference checks
    if k:
        R["checks"] = {s: check_docs(docs, base_ad.tk, ad.tk, ad, base_ad) for s, docs in checkset.items()}
        if k == C.K_MAX:
            for s, docs in bulk.items():
                R["checks"][s] = check_docs(docs, base_ad.tk, ad.tk, ad, base_ad)
        C.log(name, "checks:", {s: (v["changed"] if "changed" in v else 0, v["docs"]) for s, v in R["checks"].items()})
    R["seconds"] = round(time.time() - t0, 1)
    C.dump_json(H.clean(R), res_path)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev_docs = C.load_view("dev_strict")
    tr = C.load_view("train_D1")
    train_n = {"docs": len(tr), "bytes": sum(len(d["text"].encode("utf-8")) for d in tr)}
    del tr
    checkset = load_checkset()
    bulk = load_bulk(C.normalizer())
    C.log("bulk:", {s: len(v) for s, v in bulk.items()})
    for base in a.bases:
        ext = C.load_json(os.path.join(C.WORK, base, "extension.json"))
        for k in (a.ks or [0] + C.K_GRID):
            eval_one(base, k, ext, dev_docs, train_n, checkset, bulk, force=a.force)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
