# -*- coding: utf-8 -*-
"""Write the ready-to-use extended tokenizer folders at the chosen knee k of each base (results/knee.json).

deliver/<base>-hindko-cbpe<k>/
  tokenizer.json            base tokenizer.json + k new model tokens + their merges (sweep/<base>/k<k>/)
  tokenizer_config.json     copy of the base's (new tokens are model tokens, not added tokens); Qwen3.5 only: tokenizer_class
                            set to PreTrainedTokenizerFast (CONFIG_OVERRIDES)
  special_tokens_map.json, added_tokens.json   copied when the base has them
  new_tokens.jsonl          one row per new token: id, vocab string, text, merge (left, right), base-token
                            decomposition (base_ids, base_nbytes) for embedding initialisation, train/dev frequency
  init_embeddings.py        the PLAN 9.3 initialisation (copy of trackb/init_embeddings.py)
  EXTENSION.json            provenance, sha256 of every file, metrics and gates at this k
  README.md                 short usage notes
tokenizer.model           (SentencePiece bases with a base .model, i.e. Gemma-3) the analogous extended SentencePiece
                            model, written only if it encodes every train_D1, dev_strict and check-set document exactly
                            like tokenizer.json (the base's own .model is never copied: it lacks the new pieces)
Each folder is verified: tokenizers and transformers AutoTokenizer load it, len(tokenizer) = base + k, both give the
same ids on dev_strict, every special token keeps its id, the chat template (if any) renders, round trip exact.

    python deliver.py [--bases ...] [--k K]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402
import continued_bpe as CB  # noqa: E402

C.env_threads(2)
from tokenizers import Tokenizer  # noqa: E402

FOLDER_NAME = {"qwen-3": "qwen3", "qwen-3.5": "qwen3.5", "llama-3": "llama3", "gemma-3": "gemma3", "gemma-4": "gemma4"}
COPY = ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
# Measured with transformers 5.3.0: the Qwen3.5 base config names "Qwen2Tokenizer", whose class rebuilds the
# pre-tokenizer with the Qwen2 regex ([^\r\n\p{L}\p{N}]?\p{L}+, marks split off) instead of the Qwen3.5 regex of the
# tokenizer.json ([\p{L}\p{M}]+); even the UNEXTENDED Qwen3.5 then encodes only 650/836 dev_strict documents like its
# own tokenizer.json. "PreTrainedTokenizerFast" loads tokenizer.json as it is (transformers 4.x and 5.x).
CONFIG_OVERRIDES = {"qwen-3.5": {"set": {"tokenizer_class": "PreTrainedTokenizerFast"},
                                 "reason": "the base's 'Qwen2Tokenizer' makes transformers 5.3.0 replace the "
                                           "tokenizer.json pre-tokenizer regex with the Qwen2 one"}}


def decompositions(base, ext, k):
    """base-id decomposition of every new token: the concatenation of its merge components, recursively."""
    j = C.load_base_json(base)
    vocab = j["model"]["vocab"]
    kind = C.BASES[base]["kind"]
    dec = {}

    def of(s):
        if s in vocab:
            return [vocab[s]]
        return dec[s]
    for m in CB.select(ext, k):
        if m["kind"] == "new":
            dec[m["result"]] = of(m["left"]) + of(m["right"])
    inv = {i: s for s, i in vocab.items()}
    nbytes = {}
    for s, ids in dec.items():
        for i in ids:
            if i not in nbytes:
                nbytes[i] = len(CB.token_text_bytes(kind, inv[i]))
    return dec, nbytes


def counts(tk, texts, n):
    c = np.zeros(n, np.int64)
    for s in range(0, len(texts), 500):
        for e in tk.encode_batch(texts[s:s + 500], add_special_tokens=False):
            a = np.asarray(e.ids, dtype=np.int64)
            if a.size:
                c += np.bincount(a, minlength=n)[:n]
    return c


def deliver(base, k, train_texts, dev_docs):
    ext = C.load_json(os.path.join(C.WORK, base, "extension.json"))
    ev = C.load_json(os.path.join(C.RESULTS, base, "k%d" % k, "trackb_eval.json"))
    src = os.path.join(C.SWEEP, base, "k%d" % k, "tokenizer.json")
    sha = C.sha256_file(src)
    assert sha == ev["tokenizer_sha256"], "sweep file differs from the evaluated one"
    name = "%s-hindko-cbpe%d" % (FOLDER_NAME[base], k)
    out = os.path.join(C.DELIVER, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    shutil.copyfile(src, os.path.join(out, "tokenizer.json"))
    copied = {}
    for f in COPY:
        p = os.path.join(C.BASES[base]["dir"], f)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(out, f))
            copied[f] = C.sha256_file(p)
    config_changes = CONFIG_OVERRIDES.get(base)
    if config_changes:
        cp = os.path.join(out, "tokenizer_config.json")
        cfg = C.load_json(cp)
        cfg.update(config_changes["set"])
        with open(cp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    shutil.copyfile(os.path.join(HERE, "init_embeddings.py"), os.path.join(out, "init_embeddings.py"))
    # SentencePiece bases: the analogous extended tokenizer.model (new pieces appended as NORMAL pieces with scores
    # below every base piece, in learning order; sp_model_check.py). Delivered only if it encodes every train_D1,
    # dev_strict and check-set document exactly like the extended tokenizer.json.
    tk = Tokenizer.from_file(src)
    n = tk.get_vocab_size(with_added_tokens=True)
    sp_info = None
    if os.path.exists(os.path.join(C.BASES[base]["dir"], "tokenizer.model")):
        import sentencepiece as spm
        import sp_model_check as SPC
        import evaluate as EV
        proto = SPC.build_proto(base, ext, k)
        sp = spm.SentencePieceProcessor(model_proto=proto)
        cs = EV.load_checkset()
        bulk = EV.load_bulk(C.normalizer())
        groups = {"train_D1": train_texts, "dev_strict": [d["text"] for d in dev_docs]}
        groups.update({s: [d["text"] for d in v] for s, v in cs.items()})
        groups.update({s: [d["text"] for d in v] for s, v in bulk.items()})
        agree = {}
        for g, texts in groups.items():
            same = 0
            for st in range(0, len(texts), 500):
                hf = [e.ids for e in tk.encode_batch(texts[st:st + 500], add_special_tokens=False)]
                spi = sp.encode(texts[st:st + 500])
                same += sum(int(a == b) for a, b in zip(hf, spi))
            agree[g] = {"docs": len(texts), "identical": same}
        ok = all(v["docs"] == v["identical"] for v in agree.values())
        sp_info = {"agreement_with_tokenizer_json": agree, "delivered": ok,
                   "construction": "base tokenizer.model + HF-only added tokens beyond it as USER_DEFINED pieces + the k "
                                   "new pieces as NORMAL pieces with scores min_base_score - 1 - i (learning order)"}
        if ok:
            with open(os.path.join(out, "tokenizer.model"), "wb") as f:
                f.write(proto)
        C.log(base, "SentencePiece model agreement:", json.dumps(agree))
    # new_tokens.jsonl
    ctr = counts(tk, train_texts, n)
    cdv = counts(tk, [d["text"] for d in dev_docs], n)
    base_tk = Tokenizer.from_file(C.base_tokenizer_path(base))
    dec, nbytes = decompositions(base, ext, k)
    kind = C.BASES[base]["kind"]
    sel = CB.select(ext, k)
    comps = {m["left"] for m in sel} | {m["right"] for m in sel}
    rows = []
    for rank, m in enumerate(x for x in sel if x["kind"] == "new"):
        b = CB.token_text_bytes(kind, m["result"])
        try:
            text = b.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        iso = base_tk.encode(text, add_special_tokens=False).ids if text is not None else None
        rows.append({"id": m["id"], "token": m["result"], "text": text, "hex": b.hex(), "left": m["left"],
                     "right": m["right"], "new_rank": rank, "base_ids": dec[m["result"]],
                     "base_nbytes": [nbytes[i] for i in dec[m["result"]]], "base_ids_isolated": iso,
                     "train_freq": int(ctr[m["id"]]), "dev_freq": int(cdv[m["id"]]),
                     "leaf": m["result"] not in comps, "partial_utf8": text is None})
    assert [r["id"] for r in rows] == list(range(ext["first_new_id"], ext["first_new_id"] + k))
    with open(os.path.join(out, "new_tokens.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # verification
    ver = verify(base, out, k, ext, dev_docs)
    files = {f: C.sha256_file(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    info = {"name": name, "base": base, "base_repo": C.BASES[base]["repo"], "base_models": C.BASES[base]["models"],
            "base_license": C.BASES[base]["license"],
            "base_tokenizer_sha256": ext["base_tokenizer_sha256"], "k_new_tokens": k,
            "first_new_id": ext["first_new_id"], "len_tokenizer": ext["first_new_id"] + k,
            "method": "continued BPE (Purason et al. 2026, 'Teaching Old Tokenizers New Words', arXiv:2512.03989), "
                      "reimplemented from the paper; PUA code-point trick on the stock tokenizers BpeTrainer",
            "train_data": "Hindko train_D1 (permissive train split, manifest %s, hp.normalize %s), Arabic-script "
                          "units only" % (C.MANIFEST_SHA[:12], C.verify_frozen()["normalize_version"]),
            "extension_code_sha256": ext["code_sha256"], "extension_created_utc": ext["created_utc"],
            "files_sha256": files, "copied_from_base_sha256": copied,
            "tokenizer_model_spm": sp_info,
            "dev_strict": ev["dev"], "gates": {"G1": ev["gates"]["G1"]["pass"], "G2_new": ev["gates"]["G2_new"]["pass"],
                                               "equivalence": ev["equivalence"]["pass"]},
            "R1_new": {x: ev["R1_new"][x] for x in ("learned_tokens", "train_freq_eq0", "train_freq_lt20", "train_freq_lt100",
                                                    "pct_lt20", "pct_lt100", "median_train_freq", "partial_utf8_new")},
            "non_interference": {s: {x: v.get(x, 0) for x in ("docs", "changed", "token_reduction")}
                                 for s, v in ev["checks"].items()},
            "train_D1_tokens": {"base": C.load_json(os.path.join(C.RESULTS, base, "k0", "trackb_eval.json"))["train"]["tokens"],
                                "extended": ev["train"]["tokens"]},
            "tokenizer_config_changes": CONFIG_OVERRIDES.get(base), "verification": ver}
    C.dump_json(info, os.path.join(out, "EXTENSION.json"))
    write_readme(out, info)
    C.log(name, json.dumps(ver)[:300])
    return info


def verify(base, out, k, ext, dev_docs):
    from transformers import AutoTokenizer
    res = {}
    tk = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    at = AutoTokenizer.from_pretrained(out)
    res["transformers_class"] = type(at).__name__
    res["len_tokenizer"] = len(at)
    res["expected_len"] = ext["first_new_id"] + k
    bj = C.load_base_json(base)
    specials_all = bj.get("added_tokens", []) + C.config_only_added_tokens(base)
    res["special_ids_kept"] = all(tk.token_to_id(a["content"]) == a["id"] and at.convert_tokens_to_ids(a["content"]) == a["id"]
                                  for a in specials_all)
    res["special_tokens_checked"] = len(specials_all)
    res["config_only_added_tokens_written_to_tokenizer_json"] = [a["content"] for a in C.config_only_added_tokens(base)]
    same = rt = 0
    for d in dev_docs:
        a = tk.encode(d["text"], add_special_tokens=False).ids
        b = at(d["text"], add_special_tokens=False)["input_ids"]
        same += int(a == b)
        rt += int(at.decode(b, clean_up_tokenization_spaces=False) == d["text"])
    res["dev_docs"] = len(dev_docs)
    res["transformers_equals_tokenizers"] = same
    res["transformers_roundtrip"] = rt
    # the same comparison for the UNEXTENDED base folder with its own tokenizer_config.json (reference point)
    base_at = AutoTokenizer.from_pretrained(C.BASES[base]["dir"])
    base_tk = Tokenizer.from_file(C.base_tokenizer_path(base))
    res["base_folder_transformers_class"] = type(base_at).__name__
    res["base_folder_transformers_equals_tokenizers"] = sum(
        int(base_tk.encode(d["text"], add_special_tokens=False).ids == base_at(d["text"], add_special_tokens=False)["input_ids"])
        for d in dev_docs)
    if getattr(at, "chat_template", None):
        msgs = [{"role": "user", "content": "تساں کِداں او؟"}, {"role": "assistant", "content": "میں ٹھیک آں۔"}]
        s = at.apply_chat_template(msgs, tokenize=False)
        ids = at(s, add_special_tokens=False)["input_ids"]
        s0 = base_at.apply_chat_template(msgs, tokenize=False)
        specials = {a["id"] for a in bj.get("added_tokens", []) if a["content"] in s}
        res["chat_template_renders"] = (bool(s) and s == s0 and at.decode(ids, clean_up_tokenization_spaces=False) == s
                                        and specials <= set(ids))
        res["chat_template_tokens"] = {"base": len(base_at(s0, add_special_tokens=False)["input_ids"]), "extended": len(ids)}
    else:
        res["chat_template_renders"] = None
    res["pass"] = (res["len_tokenizer"] == res["expected_len"] and res["special_ids_kept"]
                   and same == len(dev_docs) and rt == len(dev_docs) and res["chat_template_renders"] in (True, None))
    return res


def write_readme(out, info):
    d = info["dev_strict"]
    base = info["base"]
    b0 = C.load_json(os.path.join(C.RESULTS, base, "k0", "trackb_eval.json"))["dev"]
    txt = f"""# {info['name']}

The **{base}** tokenizer ({info['base_models']}) extended with **{info['k_new_tokens']:,} Hindko tokens** by continued
BPE (Purason et al. 2026, reimplemented from the paper) on the Hindko train split. Built for continued pretraining
(CPT) of a {base} checkpoint on Hindko; details, curves and the allowed claims are in `TRACKB.md` of the Track B
work folder.

- New token ids: {info['first_new_id']:,} … {info['len_tokenizer'] - 1:,}; `len(tokenizer)` = {info['len_tokenizer']:,}. All special
  tokens keep their ids. The new tokens are ordinary BPE model tokens with merges (not added tokens).
- Text in which no new merge applies is encoded exactly as by the base tokenizer (verified on English, Python code
  and other-script check sets, see EXTENSION.json `non_interference`). Arabic-script text (Hindko, Urdu) gets
  fewer tokens; every new token is a concatenation of consecutive base tokens.
- Hindko dev_strict: {b0['bytes_per_token']:.3f} → {d['bytes_per_token']:.3f} bytes/token
  ({100 * (1 - d['nsl_vs_base']):.1f}% fewer tokens), fertility {b0['fertility']:.3f} → {d['fertility']:.3f}, STRR
  {100 * b0['strr']:.1f}% → {100 * d['strr']:.1f}%.
- License: the base tokenizer's ({info['base_license']}). Base: {info['base_repo']}.

## Use

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("{info['name']}")        # this folder
```

The model's embedding matrix must be resized to at least {info['len_tokenizer']:,} rows and the new rows initialised
before CPT: `python init_embeddings.py --model <base checkpoint> --tokenizer <this folder> --out <dir>`
(reference implementation, unit-tested on tiny random-weight models only; see TRACKB.md).
{sp_readme(info)}
Files and sha256: `EXTENSION.json`. New tokens with their base-token decompositions and frequencies: `new_tokens.jsonl`.
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


def sp_readme(info):
    sp = info.get("tokenizer_model_spm")
    if not sp:
        return ""
    if sp["delivered"]:
        a = sp["agreement_with_tokenizer_json"]
        return (chr(10) + "`tokenizer.model` is the matching SentencePiece model (new pieces appended with scores "
                "below every base piece). It encoded all %s train_D1 and %s dev_strict documents and every check-set "
                "document exactly like `tokenizer.json` (measured agreement, not a proof of equivalence on all "
                "inputs)." % ("{:,}".format(a["train_D1"]["docs"]), "{:,}".format(a["dev_strict"]["docs"])) + chr(10))
    return (chr(10) + "No `tokenizer.model`: the analogous SentencePiece model did not reproduce `tokenizer.json` on "
            "every document (EXTENSION.json). Load this folder through `tokenizer.json`." + chr(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+")
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    knees = C.load_json(os.path.join(C.RESULTS, "knee.json"))["knees"]
    train_texts = [d["text"] for d in C.load_view("train_D1")]
    dev_docs = C.load_view("dev_strict")
    man = {}
    mp = os.path.join(C.DELIVER, "deliver_manifest.json")
    if os.path.exists(mp):
        man = C.load_json(mp)
    for base in a.bases or list(knees):
        k = a.k or knees[base]["chosen_k"]
        info = deliver(base, k, train_texts, dev_docs)
        folder = os.path.join(C.DELIVER, info["name"])
        man[info["name"]] = {"base": base, "k": k,
                             "files_sha256": {f: C.sha256_file(os.path.join(folder, f)) for f in sorted(os.listdir(folder))},
                             "verification_pass": info["verification"]["pass"]}
        C.dump_json(man, mp)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
