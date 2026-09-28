"""Scan the hub for Urdu / Perso-Arabic-script (Shahmukhi, Saraiki, Pashto, Sindhi, Kashmiri) models whose TOKENIZER
differs from every tokenizer already collected, download only those tokenizer files, and measure them.

Stage 1 (metadata): search -> root tokenizer files -> signature (blob ids). Drop signatures already collected.
Stage 2 (download, tokenizer files only, <= 40 MB each): files/_urdu_candidates/<owner>__<repo>/
Stage 3 (measure): vocab size, # Arabic-script vocab entries, tokens on the 5 samples, exact round trip, UNK.
Output: scripts/_urdu_scan.json
"""
import json
import os
import re
import sys
import unicodedata

os.environ["HF_HOME"] = r"F:\Hindko\_tokenizer\hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from tiktoken_tools import bytes_to_unicode  # noqa: E402

B = r"F:\Hindko\_tokenizer\baselines"
OUT = os.path.join(B, "scripts", "_urdu_scan.json")
CAND = os.path.join(B, "files", "_urdu_candidates")
QUERIES = ["urdu", "shahmukhi", "saraiki", "punjabi", "pashto", "sindhi", "kashmiri", "balochi", "pakistan", "hindko", "lahnda"]
SKIP = re.compile(r"gguf|GGUF|lora|LoRA|adapter|qlora|awq|AWQ|gptq|GPTQ|bnb|4bit|8bit|mlx|onnx|whisper|wav2vec|w2v|mms|tts|asr|ocr|OCR|speech|sentiment|ner|NER|pos|classif|detect|hate|spam|fake|sarcas|clf", re.I)
TOKF = ["tokenizer.json", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.json", "merges.txt", "vocab.txt"]
api = HfApi()
BYTE_DEC = {v: k for k, v in bytes_to_unicode().items()}


def is_ar(c):
    o = ord(c)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def known_blobs():
    log = json.load(open(os.path.join(B, "scripts", "_download_log.json"), encoding="utf-8"))
    s = set()
    for v in log.values():
        for f in (v.get("files") or {}).values():
            if f.get("hub_blob_id"):
                s.add(f["hub_blob_id"])
    return s


def stage1():
    seen, rows = set(), []
    for q in QUERIES:
        for m in api.list_models(search=q, sort="downloads", limit=150, expand=["gated", "downloads", "likes", "pipeline_tag", "createdAt"]):
            if m.id in seen or m.gated or SKIP.search(m.id):
                continue
            if m.pipeline_tag not in (None, "text-generation", "fill-mask", "text2text-generation", "translation", "feature-extraction"):
                continue
            seen.add(m.id)
            rows.append({"repo": m.id, "downloads": m.downloads, "likes": m.likes, "pipe": m.pipeline_tag, "created": str(m.created_at)[:10], "query": q})
    return rows


def signature(repo):
    info = api.model_info(repo, files_metadata=True)
    root = {s.rfilename: s for s in (info.siblings or []) if "/" not in s.rfilename}
    files = [f for f in TOKF if f in root]
    if "tokenizer.json" in files:
        files = ["tokenizer.json"] + [f for f in files if f.endswith(".model")]
    sig = tuple(sorted(root[f].blob_id for f in files))
    return info.sha, files, sig, {f: root[f].size for f in files}, [root[f].blob_id for f in files]


def measure(d, files):
    from tokenizers import Tokenizer

    # the 5 core samples (formerly samples_hindko5.json); the scan was run before the 3 stress samples existed
    S = [s for s in json.load(open(os.path.join(B, "samples_hindko.json"), encoding="utf-8"))["samples"] if s["group"] == "core"]
    if "tokenizer.json" in files:
        tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        tk.no_truncation()
        tk.no_padding()
        js = json.loads(tk.to_str())
        mt = js["model"]["type"]
        bl = '"ByteLevel"' in json.dumps(js.get("pre_tokenizer")) or '"ByteLevel"' in json.dumps(js.get("decoder"))
        vocab = tk.get_vocab(True)
        enc = lambda t: tk.encode(t, add_special_tokens=False).ids  # noqa: E731
        dec = lambda ids: tk.decode(ids, skip_special_tokens=False)  # noqa: E731
        unk = js["model"].get("unk_token")
        unk_id = tk.token_to_id(unk) if isinstance(unk, str) else js["model"].get("unk_id")
    elif any(f.endswith(".model") for f in files):
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=os.path.join(d, [f for f in files if f.endswith(".model")][0]))
        mt, bl = "spm", False
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        enc = lambda t: sp.encode(t, out_type=int)  # noqa: E731
        dec = sp.decode
        unk_id = sp.unk_id()
    else:
        return {"note": "no tokenizer.json / .model; not measured"}
    n_ar = 0
    for t in vocab:
        s = t
        if bl:
            try:
                s = bytes(BYTE_DEC[c] for c in t).decode("utf-8", errors="ignore")
            except KeyError:
                pass
        n_ar += any(is_ar(c) for c in s)
    tot, rt, nunk = 0, 0, 0
    for smp in S:
        x = unicodedata.normalize("NFC", smp["text"])
        ids = enc(x)
        tot += len(ids)
        rt += dec(ids) == x
        nunk += sum(1 for i in ids if i == unk_id)
    return {"model_type": mt, "byte_level": bl, "vocab_size": len(vocab), "n_arabic_script_tokens": n_ar,
            "sample_tokens_total": tot, "roundtrip_exact": f"{rt}/5", "n_unk": nunk}


def main():
    rows = stage1()
    print("stage1 candidates:", len(rows), flush=True)
    known = known_blobs()
    groups = {}
    for r in rows:
        try:
            sha, files, sig, sizes, blobs = signature(r["repo"])
        except Exception as e:  # noqa: BLE001
            r["error"] = str(e)[:120]
            continue
        r.update(sha=sha, files=files, sizes=sizes)
        if not files:
            r["status"] = "no tokenizer files"
            continue
        if all(b in known for b in blobs):
            r["status"] = "tokenizer identical to an already-collected baseline"
            continue
        groups.setdefault(sig, []).append(r)
    print("unique new tokenizer signatures:", len(groups), flush=True)
    ranked = sorted(groups.values(), key=lambda g: -max((x["downloads"] or 0) + 50 * (x["likes"] or 0) for x in g))
    results = []
    for g in ranked[:45]:
        rep = max(g, key=lambda x: (x["downloads"] or 0) + 50 * (x["likes"] or 0))
        if any((v or 0) > 40 * 1024 * 1024 for v in rep["sizes"].values()):
            results.append({"repo": rep["repo"], "skipped": "file > 40MB"})
            continue
        d = os.path.join(CAND, rep["repo"].replace("/", "__"))
        try:
            for f in rep["files"]:
                hf_hub_download(rep["repo"], f, revision=rep["sha"], local_dir=d)
            for f in ["tokenizer_config.json", "special_tokens_map.json"]:
                try:
                    hf_hub_download(rep["repo"], f, revision=rep["sha"], local_dir=d)
                except Exception:  # noqa: BLE001
                    pass
            meas = measure(d, rep["files"])
        except Exception as e:  # noqa: BLE001
            meas = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        res = {"repo": rep["repo"], "revision": rep["sha"], "downloads": rep["downloads"], "likes": rep["likes"], "pipe": rep["pipe"],
               "created": rep["created"], "files": rep["files"], "also_same_tokenizer": [x["repo"] for x in g if x is not rep][:10], **meas}
        results.append(res)
        print(json.dumps({k: res.get(k) for k in ["repo", "downloads", "likes", "vocab_size", "n_arabic_script_tokens", "sample_tokens_total", "roundtrip_exact", "n_unk", "error", "note"]}, ensure_ascii=False), flush=True)
    json.dump({"queries": QUERIES, "n_stage1": len(rows), "stage1": rows, "measured": results},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
