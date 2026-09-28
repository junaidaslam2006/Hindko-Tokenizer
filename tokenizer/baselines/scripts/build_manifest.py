"""Load every baseline, record its properties, test it on Hindko text, write baselines/manifest.json.

Tests (revised 2026-09-26 after review: the 5 single-line core paragraphs never exercised line breaks, presentation
forms or U+2026, so 6 tokenizers were wrongly labelled lossless):
  1. 8 samples (samples_hindko.json): 5 core paragraphs + 3 stress samples (multi-line document, U+FDFA, U+2026).
  2. Round-trip battery on real corpus text:
     a. every validation + test document of the evaluation split (F:\\Hindko\\_tokenizer\\splits, permissive tier,
        which contains the strict tier), whole text, line breaks kept;
     b. character coverage: for every distinct code point of the permissive corpus except space and LF, the (up to) 3
        lines with the smallest sha256(f"{uid}#{line_index}") that contain it (lines of <= 2,000 chars; if the code
        point occurs only in longer lines, a 600-char window around its first occurrence in the smallest-key line).
  3. Whitespace probes: how decode(encode(.)) treats one LF and a blank line (the only whitespace besides U+0020 in
     the corpus).
A tokenizer is `lossless` iff decode(encode(x)) == x for all 8 samples and every battery text.

Run: PYTHONIOENCODING=utf-8 python build_manifest.py
Network: none (HF_HUB_OFFLINE=1). Loads tokenizers one at a time and frees them (RAM budget ~3 GB).
"""
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
import unicodedata
from collections import Counter

os.environ["HF_HOME"] = r"F:\Hindko\_tokenizer\hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "3"

B = r"F:\Hindko\_tokenizer\baselines"
SPLITS_DIR = r"F:\Hindko\_tokenizer\splits"
PERMISSIVE = r"F:\Hindko\hindko_dataset_permissive.jsonl"
sys.path.insert(0, B)
sys.path.insert(0, os.path.join(B, "scripts"))
import load_baselines as lb  # noqa: E402
import registry  # noqa: E402
from tiktoken_tools import bytes_to_unicode  # noqa: E402

LOG = json.load(open(os.path.join(B, "scripts", "_download_log.json"), encoding="utf-8"))
PROBES = {}
for f in ["_probe_hub.json", "_probe_hub2.json", "_probe_hub3.json", "_probe_hub4.json"]:
    for r in json.load(open(os.path.join(B, "scripts", f), encoding="utf-8")):
        PROBES[r["repo"]] = r
SAMPLES_FILE = "samples_hindko.json"
SAMPLES = json.load(open(os.path.join(B, SAMPLES_FILE), encoding="utf-8"))["samples"]
N_CORE = sum(1 for s in SAMPLES if s["group"] == "core")
N_SAMPLES = len(SAMPLES)
CROSS = json.load(open(os.path.join(B, "scripts", "_crosscheck.json"), encoding="utf-8"))
CROSS_ML = json.load(open(os.path.join(B, "scripts", "_crosscheck_multiline.json"), encoding="utf-8"))
KIMI = json.load(open(os.path.join(B, "scripts", "_kimi_verify.json"), encoding="utf-8"))

PROBE_SRC = r"F:\Hindko\_pipeline\_tokenizer_probe"
URDU_BASE = {"urdu-llama3-almanach": "llama-3", "urdu-llama2-almanach": "llama-2", "urdu-llama3.2-custom": "llama-3",
             "urdu-llama-bilal": None, "alif-1.0": "llama-3", "qalb-1.0": "llama-3",
             "pashto-lfm2.5": "lfm2", "sindhi-xlmr": "xlm-r"}
BYTE_DEC = {v: k for k, v in bytes_to_unicode().items()}
LF = "\n"
# whitespace probes: two common Hindko words joined by one LF / by a blank line
W1, W2 = "پشور", "اچ"
WS_PROBES = {"one line break": W1 + LF + W2, "blank line": W1 + LF + LF + W2}
BATTERY_FILE = os.path.join(B, "scripts", "_roundtrip_battery.json")
BATTERY_DEFINITION = (
    "a) every validation+test document of the evaluation split (splits/split_manifest.jsonl, permissive tier, which contains "
    "the strict tier), whole text with line breaks; b) character coverage: for every distinct code point of the permissive "
    "corpus except U+0020 and LF (lines are split on LF; both occur throughout part a), the up-to-3 lines (<= 2,000 chars) "
    "with the smallest sha256(uid#line_index) containing it, or a 600-char window of the smallest-key longer line if it "
    "occurs only in longer lines")


def is_arabic(ch):
    o = ord(ch)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def cname(c):
    return {"\n": "LINE FEED", " ": "SPACE"}.get(c) or unicodedata.name(c, "?")


def fmt(c):
    return f"U+{ord(c):04X} {cname(c)}"


# ---------------------------------------------------------------- round-trip battery
def build_battery():
    """Deterministic list of {part, uid, line_index, text}. See module docstring."""
    sys.path.insert(0, SPLITS_DIR)
    from load_split import iter_split

    items = []
    for split in ("validation", "test"):
        for uid, text, rec in iter_split(split, tier="permissive"):
            items.append({"part": f"eval:{split}", "uid": uid, "line_index": None, "source": rec["source"], "text": text})
    best, longbest = {}, {}  # char -> sorted list of up to 3 (key, uid, i, source) ; char -> (key, uid, i, source)
    lines = {}
    for line in open(PERMISSIVE, encoding="utf-8"):
        r = json.loads(line)
        for i, p in enumerate(r["text"].split(LF)):
            if not p:
                continue
            key = hashlib.sha256(f"{r['uid']}#{i}".encode()).hexdigest()
            short = len(p) <= 2000
            for c in set(p):
                if c == " ":
                    continue
                if short:
                    lst = best.setdefault(c, [])
                    if len(lst) < 3 or key < lst[-1][0]:
                        lst.append((key, r["uid"], i, r["source"]))
                        lst.sort()
                        del lst[3:]
                else:
                    cur = longbest.get(c)
                    if cur is None or key < cur[0]:
                        longbest[c] = (key, r["uid"], i, r["source"])
            lines[(r["uid"], i)] = p
    chosen = {}
    for c in sorted(set(best) | set(longbest)):
        if best.get(c):
            for key, uid, i, src in best[c]:
                chosen.setdefault((key, uid, i, None), {"source": src, "chars": []})["chars"].append(f"U+{ord(c):04X}")
        else:
            key, uid, i, src = longbest[c]
            pos = lines[(uid, i)].index(c)
            chosen.setdefault((key, uid, i, max(0, pos - 300)), {"source": src, "chars": []})["chars"].append(f"U+{ord(c):04X}")
    n_chars_covered = len(set(best) | set(longbest))
    for (key, uid, i, start), v in sorted(chosen.items(), key=lambda kv: (kv[0][0], kv[0][3] or 0)):
        p = lines[(uid, i)]
        t = p if start is None else unicodedata.normalize("NFC", p[start:start + 600])
        items.append({"part": "char-coverage", "uid": uid, "line_index": i, "window_start": start, "source": v["source"],
                      "covers": v["chars"], "text": t})
    del lines
    desc = {
        "definition": BATTERY_DEFINITION,
        "split_manifest_sha256": sha256_file(os.path.join(SPLITS_DIR, "split_manifest.jsonl")),
        "n_texts": len(items),
        "n_chars": sum(len(x["text"]) for x in items),
        "n_line_breaks": sum(x["text"].count(LF) for x in items),
        "parts": {p: {"n_texts": sum(1 for x in items if x["part"] == p),
                      "n_chars": sum(len(x["text"]) for x in items if x["part"] == p),
                      "n_line_breaks": sum(x["text"].count(LF) for x in items if x["part"] == p)}
                  for p in sorted({x["part"] for x in items})},
        "n_distinct_code_points_covered": n_chars_covered,
        "texts_sha256": hashlib.sha256("\x00".join(x["text"] for x in items).encode("utf-8")).hexdigest(),
        "at_risk_counts": {fmt(c): sum(x["text"].count(c) for x in items) for c in [LF, chr(0x2026), chr(0xFDFA), chr(0xFDF2), "\u2018", "\u2019", chr(0x0601)]},
        "file": "scripts/_roundtrip_battery.json (char-coverage texts; eval documents are referenced by uid only)",
    }
    json.dump({"description": desc,
               "eval_uids": [[x["part"], x["uid"]] for x in items if x["part"].startswith("eval")],
               "char_coverage": [x for x in items if x["part"] == "char-coverage"]},
              open(BATTERY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return items, desc


# ---------------------------------------------------------------- introspection
def summarize(node):
    """Compact human-readable summary of a tokenizers normalizer / pre-tokenizer / decoder JSON node."""
    if node is None:
        return None
    t = node.get("type")
    if t == "Sequence":
        key = "normalizers" if "normalizers" in node else "pretokenizers" if "pretokenizers" in node else "decoders"
        return " + ".join(summarize(n) for n in node.get(key, []))
    if t == "Split":
        pat = node["pattern"].get("Regex", node["pattern"].get("String"))
        return f"Split(regex={pat!r}, behavior={node.get('behavior')})"
    if t == "ByteLevel":
        return f"ByteLevel(add_prefix_space={node.get('add_prefix_space')}, use_regex={node.get('use_regex')})"
    if t == "Metaspace":
        return f"Metaspace(replacement={node.get('replacement')!r}, prepend_scheme={node.get('prepend_scheme')}, split={node.get('split')})"
    if t == "Replace":
        pat = node["pattern"].get("String", node["pattern"].get("Regex"))
        return f"Replace({pat!r}->{node.get('content')!r})"
    if t == "Prepend":
        return f"Prepend({node.get('prepend')!r})"
    if t == "Precompiled":
        return "Precompiled(SentencePiece charsmap)"
    if t == "BertNormalizer":
        return (f"BertNormalizer(clean_text={node.get('clean_text')}, chinese_chars={node.get('handle_chinese_chars')}, "
                f"strip_accents={node.get('strip_accents')}, lowercase={node.get('lowercase')})")
    if t == "Digits":
        return f"Digits(individual={node.get('individual_digits')})"
    if t == "Strip":
        return f"Strip(left={node.get('strip_left', node.get('left'))}, right={node.get('strip_right', node.get('right'))})"
    return t


def inspect_tokenizers_json(js):
    m = js["model"]
    mt = m.get("type")
    pre = json.dumps(js.get("pre_tokenizer")) if js.get("pre_tokenizer") else ""
    dec = json.dumps(js.get("decoder")) if js.get("decoder") else ""
    byte_level = mt == "BPE" and ('"ByteLevel"' in pre or '"ByteLevel"' in dec)
    bf = bool(m.get("byte_fallback"))
    if mt == "BPE":
        algo = "byte-level BPE" if byte_level else ("BPE (SentencePiece-style, byte fallback)" if bf else "BPE (character-level, no byte fallback)")
    elif mt == "Unigram":
        algo = "Unigram (SentencePiece)" + (" + byte fallback" if bf else "")
    elif mt == "WordPiece":
        algo = "WordPiece"
    else:
        algo = mt
    if byte_level:
        bfs = "n/a: byte-level (every byte is a base token)"
    else:
        bfs = bf
    return dict(algorithm=algo, model_type=mt, byte_level=byte_level, byte_fallback=bfs,
                normalizer=summarize(js.get("normalizer")), pre_tokenizer=summarize(js.get("pre_tokenizer")),
                decoder=summarize(js.get("decoder")),
                ignore_merges=m.get("ignore_merges"), unk_token=m.get("unk_token") if mt != "Unigram" else m.get("unk_id"),
                n_added_tokens=len(js.get("added_tokens", [])),
                n_special_added=sum(1 for a in js.get("added_tokens", []) if a.get("special")))


def inspect_spm(path):
    from sentencepiece import sentencepiece_model_pb2 as pb

    mp = pb.ModelProto()
    mp.ParseFromString(open(path, "rb").read())
    ts, ns = mp.trainer_spec, mp.normalizer_spec
    mtype = {1: "Unigram", 2: "BPE", 3: "Word", 4: "Char"}.get(ts.model_type, str(ts.model_type))
    return dict(algorithm=f"SentencePiece {mtype}" + (" + byte fallback" if ts.byte_fallback else ""),
                model_type=mtype, byte_level=False, byte_fallback=bool(ts.byte_fallback),
                normalizer=f"spm normalizer_spec.name={ns.name!r} add_dummy_prefix={ns.add_dummy_prefix} "
                           f"remove_extra_whitespaces={ns.remove_extra_whitespaces}",
                pre_tokenizer=f"spm split_by_whitespace={ts.split_by_whitespace} split_digits={ts.split_digits} "
                              f"split_by_unicode_script={ts.split_by_unicode_script} max_sentencepiece_length={ts.max_sentencepiece_length}",
                decoder="sentencepiece", unk_token=mp.trainer_spec.unk_id, n_pieces=len(mp.pieces),
                spm_character_coverage=ts.character_coverage, spm_vocab_size=ts.vocab_size)


def token_texts(bl, info):
    """Surface string of every vocab entry (byte-level tokens decoded to UTF-8 where possible)."""
    raw = bl.raw
    if bl.loader in ("hf_json", "tiktoken_ranks"):
        vocab = raw.get_vocab(with_added_tokens=True)
    elif bl.loader in ("spm", "local_spm"):
        vocab = {raw.id_to_piece(i): i for i in range(raw.get_piece_size())}
    elif bl.loader == "auto":
        vocab = raw.get_vocab()
    else:
        return None, None
    out = {}
    for tok, i in vocab.items():
        if info.get("byte_level"):
            try:
                s = bytes(BYTE_DEC[c] for c in tok).decode("utf-8", errors="ignore")
            except KeyError:
                s = tok  # added tokens are stored verbatim
        else:
            s = tok.replace("\u2581", " ").removeprefix("##")
        out[i] = s
    return vocab, out


# ---------------------------------------------------------------- tests
def _nows(s):
    return "".join(s.split())


def _strip_marks(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.category(c).startswith("M"))


def classify(x, y, n_unk):
    """First matching explanation of why decode(encode(x)) != x."""
    if y == x:
        return "exact"
    if unicodedata.normalize("NFC", y) == x:
        return "equal after NFC"
    if n_unk:
        return "lossy: UNK tokens"
    if "##" in y and "##" not in x:
        return "decoder leaves WordPiece '##' markers (tokenizer.json has no WordPiece decoder)"
    if " ".join(y.split()) == " ".join(x.split()):
        return "whitespace differs (line breaks not preserved)" if y.count(LF) != x.count(LF) else "whitespace differs"
    if _nows(y) == _nows(x):
        return "spacing differs (spaces inserted around punctuation)"
    if _nows(unicodedata.normalize("NFKC", y)) == _nows(unicodedata.normalize("NFKC", x)):
        return "Unicode normalization (NFKC-type) and/or whitespace"
    if _nows(_strip_marks(y)) == _nows(_strip_marks(x)):
        return "diacritics/combining marks stripped (and/or whitespace)"
    return "characters deleted or changed by the normalizer (and/or whitespace)"


def char_diff(x, y, top=8):
    """Multiset difference of characters incl. whitespace (what the tokenizer lost / invented)."""
    a, b = Counter(x), Counter(y)
    return {"lost": {fmt(c): n for c, n in (a - b).most_common(top)}, "added": {fmt(c): n for c, n in (b - a).most_common(top)}}


def first_diff(x, y):
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return {"pos": i, "expected": [f"U+{ord(c):04X}" for c in x[i:i + 4]], "got": [f"U+{ord(c):04X}" for c in y[i:i + 4]],
            "expected_ctx": x[max(0, i - 12):i + 12], "got_ctx": y[max(0, i - 12):i + 12]}


def encode_many(bl, texts):
    if bl.loader in ("hf_json", "tiktoken_ranks"):
        return [e.ids for e in bl.raw.encode_batch(texts, add_special_tokens=False)]
    return [bl.encode(t) for t in texts]


def decode_many(bl, id_lists):
    if bl.loader in ("hf_json", "tiktoken_ranks"):
        return bl.raw.decode_batch(id_lists, skip_special_tokens=False)
    return [bl.decode(ids) for ids in id_lists]


def run_samples(bl):
    res, encs = [], []
    for s in SAMPLES:
        x = unicodedata.normalize("NFC", s["text"])
        ids = bl.encode(x)
        y = bl.decode(ids)
        n_unk = sum(1 for i in ids if i in bl.unk_ids)
        c = classify(x, y, n_unk)
        r = {"group": s["group"], "slot": s["slot"], "uid": s["uid"], "n_chars": len(x), "n_words": len(x.split()),
             "n_line_breaks": x.count(LF), "n_tokens": len(ids), "n_unk": n_unk, "roundtrip_exact": y == x, "roundtrip_class": c}
        if y != x:
            r["first_diff"] = first_diff(x, y)
            r["char_diff"] = char_diff(x, y)
        res.append(r)
        encs.append(ids)
    return res, encs


def run_probes(bl):
    out, dec, unk = {}, {}, 0
    for k, x in WS_PROBES.items():
        ids = bl.encode(x)
        y = bl.decode(ids)
        u = sum(1 for i in ids if i in bl.unk_ids)
        unk += u
        dec[k] = y
        out[k] = {"input": x.replace(LF, "\\n"), "output": y.replace(LF, "\\n"), "exact": y == x, "n_unk": u}
    o1, o2 = dec["one line break"], dec["blank line"]
    if "##" in o1 + o2:  # WordPiece pieces left joined by " ##" (indicbert-v2): judge the line break, not the decoder bug
        o1, o2 = o1.replace(" ##", ""), o2.replace(" ##", "")
    if o1 == WS_PROBES["one line break"] and o2 == WS_PROBES["blank line"]:
        h = "preserved"
    elif unk:
        h = "line breaks become UNK tokens"
    elif o1.strip() == W1 + " " + W2 and o2.strip() == W1 + " " + W2:
        h = "each run of line breaks becomes one space"
    elif o1.strip() == W1 + " " + W2 and o2.strip() == W1 + "  " + W2:
        h = "each line break becomes a space"
    elif o1.strip() == W1 + W2:
        h = "line breaks deleted (words joined)"
    else:
        h = f"other: {o1!r} / {o2!r}"
    return out, h


def run_battery(bl, items):
    """Round trip on every battery text + token counts on the eval documents (as is, and with LF -> space)."""
    xs = [unicodedata.normalize("NFC", it["text"]) for it in items]
    encs = encode_many(bl, xs)
    ys = decode_many(bl, encs)
    h = hashlib.sha256()
    parts, classes, flags = {}, Counter(), Counter()
    lost, added = Counter(), Counter()
    n_unk = n_tok = 0
    lf_in = lf_out = 0
    examples = []
    for it, x, ids, y in zip(items, xs, encs, ys):
        h.update(json.dumps(ids).encode())
        u = sum(1 for i in ids if i in bl.unk_ids)
        n_unk += u
        n_tok += len(ids)
        lf_in += x.count(LF)
        lf_out += y.count(LF)
        pk = "eval documents" if it["part"].startswith("eval") else "char-coverage lines"
        p = parts.setdefault(pk, {"n_texts": 0, "n_exact": 0})
        p["n_texts"] += 1
        if y == x:
            p["n_exact"] += 1
            continue
        classes[classify(x, y, u)] += 1
        a, b = Counter(x), Counter(y)
        lost.update(a - b)
        added.update(b - a)
        if y.count(LF) < x.count(LF):
            flags["line breaks lost"] += 1
        if u:
            flags["UNK tokens"] += 1
        if _nows(x) != _nows(y):
            flags["non-whitespace characters changed"] += 1
        if len(examples) < 3:
            examples.append({"part": it["part"], "uid": it["uid"], "line_index": it.get("line_index"),
                             "class": classify(x, y, u), "first_diff": first_diff(x, y)})
    ev = [(x, ids) for it, x, ids in zip(items, xs, encs) if it["part"].startswith("eval")]
    tok_asis = sum(len(ids) for _, ids in ev)
    sp = [x.replace(LF, " ") for x, _ in ev]
    tok_sp = sum(len(ids) for ids in encode_many(bl, sp))
    n_exact = sum(p["n_exact"] for p in parts.values())
    n_texts = sum(p["n_texts"] for p in parts.values())
    return {
        "n_texts": n_texts, "n_exact": n_exact, "all_exact": n_exact == n_texts, "parts": parts,
        "failure_classes": dict(classes.most_common()), "failure_flags": dict(flags.most_common()),
        "chars_lost": {fmt(c): n for c, n in lost.most_common(12)},
        "chars_added": {fmt(c): n for c, n in added.most_common(12)},
        "line_breaks_in": lf_in, "line_breaks_out": lf_out,
        "n_tokens": n_tok, "n_unk": n_unk, "unk_rate": round(n_unk / n_tok, 6) if n_tok else None,
        "examples": examples,
        "eval_docs_tokens": {"n_docs": len(ev), "as_is": tok_asis, "line_breaks_replaced_by_spaces": tok_sp,
                             "line_break_token_share": round((tok_asis - tok_sp) / tok_asis, 5) if tok_asis else None},
    }, h.hexdigest()


def transformers_check(e, spec, encs):
    """Does transformers.AutoTokenizer (local files, no remote code) produce the same ids?"""
    if e["loader"] in ("tiktoken_ranks",):
        return "skipped: official tokenizer needs trust_remote_code (not executed)"
    if e["loader"] == "local_spm":
        return "skipped: local sentencepiece model"
    if e["loader"] == "auto":
        return f"n/a: baseline itself is loaded with transformers {e.get('hf_class') or 'AutoTokenizer'}"
    try:
        import transformers

        cls = getattr(transformers, e["hf_class"]) if e.get("hf_class") else transformers.AutoTokenizer
        tok = cls.from_pretrained(os.path.join(B, "files", e["name"]), local_files_only=True, trust_remote_code=False)
        bad = 0
        for s, ids in zip(SAMPLES, encs):
            x = unicodedata.normalize("NFC", s["text"])
            if tok.encode(x, add_special_tokens=False) != ids:
                bad += 1
        cls = type(tok).__name__
        # where they differ: which parts of transformers' rebuilt pipeline are not the published tokenizer.json's
        diag = ""
        fpath = os.path.join(B, "files", e["name"], "tokenizer.json")
        be = getattr(tok, "backend_tokenizer", None)
        if bad and be is not None and os.path.exists(fpath):
            tj = json.loads(be.to_str())
            fj = json.load(open(fpath, encoding="utf-8"))
            comps = [c for c in ("normalizer", "pre_tokenizer") if tj.get(c) != fj.get(c)]
            if tj["model"].get("vocab") != fj["model"].get("vocab"):
                comps.append("vocab")
            spec["transformers_pipeline_diff"] = {c: {"tokenizer.json": summarize(fj.get(c)), "transformers": summarize(tj.get(c))}
                                                  for c in comps if c != "vocab"}
            diag = ("; transformers' " + " and ".join(comps) + (" differs" if len(comps) == 1 else " differ") +
                    " from the published tokenizer.json") if comps else "; no normalizer/pre-tokenizer/vocab difference found"
        del tok
        how = e.get("hf_class") or "AutoTokenizer"
        n = len(SAMPLES)
        return f"agrees on {n}/{n} ({how} -> {cls})" if bad == 0 else f"DIFFERS on {bad}/{n} ({how} -> {cls}){diag}"
    except Exception as ex:  # noqa: BLE001
        return f"AutoTokenizer load failed: {type(ex).__name__}: {str(ex).splitlines()[0][:160]}"


# ---------------------------------------------------------------- main
def main():
    # self-contained copy of the leaky project probe
    dst = os.path.join(B, "files", "hindko-probe-bpe32k")
    os.makedirs(dst, exist_ok=True)
    for fn in ["hindko_bpe32k.model", "hindko_bpe32k.vocab"]:
        if not os.path.exists(os.path.join(dst, fn)):
            shutil.copy2(os.path.join(PROBE_SRC, fn), os.path.join(dst, fn))

    import tokenizers
    import transformers
    import sentencepiece
    import huggingface_hub

    t0 = time.time()
    battery, battery_desc = build_battery()
    print(f"battery: {battery_desc['n_texts']} texts, {battery_desc['n_chars']:,} chars, {battery_desc['n_line_breaks']:,} line breaks, "
          f"{battery_desc['n_distinct_code_points_covered']} code points; {time.time() - t0:.1f}s", flush=True)

    out, vocab_cache = [], {}
    for e in registry.ENTRIES:
        name = e["name"]
        t0 = time.time()
        rec = {k: e.get(k) for k in ["name", "tier", "provider", "family", "models", "year", "repo", "official", "mirror_kind", "loader", "hf_class", "notes"]}
        rec["is_mirror"] = e.get("official") is not None
        dl = LOG.get(name, {})
        if e["loader"] == "local_spm":
            rec.update(revision=None, license="project-internal (not released)", files={
                fn: {"bytes": os.path.getsize(os.path.join(dst, fn)), "sha256": sha256_file(os.path.join(dst, fn)),
                     "copied_from": os.path.join(PROBE_SRC, fn)} for fn in os.listdir(dst)})
            rec["load_path"] = "files/hindko-probe-bpe32k/hindko_bpe32k.model"
        else:
            if "error" in dl:
                rec.update(working=False, status=dl["error"])
                out.append(rec)
                continue
            rec.update(revision=dl.get("revision"), license=dl.get("license"), repo_created=dl.get("repo_created"), files=dl.get("files"))
            off = PROBES.get(e.get("official") or "", {})
            if off:
                rec["official_gated"] = off.get("gated")
                rec["official_license"] = off.get("license")
            rec["load_path"] = {"hf_json": f"files/{name}/tokenizer.json",
                                "tiktoken_ranks": f"files/{name}/tokenizer.converted.json",
                                "spm": f"files/{name}/spiece.model",
                                "auto": f"files/{name}",
                                "bytes": None}[e["loader"]]
        try:
            bl = lb.build({"name": name, "loader": e["loader"], "load_path": rec["load_path"], "hf_class": e.get("hf_class")})
            # ---- introspection
            if e["loader"] in ("hf_json", "tiktoken_ranks"):
                info = inspect_tokenizers_json(json.loads(bl.raw.to_str()))
            elif e["loader"] in ("spm", "local_spm"):
                info = inspect_spm(os.path.join(B, rec["load_path"]))
            elif e["loader"] == "auto":
                be = getattr(bl.raw, "backend_tokenizer", None)
                info = inspect_tokenizers_json(json.loads(be.to_str())) if be is not None else {"algorithm": type(bl.raw).__name__}
                info["transformers_class"] = type(bl.raw).__name__
            else:
                info = dict(algorithm="bytes (UTF-8, no merges)", model_type="bytes", byte_level=True,
                            byte_fallback="n/a: bytes", normalizer=None, pre_tokenizer=None, decoder="bytes", unk_token=2)
            rec.update(info)
            rec["vocab_size"] = bl.vocab_size
            rec["unk_ids"] = sorted(bl.unk_ids)
            # ---- vocabulary composition (Arabic-script coverage)
            vocab, surf = token_texts(bl, info)
            if vocab is not None:
                rec["id_space"] = max(vocab.values()) + 1
                ar = {i for i, s in surf.items() if any(is_arabic(c) for c in s)}
                rec["n_arabic_script_tokens"] = len(ar)
                rec["n_arabic_script_tokens_len2plus"] = sum(1 for i in ar if sum(is_arabic(c) for c in surf[i]) >= 2)
                rec["vocab_sha256"] = sha_obj(sorted(vocab.items()))
                vocab_cache[name] = (vocab, surf)
            # ---- samples (5 core + 3 stress)
            res, encs = run_samples(bl)
            core = [r for r in res if r["group"] == "core"]
            rec["samples"] = res
            rec["roundtrip_exact_core5"] = all(r["roundtrip_exact"] for r in core)
            rec["roundtrip_exact_samples"] = all(r["roundtrip_exact"] for r in res)
            rec["n_samples_exact"] = sum(1 for r in res if r["roundtrip_exact"])
            rec["roundtrip_classes"] = sorted({r["roundtrip_class"] for r in res})
            # tokens/word + 5-sample UNK rate: core paragraphs only (single-line, so line-break handling does not affect them)
            rec["sample_tokens_total"] = sum(r["n_tokens"] for r in core)
            rec["sample_unk_total"] = sum(r["n_unk"] for r in core)
            rec["unk_rate"] = round(rec["sample_unk_total"] / rec["sample_tokens_total"], 6) if rec["sample_tokens_total"] else None
            rec["sample_words_total"] = sum(r["n_words"] for r in core)
            rec["sample_tokens_per_word"] = round(rec["sample_tokens_total"] / rec["sample_words_total"], 3)
            rec["encoding_sha256_samples"] = sha_obj(encs)
            # ---- whitespace probes + round-trip battery
            rec["whitespace_probes"], rec["newline_handling"] = run_probes(bl)
            rec["roundtrip_battery"], rec["encoding_sha256_battery"] = run_battery(bl, battery)
            rec["unk_rate_battery"] = rec["roundtrip_battery"]["unk_rate"]
            rec["lossless"] = rec["roundtrip_exact_samples"] and rec["roundtrip_battery"]["all_exact"]
            rec["transformers_crosscheck"] = transformers_check(e, rec, encs)
            rec["working"] = True
            rec["status"] = "ok"
            del bl
        except Exception as ex:  # noqa: BLE001
            rec.update(working=False, status=f"load/test failed: {type(ex).__name__}: {str(ex).splitlines()[0][:300]}")
            traceback.print_exc()
        gc.collect()
        rec["seconds"] = round(time.time() - t0, 1)
        out.append(rec)
        rb = rec.get("roundtrip_battery") or {}
        print(f"{name:24s} {rec.get('status')[:20]:20s} V={rec.get('vocab_size')} core_tok={rec.get('sample_tokens_total')} "
              f"samples_exact={rec.get('n_samples_exact')}/{N_SAMPLES} battery_exact={rb.get('n_exact')}/{rb.get('n_texts')} "
              f"LF={rb.get('line_breaks_out')}/{rb.get('line_breaks_in')} nl='{rec.get('newline_handling')}' "
              f"unk={rb.get('n_unk')} lossless={rec.get('lossless')} tf={str(rec.get('transformers_crosscheck'))[:28]} {rec['seconds']}s",
              flush=True)

    # ---- behaviour key: same encodings on all 8 samples AND on the whole round-trip battery
    def bkey(r):
        return (r.get("encoding_sha256_samples"), r.get("encoding_sha256_battery"))

    groups = {}
    for r in out:
        if r.get("working"):
            groups.setdefault((r.get("vocab_sha256"),) + bkey(r), []).append(r["name"])
    for r in out:
        if r.get("working"):
            r["identical_to"] = [n for n in groups[(r.get("vocab_sha256"),) + bkey(r)] if n != r["name"]]
    enc_groups = {}
    for r in out:
        if r.get("working"):
            enc_groups.setdefault(bkey(r), []).append(r["name"])
    for r in out:
        if r.get("working"):
            r["same_encodings_as"] = [n for n in enc_groups[bkey(r)] if n != r["name"]]

    # ---- recommended_for_benchmark: working, and not behaviourally identical to an EARLIER entry
    first_seen = {}
    for r in out:
        if not r.get("working"):
            r["recommended_for_benchmark"] = False
            r["recommendation_reason"] = "not working"
            continue
        k = bkey(r)
        if k in first_seen:
            r["recommended_for_benchmark"] = False
            r["duplicate_of"] = first_seen[k]
            r["recommendation_reason"] = (f"same encodings as {first_seen[k]} on all {N_SAMPLES} samples and all "
                                          f"{battery_desc['n_texts']:,} battery texts (redundant)")
        else:
            first_seen[k] = r["name"]
            r["recommended_for_benchmark"] = True
            r["recommendation_reason"] = "unique behaviour" + (" (LEAKY reference: trained on the whole corpus)" if r["name"] == "hindko-probe-bpe32k" else "")

    # ---- for redundant entries: how different is the vocabulary from the entry it duplicates?
    for r in out:
        if r.get("working") and r.get("duplicate_of") and r["name"] in vocab_cache:
            f = r["duplicate_of"]
            if f in vocab_cache:
                va, vb = vocab_cache[r["name"]][0], vocab_cache[f][0]
                common = set(va) & set(vb)
                r["vocab_vs_duplicate_of"] = {"other": f, "n_tokens_only_here": len(set(va) - set(vb)),
                                              "n_tokens_only_in_other": len(set(vb) - set(va)),
                                              "n_common": len(common), "n_common_same_id": sum(1 for t in common if va[t] == vb[t]),
                                              "examples_only_here": sorted(set(va) - set(vb))[:8]}
                # identical encodings are still finite evidence: keep the entry if its token set differs by more than 1%
                diff = r["vocab_vs_duplicate_of"]["n_tokens_only_here"] + r["vocab_vs_duplicate_of"]["n_tokens_only_in_other"]
                if diff > 0.01 * r["vocab_size"]:
                    r["recommended_for_benchmark"] = True
                    r["recommendation_reason"] = (f"same encodings as {f} on the samples and the battery, but the token sets differ by "
                                                  f"{diff:,} tokens (>1% of vocab), so it is kept")

    # ---- manual overrides (documented in the registry)
    for r, e in zip(out, registry.ENTRIES):
        if e.get("redundant_override") and r.get("working"):
            r["recommended_for_benchmark"] = False
            r["recommendation_reason"] = "redundant: " + e["redundant_override"]

    # ---- Urdu vocab-extension check vs the base tokenizer
    for r in out:
        if r["name"] in URDU_BASE and r.get("working") and r["name"] in vocab_cache:
            vocab, surf = vocab_cache[r["name"]]
            base = URDU_BASE[r["name"]]
            cands = [base] if base else ["llama-2", "llama-3"]
            best = None
            for bname in cands:
                bv, bs = vocab_cache[bname]
                bset = set(bs.values())
                new = [i for i, s in surf.items() if s not in bset]
                new_ar = [i for i in new if any(is_arabic(c) for c in surf[i])]
                same_ids = sum(1 for t, i in bv.items() if vocab.get(t) == i)
                cand = {"base": bname, "vocab_size": r["vocab_size"], "base_vocab_size": len(bv),
                        "n_surface_forms_not_in_base": len(new), "n_new_arabic_script_tokens": len(new_ar),
                        "base_tokens_with_same_id": same_ids,
                        "examples_new_arabic": [surf[i] for i in sorted(new_ar)[:15]]}
                if best is None or cand["base_tokens_with_same_id"] > best["base_tokens_with_same_id"]:
                    best = cand
            r["urdu_extension_check"] = best
            r["urdu_vocab_extended"] = best["n_new_arabic_script_tokens"] > 0

    man = {
        "title": "Competitor tokenizer baselines for the Hindko tokenizer benchmark",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": B,
        "hf_home": os.environ["HF_HOME"],
        "download_policy": "tokenizer files only (allow-list), anonymous, revision pinned to commit sha; no weights; no login; no remote code executed",
        "environment": {"python": platform.python_version(), "tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
                        "sentencepiece": sentencepiece.__version__, "huggingface_hub": huggingface_hub.__version__},
        "samples_file": SAMPLES_FILE,
        "samples": [{k: s[k] for k in ["group", "slot", "kind", "uid", "source", "paragraph_index", "n_chars", "n_words", "n_newlines"]}
                    for s in SAMPLES],
        "roundtrip_definition": "x = NFC(text); exact iff decode(encode(x)) == x, with no special tokens added and none skipped",
        "lossless_definition": f"lossless = exact round trip on all {N_SAMPLES} samples ({N_CORE} core paragraphs + {N_SAMPLES - N_CORE} stress samples) "
                               f"AND on every text of the round-trip battery",
        "roundtrip_battery": battery_desc,
        "unk_rate_definition": "unk_rate: UNK tokens / all tokens over the 5 core samples; unk_rate_battery: the same over the round-trip "
                               "battery. UNK ids = the tokenizer model's unk token (byte-level BPE and bytes have none)",
        "tokens_per_word_definition": "sample_tokens_per_word: tokens / whitespace words over the 5 core (single-line) paragraphs only; "
                                      "a sanity check, not the benchmark",
        "newline_handling_definition": "decode(encode(w1 + LF + w2)) and decode(encode(w1 + LF + LF + w2)) for two Hindko words; the only "
                                       "whitespace in the corpus is U+0020 and LF. roundtrip_battery.eval_docs_tokens.line_break_token_share "
                                       "= share of a tokenizer's tokens on the eval documents that disappears when LF is replaced by a space",
        "redundancy_definition": f"recommended_for_benchmark=false if the encodings of all {N_SAMPLES} samples AND of all battery texts equal "
                                 "those of an earlier entry, unless the token sets differ by more than 1% of the vocabulary; plus "
                                 "documented manual overrides",
        "n_working": sum(1 for r in out if r.get("working")),
        "n_lossless": sum(1 for r in out if r.get("lossless")),
        "n_recommended_unique": sum(1 for r in out if r.get("recommended_for_benchmark")),
        "not_collected": registry.NOT_COLLECTED,
        "grok2_file_not_loaded": {"path": "files/grok-2/tokenizer.tok.json",
                                  "sha256": sha256_file(os.path.join(B, "files", "grok-2", "tokenizer.tok.json")),
                                  "repo": "xai-org/grok-2", "revision": "daf4395a80ad177386cfe39641b64fc12b1d70ed"},
        "crosschecks": {"kimi-k2 conversion vs reference": {k: v for k, v in KIMI.items() if k != "pattern"}, **CROSS, **CROSS_ML},
        "baselines": out,
    }
    json.dump(man, open(os.path.join(B, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("working:", man["n_working"], "/", len(out), " lossless:", man["n_lossless"], " recommended:", man["n_recommended_unique"])


if __name__ == "__main__":
    main()
