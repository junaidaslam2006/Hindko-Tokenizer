# -*- coding: utf-8 -*-
"""Shared helpers for the refreshed competitor sweep (sota/competitors).

- make_spec(rec): a load_baselines-style spec {name, loader, load_path, hf_class, ...} for a downloaded tokenizer
- load(spec): the same Baseline object baselines/load_baselines.py builds (tokenizer.json via `tokenizers`, SentencePiece
  natively, vocab-only repos via an explicit transformers class, local files only, never remote code)
- adapter(spec): an eval/adapters.py adapter around that object, exactly like adapters.from_baseline()
Nothing downloaded is ever imported or executed.
"""
import json
import os
import sys

os.environ.setdefault("HF_HOME", r"F:\Hindko\_tokenizer\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOK = r"F:\Hindko\_tokenizer"
EVAL = os.path.join(TOK, "eval")
BASE = os.path.join(TOK, "baselines")
for p in (EVAL, BASE, os.path.join(BASE, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import load_baselines as LB  # noqa: E402

TEST = os.path.join(TOK, "data", "test_strict.jsonl")
TEST_MANIFEST = os.path.join(TOK, "data", "test_manifest.json")
RELEASED = r"F:\Hindko\tokenizer\tokenizer.json"
RELEASED_SHA = "49f301c52363a09a1fc1925359af3ddabd7887495de02354cfa4092cba51da41"

# transformers classes allowed for vocab-only repos (no tokenizer.json). Anything else -> default by file kind.
BPE_CLASSES = {"GPT2Tokenizer", "RobertaTokenizer", "BartTokenizer", "LongformerTokenizer", "CodeGenTokenizer",
               "Qwen2Tokenizer", "BlenderbotTokenizer", "LEDTokenizer", "MvpTokenizer", "DebertaTokenizer", "GPTNeoXTokenizer"}
WP_CLASSES = {"BertTokenizer", "DistilBertTokenizer", "ElectraTokenizer", "MobileBertTokenizer", "SqueezeBertTokenizer",
              "LayoutLMTokenizer", "FunnelTokenizer", "MPNetTokenizer", "ConvBertTokenizer", "RetriBertTokenizer"}


def _cfg_class(d):
    p = os.path.join(d, "tokenizer_config.json")
    if not os.path.exists(p):
        return None
    try:
        c = json.load(open(p, encoding="utf-8")).get("tokenizer_class")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(c, str) and c.endswith("Fast"):
        c = c[:-4]
    return c


def make_spec(rec):
    """rec: a _download_log.json record. Returns (spec, note) or (None, reason)."""
    d = os.path.join(ROOT, rec["local_dir"])
    sub = os.path.join(d, rec["dir"]) if rec.get("dir") else d
    kind = rec["kind"]
    name = rec["name"]
    base = {"name": name, "repo": rec["repo"], "revision": rec.get("revision")}
    if kind == "tokenizer.json":
        return dict(base, loader="hf_json", load_path=os.path.join(sub, "tokenizer.json")), None
    if kind == "tekken.json":
        conv = os.path.join(sub, "tokenizer.converted_from_tekken.json")
        if not os.path.exists(conv):
            from tiktoken_tools import load_tekken, to_tokenizers_json
            ranks, pattern, n_special, specials, cfg = load_tekken(os.path.join(sub, "tekken.json"))
            # tekken ids: 0..n_special-1 = special tokens, ranks shifted by n_special
            sp = [(i, (s.get("token_str") if isinstance(s, dict) else str(s)) or "<SPECIAL_%d>" % i) for i, s in enumerate(specials)]
            shifted = {t: r + n_special for t, r in ranks.items()}
            tk = _tekken_tokenizer(shifted, pattern, sp)
            tk.save(conv)
        return dict(base, loader="hf_json", load_path=conv, conversion="tekken.json -> tokenizers BPE (baselines tiktoken_tools)"), None
    if kind == "spm":
        f = [x for x in rec["files"] if x.endswith(".model")][0]
        return dict(base, loader="spm", load_path=os.path.join(d, f)), None
    if kind == "source.spm":
        f = [x for x in rec["files"] if x.endswith("source.spm")][0]
        return dict(base, loader="spm", load_path=os.path.join(d, f), note="Marian source-side SentencePiece model"), None
    if kind == "vocab.json+merges.txt":
        c = _cfg_class(sub)
        cls = c if c in BPE_CLASSES else ("RobertaTokenizer" if c and "Roberta" in c else "GPT2Tokenizer")
        return dict(base, loader="auto", hf_class=cls, load_path=sub), None
    if kind == "vocab.txt":
        c = _cfg_class(sub)
        cls = c if c in WP_CLASSES else "BertTokenizer"
        return dict(base, loader="auto", hf_class=cls, load_path=sub), None
    if kind in ("tiktoken.model", "*.tiktoken"):
        return None, "tiktoken rank file: the split regex is not in a data file (it lives in the repo's Python code); not converted"
    if kind == "tokenizer.tok.json":
        return None, "xAI tokenizer.tok.json: the pre-tokenizer regex is not in the file; not converted (as baselines/manifest.json)"
    return None, "unknown kind %s" % kind


def _tekken_tokenizer(ranks, pattern, specials):
    """tiktoken_tools.to_tokenizers_json, with tekken's special-token block placed BEFORE the ranks."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from tiktoken_tools import bytes_to_unicode

    enc = bytes_to_unicode()

    def s(b):
        return "".join(enc[x] for x in b)

    vocab = {s(t): r for t, r in ranks.items()}
    for i, c in specials:
        vocab[c] = i
    merges = []
    for tok, rank in ranks.items():
        if len(tok) == 1:
            continue
        local = [(tok[:i], tok[i:], rank) for i in range(1, len(tok)) if tok[:i] in ranks and tok[i:] in ranks]
        local.sort(key=lambda x: (ranks[x[0]], ranks[x[1]]))
        merges.extend(local)
    merges.sort(key=lambda x: x[2])
    tk = Tokenizer(BPE(vocab=vocab, merges=[(s(a), s(b)) for a, b, _ in merges], ignore_merges=True, byte_fallback=False,
                       fuse_unk=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([AddedToken(c, special=True, normalized=False) for _, c in specials])
    return tk


def load(spec):
    return LB.build(spec)


def adapter(spec, extra_meta=None):
    import adapters as A
    b = load(spec)
    lp = spec.get("load_path")
    if spec["loader"] == "hf_json":
        a = A.HFAdapter(path=lp, tokenizer=b.raw, name=spec["name"])
    elif spec["loader"] == "spm":
        a = A.SPAdapter(path=lp, sp=b.raw, name=spec["name"], newline_wrapper=False)
    elif spec["loader"] == "auto":
        a = A.TransformersAdapter(b)
        a.path = lp
    else:
        raise ValueError(spec["loader"])
    a.kind = "refresh:" + spec["loader"]
    a.unk_ids = frozenset(b.unk_ids) | a.unk_ids
    a.baseline = dict(extra_meta or {}, name=spec["name"], repo=spec.get("repo"), revision=spec.get("revision"),
                      loader=spec["loader"], hf_class=spec.get("hf_class"), conversion=spec.get("conversion"), leaky=False,
                      load_baselines_vocab_size=b.vocab_size)
    a._baseline_obj = b
    return a


def released_adapter():
    import adapters as A
    import harness as H
    if H.sha256_file(RELEASED) != RELEASED_SHA:
        raise SystemExit("released tokenizer.json sha256 does not match RELEASE_MANIFEST.json")
    return A.HFAdapter(path=RELEASED, name="hindko-tokenizer-1.0.0-released")


def load_test():
    docs = [json.loads(line) for line in open(TEST, encoding="utf-8")]
    return docs
