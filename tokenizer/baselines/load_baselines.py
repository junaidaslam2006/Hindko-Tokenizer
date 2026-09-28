"""Load competitor tokenizer baselines for the Hindko tokenizer benchmark (offline; files live in ./files).

    from load_baselines import load_baseline, list_baselines
    tok = load_baseline("gpt-4o")
    ids = tok.encode(text)      # list[int]; NO special tokens (BOS/EOS/CLS/SEP) are added
    back = tok.decode(ids)      # str; special tokens are not skipped
    tok.vocab_size              # int; number of distinct token ids incl. added/special tokens
    tok.tokens(text)            # list[str] token strings (inspection only)

list_baselines(tier=None, working_only=True, recommended_only=False, lossless_only=False) -> list of names from manifest.json.
Tiers: core, frontier-2026, urdu, regional, reference.  "hindko-probe-bpe32k" is a LEAKY reference (it saw the test data).

Not every baseline is lossless on Hindko: manifest.json -> baselines[].lossless (exact round trip on 8 samples AND on the
2,619-text corpus battery), .newline_handling (several SentencePiece/WordPiece tokenizers turn line breaks into spaces,
which silently saves them tokens) and .roundtrip_battery (what each one loses).  Compare token counts on single-line
segments, or report newline_handling next to them.

Backends (chosen per baseline in manifest.json, key "loader"):
  hf_json         tokenizers.Tokenizer.from_file(tokenizer.json), truncation/padding disabled
  tiktoken_ranks  same, on a tokenizer.json converted here from the official tiktoken rank file (verified)
  spm / local_spm sentencepiece.SentencePieceProcessor on the native .model file
  auto            transformers tokenizer class given by "hf_class" (else AutoTokenizer); local files only, no remote code

PITFALL (measured 2026-09-26): transformers 5.3 AutoTokenizer does NOT reproduce 12 repos' tokenizer.json on the 8 samples
(manifest.json -> transformers_crosscheck / transformers_pipeline_diff): for LlamaTokenizerFast configs (DeepSeek-V3/R1,
Sarvam-M) it rebuilds a Metaspace pipeline and drops Arabic-script text entirely; for GPT2Tokenizer / Qwen2Tokenizer configs
(gpt-4o, gpt-4, phi-4, phi-4-mini, olmo-2, minimax-m2, claude-legacy, qwen-3.5, qwen-3.8) it replaces the published split
regex with its class default (and drops a NFC/NFKC normalizer where there is one).  This module therefore loads
tokenizer.json with the `tokenizers` library directly; always use load_baseline() for benchmarking.
  bytes           UTF-8 bytes, id = byte + 3 (ByT5 convention)
No network access is needed or attempted.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")

__all__ = ["load_baseline", "list_baselines", "manifest"]


class Baseline:
    """Uniform wrapper: encode(text)->list[int], decode(ids)->str, vocab_size."""

    def __init__(self, name, loader, vocab_size, encode_fn, decode_fn, tokens_fn, unk_ids, raw):
        self.name = name
        self.loader = loader
        self.vocab_size = int(vocab_size)
        self._enc, self._dec, self._tok = encode_fn, decode_fn, tokens_fn
        self.unk_ids = frozenset(unk_ids)
        self.raw = raw  # underlying tokenizer object (for inspection)

    def encode(self, text):
        return list(self._enc(text))

    def decode(self, ids):
        return self._dec(list(ids))

    def tokens(self, text):
        return self._tok(text)

    def __repr__(self):
        return f"<Baseline {self.name} loader={self.loader} vocab_size={self.vocab_size}>"


def _path(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def _hf_json(name, loader, path):
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(_path(path))
    tk.no_truncation()
    tk.no_padding()
    unk = set()
    try:
        m = json.loads(tk.to_str())["model"]
        ut = m.get("unk_token")
        if m.get("type") == "Unigram" and m.get("unk_id") is not None:
            unk.add(int(m["unk_id"]))
        elif ut is not None and tk.token_to_id(ut) is not None:
            unk.add(tk.token_to_id(ut))
    except Exception:  # noqa: BLE001
        pass
    return Baseline(
        name, loader, len(tk.get_vocab(with_added_tokens=True)),
        lambda t: tk.encode(t, add_special_tokens=False).ids,
        lambda ids: tk.decode(ids, skip_special_tokens=False),
        lambda t: tk.encode(t, add_special_tokens=False).tokens,
        unk, tk)


def _spm(name, loader, path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=_path(path))
    unk = {sp.unk_id()} if sp.unk_id() >= 0 else set()
    return Baseline(name, loader, sp.get_piece_size(),
                    lambda t: sp.encode(t, out_type=int),
                    lambda ids: sp.decode(ids),
                    lambda t: sp.encode(t, out_type=str), unk, sp)


def _auto(name, loader, dirpath, hf_class=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import transformers

    # explicit class: repos without tokenizer.json/config.json cannot be resolved by AutoTokenizer in transformers 5.x
    cls = getattr(transformers, hf_class) if hf_class else transformers.AutoTokenizer
    tok = cls.from_pretrained(_path(dirpath), local_files_only=True, trust_remote_code=False)
    unk = {tok.unk_token_id} if tok.unk_token_id is not None else set()
    return Baseline(name, loader, len(tok),
                    lambda t: tok.encode(t, add_special_tokens=False),
                    lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    lambda t: tok.tokenize(t), unk, tok)


def _bytes(name, loader):
    # ByT5: 0=pad, 1=eos, 2=unk, 3..258 = bytes 0..255, 259..383 = 125 sentinel ids
    def dec(ids):
        return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")

    return Baseline(name, loader, 384,
                    lambda t: [b + 3 for b in t.encode("utf-8")], dec,
                    lambda t: [bytes([b]).hex() for b in t.encode("utf-8")], set(), None)


def build(spec):
    """Build a Baseline from a manifest entry (dict with name, loader, load_path)."""
    name, loader, p = spec["name"], spec["loader"], spec.get("load_path")
    if loader in ("hf_json", "tiktoken_ranks"):
        return _hf_json(name, loader, p)
    if loader in ("spm", "local_spm"):
        return _spm(name, loader, p)
    if loader == "auto":
        return _auto(name, loader, p, spec.get("hf_class"))
    if loader == "bytes":
        return _bytes(name, loader)
    raise ValueError(f"unknown loader {loader!r}")


def manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def list_baselines(tier=None, working_only=True, recommended_only=False, lossless_only=False):
    """recommended_only=True drops baselines whose encodings equal an earlier baseline's on all samples and the battery.
    lossless_only=True keeps only baselines with decode(encode(x)) == x on all samples and the whole battery."""
    out = []
    for e in manifest()["baselines"]:
        if working_only and not e.get("working"):
            continue
        if recommended_only and not e.get("recommended_for_benchmark"):
            continue
        if lossless_only and not e.get("lossless"):
            continue
        if tier is not None and e["tier"] != tier:
            continue
        out.append(e["name"])
    return out


def load_baseline(name):
    for e in manifest()["baselines"]:
        if e["name"] == name:
            if not e.get("load_path") and e["loader"] != "bytes":
                raise RuntimeError(f"{name}: not available ({e.get('status')})")
            return build(e)
    raise KeyError(f"unknown baseline {name!r}; see list_baselines()")


if __name__ == "__main__":
    import sys

    for n in sys.argv[1:] or list_baselines():
        t = load_baseline(n)
        s = "پشور اچ"  # a short Hindko string
        ids = t.encode(s)
        print(f"{n:24s} vocab={t.vocab_size:7d} n_tokens={len(ids):3d} roundtrip={t.decode(ids) == s}")
