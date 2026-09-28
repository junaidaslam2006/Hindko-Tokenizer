# -*- coding: utf-8 -*-
"""Shared helpers for Track B (PLAN.md section 9): vocabulary extension of open LLM tokenizers by
continued BPE (Purason et al. 2026, "Teaching Old Tokenizers New Words", reimplemented from the paper).

Nothing here reads the test split: data comes from the materialised views data/train_D1.jsonl (permissive
train) and data/dev_strict.jsonl (strict validation), whose sha256 are checked against data/data_manifest.json,
whose manifest hash is checked against FROZEN.json, and every uid is checked against the frozen split manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

TB = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.dirname(TB)
DATA_DIR = os.path.join(TOK, "data")
EVAL_DIR = os.path.join(TOK, "eval")
BASELINES_FILES = os.path.join(TOK, "baselines", "files")
FROZEN_PATH = os.path.join(TOK, "FROZEN.json")
WORK = os.path.join(TB, "work")          # per-base extension builds (ordered merge lists, unit statistics)
SWEEP = os.path.join(TB, "sweep")        # materialised tokenizer.json for every (base, k)
RESULTS = os.path.join(TB, "results")    # evaluation outputs
DELIVER = os.path.join(TB, "deliver")    # ready-to-use folders at the knee k
MANIFEST_SHA = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"

os.environ.setdefault("HF_HOME", os.path.join(TOK, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# k grid of PLAN 9.2 ("1k ... 16k"), taken as powers of two like the PLAN's own vocabulary sizes
K_GRID = [1024, 2048, 4096, 8192, 16384]
K_MAX = max(K_GRID)

# Base tokenizers. 'bytelevel' = HF byte-level BPE (tiktoken/GPT-2 family); 'spbpe' = SentencePiece BPE with
# byte fallback, used through its HF tokenizer.json (the form transformers' fast tokenizer loads).
BASES: Dict[str, dict] = {
    "qwen-3":   {"kind": "bytelevel", "dir": os.path.join(BASELINES_FILES, "qwen-3"),
                 "models": "Qwen3 (0.6B-235B), Qwen2/2.5 share the encodings", "license": "apache-2.0",
                 "repo": "Qwen/Qwen3-0.6B"},
    "qwen-3.5": {"kind": "bytelevel", "dir": os.path.join(BASELINES_FILES, "qwen-3.5"),
                 "models": "Qwen3.5 (0.8B-397B), Qwen3.8 shares the encodings", "license": "apache-2.0",
                 "repo": "Qwen/Qwen3.5-0.8B"},
    "llama-3":  {"kind": "bytelevel", "dir": os.path.join(BASELINES_FILES, "llama-3"),
                 "models": "Llama 3/3.1/3.2/3.3 (also Alif-1.0, Qalb-1.0)", "license": "llama3.1",
                 "repo": "NousResearch/Meta-Llama-3.1-8B-Instruct (byte-identical mirror of meta-llama/Llama-3.1-8B-Instruct)"},
    "gemma-3":  {"kind": "spbpe", "dir": os.path.join(BASELINES_FILES, "gemma-3"),
                 "models": "Gemma 3 (1B-27B)", "license": "gemma",
                 "repo": "unsloth/gemma-3-1b-it (byte-identical mirror of google/gemma-3-1b-it)"},
    "gemma-4":  {"kind": "spbpe", "dir": os.path.join(BASELINES_FILES, "gemma-4"),
                 "models": "Gemma 4 (E2B-31B); same pieces and merges as Gemma 3, different control tokens",
                 "license": "apache-2.0", "repo": "google/gemma-4-E2B-it"},
}
BASE_ORDER = ["qwen-3", "llama-3", "gemma-3", "qwen-3.5", "gemma-4"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dump_json(obj, path: str, indent: Optional[int] = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------ frozen inputs
_FROZEN_OK = None


def verify_frozen() -> dict:
    """Check FROZEN.json, the split manifest hash, normalize.py and the materialised views."""
    global _FROZEN_OK
    if _FROZEN_OK is not None:
        return _FROZEN_OK
    fr = load_json(FROZEN_PATH)
    man = fr["split_manifest"]
    got = sha256_file(man["path"])
    if got != MANIFEST_SHA or man["sha256"] != MANIFEST_SHA:
        raise SystemExit("split manifest hash %s != frozen %s" % (got, MANIFEST_SHA))
    nz = fr["normalize"]
    if sha256_file(nz["path"]) != nz["sha256"]:
        raise SystemExit("normalize.py changed since FROZEN.json")
    dm = load_json(os.path.join(DATA_DIR, "data_manifest.json"))
    if dm["split_manifest_sha256"] != MANIFEST_SHA:
        raise SystemExit("data_manifest.json was built from another split manifest")
    _FROZEN_OK = {"split_manifest_sha256": got, "normalize_version": nz["version"], "normalize_sha256": nz["sha256"],
                  "data_manifest": {k: dm["views"][k]["sha256"] for k in ("train_D1", "dev_strict", "dev_permissive")}}
    return _FROZEN_OK


_SPLIT = None


def manifest_split() -> Dict[str, str]:
    global _SPLIT
    if _SPLIT is None:
        verify_frozen()
        _SPLIT = {}
        with open(load_json(FROZEN_PATH)["split_manifest"]["path"], encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _SPLIT[r["uid"]] = r["split"]
    return _SPLIT


def normalizer():
    sys.path.insert(0, r"F:\Hindko\_pipeline")
    from hp.normalize import normalize, NORMALIZATION_VERSION
    assert NORMALIZATION_VERSION == verify_frozen()["normalize_version"]
    return normalize


def load_view(name: str) -> List[dict]:
    """A materialised view ('train_D1', 'dev_strict' or 'dev_permissive'), verified: file sha256 == data_manifest, every uid in the
    expected split of the frozen manifest (never 'test'), and normalize(text) == text for every document
    (the views were written through hp.normalize 1.0.1; normalize() is applied again here and must be a no-op)."""
    assert name in ("train_D1", "dev_strict", "dev_permissive"), name
    fr = verify_frozen()
    path = os.path.join(DATA_DIR, name + ".jsonl")
    if sha256_file(path) != fr["data_manifest"][name]:
        raise SystemExit("%s does not match data_manifest.json" % path)
    want = "train" if name == "train_D1" else "validation"
    split = manifest_split()
    norm = normalizer()
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = split.get(d["uid"])
            if s != want:
                raise SystemExit("uid %s of %s is in split %r (expected %r)" % (d["uid"], name, s, want))
            t = norm(d["text"])
            if t != d["text"]:
                raise SystemExit("normalize() is not a no-op on %s uid %s" % (name, d["uid"]))
            d["text"] = t
            docs.append(d)
    return docs


# ------------------------------------------------------------------------------------------------ scripts
import regex as _re  # noqa: E402

# "Arabic-script character" = Script_Extensions contains Arabic, or the code point lies in an Arabic Unicode block
# (the block clause adds the few Arabic-only Common-script signs such as U+0605 and U+08E2, so that a byte prefix
# like d8 or e0 a3 counts as Arabic-only; Syriac U+0700..U+074F stays outside).
ARABIC_SCX = _re.compile(r"[\p{scx=Arabic}\p{Block=Arabic}\p{Block=Arabic_Supplement}\p{Block=Arabic_Extended_A}"
                         r"\p{Block=Arabic_Extended_B}\p{Block=Arabic_Extended_C}\p{Block=Arabic_Presentation_Forms_A}"
                         r"\p{Block=Arabic_Presentation_Forms_B}\p{Block=Arabic_Mathematical_Alphabetic_Symbols}"
                         r"\p{Block=Rumi_Numeral_Symbols}]")
_SCRIPTS = ["Arabic", "Latin", "Common", "Inherited", "Devanagari", "Gurmukhi", "Bengali", "Cyrillic", "Greek",
            "Han", "Hiragana", "Katakana", "Hangul", "Syriac", "Thaana", "Hebrew", "Armenian", "Georgian", "Thai",
            "Ethiopic", "Tamil", "Telugu", "Kannada", "Malayalam", "Gujarati", "Oriya", "Sinhala", "Tibetan"]
_SCRIPT_RX = [(s, _re.compile(r"\p{Script=%s}" % s)) for s in _SCRIPTS]
_SCRIPT_CACHE: Dict[str, str] = {}


def char_script(c: str) -> str:
    s = _SCRIPT_CACHE.get(c)
    if s is None:
        s = "Other"
        for name, rx in _SCRIPT_RX:
            if rx.match(c):
                s = name
                break
        _SCRIPT_CACHE[c] = s
    return s


def has_arabic(text: str) -> bool:
    return ARABIC_SCX.search(text) is not None


# ----------------------------------------------------------------------------------------- base loading
def base_tokenizer_path(base: str) -> str:
    return os.path.join(BASES[base]["dir"], "tokenizer.json")


def load_base_json(base: str) -> dict:
    return load_json(base_tokenizer_path(base))


def next_free_id(j: dict) -> int:
    ids = list(j["model"]["vocab"].values()) + [a["id"] for a in j.get("added_tokens", [])]
    return max(ids) + 1


def config_only_added_tokens(base: str) -> List[dict]:
    """Added tokens that the base's tokenizer_config.json defines (added_tokens_decoder) but its tokenizer.json lacks.
    transformers adds them at their config ids when it loads the base (Qwen3.5: 7 audio/TTS tokens, ids
    248,070-248,076), so new tokens must start after them and the extended tokenizer.json carries them too."""
    p = os.path.join(BASES[base]["dir"], "tokenizer_config.json")
    j = load_base_json(base)
    have = {a["id"]: a["content"] for a in j.get("added_tokens", [])}
    out = []
    if os.path.exists(p):
        for i, v in sorted(load_json(p).get("added_tokens_decoder", {}).items(), key=lambda x: int(x[0])):
            i = int(i)
            if i in have:
                assert have[i] == v["content"], (base, i)
                continue
            out.append({"id": i, "content": v["content"], "single_word": v.get("single_word", False),
                        "lstrip": v.get("lstrip", False), "rstrip": v.get("rstrip", False),
                        "normalized": v.get("normalized", False), "special": v.get("special", True)})
    return out


def first_new_id(base: str) -> int:
    j = load_base_json(base)
    return max([next_free_id(j)] + [a["id"] + 1 for a in config_only_added_tokens(base)])


def env_threads(n: int = 2):
    os.environ.setdefault("RAYON_NUM_THREADS", str(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if n > 1 else "false")
