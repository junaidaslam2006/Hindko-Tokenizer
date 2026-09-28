# -*- coding: utf-8 -*-
"""Copy the released tokenizer files and their verification records into the release folder F:\\Hindko\\tokenizer\\.

Every copied file is re-hashed after the copy and must equal (a) the source file and (b), where the build manifest
lists it, release_build/sp32k/build_manifest.json -> outputs. Writes release_card/finalize/release_files.json.

    set PYTHONIOENCODING=utf-8
    python release_card\\finalize\\copy_release.py
"""
import datetime
import hashlib
import json
import os
import shutil
import sys

TOK = r"F:\Hindko\_tokenizer"
REL = r"F:\Hindko\tokenizer"
SRC = os.path.join(TOK, r"release_build\sp32k")
OUT = os.path.join(TOK, r"release_card\finalize\release_files.json")

# (source path relative to TOK, destination relative to REL, role)
FILES = [
    (r"release_build\sp32k\tokenizer.json", "tokenizer.json", "canonical release encoder (HF tokenizers Unigram, float64 Viterbi)"),
    (r"release_build\sp32k\tokenizer_config.json", "tokenizer_config.json", "transformers config: special tokens, ChatML template, no automatic BOS/EOS"),
    (r"release_build\sp32k\special_tokens_map.json", "special_tokens_map.json", "special-token roles"),
    (r"release_build\sp32k\sp.model", "sp.model", "SentencePiece convenience copy (float32 Viterbi; long-line divergence documented)"),
    (r"release_build\sp32k\sp.vocab", "sp.vocab", "SentencePiece piece list with release scores (text)"),
    (r"release_build\sp32k\gauge.json", "gauge.json", "per-character score offsets w(c): recover log-probabilities as score - sum w(c)"),
    # verification records of the release build
    (r"release_build\sp32k\EQUIVALENCE.md", r"eval\reports\EQUIVALENCE.md", "release-build equivalence report"),
    (r"release_build\sp32k\equivalence.json", r"eval\reports\equivalence.json", "equivalence counts per set and stress category"),
    (r"release_build\sp32k\exact_check.json", r"eval\reports\exact_check.json", "independent exact Viterbi vs SentencePiece, float32 bound per set"),
    (r"release_build\sp32k\build_manifest.json", r"eval\reports\build_manifest.json", "sha256 of every build input and output"),
    (r"release_build\sp32k\stress\stress_summary.json", r"eval\reports\stress_summary.json", "stress-set categories and sizes"),
    # one-shot test LM analysis
    (r"analysis\TEST_RESULTS.md", r"eval\reports\TEST_RESULTS.md", "one-shot test LM result report"),
    (r"analysis\test_results.json", r"eval\reports\test_results.json", "one-shot test LM results (machine-readable)"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    man = json.load(open(os.path.join(SRC, "build_manifest.json"), encoding="utf-8"))["outputs"]
    rows, bad = [], []
    for src_rel, dst_rel, role in FILES:
        s = os.path.join(TOK, src_rel)
        d = os.path.join(REL, dst_rel)
        hs = sha(s)
        name = os.path.basename(s)
        if os.path.dirname(src_rel) == r"release_build\sp32k" and name in man and man[name] != hs:
            bad.append("%s: source sha256 %s != build_manifest %s" % (src_rel, hs, man[name]))
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        hd = sha(d)
        if hd != hs:
            bad.append("%s: copy sha256 differs" % dst_rel)
        rows.append({"file": dst_rel.replace("\\", "/"), "source": src_rel.replace("\\", "/"), "bytes": os.path.getsize(d),
                     "sha256": hd, "equals_build_manifest": (man.get(name) == hd) if name in man and
                     os.path.dirname(src_rel) == r"release_build\sp32k" else None, "role": role})
        print("%-28s %s %9d  %s" % (dst_rel, hd, os.path.getsize(d), role))
    if bad:
        print("\n".join(bad))
        sys.exit(1)
    json.dump({"what": "files copied into F:\\Hindko\\tokenizer\\ by copy_release.py, sha256 re-checked after the copy",
               "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "script_sha256": sha(os.path.abspath(__file__)), "files": rows},
              open(OUT, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    print("ok:", len(rows), "files ->", OUT)


if __name__ == "__main__":
    main()
