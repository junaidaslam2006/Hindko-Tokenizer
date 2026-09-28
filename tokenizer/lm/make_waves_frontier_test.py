# -*- coding: utf-8 -*-
"""Write lm/WAVES_frontier_test.json: the fixed candidate list of the REPORT-ONLY frontier LM comparison
(colab/FRONTIER_LM.md). Deterministic (no timestamps); read by colab/build_bundle.py --final-test --report-only.

Candidates, in run order (run_all.py trains candidate by candidate, so the most important finish first; the
least important are last, so they can be dropped with --only or by stopping early):
  1. hindko-1.0.0          the RELEASED tokenizer (F:/Hindko/tokenizer/tokenizer.json, the canonical encoder);
                           baseline of this bundle, so every delta in summary.txt is 'vs the released tokenizer'
  2. R2-A4-SPnat-D2-32k    reproduction anchor: the candidate sp.model the one-shot test scored (bundle
                           f54c929ba1ab). Same pieces and ids as the release; its test encodings are identical, its
                           train encodings differ from the release on 29 of 16,015 documents (exact Viterbi ties,
                           release_build/sp32k/EQUIVALENCE.md). Its seeds 1-3 must reproduce the published test runs.
  3-10. external tokenizers of baselines/manifest.json, loaded by name through baselines/load_baselines.py:
       gpt-4o (o200k), gemma-3, llama-3, qwen-3.5, deepseek-v3, llama-4, roberta-urdu, bloom.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
REL = r"F:\Hindko\tokenizer"
OUT = os.path.join(HERE, "WAVES_frontier_test.json")
sys.path.insert(0, os.path.join(TOK, "colab"))
sys.dont_write_bytecode = True


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


EXTERNAL = [  # (name, why it is in the comparison)
    ("gpt-4o", "OpenAI o200k_base (GPT-4o / GPT-4.1 / o-series; same encodings as gpt-oss and phi-4-mini)"),
    ("gemma-3", "Google Gemma 3 SentencePiece 262k (same encodings as Gemma 4)"),
    ("llama-3", "Meta Llama 3 / 3.1 tiktoken-BPE 128k (also Alif-1.0 and Qalb-1.0, the Urdu Llama-3 models)"),
    ("qwen-3.5", "Alibaba Qwen 3.5 byte-level BPE 248k (same encodings as Qwen 3.8)"),
    ("deepseek-v3", "DeepSeek-V3 byte-level BPE 128k (same encodings as DeepSeek-R1, V4, V4.1)"),
    ("llama-4", "Meta Llama 4 BPE 202k (dropped first if time is short, after BLOOM and RoBERTa-Urdu)"),
    ("roberta-urdu", "UrduHack RoBERTa-Urdu byte-level BPE 52k: best lossless external tokenizer on test by "
                     "bytes/token (5.709); monolingual Urdu (dropped second if time is short)"),
    ("bloom", "BigScience BLOOM byte-level BPE 250k: best lossless multilingual frontier-era tokenizer after "
              "RoBERTa-Urdu on test bytes/token (5.122) (dropped first if time is short)"),
]


def main():
    import build_bundle as bb
    rm = json.load(open(os.path.join(REL, "RELEASE_MANIFEST.json"), encoding="utf-8"))
    rel_tok = os.path.join(REL, "tokenizer.json")
    rel_sha = sha(rel_tok)
    if rel_sha != rm["tokenizer_files"]["tokenizer.json"]:
        raise SystemExit("released tokenizer.json sha256 differs from RELEASE_MANIFEST.json")
    anchor = os.path.join(TOK, "candidates", "round2", "standard", "tok", "R2-A4-SPnat-D2-32k", "sp.model")
    ft = json.load(open(os.path.join(HERE, "WAVES_final_test.json"), encoding="utf-8"))
    ft_anchor = next(w for w in ft["waves"] if w["id"] == "R2-A4-SPnat-D2-32k")
    if sha(anchor) != ft_anchor["tokenizer_sha256"]:
        raise SystemExit("anchor sp.model sha256 differs from WAVES_final_test.json")
    waves = [
        {"id": "hindko-1.0.0", "tokenizer_path": rel_tok, "encoder_kind": "hf", "baseline": True,
         "tokenizer_sha256": rel_sha, "vocab_size": 32768,
         "final_test_role": "released tokenizer (baseline of this report-only comparison)",
         "notes": "F:/Hindko/tokenizer/tokenizer.json, the canonical encoder of release %s (R2-A4-SPnat-D2-32k, "
                  "SentencePiece Unigram, 32,768 ids); the newline convention is inside the tokenizer.json"
                  % rm["release"].split(" (")[0]},
        {"id": "R2-A4-SPnat-D2-32k", "tokenizer_path": anchor, "encoder_kind": "sentencepiece",
         "tokenizer_sha256": ft_anchor["tokenizer_sha256"], "vocab_size": 32768,
         "final_test_role": "reproduction anchor (report-only): the candidate file scored in the one-shot test",
         "notes": "same file, encoder (eval/adapters.py SPAdapter, PLAN 1.1 newline wrapper) and arrays as bundle "
                  "f54c929ba1ab; seeds 1-3 must reproduce analysis/TEST_RESULTS.md section 3 (1.21299 / 1.20906 / "
                  "1.20803) on a T4 in fp16"},
    ]
    for name, why in EXTERNAL:
        spec = bb.baseline_spec(name)
        p, s = bb.baseline_path_sha(spec)
        if not spec.get("lossless"):
            raise SystemExit("%s is not lossless in baselines/manifest.json" % name)
        waves.append({"id": name, "tokenizer_path": p, "encoder_kind": "baseline", "baseline_name": name,
                      "tokenizer_sha256": s, "vocab_size": spec.get("vocab_size"),
                      "provider": spec.get("provider"), "family": spec.get("family"), "repo": spec.get("repo"),
                      "revision": spec.get("revision"), "license": spec.get("license"),
                      "final_test_role": "external frontier tokenizer (report-only)", "notes": why})
    doc = {
        "what": "Fixed candidate list of the REPORT-ONLY frontier LM comparison on the strict test split "
                "(colab/FRONTIER_LM.md): the same small LM (the 'confirm' recipe) trained on the same Hindko bytes "
                "with each tokenizer. Written by lm/make_waves_frontier_test.py (deterministic). No decision "
                "depends on it: the released tokenizer is fixed (analysis/AMENDMENT_1.md).",
        "baseline": "hindko-1.0.0",
        "protocol": "Stage 4 'confirm' recipe unchanged (hk_lm.py d=192 L=4 H=4, full permissive train split "
                    "train_D1, 1 epoch = 3,663 steps, byte-matched), LR 1e-3 (the dev choice), seeds 1-3, no LR "
                    "sweep, GPU fp16 training / fp32 evaluation, evaluated on test_strict (491 documents)",
        "order": "run order = priority; the last three (llama-4, roberta-urdu, bloom) are the first to drop",
        "waves": waves,
    }
    ids = [w["id"] for w in waves]
    doc["candidate_ids_sha256"] = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("wrote", OUT, ids)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
