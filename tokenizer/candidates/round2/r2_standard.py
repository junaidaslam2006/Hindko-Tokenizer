# -*- coding: utf-8 -*-
"""Round 2: A1 byte-level BPE (P1r3, D2) and A4 SentencePiece Unigram (SPnat, D2) through the Stage 1 code.

    python r2_standard.py R2-A1-P1r3-D2-48k | R2-A4-SPnat-D2-32k | R2-A4-SPnat-D2-48k

Reuses candidates/standard/sweep_lib.py (trainers, SentencePiece->HF export and its equivalence test, G2 remedy)
and candidates/standard/run_sweep.py (process(): double training for G4, harness --gates all on dev_strict with
R1 on train_D1, G1 on dev_permissive, G2 for superword pieces, the pre-declared G2 remedy, R1 on the own mix).
Two things are redirected, nothing else:
  1. output directories: sweep_lib.TOKDIR / RESDIR -> candidates/round2/standard/{tok,results};
  2. run_sweep.train_config: its SentencePiece branch hard-codes view = 'train_D1' (Stage 1 trained SentencePiece
     on D1 only). train_config_r2 is that branch verbatim with view = train_<data mix> (train_D2 here). The HF
     branch (A1) already reads c['data'] and is used unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_common as RC  # noqa: E402

sys.path.insert(0, RC.STD)
import sweep_lib as L  # noqa: E402
import run_sweep as RS  # noqa: E402

H = L.H
OUT = os.path.join(RC.R2, "standard")
L.TOKDIR = os.path.join(OUT, "tok")
L.RESDIR = os.path.join(OUT, "results")

CONFIGS = {
    RC.IDS["A1-48k"]: {"algo": "A1", "pretok": "P1r3", "data": "D2", "size": "48k"},
    RC.IDS["A4-32k"]: {"algo": "A4", "pretok": "SPnat", "data": "D2", "size": "32k"},
    RC.IDS["A4-48k"]: {"algo": "A4", "pretok": "SPnat", "data": "D2", "size": "48k"},
}
ORIG_TRAIN_CONFIG = RS.train_config


def config(cid):
    c = dict(CONFIGS[cid])
    c["id"] = cid
    c["vocab_total"] = L.SIZES[c["size"]]
    return c


def train_config_r2(c):
    """run_sweep.train_config with the SentencePiece training view taken from the config's data mix."""
    if c["algo"] not in L.SP_ALGOS:
        return ORIG_TRAIN_CONFIG(c)
    d = L.tok_dir(c)
    g4 = os.path.join(d, "g4")
    os.makedirs(g4, exist_ok=True)
    meta_path = os.path.join(d, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("trained") and all(os.path.exists(os.path.join(d, f)) for f in meta["files"]):
            L.log(c["id"], "training found, reused")
            return meta
    V, algo = c["vocab_total"], c["algo"]
    meta = {"id": c["id"], "config": c, "algorithm": L.ALGOS[algo], "vocab_total": V,
            "special_block": {"n": 64, "ids": "0..63", "tokens": L.SPECIALS},
            "created_utc": L.now_utc(), "versions": L.versions(), "code": RS.code_hashes(),
            "round2_driver": RC.code_sha("r2_standard.py", "r2_common.py"),
            "frozen": H._frozen_short()}
    times = []
    view = L.train_view(c["data"])                      # round 2: D2 (Stage 1 hard-coded 'train_D1' here)
    inp = L.lines_file(view)
    rec = L.data_manifest()["views"][view]["lines_txt"]
    meta["train_data"] = {"view": view, "file": inp, "sha256": rec["sha256"], "non_empty_lines": rec["non_empty_lines"],
                          "unit": "non-empty lines (PLAN 1.1; encoded line by line with the '\\n' piece)"}
    meta["pretokenizer"] = {"id": c["pretok"], "desc": "SentencePiece native (U+2581 whitespace, unicode-script "
                            "and number splitting, split_digits=true)" + (
                                "; split_by_whitespace=false (pieces may span words)" if algo == "A8" else "")}
    prefix = os.path.join(d, "sp")
    mt = "bpe" if algo == "A3" else "unigram"
    for k in range(2):
        t0 = time.time()
        params = L.train_sp(inp, prefix, V, mt, algo != "A8")
        times.append(round(time.time() - t0, 1))
        if k == 0:
            shutil.move(prefix + ".model", os.path.join(g4, "sp.model"))
            shutil.move(prefix + ".vocab", os.path.join(g4, "sp.vocab"))
        L.log(c["id"], "SentencePiece training %d/2: %.1fs" % (k + 1, times[-1]))
    meta["trainer"] = "sentencepiece %s SentencePieceTrainer.train" % L.versions()["sentencepiece"]
    meta["trainer_params"] = params
    meta["normalizer"] = "none (normalization_rule_name=identity, remove_extra_whitespaces=False)"
    meta["encoder"] = "SentencePiece native + harness newline wrapper (adapters.SPAdapter, newline_wrapper=True)"
    for sub in ("", "g4"):
        dd = os.path.join(d, sub) if sub else d
        with open(os.path.join(dd, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(L.sp_to_hf_json(os.path.join(dd, "sp.model")))
    meta["files"] = {f: L.sha256_file(os.path.join(d, f)) for f in ("sp.model", "sp.vocab", "tokenizer.json")}
    meta["g4_files"] = {f: L.sha256_file(os.path.join(g4, f)) for f in ("sp.model", "sp.vocab", "tokenizer.json")}
    meta["native_file"] = "sp.model"
    t0 = time.time()
    meta["hf_export"] = L.sp_hf_equivalence(os.path.join(d, "sp.model"), os.path.join(d, "tokenizer.json"),
                                            L.dev_texts())
    meta["hf_export"]["seconds"] = round(time.time() - t0, 1)
    meta["hf_export"]["what"] = ("tokenizer.json is an EXPORT of sp.model for the newline-wrapped encoder "
                                 "(sweep_lib.sp_to_hf_json); the candidate's identity is the native encoder")
    meta["randomness"] = ("none: no sampling, no shuffling (SentencePiece input_sentence_size=0, "
                          "shuffle_input_sentence=False); fixed input order; fixed thread counts")
    meta["threads"] = {"RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"), "sentencepiece_num_threads": L.SP_THREADS}
    meta["train_seconds"] = times
    meta["g4_bytes_identical"] = meta["files"][meta["native_file"]] == meta["g4_files"][meta["native_file"]]
    meta["trained"] = True
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta


RS.train_config = train_config_r2


def main():
    cid = sys.argv[1]
    c = config(cid)
    RC.verify_frozen()
    L.verify_frozen()
    t0 = time.time()
    rec = RS.process(c)
    # round-2 provenance + NSL against the sweep's reference (harness.py nsl)
    nsl = RC.add_nsl(L.res_dir(c))
    p = RS.stage1_path(c)
    rec = json.load(open(p, encoding="utf-8"))
    rec["round2"] = {"driver": RC.code_sha("r2_standard.py", "r2_common.py"),
                     "redirects": ["sweep_lib.TOKDIR/RESDIR -> candidates/round2/standard",
                                   "run_sweep.train_config: SentencePiece view = train_<data> (train_D2)"],
                     "nsl_vs_A1-P1-D1-16k": nsl.get("overall"), "wall_seconds": round(time.time() - t0, 1)}
    RC.dump(rec, p)
    RC.log("done", cid, "gates", rec["gate_pass_final"], "NSL", nsl.get("overall"))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
