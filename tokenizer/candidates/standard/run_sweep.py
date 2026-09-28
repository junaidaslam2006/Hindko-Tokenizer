# -*- coding: utf-8 -*-
"""PLAN.md Stage 1 worker: train (twice, for G4) -> [SentencePiece: HF export + equivalence test] ->
eval/harness.py on dev_strict with --gates all (G1-G5, R1 on train_D1, R2) -> extra gate work
(G1 on dev_permissive, G2 for superword pieces, R1 on the tokenizer's own training mix) -> pre-declared G2
remedy when G2 fails (PLAN 4.3) -> results/dev_strict/<id>/stage1.json.

    python run_sweep.py --worker K          claim configs from the shared queue (lock files) until none are left
    python run_sweep.py --only ID [ID ...]  process the given configs in this process
    python run_sweep.py --list              queue with status
    python run_sweep.py --clear-locks       remove claim files (only while no worker runs)
Resumable: a config whose stage1.json says complete is skipped; a finished double training (tok/<id>/meta.json)
is not repeated; the harness skips an up-to-date summary. Nothing here reads the test split (the harness
refuses test uids; train/dev views were materialised without test rows).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import sweep_lib as L

H, A = L.H, L.A
LOCKS = os.path.join(L.HERE, "locks")
REF_ID = "A1-P1-D1-16k"
# rough relative cost for longest-first ordering (refined by the timing pilot, see SWEEP.md)
ALGO_COST = {"A1": 1.0, "A2": 1.0, "A3": 2.5, "A4": 3.0, "A5": 2.0, "A8": 5.0}


def est_cost(c):
    return ALGO_COST[c["algo"]] * (0.6 + 0.4 * L.SIZES[c["size"]] / 16384)


def queue():
    """Reference first; then SentencePiece and HF jobs alternate, each longest-first (bounds how many
    memory-heavy SentencePiece trainings run at once under dynamic claiming)."""
    cs = L.all_configs()
    ref = [c for c in cs if c["id"] == REF_ID]
    key = lambda c: (-est_cost(c), c["id"])  # noqa: E731
    sp = sorted([c for c in cs if c["algo"] in L.SP_ALGOS], key=key)
    hf = sorted([c for c in cs if c["algo"] not in L.SP_ALGOS and c["id"] != REF_ID], key=key)
    out = []
    while sp or hf:
        if sp:
            out.append(sp.pop(0))
        if hf:
            out.append(hf.pop(0))
    return ref + out


def stage1_path(c):
    return os.path.join(L.res_dir(c), "stage1.json")


def is_complete(c):
    p = stage1_path(c)
    if not os.path.exists(p):
        return False
    try:
        return bool(json.load(open(p, encoding="utf-8")).get("complete"))
    except Exception:  # noqa: BLE001
        return False


def code_hashes():
    return {f: L.sha256_file(os.path.join(L.HERE, f)) for f in ("sweep_lib.py", "run_sweep.py")}


def _mem_mb():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return {"rss_mb": round(mi.rss / 2 ** 20), "peak_wset_mb": round(getattr(mi, "peak_wset", 0) / 2 ** 20)}
    except Exception:  # noqa: BLE001
        return None


# -------------------------------------------------------------------------------------------- training
def train_config(c):
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
            "created_utc": L.now_utc(), "versions": L.versions(), "code": code_hashes(),
            "frozen": H._frozen_short()}
    times = []
    if algo in L.SP_ALGOS:
        view = "train_D1"
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
    else:
        view = L.train_view(c["data"])
        texts = L.read_texts(view)
        rec = L.data_manifest()["views"][view]
        meta["train_data"] = {"view": view, "file": os.path.join(L.DATA, rec["file"]), "sha256": rec["sha256"],
                              "docs": rec["total"]["docs"], "unit": "whole documents (line breaks inside)"}
        pat = L.PRETOK[c["pretok"]]
        meta["pretokenizer"] = {"id": c["pretok"], "oniguruma": pat, "python_regex_twin": H.onig_to_py(pat),
                                "split_behavior": "isolated"}
        fn = {"A1": L.train_a1, "A2": L.train_a2, "A5": L.train_a5}[algo]
        for k, path in enumerate((os.path.join(g4, "tokenizer.json"), os.path.join(d, "tokenizer.json"))):
            t0 = time.time()
            _, params = fn(texts, V, pat, path)
            times.append(round(time.time() - t0, 1))
            L.log(c["id"], "HF training %d/2: %.1fs" % (k + 1, times[-1]))
        import tokenizers
        meta["trainer"] = "tokenizers %s %s" % (tokenizers.__version__,
                                                "UnigramTrainer" if algo == "A5" else "BpeTrainer")
        meta["trainer_params"] = params
        meta["normalizer"] = None
        meta["encoder"] = "HF tokenizers native (Tokenizer.encode, add_special_tokens=False)"
        meta["files"] = {"tokenizer.json": L.sha256_file(os.path.join(d, "tokenizer.json"))}
        meta["g4_files"] = {"tokenizer.json": L.sha256_file(os.path.join(g4, "tokenizer.json"))}
        meta["native_file"] = "tokenizer.json"
    meta["randomness"] = ("none: no sampling, no shuffling (SentencePiece input_sentence_size=0, "
                          "shuffle_input_sentence=False); fixed input order; fixed thread counts")
    meta["threads"] = {"RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"), "sentencepiece_num_threads": L.SP_THREADS}
    meta["train_seconds"] = times
    meta["g4_bytes_identical"] = meta["files"][meta["native_file"]] == meta["g4_files"][meta["native_file"]]
    meta["trained"] = True
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta


# ------------------------------------------------------------------------------------------ evaluation
def evaluate(c, ad, g4_path, out_dir, meta):
    t0 = time.time()
    summ = H.run(ad, data="dev_strict", gates="all", train="train_D1", g4_retrain=g4_path, out_dir=out_dir,
                 extra={"stage1_config": c, "tokenizer_meta": os.path.join(L.tok_dir(c), "meta.json")})
    ev = {"harness_seconds": round(time.time() - t0, 1)}
    gates = {k: summ["gates"][k] for k in ("G1", "G2", "G3", "G4", "G5")}
    ev["G1_dev_permissive"] = L.g1_light(ad, "dev_permissive")
    fam = L.ALGOS[c["algo"]]["family"]
    sw = None
    if gates["G2"].get("superword_tokens_untested"):
        if fam == "Unigram":
            sw = L.g2_superword(ad)
        else:
            sw = {"status": "superword tokens in a BPE vocabulary: not testable in isolation (none expected in Stage 1)"}
    ev["G2_superword"] = sw
    g2_pass = bool(gates["G2"]["pass"]) and (sw is None or sw.get("failures", 1) == 0)
    g5 = gates["G5"]["pass"]
    ev["gate_pass"] = {"G1": bool(gates["G1"]["pass"]) and ev["G1_dev_permissive"]["pass"], "G2": g2_pass,
                       "G3": bool(gates["G3"]["pass"]), "G4": bool(gates["G4"]["pass"]),
                       "G5": (None if g5 is None else bool(g5))}
    ev["gate_pass"]["all"] = all(v for v in ev["gate_pass"].values() if v is not None)
    return summ, ev


def process(c):
    t_start = time.time()
    L.log("=== %s" % c["id"])
    meta = train_config(c)
    d = L.tok_dir(c)
    ad = L.make_adapter(c)
    g4_path = os.path.join(d, "g4", meta["native_file"])
    summ, ev = evaluate(c, ad, g4_path, L.res_dir(c), meta)
    rec = {"id": c["id"], "config": c, "algorithm": L.ALGOS[c["algo"]], "created_utc": L.now_utc(),
           "code": code_hashes(), "tokenizer_dir": d, "files": meta["files"], "train_seconds": meta["train_seconds"],
           "g4_bytes_identical": meta["g4_bytes_identical"], "hf_export": meta.get("hf_export"),
           "harness_summary": os.path.join(L.res_dir(c), "summary.json"), "eval_initial": ev,
           "effective_vocab": c["vocab_total"], "g2_remedy": None}
    if c["algo"] == "A5" and not ev["gate_pass"]["G4"]:
        rec["g4_functional_diagnostic"] = L.hf_unigram_functional_g4(g4_path, os.path.join(d, "tokenizer.json"),
                                                                     L.dev_texts())
    # ---------------------------------------------------------------- pre-declared G2 remedy (PLAN 4.3)
    if not ev["gate_pass"]["G2"]:
        fam = L.ALGOS[c["algo"]]["family"]
        fail_ids = L.g2_all_failures(ad, test_superwords=(fam == "Unigram"))
        L.log(c["id"], "G2 failed for %d tokens -> remedy (%s family)" % (len(fail_ids), fam))
        rem = {"failing_ids": fail_ids, "n_failing": len(fail_ids), "family": fam}
        if fam == "Unigram" and fail_ids:
            pre = os.path.join(d, "pre_remedy")
            os.makedirs(pre, exist_ok=True)
            names = [f for f in meta["files"] if os.path.exists(os.path.join(d, f))]
            for f in names:
                shutil.copy2(os.path.join(d, f), os.path.join(pre, f))
            if c["algo"] in L.SP_ALGOS:
                deleted = L.remedy_sp(os.path.join(pre, "sp.model"), fail_ids, os.path.join(d, "sp.model"))
                L.remedy_sp(os.path.join(d, "g4", "sp.model"), fail_ids, os.path.join(d, "g4", "sp.remedied.model"))
                L.write_sp_vocab(os.path.join(d, "sp.model"), os.path.join(d, "sp.vocab"))
                with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(L.sp_to_hf_json(os.path.join(d, "sp.model")))
                g4_new = os.path.join(d, "g4", "sp.remedied.model")
                files = ("sp.model", "sp.vocab", "tokenizer.json")
            else:
                deleted = L.remedy_hf_unigram(os.path.join(pre, "tokenizer.json"), fail_ids,
                                              os.path.join(d, "tokenizer.json"))
                L.remedy_hf_unigram(os.path.join(d, "g4", "tokenizer.json"), fail_ids,
                                    os.path.join(d, "g4", "tokenizer.remedied.json"))
                g4_new = os.path.join(d, "g4", "tokenizer.remedied.json")
                files = ("tokenizer.json",)
            rem["deleted"] = {str(k): v for k, v in deleted.items()}
            rem["pre_remedy_files"] = {f: L.sha256_file(os.path.join(pre, f)) for f in names}
            old_res = L.res_dir(c, "__pre_remedy")
            if os.path.exists(old_res):
                shutil.rmtree(old_res)
            shutil.move(L.res_dir(c), old_res)
            ad_pre = L.make_adapter(c, "pre_remedy")
            ad = L.make_adapter(c)
            rem["reencode_check"] = L.reencode_check(ad_pre, ad, L.dev_texts())
            if c["algo"] in L.SP_ALGOS:
                rem["hf_export_after"] = L.sp_hf_equivalence(os.path.join(d, "sp.model"),
                                                              os.path.join(d, "tokenizer.json"), L.dev_texts())
            summ, ev2 = evaluate(c, ad, g4_new, L.res_dir(c), meta)
            rem["eval_after"] = ev2
            rem["applied"] = True
            rem["effective_vocab"] = c["vocab_total"] - len(fail_ids)
            rem["files_after"] = {f: L.sha256_file(os.path.join(d, f)) for f in files}
            rec["files"] = dict(rec["files"], **rem["files_after"])
            rec["effective_vocab"] = rem["effective_vocab"]
            rec["pre_remedy_summary"] = os.path.join(old_res, "summary.json")
            # the remedy must not change any encoding (Unigram argument, PLAN 4.3)
            rem["remedy_valid"] = bool(rem["reencode_check"]["identical"])
            meta["g2_remedy"] = {"n_deleted": len(fail_ids), "files_after": rem["files_after"],
                                 "pre_remedy_dir": pre, "effective_vocab": rem["effective_vocab"]}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        else:
            rem["applied"] = False
            rem["status"] = ("BPE-family G2 failure: PLAN 4.3 says delete + dev re-encoding check, retrain if any "
                             "encoding changes; not automated here, flagged for attention")
        rec["g2_remedy"] = rem
    ev_final = rec["g2_remedy"]["eval_after"] if rec["g2_remedy"] and rec["g2_remedy"].get("applied") else ev
    rec["gate_pass_final"] = ev_final["gate_pass"]
    if rec["g2_remedy"] and rec["g2_remedy"].get("applied") and not rec["g2_remedy"]["remedy_valid"]:
        rec["gate_pass_final"] = dict(rec["gate_pass_final"], G2=False, all=False)
    # ---------------------------------------------------------------- R1 on the tokenizer's own training mix
    if c["data"] != "D1":
        cnt, _, info = H.train_counts(ad, L.train_view(c["data"]))
        rec["R1_own_training_mix"] = H.prop_r1(ad, cnt, info)
    rec["seconds_total"] = round(time.time() - t_start, 1)
    rec["memory"] = _mem_mb()
    rec["complete"] = True
    json.dump(H.clean(rec), open(stage1_path(c), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    o = summ["metrics"]["overall"]
    L.log("done %s: %.4f bytes/token, gates %s, %.0fs" % (c["id"], o["bytes_per_token"], rec["gate_pass_final"],
                                                          rec["seconds_total"]))
    return rec


# ------------------------------------------------------------------------------------------ queue
def claim(c):
    os.makedirs(LOCKS, exist_ok=True)
    p = os.path.join(LOCKS, c["id"] + ".lock")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # a 'RESERVED' marker keeps configs away from workers still running older code (they cannot parse it);
        # this code version may take it over
        try:
            with open(p, encoding="utf-8") as f:
                reserved = f.read().startswith("RESERVED")
            if not reserved:
                return None
            os.replace(p, p + ".taken.%d" % os.getpid())
            os.remove(p + ".taken.%d" % os.getpid())
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return None
    os.write(fd, ("%d %s" % (os.getpid(), L.now_utc())).encode())
    os.close(fd)
    return p


def worker(k):
    L.verify_frozen()
    L.log("worker %d pid %d threads %s" % (k, os.getpid(), os.environ.get("RAYON_NUM_THREADS")))
    n = 0
    while True:
        todo = [c for c in queue() if not is_complete(c)]
        got = None
        for c in todo:
            lk = claim(c)
            if lk:
                got = (c, lk)
                break
        if got is None:
            L.log("worker %d: nothing left to claim (%d processed)" % (k, n))
            return
        c, lk = got
        try:
            process(c)
            n += 1
        except Exception:  # noqa: BLE001
            L.log("FAILED %s\n%s" % (c["id"], traceback.format_exc()))
            with open(os.path.join(L.HERE, "logs", "failed_%s.txt" % c["id"]), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            continue
        try:
            os.remove(lk)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clear-locks", action="store_true")
    a = ap.parse_args()
    if a.clear_locks:
        for f in os.listdir(LOCKS) if os.path.exists(LOCKS) else []:
            os.remove(os.path.join(LOCKS, f))
        print("locks cleared")
    if a.list:
        for c in queue():
            lk = os.path.exists(os.path.join(LOCKS, c["id"] + ".lock"))
            print("%-22s %-9s est %.2f" % (c["id"], "done" if is_complete(c) else ("claimed" if lk else "todo"),
                                          est_cost(c)))
    if a.only:
        L.verify_frozen()
        for i in a.only:
            process(L.config_by_id(i))
    if a.worker is not None:
        worker(a.worker)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
