# -*- coding: utf-8 -*-
"""Round 2: collect every result into round2_results.json and write round2_waves.json (lm/WAVES.json entry format,
consumed by colab/build_bundle.py --waves). Prints the markdown tables used in ROUND2.md.

    python r2_report.py

Reads only files under candidates/round2 plus the Stage 1/2 reference summaries (dev_strict) named in REFS.
Every wave entry is re-checked: the tokenizer file's sha256 equals the harness summary's tokenizer identity, the
encoder loads through colab/build_bundle.load_encoder(..., 'harness') (the exact encoder the bundle will use),
n_vocab equals the nominal size, <|endoftext|> = 0 and <|bos|> = 1, and a synthetic string round-trips (the
same check as lm/make_waves.py). Skipped builds are listed with the reason; no number is invented.
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_common as RC  # noqa: E402

sys.path.insert(0, os.path.join(RC.TOK, "colab"))
import build_bundle as BB  # noqa: E402  (stdlib-only at import time)

TRAIN_D1_BYTES = 45007515
DEV_BYTES = 1445513
SYN = ("".join(chr(c) for c in (0x06C1, 0x0646, 0x062F, 0x06A9, 0x0648)) + " "
       + "".join(chr(c) for c in (0x062F, 0x06D2)) + " " + "".join(chr(c) for c in (0x0646, 0x0627, 0x0644))
       + chr(0x06D4) + "\n" + "1234567" + chr(0x060C) + " " + "".join(chr(c) for c in (0x0679, 0x0628, 0x0631))
       + "\n\n" + "abc 42")                       # lm/make_waves.py SYN

STD_R = os.path.join(RC.TOK, "candidates", "standard", "results", "dev_strict")
REFS = {   # Stage 1/2 tokenizers the round-2 builds refine (dev_strict harness summaries; Stage 3 bpb from the task)
    "A1-P1r3-D2-16k": (os.path.join(STD_R, "A1-P1r3-D2-16k"), 1.4225, "Stage 3 baseline (rank 1)"),
    "A1-P1r3-D2-32k": (os.path.join(STD_R, "A1-P1r3-D2-32k"), 1.4004, "Stage 3 best"),
    "A6-SBPE-P1-D1-32k-t080": (os.path.join(RC.SBPE, "results", "dev_strict", "A6-SBPE-P1-D1-32k-t080"), 1.4053, ""),
    "A10-MinGram-P1-D1-16k": (os.path.join(RC.MING, "results", "dev_strict", "A10-mingram-P1-D1-16k"), 1.4187, ""),
    "A4-SPnat-D1-16k": (os.path.join(STD_R, "A4-SPnat-D1-16k"), 1.4213, ""),
    "A4-SPnat-D1-32k": (os.path.join(STD_R, "A4-SPnat-D1-32k"), None, "Stage 1 grid"),
    "A4-SPnat-D1-48k": (os.path.join(STD_R, "A4-SPnat-D1-48k"), None, "Stage 1 grid"),
    "A1-P1r3-D1-48k": (os.path.join(STD_R, "A1-P1r3-D1-48k"), None, "Stage 1 grid"),
    "A1-P1-D2-32k": (os.path.join(STD_R, "A1-P1-D2-32k"), None, "Stage 1 grid"),
}
REFINES = {"A6-32k": "A6-SBPE-P1-D1-32k-t080", "A10-32k": "A10-MinGram-P1-D1-16k",
           "A10-48k": "A10-MinGram-P1-D1-16k", "A4-32k": "A4-SPnat-D1-32k",
           "A4-48k": "A4-SPnat-D1-48k", "A1-48k": "A1-P1r3-D1-48k"}

# wave order = execution / cut order of the round (rank 100 + i): the 32k combinations first, 48k probes after
ORDER = ["A6-32k", "A10-32k", "A4-32k", "A1-48k", "A10-48k", "A4-48k"]


def J(p):
    return json.load(open(p, encoding="utf-8"))


def summ_metrics(sdir):
    s = J(os.path.join(sdir, "summary.json"))
    o = s["metrics"]["overall"]
    out = {"bytes_per_token": o["bytes_per_token"], "tokens": o["tokens"], "docs": o["docs"], "bytes": o["bytes"],
           "chars_per_token": o["chars_per_token"], "fertility": o["fertility"], "strr": o["strr"],
           "continued_word_rate": o["continued_word_rate"], "bytes_per_token_lines": o["bytes_per_token_lines"],
           "renyi_eff_a2.5": o.get("renyi_eff_a2.5"), "vocab_used": o.get("vocab_used"),
           "vocab_utilisation": o.get("vocab_utilisation"), "g1_fail_docs": o["g1_fail_docs"],
           "unk_tokens": o["unk_tokens"],
           "by_source_bytes_per_token": {k: v["bytes_per_token"] for k, v in s["metrics"]["by_source"].items()},
           "robustness_rel_token_change": {k: v["rel_token_change"] for k, v in o["robustness"].items()},
           "robustness_seg_change_affected": {k: v["seg_change_rate_affected"] for k, v in o["robustness"].items()},
           "nsl_vs_A1-P1-D1-16k": (s.get("nsl") or {}).get("overall")}
    r1 = s["properties"].get("R1") or {}
    if "train" in r1:
        out["train_D1_tokens"] = r1["train"]["tokens"]
        out["train_D1_bytes_per_token"] = TRAIN_D1_BYTES / r1["train"]["tokens"]
        out["R1_train_D1"] = r1["all_learned"]
        out["R1_train_D1_multichar"] = r1["multichar_learned"]
    out["R2"] = {k: s["properties"]["R2"][k] for k in ("partial_utf8_tokens", "with_train_freq_0")}
    out["gates_harness"] = {k: s["gates"][k].get("pass") for k in ("G1", "G2", "G3", "G4", "G5")}
    out["tokenizer_identity"] = s["tokenizer"]
    out["harness_code"] = s["harness"]["code"]
    out["morph_silver_high_morphscore"] = ((s.get("morphology") or {}).get("silver_high") or {}).get("morphscore")
    return out, s


def collect():
    rows = {}
    std = os.path.join(RC.R2, "standard")
    for key in ("A1-48k", "A4-32k", "A4-48k"):
        cid = RC.IDS[key]
        sdir = os.path.join(std, "results", "dev_strict", cid)
        if not os.path.exists(os.path.join(sdir, "stage1.json")):
            continue
        st = J(os.path.join(sdir, "stage1.json"))
        meta = J(os.path.join(std, "tok", cid, "meta.json"))
        m, s = summ_metrics(sdir)
        sp = key.startswith("A4")
        path = os.path.join(std, "tok", cid, "sp.model" if sp else "tokenizer.json")
        r = {"id": cid, "key": key, "algo": st["config"]["algo"], "tokenizer_path": path,
             "encoder_kind": "sentencepiece" if sp else "hf", "vocab_size": st["config"]["vocab_total"],
             "effective_vocab": st["effective_vocab"], "metrics": m, "summary_dir": sdir,
             "gates_final": st["gate_pass_final"], "g4_bytes_identical": st["g4_bytes_identical"],
             "G1_dev_permissive": st["eval_initial"]["G1_dev_permissive"]["pass"],
             "R1_own_training_mix_train_D2": st.get("R1_own_training_mix", {}).get("all_learned"),
             "train_seconds": st["train_seconds"], "files_sha256": st["files"],
             "train_data": meta["train_data"]}
        if sp:
            rem = st.get("g2_remedy") or {}
            r["g2_remedy"] = {"n_deleted": rem.get("n_failing"), "deleted": list((rem.get("deleted") or {}).values()),
                              "reencode_identical": (rem.get("reencode_check") or {}).get("identical"),
                              "remedy_valid": rem.get("remedy_valid")} if rem else None
            hx = (rem.get("hf_export_after") if rem else None) or meta["hf_export"]
            r["hf_export"] = {v: {"ids_differ_docs": hx[v]["ids_differ_docs"],
                                  "token_count_differs_docs": hx[v]["token_count_differs_docs"],
                                  "hf_decode_differs_docs": hx[v]["hf_decode_differs_docs"]}
                              for v in ("dev_strict", "dev_permissive")}
        rows[key] = r
    ming = os.path.join(RC.R2, "mingram")
    for key in ("A10-32k", "A10-48k"):
        cid = RC.IDS[key]
        sdir = os.path.join(ming, "results", "dev_strict", cid)
        if not os.path.exists(os.path.join(sdir, "r2_checks.json")):
            continue
        ch = J(os.path.join(sdir, "r2_checks.json"))
        m, s = summ_metrics(sdir)
        run = os.path.dirname(ch["tokenizer_json"])
        eq = J(os.path.join(run, "equivalence.json"))
        rows[key] = {"id": cid, "key": key, "algo": "A10", "tokenizer_path": ch["tokenizer_json"], "encoder_kind": "hf",
                     "vocab_size": RC.SIZES[key.split("-")[1]], "effective_vocab": RC.SIZES[key.split("-")[1]],
                     "metrics": m, "summary_dir": sdir, "gates_final": dict(ch["gates"], G1_dev_permissive=(
                         ch["G1_dev_permissive"]["fail"] == 0)),
                     "g4_all_outputs_identical": ch["G4_all_outputs_identical"],
                     "equivalence": {x["set"]: {k: x.get(k) for k in ("docs", "docs_identical", "tokens_hf", "tokens_ref",
                                                                     "g1_roundtrip_hf", "g1_roundtrip_ref",
                                                                     "paper_tie_rule_docs_differing",
                                                                     "paper_tie_rule_docs_with_token_count_difference",
                                                                     "min_token_check_failures")}
                                     for x in eq["sets"]},
                     "equivalence_verdict": eq["verdict"],
                     "R1_own_training_mix_train_D2": ch["R1_own_training_mix_train_D2"]["all_learned"],
                     "R1_own_training_mix_train_D2_multichar": ch["R1_own_training_mix_train_D2"]["multichar_learned"],
                     "train_log": ch["train_log"],
                     "files_sha256": {k: RC.sha256_file(os.path.join(run, k)) for k in
                                      ("tokenizer.json", "mingram_model.json", "seed_bpe.json", "vocab.tsv")}}
    sb = os.path.join(RC.R2, "superbpe")
    cid = RC.IDS["A6-32k"]
    sdir = os.path.join(sb, "results", "dev_strict", cid)
    if os.path.exists(os.path.join(sdir, "r2_checks.json")):
        ch = J(os.path.join(sdir, "r2_checks.json"))
        m, s = summ_metrics(sdir)
        rows["A6-32k"] = {"id": cid, "key": "A6-32k", "algo": "A6", "tokenizer_path": ch["tokenizer_json"],
                          "encoder_kind": "hf", "vocab_size": 32768, "effective_vocab": 32768, "metrics": m,
                          "summary_dir": sdir,
                          "gates_final": dict(ch["gates_harness"], G2=ch["G2_final"]), "G2_multiword": ch["G2_multiword"],
                          "G4_twin": ch["G4_twin"], "verify": ch["verify"], "build": ch["build"],
                          "R1_own_training_mix_train_D2": ch["R1_own_training_mix_train_D2"]["all_learned"],
                          "files_sha256": {"tokenizer.json": RC.sha256_file(ch["tokenizer_json"]),
                                           "stage2_pua.json": RC.sha256_file(os.path.join(
                                               os.path.dirname(ch["tokenizer_json"]), "stage2_pua.json"))}}
    return rows


def encoder_check(r):
    enc = BB.load_encoder(r["tokenizer_path"], r["encoder_kind"], "harness", cid=r["id"])
    ids = enc.encode(SYN)
    rt = enc.decode(ids) == SYN
    eot, bos = enc.token_to_id("<|endoftext|>"), enc.token_to_id("<|bos|>")
    ok = rt and max(ids) < enc.n_vocab and eot == 0 and bos == 1 and enc.n_vocab == r["vocab_size"]
    return {"loader": enc.source, "n_vocab": enc.n_vocab, "eot_id": eot, "bos_id": bos, "synthetic_round_trip": rt}, ok


def gates_str(g):
    return {k: ("pass" if g.get(k) else ("n/a" if g.get(k) is None else "FAIL")) for k in ("G1", "G2", "G3", "G4", "G5")}


def main():
    rows = collect()
    refs = {}
    for rid, (sdir, bpb, note) in REFS.items():
        if os.path.exists(os.path.join(sdir, "summary.json")):
            m, _ = summ_metrics(sdir)
            refs[rid] = {"summary_dir": sdir, "stage3_mean_dev_bpb": bpb, "note": note,
                         "bytes_per_token": m["bytes_per_token"], "tokens": m["tokens"],
                         "by_source_bytes_per_token": m["by_source_bytes_per_token"],
                         "train_D1_bytes_per_token": m.get("train_D1_bytes_per_token"),
                         "R1_train_D1": m.get("R1_train_D1"), "fertility": m["fertility"], "strr": m["strr"]}
    waves, skipped = [], []
    base = refs["A1-P1r3-D2-16k"]["bytes_per_token"]
    for i, key in enumerate(ORDER, 1):
        cid = RC.IDS[key]
        if key not in rows:
            skipped.append({"id": cid, "why": "not built or not evaluated (see ROUND2.md)"})
            continue
        r = rows[key]
        m = r["metrics"]
        sha = RC.sha256_file(r["tokenizer_path"])
        assert sha == m["tokenizer_identity"]["sha256"], (cid, "evaluated file differs from the shipped file")
        assert m["docs"] == 836 and m["bytes"] == DEV_BYTES and m["g1_fail_docs"] == 0, cid
        g = r["gates_final"]
        assert all(g.get(k) is not False for k in ("G1", "G2", "G3", "G4", "G5")), (cid, g)
        ec, ok = encoder_check(r)
        assert ok, (cid, ec)
        r["encoder_check"] = ec
        ref = REFINES[key]
        w = {"id": cid, "rank": 100 + i, "rank_label": "R2-%d" % i, "conditional": False,
             "tokenizer_path": r["tokenizer_path"], "encoder_kind": r["encoder_kind"]}
        if r["encoder_kind"] == "sentencepiece":
            w["newline_wrapper"] = True
        w.update({"vocab_size": r["vocab_size"], "tokenizer_sha256": sha,
                  "dev_strict_bytes_per_token": round(m["bytes_per_token"], 4), "dev_strict_tokens": m["tokens"],
                  "train_D1_bytes_per_token": round(m["train_D1_bytes_per_token"], 4),
                  "plan_config": None, "harness_gates_dev": gates_str(g)})
        if key == "A6-32k":
            gm = r["G2_multiword"]
            w["g2_multiword"] = {"multiword_tokens": gm["multiword_tokens"],
                                 "isolated_self_tokenization_failures": gm["isolated_failures"],
                                 "literal_shortest_train_chunk_clause_failures": gm["chunk_raw_failures_literal_PLAN_clause"],
                                 "literal_failures_that_pass_isolated": gm["chunk_raw_failures_that_pass_isolated"],
                                 "unreachable_under_reachability_reading": gm["unreachable"],
                                 "verdict_pass": gm["pass"]}
            vc = r["verify"]["checks"]
            w["stage2_checks"] = {"hf_vs_reference_mismatch_docs_dev_strict": vc["dev_strict"]["hf_vs_ref_mismatch_docs"],
                                  "hf_vs_reference_mismatch_docs_dev_permissive": vc["dev_permissive"]["hf_vs_ref_mismatch_docs"],
                                  "hf_vs_reference_mismatch_docs_train_D2_every_20": vc["train_D2_every_20"]["hf_vs_ref_mismatch_docs"],
                                  "g1_fail_docs_dev_strict": vc["dev_strict"]["g1_fail_docs"],
                                  "g1_fail_docs_dev_permissive": vc["dev_permissive"]["g1_fail_docs"],
                                  "g4_twin_tokenizer_identical": r["G4_twin"]["tokenizer_identical"]}
            w["plan_config"] = ("Round 2 (PLAN 2.1 'plus refinements'): A6 SuperBPE @32k, t/T = 26,214/32,768 = 0.8 (as "
                                "A6-SBPE-P1-D1-32k-t080), rebuilt with rank 1's recipe: pre-tokenizer P1r3 (stage 1 and the "
                                "digit rule of stage 2) and tokenizer training text D2.")
            w["notes"] = ("HF-native = reference encoder on 100% of dev_strict and dev_permissive; G2 multi-word part "
                          "under the reachability reading (as Stage 2). Stage 1 is a strict prefix of A1-P1r3-D2-32k.")
        elif key.startswith("A10"):
            eq = r["equivalence"]
            w["stage2_checks"] = {"hf_vs_reference_identical_docs_dev_strict": "%d/%d" % (
                                      eq["dev_strict"]["docs_identical"], eq["dev_strict"]["docs"]),
                                  "hf_vs_reference_identical_docs_dev_permissive": "%d/%d" % (
                                      eq["dev_permissive"]["docs_identical"], eq["dev_permissive"]["docs"]),
                                  "all_sets_identical": r["equivalence_verdict"]["all_sets_identical"],
                                  "paper_tie_rule_docs_differing_dev_strict": eq["dev_strict"]["paper_tie_rule_docs_differing"],
                                  "paper_tie_rule_token_count_differences": eq["dev_strict"][
                                      "paper_tie_rule_docs_with_token_count_difference"],
                                  "g4_all_outputs_identical": all(r["g4_all_outputs_identical"].values())}
            w["plan_config"] = ("Round 2: A10 MinGram @%s ('MinGram as reimplemented from the paper', Land 2026; route b) "
                                "with pre-tokenizer P1r3 and training text D2 (f = 1.15, N_em = 2, one flat prune, as A10-16k)."
                                % key.split("-")[1])
            w["notes"] = ("Stock HF Unigram = the reimplementation's reference Viterbi on 100% of dev (and perturbed/"
                          "synthetic sets). Rare-token tail on D2 at this size: see ROUND2.md (R1).")
        elif key.startswith("A4"):
            rem = r.get("g2_remedy") or {}
            hx = r["hf_export"]
            w["plan_config"] = ("Round 2: A4 SentencePiece Unigram @%s, native pre-tokenization (split_digits) as in the "
                                "Stage 1 sweep, trained on D2 (train_D2.lines.txt)." % key.split("-")[1])
            w["notes"] = ("Native sp.model with the PLAN 1.1 newline wrapper is the candidate's identity%s. Its HF "
                          "tokenizer.json export differs from native ids on %d dev_strict / %d dev_permissive documents "
                          "(equal-score ties, identical token counts), so A4 does not count as HF-native exact."
                          % ((" after the pre-declared G2 remedy (%d pieces -> CONTROL placeholders; effective "
                              "vocabulary %d; dev re-encoding identical)" % (rem["n_deleted"], r["effective_vocab"]))
                             if rem.get("n_deleted") else "", hx["dev_strict"]["ids_differ_docs"],
                             hx["dev_permissive"]["ids_differ_docs"]))
        else:
            w["plan_config"] = ("Round 2: A1 byte-level BPE @48k with rank 1's recipe (P1r3, D2): extends the vocabulary "
                                "curve past the Stage 3 best, A1-P1r3-D2-32k.")
            w["notes"] = "Same code as the Stage 1 hand-over tokenizers (candidates/standard); all gates pass."
        w["encoder_check"] = ec
        w["refines"] = ref
        w["dev_bytes_per_token_vs_baseline_pct"] = round(100 * (m["bytes_per_token"] / base - 1), 3)
        w["dev_strict_summary"] = os.path.join(r["summary_dir"], "summary.json")
        waves.append(w)
    doc = {"what": "Round 2 (exploratory refinement; PLAN 2.1 'LM stage: {8k,16k,32k} plus refinements') LM waves: "
                   "candidates that combine the Stage 3 screening findings (32k > 16k, D2 > D1, MinGram/Unigram > BPE at "
                   "equal data). Read by colab/build_bundle.py --waves: every entry with tokenizer_path + encoder_kind is a "
                   "bundle candidate, in file order. Ranks are 100 + i (outside the pre-registered PLAN 2.2 ranking).",
           "written_by": "candidates/round2/r2_report.py (deterministic; no timestamps)",
           "baseline_note": ("No top-level 'baseline' key: the Stage 3 baseline A1-P1r3-D2-16k is not a round-2 wave, and "
                             "build_bundle.py stops if a named baseline is not among the candidates. To bundle round 2 with "
                             "its references, merge these entries with lm/WAVES.json's A1-P1r3-D2-16k and A1-P1r3-D2-32k "
                             "entries (or pass --baseline with an id present in the list)."),
           "recipe": {"pre_tokenizer": "P1r3 (A4: SentencePiece native, split_digits=true)", "training_text": "D2 (train_D2)",
                      "special_block": "64 tokens at ids 0..63", "normalizer": None,
                      "vocab_total": "exactly the nominal size (32,768 / 49,152)",
                      "lm_training_text_in_bundle": "train_D1 (colab/build_bundle.py default)"},
           "frozen": {"split_manifest_sha256": RC.MANIFEST_SHA256,
                      "test_split": "never opened; all numbers are dev_strict (validation)"},
           "execution_order": [w["id"] for w in waves],
           "waves": waves, "skipped": skipped}
    RC.dump(doc, os.path.join(RC.R2, "round2_waves.json"))
    RC.dump({"what": "Round 2 results (dev_strict, 836 documents, 1,445,513 bytes; manifest 76582d3a...)",
             "candidates": rows, "references": refs, "skipped": skipped}, os.path.join(RC.R2, "round2_results.json"))
    # ---------------------------------------------------------------- markdown tables
    print("| rank | id | V (eff.) | dev bytes/token | vs baseline 16k | vs refined tokenizer | NSL | train_D1 bytes/token | book / newspaper / web | fertility | STRR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for w in waves:
        r = rows[[k for k in rows if rows[k]["id"] == w["id"]][0]]
        m = r["metrics"]
        ref = refs.get(w["refines"])
        rel = 100 * (m["bytes_per_token"] / ref["bytes_per_token"] - 1) if ref else None
        bs = m["by_source_bytes_per_token"]
        print("| %d | %s | %s | %.4f | %+.2f%% | %s %+.2f%% | %.4f | %.4f | %.3f / %.3f / %.3f | %.3f | %.3f |" % (
            w["rank"], w["id"], ("{:,}".format(r["vocab_size"]) + (" ({:,})".format(r["effective_vocab"])
                                                                   if r["effective_vocab"] != r["vocab_size"] else "")),
            m["bytes_per_token"], w["dev_bytes_per_token_vs_baseline_pct"], w["refines"], rel,
            m["nsl_vs_A1-P1-D1-16k"], m["train_D1_bytes_per_token"], bs["book"], bs["newspaper"], bs["web"],
            m["fertility"], m["strr"]))
    print()
    print("| reference | dev bytes/token | Stage 3 mean dev bpb | train_D1 bytes/token |")
    print("|---|---|---|---|")
    for rid, x in refs.items():
        print("| %s | %.4f | %s | %s |" % (rid, x["bytes_per_token"], x["stage3_mean_dev_bpb"] or "-",
                                         "%.4f" % x["train_D1_bytes_per_token"] if x["train_D1_bytes_per_token"] else "-"))
    print()
    print("| id | G1 | G2 | G3 | G4 | G5 | R1 train_D1: =0 / <20 (%) / <100 / median | R1 train_D2 (own): =0 / <20 (%) / median | R2 partial-UTF-8 (freq 0) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for w in waves:
        r = rows[[k for k in rows if rows[k]["id"] == w["id"]][0]]
        m = r["metrics"]
        a, b = m["R1_train_D1"], r["R1_own_training_mix_train_D2"]
        g = w["harness_gates_dev"]
        print("| %s | %s | %s | %s | %s | %s | %d / %s (%.2f) / %s / %g | %d / %s (%.2f) / %g | %d (%s) |" % (
            w["id"], g["G1"], g["G2"], g["G3"], g["G4"], g["G5"], a["train_freq_eq0"], "{:,}".format(a["train_freq_lt20"]),
            a["pct_lt20"], "{:,}".format(a["train_freq_lt100"]), a["median_train_freq"], b["train_freq_eq0"],
            "{:,}".format(b["train_freq_lt20"]), b["pct_lt20"], b["median_train_freq"], m["R2"]["partial_utf8_tokens"],
            m["R2"]["with_train_freq_0"]))
    print()
    for w in waves:
        print(w["id"], w["tokenizer_sha256"], w["encoder_check"])
    print("skipped:", skipped)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
