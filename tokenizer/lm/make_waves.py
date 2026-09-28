# -*- coding: utf-8 -*-
"""Write F:\\Hindko\\_tokenizer\\lm\\WAVES.json: the Stage 3 LM-screening wave list.

Applies PLAN.md 2.2 (ranked list = cut order) and PLAN 3 'After Stage 1' / 'After Stage 2' to the
Stage 1 and Stage 2 results in candidates/. For every wave it checks the tokenizer file's sha256 against
the value the candidate owner recorded, loads the encoder exactly as colab/build_bundle.py will
(build_bundle.load_encoder(..., source="harness") -> eval/harness.py build_adapter), and checks
vocabulary size, EOT/BOS ids and a round trip of a synthetic string built here.

    set PYTHONIOENCODING=utf-8
    set PYTHONDONTWRITEBYTECODE=1
    python F:\\Hindko\\_tokenizer\\lm\\make_waves.py

No split is encoded: dev numbers are read from the existing harness summaries (dev_strict). The test
split is never opened. The output has no timestamps, so a re-run gives a byte-identical file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

TOK = r"F:\Hindko\_tokenizer"
CAND = os.path.join(TOK, "candidates")
STD = os.path.join(CAND, "standard")
LM = os.path.join(TOK, "lm")
OUT = os.path.join(LM, "WAVES.json")
EXPECTED_MANIFEST = "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return round(100.0 * x, 3)


# ------------------------------------------------------------------ frozen inputs
frozen = jload(os.path.join(TOK, "FROZEN.json"))
man_sha = sha256_file(os.path.join(TOK, "splits", "split_manifest.jsonl"))
assert man_sha == EXPECTED_MANIFEST, "split manifest hash mismatch: %s" % man_sha
plan_sha = sha256_file(os.path.join(TOK, "research", "PLAN.md"))
norm_sha = sha256_file(frozen["normalize"]["path"])
assert norm_sha == frozen["normalize"]["sha256"], "normalize.py changed since freeze"

# ------------------------------------------------------------------ Stage 1 selection (read, not recomputed)
sweep = jload(os.path.join(STD, "sweep_results.json"))
sel = sweep["selection"]
rows = {r["id"]: r for r in sweep["rows"]}
hrows = {r["id"]: r for r in sweep["handover_rows"]}


def std_summary(name):
    s = jload(os.path.join(STD, "results", "dev_strict", name, "summary.json"))
    return s


def bpt(summary_path):
    s = jload(summary_path)
    o = s["metrics"]["overall"]
    g = {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in s["gates"].items()}
    return o["bytes_per_token"], o["tokens"], o["bytes"], o["docs"], g


R1_BPT = std_summary("A1-P1r3-D2-16k")["metrics"]["overall"]["bytes_per_token"]
A1P1D1_BPT = rows["A1-P1-D1-16k"]["metrics"]["bytes_per_token"]
A1P1R3D1_BPT = rows["A1-P1r3-D1-16k"]["metrics"]["bytes_per_token"]
A3_BPT = rows["A3-SPnat-D1-16k"]["metrics"]["bytes_per_token"]
A2_BPT = rows["A2-P1-D1-16k"]["metrics"]["bytes_per_token"]

q = sel["A2_A3_lm_qualification"]["per_size"]
a3_rel = {s: q["A3"][s]["rel_vs_A1"] for s in q["A3"]}
a2_rel = {s: q["A2"][s]["rel_vs_A1"] for s in q["A2"]}

SB = os.path.join(CAND, "superbpe")


def g2mw(build):
    g = jload(os.path.join(SB, build, "g2_multiword.json"))
    return {"multiword_tokens": g["multiword_tokens"], "isolated_self_tokenization_failures": g["isolated_failures"],
            "literal_shortest_train_chunk_clause_failures": g["chunk_raw_failures_literal_PLAN_clause"],
            "literal_failures_that_pass_isolated": g["chunk_raw_failures_that_pass_isolated"],
            "unreachable_under_reachability_reading": g["unreachable"], "verdict_pass": g["pass"]}


def sb_verify(build):
    v = jload(os.path.join(SB, build, "verify.json"))["checks"]
    return {"hf_vs_reference_mismatch_docs_dev_strict": v["dev_strict"]["hf_vs_ref_mismatch_docs"],
            "hf_vs_reference_mismatch_docs_dev_permissive": v["dev_permissive"]["hf_vs_ref_mismatch_docs"],
            "g1_fail_docs_dev_strict": v["dev_strict"]["g1_fail_docs"],
            "g1_fail_docs_dev_permissive": v["dev_permissive"]["g1_fail_docs"]}


mg_eq = jload(os.path.join(CAND, "mingram", "runs", "mingram_P1_D1_16384", "equivalence.json"))
mg_sets = {s["set"]: s for s in mg_eq["sets"]}
picky_g4_full = os.path.join(LM, "g4_pickybpe_fullretrain", "models", "pickybpe_P1_D1_16k_tau0.9.fullretrain.json")
picky_g4_full_sha = sha256_file(picky_g4_full) if os.path.isfile(picky_g4_full) else None

A4_EXPORT = rows["A4-SPnat-D1-16k"]["hf_export"]
A3_EXPORT = rows["A3-SPnat-D1-16k"]["hf_export"]

# ------------------------------------------------------------------ the waves (file order = rank order)
W = []


def wave(**kw):
    W.append(kw)


wave(id="A1-P1r3-D2-16k", rank=1, rank_label="1", baseline=True, conditional=False,
     tokenizer_path=os.path.join(STD, "tok", "A1-P1r3-D2-16k", "tokenizer.json"), encoder_kind="hf",
     vocab_size=16384, sha256_expected=hrows["A1-P1r3-D2-16k"]["files_sha256"]["tokenizer.json"],
     summary=os.path.join(STD, "results", "dev_strict", "A1-P1r3-D2-16k", "summary.json"),
     plan_config="PLAN 2.2 rank 1, A1 byte-level BPE @16k ('standard recipe'), with the pre-tokenizer (P1r3) and "
                 "data mix (D2) fixed by PLAN 3 'After Stage 1'. Baseline of every comparison; LR-sweep candidate.",
     notes="Hand-over tokenizer trained by the Stage 1 owner after the rules (same code, all gates pass). "
           "Not in the pre-registered grid (the grid never crosses P1r3 with D2). D2 awaits user confirmation.")
wave(id="A6-SBPE-P1-D1-16k-t090", rank=2, rank_label="2", conditional=False,
     tokenizer_path=os.path.join(SB, "sbpe_16k_t090", "tokenizer.json"), encoder_kind="hf", vocab_size=16384,
     sha256_expected=jload(os.path.join(SB, "MANIFEST.json"))["candidates"][0]["tokenizer_sha256"],
     summary=os.path.join(SB, "results", "dev_strict", "A6-SBPE-P1-D1-16k-t090", "summary.json"),
     plan_config="PLAN 2.2 rank 2, A6 SuperBPE-P1 @16k, t/T = 0.9 (reimplemented from Liu et al. 2025; "
                 "punctuation-bounded stage 2).",
     g2_multiword=g2mw("sbpe_16k_t090"), stage2_checks=sb_verify("sbpe_16k_t090"),
     notes="Kept after Stage 2: gates pass (G2 multi-word part under the reachability reading, see deviations), "
           "round trip exact, HF-native = reference encoder on 100% of dev_strict and dev_permissive.")
wave(id="A4-SPnat-D1-16k", rank=3, rank_label="3", conditional=False, newline_wrapper=True,
     tokenizer_path=os.path.join(STD, "tok", "A4-SPnat-D1-16k", "sp.model"), encoder_kind="sentencepiece",
     vocab_size=16384,
     sha256_expected=jload(os.path.join(STD, "tok", "A4-SPnat-D1-16k", "meta.json"))["g2_remedy"]["files_after"]["sp.model"],
     summary=os.path.join(STD, "results", "dev_strict", "A4-SPnat-D1-16k", "summary.json"),
     plan_config="PLAN 2.2 rank 3, Unigram @16k = the better of A4-SPnat and A5-P1 on dev bytes/token among "
                 "gate-passing tokenizers (PLAN 3 'After Stage 1').",
     notes="Native sp.model after the pre-declared G2 remedy (1 piece replaced by a CONTROL placeholder; effective "
           "vocabulary 16,383), with the PLAN 1.1 newline wrapper; this is the candidate's identity. Its HF "
           "tokenizer.json export is NOT used: it differs from the native ids on %d dev_strict / %d dev_permissive "
           "documents (equal-score ties; identical token counts), and in it the placeholder is an added special "
           "token. For PLAN 7 tie-breaker (a) A4 therefore does not count as HF-native exact."
           % (A4_EXPORT["ids_differ_docs_dev_strict"], A4_EXPORT["ids_differ_docs_dev_permissive"]))
wave(id="A10-MinGram-P1-D1-16k", rank=4, rank_label="4", conditional=False,
     tokenizer_path=os.path.join(CAND, "mingram", "runs", "mingram_P1_D1_16384", "tokenizer.json"),
     encoder_kind="hf", vocab_size=16384, sha256_expected=mg_eq["tokenizer_sha256"],
     summary=os.path.join(CAND, "mingram", "results", "dev_strict", "A10-mingram-P1-D1-16k", "summary.json"),
     plan_config="PLAN 2.2 rank 4, A10 MinGram @16k ('MinGram as reimplemented from the paper', Land 2026; "
                 "route b).",
     stage2_checks={"hf_vs_reference_identical_docs_dev_strict": "%d/%d" % (mg_sets["dev_strict"]["docs_identical"],
                                                                             mg_sets["dev_strict"]["docs"]),
                    "hf_vs_reference_identical_docs_dev_permissive": "%d/%d" % (
                        mg_sets["dev_permissive"]["docs_identical"], mg_sets["dev_permissive"]["docs"]),
                    "paper_tie_rule_docs_differing_dev_strict": mg_sets["dev_strict"]["paper_tie_rule_docs_differing"],
                    "paper_tie_rule_token_count_differences": mg_sets["dev_strict"][
                        "paper_tie_rule_docs_with_token_count_difference"]},
     notes="Kept after Stage 2: gates G1-G5 pass, round trip exact, stock HF Unigram = the reimplementation's "
           "reference Viterbi on 100% of dev. The reference uses HF's left-to-right tie rule (a disclosed "
           "deviation from the paper's 'longest leading tokens' rule, which differs on 2/836 dev_strict documents "
           "with identical token counts).")
wave(id="A6-SBPE-P1-D1-16k-t080", rank=5, rank_label="5", conditional=False,
     tokenizer_path=os.path.join(SB, "sbpe_16k_t080", "tokenizer.json"), encoder_kind="hf", vocab_size=16384,
     sha256_expected=jload(os.path.join(SB, "MANIFEST.json"))["candidates"][1]["tokenizer_sha256"],
     summary=os.path.join(SB, "results", "dev_strict", "A6-SBPE-P1-D1-16k-t080", "summary.json"),
     plan_config="PLAN 2.2 rank 5, A6 SuperBPE @16k, t/T = 0.8 (brackets the transition point).",
     g2_multiword=g2mw("sbpe_16k_t080"), stage2_checks=sb_verify("sbpe_16k_t080"),
     notes="Kept after Stage 2 on the same grounds as rank 2.")
wave(id="A1-P1r3-D2-8k", rank=6, rank_label="6a", conditional=False,
     tokenizer_path=os.path.join(STD, "tok", "A1-P1r3-D2-8k", "tokenizer.json"), encoder_kind="hf",
     vocab_size=8192, sha256_expected=hrows["A1-P1r3-D2-8k"]["files_sha256"]["tokenizer.json"],
     summary=os.path.join(STD, "results", "dev_strict", "A1-P1r3-D2-8k", "summary.json"),
     plan_config="PLAN 2.2 rank 6 (first of two waves): A1 @8k with rank 1's recipe (P1r3, D2).",
     notes="Hand-over tokenizer (outside the grid), all gates pass. Vocabulary-size curve with rank 1 and 6b.")
wave(id="A1-P1r3-D2-32k", rank=6, rank_label="6b", conditional=False,
     tokenizer_path=os.path.join(STD, "tok", "A1-P1r3-D2-32k", "tokenizer.json"), encoder_kind="hf",
     vocab_size=32768, sha256_expected=hrows["A1-P1r3-D2-32k"]["files_sha256"]["tokenizer.json"],
     summary=os.path.join(STD, "results", "dev_strict", "A1-P1r3-D2-32k", "summary.json"),
     plan_config="PLAN 2.2 rank 6 (second of two waves): A1 @32k with rank 1's recipe (P1r3, D2).",
     notes="Hand-over tokenizer (outside the grid), all gates pass. The data mix is D2 because PLAN 3 fixes one "
           "mix (decided at 16k), although the same rule applied at 32k alone would give D1 (+0.489% < 0.5%).")
wave(id="A7-PickyBPE-P1-D1-16k-tau0.9", rank=7, rank_label="7", conditional=False,
     tokenizer_path=os.path.join(CAND, "pickybpe", "models", "pickybpe_P1_D1_16k_tau0.9.json"),
     encoder_kind="custom", custom="pickybpe_factory:load_16k",
     pythonpath_required=os.path.join(CAND, "pickybpe"), vocab_size=16384,
     sha256_expected=jload(os.path.join(CAND, "pickybpe", "MANIFEST.json"))["shipped"]["sha256"],
     summary=os.path.join(CAND, "pickybpe", "results", "dev_strict", "A7_pickybpe_P1_D1_16k_tau0.9", "summary.json"),
     plan_config="PLAN 2.2 rank 7, A7 PickyBPE @16k, tau = 0.9 (removes 1.57% > 1% of learned slots, so no "
                 "fallback to 0.8). Reimplemented from Chizhov et al. 2024.",
     stage2_checks={"g4_full_uncached_retrain_sha256": picky_g4_full_sha,
                    "g4_full_uncached_retrain_identical": picky_g4_full_sha == jload(
                        os.path.join(CAND, "pickybpe", "MANIFEST.json"))["shipped"]["sha256"]},
     notes="Kept after Stage 2: gates G1-G4 pass (G5 n/a, byte-level), round trip exact. PLAN 3 (iii) (HF "
           "equivalence) applies to A6/A10 only; A7 is a custom encoder by design (PLAN 2.1) and loses tie-breaker "
           "(a). build_bundle.py imports the factory by module name: run it with PYTHONPATH=%s. This run re-did G4 "
           "with an empty pretoken cache (fresh counting from train_D1): byte-identical model."
           % os.path.join(CAND, "pickybpe"))
wave(id="A6-SBPE-P1-D1-32k-t080", rank=8, rank_label="8", conditional=True,
     condition="Admitted to the PLAN 7 ranking only if the Stage 3 mean dev bpb (3 seeds, pre-registered LR) of rank 2 "
               "(A6-SBPE-P1-D1-16k-t090) or rank 5 (A6-SBPE-P1-D1-16k-t080) is lower than that of rank 1 "
               "(A1-P1r3-D2-16k).",
     t_over_T_rule="PLAN 2.2 rank 8 takes t/T from the better of ranks 2 and 5 in Stage 3 mean dev bpb. This file "
                   "(t/T = 0.8) is the valid rank-8 tokenizer only if mean bpb(rank 5) <= mean bpb(rank 2); an exact "
                   "tie goes to 0.8 by PLAN 7 tie-breaker (b). If rank 2 is better, rank 8 must be a 32k t/T = 0.9 "
                   "build, which does not exist yet: in candidates/superbpe run 'python build_stage1.py 29491 "
                   "stage1/a1_p1_29491.json', then 'python build_stage2.py --stage1 stage1/a1_p1_29491.json --T 32768 "
                   "--out sbpe_32k_t090 --py-check', then verify.py, gates_sbpe.py and the harness, and replace this "
                   "entry's tokenizer_path before the wave runs.",
     tokenizer_path=os.path.join(SB, "sbpe_32k_t080", "tokenizer.json"), encoder_kind="hf", vocab_size=32768,
     sha256_expected=jload(os.path.join(SB, "MANIFEST.json"))["candidates"][2]["tokenizer_sha256"],
     summary=os.path.join(SB, "results", "dev_strict", "A6-SBPE-P1-D1-32k-t080", "summary.json"),
     plan_config="PLAN 2.2 rank 8, A6 SuperBPE @32k: conditional extra wave.",
     g2_multiword=g2mw("sbpe_32k_t080"), stage2_checks=sb_verify("sbpe_32k_t080"),
     notes="colab/run_all.py has no notion of 'conditional' and will train every bundle candidate. Training it "
           "anyway is harmless (the condition reads only ranks 1, 2 and 5), but its result must be excluded from "
           "PLAN 7 and from the Holm family unless the condition holds and t/T matches the rule.")
wave(id="A3-SPnat-D1-16k", rank=9, rank_label="9", conditional=False, newline_wrapper=True,
     tokenizer_path=os.path.join(STD, "tok", "A3-SPnat-D1-16k", "sp.model"), encoder_kind="sentencepiece",
     vocab_size=16384, sha256_expected=rows["A3-SPnat-D1-16k"]["files_sha256"]["sp.model"],
     summary=os.path.join(STD, "results", "dev_strict", "A3-SPnat-D1-16k", "summary.json"),
     plan_config="PLAN 2.2 rank 9: A3 SentencePiece BPE @16k gets an LM wave because it is intrinsically >= 1% "
                 "better than A1 on dev bytes/token (the rank-9 condition, resolved by Stage 1). A2 does not qualify.",
     notes="Native sp.model with the PLAN 1.1 newline wrapper. Its HF tokenizer.json export differs from the native "
           "ids on %d dev_strict and %d dev_permissive documents (exact), so it can claim tie-breaker (a). Last in "
           "the cut order (after conditional rank 8)." % (A3_EXPORT["ids_differ_docs_dev_strict"],
                                                        A3_EXPORT["ids_differ_docs_dev_permissive"]))

# ------------------------------------------------------------------ checks: hash, encoder load, synthetic round trip
sys.path.insert(0, os.path.join(TOK, "colab"))
sys.path.insert(0, os.path.join(CAND, "pickybpe"))
sys.dont_write_bytecode = True
import build_bundle as BB  # noqa: E402  (stdlib-only at import time)

SYN = ("".join(chr(c) for c in (0x06C1, 0x0646, 0x062F, 0x06A9, 0x0648)) + " "
       + "".join(chr(c) for c in (0x062F, 0x06D2)) + " " + "".join(chr(c) for c in (0x0646, 0x0627, 0x0644))
       + chr(0x06D4) + "\n" + "1234567" + chr(0x060C) + " " + "".join(chr(c) for c in (0x0679, 0x0628, 0x0631))
       + "\n\n" + "abc 42")

check_rows = []
for w in W:
    p = w["tokenizer_path"]
    assert os.path.isfile(p), p
    got = sha256_file(p)
    assert got == w["sha256_expected"], "%s: sha256 %s != recorded %s" % (w["id"], got, w["sha256_expected"])
    enc = BB.load_encoder(p, w["encoder_kind"], "harness", cid=w["id"], custom_spec=w.get("custom"),
                          no_newline_wrapper=w.get("newline_wrapper") is False)
    ids = enc.encode(SYN)
    rt = enc.decode(ids) == SYN
    eot, bos = enc.token_to_id("<|endoftext|>"), enc.token_to_id("<|bos|>")
    assert enc.n_vocab == w["vocab_size"], "%s: n_vocab %s != %s" % (w["id"], enc.n_vocab, w["vocab_size"])
    assert rt and max(ids) < enc.n_vocab and eot == 0 and bos == 1, (w["id"], rt, eot, bos)
    b, ntok, nbytes, ndocs, gates = bpt(w["summary"])
    assert ndocs == 836 and nbytes == 1445513, (w["id"], ndocs, nbytes)
    assert all(v is not False for v in gates.values()), (w["id"], gates)
    w["tokenizer_sha256"] = got
    w["dev_strict_bytes_per_token"] = round(b, 4)
    w["dev_strict_tokens"] = ntok
    w["harness_gates_dev"] = {k: ("pass" if v else "n/a") for k, v in gates.items()}
    w["encoder_check"] = {"loader": enc.source, "n_vocab": enc.n_vocab, "eot_id": eot, "bos_id": bos,
                          "synthetic_round_trip": rt}
    check_rows.append((w["id"], w["encoder_kind"], enc.n_vocab, round(b, 4), got[:12]))

# ------------------------------------------------------------------ assemble WAVES.json
ORDER = ["id", "rank", "rank_label", "baseline", "conditional", "condition", "t_over_T_rule", "tokenizer_path",
         "encoder_kind", "custom", "pythonpath_required", "newline_wrapper", "vocab_size", "tokenizer_sha256",
         "dev_strict_bytes_per_token", "dev_strict_tokens", "plan_config", "harness_gates_dev", "g2_multiword",
         "stage2_checks", "encoder_check", "notes"]
waves_out = []
for w in W:
    d = {k: w[k] for k in ORDER if k in w}
    d["dev_strict_summary"] = w["summary"]
    waves_out.append(d)

ids = [w["id"] for w in W]
rel = lambda a, b: pct(a / b - 1.0)  # noqa: E731

doc = {
    "what": "Stage 3 LM-screening waves for the Hindko tokenizer study: PLAN 2.2 ranked list (= cut order) and PLAN 3 "
            "'After Stage 1' / 'After Stage 2' applied to the Stage 1 sweep and the Stage 2 candidates. Read by "
            "colab/build_bundle.py: every entry with tokenizer_path + encoder_kind is a bundle candidate, in file order.",
    "written_by": "lm/make_waves.py (deterministic; no timestamps)",
    "baseline": "A1-P1r3-D2-16k",
    "lr_sweep": {"candidate_id": "A1-P1r3-D2-16k", "peak_lrs": [1e-3, 3e-3, 6e-3], "seeds": [1],
                 "budget": "Stage 3 screening budget (10 MB train bytes, d=128)",
                 "rule": "PLAN 5: chosen once on the baseline, then fixed for every candidate"},
    "frozen": {"split_manifest_sha256": man_sha, "split_manifest_verified": man_sha == EXPECTED_MANIFEST,
               "normalize_version": frozen["normalize"]["version"], "normalize_sha256": norm_sha,
               "plan_sha256": plan_sha, "plan_sha256_matches_FROZEN_json": plan_sha == frozen["other_files"]["plan"]["sha256"],
               "test_split": "never opened; all numbers are dev_strict (validation)"},
    "execution_order": ["LR sweep on A1-P1r3-D2-16k"] + ids,
    "cut_order": "PLAN 2.2/11: waves run in rank order and compute shortfalls are cut from the bottom: "
                 "A3-SPnat-D1-16k (rank 9) first, then rank 8 (conditional), 7, 6b (32k), 6a (8k), 5, 4, 3, 2. "
                 "The LR sweep and rank 1 are never cut. The decision (PLAN 7) covers completed waves only.",
    "selection_applied": {
        "after_stage_1": {
            "pre_tokenizer": {"choice": sel["pretokenizer"]["choice"], "decided_at": sel["pretokenizer"]["decision_at"],
                              "P1r3_vs_P1_pct_16k": pct(sel["pretokenizer"]["P1r3_rel_at_primary"]),
                              "same_outcome_all_sizes": sel["pretokenizer"]["same_outcome_at_all_sizes"]},
            "data_mix": {"choice": sel["data_mix"]["choice"], "decided_at": sel["data_mix"]["decision_at"],
                         "D2_vs_D1_pct_16k": pct(sel["data_mix"]["per_size"]["16k"]["D2"]["rel_overall"]),
                         "D2_vs_D1_pct_by_source_16k": {k: pct(v) for k, v in
                                                        sel["data_mix"]["per_size"]["16k"]["D2"]["rel_by_source"].items()},
                         "D2_vs_D1_pct_32k": pct(sel["data_mix"]["per_size"]["32k"]["D2"]["rel_overall"]),
                         "outcome_if_decided_at_32k": sel["data_mix"]["outcome_at_32k"],
                         "status": "FRAGILE; the Stage 1 owner asked for user confirmation before Stage 3"},
            "unigram_rank3": {"choice": sel["unigram_rank3"]["choice"],
                              "A4_bytes_per_token_16k": round(sel["unigram_rank3"]["A4_bytes_per_token"], 4),
                              "A5_bytes_per_token_16k": round(sel["unigram_rank3"]["A5_bytes_per_token"], 4),
                              "A5_failed_gates": sel["unigram_rank3"]["A5_failed_gates"]},
            "hard_gate_drops": ["A5-P1-D1-%s (G4)" % s for s in ("8k", "12k", "16k", "24k", "32k", "48k")],
            "A2_A3_lm_qualification": {
                "rule": "LM wave only if >= 1% better than A1 on dev bytes/token (PLAN 2.2 rank 9)",
                "A3_16k_vs_A1-P1r3-D1-16k_pct": rel(A3_BPT, A1P1R3D1_BPT),
                "A3_16k_vs_rank1_A1-P1r3-D2-16k_pct": rel(A3_BPT, R1_BPT),
                "A3_16k_vs_A1-P1-D1-16k_pct": rel(A3_BPT, A1P1D1_BPT),
                "A3_vs_A1-P1r3-D1_pct_by_size": {s: pct(v) for s, v in a3_rel.items()},
                "A2_16k_vs_A1-P1r3-D1-16k_pct": rel(A2_BPT, A1P1R3D1_BPT),
                "A2_16k_vs_rank1_A1-P1r3-D2-16k_pct": rel(A2_BPT, R1_BPT),
                "A2_16k_vs_A1-P1-D1-16k_pct": rel(A2_BPT, A1P1D1_BPT),
                "A2_vs_A1-P1r3-D1_pct_by_size": {s: pct(v) for s, v in a2_rel.items()},
                "outcome": "A3 qualifies under every reading of 'A1' (smallest margin %+.3f%%, against the rank-1 "
                           "tokenizer); A2 does not under any (largest %+.3f%%)" % (
                               min(rel(A3_BPT, R1_BPT), rel(A3_BPT, A1P1R3D1_BPT), rel(A3_BPT, A1P1D1_BPT)),
                               max(rel(A2_BPT, R1_BPT), rel(A2_BPT, A1P1R3D1_BPT), rel(A2_BPT, A1P1D1_BPT)))},
        },
        "after_stage_2": {
            "rule": "keep an A6/A7/A10 variant only if (i) G1-G5 pass, (ii) exact round trip, (iii) for A6 and A10 the "
                    "HF-native encoding reproduces the reference encoder on 100% of dev (else custom encoder, loses "
                    "tie-breaker a)",
            "A6-SBPE-P1-D1-16k-t090": "kept; HF-native (iii) holds",
            "A6-SBPE-P1-D1-16k-t080": "kept; HF-native (iii) holds",
            "A6-SBPE-P1-D1-32k-t080": "kept (rank 8 slot, conditional); HF-native (iii) holds",
            "A7-PickyBPE-P1-D1-16k-tau0.9": "kept; custom encoder (iii not required for A7); loses tie-breaker (a)",
            "A10-MinGram-P1-D1-16k": "kept; HF-native (iii) holds against the reimplementation's reference",
        },
    },
    "waves": waves_out,
    "dropped": [
        {"id": "A5-P1-D1 (8k, 12k, 16k, 24k, 32k, 48k)", "why": "hard gate G4 fails (HF UnigramTrainer not deterministic: "
         "piece order and scores differ between identical trainings). PLAN 3/7: dropped. Also 11-15% worse than A4."},
        {"id": "A2-P1-D1 (all sizes)", "why": "no LM wave: not >= 1% better than A1 (-0.17% at 16k vs A1-P1r3-D1; "
         "-1.14% vs the rank-1 tokenizer). Passes all gates; not a gate drop."},
        {"id": "A3-SPnat-D1 at 8k/12k/24k/32k/48k", "why": "rank 9 gets one wave at the 16k LM size, like ranks 1-5 and 7"},
        {"id": "A1-P1-D1-16k / -8k / -32k", "why": "superseded for ranks 1 and 6 by the Stage-1-fixed recipe "
         "(P1r3, D2). Still the byte-identical base of every A6/A7 candidate and the NSL reference."},
        {"id": "A8, A1-P0, A1-Pm, D3 variants", "why": "PLAN 2.2 rank 10: intrinsic-only probes, never LM waves"},
        {"id": "A9 BoundlessBPE", "why": "not built (needs an approved git download); not a ranked wave"},
    ],
    "deviations_and_interpretations": [
        "1. Rank 1 and rank 6 use the recipe fixed by PLAN 3 'After Stage 1' (pre-tokenizer P1r3, data mix D2): "
        "A1-P1r3-D2-{16k,8k,32k}. PLAN 2.2 labels rank 1 'A1 BPE-P1-D1 @16k'; that label states the rules' expected "
        "outcome ('Expected: P1', 'Choose D1 unless'), and PLAN 3 says to fix both after Stage 1. The three "
        "tokenizers are outside the pre-registered grid (built after the rules, same code, all gates pass).",
        "2. The Stage 2 candidates (ranks 2, 4, 5, 7, 8) were built on P1 and D1 before the Stage 1 selection existed "
        "and are not rebuilt here. Every A6/A7/A10-vs-rank-1 comparison is therefore confounded by P1r3-vs-P1 "
        "(digit runs only; +%.3f%% bytes/token at 16k) and D2-vs-D1 (+%.3f%% at 16k). In particular the "
        "A1-vs-A7 comparison of PLAN 4.3 no longer isolates the effect of PickyBPE's removals. See "
        "pending_user_decisions[2]." % (pct(sel["pretokenizer"]["P1r3_rel_at_primary"]),
                                         pct(sel["data_mix"]["per_size"]["16k"]["D2"]["rel_overall"])),
        "3. Rank 3 (A4) and rank 9 (A3) are SentencePiece-native (the pre-tokenizer rule does not apply) and trained "
        "on D1: PLAN's Stage 1 grid trains SentencePiece on D1 only, and no D2 SentencePiece model exists.",
        "4. PLAN names no size for the Stage 1 rules; they were applied at 16k (the size of ranks 1-5 and 7). D2 at "
        "16k is kept for the 32k wave (6b) because PLAN 3 fixes one data mix, although at 32k alone the rule gives D1.",
        "5. G2 for SuperBPE multi-word tokens: every multi-word token of ranks 2, 5 and 8 passes the standard G2 test "
        "encode(decode([id])) == [id] (0 failures). The PLAN 4.3 clause 'tested on the shortest train chunk that "
        "contains them', read literally, flags 224 / 569 / 1,120 tokens, all of which pass the standard test; the "
        "flags come from misaligned substrings and competition with longer superwords. Taken literally, the "
        "BPE remedy (delete, re-encode dev, else retrain) would drop ranks 2, 5 and 8 for tokens that are reachable. "
        "I applied the gate's stated purpose ('no unreachable tokens'): 0 unreachable, so they pass. Needs user "
        "confirmation (pending_user_decisions[1]).",
        "6. Rank 8's ratio: PLAN takes t/T from the better of ranks 2 and 5 in LM bpb. Only a 32k t/T = 0.8 build "
        "exists (its owner chose 0.8 on intrinsic tie-breakers). The entry names that file and states when it is "
        "valid; the t/T = 0.9 32k build was not made here.",
        "7. The rank-9 wave (A3) is added to the 8 (+1 conditional) waves, as PLAN 2.2 rank 9 pre-registers. The "
        "claim's N (PLAN 8) becomes 9 completed candidates, 10 with rank 8.",
        "8. G1 is defined over dev and test. The test half is deferred to Stage 5 for every wave (test discipline).",
        "9. The Stage 3 trigger for rank 8 uses 'beats' = lower mean dev bpb over the 3 seeds at the pre-registered "
        "LR; PLAN pre-registers no significance requirement for this trigger.",
    ],
    "pending_user_decisions": [
        "1. Confirm the data mix D2 (fragile: +0.947% at 16k, +0.489% at 32k). If D1 is preferred, replace ranks 1, "
        "6a, 6b with A1-P1r3-D1-16k / -8k / -32k (in the grid, gates pass; dev bytes/token 6.7227 / 6.2279 / "
        "7.0603; files candidates/standard/tok/<id>/tokenizer.json) and re-run make_waves.py after editing it.",
        "2. Confirm the reachability reading of the SuperBPE multi-word G2 clause (deviation 5). Under the literal "
        "reading ranks 2, 5 and 8 are dropped.",
        "3. The confound of deviation 2. Options (not applied): (a) rebuild A6/A7/A10 on P1r3 + D2 so every wave "
        "shares rank 1's recipe (the order PLAN 3 implies); (b) add A1-P1-D1-16k (candidates/standard/tok/"
        "A1-P1-D1-16k/tokenizer.json, sha256 4d018f82...) as a report-only control wave outside PLAN 7, which "
        "restores the controlled A1-vs-A6/A7/A10 comparisons and measures the P1r3+D2 effect on bpb; "
        "(c) accept the confound as recorded.",
    ],
    "not_done": [
        "No LM training, no bundle build, nothing on the test split.",
        "No 32k SuperBPE build at t/T = 0.9, no Stage 2 rebuild on P1r3/D2, no D2 SentencePiece models.",
        "Gate and equivalence results were taken from the candidate owners' outputs, except: every tokenizer file's "
        "sha256, an encoder load through eval/harness.py with vocabulary-size, EOT/BOS-id and synthetic round-trip "
        "checks (this script), and a full uncached G4 retrain of A7 (lm/g4_pickybpe_fullretrain/).",
    ],
}

tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
    f.write("\n")
os.replace(tmp, OUT)

# ------------------------------------------------------------------ self-check with build_bundle's own parser
cands, top_base, wsha = BB.load_waves(OUT)
got_ids = [c["id"] for c in cands]
assert got_ids == ids, "build_bundle would read %s, expected %s" % (got_ids, ids)
for c, w in zip(cands, W):
    assert c["kind"] == w["encoder_kind"] and os.path.normcase(c["path"]) == os.path.normcase(w["tokenizer_path"])
    assert c["custom_spec"] == w.get("custom")
    assert c["no_newline_wrapper"] is False
base, rule = BB.pick_baseline(cands, None, top_base)
assert base == "A1-P1r3-D2-16k", base

for r in check_rows:
    print("%-30s %-13s n_vocab %6d  dev b/tok %.4f  sha256 %s" % r)
print("build_bundle reads %d candidates in rank order; baseline %s (%s)" % (len(cands), base, rule))
print("WAVES.json sha256", wsha)
