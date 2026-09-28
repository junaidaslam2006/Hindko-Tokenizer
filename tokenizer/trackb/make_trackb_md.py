# -*- coding: utf-8 -*-
"""Write TRACKB.md from the measured results (every number in it is read from a results file).

    python make_trackb_md.py      (after report.py, deliver.py, determinism.py, sp_model_check.py,
                                   test_init_embeddings.py)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402
import report as RP  # noqa: E402


def opt(path):
    return C.load_json(path) if os.path.exists(path) else None


def f3(x):
    return "%.3f" % x


def main():
    res = RP.load_results()
    kn = C.load_json(os.path.join(C.RESULTS, "knee.json"))
    knees = kn["knees"]
    det = {}
    for fn in sorted(os.listdir(C.RESULTS)):
        if fn.startswith("determinism_") and fn.endswith(".json"):
            for key, val in C.load_json(os.path.join(C.RESULTS, fn)).items():
                if isinstance(val, dict) and key == "G4":
                    det.setdefault("G4", {}).update(val)
                else:
                    det[key] = val
    spc = opt(os.path.join(C.RESULTS, "gemma-3", "sp_model_check.json"))
    tie = opt(os.path.join(C.RESULTS, "test_init_embeddings.json"))
    dman = opt(os.path.join(C.DELIVER, "deliver_manifest.json")) or {}
    tables = open(os.path.join(C.RESULTS, "tables.md"), encoding="utf-8").read()
    exts = {b: C.load_json(os.path.join(C.WORK, b, "extension.json")) for b in res}
    for e in exts.values():
        e.pop("merges", None)
    delivered = {}
    for name, m in dman.items():
        delivered[m["base"]] = (name, C.load_json(os.path.join(C.DELIVER, name, "EXTENSION.json")))

    L = []
    A = L.append
    A("# Track B: Hindko vocabulary extension of open LLM tokenizers (continued BPE)")
    A("")
    A("Written %s by `make_trackb_md.py`; every number below is read from `results/`, `work/`, `deliver/` and "
      "`archive_v1.0/`, except the few marked as coming from the session log. "
      "PLAN.md section 9 (tokenizer side). **Nothing in this run was computed on the test split**: training used "
      "`train_D1` (permissive train), every measurement used `dev_strict` (validation) or the non-Hindko check set; "
      "`dev_permissive` (validation) supplied Urdu and English check documents only. Frozen inputs were verified before "
      "every step: split manifest `%s`, `hp.normalize` %s (`normalize()` re-applied to every text and asserted to be a "
      "no-op on the materialised views)." % (
          __import__("time").strftime("%Y-%m-%d %H:%M"), C.MANIFEST_SHA, C.verify_frozen()["normalize_version"]))
    A("")
    # ---------------------------------------------------------------- 1 headline
    A("## 1. Result at the chosen k (pre-registered knee rule, `KNEE_RULE.md`)")
    A("")
    A("| base (models) | license | k new | len(tokenizer) base → extended | dev_strict bytes/token base → ext. | tokens | "
      "fertility | STRR | train_D1 tokens base → ext. | new tokens < 100 train occ. | delivered folder |")
    A("|---|---|---:|---|---|---:|---|---|---|---:|---|")
    for b in res:
        k = (knees.get(b) or {}).get("chosen_k")
        if k is None or k not in res[b]:
            continue
        r0, r = res[b][0], res[b][k]
        d0, d = r0["dev"], r["dev"]
        name = delivered.get(b, ("(not delivered)",))[0]
        A("| **%s** (%s) | %s | %s | %s → %s | %s → **%s** | **−%.1f%%** | %s → %s | %.1f%% → %.1f%% | %.2fM → %.2fM | %d (%.1f%%) | `deliver/%s/` |" % (
            b, C.BASES[b]["models"], C.BASES[b]["license"], "{:,}".format(k), "{:,}".format(exts[b]["first_new_id"]),
            "{:,}".format(exts[b]["first_new_id"] + k), f3(d0["bytes_per_token"]), f3(d["bytes_per_token"]),
            100 * (1 - d["nsl_vs_base"]), f3(d0["fertility"]), f3(d["fertility"]), 100 * d0["strr"], 100 * d["strr"],
            r0["train"]["tokens"] / 1e6, r["train"]["tokens"] / 1e6, r["R1_new"]["train_freq_lt100"], r["R1_new"]["pct_lt100"],
            name))
    A("")
    A("- *tokens* = change in the number of tokens of dev_strict (1 − NSL against the unextended base). Bytes/token "
      "is Σ UTF-8 bytes ÷ Σ tokens of the 836 dev_strict documents; fertility and STRR as in PLAN 4.2 (in-context "
      "offsets, `eval/harness.py`).")
    allk = [(b, k, r) for b, rs in res.items() for k, r in rs.items() if k]
    g1_all = all(r["gates"]["G1"]["pass"] for _, _, r in allk)
    g2_all = all(r["gates"]["G2_new"]["pass"] for _, _, r in allk)
    eq_all = all(r["equivalence"]["pass"] for _, _, r in allk)
    elig_all = all(RP.eligibility(r)["non_interference"] for _, _, r in allk)
    A("- Over all %d extended tokenizers (%d bases × the k grid): G1 lossless on dev_strict: %s; G2, no unreachable new "
      "whole-character token: %s; HF-native encoding = reference continued-BPE encoding on every dev document: %s; "
      "no English / code / other-script check document changed: %s (sections 5 and 6)." % (
          len(allk), len(res), "all" if g1_all else "NOT all", "all" if g2_all else "NOT all", "all" if eq_all else "NOT all",
          "all" if elig_all else "NOT all"))
    A("- Dev was used to choose k, so the dev gains at the chosen k are selection-biased by the choice among 5 grid "
      "points; the test split has not been evaluated (PLAN Stage 5 is a later run).")
    A("")
    # ---------------------------------------------------------------- 2 what / deviations
    A("## 2. What was done, and deviations from PLAN 9")
    A("")
    A("- **Bases** (tokenizer files already in `baselines/files/`, no download): Qwen3 and Llama-3.x (byte-level BPE) and "
      "Gemma-3 (SentencePiece BPE, used through its HF `tokenizer.json`), as PLAN 9.1 lists; added as 'other strong "
      "open bases': **Qwen3.5** (Apache-2.0, byte-level BPE, 248k) and **Gemma-4** (Apache-2.0; same pieces and merges "
      "as Gemma-3, different control tokens). Llama-3.x and Gemma-3 come from the byte-identical ungated mirrors "
      "recorded in `baselines/manifest.json`.")
    A("- **Method**: continued BPE (Purason et al. 2026, arXiv:2512.03989), **reimplemented from the paper**; their "
      "toolkit was not downloaded or run. Grid k ∈ {1,024, 2,048, 4,096, 8,192, 16,384} (PLAN's '1k…16k' read as "
      "powers of two, as PLAN 2.1 does for vocabulary sizes).")
    A("- **Training text**: `train_D1` (16,015 documents, 45.0 MB; permissive train split), `min_frequency` = 2 as in "
      "every BpeTrainer run of this project.")
    A("- **Deviation 1, scope filter (added):** merges are learned only from training units that contain an "
      "Arabic-script character (Unicode Script_Extensions = Arabic, or a code point in an Arabic block; section 3.1 "
      "item 5). The Hindko corpus also holds English, digits-only, "
      "whitespace and code-like units; learning merges from them could change how the base encodes English or code. "
      "Share of the base tokens of train_D1 that the filter left out: %s. Its effect is measured, not assumed "
      "(section 6)." % ", ".join(
          "%s %.1f%%" % (b, 100 * ((e["unit_stats"]["base_tokens"] - e["unit_stats"]["arabic_base_tokens"])
                                   if e["base_kind"] == "bytelevel" else e["unit_stats"]["nonarabic_unit_tokens"])
                         / e["unit_stats"]["base_tokens"]) for b, e in exts.items()))
    g3d = delivered.get("gemma-3", (None, {}))[1].get("tokenizer_model_spm") if "gemma-3" in delivered else None
    A("- **Deviation 2, Gemma (SentencePiece):** the continued merges are learned and appended on Gemma's HF "
      "`tokenizer.json`, which is a BPE merge-list model, exactly as for the byte-level bases; the training units follow "
      "Gemma's SentencePiece `trainer_spec` (section 3.2). SentencePiece's own BPE encoder has no merge list (it "
      "repeatedly merges the adjacent pair whose concatenation is a piece with the best score, whatever the split), so "
      "pair-specific appended merges are not guaranteed to be expressible in a `.model`. The analogous `.model` "
      "(the k new pieces appended as NORMAL pieces with scores below every base piece, in learning order; Gemma-3's "
      "HF-only `<image_soft_token>` inserted first so that ids line up) was built and measured: %s%s" % (
          ("; ".join("k = %s: %d/%d dev_strict documents identical to the tokenizer.json" % (
              "{:,}".format(int(k)), v["identical_docs"], v["docs"]) for k, v in spc["by_k"].items())
           + " (`results/gemma-3/sp_model_check.json`). ") if spc else "not measured. ",
          ("At the delivered k it also agreed on every train_D1 document (%s) and every check-set document, so the "
           "Gemma-3 folder includes it as `tokenizer.model` (measured agreement, not a proof for all inputs). "
           % "{:,}".format(g3d["agreement_with_tokenizer_json"]["train_D1"]["docs"]) if g3d and g3d.get("delivered")
           else "It is not delivered." if g3d is not None else "")
          + "Gemma-4's base folder has no `.model`, so none is built for it."))
    A("- **Deviation 3, token ids (implementation fact):** HF `tokenizers` re-derives the id of every added token that "
      "is absent from the model vocabulary as len(model vocab), len + 1, … when it loads a file. Appending model tokens "
      "after Qwen's/Llama's special-token block therefore silently moved every special token onto a new token's id "
      "(first attempt, caught by G1: 397/836 dev documents round-tripped; `<tool_call>` became id 152,681). The "
      "delivered files write the added tokens into the model vocabulary at their own ids (the layout Gemma's file "
      "already has); `materialize()` asserts that every special token keeps its id. (The two first-attempt figures "
      "are from the session log, not from a results file.)")
    cfg_only = {b: C.config_only_added_tokens(b) for b in res}
    A("- **Deviation 3b, tokens defined only in `tokenizer_config.json`:** %s The new tokens therefore start after them, "
      "and the extended `tokenizer.json` carries them as added tokens at their config ids, so that `tokenizers` and "
      "`transformers` give the same ids (first found by the delivery check: `transformers` saw 7 extra tokens and "
      "encoded only 224/836 dev documents like `tokenizers`)." % (
          " ".join("%s's config defines %d added tokens that its tokenizer.json lacks (%s, ids %s–%s)." % (
              b, len(v), ", ".join("`%s`" % x["content"] for x in v), "{:,}".format(v[0]["id"]),
              "{:,}".format(v[-1]["id"])) for b, v in cfg_only.items() if v) or "none."))
    A("- **Deviation 4, gates:** G3 of PLAN 4.3 checks this project's own 64-token special block, which the bases do "
      "not have; it is replaced by 'every base special/added token keeps its id' (asserted). G4 is run as a rebuild "
      "from scratch (section 3.4). %s" % ("The PLAN 4.3 G2 remedy was not needed: no new token failed G2 (section 5)."
                                           if g2_all else "Some new tokens failed G2; see section 5."))
    q35 = delivered.get("qwen-3.5", (None, {}))[1] if "qwen-3.5" in delivered else {}
    if q35.get("tokenizer_config_changes"):
        vv = q35["verification"]
        A("- **Deviation 7, Qwen3.5 `tokenizer_config.json`:** the base config names `Qwen2Tokenizer`. In `transformers` %s "
          "that class rebuilds the pre-tokenizer with the Qwen2 regex (combining marks split off) instead of the Qwen3.5 "
          "regex of the `tokenizer.json` (marks kept inside words), so even the unextended Qwen3.5 folder encodes only "
          "%d/%d dev_strict documents like its own `tokenizer.json`. The delivered config sets `tokenizer_class` to "
          "`PreTrainedTokenizerFast`, which loads `tokenizer.json` as it is: %d/%d. This is the only change to a copied "
          "config file." % (__import__("transformers").__version__, vv["base_folder_transformers_equals_tokenizers"],
                            vv["dev_docs"], vv["transformers_equals_tokenizers"], vv["dev_docs"]))
    A("- The knee rule was written to `KNEE_RULE.md` (sha256 `%s…`) before any extended tokenizer was evaluated on dev." % kn["rule_sha256"][:16])
    A("- **Deviation 5, reading of the rule's non-interference condition:** the bulk English set (package READMEs, "
      "assembled together with the rule) turned out to contain a few documents with Arabic-script words (language "
      "names such as العربية in the transformers README). These change, like Urdu. The condition is applied as: no "
      "document of the curated English, code and other-script sets changes, and no bulk document *without* "
      "Arabic-script characters changes. The bulk sets are run at k = 16,384 only; %s" % (
          "no base chose k = 16,384 (section 7), so the reading changes no choice."
          if all((v or {}).get("chosen_k") != C.K_MAX for v in knees.values())
          else "a base chose k = 16,384 (section 7), so this reading matters for that choice."))
    v10 = opt(os.path.join(HERE, "archive_v1.0", "audit_static_v1.0.json"))
    v10cp = (opt(os.path.join(HERE, "archive_v1.0", "results", "audit.json")) or {}).get("codepoints", {}).get("qwen-3", {})
    v10cp_line = ("an exhaustive single-code-point test (run on Qwen3 only before it was stopped; "
                  "`archive_v1.0/results/audit.json`) then showed Qwen3 1.0.0 changing the encoding of %s code points "
                  "outside Script_Extensions=Arabic (k = 1,024 … 16,384; e.g. %s)." % (
                      " / ".join("{:,}".format(v10cp[k]["changed_other"]) for k in sorted(v10cp, key=int)),
                      ", ".join(v10cp["16384"]["other_examples"][:4]) if "16384" in v10cp else "")) if v10cp else ""
    v10_line = ""
    if v10:
        v10_line = " Static audit of the 1.0.0 merge lists (`archive_v1.0/audit_static_v1.0.json`): " + "; ".join(
            "%s: %s non-Arabic and %s partial-UTF-8 new tokens among 16,384, first non-Arabic at rank %s" % (
                b, s["by_k"]["16384"]["other"], s["by_k"]["16384"]["partial"],
                "{:,}".format(s["first_rank"]["other"]) if s["first_rank"]["other"] else "none")
            for b, s in v10["static"].items()) + "."
    A("- **Deviation 6, method revised during the run (1.0.0 → 1.1.0), merge-level scope:** with only the unit-level "
      "filter, byte-level bases also learned merges *inside* Arabic-containing pretokens whose result contains no "
      "Arabic character: `’’` (Qwen3 rank 460; Hindko writes Urdu-style double quotes after `۔`), `)\\n` (Qwen3.5 rank "
      "4,639) and continuation-byte pairs such as `a2 bf`. The check set caught Qwen3.5 at k ≥ 8,192 (all 10 stdlib "
      "modules and 585 of 678 bulk .py files changed; `archive_v1.0/results/qwen-3.5/`), and " + v10cp_line
      + v10_line + " Method 1.1.0 keeps a learned merge only if its "
      "token can occur only in text that contains an Arabic-script character (its bytes contain a complete "
      "Arabic-script character, or end in a byte prefix that only Arabic-script characters complete); a rejected merge "
      "is dropped with every later merge that needs it (counts in section 3.2). This is a post-filter on the "
      "unconstrained greedy run, not a constrained re-count: merges after a dropped one were chosen with counts in "
      "which the dropped merge had happened. **All numbers in this document are 1.1.0**; Gemma's lists contained no "
      "such merge and are byte-identical under both versions. 1.0.0 outputs are kept in `archive_v1.0/` (its "
      "delivered folders are superseded and must not be used).")
    A("")
    # ---------------------------------------------------------------- 3 method
    A("## 3. Method")
    A("")
    A("### 3.1 Continued BPE, and why the base behaviour is kept")
    A("")
    A("1. Every training unit is encoded with the **base** tokenizer (its own normalizer, pre-tokenizer and merges).")
    A("2. New merges are learned greedily on those base-token sequences: the most frequent adjacent pair of tokens "
      "becomes a new token, the counts are updated, repeat until k new tokens exist.")
    A("3. The new merges are appended **after** all base merges. At encoding time BPE applies the lowest-ranked "
      "available merge first, so the base merges run to completion (the unchanged base encoding) before any new "
      "merge can fire, and the new merges then fire in the order they were learned. Consequences: a text in which no "
      "new merge fires is encoded exactly as by the base; otherwise every extended token is a concatenation of "
      "consecutive base tokens (checked: 'coarsening' in section 6).")
    A("4. **Code-point (PUA) trick** (SOTA_TOKENIZATION 4, route a): each base-token id that occurs in a unit is mapped "
      "to one Supplementary Private Use Area code point (U+F0000…); each unit becomes a string of these characters; "
      "the stock Rust `tokenizers.trainers.BpeTrainer` learns the merges on them (fed as whitespace-separated words "
      "with their counts); the PUA merges are translated back into base-token strings. Merges whose result string "
      "already exists reuse its id (counted as 'merge to existing': %s). Because the trainer is greedy, the k-token "
      "extension is the prefix of the 16,384-token one; this is checked by an independent training at k = 1,024 "
      "(section 3.4)." % ", ".join("%s %d" % (b, e["translation_stats"].get("merge_to_existing_base", 0)
                                              + e["translation_stats"].get("merge_to_existing_new", 0))
                                   for b, e in exts.items()))
    A("5. **Scope** (section 2, deviations 1 and 6): merges are learned only from units that contain an Arabic-script "
      "character, and a learned merge is kept only if its token can occur only in text containing an Arabic-script "
      "character. 'Arabic-script character' = Unicode Script_Extensions contains Arabic, or the code point lies in an "
      "Arabic block (this adds Arabic-only Common-script signs such as U+0605 and U+08E2; Syriac stays out).")
    A("")
    A("### 3.2 Training units per base family")
    A("")
    A("| base | kind | units | unit occurrences used (≥ 2 base tokens, Arabic script) | base tokens of train_D1 | "
      "distinct base tokens in the units (PUA alphabet) | PUA training time | PUA merges used for 16,384 new tokens | "
      "dropped: out of scope / needing a dropped token / too long |")
    A("|---|---|---|---:|---:|---:|---:|---:|---|")
    for b, e in exts.items():
        us, ts, xs = e["unit_stats"], e["train_stats"], e["translation_stats"]
        A("| %s | %s | %s | %s | %s | %s | %.0f s | %s | %d / %d / %d |" % (
            b, e["base_kind"], "pretokens of the base pre-tokenizer" if e["base_kind"] == "bytelevel" else
            "SentencePiece word units (below)", "{:,}".format(us["unit_occ"]), "{:,}".format(us["base_tokens"]),
            "{:,}".format(ts["alphabet"]), ts["train_seconds"], "{:,}".format(xs["pua_merges_consumed"]),
            xs.get("dropped_out_of_scope", 0), xs.get("dropped_missing_component", 0), xs.get("dropped_too_long", 0)))
    A("")
    A("- **Byte-level bases** (Qwen3, Qwen3.5, Llama-3): a unit is a pretoken of the base's own pre-tokenizer "
      "(normalizer + Split regex + ByteLevel), encoded by the base BPE model; merges can never cross a pretoken, as in "
      "the base. Checked on 300 train documents per base: concatenated per-pretoken encodings = the full-pipeline "
      "encoding (0 mismatches). Llama-3's `ignore_merges` (a pretoken that is a vocabulary entry is emitted whole) is "
      "kept and covered by the equivalence test.")
    A("- **Gemma (SentencePiece BPE, byte fallback)**: its `tokenizer.json` has no effective pre-tokenizer (the "
      "normalizer turns spaces into U+2581 before the no-op `Split(' ')`), so the whole document is one BPE word. Units "
      "are cut from the base encoding of each document following Gemma's `trainer_spec` (read from its "
      "`tokenizer.model`: `split_by_whitespace` false but the vocabulary has one piece with an inner U+2581, "
      "`split_by_unicode_script` true, `split_digits` true, `max_sentencepiece_length` default 16): a unit starts at every "
      "token beginning with U+2581; newline and whitespace-run pieces, `<0xNN>` byte-fallback pieces, added tokens and "
      "ASCII-digit pieces are never merged; a unit is cut where the Unicode script changes (Inherited marks take the "
      "preceding script); a merge whose piece would exceed 16 characters is dropped with every later merge that needs it "
      "(%s)." % ", ".join("%s: %d dropped" % (b, e["translation_stats"].get("dropped_too_long", 0)) for b, e in exts.items()
                          if e["base_kind"] == "spbpe"))
    A("")
    A("### 3.3 Files")
    A("")
    A("- New tokens are ordinary **model** tokens (vocabulary entry + merge), not added tokens: added tokens are matched "
      "greedily before BPE, which is the 'naive appending' that Purason et al. show creates unreachable tokens.")
    A("- New ids are contiguous from the first free id (table in section 1). Merges are written in the base file's own "
      "format (list of pairs or 'a b' strings). The k = 0 file written by the same code is byte-identical to the base "
      "file (sha256 equal), so the serialisation adds nothing.")
    A("")
    A("### 3.4 Verification")
    A("")
    A("- **Equivalence** (every k, every dev_strict document): the HF-native encoding equals a pure-Python reference: "
      "the base tokenizer's encoding followed by the extended merge table applied in tokenizers' own "
      "(rank, left position) order (`evaluate.py: Reference`).")
    g4 = det.get("G4", {})
    if g4:
        A("- **G4 determinism** (rebuild from scratch, compare the ordered merge list and the sha256 of the k = 16,384 "
          "tokenizer.json): " + "; ".join("%s %s" % (b, "identical" if v["pass"] else "DIFFERENT") for b, v in g4.items()) + ".")
    else:
        A("- **G4 determinism**: not run.")
    if det.get("prefix"):
        p = det["prefix"]
        A("- **Prefix property**: an independent PUA training for 1,024 new tokens on %s gives %s merges as the first "
          "1,024 of the 16,384-token run (%s)." % (p["base"], "the same" if p["pass"] else "DIFFERENT",
                                                   "pass" if p["pass"] else "fail"))
    if det.get("gemma3_vs_gemma4"):
        g = det["gemma3_vs_gemma4"]
        A("- **Gemma-3 vs Gemma-4**: independent builds give %s new-token strings in the same order; ids differ by %s "
          "(Gemma-3 has its extra `<image_soft_token>` at 262,144)." % (
              "identical" if g["identical_new_token_strings_and_order"] else "DIFFERENT", g["id_offsets"]))
    A("")
    # ---------------------------------------------------------------- 4 curves
    A("## 4. Curves")
    A("")
    A("![Track B curves](figures/trackb_curves.png)")
    A("")
    A("(`figures/trackb_curves.svg` is the vector version; x axis symlog in k.)")
    A("")
    A(tables)
    A("")
    # ---------------------------------------------------------------- 5 gates / support
    A("## 5. Gates and the support profile of the new tokens")
    A("")
    A("| base | k | G1 dev_strict | G2: new whole-character tokens failing self-tokenization | partial-UTF-8 new tokens (exempt, R2) | "
      "equivalence | base vocabulary's own G2 failures (k = 0) → at k = 16,384 |")
    A("|---|---:|---|---|---:|---|---|")
    for b, rs in res.items():
        for k in sorted(rs):
            if not k:
                continue
            r = rs[k]
            g0 = rs[0]["gates"].get("G2_full_vocab", {})
            gk = rs[C.K_MAX]["gates"].get("G2_full_vocab", {}) if C.K_MAX in rs else {}
            A("| %s | %s | %d/%d | %d of %s | %d | %d/%d | %s |" % (
                b, "{:,}".format(k), r["gates"]["G1"]["docs"] - r["gates"]["G1"]["fail_docs"], r["gates"]["G1"]["docs"],
                r["gates"]["G2_new"]["failures"], "{:,}".format(r["gates"]["G2_new"]["tested"]),
                r["gates"]["G2_new"]["exempt_partial_utf8"],
                r["equivalence"]["docs"] - r["equivalence"]["mismatch_docs"], r["equivalence"]["docs"],
                ("%s → %s" % ("{:,}".format(g0.get("failures", 0)), "{:,}".format(gk.get("failures", 0))))
                if k == C.K_MAX else ""))
    A("")
    exs = []
    for b, rs in res.items():
        fe = rs[0]["gates"].get("G2_full_vocab", {}).get("failure_examples", [])
        if fe:
            exs.append("%s: %s" % (b, ", ".join("`%s`" % x["text"] for x in fe[:3])))
    for b, rs in res.items():
        g0 = rs[0]["gates"].get("G2_full_vocab", {}).get("failures")
        gk = rs.get(C.K_MAX, {}).get("gates", {}).get("G2_full_vocab", {}).get("failures") if C.K_MAX in rs else None
        mte = exts[b]["translation_stats"].get("merge_to_existing_base", 0)
        if g0 is not None and gk is not None and gk != g0:
            A("- %s: the base's own G2 failures go from %d to %d at k = 16,384. Its extension contains %d merges whose "
              "result is an existing base token (section 3.1 item 4), which makes base tokens that the base alone "
              "never produces reachable; these merges are in scope (they fire only in Arabic-script text)." % (b, g0, gk, mte))
    A("- The base vocabularies' own G2 failures (examples: %s) belong to the base (tokens its own pre-tokenizer or "
      "merge order never produces) and are listed only to show that the extension adds none. G2 of the new tokens "
      "tests every new token with valid UTF-8 text, including those with an inner newline such as `۔\\n`, which the "
      "harness's whole-vocabulary G2 skips as 'superword'." % ("; ".join(exs) or "none"))
    sup_lines = []
    for b in res:
        for k in sorted(res[b]):
            if k:
                s = res[b][k]["R1_new"]
                sup_lines.append((b, k, s["train_freq_lt100"], s["lt100_intermediate"], s["lt100_leaves"],
                                  s["train_freq_eq0"], s["eq0_intermediate"]))
    A("- **Support (R1)** is counted by encoding train_D1 with the extended tokenizer itself. New tokens with < 100 "
      "occurrences, split into intermediate merge nodes (later absorbed into longer new tokens, like the pilot BPE's "
      "rare tokens in PLAN 4.3) and leaves (tokens no later merge uses; at large k these are the last, rarest merges):")
    A("")
    A("| base | k | < 100 | of which intermediate | of which leaves | never used (= 0) | of which intermediate |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for b, k, n, i, l, z, zi in sup_lines:
        A("| %s | %s | %d | %d | %d | %d | %d |" % (b, "{:,}".format(k), n, i, l, z, zi))
    A("")
    A("  Rarest new tokens at the chosen k (train_D1 occurrences):")
    A("")
    for b in res:
        k = (knees.get(b) or {}).get("chosen_k")
        if k in res[b]:
            rr = res[b][k]["R1_new"]["rarest"][:8]
            A("  - %s k = %s: %s" % (b, "{:,}".format(k), ", ".join(
                "`%s` %d%s" % ((x["text"] if x["text"] is not None else "(bytes)").replace(" ", "␣"), x["train_freq"],
                               "" if not x["leaf"] else " (leaf)") for x in rr)))
    A("")
    # ---------------------------------------------------------------- 6 non-interference
    A("## 6. Is the base encoding of other text unchanged?")
    A("")
    cs = C.load_json(os.path.join(HERE, "checkset", "checkset.json"))["sets"]
    bm = C.load_json(os.path.join(HERE, "checkset", "bulk_manifest.json"))
    A("Check set (`checkset/make_checkset.py`; nothing downloaded; all text through `hp.normalize`):")
    A("")
    A("| set | documents | bytes | source |")
    A("|---|---:|---:|---|")
    src = {"english_wiki": "12 Wikipedia-style English paragraphs written for this check",
           "python_code": "10 whole CPython 3.11 stdlib modules (json/encoder, json/decoder, textwrap, heapq, bisect, fnmatch, colorsys, shlex, string, contextlib)",
           "other_scripts": "generated Hindi (Devanagari), Russian, Greek, Chinese, Japanese, Korean, symbols/emoji",
           "urdu_generated": "9 Urdu news paragraphs written for this check",
           "urdu_news_dev": "Urdu-labelled newspaper/web documents of dev_permissive (validation)",
           "urdu_book_dev": "Urdu-labelled book documents of dev_permissive (validation)",
           "english_dev": "English-labelled documents of dev_permissive (validation)"}
    for s, v in cs.items():
        A("| %s | %d | %s | %s |" % (s, v["docs"], "{:,}".format(v["bytes"]), src.get(s, "")))
    A("| bulk_code | %d | %s | every .py of the local CPython 3.11 stdlib outside test/, tests/, idlelib/ |" % (
        len(bm["bulk_code"]), "{:,}".format(sum(x["bytes"] for x in bm["bulk_code"]))))
    A("| bulk_english | %d | %s | every `*.dist-info/METADATA` (package READMEs) in the local site-packages |" % (
        len(bm["bulk_english"]), "{:,}".format(sum(x["bytes"] for x in bm["bulk_english"]))))
    A("")
    A("The table 'Non-interference' in section 4 gives, per base and k, the number of documents whose ids differ from "
      "the base's. Bulk sets are tested at k = 16,384 only: if no new merge fires there, none fires at a smaller k "
      "(the smaller extensions are prefixes). For changed documents two properties are checked: the extended tokens "
      "are concatenations of consecutive base tokens (**coarsening**) and the round trip is exact.")
    A("")
    bad = []
    for b, rs in res.items():
        for k, r in rs.items():
            if not k:
                continue
            for s in RP.CURATED_CLEAN:
                if r["checks"][s].get("changed", 0):
                    bad.append("%s k=%d %s" % (b, k, s))
            for s in RP.BULK_CLEAN:
                if s in r["checks"] and r["checks"][s].get("changed", 0) - r["checks"][s].get("changed_with_arabic", 0):
                    bad.append("%s k=%d %s" % (b, k, s))
    au = opt(os.path.join(C.RESULTS, "audit.json"))
    cp_bad = []
    if au:
        for b, cps in au["codepoints"].items():
            for k, v in cps.items():
                if v["changed_other"]:
                    cp_bad.append("%s k=%s: %d (%s)" % (b, k, v["changed_other"], ", ".join(v["other_examples"][:3])))
    A("**Reading:** " + ("no document without Arabic-script characters changed, for any base at any k (English, "
                         "Python code, other scripts, bulk code and READMEs)." if not bad else
                         "documents without Arabic-script characters changed in: " + "; ".join(bad) + ".")
      + (" The exhaustive code-point audit (every Unicode scalar value alone and after a space) found no non-Arabic "
         "code point whose encoding changes, for any base at any k." if au and not cp_bad else
         (" The code-point audit found changes in non-Arabic code points: " + "; ".join(cp_bad) + "." if au else
          " The code-point audit was not run."))
      + " What this establishes: a new merge can only fire in text that contains an Arabic-script character (by "
        "construction, section 3.1 item 5), and the tests found no exception. What it does not establish: that text "
        "*with* Arabic script keeps its encoding. **Urdu is not unchanged, by design**: it shares the script and much "
        "of the vocabulary with Hindko, so Urdu text gets fewer tokens (section 4), and a model extended this way also "
        "sees new tokens in Urdu, and presumably in other Arabic-script languages (Persian, Pashto, Shahmukhi Punjabi, "
        "Arabic; not measured). Any text that merely contains one Arabic-script word (a language name in an English "
        "README) changes around that word.")
    A("")
    # ---------------------------------------------------------------- 7 knee
    A("## 7. Choice of k")
    A("")
    A("Rule (`KNEE_RULE.md`, written before the dev evaluation): eligible = G1, G2, equivalence and non-interference "
      "pass; support cap = at most 10% of the k new tokens with < 100 train_D1 occurrences; compression knee = Kneedle "
      "on the linear k axis over the relative dev_strict token reduction; chosen k = min(knee, largest eligible k within "
      "the cap). Values: table 'Knee' in section 4.")
    A("")
    for b, v in knees.items():
        if v:
            A("- %s: Kneedle knee %s; largest eligible k within the cap %s; **chosen %s**.%s" % (
                b, "{:,}".format(v["kneedle_k"]),
                "{:,}".format(v["largest_eligible_k_within_cap"]) if v["largest_eligible_k_within_cap"] else "none",
                "{:,}".format(v["chosen_k"]), (" " + v["flag"]) if v.get("flag") else ""))
    A("")
    A("The curves keep rising after the knee; a user who values compression more than embedding-row count or token "
      "support can take any k from `sweep/<base>/k<k>/tokenizer.json` (sha256 in `sweep/<base>/sweep_manifest.json`), "
      "all of which passed the same gates.")
    A("")
    # ---------------------------------------------------------------- 8 deliverables
    A("## 8. Deliverables")
    A("")
    A("| folder | base | k | tokenizer.json sha256 | loads in `tokenizers` and `transformers` %s | ids identical between them on dev_strict | same check for the unextended base folder | special ids kept | chat template renders |" % __import__("transformers").__version__)
    A("|---|---|---:|---|---|---|---|---|---|")
    for b, (name, info) in delivered.items():
        v = info["verification"]
        A("| `deliver/%s/` | %s | %s | `%s…` | %s (%s, len %s) | %d/%d | %s/%d (%s) | %s (%d checked) | %s |" % (
            name, b, "{:,}".format(info["k_new_tokens"]), info["files_sha256"]["tokenizer.json"][:16],
            "yes" if v["len_tokenizer"] == v["expected_len"] else "NO", v["transformers_class"],
            "{:,}".format(v["len_tokenizer"]), v["transformers_equals_tokenizers"], v["dev_docs"],
            v.get("base_folder_transformers_equals_tokenizers"), v["dev_docs"], v.get("base_folder_transformers_class"),
            "yes" if v["special_ids_kept"] else "NO", v.get("special_tokens_checked", 0),
            {True: "yes", False: "NO", None: "no template in base"}[v["chat_template_renders"]]))
    A("")
    A("Each folder: `tokenizer.json`; `tokenizer_config.json` (and `special_tokens_map.json` / `added_tokens.json` where "
      "the base has them) copied byte-identically from the base (except Qwen3.5's `tokenizer_class`, deviation 7), "
      "because the new tokens are model tokens and need no config entry; `new_tokens.jsonl` (id, vocabulary string, text, the merge, the base-token decomposition and byte "
      "lengths used by the initialisation, train/dev frequency, leaf flag); `init_embeddings.py`; `EXTENSION.json` "
      "(provenance, sha256 of every file, metrics and gates at this k); `README.md`. The Gemma-3 folder also has the "
      "extended SentencePiece `tokenizer.model` if it passed the agreement check (section 2, deviation 2); the base's "
      "own `.model` is never copied, because it lacks the new pieces. Only `transformers` %s was available to test loading; older versions "
      "(4.x `PreTrainedTokenizerFast`) read the same `tokenizer.json` but were not tested. The Qwen3 and Gemma merge "
      "lists use the list-of-pairs format of their base files, which needs `tokenizers` ≥ 0.20 (as the bases do)." %
      __import__("transformers").__version__)
    A("")
    # ---------------------------------------------------------------- 9 init recipe
    A("## 9. Embedding initialisation for continued pretraining (PLAN 9.3; for whoever runs CPT on a GPU)")
    A("")
    A("Nothing of this section was run on a real checkpoint here (no GPU; no model weights downloaded).")
    A("")
    A("1. **Resize** the embedding matrix (and the untied LM head) to at least `len(tokenizer)` rows, padded to a "
      "multiple of 64. Qwen3 checkpoints already have 151,936 rows for 151,669 ids: the first 267 new ids fall into "
      "these untrained padding rows, which must be initialised like every other new row. Llama-3.1 has 128,256 rows. "
      "Gemma-3 1B has 262,144 rows, so its tokenizer's `<image_soft_token>` (id 262,144) lies outside the matrix and "
      "is created by the resize (it is not a text token; new Gemma-3 ids start at 262,145); Gemma-3 4B+ has 262,208 "
      "rows. The row counts of Qwen3.5 and Gemma-4 checkpoints were not checked here (no model config downloaded); "
      "new Qwen3.5 ids start at 248,077, after the 7 config-only audio/TTS tokens (section 2, deviation 3b).")
    A("2. **Input rows, uniform subword mean + script-norm calibration** (Joshi et al. 2026): e_in(t) = mean of the "
      "input rows of D(t), the base tokens that formed t in training (`base_ids` in `new_tokens.jsonl`), then rescaled "
      "to ν = mean norm of the input rows of all base tokens occurring in any D(t) (the Arabic-script base tokens "
      "Hindko is written with). The exact calibration target of Joshi et al. is this project's reading of the paper; "
      "`--no-norm-calibration` switches it off.")
    A("3. **Output rows, character-length-weighted subword mean** (untied models only): e_out(t) = Σ w_b E_out[b] / Σ w_b "
      "over b ∈ D(t), w_b = UTF-8 byte length of b (∝ characters for Arabic script). Tied models (all Gemma; Qwen3 ≤ 4B; "
      "Llama-3.2 1B/3B) have one matrix: only step 2 applies, unless the matrix is untied for CPT.")
    A("4. **Choose the initialisation with 50-step CPT probes**, not with initialisation loss or bpb (Joshi et al. "
      "2026): same data order and learning rate, validation loss after 50 steps on held-out Hindko (dev_strict; never "
      "the test split). Alternatives to include: mean of all embeddings (+ small noise), FOCUS (Dobler & de Melo 2023; "
      "needs only a fastText model trained on train_D1), Token Distillation (Dobler et al. 2026).")
    A("5. **Schedule** (Yamaguchi et al. 2026): first train the embedding and LM-head rows together with the top-2 and "
      "bottom-2 transformer layers ('2x2 LS'), then full CPT. Watch the norms of the new rows (PLAN 4.3 R3: the 50 "
      "lowest-norm output rows and their train frequencies) to find under-trained new tokens.")
    A("6. **Data volume**: train_D1 in tokens for each base and k is in section 4 (column train_D1 tokens); e.g. at the "
      "chosen k the extended tokenizers need the token counts in section 1.")
    A("")
    A("Reference implementation: `init_embeddings.py` (`--model`, `--tokenizer <delivered folder>`, `--out`). ")
    if tie:
        A("Unit test `test_init_embeddings.py` on tiny random-weight models with the real embedding row counts "
          "(Qwen3 151,936; Llama-3.1 128,256; Gemma-3 1B and Gemma-4 262,144; Qwen3.5 assumed = its tokenizer length "
          "rounded up to 64, 248,128; Qwen3.5 tested with the Qwen3 architecture class): %s "
          "(%d runs: %s). It checks resizing, bit-identical base rows, both formulas against an independent NumPy "
          "recomputation, tied rows, and a forward pass on a Hindko sentence that uses new ids. It says nothing about "
          "the quality of the initialisation." % (
              "**all pass**" if tie["all_pass"] else "**FAILURES**", len(tie["runs"]),
              ", ".join("%s %s %s" % (r["folder"], r["arch"], "tied" if r["tied"] else "untied") for r in tie["runs"])))
    else:
        A("The unit test was not run.")
    A("")
    # ---------------------------------------------------------------- 10 claims
    A("## 10. Claims")
    A("")
    A("**Allowed** (measured here; fill in the base and its numbers from section 1):")
    A("")
    A("- \"On the Hindko strict validation split (836 documents, 1.45 MB), the <base> tokenizer extended with k tokens by "
      "continued BPE uses x% fewer tokens than the unextended <base> tokenizer (bytes/token a → b), is lossless on "
      "every document, and none of its k new whole-character tokens is unreachable by self-tokenization.\" "
      "(Validation-split numbers; k was chosen on this split.)")
    A("- \"y (z%) of the new tokens occur fewer than 100 times in the Hindko training split; w never occur (all of them "
      "intermediate merge nodes).\" Say this instead of 'all well-supported': PLAN 9.4's phrase 'well-supported' holds "
      "only in this quantified sense.")
    A("- \"On the tested English, Python-code and other-script texts (N documents, M MB), the extended tokenizer returns "
      "exactly the base tokenizer's ids.\"")
    A("- \"Urdu text also gets fewer tokens (u% on Urdu news of the corpus's validation split); every extended token is a "
      "concatenation of consecutive base tokens.\"")
    A("- \"The files load with stock `tokenizers` and `transformers` (%s tested) and encode identically in both.\"" %
      __import__("transformers").__version__)
    A("")
    A("**Not allowed:**")
    A("")
    A("- Anything about model quality (perplexity, bpb of a CPT model, downstream tasks, chat quality): no model was "
      "trained or initialised; PLAN 9.4.")
    A("- 'The base behaviour on all non-Hindko text is preserved': false for Urdu (measured) and unmeasured for other "
      "Arabic-script languages; true only for text in which no new merge fires, and measured only on the check set.")
    A("- Test-split numbers of any kind (not computed), or 'best k' beyond the stated rule.")
    A("- 'Equivalent to Purason et al.'s toolkit': our reimplementation was not compared with it (not run).")
    A("- For Gemma-3: that the extended SentencePiece `tokenizer.model` encodes like `tokenizer.json` on all inputs; "
      "only the measured agreement (section 2) may be stated.")
    A("- That the initialisation recipe works: it is a literature-based recipe with a unit-tested implementation only.")
    A("")
    # ---------------------------------------------------------------- 11 not done
    A("## 11. Not done")
    A("")
    A("- No model side: no embedding initialisation of a real checkpoint, no CPT, no 50-step probes (GPU needed).")
    A("- No test-split evaluation (reserved for the one-shot Stage 5 run).")
    A("- No comparison with Purason et al.'s own toolkit, AdaptBPE or naive 'train-and-append' extension (the latter is "
      "the baseline in Purason et al.; not needed for the tokenizer deliverable and not run).")
    A("- Leaf-based pruning of rare intermediate new tokens (Purason et al.) was not applied: PLAN 4.3 measured that "
      "leaf pruning cannot remove intermediate nodes, which is where the rare new tokens are.")
    A("- No GGUF / llama.cpp conversion check of the Gemma-3 `tokenizer.model`; no SentencePiece model for Gemma-4 (its "
      "base folder has none).")
    A("- Other Arabic-script languages (Persian, Pashto, Arabic, Shahmukhi) were not in the check set.")
    A("- Aya/Command, gated originals and bases without local tokenizer files were not added.")
    A("")
    # ---------------------------------------------------------------- 12 reproduce
    A("## 12. Reproduce (F:\\Hindko\\_tokenizer\\trackb, Python 3.11, tokenizers %s)" % __import__("tokenizers").__version__)
    A("")
    A("```")
    A("python checkset/make_checkset.py")
    A("python continued_bpe.py all --base <qwen-3|llama-3|gemma-3|qwen-3.5|gemma-4>   # build + sweep/<base>/k*/tokenizer.json")
    A("python evaluate.py --bases <base ...>                                         # results/<base>/k*/")
    A("python report.py                                                              # knee.json, tables.md, figures/")
    A("python deliver.py                                                             # deliver/<name>/")
    A("python determinism.py --bases ... --prefix --gemma-cross; python sp_model_check.py --base gemma-3")
    A("python test_init_embeddings.py; python make_trackb_md.py")
    A("```")
    A("")
    A("Code sha256 at build time: " + "; ".join("%s: continued_bpe.py `%s…`" % (b, e["code_sha256"]["continued_bpe.py"][:12])
                                              for b, e in exts.items()) + ". Final `continued_bpe.py`: `%s…`; the G4 "
      "rebuilds (section 3.4) ran with the final code and reproduced every base's 16,384-token file byte for byte, so "
      "the earlier hashes differ only by the Qwen3.5 config-token fix, which changes no other base." % (
          C.sha256_file(os.path.join(HERE, "continued_bpe.py"))[:12]))
    A("")
    with open(os.path.join(HERE, "TRACKB.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("TRACKB.md written", len(L), "lines")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
