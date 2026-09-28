# -*- coding: utf-8 -*-
"""Compact equivalence.json from equivalence_full.json (verify.py), exact_check.json, build_manifest.json and the
scheme-selection runs in work/."""
import json
import os

REL = r"F:\Hindko\_tokenizer\release_build\sp32k"
W = os.path.join(REL, "work")
full = json.load(open(os.path.join(REL, "equivalence_full.json"), encoding="utf-8"))
exact = json.load(open(os.path.join(REL, "exact_check.json"), encoding="utf-8"))
man = json.load(open(os.path.join(REL, "build_manifest.json"), encoding="utf-8"))
stress_sum = json.load(open(os.path.join(REL, r"stress\stress_summary.json"), encoding="utf-8"))
old_mm = json.load(open(os.path.join(W, "old_mismatches.json"), encoding="utf-8"))
REQUIRED = ["newlines", "spaces", "digits", "latin", "codepoints", "zwnj", "tone_letters"]


def row(r, specials=False):
    n = r["items"]
    d = {"n": n, "canonical_tokens": r["tokens_canonical"],
         "hf_tokenizer_json_ids_equal": r["hf_ids_equal"], "transformers_ids_equal": r["tf_ids_equal"],
         "hf_decode_exact": r["hf_decode_exact"], "transformers_decode_exact": r["tf_decode_exact"],
         "canonical_decode_exact": r["canonical_decode_exact"],
         "changed_vs_old_model": r.get("changed_vs_old", 0),
         "changed_vs_old_model_all_changed_lines_exact_ties": r.get("changed_vs_old_all_lines_exact_ties", 0),
         "changed_vs_old_model_NOT_exact_tie": r.get("changed_vs_old_NOT_exact_tie", 0)}
    if specials:
        d["reference"] = "split-canonical: special-token strings cut out, each segment encoded by the canonical encoder"
        d["hf_ids_equal_to_plain_canonical"] = r.get("hf_ids_equal_to_plain_canonical", 0)
    else:
        d["old_model_float32_tie_items"] = r.get("old_float32_tie_items", 0)
        d["changed_vs_old_model_in_old_tie_items"] = r.get("changed_vs_old_in_old_tie_items", 0)
    return d


sets = {}
for k, r in full["sets"].items():
    sets[k] = row(r)
    sets[k]["unit"] = "documents"
    sets[k]["sha256"] = r["sha256"]
    sets[k]["lines"] = exact[k]["lines"]
    sets[k]["lines_independent_exact_viterbi_equal"] = exact[k]["exact_viterbi_equals_sentencepiece"]
    sets[k]["lines_within_float32_bound"] = exact[k]["lines_within_bound"]
    sets[k]["max_abs_compared_value"] = exact[k]["max_abs_compared_value"]
stress = {k: row(r, specials=(k == "specials")) for k, r in full["stress"].items()}
for k in stress:
    stress[k]["unit"] = "items"
    stress[k]["required_category"] = k in REQUIRED
old_lines = {}
for r in old_mm:
    old_lines.setdefault(r["set"], []).append({"line": r["line"], "old_float32": r["sp_span"], "exact": r["hf_span"]})

out = {
    "candidate": "R2-A4-SPnat-D2-32k",
    "verdict": {
        "hf_equals_canonical_on_all_required_sets": all(v["hf_tokenizer_json_ids_equal"] == v["n"] and
                                                        v["transformers_ids_equal"] == v["n"] for v in sets.values()),
        "decode_roundtrip_exact_on_all_required_sets": all(v["hf_decode_exact"] == v["n"] and
                                                           v["canonical_decode_exact"] == v["n"] for v in sets.values()),
        "hf_equals_canonical_on_all_required_stress_categories": all(
            stress[c]["hf_tokenizer_json_ids_equal"] == stress[c]["n"] and stress[c]["transformers_ids_equal"] == stress[c]["n"]
            and stress[c]["hf_decode_exact"] == stress[c]["n"] for c in REQUIRED),
        "dev_strict_encodings_changed": sets["dev_strict"]["changed_vs_old_model"],
        "test_strict_encodings_changed": sets["test_strict"]["changed_vs_old_model"],
        "changes_that_are_not_exact_ties": sum(v["changed_vs_old_model_NOT_exact_tie"] for v in sets.values()) +
                                           sum(v["changed_vs_old_model_NOT_exact_tie"] for v in stress.values()),
        "blocker": None,
        "limit": "float32 bound: equality is guaranteed while every value the SentencePiece Viterbi compares on a line "
                 "is < 1024 in magnitude (gauge-shifted scores); beyond it sentencepiece's float32 arithmetic can resolve "
                 "an exact tie differently. Observed once, on a synthetic 80,010-character single line (stress "
                 "long_lines-000062); 0 times in the four data sets (10 of 169,111 lines exceed the bound, all equal)."},
    "score_rule": {"grid": "2^-14", "rounding": "round half to even", "max_abs_rounding": man["rounding"]["max_abs_rounding"],
                   "mean_abs_rounding": man["rounding"]["mean_abs_rounding"],
                   "gauge": "exact per-character offset w(c) on the same grid (gauge.json)",
                   "new_normal_score_range": man["new_normal_score_range"],
                   "sentencepiece_unk_score": man["sentencepiece_unk_score"], "hf_low_score": man["low_score_hf"]},
    "old_export_problem": {"old_tokenizer_json_sha256": full["old_files"]["tokenizer.json"],
                           "old_sp_model_sha256": full["old_files"]["sp.model"],
                           "lines_where_old_float32_sentencepiece_differs_from_exact_arithmetic": {
                               k: len(v) for k, v in old_lines.items()},
                           "all_are_permutation_ties_X+XX_vs_XX+X": all(r["permutation"] and r["exact_equal"] for r in old_mm),
                           "detail": old_lines},
    "grid_selection_evidence": {
        "non_tie_lines_changed_vs_old (all four sets)": {"K=10": 6, "K=12": 5, "K=13": 4, "K=14": 0, "K=15": 0,
                                                         "K=16": 0, "K=18": 0, "K=20": 0},
        "note": "every non-tie change at K<=13 is a near-tie (old exact gap 1.2e-4..1.9e-4) that rounding made an exact tie; "
                "K=14 is the coarsest grid with none, i.e. the one with the largest float32-exact range (2^10)",
        "max_abs_path_score_in_four_sets": {"raw_scores_best_path": 13117.8, "gauge_best_path": 1736.0,
                                             "gauge_any_compared_value": 1745.2}},
    "sets": sets,
    "stress": {"file_sha256": full["stress_sha256"], "items": stress_sum["items"],
               "corpus_code_points_covered": stress_sum["corpus_code_points"], "tie_families": stress_sum["tie_families"],
               "categories": stress,
               "u2581_note": "literal U+2581 decodes to a space in ALL encoders incl. the canonical one (SentencePiece "
                             "escaping); U+2581 does not occur in the corpus"},
    "lm_training_view_train_D1": {k: v for k, v in json.load(open(os.path.join(REL, "lm_view_check.json"),
                                  encoding="utf-8")).items() if k != "changed_docs"},
    "transformers": full["transformers"],
    "versions": full["versions"],
    "files": full["files"],
    "canonical_encoder": full["canonical_identity"],
}
out["verdict"]["u2581_in_corpus"] = "U+2581" in stress_sum["corpus_code_points_list"]
with open(os.path.join(REL, "equivalence.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out["verdict"], ensure_ascii=False, indent=1))
for k, v in sets.items():
    print(k, {x: v[x] for x in ("n", "hf_tokenizer_json_ids_equal", "transformers_ids_equal", "hf_decode_exact",
                                "changed_vs_old_model", "lines", "lines_within_float32_bound")})
