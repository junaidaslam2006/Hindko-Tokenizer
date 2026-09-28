# -*- coding: utf-8 -*-
"""Report-only extras for MINGRAM.md (no selection, no dev decisions):
  1. tie-rule training sensitivity: run A (tie=hf, the candidate) vs run C (tie=paper, "longest leading tokens")
  2. vocabulary profile of the candidate: learned-token lengths in characters, word-initial (leading-space) share,
     whitespace pieces, digits/Latin pieces
  3. overlap of the learned vocabulary with the BPE seed's first merges and with the local char-BPE reference
Writes results/extras.json.
"""
import collections
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mingram as M  # noqa: E402

RUNS = os.path.join(HERE, "runs")


def load(run):
    return json.load(open(os.path.join(RUNS, run, "mingram_model.json"), encoding="utf-8"))


def main():
    out = {}
    a = load("mingram_P1_D1_16384")
    la = {p["piece"]: p for p in a["pieces"] if p["kind"] == "learned"}
    c_path = os.path.join(RUNS, "mingram_P1_D1_16384_tiepaper", "mingram_model.json")
    if os.path.exists(c_path):
        c = load("mingram_P1_D1_16384_tiepaper")
        lc = {p["piece"]: p for p in c["pieces"] if p["kind"] == "learned"}
        cmap = {q["piece"]: q["score_q"] for q in c["pieces"]}
        same_scores = sum(1 for p in a["pieces"] if cmap.get(p["piece"]) == p["score_q"])
        out["tie_rule_training_sensitivity"] = {
            "what": "run A (tie=hf; the candidate) vs run C (tie=paper) trained on identical inputs",
            "learned_identical_set": set(la) == set(lc), "learned_only_in_A": len(set(la) - set(lc)),
            "learned_only_in_C": len(set(lc) - set(la)), "pieces_with_identical_score": same_scores,
            "pieces": len(a["pieces"]), "same_id_order": [p["piece"] for p in a["pieces"]] == [q["piece"] for q in c["pieces"]],
            "tokenizer_json_sha256_A": M.sha256_file(os.path.join(RUNS, "mingram_P1_D1_16384", "tokenizer.json")),
            "tokenizer_json_sha256_C": M.sha256_file(os.path.join(RUNS, "mingram_P1_D1_16384_tiepaper", "tokenizer.json"))}
    lens = collections.Counter()
    lead = ws = dig = lat = 0
    for s in la:
        core = s[1:] if s.startswith(" ") else s
        lens[len(s)] += 1
        lead += s.startswith(" ")
        ws += s.isspace()
        dig += any(unicodedata.category(ch) == "Nd" for ch in s)
        lat += any("LATIN" in unicodedata.name(ch, "") for ch in s)
    counts = sorted(p["count_em"] for p in la.values())
    out["vocab_profile"] = {"learned": len(la), "length_chars_hist": dict(sorted(lens.items())),
                            "mean_length_chars": sum(k * v for k, v in lens.items()) / len(la),
                            "word_initial_leading_space": lead, "whitespace_only": ws, "with_digit": dig,
                            "with_latin_letter": lat, "final_em_count_min": counts[0],
                            "final_em_count_median": counts[len(counts) // 2]}
    seed = json.load(open(os.path.join(RUNS, "mingram_P1_D1_16384", "seed_bpe.json"), encoding="utf-8"))
    merges = seed["model"]["merges"]
    first = {("".join(x) if not isinstance(x, str) else x.replace(" ", "", 1)) for x in merges[:len(la)]}
    ranks = sorted(p["bpe_rank"] for p in la.values())
    out["vs_bpe_seed"] = {"learned_in_first_n_seed_merges": len(set(la) & first), "n": len(la),
                          "kept_with_merge_rank_ge_n": sum(1 for r in ranks if r >= len(la)),
                          "max_kept_merge_rank": ranks[-1]}
    lr = os.path.join(RUNS, "localref_charbpe_P1_D1_16384", "tokenizer.json")
    if os.path.exists(lr):
        j = json.load(open(lr, encoding="utf-8"))
        bl = {s for s in j["model"]["vocab"] if len(s) > 1 and s not in M.SPECIALS and s not in M.BYTE_PIECES}
        out["vs_local_charbpe_16k"] = {"bpe_learned": len(bl), "mingram_learned": len(la), "shared": len(bl & set(la)),
                                       "jaccard": len(bl & set(la)) / len(bl | set(la))}
    json.dump(out, open(os.path.join(HERE, "results", "extras.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
