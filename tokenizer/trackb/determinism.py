# -*- coding: utf-8 -*-
"""G4 (determinism) and the prefix property for the Track B extensions.

  G4      rebuild the K_MAX extension from scratch into work/<base>__rerun and compare the ordered merge list
          and the materialised K_MAX tokenizer.json (sha256) with the originals.
  prefix  (first base only, --prefix) train the PUA BPE directly for 1,024 new tokens and check that its merges are
          exactly the first merges of the K_MAX run, i.e. that every k of the sweep equals an independent
          training at that k.
  gemma   (--gemma-cross) Gemma 3 and Gemma 4 share their pieces and merges: their new tokens (strings, order)
          must be identical; only the ids differ (Gemma 4 has no extra image token after its vocabulary).

    python determinism.py --bases qwen-3 gemma-3 [--prefix] [--gemma-cross]  -> results/determinism_<bases>.json
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402
import continued_bpe as CB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=[])
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--gemma-cross", action="store_true")
    a = ap.parse_args()
    # one file per invocation (two invocations may run concurrently); make_trackb_md.py merges determinism_*.json
    path = os.path.join(C.RESULTS, "determinism_%s.json" % "_".join(a.bases or ["none"]))
    out = C.load_json(path) if os.path.exists(path) else {}
    for base in a.bases:
        orig = C.load_json(os.path.join(C.WORK, base, "extension.json"))
        rer = CB.build(base, tag="rerun", force=True)
        same_merges = [(m["left"], m["right"], m["id"]) for m in orig["merges"]] == \
                      [(m["left"], m["right"], m["id"]) for m in rer["merges"]]
        p1 = os.path.join(C.SWEEP, base, "k%d" % C.K_MAX, "tokenizer.json")
        p2 = CB.materialize(base, C.K_MAX, out_dir=os.path.join(C.WORK, base + "__rerun", "k%d" % C.K_MAX), ext=rer)
        s1, s2 = C.sha256_file(p1), C.sha256_file(p2)
        out.setdefault("G4", {})[base] = {"pass": bool(same_merges and s1 == s2), "same_merge_list": same_merges,
                                          "tokenizer_sha256_original": s1, "tokenizer_sha256_rerun": s2,
                                          "unit_stats_equal": {k: v for k, v in orig["unit_stats"].items() if k != "seconds"}
                                          == {k: v for k, v in rer["unit_stats"].items() if k != "seconds"}}
        C.log(base, "G4", out["G4"][base])
        shutil.rmtree(os.path.join(C.WORK, base + "__rerun", "k%d" % C.K_MAX), ignore_errors=True)
        C.dump_json(out, path)
    if a.prefix:
        base = (a.bases or ["qwen-3"])[0]
        orig = C.load_json(os.path.join(C.WORK, base, "extension.json"))
        import collections
        units = collections.Counter()
        with open(os.path.join(C.WORK, base, "units.tsv"), encoding="utf-8") as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                units[tuple(int(x) for x in s.split())] = int(c)
        merges, from_pua, tstats = CB.train_pua(units, int(1024 * CB.MARGIN))
        small, xst, _ = CB.translate(base, merges, from_pua, 1024)
        want = CB.select(orig, 1024)
        ok = [(m["left"], m["right"]) for m in small] == [(m["left"], m["right"]) for m in want]
        out["prefix"] = {"base": base, "k": 1024, "pass": ok, "direct_training_merges": len(small),
                         "prefix_merges": len(want), "train_stats": tstats}
        C.log("prefix", out["prefix"])
        C.dump_json(out, path)
    if a.gemma_cross:
        g3 = C.load_json(os.path.join(C.WORK, "gemma-3", "extension.json"))
        g4 = C.load_json(os.path.join(C.WORK, "gemma-4", "extension.json"))
        s3 = [(m["left"], m["right"], m["result"]) for m in g3["merges"]]
        s4 = [(m["left"], m["right"], m["result"]) for m in g4["merges"]]
        off = {m4["id"] - m3["id"] for m3, m4 in zip(g3["merges"], g4["merges"])}
        out["gemma3_vs_gemma4"] = {"identical_new_token_strings_and_order": s3 == s4, "id_offsets": sorted(off),
                                   "first_new_id": {"gemma-3": g3["first_new_id"], "gemma-4": g4["first_new_id"]}}
        C.log("gemma cross", out["gemma3_vs_gemma4"])
        C.dump_json(out, path)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
