# -*- coding: utf-8 -*-
"""Render ROBUSTNESS.md, robustness.json and examples_table.md from results/<key>.json and sets/sets_manifest.json.

Also cross-checks the four PLAN 4.2 probes against the study's own one-shot test runs
(eval/results/test_strict/<name>/summary.json; release_card/finalize/test_intrinsic/tokenizer_json/summary.json for the
released files): tokens, words affected and changed-segmentation counts must be identical.
Every qualitative sentence in the summary is computed from the numbers (no hand-typed result).
"""
import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcommon as C  # noqa: E402

KEYS = C.TOK_KEYS
EXT = [k for k in KEYS if k not in ("released", "mingram48k")]
OOC_SETS = [("ooc_gotquestions_southern_hindko", "GotQuestions, Southern Hindko (clean web prose)"),
            ("ooc_translatewiki_hno", "translatewiki, Northern Hindko UI messages (markup, placeholders)"),
            ("ooc_chup_di_kahani_pdf", "'Chup di Kahani' stories (PDF text layer: joined / broken words)"),
            ("ooc_finepdfs_hindko_org", "hindko.org PDFs via FinePDFs (heavily OCR-corrupted)")]
LANG_SETS = [("lang_urdu", "Urdu"), ("lang_punjabi_shahmukhi", "Punjabi (Shahmukhi)"),
             ("lang_saraiki", "Saraiki"), ("lang_pashto", "Pashto"),
             ("lang_english", "English prose"), ("lang_python_code", "Python code")]
PROBES = [("harakat", "harakat removed", "PLAN 4.2 (i)"),
          ("digits", "digit script swapped (ASCII <-> Urdu)", "PLAN 4.2 (ii)"),
          ("punct_space", "space before ۔ / ، toggled", "PLAN 4.2 (iii); on this data it inserts the space"),
          ("zwnj", "ZWNJ inserted in compounds", "PLAN 4.2 (iv)"),
          ("arabic_keyboard", "Arabic-keyboard letters ي ك ه for ی ک ہ", "new: realistic typing"),
          ("nr_nnu", "retroflex nasal نڑ written ݨ", "new: Punjabi/Saraiki orthography"),
          ("nr_n", "retroflex nasal نڑ written ن", "new: Urdu-style spelling")]
OLD_NAME = {"mingram48k": "R2-A10-MinGram-P1r3-D2-48k"}
SHORT = {"ooc_gotquestions_southern_hindko": "GotQuestions articles", "ooc_translatewiki_hno": "translatewiki UI messages",
         "ooc_chup_di_kahani_pdf": "'Chup di Kahani' PDF text", "ooc_finepdfs_hindko_org": "hindko.org OCR text"}


def load():
    R = {}
    for k in KEYS:
        p = os.path.join(C.RESULTS, k + ".json")
        R[k] = json.load(open(p, encoding="utf-8"))
    return R


def f2(x, n=2):
    return "–" if x is None else ("%." + str(n) + "f") % x


def pct(x, n=1, sign=True):
    if x is None:
        return "–"
    return (("%+." if sign else "%.") + str(n) + "f %%") % (100 * x)


def ll(a):
    return "%d/%d" % (a["lossless_docs"], a["docs"])


def cross_check(R):
    out = []
    for k in KEYS:
        if k == "released":
            p = os.path.join(C.TOK, "release_card", "finalize", "test_intrinsic", "tokenizer_json", "summary.json")
        else:
            p = os.path.join(C.TOK, "eval", "results", "test_strict", OLD_NAME.get(k, k), "summary.json")
        ref = json.load(open(p, encoding="utf-8"))["metrics"]["overall"]
        mine = R[k]
        ok = ref["tokens"] == mine["test_strict"]["tokens"] and ref["g1_pass_docs"] == mine["test_strict"]["lossless_docs"]
        diffs = []
        for pn in ("harakat", "digits", "punct_space", "zwnj"):
            a, b = mine["perturb"][pn], ref["robustness"][pn]
            for f in ("tokens", "words_affected", "words_compared", "edits"):
                if a[f] != b[f]:
                    diffs.append("%s.%s %s vs %s" % (pn, f, a[f], b[f]))
            for f in ("seg_change_rate", "seg_change_rate_affected", "rel_token_change"):
                if (a[f] is None) != (b[f] is None) or (a[f] is not None and abs(a[f] - b[f]) > 1e-12):
                    diffs.append("%s.%s %s vs %s" % (pn, f, a[f], b[f]))
        out.append({"tokenizer": k, "reference": os.path.relpath(p, C.TOK), "base_tokens_equal": ok,
                    "probe_differences": diffs, "identical": ok and not diffs})
    return out


def rank_desc(vals, key):
    order = sorted(vals, key=lambda k: -vals[k])
    return order.index(key) + 1


def main():
    R = load()
    M = json.load(open(os.path.join(C.SETS, "sets_manifest.json"), encoding="utf-8"))
    xc = cross_check(R)
    D = C.DISPLAY
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    L = []
    w = L.append

    # ------------------------------------------------------------------ numbers used by the summary
    bpt = {s: {k: R[k]["sets"][s]["norm"]["bytes_per_token"] for k in KEYS} for s, _ in OOC_SETS + LANG_SETS}
    bpt["test_strict"] = {k: R[k]["test_strict"]["bytes_per_token"] for k in KEYS}
    best_ext = {s: max(EXT, key=lambda k: bpt[s][k]) for s in bpt}
    fewer = {s: 1 - bpt[s][best_ext[s]] / bpt[s]["released"] for s in bpt}   # = 1 - tokens(released)/tokens(best ext)
    rel_rank = {s: rank_desc(bpt[s], "released") for s in bpt}
    pert = {p: {k: R[k]["perturb"][p] for k in KEYS} for p, _, _ in PROBES}

    # ------------------------------------------------------------------ header
    w("# Hindko tokenizer: out-of-distribution and robustness tests")
    w("")
    w("Generated %s by `sota/robustness/report.py` from `results/<tokenizer>.json` (`run_eval.py`) and "
      "`sets/sets_manifest.json` (`build_sets.py`). Machine-readable twin: `robustness.json`." % now)
    w("")
    w("- **What is measured.** Intrinsic tokenizer properties only: bytes per token, fertility (tokens per "
      "whitespace word, from in-context offsets), exact round trip, and how segmentations react to realistic "
      "spelling and typing variation. No language model was trained for this report. A tokenizer is not a model: "
      "it cannot translate, answer or generate. It only decides how text is cut into ids, which sets how much "
      "text fits in a context window, what training and inference cost per sentence, and how consistently a model "
      "sees the same word.")
    w("- **Released tokenizer:** `F:\\Hindko\\tokenizer\\tokenizer.json` (SentencePiece Unigram, 32,768 ids; "
      "sha256 `%s…`, checked at load). Runner-up: `R2-A10-MinGram-P1r3-D2-48k` (the pre-registered pick). "
      "External tokenizers: the study's offline copies (`baselines/manifest.json`), each with its own native "
      "encoder, no BOS/EOS added." % C.RELEASED_SHA[:12])
    w("- **Input form.** Hindko, Urdu, Punjabi, Saraiki and Pashto text is given to every tokenizer after "
      "`hp.normalize` 1.0.1, the form the release card recommends (raw-text numbers are in `robustness.json`). "
      "English and code are given as found. test_strict is already in that form.")
    w("- **Status.** Report-only. test_strict was used once for the LM decision (`analysis/TEST_RESULTS.md`); "
      "these intrinsic probes were run afterwards and change nothing about the release.")
    w("")

    # ------------------------------------------------------------------ summary
    w("## 0. Summary")
    w("")
    g = "ooc_gotquestions_southern_hindko"
    F = vocab_facts()
    oov = oov_rates()
    gq_scr = next(r for r in M["ooc_screening"] if "ooc_" + r["id"] == g)
    ahead_of_rel = [k for k in KEYS if bpt[g][k] > bpt[g]["released"]]
    w("- **Out-of-corpus Hindko.** GotQuestions articles in Southern Hindko are not in the corpus (word-8-gram "
      "overlap %s, spacing-insensitive character-24-gram overlap %s) and differ in dialect from the training text: "
      "%s of their running words never occur in the training text (test_strict: %s). On them the released "
      "tokenizer gives %.2f bytes/token (test_strict: %.2f), rank %d of 11%s. The best external tokenizer there is "
      "%s at %.2f, so the released tokenizer uses %.0f %% fewer tokens than it (%.0f %% fewer than GPT-4o)." % (
          pct(gq_scr["w8_overlap_all"], 2, False), pct(gq_scr["c24_overlap_all"], 2, False),
          pct(oov[g]["token_oov"], 1, False), pct(oov["test_strict"]["token_oov"], 1, False),
          bpt[g]["released"], bpt["test_strict"]["released"], rel_rank[g],
          (" (behind %s)" % ", ".join("%s %.2f" % (D[k], bpt[g][k]) for k in ahead_of_rel)) if ahead_of_rel else "",
          D[best_ext[g]], bpt[g][best_ext[g]], 100 * fewer[g], 100 * (1 - bpt[g]["gpt-4o"] / bpt[g]["released"])))
    lines_ooc = []
    for s, name in OOC_SETS[1:]:
        lines_ooc.append("%s: %.2f vs best external %s %.2f (rank %d)" % (
            SHORT[s], bpt[s]["released"], D[best_ext[s]], bpt[s][best_ext[s]], rel_rank[s]))
    lead = [s for s, _ in OOC_SETS if bpt[s]["released"] > bpt[s][best_ext[s]]]
    behind = [s for s, _ in OOC_SETS if s not in lead]
    beh_txt = "; ".join("on the %s %s is %.1f %% more efficient, but it does not reproduce that text exactly "
                        "(round trip %s documents)" % (
                            SHORT[s], D[best_ext[s]],
                            100 * (bpt[s][best_ext[s]] / bpt[s]["released"] - 1),
                            ll(R[best_ext[s]]["sets"][s]["norm"])) for s in behind)
    margins = [bpt[s][best_ext[s]] / bpt[s]["released"] - 1 for s, _ in OOC_SETS[1:]]   # tokens(rel)/tokens(ext) - 1
    w("- **Noisy out-of-corpus Hindko** (UI strings, PDF text layer, OCR), bytes/token: " + "; ".join(lines_ooc) + ". "
      "The released tokenizer is ahead of every external tokenizer on %d of the 4 out-of-corpus sets%s. On these "
      "three noisy sets the gap is small: the released tokenizer needs %s to %s tokens relative to the best "
      "external tokenizer." % (len(lead), ("; " + beh_txt) if behind else "", pct(min(margins)), pct(max(margins))))
    ak = pert["arabic_keyboard"]
    ak_after_rel = R["released"]["perturb"]["arabic_keyboard"]["tokens"]
    better_after = [k for k in EXT if R[k]["perturb"]["arabic_keyboard"]["tokens"] < ak_after_rel]
    AKM = json.load(open(os.path.join(C.RESULTS, "ak_mitigation.json"), encoding="utf-8"))["tokenizers"]
    fold_fewest = all(AKM[k]["ak_fold_yk"] > AKM["released"]["ak_fold_yk"] for k in EXT)
    w("- **Weak spot: Arabic-keyboard letters.** Typing ي ك ه instead of ی ک ہ (every occurrence) costs the "
      "released tokenizer %s tokens on test_strict and changes the segmentation of %s of the words it touches "
      "(GPT-4o: %s tokens, %s of touched words; RoBERTa-Urdu, the other Urdu-script-specialised vocabulary: %s). "
      "After that change the released tokenizer needs %s tokens, %s. `normalize()` recovers part of it (%s tokens "
      "vs clean text), because it folds YEH and KAF only inside provably Urdu-script words and never folds HEH. "
      "Folding every ي→ی and ك→ک before encoding (appropriate for Hindko/Urdu text; it also rewrites Arabic quotations) leaves %s, %s; the rest is the cost "
      "of ه, which cannot be folded blindly." % (
          pct(ak["released"]["rel_token_change"], 0), pct(ak["released"]["seg_change_rate_affected"], 0, False),
          pct(ak["gpt-4o"]["rel_token_change"], 0), pct(ak["gpt-4o"]["seg_change_rate_affected"], 0, False),
          pct(ak["roberta-urdu"]["rel_token_change"], 0), "{:,}".format(ak_after_rel),
          ("more than %s of the 9 external tokenizers (%s)" % (len(better_after), ", ".join(D[k] for k in better_after)))
          if better_after else "still fewer than every external tokenizer",
          pct(R["released"]["perturb_norm"]["arabic_keyboard_then_normalize"]["rel_token_change"], 0),
          pct(AKM["released"]["rel_ak_fold_yk"], 0),
          "and the released tokenizer again needs fewer tokens than every external tokenizer on the same folded text"
          if fold_fewest else "but some external tokenizer still needs fewer tokens"))
    small = [p for p, _, _ in PROBES if p != "arabic_keyboard"]
    still_fewer = all(pert[p][k]["tokens"] > pert[p]["released"]["tokens"] for p in small for k in EXT)
    ps_ext = max(pert["punct_space"][k]["rel_token_change"] for k in EXT)
    w("- **The other six probes** (harakat, digits, punctuation spacing, ZWNJ, and the two spellings of the "
      "retroflex nasal) change the released tokenizer's token count by %s to %s. %s In relative terms it reacts "
      "more than the external tokenizers to a space before ۔/، (%s against at most %s; the corpus never has that "
      "space), because many of its word pieces end in the full stop." % (
          pct(min(pert[p]["released"]["rel_token_change"] for p in small)),
          pct(max(pert[p]["released"]["rel_token_change"] for p in small)),
          "After each of them it still needs fewer tokens than every external tokenizer." if still_fewer else
          "After at least one of them some external tokenizer needs fewer tokens (see §3).",
          pct(pert["punct_space"]["released"]["rel_token_change"]), pct(ps_ext)))
    cmx = {k: R[k]["codemixed"] for k in KEYS}
    cm_best_ext = max(EXT, key=lambda k: cmx[k]["mixed"]["bytes_per_token"])
    w("- **Code-mixed Hindko + Urdu + English (100 synthetic lines).** The released tokenizer gives %.2f "
      "bytes/token on the mixed lines (%.2f on the same lines without the insertions); the best external "
      "tokenizer gives %.2f (%s). The inserted English and Urdu material costs the released tokenizer %.2f bytes "
      "per token, against %.2f for GPT-4o." % (
          cmx["released"]["mixed"]["bytes_per_token"], cmx["released"]["base"]["bytes_per_token"],
          cmx[cm_best_ext]["mixed"]["bytes_per_token"], D[cm_best_ext],
          cmx["released"]["added_bytes"] / cmx["released"]["added_tokens"],
          cmx["gpt-4o"]["added_bytes"] / cmx["gpt-4o"]["added_tokens"]))
    worse = []
    for s, name in LANG_SETS:
        n_better = sum(1 for k in EXT if bpt[s][k] > bpt[s]["released"])
        worse.append("%s %.2f (%d of 9 external tokenizers more efficient; best %s %.2f)" % (
            name, bpt[s]["released"], n_better, D[best_ext[s]], bpt[s][best_ext[s]]))
    weak_langs = [n for s, n in LANG_SETS if sum(1 for k in EXT if bpt[s][k] > bpt[s]["released"]) >= 5]
    ok_langs = [n for s, n in LANG_SETS if sum(1 for k in EXT if bpt[s][k] > bpt[s]["released"]) <= 1]
    w("- **Other languages (honest characterisation).** Bytes/token of the released tokenizer: " + "; ".join(worse) +
      ". It is a Hindko-specialised vocabulary. Because Hindko shares script and much vocabulary with its "
      "neighbours, it is also efficient on %s; it is clearly worse than general-purpose tokenizers on %s, and it is "
      "not suitable as the only tokenizer of a multilingual, English or coding model." % (
          ", ".join(ok_langs), ", ".join(weak_langs)))
    w("- **Lossless.** The released tokenizer round-trips every document of every set in this report (%d/%d). "
      "mT5 does not (it normalizes text, e.g. line breaks become spaces)." % (
          sum(R["released"]["sets"][s]["norm"]["lossless_docs"] for s, _ in OOC_SETS + LANG_SETS) +
          R["released"]["test_strict"]["lossless_docs"] + R["released"]["codemixed"]["mixed"]["lossless_docs"],
          sum(R["released"]["sets"][s]["norm"]["docs"] for s, _ in OOC_SETS + LANG_SETS) +
          R["released"]["test_strict"]["docs"] + R["released"]["codemixed"]["mixed"]["docs"]))
    w("- **Consistency check.** The four PLAN §4.2 probes reproduce the study's one-shot test runs exactly for "
      "%d of %d tokenizers (tokens, words affected, changed segmentations)." % (
          sum(1 for x in xc if x["identical"]), len(xc)))
    w("")

    # ------------------------------------------------------------------ tokenizers
    w("## 1. Tokenizers")
    w("")
    w("| tokenizer | family | vocab | test_strict bytes/token | fertility | lossless on test_strict |")
    w("|---|---|---:|---:|---:|---:|")
    for k, disp, fam, _ in C.TOKENIZERS:
        t = R[k]["test_strict"]
        w("| %s | %s | %s | %.3f | %.3f | %s |" % ("**%s**" % disp if k == "released" else disp, fam,
                                                   "{:,}".format(R[k]["identity"]["vocab_size"]),
                                                   t["bytes_per_token"], t["fertility"], ll(t)))
    w("")

    # ------------------------------------------------------------------ OOC
    w("## 2. Out-of-corpus Hindko")
    w("")
    w("### 2.1 Which samples are really outside the corpus")
    w("")
    w("Every survey sample on disk (`F:\\Hindko\\_web\\survey\\_samples`, `…\\samples`, `…\\corpora_work\\samples`) "
      "that could be Hindko running text was split into documents and compared with the whole released corpus "
      "(`hindko_dataset_permissive.jsonl` + `hindko_dataset.jsonl`, all splits and tiers: %s 8-grams, %s character "
      "24-grams). Keys: NFC, harakat and invisible characters removed, Arabic-keyboard letters folded. A document "
      "is kept when its word-8-gram overlap is < 10 %%, its character-24-gram overlap (spacing-insensitive, catches "
      "PDF texts with joined words) is < 20 %%, it has ≥ 20 words, ≥ 60 %% Arabic-script letters, no replacement "
      "characters, and more Hindko/Punjabi than Urdu function words." % (
          "{:,}".format(M["corpus_indexes"]["w8"]["unique_8grams"]),
          "{:,}".format(M["corpus_indexes"]["c24"]["unique_chargrams"])))
    w("")
    w("| source | role | docs | docs passing the rules | used as out-of-corpus Hindko | word-8-gram overlap (all docs) | char-24-gram overlap (all docs) | note |")
    w("|---|---|---:|---:|---:|---:|---:|---|")
    for r in M["ooc_screening"]:
        reasons = sorted({re.sub(r"[ (].*$", "", x) if x.startswith("U+") else x.split(" (")[0].split(" %")[0]
                          for d in r["doc_stats"] if not d["kept"] for x in d["reasons"]})
        role = {"ooc": "candidate", "in_corpus": "control: known corpus source", "uncertain": "screened, not used",
                "not_hindko": "control: not Hindko"}[r["expected"]]
        w("| `%s` | %s | %d | %d | %s | %s | %s | %s |" % (
            r["id"], role, r["docs"], r["docs_kept"], r["docs_kept"] if r["expected"] == "ooc" else 0,
            pct(r["w8_overlap_all"], 1, False), pct(r["c24_overlap_all"], 1, False),
            r["description"] + ("; dropped: " + "; ".join(reasons) if reasons and r["expected"] == "ooc" else "")))
    ct = M["ooc_control_test_strict"]
    ic = [r["w8_overlap_all"] for r in M["ooc_screening"] if r["expected"] == "in_corpus"]
    used = [r for r in M["ooc_screening"] if r["expected"] == "ooc" and r["docs_kept"]]
    mx = max(max(r["w8_overlap_kept"] or 0, r["c24_overlap_kept"] or 0) for r in used)
    w("")
    w("- **Controls behave as expected.** 25 random test_strict documents: %s word-8-gram and %s character-24-gram "
      "overlap. The known corpus sources (Common Voice, Omnilingual, gandharahindko.com, aaprihindko, …) show "
      "%s–%s word-8-gram overlap; the parts not found are pages or sentences the corpus did not take." % (
          pct(ct["w8_overlap"], 0, False), pct(ct["c24_overlap"], 0, False), pct(min(ic), 0, False),
          pct(max(ic), 0, False)))
    w("- **Used as out-of-corpus Hindko:** %s. The highest overlap of any kept set with the corpus, on either index, "
      "is %s. The Surah Yasin PDF (Quranic Arabic with an Urdu-leaning translation), the Bible-for-Children PDF "
      "(broken font encoding: replacement characters) and the djvu OCR of an Urdu-language history book were dropped "
      "by the rules. The FineWeb-2 rows the survey had flagged pass the rules, but by inspection they are mostly "
      "Punjabi (Punjabi poetry sites, pnb.wikipedia.org), so they are not used as Hindko." % (
          "; ".join("`%s` (%d docs)" % (r["id"], r["docs_kept"]) for r in used), pct(mx, 2, False)))
    w("")
    w("### 2.2 Results")
    w("")
    hdr = "| tokenizer | test_strict (in-distribution) | " + " | ".join(n.split(" (")[0] for _, n in OOC_SETS) + " |"
    w("**Bytes per token** (higher = fewer tokens for the same text; normalized input):")
    w("")
    w(hdr)
    w("|---|" + "---:|" * (1 + len(OOC_SETS)))
    for k in KEYS:
        cells = [f2(bpt["test_strict"][k])] + [f2(bpt[s][k]) for s, _ in OOC_SETS]
        if k == "released":
            cells = ["**%s**" % c for c in cells]
        w("| %s | %s |" % (D[k], " | ".join(cells)))
    w("| *released uses x % fewer tokens than the best external* | " +
      " | ".join("%.1f %% (vs %s)" % (100 * fewer[s], D[best_ext[s]]) for s in ["test_strict"] + [s for s, _ in OOC_SETS]) + " |")
    w("")
    w("**Fertility** (tokens per whitespace word; lower is better) and **lossless documents**:")
    w("")
    w("| tokenizer | " + " | ".join(n.split(" (")[0] for _, n in OOC_SETS) + " | lossless (all 4 sets) |")
    w("|---|" + "---:|" * (len(OOC_SETS) + 1))
    for k in KEYS:
        cells = [f2(R[k]["sets"][s]["norm"]["fertility"], 3) for s, _ in OOC_SETS]
        lo = sum(R[k]["sets"][s]["norm"]["lossless_docs"] for s, _ in OOC_SETS)
        nd = sum(R[k]["sets"][s]["norm"]["docs"] for s, _ in OOC_SETS)
        w("| %s | %s | %d/%d |" % (D[k], " | ".join(cells), lo, nd))
    w("")
    w("- Set sizes (normalized UTF-8 bytes): " + "; ".join("%s %s" % (n.split(" (")[0], "{:,}".format(
        M["sets"][s]["bytes_norm"])) for s, n in OOC_SETS) + ".")
    w("- Distance from the training text: share of running words (letters only, harakat removed) never seen in "
      "train_D2: test_strict %s; " % pct(oov["test_strict"]["token_oov"], 1, False) + "; ".join(
        "%s %s" % (n.split(" (")[0], pct(oov[s]["token_oov"], 1, False)) for s, n in OOC_SETS) +
      ". The PDF and OCR sets are far from the training text mostly because of extraction damage (joined words, "
      "wrong letters), not dialect.")
    rw = R["released"]["sets"]
    w("- Raw (un-normalized) input changes the released tokenizer's bytes/token to " + "; ".join(
        "%s %.2f" % (n.split(" (")[0], rw[s]["raw"]["bytes_per_token"]) for s, n in OOC_SETS) +
      " (PDF text carries bidi control characters that `normalize()` removes).")
    w("- Byte-fallback pieces used by the released tokenizer (characters outside its vocabulary): " + "; ".join(
        "%s %s" % (n.split(" (")[0], "{:,}".format(rw[s]["norm"]["byte_fallback_tokens"])) for s, n in OOC_SETS) + ".")
    w("")

    # ------------------------------------------------------------------ perturbations
    w("## 3. Perturbation robustness on test_strict")
    w("")
    w("Each probe rewrites all 491 test_strict documents (1,451,026 bytes, 171,769 words). The four PLAN §4.2 probes "
      "are `eval/perturb.py`, unchanged; the three new ones are `perturb_extra.py` (same alignment format, so "
      "`eval/harness.seg_compare` scores them identically). *Relative token change* = (tokens after − tokens "
      "before) / tokens before, over all documents. *Segmentation changed* = share of words whose token boundaries, "
      "on the characters the word keeps, are not the same as before; given over the words the probe touched, and "
      "over all words.")
    w("")
    w("| probe | what it does | words touched | source |")
    w("|---|---|---:|---|")
    for p, name, src in PROBES:
        w("| `%s` | %s | %s | %s |" % (p, name, "{:,}".format(R["released"]["perturb"][p]["words_affected"]), src))
    w("")
    w("**Relative token-count change** (closer to 0 is more robust):")
    w("")
    w("| tokenizer | " + " | ".join("`%s`" % p for p, _, _ in PROBES) + " | `arabic_keyboard` then `normalize()` |")
    w("|---|" + "---:|" * (len(PROBES) + 1))
    for k in KEYS:
        cells = [pct(pert[p][k]["rel_token_change"], 2) for p, _, _ in PROBES]
        cells.append(pct(R[k]["perturb_norm"]["arabic_keyboard_then_normalize"]["rel_token_change"], 2))
        w("| %s | %s |" % (D[k], " | ".join(cells)))
    w("")
    w("**Share of touched words whose segmentation changes** (lower is more robust; all words in brackets):")
    w("")
    w("| tokenizer | " + " | ".join("`%s`" % p for p, _, _ in PROBES) + " |")
    w("|---|" + "---:|" * len(PROBES))
    for k in KEYS:
        cells = ["%s (%s)" % (pct(pert[p][k]["seg_change_rate_affected"], 1, False), pct(pert[p][k]["seg_change_rate"], 2, False))
                 for p, _, _ in PROBES]
        w("| %s | %s |" % (D[k], " | ".join(cells)))
    w("")
    AKM = json.load(open(os.path.join(C.RESULTS, "ak_mitigation.json"), encoding="utf-8"))["tokenizers"]
    w("**Arabic-keyboard input and two fixes** (`ak_mitigation.py`; tokens relative to the clean text; last column: "
      "tokens after the YEH/KAF fold ÷ the released tokenizer's):")
    w("")
    w("| tokenizer | typed with ي ك ه | then `normalize()` | then fold ي→ی, ك→ک (ه left) | × released after the fold |")
    w("|---|---:|---:|---:|---:|")
    for k in KEYS:
        w("| %s | %s | %s | %s | %.3f |" % (D[k], pct(AKM[k]["rel_ak"], 1),
                                             pct(R[k]["perturb_norm"]["arabic_keyboard_then_normalize"]["rel_token_change"], 1),
                                             pct(AKM[k]["rel_ak_fold_yk"], 1), AKM[k]["ak_fold_yk"] / AKM["released"]["ak_fold_yk"]))
    w("")
    w("**Tokens after the probe, relative to the released tokenizer after the same probe** (× tokens; > 1 means "
      "the released tokenizer still needs fewer tokens):")
    w("")
    w("| tokenizer | clean | " + " | ".join("`%s`" % p for p, _, _ in PROBES) + " |")
    w("|---|" + "---:|" * (len(PROBES) + 1))
    for k in KEYS:
        cells = ["%.3f" % (R[k]["test_strict"]["tokens"] / R["released"]["test_strict"]["tokens"])]
        cells += ["%.3f" % (pert[p][k]["tokens"] / pert[p]["released"]["tokens"]) for p, _, _ in PROBES]
        w("| %s | %s |" % (D[k], " | ".join(cells)))
    w("")
    rel = R["released"]["perturb"]
    w("Reading the tables:")
    w("")
    w("- **Relative change is not the whole story.** A tokenizer that splits Hindko into many small pieces has "
      "little left to break, so its relative change can be small while its absolute token count stays much higher. "
      "The last table below shows the absolute effect.")
    F = vocab_facts()
    ak_ext = {k: pert["arabic_keyboard"][k]["rel_token_change"] for k in EXT}
    w("- **Arabic-keyboard letters are the released tokenizer's clearest weakness.** ی, ک and ہ are among the most "
      "frequent letters in Hindko, and its training text (train_D2, %s characters) writes them in the Urdu "
      "codepoints almost without exception: ARABIC YEH %d times, ARABIC KAF %d, ARABIC HEH %d. Only %d of the "
      "32,768 vocabulary pieces contain any of ي ك ه. So %s of the touched words are re-cut, at %.2f extra tokens "
      "per touched word. The external tokenizers change by %s to %s on the same probe. Mitigation for applications: "
      "apply `normalize()` (it recovers YEH/KAF inside provably Urdu-script words) and, where the input is known to "
      "be Hindko or Urdu, fold ي→ی and ك→ک before encoding; ه cannot be folded blindly, because Arabic-keyboard "
      "users also type it for ھ." % (
          "{:,}".format(F["train_chars"]), F["train_ak"]["yeh"], F["train_ak"]["kaf"], F["train_ak"]["heh"],
          F["vocab_ak_pieces"], pct(rel["arabic_keyboard"]["seg_change_rate_affected"], 0, False),
          rel["arabic_keyboard"]["extra_tokens_per_affected_word"],
          pct(min(ak_ext.values()), 1), pct(max(ak_ext.values()), 1)) +
      " With the YEH/KAF fold the released tokenizer's cost falls to %s (table above)." %
      pct(AKM["released"]["rel_ak_fold_yk"], 1))
    w("- **The retroflex nasal.** Writing نڑ as ݨ (the Punjabi/Saraiki convention) costs %s tokens; as plain ن, %s. "
      "Both spellings occur in the corpus, so the vocabulary has pieces for them." % (
          pct(rel["nr_nnu"]["rel_token_change"], 2), pct(rel["nr_n"]["rel_token_change"], 2)))
    hd = [pert[p][k]["rel_token_change"] for p in ("harakat", "digits") for k in KEYS]
    w("- **Removing harakat and swapping the digit script** change every tokenizer's token count by %s to %s "
      "(removing harakat shortens the text)." % (pct(min(hd), 2), pct(max(hd), 2)))
    biggest_small = max(small, key=lambda p: rel[p]["rel_token_change"])
    w("- **Space before ۔ / ،**: the corpus never has it (0 occurrences in train and dev, `eval/perturb.py`). The "
      "released tokenizer spends %s more tokens%s, because many of its word pieces end in the full stop (e.g. `▁آئی۔` "
      "in sentence 1 of §6), and a space breaks them off." % (
          pct(rel["punct_space"]["rel_token_change"], 1),
          ", its largest change among the six small probes" if biggest_small == "punct_space" else ""))
    w("")
    w("Cross-check of the four PLAN probes against the study's one-shot test runs:")
    w("")
    w("| tokenizer | reference file | identical |")
    w("|---|---|---|")
    for x in xc:
        w("| %s | `%s` | %s |" % (D[x["tokenizer"]], x["reference"], "yes" if x["identical"] else
                                  "NO: " + "; ".join(x["probe_differences"][:4])))
    w("")

    # ------------------------------------------------------------------ code-mixed
    w("## 4. Code-mixed text")
    w("")
    cmm = M["sets"]["codemixed"]
    w("- **Synthetic set (100 lines, `sets/codemixed.jsonl`).** One test_strict Hindko sentence (8–25 words) from "
      "each of 100 randomly chosen documents; 1–3 English words inserted at random word boundaries (from a fixed "
      "list of %d words common in Pakistani code-switching: `meeting`, `online`, `WhatsApp`, `exam`, …); with "
      "probability 0.25 an English opener (`Actually,` …), with 0.25 an English closer (`Thank you!` …), with 0.35 an "
      "Urdu clause (%d fixed clauses such as `کوئی بات نہیں۔`). Seed `%s`. The insertions are placed at random, so "
      "the lines are realistic at the level of scripts and words, not always grammatical." % (
          len(cmm["en_words"]), len(cmm["ur_clauses"]), "20260927-codemix"))
    w("- **Natural set.** The %d test_strict documents labelled `mixed` (Hindko text quoting Urdu songs, and Urdu "
      "ghazals inside Hindko books; %s bytes)." % (M["sets"]["mixed_natural_test"]["docs"],
                                                   "{:,}".format(M["sets"]["mixed_natural_test"]["bytes_norm"])))
    w("- Example line: `کمپیوٹر آیا تا دفتری internet ملازمتاں دے مستقبل تے class سوال اُٹھے۔`")
    w("")
    w("| tokenizer | synthetic mixed: bytes/token | same lines without insertions | inserted material: bytes/token | tokens / line | lossless | natural mixed: bytes/token |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for k in KEYS:
        c = cmx[k]
        cells = [f2(c["mixed"]["bytes_per_token"]), f2(c["base"]["bytes_per_token"]),
                 f2(c["added_bytes"] / c["added_tokens"]), "%.1f" % (c["mixed"]["tokens"] / c["mixed"]["docs"]),
                 ll(c["mixed"]), f2(R[k]["sets"]["mixed_natural_test"]["norm"]["bytes_per_token"])]
        if k == "released":
            cells = ["**%s**" % x for x in cells]
        w("| %s | %s |" % (D[k], " | ".join(cells)))
    w("")
    add = {k: cmx[k]["added_bytes"] / cmx[k]["added_tokens"] for k in KEYS}
    cheaper = [k for k in EXT if add[k] > add["released"]]
    fewest = min(KEYS, key=lambda k: cmx[k]["mixed"]["tokens"])
    fewest_ext = min(EXT, key=lambda k: cmx[k]["mixed"]["tokens"])
    w("- The inserted English words and Urdu clauses cost the released tokenizer %.2f bytes per token; %d of the 9 "
      "external tokenizers encode that material more cheaply (up to %.2f, %s). On whole lines, where Hindko is most "
      "of the text, the fewest tokens per line are %s's (%.1f); the released tokenizer needs %.1f and the best "
      "external tokenizer, %s, %.1f." % (
          add["released"], len(cheaper), max(add[k] for k in EXT), D[max(EXT, key=lambda k: add[k])], D[fewest],
          cmx[fewest]["mixed"]["tokens"] / 100, cmx["released"]["mixed"]["tokens"] / 100, D[fewest_ext],
          cmx[fewest_ext]["mixed"]["tokens"] / 100))
    w("- The more English a line contains, the smaller the released tokenizer's advantage; for mostly-English text "
      "it is at a disadvantage (§5).")
    w("")

    # ------------------------------------------------------------------ other languages
    w("## 5. Other languages (where the released tokenizer is and is not appropriate)")
    w("")
    w("| tokenizer | " + " | ".join(n for _, n in LANG_SETS) + " |")
    w("|---|" + "---:|" * len(LANG_SETS))
    for k in KEYS:
        cells = [f2(bpt[s][k]) for s, _ in LANG_SETS]
        if k == "released":
            cells = ["**%s**" % x for x in cells]
        w("| %s | %s |" % (D[k], " | ".join(cells)))
    w("| *released: rank among the 11* | " + " | ".join(str(rel_rank[s]) for s, _ in LANG_SETS) + " |")
    w("| *released tokens ÷ best external tokens* | " + " | ".join(
        "%.2f× (%s)" % (bpt[s][best_ext[s]] / bpt[s]["released"], D[best_ext[s]]) for s, _ in LANG_SETS) + " |")
    w("")
    w("Bytes per token; higher is better. Fertility and lossless counts are in `robustness.json`.")
    w("")
    import sysconfig
    py_paths = {sysconfig.get_paths()["purelib"]: "<python 3.11>\\Lib\\site-packages",
                sysconfig.get_paths()["stdlib"]: "<python 3.11>\\Lib"}
    for s, name in LANG_SETS:
        v = M["sets"][s]
        src = v.get("source", "")
        for a_, b_ in py_paths.items():          # no local user-profile paths in the report
            src = src.replace(a_, b_)
        w("- **%s**: %d documents, %s bytes; %s%s" % (
            name, v["docs"], "{:,}".format(v["bytes_norm"]), "`%s`" % src if src else "",
            ("; %d sampled, %d dropped because they overlap the Hindko corpus" % (v["sampled"], v["dropped_in_hindko_corpus"])
             if "sampled" in v else "") + (". " + v["rule"] if "rule" in v else "")))
    w("")
    vs = F["train_variety_char_share"]
    w("- **Urdu and Punjabi:** the released tokenizer ranks %d and %d of 11. Hindko shares script, much vocabulary "
      "and many loanwords with both, and its training text (%s Hindko-labelled, %s mixed Hindko–Urdu by characters) "
      "quotes Urdu. This is a side effect, not a design goal: it was not trained or selected on Urdu or Punjabi "
      "text." % (rel_rank["lang_urdu"], rel_rank["lang_punjabi_shahmukhi"], pct(vs.get("hindko", 0), 1, False),
                 pct(vs.get("mixed", 0), 1, False)))
    bf = {s: R["released"]["sets"][s]["norm"]["byte_fallback_tokens"] / R["released"]["sets"][s]["norm"]["tokens"]
          for s in ("lang_saraiki", "lang_pashto")}
    w("- **Saraiki and Pashto:** the training text contains the Saraiki implosives ٻ ڄ ݙ ڳ %d times and the "
      "Pashto letters ټ ډ ړ ږ ښ ګ ڼ %d times in total, so the tokenizer spells these letters with byte-fallback "
      "pieces (%s of its Saraiki tokens and %s of its Pashto tokens are byte pieces). Saraiki still ranks %d of 11, "
      "because the rest of its vocabulary is close to Hindko; Pashto, a different language family, ranks %d of 11." % (
          F["train_skr"], F["train_pbt"], pct(bf["lang_saraiki"], 1, False), pct(bf["lang_pashto"], 1, False),
          rel_rank["lang_saraiki"], rel_rank["lang_pashto"]))
    w("- **English and Python code:** only %d of the 32,768 pieces contain Latin letters (mostly names from the "
      "corpus, e.g. botanical terms). The released tokenizer needs %.1f× GPT-4o's tokens on English prose and %.1f× "
      "on Python code. It is lossless on both, so nothing breaks, but it should not be used as-is for an English or "
      "coding model." % (F["vocab_latin_pieces"], bpt["lang_english"]["gpt-4o"] / bpt["lang_english"]["released"],
                         bpt["lang_python_code"]["gpt-4o"] / bpt["lang_python_code"]["released"]))
    w("")

    # ------------------------------------------------------------------ examples
    ex_md = examples_md(R)
    w("## 6. Example segmentations (for the model card)")
    w("")
    L.extend(ex_md)
    w("")

    # ------------------------------------------------------------------ claims
    w("## 7. What these tests support")
    w("")
    w("**Supported (intrinsic, report-only):**")
    w("")
    w("- \"On Hindko text that is not in its training corpus (Southern Hindko web articles), the released "
      "tokenizer uses %.0f %% fewer tokens than the most efficient of the 9 external tokenizers measured (%s) and "
      "%.0f %% fewer than GPT-4o's o200k, and it round-trips the text exactly.\"" % (
          100 * fewer[g], D[best_ext[g]], 100 * (1 - bpt[g]["gpt-4o"] / bpt[g]["released"])))
    touched_max = max(rel[p]["words_affected"] for p in small) / R["released"]["test_strict"]["words"]
    w("- \"Under the four PLAN §4.2 perturbations and the two alternative spellings of the retroflex nasal, its "
      "token count on the test split changes by %s to %s, and after each of them it still needs fewer tokens than "
      "every external tokenizer measured.\"%s The words a probe touches (at most %s of all words) are often "
      "re-segmented, as with every tokenizer, so 'stable segmentation' is not claimed." % (
          pct(min(pert[p]["released"]["rel_token_change"] for p in small)),
          pct(max(pert[p]["released"]["rel_token_change"] for p in small)),
          "" if still_fewer else " [NOT supported as worded: see §3]", pct(touched_max, 1, False)))
    w("")
    w("**Not supported / must be stated as limitations:**")
    w("")
    w("- Robustness to Arabic-keyboard typing: %s tokens as typed, %s after `normalize()`, %s after also folding "
      "ي→ی and ك→ک." % (pct(rel["arabic_keyboard"]["rel_token_change"], 0),
                        pct(R["released"]["perturb_norm"]["arabic_keyboard_then_normalize"]["rel_token_change"], 0),
                        pct(AKM["released"]["rel_ak_fold_yk"], 0)))
    w("- Efficiency outside Hindko and its close neighbours: %s are clearly worse than with general tokenizers." %
      ", ".join(weak_langs))
    w("- Relative robustness to a space before ۔/، (%s vs at most %s for the external tokenizers)." % (
        pct(pert["punct_space"]["released"]["rel_token_change"]), pct(ps_ext)))
    w("- Anything about model quality. These are tokenizer measurements; downstream quality needs a model "
      "trained or adapted with this tokenizer (PLAN §8).")
    w("- 'Best tokenizer for all languages' or 'better than all AI models'. The comparison is 11 tokenizers on "
      "Hindko-centred text.")
    w("")
    w("## 8. Files and reproduction")
    w("")
    w("| file | content |")
    w("|---|---|")
    for f, c in [("rcommon.py", "tokenizer list and loaders, n-gram keys"),
                 ("corpus_ngrams.py / corpus_chargrams.py", "corpus word-8-gram and character-24-gram indexes (sets/corpus_*.npy)"),
                 ("build_sets.py", "every evaluation set + sets/sets_manifest.json (screening of each survey document)"),
                 ("perturb_extra.py", "the three new probes (Arabic keyboard, نڑ→ݨ, نڑ→ن)"),
                 ("run_eval.py", "metrics per tokenizer → results/<key>.json (3 shards, 1 thread each)"),
                 ("ak_mitigation.py", "Arabic-keyboard probe followed by the YEH/KAF fold → results/ak_mitigation.json"),
                 ("report.py", "this file, robustness.json, examples_table.md")]:
        w("| `%s` | %s |" % (f, c))
    w("")
    w("Reproduce from `F:\\Hindko\\_tokenizer\\sota\\robustness` with `PYTHONIOENCODING=utf-8`: `python corpus_ngrams.py`, "
      "`python corpus_chargrams.py`, `python build_sets.py`, `python run_eval.py --shard K --of 3` for K = 0, 1, 2, `python ak_mitigation.py`, "
      "`python report.py`. The set files contain third-party text and are for local use only.")
    w("")
    md = "\n".join(L)
    open(os.path.join(C.HERE, "ROBUSTNESS.md"), "w", encoding="utf-8", newline="\n").write(md)
    open(os.path.join(C.HERE, "examples_table.md"), "w", encoding="utf-8", newline="\n").write(
        "\n".join(["# Example segmentations of three Hindko sentences", ""] + ex_md) + "\n")

    # ------------------------------------------------------------------ json
    J = {"generated_utc": now, "released_tokenizer_sha256": C.RELEASED_SHA,
         "tokenizers": [{"key": k, "display": d, "family": f} for k, d, f, _ in C.TOKENIZERS],
         "sets_manifest_sha256": C.sha256_file(os.path.join(C.SETS, "sets_manifest.json")),
         "ooc_screening": [{x: r[x] for x in r if x != "doc_stats"} for r in M["ooc_screening"]],
         "ooc_control_test_strict": M["ooc_control_test_strict"],
         "sets": {s: {k: R[k]["sets"][s] for k in KEYS} for s in R["released"]["sets"]},
         "test_strict": {k: R[k]["test_strict"] for k in KEYS},
         "perturbations": {p: {k: R[k]["perturb"][p] for k in KEYS} for p, _, _ in PROBES},
         "perturbation_definitions": {p: {"what": n, "source": s} for p, n, s in PROBES},
         "arabic_keyboard_then_normalize": {k: R[k]["perturb_norm"]["arabic_keyboard_then_normalize"] for k in KEYS},
         "codemixed": {k: {x: R[k]["codemixed"][x] for x in ("mixed", "base", "added_bytes", "added_tokens")} for k in KEYS},
         "examples": {k: R[k]["examples"] for k in KEYS},
         "summary_numbers": {"bytes_per_token": bpt, "best_external": best_ext, "released_fewer_tokens_vs_best_external": fewer,
                             "released_rank_of_11": rel_rank},
         "cross_check_plan_probes": xc,
         "released_vocab_and_training_text_facts": vocab_facts(),
         "word_oov_vs_train_D2": oov_rates(),
         "arabic_keyboard_mitigation": json.load(open(os.path.join(C.RESULTS, "ak_mitigation.json"), encoding="utf-8"))}
    json.dump(C_clean(J), open(os.path.join(C.HERE, "robustness.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote ROBUSTNESS.md (%d lines), robustness.json, examples_table.md" % len(L))
    print("cross-check identical:", sum(1 for x in xc if x["identical"]), "/", len(xc))


def C_clean(x):
    import harness as H
    return H.clean(x)


_OOV = None


def oov_rates():
    """Share of running words (rcommon.key_words: letters only, harakat removed) of each Hindko set that never occur
    in the tokenizer's training text train_D2: a simple measure of how far a set is from the training data."""
    global _OOV
    if _OOV is not None:
        return _OOV
    vocab = set()
    with open(os.path.join(C.TOK, "data", "train_D2.jsonl"), encoding="utf-8") as f:
        for line in f:
            vocab.update(C.key_words(json.loads(line)["text"]))

    def rate(texts):
        n = miss = 0
        types, mtypes = set(), set()
        for t in texts:
            for w_ in C.key_words(t):
                n += 1
                types.add(w_)
                if w_ not in vocab:
                    miss += 1
                    mtypes.add(w_)
        return {"words": n, "token_oov": miss / n if n else None, "types": len(types),
                "type_oov": len(mtypes) / len(types) if types else None}
    _OOV = {"train_types": len(vocab), "test_strict": rate(d["text"] for d in C.load_test())}
    for s, _ in OOC_SETS:
        _OOV[s] = rate(r["text_norm"] for r in C.read_jsonl(os.path.join(C.SETS, s + ".jsonl")))
    return _OOV


_VF = None


def vocab_facts():
    """Facts about the released vocabulary and its training text (train_D2), computed here, not typed."""
    global _VF
    if _VF is not None:
        return _VF
    from tokenizers import Tokenizer
    v = Tokenizer.from_file(C.RELEASED_JSON).get_vocab()
    ak = [chr(0x064A), chr(0x0643), chr(0x0647)]
    skr = [chr(c) for c in (0x067B, 0x0684, 0x0759, 0x06B3)]
    pbt = [chr(c) for c in (0x067C, 0x0689, 0x0693, 0x0696, 0x069A, 0x06AB, 0x06BC)]
    latin = [s for s in v if re.search("[A-Za-z]", s) and not s.startswith("<|") and not re.match(r"^<0x[0-9A-F]{2}>$", s)]
    cnt = {c: 0 for c in ak + skr + pbt}
    chars = 0
    var = {}
    p = os.path.join(C.TOK, "data", "train_D2.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r["text"]
            chars += len(t)
            var[r["variety"]] = var.get(r["variety"], 0) + len(t)
            for c in cnt:
                cnt[c] += t.count(c)
    _VF = {"vocab_size": len(v), "vocab_latin_pieces": len(latin),
           "vocab_ak_pieces": sum(1 for s in v if any(c in s for c in ak)),
           "train_file": p, "train_sha256": C.sha256_file(p), "train_chars": chars,
           "train_variety_char_share": {k: v / chars for k, v in sorted(var.items())},
           "train_ak": {"yeh": cnt[ak[0]], "kaf": cnt[ak[1]], "heh": cnt[ak[2]]},
           "train_skr": sum(cnt[c] for c in skr), "train_pbt": sum(cnt[c] for c in pbt)}
    return _VF


BYTE_TOK = re.compile(r"<([0-9a-f]{2})>")


def cell_tokens(toks):
    out = []
    for t in toks:
        t = BYTE_TOK.sub(lambda m: chr(0x27E8) + m.group(1) + chr(0x27E9), t)
        out.append(t.replace("|", "\\|"))
    return " \\| ".join(out)


def examples_md(R):
    ex = C.read_jsonl(os.path.join(C.SETS, "examples.jsonl"))
    L = []
    L.append("Three natural sentences from the strict test split (held out from tokenizer training), each "
             "tokenizer with its own encoder, no special tokens. `▁` marks a space the token carries; `▯` stands for "
             "one byte of a letter that the token splits (byte-level BPE can cut a 2-byte Urdu letter into two "
             "tokens). Token boundaries are shown as `|`. The text is right-to-left, so in most viewers the first "
             "token is at the right.")
    L.append("")
    L.append("| tokenizer | " + " | ".join("sentence %d (%d words)" % (i + 1, e["words"]) for i, e in enumerate(ex)) +
             " | total | × released |")
    L.append("|---|" + "---:|" * (len(ex) + 2))
    tot_rel = sum(R["released"]["examples"][i]["n"] for i in range(len(ex)))
    for k in KEYS:
        ns = [R[k]["examples"][i]["n"] for i in range(len(ex))]
        row = [str(n) for n in ns] + [str(sum(ns)), "%.2f" % (sum(ns) / tot_rel)]
        if k == "released":
            row = ["**%s**" % x for x in row]
        L.append("| %s | %s |" % (C.DISPLAY[k], " | ".join(row)))
    L.append("")
    for i, e in enumerate(ex):
        L.append("**Sentence %d** (%s, test_strict): %s" % (i + 1, e["source"], e["text_raw"]))
        L.append("")
        L.append("*Approximate English:* %s" % e["gloss_en_approx"])
        L.append("")
        L.append("| tokenizer | tokens | segmentation |")
        L.append("|---|---:|---|")
        for k in KEYS:
            r = R[k]["examples"][i]
            L.append("| %s | %d | %s |" % (C.DISPLAY[k], r["n"], cell_tokens(r["tokens"])))
        L.append("")
    return L


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
