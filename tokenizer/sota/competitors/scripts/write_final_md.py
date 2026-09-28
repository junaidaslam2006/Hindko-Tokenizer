# -*- coding: utf-8 -*-
"""Write ../COMPETITORS_FINAL.md from ../competitors_final.json (run build_final.py first)."""
import json
import os

C = r"F:\Hindko\_tokenizer\sota\competitors"
D = json.load(open(os.path.join(C, "competitors_final.json"), encoding="utf-8"))
R = D["released"]
V = D["verdict"]
N = D["counts"]
S = D["search_coverage"]


def f3(x):
    return "%.3f" % x


def pc(x):
    return "%.1f %%" % (100 * x)


def trunc(s, n=60):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def aliases(g, n=3):
    a = g["aliases"]
    if not a:
        return "–"
    return ", ".join("`%s`" % x for x in a[:n]) + (" +%d" % (len(a) - n) if len(a) > n else "")


L = []
w = L.append
w("# Final competitor ranking: released Hindko tokenizer vs every external tokenizer measured (2026-09-27)")
w("")
w("Generated %s by `sota/competitors/scripts/finalize.py` + `build_final.py` + `write_final_md.py` from tokenizers **already measured** "
  "(no new search or download for this file). Text: `data/test_strict.jsonl` (491 documents, 1,451,026 UTF-8 bytes, 171,769 words, "
  "sha256 `a74c33ad008f…`). Metrics from `eval/harness.py` (same code as `eval/TEST_COMPETITORS.md`), each tokenizer with its own native "
  "encoder, no BOS/EOS/CLS. Machine-readable: `competitors_final.json`. Reporting only; nothing here changes a decision." % D["generated_utc"])
w("")
w("## Answer")
w("")
yes_b, yes_f = V["any_lossless_external_beats_released_on_bytes_per_token"], V["any_lossless_external_beats_released_on_fertility"]
w("**Does any lossless external tokenizer beat the released one on bytes/token or fertility? %s.**" %
  ("No" if not (yes_b or yes_f) else "YES — see below"))
w("")
b = V["best_lossless_external"]
bf = V["best_lossless_external_by_fertility"]
w("- Released tokenizer (SentencePiece Unigram, 32,768 ids, `tokenizer.json` sha256 `%s…`): **%s bytes/token**, fertility **%s**, "
  "STRR %s, G1 %s, %s tokens." % (R["sha256_tokenizer_json"][:8], f3(R["bytes_per_token"]), f3(R["fertility"]), pc(R["strr"]), R["g1"],
                                    "{:,}".format(R["tokens"])))
w("- Rank **%d of %d** distinct lossless encoding behaviours (%d lossless external behaviours + the released one) on both bytes/token and fertility." %
  (V["released_rank_among_lossless_behaviours"], V["n_lossless_behaviours"] + 1, V["n_lossless_behaviours"]))
w("- Best lossless external tokenizer: `%s` (%s) at **%s bytes/token**, fertility %s — the released tokenizer uses **%s fewer tokens** on the same 491 documents." %
  (b["representative"], b["repo"], f3(b["bytes_per_token"]), f3(b["fertility"]), pc(b["released_uses_fewer_tokens"])))
if bf["representative"] != b["representative"]:
    w("- Best lossless external by fertility: `%s` at %s (released: %s)." % (bf["representative"], f3(bf["fertility"]), f3(R["fertility"])))
else:
    w("- It is also the best lossless external tokenizer by fertility (%s vs released %s)." % (f3(bf["fertility"]), f3(R["fertility"])))
w("- %d lossy tokenizers show a higher *raw* bytes/token (and %d a lower-or-equal fertility). None is a competitor: each fails the round trip on "
  "(almost) every document, mostly by emitting UNK for words it cannot spell, so it encodes less text than it was given (table below)." %
  (V["lossy_with_higher_raw_bytes_per_token"], V["lossy_with_lower_or_equal_fertility"]))
cf = V["best_lossy_with_zero_unk_and_zero_nonspace_loss"]
if cf:
    w("- Robustness check: the best tokenizer that fails G1 but emits no UNK and loses no non-space character (`%s`, G1 %d/491) reaches %s bytes/token — also below the released tokenizer." %
      (cf["representative"], cf["g1_pass_docs"], f3(cf["bytes_per_token"])))
w("")
w("**Claim this supports** (PLAN §8 / Amendment 1 §5 / TEST_RESULTS §12 framing): *on the strict Hindko test split, the released tokenizer "
  "produces fewer tokens than every external tokenizer measured here that round-trips the text.* It is a statement about token counts on "
  "this corpus, for the tokenizers found and measured by 2026-09-27. A tokenizer is not a model: it cannot generate, translate or answer "
  "anything, and this ranking says nothing about downstream model quality. Closed tokenizers (Gemini, Claude 3+, OpenAI encodings newer "
  "than o200k) are not public and are not covered.")
w("")
w("## What was compared")
w("")
w("| count | value |")
w("|---|---:|")
w("| external harness rows (full metrics, test_strict) | %d |" % N["harness_rows_external"])
w("| &nbsp;&nbsp;of which new (hub refresh, `results/`) | %d |" % N["harness_rows_new"])
w("| &nbsp;&nbsp;of which study baselines (`eval/results/test_strict/`) | %d |" % N["harness_rows_study_baselines"])
w("| **distinct encoding behaviours** (identical per-document token counts + round-trip outcome merged) | **%d** |" % N["distinct_encoding_behaviours"])
w("| &nbsp;&nbsp;lossless (G1 491/491) / lossy | %d / %d |" % (N["distinct_behaviours_lossless"], N["distinct_behaviours_lossy"]))
w("| **vocabulary families** (heuristic, see Method) | **%d** |" % N["vocabulary_families"])
w("| &nbsp;&nbsp;families with at least one lossless member | %d |" % N["vocabulary_families_with_a_lossless_member"])
w("| hub repos whose tokenizer is represented by a measured row (byte-identical file or identical behaviour) | %s |" % "{:,}".format(N["repos_represented_by_measured_rows"]))
w("| tokenizer files loaded and screened in the refresh | %d |" % N["tokenizer_files_screened_ok_in_refresh"])
w("")
w("Excluded: `hindko-probe-bpe32k` (the study's in-corpus probe, trained on text that includes the test split; never ranked) and the released tokenizer itself.")
w("The study's baseline notes missed that `sarvam-30b` encodes the test text exactly like `gemma-3`/`gemma-4` (identical per-document vectors); here they are merged.")
w("")
w("## Top 15 lossless external tokenizers (distinct behaviours, by bytes/token)")
w("")
w("*fewer tokens* = 1 − tokens(released) ÷ tokens(tokenizer) on the same 491 documents. *aliases* = other measured rows with identical encodings; "
  "*repos* = hub repos carrying this tokenizer (byte-identical file or identical behaviour).")
w("")
w("| # | tokenizer | repo / provider | origin | vocab family | algorithm | vocab | bytes/tok | fertility | STRR | released uses fewer tokens | aliases | repos |")
w("|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|")
w("| – | **released (Hindko 1.0.0)** | this study | released | – | Unigram | 32,768 | **%s** | **%s** | %s | – | – | – |" %
  (f3(R["bytes_per_token"]), f3(R["fertility"]), pc(R["strr"])))
for i, g in enumerate(D["top15_lossless_behaviours"], 1):
    w("| %d | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
        i, g["representative"], trunc(g["repo"], 50), g["origin"], trunc(g["vocab_family_label"], 45), g["model_type"],
        "{:,}".format(g["vocab_size"] or 0), f3(g["bytes_per_token"]), f3(g["fertility"]), pc(g["strr"]),
        pc(g["released_uses_fewer_tokens"]), aliases(g), g["n_repos"]))
w("")
w("## Top 15 lossless vocabulary families (best member of each family)")
w("")
w("| # | family (label) | best member | bytes/tok | fertility | released uses fewer tokens | measured rows in family |")
w("|---:|---|---|---:|---:|---:|---:|")
for i, g in enumerate(D["top15_lossless_families"], 1):
    w("| %d | %s | `%s` | %s | %s | %s | %d |" % (i, trunc(g["vocab_family_label"], 60), g["representative"], f3(g["bytes_per_token"]),
                                                f3(g["fertility"]), pc(g["released_uses_fewer_tokens"]), g["family_size"]))
w("")
w("## Notable lossy tokenizers")
w("")
w("Lossy = fails G1 on at least one document. These are ranked but never counted as competitors: UNK tokens and dropped characters inflate raw bytes/token.")
w("")
w("| tokenizer | repo | algorithm | vocab | bytes/tok | fertility | G1 | UNK share of tokens | non-space chars lost | note |")
w("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
NL = D["notable_lossy"]
for tag, grp in (("raw bytes/token above released", NL["higher_raw_bytes_per_token_than_released"]),
                 ("best lossy below released", NL["next_best_lossy_below_released"])):
    for g in grp:
        w("| `%s` | %s | %s | %s | %s | %s | %d/491 | %s | %s | %s |" % (
            g["representative"], trunc(g["repo"], 50), g["model_type"], "{:,}".format(g["vocab_size"] or 0), f3(g["bytes_per_token"]),
            f3(g["fertility"]), g["g1_pass_docs"], pc(g["unk_share"]), "{:,}".format(g["nonws_chars_lost"]), tag))
w("")
w("## Study baselines (frontier and regional tokenizers), deduplicated")
w("")
w("| tokenizer | aliases (identical encodings) | provider | bytes/tok | fertility | G1 | released uses fewer tokens |")
w("|---|---|---|---:|---:|---:|---:|")
for g in D["study_baselines_deduplicated"]:
    w("| `%s` | %s | %s | %s | %s | %d/491%s | %s |" % (g["representative"], aliases(g, 5), g["provider"], f3(g["bytes_per_token"]),
                                                     f3(g["fertility"]), g["g1_pass_docs"], "" if g["g1_pass_docs"] == 491 else " ⚠",
                                                     pc(g["released_uses_fewer_tokens"])))
w("")
w("## Search coverage — the search was cut short")
w("")
p1, p2, p3 = S["pass1"], S["pass2"], S["pass3"]
sw = S["sweep_totals_at_snapshot"]
w("The ranking uses the measurement snapshot of %s (the refresh report `COMPETITORS_REFRESH.md`). The web/hub search was then **stopped for time**; "
  "anything found or downloaded after the snapshot is **not** in this ranking." % S["ranking_snapshot_utc"])
w("")
w("- **Pass 1** (`_search.json`, %s): %d repo-id keywords (Hindko: hindko, hindku, hinko, hazarewal, hazara, potohari/pothwari, pahari, lahnda; "
  "Saraiki, Shahmukhi/Punjabi, Urdu, Pashto, Sindhi, Kashmiri, Balochi, Brahui, Khowar, Shina, Gojri, regional/script terms), %d ISO 639 language tags "
  "(%s), %d model-lab authors, %d named frontier/regional repos (%d not on the hub); %d queries → %s repos, %s with tokenizer files." %
  (p1["generated_utc"], p1["n_keywords"], p1["n_lang_tags"], ", ".join(p1["lang_tags"]), p1["n_authors"], p1["n_named_repos"],
   p1["n_named_missing"], p1["n_queries"], "{:,}".format(p1["n_repos"]), "{:,}".format(p1["n_with_tokenizer_files"])))
w("- **Pass 2** (`_search2.json`, %s): %d keywords, %d authors, %d named repos from web/GitHub leads (UrduLM/ALIF, Markhor, PakMosaic, "
  "BrahmicTokenizer, 2026 lab releases); %d queries → %s repos, %d kept after filtering." %
  (p2["generated_utc"], p2["n_keywords"], p2["n_authors"], len(p2["named"]), p2["n_queries"], "{:,}".format(p2["n_repos"]), p2["n_kept_after_filter"]))
w("- **Totals at the snapshot:** %s repos found, %s with tokenizer files, %s probed by file hash, %d new files downloaded + %d reconstructed, "
  "%d screened OK, %d behaviour groups, 0 download errors; 302 gated repos and 563 speech/image repos skipped; 392 large files classified by range reads "
  "(identical or expected-identical to a measured tokenizer) instead of downloaded." %
  ("{:,}".format(sw["n_repos_found"]), "{:,}".format(sw["n_repos_with_tokenizer_files"]), "{:,}".format(sw["n_probed"]), sw["n_downloaded"],
   sw["n_reconstructed"], sw["n_screened_ok"], sw["n_behaviour_groups"]))
w("- **GitHub and web:** %s" % S["github_and_web"])
w("- **Pass 3 — stopped** (`_search3.json`, %s): %d author queries for the regional authors surfaced by passes 1–2 (%s) → %s repos, %s with tokenizer files, "
  "%d kept after filtering. The last line of the probe log (`logs/probe_pass3.log`) reports 458 repos still to probe when the step was stopped; nothing from pass 3 was screened or measured (all `screen/` and `results/` files predate the snapshot)." %
  (p3["generated_utc"], p3["n_queries"], ", ".join(p3["authors"]), "{:,}".format(p3["n_repos"]), "{:,}".format(p3["n_with_tokenizer_files"]), p3["n_kept_after_filter"]))
dlg = S["download_log"]
w("- **Downloads after the snapshot:** `_download_log.json` holds %d entries, %d of them logged after the snapshot (a low-priority download batch, "
  "`logs/download_final_lo.log`, was still writing when this file was generated); they are not screened or measured. Download errors in the log: %s." %
  (dlg["entries"], dlg["after_snapshot"], "; ".join("%s (%s)" % (e["repo"], e["error"][:60]) for e in dlg["errors"]) or "none"))
w("- **Not collected:** " + "; ".join(S["not_collected"]) + ".")
w("")
w("## Method and caveats")
w("")
for k, v in D["method"].items():
    w("- **%s:** %s" % (k, v))
w("- Bytes/token and fertility are screening metrics; the study's decision metric is LM bits-per-byte (PLAN §4.2), which was not re-run for these competitors. "
  "The frontier LM comparison (`colab/build_frontier_test/`, `FRONTIER_LM.md`) is **planned**, not run (Colab GPU quota exhausted).")
w("- The hub search covers repo ids, language tags and listed labs; a tokenizer published under an unrelated name, without a language tag, gated, or outside the hub cannot be excluded.")
w("- A vocabulary family is a lineage heuristic (same vocabulary prefix); families whose first entries are generic (byte alphabets, `[unused]` slots) may be split or merged imperfectly. Deduplication itself uses the exact per-document fingerprint, not the family key.")
w("")
open(os.path.join(C, "COMPETITORS_FINAL.md"), "w", encoding="utf-8").write("\n".join(L))
print("wrote", len(L), "lines")
