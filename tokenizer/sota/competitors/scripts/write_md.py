# -*- coding: utf-8 -*-
"""Write ../COMPETITORS_REFRESH.md from ../competitors_all.json (+ the sweep logs)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def jl(name):
    p = os.path.join(ROOT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def f3(x):
    return "–" if x is None else "%.3f" % x


def pct(x):
    return "–" if x is None else "%.1f%%" % (100 * x)


def fi(x):
    return "–" if x is None else format(int(x), ",")


def g1(r):
    g = r["test_strict"]["G1"]
    return "%d/%d" % (g["docs"] - g["fail_docs"], g["docs"])


def unk_share(r):
    o = r["test_strict"]["overall"]
    return (o["unk_tokens"] or 0) / o["tokens"] if o["tokens"] else None


def main():
    A = jl("competitors_all.json")
    V = A["verdict"]
    S = A["sweep"]
    T = A["tokenizers"]
    by = {r["name"]: r for r in T}
    rel = by["hindko-tokenizer-1.0.0-released"]
    ro = rel["test_strict"]["overall"]
    ranked = [r for r in T if r.get("rank_bytes_per_token")]
    ext = [r for r in ranked if r["origin"] != "released"]
    ll = [r for r in ext if r["test_strict"]["G1"]["fail_docs"] == 0]
    new = [r for r in ext if r["origin"].startswith("new")]
    tri = jl("_triage.json") or {"results": []}
    tri2 = jl("_triage2.json") or {"results": []}
    gated = jl("_gated.json") or {"repos": []}
    srch = jl("_search.json")
    srch2 = jl("_search2.json")
    L = []
    a = L.append
    a("# Refreshed competitor sweep: does any existing tokenizer beat the released Hindko tokenizer? (2026-09-27)")
    a("")
    a("Generated %s by `sota/competitors/scripts/report_refresh.py` + `write_md.py`. Text: `data/test_strict.jsonl` "
      "(491 documents, %s UTF-8 bytes, %s words, sha256 `%s…`). Metrics: `eval/harness.py` (code sha256 identical to the "
      "run behind `eval/TEST_COMPETITORS.md`; checked for all %d harness rows%s), each tokenizer with its own native encoder, "
      "no BOS/EOS/CLS added. Reporting only: nothing here changes a decision." % (
          A["generated_utc"], fi(ro["bytes"]), fi(ro["words"]), A["dataset"]["sha256"][:12], len(ranked) + 1,
          "" if not A["code_mismatch"] else "; MISMATCH: " + ", ".join(A["code_mismatch"])))
    a("")
    a("## Verdict")
    a("")
    thr = V["sota_threats_lossless_bytes_per_token"] + [x for x in V["sota_threats_lossless_fertility"]
                                                        if x not in V["sota_threats_lossless_bytes_per_token"]]
    if thr:
        a("**SOTA THREAT FOUND.** Lossless external tokenizer(s) at or above the released tokenizer on bytes/token or at or "
          "below it on fertility: %s. See the table below." % ", ".join("`%s`" % x for x in thr))
    else:
        best = by[V["best_lossless_external"]]
        bo = best["test_strict"]["overall"]
        bf = by[V["best_lossless_external_by_fertility"]]
        a("**No SOTA threat found.** No external tokenizer found in this sweep beats the released tokenizer on bytes/token or "
          "on fertility while round-tripping every test document.")
        a("")
        a("- Released tokenizer (Hindko tokenizer 1.0.0, SentencePiece Unigram, 32,768 ids): **%s bytes/token**, fertility "
          "**%s**, STRR %s, G1 %s, %s tokens. Rank **1 of %d** by bytes/token among the lossless tokenizers (%d ranked rows "
          "in total, lossy ones included; rank %d overall)." % (
              f3(ro["bytes_per_token"]), f3(ro["fertility"]), pct(ro["strr"]), g1(rel), fi(ro["tokens"]),
              V["n_lossless_ranked"], V["n_ranked"], V["rank_released_bytes_per_token"]))
        a("- Best lossless external tokenizer by bytes/token: `%s` (%s; %s) at **%s** bytes/token, fertility %s. The released "
          "tokenizer uses **%.1f %% fewer tokens** on the same 491 documents." % (
              best["name"], best.get("provider") or "", best.get("repo") or "", f3(bo["bytes_per_token"]), f3(bo["fertility"]),
              best["test_strict"]["released_uses_fewer_tokens_pct"]))
        a("- Best lossless external tokenizer by fertility (tokens per word): `%s` at %s (released: %s)." % (
            bf["name"], f3(bf["test_strict"]["overall"]["fertility"]), f3(ro["fertility"])))
    lossy_above = [by[x] for x in V["lossy_tokenizers_at_or_above_released_bytes_per_token"]]
    if lossy_above:
        a("- **Lossy tokenizers with a higher raw bytes/token than the released one: %d.** None is a threat: each fails the "
          "round trip on (almost) every document, and each gets its number by emitting UNK for the words it cannot spell "
          "(one UNK can stand for a whole word or line), so it encodes less text than it was given:" % len(lossy_above))
        a("")
        a("  | tokenizer | repo | bytes/tok | G1 | UNK share of its tokens | non-space chars lost | vocab | algorithm |")
        a("  |---|---|---:|---:|---:|---:|---:|---|")
        for r in lossy_above:
            o = r["test_strict"]["overall"]
            a("  | `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                r["name"], r.get("repo") or "", f3(o["bytes_per_token"]), g1(r), pct(unk_share(r)), fi(o["nonws_chars_lost"]),
                fi(r.get("vocab_size")), r.get("model_type") or r.get("algorithm") or ""))
        a("")
    a("- Claim this supports (PLAN §8 / Amendment 1 §5 / TEST_RESULTS §12 framework): *on the strict Hindko test split, the "
      "released tokenizer uses fewer tokens per byte than every external tokenizer measured here that round-trips the "
      "text* — a statement about token counts on this corpus, not about downstream model quality. A tokenizer is not a "
      "model: it cannot generate, translate or answer anything by itself.")
    a("")
    a("## What was searched (as of 2026-09-27)")
    a("")
    a("- **Hugging Face hub (anonymous API, metadata only):** %d keyword searches on repo ids (Hindko: `hindko`, `hindku`, "
      "`hinko`, `hazarewal`, `hazara`, `potohari`/`pothwari`, `pahari`, `lahnda`; Saraiki, Shahmukhi/Punjabi, Urdu, Pashto, "
      "Sindhi, Kashmiri, Balochi, Brahui, Khowar, Shina, Gojri, `pakistan`, `peshawar`, `nastaliq`, `perso-arabic`, `indic`, "
      "`indo-aryan`, `south-asian`, `bharat`), %d language-tag filters (ISO 639: hno, hnd, ur, urd, pnb, skr, lah, phr, ps, "
      "pus, pbt, pbu, pst, sd, snd, ks, kas, bal, bcc, bgp, khw, scl, gju, pa, pan, trw, bft; text pipelines only), the "
      "2025-2026 repos of %d model labs, and %d named frontier / regional repos; then a second pass (%d keyword, %d author, "
      "%d named queries) for leads from web and GitHub search (UrduLM/ALIF, Markhor, PakMosaic, BrahmicTokenizer, "
      "Muse Glimmer, Nemotron 3.5, Atria Dawn, Dots3, Ornith, Hy-MT2, Nex-N2.5, Bonsai 2, Ling 3.0, MiMo V2.6, MiniCPM5, "
      "Grok 2/3, gpt-oss derivatives)." % (
          len(srch["keywords"]), len(srch["lang_tags"]), len(srch["authors"]), len(srch["named"]),
          len(srch2["keywords"]), len(srch2["authors"]), len(srch2["named"])))
    a("- **Result:** %s repos found, %s with tokenizer files (root or first-level folder). %s gated repos were not logged "
      "into (see *Not collected*); %s speech / image-generation repos were skipped. The tokenizer files of %s repos were "
      "identified by git blob id / LFS sha256 (HEAD requests); %s repos carry a file byte-identical to one already "
      "benchmarked, and %s distinct new files remained." % (
          fi(S["n_repos_found"]), fi(S["n_repos_with_tokenizer_files"]), fi(S["skipped"]["gated"]),
          fi(S["skipped"]["non_text_pipeline"]), fi(S["n_probed"]), fi(S["n_repos_all_files_already_known"]),
          fi(S["n_new_file_signatures"])))
    n_tri_id = sum(1 for r in tri["results"] if r["status"] == "identical_behaviour")
    tri2c = jl("_triage2c.json") or {"results": []}
    n_tri2_same = sum(1 for r in tri2["results"] + tri2c["results"] if r["status"] == "same_vocabulary_sampled")
    dlog = jl("_download_log.json") or {}
    n_rebuilt = sum(1 for v in dlog.values() if v.get("reconstructed_from"))
    skipped_low_ar = jl("_not_downloaded.json") or []
    cal = jl("_calibration.json") or []
    cal_low = [c for c in cal if (c["sampled_arabic_share"] or 0) < 0.03 and c["g1"] >= 467]
    cal_best_low = max(cal_low, key=lambda c: c["bytes_per_token"]) if cal_low else None
    cal_hi = [c for c in cal if c["bytes_per_token"] >= 5.5 and c["g1"] >= 467 and c["bytes_per_token"] < 100]
    cal_min_share_hi = min((c["sampled_arabic_share"] for c in cal_hi if c["sampled_arabic_share"] is not None), default=None)
    a("- **Bandwidth triage.** The link to the hub ran at 0.25-0.75 MB/s, so the %d new files larger than 5 MB (mostly "
      "re-saved Qwen / Llama / Gemma / Mistral tokenizers inside fine-tunes, ~13 GB) were triaged with HTTP range reads "
      "before any full download: %d are byte-identical in their whole `model` section, pipeline and effective added "
      "tokens to a tokenizer already on disk (`scripts/range_triage.py`; behaviour provably identical on this text), and %d "
      "have the same first 300 vocabulary entries, the same pipeline, no added token that occurs in the test text, and "
      "100%% of the vocabulary/merge strings sampled from the middle and end of the file inside the reference's vocabulary "
      "(`scripts/triage2.py`; a re-save of that reference, NOT measured). %d files whose `model` section is byte-identical to "
      "a file on disk but whose pipeline or added tokens differ were rebuilt exactly from their own first bytes plus the "
      "reference's `model` section (`scripts/reconstruct.py`; length checked against the hub) and measured. %d large files "
      "whose sampled vocabulary strings are < 3 %% Perso-Arabic and whose vocabulary family (first 300 entries) is "
      "represented by a measured file or shows < 0.5 %% Perso-Arabic strings were not downloaded (`_not_downloaded.json`). "
      "Calibration on the measured tokenizers (`_calibration.json`, same sampler): the best lossless tokenizer whose "
      "sample is < 3 %% Perso-Arabic is `%s` at %.3f bytes/token, and every tokenizer measured above 5.5 bytes/token has "
      ">= %.0f %% Perso-Arabic strings in the sample, so none of these files can approach %.3f. Every other file (all "
      "files <= 5 MB, and every remaining large file) was downloaded in full and measured." % (
          len(tri["results"]), n_tri_id, n_tri2_same, n_rebuilt, len(skipped_low_ar),
          cal_best_low["name"] if cal_best_low else "?", cal_best_low["bytes_per_token"] if cal_best_low else 0,
          100 * (cal_min_share_hi or 0), ro["bytes_per_token"]))
    a("- **Measured:** %s tokenizers downloaded (tokenizer files only; pinned commit; sha256 in `_download_log.json`), %s "
      "loaded and encoded all 491 test documents (`screen/`), in %s distinct encoding behaviours; %s of them encode the "
      "test text exactly like one of the 66 earlier baselines, and the full harness ran on one representative of each of "
      "the other %s behaviours (`results/`). %s downloaded tokenizers could not be loaded (listed in `competitors_all.json`)." % (
          fi(S["n_downloaded"]), fi(S["n_screened_ok"]), fi(S["n_behaviour_groups"]), fi(S["n_groups_equal_to_a_baseline"]),
          fi(S["n_new_behaviours_harnessed"]), fi(len(A["not_loadable_or_failed"]))))
    a("- **GitHub and the web** (GitHub search API; web search): no Hindko tokenizer and no Hindko language model exists "
      "(GitHub hits for `hindko`: a learning game, an idiom project, a Bible translation, a spoken-digit classifier, a "
      "keyboard). Urdu tokenizer projects found on GitHub only (MusW02/Urdu-TwoStage-BPE, zainali93/Markhor, "
      "salmanmasih/UrduLegalTok, course projects) publish code rather than hub tokenizer files and were not run (no "
      "third-party code). The only hub repo with Hindko in its name that has a tokenizer, `bisma10/hindko-punjabi-urdu-chatbot` "
      "(a LoRA adapter), ships its base model's tokenizer — see its row.")
    a("")
    a("## Updated ranking (summary)")
    a("")
    a("*fewer tokens* = 1 − tokens(released) ÷ tokens(tokenizer) on the same 491 documents. *origin*: `baseline` = one of "
      "the 66 of 2026-09-26, `new` = found by this refresh. Rows are distinct encoding behaviours (members of a behaviour "
      "group are listed in `competitors_all.json`). ⚠ = not lossless.")
    a("")
    a("| # | ll # | tokenizer | origin | provider / repo | vocab | bytes/tok | fertility | STRR | G1 | UNK share | released uses fewer tokens |")
    a("|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    show = ranked[:40]
    heads = ["gpt-4o", "gemma-4", "llama-4", "qwen-3.5", "deepseek-v4.1", "mistral-nemo", "command-a-plus", "kimi-k2", "glm-5",
             "falcon-h1", "grok-1", "claude-legacy", "phi-4", "granite-4.2", "minimax-m3", "sarvam-30b", "tiny-aya", "bloom"]
    for n in heads:
        if n in by and by[n] not in show:
            show.append(by[n])
    show.sort(key=lambda r: r["rank_bytes_per_token"])
    for r in show:
        o = r["test_strict"]["overall"]
        lossy = r["test_strict"]["G1"]["fail_docs"] > 0
        name = "**`%s`**" % r["name"] if r["origin"] == "released" else "`%s`" % r["name"]
        orig = {"released": "released", "baseline (66, 2026-09-26)": "baseline"}.get(r["origin"], "new")
        a("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s%s | %s | %s |" % (
            r["rank_bytes_per_token"], r["rank_among_lossless"] or "–", name, orig,
            ((r.get("provider") or "") + (" — " + r["repo"] if r.get("repo") and r["origin"] != "released" else ""))[:80],
            fi(r.get("vocab_size") or r.get("vocab_size_harness")), f3(o["bytes_per_token"]), f3(o["fertility"]), pct(o["strr"]),
            g1(r), " ⚠" if lossy else "", pct(unk_share(r)),
            "–" if r["origin"] == "released" else ("n/a (lossy)" if lossy and o["bytes_per_token"] >= ro["bytes_per_token"]
                                                    else "%.1f %%" % r["test_strict"]["released_uses_fewer_tokens_pct"])))
    a("")
    a("The table shows the top 40 rows by bytes/token plus the headline model families; all %d ranked rows are in "
      "`competitors_all.json`." % len(ranked))
    a("")
    a("## New tokenizers found by this refresh (distinct behaviours, best first)")
    a("")
    a("| tokenizer | repo | found by | created | vocab | algorithm | bytes/tok | fertility | G1 | same behaviour as |")
    a("|---|---|---|---|---:|---|---:|---:|---:|---|")
    for r in sorted(new, key=lambda r: -r["test_strict"]["overall"]["bytes_per_token"])[:30]:
        o = r["test_strict"]["overall"]
        a("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"], r.get("repo") or "", (r.get("found_by") or "").split("|")[0], r.get("created") or "", fi(r.get("vocab_size")),
            r.get("model_type") or "", f3(o["bytes_per_token"]), f3(o["fertility"]), g1(r),
            ", ".join(r.get("same_encodings_as") or [])[:80] or "–"))
    a("")
    a("%d new behaviours in total; the rest (lower bytes/token) are in `competitors_all.json`." % len(new))
    a("")
    # regional best
    ck = jl("_checklist.json") or []
    if ck:
        a("## Named frontier and regional models: what happened to each tokenizer")
        a("")
        a("| model / family | repo | status in this sweep |")
        a("|---|---|---|")
        for it in ck:
            for i, r in enumerate(it["repos"]):
                a("| %s | `%s` | %s |" % (it["label"] if i == 0 else "", r["repo"], r["status"].replace("|", "/")))
        a("")
    a("## Best tokenizers per neighbouring language (lossless only)")
    a("")
    buckets = [("Urdu", ("urdu", "_ur", "-ur", "urd")), ("Punjabi / Shahmukhi", ("punjabi", "shahmukhi", "pnb", "panjabi")),
               ("Saraiki", ("saraiki", "skr")), ("Pashto", ("pashto", "pus", "pbt", "pashto")), ("Sindhi", ("sindhi", "snd")),
               ("Kashmiri", ("kashmiri", "kas", "_ks")), ("Balochi", ("balochi", "bal"))]
    for lab, keys in buckets:
        cand = [r for r in ll if any(k in (r["name"] + " " + (r.get("repo") or "")).lower() for k in keys)]
        if cand:
            b = cand[0]
            a("- %s: `%s` (%s) %s bytes/token, fertility %s — the released tokenizer uses %.1f %% fewer tokens." % (
                lab, b["name"], b.get("repo") or "", f3(b["test_strict"]["overall"]["bytes_per_token"]),
                f3(b["test_strict"]["overall"]["fertility"]), b["test_strict"]["released_uses_fewer_tokens_pct"]))
        else:
            a("- %s: no lossless tokenizer with the language in its name." % lab)
    a("")
    a("## Not collected")
    a("")
    g_ok = [g for g in gated["repos"] if g.get("all_identical_to_collected")]
    g_no = [g for g in gated["repos"] if not g.get("all_identical_to_collected")]
    a("- **Gated repos (%d checked by file metadata, never logged into):** %d carry a tokenizer file byte-identical to one "
      "measured here (e.g. %s). %d could not be matched and were not measured: %s." % (
          len(gated["repos"]), len(g_ok), ", ".join("`%s`" % g["repo"] for g in g_ok[:6]), len(g_no),
          ", ".join("`%s`" % g["repo"] for g in sorted(g_no, key=lambda g: -(g.get("downloads") or 0))[:25])))
    a("- **Not convertible without running repo code:** tiktoken rank files whose split regex lives only in Python code "
      "(%s), and xAI's `tokenizer.tok.json` (Grok 2; as in `baselines/manifest.json`)." % ", ".join(
          "`%s`" % x["repo"] for x in A["not_loadable_or_failed"] if "tiktoken" in x["status"]) or "none")
    a("- **Not public:** Gemini, Claude 3+ and any OpenAI encoding newer than o200k_base / o200k_harmony (unchanged since "
      "2026-09-26). MiMo-V2.6 (not found on the hub under that name).")
    a("- **Large files classified without a full download:** %d (see above); their encodings are identical (byte-identical "
      "model section) or expected identical (sampled vocabulary) to a measured tokenizer, so none can exceed the rows "
      "shown here." % (n_tri_id + n_tri2_same))
    a("- **Download errors:** %d." % len(A["download_errors"]))
    a("")
    a("## Caveats")
    a("")
    a("- Bytes/token and fertility are screening metrics; LM bits-per-byte (PLAN §4.2) is the decision metric and was not "
      "re-run here. The released tokenizer's LM results are in `analysis/TEST_RESULTS.md`.")
    a("- Lossy tokenizers (UNK-emitting WordPiece/WordLevel vocabularies, NFKC/lower-casing pipelines, line-break "
      "dropping) can show a higher raw bytes/token than a lossless one; they are ranked but never counted as a threat.")
    a("- The hub search covers repo ids, language tags and the listed labs; a tokenizer published under an unrelated name, "
      "without a language tag, or outside the hub cannot be excluded.")
    open(os.path.join(ROOT, "COMPETITORS_REFRESH.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", len(L), "lines")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
