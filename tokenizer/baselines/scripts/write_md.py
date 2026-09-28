"""Render baselines/BASELINES.md from manifest.json (+ scan / crosscheck logs). All numbers come from those files."""
import json
import os

B = r"F:\Hindko\_tokenizer\baselines"
M = json.load(open(os.path.join(B, "manifest.json"), encoding="utf-8"))
SCAN = json.load(open(os.path.join(B, "scripts", "_urdu_scan.json"), encoding="utf-8")) if os.path.exists(os.path.join(B, "scripts", "_urdu_scan.json")) else None
BL = M["baselines"]
W = [b for b in BL if b.get("working")]
REC = [b for b in W if b.get("recommended_for_benchmark")]
LOSSLESS = [b for b in W if b.get("lossless")]
BAT = M["roundtrip_battery"]
LF = "\n"
NS = len(M["samples"])
NCORE = sum(1 for s in M["samples"] if s["group"] == "core")
# Labelled lossless by the first version of this file (5 single-line samples only); kept to state what changed.
PREVIOUSLY_LABELLED_LOSSLESS = ["mt5", "xlm-r", "sindhi-xlmr", "claude-legacy", "urdu-gpt2-20k", "hindko-probe-bpe32k"]
NL_SHORT = {"preserved": "kept", "each run of line breaks becomes one space": "→ 1 space per run",
            "each line break becomes a space": "→ space each", "line breaks deleted (words joined)": "deleted",
            "line breaks become UNK tokens": "→ UNK"}


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def trunc(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def src(b):
    if b["repo"].startswith("local:"):
        return "local file (project probe)"
    k = b.get("mirror_kind")
    r = f"`{b['repo']}`"
    if k:
        r += f" — {k} of `{b['official']}`" if b.get("official") else f" — {k}"
    return r


def bf(b):
    v = b.get("byte_fallback")
    if isinstance(v, str):
        return "byte-level" if "byte-level" in v else v
    return "yes" if v else "no"


def nl(b):
    return NL_SHORT.get(b.get("newline_handling"), "other")


def names(bs):
    return ", ".join(f"`{b['name']}`" for b in bs)


L = []
A = L.append
A("# Competitor tokenizer baselines for the Hindko tokenizer")
A("")
A(f"Generated {M['generated_utc']} by `scripts/build_manifest.py` (numbers in this file are copied from `manifest.json`).")
A("")
_manual = [b for b in W if b["recommendation_reason"].startswith("redundant: ")]
_rule = [b for b in W if not b.get("recommended_for_benchmark") and b not in _manual]
A(f"- **{len(W)} working baselines** out of {len(BL)} registered. **{len(REC)} are recommended for the benchmark.** The other "
  f"{len(W) - len(REC)} are redundant, flagged `recommended_for_benchmark: false`: {len(_rule)} encode all {NS} samples and all "
  f"{BAT['n_texts']:,} round-trip battery texts token-for-token like an earlier entry, with a token set that differs from it by at "
  f"most 1%, and {names(_manual)} {'is' if len(_manual) == 1 else 'are'} marked redundant by a documented override. "
  "See *Baselines whose encodings match*.")
A(f"- **{len(LOSSLESS)} of {len(W)} are lossless on Hindko**: `decode(encode(x)) == x` for all {NS} samples *and* for every text of "
  f"the round-trip battery ({BAT['n_texts']:,} real corpus texts, {BAT['n_chars']:,} characters, {BAT['n_line_breaks']:,} line breaks; "
  f"see *Samples, battery and definitions*). The other {len(W) - len(LOSSLESS)} are listed in *Tokenizers that are not lossless on Hindko*.")
fams = sorted({b["provider"] for b in W})
A(f"- Providers covered ({len(fams)}): " + ", ".join(fams) + ".")
A("- Files: tokenizer files only (allow-list), downloaded anonymously, revision pinned to the commit sha recorded in the manifest. "
  "No model weights, no login, no remote code executed.")
A("- `hindko-probe-bpe32k` is the project's earlier SentencePiece probe. It was trained on the whole corpus, including the test split, "
  "so it is a **leaky reference and not a fair competitor**.")
A("")
A("**Revision (2026-09-26, after review).** The first version tested losslessness on 5 single-line paragraphs only. None of them "
  "contained a line break, an Arabic presentation form or U+2026, so it labelled "
  + ", ".join(f"`{n}`" for n in PREVIOUSLY_LABELLED_LOSSLESS) + " lossless. Measured now: "
  + "; ".join(f"`{b['name']}` {'lossless' if b.get('lossless') else 'NOT lossless'} "
              f"({b['roundtrip_battery']['n_exact']:,}/{b['roundtrip_battery']['n_texts']:,} battery texts exact)"
              for b in W if b["name"] in PREVIOUSLY_LABELLED_LOSSLESS)
  + ". Three stress samples and the corpus battery were added, the redundancy rule now also compares battery encodings, "
  "and every tokenizer's line-break handling is recorded.")
A("")
A("## Usage")
A("")
A("```python")
A("import sys; sys.path.insert(0, r\"F:\\Hindko\\_tokenizer\\baselines\")")
A("from load_baselines import load_baseline, list_baselines")
A("names = list_baselines(recommended_only=True)   # de-duplicated set; list_baselines() gives all working ones")
A("lossless = list_baselines(lossless_only=True)   # exact round trip on the samples and the whole battery")
A("tok = load_baseline(\"gemma-4\")")
A("ids = tok.encode(text)      # list[int], no BOS/EOS/CLS/SEP added")
A("text2 = tok.decode(ids)     # str")
A("tok.vocab_size")
A("```")
A("")
A("`load_baseline` works offline from `files/`. It reads `tokenizer.json` with the `tokenizers` library, not with "
  "`transformers.AutoTokenizer`, because transformers 5.3 does not reproduce some repos' tokenizers. See *Verification* below.")
A("")
A("## Samples, battery and definitions")
A("")
A(f"**{NS} samples** (`samples_hindko.json`) from `F:\\Hindko\\hindko_dataset.jsonl` (strict subset), chosen deterministically. "
  f"The {NCORE} *core* slots each take the paragraph (a line of 200–900 chars) with the smallest `sha256(uid#paragraph_index)` that "
  f"satisfies the slot. The {NS - NCORE} *stress* slots each have their own required predicate: a whole multi-line record "
  "(600–3,000 chars, at least 4 line breaks including a blank line, smallest `sha256(uid#doc)`), a paragraph containing U+FDFA `ﷺ`, "
  "and a paragraph containing U+2026 `…`.")
A("")
A("| group | slot | uid | source | chars | words | line breaks |")
A("|---|---|---|---|---:|---:|---:|")
for s in M["samples"]:
    A(f"| {s['group']} | {s['slot']} | `{s['uid']}` | {s['source']} | {s['n_chars']} | {s['n_words']} | {s['n_newlines']} |")
A("")
A(f"**Round-trip battery** (`manifest.json → roundtrip_battery`): {BAT['definition']}.")
A("")
A("| part | texts | characters | line breaks |")
A("|---|---:|---:|---:|")
for p, v in BAT["parts"].items():
    A(f"| {p} | {v['n_texts']:,} | {v['n_chars']:,} | {v['n_line_breaks']:,} |")
A(f"| **total** | {BAT['n_texts']:,} | {BAT['n_chars']:,} | {BAT['n_line_breaks']:,} |")
A("")
A(f"The coverage lines contain {BAT['n_distinct_code_points_covered']} distinct code points: every code point of the permissive "
  "corpus except the space and LF, which occur throughout the eval documents. "
  "Occurrences in the battery of characters that some tokenizers lose: " + ", ".join(f"{k} ×{v:,}" for k, v in BAT["at_risk_counts"].items()) + ".")
A("")
A(f"- **Round trip:** {M['roundtrip_definition']}.")
A(f"- **Lossless:** {M['lossless_definition']}.")
A(f"- **UNK rate:** {M['unk_rate_definition']}.")
A(f"- **Line breaks:** {M['newline_handling_definition']}.")
A(f"- **tok/word** below is total tokens ÷ whitespace words over the {NCORE} core paragraphs "
  f"({W[0]['sample_words_total']} words, no line breaks). It is a quick sanity check, "
  "**not the benchmark**; ranking the competitors needs the held-out split and the full metric suite.")
A("")
A("## Baselines")
A("")
A(f"Lossless = exact round trip on all {NS} samples and all {BAT['n_texts']:,} battery texts. Line breaks = what `decode(encode(x))` "
  "makes of LF. UNK rate is over the battery. \"Same as\" = redundant with the named, earlier baseline.")
A("")
A("| name | provider — tokenizer | year | source repo | vocab | algorithm | byte fallback | licence (repo) | lossless | line breaks | UNK rate (battery) | tok/word (core) | same as |")
A("|---|---|---:|---|---:|---|---|---|:---:|---|---:|---:|---|")
order = {"core": 0, "frontier-2026": 1, "urdu": 2, "regional": 3, "reference": 4}
for b in sorted(W, key=lambda b: (order[b["tier"]], b["provider"], -b["year"], b["name"])):
    same = "" if b.get("recommended_for_benchmark") else (b.get("duplicate_of") or b["recommendation_reason"])
    A(f"| `{b['name']}` | {esc(b['provider'])} — {esc(b['family'])} | {b['year']} | {esc(src(b))} | {b['vocab_size']:,} | {esc(b['algorithm'])} | "
      f"{bf(b)} | {esc(b.get('license'))} | {'yes' if b['lossless'] else '**no**'} | {nl(b)} | {b['unk_rate_battery']:.5f} | "
      f"{b['sample_tokens_per_word']:.2f} | {esc(same)} |")
A("")
A("Tiers: rows are grouped as core (the requested list), frontier-2026 (newest releases found on the hub on 2026-09-26), "
  "urdu (Urdu-specific), regional (Perso-Arabic neighbours with vocabulary extensions: Pashto, Sindhi), and reference (legacy or leaky).")
A("")
bad = [b for b in W if not b["lossless"]]
if bad:
    A(f"### Tokenizers that are not lossless on Hindko ({NS} samples + {BAT['n_texts']:,}-text battery)")
    A("")
    A(f"Tested on the {NS} samples ({NCORE} core + {NS - NCORE} stress) and on every battery text "
      f"({BAT['parts'].get('eval documents', {}).get('n_texts', sum(v['n_texts'] for k, v in BAT['parts'].items() if k.startswith('eval'))):,} "
      "eval-split documents plus the character-coverage lines). Characters lost are summed over the failing battery texts (multiset "
      "difference input − output, so a character that is changed into another counts as lost). A tokenizer can fail in several ways at once.")
    A("")
    A("| name | samples exact | battery: eval docs exact | battery: coverage lines exact | line breaks (in → out) | UNK tokens (battery) | characters lost (battery, top 5) | first difference (expected → got) |")
    A("|---|---:|---:|---:|---|---:|---|---|")
    for b in bad:
        rb = b["roundtrip_battery"]
        ev, cv = rb["parts"].get("eval documents", {}), rb["parts"].get("char-coverage lines", {})
        fd = (rb.get("examples") or [{}])[0].get("first_diff") or next((s.get("first_diff") for s in b["samples"] if s.get("first_diff")), None)
        # LF shown as ⏎ so that a line break turned into a space is visible
        fds = (f"`{esc(fd['expected_ctx'].replace(LF, '⏎'))}` → `{esc(fd['got_ctx'].replace(LF, '⏎'))}` "
               f"({fd['expected'][0] if fd['expected'] else 'end'} → {fd['got'][0] if fd['got'] else 'end'})") if fd else ""
        lost_s = "; ".join(f"{k} ×{v:,}" for k, v in list(rb["chars_lost"].items())[:5]) or "none (spacing only)"
        A(f"| `{b['name']}` | {b['n_samples_exact']}/{NS} | {ev.get('n_exact', 0):,}/{ev.get('n_texts', 0):,} | {cv.get('n_exact', 0)}/{cv.get('n_texts', 0)} | "
          f"{nl(b)} ({rb['line_breaks_in']:,} → {rb['line_breaks_out']:,}) | {rb['n_unk']:,} | {esc(lost_s)} | {fds} |")
    A("")
    A("Failure flags per tokenizer (number of battery texts): "
      + "; ".join(f"`{b['name']}`: " + ", ".join(f"{k} {v:,}" for k, v in b["roundtrip_battery"]["failure_flags"].items()) for b in bad) + ".")
    A("")
A("### Line breaks and token counts")
A("")
drop = [b for b in W if b.get("newline_handling") != "preserved"]
keep = [b for b in W if b.get("newline_handling") == "preserved"]
shares = sorted(b["roundtrip_battery"]["eval_docs_tokens"]["line_break_token_share"] for b in keep)
A(f"The corpus is multi-line: {BAT['n_line_breaks']:,} line breaks in the battery. {len(drop)} tokenizers do not keep them: "
  + "; ".join(f"{names([b for b in drop if b.get('newline_handling') == h])} ({h})"
              for h in sorted({b.get('newline_handling') for b in drop})) + ". "
  f"The other {len(keep)} keep every line break in the probes and in the battery.")
A("")
if shares:
    A(f"Dropping line breaks is a free saving in any token count, because a tokenizer that turns LF into a space effectively "
      f"encodes the text with every LF replaced by a space. For the {len(keep)} tokenizers that keep line breaks, the original eval "
      f"documents cost {shares[0] * 100:.2f}–{shares[-1] * 100:.2f}% more tokens than that space-joined text (median "
      f"{shares[len(shares) // 2] * 100:.2f}%; share = (tokens(text) − tokens(text with LF → space)) / tokens(text)"
      + ("; `byt5` gets 0 because LF and a space are each one byte" if [b["name"] for b in keep if not b["roundtrip_battery"]["eval_docs_tokens"]["line_break_token_share"]] == ["byt5"] else "")
      + "). The benchmark must therefore either compare token counts on "
      "single-line segments (split on LF, as the 5 core samples are), or count on the original text and report each tokenizer's "
      "`newline_handling` next to its numbers. Per-tokenizer values are in `manifest.json → baselines[].roundtrip_battery.eval_docs_tokens`.")
    A("")
red = [b for b in W if b.get("vocab_vs_duplicate_of")]
if red:
    A("### Baselines whose encodings match an earlier baseline")
    A("")
    A(f"These encode all {NS} samples and all {BAT['n_texts']:,} battery texts ({BAT['n_chars']:,} characters) exactly like an earlier "
      "baseline. The vocabularies were also compared directly. Columns: tokens only in this entry / only in the other entry / shared "
      "tokens with the same id.")
    A("")
    A("| name | duplicate of | only here | only in other | shared, same id / shared |")
    A("|---|---|---:|---:|---|")
    for b in red:
        v = b["vocab_vs_duplicate_of"]
        A(f"| `{b['name']}` | `{v['other']}` | {v['n_tokens_only_here']:,} | {v['n_tokens_only_in_other']:,} | {v['n_common_same_id']:,} / {v['n_common']:,} |")
    A("")
    kept = [b for b in red if b.get("recommended_for_benchmark")]
    manual = [b for b in red if b["recommendation_reason"].startswith("redundant: ")]
    A("Small differences are usually special or reserved tokens. Rule: an entry is marked redundant "
      "(`recommended_for_benchmark: false`) only if its sample and battery encodings match *and* its token set differs from the "
      "other entry's by at most 1% of its vocabulary. "
      + (("Kept despite identical encodings: " + ", ".join(
          f"`{b['name']}` (token sets differ by {b['vocab_vs_duplicate_of']['n_tokens_only_here'] + b['vocab_vs_duplicate_of']['n_tokens_only_in_other']:,} tokens)"
          for b in kept) + ". ") if kept else "")
      + " ".join(f"`{b['name']}` is marked redundant by hand: {b['recommendation_reason'].removeprefix('redundant: ')}." for b in manual) + " "
      "The benchmark can score one representative per redundant group, but should name every member, since each is a different model family.")
    A("")
A("## Pipelines (normalizer / pre-tokenizer)")
A("")
A("| name | normalizer | pre-tokenizer (regexes truncated) | decoder |")
A("|---|---|---|---|")
for b in sorted(W, key=lambda b: b["name"]):
    A(f"| `{b['name']}` | {esc(trunc(b.get('normalizer'), 90))} | {esc(trunc(b.get('pre_tokenizer'), 150))} | {esc(trunc(b.get('decoder'), 60))} |")
A("")
A("Full pipeline strings, added-token counts, Arabic-script vocabulary counts, file sha256s and pinned revisions are in `manifest.json`.")
A("")
A("### Arabic-script vocabulary")
A("")
A("Count of vocabulary entries containing at least one Arabic-script code point (U+0600–06FF, 0750–077F, 08A0–08FF, FB50–FDFF, FE70–FEFF). "
  "For byte-level BPE, tokens are decoded from the byte alphabet first, and partial UTF-8 sequences are ignored.")
A("")
A("| name | vocab | Arabic-script tokens | of which ≥2 Arabic letters |")
A("|---|---:|---:|---:|")
for b in sorted([b for b in W if b.get("n_arabic_script_tokens") is not None], key=lambda b: -b["n_arabic_script_tokens"]):
    A(f"| `{b['name']}` | {b['vocab_size']:,} | {b['n_arabic_script_tokens']:,} | {b['n_arabic_script_tokens_len2plus']:,} |")
A("")
A("## Verification")
A("")
A("**Mirrors.** For gated official repos, the hub still exposes file metadata (git blob id and LFS sha256) without login. "
  "The mirrors marked *byte-identical* have the same blob id or LFS sha256 as the official file:")
A("")
for b in W:
    if b.get("mirror_kind") == "byte-identical mirror":
        A(f"- `{b['name']}`: `{b['repo']}` = `{b['official']}` (official gated: {b.get('official_gated')}).")
A("- `eurollm`: the ungated official re-release `utter-project/EuroLLM-9B-Instruct-2512` has the same `tokenizer.model` LFS sha256 as the gated 2024 `EuroLLM-9B-Instruct`.")
A("- `tiny-aya`: the ungated mirror's `tokenizer.json` is **not** byte-identical to the gated official one (187 bytes smaller). "
  "Equivalence could not be checked without logging in, so treat this row as provisional.")
A("")
A("**Conversions** (all numbers from `manifest.json → crosschecks`). The first checks ran on 2,000 single-line paragraphs, the 5 core "
  "samples and one ASCII stress string; the *(multi-line)* checks repeat them on 200 multi-line strict records (first 3,000 characters "
  "each, line breaks kept) plus the 3 stress samples:")
A("")
for k, v in M["crosschecks"].items():
    if k.startswith("kimi-k2 conversion vs reference (multi"):
        A(f"- Kimi K2 (multi-line): {v['n_texts_mismatch']} mismatches vs the reference and {v['n_roundtrip_failures_a']} round-trip failures "
          f"on {v['n_texts']:,} texts ({v['n_tokens_b']:,} tokens, {v['n_line_breaks']:,} line breaks).")
    elif k.startswith("kimi-k2 conversion"):
        A(f"- Kimi K2: the official `tiktoken.model` was converted to a `tokenizers` BPE here (`files/kimi-k2/tokenizer.converted.json`). "
          f"Its split regex was read from `tokenization_kimi.py` as text with `ast.literal_eval`; the file was never imported or run. "
          f"Checked against a pure-Python implementation of tiktoken's algorithm on {v['n_texts']:,} texts ({v['n_reference_tokens']:,} tokens): "
          f"{v['n_texts_mismatch_vs_reference']} mismatches and {v['n_roundtrip_failures']} round-trip failures.")
    elif k == "kimi split regex vs K2":
        A("- Kimi K2.5, K2.6 and K3 publish the same `tiktoken.model` (identical LFS sha256), and their split regex matches K2's: "
          + ", ".join(f"{r.split('/')[1]}={'same' if ok else 'DIFFERENT'}" for r, ok in v.items()) + ".")
    elif "tekken" in k:
        A(f"- {k}: {v['n_texts_mismatch']} mismatching texts out of {v['n_texts']:,} ({v['n_tokens_b']:,} tokens; tekken {v.get('tekken_version')}"
          + (f", {v.get('n_special')} special ids" if v.get("n_special") is not None else "")
          + (f", {v['n_line_breaks']:,} line breaks" if v.get("n_line_breaks") else "") + ").")
    elif " vs " in k:
        bv = v.get("base_vocab")
        A(f"- {k}: {v['n_texts_mismatch']} mismatching texts out of {v['n_texts']:,}"
          + (f"; base vocab ids < {bv.get('n_checked'):,} with a different token: {bv.get('n_ids_differ')}" if bv else "")
          + (f" ({v['n_line_breaks']:,} line breaks)" if v.get("n_line_breaks") else "") + ".")
A("")
A(f"**transformers cross-check.** Each baseline was also loaded with transformers 5.3 (`AutoTokenizer`, or the explicit class "
  f"where AutoTokenizer cannot resolve a bare vocab file), and its ids on the {NS} samples were compared with ours:")
A("")
agree = [b["name"] for b in W if str(b.get("transformers_crosscheck", "")).startswith("agrees")]
differ = [b for b in W if str(b.get("transformers_crosscheck", "")).startswith("DIFFERS")]
other = [b for b in W if not str(b.get("transformers_crosscheck", "")).startswith(("agrees", "DIFFERS"))]
A(f"- Agree on {NS}/{NS}: {len(agree)} baselines.")
for b in differ:
    A(f"- **{b['name']}**: {b['transformers_crosscheck']}.")
for b in other:
    A(f"- {b['name']}: {b['transformers_crosscheck']}.")
A("")
diag = [b for b in differ if b.get("transformers_pipeline_diff")]
A(f"The multi-line and stress samples raised the number of disagreements from 5 (on the 5 core paragraphs) to {len(differ)}. "
  f"In {len(diag)} of the {len(differ)} cases, the pipeline transformers builds differs from the published `tokenizer.json` "
  "(`manifest.json → transformers_pipeline_diff` records both versions). In 5.3, classes such as `LlamaTokenizerFast` "
  "(named in the DeepSeek-V3/R1 and Sarvam-M configs), `GPT2Tokenizer` and `Qwen2Tokenizer` rebuild their class-default "
  "pre-tokenizer instead of loading the repo's, and drop a normalizer the file declares. Examples: for Qwen 3.5/3.8 the published "
  "regex keeps combining marks inside words (`[\\p{L}\\p{M}]+`) and transformers substitutes Qwen2's `\\p{L}+`; for `claude-legacy` "
  "it drops the file's NFKC normalizer; for DeepSeek-V3 and Sarvam-M it builds a Metaspace pipeline, and the first 120 characters "
  "of a sample came out as the single token `()`, so all Arabic-script text was dropped. Our loader runs the published files "
  "as they are. DeepSeek-V4 loads through the generic backend, agrees with transformers, and encodes the samples identically to "
  "the V3 file. The Mistral files match `tekken.json`, and Xenova/gpt-4o matches OpenAI's gpt-oss, on single-line and multi-line "
  "text. **Benchmark code must use `load_baseline()`, not `AutoTokenizer`.**")
A("")
A("## Urdu-specific tokenizers")
A("")
A("The task asked for Urdu LLMs with an **extended** Urdu vocabulary. Every Urdu candidate below was measured, not assumed:")
A("")
A("| name | repo | vocab | extended vs base? | new Arabic-script tokens | lossless | encodings (samples + battery) |")
A("|---|---|---:|---|---:|:---:|---|")
for b in W:
    if b["tier"] not in ("urdu", "regional"):
        continue
    u = b.get("urdu_extension_check")
    if u:
        ext = f"{'yes' if b.get('urdu_vocab_extended') else 'no'} (base {u['base']}, {u['n_surface_forms_not_in_base']:,} new surface forms)"
        newar = f"{u['n_new_arabic_script_tokens']:,}"
    else:
        ext, newar = "n/a (own vocabulary)", "–"
    eff = ("identical to " + ", ".join(b["same_encodings_as"][:2])) if b.get("same_encodings_as") else \
        f"{b['sample_tokens_total']} tokens on the 5 core samples"
    A(f"| `{b['name']}` | `{b['repo']}` | {b['vocab_size']:,} | {ext} | {newar} | {'yes' if b['lossless'] else '**no**'} | {esc(eff)} |")
A("")
A("Findings:")
A("")
byname = {b["name"]: b for b in W}
same3 = [n for n in ["alif-1.0", "qalb-1.0", "urdu-llama3-almanach"] if "llama-3" in byname[n].get("same_encodings_as", [])]
not3 = [n for n in ["alif-1.0", "qalb-1.0", "urdu-llama3-almanach"] if n not in same3]
A(f"- {', '.join('`' + n + '`' for n in same3)} have Llama 3's vocabulary (no new surface forms containing Arabic script) "
  f"and encode all {NS} samples and all {BAT['n_texts']:,} battery texts identically to `llama-3`"
  + (f" ({', '.join(not3)} differ)" if not3 else "") + ". `urdu-llama2-almanach` "
  f"{'likewise matches `llama-2`' if 'llama-2' in byname['urdu-llama2-almanach'].get('same_encodings_as', []) else 'differs from llama-2'}. "
  "None of these Urdu LLMs extends the vocabulary.")
c = byname["urdu-llama3.2-custom"]
A(f"- `urdu-llama3.2-custom` has {c['vocab_size'] - byname['llama-3']['vocab_size']:,} more ids than Llama 3. The extra tokens are "
  "stored as *added tokens* written in the byte-level alphabet (for example `ĠÙ¾ÛĮØª`), which never match raw text, so it encodes "
  f"{'everything identically to Llama 3' if 'llama-3' in c.get('same_encodings_as', []) else 'the 5 core samples identically to Llama 3'}. "
  "The extension does nothing.")
c = byname["urdu-llama-bilal"]
A(f"- `urdu-llama-bilal` has {c['vocab_size'] - byname['llama-3']['vocab_size']:,} more ids than Llama 3, as raw-Urdu *added tokens*. "
  "Added tokens are matched before pre-tokenization, even inside words, so the token count goes up: "
  f"{c['sample_tokens_total']} vs {byname['llama-3']['sample_tokens_total']} for Llama 3 on the 5 core samples.")


def lossdesc(n):
    b = byname[n]
    if b["lossless"]:
        return "lossless"
    rb = b["roundtrip_battery"]
    top = ", ".join(list(rb["chars_lost"])[:3])
    return (f"**not lossless**: {rb['n_exact']:,}/{rb['n_texts']:,} battery texts exact; line breaks {nl(b)}; loses {top}")


A("- There is no strong Urdu LLM with a working vocabulary extension on the hub. The Urdu-script competitors worth benchmarking are "
  f"therefore the Urdu-native tokenizers of small Urdu models: `roberta-urdu` (byte-level BPE 52k; {lossdesc('roberta-urdu')}), "
  f"`urdu-gpt2-20k` (20k, {esc(trunc(byname['urdu-gpt2-20k'].get('normalizer'), 60))}; {lossdesc('urdu-gpt2-20k')}) and "
  f"`urdu-bert-64k` (WordPiece; {lossdesc('urdu-bert-64k')}). Two Perso-Arabic neighbours with real extensions were added from the "
  f"scan: `pashto-lfm2.5` (Pashto-extended LFM2.5 LLM; {lossdesc('pashto-lfm2.5')}) and `sindhi-xlmr` (Sindhi-extended XLM-R; "
  f"{lossdesc('sindhi-xlmr')}). Add to these the multilingual large-vocabulary models (Gemma 3/4, Sarvam-30B, BLOOM, XLM-R, NLLB, "
  "MuRIL, IndicBERT v2).")
if SCAN:
    meas = [m for m in SCAN["measured"] if m.get("vocab_size")]
    A(f"- A wider hub scan (`scripts/urdu_scan.py`; queries: {', '.join(SCAN['queries'])}) looked at {SCAN['n_stage1']} ungated repos "
      f"(speech, classification, GGUF and adapter repos excluded). It found {SCAN.get('n_unique_new_signatures')} tokenizer files "
      f"that differ from everything above, and downloaded and measured the {len(SCAN['measured'])} with the most downloads and likes. "
      "The rest were not examined. See `scripts/_urdu_scan.json` and the table below. The scan ran before the stress samples and the "
      "battery existed: its RT column covers the 5 single-line core paragraphs only, so `True` there does **not** mean lossless "
      "(the scan candidates were not run on the battery; the selected ones above were).")
    A("")
    A("| repo | downloads | vocab | Arabic-script tokens | tokens on 5 core samples | RT (5 core paragraphs only) | UNK |")
    A("|---|---:|---:|---:|---:|:---:|---:|")
    for m in sorted(meas, key=lambda m: m.get("sample_tokens_total") or 1e9):
        A(f"| `{m['repo']}` | {m.get('downloads')} | {m['vocab_size']:,} | {m.get('n_arabic_script_tokens'):,} | {m.get('sample_tokens_total')} | {m.get('roundtrip_exact')} | {m.get('n_unk')} |")
A("")
A("## Not collected")
A("")
for line in M.get("not_collected", []):
    A(f"- {line}")
A("")
A("## Caveats")
A("")
A(f"- Losslessness is measured on the released text (NFC, before the canonical normalisation in `_tokenizer/normalization`). That "
  "normalisation folds `ﷲ` but keeps `ﷺ`, so NFKC-type tokenizers stay lossy after it. The battery covers every code point of the "
  "corpus, but only the eval split in full; a tokenizer could still fail on a context that occurs only in train.")
A(f"- The {NCORE}-paragraph token counts are a sanity check, not a ranking. They are single-line, so line-break handling does not affect them.")
A("- `vocab` is the number of distinct token ids including added/special tokens (`len(get_vocab(with_added_tokens=True))`). "
  "The model's embedding matrix can be larger (padding), and `id_space` in the manifest gives max id + 1.")
A("- Licences are copied from each repo's model card metadata. `None` means the card declares no licence tag (for example DeepSeek-V3 "
  "and Hunyuan, which link a custom licence file instead). Mirrors inherit the official licence, recorded as `official_license` where known. "
  "Several licences (cc-by-nc-4.0, Llama, Gemma, Falcon, BLOOM RAIL) carry use restrictions. `files/` is a local working copy: "
  "check each licence before redistributing it.")
A("- Release years come from the model releases (the repo creation date is in the manifest). They are not audited beyond that.")
A("- Several frontier tokenizers are not public (Gemini, Claude 3 and later, OpenAI models after o200k/o200k_harmony). "
  "Gemma 3 is the closest public proxy for Gemini 2.0: the Gemma 3 report says they share a tokenizer.")
A("")
A("## Reproduce")
A("")
A("```")
A("cd F:\\Hindko\\_tokenizer\\baselines\\scripts")
A("set HF_HOME=F:\\Hindko\\_tokenizer\\hf_cache & set PYTHONIOENCODING=utf-8")
A("python probe_hub.py candidates.txt _probe_hub.json     # hub metadata (also candidates2/3/4.txt -> _probe_hub2/3/4.json)")
A("python download.py                                      # tokenizer files only, pinned revisions")
A("python make_samples.py                                  # 8 samples + single-line and multi-line verification sets")
A("python convert_kimi.py && python crosscheck.py          # conversions and cross-checks (crosscheck.py queries the hub for Kimi versions)")
A("python crosscheck_multiline.py                          # the same cross-checks on multi-line text (offline)")
A("python urdu_scan.py                                     # wider Urdu / Perso-Arabic scan")
A("python build_manifest.py && python write_md.py          # needs ../../splits/split_manifest.jsonl for the battery")
A("```")
A("")
open(os.path.join(B, "BASELINES.md"), "w", encoding="utf-8").write("\n".join(L))
print("wrote BASELINES.md", len(L), "lines")
