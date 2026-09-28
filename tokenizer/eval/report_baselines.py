# -*- coding: utf-8 -*-
"""Collect the harness results of the external baselines into eval/baselines_dev.json and eval/BASELINES_DEV.md.

Reads eval/results/dev_strict/<name>/summary.json (ranking set) and eval/results/dev_permissive/<name>/summary.json
(reporting). Ranked by dev_strict bytes/token. The leaky probe (hindko-probe-bpe32k: trained on all data,
including dev and test) is reported in its own section and never ranked. NSL against A1-P1-16k is pending
(that tokenizer does not exist yet); a PROVISIONAL NSL against GPT-4o (o200k) is given instead.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
import adapters as A  # noqa: E402

sys.path.insert(0, A.BASELINES_DIR)
from load_baselines import list_baselines, manifest  # noqa: E402

NSL_REF = "gpt-4o"
PERTS = H.PERT_NAMES


def load(data, name):
    p = os.path.join(H.RESULTS, data, H.safe_name(name), "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def f3(x, nd=3):
    return "–" if x is None else ("%.*f" % (nd, x))


def pct(x, nd=1):
    return "–" if x is None else ("%.*f%%" % (nd, 100 * x))


def slim(m):
    keys = ["docs", "bytes", "chars", "words", "tokens", "bytes_per_token", "chars_per_token", "fertility",
            "tokens_per_word", "continued_word_rate", "strr", "words_without_token", "g1_pass_docs", "g1_fail_docs",
            "g1_fail_docs_modulo_whitespace", "nonws_chars_lost", "nonws_chars_added", "unk_tokens", "unk_rate",
            "bytes_per_token_lines", "renyi_eff_a2.5", "renyi_eff_a2.0", "vocab_used", "vocab_utilisation"]
    out = {k: m.get(k) for k in keys}
    out["robustness"] = {p: {k: m["robustness"][p][k] for k in ("rel_token_change", "extra_tokens_per_affected_word",
                                                                "seg_change_rate", "seg_change_rate_affected",
                                                                "words_affected", "words_vanished",
                                                                "seg_changed_unaffected_words")}
                         for p in PERTS}
    return out


def diagnose_lossy(names) -> dict:
    """For each tokenizer that fails G1 on dev_strict: which non-whitespace characters decode() loses or adds
    (by Unicode name, top 8), and what happens to line breaks. Measured on all dev_strict documents; cached in
    eval/lossy_diagnosis.json."""
    import collections
    import unicodedata
    cache = os.path.join(HERE, "lossy_diagnosis.json")
    docs, info = H.load_docs("dev_strict")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("dataset_sha256") == info["sha256"] and sorted(c["tokenizers"]) == sorted(names):
            return c
    out = {"dataset_sha256": info["sha256"], "tokenizers": {}}
    nm = lambda ch: "%s U+%04X" % (unicodedata.name(ch, "?"), ord(ch))
    for n in names:
        ad = A.from_baseline(n)
        lost, added = collections.Counter(), collections.Counter()
        nl_in = nl_out = 0
        for d in docs:
            t = d["text"]
            dec = ad.decode(ad.encode(t))
            if dec == t:
                continue
            a = collections.Counter(ch for ch in t if not ch.isspace())
            b = collections.Counter(ch for ch in dec if not ch.isspace())
            lost.update(a - b)
            added.update(b - a)
            nl_in += t.count("\n")
            nl_out += dec.count("\n")
        out["tokenizers"][n] = {"lost_top": [[nm(ch), k] for ch, k in lost.most_common(8)],
                                "added_top": [[nm(ch), k] for ch, k in added.most_common(8)],
                                "line_breaks_in_failing_docs": nl_in, "line_breaks_after_decode": nl_out}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def main():
    man = {e["name"]: e for e in manifest()["baselines"]}
    names = list_baselines(working_only=True)
    ref = load("dev_strict", NSL_REF)
    ref_tok = ref["metrics"]["overall"]["tokens"] if ref else None
    ref_src = {s: v["tokens"] for s, v in ref["metrics"]["by_source"].items()} if ref else {}
    rows, missing = [], []
    ds_info = {}
    harness_code = set()
    for n in names:
        s = load("dev_strict", n)
        if s is None:
            missing.append(n)
            continue
        p = load("dev_permissive", n)
        e = man[n]
        harness_code.add(json.dumps(s["harness"]["code"], sort_keys=True))
        ds_info["dev_strict"] = s["dataset"]
        if p:
            ds_info["dev_permissive"] = p["dataset"]
        o = s["metrics"]["overall"]
        row = {"name": n, "leaky": n in A.LEAKY_BASELINES, "provider": e.get("provider"), "family": e.get("family"),
               "tier": e.get("tier"), "year": e.get("year"), "repo": e.get("repo"), "revision": e.get("revision"),
               "algorithm": e.get("algorithm"), "loader": e.get("loader"), "vocab_size": e.get("vocab_size"),
               "tokenizer_sha256": s["tokenizer"].get("sha256"),
               "manifest_lossless": e.get("lossless"), "newline_handling": e.get("newline_handling"),
               "recommended_for_benchmark": e.get("recommended_for_benchmark"),
               "same_encodings_as": e.get("same_encodings_as") or e.get("identical_to"),
               "dev_strict": {"overall": slim(o),
                              "by_source": {k: slim(v) for k, v in s["metrics"]["by_source"].items()},
                              "by_variety": {k: slim(v) for k, v in s["metrics"]["by_variety"].items()},
                              "G1": s["gates"]["G1"], "R2": {k: v for k, v in s["properties"]["R2"].items() if k != "list"},
                              "morphology": {k: (v if not isinstance(v, dict) else
                                                 {kk: vv for kk, vv in v.items() if kk != "unaligned_examples"})
                                             for k, v in s.get("morphology", {}).items()},
                              "speed": s["speed"],
                              "nsl_provisional_vs_" + NSL_REF: (o["tokens"] / ref_tok if ref_tok else None),
                              "nsl_provisional_by_source": {k: v["tokens"] / ref_src[k] for k, v in
                                                            s["metrics"]["by_source"].items() if ref_src.get(k)},
                              "offset_id_mismatch_docs": s["tokenizer"].get("offset_id_mismatch_docs")},
               "dev_permissive": ({"overall": slim(p["metrics"]["overall"]),
                                   "by_source": {k: slim(v) for k, v in p["metrics"]["by_source"].items()},
                                   "by_variety": {k: slim(v) for k, v in p["metrics"]["by_variety"].items()},
                                   "G1": p["gates"]["G1"]} if p else None)}
        rows.append(row)
    comp = sorted([r for r in rows if not r["leaky"]], key=lambda r: -r["dev_strict"]["overall"]["bytes_per_token"])
    leaky = [r for r in rows if r["leaky"]]
    for k, r in enumerate(comp, 1):
        r["rank_dev_strict_bytes_per_token"] = k
    k = 0
    for r in comp:
        if r["dev_strict"]["G1"]["fail_docs"] == 0:
            k += 1
            r["rank_among_lossless"] = k
        else:
            r["rank_among_lossless"] = None
    lossy = [r["name"] for r in comp + leaky if r["dev_strict"]["G1"]["fail_docs"]]
    diag = diagnose_lossy(lossy)["tokenizers"]
    for r in comp + leaky:
        if r["name"] in diag:
            r["dev_strict"]["G1_diagnosis"] = diag[r["name"]]
    fr = H._frozen_short()
    out = {"what": "External tokenizer baselines on the Hindko dev split (PLAN.md Stage 0, step 5)",
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "frozen": fr, "datasets": ds_info, "harness_code_sha256": [json.loads(x) for x in sorted(harness_code)],
           "ranking": "dev_strict bytes/token, descending; the leaky probe is excluded",
           "nsl": "NSL vs A1-P1-16k is pending (not trained yet). nsl_provisional_vs_%s = tokens / tokens(%s) on the "
                  "same dev_strict documents" % (NSL_REF, NSL_REF),
           "n_baselines_evaluated": len(rows), "n_competitors_ranked": len(comp), "missing": missing,
           "baselines": comp + leaky}
    with open(os.path.join(HERE, "baselines_dev.json"), "w", encoding="utf-8") as f:
        json.dump(H.clean(out), f, ensure_ascii=False, indent=1)
    write_md(out, comp, leaky)
    print("ranked", len(comp), "leaky", len(leaky), "missing", missing)


def write_md(out, comp, leaky):
    ds = out["datasets"]["dev_strict"]
    L = []
    a = L.append
    first = comp[0]["dev_strict"]["overall"] if comp else None
    n_lossless = sum(1 for r in comp if r["dev_strict"]["G1"]["fail_docs"] == 0)
    a("# External tokenizers on Hindko dev (Stage 0 baselines)")
    a("")
    a("Generated %s by `eval/report_baselines.py` from the harness outputs in `eval/results/`. Every number is "
      "measured; nothing on this page touches the test split." % out["generated_utc"])
    a("")
    a("- **Text:** `data/dev_strict.jsonl`, the evaluation set: %d documents, %s UTF-8 bytes, %s whitespace words, "
      "sha256 `%s…`. It is the strict validation split of manifest `%s…`, in canonical form (`hp.normalize` %s)."
      % (ds["docs"], format(first["bytes"], ","), format(first["words"], ","), ds["sha256"][:12],
         out["frozen"]["split_manifest_sha256"][:8], out["frozen"]["normalize_version"]))
    a("- **Tokenizers:** %d external tokenizers (all working entries of `baselines/manifest.json`), each loaded "
      "through `baselines/load_baselines.py` and used with its own native encoder. No BOS/EOS/CLS tokens are added." % (len(comp) + len(leaky)))
    a("  - %d are ranked. **%d of the %d round-trip every dev_strict document exactly** (G1)." % (len(comp), n_lossless, len(comp)))
    top_lossy = [r for r in comp if r["rank_among_lossless"] is None and r["rank_dev_strict_bytes_per_token"] <= 10]
    best_ll = [r for r in comp if r["rank_among_lossless"] is not None][:3]
    a("  - **The raw ranking flatters lossy tokenizers.** %d of the top 10 by bytes/token are not lossless (%s): they "
      "drop line breaks, strip or fold characters, or emit UNK, so they encode less text than they were given. "
      "Among lossless tokenizers the best are %s (column *ll #*)." % (
          len(top_lossy), ", ".join("`%s`" % r["name"] for r in top_lossy),
          ", ".join("`%s` %s bytes/token" % (r["name"], f3(r["dev_strict"]["overall"]["bytes_per_token"])) for r in best_ll)))
    a("  - `hindko-probe-bpe32k` is **LEAKY**: it is this project's earlier probe, trained on the whole corpus "
      "including dev and test. It is reported in its own section and never ranked.")
    a("- **Ranking:** by bytes/token on dev_strict, descending. Bytes/token is a screening metric, never decisive "
      "on its own (PLAN §4.2); the decision metric is LM bits-per-byte (PLAN §5), which needs trained candidates.")
    a("- **NSL:** NSL against A1-P1-16k is pending, because that tokenizer is not trained yet (`harness.py nsl` adds "
      "it later). The column *NSL vs o200k* is a **provisional** substitute: tokens ÷ GPT-4o's tokens on the same documents.")
    a("")
    a("## How to read the columns")
    a("")
    a("- **bytes/tok, chars/tok:** Σ UTF-8 bytes (or characters) ÷ Σ tokens. Higher = fewer tokens.")
    a("- **fertility:** the mean number of tokens that overlap a whitespace word. It is measured on the in-context "
      "encoding of whole documents, through the tokenizers' character offsets. Tokens made only of whitespace "
      "(a lone `▁`, `\\n`) belong to no word.")
    a("- **STRR:** the share of words that are a single token. **cont.:** the share split into ≥ 2 tokens (= 1 − STRR).")
    a("- **G1:** documents with `decode(encode(doc)) == doc`, out of %d. **UNK:** unknown-token count." % ds["docs"])
    a("- **lines b/tok:** bytes/token when every line is encoded on its own, so no line-break tokens exist at all. "
      "It compares tokenizers that silently drop line breaks (turning them into spaces) with lossless ones on equal terms.")
    a("")
    a("## Ranking on dev_strict")
    a("")
    a("*#* = rank by bytes/token; *ll #* = rank among the tokenizers that round-trip all %d documents." % ds["docs"])
    a("")
    a("| # | ll # | tokenizer | provider / family | vocab | algorithm | bytes/tok | chars/tok | NSL vs o200k | fertility | STRR | G1 | UNK | lines b/tok | note |")
    a("|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in comp:
        o = r["dev_strict"]["overall"]
        g = r["dev_strict"]["G1"]
        note = []
        if g["fail_docs"]:
            note.append("**not lossless**")
        if r["newline_handling"] and r["newline_handling"] != "preserved":
            note.append("line breaks: " + r["newline_handling"])
        if r["same_encodings_as"]:
            same = r["same_encodings_as"]
            same = same if isinstance(same, list) else [same]
            note.append("same encodings as " + ", ".join("`%s`" % x for x in same))
        a("| %d | %s | `%s` | %s — %s | %s | %s | **%s** | %s | %s | %s | %s | %s/%d | %s | %s | %s |" % (
            r["rank_dev_strict_bytes_per_token"], r["rank_among_lossless"] or "–", r["name"], r["provider"] or "", r["family"] or "",
            format(r["vocab_size"], ","), r["algorithm"] or "", f3(o["bytes_per_token"]), f3(o["chars_per_token"]),
            f3(r["dev_strict"]["nsl_provisional_vs_gpt-4o"]), f3(o["fertility"]), pct(o["strr"]), g["docs"] - g["fail_docs"],
            g["docs"], format(o["unk_tokens"], ","), f3(o["bytes_per_token_lines"]), "; ".join(note)))
    a("")
    a("## Leaky reference (not a competitor)")
    a("")
    a("| tokenizer | why it is not ranked | vocab | bytes/tok | fertility | STRR | G1 | UNK | lines b/tok |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in leaky:
        o = r["dev_strict"]["overall"]
        g = r["dev_strict"]["G1"]
        a("| `%s` | **LEAKY**: trained on the whole corpus, including dev and test | %s | %s | %s | %s | %s/%d | %s | %s |" % (
            r["name"], format(r["vocab_size"], ","), f3(o["bytes_per_token"]), f3(o["fertility"]), pct(o["strr"]),
            g["docs"] - g["fail_docs"], g["docs"], o["unk_tokens"], f3(o["bytes_per_token_lines"])))
    a("")
    a("Its numbers show what an in-domain vocabulary can reach. They are optimistic, because it saw this very text.")
    a("")
    # lossless detail
    a("## Tokenizers that are not lossless on dev_strict")
    a("")
    a("The table counts, over all %d documents: failed round trips; failures that remain after all whitespace is "
      "collapsed to single spaces (i.e. more than line-break or space handling is lost); non-whitespace characters "
      "lost and added (multiset difference); and UNK tokens. A tokenizer that drops line breaks spends no tokens on "
      "them, which flatters its bytes/token; compare *lines b/tok* instead." % ds["docs"])
    a("")
    a("| tokenizer | G1 fail docs | fail modulo whitespace | chars lost | chars added | UNK | line breaks (measured in failing docs: in → after decode) | most lost / added characters |")
    a("|---|---:|---:|---:|---:|---:|---|---|")

    def short(lst, k=3):
        return ", ".join("%s ×%s" % (x.split(" U+")[0].title().replace("Arabic ", "").replace("Letter ", ""), format(n, ","))
                         for x, n in lst[:k]) or "none"
    for r in comp + leaky:
        g = r["dev_strict"]["G1"]
        if g["fail_docs"] or g["unk_tokens"]:
            dg = r["dev_strict"].get("G1_diagnosis", {})
            nl = ("%s → %s" % (format(dg["line_breaks_in_failing_docs"], ","), format(dg["line_breaks_after_decode"], ","))
                  if dg else "–")
            a("| `%s`%s | %d | %d | %s | %s | %s | %s | lost: %s; added: %s |" % (
                r["name"], " (leaky)" if r["leaky"] else "", g["fail_docs"], g["fail_docs_modulo_whitespace"],
                format(g["nonws_chars_lost"], ","), format(g["nonws_chars_added"], ","), format(g["unk_tokens"], ","), nl,
                short(dg.get("lost_top", [])), short(dg.get("added_top", []))))
    a("")
    a("Full character lists (top 8, with code points): `eval/lossy_diagnosis.json`. What the lists show:")
    a("")
    a("- **NFKC normalizers** (`mt5`, `xlm-r`, `sindhi-xlmr`, `nllb-200`, `claude-legacy`, and the leaky probe) "
      "rewrite every `…` as three full stops (1,149 → 3,447) and expand `ﷺ` into Arabic words.")
    a("- **`urdu-bert-64k`** folds letters (`ئ` → Arabic `ي` ×6,034; `آ` → `ا` ×3,905) and strips harakat "
      "(damma ×4,065, kasra ×2,284). Its top bytes/token is therefore measured on text it partly discards.")
    a("- **`indicbert-v2`**'s tokenizer.json has no WordPiece decoder, so `decode()` leaves the `##` continuation "
      "markers in the text (156,308 `#` characters added).")
    a("- **UNK tokens** (`[UNK]`, `<unk>`) replace characters outside the vocabulary: `mbert` ×2,639, `urdu-gpt2-20k` "
      "×6,875 (mostly line breaks), `nllb-200` ×690, `muril` ×152. `xlm-r` and `sindhi-xlmr` also lose Hindko tone "
      "letters (U+08BF ×16, U+08BE ×10 among their top losses).")
    a("- Every tokenizer above except `claude-legacy` loses all 6,826 line breaks of dev_strict: as spaces, or as "
      "UNK tokens in `urdu-gpt2-20k`. `claude-legacy` keeps them.")
    a("")
    # per source
    a("## Per source (dev_strict)")
    a("")
    srcs = ["newspaper", "book", "web"]
    sizes = {s: comp[0]["dev_strict"]["by_source"][s] for s in srcs}
    a("dev_strict holds %s. Web is small (%d documents), so its numbers are noisy." % (
        ", ".join("%s %d docs / %s bytes" % (s, sizes[s]["docs"], format(sizes[s]["bytes"], ",")) for s in srcs),
        sizes["web"]["docs"]))
    a("")
    a("| # | tokenizer | " + " | ".join("%s b/tok | %s fert." % (s, s) for s in srcs) + " |")
    a("|---:|---|" + "---:|---:|" * len(srcs))
    for r in comp:
        bs = r["dev_strict"]["by_source"]
        a("| %d | `%s` | " % (r["rank_dev_strict_bytes_per_token"], r["name"]) +
          " | ".join("%s | %s" % (f3(bs[s]["bytes_per_token"]), f3(bs[s]["fertility"])) for s in srcs) + " |")
    for r in leaky:
        bs = r["dev_strict"]["by_source"]
        a("| – | `%s` (LEAKY) | " % r["name"] + " | ".join("%s | %s" % (f3(bs[s]["bytes_per_token"]), f3(bs[s]["fertility"])) for s in srcs) + " |")
    a("")
    # per variety
    vs = sorted(comp[0]["dev_strict"]["by_variety"], key=lambda v: -comp[0]["dev_strict"]["by_variety"][v]["docs"])
    perm = comp[0]["dev_permissive"]
    vp = sorted(perm["by_variety"], key=lambda v: -perm["by_variety"][v]["docs"]) if perm else []
    a("## Per language_variety")
    a("")
    a("- dev_strict varieties: %s." % ", ".join("%s %d docs" % (v, comp[0]["dev_strict"]["by_variety"][v]["docs"]) for v in vs))
    if perm:
        a("- dev_permissive (reporting only; it adds the Urdu-variety and other non-strict documents): %s." %
          ", ".join("%s %d docs" % (v, perm["by_variety"][v]["docs"]) for v in vp))
    a("")
    cols = [("strict", v) for v in vs if comp[0]["dev_strict"]["by_variety"][v]["docs"] >= 10] + \
           [("perm.", v) for v in vp if perm and perm["by_variety"][v]["docs"] >= 10]
    a("Bytes/token per variety (varieties with ≥ 10 documents). *perm. all* is the whole dev_permissive set.")
    a("")
    a("| # | tokenizer | " + " | ".join("%s %s" % c for c in cols) + " | perm. all |")
    a("|---:|---|" + "---:|" * (len(cols) + 1))
    for r in comp + leaky:
        cells = []
        for tag, v in cols:
            d = r["dev_strict"]["by_variety"] if tag == "strict" else (r["dev_permissive"] or {}).get("by_variety", {})
            cells.append(f3(d.get(v, {}).get("bytes_per_token")))
        pa = r["dev_permissive"]["overall"]["bytes_per_token"] if r["dev_permissive"] else None
        a("| %s | `%s`%s | %s | %s |" % (r.get("rank_dev_strict_bytes_per_token", "–"), r["name"],
                                        " (LEAKY)" if r["leaky"] else "", " | ".join(cells), f3(pa)))
    a("")
    # robustness
    a("## Robustness under the four perturbations (dev_strict)")
    a("")
    a("For each perturbation: the relative change in the total token count, and the share of the **affected** "
      "words whose segmentation changed. A word's segmentation is compared on the characters it shares with its "
      "perturbed form: which of them fall in the same token, which are split inside a character, and whether a token "
      "crosses into a neighbouring word. The perturbations (`eval/perturb.py`):")
    a("")
    cen = json.load(open(os.path.join(HERE, "census_dev_strict.json"), encoding="utf-8"))["perturbations"]
    a("- **harakat:** delete every Arabic mark U+064B–U+065F and U+0670 (%s marks in %d documents; %s words affected)." % (
        format(cen["harakat"]["edits"], ","), cen["harakat"]["docs_affected"], format(cen["harakat"]["words_affected"], ",")))
    a("- **digits:** swap ASCII ↔ Extended Arabic-Indic digits (%s digits, %s words)." % (
        format(cen["digits"]["edits"], ","), format(cen["digits"]["words_affected"], ",")))
    a("- **punct_space:** toggle the space before `۔` and `،` (%s edits, %s words). The canonical corpus never has "
      "a space there, so on this text the probe *inserts* one. The PLAN wording is \"delete the space\", which would "
      "change nothing here." % (format(cen["punct_space"]["edits"], ","), format(cen["punct_space"]["words_affected"], ",")))
    a("- **zwnj:** insert ZWNJ at the element boundary of %d Perso-Urdu compound words (%d types, e.g. `روزنامہ`, "
      "`زمیندار`, `خوشحال`) found by a fixed list of 3 prefixes and 12 head elements. The corpus has no ZWNJ at all." % (
          cen["zwnj"]["words_affected"], cen["zwnj"]["compound_word_types"]))
    a("")
    a("| # | tokenizer | " + " | ".join("%s Δtok | %s seg-chg" % (p, p) for p in PERTS) + " |")
    a("|---:|---|" + "---:|---:|" * len(PERTS))
    for r in comp + leaky:
        rb = r["dev_strict"]["overall"]["robustness"]
        a("| %s | `%s`%s | " % (r.get("rank_dev_strict_bytes_per_token", "–"), r["name"], " (LEAKY)" if r["leaky"] else "") +
          " | ".join("%s | %s" % (pct(rb[p]["rel_token_change"], 2), pct(rb[p]["seg_change_rate_affected"], 1)) for p in PERTS) + " |")
    a("")
    a("Byte-level and byte-fallback tokenizers show 100% *digits* seg-chg when an ASCII digit (1 byte) becomes an "
      "Urdu digit (2 bytes) that they split into 2 byte tokens. That is a real change of segmentation, not an error.")
    a("")
    # distributional and morphology
    a("## Report-only metrics (dev_strict)")
    a("")
    a("Rényi efficiency rated the worse GPT-2 regex higher in our pilot (PLAN §4.2); it is never used to decide. "
      "Morphology is on the SILVER sets (518 high-confidence and 236 low-confidence words, `morphology/morph_eval.py`). "
      "It is report-only, because MorphScore does not predict LM quality (Arnett et al. 2025).")
    a("")
    a("| # | tokenizer | Rényi α=2.5 | Rényi α=2 | vocab used | utilisation | morph F1 (high) | MorphScore (high) | stem kept (high) | tok/word (high) | docs/s |")
    a("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in comp + leaky:
        o = r["dev_strict"]["overall"]
        m = r["dev_strict"]["morphology"].get("silver_high", {})
        a("| %s | `%s`%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("rank_dev_strict_bytes_per_token", "–"), r["name"], " (LEAKY)" if r["leaky"] else "",
            f3(o["renyi_eff_a2.5"]), f3(o["renyi_eff_a2.0"]), format(o["vocab_used"], ","), pct(o["vocab_utilisation"], 2),
            f3(m.get("boundary_f1")), f3(m.get("morphscore")), f3(m.get("stem_boundary_respected")), f3(m.get("tokens_per_word"), 2),
            "%.0f" % r["dev_strict"]["speed"]["docs_per_s"] if r["dev_strict"]["speed"]["docs_per_s"] else "–"))
    a("")
    a("*stem kept* = `stem_boundary_respected`: the word's stem|suffix boundary is a token boundary and the stem is not split. "
      "*docs/s* is plain `encode()` on this shared laptop CPU, one thread; it is a sanity check only.")
    a("")
    a("## Partial-UTF-8 tokens in the vocabularies (R2, byte-level tokenizers)")
    a("")
    a("| tokenizer | partial-UTF-8 tokens in vocab | occurrences on dev_strict |")
    a("|---|---:|---:|")
    for r in comp + leaky:
        r2 = r["dev_strict"]["R2"]
        if r2.get("partial_utf8_tokens"):
            a("| `%s` | %s | %s |" % (r["name"], format(r2["partial_utf8_tokens"], ","), format(r2["dev_occurrences"], ",")))
    a("")
    a("## What this page does not cover")
    a("")
    a("- **No test-split number.** The test split is evaluated once, in Stage 5.")
    a("- **G2–G5 and R1 were not run on the baselines.** They are health gates for the project's own *candidates* "
      "(PLAN §4.3). The harness runs them with `--gates all`; for external tokenizers, trained on other data, "
      "they would say nothing about our choice.")
    a("- **NSL against A1-P1-16k** waits for that tokenizer (Stage 1).")
    a("- **Bits-per-byte** needs LMs (Stage 3). Every column here is intrinsic.")
    a("- The *transformers* baselines (`muril`, `roberta-urdu`) take their character offsets from the same "
      "tokenizer object's `return_offsets_mapping`. The harness checks that those ids equal the baseline's own "
      "`encode()`; mismatching documents fall back to string alignment (%s)." % ", ".join(
          "`%s`: %s mismatching docs" % (r["name"], r["dev_strict"]["offset_id_mismatch_docs"]) for r in comp + leaky
          if r["dev_strict"]["offset_id_mismatch_docs"] is not None))
    a("")
    a("## Reproduce")
    a("")
    a("```")
    a("set PYTHONIOENCODING=utf-8")
    a("python eval\\run_baselines.py --shard 0 --of 3   (and --shard 1, --shard 2, in parallel; ~10 min each)")
    a("python eval\\report_baselines.py")
    a("```")
    a("")
    a("Harness code sha256 (harness.py / adapters.py / perturb.py / morph_eval.py) for every result on this page: " +
      "; ".join(", ".join("%s `%s…`" % (k, v[:10]) for k, v in hc.items()) for hc in out["harness_code_sha256"]) + ".")
    with open(os.path.join(HERE, "BASELINES_DEV.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
