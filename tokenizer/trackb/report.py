# -*- coding: utf-8 -*-
"""Aggregate the Track B evaluation, apply the pre-registered knee rule (KNEE_RULE.md) and draw the curves.

    python report.py      -> results/summary.json, results/knee.json, figures/*.png|svg, results/tables.md
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402

FIG = os.path.join(HERE, "figures")
SUPPORT_CAP_PCT = 10.0          # KNEE_RULE.md 2
SUPPORT_FREQ = 100
CURATED_CLEAN = ("english_wiki", "python_code", "other_scripts", "english_dev")
BULK_CLEAN = ("bulk_code", "bulk_english")
URDU = ("urdu_generated", "urdu_news_dev", "urdu_book_dev")


def load_results():
    out = {}
    for base in C.BASE_ORDER:
        for k in [0] + C.K_GRID:
            p = os.path.join(C.RESULTS, base, "k%d" % k, "trackb_eval.json")
            if os.path.exists(p):
                out.setdefault(base, {})[k] = C.load_json(p)
    return out


def eligibility(r):
    if r["k"] == 0:
        return {"eligible": None}
    g1 = r["gates"]["G1"]["pass"]
    g2 = r["gates"]["G2_new"]["pass"]
    eq = r["equivalence"]["pass"]
    # curated English / code / other-script sets: no document may change. Bulk sets (k = K_MAX): no document
    # WITHOUT Arabic-script characters may change (a few package READMEs contain Arabic-script words, e.g. language
    # names; those are expected to change like Urdu and are reported separately).
    ni = all(r["checks"][s].get("changed", 0) == 0 for s in CURATED_CLEAN)
    if r["k"] == C.K_MAX:
        ni = ni and all(r["checks"][s].get("changed", 0) - r["checks"][s].get("changed_with_arabic", 0) == 0
                        for s in BULK_CLEAN if s in r["checks"])
    return {"eligible": bool(g1 and g2 and eq and ni), "G1": g1, "G2_new": g2, "equivalence": eq,
            "non_interference": ni}


def kneedle(ks, tokens):
    t0 = tokens[0]
    red = [1 - t / t0 for t in tokens]
    rmax = red[-1]
    x = [k / ks[-1] for k in ks]
    y = [r / rmax for r in red]
    d = [yy - xx for xx, yy in zip(x, y)]
    i = max(range(len(ks)), key=lambda j: d[j])
    return ks[i], {"x": x, "y": y, "diff": d, "token_reduction": red}


def knee_for(base, rs):
    ks = [0] + [k for k in C.K_GRID if k in rs]
    if ks[-1] != C.K_MAX:
        return None
    tokens = [rs[k]["dev"]["tokens"] for k in ks]
    kk, kd = kneedle(ks, tokens)
    elig = {k: eligibility(rs[k]) for k in ks[1:]}
    sup = {k: rs[k]["R1_new"]["pct_lt100"] for k in ks[1:]}
    ok = [k for k in ks[1:] if elig[k]["eligible"] and sup[k] <= SUPPORT_CAP_PCT]
    cap = max(ok) if ok else None
    chosen = min(kk, cap) if cap is not None else 1024
    return {"kneedle_k": kk, "kneedle": kd, "support_pct_lt100": sup, "support_cap_pct": SUPPORT_CAP_PCT,
            "largest_eligible_k_within_cap": cap, "eligibility": elig, "chosen_k": chosen,
            "flag": None if cap is not None else "no k meets the support cap; k = 1024 by rule"}


def fmt(x, nd=3):
    return "–" if x is None else ("{:,.%df}" % nd).format(x)


def pct(x, nd=1):
    return "–" if x is None else ("%." + str(nd) + "f%%") % (100 * x)


def draw(res, knees):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    colors = {"qwen-3": "#1f77b4", "qwen-3.5": "#17becf", "llama-3": "#d62728", "gemma-3": "#2ca02c",
              "gemma-4": "#98df8a"}
    panels = [("bytes_per_token", "dev_strict bytes/token (higher = fewer tokens)"),
              ("fertility", "fertility (tokens per word)"), ("strr", "STRR (share of words that are 1 token)"),
              ("pct_lt100", "new tokens with < 100 train occurrences (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for base, rs in res.items():
            ks = sorted(rs)
            if key == "pct_lt100":
                ks = [k for k in ks if k]
                ys = [rs[k]["R1_new"]["pct_lt100"] for k in ks]
            else:
                ys = [rs[k]["dev"][key] for k in ks]
            ls = "--" if base == "gemma-4" else "-"
            ax.plot(ks, ys, ls, marker="o", color=colors.get(base), label=base)
            kn = knees.get(base, {}) or {}
            if kn.get("chosen_k") in rs and (key != "pct_lt100" or kn["chosen_k"]):
                kc = kn["chosen_k"]
                yc = rs[kc]["R1_new"]["pct_lt100"] if key == "pct_lt100" else rs[kc]["dev"][key]
                ax.plot([kc], [yc], marker="*", markersize=16, color=colors.get(base), markeredgecolor="black")
        if key == "pct_lt100":
            ax.axhline(SUPPORT_CAP_PCT, color="grey", lw=0.8, ls=":")
        ax.set_xscale("symlog", linthresh=1024)
        ax.set_xticks([0] + C.K_GRID)
        ax.set_xticklabels(["0", "1k", "2k", "4k", "8k", "16k"])
        ax.set_xlabel("new tokens k")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Track B: continued-BPE extension on Hindko (dev_strict; star = chosen knee)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "trackb_curves.png"), dpi=110)
    fig.savefig(os.path.join(FIG, "trackb_curves.svg"))
    plt.close(fig)


def tables(res, knees):
    L = []
    L.append("### Dev curves (dev_strict, 836 documents, 1,445,513 bytes)\n")
    for base, rs in res.items():
        L.append("**%s** (%s)\n" % (base, C.BASES[base]["models"]))
        L.append("| k | vocab | bytes/token | vs base | NSL | fertility | STRR | cont. words | new tokens on dev | "
                 "train_D1 tokens | new < 20 | new < 100 | new = 0 | < 100 intermediate | partial UTF-8 | G1 | G2 new | equiv. |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        b0 = rs[0]["dev"]["bytes_per_token"]
        for k in sorted(rs):
            r = rs[k]
            d = r["dev"]
            if k:
                s = r["R1_new"]
                u = r["dev_new_usage"]
                g2 = r["gates"]["G2_new"]
                row = ["%s" % "{:,}".format(k), "{:,}".format(r["vocab_size_total"]), fmt(d["bytes_per_token"]),
                       "+%.1f%%" % (100 * (d["bytes_per_token"] / b0 - 1)), fmt(d["nsl_vs_base"]),
                       fmt(d["fertility"]), pct(d["strr"]), pct(d["continued_word_rate"]),
                       "%s (%s of dev tokens)" % (pct(u["share_new_seen_on_dev"]), pct(u["share_dev_tokens_that_are_new"])),
                       "{:,}".format(r["train"]["tokens"]),
                       "%d (%.1f%%)" % (s["train_freq_lt20"], s["pct_lt20"]),
                       "%d (%.1f%%)" % (s["train_freq_lt100"], s["pct_lt100"]), "%d" % s["train_freq_eq0"],
                       "%d" % s["lt100_intermediate"], "%d" % s["partial_utf8_new"],
                       "%d/%d" % (r["gates"]["G1"]["docs"] - r["gates"]["G1"]["fail_docs"], r["gates"]["G1"]["docs"]),
                       "%d fail / %d" % (g2["failures"], g2["tested"]),
                       "%d/%d" % (r["equivalence"]["docs"] - r["equivalence"]["mismatch_docs"], r["equivalence"]["docs"])]
            else:
                row = ["0 (base)", "{:,}".format(r["vocab_size_total"]), fmt(d["bytes_per_token"]), "–", "1.000",
                       fmt(d["fertility"]), pct(d["strr"]), pct(d["continued_word_rate"]), "–",
                       "{:,}".format(r["train"]["tokens"]), "–", "–", "–", "–", "–",
                       "%d/%d" % (r["gates"]["G1"]["docs"] - r["gates"]["G1"]["fail_docs"], r["gates"]["G1"]["docs"]),
                       "–", "–"]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L.append("### Per source at the chosen k (bytes/token; base → extended)\n")
    L.append("| base | k | book | newspaper | web |")
    L.append("|---|---:|---|---|---|")
    for base, rs in res.items():
        kc = (knees.get(base) or {}).get("chosen_k")
        if kc not in rs:
            continue
        cells = []
        for s in ("book", "newspaper", "web"):
            a = rs[0]["dev"]["by_source"][s]["bytes_per_token"]
            b = rs[kc]["dev"]["by_source"][s]["bytes_per_token"]
            cells.append("%.3f → %.3f (+%.1f%%)" % (a, b, 100 * (b / a - 1)))
        L.append("| %s | %s | %s |" % (base, "{:,}".format(kc), " | ".join(cells)))
    L.append("")
    L.append("### Non-interference (documents whose ids differ from the base; bulk sets tested at k = 16,384)\n")
    L.append("| base | k | English (12 gen.) | English in corpus dev (7) | Python stdlib (10) | other scripts (7) | "
             "bulk stdlib .py (678) | bulk English READMEs (228) | Urdu generated (9) | Urdu news dev (22) | Urdu books dev (358) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for base, rs in res.items():
        for k in sorted(rs):
            if not k:
                continue
            c = rs[k]["checks"]

            def ch(s):
                if s not in c:
                    return "(not run)"
                x = c[s]
                n_ar = x.get("docs_with_arabic", 0)
                if not n_ar:
                    return "%d" % x.get("changed", 0)
                return "%d of %d without Arabic script (%d of %d with it)" % (
                    x.get("changed", 0) - x.get("changed_with_arabic", 0), x["docs"] - n_ar,
                    x.get("changed_with_arabic", 0), n_ar)

            def ur(s):
                x = c[s]
                return "%d changed, %s fewer tokens, coarsening %d/%d, round trip %d/%d" % (
                    x.get("changed", 0), pct(x["token_reduction"]), x.get("coarsening_ok", 0), x.get("changed", 0),
                    x.get("roundtrip_ok", 0), x.get("changed", 0))
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                base, "{:,}".format(k), ch("english_wiki"), ch("english_dev"), ch("python_code"), ch("other_scripts"),
                ch("bulk_code") if k == C.K_MAX else "(= 16k)", ch("bulk_english") if k == C.K_MAX else "(= 16k)",
                ur("urdu_generated"), ur("urdu_news_dev"), ur("urdu_book_dev")))
    L.append("")
    ap = os.path.join(C.RESULTS, "audit.json")
    if os.path.exists(ap):
        au = C.load_json(ap)
        L.append("### Audit (`audit.py`): scope of every new merge, and every Unicode code point\n")
        L.append("Static: new tokens by class of their bytes (*Arabic* = contains a complete Arabic-script character or "
                 "ends in a byte prefix that only Arabic-script characters complete; *other* = can occur in text "
                 "without Arabic script). Code points: all 1,112,064 Unicode scalar values, alone and after a space "
                 "(2,224,128 strings), encoded by base and extension; strings whose ids change.\n")
        L.append("| base | k | new tokens: Arabic / partial-UTF-8 with Arabic-only prefix / other partial-UTF-8 / other | code points changed: Arabic-script / other |")
        L.append("|---|---:|---|---|")
        for base in res:
            st = au["static"].get(base)
            cp = au["codepoints"].get(base, {})
            if not st:
                continue
            for k in C.K_GRID:
                s = st["by_k"][str(k)] if str(k) in st["by_k"] else st["by_k"][k]
                c = cp.get(str(k)) or cp.get(k)
                L.append("| %s | %s | %d / %d / %d / %d | %s |" % (
                    base, "{:,}".format(k), s["arabic"], s["partial"], s.get("partial_out", 0), s["other"],
                    ("%d / **%d**" % (c["changed_arabic_script"], c["changed_other"])) if c else "not run"))
        L.append("")
    L.append("### Knee (KNEE_RULE.md)\n")
    L.append("| base | Kneedle y − x at 1k / 2k / 4k / 8k / 16k | Kneedle knee | largest eligible k with ≤ 10% new tokens < 100 | chosen k |")
    L.append("|---|---|---:|---:|---:|")
    for base, kn in knees.items():
        if not kn:
            continue
        L.append("| %s | %s | %s | %s | **%s** |" % (
            base, " / ".join("%.3f" % v for v in kn["kneedle"]["diff"][1:]), "{:,}".format(kn["kneedle_k"]),
            "{:,}".format(kn["largest_eligible_k_within_cap"]) if kn["largest_eligible_k_within_cap"] else "none",
            "{:,}".format(kn["chosen_k"])))
    L.append("")
    return "\n".join(L)


def main():
    res = load_results()
    knees = {b: knee_for(b, rs) for b, rs in res.items()}
    kn_out = {"rule_file": os.path.join(HERE, "KNEE_RULE.md"), "rule_sha256": C.sha256_file(os.path.join(HERE, "KNEE_RULE.md")),
              "knees": knees}
    C.dump_json(kn_out, os.path.join(C.RESULTS, "knee.json"))
    summ = {}
    for b, rs in res.items():
        summ[b] = {}
        for k, r in rs.items():
            summ[b][k] = {"vocab": r["vocab_size_total"], "tokenizer_sha256": r["tokenizer_sha256"], "dev": {
                x: r["dev"][x] for x in ("tokens", "bytes_per_token", "chars_per_token", "fertility", "strr",
                                        "continued_word_rate", "nsl_vs_base")},
                "train_tokens": r["train"]["tokens"], "R1_new": r.get("R1_new"), "eligibility": eligibility(r)}
    C.dump_json(summ, os.path.join(C.RESULTS, "summary.json"))
    draw(res, knees)
    t = tables(res, knees)
    with open(os.path.join(C.RESULTS, "tables.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(t)
    print(t)
    print(json.dumps({b: (k or {}).get("chosen_k") for b, k in knees.items()}))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
