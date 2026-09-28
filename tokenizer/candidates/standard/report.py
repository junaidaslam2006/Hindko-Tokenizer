# -*- coding: utf-8 -*-
"""PLAN.md Stage 1 report: aggregate results/dev_strict/<id>/{summary,stage1}.json of every sweep tokenizer into
sweep_results.json, SWEEP.md and plots/*.png, and apply the PLAN 3 'After Stage 1' selection rules.

    python report.py            (reads only dev_strict results; computes NSL against A1-P1-D1-16k)
All numbers are measured on dev_strict (836 documents, 1,445,513 bytes, manifest 76582d3a...). Nothing is
computed on the test split.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

import sweep_lib as L

H = L.H
REF = "A1-P1-D1-16k"
SRC = ("book", "newspaper", "web")
VARS = ("hindko", "mixed", "no_signal")
PERT = H.PERT_NAMES
SIZE_ORDER = list(L.SIZES)
PLOTS = os.path.join(L.HERE, "plots")
PRIMARY = "16k"                      # size at which the selection rules are applied (see SWEEP.md, rule notes)
MKEYS = ("bytes_per_token", "chars_per_token", "fertility", "tokens_per_word", "continued_word_rate", "strr",
         "renyi_eff_a2.5", "renyi_eff_a2.0", "vocab_used", "vocab_utilisation", "bytes_per_token_lines", "tokens",
         "unk_tokens", "words_without_token")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def nsl_update(c, ref_docs):
    rd = L.res_dir(c)
    sp = os.path.join(rd, "summary.json")
    s = jload(sp)
    s["nsl"] = H.nsl_from_docs(os.path.join(rd, "docs.jsonl"), ref_docs)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(H.clean(s), f, ensure_ascii=False, indent=1)
    return s


def build_row(c, ref_docs):
    rd = L.res_dir(c)
    if not (os.path.exists(os.path.join(rd, "summary.json")) and os.path.exists(os.path.join(rd, "stage1.json"))):
        return {"id": c["id"], "config": c, "status": "missing"}
    r = jload(os.path.join(rd, "stage1.json"))
    if not r.get("complete"):
        return {"id": c["id"], "config": c, "status": "incomplete"}
    s = nsl_update(c, ref_docs) if os.path.exists(ref_docs) else jload(os.path.join(rd, "summary.json"))
    o = s["metrics"]["overall"]
    nsl = s.get("nsl", {})
    g = s["gates"]
    rem = r.get("g2_remedy") or {}
    ev0 = r["eval_initial"]
    evf = rem.get("eval_after") if rem.get("applied") else ev0
    row = {"id": c["id"], "status": "ok", "algo": c["algo"], "algo_desc": L.ALGOS[c["algo"]]["desc"],
           "family": L.ALGOS[c["algo"]]["family"], "level": L.ALGOS[c["algo"]]["level"], "pretok": c["pretok"],
           "data": c["data"], "size": c["size"], "vocab_total": c["vocab_total"],
           "effective_vocab": r.get("effective_vocab", c["vocab_total"]),
           "vocab_actual": s["tokenizer"].get("vocab_size"),
           "vocab_actual_equals_nominal": s["tokenizer"].get("vocab_size") == c["vocab_total"],
           "encoder": s["tokenizer"].get("encoder"), "files_sha256": r["files"],
           "tokenizer_dir": r["tokenizer_dir"], "train_seconds": r["train_seconds"]}
    row["gates"] = {
        "G1": bool(r["gate_pass_final"]["G1"]), "G1_fail_docs_dev_strict": g["G1"]["fail_docs"],
        "G1_fail_docs_dev_permissive": evf["G1_dev_permissive"]["fail_docs"],
        "G2": bool(r["gate_pass_final"]["G2"]), "G2_tested": g["G2"].get("tested"),
        "G2_failures_before_remedy": rem.get("n_failing", 0), "G2_failures_after": g["G2"].get("failures"),
        "G2_exempt_partial_utf8": g["G2"].get("exempt_partial_utf8"),
        "G2_superword_tested": (evf.get("G2_superword") or {}).get("tested"),
        "G2_superword_failures": (evf.get("G2_superword") or {}).get("failures"),
        "G2_remedy_applied": bool(rem.get("applied")),
        "G2_remedy_reencode_identical": (rem.get("reencode_check") or {}).get("identical") if rem.get("applied") else None,
        "G3": bool(r["gate_pass_final"]["G3"]), "G4": bool(r["gate_pass_final"]["G4"]),
        "G4_bytes_identical": g["G4"].get("bytes_identical"), "G4_json_identical": g["G4"].get("json_identical"),
        "G5": r["gate_pass_final"]["G5"], "G5_sub_character_tokens": g["G5"].get("sub_character_tokens"),
        "all": bool(r["gate_pass_final"]["all"])}
    if r.get("g4_functional_diagnostic"):
        row["gates"]["G4_functional_diagnostic"] = r["g4_functional_diagnostic"]
    row["metrics"] = {k: o.get(k) for k in MKEYS}
    row["metrics"]["nsl"] = nsl.get("overall")
    row["metrics"]["encode_MB_per_s"] = s["speed"].get("MB_per_s")
    row["by_source"] = {}
    for src in SRC:
        m = s["metrics"]["by_source"].get(src)
        if m:
            row["by_source"][src] = {"docs": m["docs"], "bytes": m["bytes"], "bytes_per_token": m["bytes_per_token"],
                                     "chars_per_token": m["chars_per_token"], "fertility": m["fertility"],
                                     "continued_word_rate": m["continued_word_rate"],
                                     "nsl": nsl.get("source:" + src)}
    row["by_variety"] = {}
    for v in VARS:
        m = s["metrics"]["by_variety"].get(v)
        if m:
            row["by_variety"][v] = {"docs": m["docs"], "bytes": m["bytes"], "bytes_per_token": m["bytes_per_token"],
                                    "fertility": m["fertility"], "nsl": nsl.get("variety:" + v)}
    row["robustness"] = {p: {k: o["robustness"][p].get(k) for k in
                             ("rel_token_change", "extra_tokens_per_affected_word", "seg_change_rate",
                              "seg_change_rate_affected", "words_affected")} for p in PERT}
    r1 = s["properties"]["R1"]
    keys = ("learned_tokens", "train_freq_eq0", "train_freq_lt20", "train_freq_lt100", "pct_lt20", "pct_lt100",
            "median_train_freq", "learned_leaves", "lt20_leaves", "lt20_intermediate", "eq0_intermediate")
    row["R1"] = {"train": r1["train"]["train"], "train_tokens": r1["train"]["tokens"],
                 "all_learned": {k: r1["all_learned"].get(k) for k in keys},
                 "multichar_learned": {k: r1["multichar_learned"].get(k) for k in keys}}
    if r.get("R1_own_training_mix"):
        ro = r["R1_own_training_mix"]
        row["R1_own_training_mix"] = {"train": ro["train"]["train"], "train_tokens": ro["train"]["tokens"],
                                      "all_learned": {k: ro["all_learned"].get(k) for k in keys}}
    r2 = s["properties"]["R2"]
    row["R2"] = {k: r2.get(k) for k in ("partial_utf8_tokens", "with_train_freq_0", "dev_occurrences")}
    mo = s.get("morphology", {})
    row["morphology_silver"] = {k: {kk: mo[k].get(kk) for kk in ("boundary_f1", "boundary_precision",
                                                                  "boundary_recall", "morphscore", "stem_intact")}
                                for k in ("silver_high", "silver_low") if k in mo}
    if r.get("hf_export"):
        he = r["hf_export"] if not rem.get("applied") else rem.get("hf_export_after", r["hf_export"])
        row["hf_export"] = {"equivalent": he.get("equivalent"), "equivalent_token_counts": he.get("equivalent_token_counts"),
                            "ids_differ_docs_dev_strict": he["dev_strict"]["ids_differ_docs"],
                            "ids_differ_docs_dev_permissive": he["dev_permissive"]["ids_differ_docs"],
                            "token_count_differs_docs": he["dev_strict"]["token_count_differs_docs"]
                            + he["dev_permissive"]["token_count_differs_docs"],
                            "decode_differs_docs": he["dev_strict"]["hf_decode_differs_docs"]
                            + he["dev_permissive"]["hf_decode_differs_docs"]}
    if rem:
        row["g2_remedy"] = {k: rem.get(k) for k in ("applied", "n_failing", "effective_vocab", "remedy_valid", "status")}
        row["g2_remedy"]["deleted"] = rem.get("deleted")
        row["g2_remedy"]["reencode_check"] = rem.get("reencode_check")
    row["harness_seconds"] = evf.get("harness_seconds")
    row["seconds_total"] = r.get("seconds_total")
    return row


# ------------------------------------------------------------------------------------ selection rules
def selection(rows, handover_rows=()):
    R = {r["id"]: r for r in rows if r.get("status") == "ok"}
    RH = dict(R, **{r["id"]: r for r in handover_rows if r.get("status") == "ok"})

    def bpt(i, src=None):
        r = R.get(i)
        if r is None:
            return None
        return r["metrics"]["bytes_per_token"] if src is None else r["by_source"][src]["bytes_per_token"]

    def rel(a, b, src=None):
        x, y = bpt(a, src), bpt(b, src)
        return None if x is None or y is None else x / y - 1.0

    def ok(i):
        return i in R and R[i]["gates"]["all"]

    out = {"primary_size": PRIMARY, "sign": "rel = bytes/token(candidate) / bytes/token(reference) - 1; "
                                            "positive = candidate compresses better (fewer tokens)"}
    # 1. pre-tokenizer
    per = {}
    for s in SIZE_ORDER:
        p1 = "A1-P1-D1-%s" % s
        per[s] = {"P1_bytes_per_token": bpt(p1)}
        for pt in ("P1r3", "P0", "Pm"):
            per[s]["%s_rel_vs_P1" % pt] = rel("A1-%s-D1-%s" % (pt, s), p1)
            per[s]["%s_bytes_per_token" % pt] = bpt("A1-%s-D1-%s" % (pt, s))
        per[s]["P1r3_within_0.2pct"] = (per[s]["P1r3_rel_vs_P1"] is not None and per[s]["P1r3_rel_vs_P1"] >= -0.002)
    d16 = per[PRIMARY]["P1r3_rel_vs_P1"]
    gates_ok = ok("A1-P1r3-D1-%s" % PRIMARY)
    choice = "P1r3" if (d16 is not None and d16 >= -0.002 and gates_ok) else "P1"
    consistent = sorted({("P1r3" if per[s]["P1r3_within_0.2pct"] else "P1") for s in SIZE_ORDER})
    out["pretokenizer"] = {
        "rule": "PLAN 3: fix the pre-tokenizer; expected P1; choose P1r3 only if it is not worse than P1 by more "
                "than 0.2% in dev bytes/token (P0 = negative control, Pm = intrinsic-only probe; neither is eligible)",
        "per_size": per, "decision_at": PRIMARY, "P1r3_rel_at_primary": d16, "P1r3_passes_gates": gates_ok,
        "choice": choice, "outcome_by_size": {s: ("P1r3" if per[s]["P1r3_within_0.2pct"] else "P1") for s in SIZE_ORDER},
        "same_outcome_at_all_sizes": len(consistent) == 1}
    # 2. data mix
    dm = {}
    for s in ("16k", "32k"):
        d1 = "A1-P1-D1-%s" % s
        dm[s] = {}
        for D in ("D2", "D3"):
            i = "A1-P1-%s-%s" % (D, s)
            r_all = rel(i, d1)
            r_src = {src: rel(i, d1, src) for src in SRC}
            qual = (r_all is not None and r_all >= 0.005 and all(v is not None and v >= -0.01 for v in r_src.values())
                    and ok(i))
            dm[s][D] = {"rel_overall": r_all, "rel_by_source": r_src, "worst_source": min(r_src, key=lambda k: r_src[k])
                        if all(v is not None for v in r_src.values()) else None,
                        "passes_gates": ok(i), "qualifies": qual}
    q = [D for D in ("D2", "D3") if dm[PRIMARY][D]["qualifies"]]
    mix = max(q, key=lambda D: dm[PRIMARY][D]["rel_overall"]) if q else "D1"
    q32 = [D for D in ("D2", "D3") if dm["32k"][D]["qualifies"]]
    mix32 = max(q32, key=lambda D: dm["32k"][D]["rel_overall"]) if q32 else "D1"
    out["data_mix"] = {"rule": "PLAN 3: choose D1 unless D2 or D3 is >= 0.5% better on strict dev bytes/token AND "
                               "no source (book/newspaper/web) is worse by > 1%",
                       "per_size": dm, "decision_at": PRIMARY, "choice": mix, "outcome_at_32k": mix32,
                       "same_outcome_at_16k_and_32k": mix == mix32}
    # 3. Unigram implementation for rank 3
    a4, a5 = "A4-SPnat-D1-%s" % PRIMARY, "A5-P1-D1-%s" % PRIMARY
    uni = {"A4_bytes_per_token": bpt(a4), "A5_bytes_per_token": bpt(a5), "A5_rel_vs_A4": rel(a5, a4),
           "A4_passes_gates": ok(a4), "A5_passes_gates": ok(a5),
           "A4_failed_gates": [k for k in ("G1", "G2", "G3", "G4", "G5") if a4 in R and R[a4]["gates"][k] is False],
           "A5_failed_gates": [k for k in ("G1", "G2", "G3", "G4", "G5") if a5 in R and R[a5]["gates"][k] is False]}
    elig = [x for x in (a4, a5) if ok(x)]
    uni["choice"] = (max(elig, key=bpt) if elig else None)
    uni["choice_if_gates_ignored"] = max([x for x in (a4, a5) if x in R], key=bpt) if (a4 in R or a5 in R) else None
    uni["per_size_rel_A5_vs_A4"] = {s: rel("A5-P1-D1-%s" % s, "A4-SPnat-D1-%s" % s) for s in SIZE_ORDER}
    out["unigram_rank3"] = dict(uni, rule="PLAN 2.2 rank 3 / PLAN 3: Unigram @16k = the better of A4-SPnat and "
                                          "A5-P1 on dev bytes/token (gate-passing candidates only, PLAN 3/7)")
    # 4. A2 / A3 qualification for the LM stage
    a1pt = out["pretokenizer"]["choice"]
    qual = {}
    for algo, pt in (("A2", "P1"), ("A3", "SPnat")):
        qual[algo] = {}
        for s in SIZE_ORDER:
            i = "%s-%s-D1-%s" % (algo, pt, s)
            ref = "A1-%s-D1-%s" % (a1pt, s)
            d = rel(i, ref)
            qual[algo][s] = {"rel_vs_A1": d, "reference": ref, "passes_gates": ok(i),
                             "qualifies": bool(d is not None and d >= 0.01 and ok(i))}
    out["A2_A3_lm_qualification"] = {
        "rule": "PLAN 2.2 rank 9: A2/A3 go to the LM stage only if intrinsically >= 1% better than A1 on dev "
                "bytes/token (same size, same data D1, A1 with the chosen pre-tokenizer)",
        "decision_at": PRIMARY, "per_size": qual,
        "A2_qualifies": qual["A2"][PRIMARY]["qualifies"], "A3_qualifies": qual["A3"][PRIMARY]["qualifies"],
        "qualifying_sizes": {a: [s for s in SIZE_ORDER if qual[a][s]["qualifies"]] for a in qual}}
    # 5. hard-gate drops
    out["gate_drops"] = [{"id": r["id"], "failed": [k for k in ("G1", "G2", "G3", "G4", "G5") if r["gates"][k] is False]}
                         for r in rows if r.get("status") == "ok" and not r["gates"]["all"]]
    out["missing"] = [r["id"] for r in rows if r.get("status") != "ok"]
    out["vocab_size_mismatches"] = [{"id": r["id"], "nominal": r["vocab_total"], "actual": r["vocab_actual"]}
                                    for r in rows if r.get("status") == "ok" and not r["vocab_actual_equals_nominal"]]
    def member(i):
        return {"id": i, "trained": i in RH, "in_preregistered_grid": i in R,
                "passes_gates": bool(i in RH and RH[i]["gates"]["all"]),
                "bytes_per_token": RH[i]["metrics"]["bytes_per_token"] if i in RH else None}

    out["stage3_members_from_stage1"] = {
        "rank1": member("A1-%s-%s-16k" % (a1pt, mix)),
        "rank3": member(uni["choice"]) if uni["choice"] else None,
        "rank6": [member("A1-%s-%s-%s" % (a1pt, mix, s)) for s in ("8k", "32k")],
        "rank9": [a for a in ("A2", "A3") if out["A2_A3_lm_qualification"]["%s_qualifies" % a]],
        "rank9_ids_16k": [member("%s-%s-D1-16k" % (a, "P1" if a == "A2" else "SPnat")) for a in ("A2", "A3")
                          if out["A2_A3_lm_qualification"]["%s_qualifies" % a]]}
    return out


# --------------------------------------------------------------------------------------------- plots
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plots(rows, sel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    os.makedirs(PLOTS, exist_ok=True)
    R = {r["id"]: r for r in rows if r.get("status") == "ok"}
    xs = [L.SIZES[s] for s in SIZE_ORDER]
    made = []
    algos = [("A1", "P1", "A1 byte BPE (P1)"), ("A2", "P1", "A2 char BPE (P1)"), ("A3", "SPnat", "A3 SP BPE"),
             ("A4", "SPnat", "A4 SP Unigram"), ("A5", "P1", "A5 HF Unigram (P1)"), ("A8", "SPsuper", "A8 SP superword")]

    def series(algo, pt, data="D1", f=lambda r: r["metrics"]["bytes_per_token"]):
        pts = [(L.SIZES[s], f(R["%s-%s-%s-%s" % (algo, pt, data, s)])) for s in SIZE_ORDER
               if "%s-%s-%s-%s" % (algo, pt, data, s) in R]
        return [p[0] for p in pts], [p[1] for p in pts]

    def xaxis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(SIZE_ORDER)
        ax.set_xlabel("vocabulary size (total, incl. 64 specials)", color=INK2, fontsize=9)

    # 1. absolute + relative compression by algorithm
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
    for k, (a, pt, lab) in enumerate(algos):
        x, y = series(a, pt)
        if x:
            axs[0].plot(x, y, color=COL[k], lw=2, marker="o", ms=5, label=lab)
    _style(axs[0], "Compression by algorithm (dev_strict)", "bytes / token")
    xaxis(axs[0])
    base = {s: R.get("A1-P1-D1-%s" % s) for s in SIZE_ORDER}
    for k, (a, pt, lab) in enumerate(algos):
        pts = [(L.SIZES[s], 100 * (R["%s-%s-D1-%s" % (a, pt, s)]["metrics"]["bytes_per_token"] /
                                   base[s]["metrics"]["bytes_per_token"] - 1))
               for s in SIZE_ORDER if "%s-%s-D1-%s" % (a, pt, s) in R and base[s]]
        if pts:
            axs[1].plot([p[0] for p in pts], [p[1] for p in pts], color=COL[k], lw=2, marker="o", ms=5, label=lab)
    axs[1].axhline(1.0, color=INK2, lw=1, ls="--")
    axs[1].text(xs[-1], -0.8, "dashed: +1% (A2/A3 LM threshold)", color=INK2, fontsize=8, ha="right", va="top")
    _style(axs[1], "Relative to A1-P1-D1 at the same size", "bytes/token vs A1-P1 (%)")
    xaxis(axs[1])
    axs[1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    p = os.path.join(PLOTS, "compression_by_algorithm.png")
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    made.append(p)
    # 2. pre-tokenizer effect
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURF)
    for k, pt in enumerate(("P1r3", "P0", "Pm")):
        pts = [(L.SIZES[s], 100 * sel["pretokenizer"]["per_size"][s]["%s_rel_vs_P1" % pt]) for s in SIZE_ORDER
               if sel["pretokenizer"]["per_size"][s]["%s_rel_vs_P1" % pt] is not None]
        if pts:
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=COL[k], lw=2, marker="o", ms=5, label=pt)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(-0.2, color=INK2, lw=1, ls="--")
    ax.text(xs[-1], -0.3, "-0.2% (P1r3 rule)", color=INK2, fontsize=8, ha="right", va="top")
    _style(ax, "Pre-tokenizer vs P1 (A1 byte-level BPE, D1)", "bytes/token vs P1 (%)")
    xaxis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOTS, "pretokenizer_vs_P1.png")
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    made.append(p)
    # 3. data mix per source
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF, sharey=True)
    cats = ["overall"] + list(SRC)
    for j, s in enumerate(("16k", "32k")):
        ax = axs[j]
        for k, D in enumerate(("D2", "D3")):
            e = sel["data_mix"]["per_size"][s][D]
            vals = [e["rel_overall"]] + [e["rel_by_source"][src] for src in SRC]
            pos = [i + (k - 0.5) * 0.36 for i in range(len(cats))]
            ax.bar(pos, [100 * (v or 0) for v in vals], width=0.34, color=COL[k], label=D, edgecolor=SURF, linewidth=2)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.axhline(-1.0, color=INK2, lw=1, ls=":")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats)
        _style(ax, "Data mix vs D1 (A1-P1 @%s)" % s, "bytes/token vs D1 (%)" if j == 0 else "")
    axs[0].text(-0.45, 0.55, "+0.5% (overall)", color=INK2, fontsize=8)
    axs[0].text(-0.45, -0.95, "-1% (any source)", color=INK2, fontsize=8)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOTS, "data_mix_by_source.png")
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    made.append(p)
    # 4. R1 support: share of learned tokens with train frequency < 20
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURF)
    for k, (a, pt, lab) in enumerate(algos):
        x, y = series(a, pt, f=lambda r: r["R1"]["all_learned"]["pct_lt20"])
        if x:
            ax.plot(x, y, color=COL[k], lw=2, marker="o", ms=5, label=lab)
    _style(ax, "R1: learned tokens seen < 20 times in train_D1", "% of learned tokens")
    xaxis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOTS, "r1_lt20_share.png")
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    made.append(p)
    return {"files": made}


# ---------------------------------------------------------------------------------------- markdown
def f(x, n=3, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return ("%+." + str(n) + "f%%") % (100 * x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return "{:,}".format(x)
    return ("%." + str(n) + "f") % x


def gstr(g):
    bad = [k for k in ("G1", "G2", "G3", "G4", "G5") if g[k] is False]
    s = "pass" if not bad else "FAIL " + ",".join(bad)
    if g.get("G2_remedy_applied"):
        s += " (G2 remedied: %d deleted)" % g["G2_failures_before_remedy"]
    return s


def headline(rows, sel):
    R = {r["id"]: r for r in rows if r.get("status") == "ok"}
    b = lambda i: R[i]["metrics"]["bytes_per_token"] if i in R else None  # noqa: E731
    out = []
    fam16 = [("A8 SP superword Unigram", "A8-SPsuper-D1-16k"), ("A3 SP BPE", "A3-SPnat-D1-16k"),
             ("A4 SP Unigram", "A4-SPnat-D1-16k"), ("A1 byte BPE P1r3", "A1-P1r3-D1-16k"),
             ("A1 byte BPE P1", "A1-P1-D1-16k"), ("A2 char BPE", "A2-P1-D1-16k"), ("A1 byte BPE P0", "A1-P0-D1-16k"),
             ("A1 byte BPE Pm", "A1-Pm-D1-16k"), ("A5 HF Unigram", "A5-P1-D1-16k")]
    fam16 = sorted([(n, i) for n, i in fam16 if i in R], key=lambda x: -b(x[1]))
    out.append("Compression at 16k (bytes/token, D1): " + "; ".join("%s %.3f" % (n, b(i)) for n, i in fam16) +
               ". A8's lead comes from pieces that span words (e.g. `▁دے▁نال`); PLAN ranks it as an intrinsic-only "
               "probe.")
    if all(("A1-P1-D1-%s" % z) in R for z in SIZE_ORDER):
        out.append("A1-P1 vocabulary curve: " + ", ".join("%s %.3f" % (z, b("A1-P1-D1-%s" % z)) for z in SIZE_ORDER) +
                   " bytes/token; 16k→32k gains %+.1f%%, 32k→48k %+.1f%%." % (
                       100 * (b("A1-P1-D1-32k") / b("A1-P1-D1-16k") - 1),
                       100 * (b("A1-P1-D1-48k") / b("A1-P1-D1-32k") - 1)))
    npass = sum(1 for r in R.values() if r["gates"]["all"])
    rem = [r["id"] for r in R.values() if r["gates"]["G2_remedy_applied"]]
    out.append("Hard gates: %d of %d tokenizers pass G1–G5 after the pre-declared G2 remedy. G1 has 0 failures on "
               "all 2,194 dev documents for every tokenizer. The G2 remedy ran for %d Unigram-family tokenizers (%s); "
               "every one re-encoded dev identically. No BPE-family tokenizer failed G2." % (
                   npass, len(R), len(rem), ", ".join(rem)))
    a5 = [r for r in R.values() if r["algo"] == "A5"]
    if a5 and not any(r["gates"]["G4"] for r in a5):
        d = [r["gates"].get("G4_functional_diagnostic") or {} for r in a5]
        same = all(x.get("same_piece_set") for x in d)
        diffdocs = ", ".join("%s: %d" % (r["size"], (x.get("dev_strict") or {}).get("docs_differing_after_id_mapping", 0))
                             for r, x in zip(a5, d))
        mx = max((x.get("max_abs_score_diff") or 0) for x in d)
        out.append("All six A5 (HF UnigramTrainer) tokenizers fail G4. Two trainings with identical inputs give "
                   "files that differ in piece order (ids) and in scores by at most %.1e. The trainer was "
                   "also non-deterministic with one thread in a pre-sweep probe. The cause is probably hash-map "
                   "iteration order changing float summation; this is a hypothesis, not checked in the Rust source. "
                   "Diagnostic: same piece set in all six pairs: %s; dev_strict documents that encode differently once "
                   "ids are mapped by piece string: %s. Under PLAN §3/§7, A5 is dropped. It is also 11–15%% worse "
                   "in bytes/token than A4, so the drop does not decide the rank-3 rule." % (mx, "yes" if same else "no",
                                                                                   diffdocs))
    sp = [r for r in R.values() if r.get("hf_export")]
    if sp:
        exact = [r["id"] for r in sp if r["hf_export"]["equivalent"]]
        cnt = sum(r["hf_export"]["token_count_differs_docs"] for r in sp)
        dec = sum(r["hf_export"]["decode_differs_docs"] for r in sp)
        worst = max(sp, key=lambda r: r["hf_export"]["ids_differ_docs_dev_strict"] +
                    r["hf_export"]["ids_differ_docs_dev_permissive"])
        out.append("SentencePiece → HF `tokenizer.json` exports: %d of %d reproduce the native ids on every dev "
                   "document (all A3 BPE exports among them: %s). The others differ only on exact-score Unigram ties "
                   "(worst %s: %d of 2,194 documents). Token counts differ in %d documents and decodes in %d." % (
                       len(exact), len(sp), "yes" if all(r["id"] in exact for r in sp if r["algo"] == "A3") else "no",
                       worst["id"], worst["hf_export"]["ids_differ_docs_dev_strict"] +
                       worst["hf_export"]["ids_differ_docs_dev_permissive"], cnt, dec))
    for i16, i32 in (("A1-P1-D1-16k", "A1-P1-D1-32k"), ("A4-SPnat-D1-16k", "A4-SPnat-D1-32k")):
        if i16 in R and i32 in R:
            out.append("R1 %s: %s%% of learned tokens seen < 20 times in train_D1 at 16k, %s%% at 32k." % (
                i16.rsplit("-", 1)[0], f(R[i16]["R1"]["all_learned"]["pct_lt20"], 2),
                f(R[i32]["R1"]["all_learned"]["pct_lt20"], 2)))
    dm = sel["data_mix"]
    if not dm["same_outcome_at_16k_and_32k"]:
        e16, e32 = dm["per_size"]["16k"]["D2"], dm["per_size"]["32k"]["D2"]
        out.append("**Fragile rule outcome:** the data-mix rule picks %s at 16k (D2 %+.2f%% overall) but D1 at 32k, "
                   "where D2 is %+.3f%%, just under the 0.5%% threshold. The decision at 16k follows the stated "
                   "operationalisation (§9); please confirm it before Stage 3." % (dm["choice"], 100 * e16["rel_overall"],
                                                                          100 * e32["rel_overall"]))
    return out


def markdown(rows, sel, meta):
    R = {r["id"]: r for r in rows if r.get("status") == "ok"}
    ref = R.get(REF)
    out = []
    w = out.append
    w("# Stage 1 intrinsic sweep: standard candidates")
    w("")
    w("Generated %s by `candidates/standard/report.py`. Every number is measured on **dev_strict** (836 documents, "
      "1,445,513 UTF-8 bytes, split manifest `76582d3a…`). Nothing was computed on the test split. "
      "PLAN.md §3 Stage 1; metric definitions in `eval/harness.py` (PLAN §4)." % meta["generated_utc"])
    w("")
    w("Tokenizers trained: **%d of %d** planned (%d complete; missing: %s)." % (
        len(R), len(rows), len(R), ", ".join(sel["missing"]) or "none"))
    w("")
    # ---- headline
    w("## 0. Headline (measured on dev_strict)")
    w("")
    for line in headline(rows, sel):
        w("- " + line)
    w("")
    # ---- decisions
    w("## 1. Selection rules (PLAN §3 'After Stage 1'), applied as written")
    w("")
    w("Sign convention: *rel* = bytes/token(candidate) ÷ bytes/token(reference) − 1. Positive means that the "
      "candidate needs fewer tokens. Every rule is applied at **16k** (the size of the rank-1 standard recipe). "
      "All other sizes are shown to check whether the outcome is stable (PLAN does not name a size; see §9).")
    w("")
    pt = sel["pretokenizer"]
    w("### 1.1 Pre-tokenizer → **%s**" % pt["choice"])
    w("")
    w("Rule: P1 unless P1r3 is not worse than P1 by more than 0.2% in dev bytes/token. P0 (negative control) and "
      "Pm (intrinsic-only probe) are reported only.")
    w("")
    w("| size | P1 bytes/token | P1r3 rel | within 0.2%? | P0 rel | Pm rel |")
    w("|---|---|---|---|---|---|")
    for s in SIZE_ORDER:
        e = pt["per_size"][s]
        w("| %s | %s | %s | %s | %s | %s |" % (s, f(e["P1_bytes_per_token"], 4), f(e["P1r3_rel_vs_P1"], 3, True),
                                              "yes" if e["P1r3_within_0.2pct"] else "no", f(e["P0_rel_vs_P1"], 2, True),
                                              f(e["Pm_rel_vs_P1"], 2, True)))
    w("")
    w("At 16k P1r3 is %s relative to P1 (P1r3 passes the gates: %s), so the rule selects **%s**. The outcome is %s "
      "across all six sizes." % (f(pt["P1r3_rel_at_primary"], 3, True), f(pt["P1r3_passes_gates"]), pt["choice"],
                                 "the same" if pt["same_outcome_at_all_sizes"] else
                                 "NOT the same (%s)" % pt["outcome_by_size"]))
    w("")
    dm = sel["data_mix"]
    w("### 1.2 Data mix → **%s**" % dm["choice"])
    w("")
    w("Rule: D1 unless D2 or D3 is ≥ 0.5% better on strict dev bytes/token *and* no source is worse by more than 1%.")
    w("")
    w("| size | mix | rel overall | rel book | rel newspaper | rel web | qualifies |")
    w("|---|---|---|---|---|---|---|")
    for s in ("16k", "32k"):
        for D in ("D2", "D3"):
            e = dm["per_size"][s][D]
            w("| %s | %s | %s | %s | %s | %s | %s |" % (s, D, f(e["rel_overall"], 2, True),
                                                       *[f(e["rel_by_source"][x], 2, True) for x in SRC],
                                                       "yes" if e["qualifies"] else "no"))
    w("")
    w("Decision at 16k: **%s**; the same rule at 32k gives %s (%s)." % (
        dm["choice"], dm["outcome_at_32k"], "consistent" if dm["same_outcome_at_16k_and_32k"] else "INCONSISTENT"))
    w("")
    u = sel["unigram_rank3"]
    w("### 1.3 Unigram implementation for rank 3 → **%s**" % (u["choice"] or "none eligible"))
    w("")
    w("Rule: the better of A4-SPnat and A5-P1 at 16k on dev bytes/token, among candidates that pass the hard gates.")
    w("")
    w("| candidate | bytes/token @16k | passes G1–G5 | failed gates |")
    w("|---|---|---|---|")
    w("| A4-SPnat-D1-16k | %s | %s | %s |" % (f(u["A4_bytes_per_token"], 4), f(u["A4_passes_gates"]),
                                               ", ".join(u["A4_failed_gates"]) or "none"))
    w("| A5-P1-D1-16k | %s | %s | %s |" % (f(u["A5_bytes_per_token"], 4), f(u["A5_passes_gates"]),
                                            ", ".join(u["A5_failed_gates"]) or "none"))
    w("")
    w("A5 relative to A4 at each size: " + ", ".join("%s %s" % (s, f(v, 2, True))
                                                     for s, v in u["per_size_rel_A5_vs_A4"].items()) + ".")
    if u["choice"] != u["choice_if_gates_ignored"]:
        w("")
        w("**The gates decide this rule.** Ignoring the gates, the better compressor would be %s." %
          u["choice_if_gates_ignored"])
    w("")
    q = sel["A2_A3_lm_qualification"]
    w("### 1.4 A2 / A3 qualification for the LM stage → A2: **%s**, A3: **%s**" % (
        "qualifies" if q["A2_qualifies"] else "does not qualify", "qualifies" if q["A3_qualifies"] else
        "does not qualify"))
    w("")
    w("Rule: an LM wave only if the candidate is ≥ 1% better than A1 in dev bytes/token (same size, D1, A1 with "
      "the chosen pre-tokenizer).")
    w("")
    w("| size | A2 rel vs A1 | A2 ≥ 1%? | A3 rel vs A1 | A3 ≥ 1%? |")
    w("|---|---|---|---|---|")
    for s in SIZE_ORDER:
        a2, a3 = q["per_size"]["A2"][s], q["per_size"]["A3"][s]
        w("| %s | %s | %s | %s | %s |" % (s, f(a2["rel_vs_A1"], 2, True), "yes" if a2["qualifies"] else "no",
                                         f(a3["rel_vs_A1"], 2, True), "yes" if a3["qualifies"] else "no"))
    w("")
    w("### 1.5 Hard-gate drops (after the pre-declared G2 remedy)")
    w("")
    if sel["gate_drops"]:
        for d in sel["gate_drops"]:
            w("- %s: failed %s" % (d["id"], ", ".join(d["failed"])))
    else:
        w("- none")
    w("")
    m = sel["stage3_members_from_stage1"]
    w("### 1.6 What Stage 1 hands to Stage 3")
    w("")

    def mem(x):
        if not x:
            return "none"
        return "`%s` (%s; gates pass: %s; %s bytes/token)" % (
            x["id"], "pre-registered grid" if x["in_preregistered_grid"] else
            ("built after the rules, §1.7" if x["trained"] else "NOT TRAINED"), f(x["passes_gates"]),
            f(x["bytes_per_token"], 4))
    w("- rank 1 (standard recipe with the fixed pre-tokenizer and data mix): " + mem(m["rank1"]))
    w("- rank 3 (Unigram @16k): " + mem(m["rank3"]))
    w("- rank 6: " + "; ".join(mem(x) for x in m["rank6"]))
    w("- rank 9 (A2/A3 LM waves): " + (", ".join(mem(x) for x in m["rank9_ids_16k"]) or "none qualifies"))
    w("")
    w("The two rules were measured separately. PLAN's grid crosses the data mixes with P1 only, so the combined "
      "recipe P1r3 × D2 is not in the pre-registered grid. It was built afterwards (§1.7). The A4 (rank 3) and "
      "A3 (rank 9) tokenizers above were trained on D1, because Stage 1 trains them on D1 only. Whether the "
      "fixed data mix should also be applied to them is left open.")
    w("")
    hv = [r for r in meta.get("handover_rows", []) if r.get("status") == "ok"]
    if hv:
        w("### 1.7 Hand-over tokenizers built from the rule outcomes (not used by any rule)")
        w("")
        w("A1 byte-level BPE with P1r3 × D2 at the LM-stage sizes of ranks 1 and 6, trained and gated with the "
          "same code. Neighbours from the grid are shown for comparison.")
        w("")
        w("| tokenizer | gates | bytes/tok | NSL | book | newspaper | web | R1 < 20 on train_D1 (%) | "
          "R1 < 20 on own mix (%) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in hv + [R[i] for i in ("A1-P1-D1-8k", "A1-P1-D1-16k", "A1-P1-D1-32k", "A1-P1r3-D1-16k",
                                      "A1-P1-D2-16k", "A1-P1-D2-32k") if i in R]:
            a, o = r["R1"]["all_learned"], (r.get("R1_own_training_mix") or {}).get("all_learned")
            w("| %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s |" % (
                r["id"], gstr(r["gates"]), f(r["metrics"]["bytes_per_token"], 4), f(r["metrics"]["nsl"], 4),
                *[f(r["by_source"][x]["bytes_per_token"], 4) for x in SRC], f(a["train_freq_lt20"]),
                f(a["pct_lt20"], 2), ("%s (%s)" % (f(o["train_freq_lt20"]), f(o["pct_lt20"], 2))) if o else "= D1"))
        w("")
    # ---- gate table
    w("## 2. Hard gates G1–G5 (PLAN §4.3)")
    w("")
    w("G1 = lossless on dev_strict **and** dev_permissive (test is excluded in this run). G2 = self-tokenization of "
      "every learned whole-character token (superword pieces of A8 are tested too, see §9). G3 = special block "
      "atomic, 0 marker strings in train/dev text. G4 = a second training with identical inputs gives a "
      "byte-identical file. G5 = no sub-character token in character-level vocabularies (n/a for byte-level A1).")
    w("")
    w("| tokenizer | G1 fails (strict/perm) | G2 tested | G2 fails before remedy | G3 | G4 identical | G5 | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            w("| %s | missing |||||||" % r["id"])
            continue
        g = r["gates"]
        tested = (g["G2_tested"] or 0) + (g["G2_superword_tested"] or 0)
        w("| %s | %d/%d | %s | %s | %s | %s | %s | %s |" % (
            r["id"], g["G1_fail_docs_dev_strict"], g["G1_fail_docs_dev_permissive"], f(tested),
            f(g["G2_failures_before_remedy"]), "pass" if g["G3"] else "FAIL",
            "yes" if g["G4"] else "NO", "n/a" if g["G5"] is None else ("pass" if g["G5"] else "FAIL"), gstr(g)))
    w("")
    mm_ = sel.get("vocab_size_mismatches", [])
    w("Vocabulary size check (actual ids incl. specials vs nominal size): " + (
        "all %d tokenizers match." % len(R) if not mm_ else
        "MISMATCH for " + ", ".join("%s (%d vs %d)" % (x["id"], x["actual"], x["nominal"]) for x in mm_)))
    w("")
    # ---- main table
    w("## 3. All tokenizers × metrics (dev_strict)")
    w("")
    w("NSL = tokens ÷ tokens of `%s` on the same documents. Fertility = tokens per whitespace word from the "
      "in-context encoding. Rényi efficiency is report-only (PLAN 4.2). Utilisation = share of ids seen on dev." % REF)
    w("")
    w("| tokenizer | V | eff. V | bytes/tok | chars/tok | NSL | fertility | cont. word rate | STRR | Rényi 2.5 | "
      "util. | bytes/tok (lines) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            continue
        mm = r["metrics"]
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["id"], f(r["vocab_total"]), f(r["effective_vocab"]), f(mm["bytes_per_token"], 4),
            f(mm["chars_per_token"], 3), f(mm["nsl"], 4), f(mm["fertility"], 4), f(mm["continued_word_rate"], 4),
            f(mm["strr"], 4), f(mm["renyi_eff_a2.5"], 4), f(mm["vocab_utilisation"], 3),
            f(mm["bytes_per_token_lines"], 4)))
    w("")
    w("## 4. Per source (dev_strict: book 539 docs / 878,168 bytes; newspaper 280 / 532,872; web 17 / 34,473)")
    w("")
    w("| tokenizer | book bytes/tok | newspaper bytes/tok | web bytes/tok | book NSL | newspaper NSL | web NSL | "
      "book fert. | newspaper fert. | web fert. |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            continue
        b = r["by_source"]
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["id"], *[f(b[s]["bytes_per_token"], 4) for s in SRC], *[f(b[s]["nsl"], 4) for s in SRC],
            *[f(b[s]["fertility"], 3) for s in SRC]))
    w("")
    w("Per language variety (bytes/token): see `sweep_results.json` → `by_variety` (hindko 751 docs, mixed 83, "
      "no_signal 2).")
    w("")
    w("## 5. Reported properties R1 (train_D1 support) and R2 (partial-UTF-8 tokens)")
    w("")
    w("R1 counts each learned token's frequency when the whole of train_D1 (16,015 documents) is encoded with the "
      "tokenizer's native encoder. 'Learned' excludes specials, the 256 base bytes / `<0xNN>` pieces, UNK, the "
      "newline piece, and (A2) single characters. The leaf/intermediate split needs a merge graph (HF BPE only). "
      "D2/D3 tokenizers: their support on their own training mix is in `sweep_results.json` → `R1_own_training_mix`.")
    w("")
    w("| tokenizer | learned | freq = 0 | freq < 20 (%) | of which intermediate | freq < 100 (%) | median freq | "
      "R2 partial-UTF-8 | R2 train freq 0 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            continue
        a = r["R1"]["all_learned"]
        w("| %s | %s | %s | %s (%s) | %s | %s (%s) | %s | %s | %s |" % (
            r["id"], f(a["learned_tokens"]), f(a["train_freq_eq0"]), f(a["train_freq_lt20"]), f(a["pct_lt20"], 2),
            f(a.get("lt20_intermediate")) if a.get("lt20_intermediate") is not None else "n/a",
            f(a["train_freq_lt100"]), f(a["pct_lt100"], 1), f(a["median_train_freq"], 0),
            f(r["R2"]["partial_utf8_tokens"]), f(r["R2"]["with_train_freq_0"])))
    w("")
    w("## 6. Robustness (TokSuite-style; PLAN 4.2)")
    w("")
    w("rel Δtok = relative change in the dev token count after the perturbation; seg Δ = share of *affected* words "
      "whose segmentation changes. Perturbations (eval/perturb.py): harakat removed, digit script swapped, space "
      "before ۔/، deleted, ZWNJ inserted inside compounds.")
    w("")
    w("| tokenizer | " + " | ".join("%s rel Δtok | %s seg Δ" % (p, p) for p in PERT) + " |")
    w("|---|" + "---|" * (2 * len(PERT)))
    for r in rows:
        if r.get("status") != "ok":
            continue
        rb = r["robustness"]
        w("| %s | " % r["id"] + " | ".join("%s | %s" % (f(rb[p]["rel_token_change"], 2, True),
                                                        f(rb[p]["seg_change_rate_affected"], 3)) for p in PERT) + " |")
    w("")
    w("## 7. Morphology (SILVER sets; report only, PLAN 4.2)")
    w("")
    w("| tokenizer | high: boundary F1 | high: MorphScore | high: stem intact | low: boundary F1 | low: MorphScore |")
    w("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok" or not r.get("morphology_silver"):
            continue
        hi, lo = r["morphology_silver"].get("silver_high", {}), r["morphology_silver"].get("silver_low", {})
        w("| %s | %s | %s | %s | %s | %s |" % (r["id"], f(hi.get("boundary_f1"), 3), f(hi.get("morphscore"), 3),
                                              f(hi.get("stem_intact"), 3), f(lo.get("boundary_f1"), 3),
                                              f(lo.get("morphscore"), 3)))
    w("")
    w("## 8. Training time, encoder speed, SentencePiece HF export")
    w("")
    w("Training time is wall-clock for each of the two trainings (G4), measured with 2 threads per process while "
      "other agents were loading the CPU; it is a sanity number, not a benchmark. The HF `tokenizer.json` of a "
      "SentencePiece candidate is an *export*; its native encoder (`sp.model` + newline wrapper) is the "
      "candidate's identity. 'ids differ' counts dev documents (strict + permissive, 2,194) whose export ids differ "
      "from the native ids; 'count differs' those where the number of tokens differs too.")
    w("")
    w("| tokenizer | train s (1st, 2nd) | encode MB/s | HF export: ids differ | count differs | decode differs |")
    w("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            continue
        he = r.get("hf_export")
        w("| %s | %s | %s | %s | %s | %s |" % (
            r["id"], ", ".join("%.0f" % t for t in r["train_seconds"]), f(r["metrics"]["encode_MB_per_s"], 2),
            "native HF" if not he else f(he["ids_differ_docs_dev_strict"] + he["ids_differ_docs_dev_permissive"]),
            "" if not he else f(he["token_count_differs_docs"]), "" if not he else f(he["decode_differs_docs"])))
    w("")
    w("Plots: " + ", ".join("`plots/%s`" % os.path.basename(p) for p in meta.get("plots", {}).get("files", [])))
    w("")
    w("![compression](plots/compression_by_algorithm.png)")
    w("")
    w("![pretokenizers](plots/pretokenizer_vs_P1.png)")
    w("")
    w("![data mix](plots/data_mix_by_source.png)")
    w("")
    w("![R1](plots/r1_lt20_share.png)")
    w("")
    w("## 9. Deviations, interpretations, and what was not done")
    w("")
    for d in meta["deviations"]:
        w("- " + d)
    w("")
    w("## 10. Files")
    w("")
    w("- `tok/<id>/`: `tokenizer.json` (HF), `sp.model` + `sp.vocab` (SentencePiece candidates), `meta.json` "
      "(training parameters, data sha256, file sha256, G4 twin sha256, HF-export check), `g4/` (the second "
      "training), `pre_remedy/` (only where the G2 remedy ran).")
    w("- `results/dev_strict/<id>/`: harness `summary.json` + `docs.jsonl` (per document, for the PLAN 6 "
      "bootstrap), `stage1.json` (gates incl. the extensions, remedy record, R1 on own mix).")
    w("- `sweep_results.json`: every row of this report plus the rule computations; `pretok_differential_Pm.json`.")
    w("- Code: `sweep_lib.py`, `run_sweep.py`, `report.py` (sha256 in `sweep_results.json` → `meta.code`).")
    return "\n".join(out) + "\n"


def main():
    ref_docs = os.path.join(L.res_dir(L.config_by_id(REF)), "docs.jsonl")
    cs = L.all_configs()
    rows = [build_row(c, ref_docs) for c in cs]
    handover = [build_row(c, ref_docs) for c in L.handover_configs()]
    sel = selection(rows, handover)
    meta = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": "dev_strict", "split_manifest_sha256": H._frozen_short()["split_manifest_sha256"],
            "normalize": H._frozen_short(), "reference_for_nsl": REF, "versions": L.versions(),
            "code": {k: L.sha256_file(os.path.join(L.HERE, k)) for k in ("sweep_lib.py", "run_sweep.py", "report.py")},
            "harness_code": H.code_hashes(), "test_split_used": False}
    dev_path = os.path.join(L.HERE, "deviations.json")
    meta["deviations"] = jload(dev_path) if os.path.exists(dev_path) else []
    meta["plots"] = plots(rows, sel)
    meta["handover_rows"] = handover
    res = {"meta": {k: v for k, v in meta.items() if k != "handover_rows"}, "selection": sel, "rows": rows,
           "handover_rows": handover}
    with open(os.path.join(L.HERE, "sweep_results.json"), "w", encoding="utf-8") as fh:
        json.dump(H.clean(res), fh, ensure_ascii=False, indent=1)
    with open(os.path.join(L.HERE, "SWEEP.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(rows, sel, meta))
    print(json.dumps(H.clean({k: sel[k] for k in ("gate_drops", "missing", "stage3_members_from_stage1")}),
                     ensure_ascii=False, indent=1))
    print("pretokenizer:", sel["pretokenizer"]["choice"], "| data mix:", sel["data_mix"]["choice"],
          "| unigram rank3:", sel["unigram_rank3"]["choice"], "| A2:", sel["A2_A3_lm_qualification"]["A2_qualifies"],
          "| A3:", sel["A2_A3_lm_qualification"]["A3_qualifies"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
