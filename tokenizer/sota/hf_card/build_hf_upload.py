"""Build F:\\Hindko\\hf_upload\\ (Hugging Face upload folder) from already-verified study outputs.

Reads only; writes only under F:\\Hindko\\hf_upload\\. No corpus text except the three short
example sentences of ROBUSTNESS.md section 6. Run: PYTHONIOENCODING=utf-8 python build_hf_upload.py
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

REL = Path(r"F:\Hindko\tokenizer")
SOTA = Path(r"F:\Hindko\_tokenizer\sota")
OUT = Path(r"F:\Hindko\hf_upload")
EVAL = OUT / "eval"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


OUT.mkdir(exist_ok=True)
EVAL.mkdir(exist_ok=True)
(OUT / "examples").mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. tokenizer files
manifest = jload(REL / "RELEASE_MANIFEST.json")["tokenizer_files"]
copied = {}
for name, want in manifest.items():
    src, dst = REL / name, OUT / name
    s = sha(src)
    assert s == want, f"{name}: source sha {s} != manifest {want}"
    shutil.copyfile(src, dst)
    d = sha(dst)
    assert d == want, f"{name}: copy sha {d} != manifest {want}"
    copied[name] = d
print("copied + verified:", len(copied), "files")

# ---------------------------------------------------------------- 2. .gitattributes (HF default LFS patterns)
GITATTR = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
tokenizer.json filter=lfs diff=lfs merge=lfs -text
"""
(OUT / ".gitattributes").write_text(GITATTR, encoding="utf-8", newline="\n")

# ---------------------------------------------------------------- 3. competitor tables
cf = jload(SOTA / "competitors" / "competitors_final.json")
rel = cf["released"]
cols = ["rank_lossless", "tokenizer", "repo", "provider", "origin", "algorithm", "vocab_size",
        "bytes_per_token", "fertility", "strr", "tokens", "g1_lossless_docs_of_491", "lossless",
        "unk_share", "released_uses_fewer_tokens_pct", "aliases", "n_hub_repos"]
rows = []
for b in cf["all_behaviours"]:
    rows.append({
        "tokenizer": b["representative"], "repo": b.get("repo", ""), "provider": b.get("provider", ""),
        "origin": b.get("origin", ""), "algorithm": b.get("model_type", ""), "vocab_size": b.get("vocab_size", ""),
        "bytes_per_token": round(b["bytes_per_token"], 4), "fertility": round(b["fertility"], 4),
        "strr": round(b["strr"], 4) if b.get("strr") is not None else "", "tokens": b["tokens"],
        "g1_lossless_docs_of_491": b["g1_pass_docs"], "lossless": b["g1_pass_docs"] == 491,
        "unk_share": round(b.get("unk_share") or 0.0, 4),
        "released_uses_fewer_tokens_pct": round(100 * b["released_uses_fewer_tokens"], 2),
        "aliases": ";".join(b.get("aliases") or []), "n_hub_repos": b.get("n_repos", "")})
rows.append({"tokenizer": "hindko-tokenizer 1.0.0 (released)", "repo": "junaid008/hindko-tokenizer",
             "provider": "this release", "origin": "released", "algorithm": "Unigram",
             "vocab_size": rel["vocab_size"], "bytes_per_token": round(rel["bytes_per_token"], 4),
             "fertility": round(rel["fertility"], 4), "strr": round(rel["strr"], 4), "tokens": rel["tokens"],
             "g1_lossless_docs_of_491": 491, "lossless": True, "unk_share": 0.0,
             "released_uses_fewer_tokens_pct": 0.0, "aliases": "", "n_hub_repos": ""})
lossless = sorted([r for r in rows if r["lossless"]], key=lambda r: -r["bytes_per_token"])
lossy = sorted([r for r in rows if not r["lossless"]], key=lambda r: -r["bytes_per_token"])
for i, r in enumerate(lossless, 1):
    r["rank_lossless"] = i
for r in lossy:
    r["rank_lossless"] = ""
with open(EVAL / "competitors_test_strict.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in lossless + lossy:
        w.writerow(r)
assert lossless[0]["origin"] == "released"
print("competitors csv:", len(lossless), "lossless +", len(lossy), "lossy rows")

keep = ["representative", "aliases", "provider", "repo", "model_type", "vocab_size", "bytes_per_token",
        "fertility", "strr", "tokens", "g1_pass_docs", "released_uses_fewer_tokens", "n_repos"]
slim = lambda L: [{k: x.get(k) for k in keep} for x in L]
jdump({
    "what": "Held-out Hindko (test_strict) token-count ranking of the released tokenizer against every external "
            "tokenizer measured; compact copy of sota/competitors/competitors_final.json (full per-behaviour table: "
            "competitors_test_strict.csv). Screening metrics only; a tokenizer is not a model.",
    "generated_utc_source": cf["generated_utc"],
    "dataset": {**cf["dataset"], "path": "data/test_strict.jsonl (study folder; not published)"},
    "method": {**cf["method"], "metrics": "bytes/token = UTF-8 bytes / tokens; fertility = harness definition: sum over whitespace words of the number of tokens overlapping the word, divided by words (standalone '▁' and newline tokens belong to no word; the plain ratio tokens / whitespace words is reported separately as tokens_per_word, released 1.161 vs fertility 1.135); every tokenizer scored with the same definition; each tokenizer with its own native encoder, no BOS/EOS/CLS (eval/harness.py)"},
    "released": rel, "verdict": cf["verdict"], "counts": cf["counts"],
    "top15_lossless_behaviours": slim(cf["top15_lossless_behaviours"]),
    "study_baselines_deduplicated": slim(cf["study_baselines_deduplicated"]),
    "search_coverage": cf["search_coverage"],
}, EVAL / "competitors_summary.json")

# ---------------------------------------------------------------- 4. LM tables (copies of release eval/ CSVs + key contrasts)
for n in ["dev_lm_confirm_ranking.csv", "test_lm_confirm.csv", "dev_lm_large.csv", "test_lm_large.csv",
          "health_gates_and_support.csv"]:
    shutil.copyfile(REL / "eval" / n, EVAL / n)
dl = jload(REL / "eval" / "dev_lm_large.json")
tl = jload(REL / "eval" / "test_lm_large.json")
jdump({
    "what": "Key LM bits-per-byte contrasts cited in README.md (hierarchical cluster bootstrap, 10,000 replicates). "
            "Full tables: dev_lm_confirm_ranking.csv, test_lm_confirm.csv, dev_lm_large.csv, test_lm_large.csv.",
    "dev_large_top_set_pairs": dl.get("top_set_pairs"),
    "test_large_boot_2seeds_3cands": tl.get("boot_2seeds_3cands"),
    "test_large_boot_seed1_4cands": tl.get("boot_seed1_4cands"),
    "test_large_parametric_all_seeds": tl.get("parametric_all_seeds"),
}, EVAL / "lm_key_contrasts.json")

# ---------------------------------------------------------------- 5. robustness (numbers only) + example segmentations
rb = jload(SOTA / "robustness" / "robustness.json")


def numeric_only(o):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(v, (int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, (dict, list)):
                nv = numeric_only(v)
                if nv not in ({}, []):
                    out[k] = nv
            elif isinstance(v, str) and len(v) <= 80 and all(ord(c) < 0x590 for c in v):
                out[k] = v  # short ASCII/Latin labels only
        return out
    if isinstance(o, list):
        return [numeric_only(x) if isinstance(x, (dict, list)) else x for x in o
                if isinstance(x, (dict, list, int, float))]
    return o


jdump({
    "what": "Out-of-distribution, perturbation, code-mixed and other-language token metrics (numbers only) from "
            "sota/robustness/robustness.json; report: ROBUSTNESS.md. Input normalized with hp.normalize 1.0.1 for "
            "Hindko/Urdu/Punjabi/Saraiki/Pashto; English and code as found.",
    "generated_utc_source": rb["generated_utc"],
    "tokenizers": rb["tokenizers"],
    "test_strict": numeric_only(rb["test_strict"]),
    "sets": numeric_only(rb["sets"]),
    "perturbations": numeric_only(rb["perturbations"]),
    "perturbation_definitions": rb["perturbation_definitions"],
    "arabic_keyboard_then_normalize": rb["arabic_keyboard_then_normalize"],
    "arabic_keyboard_mitigation": rb["arabic_keyboard_mitigation"]["tokenizers"],
    "codemixed": numeric_only(rb["codemixed"]),
    "summary_numbers": rb["summary_numbers"],
    "word_oov_vs_train_D2": rb["word_oov_vs_train_D2"],
}, EVAL / "robustness_summary.json")

from tokenizers import Tokenizer  # noqa: E402

tk = Tokenizer.from_file(str(OUT / "tokenizer.json"))
ex = rb["examples"]
sents = [tk.decode(e["ids"], skip_special_tokens=False) for e in ex["released"]]
for s, e in zip(sents, ex["released"]):
    assert tk.encode(s).ids == e["ids"], "example ids do not reproduce"
jdump({
    "what": "Three natural sentences from the strict test split (held out from tokenizer training), segmented by each "
            "tokenizer's own encoder, no special tokens. '\u2581' = space carried by the token; '\u25af' = one byte of a "
            "letter split across tokens (byte-level BPE). Source: sota/robustness/robustness.json -> examples.",
    "sentences": sents,
    "tokenizers": {t["key"]: t["display"] for t in rb["tokenizers"]},
    "segmentations": {k: [{"n": e["n"], "tokens": e["tokens"]} for e in v] for k, v in ex.items()},
}, EVAL / "examples_segmentations.json")

# ---------------------------------------------------------------- 6. charts
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GRAY, INK, INK2, MUTED, SURF = "#2a78d6", "#eb6834", "#b9b7b0", "#0b0b0b", "#52514e", "#898781", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

base = {b["representative"]: b for b in cf["study_baselines_deduplicated"]}
top = {b["representative"]: b for b in cf["top15_lossless_behaviours"]}
pick = [
    ("hindko-tokenizer (this, 32k)", rel["bytes_per_token"], None),
    ("urnova-95m (best external)", top["ProximaAI__urnova-95m"]["bytes_per_token"], top["ProximaAI__urnova-95m"]),
    ("RoBERTa-Urdu (UrduHack)", base["roberta-urdu"]["bytes_per_token"], base["roberta-urdu"]),
    ("BLOOM", base["bloom"]["bytes_per_token"], base["bloom"]),
    ("Gemma 3 / Gemma 4", base["gemma-3"]["bytes_per_token"], base["gemma-3"]),
    ("GPT-4o (o200k)", base["gpt-4o"]["bytes_per_token"], base["gpt-4o"]),
    ("Mistral Tekken (NeMo, Small 4)", base["apertus"]["bytes_per_token"], base["apertus"]),
    ("Llama 4", base["llama-4"]["bytes_per_token"], base["llama-4"]),
    ("Qwen 3.5", base["qwen-3.5"]["bytes_per_token"], base["qwen-3.5"]),
    ("DeepSeek-V3 / R1 / V4", base["deepseek-r1"]["bytes_per_token"], base["deepseek-r1"]),
    ("Llama 3", base["alif-1.0"]["bytes_per_token"], base["alif-1.0"]),
]
chart1 = []
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ys = list(range(len(pick)))[::-1]
for y, (lab, v, b) in zip(ys, pick):
    ax.barh(y, v, height=0.62, color=BLUE if b is None else GRAY, edgecolor=SURF, linewidth=2)
    ratio = 1.0 if b is None else 1.0 / (1.0 - b["released_uses_fewer_tokens"])
    txt = f"{v:.2f}" + ("" if b is None else f"   ({ratio:.2f}\u00d7 the tokens)")
    ax.text(v + 0.08, y, txt, va="center", ha="left", color=INK if b is None else INK2,
            fontsize=9, fontweight="bold" if b is None else "normal")
    chart1.append({"label": lab, "bytes_per_token": v, "tokens_vs_released": round(ratio, 4)})
ax.set_yticks(ys)
ax.set_yticklabels([p[0] for p in pick])
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(0, 9.6)
ax.set_xlabel("UTF-8 bytes per token (higher = fewer tokens for the same text)")
ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.suptitle("Held-out Hindko: bytes per token", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
ax.set_title("Strict test split: 491 documents (1.45 MB), never seen in training.\n"
             "Lossless tokenizers only, each with its own encoder. Ratio = tokens needed vs this tokenizer.",
             loc="left", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(EVAL / "chart_bytes_per_token_test.png", facecolor=SURF)
plt.close(fig)

# chart 2: LM bpb deltas vs baseline, dev and test
def rd(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


devc = {r["id"]: r for r in rd(REL / "eval" / "dev_lm_confirm_ranking.csv")}
tstc = {r["id"]: r for r in rd(REL / "eval" / "test_lm_confirm.csv")}
devl = {r["id"]: r for r in rd(REL / "eval" / "dev_lm_large.csv")}
tstl = {r["id"]: r for r in rd(REL / "eval" / "test_lm_large.csv")}
NAMES = {"R2-A4-SPnat-D2-32k": "SP-Unigram 32k (released)",
         "R2-A10-MinGram-P1r3-D2-48k": "MinGram 48k (pre-registered pick)",
         "R2-A10-MinGram-P1r3-D2-32k": "MinGram 32k",
         "R2-A1-P1r3-D2-48k": "BPE 48k",
         "A1-P1r3-D2-32k": "BPE 32k"}
order_c = ["R2-A4-SPnat-D2-32k", "R2-A10-MinGram-P1r3-D2-48k", "R2-A10-MinGram-P1r3-D2-32k",
           "R2-A1-P1r3-D2-48k", "A1-P1r3-D2-32k"]
order_l = ["R2-A4-SPnat-D2-32k", "R2-A10-MinGram-P1r3-D2-32k", "R2-A10-MinGram-P1r3-D2-48k", "A1-P1r3-D2-32k"]
b2 = tl["boot_2seeds_3cands"]
chart2 = {"confirm": [], "large": []}
fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), dpi=150, sharey=False)
fig.patch.set_facecolor(SURF)
for ax, order, scale in ((axs[0], order_c, "confirm"), (axs[1], order_l, "large")):
    ax.set_facecolor(SURF)
    ys = list(range(len(order)))[::-1]
    for y, i in zip(ys, order):
        if scale == "confirm":
            d = devc[i]
            dv, dlo, dhi = float(d["delta_vs_baseline_pct"]), float(d["delta_vs_baseline_ci95_lo"]), float(d["delta_vs_baseline_ci95_hi"])
            t = tstc[i]
            tv = float(t["test_delta_vs_baseline_pct"])
            tlo, thi = json.loads(t["test_delta_vs_baseline_ci95_pct"])
        else:
            d = devl[i]
            dv, dlo, dhi = float(d["delta_vs_baseline_pct"]), float(d["delta_vs_baseline_ci95_pct_lo"]), float(d["delta_vs_baseline_ci95_pct_hi"])
            tv = tlo = thi = None
            if i in tstl:
                tv = float(tstl[i]["test_delta_vs_baseline_pct_means"])
                key = f"{i} - A1-P1r3-D2-16k"
                if key in b2:
                    tlo, thi = b2[key]["ci95_pct"]
        ax.errorbar(dv, y + 0.14, xerr=[[dv - dlo], [dhi - dv]], fmt="o", color=BLUE, ms=8, capsize=3,
                    elinewidth=2, markeredgecolor=SURF, markeredgewidth=1.5, label="dev" if y == ys[0] else None)
        if tv is not None:
            if tlo is not None:
                ax.errorbar(tv, y - 0.14, xerr=[[tv - tlo], [thi - tv]], fmt="s", color=ORANGE, ms=8, capsize=3,
                            elinewidth=2, markeredgecolor=SURF, markeredgewidth=1.5,
                            label="test (sealed, used once)" if y == ys[0] else None)
            else:
                ax.plot(tv, y - 0.14, "s", color=ORANGE, ms=8, markerfacecolor=SURF, markeredgewidth=2)
                ax.text(tv, y - 0.42, "1 seed, no CI", ha="center", va="center", fontsize=7.5, color=INK2)
        chart2[scale].append({"id": i, "dev_delta_pct": dv, "dev_ci95": [dlo, dhi], "test_delta_pct": tv,
                              "test_ci95": None if tlo is None else [tlo, thi]})
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[i] for i in order])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("\u0394 bpb vs BPE-16k baseline, % (left = better)")
    ax.set_ylim(-0.7, len(order) - 0.4)
axs[0].set_title("Decision scale (pre-registered)\n1.77M non-embedding parameters, 5 seeds", loc="left", fontsize=9.5, color=INK)
axs[1].set_title("Large arbiter (report-only)\n10.6M non-embedding parameters, 2 seeds", loc="left", fontsize=9.5, color=INK)
fig.text(0.01, 0.015, "Right panel: BPE 32k was not run on test at this scale; MinGram 32k has 1 test seed (Colab quota exhausted). "
         "Baseline = byte-level BPE, 16k vocabulary.", fontsize=7.5, color=INK2)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.905), ncol=2, frameon=False, fontsize=9)
fig.suptitle("Language-model evidence: held-out bits per byte, relative to the standard BPE recipe",
             x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
fig.text(0.01, 0.875, "Small GPT-style LMs trained from scratch with each tokenizer; 95 % CIs from a hierarchical "
         "cluster bootstrap (10,000 replicates).", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.85))
fig.savefig(EVAL / "chart_lm_bpb_dev_test.png", facecolor=SURF)
plt.close(fig)
jdump({"chart_bytes_per_token_test.png": chart1, "chart_lm_bpb_dev_test.png": chart2}, EVAL / "chart_data.json")

# ---------------------------------------------------------------- 7. vocab audit counts (no pieces listed)
jdump({
    "what": "Headline counts of the vocabulary audit (F:/Hindko/_tokenizer/sota/vocab_audit/VOCAB_AUDIT.md, 2026-09-27). "
            "The piece lists themselves are kept private (they are for native-speaker review).",
    "verdict": "no fix to the tokenizer needed before publishing",
    "learned_pieces": 32441,
    "arabic_script_learned_pieces_pct": 90.9,
    "junk_artifact_pieces": 478, "junk_artifact_pieces_pct_of_learned": 1.47, "junk_test_token_share_pct": 0.29,
    "longest_digit_run_in_any_piece": 4, "ascii_digit_pieces": "single digits 0-9 only",
    "urdu_digit_run_pieces_2_to_4": 345,
    "pii_placeholder_dedicated_pieces": 0,
    "pieces_identifying_private_individuals_found": 0,
    "offensive_sense_attested": 13, "sensitive_group_labels": 17, "ordinary_words_usable_as_insults": 24,
    "false_positive_candidates": 45, "pieces_removed": 0,
    "special_ids_emitted_on_11786145_real_tokens": 0,
    "split_special_tokens_true_prevents_literal_special_match": False,
    "sections": {"composition": "2", "junk": "3", "privacy": "4", "offensive": "5", "specials": "7"},
}, EVAL / "vocab_audit_summary.json")

print("done:", sorted(p.name for p in OUT.rglob("*") if p.is_file()))
