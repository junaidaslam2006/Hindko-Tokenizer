"""Render F:\\Hindko\\hf_upload\\README.md from README.template.md, then write SHA256SUMS.

Fills: {{SEG1..3}} (token splits from eval/examples_segmentations.json), {{SHA_<file>}} (sha256 of the
copied files), {{SOURCES}} (hidden HTML comment mapping every number to its source)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path(r"F:\Hindko\hf_upload")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


ex = json.loads((OUT / "eval" / "examples_segmentations.json").read_text(encoding="utf-8"))
SHOW = [("released", "**This tokenizer**"), ("gemma-3", "Gemma 3 / Gemma 4"), ("gpt-4o", "GPT-4o (o200k)"),
        ("llama-4", "Llama 4"), ("qwen-3.5", "Qwen 3.5"), ("deepseek-v3", "DeepSeek-V3"), ("llama-3", "Llama 3")]


def seg_table(i):
    rows = ["| Tokenizer | Tokens | Split (token boundaries shown as \\|) |", "|---|---:|---|"]
    for key, name in SHOW:
        e = ex["segmentations"][key][i]
        split = " \\| ".join(t.replace("|", "\\|") for t in e["tokens"])
        n = f"**{e['n']}**" if key == "released" else str(e["n"])
        rows.append(f"| {name} | {n} | {split} |")
    return "\n".join(rows)


SOURCES = r"""<!--
NUMBER-TO-SOURCE MAP (study folder = F:\Hindko\_tokenizer\ ; release folder = F:\Hindko\tokenizer\ ; eval/ = this repo)
- 32,768 ids; 32,441 learned; id layout; 5 placeholders at 7183,16359,16629,16924,19063: release README.md s1/s2.5; _tokenizer/sota/vocab_audit/VOCAB_AUDIT.md s2.1
- 7.277 bytes/token; fertility 1.135; STRR 89.1 %; 199,390 tokens; G1 491/491; 4.068 chars/token: release eval/test_intrinsic_released.json; release README s7.5; sota/competitors/competitors_final.json -> released
- test split 491 docs, 1,451,026 bytes, 171,769 words, 27 clusters: sota/competitors/COMPETITORS_FINAL.md header; release README s4
- 550 external rows (485 new + 65 baselines); 528 distinct encodings; 130 lossless (131 incl. released); rank 1 of 131; 14,856 repos found; 2,608 repos represented: sota/competitors/COMPETITORS_FINAL.md "What compared", "Search coverage"; eval/competitors_summary.json -> counts
- urnova-95m 6.061 / 1.373 / 16.7 %; top-12 table rows: sota/competitors/COMPETITORS_FINAL.md "Top 15 lossless"; eval/competitors_test_strict.csv
- Frontier rows (Gemma 3 4.610 36.6 %; GPT-4o 4.338 40.4 %; Tekken 3.901 46.4 %; Llama 4 3.727 48.8 %; Qwen 3.5 3.598 50.6 %; Kimi K2 3.530 51.5 %, vocab 163,840; DeepSeek 3.372 53.7 %; Llama 3 2.641 63.7 %; vocab sizes): COMPETITORS_FINAL.md "Study baselines"; eval/competitors_summary.json -> study_baselines_deduplicated; Tekken vocab 131,072: release README s7.4
- 1.68x (GPT-4o), 2.02x (Qwen 3.5), and all "x the tokens" ratios = 1/(1 - released_uses_fewer_tokens): eval/chart_data.json
- Example sentences and counts (14/20/20 = 54; GPT-4o 106 1.96x; Gemma 98 1.81x; Llama 4 119 2.20x; Qwen 128 2.37x; DeepSeek 131 2.43x; Llama 3 177 3.28x): sota/robustness/ROBUSTNESS.md s6; eval/examples_segmentations.json
- 29.8 KB / 17.8 KB per 4,096 tokens: derived = 4096 x 7.277 bytes and 4096 x 4.338 bytes
- MinGram 48k 7.373 bytes/token (test): COMPETITORS_FINAL / release README s7.4; SuperBPE 8.40 bytes/token (dev): eval/dev_lm_confirm_ranking.csv (R2-A6-SBPE-P1r3-D2-32k-t080 dev_bytes_per_token 8.399)
- Decision-scale dev bpb and deltas (1.22584, -1.62 [-2.11,-1.24], MinGram48k 1.22636 -1.58, MinGram32k 1.22844 -1.41, BPE48k 1.23374, BPE32k 1.23597, baseline 1.24607; +0.04 / +0.21 gaps; rank 1 of 17): eval/dev_lm_confirm_ranking.csv; analysis/DECISION.md
- resolution about 0.6-0.7 %: release README s6 step 5 (from analysis/DECISION.md)
- Decision-scale test bpb (1.20911 -1.01 [-1.33,-0.72]; 1.20543 -1.32 [-1.47,-1.16]; 1.20620 -1.25; 1.21111 -0.85; 1.21459 -0.57; 1.22150): eval/test_lm_confirm.csv; analysis/TEST_RESULTS.md s4-s5
- +0.31 [-0.02,+0.63] vs MinGram48k; +0.24 [-0.13,+0.60] vs MinGram32k; newspaper +0.59 / +0.51; shrink -1.62 -> -1.01 (largest): analysis/TEST_RESULTS.md s5, s8; release README s7.3
- Large arbiter dev (1.13373, 1.14086, 1.14407, 1.14908, 1.14940; deltas and CIs): eval/dev_lm_large.csv; top-set pairs -0.62 [-1.04,-0.31], -0.90 [-1.35,-0.56]: eval/lm_key_contrasts.json -> dev_large_top_set_pairs
- Large arbiter test (1.11898, 1.12240 n=1, 1.12477, 1.12767; -0.77 [-1.15,-0.40]; -0.51 [-0.86,-0.22]; -0.26 [-0.44,-0.08]): eval/test_lm_large.csv; eval/lm_key_contrasts.json -> test_large_boot_2seeds_3cands
- 1.77M / 10.6M non-embedding params; 5 / 2 seeds; 45 MB LM train stream (train_D1, 16,015 docs); 10,000 bootstrap replicates; +-0.3 % margin: release README s6, s7; analysis/AMENDMENT_1.md
- 75 tokenizer builds measured on dev: count of distinct candidates/**/results/dev_strict/<id>/summary.json folders (86) minus 11 "__pre_remedy" duplicates; includes 2 reference builds
- 18 LM candidates, 17 ranked, 12 (10 pre-registered + 2 reference arms added before the first LM run) + 6 exploratory: eval/dev_lm_confirm_ranking.csv -> origin; release README s6; analysis/DECISION.md
- Released = tied top set, chosen by amended rule; pre-registered pick MinGram 48k: analysis/AMENDMENT_1.md s5
- plain tokens/word 1.161 (199,390 / 171,769): release eval/test_intrinsic_released.json -> overall.tokens_per_word
- `2024` = 5 tokens at word start (standalone piece + 4 digits): measured with tokenizer.json; VOCAB_AUDIT.md
- 230 leak-prone records from 40 evaluation groups moved to train: release README s4
- Training data (11,066 docs; 14,731,915 chars; 26,330,463 bytes; per-source rows; 61.8 % books; 0 leaking records; 90/5/5): release README s4 (data/data_manifest.json, splits/splits_report.json)
- Out-of-corpus, noisy, code-mixed, other-language and perturbation numbers (6.18/4.74, 23 %, 35 %; 4.41/4.20; 4.97/4.91; 3.62/3.73; 5.66/5.55, 2.77/5.93; 1,692/1,692; 6.7 % unseen words; all perturbation % and ranges; +92.15 / +60.72 / +25.8; other-language table and 1.51x / 2.61x / 3.05x): sota/robustness/ROBUSTNESS.md s0, s2.2, s3, s4, s5; eval/robustness_summary.json
- 9 of 32,768 pieces contain Arabic-keyboard letters: sota/robustness/ROBUSTNESS.md s3 (robustness.json -> released_vocab_and_training_text_facts.vocab_ak_pieces)
- normalize() equivalence (2,194 dev docs, 86,516 stress items, 50,000 fuzz, 0 mismatches): sota/hf_card/check_normalize.py output
- Vocabulary audit numbers (90.9 % / 90.3 % Arabic-script; 530 Latin; 345 Urdu-digit runs; longest digit run 4; 478 artifacts 1.47 % / 0.29 %; 13 / 17 / 24 sensitive; 6 test tokens; split_special_tokens finding): sota/vocab_audit/VOCAB_AUDIT.md s0, s2, s3, s4, s5; eval/vocab_audit_summary.json
- 40.6 % pieces < 20 occurrences in train_D1 (7.3M tokens): eval/health_gates_and_support.csv; release README s7.7
- sp.model float32 limit (about 3,400 characters): release README s3.2, s8
- File sha256 values: computed on this folder by sota/hf_card/render_readme.py; equal to release RELEASE_MANIFEST.json (checked by sota/hf_card/build_hf_upload.py)
- Frontier LM comparison "planned, not run": _tokenizer/colab/FRONTIER_LM.md (bundle 527e1023adb5)
-->"""

t = (HERE / "README.template.md").read_text(encoding="utf-8")
for i in range(3):
    t = t.replace("{{SEG%d}}" % (i + 1), seg_table(i))
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sp.model", "sp.vocab", "gauge.json"]:
    t = t.replace("{{SHA_%s}}" % f, sha(OUT / f))
t = t.replace("{{SOURCES}}", SOURCES)
assert "{{" not in t, "unfilled placeholder"
(OUT / "README.md").write_text(t, encoding="utf-8", newline="\n")

files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
(OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT).as_posix()}\n" for p in files),
                                encoding="utf-8", newline="\n")
print("README.md", len(t), "chars;", len(files), "files in SHA256SUMS")
