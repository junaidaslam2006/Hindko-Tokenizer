# A6 SuperBPE for Hindko (PLAN Stage 2)

SuperBPE, **reimplemented from Liu et al. 2025, "SuperBPE: Space Travel for Language Models" (COLM 2025)**. No third-party SuperBPE code was cloned, installed or run.

Written 2026-09-26. Everything below was measured on this machine, on `dev_strict` unless stated otherwise:
- split manifest sha256 `76582d3a…` (verified before every run);
- `normalize()` 1.0.1, checked as a fixed point on every train and dev text used;
- **the test split was never read** (`common.load_view` refuses it, and the harness refuses test uids).

## 0. Summary

| id | t / T | tokenizer.json sha256 | dev bytes/token | vs A1-P1-16k | NSL | encoder |
|---|---|---|---|---|---|---|
| A6-SBPE-P1-D1-16k-t090 | 14,746 / 16,384 | `241393b316eeb8f2…` | 7.359 | **+9.64%** | 0.9120 | HF-native, exact |
| A6-SBPE-P1-D1-16k-t080 | 13,107 / 16,384 | `f5859d293e0fad3a…` | 7.465 | **+11.22%** | 0.8991 | HF-native, exact |
| A6-SBPE-P1-D1-32k-t080 | 26,214 / 32,768 | `d57ca6031e7283dd…` | 8.190 | +22.02% (+16.35% vs A1-P1-32k) | 0.8195 | HF-native, exact |

- **All three pass G1–G4 and R1/R2 are reported.** G5 does not apply to byte-level vocabularies. G2 for multi-word tokens uses the operationalisation in §4, which is a documented deviation.
- **The shipped `tokenizer.json` is exact.** Loaded as stock HF `tokenizers`, it gives the same ids as an independent pure-Python reference encoder on:
  - 100% of dev_strict (836 documents) and dev_permissive (1,358 documents);
  - every 20th train document (801 documents).
  `transformers.PreTrainedTokenizerFast(tokenizer_file=…)` also gives identical ids on dev_strict. No custom encoder is needed, so tie-breaker (a) of PLAN §7 holds.
- All three meet the PLAN §3 "after Stage 2" conditions: they pass the hard gates, round-trip exactly, and the HF-native encoding reproduces the reference encoder on 100% of dev.
- **32k ratio: t/T = 0.8.** It was chosen by the rule fixed in `rule_32k.json` before any harness metric existed: gates, then PLAN §7's tie-breakers, where (b), bytes/token, decided (7.465 vs 7.359). Liu et al. found that the most compressive transition was *not* the best downstream, so Stage 3 bpb may reverse this choice (§6).
- **Compression is not the objective.** PLAN §4.2 says these intrinsic numbers screen candidates; the LM bpb of Stage 3 decides.

## 1. Method

**Stage 1 = A1-P1 up to t.**
- Recipe: HF `BpeTrainer`, `Split(P1, isolated)` + `ByteLevel(use_regex=False)`, `min_frequency=2`, ByteLevel initial alphabet, the 64-token special block at ids 0–63, trained on `train_D1` (whole documents).
- t is the vocabulary size at the transition, *including* the 64 specials and the 256 bytes: t = round(r·T).
- The stage-1 file at t is a strict prefix of A1 trained to T: identical merges and ids, checked for 13,107 and 14,746 inside 16,384, and for 26,214 and 16,384 inside 32,768.
- `stage1/a1_p1_16384.json` is **byte-identical** to `candidates/standard/tok/A1-P1-D1-16k/tokenizer.json` (sha256 `4d018f82…`). Stage 1 is therefore exactly the standard recipe.
- The candidates/standard sweep had not produced A1-P1-D1-32k yet, so `stage1/a1_p1_32768.json` (sha256 `c6198260…`) was built and evaluated here.

**Stage 2 = BPE continued to T without whitespace pre-tokenization.**
- Stage-2 pre-tokenizer **S2** (Oniguruma; the Python twin replaces `\x{200C}` with `\u200C`):
  ` ?[\p{L}\p{M}\x{200C}\x{200D}]+(?: [\p{L}\p{M}\x{200C}\x{200D}]+)*| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+`
  - It is P1 with its first alternative widened to a *run of letter-words joined by single spaces*.
  - A chunk therefore ends only at punctuation or symbols (`۔ ، ؟ " …`), digits and line breaks, so superwords span spaces but never sentence punctuation, digits or lines.
  - The canonical form has no double spaces, tabs, or leading or trailing spaces on a line (measured on train_D1: 0 of each).
- **Chunk preparation.**
  - Every train_D1 document is encoded with the stage-1 merges under S2, and the tokens are grouped into chunks by `word_ids`.
  - This equals the stage-1 (P1) encoding of the document in **all 16,015 train documents** (0 mismatches, every build). No stage-1 merge crosses a P1 boundary inside a chunk, so "encode each chunk with the stage-1 tokenizer" (SOTA §4) and "apply stage-1 merges inside the chunk" give the same result.
  - train_D1 has 2,348,237 S2 chunks, of which 613,757 contain several words. There are 589,087 chunk types with ≥2 stage-1 tokens at t = 14,746.
- **PUA code-point trick (SOTA §4, route a).**
  - Stage-1 id i becomes U+F0000+i (plane-15 PUA), and every chunk becomes a PUA string.
  - The stock Rust `BpeTrainer` then runs on these strings: no normalizer, no pre-tokenizer, `min_frequency=2`, `initial_alphabet` = the t−64 PUA characters, `vocab_size` = (t−64) + (T−t).
- **Independent pure-Python trainer** (`build_stage2.py:py_train`).
  - It follows the same algorithm with HF `BpeTrainer` semantics: highest pair count first, ties broken by the smallest id pair; left-to-right non-overlapping merges; `min_frequency` 2; an existing token string is reused on duplicates.
  - **Run uncapped, it reproduced the Rust merges exactly in all three builds** (1,638 / 3,277 / 6,554 merges).
- **4-word cap (Liu et al.).**
  - The stock trainer cannot enforce it. `py_train(max_words=4)` never merges a pair whose result would contain more than 4 whitespace-separated words, counted as runs of non-space bytes, where a word fragment counts as a word. Training then continues with the next pair.
  - The cap **binds** here. The uncapped runs produced 14 / 37 / 103 tokens of more than 4 words, up to 9 / 13 / 18 words. Most are newspaper boilerplate such as ` گندھارا ہندکو بورڈ دے جنرل سیکرٹری محمد ضیاء الدین`, idiom-entry formulas such as ` دا مطلب ایہہ وے کہ`, and ` صلی اللہ علیہ وآلہ وسلم`.
  - **The shipped merges are therefore the capped pure-Python ones** (`route: pure_python_capped` in `build_info.json`). The Rust run serves as the uncapped cross-check.
- **Assembly.**
  - The PUA merges are translated back to byte-level strings and appended after all stage-1 merges.
  - The final `tokenizer.json` is the stage-1 file with S2 as the Split regex and the extended vocab and merges.
  - 0 stage-2 tokens duplicate the bytes of another token, and 0 merge pairs repeat, in every build. This condition makes one-pass rank-ordered HF encoding equal to the two-phase encoding that training assumed.
- **Every stage-2 token is multi-word.**
  - Word counts of the stage-2 tokens:

    | build | 2 words | 3 words | 4 words |
    |---|---|---|---|
    | 16k-t0.9 | 1,476 | 124 | 38 |
    | 16k-t0.8 | 2,928 | 287 | 62 |
    | 32k-t0.8 | 5,661 | 751 | 142 |

  - After stage 1, cross-word pairs (` دے`+` نال`) outnumber every remaining within-word pair.

**Reference encoder** (`ref_encoder.py`, pure Python; it reads the JSON files directly and never calls HF `tokenizers`).
- Steps: cut out special tokens → Python-`regex` S2 → stage-1 merges by rank on the chunk's bytes → stage-2 merges by rank in PUA space (tokens identified by their stage-1 id *sequence*) → final ids.
- The rank semantics are those of HF `Word::merge_all`: lowest rank first, then leftmost position.

## 2. Fidelity notes: what differs from Liu et al. 2025

The paper was read through the arXiv HTML page, using a web-fetch summary with short quotes, not the full PDF. The points it states:
- stage 2 "resumes from the vocabulary learned thus far" with "whitespace pretokenization … skipped";
- digit pretokenization (groups of 3 from the right) and a colon special case stay active;
- an upper bound of 4 words per token, reported to have "no measurable impact";
- the same data in both stages;
- t is a vocabulary size (t = 80k / 160k / 180k of 200k).

| aspect | paper | here | why |
|---|---|---|---|
| stage-2 boundaries | whitespace splitting removed. The text read does not mention punctuation or sentence splitting | split at **every** punctuation/symbol run, digit and line break | task instruction ("split only at sentence and line boundaries and punctuation") and SOTA §4. Stricter: no superword contains punctuation |
| digits | right-to-left groups of 3, both stages | P1: single digits, both stages | PLAN §2.1 pre-tokenizer choice (P1r3 is a Stage 1 ablation) |
| stage-1 regex | GPT-2-style | P1 (marks, ZWNJ and ZWJ kept inside words) | PLAN §2.1; the GPT-2 regex is 6–7% worse here |
| 4-word cap | stated; mechanism not given | a forbidden pair is skipped permanently; the cap binds for 14 / 37 / 103 tokens | only the mechanism is ours |
| colon special case | no token containing `: ` | not needed: S2 punctuation pre-tokens never contain a space after the punctuation | — |
| implementation | HF fork (Rust) | stock Rust trainer via PUA (uncapped) + pure-Python trainer (capped); identical when uncapped | no Rust toolchain; no third-party code |
| scale | 200k vocabulary, 10 GB | 16k / 32k, 45 MB (train_D1) | PLAN |
| t | 180k/200k = 0.9 was best downstream | 0.9 and 0.8 at 16k; 0.8 at 32k | task |

## 3. Exactness checks (`verify.py`, `verify.json` per build)

| check | 16k-t0.9 | 16k-t0.8 | 32k-t0.8 |
|---|---|---|---|
| HF-native ids = reference ids: dev_strict (836) / dev_permissive (1,358) / train every 20th (801) | 0 / 0 / 0 mismatching documents | 0 / 0 / 0 | 0 / 0 / 0 |
| reference phase 1 = HF stage-1 (P1) tokenizer, same documents | 0 mismatches | 0 | 0 |
| every final token boundary is a stage-1 boundary | 0 violations | 0 | 0 |
| G1 round trip, HF and reference, dev_strict + dev_permissive | 0 failures | 0 | 0 |
| `PreTrainedTokenizerFast` (transformers 5.3.0) = HF, dev_strict | 0 mismatches | 0 | 0 |
| special-token and edge strings (`<\|im_start\|>` inside text, `""`, `"\n"`, digits) | equal and round-trip | equal and round-trip | equal and round-trip |

The Oniguruma and Python `regex` versions of S2 give identical pre-tokens (`s2_differential.json`) on:
- dev_strict (0/836 documents differ; 39,231 chunks);
- dev_permissive (0/1,358);
- ZWNJ-perturbed dev_strict (0/836);
- all of train_D1 (0/16,015; 2,348,237 chunks).

## 4. Gates (PLAN §4.3)

- **Source:** harness `eval/harness.py` 1.0.0 with `--gates all --train-data train_D1`, plus `gates_sbpe.py` for multi-word G2.
- **Harness code hashes:** identical to the candidates/standard A1-16k run (`harness.py` `31db30dd…`, `adapters.py` `bc377b78…`).

| tokenizer | G1 | G2 (harness: tokens without inner space) | G2 multi-word | G3 | G4 | G5 |
|---|---|---|---|---|---|---|
| A1-P1-D1-16k (standard) | pass | pass (16,055 tested, 0 fail) | n/a | pass | pass | n/a (byte-level) |
| A6 16k-t0.9 | pass (836/836) | pass (14,418 tested, 0 fail) | pass (1,638 tokens, 0 unreachable) | pass | pass | n/a |
| A6 16k-t0.8 | pass | pass (12,779 tested, 0 fail) | pass (3,277, 0 unreachable) | pass | pass | n/a |
| A1-P1-D1-32k (built here) | pass | pass (32,439, 0 fail) | n/a | pass | pass | n/a |
| A6 32k-t0.8 | pass | pass (25,885, 0 fail) | pass (6,554, 0 unreachable) | pass | pass | n/a |

**G4.** The whole pipeline was retrained from scratch into `g4/`: a new stage-1 training, a new chunk extraction in a fresh work directory, the Rust trainer, and the capped Python trainer. The results are byte-identical `tokenizer.json` files for all three builds, and for A1-P1-32k. These retrains ran with the final code (`build_stage2.py` `06122cfa…`, `common.py` `bd178114…`).
- The two official 16k builds started before one cosmetic edit, which added the `--work` cache-directory option and did not change behaviour.
- The 16k-t0.9 `build_info.json` records the older hash `f6cee016…`.
- The 16k-t0.8 `build_info.json` records the new hash, because the hash is taken when the build finishes.

The final code reproduces both files byte for byte.

**G2 for multi-word tokens: documented deviation.**
- **The clause.** PLAN §4.3 says multi-word tokens "are tested on the shortest train chunk that contains them".
- **Read literally**, as the shortest distinct train chunk containing the token's *text* as a raw substring, the test "fails" 224 / 569 / 1,120 tokens. Every one of them encodes to itself in isolation and is used in training:
  - 16k-t0.9: train frequency 13–1,578, median 174;
  - 16k-t0.8: frequency 0–1,584, median 106;
  - 32k-t0.8: frequency 0–2,280, median 61.
- **The literal test fails for two reasons that are not unreachability:**
  - *Substring misalignment.* ` تے اس` is "contained" in ` تے اسی`, where the last word is only a prefix.
  - *Context competition and absorbed intermediates*, exactly as for BPE intermediates in §4.3 R1. ` طور تے` inside ` خاص طور تے` loses to the longer superword. Even with stage-1-aligned containment (the chunk's stage-1 segmentation contains the token), 111 / 278 / 678 tokens are not emitted in their shortest chunk.
- **Operationalisation used for the verdict** (`gates_sbpe.py`): a multi-word token is reachable iff *any* of these holds:
  - (i) its own text encodes to exactly `[id]`, the ordinary G2 self-tokenization. The text is itself a valid S2 chunk;
  - (ii) it is emitted in the shortest train chunk whose stage-1 segmentation contains it;
  - (iii) its train frequency is > 0.
- **Result:** test (i) alone passes for **every** multi-word token in all three builds (0 isolated failures), so 0 are unreachable. The pre-declared remedy (deletion) was therefore not triggered.
- **If the clause is read literally,** no SuperBPE could pass it, and deleting hundreds of frequently used tokens would change encodings, so the remedy's "retrain" would restore them. This deserves a PLAN amendment, like the gate revisions already made there.

**R1 (train support, whole train_D1 documents) and R2:**

| tokenizer | learned | train freq = 0 | < 20 | < 100 | median | multi-word tokens < 20 / = 0 | partial-UTF-8 (freq 0) |
|---|---|---|---|---|---|---|---|
| A1-P1-D1-16k | 16,064 | 25 | 447 (2.78%) | 8,125 | 98 | — | 9 (4) |
| A6 16k-t0.9 | 16,064 | 22 | 369 (2.30%) | 6,473 | 129 | 6 / 0 | 8 (3) |
| A6 16k-t0.8 | 16,064 | 19 | 326 (2.03%) | 5,928 | 129 | 20 / 2 | 8 (3) |
| A1-P1-D1-32k | 32,448 | 88 | 8,760 (27.0%) | 25,225 | 34 | — | 9 (4) |
| A6 32k-t0.8 | 32,448 | 68 | 1,838 (5.66%) | 23,417 | 53 | 67 / 7 | 9 (4) |

**Leaves vs intermediates.** At 16k every token with train frequency < 20 is an intermediate merge-graph node (0 leaves), as for A1-16k. At 32k they split as follows:

| tokenizer | leaves < 20 | intermediates < 20 |
|---|---|---|
| A1-P1-D1-32k | 6,862 | 1,898 |
| A6 32k-t0.8 | 423 | 1,415 |

SuperBPE spends its last (T−t) ids on frequent superwords instead of rare subwords. At 32k this cuts tokens with train frequency < 20 from 8,760 to 1,838.

## 5. Harness results (dev_strict; NSL relative to A1-P1-D1-16k)

| tokenizer | dev tokens | bytes/token | Δ vs A1-16k | NSL | chars/token | fertility | STRR | Rényi α=2.5 | vocab used | train bytes/token |
|---|---|---|---|---|---|---|---|---|---|---|
| A1-P1-D1-16k | 215,371 | 6.712 | — | 1.0000 | 3.763 | 1.211 | 0.838 | 0.4995 | 69.3% | 5.955 |
| A6 16k-t0.9 | 196,428 | 7.359 | +9.64% | 0.9120 | 4.126 | 1.223 | 0.830 | 0.5255 | 72.6% | 6.343 |
| A6 16k-t0.8 | 193,636 | 7.465 | +11.22% | 0.8991 | 4.185 | 1.238 | 0.819 | 0.5290 | 74.4% | 6.403 |
| A1-P1-D1-32k | 205,344 | 7.039 | +4.88% | 0.9534 | 3.947 | 1.153 | 0.878 | 0.4589 | 46.1% | 6.267 |
| A6 32k-t0.8 | 176,498 | 8.190 | +22.02% | 0.8195 | 4.592 | 1.168 | 0.868 | 0.4827 | 55.1% | 6.988 |

| tokenizer | bytes/token book | newspaper | web | NSL book | NSL newspaper | NSL web | bytes/token, line by line |
|---|---|---|---|---|---|---|---|
| A1-P1-D1-16k | 6.471 | 7.235 | 5.744 | 1 | 1 | 1 | 6.899 |
| A6 16k-t0.9 | 6.960 | 8.263 | 5.975 | 0.930 | 0.876 | 0.961 | 7.588 |
| A6 16k-t0.8 | 7.037 | 8.442 | 6.029 | 0.920 | 0.857 | 0.953 | 7.701 |
| A1-P1-D1-32k | 6.799 | 7.559 | 6.053 | 0.952 | 0.957 | 0.949 | 7.247 |
| A6 32k-t0.8 | 7.685 | 9.357 | 6.523 | 0.842 | 0.773 | 0.881 | 8.479 |

**Superword use on dev_strict:**

| build | share of tokens | share of bytes | distinct superword ids used |
|---|---|---|---|
| 16k-t0.9 | 9.6% | 18.8% | 1,232 |
| 16k-t0.8 | 12.2% | 23.7% | 2,323 |
| 32k-t0.8 | 15.7% | 28.8% | 4,123 |

The most frequent superwords (t0.9): ` دے نال` 226, ` اس دے` 198, ` اے کہ` 196, ` اس دی` 189, ` اے تے` 172, ` دی اے` 138, ` اس نوں` 138, ` اُناں نے` 137, ` کر کے` 132, ` دے مطابق` 119.

**Robustness (PLAN §4.2).** Δtok is the relative change in token count. "Seg-change" is the share of affected words whose segmentation changes.

| tokenizer | harakat Δtok | harakat seg-change | digits Δtok | punct_space Δtok | punct_space seg-change | zwnj Δtok | zwnj seg-change |
|---|---|---|---|---|---|---|---|
| A1-P1-D1-16k | −0.0045 | 0.121 | +0.000005 | +0.0380 | 0.349 | +0.0035 | 0.698 |
| A6 16k-t0.9 | −0.0062 | **0.285** | +0.000005 | +0.0416 | 0.347 | +0.0038 | 0.690 |
| A6 16k-t0.8 | −0.0062 | **0.284** | +0.000005 | +0.0422 | 0.347 | +0.0037 | 0.552 |
| A1-P1-D1-32k | −0.0032 | 0.086 | +0.000005 | +0.0399 | 0.350 | +0.0038 | 0.808 |
| A6 32k-t0.8 | −0.0056 | **0.264** | +0.000006 | +0.0464 | 0.350 | +0.0046 | 0.794 |

**Observations (intrinsic only, report-grade):**
- **The gain is largest on newspaper text** (−12.4% tokens at 16k-t0.9), which is rich in fixed phrases and bylines, and smallest on web text (−3.9%).
- **The train-side gain is smaller than the dev_strict gain** (+6.5% vs +9.6% bytes/token at t0.9), because book text (71% of train characters) gains less.
- **For the LM protocol (PLAN §5), measured train bytes/token replaces the A1 value that PLAN used as an upper bound for SuperBPE.** It gives ctx_tokens = round(1,536 / bytes/token):

  | tokenizer | train bytes/token | ctx_tokens |
  |---|---|---|
  | A1-P1-D1-16k | 5.955 | 258 |
  | A6 16k-t0.9 | 6.343 | 242 |
  | A6 16k-t0.8 | 6.403 | 240 |
  | A6 32k-t0.8 | 6.988 | 220 |

- **Fertility rises slightly** (1.211 → 1.223 at 16k-t0.9), for two reasons. Stage 1 stops at t < T, so it has fewer within-word merges. The harness also counts a superword once for every word it touches.
- **Removing harakat changes the segmentation of about 28% of affected words** under SuperBPE, against 12% under A1-16k. A changed stage-1 segmentation of one word changes which superword forms with its neighbour. PLAN §7 lists robustness as tie-breaker (d).
- **Morphology** (SILVER sets, report only): silver-high boundary F1 is 0.105 (A1-16k), 0.129 (t0.9) and 0.164 (t0.8); MorphScore is 0.600, 0.603 and 0.641. Details are in each `summary.json`.

## 6. The 32k ratio

- **The rule** (`rule_32k.json`, written 2026-09-26T12:22:43Z, before any harness metric of either 16k build existed):
  1. drop any variant that fails a gate;
  2. otherwise apply PLAN §7's tie-breakers in order: (a) HF-native, (b) dev bytes/token, (c) R1 < 20, (d) robustness, (e) vocabulary size.
- **Outcome:** both variants pass every gate and both are HF-native, so (b) decides: 0.8 (7.465) over 0.9 (7.359). Criterion (c) agrees (326 vs 369).
- **Caveat:** PLAN §2.2's prior ranks 0.9 above 0.8, and rank 8 is meant to take its ratio from Stage 3 bpb. If bpb favours 0.9, build the other 32k variant:
  `python build_stage1.py 29491 stage1/a1_p1_29491.json` then
  `python build_stage2.py --stage1 stage1/a1_p1_29491.json --T 32768 --out sbpe_32k_t090 --py-check`
  That takes about 10 minutes plus about 12 minutes of checks.

## 7. Runtimes (wall clock, shared laptop CPU, ≤ 2 processes, 3 rayon threads for training)

| step | 16k-t0.9 | 16k-t0.8 | 32k-t0.8 |
|---|---|---|---|
| stage 1 (A1 to t) | 29.7 s | 26.0 s | 89.1 s |
| chunk extraction (encode train_D1 under S2 + P1 check) | 68.1 s | 125.1 s | 114.9 s |
| Rust PUA trainer (uncapped) | 35.6 s | 68.5 s | 102.5 s |
| pure-Python trainer, uncapped (cross-check) | 72.2 s | 108.9 s | 173.9 s |
| pure-Python trainer, capped (shipped) | 70.6 s | 162.2 s | 158.7 s |
| whole official build (with the cross-check) | 190.6 s | 525.3 s | 628.0 s |
| G4 rebuild (no cross-check) | 361.8 s | 334.9 s | 369.7 s |
| verify.py (all checks) | about 4 min | about 3 min | about 3 min |
| multi-word G2 | 228 s | 206 s | 208 s |
| harness `--gates all` | 141 s | 127 s | 118 s |

- The spread between builds reflects contention: 16k-t0.8 ran beside verify and G2 jobs.
- Plain HF encoding of dev_strict takes 2.2–3.4 s. The pure-Python reference takes 8.6–14.8 s.
- Peak memory was not measured.

## 8. Files (`F:\Hindko\_tokenizer\candidates\superbpe\`; sha256 of every file in `MANIFEST.json`)

| path | content |
|---|---|
| `sbpe_16k_t090/`, `sbpe_16k_t080/`, `sbpe_32k_t080/` | `tokenizer.json` (the candidate; stock HF), `stage2_pua.json` (PUA-space merges as stage-1 id sequences), `build_info.json` (parameters, runtimes, uncapped vs capped, hashes), `verify.json`, `g2_multiword.json` |
| `stage1/` | A1-P1 at 13,107 / 14,746 / 16,384 / 26,214 / 32,768 (+ `.info.json`) |
| `results/dev_strict/<id>/` | harness `summary.json` + per-document `docs.jsonl`, for the bootstrap of PLAN §6 |
| `g4/` | from-scratch retrains used by G4 |
| `results_superbpe.json`, `s2_differential.json`, `rule_32k.json`, `MANIFEST.json` | aggregated results, regex differential, 32k rule, hashes plus candidate entries (`encoder_kind: hf`) for the LM stage |
| `common.py`, `build_stage1.py`, `build_stage2.py`, `ref_encoder.py`, `verify.py`, `gates_sbpe.py`, `report.py` | code |
| `work/`, `g4/work/`, `logs/` | chunk caches (pickles; can be regenerated) and logs |

**Reproduce one build.**
1. `python build_stage1.py 14746 stage1/a1_p1_14746.json`
2. `python build_stage2.py --stage1 stage1/a1_p1_14746.json --T 16384 --out sbpe_16k_t090 --py-check`
3. `python verify.py sbpe_16k_t090 --train-every 20`
4. `python gates_sbpe.py sbpe_16k_t090`
5. `python ../../eval/harness.py run --tokenizer-json sbpe_16k_t090/tokenizer.json --name A6-SBPE-P1-D1-16k-t090 --gates all --g4-retrain g4/sbpe_16k_t090/tokenizer.json --nsl-ref ../standard/results/dev_strict/A1-P1-D1-16k --out results/dev_strict/A6-SBPE-P1-D1-16k-t090 --threads 2`

Run every step with `PYTHONIOENCODING=utf-8`.

## 9. What I did not do, and open questions

**Not done:**
- No LM training and no bpb (Stage 3).
- Nothing on the test split.
- No 32k build with t/T = 0.9, and no other t/T values.
- No BoundlessBPE or SuperBPE-fork cross-check. That would need a git clone, and none was made.
- No R3 (post-LM embedding norms).
- No MorphScore download.
- The paper was read only through the arXiv HTML page (web-fetch summary), not the PDF.
- The Rust trainer cannot run the capped variant, so the shipped merges rest on the pure-Python trainer. That trainer is validated only in uncapped mode, where it matched Rust exactly; capped mode differs only by the forbidden-pair skip.

**Open questions for the plan owner:**
1. Amend PLAN §4.3's multi-word G2 clause to the reachability reading used here (§4). The literal reading cannot be passed by any SuperBPE.
2. Is the stricter S2, which splits at every punctuation mark, wanted? The alternative is the paper's looser stage 2, where only whitespace splitting is dropped, which would allow e.g. `، تے`-type superwords. The task specified the stricter form.
3. The 32k ratio should be revisited with Stage 3 bpb (§6).
4. The candidates/standard sweep should confirm that its A1-P1-D1-32k matches `stage1/a1_p1_32768.json` (sha256 `c619826034785f68…`), or replace the A1-32k row here.
