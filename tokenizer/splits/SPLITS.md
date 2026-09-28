# Hindko corpus: train / validation / test splits

Leak-free, group-disjoint splits of the **permissive** dataset
(`hindko_dataset_permissive.jsonl`, 18,283 records, 5.75 M words). They are
built for evaluating tokenizers and language models.

Every strict record (`hindko_dataset.jsonl`, 12,393 records) inherits its
permissive record's split, so strict-only evaluation works out of the box.
No text files are written here. Text is materialised later, after the
normalisation spec is fixed (see *Using the splits*).

| file | what |
|---|---|
| `split_manifest.jsonl` | one line per permissive record: `uid, split, group, source, site_or_folder, quality_tier, language_variety, n_words`, plus `assignment` (`group` or `leak_moved_to_train`) and `upstream_split` (Omnilingual only) |
| `splits_report.json` | every number below, plus: rules, seed, config, input and code hashes, group lists per split, every moved record with its best match, and leakage distributions |
| `make_splits.py` | the reproducible builder (deterministic: two runs give a byte-identical manifest) |
| `load_split.py` | `iter_split(split, tier, sources, varieties, drop_upstream_heldout)` yields `(uid, text, record)`; `python load_split.py` verifies the manifest and prints the table below |
| `test_load_split.py` | 34 checks of the loader and the split invariants (all pass) |
| `probe_*.py`, `probe_*_out.json`, `probe_*_log.txt` | the diagnostics behind the grouping decisions, run on the rule groups before the near-duplicate merge (see *Why these groups*) |
| `make_splits.log` | console log of the published run |

## Result

Word shares per source (target 90 / 5 / 5, tolerance +/-1.5 points):

| source | train | validation | test | strict val / test |
|---|---|---|---|---|
| newspaper (1.36 M words) | 1,223,173 (90.02 %) | 67,796 (4.99 %) | 67,810 (4.99 %) | 5.05 % / 5.03 % |
| book (4.08 M) | 3,673,298 (90.01 %) | 203,824 (4.99 %) | 204,032 (5.00 %) | 5.01 % / 5.00 % |
| web (0.31 M) | 283,437 (90.00 %) | 15,659 (4.97 %) | 15,821 (5.02 %) | 5.00 % / 5.00 % |
| **all** | 5,179,908 words, 16,015 records | 287,279 words, 1,358 records | 287,663 words, 910 records | 172,226 / 171,769 words |

Groups per split: newspaper 226 / 15 / 15, book 89 / 13 / 9, web 247 / 28 / 29.
No group holds more than 22 % of its source's words in an evaluation split.

**Language variety of the test set** (words): 64.7 % hindko, 29.8 % urdu,
5.4 % mixed. Validation: 64.8 % hindko, 31.4 % urdu. Almost all the
Urdu-dominant text is from books, which are 40 % Urdu-variety as a whole (for
example Urdu criticism, translations, and poetry collections).

- Newspaper test is 97.0 % hindko; web test is 85.7 % hindko.
- The **strict** test set is 99.5 % hindko (0.5 % mixed).
- Each evaluation split matches its source's mix to within about 1.5 points:
  - variety (all sources);
  - numbered-issue vs dated-folder newspaper text (20.1 % and 20.5 % vs 20.4 %);
  - book verse share (14.0 % and 14.2 % vs 13.8 %);
  - web register, within the granularity of a whole speaker: spontaneous speech 21.8 % / 37.0 % vs 31.5 %.
- Report both: headline Hindko numbers with `varieties={'hindko'}` and/or `tier='strict'`, and the full sets for completeness.

## Leakage

A validation/test record counts as **leaked** if either:

- **containment ≥ 0.30:** the share of its distinct word 8-grams found in any train record; or
- **Jaccard ≥ 0.50:** the exact 5-character-shingle Jaccard with any train record.

Both are measured on an aggressive *leak normalisation*, used for matching only:

- NFKD;
- all marks and harakat removed;
- Arabic yeh/kaf/heh variants folded to the Urdu letters;
- Arabic-Indic digits folded to ASCII;
- tatweel, ZWNJ and format characters removed;
- punctuation dropped.

So the splits stay leak-free whatever digit and presentation-form policy the normalisation stage adopts.

| | validation | test |
|---|---|---|
| leaked if whole groups had been assigned | 93 of 1,452 records (9,119 words) | 112 of 1,046 records (38,839 words) |
| leaked after resolution | **0** of 1,358 | **0** of 910 |

- Resolution moved **230 records (55,417 words)** from 40 evaluation groups to train.
- For comparison, the literal procedure ("move what leaks, re-measure, repeat") converges in 4 rounds (205 → 5 → 3 → 0) and moves 213 records.
- The published rule is stricter. Each record is tested against **everything outside its own group**, not just train. That removes 17 more records, so evaluation records also have no ≥-threshold overlap with the other evaluation split or with other groups of their own split.

Final distribution over the 2,268 evaluation records:

- **Containment in train:** median 0; p95 0.090; p99 0.228; max 0.294.
  - 1,811 records share no 8-gram with train;
  - 32 records fall in (0.2, 0.3].
- **Best exact Jaccard in train:** max 0.471.
  - 2,229 records have no train candidate with a MinHash estimate ≥ 0.25.
- **Validation vs test:** 0 records over either threshold; containment max 0.24 (p99 0.05), Jaccard max 0.25.

Independent cross-checks, all 0 over threshold:

- hp.clean.Deduper's own normalisation and shingles: max Jaccard 0.466;
- 8-grams on raw, un-normalised whitespace tokens: max containment 0.289;
- exact Jaccard against **every** train record that shares ≥ 1 8-gram with an evaluation record (11,221 pairs, no MinHash involved): max 0.465.

MinHash uses 128 permutations and compares **all** pairs (no LSH banding). Its measured error SD is 0.040 (theory 0.044). The lowest estimate seen for any pair whose exact Jaccard is ≥ 0.5 was 0.42, well above the 0.25 verification floor.

Corpus-wide (independent of the split), the share of each source's words that duplicates text outside its own group is:

- newspaper **19.3 %**: syndication and recurring weekly columns and reports;
- book 2.3 %;
- web 0.75 %.

## Why these groups

- **Newspaper = edition.**
  - The group is the issue number, else the `source_path` folder.
  - Editions sharing a day-precision date are merged: 10 merges. The numbered tree (`2023-2024-2025/<issue>`) and the dated tree (`2024 to 2026/<month>/<d.m.y>`) hold the same editions; for example, issue 321 and folder `8.5.2024` both carry 2024-05-08.
  - Numbered issues hold only a few pages each, and most have no date. `probe_group_overlap.py` found no general one-to-one twin structure, so no date-free edition matching was attempted beyond the next rule.
- **Near-duplicate groups (all sources).** Two groups of the same source that share ≥ 25 % of the smaller one's distinct 8-grams are merged (8-grams found in more than 10 groups are formulae and not counted). This collapsed 851 rule groups to 671. Merges include:
  - issue 307 with folder `17.1.24` (35 %; issue 306 ↔ `10.1.2024` fits consecutive weeks);
  - the 98th Farma dictionary with "Dictionary with matlan adition" (37 %);
  - the 106th and 59th Farma anthologies;
  - chains of numbered issues re-running the same columns;
  - 118 tvshia pages that are near-copies of each other.

  The dry run is `probe_overlap_merge.py`. Cross-source copies, such as the 319 newspaper serials in books, are **not** merged, to keep per-source stratification; the leak rule handles them.
- **Book = `book_folder`.**
- **Omnilingual ASR = speaker.**
- **Common Voice = record.**
- **Other web sites = the whole site, unless the site exceeds half of one evaluation split's web target (7,873 words).** Such a site is split per record (uid). This applies to noukeqalam, tvshia, gandharahindko, iattock and aaprihindko.
  - *Why:* a group above that size can never enter evaluation, because it would dominate the split and break the ±1.5-point target. noukeqalam alone is 34 % of web words and would otherwise never be evaluated.
  - Site pages are separate posts, and page-to-page copying is caught twice: by the near-duplicate merge and by the record-level leak check.

**Eligibility for evaluation.** A group can enter validation or test only if both hold:

- its leak-prone words are < 50 % of its words (64 groups fail this);
- its remaining words are ≤ half of one evaluation split's per-source target (17 groups fail this).

The second rule excludes, for example, the 920 K-word dictionary, the Pothohari dictionary and 8 large Omnilingual speakers.

**Selection** is done per source, with seed `20260926`: 200 seeded restarts of a greedy fill followed by best-improvement local search (moves and swaps). The cost combines:

- word-share and strict-share error against 5 %;
- mix distance for the attributes above;
- a hinge on the Herfindahl index of group sizes, so that no single book, edition or speaker dominates, without favouring small groups.

## Using the splits

```python
import sys; sys.path.insert(0, r'F:\Hindko\_tokenizer\splits')
from load_split import iter_split

# tokenizer / LM training data (normalise `text` with the canonical spec first)
for uid, text, rec in iter_split('train'):
    ...

# evaluation views
iter_split('test')                                  # full test, all sources
iter_split('test', tier='strict')                   # research-grade subset
iter_split('test', varieties={'hindko'})            # Hindko-variety only
iter_split('validation', sources=['newspaper'])     # per-source
iter_split('train', drop_upstream_heldout=True)     # see Omnilingual note
```

Use validation for every choice (vocabulary size, merges, hyper-parameters)
and touch test once. Report per source as well as overall: the book, newspaper
and web registers differ a lot.

## Known limitations (honest list)

- **Lexical leakage only.** Paraphrases, translations (for example an Urdu and a Hindko version of the same story) and the same news event written up afresh in another edition are not detected. The thresholds are the requested 0.30 / 0.50; 32 evaluation records sit between 0.2 and 0.3 containment.
- **Lexicon text is barely evaluated.** Dictionaries are 27.6 % of book words but mostly sit in groups too big to hold out. Book validation has 0 % lexicon and test 9.2 %.
- **Omnilingual upstream split not honoured.**
  - Omnilingual ASR's own dev+test speakers hold 44,215 words (14.0 % of web), more than the whole 5 % + 5 % web evaluation budget.
  - Here, upstream test speakers spk03, spk07 and spk13, and dev speakers spk08 and spk12, are in **our train**. spk04 (upstream dev) is in our test.
  - If a model will be scored on the Omnilingual ASR benchmark, train with `drop_upstream_heldout=True`.
- **Web evaluation is page-disjoint, not site-disjoint,** for the five large sites.
- **Newspaper edition identity is only partly recoverable** between the two folder trees: through shared dates and ≥ 25 % text overlap. A remaining same-edition pair would share less than that, and any shared text is still removed by the leak rule.
- **Partial groups:** 40 evaluation groups had leak-prone records moved to train (`assignment = leak_moved_to_train`). Those moved records duplicate text found in other groups, so train gains no evaluation text. Strictly speaking, though, those books and editions are not 100 % held out.
- **Not done here:** text materialisation (it waits for normalisation), and any decontamination against external benchmarks.

## Reproduce

```bat
set PYTHONIOENCODING=utf-8
python F:\Hindko\_tokenizer\splits\make_splits.py      :: ~6-11 min on this shared machine, peak ~1.3 GB, 3 threads
python F:\Hindko\_tokenizer\splits\test_load_split.py
python F:\Hindko\_tokenizer\splits\load_split.py
```

Inputs are read-only. `splits_report.json` records their sha256 and the
builder's sha256. If the dataset is rebuilt, re-run `make_splits.py`:
`load_split.py` refuses to serve a record whose uid, tier or word count no
longer matches the manifest.
