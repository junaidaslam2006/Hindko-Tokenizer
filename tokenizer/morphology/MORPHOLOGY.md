# Hindko inflectional morphology — SILVER segmentation set for tokenizer evaluation

> **Status: SILVER, not gold.** No native Hindko annotator was available. Every
> segmentation was induced automatically from paradigm evidence in the corpus and
> then read by a reviewer who is **not** a native speaker (an AI model with working
> knowledge of Urdu / Panjabi / Hindko). Use the numbers as a *relative* signal for
> comparing tokenizers, not as an absolute measure of morphological correctness.
> A native Hindko speaker should check `morph_silver_high.tsv` before any
> published claim.
>
> Built 2026-09-26 from the **strict** release (`F:\Hindko\hindko_dataset.jsonl`,
> read-only). Deterministic: re-running the scripts reproduces the files byte for
> byte (verified with md5 over two full runs).

## 1. What is here

| file | content |
|---|---|
| `morph_silver_high.tsv` | 518 word forms whose stem\|suffix boundary is unambiguous by paradigm evidence **and** by the word's own contexts (§4.4, §4.6) |
| `morph_silver_low.tsv` | 236 plausible but uncertain word forms (3 with a KWIC-confirmed alternative segmentation) |
| `morph_eval.py` | MorphScore-style evaluator (library + CLI) |
| `tests/test_morph_eval.py` | 11 checks: the evaluator (trivial tokenizers, markers, byte-level, alternatives) and the gold files (no reported homograph, no one-letter-stem word, every alternative reviewed, every high verb passes the context check) |
| `inventory/hindko_inflection_inventory.tsv` / `.json` / `inventory_table.md` | the inflection inventory with sources, dialect notes, corpus counts and examples (146 rows) |
| `eval_results/` | outputs of the evaluator on trivial tokenizers and (illustration only) on some existing tokenizers |
| `review/review_decisions.tsv` | every manual review decision with its reason: 232 decisions on 232 words (190 drop, 24 demote to low, 3 alternative confirmed, 15 alternative rejected); the 80 rows after the `#round2` marker are the second round, and 11 round-1 rows were changed to drop in round 2 (their round-2 reason follows `\|\| round 2:`) (§4.6) |
| `review/context_sheet_high.tsv`, `context_sheet_low.tsv` | every gold entry with its context metric and its six most frequent left and right neighbours: the sheet read in round 2 |
| `data/candidates_all.tsv` | all 5305 analysed candidate words with parse, evidence, context metric, offered alternatives and decision (audit trail) |
| `data/induce_params.json` | every parameter and count of the induction run |
| `data/wordfreq_strict.*`, `data/urdu_vs_hindko_lines.json` | corpus counts the method uses |
| `data/context_profiles.json` | for every word with frequency ≥ 10: classes of the next and previous token inside the sentence, and the top-10 neighbours (§4.4) |
| `scripts/` | `build_counts.py` → `context_profiles.py` → `induce.py` → `build_inventory.py` → `context_sheet.py` → `render_docs.py`; `morph_inventory.py` is the curated grammar data; review helpers `_kwic.py` (KWIC lines), `_merge_review.py`, `_diff_gold.py`, `_show.py` |
| `sources/` | the open-access paper used (Language in India 2011, PDF + text) and the Hindko phrasebook text extracted from the corpus |
| `_pylib/` | `pypdf` 6.19.0 (pure Python, installed with `--target` here only to read that PDF) |

### Gold file format (both TSVs)

| column | meaning |
|---|---|
| `word` | the word form, NFC, **harakat stripped** (`hp.lang.strip_marks`), as the corpus spells it |
| `segmentation` | required boundaries shown with `\|`, e.g. `کر\|دیاں`, `بنڑ\|ا\|یا` |
| `boundaries` | the same boundaries as character offsets into `word` (`2`, `3,4`) |
| `optional_boundaries` | offsets inside a suffix where a split is defensible (`-د\|یاں`, `-س\|ی`, `-نڑ\|اں`): neither rewarded nor penalised |
| `alternatives` | low set only: other acceptable boundary sets, `;`-separated, each one a reading the review found in the corpus by KWIC (`دوریاں`: `3` primary = `دور\|یاں` 'tours', `4` alternative = `دوری\|اں` 'distances') |
| `stem`, `suffixes`, `gloss`, `category`, `family` | analysis (Leipzig-style gloss) |
| `freq`, `PESH`, `BOOK`, `HAZ`, `n_units` | strict-corpus frequency, per dialect group, and number of issues / books / sites it occurs in |
| `dialect` | distributional label of the word (see §3.3) with its group counts |
| `urdu_ratio` | relative frequency in Urdu-dominant lines ÷ in Hindko-dominant lines (§4.5) |
| `next_is_postposition` | share of occurrences followed by a postposition (used to label -اں nouns vs verbs) |
| `context` | verbs only: the word's own-context metric of §4.4 (`nom`, or `obl` and `aux` for oblique infinitives, or `detq` for other infinitives) |
| `slots`, `evidence`, `caus_forms` | **the paradigm evidence**: attested members with their counts |
| `notes`, `confidence`, `top_surface` | why the word is low (if it is), and its most frequent spelling with harakat |

## 2. Sources and what was (not) consulted

| key | source | how it was used |
|---|---|---|
| RRS2011 | Raja, N. A., Haroon-ur-Rashid & A. Sohail (2011). *A Brief Introduction of Hindko Language.* Language in India 11(11): 471–482 (open access; variety: Muzaffarabad / Hazara belt). | Read in full. Plural `-an` (chowkan, zatan, gallan); verb `paR-daa` (present), `paR-yaa` (past), `paR-sii` (future); agreement `likh-Daa / likh-Dii / likh-De`; case markers `-ne` / `sun` (ergative), `-te` (locative), `nu` (dative). |
| SH1980 | Shackle, C. (1980). *Hindko in Kohat and Peshawar.* BSOAS 43(3): 482–510. | **Not read** (paywalled). Only the statements quoted from it on Wikipedia *Kohati*: oblique `-e` (pʊttʊr → pʊtre), dative postposition `ã`, oblique plural `-ã`. |
| SH2010 | Shackle (2010), as quoted on Wikipedia *Lahnda*. | Lahnda varieties have a future in `-s-`. |
| BC2019 | Bashir, E. & T. J. Conners (2019). *A Descriptive Grammar of Hindko, Panjabi, and Saraiki.* De Gruyter Mouton. | **Content not consulted.** Only the publisher / LINGUIST List description was legitimately reachable (it describes the Abbottabad variety). The Google Books search-inside page answered with a CAPTCHA, which was not bypassed; the De Gruyter page refused automated access (HTTP 405). The unauthorised archive.org copy and a Scribd upload that appeared in search results were **not** used. No item below is attributed to this grammar. |
| BANO | Aftab Iqbal Bano, *آؤ ہندکو زبان سکھنے آں* (Hindko Bol Chal; Gandhara Hindko Academy, Peshawar; Farma 131). | A Hindko phrasebook **inside this corpus** (permissive tier) with line-by-line Urdu glosses. Lesson 5 (pronouns) and lesson 8 (present / past / future) give Peshawari forms: `منے = میں نے`, `سانے = ہم نے`, `اُزا = اُس کا`, `سُواڈا = آپ کا`, `میرے وسے = میرے لئے`, `دفتر سی = دفتر سے`, `دیساں / آسیں / جاسی / جاسُن` (future 1SG / 2SG / 3SG / 3PL), `جاسی ایں` (1PL, written apart), `گیا ایا = گیا تھا`, `کرنا واں` (PRS.1SG). Extract: `sources/corpus_grammar_book_aao_hindko.txt`. |
| CORPUS | the strict release, counted per dialect group (§3.3) | distributional evidence only |

**Consequence.** The Hazara (Abbottabad) side of the inventory rests on RRS2011 and
on corpus distribution, not on the standard reference grammar. Checking the
inventory against Bashir & Conners (2019), chapters on nouns, pronouns and verbs,
is the first thing to do with library access.

## 3. The inventory

### 3.1 Key findings

* **Retroflex ṇ is spelled `نڑ`** in every source of this corpus: 54,541
  occurrences of `ن+ڑ` in the strict corpus, against 15 of `ݨ` (U+0768,
  all in the Omnilingual transcripts) and 0 of `ڻ`. The 496 `ن+ط` in book
  decodes are an InPage glyph (ن with small ط) decoded as two letters (e.g. BANO
  `ہونیط ولی` = `ہونڑیں ولی`). Consequence: the infinitive is `-نڑاں / -نڑا / -نڑیں /
  -نڑے`, and after `ر` it is `-نا / -نے / -ناں` (`کرنا`, `کرنے`, `کرناں`).
* **Peshawari vs Hazara, by corpus distribution** (Hazara group = 64K tokens, mostly
  transcribed speech, so register differs; see §3.3):
  locative `اچ` (PESH 25,725 / HAZ 7) vs `بچ` (HAZ 672) and book `وچ`; dative `نوں`
  (HAZ 8) vs `کو` (HAZ 746, Hazara-enriched); benefactive `وسے` (HAZ 1) vs `واسطے`;
  past copula `ایا / ائی / ائے` (HAZ ≤ 4; BANO) vs `آسا / اسا / سا / آسی / آسے`
  (all Hazara-enriched); oblique infinitive `-نڑیں` (HAZ 19 where 134 are expected;
  BANO attests it) vs `-نڑا` (Hazara-enriched); `آپڑا` (Peshawari newspaper spelling)
  vs `اپڑا` (books) vs `اپنا` (Hazara-enriched / Urdu); 1SG dative `منوں` vs Hazara
  `موکو`; 3SG genitive `ازا` (BANO) vs Hazara `ایندا`.
* The `-s-` future (`-سی`, `-ساں`, `-سن`) is attested in every dialect group; it is also
  often written as a separate word (`ہو سی`, `کر سُن`), and only joined spellings enter the gold set.
* Oblique plural of masculine `-ا/-ہ` nouns is `-یاں` (`علاقیاں`, `اداریاں`, `حلقیاں`,
  Peshawari `کیہنٹیاں` 'hours'); Urdu has `-وں`. BANO writes the same ending apart
  after `-ے`: `بچے اں`, `جملے آں`.

### 3.2 Bound morphology (suffixes), with corpus evidence

Columns: literature statement [source] · corpus distribution label · analysed word
types / tokens · most frequent examples with their strict-corpus counts.

| id | form | gloss | dialect (literature) | dialect (corpus) | types / tokens | corpus examples |
|---|---|---|---|---|---|---|
| V.IPFV.دا | -دا | IPFV.PTCP.M.SG | shared [RRS2011 paR-daa, tur-Daa (Muzaffarabad); BANO کردا وے (Peshawar)] | Hazara-enriched | 67 / 6507 | کر\|دا:1240 سک\|دا:1141 لگ\|دا:482 مل\|دا:333 جل\|دا:285 |
| V.IPFV.دی | -دی | IPFV.PTCP.F.SG | shared [RRS2011 likh-Dii (3b); BANO کردی اے] | shared | 56 / 3459 | کر\|دی:554 سک\|دی:548 لگ\|دی:341 مل\|دی:286 چل\|دی:121 |
| V.IPFV.دے | -دے | IPFV.PTCP.M.PL/OBL | shared [RRS2011 likh-De (4a); BANO کردے] | shared | 77 / 9289 | کر\|دے:2830 جل\|دے:781 سک\|دے:596 آخ\|دے:469 رکھ\|دے:409 |
| V.IPFV.دیاں | -دیاں | IPFV.PTCP.F.PL | shared [RRS2011 (paR-daa, likh-Daa); BANO (کردا، ہوندا)] | Hazara-enriched | 21 / 663 | کر\|دیاں:174 جل\|دیاں:64 مل\|دیاں:53 سک\|دیاں:49 دیکھ\|دیاں:41 |
| V.IPFV.ندا | -ندا | IPFV.PTCP.M.SG | shared [BANO ہوندا وے (Peshawar)] | shared | 45 / 9703 | ہو\|ندا:3491 جا\|ندا:2450 دی\|ندا:612 رہ\|ندا:541 پی\|ندا:331 |
| V.IPFV.ندی | -ندی | IPFV.PTCP.F.SG | shared [RRS2011 (paR-daa, likh-Daa); BANO (کردا، ہوندا)] | shared | 38 / 8828 | ہو\|ندی:3289 جا\|ندی:2886 دی\|ندی:453 رہ\|ندی:364 کہ\|ندی:250 |
| V.IPFV.ندے | -ندے | IPFV.PTCP.M.PL/OBL | shared [RRS2011 (paR-daa, likh-Daa); BANO (کردا، ہوندا)] | shared | 65 / 11048 | ہو\|ندے:3254 جا\|ندے:1532 دی\|ندے:1167 رہ\|ندے:560 کہ\|ندے:489 |
| V.IPFV.ندیاں | -ندیاں | IPFV.PTCP.F.PL | shared [RRS2011 (paR-daa, likh-Daa); BANO (کردا، ہوندا)] | shared | 14 / 1165 | ہو\|ندیاں:571 جا\|ندیاں:234 رہ\|ندیاں:70 دی\|ندیاں:68 رو\|ندیاں:39 |
| V.PFV.یا | -یا | PFV.PTCP.M.SG | shared [RRS2011 paR-yaa, tur-yaa, lang-yaa (Muzaffarabad)] | shared | 139 / 20866 | ہو\|یا:4999 کہ\|یا:3295 آخ\|یا:1388 فرما\|یا:1276 رہ\|یا:745 |
| V.PFV.ی | -ی | PFV.PTCP.F.SG | shared [RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)] | shared | 86 / 8786 | لگ\|ی:764 رکھ\|ی:666 مل\|ی:537 لکھ\|ی:410 ہک\|ی:402 |
| V.PFV.ے | -ے | PFV.PTCP.M.PL \| SBJV.3SG | shared [RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)] | shared | 92 / 14066 | کر\|ے:1703 لگ\|ے:1083 پچھ\|ے:945 جل\|ے:813 سک\|ے:704 |
| V.PFV.یاں | -یاں | PFV.PTCP.F.PL | shared [RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)] | shared | 51 / 2790 | ہو\|یاں:771 لگ\|یاں:235 رہ\|یاں:211 لکھ\|یاں:148 بول\|یاں:112 |
| V.PFV.ا | -ا | PFV.PTCP.M.SG (-aa) | shared (Urdu-like) [BANO پڑھا ایا (Peshawar)] | shared | 87 / 11387 | لگ\|ا:1341 چل\|ا:790 بنڑ\|ا:789 رکھ\|ا:717 لکھ\|ا:688 |
| V.PFV.ئی | -ئی | PFV.PTCP.F.SG | shared [RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)] | shared | 52 / 6632 | ہو\|ئی:3278 پا\|ئی:482 بنڑ\|ا\|ئی:312 لا\|ئی:202 سو\|ئی:160 |
| V.PFV.ئے | -ئے | PFV.PTCP.M.PL | shared [RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)] | shared | 42 / 6386 | ہو\|ئے:4804 پا\|ئے:194 لگ\|ا\|ئے:147 بنڑ\|ا\|ئے:146 لیا\|ئے:143 |
| V.PFV.ئیاں | -ئیاں | PFV.PTCP.F.PL | shared [RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)] | shared | 12 / 783 | ہو\|ئیاں:438 جا\|ئیاں:119 لا\|ئیاں:46 پا\|ئیاں:42 بنڑ\|ا\|ئیاں:38 |
| V.FUT.سی | -سی | FUT.3SG | shared [SH2010 (-s- future of Lahnda); RRS2011 paR-sii; BANO جاسی، ہوسی] | shared | 34 / 6434 | ہو\|سی:2168 جا\|سی:1742 کر\|سی:575 رہ\|سی:336 دی\|سی:264 |
| V.FUT.ساں | -ساں | FUT.1SG | Peshawari attested [BANO دیساں (1SG), دیساں = دونگی] | Hazara-enriched | 33 / 1850 | کر\|ساں:543 دی\|ساں:268 جا\|ساں:184 رہ\|ساں:133 لی\|ساں:84 |
| V.FUT.سیں | -سیں | FUT.2SG | Peshawari attested [BANO آسیں (2SG), کرسیں] | absent/rare in Hazara sources (HAZ 1, expected 6) | 8 / 340 | جا\|سیں:83 کر\|سیں:74 ہو\|سیں:68 رہ\|سیں:39 دی\|سیں:34 |
| V.FUT.سن | -سن | FUT.3PL | Peshawari attested [BANO جاسُن (3PL); written apart: پڑھ سُن] | shared | 15 / 1331 | جا\|سن:424 ہو\|سن:342 کر\|سن:195 رہ\|سن:102 دی\|سن:66 |
| V.FUT.سو | -سو | FUT.2PL | - [CORPUS only] | Hazara-enriched | 6 / 150 | کر\|سو:55 ہو\|سو:30 جا\|سو:19 دی\|سو:19 جل\|سو:16 |
| V.INF.نا | -نا | INF \| PRS.1 (-naa, BANO: دینا واں) | shared (also Urdu INF) [BANO دینا واں, کرنا واں = PRS.1SG] | shared | 38 / 6359 | کر\|نا:4291 چاہ\|نا:265 مار\|نا:174 پڑھ\|نا:159 سمجھ\|نا:149 |
| V.INF.نے | -نے | INF.OBL | -نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara [CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)] | shared | 49 / 9605 | کر\|نے:6286 پڑھ\|نے:472 مل\|نے:345 بول\|نے:292 سنڑ\|نے:223 |
| V.INF.نڑاں | -نڑاں | INF | - [CORPUS only] | shared | 53 / 3122 | دی\|نڑاں:457 جا\|نڑاں:365 کھا\|نڑاں:228 کہ\|نڑاں:196 لی\|نڑاں:180 |
| V.INF.نڑا | -نڑا | INF | - [CORPUS only] | Hazara-enriched | 28 / 1272 | ہو\|نڑا:236 جا\|نڑا:174 کھا\|نڑا:169 دی\|نڑا:125 رہ\|نڑا:70 |
| V.INF.نڑیں | -نڑیں | INF.OBL | Peshawari attested [BANO ہونیط ولی, دینیط دا (decode writes the n-with-small-tah glyph as ن+ط)] | absent/rare in Hazara sources (HAZ 19, expected 131) | 64 / 7094 | ہو\|نڑیں:1880 دی\|نڑیں:784 جا\|نڑیں:773 پا\|نڑیں:509 لی\|نڑیں:368 |
| V.INF.نڑے | -نڑے | INF.OBL | - [CORPUS only] | shared | 59 / 4394 | ہو\|نڑے:1089 جا\|نڑے:793 دی\|نڑے:260 رکھ\|نڑے:153 کھا\|نڑے:152 |
| V.INF.نڑ | -نڑ | INF.SHORT | -نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara [CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)] | no analysed words | 0 / 0 |  |
| V.INF.ناں | -ناں | INF | -نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara [CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)] | Hazara-enriched | 26 / 1065 | کر\|ناں:294 سک\|ناں:158 کہ\|ناں:87 منگ\|ناں:74 رکھ\|ناں:61 |
| V.INF.نی | -نی | INF.F.SG | Peshawari attested [BANO پریکٹس کرنی وے (obligation, F)] | shared | 29 / 1180 | کر\|نی:569 چاہ\|نی:72 سک\|نی:57 کہ\|نی:49 رہ\|نی:40 |
| V.INF.نیاں | -نیاں | INF.F.PL | -نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara [CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)] | shared | 10 / 349 | کر\|نیاں:103 سک\|نیاں:72 چاہ\|نیاں:72 دی\|نیاں:23 ہو\|نیاں:21 |
| V.INF.نڑی | -نڑی | INF.F.SG | -نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara [CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)] | shared | 10 / 430 | ہو\|نڑی:195 دی\|نڑی:69 جا\|نڑی:41 او\|نڑی:32 گہا\|نڑی:26 |
| V.INF.نڑیاں | -نڑیاں | INF.F.PL | -نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara [CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)] | Hazara-enriched | 7 / 184 | گہا\|نڑیاں:51 سو\|نڑیاں:38 پا\|نڑیاں:37 گا\|نڑیاں:23 کھا\|نڑیاں:13 |
| V.SBJV.اں | -اں | SBJV.1SG | shared [BANO (کردیوے، ہووے); CORPUS] | Hazara-enriched | 43 / 3603 | کر\|اں:883 لکھ\|اں:409 سوچ\|اں:268 دکھ\|اں:209 جل\|اں:147 |
| V.SBJV.یں | -یں | SBJV.2SG | shared [BANO (کردیوے، ہووے); CORPUS] | Hazara-enriched | 11 / 305 | بنڑ\|یں:87 سنڑ\|یں:63 جل\|یں:29 ویکھ\|یں:22 دس\|یں:19 |
| V.SBJV.و | -و | SBJV/IMP.2PL | shared [BANO (کردیوے، ہووے); CORPUS] | Hazara-enriched | 50 / 4006 | کر\|و:1044 دی\|و:367 دیکھ\|و:265 دیخ\|و:241 پچھ\|و:158 |
| V.SBJV.ن | -ن | SBJV.3PL \| INF.OBL | shared [BANO (کردیوے، ہووے); CORPUS] | shared | 42 / 2079 | کر\|ن:820 سک\|ن:149 جل\|ن:128 آخ\|ن:72 مل\|ن:67 |
| V.SBJV.واں | -واں | SBJV.1SG | shared [BANO (کردیوے، ہووے); CORPUS] | Hazara-enriched | 26 / 1323 | جا\|واں:329 دی\|واں:216 ہو\|واں:124 کہ\|واں:71 پا\|واں:67 |
| V.SBJV.ویں | -ویں | SBJV.2SG | shared [BANO (کردیوے، ہووے); CORPUS] | Hazara-enriched | 10 / 418 | دی\|ویں:120 جا\|ویں:101 ہو\|ویں:65 رہ\|ویں:30 چاہ\|ویں:25 |
| V.SBJV.وے | -وے | SBJV.3SG | Peshawari attested [BANO کردیوے، ہووے] | shared | 38 / 9698 | ہو\|وے:4090 جا\|وے:3176 دی\|وے:582 رہ\|وے:359 ہوجا\|وے:222 |
| V.SBJV.ون | -ون | SBJV.3PL | shared [BANO (کردیوے، ہووے); CORPUS] | shared | 19 / 1793 | ہو\|ون:668 جا\|ون:580 دی\|ون:151 کھا\|ون:51 پا\|ون:46 |
| V.CONJ.کے | -کے | CONJ.PTCP | shared [BANO جاکے فاتحہ] | shared | 21 / 2824 | کر\|کے:1168 ہو\|کے:517 لی\|کے:413 جا\|کے:175 پا\|کے:128 |
| V.CAUS.وا | -وا | CAUS2 formative (+ vowel-stem endings) | shared [CORPUS] | shared | 13 / 335 | کر\|وا\|یا:77 کر\|وا\|ئی:69 کر\|وا\|ندے:39 کر\|وا\|نڑیں:29 کر\|وا\|ئے:24 |
| V.CAUS.ا | -ا | CAUS formative (+ vowel-stem endings) | shared [CORPUS] | shared | 165 / 6004 | بنڑ\|ا\|یا:626 بنڑ\|ا\|ئی:312 لگ\|ا\|یا:247 کر\|ا\|ئی:148 لگ\|ا\|ئے:147 |
| N.NOUN_C.اں | -اں | PL (F) \| OBL.PL (M) | shared [RRS2011 chowkan, zatan, gallan; SH1980 obl.pl -a~ (Kohat); BANO شاگرداں، کتاباں] | shared | 652 / 48744 | لوک\|اں:3541 کتاب\|اں:1775 گل\|اں:1741 زبان\|اں:1629 قیمت\|اں:858 |
| N.NOUN_I.اں | -اں | PL of F -ii nouns (کڑی -> کڑیاں) | shared [CORPUS] | shared | 164 / 8523 | شہری\|اں:477 کڑی\|اں:402 کہانڑی\|اں:298 گڈی\|اں:288 خوشی\|اں:285 |
| N.NOUN_A.ے | -ے | M.OBL.SG \| M.PL | shared [SH1980 pUttUr -> obl pUtre (Kohat)] | shared | 400 / 39383 | بچ\|ے:1592 صوب\|ے:1198 بند\|ے:1074 علاق\|ے:913 موقع\|ے:773 |
| N.NOUN_A.یاں | -یاں | M.OBL.PL | - [CORPUS; BANO writes the ending apart after -e: بچے اں، جملے آں] | shared | 122 / 7164 | علاق\|یاں:821 بچ\|یاں:705 موقع\|یاں:594 ادار\|یاں:486 حلق\|یاں:223 |
| AGR.ا | -ا | M.SG | shared [BANO میرا، ساڈا، سُواڈا] | shared | 519 / 64385 | کیت\|ا:10660 دت\|ا:4705 میر\|ا:3122 نینگ\|ا:1801 تیر\|ا:1727 |
| AGR.ی | -ی | F.SG | shared (lexical choice differs: بڈا Peshawari, وڈا general) [BANO; CORPUS] | shared | 420 / 62010 | کیت\|ی:7692 میر\|ی:3923 آپڑ\|ی:3854 دت\|ی:2977 تیر\|ی:2257 |
| AGR.ے | -ے | M.PL/OBL | shared (lexical choice differs: بڈا Peshawari, وڈا general) [BANO; CORPUS] | shared | 482 / 74394 | میر\|ے:6405 آپڑ\|ے:5557 سار\|ے:4017 پہل\|ے:3573 تیر\|ے:3551 |
| AGR.یاں | -یاں | F.PL | shared (lexical choice differs: بڈا Peshawari, وڈا general) [BANO; CORPUS] | shared | 138 / 9281 | سار\|یاں:1339 وال\|یاں:670 کیت\|یاں:649 دت\|یاں:308 تیر\|یاں:290 |
| PRON.GEN.دا | -دا | GEN written joined to a pronoun (اسدا) | - [] | Hazara-enriched | 2 / 395 | اس\|دا:305 جس\|دا:90 |
| PRON.GEN.دی | -دی | GEN written joined to a pronoun (اسدا) | - [] | Hazara-enriched | 2 / 505 | اس\|دی:436 جس\|دی:69 |
| PRON.GEN.دے | -دے | GEN written joined to a pronoun (اسدا) | - [] | Hazara-enriched | 3 / 528 | اس\|دے:434 جس\|دے:84 کس\|دے:10 |
| PRON.GEN.دیاں | -دیاں | GEN written joined to a pronoun (اسدا) | - [] | insufficient data | 1 / 62 | اس\|دیاں:62 |
| V.COP.CLITIC | -ین / -ن (کردین، گئین، سکدین) | contracted 3PL copula on participles (کردے ہن -> کردین) | Peshawari (BANO: کہندین، کھاندین، آگئین) [BANO; CORPUS] | shared | 7 / 2834 | کردین:437 سکدین:248 گئین:470 آگئین:10 ہوگئین:44 |
| ORTH.RETRO_N | نڑ (ṇ) | retroflex nasal written ن + ڑ; ݨ (U+0768) almost unused | all sources of this corpus [CORPUS (_probe_chars.py)] | see note | 0 / 0 | strict corpus character counts: ن+ڑ 54541, U+0768 15, U+06BB 0, ن+ط 496; the book-decode artefact ن+ط comes from an InPage glyph for ن with small ط |

### 3.3 Free forms: postpositions, pronouns, auxiliaries (not segmented)

Counts are strict-corpus tokens as PESH / BOOK / HAZ.

| item | words (strict-corpus count, PESH / BOOK / HAZ) | gloss | dialect (literature) | dialect (corpus, per word) |
|---|---|---|---|---|
| P.GEN | دا 57832 (20754/35896/1085); دی 76895 (34553/41020/1169); دے 102808 (44479/56978/1173); دیاں 5069 (786/4136/143) | GEN (agrees with possessum: M.SG / F.SG / M.PL,OBL / F.PL) | shared [BANO (اُس دا = اُس کا); CORPUS] | دا: shared; دی: shared; دے: shared; دیاں: Hazara-enriched |
| P.DAT.nu | نوں 37883 (16171/21624/8) | DAT/ACC | Peshawari (BANO: سانوں، تنوں, اُس نوں) [BANO; RRS2011 (nadeem-nu, Muzaffarabad); CORPUS] | نوں: absent/rare in Hazara sources |
| P.DAT.aan | آں 11480 (3829/7399/245) | DAT/ACC (also 1SG copula / PL written apart) | Kohat/Peshawar (SH1980: Kohati a~ dative) [SH1980 via Wikipedia; CORPUS] | آں: shared |
| P.DAT.ko | کو 3540 (560/2231/746); موکو 24 (1/0/23) | DAT/ACC | unknown (also Urdu); corpus: Hazara-enriched [CORPUS] | کو: Hazara-enriched; موکو: Hazara-enriched |
| P.COM | نال 29126 (13010/15605/447); سنگ 312 (60/239/13) | with (COM/INS) | shared [CORPUS] | نال: shared; سنگ: Hazara-enriched |
| P.LOC.ich | اچ 38502 (25725/12768/7) | in (LOC) | Peshawari (newspaper standard) [CORPUS] | اچ: absent/rare in Hazara sources |
| P.LOC.bich | بچ 12416 (4820/6913/672) | in (LOC) | shared (Peshawari and Hazara) [CORPUS] | بچ: shared |
| P.LOC.vich | وچ 18533 (1211/17005/310); وچوں 274 (5/265/4) | in (LOC); وچوں from within (-وں ABL) | book / Panjabi-like spelling; BANO uses وچ [BANO; CORPUS] | وچ: Hazara-enriched; وچوں: shared |
| P.LOC.te | تے 113157 (42748/68721/1537) | on (LOC); homograph of تے "and" | shared [RRS2011 (-te locative); CORPUS] | تے: shared |
| P.ABL.ton | توں 6475 (1518/4827/129) | from (ABL); homograph of توں "you" | shared [CORPUS] | توں: shared |
| P.ABL.thin | تھیں 207 (0/207/0) | from (ABL) | Hazara (hp/lang.py R10 note: Hazara-Hindko form) [hp/lang.py; CORPUS] | تھیں: insufficient data |
| P.ABL.si | سی 26367 (12474/13589/200) | from (ABL); homograph of FUT.3SG سی and Hazara "was" | Peshawari (BANO: سواڈے سی = آپ سے; دفتر سی = دفتر سے) [BANO; CORPUS] | سی: shared |
| P.ABL.kolon | کولوں 1329 (210/1085/32) | from / by | shared (BANO: میرے کولوں = مجھ سے) [BANO; CORPUS] | کولوں: shared |
| P.BEN | وسے 8665 (5723/2941/1); واسطے 1673 (574/1003/95); آسطے 526 (272/219/35); لئی 229 (18/211/0) | for (BEN) | وسے Peshawari (BANO: میرے وسے = میرے لئے) [BANO; CORPUS] | وسے: absent/rare in Hazara sources; واسطے: Hazara-enriched; آسطے: shared; لئی: insufficient data |
| P.ERG | نے 28133 (13103/14884/129) | ERG | shared (RRS2011 also reports ergative sun) [RRS2011; BANO; CORPUS] | نے: shared |
| PR.1PL | اسی 7817 (2583/4806/419); اسیں 82 (1/53/28); اساں 3505 (852/2293/358); سانوں 1473 (463/1002/8); سانے 733 (189/544/0) | 1PL: NOM اسی/اسیں, OBL/ERG اساں, DAT سانوں, ERG سانے | اسی/سانے Peshawari (BANO: اسی = ہم, سانے = ہم نے); اسیں Hazara-enriched in corpus [BANO; CORPUS] | اسی: Hazara-enriched; اسیں: Hazara-enriched; اساں: Hazara-enriched; سانوں: shared; سانے: absent/rare in Hazara sources |
| PR.2PL | تسی 2397 (544/1713/126); تسیں 48 (4/21/23); تساں 1062 (197/753/111); سوانوں 115 (1/114/0) | 2PL/HON: NOM تسی/تسیں, OBL تساں, DAT سوانوں | تسی/سوانوں Peshawari (BANO: تسی = آپ, سوانوں = آپ کو) [BANO; CORPUS] | تسی: Hazara-enriched; تسیں: Hazara-enriched; تساں: Hazara-enriched; سوانوں: insufficient data |
| PR.3PL | انہاں 10205 (1868/8168/169); اناں 13016 (6957/5878/152); جنہاں 1159 (237/892/27); جناں 1799 (909/888/2); کنہاں 212 (3/209/0) | 3PL / REL / INTERR oblique (-ہاں) | اناں spelling Peshawari newspaper; انہاں elsewhere [BANO (اُنہاں = اُنھیں / اُن; کنہاں = کنہوں); CORPUS] | انہاں: shared; اناں: shared; جنہاں: shared; جناں: absent/rare in Hazara sources; کنہاں: insufficient data |
| PR.1SG | میں 15769 (2470/12906/390); منوں 4419 (580/3838/1); مینوں 106 (5/101/0); منے 2701 (345/2354/2); موکو 24 (1/0/23) | 1SG: NOM میں, DAT منوں/مینوں/موکو, ERG منے | منوں/منے Peshawari (BANO); مینوں Panjabi-like; موکو Hazara-enriched in corpus [BANO; CORPUS] | میں: Hazara-enriched; منوں: absent/rare in Hazara sources; مینوں: insufficient data; منے: absent/rare in Hazara sources; موکو: Hazara-enriched |
| PR.2SG | تو 6310 (930/4892/486); توں 6475 (1518/4827/129); تنوں 1550 (112/1437/1); تینوں 68 (6/62/0); تنے 742 (53/681/2) | 2SG: NOM تو/توں, DAT تنوں/تینوں, ERG تنے | تنوں/تنے Peshawari (BANO) [BANO; CORPUS] | تو: Hazara-enriched; توں: shared; تنوں: absent/rare in Hazara sources; تینوں: insufficient data; تنے: absent/rare in Hazara sources |
| PR.3SG | اوہ 10993 (3655/7196/135); او 8821 (1237/6856/704); اس 49270 (14689/32964/1539); ایس 744 (569/164/11); ایہہ 16750 (2977/13603/91) | 3SG / DEM | shared [BANO; CORPUS] | اوہ: shared; او: Hazara-enriched; اس: shared; ایس: shared; ایہہ: shared |
| PR.GEN3 | ازا 247 (42/205/0); ازی 460 (55/405/0); ازے 755 (132/623/0); اسدا 305 (5/285/15); ایندا 22 (0/3/19); ایندے 29 (0/3/26) | 3SG.GEN (his/her/its), agreeing | ازا Peshawari (BANO: اُزا = اُس کا); ایندا Hazara-enriched in corpus [BANO; CORPUS] | ازا: insufficient data; ازی: absent/rare in Hazara sources; ازے: absent/rare in Hazara sources; اسدا: Hazara-enriched; ایندا: Hazara-enriched; ایندے: Hazara-enriched |
| PR.POSS | میرا 3122 (414/2684/15); تیرا 1727 (202/1517/8); ساڈا 757 (216/537/4); سواڈا 44 (1/43/0); تہاڈا 11 (5/6/0); آپڑا 1249 (438/810/1); اپڑا 659 (87/568/4); اپنا 199 (24/151/24) | possessives, inflect like -aa adjectives (-ا/-ی/-ے/-یاں) | سواڈا Peshawari HON (BANO: سُواڈا = آپ کا); آپڑا Peshawari newspaper; اپنا Urdu/Hazara [BANO; CORPUS] | میرا: shared; تیرا: shared; ساڈا: shared; سواڈا: insufficient data; تہاڈا: insufficient data; آپڑا: absent/rare in Hazara sources; اپڑا: shared; اپنا: Hazara-enriched |
| AUX.PRS | وے 23442 (11112/12202/93); اے 51457 (19771/30001/1642); ہے 4507 (453/3070/977); ون 6586 (3497/3070/5); ہن 1908 (978/264/659); ان 4377 (1946/2313/108); واں 2778 (703/2070/5); ویں 1074 (357/710/7) | present copula: 3SG وے/اے, 3PL ون/ہن/ان, 1SG واں, 2SG ویں | وے/ون Peshawari (BANO: کردا وے, ہسپتال وچ داخل اے) [BANO; CORPUS] | وے: shared; اے: shared; ہے: Hazara-enriched; ون: absent/rare in Hazara sources; ہن: Hazara-enriched; ان: shared; واں: absent/rare in Hazara sources; ویں: shared |
| AUX.PST.P | ایا 6889 (2437/4449/1); ائی 3952 (1260/2684/4); ائے 5870 (1885/3980/2); ایاں 712 (143/546/23) | past copula (was): M.SG / F.SG / M.PL / F.PL | Peshawari (BANO: گیا ایا = گیا تھا; گئے ائے = گئے تھے) [BANO; CORPUS] | ایا: absent/rare in Hazara sources; ائی: absent/rare in Hazara sources; ائے: absent/rare in Hazara sources; ایاں: Hazara-enriched |
| AUX.PST.H | آسا 642 (411/41/185); اسا 200 (35/56/109); سا 254 (55/100/98); آسی 861 (408/346/105); آسے 718 (490/143/84) | past copula (was), Hazara | Hazara (corpus: over-represented in Hazara web/speech sources) [CORPUS] | آسا: Hazara-enriched; اسا: Hazara-enriched; سا: Hazara-enriched; آسی: Hazara-enriched; آسے: Hazara-enriched |
| AUX.PROG | پیا 3751 (1546/2177/26); پئی 2478 (974/1494/10); پئے 2223 (710/1503/10); پئیاں 101 (20/81/0) | progressive auxiliary (کردا پیا وے "is doing") | Peshawari (BANO: پیا دینا واں; کے پئے کردین) [BANO; CORPUS] | پیا: shared; پئی: shared; پئے: shared; پئیاں: insufficient data |

**Dialect groups** (per source, from the release documentation):
PESH = newspaper (Weekly Hindkowan, Peshawar) + gandharahindko + tvshia (1,233,628 tokens);
BOOK = Gandhara Hindko Academy books (2,149,019; Peshawar publisher, authors from Peshawar
*and* Hazara, so not dialect-pure); HAZ = Omnilingual ASR + Common Voice transcripts,
aaprihindko, hindko.org, hindkomaza, hazarewall (63,862); OTHER = Southern Hindko
(botanix), FineWeb-2 pages, blogs (6,227).
**Corpus label**: *Hazara-enriched* = HAZ ≥ 5 and HAZ rate ≥ 3 × PESH rate; *absent/rare in
Hazara sources* = at least 5 HAZ tokens expected at the corpus-wide rate but ≤ 1/5 of that
observed, and PESH ≥ 10; *shared* = ≥ 3 in both; otherwise *insufficient data*. These are
distributional facts, not grammar claims: the HAZ group is small and mostly speech.

## 4. How the silver set was induced

### 4.1 Candidates
Strict-corpus word types (tokenised with `hp.lang.TOKEN_SPLIT_RE`, harakat stripped)
with frequency ≥ 10, spread over ≥ 3 units (issues / books / web sites), Arabic letters
only, ≥ 3 letters, not in the closed-class list (`morph_inventory.FUNCTION_WORDS`:
pronouns, postpositions, auxiliaries, and homographs such as `ہونڑ` 'now'):
15438 types. 5305 of them have at least one parse.

### 4.2 Parses and paradigm tests
A parse writes the word as STEM + suffix(es) from the inventory, with a stem of **≥ 2
letters**. Each family has its own test on corpus counts of the *other* members of the
paradigm (a member counts at ≥ 3 occurrences and ≥ 2 % of the paradigm's largest cell):

| family | strong evidence |
|---|---|
| VERB | ≥ 2 imperfective cells (`-دا/-دی/-دے/-دیاں` or `-ندا…`) **and** ≥ 3 slots among IPFV / PFV / FUT / INF / SBJV / CONJ **and** the stem's imperfective forms are followed by an auxiliary (`وے اے ایا ائی پیا ہن ون آسا …`) in ≥ 12 % of their occurrences |
| CAUS | base verb strong **and** the causative stem (`base+ا`, `base+وا`) has ≥ 2 attested forms of its own → two boundaries (`بنڑ\|ا\|یا`, `کر\|وا\|ندے`) |
| AGR | `X+ا` attested (the masculine citation form) and ≥ 3 of `X+ا / X+ی / X+ے / X+یاں`, with `X+ہ` rare (adjective-like); possessives (`میر-`, `ساڈ-`, `آپڑ-` …) and suppletive perfective stems (`کیت-`, `دت-`, `لت-`) go through this test |
| NOUN_A | `X+ہ` or `X+ا`, `X+ے`, `X+یاں` all attested (`علاقہ / علاقے / علاقیاں`) |
| NOUN_I | `Xی` frequent, masculine `X+ا/ہ/ے` ≤ 5 % of it, bare `X` rare (`کڑی / کڑیاں`) |
| NOUN_C | `X` attested ≥ 10 times and ≥ 20 % as often as `X+اں` (`کتاب / کتاباں`) |
| PRON_GEN | pronoun + joined genitive (`اس\|دا`, `جس\|دے`) |

The auxiliary gate was added after the first review round, when coincidental
look-alikes passed the count tests. Its evidence (share of tokens followed by an
auxiliary; `scripts/_probe_ctx.py`): genuine participles `کردا` 0.36, `ہوندی` 0.67, `جاندی`
0.85, `لکھدے` 0.39, `سکدے` 0.17; look-alikes `پردے` 'curtains' 0.02, `گردی` 0.01,
`سجدے` 'prostrations' 0.00, `مندا` 'bad' 0.05, `چکدے` 0.05.

Two guards were added in round 2 (§4.6):

* **Vowel-final verb stems.** AGR and NOUN_A are not tried on a stem that is itself a
  strong verb ending in a vowel letter (`ا آ و ے ی`). Such a verb makes its participles
  with `-یا / -ئی / -ئے` and its subjunctive with `-وے`, so `X+ا / X+ی / X+ے / X+یاں`
  belong to another lexeme: `روے` is `رہوے` 'may remain' written without `ہ`, not `رو`
  'cry' + `-ے`, and `ہوا` is 'air'. Stems in `ہ` (`رہ`, `کہ`) take `-ی / -ے` like
  consonant stems and are not affected.
* **One-letter verb stems.** If a verb stem shorter than 2 letters plus an inventory suffix
  explains the word, the word is **dropped**. In practice this is `آ` 'come', also written
  `ا`: `آنڑیں = آ|نڑیں`, `آنڑے = آ|نڑے`, `آنڑاں = آ|نڑاں`, `آندا = آ|ندا`. Before this rule
  the minimum stem length silently pushed such words into the only longer-stem parse
  (`آنڑ|یں` 'SBJV.2SG' of a rare stem `آنڑ`), which is wrong for almost all their tokens
  (21 candidates are dropped by this rule).

### 4.3 Choosing and scoring
The strong parse of highest priority (PRON_GEN > CAUS > VERB > AGR > NOUN_A > NOUN_I >
NOUN_C) wins. Another parse that **requires** a boundary the winner neither requires nor
lists as optional is a *conflict*, and makes the word low. For `-یاں` nouns the masculine
reading (`بچ|یاں`, from `بچہ`) and the feminine one (`بچی|اں`, from `بچی`) are weighed by
the frequency of their base forms: the larger is primary and, unless the other is
negligible (≤ 5 %), the word is ambiguous. Two more low-confidence causes: a compound verb
written as one word (`ہوجا|سی` = `ہو+جا+سی`: the stem itself contains a boundary) and a
weak paradigm.

**Alternatives.** An alternative is scored as fully correct (§7), so a conflict is *not*
turned into an alternative automatically. The first version did that, and the external
review counted about 14 of its 51 alternatives as wrong splits (`بیمار|یاں` for
`بیماری|اں`, `شہر|یاں` for `شہری|اں`, `اسدی|اں` for `اس|دیاں`). Now a conflicting parse
becomes an alternative only if (a) its own lexeme is attested (for `-یاں`: the masculine
reading needs `X+ا/ہ` and `X+ے`, each ≥ 10 times; the feminine reading needs `Xی` ≥ 10
times; the smaller reading ≥ 10 % of the larger; other families: the competing parse is
strong) **and** (b) the review read the word's KWIC lines and found that reading in at
least about 2 of 12 lines (action `alt_ok`). Conflicts that fail are kept in `notes` only
("not accepted as alternative"). 3 alternatives survive; 15
offered ones were rejected by the review.

### 4.4 Per-word context check (round 2)
The paradigm tests pool evidence over the **stem**, so a word whose stem is a genuine verb
passes them even when the word itself is mostly another lexeme: `جلدی` 'quickly' (not
`جل|دی` 'going'), `پاسو` 'side' (not `پا|سو` 'you will get'), `سونڑے` 'beautiful' (not
`سو|نڑے` 'sleeping'), `ملاں` 'mullah / mills' (not `مل|اں` 'that I may meet'). The external
review found these in the first high set. Every **verb** entry is therefore also tested on
its **own** tokens (`data/context_profiles.json`: the class of the next and the previous
token inside the sentence; lines are split at `۔ ؟ ! ? .`):

| entry | own-context metric | kept high only if |
|---|---|---|
| finite form (SBJV, FUT) or conjunctive participle (`-کے`) | `nom` = share followed by a genitive `دا/دی/دے/دیاں`, a case postposition (`نوں کو نال اچ بچ وچ توں کولوں وسے واسطے تک …`) or `ولا/والا/آلا …`, **plus** share preceded by a quantifier / intensifier (`ہر دونوں ہک اک کئی کسے بوت بہت اتنی کتنی سجے کھبے دوسرے دوجے کجھ تھوڑا زیادہ کافی …`) | `nom` ≤ 0.10 |
| participle (IPFV, PFV, and the PFV causatives) | the same without the genitive (participles take an attributive genitive: `بنڑی دی مسیت`) | `nom` ≤ 0.10 |
| oblique infinitive (`-نڑیں -نڑے -نے`) | `obl` = share followed by a genitive, a case postposition, `ولا …`, `لگ-` 'begin' or `جوگا` 'able' | `obl` ≥ 0.30, or `aux` ≥ 0.30 (Peshawari present: `کر سکنے آں`) |
| other infinitive | `detq` = share preceded by a quantifier / intensifier | `detq` ≤ 0.10 |

On the reported words: `پاسو` nom 0.43 (preceded by `ہر / دونوں / دوسرے / ہک`), `ملاں` nom
0.24 (followed by `دی / دے / نوں`), `جلدی` nom 0.15 (`بوت جلدی`, `جلدی نال`), `سونڑے` obl 0.15;
genuine forms of the same stems score 0.00–0.02 (`جلدا`, `پاساں`, `ملسی`). A word that fails
goes to the low set, and the review reads its KWIC lines (§4.6). 172 analysed
candidates fail the check. The thresholds were set by reading the per-suffix distributions of
the selected words and the KWIC lines of the words above them: they are heuristics. The
check has false positives (`کہساں` 'I will say' after `ایہہ / اتنا`, which simply stays low)
and cannot see a homograph whose other reading also stands in verbal contexts, and it tests
verbs only; that is why every high and low entry was also read in the context sheet (§4.6).

### 4.5 Urdu filter
From the *permissive* superset, lines with ≥ 3 whole-token Hindko/Urdu markers
(`hp.lang.marker_counts`) were split into Hindko-dominant (37,726 lines,
2,766,172 tokens; score ≥ 0.8) and Urdu-dominant (22,735 lines, 895,642
tokens; score ≤ 0.2). `urdu_ratio` = relative frequency in Urdu lines ÷ in Hindko lines.
Ratio > 3 → **Urdu-only, dropped** (`ہونا`, `جانا`, `اپنا`, `کریں`); 1 < ratio ≤ 3 →
shared form, low (`کرنا`, `ہوئے`, `لکھا`); a word whose suffix does not exist in Urdu
(`-اں`, `-نڑیں`, `-سی`, `-دا` …) but which is nevertheless Urdu-leaning is a look-alike
and is dropped (`جہاں`, `رواں`, `کیسی`, `ہنسی`).

### 4.6 Manual review
All decisions are in `review/review_decisions.tsv` with their reasons: 232 rows
(190 drop, 24 demote, 3 alternative confirmed,
15 alternative rejected). Review decisions change confidence or confirm /
reject an alternative the method offered; they never add a segmentation the method did not
produce (`induce.py` asserts this).

**Round 1** (152 rows; 11 of them changed from low to drop in round 2). The selected entries and the next candidates in line were read as
word lists with their paradigm evidence. Typical drops: homographs and names (`پیو` 'father',
`بولی` 'language', `پانڑی` 'water', `کرسی` 'chair' → low, `میراں`), loans that only look
inflected (`رسوائی`, `سٹی` 'city', `کریسی` 'crisis'), adverbs (`ذرا`, `ہولے`, `ہمیشاں`), and
wrong analyses (`بلاندے` 'calling' is not a causative of `بل` 'burn', `کہدا` 'whose' is
`کہ + دا`). Words were marked as read (`review/_reviewed_words.txt`, `_reviewed_low_words.txt`)
without a per-word context check, and an external review
then found wrong entries among them (`روے`, `پاسو`, `سونڑے`, `جلدی`, `ملاں` in the high set,
the `آ`-stem words `آنڑیں / آنڑاں / آنڑے` in the low set, and wrong alternatives).

**Round 2** (80 rows after the `#round2` marker, 61 of them drops). Every
decision rests on corpus lines read with `scripts/_kwic.py` (8–14 occurrences evenly spaced
over the word's tokens in corpus order):

1. every word that fails the context check (§4.4) inside the quotas or entering them;
2. every entry of the final high and low sets in `review/context_sheet_*.tsv` (the six most
   frequent left and right neighbours of the word), with KWIC for any doubtful line — this
   found, e.g., `حویلیاں` = the town Havelian, `پچھاں` = 'back' (`پچھاں مڑ`), `بنڑاں` = the
   causative stem `بنڑا` with `ں` (`یقینی بنڑاں دتا`), `برے` / `ورے` = 'but', `سونے / سونا`
   = 'gold', `دینی` 'religious';
3. every alternative the method still offered.

Rules: **drop** when the gold reading is not the majority of the KWIC lines (also when two
readings are about equal and the other one has a different boundary, e.g. `تہوکے`,
`ملائی`); **alt_ok** when the alternative's reading occurs in about 2 of 12 lines or more;
**noalt** otherwise. When KWIC showed that the method's *primary* `-یاں` reading does not
occur (`گلیاں` 'lanes' = `گلی|اں`, `تختیاں`, `گیٹیاں`, `پکھیاں`, `ٹولیاں`), the word was
dropped rather than re-segmented by hand. Every drop refills the quota with the next word,
which was read in turn until no new entry needed a decision.

**Check after round 2.** A new random sample of 40 high entries (`random.Random(20260927)`
over the high set once the review had converged) was read in KWIC (5 lines each): in all 40 the gold reading is
the majority reading. One of them, `نینگی` 'is not (F)', rests on a convention: it analyses the
Peshawari negative copula `نینگا / نینگی / نینگے` as stem + agreement. The external review's
own sample had also flagged `ملایا` as partly the place name Malaya: reading all 44 tokens,
8 are 'Malaya' (`ملایا دا ویزا`, `ملایا جل پہنچیا`) and 36 the causative 'joined' (`ہتھ
ملایا`, `نمبر ملایا`). The segmentation is right for the majority, but an 18 % homograph share
is not "unambiguous", so it was moved to the low set (this was the last change; the sample
above was drawn before it, and the entry that replaced it, `بنڑانڑاں` 'to make', was read in
KWIC).

### 4.7 Selection
Per-category quotas, most frequent first (`PARAMS['QUOTA_HIGH']`, `QUOTA_LOW`).

## 5. Conventions (read before interpreting scores)

* **Stem = everything before the first required boundary.** Stems have ≥ 2 letters.
  Words that the one-letter stem `آ` 'come' (also written `ا`) plus an inventory suffix
  explains are dropped, not re-parsed with a longer stem, so `آنڑیں / آنڑاں / آنڑے / آندا`
  are not in the set; `گیا / گئی` (stem `گ-`) are closed-class words.
* **Fusional endings count as one suffix** (`-دیاں`, `-ندا`, `-یاں`, `-نڑاں`, `-سی`); their
  defensible inner splits are *optional* boundaries. Causatives have two required
  boundaries (`کر|وا|یا`).
* Vowel-stem participles are segmented `ہو|ندا`, `جا|ندی`, `دی|ندے` (stem allomorph `دی` of `دے`).
* `-یاں`: primary boundary follows the larger paradigm (`علاق|یاں`, `کڑی|اں`). Adjective
  and possessive F.PL is `X|یاں` with an optional `ی|اں` (`سار|یاں`, `نک|یاں`). The other
  reading is an alternative only where KWIC showed both in the corpus (`دوریاں` 'tours' /
  'distances', `شہزادیاں` 'princes' / 'princesses', `شماریاں` 'issues' / `مردم شماریاں`
  'censuses'); all of these are in the low set.
* Suppletive perfective stems are stems: `کیت|ا`, `دت|ی`, `لت|ے`.
* Postpositions and the future written as separate words are not in the set (a
  one-morph word has no internal boundary to test).
* Words are evaluated **in isolation**; `--prefix-space` encodes `' ' + word` and ignores
  the leading space, which is how most tokenizers see a word inside running text.

## 6. Statistics

| category | high | low | high examples |
|---|---|---|---|
| ADJ.AGR | 40 | 25 | سار\|ے پہل\|ے سار\|ی |
| N.F.PL | 25 | 15 | کڑی\|اں کہانڑی\|اں لکھاری\|اں |
| N.OBL | 30 | 15 | بچ\|ے صوب\|ے بند\|ے |
| N.OBL.PL | 35 | 30 | علاق\|یاں موقع\|یاں ادار\|یاں |
| N.PL | 75 | 3 | لوک\|اں کتاب\|اں گل\|اں |
| PRON.GEN | 7 | 1 | اس\|دی اس\|دے اس\|دا |
| PRON.POSS | 16 | 1 | میر\|ے آپڑ\|ے میر\|ی |
| V.CAUS | 25 | 20 | بنڑ\|ا\|یا بنڑ\|ا\|ئی بنڑ\|ا\|ئے |
| V.CONJ | 10 | 3 | ہو\|کے لی\|کے جا\|کے |
| V.FUT | 45 | 18 | ہو\|سی جا\|سی کر\|ساں |
| V.INF | 45 | 25 | کر\|نے ہو\|نڑیں ہو\|نڑے |
| V.IPFV | 70 | 25 | ہو\|ندا ہو\|ندی ہو\|ندے |
| V.PFV | 55 | 30 | کیت\|ا کیت\|ی ہو\|یا |
| V.SBJV | 40 | 25 | ہو\|وے جا\|وے کر\|و |
| **total** | **518** | **236** | |

* Frequency in the strict corpus: high min 10, median 223, max 10660, sum 276,461 tokens; low min 10, median 92, max 4804, sum 73,057 tokens.
* Stem length in letters: high 2: 224, 3: 171, 4: 79, 5: 30, 6: 9, 7: 3, 8: 1, 9: 1; low 2: 75, 3: 80, 4: 67, 5: 11, 6: 3.
* Words with two required boundaries (causatives): high 25, low 20. Words with optional boundaries: high 244, low 107.
* Per-word dialect label: high — Hazara-enriched: 74, absent/rare in Hazara sources: 56, insufficient: 223, shared: 165; low — Hazara-enriched: 21, absent/rare in Hazara sources: 18, insufficient: 148, shared: 49.
* Why low entries are low (a word can have several reasons): Urdu-leaning shared form 94, weak paradigm evidence 52, competing segmentation 43, not accepted as alternative 40, compound verb stem written jointly 33, context check failed 23, review 16.
* Candidate pool before quotas: 3016 high-eligible, 1562 low-eligible, 727 dropped (of 5305 analysed, 15438 candidates). Most high-eligible words were not selected only because of the quotas; `data/candidates_all.tsv` lists them, but only the selected ones (and every word that entered the selection during the review) were reviewed.

## 7. The MorphScore-style metric (`morph_eval.py`)

For one word let **R** be the required boundaries, **O** the optional ones, **P** the
tokenizer's internal token boundaries, and the stem the span before min(R). Alignment is
done in UTF-8 bytes, so a byte-level split inside a two-byte Arabic letter can never be
correct.

| metric | definition |
|---|---|
| boundary precision / recall / F1 | micro-averaged: TP = \|P ∩ R\|, FP = \|P − R − O\|, FN = \|R − P\| |
| `morphscore` | share of **multi-token** words whose stem\|suffix boundary min(R) ∈ P. Single-token words are excluded, as in MorphScore (Arnett & Bergen, COLING 2025), as far as we understand its definition |
| `morphscore_all` | the same with single-token words counted as failures |
| `stem_intact` | share of words with no boundary strictly inside the stem |
| **`stem_boundary_respected`** | min(R) ∈ P **and** the stem is intact — *the headline number*: the tokenizer separates the inflection without over-splitting the stem |
| `exact_match` | P − O = R |
| `single_token_rate`, `tokens_per_word` | how often the word is one token; fertility on these words |

For words with alternatives the best-scoring alternative is used; alternatives exist only
where the review confirmed both readings in the corpus (§4.3), so this credits a split that
is right for a real share of the word's tokens, not a mechanical competing parse. `weight='freq'`
weights each word by its corpus frequency. Why the headline is not MorphScore alone:
MorphScore rewards over-segmentation (a character tokenizer gets 1.0).

```python
import sys; sys.path.insert(0, r'F:\Hindko\_tokenizer\morphology')
from morph_eval import load_gold, evaluate, report, tokenizers_json_encoder
gold = load_gold(r'F:\Hindko\_tokenizer\morphology\morph_silver_high.tsv')
enc = tokenizers_json_encoder('my_tokenizer.json', prefix_space=True)   # or any f(word) -> tokens / offsets
print(report(evaluate(enc, gold), 'my tokenizer'))
```
```
python morph_eval.py --demo
python morph_eval.py --tokenizer-json A.json B.json --prefix-space [--weight freq] [--json out.json]
python morph_eval.py --spm model.model          python morph_eval.py --hf path_or_name --prefix-space
```
`encode(word)` may return token strings (SentencePiece `▁`, WordPiece `##`, byte-fallback
`<0xNN>` and GPT-2 byte-level strings are handled) or character offsets. Words whose tokens
do not spell the word (e.g. `[UNK]`) are reported as `unaligned` and left out.

### 7.1 Sanity check with trivial tokenizers (`eval_results/trivial_tokenizers.txt`)

| set :: tokenizer | P | R | F1 | MorphScore | MorphScore (all) | stem intact | **respected** | exact | tok/word |
|---|---|---|---|---|---|---|---|---|---|
| high :: char-level | 0.297 | 1.000 | 0.458 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 5.085 |
| high :: whole-word | n/a | 0.000 | 0.000 | n/a | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| high :: random-split p=0.3 (seed 13) | 0.302 | 0.308 | 0.305 | 0.405 | 0.307 | 0.527 | 0.178 | 0.143 | 2.245 |
| high :: naive longest-suffix stripper | 0.907 | 0.866 | 0.886 | 0.859 | 0.859 | 0.907 | 0.859 | 0.859 | 2.000 |
| high :: oracle (gold segmentation) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.048 |
| low :: char-level | 0.300 | 1.000 | 0.462 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 5.106 |
| low :: whole-word | n/a | 0.000 | 0.000 | n/a | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| low :: random-split p=0.3 (seed 13) | 0.273 | 0.293 | 0.282 | 0.374 | 0.297 | 0.466 | 0.136 | 0.102 | 2.322 |
| low :: naive longest-suffix stripper | 0.894 | 0.824 | 0.858 | 0.809 | 0.809 | 0.894 | 0.809 | 0.809 | 2.000 |
| low :: oracle (gold segmentation) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.085 |

Char-level finds every boundary (recall 1, MorphScore 1) but splits every stem (respected 0);
whole-word keeps every stem (stem_intact 1) but finds no boundary (recall 0, respected 0);
random splitting lies in between; the oracle scores 1 everywhere. The *naive suffix stripper*
uses this resource's own suffix list, so it is circular and is **not** a baseline: it only
shows that the metric rewards the intended behaviour (it fails on `N.F.PL`, where it strips
`-یاں` instead of `-اں` after `ی`, and on causatives, where it strips only one suffix).

### 7.2 Illustration on existing tokenizers (not the benchmark)

Tokenizer files read offline from `F:\Hindko\_tokenizer\baselines\files\` (staged by another
workstream), `--prefix-space`, high set, type-weighted:

| set :: tokenizer | P | R | F1 | MorphScore | MorphScore (all) | stem intact | **respected** | exact | tok/word |
|---|---|---|---|---|---|---|---|---|---|
| high :: gpt-2 | 0.164 | 0.978 | 0.281 | 0.985 | 0.985 | 0.000 | 0.000 | 0.000 | 7.805 |
| high :: gpt-4o | 0.413 | 0.543 | 0.469 | 0.586 | 0.568 | 0.510 | 0.402 | 0.349 | 2.591 |
| high :: mbert | 0.376 | 0.578 | 0.455 | 0.595 | 0.575 | 0.413 | 0.295 | 0.257 | 2.737 |
| high :: xlm-r | 0.657 | 0.663 | 0.660 | 0.747 | 0.678 | 0.766 | 0.577 | 0.531 | 2.239 |
| high :: llama-3 | 0.176 | 0.538 | 0.265 | 0.568 | 0.564 | 0.210 | 0.160 | 0.050 | 4.546 |
| high :: gemma-3 | 0.380 | 0.462 | 0.417 | 0.514 | 0.483 | 0.496 | 0.344 | 0.303 | 2.490 |
| high :: qwen-3 | 0.256 | 0.661 | 0.369 | 0.691 | 0.689 | 0.031 | 0.029 | 0.019 | 4.153 |

GPT-2's byte-level vocabulary has few Arabic merges (about 7.8 tokens per gold word, so most
letters are split into their two bytes): MorphScore close to 1, but no stem survives intact. This is only an illustration that the metric separates tokenizers; the full
comparison belongs to the tokenizer benchmark.

## 8. Limitations (read these)

1. **Silver, single non-native reviewer.** Segmentations are consistent with corpus
   paradigms and with each word's own contexts, and were read by an AI reviewer; they have
   not been checked by a Hindko speaker. The first version still had wrong high entries
   (5 found by the external review, 2 of 40 in its random sample); after round 2 a new
   random sample of 40 had none (§4.6), which bounds the residue only loosely (the one-sided
   95 % upper bound for 0 errors in 40 is about 7 %). Expect a residue of wrong analyses,
   especially homographs whose readings share contexts, and poetry-specific uses.
2. **Peshawari-dominant evidence.** The Peshawar group (newspaper + two Peshawari sites) and the Peshawar-published books supply
   98.0 % of the tokens; Hazara sources only 1.8 %. Hazara-only paradigms are
   under-represented, and the per-word dialect label is *insufficient* for 223 of the
   518 high entries.
3. **The standard Hazara grammar (Bashir & Conners 2019) was not consulted** (§2).
4. **Orthography varies**: `آپڑا/اپڑا/اپنڑا/اپنا`, `ہوئے/ہویے`, `کہنٹے/کیہنٹے`, and the future
   and genitive are often written as separate words. Only the spelling actually found is
   segmented; harakat are stripped, so tokenizers must be fed the stripped form (or map
   offsets themselves).
5. **Omitted by design**: words with a one-letter stem (`آنڑیں`, `آندا`; dropped, §5), derivational morphology, contracted
   copula clitics (`کردین` = `کردے ہن`), joined compound verbs (`ہوجاسی`, only in the low set),
   pronoun paradigms (inventory only).
6. **Conventions matter**: `-یاں` vs `-ی|اں`, fusional endings as one suffix, causatives as two
   boundaries. Another reasonable convention would move some boundaries; optional
   boundaries and alternatives soften, but do not remove, this.
7. **The Urdu filter is itself heuristic**: lines are classified by marker words, and a
   shared form's ratio also reflects genre (Urdu lines are often poetry).
8. **Causatives are judged by transparency**, not etymology; some (`دکھا-` from `دیکھ`) were
   demoted, others may still be lexicalised.
9. **Frequency ≥ 10 and dispersion ≥ 3 units** bias the set to common, newspaper-frequent forms.
10. **The context check (§4.4) is a heuristic** with hand-set thresholds and closed word lists
    for the context classes; it tests verbs only. Some correct forms fail it and sit in the
    low set (23 low entries carry a failed check, e.g. `کہساں`, `مرساں`, `جلسیں`).
11. **Gloss labels are approximate** where the boundary is shared by two readings: `سک|نے`
    is labelled INF.OBL but is mostly the Peshawari present (`کر سکنے آں`), `لکھ|اں` is labelled
    SBJV.1SG but is mostly 'lakhs'. The segmentation is right for both readings; only the
    `gloss` / `category` columns are uncertain there.

## 9. Reproduce

Everything below in one go: `bash scripts/run_all.sh` (Git Bash; about 4 minutes).

```bat
set PYTHONIOENCODING=utf-8
python scripts\build_counts.py        :: word counts per dialect group; Urdu/Hindko line counts (~30 s)
python scripts\context_profiles.py    :: next / previous token classes per word (~85 s)
python scripts\induce.py              :: parses, evidence, context check, review, selection (~35 s)
python scripts\build_inventory.py     :: inventory with corpus evidence
python scripts\context_sheet.py       :: review sheets
python morph_eval.py --demo --json eval_results\trivial_tokenizers.json > eval_results\trivial_tokenizers.txt
python morph_eval.py --demo --weight freq --json eval_results\trivial_tokenizers_freqweighted.json > eval_results\trivial_tokenizers_freqweighted.txt
python tests\test_morph_eval.py
python scripts\render_docs.py         :: this file
```
Illustration (§7.2), with `B=F:\Hindko\_tokenizer\baselines\files`:
```bat
python morph_eval.py --tokenizer-json %B%\gpt-2\tokenizer.json %B%\gpt-4o\tokenizer.json %B%\mbert\tokenizer.json %B%\xlm-r\tokenizer.json %B%\llama-3\tokenizer.json %B%\gemma-3\tokenizer.json %B%\qwen-3\tokenizer.json --prefix-space --json eval_results\existing_tokenizers_illustration.json > eval_results\existing_tokenizers_illustration.txt
```
