# Hindko inflectional morphology — SILVER segmentation set for tokenizer evaluation

> **Status: SILVER, not gold.** No native Hindko annotator was available. Every
> segmentation was induced automatically from paradigm evidence in the corpus and
> then read by a reviewer who is **not** a native speaker (an AI model with working
> knowledge of Urdu / Panjabi / Hindko). Use the numbers as a *relative* signal for
> comparing tokenizers, not as an absolute measure of morphological correctness.
> A native Hindko speaker should check `morph_silver_high.tsv` before any
> published claim.
>
> Built {{DATE}} from the **strict** release (`F:\Hindko\hindko_dataset.jsonl`,
> read-only). Deterministic: re-running the scripts reproduces the files byte for
> byte (verified with md5 over two full runs).

## 1. What is here

| file | content |
|---|---|
| `morph_silver_high.tsv` | {{N_HIGH}} word forms whose stem\|suffix boundary is unambiguous by paradigm evidence **and** by the word's own contexts (§4.4, §4.6) |
| `morph_silver_low.tsv` | {{N_LOW}} plausible but uncertain word forms ({{N_LOW_ALT}} with a KWIC-confirmed alternative segmentation) |
| `morph_eval.py` | MorphScore-style evaluator (library + CLI) |
| `tests/test_morph_eval.py` | {{N_TESTS}} checks: the evaluator (trivial tokenizers, markers, byte-level, alternatives) and the gold files (no reported homograph, no one-letter-stem word, every alternative reviewed, every high verb passes the context check) |
| `inventory/hindko_inflection_inventory.tsv` / `.json` / `inventory_table.md` | the inflection inventory with sources, dialect notes, corpus counts and examples ({{N_INV}} rows) |
| `eval_results/` | outputs of the evaluator on trivial tokenizers and (illustration only) on some existing tokenizers |
| `review/review_decisions.tsv` | every manual review decision with its reason: {{N_REVIEW}} decisions on {{N_REVIEW_WORDS}} words ({{N_REVIEW_DROP}} drop, {{N_REVIEW_LOW}} demote to low, {{N_REVIEW_ALTOK}} alternative confirmed, {{N_REVIEW_NOALT}} alternative rejected); the {{N_R2}} rows after the `#round2` marker are the second round, and {{N_R2_UPDATED}} round-1 rows were changed to drop in round 2 (their round-2 reason follows `\|\| round 2:`) (§4.6) |
| `review/context_sheet_high.tsv`, `context_sheet_low.tsv` | every gold entry with its context metric and its six most frequent left and right neighbours: the sheet read in round 2 |
| `data/candidates_all.tsv` | all {{N_ANALYSED}} analysed candidate words with parse, evidence, context metric, offered alternatives and decision (audit trail) |
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

* **Retroflex ṇ is spelled `نڑ`** in every source of this corpus: {{RETRO_NR}}
  occurrences of `ن+ڑ` in the strict corpus, against {{RETRO_0768}} of `ݨ` (U+0768,
  all in the Omnilingual transcripts) and 0 of `ڻ`. The {{RETRO_NT}} `ن+ط` in book
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

{{INVENTORY_BOUND}}

### 3.3 Free forms: postpositions, pronouns, auxiliaries (not segmented)

Counts are strict-corpus tokens as PESH / BOOK / HAZ.

{{INVENTORY_FREE}}

**Dialect groups** (per source, from the release documentation):
PESH = newspaper (Weekly Hindkowan, Peshawar) + gandharahindko + tvshia ({{TOK_PESH}} tokens);
BOOK = Gandhara Hindko Academy books ({{TOK_BOOK}}; Peshawar publisher, authors from Peshawar
*and* Hazara, so not dialect-pure); HAZ = Omnilingual ASR + Common Voice transcripts,
aaprihindko, hindko.org, hindkomaza, hazarewall ({{TOK_HAZ}}); OTHER = Southern Hindko
(botanix), FineWeb-2 pages, blogs ({{TOK_OTHER}}).
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
{{N_CAND}} types. {{N_ANALYSED}} of them have at least one parse.

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
  ({{N_SUBMIN}} candidates are dropped by this rule).

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
("not accepted as alternative"). {{N_REVIEW_ALTOK}} alternatives survive; {{N_REVIEW_NOALT}}
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
goes to the low set, and the review reads its KWIC lines (§4.6). {{N_CTX_FLAGGED}} analysed
candidates fail the check. The thresholds were set by reading the per-suffix distributions of
the selected words and the KWIC lines of the words above them: they are heuristics. The
check has false positives (`کہساں` 'I will say' after `ایہہ / اتنا`, which simply stays low)
and cannot see a homograph whose other reading also stands in verbal contexts, and it tests
verbs only; that is why every high and low entry was also read in the context sheet (§4.6).

### 4.5 Urdu filter
From the *permissive* superset, lines with ≥ 3 whole-token Hindko/Urdu markers
(`hp.lang.marker_counts`) were split into Hindko-dominant ({{HI_LINES}} lines,
{{HI_TOKS}} tokens; score ≥ 0.8) and Urdu-dominant ({{UR_LINES}} lines, {{UR_TOKS}}
tokens; score ≤ 0.2). `urdu_ratio` = relative frequency in Urdu lines ÷ in Hindko lines.
Ratio > 3 → **Urdu-only, dropped** (`ہونا`, `جانا`, `اپنا`, `کریں`); 1 < ratio ≤ 3 →
shared form, low (`کرنا`, `ہوئے`, `لکھا`); a word whose suffix does not exist in Urdu
(`-اں`, `-نڑیں`, `-سی`, `-دا` …) but which is nevertheless Urdu-leaning is a look-alike
and is dropped (`جہاں`, `رواں`, `کیسی`, `ہنسی`).

### 4.6 Manual review
All decisions are in `review/review_decisions.tsv` with their reasons: {{N_REVIEW}} rows
({{N_REVIEW_DROP}} drop, {{N_REVIEW_LOW}} demote, {{N_REVIEW_ALTOK}} alternative confirmed,
{{N_REVIEW_NOALT}} alternative rejected). Review decisions change confidence or confirm /
reject an alternative the method offered; they never add a segmentation the method did not
produce (`induce.py` asserts this).

**Round 1** ({{N_R1}} rows; {{N_R2_UPDATED}} of them changed from low to drop in round 2). The selected entries and the next candidates in line were read as
word lists with their paradigm evidence. Typical drops: homographs and names (`پیو` 'father',
`بولی` 'language', `پانڑی` 'water', `کرسی` 'chair' → low, `میراں`), loans that only look
inflected (`رسوائی`, `سٹی` 'city', `کریسی` 'crisis'), adverbs (`ذرا`, `ہولے`, `ہمیشاں`), and
wrong analyses (`بلاندے` 'calling' is not a causative of `بل` 'burn', `کہدا` 'whose' is
`کہ + دا`). Words were marked as read (`review/_reviewed_words.txt`, `_reviewed_low_words.txt`)
without a per-word context check, and an external review
then found wrong entries among them (`روے`, `پاسو`, `سونڑے`, `جلدی`, `ملاں` in the high set,
the `آ`-stem words `آنڑیں / آنڑاں / آنڑے` in the low set, and wrong alternatives).

**Round 2** ({{N_R2}} rows after the `#round2` marker, {{N_R2_DROP}} of them drops). Every
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

{{STATS}}

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

{{TRIVIAL}}

Char-level finds every boundary (recall 1, MorphScore 1) but splits every stem (respected 0);
whole-word keeps every stem (stem_intact 1) but finds no boundary (recall 0, respected 0);
random splitting lies in between; the oracle scores 1 everywhere. The *naive suffix stripper*
uses this resource's own suffix list, so it is circular and is **not** a baseline: it only
shows that the metric rewards the intended behaviour (it fails on `N.F.PL`, where it strips
`-یاں` instead of `-اں` after `ی`, and on causatives, where it strips only one suffix).

### 7.2 Illustration on existing tokenizers (not the benchmark)

Tokenizer files read offline from `F:\Hindko\_tokenizer\baselines\files\` (staged by another
workstream), `--prefix-space`, high set, type-weighted:

{{EXISTING}}

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
   {{SHARE_PB}} % of the tokens; Hazara sources only {{SHARE_H}} %. Hazara-only paradigms are
   under-represented, and the per-word dialect label is *insufficient* for {{N_INSUFF_HIGH}} of the
   {{N_HIGH}} high entries.
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
    low set ({{N_LOW_CTX}} low entries carry a failed check, e.g. `کہساں`, `مرساں`, `جلسیں`).
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
