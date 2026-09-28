# Hindko canonical text form (normalization spec) — v1.0.1

Roadmap §4 item 4. Written 2026-09-26. Revised the same day to 1.0.1, an idempotency fix (§15). The output on the released corpus is byte-identical to 1.0.0.

- **Implementation:** `F:\Hindko\_pipeline\hp\normalize.py` (`normalize`, `normalize_with_report`, `change_report`, `is_canonical`, `NORMALIZATION_VERSION = '1.0.1'`).
- **Tests:** `test_normalize.py`. It has 188 checks and all pass:
  - 184 unit checks cover every rule and the 1.0.1 regressions;
  - three seeded fuzz sets (60,000 basic, 400,000 focused on ZWJ/ZWNJ and combining marks, 300,000 over every assigned Arabic-block codepoint) all pass the fixed-point check;
  - `--corpus` adds the same check over all 18,283 corpus records.
  - The fixed-point check requires more than `normalize(normalize(x)) == normalize(x)`: normalizing the output again must produce an empty change report.
- **Evidence:** every number below comes from a script in this folder. §13 lists them.
- **Input measured:** `hindko_dataset_permissive.jsonl`, read-only:
  - 18,283 records and 28,057,099 characters;
  - newspaper 6,098 records / 6,566,782 characters;
  - book 11,223 / 20,035,176;
  - web 962 / 1,455,141.
- **Unicode version:** Python 3.11, so Unicode 14.0.0.

---

## 0. Decisions at a glance

| Question | Decision | Deciding number |
|---|---|---|
| NFC | **Keep** (settled) | 0 of 18,283 records were non-NFC |
| ZWNJ U+200C | **Keep between letters** (settled). New: removed at word edges and runs collapsed, because there it is invisible | 0 occurrences in the corpus |
| Kashida U+0640 | **Remove** (settled) | 0 occurrences (the decoder strips it) |
| Arabic presentation forms | **Fold to letters.** Allah ligature → `اللہ` (HEH GOAL, the corpus spelling). Symbols are kept: `ﷺ`, `﴾ ﴿`, `﷽`, `﷼`. | only 4 codepoints occur: `ﷺ` 2,845 (kept), `﴾ ﴿` 20 each (kept), `ﷲ` 15 (folded) |
| Arabic YEH / ALEF MAKSURA / KAF / TEH MARBUTA / HIGH HAMZA YEH | **Fold only inside a word proven to be Urdu/Hindko orthography** | 89 folds in 12 web records. All 89 were checked by eye and all are correct |
| Arabic HEH U+0647 | **Never fold** | 264 occurrences. In Urdu-orthography words it stands for HEH GOAL or for HEH DOACHASHMEE, and the codepoints cannot tell which |
| HEH GOAL vs HEH DOACHASHMEE | **Never merge** | 553 attested word pairs differ only by this letter (§10) |
| Digits: ASCII vs Urdu (Extended Arabic-Indic) | **Keep both as written** | 61,824 ASCII vs 59,397 Urdu digits. With single-digit pre-tokenization, keeping both costs 7 tokens in 495,871 and 10 vocabulary ids |
| Arabic-Indic digits U+0660–0669 | **Fold to Extended Arabic-Indic** | 65 digits in 2 web records |
| Punctuation, quotes, Latin text | **Keep as written** | e.g. 669 ASCII `.` vs 399,646 `۔` at sentence end |
| Harakat, their order, doubled or detached marks | **Keep** (NFC fixes the order) | 8.3% of word tokens carry marks. The marks tell words apart, e.g. `اُس` vs `اِس` |
| Hindko tone letters U+08BE–U+08C2 | **Keep.** Compose a base letter + U+065A into the tone letter, as the decoder does | 5,030 occurrences in the corpus, 0 in decomposed form |
| `ݨ` vs `نڑ` (retroflex nasal) | **Never fold** | 436 vs 63,958 word tokens, 84 word pairs attested both ways |
| Tokenizer (encode time) | **Lossless byte-level BPE, no normalizer, round-trip exact.** It needs a pre-tokenizer that keeps marks inside words, and should split digits one at a time. `normalize()` is exported for users | the GPT-2 regex makes 16.5% more pre-tokens and cuts every haraka out of its word |

**Net effect on the released corpus:** 233 characters change, in 20 records, all of them web:

- Newspaper and book text is already in canonical form. 0 characters change there.
- Web: 233 of 1,455,141 characters (0.016%) change, in 20 of 962 records (2.08%).
- Overall: 0.00083% of characters and 0.11% of records.

The rules matter mainly for future data and for user input at inference time. Real-world Urdu-script text is full of the noise these rules remove.

---

## 1. Two levels

**(a) The canonical DATA form** is `normalize(text)`. It is applied to corpus text before tokenizer training and before any evaluation, so a tokenizer is trained and measured on one form.

It is conservative:

- It folds only variants that are *encoding noise*:
  - presentation forms;
  - invisible and control characters;
  - non-standard spaces;
  - Arabic-keyboard letters inside words that provably use Urdu/Hindko orthography;
  - Arabic-Indic digits;
  - decomposed tone letters.
- It never touches a distinction a reader could see or pronounce.

**(b) The TOKENIZER** does no normalization at encode time. It is a byte-level BPE whose `decode(encode(x)) == x` for every string, as in modern LLM tokenizers.

- Text that has not been normalized still round-trips exactly. It may simply tokenize less efficiently.
- `hp.normalize.normalize` is exported, with `NORMALIZATION_VERSION`, so users can bring their input into the training form.

§6 has the tokenizer contract.

---

## 2. What the corpus contains (census)

Each cell below is occurrences / records containing the character. The script is `census.py` and the full table is `census/census_out.txt`.

**Yeh**

| Character | Newspaper | Book | Web |
|---|---|---|---|
| `ي` U+064A ARABIC YEH | 0 | 2/2 | 168/18 |
| `ى` U+0649 ALEF MAKSURA | 0 | 0 | 5/3 |
| `ی` U+06CC FARSI YEH | 500,071/6,098 | 1,306,023/11,068 | 97,402/962 |
| `ے` U+06D2 YEH BARREE | 258,172/6,092 | 701,070/10,509 | 57,834/952 |
| `ۓ` U+06D3 | 0 | 32/19 | 103/57 |
| `ئ` U+0626 | 57,814/5,931 | 147,483/9,411 | 8,783/867 |

**Kaf**

| Character | Newspaper | Book | Web |
|---|---|---|---|
| `ك` U+0643 ARABIC KAF | 0 | 0 | 43/5 |
| `ک` U+06A9 KEHEH | 211,946/6,092 | 706,182/10,720 | 49,242/956 |

**Heh**

| Character | Newspaper | Book | Web |
|---|---|---|---|
| `ه` U+0647 ARABIC HEH | 0 | 0 | 264/35 (254 of them from tvshia) |
| `ہ` U+06C1 HEH GOAL | 198,399/6,088 | 702,797/10,792 | 53,344/962 |
| `ھ` U+06BE HEH DOACHASHMEE | 41,388/5,268 | 257,522/10,065 | 18,240/873 |
| `ۃ` U+06C3 | 109/60 | 1,614/572 | 22/18 |
| `ة` U+0629 | 0 | 0 | 24/11 |
| `ۂ` U+06C2 | 124/90 | 1,338/869 | 0 |
| `ۀ` U+06C0 | 0 | 0 | 0 |

**Hamza and alef**

| Character | Newspaper | Book | Web |
|---|---|---|---|
| `ء` U+0621 | 3,649/1,771 | 11,582/3,477 | 395/176 |
| U+0654 combining hamza (82 after a space, 83 after a letter or mark with no precomposed form) | 4/3 | 161/129 | 0 |
| `أ` | 36/29 | 250/193 | 14/4 |
| `ؤ` | 1,952/1,157 | 13,360/3,153 | 676/274 |
| `إ` | 0 | 0 | 1/1 |
| `آ` | 33,355/5,203 | 104,544/8,638 | 11,598/832 |
| `ٸ` U+0678 | 0 | 0 | 5/1 |
| `ٱ` U+0671 | 0 | 0 | 0 |

**Digits** (occurrences, then runs)

| Digit set | Newspaper | Book | Web |
|---|---|---|---|
| ASCII `0-9` | 20,550 (8,467 runs) | 30,321 (11,944 runs) | 10,953 (5,432 runs) |
| Extended Arabic-Indic `۰-۹` | 1,017 (379 runs) | 58,121 (40,023 runs) | 259 (185 runs) |
| Arabic-Indic `٠-٩` | 0 | 0 | 65 (25 runs, 2 records) |
| Mixed-script runs | 0 | 1 | 1 |

**Presentation forms (U+FB50–FDFF, U+FE70–FEFF)**

Only 4 codepoints occur:

- `ﷺ` U+FDFA: 2,845 occurrences in 772 records (newspaper 166, book 2,658, web 21);
- `﴾` and `﴿`: 1 + 19 each (newspaper + book);
- `ﷲ` U+FDF2: 15 occurrences in 6 web records (gandharahindko).

**Invisible characters and spaces**

- ZWNJ, ZWJ, ZWSP, LRM, RLM, ALM, the bidi embeddings and isolates, BOM, SOFT HYPHEN, NBSP and every other space separator: **0**.
- The only whitespace in the corpus is U+0020 and LF.
- There are no tabs, no double spaces, no spaces at line ends and no runs of 3 or more newlines.
- The only control character is U+0095: 50 occurrences in 1 web record (hazarewall). It is a cp1252 bullet decoded wrongly, and every occurrence is followed by a space.

**Kashida U+0640:** 0.

**Tone letters**

| Letter | Newspaper | Book |
|---|---|---|
| `ࢾ` U+08BE | 6 | 1,260 |
| `ࢿ` U+08BF | 3 | 1,505 |
| `ࣀ` U+08C0 | 4 | 580 |
| `ࣁ` U+08C1 | 2 | 598 |
| `ࣂ` U+08C2 | 2 | 1,070 |

- Total: 5,030. The web has none.
- The combining U+065A appears on other letters: 15,882 times, in the books' dictionary jazm convention.
- The decomposed form (base letter + U+065A on `پ ت ٹ چ ک`) occurs 0 times.

**Retroflex nasal**

| Spelling | Newspaper | Book | Web |
|---|---|---|---|
| `ݨ` U+0768 | 0 | 0 | 439/73 |
| `نڑ` | 19,624 | 40,440 | 4,238 |
| `نْ` (NOON + SUKUN) | 706 | 13,296 | 15 |

`ڻ` and `ڼ` do not occur.

**Punctuation**

| Character | Newspaper | Book | Web |
|---|---|---|---|
| `۔` | 39,780 | 348,705 | 11,161 |
| `،` | 35,706 | 108,563 | 5,242 |
| `؛` | 65 | 22,545 | 34 |
| `؟` | 812 | 4,323 | 738 |
| `٪` | 0 | 0 | 0 |
| `٫` | 0 | 0 | 3 |
| `٬` | 0 | 0 | 0 |
| `…` | 1,195 | 13,778 | 40 |
| ASCII `.` | 1,003 | 4,091 | 483 |
| ASCII `,` | 71 | 316 | 188 |
| ASCII `?` | 4 | 14 | 3 |
| ASCII `;` | 0 | 6 | 0 |
| ASCII `%` | 0 | 5 | 0 |
| ASCII `:` | 1,316 | 79,654 | 3,576 |

**Quotes**

| Character | Newspaper | Book | Web |
|---|---|---|---|
| `‘` | 6,106 | 21,555 | 126 |
| `’` | 6,002 | 26,002 | 142 |
| `“` | 0 | 0 | 75 |
| `”` | 0 | 0 | 123 |
| `«` | 0 | 0 | 25 |
| `»` | 0 | 0 | 31 |
| `"` | 559 | 1,386 | 129 |
| `'` | 577 | 125 | 1,229 |

**Latin letters:** 8,807 in newspaper text (347 records), 53,304 in books (1,165 records) and 12,911 on the web (155 records). PII placeholders: `<PHONE>` 125, `<EMAIL>` 7, `<ID_NUMBER>` 1.

**Harakat** (newspaper / book / web)

| Mark | Newspaper | Book | Web |
|---|---|---|---|
| fatha | 4,512 | 125,504 | 2,765 |
| damma | 37,671 | 135,432 | 10,622 |
| kasra | 7,908 | 95,003 | 5,119 |
| shadda | 1,016 | 22,745 | 680 |
| sukun | 757 | 68,853 | 70 |
| superscript alef | 2,589 | 8,284 | 324 |
| honorific signs U+0610–0613 | 1,063 | 5,049 | 48 |
| takhallus U+0614 | 134 | 5,067 | 52 |

Mark order is always NFC: vowel + shadda occurs 8,696 times, and shadda + vowel 0 times.

---

## 3. The data-form rules

Each rule reports the input characters it changed (`change_report`). The execution order is given below the table.

| Rule | What it does | Why | Corpus effect |
|---|---|---|---|
| **R01 nfc** | Unicode NFC | Settled. Canonical equivalents such as `ا`+U+0653 vs `آ`, `ہ`+U+0654 vs `ۂ`, or shadda/fatha typed in either order must be one byte sequence, or the tokenizer learns two spellings of the same thing | 0: every record was already NFC |
| **R02 newlines** | CRLF, CR, VT, FF, NEL, LS, PS → LF | Invisible line-break variants | 0 |
| **R03 controls** | C0 controls (except TAB and LF), DEL and C1 controls are removed | Not text. Protects against encoding accidents | **50 characters, 1 web record** (U+0095, hazarewall) |
| **R04 presentation** | U+FB50–FDFF and U+FE70–FEFF → their NFKC letters. See the notes below the table | PDF copy-paste and old fonts produce these. Each is a 3-byte glyph code for a letter the corpus already has | **29 characters, 6 web records (1 strict)**: 15 `ﷲ`, 14 of them preceded by `ا` |
| **R05 kashida** | U+0640 removed | Settled. Justification only. The one real use, citing a letter's joining form (`ـوں`), is not attested because the decoder strips kashida | 0 |
| **R06 invisible** | Removes ZWSP, LRM, RLM, ALM (U+061C), LRE/RLE/PDF/LRO/RLO, LRI/RLI/FSI/PDI, WORD JOINER, U+2061–2064, BOM, SOFT HYPHEN, CGJ, MVS and U+206A–206F | Settled (clean.py already removed most of these). Added: ALM, WJ, SHY, CGJ and the invisible operators, which clean.py missed. All are invisible format characters | 0 |
| **R07 zwj** | ZWJ is removed unless both neighbours are non-Arabic, non-space characters. Neighbours are judged in NFC order | Settled: removed in Arabic text. The exception keeps emoji and Indic ZWJ sequences intact in user input | 0 |
| **R08 zwnj** | ZWNJ is kept between two letters or marks (settled). A run collapses to one. At a word edge, next to a space, punctuation or a digit, it is removed. Neighbours are judged in NFC order | Only between two joinable characters does ZWNJ change what the reader sees. Elsewhere it is invisible junk that would split tokens | 0: there is no ZWNJ in the corpus |
| **R09 spaces** | Every Zs space and TAB → U+0020. Runs collapse to one. Spaces at line ends are stripped. 3 or more LF → 2 LF (the blank line that separates book units is kept) | NBSP, thin spaces etc. are invisible variants. This matches clean.py | 0 |
| **R10 nfc2** | NFC, run once before R07 and again after every R07–R09 pass | R03–R08 removals can put a base letter next to its mark again, and R04 can turn a presentation form into a haraka that NFC must reorder | 0 |
| **R11 arabic_letters** | Inside a *proven* Urdu/Hindko word: `ي`→`ی`, `ى`→`ی`, `ك`→`ک`, `ة`→`ۃ`, `ٸ`→`ئ`. The proof is defined below the table | These are Arabic-keyboard codepoints. Urdu and Hindko never use U+064A, U+0649 or U+0643. `ٸ` (a Kazakh letter) is an Urdu-keyboard substitute for `ئ`. The per-word proof makes the rule safe for Arabic quotations | **89 characters, 12 web records (2 strict)**: `ي` 69, `ك` 12, `ٸ` 5, `ة` 3 |
| **R12 tone_letters** | `پ ت ٹ چ ک` (+ harakat) + U+065A → `ࢾ ࢿ ࣀ ࣁ ࣂ` (U+08BE–U+08C2), the same regex as `hp/inpage.py` | NFC does not compose these (they have no canonical decomposition). A user typing base + small v must get the corpus codepoint. U+065A on any other letter stays (dictionary jazm) | 0: the decoder already composed them |
| **R13 digits** | `٠-٩` U+0660–0669 → `۰-۹` U+06F0–06F9 | Same values and the same Perso-Arabic digit family. The Urdu set is the native one (59,397 vs 65). Digits have no spelling to preserve. See §5 | **65 characters, 2 web records** (tvshia verse numbers) |
| **R14 nfc_final** | NFC guard | Guarantees NFC output. It fired 0 times on the whole corpus and on all 760,000 fuzz strings | 0 |

**Execution order (1.0.1).**

1. R01–R06 run once.
2. R10 (NFC) runs, so that R07 and R08 judge each ZWJ/ZWNJ by the neighbours it will have in the output.
3. The block R07 → R08 → R09 → R10 repeats until R07 and R08 remove nothing.
   - A removal lets NFC reorder or compose the marks on both sides of the gap, which can give another ZWJ/ZWNJ a new neighbour.
   - Every extra pass removes at least one ZWJ/ZWNJ and no rule adds one, so there are at most (#ZWJ + #ZWNJ + 1) passes. Text without ZWJ/ZWNJ, which is the whole corpus, takes exactly one pass.
4. R11–R14 run once.

§15 has the defect this order fixes.

**R04 details.**

- The Allah ligature `ﷲ` becomes `اللہ` (U+0627 U+0644 U+0644 U+06C1). That is the corpus spelling: `hp/inpage.py` decodes the InPage Allah ligature the same way.
- A directly preceding `ا` is absorbed, because 14 of the 15 occurrences are written `اﷲ`: `انعام اﷲ`, `عبداﷲ`, `ثناء اﷲ`. Old Urdu fonts drew U+FDF2 without its alef, and plain NFKC would produce the non-word `االله`. Since 1.0.1, a preceding alef presentation form (U+FE8D, U+FE8E) is absorbed too. It is the same letter, and 1.0.0 produced `االلہ` for it. The corpus has neither form.
- Spacing and tatweel forms of harakat (U+FE70–FE7F, U+FC5E–FC63) become the bare mark. NFKC would insert a space inside the word.
- These are kept as they are:
  - `ﷺ ﷻ ﷼ ﷽`;
  - `﴾ ﴿` and the Unicode 14 honorific ligatures (U+FD40–FD4F, U+FDCF, U+FDFE, U+FDFF);
  - any multi-word ligature.
- `ﷺ` is a symbol in its own right. NFKC would expand each one into the 18-character Arabic-orthography phrase `صلى الله عليه وسلم`.

**R11 proof.** A word is a maximal run of Arabic-script letters, marks and ZWNJ. The rule fires only when both of these hold:

1. **At least one letter proves non-Arabic orthography.** The proving characters are:
   - the letters `پ ٹ چ ڈ ڑ ژ ک گ ں ھ ہ ۂ ۃ ی ے ۓ ݨ`;
   - the tone letters;
   - `ٸ`;
   - the mark U+065A.
2. **Every letter belongs to Arabic + Urdu/Hindko.** A letter from another orthography blocks the rule. This covers Pashto `ښ ګ ټ …`, Sindhi `ڪ …` and Saraiki/Sindhi implosives `ݙ ڄ ٻ ڳ`, because Sindhi and Pashto use ARABIC YEH on purpose.

**R11 is per word.** A proven neighbour is not proof, so an Arabic phrase inside an Urdu sentence keeps its letters.

R11 is not a spelling repair. It swaps codepoints that render identically in the word's own orthography.

**Properties:**

- **Deterministic and idempotent:** `normalize(normalize(x)) == normalize(x)`. In fact no rule fires on canonical text: `change_report(normalize(x)) == {}`.
  - This was checked on all 18,283 records and on 760,000 seeded fuzz strings (60,000 basic, 400,000 focused on ZWJ/ZWNJ and combining marks, 300,000 over all assigned Arabic-block codepoints).
  - 1.0.0 was not idempotent on some ZWJ/ZWNJ inputs. The corpus was never affected. See §15.
  - Why 1.0.1 is idempotent: after step 3 the text is NFC and no rule of the block changes it. R11–R13 only swap an Arabic-script letter for another Arabic-script letter, and R12 also drops U+065A after a tone base. None of these changes the category of a ZWJ/ZWNJ neighbour or creates an NFC composition or reordering.
- **NFC on input and output:** the input is NFC-normalized first, and the output is NFC.
- **Commuting rules:** U+065A counts as evidence in R11 so that R11 and R12 commute (a fuzz case found that `تٚي` needed this).

---

## 4. Deliberately NOT folded

| Variation | Counts | Why it stays |
|---|---|---|
| Arabic HEH `ه` → `ہ` or `ھ` | 264 (web). In words: `علیه` 64, `قصه` 29, `واقعه` 21, `آخونزاده` 21, `ابراهیم` 16, `سوره` 15, `چهلم` 10 (Persian-style tvshia pages), Quranic `اللّٰهِ`, and Urdu-web `چاهی`, `کوتهاریہ`, `اُتاهی` | In Urdu-keyboard-less text `ه` stands for **either** HEH GOAL (`چهلم` = `چہلم`) **or** HEH DOACHASHMEE (`ته` = `تھ`), even word-finally (`ساتھ`, `کچھ`). Choosing between them is spelling repair, and a wrong choice merges different words (§10: `کہا` vs `کھا`). In Arabic quotations `ه` is simply correct |
| `ہ` vs `ھ` | 553 word pairs attested ≥ 5 times each way | Different letters. Sometimes different words, sometimes spelling variants of one word (§10). A tokenizer must see both |
| ASCII vs Urdu digits | 61,824 vs 59,397 | §5 |
| ASCII `. , ? ; %` vs `۔ ، ؟ ؛ ٪` | e.g. 669 ASCII full stops after an Arabic letter vs 399,646 `۔` | ASCII punctuation is also used in decimals, abbreviations, URLs and Latin text, so a fold is context-dependent, not encoding noise. Each form is a single cheap token |
| Quotes `‘‘ ’’` (InPage) / `“ ”` / `« »` / `" '` | 59,933 / 198 / 56 / 4,005 | Typographic choices. Not noise |
| `…` vs `...` | 15,013 `…` | `…` is a deliberate decoder output (README) |
| `ﷺ` vs the combining honorific U+0610 | 2,845 vs 1,656 | Different presentations (an inline symbol vs a mark above a name) |
| Izafat and hamza encodings: `ۂ` / `ہء`, word-final `ئ` / `یء` / `یٔ`, `ۓ` / `ئے` | 1,408 / 69; 318 / 11 / 0; 135 / 68,423 | `یء` is also the real word `شیء`, and `ئے` differs visibly from `ۓ` (it has an extra tooth). Choosing one is an orthographic decision, not codepoint noise |
| Combining hamza on consonants (`مسٔلہ`, `تاثٔر`) or after a space | 83 / 82 | Typists' spellings. There is no precomposed target |
| Doubled identical harakat (`ََ` for tanween, `ْْ`) | 813 (6 / 800 / 7) | `ََ` is a typist's tanween substitute, and folding it to either `َ` or `ً` would guess |
| Marks after a space or at line start (izafat `اصلاح ِ احوال`) | 5,110 (302 / 4,754 / 54) | Attaching them to the previous word would be a repair. The tokenizer learns ` ِ` as a unit |
| U+065A on letters other than the five tone bases | 15,882 (books) | The decoder deliberately keeps it (dictionary jazm) |
| `ݨ` vs `نڑ` vs `نْ` vs `ط` (retroflex nasal) | §10 | Orthographic schools, not encoding noise. `نڑ` can also be a real n + ṛ sequence |
| U+FFFD | 49 (book) | Project principle 3: corruption is surfaced, never silently repaired |
| `<PHONE>`, `<EMAIL>`, `<ID_NUMBER>` | 133 | PII placeholders, kept byte-exact |
| Latin letters, case | 75,022 letters | The tokenizer is lossless. Case carries information (acronyms) |
| `ۀ` U+06C0 → `ۂ`, `ە` U+06D5 → `ہ` | 0 occurrences | No evidence, so no rule |

---

## 5. The digit decision: keep ASCII and Urdu digits apart; fold only Arabic-Indic

**Counts.**

- Both systems are native and both are frequent:
  - the newspaper is 96% ASCII by run (8,467 vs 379);
  - the books are 77% Urdu digits (40,023 vs 11,944 runs), for verse references, `۱۹۸۵ء` years and page numbers;
  - the web is 96% ASCII by run (5,432 of 5,643).
- ASCII digits also live in Latin contexts, such as `COVID-19`, URLs and `<PHONE>` masks. There, Urdu digits would be wrong.
- Folding ASCII to Urdu would damage those contexts. Folding Urdu to ASCII would erase a real, frequent typesetting choice. Both folds lose information.
- Arabic-Indic digits (U+0660–0669, 65 characters, 2 records) are neither native to Urdu/Hindko nor used in Latin context. They are an Arabic-keyboard artifact, and they fold into the Urdu set.

**Tokenization argument (measured, `digits_experiment.py`).** The experiment trained byte-level BPE with a 32,000 vocabulary on the canonical form. The split was 90/10 by `sha1(uid)`: 17,140 training documents and 1,143 held-out documents with 1,787,824 characters. The held-out part has 4,558 digit runs (1,558 ASCII and 2,999 Urdu) and 7,984 digit characters.

| Pre-tokenization of numbers | Digits | Held-out tokens | Tokens on digit runs | Vocab ids containing a digit |
|---|---|---|---|---|
| each digit alone (LLaMA style) | **keep** | 495,871 | 7,984 (1.000 per digit) | 20 |
| each digit alone | fold → ASCII | 495,864 | 7,984 (1.000 per digit) | 10 |
| ≤ 3 digits per pre-token (GPT-4 style) | keep | 493,577 | 5,315 (0.666 per digit) | 400 |
| ≤ 3 digits per pre-token | fold → ASCII | 493,347 | 5,250 (0.658 per digit) | 243 |

**Conclusion.**

- **With single-digit splitting, keeping both digit sets is free.** It costs 7 tokens out of 495,871 (0.0014%) and 10 vocabulary ids, and every digit is exactly one token either way.
- With 3-digit grouping, keeping both costs 157 vocabulary ids and 1.2% more digit tokens.
- The tokenizer should therefore split digits one at a time (§6). The data form keeps the writer's digits.
- A training/inference mismatch is avoided too: users type both digit sets, and a lossless tokenizer sees what they type.

---

## 6. Level (b): the tokenizer contract

1. **Lossless.** Use byte-level BPE with no normalizer, no lowercasing and no whitespace rewriting.
   - `decode(encode(x)) == x` must hold for every string.
   - It held for 1,143 of 1,143 held-out documents in all four trial tokenizers.
2. **The pre-tokenizer must keep combining marks and ZWNJ inside words.**
   - The GPT-2 pattern (` ?\p{L}+`) cuts every haraka out of its word: `کُن` → `ک` + `ُ` + `ن`. BPE can then never learn the vocalised word.
   - On the held-out text, 30,809 of 371,850 Arabic-script words (8.3%) carry a mark.
   - The GPT-2 pattern produces 520,886 pre-tokens against 447,296 for a mark-aware o200k-style pattern (+16.5%), even though the mark-aware pattern splits digits one at a time.
   - The pattern used in the trials (Oniguruma syntax):
     ```
     [^\r\n\p{L}\p{M}\p{N}\x{200C}]?[\p{L}\p{M}\x{200C}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}\x{200C}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
     ```
   - The tone letters are `\p{Lo}` and need no special case.
3. **Split digits one at a time** (§5).
4. **Recommended:** register `<PHONE>`, `<EMAIL>` and `<ID_NUMBER>` as added (non-special) tokens, so each mask is one token.
5. **Ship `normalize()` with the tokenizer.** Record `NORMALIZATION_VERSION` in the tokenizer config, and recommend applying it to inputs. Without it, the lossless tokenizer still works but meets forms it never saw in training:
   - input typed shadda-first (the corpus has 8,696 vowel+shadda sequences and 0 shadda+vowel);
   - PDF presentation forms (3-byte codes for ordinary letters);
   - Arabic-keyboard `ي ك`;
   - bidi and zero-width junk.
6. **If a SentencePiece/Unigram model is trained, use `normalization_rule_name=identity`, not the default `nmt_nfkc`.**
   - On the canonical corpus, NFKC would rewrite all 15,013 `…` (840 records) to `...` and all 2,845 `ﷺ` (772 records) to `صلى الله عليه وسلم`. That adds 78,392 characters and puts Arabic `ى`/`ه` into a corpus that otherwise has none (`nfkc_check.py`).
   - Use `byte_fallback=true`, `split_digits=true` and `remove_extra_whitespaces=false`.
   - `character_coverage=1.0` is affordable: the canonical permissive corpus has 254 distinct codepoints, and 158 of them cover 99.99% of characters (`alphabet.py`).
7. **Evaluation.** Measure fertility and every other metric on the canonical form of the held-out split. Baseline tokenizers get the same text, and their own normalizers apply inside them.

---

## 7. Measured effect on the permissive corpus (`apply_corpus.py`)

The table gives characters changed, then records changed, per rule and source. Rules not listed changed nothing anywhere.

| Rule | Newspaper | Book | Web | Web share of characters | Strict tier |
|---|---|---|---|---|---|
| R03 controls | 0 | 0 | 50 / 1 | 0.0034% | 0 / 0 |
| R04 presentation | 0 | 0 | 29 / 6 | 0.0020% | 1 / 1 |
| R11 arabic_letters | 0 | 0 | 89 / 12 | 0.0061% | 15 / 2 |
| R13 digits | 0 | 0 | 65 / 2 | 0.0045% | 0 / 0 |
| **Any rule** | **0 / 0** | **0 / 0** | **233 / 20** (2.08% of web records) | **0.016%** | **16 / 3** |

- **Overall:** 233 of 28,057,099 characters (0.00083%) and 20 of 18,283 records (0.11%). The total length changes by −19 characters.
- **Idempotency:** all 18,283 records pass. 0 outputs are non-NFC.
- **1.0.1 re-run:** `apply/changes.jsonl` and `apply/examples.txt` are byte-identical to the 1.0.0 run. `apply/apply_stats.json` differs only in `normalization_version`.
- **By site:** hazarewall 88 spans, tvshia 34, noukeqalam 26, gandharahindko 15, omnilingual 10, iattock 5.
- Every changed record, with its spans, is in `apply/changes.jsonl`.

---

## 8. Twenty before/after examples (changed spans, from `apply/changes.jsonl`)

| # | Rule | Source (site, tier) | Before | After |
|---|---|---|---|---|
| 1 | R03 | web (hazarewall, perm.) | `شیخ البانڈی`+U+0095+` باغ` | `شیخ البانڈی باغ` |
| 2 | R04 | web (gandharahindko, **strict**) | `کہیا کہ ﷲ تعالیٰ دا` | `کہیا کہ اللہ تعالیٰ دا` |
| 3 | R04 | web (gandharahindko, perm.) | `لکھاری انعام اﷲ کوہستانی` | `لکھاری انعام اللہ کوہستانی` |
| 4 | R04 | web (gandharahindko, perm.) | `علی عبداﷲ نے حاصل کی` | `علی عبداللہ نے حاصل کی` |
| 5 | R04 | web (gandharahindko, perm.) | `حافظ ثناء اﷲ،شجاعت علی` | `حافظ ثناء اللہ،شجاعت علی` |
| 6 | R11 `ي` | web (omnilingual, perm.) | `بھائی کو ايسے کرن` | `بھائی کو ایسے کرن` |
| 7 | R11 `ي` | web (omnilingual, perm.) | `سُن کے کيتا کیا` | `سُن کے کیتا کیا` |
| 8 | R11 `ي` | web (omnilingual, perm.) | `ماسیاں وی ميں کروڑی` | `ماسیاں وی میں کروڑی` |
| 9 | R11 `ك` | web (omnilingual, perm.) | `اوہ کافی ویك ہو` | `اوہ کافی ویک ہو` |
| 10 | R11 `ة` | web (omnilingual, perm.) | `کسی دا زکاة نکلی` | `کسی دا زکاۃ نکلی` |
| 11 | R11 `ة` | web (tvshia, **strict**) | `تذکرة الحفاظ ج 4` | `تذکرۃ الحفاظ ج 4` |
| 12 | R11 `ٸ` | web (iattock, perm.) | `ایجا نٸیں ملڑاں` | `ایجا نئیں ملڑاں` |
| 13 | R11 `ٸ` | web (iattock, perm.) | `ہک ملاٸی ٹولہ وسدا` | `ہک ملائی ٹولہ وسدا` |
| 14 | R11 `ٸ` | web (iattock, perm.) | `تینوں قاٸم آکھے` | `تینوں قائم آکھے` |
| 15 | R11 `ي ك` | web (hazarewall, perm.) | `ہر سال سيكڑون بچے` | `ہر سال سیکڑون بچے` |
| 16 | R11 `ك` | web (hazarewall, perm.) | `پاكستان كے شہر كراچی كو` | `پاکستان کے شہر کراچی كو` (`كو` has no proof and stays) |
| 17 | R11 `ي` | web (hazarewall, **strict**) | `اتھے پيش کيتا جلسي` | `اتھے پیش کیتا جلسي` (`جلسي` stays) |
| 18 | R11 `ي` | web (tvshia, perm.) | `مولا علي عليہ السلام کي` | `مولا علي علیہ السلام کی` (`علي` stays) |
| 19 | R13 | web (tvshia, perm.) | `ناہیں۔ (٧٤)` | `ناہیں۔ (۷۴)` |
| 20 | R13 | web (tvshia, perm.) | `چوکھے۔(١٠٠)` | `چوکھے۔(۱۰۰)` |

Rows 16–18 show the cost of per-word proof: Arabic-keyboard words made only of letters shared with Arabic stay as written (§11).

---

## 9. Characters never seen in the NFC strict corpus

Before normalization, 50 codepoints occur in the permissive corpus and never in the NFC strict corpus. After normalization, 38 remain: the 10 Arabic-Indic digits are folded, `ٸ` is folded and U+0095 is removed. `alphabet/alphabet.json` lists them all.

| Class | Codepoints (count) | Where | Decision |
|---|---|---|---|
| Vietnamese letters | 20 codepoints, 30 characters (`ạ ấ ầ ế ệ ộ ớ ợ ủ ỹ đ ơ ư à á ù ú …`) | 1 record, aaprihindko home page (`Với kiến trúc phong cách …`, uid 710f3d37c04723f4) | Text, so no fold. **The web stage should drop this spam paragraph** |
| Devanagari `त थ र` | 4 | Omnilingual transcripts (`اُत्तर`, `س्थان`) | Transcription errors. Keep (no normalization can repair them) |
| Saraiki/Pashto letters `ڄ ګ ݗ ݪ` | 10, 2, 1, 6 | Omnilingual transcripts (Southern Hindko) | Real orthography. Keep |
| ZWARAKAY U+0659 | 30 | iattock (a Chhachhi writer's vowel sign, e.g. `پٙنج`) | Authorial. Keep |
| `٫` U+066B | 3 | tvshia, misused as end punctuation (`ویے٫`) | Keep. It is not encoding noise |
| `× % ; ^` | 30, 5, 6, 1 | books (decoder symbols) | Keep |
| `— ″ ← 。 Δ` | 8, 1, 1, 1, 1 | web (`Δ` is the WordPress comment-form label) | Keep. `Δ` and `←` are web furniture |

Normalization never deletes text because it is rare. Rare characters are the byte-level tokenizer's job.

---

## 10. Variation a tokenizer will fragment but normalization must NOT fold

These are measured on the canonical form (`variation.py`, `variation/variation.json`). Every item below makes one word appear under several byte sequences, so a tokenizer spends extra vocabulary or tokens on it. Every item is linguistically or orthographically meaningful and must reach the model as written.

1. **Retroflex nasal ṇ: four spellings.**
   - `نڑ` is the Gandhara Hindko Board convention: 63,958 tokens, 5,118 types.
   - `ݨ` U+0768 is the Punjabi/Saraiki convention, found only in web transcripts: 436 tokens, 167 types.
   - `نْ` (noon + sukun) is the decoder's ṇ marker and also ordinary sukun: 13,957 tokens.
   - `ط` is some book typists' convention, about 5,600 words (README).
   - 84 word pairs are attested both ways: `اپنڑے` 351 / `اپݨے` 43, `پانڑی` 1,257 / `پاݨی` 12, `بنڑ` 2,466 / `بݨ` 11, `ہونڑ` 3,026 / `ہوݨ` 8, `ہُنڑ` 587 / `ہُݨ` 11.
   - It is not foldable. The choice marks the writer's orthographic school and dialect, and `نڑ` is also an ordinary n + ṛ sequence.
2. **Hindko tone letters `ࢾ ࢿ ࣀ ࣁ ࣂ`** (5,024 tokens, 1,299 types).
   - The dictionaries respell voiced aspirates with them. 531 pairs coexist with the aspirate spelling, e.g. `ࢾر` 70 / `بھر` 1,165, `ࢾرنا` 49 / `بھرنا` 188, `ࢾائی` 16 / `بھائی` 794.
   - 798 pairs coexist with the plain voiceless letter, e.g. `ࢾر` 70 / `پر` 14,816.
   - The tone letter encodes the Hindko low tone. Folding it into either neighbour destroys that information.
3. **HEH GOAL vs HEH DOACHASHMEE.** 553 word pairs differ by one `ہ`↔`ھ` swap, with both spellings attested at least 5 times.
   - Some pairs are different words: `کہا` 1,425 'said' / `کھا` 1,036 'eat'.
   - Some are one word spelled two ways: `پُہل` 398 / `پُھل` 579 'flower', `ہندکو` 20,961 / `ھندکو` 309, `انہاں` 5,204 / `انھاں` 103, `ہِک` 3,267 / `ھِک` 382.
   - Only a lexicon could separate the two cases. A codepoint rule would merge words.
4. **Harakat.** 489,134 of 5,859,752 Arabic-script word tokens (8.3%) carry marks, and 26,953 bare letter strings are written with two or more mark patterns.
   - Marks tell words apart: `اس` 51,136 / `اُس` 10,472 ('that') / `اِس` 7,227 ('this').
   - Marks also record Hindko vowels: `اچ` 29,143 / `اُچ` 9,547 / `اِچ` 2,681, and `نوں` 33,807 / `نُوں` 8,534.
   - Stripping them for the tokenizer is the job of a feature function (`hp.lang.strip_marks`), not of the data form.
5. **Izafat and hamza spellings.**
   - `ۂ` 1,408 / `ہء` 69 word-final.
   - `ئ` 318 / `یء` 11 word-final.
   - The izafat kasra is written after a space 681 times.
   - `ۓ` 135 / `ئے` 68,423, with 35 pairs such as `گۓ` 20 / `گئے` 7,250 and `آۓ` 15 / `آئے` 3,205.
6. **Dialect and register spellings of the same word.** These are lexical, not orthographic, and no normalization may touch them:
   - 'not': `نہ` 26,141 / `نہیں` 6,116 / `نئیں` 4,823 / `نیں` 3,420;
   - 'in': `اچ` 29,143 / `وچ` 17,680 / `وِچ` 3,601 / `اِچ` 2,681;
   - 'is': `اے` 61,473 / `ہے` 30,070 / `وے` 25,574;
   - 'Peshawar': `پشور` 9,066 / `پشاور` 3,879;
   - 'we': `اسی` 6,639 / `اساں` 2,891 / `اَسی` 1,570;
   - 'to be': `ہونا` 8,173 / `ہونڑ` 3,026 / `ہون` 233 / `ہوݨ` 8;
   - 'water': `پانی` 1,783 / `پانڑی` 1,257 / `پاݨی` 12.
7. **Typist conventions kept on purpose:**
   - doubled harakat, 813;
   - marks after a space or at line start, 5,110;
   - combining hamza on consonants, 83;
   - ASCII `.` ending Urdu sentences, 669.

---

## 11. Residual noise the data form leaves (by design)

These counts are after normalization (`apply/apply_stats.json`):

- `ي` 101 (65 types; web 99, book 2). Top words: `يوسف` 17, `علي` 4, `اسي` 4, `ايس` 3, `دي` 3, `شاعري` 3, `يا` 3, `مين` 2, `فير` 2, and Arabic `قُصَيّ`, `لُؤَي`.
- `ك` 31 (25 types). Examples: `اكتوبر` 4, `كو` 2, `كا` 2, `ايك`, `كلو`, and Arabic `ذَلِكَ`, `مالك`.
- `ى` 5, `ة` 21 (`فاطمة` 6, Arabic `القراءة`), and `ه` 264 (§4).

These words contain no letter that proves Urdu orthography. Many are Urdu or Hindko (`كو`, `فير`, `مين`) and some are Arabic (`ذَلِكَ`, `حَتَّى`).

A *line-level* rule would fold the Urdu ones: fold every word in a line that holds a proven Urdu word and no Arabic-only letter. It was considered and **not adopted**, because it is an inference, not a proof. It would reach Arabic phrases quoted without `ة ى ه`. The residual is 158 characters in a corpus of 28 million, and the lossless tokenizer encodes it anyway. A future version can adopt the line rule if the tokenizer stage finds it matters.

**Known limitation.** The proof treats `ک` and `ی` as Urdu evidence. In Sindhi, `ک` is kh and ARABIC YEH is standard, so a Sindhi word made of shared letters, such as `کي`, would be folded. Sindhi-specific letters block the rule, and Sindhi is out of domain for this corpus.

---

## 12. Findings for other pipeline stages (not normalization)

1. **Web furniture is in two STRICT records.** tvshia records 62c75dbcc47dd7fb and 0d62f65f648cd967 contain the site's language menu (`العربیه / بلتی / فارسی / پښتو / हिन्दी / ภาษาไทย / جستجو / You are here / Home`). That is where the Thai and Devanagari letters of the strict corpus come from. Fix it in `hp/web.py` furniture removal.
2. **Vietnamese spam** (30 accented letters, plus the ASCII ones) sits in the aaprihindko home page record 710f3d37c04723f4 (permissive).
3. **The WordPress comment-form `Δ`** remains in hindkomaza record 87d3216730c29555 (permissive).
4. **U+0095 bullets in the hazarewall forum** (record b02b491d30e5f6c8) show a cp1252 decoding leftover. The web stage could map them to `•` at the source. R03 only removes them.

---

## 13. Reproduce

Run everything with `PYTHONIOENCODING=utf-8`, from `F:\Hindko\_tokenizer\normalization`. Every run is deterministic.

| Command | What it does | Time |
|---|---|---|
| `python census.py` | Writes `census/census.json` and `census/census_out.txt` | 38 s |
| `python probe.py [--words] REGEX…` | Prints contexts for any codepoint or sequence | — |
| `python test_normalize.py [--corpus]` | Runs 187 checks: 184 unit checks and the 3 fuzz sets (760,000 strings). `--corpus` adds one more check over all records (188). Output: `test_normalize_out.txt` | 82 s without `--corpus` |
| `python regression/regression_idempotency.py` | Runs the 3 fuzz sets against the frozen 1.0.0 module (`regression/normalize_v1_0_0.py`) and against the current one, then compares both versions on every corpus record. Writes `regression/idempotency_regression.json` and `regression/regression_out.txt` | 5.5 min |
| `python apply_corpus.py` | Writes `apply/apply_stats.json`, `apply/changes.jsonl` and `apply/examples.txt` | 34 s |
| `python variation.py` | Writes `variation/variation.json` and `variation/variation_out.txt` | 71 s |
| `python alphabet.py` | Writes `alphabet/alphabet.json` | — |
| `python nfkc_check.py` | Writes `nfkc_check.json` | — |
| `python digits_experiment.py` | Writes `digits/digits_results.json` and 4 trial tokenizers `digits/bpe32k_*.json` | 148 s, 3 threads |

To produce the canonical corpus for training:

```python
import sys; sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp.normalize import normalize, NORMALIZATION_VERSION
```

Then apply `normalize` to each record's `text` field. Record `NORMALIZATION_VERSION` with the tokenizer.

---

## 14. What was not done

- **No native-speaker review.** The R11 folds (89) were checked by eye against Urdu/Hindko orthography, and none looked wrong. The variation analysis in §10 is by codepoint pattern, and its "same word" readings (`پُہل`/`پُھل`, `ہندکو`/`ھندکو`) are orthographic judgments.
- **No full tokenizer was trained.** The four 32k BPEs exist only as evidence for §5 and §6. That is roadmap item 5.
- **The corpus files were not changed.** Normalization is applied downstream. The release files are untouched and no canonical copy of the corpus was written.
- **The `ط`-for-ṇ count (about 5,600 words)** is quoted from the README, not re-measured.
- **The per-rule character counts are "input characters changed"** (a removed or replaced codepoint counts 1). NFC changes would be counted by diff, but none occurred.
- **Unicode version.** Character classes depend on Python's Unicode version (14.0.0 here). NFC itself is stable across versions for assigned characters.
- **1.0.1 did not re-run `census.py`, `variation.py`, `nfkc_check.py` or `digits_experiment.py`.**
  - `census.py` measures the raw corpus and does not call `normalize()`.
  - The other three measure normalized corpus text, which is byte-identical under 1.0.0 and 1.0.1: the sha256 of all 18,283 outputs matches, and 0 records differ in text or in change report.
  - `alphabet.py` was re-run, and only its version stamp changed.

---

## 15. Changelog

### 1.0.1 (2026-09-26): idempotency fix

**The defect.** A reviewer showed that 1.0.0 `normalize()` was not idempotent, although idempotency is a hard requirement.

- R07 decided whether to keep a ZWJ from its neighbours before NFC had run.
- R04 can turn a presentation form into a haraka (U+FE7E → SUKUN), and R10 then reordered that haraka next to a ZWJ that R07 had kept. The next call removed the ZWJ, and NFC could compose again.
- Reviewer's reproducers, 1.0.0 first → second call:
  - `a` ZWJ U+0301 U+FE7E gave `0061 200D 0652 0301` → `00E1 0652`;
  - U+1F468 ZWJ U+0301 U+FE76 gave `1F468 200D 064E 0301` → `1F468 064E 0301`.

**Found while fixing:**

- **R08 has the same flaw with compositions.** `=` ZWSP U+0338 ZWNJ `a` gave `2260 200C 0061`, then `2260 0061`: `=` + U+0338 composes to `≠`, which is a symbol, so the ZWNJ no longer sits between letters or marks.
- **Removals cascade.** In BEH HAMZA-ABOVE (ZWJ U+0334)×k ZWJ `b`, each ZWJ removal lets NFC move HAMZA ABOVE (ccc 230) past U+0334 (ccc 1), next to the following ZWJ. A fixed cap of 3 passes would not reach the fixed point, so 1.0.1 repeats the block until nothing changes. The unit tests include a 50-link chain.

**The fix** is the execution order in §3: NFC before R07, then R07–R10 repeated until R07 and R08 remove nothing. R04 also absorbs an alef presentation form before `ﷲ` (§3, R04 details). No other rule changed.

**Evidence** (`regression/idempotency_regression.json`):

| Fuzz set (seed) | Strings | 1.0.0 not a fixed point | 1.0.1 not a fixed point | Outputs that differ between versions |
|---|---|---|---|---|
| basic (20260926) | 60,000 | 0 | 0 | 0 |
| focused (20260927) | 400,000 | 1,109 (R07+R10: 829, R07: 280) | 0 | 2,245 |
| broad (20260928) | 300,000 | 3 (R07+R10: 2, R07: 1) | 0 | 15 |

- Every input whose output changed between versions contains ZWJ, ZWNJ or `ﷲ`.
- The fuzz sets found only the R07 class. The R08 composition class and the cascades were found by analysis and are pinned by unit tests (`test_r07_r08_fixed_point`).
- All 7 reproducers fail under 1.0.0, on both exact output and fixed point, and pass under 1.0.1. 18 of the 19 unit checks added in 1.0.1 fail under 1.0.0. The one that passes checks that a ZWJ between non-Arabic marks is kept.
- **Corpus:** on 18,283 records, 0 records differ in text and 0 differ in change report. The sha256 of the concatenated outputs is identical. The corpus has no ZWJ, ZWNJ or alef presentation form, so every corpus number in this document stands.
