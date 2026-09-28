# State of the art in subword tokenization and tokenizer evaluation (2023–2026), applied to Hindko

Written 2026-09-26 for the Hindko tokenizer effort (`F:\Hindko\_tokenizer\research\`).
Companion file: `PLAN.md` (the executable plan). Scripts and raw numbers that back the
"measured here" statements are in this folder (listed in §7).

**How this was researched, and its limits.**
- For most 2024–2026 papers I read the arXiv HTML full text; the key ones are SuperBPE, MinGram (added after review), Faster Superword Tokenization, BoundlessBPE, PickyBPE, Scaffold-BPE, LiteToken, Length-MAX, TokEval, Objective-vs-Search, Explaining Crosslingual Inequities, Teaching Old Tokenizers New Words, the 0.01GB vocabulary-expansion paper, the NVIDIA Hindi initialization study, TokSuite, Tokenizer Choice, Getting the Most out of your Tokenizer, and UrduLM. I also read the READMEs of the code repositories.
- I could only read the abstract or the ACL landing page for papers served as PDF, because PDF text extraction is not available on this machine. These are marked **[abstract]** below.
- Every full-text reading went through a summarising fetch tool. Numbers quoted from papers are "as reported". Re-check any number before you quote it in a publication.
- Statements marked **measured here** come from scripts in this folder, run on this machine on 2026-09-26.

---

## 0. The facts about our corpus that drive the design (measured here)

`corpus_charstats.py`, `corpus_charstats2.py` on `hindko_dataset_permissive.jsonl`:

| quantity | value |
|---|---|
| documents | 18,283 (newspaper 6,098 · book 11,223 · web 962) |
| characters / UTF-8 bytes | 28,057,099 chars / 49,920,184 bytes (1.78 bytes per char) |
| characters by source | book 20.0M (71%) · newspaper 6.6M (23%) · web 1.5M (5%) |
| whitespace words / distinct | 5,754,850 / 381,357 (250,846 = 66% of types are hapaxes) |
| coverage of word tokens by the top-k word types | 100 → 30.6%, 1k → 56.7%, 8k → 80.6%, 16k → 86.0%, 32k → 90.1%, 64k → 93.1% |
| distinct code points | 267; all documents are already NFC |
| combining marks (harakat etc.) | 564,186 chars (2.0%); **8.2% of whitespace words contain a mark**; 24.4 marks per 1000 chars in books vs 8.6 in newspaper |
| digits | ASCII 61,824 · Extended Arabic-Indic (U+06F0–9) 59,397 · Arabic-Indic (U+0660–9) 65; runs: 1-digit 35.8k, 2-digit 15.7k, 3-digit 6.0k, 4-digit 8.9k, ≥5 digits 96 |
| ZWNJ U+200C / ZWJ / ZWSP | **0 / 0 / 0**: the ZWNJ-preservation rule never fires in the current release |
| kashida U+0640 | 0, because it is removed upstream |
| presentation forms | 2,900 chars, 98% of them U+FDFA ﷺ (2,845). NFKC would expand each one to an 18-character phrase. |
| Hindko-specific letters U+08BE–U+08C2 (letters with small v) | 5,030 |
| other format characters | U+0601 ARABIC SIGN SANAH (year sign, Cf) × 1,542 |
| Latin letters | 75,022 (0.27%) |
| most frequent words | تے 124.6k, دے 111.7k, دی 84.6k, کے 70.1k, دا 64.2k, میں, اس, کہ, کی, نے, اے, نوں, نال … (short, space-separated function words) |

Consequences, used throughout:
1. The data is small, about 8.3M tokens at a 16k vocabulary (pilot, §7). Every vocabulary entry must earn enough occurrences to be trained. This is the low-resource regime, not the web-scale regime most 2025 papers study.
2. Combining marks are common and unevenly spread (books ≫ news). A pre-tokenizer that does not treat `\p{M}` as part of a word fragments 8% of words. This is measured in §7.
3. The very frequent postpositions and auxiliaries are written as separate words (دے نال, اچ, اے, سی, نوں). That is exactly the pattern that superword tokenizers compress.
4. Two digit scripts are used about equally, and numbers are short (the largest group is 4-digit years). Digit policy is a small but visible design choice.

---

## 1. Algorithms

### 1.1 The classical three, and what 2024–2026 evidence says about them

- **BPE**: bottom-up merges. Sennrich et al. 2016 (ACL). Byte-level BPE is GPT-2 (Radford et al. 2019); tiktoken is the fast runtime used by GPT-4/4o.
- **Unigram LM**: top-down pruning of a seed vocabulary by likelihood. Kudo 2018 (ACL); SentencePiece is Kudo & Richardson 2018 (EMNLP demo).
- **WordPiece**: likelihood-driven merges. It is used for BERT-style encoders and is not relevant to a generative Hindko LM.
- **SentencePiece `byte_fallback`**: character-level vocabulary, and unseen characters fall back to 256 byte tokens. This is the Llama-2 / Gemma style.

Recent evidence:

- **Objective vs. Search: Decomposing What Makes a Good Tokeniser**. Yavuz, Meister, Pimentel. 2026. EMNLP 2026. https://arxiv.org/abs/2609.19145
  - Finding: they split BPE and Unigram into *objective* (compression vs likelihood) and *search* (bottom-up vs top-down). The search procedure dominates, and bottom-up (BPE-like) tokenizers give lower bits-per-byte in most settings.
  - Numbers (128k vocab, 1B params, English): BPE 0.7790 ± 0.0004 bpb vs UnigramLM 0.7872 ± 0.0001.
  - Caveats: top-down methods "may be helpful for morphologically rich languages" (German, Spanish, Turkish). At 8k vocabulary the ranking follows each method's own objective. BLiMP showed no consistent winner.
  - Method: significance by paired document-level bootstrap; 100M-parameter models averaged over 3 seeds.
- **MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment**. Land. 2026. arXiv (v2, 12 Aug 2026). https://arxiv.org/abs/2606.27019 · code https://github.com/sanderland/script_tok. Full entry in §1.3. Its controlled LM experiment points the *other* way from Yavuz et al.:
  - Abstract, verbatim: "In controlled downstream language-model training, Unigram-family tokenizers, with MinGram among the best, consistently beat BPE in bits-per-byte."
  - Table 4 (read in the arXiv HTML v2; numbers as reported): depth-24 nanochat models, 5.86B tokens of ClimbMix (mostly English), 32,768 learned tokens, **20 seeds per method**, Welch's t-tests. MinGram 0.7123 bpb, Unigram-BPE-Init 0.7123, standard Unigram 0.7127, BPE 0.7138. BPE is 0.22% worse than the best (p < .001); the Unigram variants (MinGram, MinGram-PP, Unigram-BPE-Init, standard Unigram) are within 0.05% of it.
  - Caveat by the author: "Our downstream evaluation is narrow": one small model, one corpus, one vocabulary size, English only.
- *Implication (revised after review): the BPE-vs-Unigram evidence for bpb is **mixed**, not a BPE prior.*
  - Yavuz et al. 2026 favour BPE (128k vocabulary, 1B parameters, English); Land 2026 favours the Unigram family (32k vocabulary, a depth-24 nanochat model whose parameter count the paper does not state, English, 20 seeds). Both gaps are small: 0.2–1% of bpb.
  - Neither study covers an inflected, low-resource, Arabic-script language at 8k–32k. Our pilot shows Unigram compressing 2.6–2.8% better than BPE at equal size (§7).
  - Unigram-family tokenizers therefore enter the LM stage on an equal footing with BPE: standard Unigram *and* MinGram (PLAN §2).
  - Both papers needed 3–20 seeds to resolve gaps of 0.2–1%. Our arbiter is tuned to detect ≥0.5% (PLAN §6). A gap the size of Land's BPE-vs-MinGram one (0.22%) would probably be reported as "not distinguishable" here.
- **Tokenizer Choice For LLM Training: Negligible or Crucial?** Ali et al. 2024. Findings of NAACL 2024. https://arxiv.org/abs/2310.08754
  - Setup: 24 models of 2.6B parameters.
  - SentencePiece BPE beat HF BPE on average.
  - Germanic languages benefited from BPE and Romance languages from Unigram.
  - Low fertility is "necessary but not sufficient". Fertility and parity did not consistently predict downstream scores.
  - *Implication:* the *library* is a variable too, so train both SentencePiece and HF variants. Do not pick by fertility alone.
- **Tokenization Is More Than Compression**. Schmidt, Reddy, Zhang, Alameddine, Uzan, Pinter, Tanner. 2024. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.40/ **[abstract]**
  - PathPiece (minimum-token segmentation) disproves "fewer tokens ⇒ better".
  - Pre-tokenization matters a lot, and initialising the vocabulary from BPE helps.
  - Scale: 64 models of 350M–2.4B parameters.
  - *Implication:* compression is a screening metric, not the objective. The LM is the arbiter (§2.5).
- **Multilingual Tokenization through the Lens of Indian Languages**. Brahma et al. 2025/2026. Findings of ACL 2026. https://arxiv.org/abs/2506.17789 **[abstract]**
  - Across 17 Indic languages: script-specific normalization helps, and Unigram preserves morphological boundaries better than BPE.
- **Greed is All You Need: An Evaluation of Tokenizer Inference Methods**. Uzan, Schmidt, Tanner, Pinter. 2024. ACL 2024 (short). https://aclanthology.org/2024.acl-short.73/ **[abstract]**
  - The inference (segmentation) method matters separately from the vocabulary, and greedy inference is surprisingly good.
  - *Implication:* keep the native encoder of each algorithm. Record the encoder as part of the tokenizer's identity.

### 1.2 Superword tokenizers (tokens that cross whitespace)

- **SuperBPE: Space Travel for Language Models**. Liu, Hayase, Hofmann, Oh, Smith, Choi. 2025. COLM 2025. https://arxiv.org/abs/2503.13423 · code https://github.com/PythonNut/superbpe
  - Two-stage curriculum:
    - Stage 1: ordinary BPE with whitespace pre-tokenization, up to a transition point *t*.
    - Stage 2: BPE continues without whitespace pre-tokenization, up to the full size *T*.
  - Constraints: digits are still split into right-to-left groups of 3; superwords are capped at 4 words.
  - Results (200k vocab, 8B model): up to 33% fewer tokens. t = 180k gave +4.0% average over 30 tasks (+8.2% MMLU) and 27% less inference compute. t = 80k compressed most (6.63 vs 4.45 bytes/token) but was *not* best downstream, so the most compressive tokenizer is not the best one.
  - Loss distribution: SuperBPE makes fewer very-high-loss and very-low-loss predictions.
  - Tokenizer training: 10 GB of data, "a few hours on 100 CPUs".
  - Only English was tested, at 200k vocabulary.
  - Training requires a *fork* of HF `tokenizers` (Rust). Inference works with stock `transformers` (the released models load with `AutoTokenizer`).
- **Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier**. Schmidt, Reddy, Tanner, Pinter. 2025. COLM 2025. https://arxiv.org/abs/2504.00178 · code https://github.com/kensho-technologies/boundlessbpe
  - Single-pass algorithm. It "supermerges" two adjacent pretokens only when each is already a single token, and only for pretokens that match `^(?=.+\p{L})(?:\p{L}\p{M}*|[ _'’])+$`. That regex handles combining marks, which matters for us.
  - Adds PickyBPE-style deletions (τ = 0.9).
  - Results: +9–15% bytes/token at 131k vocab, and at least 3% higher Rényi efficiency (α = 2.5).
  - **No LM was trained.**
  - Cost: 4.7 CPU-days on 1 GB (vs 59 s for HF BPE).
- **Faster Superword Tokenization**. Schmidt, Tanner, Pinter. 2026. COLM 2026. https://arxiv.org/abs/2604.05192 · code https://github.com/kensho-technologies/fastboundlessbpe
  - Supermerge candidates can be aggregated by frequency like ordinary pretokens, which gives a >600× speed-up (about 600 s per GB).
  - A two-phase BoundlessBPE is shown to be near-equivalent to SuperBPE (identical vocabulary and merges in the matched setting).
  - Code: Rust plus a pure-Python reference with identical output (Apache-2.0). It is built with `maturin`, there is no PyPI wheel, and it needs `regex` + `heapdict`.
  - Formats: `.model` files. **Exact superword behaviour cannot be represented in HF/tiktoken formats**; exports are "close but not identical".
- **Explaining and Mitigating Crosslingual Tokenizer Inequities**. Arnett, Chang, Biderman, Bergen. 2025. arXiv. https://arxiv.org/abs/2510.21909
  - About 7,000 monolingual tokenizers over 97 languages, including Urdu.
  - The proportion of whitespace explains a share of cross-language token-count differences. SuperBPE gives lower and less variable token counts at every vocabulary size tested (16k–115k). The "optimal" vocabulary size differs per language.
  - *Implication:* superwords are not an English-only trick. They are the mechanism most likely to help a language whose most frequent words are short, separate function words.

*Implications for Hindko:*
- Superwords are the highest-upside algorithmic change available, and the least proven at small vocabulary sizes and small data.
- Liu et al.'s own ablation shows that a late transition (t = 90% of T) beat the most compressive setting. We should test t/T ∈ {0.8, 0.9} rather than chase compression.
- Superword tokens interact badly with tasks that need word-boundary control (a prompt that ends in a space, per-word tagging). Keep a plain BPE/Unigram winner as the fallback for such uses.

### 1.3 Vocabulary refinement: removing intermediate / "scaffold" / residue tokens

- **BPE Gets Picky**. Chizhov, Arnett, Korotkova, Yamshchikov. 2024. EMNLP 2024. https://arxiv.org/abs/2409.04599 · code https://github.com/pchizhov/picky_bpe
  - During training, it removes a token when IoS = f(pair)/f(token) ≥ τ, meaning the token almost only occurs inside one larger merge.
  - τ ∈ {0.6, 0.7, 0.8, 0.9}; no universal best (τ = 0.7 often good).
  - Compression is unchanged (0.989–1.000 relative), and removed tokens had low embedding norms, i.e. they were under-trained.
  - Gains were largest at **small vocabularies (8k)**. That is our regime.
- **Scaffold-BPE**. Lian et al. 2025. AAAI 2025. https://arxiv.org/abs/2404.17808
  - Parameter-free marking of "scaffold" tokens: a token whose frequency falls below the current top of the pair queue. About 6.07% of a 32k vocabulary.
  - They are demolished into non-scaffold children at encode time.
  - Gains on 468M–6.7B models (p < 0.01).
  - No code link in the paper.
- **LiteToken: Removing Intermediate Merge Residues From BPE Tokenizers**. Sun, Yang, Lin, Zhang. 2026. ICML 2026. https://arxiv.org/abs/2602.04706
  - Post-hoc removal: a low final-to-intermediate frequency ratio plus a low neighbour-entropy filter.
  - Measured residue share: Llama-3 8.1%, Gemma-3 10.4%, o200k 8.0%, Qwen 5.1%.
  - Removal needs a custom split-and-re-merge encoder and costs about 3% more tokens.
- **Teaching Old Tokenizers New Words**. Purason, Chizhov, Yamshchikov, Fishel. 2025/2026. Findings of EACL 2026. https://arxiv.org/abs/2512.03989 · code https://github.com/taidopurason/tokenizer-extension
  - "Leaf-based" pruning removes only leaves of the merge graph, so no token becomes unreachable. HF-compatible.
- **DH-BPE / vocabulary pruning**. Shao. 2026. arXiv. https://arxiv.org/abs/2609.06898 **[abstract]**. Pruning guided by exposure under minimum-token segmentation beats plain and pruned BPE and MinGram on compression at 12k/16k.
- **MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment**. Land. 2026. arXiv (v1 25 Jun 2026, v2 12 Aug 2026). https://arxiv.org/abs/2606.27019 · code https://github.com/sanderland/script_tok (Apache-2.0). Read in the arXiv HTML v2 on 2026-09-26 (it was wrongly marked [abstract] in the first version of this document; an HTML version exists).
  - Method: a BPE seed vocabulary of ⌈1.15·n⌉ tokens, log-probabilities initialised from BPE token frequencies, **2 iterations of hard EM on the minimum-token path** (token count is the primary objective, the Unigram score only a tie-break), then **one flat score-pruning step** to n. No suffix array, no forward-backward pass, no iterative prune loop.
  - Compression: better than both BPE and standard Unigram in six languages (English, German, Finnish, Russian, **Arabic**, Korean). A compression-oriented variant (MinGram-PP) matches the strongest token-count compressors while keeping higher morphological alignment.
  - **Downstream bpb: the Unigram family beat BPE** (numbers in §1.1: MinGram 0.7123 vs BPE 0.7138, 20 seeds, English only, 32k).
  - Rare tokens (Table 4, "rare tokens" column, as reported): MinGram 9, Unigram 24, **BPE 172**. That is the same effect our gate audit measures on Hindko: byte-level BPE leaves hundreds of learned tokens with fewer than 20 training occurrences (§7).
  - Code: pure Python, managed with `uv`, no Rust/C++ build (from the README). The output format (HF `tokenizer.json` or custom) is not documented in the README. Downloading it needs your approval.
  - *Implication:* MinGram is a Unigram-family candidate for the LM stage (PLAN §2, A10). Its inference is "fewest tokens, Unigram score as tie-break". That can probably be expressed as a stock HF `Unigram` model by giving every piece the score −C + log p with C larger than any per-pretoken score range, so that Viterbi minimises token count first. This is my proposal, not the paper's, and it must pass an encoder-equivalence test against the reference implementation before it counts as HF-native.
- **Length-MAX Tokenizer**. Dong & Su. 2025. arXiv. https://arxiv.org/abs/2511.20849
  - Maximises average token length, formulated as graph partitioning; greedy longest-match DFA encoder (Rust).
  - 14–18% fewer tokens than BPE at 10k–50k; 18% fewer steps to a fixed loss on GPT-2 124M–1.3B, with 5 seeds and paired tests.
  - English only; code promised, but no URL found.

*Implication:*
- Removing under-used intermediate tokens is cheap insurance against glitch tokens, and the evidence is strongest at small vocabularies.
- HF/SentencePiece cannot represent deletions natively, except by leaf-pruning. PickyBPE therefore needs a custom encoder, which is fine for a from-scratch Hindko model but not for extending an existing LLM.
- Unigram LM avoids the problem by construction (top-down pruning), and so does Length-MAX.

### 1.4 Context- and morphology-aware vocabularies

- **SaGe**. Yehezkel & Pinter. 2023. EACL 2023. https://aclanthology.org/2023.eacl-main.45/ · code https://github.com/MeLeLBGU/SaGe **[abstract]**. Unigram-like pruning with a skip-gram context objective, giving more cohesive tokens. The best morphological alignment in Uzan et al. 2024.
- **MorphBPE**. Asgari, El Kheir, Sadraei Javaheri. 2025. arXiv. https://arxiv.org/abs/2502.00894 **[abstract]**. BPE that never merges across morpheme boundaries. Needs a morphological segmenter. Lower loss and faster convergence at 300M/1B in 4 languages incl. Arabic; used in Fanar.
- **MorphPiece**. Jabbar. 2023. arXiv. https://arxiv.org/abs/2307.07262 **[abstract]**. Partly morphology-based (English).
- **Unsupervised Morphological Tree Tokenizer (TreeTok)**. Zhu, Hu, Ji, Wu, Tu. 2025. Findings of ACL 2025. https://arxiv.org/abs/2406.15245 **[abstract]**. A self-supervised deep model induces character-level trees, then tree-constrained BPE and Unigram. Requires training a neural model.
- **MorphTok**. Brahma et al. 2025. ICML 2025 TokShop. https://arxiv.org/abs/2504.10335 **[abstract]**. Constrained BPE keeps Devanagari dependent vowel signs attached to their base. That is the Indic analogue of our "keep \p{M} with its letter" rule.
- **Confounding factors / MorphScore** (see §2.4): morphological alignment explains little of LM performance.

*Implication:*
- There is no Hindko morphological analyser or gold segmentation, so supervised morphology-aware tokenizers (MorphBPE, MorphPiece) are **not feasible now**.
- TreeTok and SaGe are feasible in principle, but heavy and CPU-unfriendly.
- The one cheap, clearly relevant lesson is to keep combining marks attached to their base letter (MorphTok's CBPE, Velayuthan & Sarveswaran's graphemes).

### 1.5 Arabic-script, Urdu and Indic tokenizers

- **Egalitarian Language Representation … It All Begins with Tokenizers**. Velayuthan & Sarveswaran. 2025. COLING 2025. https://aclanthology.org/2025.coling-main.400/ **[abstract]**. For Tamil, Sinhala and Hindi, *pre-tokenization matters more than the algorithm*. Grapheme-based BPE (GPE) beats byte-level for complex scripts.
- **Tokenization is Sensitive to Language Variation**. Wegmann, Nguyen, Jurgens. 2025. Findings of ACL 2025. https://arxiv.org/abs/2502.15343 **[abstract]**. The pre-tokenizer has the biggest effect, including on tasks that need sensitivity to dialect and spelling variation. Hindko spelling varies.
- **UrduLM**. Naqvi et al. 2026. arXiv. https://arxiv.org/abs/2601.17664
  - Urdu BPE at 10k/20k/32k, with GPT-4-style regex adapted to U+0600–06FF.
  - Normalises Latin digits to Urdu digits and uses UrduHack character maps.
  - Fertility 1.11 (32k) vs 1.566 for o200k.
  - *Implication:* a monolingual Urdu-script tokenizer beats frontier-LLM tokenizers by 20–30% tokens. Expect the same for Hindko.
- **Qalb**. Hassan, Ahmed, Awais. 2026. arXiv. https://arxiv.org/abs/2601.08141. Urdu CPT of Llama-3.1-8B on 1.97B tokens with LoRA on one A100. The paper does not describe any tokenizer change. **Alif** (https://arxiv.org/abs/2510.09051) is the previous Urdu SOTA **[not read]**.
- **Arabic**:
  - Fanar (https://arxiv.org/abs/2501.13944) uses MorphBPE.
  - Arabic LLMs typically strip diacritics in pre-training but keep them in the vocabulary (search-result summaries; **not read in full**).
  - We **cannot** strip harakat: it would change the text the model is scored on, and project rule 3 forbids silent repair.
- **MUTANT: A Recipe for Multilingual Tokenizer Design**. Rana et al. 2025/2026. arXiv. https://arxiv.org/abs/2511.03237 **[abstract]**. Language-aware pre-tokenization plus multiword awareness. 39.5% better fertility than Llama-4 on 22 Indic languages.
- **TokSuite**. Altıntaş et al. 2026. ICML 2026. https://arxiv.org/abs/2512.20757
  - 14 otherwise-identical 1B models with different tokenizers, plus a robustness benchmark that includes Farsi (Perso-Arabic script, optional diacritics).
  - Non-English degrades more under noise (0.21 vs 0.15 average drop).
  - Byte-level/ByT5 is most robust; *tokenization consistency* (same surface form → same tokens) drives robustness.
  - *Implication:* add a diacritic/digit-script/spacing perturbation test.
- **FLORES+** (https://huggingface.co/datasets/openlanguagedata/flores_plus): checked via the HF API on 2026-09-26. The devtest split has 512 files, `urd_Arab` and `hin_Deva` are present, and there is **no `hno`/`hnd` file**. No parallel Hindko benchmark exists, so Petrov-style parity for Hindko cannot be computed without commissioning a translation.
- No Hindko tokenizer or Hindko LM turned up in searches, consistent with `prev context.md` §4.2(5).

### 1.6 Vocabulary size vs data size (2024–2026)

- **Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies**. Tao et al. 2024. NeurIPS 2024. https://arxiv.org/abs/2407.13623
  - The optimal vocabulary grows with compute, sub-linearly in non-vocabulary parameters (reported exponent γ ≈ 0.83).
  - **The optimal vocabulary shrinks when data is scarce.** That is our case.
  - Loss is measured in unigram-normalised units and correlates with bits/char at ρ ≈ 0.99.
- **Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling**. Huang et al. 2025. ICML 2025. https://arxiv.org/abs/2501.16975 **[abstract]**. Log-linear gains from scaling the *input* vocabulary (hashed n-gram embeddings) with the output vocabulary decoupled. This is an architecture change, not a tokenizer change; it is noted for later, with no action now.
- **Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training**. Chung & Kim. 2025. NeurIPS 2025. https://arxiv.org/abs/2508.15390 **[abstract]**
  - Bigger vocabularies lower loss almost only on the most frequent words.
  - Once the vocabulary reaches about 24k, "all commonly-used words" already have their own token (English).
- **How Much is Enough? The Diminishing Returns of Tokenization Training Data**. Reddy et al. 2025. ICML 2025 TokShop. https://arxiv.org/abs/2502.20273 **[abstract]**
  - Tokenizer quality saturates only at about 150–200 GB (English/Russian), and pre-tokenization is the limiting factor.
  - *Implication for us:* at 50 MB we are far below saturation, so every document (the permissive tier too) is likely to help the tokenizer. Measure it (data-mix axis in PLAN).
- **Finding the Optimal Vocabulary Size for NMT**. Gowda & May. 2020. Findings of EMNLP 2020. https://aclanthology.org/2020.findings-emnlp.352/ **[abstract]**
  - Frames vocabulary choice as class imbalance.
  - The rule of thumb usually attributed to it ("largest vocabulary such that ~95% of types have ≥100 occurrences") **could not be re-verified here** because the PDF is unreadable. Treat it as a heuristic, not a citation.

*Implication:*
- For a from-scratch Hindko model trained on about 8M tokens, the right vocabulary is almost certainly **far below** the 100k–260k of frontier models. The candidates to test are 8k–32k.
- In our pilot, a 16k BPE already puts 77% of held-out word tokens in a single token (§7).
- For vocabulary *extension* of a big multilingual LLM the question is different: how many Hindko tokens to append (§3.5).

---

## 2. Evaluation

### 2.1 Length-based intrinsic metrics

- **Fertility** (tokens per whitespace word) and **proportion of continued words**. Rust et al. 2021, "How Good is Your Tokenizer?", ACL 2021, https://aclanthology.org/2021.acl-long.243/ (not re-read). Fertility depends on what counts as a word; for Urdu script, spacing is inconsistent.
- **Characters per token / bytes per token** (compression), and **normalized sequence length** (NSL = tokens(candidate) / tokens(reference) on the same text). Bytes per token is the only length metric that is directly comparable across tokenizers *and* can be used to turn per-token loss into bits-per-byte.
- **Parity / tokenization premium**. Petrov, La Malfa, Torr, Bibi. 2023. NeurIPS 2023. https://arxiv.org/abs/2305.15425
  - Parity is the ratio of token counts for the *same content* in two languages (parallel FLORES-200). Premiums reach 15×, and even byte-level models show 4×.
  - For Hindko: no parallel data (§1.5). We can report *within-Hindko* NSL against Urdu/multilingual tokenizers, but must not call it parity.
- **STRR** (single-token retention rate: share of words kept whole). Beyond Fertility, 2025, https://arxiv.org/abs/2510.09947 **[abstract]**.

### 2.2 Distributional metrics: Rényi efficiency and its critics

- **Tokenization and the Noiseless Channel**. Zouhar, Meister, Gastaldi, Du, Sachan, Cotterell. 2023. ACL 2023. https://aclanthology.org/2023.acl-long.284/ **[abstract]**
  - Rényi efficiency H_α(p)/log|V| of the unigram token distribution correlated with BLEU at 0.82, vs −0.30 for sequence length.
  - α = 2.5 is the value used in the authors' `tokenization-scorer` package example.
- **Two Counterexamples to Tokenization and the Noiseless Channel**. Cognetta, Zouhar, Moon, Okazaki. 2024. LREC-COLING 2024. https://arxiv.org/abs/2402.14614 **[abstract]**. Random-drop BPE and duplication BPE *raise* Rényi efficiency while *lowering* BLEU.
- **TokEval: A Tokenizer Evaluation Suite**. Meister. 2026. COLM 2026. https://arxiv.org/abs/2608.18062 · code https://github.com/cimeister/tokenizer-intrinsic-evals (MIT)
  - 46 tokenizers × a 1.27B model.
  - Rényi efficiency (α = 2) had the strongest correlation with held-out bpb (Spearman ρ = −0.80), compression ρ = −0.51, trigram entropy −0.66.
  - Structure metrics (digit-boundary F1, AST alignment) predicted *task* accuracy.
  - The paper's own caveat: intrinsic fits are "suitable for screening but lack precision for ranking close candidates".
  - Also defines UTF-8 integrity, character-split rate and round-trip exactness, which we adopt.
- **Beyond Text Compression: Evaluating Tokenizers Across Scales**. Lotz, Lopes, Peitz, Setiawan, Emili. 2025. ACL 2025. https://aclanthology.org/2025.acl-long.1546/ **[abstract]**
  - 350M models predict significant tokenizer differences at 2.7B.
  - Tokenizer choice barely matters for English but consistently matters multilingually.
  - Proposes Zipf-inspired intrinsic metrics that beat compression as predictors.
- **Measured here** (§7): the GPT-2 regex, which splits off combining marks, compresses 6% *worse* than our mark-aware regex, yet has the *higher* Rényi efficiency (0.4899 vs 0.4847 at 16k). This is a live Cognetta-style counterexample on our own data. Rényi must never decide a comparison by itself.

### 2.3 "Compression is not the objective"

- Schmidt et al. 2024 (§1.1), Ali et al. 2024 (§1.1), and SuperBPE's t = 80k vs 180k result (§1.2) all show that the most compressive tokenizer is not reliably the best.
- **Getting the Most out of your Tokenizer for Pre-training and Domain Adaptation**. Dagan, Synnaeve, Rozière. 2024. ICML 2024. https://arxiv.org/abs/2402.01035
  - Removing the pre-tokenization regex compresses about 30% better on code but collapses HumanEval.
  - The GPT-4 regex gave +5% compression with no loss.

### 2.4 Morphological alignment

- **MorphScore**. Arnett & Bergen 2025 (COLING 2025), extended to 70 languages in "Evaluating Morphological Alignment of Tokenizers in 70 Languages" (Arnett et al. 2025, https://arxiv.org/abs/2507.06378) · code https://github.com/catherinearnett/morphscore
  - Boundary precision/recall against UD-derived segmentations. **Urdu (`urd_arab`) is included; Hindko is not.**
  - The authors' own finding: morphological alignment explains little variance in LM performance and "was not predictive of model performance". See also "Confounding Factors in Relating Model Performance to Morphology" (https://arxiv.org/abs/2511.01380, **[abstract]**).
  - MorphBPE's consistency-F1 and morphological edit distance also require gold segmentations.
- *Implication:* a report-only secondary metric. Options:
  - (a) MorphScore-Urdu as a proxy (requires downloading its data, which needs your approval);
  - (b) a 300–500 word Hindko gold set annotated by a native speaker (the §4.2(6) human spot-check could produce it).
  - It must not drive the decision.

### 2.5 The arbiter: bits-per-byte from small LMs

- bpb = (total negative log-likelihood of held-out text, in bits) / (UTF-8 bytes of that text). It is tokenizer-independent as long as every tokenizer encodes *exactly the same text* (lossless round-trip on the canonical form).
  - Used by SuperBPE, Objective-vs-Search, TokEval and Tao et al. (bits/char).
- **Small models are informative.**
  - Lotz et al. 2025 (350M → 2.7B).
  - Regional Tiny Stories (Patil et al. 2025, https://arxiv.org/abs/2504.07989, **[abstract]**) compared Indic tokenizers with 1–10M-parameter models and found language-specific tokenizers better.
- **Statistics used by the SOTA papers:**
  - Objective-vs-Search: 3 seeds (at 100M, and for the 1B English bpb runs), *paired document-level bootstrap*. The reported ± at 1B / 128k vocabulary ranged from 1e-4 to 2.6e-3 bpb (≈0.01–0.3% relative).
  - Length-MAX: 5 seeds; paired t-test and paired bootstrap with 10,000 resamples.
  - MinGram (Land 2026): 20 seeds per tokenizer, chosen by a power calculation after 5 seeds; Welch's t-tests. The gaps it resolved were 0.03–0.22% of bpb.
  - SuperBPE: matched the *context length in bytes* across tokenizers (4,096 vs 3,000 tokens).

### 2.6 Under-trained ("glitch") tokens

- **Fishing for Magikarp: Automatically Detecting Under-trained Tokens in LLMs**. Land & Bartolo. 2024. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.649/ **[abstract; PDF unreadable here]**. Combines tokenizer analysis (unreachable or partial-UTF-8 tokens), embedding-weight indicators and prompting. Such tokens are found across many open LLMs.
- PickyBPE and LiteToken (§1.3) remove them at the source. Purason et al.'s *self-tokenization test* (encode(token string) ≠ [token] ⇒ unreachable) finds unreachable tokens created by naive vocabulary extension.
- *Implication (revised after review; measured in §7):*
  - At tokenizer level, a hard requirement is only defensible for 0 unreachable *whole-character* tokens (self-tokenization test).
  - Partial-UTF-8 tokens are inherent to byte-level BPE: our 16k pilot has 9, e.g. space + the Arabic lead byte `d8`. The same holds for GPT-2, Llama-3 and Qwen. Report them rather than gate on them.
  - A minimum training frequency per token cannot be met by BPE with standard remedies. At 16k, 404 learned tokens occur fewer than 20 times in training, all of them intermediate tokens absorbed by later merges. `min_frequency` and leaf-pruning leave them unchanged.
  - Only deleting intermediates (PickyBPE; custom encoder) or a Unigram-family vocabulary removes them. So "under-trained tokens" becomes a reported property, and whether removing them helps becomes an LM comparison.
  - At model level: after the LM runs, check the embedding norms of the rarest tokens.

---

## 3. Engineering

### 3.1 Canonical form (must be fixed *before* any tokenizer is trained)

- Settled in `prev context.md`: NFC, ZWNJ preserved, kashida removed.
  - Our data is already 100% NFC. ZWNJ does not occur, but the rule must still hold for future text.
- **Never let the tokenizer normalise.**
  - SentencePiece's default `nmt_nfkc` would rewrite text. For example, NFKC turns U+FDFA into an 18-character phrase and folds presentation forms. Use `normalization_rule_name=identity`.
  - `remove_extra_whitespaces=True` (the default) collapses whitespace, so use `False`. Otherwise encode→decode is lossy and bpb is computed on different text.
- **Presentation forms.** 2,900 characters, almost all ﷺ. Folding them is a corpus decision (a normalisation spec), not a tokenizer decision. Recommendation: keep them, because they are real authorial choices; the tokenizer learns ﷺ as a single token.
- **Digits.** Keep both scripts as written. Normalising script is a corpus-level choice; UrduLM normalises Latin → Urdu digits.
  - Dual-script digits cost only 10 extra tokens.
  - Merging the scripts would make the LM unable to reproduce the source script, and would change the bpb target.

### 3.2 Pre-tokenization for Arabic-script Hindko

- **Combining marks.**
  - The GPT-2 and cl100k/Llama-3 regexes use `\p{L}+`, which excludes `\p{M}`. A diacritised word such as کُن is therefore cut at every mark.
  - o200k (GPT-4o) added `\p{M}` to its letter classes; BoundlessBPE's eligibility regex uses `\p{L}\p{M}*`.
  - **Measured here:** GPT-2 regex vs mark-aware regex = 5.65 vs 6.01 bytes/token at 16k, fertility 1.469 vs 1.376.
- **ZWNJ/ZWJ (U+200C/D)** are format characters (Cf) and must be allowed *inside* a word class, or they become separate tokens that break words. It costs nothing to include them now.
- **Punctuation.** ۔ U+06D4 (full stop), ، U+060C, ؟ U+061F, ؛ U+061B, ٪ U+066A, quotation marks ‘ ’ (60k), ( ) (250k, dictionary sense numbers such as `(۲)`), `:` (85k), ellipsis … (15k).
  - Urdu writers often attach ۔ and ، to the word with no space (اے۔ 13.8k, ہے۔ 11.6k as whitespace "words"). A punctuation class `[^\s\p{L}\p{N}\p{M}]+` separates them.
  - U+0601 ARABIC SIGN SANAH is Cf; it lands in the punctuation class, which is acceptable.
- **Digits.** Options are single digits (Llama-1/2, Gemma, Qwen) or groups of 3 (cl100k: left-to-right `\p{N}{1,3}`; SuperBPE and TokEval's "ideal": right-to-left).
  - **Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs**. Singh & Strouse. 2024. arXiv. https://arxiv.org/abs/2402.14903 **[abstract]**. Right-to-left grouping helps arithmetic, and the gap shrinks with scale.
  - TokEval's digit-boundary F1 correlated with bpb and BLiMP.
  - Our numbers are short (35.8k 1-digit, 8.9k 4-digit years) and digits are only 0.43% of characters, so the choice barely moves bpb.
  - **Default: single digits.** Only 20 digit tokens (two scripts), always well trained, no place-value ambiguity. Right-to-left groups of 3 is an ablation.
- **Leading space.** GPT-style ` ?\p{L}+` attaches the space to the following word, which is standard and lossless. SentencePiece uses ▁ with `add_dummy_prefix`.
- **Do not remove pre-tokenization entirely.**
  - Dagan et al. 2024: identity regex collapses code tasks. The Alqahtani et al. 2026 survey (https://arxiv.org/abs/2601.13260): "removing pre-tokenization consistently degrades performance".
  - Superwords should come from a *curriculum* (SuperBPE) or a *constrained* supermerge (BoundlessBPE), not from dropping the regex. SentencePiece's `split_by_whitespace=false` is the "drop it from the start" variant; include it only as a low-prior probe.

### 3.3 Byte-level BPE vs character-level + byte_fallback

- Byte-level BPE (GPT-2/Llama-3/Qwen) never produces `<unk>`, but can learn tokens that are *partial UTF-8 characters*. Urdu-script letters are 2 bytes. TokEval measures this as the UTF-8 boundary-crossing and character-split rate.
- Character-level vocabulary + `byte_fallback` (SentencePiece; HF `BPE(byte_fallback=True)`) keeps whole characters, and our alphabet is only 267 code points.
- *Implication:* both are cheap; train both.
  - For **extending** an existing LLM we must use *its* scheme: byte-level for Llama-3/Qwen/GPT-style, SentencePiece + byte_fallback for Gemma/Llama-2 style.

### 3.4 Special tokens and chat templates

- Reserve these from day one, so vocabulary IDs never shift:
  - `<|endoftext|>`/EOS, BOS, PAD;
  - chat-role markers (`<|im_start|>`, `<|im_end|>` ChatML-style, or `<|start_header_id|>`, `<|eot_id|>` Llama-3-style);
  - a block of reserved placeholders. Llama-3 reserves 256 special tokens; 64–256 is typical.
- Make them *added/special* tokens that the pre-tokenizer never splits and training never merges.
- Store a Jinja `chat_template` in `tokenizer_config.json` (HF `transformers` convention).
- Pad the total vocabulary to a multiple of 64 (or 128) for matmul efficiency. The padding slots are reserved tokens and must be excluded from the glitch-token check.
- No special token may be learnable from raw text: check that no training document contains the literal marker strings.

### 3.5 Vocabulary extension of an existing LLM (the likely route to a useful Hindko model)

Continued pretraining (CPT) of an Urdu-capable multilingual model is how Hindko models will realistically be built (`prev context.md` §4.1(A)). The tokenizer question then becomes which Hindko tokens to *append*, and how to initialise them.

- **Which tokens.**
  - **Continued BPE training** (Purason et al. 2026, above): resume merge learning on Hindko text *on top of the base tokenizer's merges*. The toolkit supports byte-level BPE and SentencePiece BPE via HF.
    - Up to 9.6% better tokenization efficiency than naive "train a separate tokenizer and append".
    - Naive appending creates *unreachable* tokens; continued training creates none.
  - **AdaptBPE** (Liyanage & Yvon, EACL 2026, https://arxiv.org/abs/2601.21665, code https://github.com/vijini/Adapt-BPE) swaps under-used tokens for target-domain ones within a fixed budget **[abstract]**.
  - **FragMend** (Mehta et al. 2026, https://arxiv.org/abs/2604.16656) chooses tokens by interpretability signals; about 20 points better initialisation for non-Latin scripts **[abstract]**.
- **How many.**
  - Yamaguchi, Villavicencio, Aletras, "How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?", Computational Linguistics 52(1), 2026, https://arxiv.org/abs/2406.11477:
    - Models: Llama-2-7B, Llama-3-8B, Gemma-2-9B.
    - Languages: 10, incl. Arabic, Hindi, Sinhala, Telugu.
    - Tested +50 to +5k tokens with 30k sentences of target text.
    - Recommendation: "|V_new| between 500 and 1K for MT and 100 for summarisation".
    - Mean or Align initialisation was best for generation.
    - Tuning embeddings + top-2 + bottom-2 layers ("2x2 LS") with 512-token sequences was the best training setup.
  - Alqahtani et al. 2026 (survey): "targeted vocabulary expansion (5–10K tokens)" beats aggressive scaling.
  - Our corpus is about 50 MB, i.e. 5,000× more than 0.01 GB, so the upper end (1k–10k) is plausible. It must be measured.
- **Initialisation (2022 → 2026).**
  - Mean of all embeddings; **mean of subtokens / FVT** (Gee et al. 2022).
  - **WECHSEL**: Minixhofer et al., NAACL 2022, https://aclanthology.org/2022.naacl-main.293/. Needs aligned static embeddings; Urdu exists, Hindko does not.
  - **FOCUS**: Dobler & de Melo, EMNLP 2023, https://arxiv.org/abs/2305.14481, code https://github.com/konstantinjdobler/focus. Needs only a fastText model trained on *our* corpus, so it is feasible on CPU.
  - **OFA**: Liu, Lin, Wang, Schütze, Findings of NAACL 2024, https://aclanthology.org/2024.findings-naacl.68/.
  - **Constrained Word2Vec / convex-hull argument**: Mundra et al., CoNLL 2024, https://aclanthology.org/2024.conll-1.8/. Simple mean-in-convex-hull initialisation is competitive.
  - **Zero-Shot Tokenizer Transfer (hypernetworks)**: Minixhofer, Ponti, Vulić, NeurIPS 2024, https://arxiv.org/abs/2405.07883. Hypernetworks exist only for specific base models; transfer to unseen languages is not established.
  - **OMP tokenizer transplantation**: Goddard & Neto 2025, https://arxiv.org/abs/2506.06607, in `mergekit-tokensurgeon`.
  - **Model-Aware Tokenizer Transfer (MATT)**: Haltiuk & Smywiński-Pohl 2025/2026, https://arxiv.org/abs/2510.21954.
  - **Token Distillation**: Dobler et al., ICLR 2026, https://openreview.net/forum?id=n20ml5nGEo. Distils hidden states of the fragmented form into the new embedding; about 2,500 tokens in under 10 GPU-minutes.
  - **Beyond Initialization Loss** (Joshi et al., NVIDIA, 2026, https://arxiv.org/abs/2608.03494): the most directly analogous study. Setup: Hindi, +25,600 tokens on Nemotron-3-Nano-30B-A3B, 20+ strategies.
    - Best: *asymmetric* initialisation, with uniform subword-mean + Hindi-norm calibration for input embeddings and character-length-weighted subword mean for output embeddings.
    - Validation loss at step 50: 2.722 vs 5.527 for mean-of-all. About 6× fewer CPT steps to equal loss.
    - Key warning: **initialisation loss and initialisation bpb are unreliable predictors**; use a 50-step CPT probe instead.
- **Cost of *replacing* a tokenizer outright:** Dagan et al. 2024 found more than 50B tokens of continued training are needed to erase the gap. Appending a few thousand tokens is far cheaper.
- *Feasibility here:*
  - All tokenizer-side steps (choosing tokens by continued BPE, checking reachability, measuring fertility gains) run on this CPU in minutes.
  - Embedding initialisation for a 1–9B model and any CPT or 50-step probe need a GPU; they are **out of scope for this machine**.

---

## 4. Feasibility on this machine (measured here)

**Hardware reality.**
- The CPU is an **Intel i7-10610U laptop part: 4 physical cores, 8 threads**. The task brief's "8 cores" are logical.
- It is shared with other agents.
- Free RAM was 5.8 GB when checked.

**Libraries (probed with `probe_libs.py`).**

| library | options that matter here |
|---|---|
| `tokenizers` 0.22.2 | `BpeTrainer(max_token_length, initial_alphabet, limit_alphabet, min_frequency)`, `UnigramTrainer(max_piece_length, shrinking_factor)`, `BPE(byte_fallback, ignore_merges, dropout)`, pre-tokenizers `Split` (Oniguruma regex with `\p{M}`, lookahead), `ByteLevel(use_regex=False)`, `Digits`, `UnicodeScripts` |
| `sentencepiece` 0.2.1 | fields include `byte_fallback`, `split_digits`, `split_by_whitespace`, `split_by_unicode_script`, `max_sentencepiece_length`, `pretokenization_delimiter`, `normalization_rule_name`, `remove_extra_whitespaces`, `user_defined_symbols`, `required_chars`, `allow_whitespace_only_pieces`, `treat_whitespace_as_suffix` |

**Released implementations vs this machine.**
- No Rust toolchain (`cargo`/`rustc`/`maturin` absent).
- `git` is present.
- `regex` 2026.2.28 is installed; `heapdict` is not (it is a tiny pure-Python package).

| method | released code | runs here as-is? | HF-native encoding? |
|---|---|---|---|
| BPE / Unigram / WordPiece | HF `tokenizers`, `sentencepiece` | yes | yes |
| SuperBPE | PythonNut/superbpe (HF fork, Rust build) | **no** (needs Rust) | **yes**, stock `AutoTokenizer` loads released SuperBPE tokenizers |
| SuperBPE / BoundlessBPE / BPE | kensho fastboundlessbpe, pure-Python reference, "identical results" | yes after `git clone` + `pip install heapdict`; **download requires your approval** | no; own `.model`, HF export approximate |
| BoundlessBPE (original) | kensho boundlessbpe (pure Python) | yes, but 4.7 CPU-days/GB ⇒ ~5–6 h at our 50 MB if linear (extrapolated, not run) | no |
| PickyBPE | pchizhov/picky_bpe (Python prototype) | likely (not tested) | no; deletions need its encoder |
| Scaffold-BPE, LiteToken, Length-MAX | no usable code found | — | no |
| Continued-BPE vocabulary extension | taidopurason/tokenizer-extension | likely (HF-based; not tested) | yes |
| MinGram (and BPE/Unigram/PathPiece baselines) | sanderland/script_tok (pure Python, `uv`-managed, Apache-2.0) | likely after `git clone` + dependency install (`uv` is installed); **download requires your approval**; not tested | output format not documented; an HF `Unigram` with shifted scores is my untested proposal (§1.3) |

**Two routes to SuperBPE on stock `tokenizers` without the fork (my proposal; neither has been tested yet).**
- **(a) Code-point remapping.**
  - Encode each stage-2 chunk (split at punctuation/newlines, *not* at spaces) with the stage-1 tokenizer, and map every stage-1 token id to one Private-Use-Area code point (BMP PUA has 6,400; plane-15 PUA has 65,534).
  - Run the stock Rust `BpeTrainer` on those strings for T − t more merges, then translate the merges back into token strings and append them to the stage-1 merges.
  - Encoding then uses stock HF BPE with the stage-2 pre-tokenizer, with merges applied by rank. Stage-1 merges never cross a space except via the leading-space attachment, so they reproduce stage-1 segmentation inside chunks. The exception is runs of ≥2 spaces, which must be pinned by the canonical form or accepted as a documented deviation.
  - `max_token_length` in PUA characters caps a superword at k stage-1 tokens, which is stricter than SuperBPE's 4-word cap.
  - Runtime: seconds to minutes.
- **(b) Pure-Python stage-2 trainer.**
  - Same data preparation, then incremental pair counting over about 8M stage-1 tokens.
  - Runtime estimate from the measured pure-Python BPE speed is in §7.
- For both routes, a **mandatory verification**: re-encode the dev set with the final HF tokenizer and confirm that (i) round-trip is exact and (ii) no stage-2 token appears where the stage-1 tokenizer would split *inside* a word.

**Pure-Python reimplementation of PickyBPE / Scaffold-BPE.** Both are "BPE plus a per-merge check" on word-type counts. Our corpus has about 400k pretoken types, so a Python incremental trainer is feasible (measured speed in §7). A faithful PickyBPE also needs its event-ordered encoder (merges and deletions in training order), about 100 lines of Python.

**Tiny GPT on this CPU.**
- Denormals: without flush-to-zero (`torch.set_flush_denormal(True)`), denormal floats made training steps **2–11× slower and erratic** (0.8–5.7 s vs a steady 0.44–0.55 s for the same step, `denormal_test.py`). This is a must-have setting.
- Measured throughput with FTZ, fp32, chunked output layer, 2,048-token steps, ctx 256, under load from other agents:

| model | vocab | 3 threads train tok/s | 1 thread train tok/s | 3 threads eval tok/s |
|---|---|---|---|---|
| d=192, L=4 (1.77M non-emb) | 8,192 | 2,286 | 1,439 | 9,085 |
| d=192, L=4 | 16,384 | 1,683 | 1,096 | 4,528 |
| d=192, L=4 | 32,768 | 959 | 583 | 2,994 |
| d=256, L=4 (3.15M non-emb) | 16,384 | 1,045 | 835 | 2,858 |
| d=256, L=4 | 32,768 | 697 | — | 2,532 |
| d=256, L=6, ctx 512 (4.73M) | 16,384 | 860 | — | 2,719 |

- **Correction (after review).** The first version said here that "three single-thread jobs in parallel give about 2× the aggregate throughput of one 3-thread job". That was arithmetic (3 × the solo rate of 1,096 vs 1,683 tok/s), not a measurement, and it is wrong. Measured with `bench_concurrent.py` (three simultaneous 1-thread processes training the same configuration, FTZ on, 10 timed steps, machine shared):

| model | vocab | solo 1-thread tok/s | each of 3 simultaneous 1-thread processes | ratio to solo | aggregate | one 3-thread process |
|---|---|---|---|---|---|---|
| d=128, L=4 | 8,192 | 1,960 | 1,474–1,504 (mean 1,487) | 0.76 | 4,462 | — |
| d=128, L=4 | 16,384 | 1,373 | 931–951 (mean 940) | 0.68 | 2,821 | 1,926 |
| d=128, L=4 | 32,768 | 926 | 570–576 (mean 573) | 0.62 | 1,720 | — |
| d=192, L=4 | 8,192 | 1,701 | 1,070–1,100 (mean 1,087) | 0.64 | 3,262 | — |
| d=192, L=4 | 16,384 | 1,099 | 669–672 (mean 670) | 0.61 | 2,011 | 1,382 |
| d=192, L=4 | 32,768 | 706 | 375–385 (mean 381) | 0.54 | 1,144 | — |

  - Simultaneous single-thread seeds are still the better way to run a 3-seed wave: 1.46× the aggregate of one 3-thread job at 16k (the reviewer measured 1.28× for d=192/16k, with 830–835 tok/s per process). They are not 2× better, and each process runs at only 0.54–0.76× of its solo speed: memory bandwidth is shared, and other agents load the machine.
  - Throughput varies by up to ≈25% between measurements (d=192/16k per process: 670 here vs 832 in the reviewer's run). PLAN.md's budget uses the slower, concurrent numbers from this table.
- The output layer dominates at large vocabularies. Cost per *byte* rises with vocabulary size even though tokens per byte falls.
- Is it useful? Yes, as a proxy.
  - Our entire corpus is about 8M tokens. A compute-optimal from-scratch model for that much data has well under 10M parameters, so a 2–5M-parameter arbiter is not a toy relative to the real use-case.
  - It is *not* evidence about how the tokenizer behaves inside a 7B CPT model. PLAN.md states the claim accordingly.

---

## 5. What this means for Hindko: prioritised summary

| # | Finding | Strength of evidence | Action |
|---|---|---|---|
| 1 | Pre-tokenization is the biggest lever: keep `\p{M}` (and ZWNJ/ZWJ) inside words; split punctuation; never drop the regex | Strong (Velayuthan 2025; Wegmann 2025; Dagan 2024; Schmidt 2024; **measured here: +6% bytes/token**) | Fix the regex first (P1) |
| 2 | Use bpb from small LMs as the arbiter; use intrinsic metrics only to screen | Strong (Schmidt 2024; Ali 2024; Cognetta 2024; TokEval 2026; **measured Rényi inversion**) | LM protocol in PLAN |
| 3 | Vocabulary must be small for 8M tokens of data | Medium (Tao 2024; data-scarce regime) | Test 8k–32k, not 100k+ |
| 4 | Superword tokenization (SuperBPE curriculum, late transition) | Medium-high for English LMs; untested for small-data, non-English LMs | Top algorithmic candidate; implement via PUA trick or pure Python |
| 5 | BPE vs Unigram for bpb: **mixed evidence** (BPE ahead in Yavuz 2026 at 128k/1B; Unigram family, MinGram among the best, ahead in Land 2026 at 32k with 20 seeds); Unigram possibly better for inflected languages at small vocabulary | Medium, conflicting (Yavuz 2026; Land 2026; Brahma 2026; Ali 2024); gaps of 0.2–1% | Test both families on equal footing: standard Unigram and MinGram as LM candidates beside BPE; SentencePiece and HF variants |
| 6 | Remove intermediate tokens (PickyBPE τ≈0.7–0.9) | Medium, strongest at 8k (Chizhov 2024); **measured here: all 404 under-supported tokens of a 16k BPE are intermediates** | Candidate at 16k; needs custom encoder |
| 7 | Glitch/under-trained tokens | Strong (Land & Bartolo 2024; LiteToken 2026) | Hard gate on whole-character reachability; support profile and partial-UTF-8 count *reported* (a frequency floor is not attainable for BPE, §7); embedding-norm checks |
| 8 | Digits: single-digit split; keep both digit scripts | Weak-medium (Singh 2024; TokEval) | Default single digits; R2L-3 ablation |
| 9 | Morphology-aware tokenizers | Weak predictive value (MorphScore 2025); no Hindko resources | Report-only; no gold data yet |
| 10 | CPT route: continued-BPE extension (+~1k–10k tokens), asymmetric subword-mean init, 50-step probes | Medium-strong (Purason 2026; Yamaguchi 2026; Joshi 2026) | Separate "extension track"; GPU needed for the model side |

---

## 6. Things I did not do

- I did not read the full text of the **[abstract]**-marked papers; PDF extraction is unavailable here.
- I did not download any repository, dataset or base-model tokenizer. The ones that would help are listed in PLAN.md §10, and each needs your approval. This includes `sanderland/script_tok` (MinGram).
- I did not run SuperBPE, BoundlessBPE, PickyBPE, MinGram or LiteToken. The feasibility statements are based on repository READMEs, papers and the measured speed of a pure-Python BPE trainer. The HF-native MinGram route (Unigram scores −C + log p) is an untested proposal.
- I trained no LM for learning: the GPT benchmarks used random tokens and measure speed only.
- The pilot tokenizer numbers in §7, including the gate audit, use an md5(uid) pseudo-split that is **not** group-disjoint. They calibrate the plan; they are not results. The real splits now exist (`_tokenizer/splits/`, built by another agent); nothing in this folder has been run on them yet except `split_facts.py`.
- Corrections made after review (2026-09-26): the budget's parallel-scaling claim (now measured, §4), the BPE-vs-Unigram framing and the MinGram entry (§1.1, §1.3), the glitch-token implications (§2.6), and PLAN.md's health gates, newspaper cluster key and budget.

---

## 7. Pilot measurements (this machine, 2026-09-26)

**Setup.**
- Split: md5(uid) % 20 == 0 is held out (950 docs); the other 17,333 are used for training. This is **not** group-disjoint, so the numbers calibrate the plan and must not be reported as results.
- Script: `pilot_tokenizers.py` (+ `pilot_sp_fertility_fix.py`).
- Outputs: `pilot/pilot_results_hf_sp.json`, `pilot/pilot_sp_fertility_fixed.json`, and the tokenizer files in `pilot/`.

| tokenizer | vocab | train time | bytes/token | chars/token | fertility (tok/word) | words = 1 token | Rényi eff. (α=2.5) | distinct ids used on held-out | round-trip failures |
|---|---|---|---|---|---|---|---|---|---|
| HF byte-level BPE, P1 (mark-aware regex) | 16,000 | 14.7 s | 6.009 | 3.379 | 1.376 | 77.1% | 0.4847 | 14,673 | 0/950 |
| HF byte-level BPE, P0 (GPT-2 regex) | 16,000 | 16.0 s | 5.649 | 3.177 | 1.469 | 73.9% | **0.4899** | 14,549 | 0/950 |
| SentencePiece Unigram (identity norm., byte_fallback, split_digits) | 16,000 | 64.8 s | **6.168** | 3.469 | 1.326 | 80.1% | 0.5234 | 14,566 | 0/950 |
| HF byte-level BPE, P1 | 32,000 | 14.6 s | 6.318 | 3.553 | 1.309 | 81.0% | 0.4543 | 23,345 | 0/950 |
| HF byte-level BPE, P0 (GPT-2 regex) | 32,000 | 15.9 s | 5.886 | 3.310 | 1.411 | 77.2% | 0.4618 | 22,651 | 0/950 |
| SentencePiece Unigram | 32,000 | 66.0 s | **6.497** | 3.654 | 1.264 | 83.4% | 0.4936 | 23,054 | 0/950 |

- P1 = ` ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+` (Oniguruma syntax). In Python `regex`, write `‌` instead of `\x{200C}`: the Python module rejects `\x{…}`, which was hit and fixed during this pilot. A differential test between the two engines belongs in the harness.
- Checked with `pre_tokenize_str`: HF keeps ZWNJ and marks inside the word (`می‌خواہم`, ` کُن`), separates `۔`, and splits `2023` into single digits.

**Observations (pilot-grade):**
1. **The mark-aware regex compresses 6.4% (16k) / 7.3% (32k) better** than the GPT-2 regex, with lower fertility. The GPT-2 regex nevertheless has **higher Rényi efficiency**: a Cognetta-style counterexample on our own data.
2. **SentencePiece Unigram compresses 2.6–2.8% better** than HF BPE-P1 at equal vocabulary, and keeps more words whole. This is *not* an algorithm-only comparison: SentencePiece uses its own whitespace (▁) and script-boundary pre-tokenization, which can keep `ہے۔` together. PLAN.md therefore adds HF Unigram with P1, and SentencePiece BPE, to separate the algorithm from the pre-tokenizer.
3. Going from 16k to 32k gains 5.1% (BPE) / 5.3% (Unigram) in bytes/token. At 32k, about 73% of the vocabulary appears in a 5% sample, so a sizeable tail of the 32k vocabulary is rare.
4. Corpus size in tokens: about 8.3M (16k BPE) and 7.9M (32k BPE) for the whole permissive corpus.
5. Tokenizer training is cheap: HF BPE takes 15 s and SentencePiece Unigram about 65 s (3 threads). A 50-configuration intrinsic sweep fits in well under an hour.

**Pure-Python BPE speed** (`pybpe_bench.py` → `pybpe_bench.json`; CPython, 1 thread, character-level, P1 pretokens):
- 246,839 pretoken types.
- Timings: pre-tokenization 10.6 s, index build 2.9 s, **16,000 merges in 19.7 s** (the first 500 merges take 10 s, then it speeds up).
- A faithful pure-Python PickyBPE or Scaffold-BPE (a per-merge check plus occasional token removals) should therefore train in about 1 minute per configuration. This is an estimate.
- A pure-Python SuperBPE stage 2 works on about 1.08M punctuation/newline-delimited chunks holding about 8M stage-1 tokens. It is estimated at 5–20 min and 1–2 GB RAM (not run).

**Corpus facts relevant to the engineering choices** (measured):
- No runs of ≥2 spaces and no tabs. The "multiple-space" caveat of the SuperBPE code-point route therefore does not arise today; keep it as an assertion.
- 783 lines exceed 4,192 bytes. **SentencePiece's default `max_sentence_length` (4192) would silently skip them**, so the pilot set 65,536.
- 50 × U+0095 (a C1 control character). Most likely a mis-decoded cp1252 bullet; report it to the corpus pipeline. The tokenizer must still round-trip it (byte fallback does).

**Tiny-GPT throughput for d=128** (1 thread, `bench_tiny_gpt2_d128.py`): 8k 1,989 tok/s · 16k 1,469 · 32k 925 (0.79M non-embedding parameters). The d=192/256 numbers are in §4 (`bench_tiny_gpt2_t3.json`, `bench_tiny_gpt2_stdout.jsonl`). These are **solo** figures. Three simultaneous processes each run at 0.54–0.76× of them (§4, `bench_concurrent.json`), and PLAN.md's budget uses the concurrent figures (`budget_estimate.py` → `budget_estimate.json`).

**Health-gate audit of the pilot tokenizers** (added after review; `gate_audit.py` → `gates/gate_audit_pilot.json`, `gates/minfreq_test.json`, `gates/a2_audit.json`, `gates/leafprune_test.json`; train = the 17,333 pilot training documents):

| tokenizer | learned | train freq = 0 | < 20 | < 20 that are intermediate merge nodes | partial-UTF-8 tokens | self-tokenization failures (whole-character tokens) | held-out bytes/token |
|---|---|---|---|---|---|---|---|
| HF byte-level BPE-P1 16k | 15,743 | 20 | 404 (2.6%) | 404 | 9 (3 with freq 0) | 0 | 6.009 |
| HF byte-level BPE-P1 32k | 31,743 | 78 | 7,218 (22.7%) | 1,820 | 10 (5 with freq 0) | 0 | 6.318 |
| HF char-level BPE + byte_fallback 16k (A2) | 15,478 | 15 | 419 (2.7%) | 419 | 0 | 0 | 6.007 |
| SP Unigram 16k | 15,741 | 0 | 127 (51 multi-character) | — | 0 | 0 | 6.168 |
| SP Unigram 32k | 31,741 | 30 | 8,437 (26.6%) | — | 0 | **5** (e.g. `▁نے۔`, always split as `▁نے` + `۔`) | 6.497 |
| BPE-P1 leaf-pruned to 16k from 18k / from 20k (both runs give the same tokenizer; it differs from plain 16k in one token) | 15,743 | 20 | 403 | 403 | 9 | 0 | 6.009 |

- The review's numbers for the byte-level BPE reproduce exactly: 9 partial-UTF-8 tokens at 16k (ids 258 `20 d8` × 52, 260 `20 d9` × 165, 282 `d8 a7 d8` × 51, 1148 `e0 a2` × 0, 15887 `bf bd` × 0, …), 404 below 20, 20 at 0, 7,218 below 20 at 32k.
- SentencePiece support is counted on whole training documents, which reproduces the review's 127 at 16k. Counted on the non-empty lines the pilot trained SentencePiece on, it is 79 (1 multi-character) at 16k and 8,385 at 32k.
  - The gap reveals a pilot flaw: SentencePiece was trained on lines but encodes whole documents. Each `\n` becomes a byte-fallback piece (322,852 in train), and the word after it loses its `▁` prefix.
  - PLAN.md §1.1 now requires SentencePiece candidates to encode line by line with a dedicated newline piece, as they are trained.
- `min_frequency` = 20 reproduces the `min_frequency` = 2 merges exactly (one plain retrain also reproduced the saved pilot file exactly, which is evidence of determinism, from one repetition only). At 100 / 400 / 1,000 training stops after 9,500 / 3,685 / 1,903 merges.
- `bf bd` is the tail of U+FFFD: the corpus holds 49 U+FFFD in 20 book documents, a corpus defect reported in PLAN.md §1.1.
- Consequences are drawn in PLAN.md §4.3.

---

## 8. Files in this folder

| file | what it is |
|---|---|
| `SOTA_TOKENIZATION.md` | this document |
| `PLAN.md` | the executable plan |
| `corpus_charstats.py` / `.json`, `corpus_charstats2.py` / `.json` | character, word and digit inventory of the corpus |
| `probe_libs.py` | option probe for `tokenizers` / `sentencepiece` |
| `pilot_tokenizers.py`, `pilot_sp_fertility_fix.py`, `pilot/` | pilot tokenizers and their metrics. `pilot/train_lines.txt` (48 MB) is a scratch copy of training text and can be removed. |
| `pybpe_bench.py` / `.json` | pure-Python BPE speed |
| `bench_tiny_gpt.py` | v1 GPT benchmark (materialised logits; superseded, kept for the record) |
| `bench_tiny_gpt2.py`, `bench_tiny_gpt2_d128.py`, `bench_tiny_gpt2_*.json`, `bench_tiny_gpt2_stdout.jsonl` | GPT throughput with FTZ + chunked loss |
| `mm_speed.py`, `prof_tiny_gpt.py`, `denormal_test.py` | diagnosis of the denormal slowdown |
| `bench_concurrent.py` / `.json`, `bench_concurrent_stdout.jsonl` | throughput of 3 simultaneous 1-thread processes vs solo vs one 3-thread process (added after review) |
| `budget_estimate.py` / `.json`, `pilot/hf_bpe_P1_8192.json` | Stage 3/4 wall-clock from the concurrent throughput; measures 8k bytes/token (added after review) |
| `gate_audit.py`, `gates/` | health-gate audit of the pilot tokenizers, `min_frequency` test, A2 build, leaf-pruning test (added after review) |
| `split_facts.py` / `.json` | newspaper `issue` coverage, split sizes and bootstrap clusters on the split manifest (added after review) |
