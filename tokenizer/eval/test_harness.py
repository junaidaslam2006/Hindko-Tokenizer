# -*- coding: utf-8 -*-
"""Unit tests for eval/harness.py, adapters.py and perturb.py (PLAN Stage 0, step 4).

Hand-computed tiny cases: a CHARACTER tokenizer and a WHOLE-WORD tokenizer (custom Python encoders) on a
3-document corpus whose every metric is worked out by hand below; gate tests G1-G5 and properties R1/R2 on
tiny custom tokenizers with known answers; perturbation alignment; differential tests of the offset-based
fertility against an independent byte-span computation (HF byte-level BPE, SentencePiece + newline wrapper).

Run:  python test_harness.py          (about 1 minute; trains two tiny tokenizers in a temp dir)
"""
import json
import math
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402

H._setup_threads(1)
import adapters as A  # noqa: E402
import perturb as P  # noqa: E402
import numpy as np  # noqa: E402

# C: is nearly full on this machine: keep the test scratch on F:
tempfile.tempdir = os.path.join(HERE, "_tmp_tests")
os.makedirs(tempfile.tempdir, exist_ok=True)

c = chr
# letters (built from codepoints so the file bytes stay unambiguous)
ALEF, BEH, PEH, TEH, KEHEH, NOON, DAMMA = c(0x627), c(0x628), c(0x67E), c(0x62A), c(0x6A9), c(0x646), c(0x64F)
KHAH, WAW, SHEEN, HAH, LAM, YEH_BARREE, FULL_STOP = c(0x62E), c(0x648), c(0x634), c(0x62D), c(0x644), c(0x6D2), c(0x6D4)
U1, U2 = c(0x6F1), c(0x6F2)          # Extended Arabic-Indic 1, 2
ZWNJ = c(0x200C)

AB = ALEF + BEH                        # 'ab'
PT = PEH + TEH                         # 'pt'
KUN = KEHEH + DAMMA + NOON             # kun with damma (3 chars)
KN = KEHEH + NOON
N12 = U1 + U2
KHUSHHAL = KHAH + WAW + SHEEN + HAH + ALEF + LAM      # 6 chars; khush + haal (compound rule: prefix khush, rest 3 letters)
AE = ALEF + YEH_BARREE                 # 'ae' (Hindko 'is')

DOC1 = AB + " " + PT + "\n" + AB                        # 8 chars, 14 bytes, 3 words
DOC2 = KUN + " " + N12                                  # 6 chars, 11 bytes, 2 words
DOC3 = KHUSHHAL + " " + AE + FULL_STOP                  # 10 chars, 19 bytes, 2 words ('ae.' is one whitespace word)
DOCS = [DOC1, DOC2, DOC3]


class CharTok:
    """One token per character. vocab: <unk>=0, specials, then a fixed alphabet."""
    model_type = "char"

    def __init__(self, alphabet, specials=(), atomic_specials=True):
        self.itos = ["<unk>"] + list(specials) + sorted(set(alphabet))
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.special_tokens = {s: self.stoi[s] for s in specials}
        self.unk_ids = {0}
        self.vocab_size = len(self.itos)
        self.char_level = True
        self._re = re.compile("(" + "|".join(map(re.escape, specials)) + ")") if (specials and atomic_specials) else None

    def encode(self, text):
        out = []
        for part in (self._re.split(text) if self._re else [text]):
            if part in self.special_tokens and self._re:
                out.append(self.special_tokens[part])
            else:
                out += [self.stoi.get(ch, 0) for ch in part]
        return out

    def decode(self, ids):
        return "".join(self.itos[i] if i else "�" for i in ids)

    def token_bytes(self, i):
        return None if i == 0 else self.itos[i].encode("utf-8")


class WordTok:
    """Whitespace runs and non-whitespace runs are tokens; unknown runs -> <unk> (decodes to '<unk>')."""
    model_type = "word"

    def __init__(self, vocab):
        self.itos = ["<unk>"] + sorted(set(vocab))
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk_ids = {0}
        self.vocab_size = len(self.itos)
        self.char_level = True

    def encode_offsets(self, text):
        ids, st, en = [], [], []
        for m in re.finditer(r"\S+|\s+", text):
            ids.append(self.stoi.get(m.group(), 0)); st.append(m.start()); en.append(m.end())
        return ids, st, en

    def encode(self, text):
        return self.encode_offsets(text)[0]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def token_bytes(self, i):
        return None if i == 0 else self.itos[i].encode("utf-8")


class SuperTok(WordTok):
    """One token for the whole text (a 'superword' covering several words)."""

    def encode_offsets(self, text):
        return [self.stoi.get(text, 0)], [0], [len(text)]


def _gold_chars():
    import csv
    s = set()
    for fn in ("morph_silver_high.tsv", "morph_silver_low.tsv"):
        with open(os.path.join(H.MORPH_DIR, fn), encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                s |= set(r["word"])
    return s


# the character tokenizer knows every character of the test docs and of the silver morphology words
ALPHABET = set("".join(DOCS)) | {" ", "\n", ZWNJ, "1", "2"} | _gold_chars()
WORD_VOCAB = [AB, PT, KUN, KN, N12, "12", KHUSHHAL, AE + FULL_STOP, AE, FULL_STOP, " ", "\n"]


def write_docs(tmp, texts, name="unit_docs"):
    path = os.path.join(tmp, name + ".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for k, t in enumerate(texts):
            f.write(json.dumps({"uid": "unit%d" % k, "group": "g%d" % k, "cluster": "g%d" % k, "source": "book",
                                "variety": "hindko", "tier": "strict", "text": t}, ensure_ascii=False) + "\n")
    return path


def run_on(adapter, texts, tmp, gates="g1", **kw):
    path = write_docs(tmp, texts)
    return H.run(adapter, data=path, gates=gates, out_dir=os.path.join(tmp, "out_" + H.safe_name(adapter.name)),
                 force=True, **kw)


class TestHandComputed(unittest.TestCase):
    """Every expected value below is computed by hand in the comments."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hk_eval_test_")
        cls.char = A.CustomAdapter(CharTok(ALPHABET), name="char")
        cls.word = A.CustomAdapter(WordTok(WORD_VOCAB), name="word")
        cls.s_char = run_on(cls.char, DOCS, cls.tmp)
        cls.s_word = run_on(cls.word, DOCS, cls.tmp)

    def test_sizes(self):
        o = self.s_char["metrics"]["overall"]
        # bytes 14 + 11 + 19 = 44; chars 8 + 6 + 10 = 24; words 3 + 2 + 2 = 7
        self.assertEqual((o["bytes"], o["chars"], o["words"], o["docs"]), (44, 24, 7, 3))

    def test_char_tokenizer(self):
        o = self.s_char["metrics"]["overall"]
        # 24 tokens (one per char); pieces per word: doc1 2,2,2; doc2 3,2; doc3 6,3 -> 20
        self.assertEqual(o["tokens"], 24)
        self.assertAlmostEqual(o["bytes_per_token"], 44 / 24)
        self.assertAlmostEqual(o["chars_per_token"], 1.0)
        self.assertAlmostEqual(o["fertility"], 20 / 7)
        self.assertAlmostEqual(o["tokens_per_word"], 24 / 7)
        self.assertAlmostEqual(o["continued_word_rate"], 1.0)
        self.assertAlmostEqual(o["strr"], 0.0)
        self.assertEqual((o["g1_pass_docs"], o["g1_fail_docs"], o["unk_tokens"]), (3, 0, 0))
        # lines view: doc1 lines 'ab pt','ab' -> 5 + 2 = 7 tokens, 13 bytes; docs 2,3 unchanged (6 + 10 tokens)
        self.assertAlmostEqual(o["bytes_per_token_lines"], (13 + 11 + 19) / (7 + 6 + 10))

    def test_word_tokenizer(self):
        o = self.s_word["metrics"]["overall"]
        # tokens: doc1 ab,' ',pt,'\n',ab = 5; doc2 kun,' ',12 = 3; doc3 khushhal,' ','ae.' = 3 -> 11
        self.assertEqual(o["tokens"], 11)
        self.assertAlmostEqual(o["bytes_per_token"], 44 / 11)
        self.assertAlmostEqual(o["fertility"], 1.0)
        self.assertAlmostEqual(o["strr"], 1.0)
        self.assertAlmostEqual(o["continued_word_rate"], 0.0)
        self.assertAlmostEqual(o["tokens_per_word"], 11 / 7)
        self.assertEqual(o["g1_pass_docs"], 3)

    def test_renyi_and_utilisation(self):
        o = self.s_word["metrics"]["overall"]
        # counts: ab 2, ' ' 3, pt 1, '\n' 1, kun 1, 12(urdu) 1, khushhal 1, 'ae.' 1 -> 11 tokens, 8 types
        p = np.array([2, 3, 1, 1, 1, 1, 1, 1]) / 11
        V = len(WORD_VOCAB) + 1
        h2 = -math.log2((p ** 2).sum())
        h25 = math.log2((p ** 2.5).sum()) / (1 - 2.5)
        self.assertAlmostEqual(o["renyi_eff_a2.0"], h2 / math.log2(V))
        self.assertAlmostEqual(o["renyi_eff_a2.5"], h25 / math.log2(V))
        self.assertEqual(o["vocab_used"], 8)
        self.assertAlmostEqual(o["vocab_utilisation"], 8 / V)

    def test_robustness_char(self):
        r = self.s_char["metrics"]["overall"]["robustness"]
        # harakat: damma removed from kun -> 24 -> 23 tokens; the word 'kun' is affected, its letters stay split
        self.assertAlmostEqual(r["harakat"]["rel_token_change"], -1 / 24)
        self.assertEqual((r["harakat"]["words_affected"], r["harakat"]["seg_change_rate"]), (1, 0.0))
        # digits: 2 urdu digits -> ascii, same count, same partition
        self.assertEqual((r["digits"]["tokens"], r["digits"]["words_affected"], r["digits"]["seg_change_rate"]), (24, 1, 0.0))
        # punct_space: a space inserted before the full stop in doc3 -> +1 token, partition unchanged
        self.assertAlmostEqual(r["punct_space"]["rel_token_change"], 1 / 24)
        self.assertEqual((r["punct_space"]["words_affected"], r["punct_space"]["seg_change_rate"]), (1, 0.0))
        # zwnj: khush|haal -> +1 token
        self.assertAlmostEqual(r["zwnj"]["rel_token_change"], 1 / 24)
        self.assertEqual((r["zwnj"]["words_affected"], r["zwnj"]["seg_change_rate_affected"]), (1, 0.0))
        for p in H.PERT_NAMES:
            self.assertEqual(r[p]["words_compared"], 7)

    def test_robustness_word(self):
        r = self.s_word["metrics"]["overall"]["robustness"]
        # harakat: 'kun' -> 'kn' (in vocab) one token -> no change
        self.assertEqual((r["harakat"]["tokens"], r["harakat"]["seg_change_rate_affected"]), (11, 0.0))
        # digits: urdu 12 -> ascii 12 (in vocab)
        self.assertEqual((r["digits"]["tokens"], r["digits"]["seg_change_rate_affected"]), (11, 0.0))
        # punct_space: 'ae.' (1 token) -> 'ae',' ','.' (3 tokens): +2; the word's partition {ae.} -> {ae}{.} changes
        self.assertEqual(r["punct_space"]["tokens"], 13)
        self.assertAlmostEqual(r["punct_space"]["extra_tokens_per_affected_word"], 2.0)
        self.assertAlmostEqual(r["punct_space"]["seg_change_rate_affected"], 1.0)
        self.assertAlmostEqual(r["punct_space"]["seg_change_rate"], 1 / 7)
        # zwnj: khush+ZWNJ+haal is unknown -> ONE <unk> token over the whole word: count and partition unchanged
        self.assertEqual((r["zwnj"]["tokens"], r["zwnj"]["seg_change_rate_affected"]), (11, 0.0))

    def test_morphology_extremes(self):
        m = self.s_char["morphology"]["silver_high"]
        self.assertEqual(m["n_unaligned"], 0)
        self.assertAlmostEqual(m["boundary_recall"], 1.0)      # every character boundary predicted
        self.assertAlmostEqual(m["single_token_rate"], 0.0)
        m = self.s_word["morphology"]["silver_high"]
        self.assertAlmostEqual(m["single_token_rate"], 1.0)    # one token per word, unknown or not
        self.assertAlmostEqual(m["boundary_recall"], 0.0)

    def test_superword_fertility(self):
        ad = A.CustomAdapter(SuperTok([AB + " " + PT]), name="super")
        s = run_on(ad, [AB + " " + PT], self.tmp)
        o = s["metrics"]["overall"]
        # one token covering two words: fertility counts it once per word -> 2/2; tokens/word 1/2; STRR 1
        self.assertEqual(o["tokens"], 1)
        self.assertAlmostEqual(o["fertility"], 1.0)
        self.assertAlmostEqual(o["tokens_per_word"], 0.5)
        self.assertAlmostEqual(o["strr"], 1.0)

    def test_g1_failure_and_unk(self):
        # 'nawan' is not in the word vocabulary -> <unk>, decoded as '<unk>' -> round trip fails
        new = NOON + WAW + ALEF + c(0x6BA)
        s = run_on(A.CustomAdapter(WordTok(WORD_VOCAB), name="word_unk"), [DOC1, AB + " " + new], self.tmp)
        g = s["gates"]["G1"]
        self.assertEqual((g["pass"], g["fail_docs"], g["unk_tokens"]), (False, 1, 1))
        self.assertEqual(g["nonws_chars_lost"], 4)                  # the 4 letters of 'nawan'
        self.assertEqual(g["nonws_chars_added"], len("<unk>"))


class TestGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hk_eval_gates_")

    def test_g2_unreachable(self):
        class Greedy(CharTok):
            """Vocabulary has 'ab' but the encoder emits single characters only -> 'ab' is unreachable."""
            def __init__(self):
                super().__init__(ALPHABET)
                self.itos.append(AB); self.stoi[AB] = len(self.itos) - 1; self.vocab_size = len(self.itos)
        g = H.gate_g2(A.CustomAdapter(Greedy(), name="greedy"))
        self.assertEqual((g["pass"], g["failures"]), (False, 1))
        self.assertEqual(g["failure_list"][0]["text"], AB)
        self.assertTrue(H.gate_g2(A.CustomAdapter(CharTok(ALPHABET), name="c"))["pass"])

    def test_g3(self):
        ok = A.CustomAdapter(CharTok(ALPHABET, specials=H.SPECIAL_TOKENS), name="c_sp")
        g = H.gate_g3(ok)
        self.assertTrue(g["pass"], g)
        self.assertEqual((g["block_size_present"], g["corpus_marker_occurrences"]), (64, 0))
        bad = A.CustomAdapter(CharTok(ALPHABET | set("".join(H.SPECIAL_TOKENS)), specials=H.SPECIAL_TOKENS,
                                      atomic_specials=False), name="c_sp_split")
        g = H.gate_g3(bad)
        self.assertFalse(g["pass"])
        self.assertEqual(g["not_atomic_n"], 64)
        g = H.gate_g3(A.CustomAdapter(CharTok(ALPHABET, specials=H.SPECIAL_TOKENS[:5]), name="c_sp5"))
        self.assertEqual((g["pass"], g["block_missing_n"]), (False, 59))

    def test_g4(self):
        a = os.path.join(self.tmp, "a.json"); b = os.path.join(self.tmp, "b.json"); d = os.path.join(self.tmp, "d.json")
        for p, t in ((a, '{"x": 1}'), (b, '{"x": 1}'), (d, '{"x": 2}')):
            open(p, "w").write(t)
        self.assertTrue(H.gate_g4(a, b)["pass"])
        self.assertFalse(H.gate_g4(a, d)["pass"])
        self.assertIsNone(H.gate_g4(a, None)["pass"])

    def test_g5_and_r2(self):
        class Partial(CharTok):
            def __init__(self):
                super().__init__(ALPHABET)
                self.itos.append("<partial>"); self.stoi["<partial>"] = len(self.itos) - 1; self.vocab_size = len(self.itos)

            def token_bytes(self, i):
                return b"\xd8" if self.itos[i] == "<partial>" else super().token_bytes(i)
        ad = A.CustomAdapter(Partial(), name="partial")
        g = H.gate_g5(ad)
        self.assertEqual((g["pass"], g["sub_character_tokens"]), (False, 1))
        r2 = H.prop_r2(ad, None, np.zeros(ad.id_upper, np.int64))
        self.assertEqual(r2["partial_utf8_tokens"], 1)
        self.assertTrue(H.gate_g5(A.CustomAdapter(CharTok(ALPHABET), name="c"))["pass"])

    def test_r1_support(self):
        ad = A.CustomAdapter(CharTok(ALPHABET), name="c")
        path = write_docs(self.tmp, [AB + " " + AB + " " + PT], name="unit_train")
        cnt, n, info = H.train_counts(ad, path)
        # 8 tokens: a 2, b 2, space 2, p 1, t 1
        self.assertEqual(n, 8)
        r1 = H.prop_r1(ad, cnt, info)["all_learned"]
        # learned = every alphabet char (no specials/base): |ALPHABET|; 4 letters + space seen -> the rest have freq 0
        self.assertEqual(r1["learned_tokens"], len(ALPHABET))
        self.assertEqual(r1["train_freq_eq0"], len(ALPHABET) - 5)
        self.assertEqual(r1["train_freq_lt20"], len(ALPHABET))


class TestTestSplitGuard(unittest.TestCase):
    def test_refuses_test_uids(self):
        """A dataset holding a test-split uid is refused (only the manifest's uid -> split map is read; the
        document text here is a dummy, no test text is touched)."""
        split = H._manifest_split()
        test_uid = next(u for u, s in sorted(split.items()) if s == "test")
        tmp = tempfile.mkdtemp(prefix="hk_eval_guard_")
        path = os.path.join(tmp, "guard.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"uid": test_uid, "source": "book", "variety": "hindko", "text": "x"}) + "\n")
        with self.assertRaises(SystemExit):
            H.load_docs(path)


class TestPerturb(unittest.TestCase):
    def test_definitions(self):
        p = P.perturb_harakat(KUN + " " + DAMMA + " " + AB)          # a mark-only 'word' vanishes, spaces fixed
        self.assertEqual(p.text, KN + " " + AB)
        P.check_alignment(KUN + " " + DAMMA + " " + AB, p)
        p = P.perturb_digits("x1 " + N12)
        self.assertEqual(p.text, "x" + U1 + " 12")
        p = P.perturb_punct_space(AE + FULL_STOP + " " + AB + " " + c(0x60C))
        self.assertEqual(p.text, AE + " " + FULL_STOP + " " + AB + c(0x60C))   # inserted where absent, deleted where present
        P.check_alignment(AE + FULL_STOP + " " + AB + " " + c(0x60C), p)
        self.assertEqual(P.perturb_zwnj(KHUSHHAL).text, KHAH + WAW + SHEEN + ZWNJ + HAH + ALEF + LAM)
        self.assertEqual(P.perturb_zwnj(KHAH + WAW + SHEEN + c(0x6CC)).n_edits, 0)     # khushi: not a compound

    def test_alignment_on_dev(self):
        docs, _ = H.load_docs("dev_strict")
        for d in docs[:200]:
            for f in P.PERTURBATIONS.values():
                P.check_alignment(d["text"], f(d["text"]))


def brute_pieces(text, spans_bytes):
    """Independent fertility: words and tokens as UTF-8 byte spans; a token counts for a word if it shares a
    non-whitespace byte with it."""
    tb = text.encode("utf-8")
    words = [(m.start(), m.end()) for m in re.finditer(rb"\S+", tb)]
    out = []
    for a, b in words:
        n = 0
        for s, e in spans_bytes:
            lo, hi = max(s, a), min(e, b)
            if hi > lo and tb[lo:hi].strip():
                n += 1
        out.append(n)
    return out


class TestDifferential(unittest.TestCase):
    """The offset-based piece counts must equal an independent byte-span computation."""

    def check(self, ad, texts):
        for t in texts:
            ids, st, en = ad.encode_offsets(t)
            ws, we = H.word_spans(t)
            got = H.word_piece_counts(ws, we, np.asarray(st), np.asarray(en), ad.notext[np.asarray(ids, dtype=np.int64)])
            spans = ad.byte_spans(t, ids)
            self.assertIsNotNone(spans)
            want = brute_pieces(t, spans.tolist())
            self.assertEqual(got.tolist(), want, ad.name)
            self.assertEqual(ad.decode(ids), t)

    def test_hf_bytelevel_gpt4o(self):
        docs, _ = H.load_docs("dev_strict")
        self.check(A.from_baseline("gpt-4o"), [d["text"] for d in docs[:60]])

    def test_hf_tiny_bpe_and_sp_wrapper(self):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
        docs, _ = H.load_docs("dev_strict")
        texts = [d["text"] for d in docs[:40]]
        train = [json.loads(l)["text"] for _, l in zip(range(400), open(H.resolve_data("train_D1"), encoding="utf-8"))]
        P1 = r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(P1), behavior="isolated"),
                                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator(train, trainer=trainers.BpeTrainer(
            vocab_size=1000, min_frequency=2, show_progress=False, special_tokens=H.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tmp = tempfile.mkdtemp(prefix="hk_eval_tiny_")
        path = os.path.join(tmp, "tiny_bpe.json")
        tk.save(path)
        ad = A.HFAdapter(path=path, name="tiny_bpe")
        self.check(ad, texts)
        self.assertTrue(H.gate_g2(ad)["pass"])
        self.assertTrue(H.gate_g3(ad)["pass"])
        self.assertGreater(H.prop_r2(ad, None, np.zeros(ad.id_upper, np.int64))["partial_utf8_tokens"], 0)
        # SentencePiece unigram with the '\n' user symbol (PLAN 1.1) + newline wrapper
        import sentencepiece as spm
        lines = os.path.join(tmp, "lines.txt")
        with open(lines, "w", encoding="utf-8", newline="\n") as f:
            for t in train:
                for ln in t.split("\n"):
                    if ln:
                        f.write(ln + "\n")
        spm.SentencePieceTrainer.train(input=lines, model_prefix=os.path.join(tmp, "tiny_sp"), model_type="unigram",
                                       vocab_size=1500, character_coverage=1.0, byte_fallback=True, split_digits=True,
                                       normalization_rule_name="identity", remove_extra_whitespaces=False,
                                       max_sentence_length=65536, user_defined_symbols=["\n"] + H.SPECIAL_TOKENS,
                                       minloglevel=2, num_threads=1)
        sp = A.SPAdapter(path=os.path.join(tmp, "tiny_sp.model"), name="tiny_sp", newline_wrapper=True)
        edge = ["\n" + AB, AB + "\n", AB + "\n\n" + PT, "\n\n", "", AB + " \n " + PT]
        for t in edge:
            self.assertEqual(sp.decode(sp.encode(t)), t, repr(t))
        self.check(sp, texts)
        self.assertTrue(H.gate_g3(sp)["pass"])
        self.assertTrue(H.gate_g5(sp)["pass"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
