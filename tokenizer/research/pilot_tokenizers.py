# -*- coding: utf-8 -*-
"""PILOT ONLY (not the final protocol): training time and compression of a few
baseline tokenizers on the Hindko corpus, to calibrate PLAN.md.

Split: a deterministic pseudo-split by md5(uid) (5% held out). This is NOT
issue/book-disjoint; the real protocol uses F:\\Hindko\\_tokenizer\\splits.
Reads the released dataset read-only; writes only under research/pilot/.
"""
import hashlib, json, math, os, sys, time, collections
import regex
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
import sentencepiece as spm

SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pilot"); os.makedirs(OUT, exist_ok=True)
THREADS = 3
os.environ["RAYON_NUM_THREADS"] = str(THREADS)

# P1: letters+combining marks (+ZWNJ/ZWJ) stay together; single digits; punctuation runs.
P1 = r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
# P0: GPT-2 regex (combining marks are NOT \p{L}, so diacritised words get split)
P0 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

def load():
    tr, te = [], []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h = int(hashlib.md5(str(r["uid"] if "uid" in r else r["id"]).encode()).hexdigest(), 16)
            (te if h % 20 == 0 else tr).append(r["text"])
    return tr, te

def renyi_eff(counts, alpha=2.5):
    tot = sum(counts.values()); ps = [c / tot for c in counts.values() if c > 0]
    h = math.log(sum(p ** alpha for p in ps)) / (1 - alpha)
    return h / math.log(len(counts)) if len(counts) > 1 else 0.0

def evaluate(name, encode, vocab_size, te, train_s):
    n_tok = 0; n_bytes = 0; n_chars = 0; n_words = 0; single = 0; cnt = collections.Counter()
    rt_fail = 0
    for t in te:
        ids, dec = encode(t)
        n_tok += len(ids); cnt.update(ids); n_bytes += len(t.encode("utf-8")); n_chars += len(t)
        if dec is not None and dec != t: rt_fail += 1
    # fertility on whitespace words (encode each word with a leading space, as in running text)
    wc = collections.Counter(w for t in te for w in t.split())
    tot_w = sum(wc.values())
    fert = 0
    for w, c in wc.items():
        k = len(encode(" " + w)[0]); fert += k * c
        if k == 1: single += c
    row = {"tokenizer": name, "vocab": vocab_size, "train_s": round(train_s, 1),
           "bytes_per_token": round(n_bytes / n_tok, 3), "chars_per_token": round(n_chars / n_tok, 3),
           "fertility_tok_per_word": round(fert / tot_w, 3), "pct_words_single_token": round(100 * single / tot_w, 1),
           "renyi_eff_a2.5": round(renyi_eff(cnt), 4), "vocab_used_in_heldout": len(cnt),
           "roundtrip_fail_docs": rt_fail, "heldout_docs": len(te)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row

def hf_bpe(tr, te, V, pattern, tag):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trn = trainers.BpeTrainer(vocab_size=V, min_frequency=2, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"])
    t0 = time.time(); tok.train_from_iterator(tr, trainer=trn); dt = time.time() - t0
    tok.save(os.path.join(OUT, "hf_bpe_%s_%d.json" % (tag, V)))
    enc = lambda s: (lambda e: (e.ids, tok.decode(e.ids)))(tok.encode(s))
    return evaluate("hf_bytelevel_bpe_%s" % tag, enc, V, te, dt)

def sp_train(tr_path, te, V, mtype):
    prefix = os.path.join(OUT, "sp_%s_%d" % (mtype, V))
    t0 = time.time()
    spm.SentencePieceTrainer.train(input=tr_path, model_prefix=prefix, model_type=mtype, vocab_size=V,
        character_coverage=1.0, byte_fallback=True, split_digits=True, normalization_rule_name="identity",
        remove_extra_whitespaces=False, add_dummy_prefix=True, max_sentence_length=1 << 16,
        num_threads=THREADS, input_sentence_size=0, shuffle_input_sentence=False, minloglevel=2,
        unk_surface="\u2047")
    dt = time.time() - t0
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    enc = lambda s: (lambda ids: (ids, sp.decode(ids)))(sp.encode(s))
    return evaluate("sp_%s" % mtype, enc, V, te, dt)

if __name__ == "__main__":
    tr, te = load()
    print("train docs", len(tr), "heldout docs", len(te), flush=True)
    tr_path = os.path.join(OUT, "train_lines.txt")
    with open(tr_path, "w", encoding="utf-8") as g:
        for t in tr:
            for ln in t.split("\n"):
                if ln.strip(): g.write(ln + "\n")
    rows = []
    which = sys.argv[1:] or ["hf", "sp"]
    for V in (16000, 32000):
        if "hf" in which:
            rows.append(hf_bpe(tr, te, V, P1, "P1"))
            rows.append(hf_bpe(tr, te, V, P0, "P0gpt2"))
        if "sp" in which:
            rows.append(sp_train(tr_path, te, V, "unigram"))
    with open(os.path.join(OUT, "pilot_results_%s.json" % "_".join(which)), "w", encoding="utf-8") as g:
        json.dump({"note": "pilot on md5(uid)%20==0 held-out; not issue/book-disjoint", "P1": P1, "P0": P0,
                   "rows": rows}, g, ensure_ascii=False, indent=1)
