"""Digit decision evidence: does keeping ASCII and Urdu digits apart cost tokens?

Trains byte-level BPE tokenizers (HF tokenizers 0.22) on the canonical data form
of the permissive corpus, 90/10 split by uid (last hex digit of sha1(uid) == '0'
-> held-out; deterministic), in four settings:

  digits  : keep   = digits as written (ASCII and Extended Arabic-Indic apart)
            fold   = every digit folded to ASCII (the alternative being judged)
  numbers : single = each digit its own pre-token (LLaMA / DeepSeek style)
            group3 = runs of up to 3 digits per pre-token (GPT-4 cl100k / o200k style)

Pre-tokenizer: an o200k-like split that keeps combining marks and ZWNJ inside
words (the GPT-2 pattern would cut every haraka out of its word), then
ByteLevel(use_regex=False). Vocab 32,000, min_frequency 2, no normalizer.

Reports on the held-out part: total tokens, tokens spent on digit runs, digit
tokens per digit, vocabulary entries that contain a digit, round-trip
exactness. Also measures pre-token fragmentation of the GPT-2 regex vs the
mark-aware regex on the whole held-out text.
Environment: RAYON_NUM_THREADS=3 (set below), HF_HOME on F:.
"""
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault('RAYON_NUM_THREADS', '3')
os.environ.setdefault('HF_HOME', r'F:\Hindko\_tokenizer\hf_cache')
sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import normalize as N  # noqa: E402
from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, trainers  # noqa: E402

PERM = r'F:\Hindko\hindko_dataset_permissive.jsonl'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'digits')
os.makedirs(OUT, exist_ok=True)
VOCAB = 32000

WORD = r"[\p{L}\p{M}\x{200C}]"
NOTWORD = r"[^\r\n\p{L}\p{M}\p{N}\x{200C}]"
PAT = {
    'single': NOTWORD + '?' + WORD + r"+|\p{N}| ?[^\s\p{L}\p{M}\p{N}\x{200C}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    'group3': NOTWORD + '?' + WORD + r"+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}\x{200C}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

FOLD_ASCII = str.maketrans({**{chr(0x06F0 + i): str(i) for i in range(10)},
                            **{chr(0x0660 + i): str(i) for i in range(10)}})
DIGIT_RUN = re.compile('[0-9\u06F0-\u06F9\u0660-\u0669]+')


def load():
    train, held = [], []
    with open(PERM, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            t = N.normalize(r['text'])
            (held if hashlib.sha1(r['uid'].encode()).hexdigest()[-1] == '0' else train).append(t)
    return train, held


def build(pattern):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    return tok


def main():
    t0 = time.time()
    train, held = load()
    print('train docs %d chars %d | held docs %d chars %d | %.0fs' % (
        len(train), sum(map(len, train)), len(held), sum(map(len, held)), time.time() - t0))
    results = {'vocab_size': VOCAB, 'train_docs': len(train), 'held_docs': len(held),
               'train_chars': sum(map(len, train)), 'held_chars': sum(map(len, held)), 'runs': {}}
    # digit inventory of the held-out part
    runs = [m.group() for t in held for m in DIGIT_RUN.finditer(t)]
    results['held_digit_runs'] = len(runs)
    results['held_digit_chars'] = sum(map(len, runs))
    results['held_digit_runs_by_script'] = {
        'ascii': sum(1 for r in runs if r.isascii()),
        'ext_arabic_indic': sum(1 for r in runs if all('\u06F0' <= c <= '\u06F9' for c in r)),
        'mixed_or_other': sum(1 for r in runs if not r.isascii() and not all('\u06F0' <= c <= '\u06F9' for c in r))}
    for numbers in ('single', 'group3'):
        for digits in ('keep', 'fold'):
            name = '%s_%s' % (digits, numbers)
            tr = train if digits == 'keep' else [t.translate(FOLD_ASCII) for t in train]
            he = held if digits == 'keep' else [t.translate(FOLD_ASCII) for t in held]
            tok = build(PAT[numbers])
            trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2, show_progress=False,
                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
            t1 = time.time()
            tok.train_from_iterator(tr, trainer=trainer, length=len(tr))
            secs = time.time() - t1
            enc = tok.encode_batch(he)
            total = sum(len(e.ids) for e in enc)
            rt_ok = sum(1 for t, e in zip(he, enc) if tok.decode(e.ids) == t)
            # tokens covering digit runs: count tokens whose offsets fall inside a digit run
            dig_tok = 0
            for t, e in zip(he, enc):
                spans = [(m.start(), m.end()) for m in DIGIT_RUN.finditer(t)]
                if not spans:
                    continue
                k = 0
                for (a, b) in e.offsets:
                    while k < len(spans) and spans[k][1] <= a:
                        k += 1
                    if k < len(spans) and spans[k][0] <= a < spans[k][1]:
                        dig_tok += 1
            vocab = tok.get_vocab()
            dec = decoders.ByteLevel()
            vd = [dec.decode([v]) for v in vocab]
            with_digit = [s for s in vd if any(c.isdigit() for c in s)]
            results['runs'][name] = {
                'train_seconds': round(secs, 1), 'held_tokens': total,
                'held_chars_per_token': round(sum(map(len, he)) / total, 4),
                'held_tokens_on_digit_runs': dig_tok,
                'digit_tokens_per_digit_char': round(dig_tok / max(1, results['held_digit_chars']), 4),
                'vocab_entries_with_digit': len(with_digit),
                'vocab_entries_digit_only': sum(1 for s in with_digit if s.strip().isdigit()),
                'roundtrip_exact_docs': rt_ok, 'roundtrip_docs': len(he)}
            tok.save(os.path.join(OUT, 'bpe32k_%s.json' % name))
            print(name, json.dumps(results['runs'][name]), flush=True)
    # pre-token fragmentation: GPT-2 regex vs mark-aware regex
    words_with_marks = 0
    words = 0
    for t in held:
        for w in N.WORD_RE.finditer(t):
            words += 1
            if any(0x0610 <= ord(c) <= 0x061A or 0x064B <= ord(c) <= 0x065F or ord(c) == 0x0670 for c in w.group()):
                words_with_marks += 1
    frag = {}
    for label, pat in (('gpt2', GPT2), ('mark_aware_single', PAT['single'])):
        sp = pre_tokenizers.Split(Regex(pat), behavior='isolated')
        frag[label] = sum(len(sp.pre_tokenize_str(t)) for t in held)
    results['held_arabic_words'] = words
    results['held_arabic_words_with_marks'] = words_with_marks
    results['held_pretokens'] = frag
    with open(os.path.join(OUT, 'digits_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != 'runs'}, indent=1))
    print('total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
