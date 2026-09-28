"""Per-word context profiles for the homograph check (step 1b).

For every word type of the STRICT corpus (read-only) with freq >= MIN_FREQ,
count the class of the token that FOLLOWS and of the token that PRECEDES each
occurrence, inside a sentence (lines are split at ۔ ؟ ! ? . first, so a
sentence end is a context of its own). Also keeps the most frequent actual
neighbours, for reading.

Output: data/context_profiles.json
  meta  : classes and their word lists, parameters
  words : {form: {n, next: {class: count}, prev: {class: count},
                  top_next: [[tok, count], ...], top_prev: [...]}}

Tokenisation is the same as build_counts.py (hp.lang.TOKEN_SPLIT_RE, harakat
stripped with hp.lang.strip_marks). Deterministic (no randomness; sorted output).
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, 'F:/Hindko/_pipeline')
from hp.lang import TOKEN_SPLIT_RE, strip_marks  # noqa: E402

STRICT = 'F:/Hindko/hindko_dataset.jsonl'
MIN_FREQ = 10
TOP_K = 10
SENT_SPLIT_RE = re.compile('[' + chr(0x06D4) + chr(0x061F) + '!?.]+')   # ۔ ؟ ! ? .

# ---- context classes (closed lists; a token belongs to the first class that lists it)
AUX = ('وے اے ہے ہیں ہن ون ان ایا ائی ائے ایاں ایہا ایہی ایہے ایہیاں آسا آسی آسے آسیاں اسا سا '
       'پیا پئی پئے پئیاں واں آں ایں ویں ہاں رہیا رئے رئی').split()          # = induce.AUX
GENP = 'دا دی دے دیاں'.split()        # genitive after the word: a noun (or an attributive participle) precedes
CASE = ('نوں کو نال اچ بچ وچ توں کولوں وسے واسطے آسطے تک '
        'وچوں بچوں اچوں تھیں کنوں').split()          # other case postpositions: a noun (or infinitive) precedes
AMB = 'تے سی نے'.split()             # also 'and' / FUT.3SG, Hazara 'was' / ERG: ambiguous after verbs
NMLZ = 'ولا ولی ولے ولیاں والا والی والے والیاں آلا آلی آلے آلیاں'.split()    # agentive / prospective
KE = ['کہ']
INFGOV = ('لگا لگی لگے لگیاں لگ لگدا لگدی لگدے لگدیاں لگیا لگسی لگن جوگا جوگی جوگے جوگیاں').split()   # 'begin to' / 'able to': take an oblique infinitive (کہنڑے لگے)
GEN = ('دا دی دے دیاں میرا میری میرے میریاں تیرا تیری تیرے تیریاں ساڈا ساڈی ساڈے ساڈیاں '
       'سواڈا سواڈی سواڈے آپڑا آپڑی آپڑے آپڑیاں اپڑا اپڑی اپڑے اپڑیاں ازا ازی ازے').split()
DETQ = ('ہر دونوں ہک اک کئی کسے بوت بہت بڑی بڑا بڑے اتنی اتنا اتنے کتنی کتنا کتنے جتنی جتنا '
        'جتنے سجے کھبے سجی کھبی سجا کھبا دوسرے دوسری دوسرا دوجے دوجی دوجا کجھ تھوڑا '
        'تھوڑی تھوڑے زیادہ کافی').split()   # quantifiers, intensifiers, 'right/left/other': a noun/adverb follows
# (کوئی 'someone' and سارے 'all' are left out: as subjects they often precede a verb)
DEM = 'ایہہ اوہ ایہ او اس ایس اوس'.split()
NEG = 'نہ نئیں نہیں نیں'.split()

NEXT_CLASSES = [('AUX', AUX), ('GENP', GENP), ('CASE', CASE), ('AMB', AMB), ('NMLZ', NMLZ), ('KE', KE),
                ('INFGOV', INFGOV)]  # + END, SAME, OTHER
PREV_CLASSES = [('GEN', GEN), ('DETQ', DETQ), ('DEM', DEM), ('NEG', NEG)]              # + START, SAME, OTHER


def classifier(classes):
    table = {}
    for name, words in classes:
        for w in words:
            table.setdefault(w, name)
    return table


NEXT_OF = classifier(NEXT_CLASSES)
PREV_OF = classifier(PREV_CLASSES)


def sentences(text):
    for ln in text.splitlines():
        for seg in SENT_SPLIT_RE.split(ln):
            toks = [strip_marks(t) for t in TOKEN_SPLIT_RE.split(seg) if t]
            toks = [t for t in toks if t]
            if toks:
                yield toks


def main():
    freq = collections.Counter()
    for line in open(STRICT, encoding='utf-8'):
        for toks in sentences(json.loads(line)['text']):
            freq.update(toks)
    keep = {w for w, c in freq.items() if c >= MIN_FREQ}
    nxt = collections.defaultdict(collections.Counter)
    prv = collections.defaultdict(collections.Counter)
    nxt_tok = collections.defaultdict(collections.Counter)
    prv_tok = collections.defaultdict(collections.Counter)
    for line in open(STRICT, encoding='utf-8'):
        for toks in sentences(json.loads(line)['text']):
            for i, t in enumerate(toks):
                if t not in keep:
                    continue
                if i + 1 < len(toks):
                    x = toks[i + 1]
                    nxt[t]['SAME' if x == t else NEXT_OF.get(x, 'OTHER')] += 1
                    nxt_tok[t][x] += 1
                else:
                    nxt[t]['END'] += 1
                if i > 0:
                    x = toks[i - 1]
                    prv[t]['SAME' if x == t else PREV_OF.get(x, 'OTHER')] += 1
                    prv_tok[t][x] += 1
                else:
                    prv[t]['START'] += 1

    def top(c):
        return [[k, v] for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]]
    words = {}
    for w in sorted(keep):
        words[w] = dict(n=freq[w], next=dict(sorted(nxt[w].items())), prev=dict(sorted(prv[w].items())),
                        top_next=top(nxt_tok[w]), top_prev=top(prv_tok[w]))
    meta = dict(source=STRICT, min_freq=MIN_FREQ, top_k=TOP_K, sentence_split=SENT_SPLIT_RE.pattern,
                next_classes={k: v for k, v in NEXT_CLASSES}, prev_classes={k: v for k, v in PREV_CLASSES},
                note='next: + END (sentence end), SAME (word repeated), OTHER; prev: + START, SAME, OTHER')
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    with open(os.path.join(ROOT, 'data', 'context_profiles.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, words=words), f, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    print('words', len(words), 'tokens', sum(freq.values()))


if __name__ == '__main__':
    main()
