"""MorphScore-style evaluation of tokenizer / morpheme-boundary alignment
on the Hindko SILVER morphology set (morph_silver_high.tsv, morph_silver_low.tsv).

    from morph_eval import load_gold, evaluate, report
    gold = load_gold('F:/Hindko/_tokenizer/morphology/morph_silver_high.tsv')
    res = evaluate(encode, gold)        # encode(word) -> tokens
    print(report(res))

`encode(word)` may return
  * a list of token STRINGS whose concatenation spells the word, after the
    tokenizer's own markers are removed: SentencePiece '▁', WordPiece '##',
    GPT-2 byte-level strings (pass byte_level=True), byte-fallback pieces
    '<0xNN>'; or
  * a list of (start, end) CHARACTER offsets into the word (HF fast tokenizers).
Alignment is done on UTF-8 bytes, so a byte-level tokenizer that splits a
two-byte Arabic letter produces a boundary that can never match a gold
boundary (it is counted as a false positive, and inside the stem as an
over-split).

Gold (per word): required boundaries R (stem|suffix, and suffix|suffix for
causatives), optional boundaries O (linguistically defensible splits inside a
suffix, e.g. -د|ا, -س|ی, -نڑ|اں: neither rewarded nor penalised), and for
ambiguous words alternative required sets A (بچیاں = بچ|یاں or بچی|اں). The
stem is the span before the first required boundary.

Metrics (P = predicted internal boundaries of one word):
  boundary_precision / recall / f1  micro-averaged over words:
        TP = |P ∩ R|, FP = |P \\ (R ∪ O)|, FN = |R \\ P|
  morphscore            share of MULTI-token words whose first (stem|suffix)
                        boundary is in P (single-token words excluded, as in
                        MorphScore, Arnett & Bergen 2025)
  morphscore_all        same, single-token words counted as failures
  stem_intact           share of words with no boundary strictly inside the stem
  stem_boundary_respected  share of words whose stem|suffix boundary is in P
                        AND whose stem is not split: the headline number
  exact_match           P minus optional == R
  single_token_rate, tokens_per_word
For words with alternatives, the alternative giving the best
(stem_boundary_respected, f1) is used. With weight='freq' every word counts
with its strict-corpus frequency instead of once.

CLI:
  python morph_eval.py --demo                      trivial tokenizers (char, whole-word, oracle, random)
  python morph_eval.py --hf NAME_OR_PATH [--prefix-space]   HF tokenizer (offsets)
  python morph_eval.py --spm MODEL.model           SentencePiece model
  python morph_eval.py --tokenizer-json FILE.json  `tokenizers` JSON file
  add --gold FILE.tsv to use another gold file, --weight freq, --json OUT.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLD = [os.path.join(HERE, 'morph_silver_high.tsv'), os.path.join(HERE, 'morph_silver_low.tsv')]
SP_MARK = '\u2581'           # SentencePiece word-boundary marker
BYTE_PIECE = re.compile(r'^<0x([0-9A-Fa-f]{2})>$')


# --------------------------------------------------------------------------- gold
class GoldWord:
    __slots__ = ('word', 'req', 'opt', 'alts', 'category', 'freq', 'id', 'confidence', 'segmentation')

    def __init__(self, word, req, opt, alts, category='', freq=1, id='', confidence='', segmentation=''):
        self.word, self.req, self.opt, self.alts = word, tuple(req), tuple(opt), [tuple(a) for a in alts]
        self.category, self.freq, self.id, self.confidence = category, freq, id, confidence
        self.segmentation = segmentation


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()] if s else []


def load_gold(paths: Sequence[str] | str) -> List[GoldWord]:
    """Read one or more silver TSV files and validate every row."""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                w = r['word']
                req = _ints(r['boundaries'])
                opt = _ints(r.get('optional_boundaries', ''))
                alts = [_ints(a) for a in r.get('alternatives', '').split(';') if a.strip()]
                seg = r.get('segmentation', '')
                # validation: offsets inside the word, segmentation string agrees
                for b in req + opt + [x for a in alts for x in a]:
                    assert 0 < b < len(w), (p, r['id'], w, b)
                assert not set(req) & set(opt), (p, r['id'])
                if seg:
                    assert seg.replace('|', '') == w, (p, r['id'], seg, w)
                    pos, cuts = 0, []
                    for piece in seg.split('|')[:-1]:
                        pos += len(piece); cuts.append(pos)
                    assert cuts == req, (p, r['id'], seg, req)
                assert unicodedata.is_normalized('NFC', w), (p, r['id'])
                out.append(GoldWord(w, req, opt, alts, r.get('category', ''), int(r.get('freq') or 1),
                                    r.get('id', ''), r.get('confidence', ''), seg))
    return out


# ------------------------------------------------------------------ alignment
def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte<->unicode table (same construction as in GPT-2 / HF)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + \
        list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def _token_bytes(tok: str, byte_level: bool) -> bytes:
    m = BYTE_PIECE.match(tok)
    if m:
        return bytes([int(m.group(1), 16)])
    if byte_level:
        return bytes(_BYTE_DECODER[c] for c in tok if c in _BYTE_DECODER)
    if tok.startswith('##'):
        tok = tok[2:]
    return tok.replace(SP_MARK, '').encode('utf-8')


def char_to_byte(word: str) -> List[int]:
    """offset table: char index -> byte index (len(word)+1 entries)"""
    out, n = [0], 0
    for ch in word:
        n += len(ch.encode('utf-8')); out.append(n)
    return out


def boundaries_from_offsets(word: str, spans) -> Tuple[Optional[List[int]], int]:
    """Character (start, end) offsets -> internal boundaries in UTF-8 byte offsets.
    Returns (None, n) if the spans do not tile the word."""
    c2b = char_to_byte(word)
    spans = [(int(s), int(e)) for s, e in spans if int(e) > int(s)]
    if not spans or spans[0][0] != 0 or spans[-1][1] != len(word):
        return None, len(spans)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        if e1 != s2:
            return None, len(spans)
    return [c2b[s] for s, _ in spans[1:]], len(spans)


def boundaries_from_tokens(word: str, toks: Sequence[str], byte_level: bool = False
                           ) -> Tuple[Optional[List[int]], int]:
    """Token strings -> internal boundaries in UTF-8 byte offsets (empty tokens,
    e.g. a lone '▁', are ignored). Returns (None, n) when the tokens do not
    spell the word."""
    wb = word.encode('utf-8')
    pieces = [_token_bytes(t, byte_level) for t in toks]
    pieces = [p for p in pieces if p]
    if b''.join(pieces) == b' ' + wb:          # word encoded with a leading space (prefix_space / Ġ)
        pieces[0] = pieces[0][1:]
        pieces = [p for p in pieces if p]
    if b''.join(pieces) != wb:
        # tolerate a normalisation that keeps the byte length (e.g. NFKC of NFC Arabic letters)
        joined = unicodedata.normalize('NFKC', b''.join(pieces).decode('utf-8', 'replace'))
        if joined != unicodedata.normalize('NFKC', word) or len(joined.encode('utf-8')) != len(wb):
            return None, len(pieces)
    cuts, pos = [], 0
    for p in pieces[:-1]:
        pos += len(p); cuts.append(pos)
    return cuts, len(pieces)


# ------------------------------------------------------------------- scoring
def score_word(g: GoldWord, P: List[int]) -> Dict[str, float]:
    c2b = char_to_byte(g.word)
    Pset = set(P)
    best = None
    for req in [g.req] + g.alts:
        R = {c2b[b] for b in req}
        O = {c2b[b] for b in g.opt} - R
        stem_end = min(R)
        tp = len(Pset & R)
        fp = len(Pset - R - O)
        fn = len(R - Pset)
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        stem_intact = not any(0 < p < stem_end for p in Pset)
        stem_b = stem_end in Pset
        d = dict(tp=tp, fp=fp, fn=fn, f1=f1, prec=prec, rec=rec, stem_intact=stem_intact,
                 stem_boundary=stem_b, respected=stem_b and stem_intact,
                 exact=(Pset - O) == R, alt=req != g.req)
        key = (d['respected'], d['f1'], not d['alt'])
        if best is None or key > best[0]:
            best = (key, d)
    return best[1]


def evaluate(encode: Callable[[str], list], gold: List[GoldWord], byte_level: bool = False,
             weight: str = 'type', by_category: bool = True) -> Dict:
    """Run `encode` on every gold word and aggregate the metrics."""
    agg = _new_agg()
    cats: Dict[str, Dict] = {}
    unaligned = []
    for g in gold:
        enc = encode(g.word) or []
        if enc and isinstance(enc[0], (tuple, list)):
            P, ntok = boundaries_from_offsets(g.word, enc)
        else:
            P, ntok = boundaries_from_tokens(g.word, list(enc), byte_level or getattr(encode, 'byte_level', False))
        if P is None:
            unaligned.append(g.word)
            continue
        w = g.freq if weight == 'freq' else 1
        d = score_word(g, P)
        _add(agg, d, ntok, w)
        if by_category:
            _add(cats.setdefault(g.category, _new_agg()), d, ntok, w)
    res = _finish(agg)
    res['n_gold'] = len(gold)
    res['n_unaligned'] = len(unaligned)
    res['unaligned_examples'] = unaligned[:10]
    res['weight'] = weight
    if by_category:
        res['by_category'] = {k: _finish(v) for k, v in sorted(cats.items())}
    return res


def _new_agg():
    return dict(n=0.0, tp=0.0, fp=0.0, fn=0.0, f1_sum=0.0, multi=0.0, multi_hit=0.0, hit=0.0, intact=0.0,
                respected=0.0, exact=0.0, single=0.0, tokens=0.0, words=0)


def _add(a, d, ntok, w):
    a['words'] += 1
    a['n'] += w
    a['tp'] += w * d['tp']; a['fp'] += w * d['fp']; a['fn'] += w * d['fn']
    a['f1_sum'] += w * d['f1']
    a['hit'] += w * d['stem_boundary']
    a['intact'] += w * d['stem_intact']
    a['respected'] += w * d['respected']
    a['exact'] += w * d['exact']
    a['tokens'] += w * ntok
    if ntok >= 2:
        a['multi'] += w; a['multi_hit'] += w * d['stem_boundary']
    else:
        a['single'] += w


def _finish(a):
    n = a['n'] or float('nan')
    p = a['tp'] / (a['tp'] + a['fp']) if (a['tp'] + a['fp']) else float('nan')
    r = a['tp'] / (a['tp'] + a['fn']) if (a['tp'] + a['fn']) else float('nan')
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else (0.0 if r == 0 else float('nan'))
    return dict(words=a['words'], boundary_precision=p, boundary_recall=r, boundary_f1=f,
                boundary_f1_macro=a['f1_sum'] / n,
                morphscore=(a['multi_hit'] / a['multi']) if a['multi'] else float('nan'),
                morphscore_all=a['hit'] / n, stem_intact=a['intact'] / n,
                stem_boundary_respected=a['respected'] / n, exact_match=a['exact'] / n,
                single_token_rate=a['single'] / n, tokens_per_word=a['tokens'] / n)


def _fmt(x):
    return '  n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:5.3f}'


def report(res: Dict, name: str = '') -> str:
    keys = ['boundary_precision', 'boundary_recall', 'boundary_f1', 'morphscore', 'morphscore_all',
            'stem_intact', 'stem_boundary_respected', 'exact_match', 'single_token_rate', 'tokens_per_word']
    lines = [f'== {name}  (gold words {res["n_gold"]}, unaligned {res["n_unaligned"]}, weight={res["weight"]})']
    lines.append('   ' + '  '.join(f'{k}={_fmt(res[k])}' for k in keys))
    for cat, v in res.get('by_category', {}).items():
        lines.append(f'   {cat:10s} n={v["words"]:4d}  F1={_fmt(v["boundary_f1"])}  '
                     f'MorphScore={_fmt(v["morphscore"])}  respected={_fmt(v["stem_boundary_respected"])}  '
                     f'tok/word={_fmt(v["tokens_per_word"])}')
    return '\n'.join(lines)


# --------------------------------------------------------------- tokenizers
def char_tokenizer(word: str) -> List[str]:
    return list(word)


def whole_word_tokenizer(word: str) -> List[str]:
    return [word]


def make_oracle(gold: List[GoldWord]) -> Callable[[str], List[str]]:
    seg = {g.word: g.segmentation.split('|') for g in gold}
    return lambda w: seg[w]


def make_random(p: float = 0.3, seed: int = 13) -> Callable[[str], List[str]]:
    """Split between characters with probability p (deterministic per word)."""
    def enc(word):
        rng = random.Random(f'{seed}:{word}')
        out, cur = [], word[0]
        for ch in word[1:]:
            if rng.random() < p:
                out.append(cur); cur = ch
            else:
                cur += ch
        out.append(cur)
        return out
    return enc


def make_suffix_stripper(suffixes: Iterable[str]) -> Callable[[str], List[str]]:
    """Naive baseline: split off the longest listed suffix, never touching the stem."""
    sufs = sorted(set(suffixes), key=len, reverse=True)

    def enc(word):
        for s in sufs:
            if word.endswith(s) and len(word) > len(s) + 1:
                return [word[:-len(s)], s]
        return [word]
    return enc


def _is_byte_level(tok_json: dict) -> bool:
    """GPT-2-style byte-level BPE (token strings are byte-mapped)?"""
    t = json.dumps(tok_json.get('pre_tokenizer')) + json.dumps(tok_json.get('decoder'))
    return '"ByteLevel"' in t


def tokenizers_json_encoder(path: str, prefix_space: bool = False):
    """Encoder for a `tokenizers` tokenizer.json. Returns token strings; the
    function carries .byte_level so evaluate() decodes byte-level tokens."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    for fn in ('no_truncation', 'no_padding'):
        try:
            getattr(tok, fn)()
        except Exception:
            pass

    def enc(word):
        return tok.encode((' ' + word) if prefix_space else word, add_special_tokens=False).tokens
    enc.byte_level = _is_byte_level(json.load(open(path, encoding='utf-8')))
    return enc


def hf_encoder(name_or_path: str, prefix_space: bool = False):
    """transformers AutoTokenizer (fast: backend token strings; slow: tokenize())."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if getattr(tok, 'is_fast', False):
        backend = tok.backend_tokenizer

        def enc(word):
            return backend.encode((' ' + word) if prefix_space else word, add_special_tokens=False).tokens
        enc.byte_level = _is_byte_level(json.loads(backend.to_str()))
    else:
        def enc(word):
            return tok.tokenize((' ' + word) if prefix_space else word)
        enc.byte_level = False
    return enc


def spm_encoder(model_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return lambda w: sp.encode(w, out_type=str)


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gold', nargs='*', default=None, help='gold TSV file(s); default: high and low separately')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--hf')
    ap.add_argument('--prefix-space', action='store_true')
    ap.add_argument('--spm')
    ap.add_argument('--tokenizer-json', nargs='*')
    ap.add_argument('--byte-level', action='store_true')
    ap.add_argument('--weight', default='type', choices=['type', 'freq'])
    ap.add_argument('--json', help='write all results to this JSON file')
    a = ap.parse_args(argv)
    gold_sets = {os.path.basename(p): load_gold(p) for p in (a.gold or DEFAULT_GOLD)}
    encoders = {}
    if a.demo:
        sys.path.insert(0, os.path.join(HERE, 'scripts'))
        from morph_inventory import VERB_SUFFIXES, NOUN_SUFFIXES, AGR_SUFFIXES
        sufs = [s['suffix'] for s in VERB_SUFFIXES + NOUN_SUFFIXES + AGR_SUFFIXES]
        encoders['char-level'] = char_tokenizer
        encoders['whole-word'] = whole_word_tokenizer
        encoders['random-split p=0.3 (seed 13)'] = make_random(0.3, 13)
        encoders['naive longest-suffix stripper'] = make_suffix_stripper(sufs)
    if a.hf:
        encoders[f'hf:{a.hf}' + (' +prefix-space' if a.prefix_space else '')] = hf_encoder(a.hf, a.prefix_space)
    if a.spm:
        encoders[f'spm:{os.path.basename(a.spm)}'] = spm_encoder(a.spm)
    for tj in (a.tokenizer_json or []):
        label = os.path.basename(os.path.dirname(os.path.abspath(tj))) + '/' + os.path.basename(tj)
        encoders[f'tokenizers:{label}' + (' +prefix-space' if a.prefix_space else '')] = \
            tokenizers_json_encoder(tj, a.prefix_space)
    if not encoders:
        ap.error('nothing to evaluate: pass --demo, --hf, --spm or --tokenizer-json')
    results = {}
    for gname, gold in gold_sets.items():
        encs = dict(encoders)
        if a.demo:
            encs['oracle (gold segmentation)'] = make_oracle(gold)
        for ename, enc in encs.items():
            res = evaluate(enc, gold, byte_level=a.byte_level, weight=a.weight)
            results[f'{gname} :: {ename}'] = res
            print(report(res, f'{gname} :: {ename}'))
            print()
    if a.json:
        def clean(x):          # strict JSON: NaN (undefined metric) -> null
            if isinstance(x, float) and math.isnan(x):
                return None
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [clean(v) for v in x]
            return x
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(clean(results), f, ensure_ascii=False, indent=1)
    return results


if __name__ == '__main__':
    main()
