"""Vocabulary audit, step 1: per-piece statistics (read-only on the tokenizer).

Loads F:\\Hindko\\tokenizer\\tokenizer.json (canonical encoder) and sp.vocab / sp.model,
checks that the three agree on the piece list, classifies every piece, and encodes
train_D2 (the tokenizer's training text), dev_strict and test_strict to get
per-piece token counts, document counts, unit (book / newspaper issue / web site)
counts and the share of occurrences in boilerplate lines (exact lines that recur
in >= 3 distinct units).

Output: piece_stats.json (list, one row per id), composition.json.
Run: PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=3 python s1_stats.py
"""
import json, os, sys, collections, unicodedata, hashlib
from tokenizers import Tokenizer
from sentencepiece import sentencepiece_model_pb2 as pb

REL = r'F:\Hindko\tokenizer'
DATA = r'F:\Hindko\_tokenizer\data'
OUT = os.path.dirname(os.path.abspath(__file__))
SP = chr(0x2581)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

# ---------------------------------------------------------------- load
tj_path = os.path.join(REL, 'tokenizer.json')
tj = json.load(open(tj_path, encoding='utf-8'))
vocab = [p for p, s in tj['model']['vocab']]
scores = [s for p, s in tj['model']['vocab']]
added = {a['id']: a for a in tj['added_tokens']}
m = pb.ModelProto(); m.ParseFromString(open(os.path.join(REL, 'sp.model'), 'rb').read())
sp_pieces = [p.piece for p in m.pieces]
sp_types = [pb.ModelProto.SentencePiece.Type.Name(p.type) for p in m.pieces]
sv = [l.rstrip('\n').split('\t') for l in open(os.path.join(REL, 'sp.vocab'), encoding='utf-8').read().split('\n') if l]
# sp.vocab: the newline piece spans two physical lines ("\n\t0.0"); rebuild robustly
raw = open(os.path.join(REL, 'sp.vocab'), encoding='utf-8').read()
sv_pieces = []
i = 0
rows = raw.split('\n')
k = 0
while k < len(rows):
    r = rows[k]
    if r == '' and k + 1 < len(rows) and rows[k + 1].startswith('\t'):
        sv_pieces.append('\n'); k += 2; continue
    if r == '':
        k += 1; continue
    sv_pieces.append(r.rsplit('\t', 1)[0]); k += 1

agree = {
    'n_tokenizer_json': len(vocab), 'n_sp_model': len(sp_pieces), 'n_sp_vocab': len(sv_pieces),
    'tokenizer_json_eq_sp_model': vocab == sp_pieces,
    'sp_vocab_eq_sp_model': sv_pieces == sp_pieces,
    'duplicate_piece_strings': [p for p, c in collections.Counter(vocab).items() if c > 1],
    'sha256': {f: sha(os.path.join(REL, f)) for f in ('tokenizer.json', 'sp.model', 'sp.vocab')},
}
if not agree['tokenizer_json_eq_sp_model']:
    agree['first_diffs'] = [(i, vocab[i], sp_pieces[i]) for i in range(min(len(vocab), len(sp_pieces))) if vocab[i] != sp_pieces[i]][:20]
print(json.dumps({k: v for k, v in agree.items() if k != 'sha256'}, ensure_ascii=False))

# ---------------------------------------------------------------- classification
ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
def is_arabic_block(c):
    o = ord(c)
    return any(a <= o <= b for a, b in ARABIC_RANGES)

def char_class(c):
    cat = unicodedata.category(c)
    o = ord(c)
    if cat == 'Nd':
        return 'digit'
    if cat[0] == 'P':
        return 'punct'
    if cat[0] == 'S':
        return 'symbol'
    if cat in ('Cf',):
        return 'format'  # ZWNJ/ZWJ/bidi etc.
    if cat[0] == 'C':
        return 'control_other'
    if cat[0] == 'Z':
        return 'space'
    if cat[0] in ('L', 'M'):
        if is_arabic_block(c):
            return 'arabic'
        name = unicodedata.name(c, '')
        if name.startswith('LATIN') or (o < 0x250 and cat[0] == 'L'):
            return 'latin'
        if cat[0] == 'M' and o < 0x0370:
            return 'latin'  # combining diacritics used with Latin
        return 'other_script:' + (name.split(' ')[0] if name else 'UNNAMED')
    if cat == 'No':
        return 'digit_other'
    return 'other:' + cat

def piece_class(i):
    p = vocab[i]
    t = sp_types[i]
    if t == 'USER_DEFINED':
        return 'newline' if p == '\n' else 'special'
    if t == 'CONTROL':
        return 'special_placeholder'
    if t == 'UNKNOWN':
        return 'unk'
    if t == 'BYTE':
        return 'byte_fallback'
    body = p[1:] if p.startswith(SP) else p
    if body == '':
        return 'space_marker'
    cls = set(char_class(c) for c in body)
    # ZWNJ inside an Arabic-script word counts as Arabic-script
    if cls <= {'arabic', 'format'} and 'arabic' in cls:
        return 'arabic_script_word'
    if len(cls) == 1:
        c = next(iter(cls))
        if c == 'arabic':
            return 'arabic_script_word'
        if c == 'latin':
            return 'latin'
        if c == 'digit':
            return 'digit'
        if c == 'punct':
            # Arabic-block punctuation vs other
            return 'punct_arabic' if all(is_arabic_block(ch) for ch in body) else 'punct'
        if c == 'symbol':
            return 'symbol'
        return c
    return 'mixed:' + '+'.join(sorted(cls))

classes = [piece_class(i) for i in range(len(vocab))]

# ---------------------------------------------------------------- encoding
tok = Tokenizer.from_file(tj_path)

def unit_of(r):
    g = r['group']
    if g.startswith('web:'):
        return ':'.join(g.split(':')[:2])
    return g

def load(fn):
    return [json.loads(l) for l in open(os.path.join(DATA, fn), encoding='utf-8')]

def encode_lines(lines):
    uniq = list(dict.fromkeys(lines))
    enc = tok.encode_batch(uniq, add_special_tokens=False)
    return {u: e.ids for u, e in zip(uniq, enc)}

N = len(vocab)
def fresh():
    return {'count': [0] * N, 'docs': [0] * N}

stats = {}
# train_D2 with unit/source detail and boilerplate share
d2 = load('train_D2.jsonl')
line_units = collections.defaultdict(set)
for r in d2:
    u = unit_of(r)
    for ln in r['text'].split('\n'):
        if ln:
            line_units[ln].add(u)
boiler = {ln for ln, us in line_units.items() if len(us) >= 3}
print('D2 docs', len(d2), 'distinct lines', len(line_units), 'boilerplate lines (>=3 units)', len(boiler))
lmap = encode_lines(list(line_units.keys()))
cnt = [0] * N; docs = [0] * N; boil = [0] * N
units = [None] * N; srcs = [None] * N; docuids = [None] * N
for di, r in enumerate(d2):
    u = unit_of(r); s = r['source']
    seen = collections.Counter()
    for ln in r['text'].split('\n'):
        if not ln:
            continue
        ids = lmap[ln]
        b = ln in boiler
        for t in ids:
            seen[t] += 1
            if b:
                boil[t] += 1
    nl = r['text'].count('\n')
    if nl:
        seen[64] += nl
    for t, c in seen.items():
        cnt[t] += c; docs[t] += 1
        if units[t] is None:
            units[t] = collections.Counter(); srcs[t] = collections.Counter(); docuids[t] = []
        units[t][u] += c; srcs[t][s] += c
        if len(docuids[t]) < 5:
            docuids[t].append(r['uid'])
stats['train_D2'] = {'count': cnt, 'docs': docs}
print('D2 tokens', sum(cnt))

for view in ('dev_strict', 'test_strict', 'dev_permissive'):
    rows = load(view + '.jsonl')
    allines = [ln for r in rows for ln in r['text'].split('\n') if ln]
    lm = encode_lines(allines)
    c = [0] * N; d = [0] * N
    for r in rows:
        seen = collections.Counter()
        for ln in r['text'].split('\n'):
            if ln:
                for t in lm[ln]:
                    seen[t] += 1
        nl = r['text'].count('\n')
        if nl:
            seen[64] += nl
        for t, k in seen.items():
            c[t] += k; d[t] += 1
    stats[view] = {'count': c, 'docs': d, 'n_docs': len(rows), 'tokens': sum(c)}
    print(view, len(rows), 'docs', sum(c), 'tokens')

# cross-check: full-document encoding equals the line-wise reconstruction on test_strict
rows = load('test_strict.jsonl')
mism = 0
for r in rows:
    full = tok.encode(r['text'], add_special_tokens=False).ids
    rec = []
    for k, ln in enumerate(r['text'].split('\n')):
        if k:
            rec.append(64)
        if ln:
            rec.extend(tok.encode(ln, add_special_tokens=False).ids)
    mism += (full != rec)
print('test_strict full-doc vs line-wise mismatches:', mism)

out = []
for i in range(N):
    p = vocab[i]
    body = p[1:] if p.startswith(SP) else p
    u = units[i] or collections.Counter()
    top = u.most_common(1)
    out.append({
        'id': i, 'piece': p, 'type': sp_types[i], 'class': classes[i],
        'word_initial': p.startswith(SP) and sp_types[i] == 'NORMAL',
        'len_chars': len(body) if sp_types[i] == 'NORMAL' else None,
        'len_bytes': len(body.encode('utf-8')) if sp_types[i] == 'NORMAL' else None,
        'score': scores[i],
        'd2_count': stats['train_D2']['count'][i], 'd2_docs': stats['train_D2']['docs'][i],
        'd2_units': len(u), 'd2_top_unit': top[0][0] if top else None,
        'd2_top_unit_share': (top[0][1] / stats['train_D2']['count'][i]) if top else None,
        'd2_sources': dict(srcs[i]) if srcs[i] else {},
        'd2_boiler_share': (boil[i] / stats['train_D2']['count'][i]) if stats['train_D2']['count'][i] else None,
        'd2_example_uids': docuids[i] or [],
        'test_count': stats['test_strict']['count'][i], 'test_docs': stats['test_strict']['docs'][i],
        'dev_count': stats['dev_strict']['count'][i], 'dev_docs': stats['dev_strict']['docs'][i],
        'devp_count': stats['dev_permissive']['count'][i],
    })
json.dump(out, open(os.path.join(OUT, 'piece_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False)
json.dump({'agreement': agree,
           'views': {v: {'tokens': sum(stats[v]['count']), 'n_docs': stats[v].get('n_docs', len(d2))} for v in stats},
           'boilerplate_lines': len(boiler), 'd2_distinct_lines': len(line_units),
           'test_fulldoc_vs_linewise_mismatch': mism},
          open(os.path.join(OUT, 's1_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done')
