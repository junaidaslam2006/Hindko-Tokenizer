"""Vocabulary audit, step 3: one pass over train_D2 with token offsets.

Per piece it computes
  url_share          share of occurrences inside a whitespace word that looks like a URL/e-mail/handle
  placeholder_share  share of occurrences inside <PHONE>/<EMAIL>/<ID_NUMBER>
  whole_word_share   share of occurrences where a word-initial piece is the whole word
                     (next token starts a new word, or is punctuation, or the line ends)
  after_punct_glued  share of occurrences of a continuation piece that starts right after
                     sentence punctuation with no space (e.g. '...کیتا۔اِس')
and, for the candidate set (word-initial Arabic-script pieces with <= 2 D2 documents,
pieces with a takhallus/honorific sign, and all flagged by later steps via --extra),
the neighbouring words (previous and next whitespace word) as counters.

Nothing from the corpus is written except these aggregates and neighbour-word counts
for the candidate set (scan.json), which stays in this private study folder.
"""
import json, os, re, sys, collections
from tokenizers import Tokenizer

REL = r'F:\Hindko\tokenizer'
DATA = r'F:\Hindko\_tokenizer\data'
OUT = os.path.dirname(os.path.abspath(__file__))
SP = chr(0x2581)
P = json.load(open(os.path.join(OUT, 'piece_stats.json'), encoding='utf-8'))
tok = Tokenizer.from_file(os.path.join(REL, 'tokenizer.json'))
N = len(P)

URLISH = re.compile(r'(https?:|www\.|\.com|\.pk|\.org|\.net|@|\.gov|\.edu|/[A-Za-z])', re.I)
PLACE = re.compile(r'<(PHONE|EMAIL|ID_NUMBER)>')
PUNCT_END = set('۔،؟!؛:.,)(]["\'' + chr(0x2018) + chr(0x2019) + chr(0x201C) + chr(0x201D) + chr(0x2026))
SIGNS = {chr(0x0610), chr(0x0611), chr(0x0612), chr(0x0613), chr(0x0614)}

cand = set()
for r in P:
    if r['type'] != 'NORMAL':
        continue
    body = r['piece'][1:] if r['piece'].startswith(SP) else r['piece']
    if r['word_initial'] and r['class'] == 'arabic_script_word' and r['d2_docs'] <= 2:
        cand.add(r['id'])
    if any(c in SIGNS for c in body):
        cand.add(r['id'])
extra = sys.argv[1:]
for a in extra:
    cand.add(int(a))

occ = [0] * N; url = [0] * N; ph = [0] * N; whole = [0] * N; glued = [0] * N
prevw = collections.defaultdict(collections.Counter)
nextw = collections.defaultdict(collections.Counter)
wordform = collections.defaultdict(collections.Counter)   # the whitespace word that contains the piece

lines = collections.Counter()
for l in open(os.path.join(DATA, 'train_D2.jsonl'), encoding='utf-8'):
    r = json.loads(l)
    for ln in r['text'].split('\n'):
        if ln:
            lines[ln] += 1
uniq = list(lines)
encs = tok.encode_batch(uniq, add_special_tokens=False)
for ln, e in zip(uniq, encs):
    mult = lines[ln]
    # whitespace word spans
    spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', ln)]
    starts = [s for s, _ in spans]
    import bisect
    phs = [(m.start(), m.end()) for m in PLACE.finditer(ln)]
    ids = e.ids; offs = e.offsets; toks = e.tokens
    for k, (t, (a, b)) in enumerate(zip(ids, offs)):
        occ[t] += mult
        # locate containing word
        a2 = a + 1 if (a < len(ln) and ln[a] == ' ') else a
        wi = bisect.bisect_right(starts, a2) - 1
        word = ln[spans[wi][0]:spans[wi][1]] if wi >= 0 and spans[wi][0] <= a2 < spans[wi][1] else ''
        if word and URLISH.search(word):
            url[t] += mult
        if any(s <= a2 < en for s, en in phs):
            ph[t] += mult
        if toks[k].startswith(SP):
            nxt = toks[k + 1] if k + 1 < len(toks) else None
            if nxt is None or nxt.startswith(SP) or (nxt and nxt[0] in PUNCT_END):
                whole[t] += mult
        else:
            if a > 0 and ln[a - 1] in '۔،؟!؛' and (a < len(ln) and ln[a] not in PUNCT_END):
                glued[t] += mult
        if t in cand:
            prevw[t][ln[spans[wi - 1][0]:spans[wi - 1][1]] if wi >= 1 else '<BOL>'] += mult
            nextw[t][ln[spans[wi + 1][0]:spans[wi + 1][1]] if 0 <= wi < len(spans) - 1 else '<EOL>'] += mult
            wordform[t][word] += mult

res = {'per_piece': [], 'cand': {}}
for i in range(N):
    o = occ[i]
    res['per_piece'].append({'id': i, 'occ': o,
                             'url_share': url[i] / o if o else None,
                             'placeholder_share': ph[i] / o if o else None,
                             'whole_word_share': whole[i] / o if o else None,
                             'glued_after_punct_share': glued[i] / o if o else None})
for t in cand:
    res['cand'][t] = {'prev': prevw[t].most_common(8), 'next': nextw[t].most_common(8),
                      'words': wordform[t].most_common(5)}
json.dump(res, open(os.path.join(OUT, 'scan.json'), 'w', encoding='utf-8'), ensure_ascii=False)
print('pieces', N, 'candidates', len(cand), 'check occ == d2_count:',
      sum(1 for i in range(N) if i != 64 and occ[i] != P[i]['d2_count']))
