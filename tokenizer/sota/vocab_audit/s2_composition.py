"""Vocabulary audit, step 2: composition by script/class and by length, plus the
character inventory of the learned pieces. Reads piece_stats.json (step 1).
Output: composition.json, char_inventory.tsv
"""
import json, os, collections, unicodedata

OUT = os.path.dirname(os.path.abspath(__file__))
SP = chr(0x2581)
P = json.load(open(os.path.join(OUT, 'piece_stats.json'), encoding='utf-8'))

by_type = collections.Counter(r['type'] for r in P)
by_class = collections.Counter(r['class'] for r in P)
learned = [r for r in P if r['type'] == 'NORMAL']

# collapse classes for the headline table
def headline(c):
    if c == 'arabic_script_word':
        return 'Arabic-script letters (incl. harakat, ZWNJ)'
    if c == 'latin':
        return 'Latin letters'
    if c == 'digit':
        return 'digits (single digit)'
    if c in ('punct', 'punct_arabic'):
        return 'punctuation'
    if c == 'symbol':
        return 'symbols'
    if c == 'space_marker':
        return 'bare word marker'
    if c.startswith('mixed'):
        return 'mixed classes'
    if c.startswith('other_script'):
        return 'other scripts'
    return c

head = collections.Counter(headline(r['class']) for r in learned)
wi = collections.Counter((headline(r['class']), r['word_initial']) for r in learned)

# length distributions (characters after the word marker)
len_all = collections.Counter(r['len_chars'] for r in learned)
len_ar = collections.Counter(r['len_chars'] for r in learned if r['class'] == 'arabic_script_word')
len_wi = collections.Counter(r['len_chars'] for r in learned if r['word_initial'])
len_cont = collections.Counter(r['len_chars'] for r in learned if not r['word_initial'])
bytes_all = collections.Counter(r['len_bytes'] for r in learned)

def summary(c):
    xs = sorted(k for k, v in c.items() for _ in range(v))
    n = len(xs)
    return {'n': n, 'mean': round(sum(xs) / n, 3), 'median': xs[n // 2], 'p90': xs[int(0.9 * n)], 'max': xs[-1]}

# token-weighted class shares on test_strict and train_D2
tok_test = collections.Counter(); tok_d2 = collections.Counter()
for r in P:
    tok_test[headline(r['class']) if r['type'] == 'NORMAL' else r['class']] += r['test_count']
    tok_d2[headline(r['class']) if r['type'] == 'NORMAL' else r['class']] += r['d2_count']

# character inventory of learned pieces
chars = collections.Counter(); single = set()
for r in learned:
    body = r['piece'][1:] if r['piece'].startswith(SP) else r['piece']
    for ch in set(body):
        chars[ch] += 1
    if len(body) == 1:
        single.add(body)
inv = []
for ch, n in sorted(chars.items(), key=lambda x: ord(x[0])):
    inv.append({'cp': 'U+%04X' % ord(ch), 'char': ch, 'name': unicodedata.name(ch, '?'),
                'cat': unicodedata.category(ch), 'pieces_containing': n, 'has_single_char_piece': ch in single})
with open(os.path.join(OUT, 'char_inventory.tsv'), 'w', encoding='utf-8') as f:
    f.write('cp\tname\tcat\tpieces_containing\thas_single_char_piece\n')
    for x in inv:
        f.write('%s\t%s\t%s\t%d\t%s\n' % (x['cp'], x['name'], x['cat'], x['pieces_containing'], x['has_single_char_piece']))

blocks = collections.Counter()
for x in inv:
    o = int(x['cp'][2:], 16)
    if o < 0x80: b = 'ASCII'
    elif o < 0x250: b = 'Latin-1/Extended'
    elif 0x0300 <= o < 0x0370: b = 'Combining diacritics'
    elif 0x0600 <= o < 0x0700: b = 'Arabic'
    elif 0x0750 <= o < 0x0780: b = 'Arabic Supplement'
    elif 0x08A0 <= o < 0x0900: b = 'Arabic Extended-A'
    elif 0xFB50 <= o < 0xFE00: b = 'Arabic Presentation Forms-A'
    elif 0xFE70 <= o < 0xFF00: b = 'Arabic Presentation Forms-B'
    elif 0x2000 <= o < 0x2070: b = 'General Punctuation'
    else: b = 'other (%s)' % x['name'].split(' ')[0]
    blocks[b] += 1

res = {
    'ids_by_sp_type': dict(by_type),
    'learned_by_class_fine': dict(by_class.most_common()),
    'learned_by_class': dict(head.most_common()),
    'learned_word_initial_vs_continuation': {f'{k[0]} | {"word-initial" if k[1] else "continuation"}': v for k, v in sorted(wi.items())},
    'n_word_initial': sum(1 for r in learned if r['word_initial']),
    'n_continuation': sum(1 for r in learned if not r['word_initial']),
    'length_chars_all': dict(sorted(len_all.items())), 'length_chars_all_summary': summary(len_all),
    'length_chars_arabic': dict(sorted(len_ar.items())), 'length_chars_arabic_summary': summary(len_ar),
    'length_chars_word_initial_summary': summary(len_wi),
    'length_chars_continuation_summary': summary(len_cont),
    'length_bytes_summary': summary(bytes_all),
    'token_share_test_strict': {k: v for k, v in tok_test.most_common() if v},
    'token_share_train_D2': {k: v for k, v in tok_d2.most_common() if v},
    'char_inventory_size': len(inv), 'char_inventory_by_block': dict(blocks.most_common()),
    'chars_without_single_piece': [x for x in inv if not x['has_single_char_piece']],
}
json.dump(res, open(os.path.join(OUT, 'composition.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('length_chars_all', 'length_chars_arabic')}, ensure_ascii=False, indent=1))
