"""Vocabulary audit, step 4: junk / corpus-artifact flags.

Flags (a piece can carry several):
  other_script        letters of a script other than Arabic or Latin (Thai, Devanagari)
  latin_accented      Latin letters outside ASCII (web language-menu residue)
  replacement_char    U+FFFD (corpus decoding defect)
  odd_symbol          ©, ۩, °
  url_fragment        >= 50 % of D2 occurrences inside a URL / e-mail / handle word
  placeholder_part    >= 50 % of D2 occurrences inside <PHONE>/<EMAIL>/<ID_NUMBER>
  boilerplate         >= 80 % of D2 occurrences in exact lines that recur in >= 3 units
  punct_glued_word    sentence punctuation followed by letters inside the piece
                      ('۔جدُوں', '▁نے۔اِس'): a missing space after ۔ ، ؟ fused two words
  word_plus_punct     letters followed only by trailing punctuation ('▁والے۔'): not junk,
                      counted because it duplicates a word with its full stop
  repeated_punct      a punctuation character repeated (…… ۔۔ ؟؟ !!)
  mark_initial        the piece body starts with a combining mark (separates a harakat
                      from its letter)
  multi_digit         >= 2 digits (Extended Arabic-Indic digits are not split by
                      split_digits); subflag year_like
  single_unit         all D2 occurrences from one book / newspaper issue / web site
  single_unit_heavy   single_unit and >= 20 occurrences (the unit's own spellings)
  tah_convention      ends in the typists' ط-for-retroflex-nasal convention (README of the
                      corpus: 'ط written for the Hindko retroflex nasal in some books')
  unused_in_D2        never produced when encoding the training text
Output: junk_flags.tsv (every flagged piece), junk_summary.json
"""
import json, os, re, collections, unicodedata

OUT = os.path.dirname(os.path.abspath(__file__))
SP = chr(0x2581)
P = json.load(open(os.path.join(OUT, 'piece_stats.json'), encoding='utf-8'))
S = json.load(open(os.path.join(OUT, 'scan.json'), encoding='utf-8'))['per_piece']
SENT = set('۔،؟!؛')
TAH = chr(0x0637)

def is_letter(c):
    return unicodedata.category(c)[0] in 'LM'

flags = {}
for r, s in zip(P, S):
    if r['type'] != 'NORMAL':
        continue
    p = r['piece']; body = p[1:] if p.startswith(SP) else p
    f = []
    cls = r['class']
    if 'other_script' in cls:
        f.append('other_script')
    if any(unicodedata.category(c)[0] == 'L' and 0x80 <= ord(c) < 0x250 for c in body):
        f.append('latin_accented')
    if chr(0xFFFD) in body:
        f.append('replacement_char')
    if any(c in body for c in (chr(0xA9), chr(0x06E9), chr(0xB0))):
        f.append('odd_symbol')
    if s['occ'] and s['url_share'] >= 0.5:
        f.append('url_fragment')
    if s['occ'] and s['placeholder_share'] >= 0.5:
        f.append('placeholder_part')
    if r['d2_boiler_share'] is not None and r['d2_boiler_share'] >= 0.8:
        f.append('boilerplate')
    # punctuation followed by a letter inside the piece
    if any(body[k] in SENT and is_letter(body[k + 1]) for k in range(len(body) - 1)):
        f.append('punct_glued_word')
    elif len(body) >= 2 and body[-1] in SENT and any(is_letter(c) for c in body):
        f.append('word_plus_punct')
    if len(body) >= 2 and all(unicodedata.category(c)[0] == 'P' for c in body) and len(set(body)) == 1:
        f.append('repeated_punct')
    if body and unicodedata.category(body[0]) == 'Mn':
        f.append('mark_initial')
    digs = re.findall(r'\d+', body)
    if digs and max(len(d) for d in digs) >= 2:
        f.append('multi_digit')
        d = max(digs, key=len)
        dd = d.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
        if len(dd) == 4 and (dd.startswith('1') or dd.startswith('20')):
            f.append('year_like')
    if r['d2_units'] == 1:
        f.append('single_unit')
        if r['d2_count'] >= 20:
            f.append('single_unit_heavy')
    b2 = body.rstrip('۔،')
    if len(b2) >= 3 and b2.endswith(TAH) and (b2[-2] in ('ا', 'ی', 'ے')):
        f.append('tah_convention')
    if r['d2_count'] == 0:
        f.append('unused_in_D2')
    if f:
        flags[r['id']] = f

cnt = collections.Counter(x for f in flags.values() for x in f)
learned = [r for r in P if r['type'] == 'NORMAL']
tot_test = sum(r['test_count'] for r in P)
tot_d2 = sum(r['d2_count'] for r in P)
def tokshare(flag):
    ids = [i for i, f in flags.items() if flag in f]
    return {'pieces': len(ids),
            'test_tokens': sum(P[i]['test_count'] for i in ids),
            'test_token_share_pct': round(100 * sum(P[i]['test_count'] for i in ids) / tot_test, 3),
            'd2_token_share_pct': round(100 * sum(P[i]['d2_count'] for i in ids) / tot_d2, 3)}
summary = {k: tokshare(k) for k in sorted(cnt)}
# single-unit breakdown by unit kind and top units
su = [i for i, f in flags.items() if 'single_unit' in f]
summary['single_unit_by_source'] = dict(collections.Counter(next(iter(P[i]['d2_sources'])) for i in su))
summary['single_unit_top_units'] = collections.Counter(P[i]['d2_top_unit'] for i in su).most_common(15)
suh = [i for i, f in flags.items() if 'single_unit_heavy' in f]
summary['single_unit_heavy_top_units'] = collections.Counter(P[i]['d2_top_unit'] for i in suh).most_common(15)
# harakat density in single-unit pieces (one book's vocalised spelling)
HAR = set(chr(c) for c in range(0x064B, 0x0653))
summary['single_unit_with_harakat'] = sum(1 for i in su if any(c in HAR for c in P[i]['piece']))
summary['all_learned_with_harakat'] = sum(1 for r in learned if any(c in HAR for c in r['piece']))
summary['word_plus_punct_by_final'] = dict(collections.Counter(P[i]['piece'][-1] for i, f in flags.items() if 'word_plus_punct' in f))
summary['punct_glued_by_unit_share'] = {
    'single_unit': sum(1 for i, f in flags.items() if 'punct_glued_word' in f and 'single_unit' in f),
    'top_units': collections.Counter(P[i]['d2_top_unit'] for i, f in flags.items() if 'punct_glued_word' in f).most_common(8)}
summary['max_repeated_punct_len'] = max((len(P[i]['piece'].lstrip(SP)) for i, f in flags.items() if 'repeated_punct' in f), default=0)
summary['multi_digit_max_len'] = max((max(len(d) for d in re.findall(r'\d+', P[i]['piece'])) for i, f in flags.items() if 'multi_digit' in f), default=0)
summary['n_learned'] = len(learned)
json.dump(summary, open(os.path.join(OUT, 'junk_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(os.path.join(OUT, 'junk_flags.tsv'), 'w', encoding='utf-8') as fo:
    fo.write('id\tpiece\tflags\td2_count\td2_docs\td2_units\ttop_unit\ttop_unit_share\ttest_count\n')
    for i in sorted(flags):
        r = P[i]
        fo.write('%d\t%s\t%s\t%d\t%d\t%d\t%s\t%.2f\t%d\n' % (i, r['piece'], ','.join(flags[i]), r['d2_count'], r['d2_docs'], r['d2_units'],
                                                         r['d2_top_unit'] or '', r['d2_top_unit_share'] or 0, r['test_count']))
print(json.dumps(summary, ensure_ascii=False, indent=1))
