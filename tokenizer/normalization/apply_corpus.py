"""Apply hp.normalize to the permissive corpus and measure what changes.

Outputs (this folder):
  apply/apply_stats.json   per rule x source: characters changed, records changed; totals;
                           idempotency and NFC checks; residual Arabic-variant letters
  apply/changes.jsonl      one line per changed record: uid, source, tier, rule counts, changed spans
  apply/examples.txt       deterministic before/after examples (round-robin over rules)
Reads the released corpus read-only. Deterministic (file order, sorted keys).
"""
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import normalize as N  # noqa: E402

PERM = r'F:\Hindko\hindko_dataset_permissive.jsonl'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'apply')
os.makedirs(OUT, exist_ok=True)
SOURCES = ('newspaper', 'book', 'web')
WORD_RE = N.WORD_RE


def spans(a, b, ctx=12):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        out.append({'tag': tag, 'before': a[max(0, i1 - ctx):i2 + ctx], 'after': b[max(0, j1 - ctx):j2 + ctx],
                    'removed': a[i1:i2], 'inserted': b[j1:j2],
                    'removed_cps': ['U+%04X' % ord(c) for c in a[i1:i2]][:12],
                    'inserted_cps': ['U+%04X' % ord(c) for c in b[j1:j2]][:12]})
    return out


def rule_of_span(sp):
    """Best-effort attribution of a diff span to a rule, for picking examples."""
    r = set(sp['removed'])
    i = set(sp['inserted'])
    if chr(0x95) in r:
        return 'R03_controls'
    if any(0xFB50 <= ord(c) <= 0xFEFF for c in r):
        return 'R04_presentation'
    if any(0x0660 <= ord(c) <= 0x0669 for c in r):
        return 'R13_digits'
    if any(0x08BE <= ord(c) <= 0x08C2 for c in i):
        return 'R12_tone_letters'
    return 'R11_arabic_letters'


def main():
    chars = Counter()
    recs = Counter()
    strict_chars = Counter()
    strict_recs = Counter()
    rule_chars = defaultdict(Counter)       # rule -> source -> chars
    rule_recs = defaultdict(Counter)        # rule -> source -> records
    strict_rule_chars = Counter()
    strict_rule_recs = Counter()
    any_recs = Counter()
    any_chars = Counter()
    not_idem = 0
    not_nfc = 0
    len_delta = Counter()
    residual = defaultdict(Counter)         # cp -> word -> n  (after normalization)
    residual_occ = defaultdict(Counter)     # cp -> source -> n
    watch = {0x064A: 'ARABIC YEH', 0x0649: 'ALEF MAKSURA', 0x0643: 'ARABIC KAF', 0x0629: 'TEH MARBUTA',
             0x0647: 'ARABIC HEH', 0x0678: 'HIGH HAMZA YEH', 0x0660: 'ARABIC-INDIC ZERO'}
    by_rule_examples = defaultdict(list)
    fw = open(os.path.join(OUT, 'changes.jsonl'), 'w', encoding='utf-8')
    with open(PERM, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            s, t, tier = r['source'], r['text'], r['quality_tier']
            chars[s] += len(t)
            recs[s] += 1
            if tier == 'strict':
                strict_chars[s] += len(t)
                strict_recs[s] += 1
            out, rep = N.normalize_with_report(t)
            if N.normalize(out) != out:
                not_idem += 1
            if not unicodedata.is_normalized('NFC', out):
                not_nfc += 1
            for rid, k in rep.items():
                rule_chars[rid][s] += k
                rule_recs[rid][s] += 1
                if tier == 'strict':
                    strict_rule_chars[rid] += k
                    strict_rule_recs[rid] += 1
            if rep:
                any_recs[s] += 1
                any_chars[s] += sum(rep.values())
                len_delta[s] += len(out) - len(t)
                sp = spans(t, out)
                fw.write(json.dumps({'uid': r['uid'], 'source': s, 'tier': tier, 'site': r.get('site'),
                                     'book_folder': r.get('book_folder'), 'issue': r.get('issue'),
                                     'rules': rep, 'spans': sp}, ensure_ascii=False) + '\n')
                for x in sp:
                    by_rule_examples[rule_of_span(x)].append((s, r['uid'], r.get('site') or r.get('book_folder')
                                                              or r.get('issue'), x))
            # residual Arabic-variant letters after normalization
            for cp in watch:
                c = chr(cp)
                if cp == 0x0660:
                    k = sum(out.count(chr(0x0660 + i)) for i in range(10))
                    if k:
                        residual_occ['U+0660-0669'][s] += k
                    continue
                if c in out:
                    residual_occ['U+%04X' % cp][s] += out.count(c)
                    for m in WORD_RE.finditer(out):
                        if c in m.group():
                            residual['U+%04X' % cp][m.group()] += 1
    fw.close()

    tot_chars = sum(chars.values())
    stats = {
        'normalization_version': N.NORMALIZATION_VERSION, 'unicode_version': N.UNICODE_VERSION,
        'input': PERM,
        'records': dict(recs), 'chars': dict(chars), 'total_chars': tot_chars,
        'strict_records': dict(strict_recs), 'strict_chars': dict(strict_chars),
        'records_changed_any_rule': dict(any_recs), 'chars_changed_any_rule': dict(any_chars),
        'length_delta_after_normalization': dict(len_delta),
        'per_rule': {rid: {'chars': dict(rule_chars[rid]), 'records': dict(rule_recs[rid]),
                           'chars_total': sum(rule_chars[rid].values()),
                           'records_total': sum(rule_recs[rid].values()),
                           'strict_chars': strict_rule_chars[rid], 'strict_records': strict_rule_recs[rid]}
                     for rid in N.RULE_IDS},
        'not_idempotent_records': not_idem, 'not_nfc_output_records': not_nfc,
        'residual_after_normalization': {k: dict(v) for k, v in sorted(residual_occ.items())},
        'residual_words_top': {k: v.most_common(40) for k, v in sorted(residual.items())},
        'residual_word_types': {k: len(v) for k, v in sorted(residual.items())},
    }
    with open(os.path.join(OUT, 'apply_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    # deterministic example pick: round-robin over rules, first unseen (uid, before) per rule
    L = []
    rules = sorted(by_rule_examples)
    idx = {k: 0 for k in rules}
    seen = set()
    picked = []
    while len(picked) < 40 and any(idx[k] < len(by_rule_examples[k]) for k in rules):
        for k in rules:
            while idx[k] < len(by_rule_examples[k]):
                s, uid, tag, x = by_rule_examples[k][idx[k]]
                idx[k] += 1
                key = (x['removed'], x['inserted'])
                if key in seen:
                    continue
                seen.add(key)
                picked.append((k, s, uid, tag, x))
                break
    for k, s, uid, tag, x in picked:
        L.append('%s | %s | %s | %s' % (k, s, uid, tag))
        L.append('   before: %s' % x['before'].replace('\n', ' / '))
        L.append('   after : %s' % x['after'].replace('\n', ' / '))
        L.append('   removed %s -> inserted %s' % (x['removed_cps'], x['inserted_cps']))
    with open(os.path.join(OUT, 'examples.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    # console summary
    print('records', dict(recs), 'chars', dict(chars))
    print('changed records', dict(any_recs), 'changed chars', dict(any_chars))
    print('not idempotent', not_idem, 'not NFC', not_nfc)
    for rid in N.RULE_IDS:
        pr = stats['per_rule'][rid]
        if pr['chars_total']:
            print('%-20s chars %-40s recs %-40s strict %d/%d' % (rid, pr['chars'], pr['records'], pr['strict_chars'],
                                                                  pr['strict_records']))
    print('residual', json.dumps(stats['residual_after_normalization']))
    print('residual types', stats['residual_word_types'])
    for k, v in stats['residual_words_top'].items():
        print(k, v[:25])
    print('examples by rule', {k: len(v) for k, v in by_rule_examples.items()})


if __name__ == '__main__':
    main()
