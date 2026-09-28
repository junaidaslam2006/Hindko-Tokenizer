"""Dataset QA: length distribution, over-merge detection, rejection review."""
import io, json, re, collections, statistics

R = r'F:\Hindko'
strict = [json.loads(l) for l in io.open(R + r'\hindko_dataset.jsonl', encoding='utf-8') if l.strip()]
perm = [json.loads(l) for l in io.open(R + r'\hindko_dataset_permissive.jsonl', encoding='utf-8') if l.strip()]
rej = [json.loads(l) for l in io.open(R + r'\rejected\rejected.jsonl', encoding='utf-8') if l.strip()]

print('=== COUNTS ===')
print('strict=%d permissive(incl strict)=%d rejected=%d' % (len(strict), len(perm), len(rej)))

L = sorted(r['n_chars'] for r in strict)
print('\n=== STRICT LENGTH DISTRIBUTION (chars) ===')
print('min=%d p10=%d median=%d p90=%d p99=%d max=%d mean=%.0f'
      % (L[0], L[len(L)//10], L[len(L)//2], L[len(L)*9//10],
         L[int(len(L)*0.99)], L[-1], statistics.mean(L)))

# --- over-merge detection -------------------------------------------------
# A record that swallowed several articles will contain multiple datelines.
DATELINE = re.compile(
    r'(?:پشور|پشاو ر|لاہور|کراچی|اسلام آباد|ہزارہ|مانسہرہ|ایبٹ آباد|کوہاٹ|بنوں|ٹانک|صوابی|مردان|سوات|چترال)'
    r'\s*\(|\((?:ہندکووان نیوز|ہندکووا ن نیوز|ویب ڈیسک|نمائندہ ہندکووان|سٹاف رپورٹر)\)')

multi = []
for r in strict:
    n = len(DATELINE.findall(r['text']))
    if n >= 2:
        multi.append((n, r))
multi.sort(key=lambda x: -x[0])
print('\n=== OVER-MERGE CHECK: records containing >=2 datelines ===')
print('count: %d / %d (%.1f%%)' % (len(multi), len(strict), 100*len(multi)/max(len(strict),1)))
for n, r in multi[:5]:
    print('  datelines=%d chars=%d id=%s file=%s' % (n, r['n_chars'], r['id'], r['source_file'][:34]))
    print('     title=%r' % (r['title'] or '')[:70])

print('\n=== LONGEST 5 STRICT RECORDS ===')
for r in sorted(strict, key=lambda x: -x['n_chars'])[:5]:
    nd = len(DATELINE.findall(r['text']))
    print('  %d chars, %d datelines, %s  title=%r'
          % (r['n_chars'], nd, r['source_file'][:30], (r['title'] or '')[:50]))

print('\n=== REJECTED (all %d) ===' % len(rej))
for r in rej:
    print('  chars=%-6d %-42s %s' % (r['n_chars'], r['source_file'][:42],
                                     ';'.join(r['reasons'])[:60]))

print('\n=== PERMISSIVE-ONLY (flagged) ===')
flagged = [p for p in perm if p['quality_tier'] != 'strict']
print('count:', len(flagged))
fc = collections.Counter()
for p in flagged:
    for f in p['quality_flags']:
        fc[f.split('(')[0]] += 1
print('flag types:', dict(fc.most_common()))
for p in flagged[:5]:
    print('  %-14s chars=%-5d %s' % (p['id'], p['n_chars'], p['quality_flags']))
    print('      %s' % p['text'][:110].replace('\n', ' / '))

print('\n=== METADATA COVERAGE (strict) ===')
for k in ['title', 'date', 'issue', 'section', 'date_precision']:
    print('  %-16s %d/%d (%.0f%%)' % (k, sum(1 for r in strict if r.get(k)),
                                       len(strict),
                                       100*sum(1 for r in strict if r.get(k))/max(len(strict),1)))
print('  distinct issues:', len({r['issue'] for r in strict if r.get('issue')}))
print('  roles:', dict(collections.Counter(r['role'] for r in strict)))
print('  date_source:', dict(collections.Counter(r.get('date_source') for r in strict)))

# script sanity: confirm no replacement chars and no residue symbols leaked
bad_repl = sum(r['text'].count('\ufffd') for r in perm)
bad_res = sum(len(re.findall(r'[_|~`{}<>]', r['text'])) for r in perm)
print('\n=== SANITY ===')
print('  U+FFFD in all accepted text:', bad_repl)
print('  residue symbols in all accepted text:', bad_res)
print('  records with zero digits:', sum(1 for r in strict if not re.search(r'[0-9\u06f0-\u06f9]', r['text'])))
