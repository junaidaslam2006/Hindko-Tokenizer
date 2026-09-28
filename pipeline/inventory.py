import os, collections, json

roots = [r'F:\Hindko\raw_files', r'F:\Hindko\_pipeline\samples']
ext = collections.Counter()
inps = []
for R in roots:
    for root, _, files in os.walk(R):
        for f in files:
            if f.startswith('_') or f.endswith('.part'):
                continue
            p = os.path.join(root, f)
            e = os.path.splitext(f)[1].lower().lstrip('.')
            ext[e] += 1
            if e in ('inp', 'b01'):
                inps.append((p, os.path.getsize(p)))

print('local inventory by extension:')
for e, c in ext.most_common():
    print('   %4d  .%s' % (c, e))
print('\nlocal .inp/.B01 files (%d):' % len(inps))
for p, s in sorted(inps):
    print('   %8d  %s' % (s, p.replace('F:\\Hindko\\', '')))

tot = 0
for R in roots:
    for root, _, files in os.walk(R):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
print('\ntotal local bytes: %.1f MB' % (tot / 1e6))

# revised size estimate from what we actually pulled
M = r'F:\Hindko\raw_files\_manifest.jsonl'
recs = [json.loads(l) for l in open(M, encoding='utf-8') if l.strip()]
by = collections.defaultdict(list)
for r in recs:
    by[os.path.splitext(r['path'])[1].lower().lstrip('.')].append(r['size'])
listing = json.load(open(r'F:\Hindko\_pipeline\drive_listing.json', encoding='utf-8'))
counts = collections.Counter(os.path.splitext(e['path'])[1].lower().lstrip('.') for e in listing)
print('\nrevised total-size estimate (mean sampled size x file count):')
grand = 0
for e, sizes in sorted(by.items(), key=lambda x: -sum(x[1])):
    mean = sum(sizes) / len(sizes)
    est = mean * counts[e]
    grand += est
    print('   .%-5s n=%-5d sampled=%-3d mean=%8.1f MB  est=%8.1f MB'
          % (e, counts[e], len(sizes), mean/1e6, est/1e6))
print('   ESTIMATED TOTAL: %.1f GB' % (grand / 1e9))
