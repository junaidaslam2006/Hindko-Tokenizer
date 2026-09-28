"""Helper: print where given pieces occur in the encoded train_D2 (and optionally
test/dev) text, with a short character window. For the auditor's inspection only;
nothing is written to disk.

Usage: python ctx.py [--n 5] [--w 30] [--views train_D2,test_strict] ID_OR_PIECE ...
A piece argument may use '_' for the word marker U+2581 only if prefixed with 'sp:'.
"""
import json, os, sys, collections
from tokenizers import Tokenizer

REL = r'F:\Hindko\tokenizer'
DATA = r'F:\Hindko\_tokenizer\data'
SP = chr(0x2581)
args = sys.argv[1:]
n = 5; w = 30; views = ['train_D2']
while args and args[0].startswith('--'):
    k = args.pop(0)
    if k == '--n': n = int(args.pop(0))
    elif k == '--w': w = int(args.pop(0))
    elif k == '--views': views = args.pop(0).split(',')
tok = Tokenizer.from_file(os.path.join(REL, 'tokenizer.json'))
targets = []
for a in args:
    if a.isdigit():
        targets.append(int(a))
    else:
        if a.startswith('sp:'):
            a = SP + a[3:]
        i = tok.token_to_id(a)
        if i is None:
            print('not a piece:', repr(a)); continue
        targets.append(i)
tset = set(targets)
found = collections.defaultdict(list)
for v in views:
    for line in open(os.path.join(DATA, v + '.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        for ln in r['text'].split('\n'):
            if not ln:
                continue
            e = tok.encode(ln, add_special_tokens=False)
            for t, (s, en) in zip(e.ids, e.offsets):
                if t in tset and len(found[t]) < n:
                    # offsets refer to the normalized string (with the prefix marker); approximate
                    s0 = max(0, s - 1)
                    found[t].append((v, r['uid'], r['group'][:60], ln[max(0, s0 - w):s0 + w + 10].replace('\t', ' ')))
        if all(len(found[t]) >= n for t in tset):
            break
for t in targets:
    print('==', t, repr(tok.id_to_token(t)))
    for x in found[t]:
        print('   ', x)
