"""Vocabulary audit, step 7: special tokens and reserved ids must be unreachable from normal text.

Checks
  1. Declarations: tokenizer.json added_tokens (special flag, normalized/lstrip/rstrip),
     sp.model piece types, tokenizer_config / special_tokens_map roles.
  2. Real text: encode train_D2, train_D1, dev_strict, dev_permissive, test_strict with
     tokenizer.json and count every id in the 'never from text' set
     {0..63, 65, 7183, 16359, 16629, 16924, 19063}; count literal '<|' in the text (gate G3).
     The same for sp.model with the newline convention, on dev_strict and test_strict.
  3. Synthetic text: every learned piece string on its own; 50,000 random concatenations of
     1-8 learned pieces with random spaces; the 256 single bytes as characters; near-miss
     strings ('<|bos', '< |bos| >', fullwidth, '<unk>', '<0x41>', '<s>', '[UNK]', '<|reserved_59|>',
     '<|unused_5|>', upper-case variants ...).
  4. Literal special strings (documented behaviour, not normal text): tokenizer.json and
     transformers cut them out as special ids; transformers' split_special_tokens=True
     keeps them as text.
Output: specials.json
"""
import json, os, random, re, collections
from tokenizers import Tokenizer
import sentencepiece as spm

REL = r'F:\Hindko\tokenizer'
DATA = r'F:\Hindko\_tokenizer\data'
OUT = os.path.dirname(os.path.abspath(__file__))
SP = chr(0x2581)
tj = json.load(open(os.path.join(REL, 'tokenizer.json'), encoding='utf-8'))
tok = Tokenizer.from_file(os.path.join(REL, 'tokenizer.json'))
sp = spm.SentencePieceProcessor(model_file=os.path.join(REL, 'sp.model'))
cfg = json.load(open(os.path.join(REL, 'tokenizer_config.json'), encoding='utf-8'))
stm = json.load(open(os.path.join(REL, 'special_tokens_map.json'), encoding='utf-8'))
UNUSED = [7183, 16359, 16629, 16924, 19063]
NEVER = set(range(0, 64)) | {65} | set(UNUSED)
res = {}

# 1 declarations ---------------------------------------------------------------------------
at = tj['added_tokens']
res['declarations'] = {
    'added_tokens': len(at),
    'added_special': sum(a['special'] for a in at),
    'added_ids': sorted(a['id'] for a in at) == sorted(list(range(65)) + UNUSED),
    'newline_added_not_special': [a for a in at if a['id'] == 64][0]['special'] is False,
    'all_normalized_false': all(a['normalized'] is False for a in at),
    'all_lstrip_rstrip_false': all(not a['lstrip'] and not a['rstrip'] for a in at),
    'all_single_word_false': all(not a['single_word'] for a in at),
    'model_unk_id': tj['model']['unk_id'], 'byte_fallback': tj['model']['byte_fallback'],
    'model_vocab_scores_of_never_ids': sorted(set(tj['model']['vocab'][i][1] for i in NEVER)),
    'sp_types_of_never_ids': dict(collections.Counter(
        ('CONTROL' if sp.is_control(i) else 'UNKNOWN' if sp.is_unknown(i) else 'BYTE' if sp.is_byte(i) else 'USER_DEFINED/NORMAL') for i in NEVER)),
    'roles': {k: stm.get(k) for k in ('bos_token', 'eos_token', 'pad_token', 'unk_token')},
    'config_add_bos_eos': (cfg.get('add_bos_token'), cfg.get('add_eos_token')),
    'config_unk_token': cfg.get('unk_token'),
    'no_learned_piece_contains_pipe_or_angle_pipe': not any(('|' in p) for i, (p, s) in enumerate(tj['model']['vocab']) if i not in NEVER and not p.startswith('<0x')),
}

# 2 real text ------------------------------------------------------------------------------
real = {}
for v in ('train_D2', 'train_D1', 'dev_strict', 'dev_permissive', 'test_strict'):
    docs = [json.loads(l)['text'] for l in open(os.path.join(DATA, v + '.jsonl'), encoding='utf-8')]
    lit = sum(d.count('<|') for d in docs)
    c = collections.Counter(); n = 0
    for e in tok.encode_batch(docs, add_special_tokens=False):
        n += len(e.ids)
        for t in e.ids:
            if t in NEVER:
                c[t] += 1
    real[v] = {'docs': len(docs), 'tokens': n, 'never_ids_emitted': dict(c), 'literal_<|_in_text': lit}
for v in ('dev_strict', 'test_strict'):
    docs = [json.loads(l)['text'] for l in open(os.path.join(DATA, v + '.jsonl'), encoding='utf-8')]
    c = collections.Counter()
    for d in docs:
        for ids in sp.encode(d.split('\n')):
            for t in ids:
                if t in NEVER:
                    c[t] += 1
    real[v]['sp_model_never_ids_emitted'] = dict(c)
res['real_text'] = real

# 3 synthetic ------------------------------------------------------------------------------
vocab = [p for p, s in tj['model']['vocab']]
learned = [i for i in range(len(vocab)) if i not in NEVER and i != 64 and not vocab[i].startswith('<0x')]
rng = random.Random(20260927)
synth = []
for i in learned:
    synth.append(vocab[i].replace(SP, ' '))
for _ in range(50000):
    k = rng.randint(1, 8)
    s = ''
    for _ in range(k):
        s += vocab[rng.choice(learned)].replace(SP, ' ')
        if rng.random() < 0.3:
            s += ' '
    synth.append(s)
synth += [chr(b) for b in range(256) if b not in (10,)]
near = ['<|bos', 'bos|>', '< |bos| >', '<| bos |>', '<|BOS|>', '<|Bos|>', '<||>', '<|>', '|>', '<|',
        '<|reserved_59|>', '<|reserved_-1|>', '<|reserved_00|>', '<|unused_5|>', '<|UNUSED_0|>', '<|unused0|>',
        '<unk>', '<UNK>', '[UNK]', '<s>', '</s>', '<pad>', '<0x41>', '<0x0A>', '<|endoftext', 'endoftext|>',
        '<|im_start', '<|im_end', '<|im_start |>', '<\u200c|bos|>', '<|bos|\u200c>', '\uff1c\uff5cbos\uff5c\uff1e',
        '\u2039|bos|\u203a', '<\u2502bos\u2502>', '<\u01c0bos\u01c0>', '\u2581', '\u2581\u2581', '\n\n', '\r\n', '\t']
synth += near
c = collections.Counter(); n = 0
for e in tok.encode_batch(synth, add_special_tokens=False):
    n += len(e.ids)
    for t in e.ids:
        if t in NEVER:
            c[t] += 1
csp = collections.Counter()
for s in synth[:len(learned)] + synth[-len(near):]:
    for ids in sp.encode(s.split('\n')):
        for t in ids:
            if t in NEVER:
                csp[t] += 1
res['synthetic'] = {'strings': len(synth), 'tokens': n, 'never_ids_emitted_tokenizer_json': dict(c),
                    'sp_model_checked_strings': len(learned) + len(near), 'never_ids_emitted_sp_model': dict(csp),
                    'near_miss_examples': {s: tok.encode(s, add_special_tokens=False).tokens for s in near[:12]}}

# 4 literal special strings (documented behaviour) --------------------------------------------
lits = ['<|endoftext|>', '<|bos|>', '<|reserved_17|>', '<|unused_2|>', 'متن <|im_start|> متن']
beh = {}
for s in lits:
    beh[s] = {'tokenizer_json': tok.encode(s, add_special_tokens=False).ids,
              'sp_model_plain': sp.encode(s)}
try:
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(REL)
    for s in lits:
        beh[s]['transformers'] = hf(s, add_special_tokens=False)['input_ids']
        try:
            beh[s]['transformers_split_special_tokens'] = hf(s, add_special_tokens=False, split_special_tokens=True)['input_ids']
        except Exception as ex:
            beh[s]['transformers_split_special_tokens'] = 'error: %s' % ex
    res['transformers'] = {'class': type(hf).__name__, 'len': len(hf), 'n_special_ids': len(hf.all_special_ids)}
except Exception as ex:
    res['transformers'] = 'not run: %s' % ex
res['literal_special_strings'] = beh
json.dump(res, open(os.path.join(OUT, 'specials.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
