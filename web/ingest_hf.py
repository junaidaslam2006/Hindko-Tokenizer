"""Pull the openly licensed Hindko transcript/sentence sets into raw_files/web.

1. Meta Omnilingual ASR corpus, config hno_Arab (CC BY 4.0): the raw_text of
   every segment in train/dev/test, fetched from the HF datasets-server rows
   API (text columns only, no audio). One document per segment.
2. Mozilla Common Voice 26, locale hno (CC0-1.0): the unique validated
   sentences, read by the survey from the text columns of the CV26 parquet on
   the HF mirror (range requests, no audio). Sentences are grouped in their
   original order into documents of ~300 words, since single sentences are too
   short to stand as corpus records.

Output: F:/Hindko/raw_files/web/<source_id>/docs.jsonl with the common web
schema: doc_id, source_id, url, title, text, license, attribution, meta.
"""
import json
import os
import re
import time

import requests

ROOT = r'F:\Hindko\raw_files\web'
SURVEY = r'F:\Hindko\_web\survey\_samples'
TAG_RE = re.compile(r'<[^<>]{0,40}>')
SPACES_RE = re.compile(r'[ \t]{2,}')


def write(source_id, docs):
    d = os.path.join(ROOT, source_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'docs.jsonl'), 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(source_id, len(docs), 'docs', sum(len(x['text'].split()) for x in docs), 'words')


def omnilingual():
    ds = 'facebook/omnilingual-asr-corpus'
    docs = []
    for split in ('train', 'dev', 'test'):
        off = 0
        while True:
            for attempt in range(6):
                r = requests.get('https://datasets-server.huggingface.co/rows',
                                 params={'dataset': ds, 'config': 'hno_Arab', 'split': split,
                                         'offset': off, 'length': 100}, timeout=120)
                if r.status_code == 200:
                    break
                time.sleep(10 * (attempt + 1))
            r.raise_for_status()
            j = r.json()
            for row in j['rows']:
                x = row['row']
                raw = x['raw_text'] or ''
                text = SPACES_RE.sub(' ', TAG_RE.sub(' ', raw)).strip()
                if not text or set(text) <= set('? .'):
                    continue
                docs.append({
                    'doc_id': 'omni_hno_%s_%s_%s_%s' % (split, x['speaker_id'], x['prompt_id'], x['segment_id']),
                    'source_id': 'omnilingual_asr_hno',
                    'url': 'https://huggingface.co/datasets/%s' % ds,
                    'title': None,
                    'text': text,
                    'license': 'CC-BY-4.0',
                    'attribution': 'Meta FAIR, Omnilingual ASR corpus (2025), config hno_Arab',
                    'register': 'transcribed_spontaneous_speech',
                    'meta': {'split': split, 'speaker_id': x['speaker_id'], 'prompt_id': x['prompt_id'],
                             'prompt_en': x['prompt'], 'segment_id': x['segment_id'],
                             'duration_s': float(x['duration']), 'glottocode': x.get('glottocode'),
                             'tags_removed': len(TAG_RE.findall(raw))},
                })
            off += len(j['rows'])
            if off >= j['num_rows_total'] or not j['rows']:
                break
    write('omnilingual_asr_hno', docs)


def common_voice(words_per_doc=300):
    seen, sents = set(), []
    for l in open(os.path.join(SURVEY, 'cv26_hno_validated_sentences.jsonl'), encoding='utf-8'):
        x = json.loads(l)
        if x['sentence_id'] in seen:
            continue
        seen.add(x['sentence_id'])
        sents.append(x)
    docs, buf, n = [], [], 0
    for x in sents + [None]:
        if x is not None:
            buf.append(x)
            n += len(x['sentence'].split())
        if buf and (n >= words_per_doc or x is None):
            docs.append({
                'doc_id': 'cv26_hno_%03d' % len(docs),
                'source_id': 'common_voice_hno',
                'url': 'https://commonvoice.mozilla.org/hno',
                'title': None,
                'text': '\n'.join(s['sentence'].strip() for s in buf),
                'license': 'CC0-1.0',
                'attribution': 'Mozilla Common Voice Scripted Speech 26.0, locale hno (validated sentences)',
                'register': 'crowdsourced_sentences',
                'meta': {'n_sentences': len(buf), 'sentence_ids': [s['sentence_id'] for s in buf]},
            })
            buf, n = [], 0
    write('common_voice_hno', docs)


if __name__ == '__main__':
    omnilingual()
    common_voice()
