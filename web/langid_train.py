"""Train a Hindko-vs-neighbours language classifier for filtering web text.

Classes: hindko (our released strict corpus), punjabi (FineWeb-2 pnb_Arab),
urdu (FineWeb-2 urd_Arab), saraiki (skr_Arab), kashmiri (kas_Arab).
Held-out evaluation is by GROUP (book folder / newspaper issue / web domain)
so that topic and source leakage do not inflate the scores.

Features: whole-token unigrams + bigrams restricted to tokens seen in >= MIN_GROUPS
distinct groups (drops names and topic words that live in one book or site),
plus character 2-4 grams inside words (orthography: نڑ, ہک, ٻ ...).

    python langid_train.py      -> F:/Hindko/_web/langid/model.pkl + report
"""
import collections
import json
import os
import pickle
import random
import re
import sys
from urllib.parse import urlparse

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, r'F:\Hindko\_pipeline')
sys.path.insert(0, r'F:\Hindko\_web')
from hp.langid import MODEL, WordFeats, norm  # noqa: E402

REF = r'F:\Hindko\_web\ref'
OUT = r'F:\Hindko\_web\langid'
CORPUS = r'F:\Hindko\hindko_dataset.jsonl'
PERMISSIVE = r'F:\Hindko\hindko_dataset_permissive.jsonl'
MIN_GROUPS = 4
CHUNK = 1200          # train on ~paragraph-sized chunks, the unit we will classify on the web
random.seed(13)
# names of the language and its cities: an Urdu article ABOUT Hindko must not
# look like Hindko because it says so
TOPIC_STOP = frozenset({'ہندکو', 'پشور', 'پشاور', 'ہزارہ', 'ایبٹ', 'آباد', 'مانسہرہ', 'کوہاٹ', 'ڈیسک', 'ویب'})
NEWS_GROUPS = set()
# FineWeb-2 reference sets and the class they stand for. Persian/Arabic/Pashto/
# Sindhi and the small Arabic-script languages are there so that text the
# classifier has never seen is not forced into 'hindko' (the first FineWeb-2
# scan scored Iranian .ir pages as Hindko before these were added).
REF_SETS = [('pnb_Arab', 'punjabi'), ('urd_Arab', 'urdu'), ('skr_Arab', 'saraiki'), ('kas_Arab', 'kashmiri'),
            ('fas_Arab', 'persian'), ('arb_Arab', 'arabic'), ('arz_Arab', 'arabic'), ('pbt_Arab', 'pashto'),
            ('snd_Arab', 'sindhi'), ('ckb_Arab', 'other'), ('azb_Arab', 'other'), ('brh_Arab', 'other'),
            ('uig_Arab', 'other'), ('glk_Arab', 'other'), ('sdh_Arab', 'other'), ('hac_Arab', 'other'),
            ('bal_Arab', 'other')]
# Hindko sites found inside pnb_Arab/skr_Arab by the survey (_web/survey/corpora.md)
HINDKO_DOMAINS = frozenset({'gandharahindko.com', 'hindko.org', 'aaprihindko.com', 'hindkomaza.home.blog',
                            'noukeqalam.com', 'iattock.com', 'tvshia.com', 'mansehra.com', 'hazar-e-wall.com'})


def chunks(text, size=CHUNK):
    """Split into ~size-char pieces on line boundaries."""
    buf, n = [], 0
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        n += len(line)
        if n >= size:
            yield ' '.join(buf)
            buf, n = [], 0
    if n >= 200:
        yield ' '.join(buf)


def load():
    data = []   # (label, group, text)
    for l in open(CORPUS, encoding='utf-8'):
        d = json.loads(l)
        if d.get('language_variety') != 'hindko':
            continue
        g = d.get('book_folder') or d.get('issue') or d.get('source_file')
        if d.get('source') == 'newspaper':
            NEWS_GROUPS.add('h:' + str(g))
        data.append(('hindko', 'h:' + str(g), d['text']))
    for l in open(PERMISSIVE, encoding='utf-8'):
        d = json.loads(l)
        if d.get('language_variety') == 'pothohari':
            data.append(('pothohari', 'b:' + str(d.get('book_folder')), d['text']))
    for fn, label in REF_SETS:
        path = os.path.join(REF, fn + '.jsonl')
        if not os.path.exists(path):
            print('missing reference set', fn)
            continue
        for l in open(path, encoding='utf-8'):
            d = json.loads(l)
            host = urlparse(d.get('url') or '').netloc or 'unknown'
            if host.replace('www.', '') in HINDKO_DOMAINS:
                continue            # GlotLID filed these Hindko sites under pnb/skr: label noise
            data.append((label, 'w:' + host, d['text']))
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = load()
    groups = collections.defaultdict(set)
    for lab, g, _ in docs:
        groups[lab].add(g)
    held = set()
    for lab, gs in groups.items():
        gs = sorted(gs)
        random.shuffle(gs)
        held.update(gs[:max(1, len(gs) // 5)])
    tr, te = [], []
    cap = {'hindko': 60000, 'urdu': 20000, 'punjabi': 30000, 'persian': 8000, 'arabic': 8000,
           'pashto': 8000, 'sindhi': 8000, 'other': 8000}
    per = collections.Counter()
    for lab, g, t in docs:
        for c in chunks(t):
            c = norm(c)
            if len(c) < 150:
                continue
            if g in held:
                te.append((lab, g, c))
            elif lab != 'pothohari':          # pothohari: evaluation only (125 records)
                if per[lab] < cap.get(lab, 10 ** 9):
                    tr.append((lab, g, c))
                    per[lab] += 1
    print('train chunks', collections.Counter(x[0] for x in tr))
    print('test chunks ', collections.Counter(x[0] for x in te))

    # vocabulary: tokens occurring in >= MIN_GROUPS distinct groups
    # all newspaper issues count as ONE group here, so newspaper furniture
    # (web desk, bylines) cannot enter the vocabulary on issue count alone
    tok_groups = collections.defaultdict(set)
    for lab, g, c in tr:
        vg = 'h:newspaper' if g.startswith('h:') and not g.startswith('h:book') and g in NEWS_GROUPS else g
        for w in set(c.split()):
            tok_groups[w].add(vg)
    vocab_ok = {w for w, gs in tok_groups.items() if len(gs) >= MIN_GROUPS} - TOPIC_STOP
    print('vocab kept', len(vocab_ok), 'of', len(tok_groups))

    wv = TfidfVectorizer(analyzer=WordFeats(vocab_ok), min_df=3, sublinear_tf=True, max_features=200000)
    cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, sublinear_tf=True,
                         max_features=300000)
    Xtr = hstack([wv.fit_transform([c for _, _, c in tr]), cv.fit_transform([c for _, _, c in tr])]).tocsr()
    ytr = [l for l, _, _ in tr]
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight='balanced')
    clf.fit(Xtr, ytr)
    Xte = hstack([wv.transform([c for _, _, c in te]), cv.transform([c for _, _, c in te])]).tocsr()
    yte = [l for l, _, _ in te]
    pred = clf.predict(Xte)
    rep = classification_report(yte, pred, digits=4, zero_division=0)
    labels = sorted(set(yte) | set(pred))
    cm = confusion_matrix(yte, pred, labels=labels)
    print(rep)
    print(labels)
    print(cm)
    # top word features per class
    names = wv.get_feature_names_out()
    tops = {}
    for i, lab in enumerate(clf.classes_):
        coef = clf.coef_[i][:len(names)]
        tops[lab] = [names[j] for j in np.argsort(-coef)[:40]]
        print(lab, ' '.join(tops[lab]))
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    with open(MODEL, 'wb') as f:
        pickle.dump({'wv': wv, 'cv': cv, 'clf': clf, 'vocab_ok': vocab_ok}, f)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(rep + '\n' + str(labels) + '\n' + str(cm) + '\n\n')
        for lab, t in tops.items():
            f.write(lab + ': ' + ' '.join(t) + '\n')


if __name__ == '__main__':
    main()
