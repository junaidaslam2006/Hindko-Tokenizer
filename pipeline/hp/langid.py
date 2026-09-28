"""Hindko-vs-neighbours language classifier for web text.

Classes: hindko, punjabi, urdu, saraiki, kashmiri. Trained by
_web/langid_train.py on the released strict Hindko corpus against FineWeb-2
pnb_Arab / urd_Arab / skr_Arab / kas_Arab samples, evaluated on held-out
groups (book folder, newspaper issue, web domain). Word uni+bigrams over a
vocabulary seen in >= 4 groups (names and topic words excluded) plus
character 2-4 grams. It does not know Pothohari/Pahari (those score as
Hindko), which is why web sources keep a per-site dialect note.
"""
import os
import pickle
import re

from scipy.sparse import hstack

from . import lang

AR_WORD_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')
MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'langid_hno_v1.pkl')


def norm(text):
    """Arabic-script words only, harakat stripped."""
    return ' '.join(AR_WORD_RE.findall(lang.strip_marks(text)))


class WordFeats:
    """Unigrams + bigrams over a fixed vocabulary (picklable analyzer)."""

    def __init__(self, vocab):
        self.vocab = frozenset(vocab)

    def __call__(self, s):
        w = [t for t in s.split() if t in self.vocab]
        return w + [a + ' ' + b for a, b in zip(w, w[1:])]


class Model:
    def __init__(self, path=MODEL):
        with open(path, 'rb') as f:
            m = pickle.load(f)
        self.wv, self.cv, self.clf = m['wv'], m['cv'], m['clf']
        self.classes = [str(c) for c in self.clf.classes_]

    def proba(self, texts):
        """texts are raw strings; returns list of {label: p}."""
        n = [norm(t) for t in texts]
        X = hstack([self.wv.transform(n), self.cv.transform(n)]).tocsr()
        P = self.clf.predict_proba(X)
        return [dict(zip(self.classes, map(float, row))) for row in P]


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = Model()
    return _MODEL
