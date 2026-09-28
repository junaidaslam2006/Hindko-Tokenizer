"""Shared feature code for the web language classifier (see langid_train.py)."""
import pickle
import re
import sys

from scipy.sparse import hstack

sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import lang  # noqa: E402

AR_WORD_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')
MODEL = r'F:\Hindko\_web\langid\model.pkl'


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
        self.classes = list(self.clf.classes_)

    def proba(self, texts):
        """texts are raw strings; returns list of {label: p}."""
        n = [norm(t) for t in texts]
        X = hstack([self.wv.transform(n), self.cv.transform(n)]).tocsr()
        P = self.clf.predict_proba(X)
        return [dict(zip(self.classes, map(float, row))) for row in P]
