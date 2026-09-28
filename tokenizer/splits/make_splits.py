#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_splits.py - leak-free, group-disjoint train / validation / test splits
for the Hindko corpus (tokenizer + language-model evaluation).

Run (from anywhere):
    set PYTHONIOENCODING=utf-8
    python F:\\Hindko\\_tokenizer\\splits\\make_splits.py

Reads   F:\\Hindko\\hindko_dataset_permissive.jsonl  (superset, 18,283 records)
        F:\\Hindko\\hindko_dataset.jsonl             (strict subset; only verified)
Writes  split_manifest.jsonl, splits_report.json     (next to this file, nothing else)

Deterministic: fixed seed, sorted iteration, seeded numpy Generators, no
salted Python hash(). Re-running on the same inputs gives byte-identical
outputs (the report's timing block aside).

Method in one paragraph
-----------------------
Records are grouped (newspaper edition, book folder, Omnilingual speaker,
Common Voice record, web site / web page); groups of one source that share
>= 25 % of the smaller group's word 8-grams are then merged, so the same text
filed twice is one unit. Every record is compared with the
REST OF THE CORPUS OUTSIDE ITS OWN GROUP: word-8-gram containment (share of
its distinct 8-grams found in any record of another group) and the best exact
Jaccard of 5-character shingles (MinHash all-pairs for candidates, then exact
verification). Both are computed on a deliberately aggressive "leak
normalisation" (NFKD, all marks/harakat removed, Arabic/Urdu letter variants
folded, Arabic-Indic digits -> ASCII, tatweel/ZWNJ/format chars removed,
punctuation dropped, Latin casefolded) so the result stays leak-free whatever
digit/presentation-form policy the normalisation stage later chooses. A record
reaching containment >= 0.30 or Jaccard >= 0.50 against anything outside its
group is "leak-prone" (set P, closed under same-group moves). Whole groups are
then chosen for validation/test per source with a seeded multi-restart local
search on the words that would remain after P is moved to train; a group whose
P share is >= 50 % never enters evaluation. Finally the actual assignment is
verified against the actual train set and any leak is moved to train and the
whole procedure repeats until zero evaluation records exceed a threshold.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))          # F:\Hindko
PIPELINE = os.path.join(ROOT, '_pipeline')
PERMISSIVE_PATH = os.path.join(ROOT, 'hindko_dataset_permissive.jsonl')
STRICT_PATH = os.path.join(ROOT, 'hindko_dataset.jsonl')
MANIFEST_PATH = os.path.join(HERE, 'split_manifest.jsonl')
REPORT_PATH = os.path.join(HERE, 'splits_report.json')

SPLITS = ('train', 'validation', 'test')
EVAL = ('validation', 'test')
SPLIT_CODE = {'train': 0, 'validation': 1, 'test': 2}
SOURCES = ('newspaper', 'book', 'web')

CONFIG = dict(
    seed=20260926,
    fractions={'validation': 0.05, 'test': 0.05},
    tolerance_points=1.5,               # required: |share - 5 %| <= 1.5 points per source
    ngram=8,                            # word n-gram for containment
    containment_threshold=0.30,         # leaked if >= this
    jaccard_threshold=0.50,             # leaked if >= this
    shingle_chars=5,                    # same shingle width as hp.clean.Deduper
    num_perm=128,
    minhash_candidate_floor=0.25,       # pairs with estimate >= this are verified exactly
    max_group_share_of_target=0.50,     # a group may fill at most half of one eval split (per source)
    group_max_leak_share=0.50,          # a group with >= 50 % leak-prone words never enters eval
    restarts=200,                       # seeded restarts of the group-selection search per source
    # selection cost per evaluation split = (share error/0.001)^2 + (strict share error/0.002)^2
    # + sum over composition attributes (L2 distance of the split's category mix from the
    # source's mix / 0.05)^2 + (max(0, Herfindahl index of group sizes - 0.12)/0.03)^2
    # (+ a steep penalty beyond the 1.5-point tolerance)
    cost_scales=dict(share=0.001, strict_share=0.002, composition=0.05,
                     concentration_free=0.12, concentration=0.03),
    # composition attributes matched per source (categories -> see record_strata())
    composition={'newspaper': ['variety', 'tree'], 'book': ['variety', 'verse'],
                 'web': ['variety', 'register']},
    # near-duplicate groups: two groups of the same source that share >= 25 % of the
    # smaller group's distinct word 8-grams are merged into one group (union-find).
    # 8-grams found in more than 10 groups are formulae (datelines, honorifics,
    # recurring headers) and are not counted as evidence of a shared document.
    group_merge_overlap=0.25,
    group_merge_max_groups_per_gram=10,
    threads=3,
    max_resolution_iterations=10,
)

# ---------------------------------------------------------------------------
# leak normalisation (used ONLY for matching; never for output text)
# ---------------------------------------------------------------------------
FOLD = {
    0x064A: 0x06CC,   # ARABIC YEH            -> FARSI/URDU YEH
    0x0649: 0x06CC,   # ALEF MAKSURA          -> FARSI/URDU YEH
    0x0643: 0x06A9,   # ARABIC KAF            -> KEHEH
    0x0647: 0x06C1,   # ARABIC HEH            -> HEH GOAL
    0x0629: 0x06C1,   # TEH MARBUTA           -> HEH GOAL
    0x06C3: 0x06C1,   # TEH MARBUTA GOAL      -> HEH GOAL
    0x06D5: 0x06C1,   # AE                    -> HEH GOAL
    0x0640: None,     # TATWEEL (kashida)
}
for _i in range(10):
    FOLD[0x0660 + _i] = 0x30 + _i   # Arabic-Indic digits
    FOLD[0x06F0 + _i] = 0x30 + _i   # Extended Arabic-Indic (Urdu) digits


class _FoldTable(dict):
    """str.translate table built lazily: folds above, and deletes every
    combining mark (Mn/Me: harakat, hamza/madda above after NFKD, ...) and
    format character (Cf: ZWNJ, ZWJ, bidi marks, ...)."""

    def __missing__(self, cp):
        if cp in FOLD:
            v = FOLD[cp]
        else:
            v = None if unicodedata.category(chr(cp)) in ('Mn', 'Me', 'Cf') else cp
        self[cp] = v
        return v


_TABLE = _FoldTable()
TOKEN_RE = re.compile(r'[^\W_]+')     # runs of letters/digits; punctuation dropped


def leak_tokens(text: str):
    t = unicodedata.normalize('NFKD', text).translate(_TABLE).casefold()
    return TOKEN_RE.findall(t)


# ---------------------------------------------------------------------------
# deterministic 64-bit hashing helpers (numpy, wrap-around arithmetic)
# ---------------------------------------------------------------------------
U64 = np.uint64
_MULT = U64(0x9E3779B97F4A7C15)
_M1 = U64(0xBF58476D1CE4E5B9)
_M2 = U64(0x94D049BB133111EB)


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised."""
    x = x.astype(np.uint64, copy=True)
    x ^= x >> U64(30)
    x *= _M1
    x ^= x >> U64(27)
    x *= _M2
    x ^= x >> U64(31)
    return x


_LEN_SALT = mix64(np.arange(1, 65, dtype=np.uint64) * U64(0xD6E8FEB86659FD93))


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Hashes of every contiguous n-gram of `ids` (uint64, 1-based ids)."""
    L = len(ids) - n + 1
    if L <= 0:
        return np.empty(0, np.uint64)
    h = np.zeros(L, np.uint64)
    for j in range(n):
        h = h * _MULT + ids[j:j + L]
    return mix64(h ^ _LEN_SALT[n - 1])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def log(*a):
    print('[%s]' % time.strftime('%H:%M:%S'), *a, flush=True)


def peak_memory_mb():
    """Peak working set of this process (Windows), else ru_maxrss; None if unknown."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
    except Exception:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. load
# ---------------------------------------------------------------------------
def load_records():
    recs = []
    with open(PERMISSIVE_PATH, encoding='utf-8') as f:
        for line in f:
            recs.append(json.loads(line))
    uids = [r['uid'] for r in recs]
    assert len(set(uids)) == len(uids), 'duplicate uid in permissive dataset'
    # the strict file must be exactly the permissive records tagged strict
    perm = {r['uid']: r for r in recs}
    n_strict = 0
    with open(STRICT_PATH, encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            n_strict += 1
            p = perm.get(s['uid'])
            assert p is not None, 'strict uid %s not in permissive' % s['uid']
            assert p['quality_tier'] == 'strict' and p['text'] == s['text'], s['uid']
    assert n_strict == sum(1 for r in recs if r['quality_tier'] == 'strict')
    return recs, n_strict


# ---------------------------------------------------------------------------
# 2. groups
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def site_or_folder(r):
    if r['source'] == 'newspaper':
        return os.path.dirname(r['source_path'])
    if r['source'] == 'book':
        return r['book_folder']
    return r['site']


def assign_groups(recs, cfg):
    """Returns (group key per record, grouping audit dict)."""
    frac_v = cfg['fractions']['validation']
    words_by_source = collections.Counter()
    site_words = collections.Counter()
    for r in recs:
        words_by_source[r['source']] += r['n_words']
        if r['source'] == 'web':
            site_words[r['site']] += r['n_words']

    # --- newspaper: edition = issue number, else source_path folder; editions
    # that share a day-precision publication date are merged (the numbered
    # tree 2023-2024-2025/<issue>/ and the dated tree 2024 to 2026/<month>/<d.m.y>/
    # hold the same editions, e.g. issue 321 and folder 8.5.2024 both carry
    # 2024-05-08). Merging only makes groups coarser, never finer.
    uf = UnionFind()
    base = {}
    for r in recs:
        if r['source'] != 'newspaper':
            continue
        if r.get('issue'):
            b = 'news:issue:%s' % r['issue']
        else:
            b = 'news:dir:%s' % os.path.dirname(r['source_path'])
        base[r['uid']] = b
        uf.find(b)
        if r.get('date_precision') == 'day' and r.get('date'):
            uf.union(b, 'date:%s' % r['date'])
    comp_members = collections.defaultdict(set)
    for b in set(base.values()):
        comp_members[uf.find(b)].add(b)
    comp_name = {}
    for root, members in comp_members.items():
        issues = sorted(m for m in members if m.startswith('news:issue:'))
        name = issues[0] if issues else sorted(members)[0]
        comp_name[root] = name
    merges = {comp_name[root]: sorted(m) for root, m in comp_members.items() if len(m) > 1}

    # --- web: which ordinary sites are "too big to move whole"
    per_split_web_target = frac_v * words_by_source['web']
    big_site_limit = cfg['max_group_share_of_target'] * per_split_web_target
    big_sites = sorted(s for s, w in site_words.items()
                       if s not in ('omnilingual_asr_hno', 'common_voice_hno')
                       and w > big_site_limit)

    keys = []
    for r in recs:
        src = r['source']
        if src == 'newspaper':
            k = comp_name[uf.find(base[r['uid']])]
        elif src == 'book':
            k = 'book:%s' % r['book_folder']
        elif src == 'web':
            site = r['site']
            if site == 'omnilingual_asr_hno':
                k = 'web:omnilingual_asr_hno:speaker:%s' % r['web_meta']['speaker_id']
            elif site == 'common_voice_hno':
                k = 'web:common_voice_hno:record:%s' % r['uid']
            elif site in big_sites:
                k = 'web:%s:record:%s' % (site, r['uid'])
            else:
                k = 'web:%s' % site
        else:
            raise ValueError('unknown source %r' % src)
        keys.append(k)

    audit = dict(
        newspaper_edition_merges=merges,
        newspaper_base_groups=len(set(base.values())),
        newspaper_groups_after_date_merge=len({keys[i] for i, r in enumerate(recs)
                                          if r['source'] == 'newspaper'}),
        web_site_words=dict(sorted(site_words.items(), key=lambda x: -x[1])),
        web_big_site_limit_words=round(big_site_limit, 1),
        web_sites_split_by_uid=big_sites,
    )
    return keys, audit


def merge_overlapping_groups(keys, recs, H, R, OWN, cfg):
    """Merge same-source groups whose shared distinct word 8-grams reach
    `group_merge_overlap` of the smaller group's distinct 8-grams (8-grams in
    more than `group_merge_max_groups_per_gram` groups are ignored). Such
    groups hold substantially the same text (the same edition filed in both
    newspaper trees, one dictionary copied into another, reposted web pages)
    and must be held out together or not at all. Returns (new keys, audit)."""
    import itertools
    names = sorted(set(keys))
    gi = {g: i for i, g in enumerate(names)}
    gid = np.array([gi[k] for k in keys], np.int32)
    src_of = {}
    for k, r in zip(keys, recs):
        src_of[k] = r['source']
    Hh, G = H[OWN], gid[R[OWN]]
    order = np.lexsort((G, Hh))
    Hs, Gs = Hh[order], G[order]
    keep = np.ones(len(Hs), bool)
    keep[1:] = (Hs[1:] != Hs[:-1]) | (Gs[1:] != Gs[:-1])
    Hs, Gs = Hs[keep], Gs[keep]
    gpg = np.bincount(Gs, minlength=len(names))
    new_h = np.ones(len(Hs), bool)
    new_h[1:] = Hs[1:] != Hs[:-1]
    st = np.nonzero(new_h)[0]
    en = np.append(st[1:], len(Hs))
    ng = en - st
    sel = (ng >= 2) & (ng <= cfg['group_merge_max_groups_per_gram'])
    S = collections.Counter()
    for s, e in zip(st[sel].tolist(), en[sel].tolist()):
        for a, b in itertools.combinations(Gs[s:e].tolist(), 2):
            S[(a, b)] += 1
    uf = UnionFind()
    edges = []
    for (a, b), c in sorted(S.items()):
        ga, gb = names[a], names[b]
        if src_of[ga] != src_of[gb]:
            continue
        ov = c / max(1, min(gpg[a], gpg[b]))
        if ov >= cfg['group_merge_overlap']:
            uf.union(ga, gb)
            edges.append(dict(a=ga, b=gb, shared_8grams=int(c), overlap=round(float(ov), 4)))
    comps = collections.defaultdict(list)
    for g in names:
        comps[uf.find(g)].append(g)
    rename = {}
    merged = {}
    for members in comps.values():
        if len(members) == 1:
            continue
        members = sorted(members)
        issues = [m for m in members if m.startswith('news:issue:')]
        rep = (issues or members)[0]
        new = '%s [+%d]' % (rep, len(members) - 1)
        merged[new] = members
        for m in members:
            rename[m] = new
    audit = dict(rule='same-source groups sharing >= %.2f of the smaller group\'s distinct word 8-grams '
                      '(8-grams in <= %d groups) are one group'
                      % (cfg['group_merge_overlap'], cfg['group_merge_max_groups_per_gram']),
                 edges=sorted(edges, key=lambda e: -e['overlap']),
                 merged_groups=merged,
                 merged_by_source=dict(collections.Counter(src_of[v[0]] for v in merged.values())))
    return [rename.get(k, k) for k in keys], audit


# ---------------------------------------------------------------------------
# 3. features: word n-grams, char shingles, MinHash
# ---------------------------------------------------------------------------
def build_features(recs, cfg):
    n_gram = cfg['ngram']
    k_sh = cfg['shingle_chars']
    vocab = {}
    tok_ids = []
    shingles = []
    for r in recs:
        toks = leak_tokens(r['text'])
        ids = np.fromiter((vocab.setdefault(t, len(vocab) + 1) for t in toks),
                          dtype=np.uint64, count=len(toks))
        tok_ids.append(ids)
        joined = ''.join(toks)
        cps = np.frombuffer(joined.encode('utf-32-le'), dtype=np.uint32).astype(np.uint64) + U64(1)
        if len(cps) >= k_sh:
            sh = np.unique(ngram_hashes(cps, k_sh))
        elif len(cps):
            sh = ngram_hashes(cps, len(cps))
        else:
            sh = np.empty(0, np.uint64)
        shingles.append(sh)
    grams = []
    short = collections.defaultdict(list)   # length -> record indices with < n tokens
    for i, ids in enumerate(tok_ids):
        if len(ids) >= n_gram:
            grams.append(np.unique(ngram_hashes(ids, n_gram)))
        elif len(ids):
            grams.append(ngram_hashes(ids, len(ids)))     # the whole record as one gram
            short[len(ids)].append(i)
        else:
            grams.append(np.empty(0, np.uint64))
    return tok_ids, grams, shingles, short, len(vocab)


def gram_occurrences(tok_ids, grams, short):
    """Flat (hash, record, own) occurrence arrays. A record with fewer than n
    tokens contributes its whole token sequence as one gram; every other
    record that contains that sequence gets a SYNTHETIC occurrence of it
    (own=False): it makes 'record j contains this gram' true for membership
    tests, but never counts in record j's own containment numerator or
    denominator."""
    H = [g for g in grams]
    R = [np.full(len(g), i, np.int32) for i, g in enumerate(grams)]
    O = [np.ones(len(g), bool) for g in grams]
    extra_pairs = 0
    for k, idxs in sorted(short.items()):
        want = np.unique(np.concatenate([grams[i] for i in idxs]))
        for j, ids in enumerate(tok_ids):
            if len(ids) < k:
                continue
            hk = ngram_hashes(ids, k)
            j_hits = np.unique(hk[np.isin(hk, want)])
            if len(ids) == k:
                # a k-token record already owns its whole-sequence gram
                j_hits = j_hits[~np.isin(j_hits, grams[j])]
            if len(j_hits):
                H.append(j_hits)
                R.append(np.full(len(j_hits), j, np.int32))
                O.append(np.zeros(len(j_hits), bool))
                extra_pairs += len(j_hits)
    return np.concatenate(H), np.concatenate(R), np.concatenate(O), extra_pairs


def minhash_signatures(shingles, cfg):
    rng = np.random.default_rng([cfg['seed'], 1])
    A = rng.integers(0, np.iinfo(np.uint64).max, cfg['num_perm'], dtype=np.uint64,
                     endpoint=True) | U64(1)
    B = rng.integers(0, np.iinfo(np.uint64).max, cfg['num_perm'], dtype=np.uint64,
                     endpoint=True)
    sigs = np.full((len(shingles), cfg['num_perm']), 0xFFFFFFFF, np.uint32)
    empty = np.zeros(len(shingles), bool)
    for i, x in enumerate(shingles):
        if not len(x):
            empty[i] = True
            continue
        best = np.full(cfg['num_perm'], np.iinfo(np.uint64).max, np.uint64)
        for c0 in range(0, len(x), 16384):
            xc = x[c0:c0 + 16384]
            v = (A[:, None] * xc[None, :] + B[:, None]) >> U64(32)
            best = np.minimum(best, v.min(axis=1))
        sigs[i] = best.astype(np.uint32)
    return sigs, empty


def minhash_candidates(sigs, empty, cfg):
    """All-pairs MinHash estimates (blocked, upper triangle), keeping pairs
    whose estimate >= floor. Brute force, so no LSH recall loss."""
    N, P = sigs.shape
    ST = np.ascontiguousarray(sigs.T)
    floor_cnt = int(np.ceil(cfg['minhash_candidate_floor'] * P))
    blk = 96

    def block(i0):
        i1 = min(i0 + blk, N)
        cnt = np.zeros((i1 - i0, N - i0), np.uint8)
        for k in range(P):
            row = ST[k]
            cnt += (row[i0:i1, None] == row[None, i0:])
        ii, jj = np.nonzero(cnt >= floor_cnt)
        vals = cnt[ii, jj]
        ii = ii + i0
        jj = jj + i0
        keep = jj > ii
        return ii[keep].astype(np.int32), jj[keep].astype(np.int32), vals[keep]

    with ThreadPoolExecutor(max_workers=cfg['threads']) as ex:
        parts = list(ex.map(block, range(0, N, blk)))
    I = np.concatenate([p[0] for p in parts])
    J = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    ok = ~(empty[I] | empty[J])
    return I[ok], J[ok], C[ok].astype(np.float64) / P


def exact_jaccard(shingles, I, J):
    out = np.empty(len(I), np.float64)
    for n, (i, j) in enumerate(zip(I.tolist(), J.tolist())):
        a, b = shingles[i], shingles[j]
        inter = np.intersect1d(a, b, assume_unique=True).size
        union = len(a) + len(b) - inter
        out[n] = inter / union if union else 0.0
    return out


# ---------------------------------------------------------------------------
# 4. leakage measurement
# ---------------------------------------------------------------------------
class LeakIndex:
    def __init__(self, H, R, OWN, gid, n_rec, pairs_i, pairs_j, pairs_jac):
        self.H, self.R, self.OWN = H, R, OWN
        self.gid = gid
        self.n_rec = n_rec
        self.G = gid[R]
        self.n_grams = np.bincount(R[OWN], minlength=n_rec).astype(np.float64)
        order = np.lexsort((self.G, H))
        Hs, Gs = H[order], self.G[order]
        new_h = np.ones(len(Hs), bool)
        new_h[1:] = Hs[1:] != Hs[:-1]
        new_pair = new_h.copy()
        new_pair[1:] |= Gs[1:] != Gs[:-1]
        hid = np.cumsum(new_h) - 1
        n_groups = np.bincount(hid, weights=new_pair)
        ext_sorted = n_groups[hid] >= 2
        self.external = np.empty(len(H), bool)
        self.external[order] = ext_sorted            # gram also occurs in ANOTHER group
        self.gkey = mix64(H ^ mix64((self.G.astype(np.uint64) + U64(1)) * _MULT))
        self.pi, self.pj, self.pjac = pairs_i, pairs_j, pairs_jac
        self.same_group_pair = gid[pairs_i] == gid[pairs_j]
        self.sorted_order = np.argsort(H, kind='stable')
        self.H_sorted = H[self.sorted_order]

    def _per_record(self, occ_mask):
        occ_mask = occ_mask & self.OWN
        num = np.bincount(self.R, weights=occ_mask.astype(np.float64), minlength=self.n_rec)
        with np.errstate(invalid='ignore', divide='ignore'):
            c = np.where(self.n_grams > 0, num / np.maximum(self.n_grams, 1), 0.0)
        return c

    def worst_case(self, inP):
        """Containment/Jaccard of every record against everything outside its
        own group plus the leak-prone records of its own group."""
        occP = inP[self.R]
        keyP = np.unique(self.gkey[occP])
        own = np.isin(self.gkey, keyP) & ~occP
        cont = self._per_record(self.external | own)
        jac = np.zeros(self.n_rec)
        for a, b in ((self.pi, self.pj), (self.pj, self.pi)):
            # for record a, partner b counts if it is in another group, or in
            # the same group but itself leak-prone (it will sit in train)
            ok = ~self.same_group_pair | inP[b]
            np.maximum.at(jac, a[ok], self.pjac[ok])
        return cont, jac

    def actual(self, split_arr, eval_codes=(1, 2), ref_code=0):
        """Containment of each record whose split is in eval_codes, against the
        union of records whose split == ref_code; plus the best exact Jaccard
        (among verified MinHash candidates) against those records."""
        ref_occ = split_arr[self.R] == ref_code
        ref_h = np.unique(self.H[ref_occ])
        in_ref = np.isin(self.H, ref_h, assume_unique=False)
        tgt_occ = np.isin(split_arr[self.R], eval_codes)
        cont = self._per_record(in_ref & tgt_occ)
        jac = np.zeros(self.n_rec)
        partner = np.full(self.n_rec, -1, np.int64)
        for a, b in ((self.pi, self.pj), (self.pj, self.pi)):
            ok = np.isin(split_arr[a], eval_codes) & (split_arr[b] == ref_code)
            aa, bb, ss = a[ok], b[ok], self.pjac[ok]
            order = np.lexsort((-bb, ss))    # ascending by sim; ties -> smaller partner index last
            for x, y, s in zip(aa[order].tolist(), bb[order].tolist(), ss[order].tolist()):
                if s >= jac[x]:
                    jac[x] = s
                    partner[x] = y
        return cont, jac, partner

    def shared_counts(self, r, pool_mask):
        """Counter: record q in pool_mask -> number of record r's own grams q contains."""
        sel = (self.R == r) & self.OWN
        hs = np.unique(self.H[sel])
        lo = np.searchsorted(self.H_sorted, hs, 'left')
        hi = np.searchsorted(self.H_sorted, hs, 'right')
        cnt = collections.Counter()
        for a, b in zip(lo.tolist(), hi.tolist()):
            for o in self.sorted_order[a:b].tolist():
                q = int(self.R[o])
                if q != r and pool_mask[q]:
                    cnt[q] += 1
        return cnt, len(hs)

    def top_container(self, r, pool_mask):
        """The single record in pool_mask sharing the most grams with record r."""
        cnt, n = self.shared_counts(r, pool_mask)
        if not cnt:
            return None, 0.0
        q, c = min(cnt.items(), key=lambda x: (-x[1], x[0]))
        return q, c / max(n, 1)


def leak_prone_fixpoint(idx: LeakIndex, cfg, seed_P):
    inP = seed_P.copy()
    rounds = 0
    while True:
        rounds += 1
        cont, jac = idx.worst_case(inP)
        newP = inP | (cont >= cfg['containment_threshold']) | (jac >= cfg['jaccard_threshold'])
        if (newP == inP).all():
            return inP, cont, jac, rounds
        inP = newP


# ---------------------------------------------------------------------------
# 5. group selection (per source, seeded multi-restart local search)
# ---------------------------------------------------------------------------
STRATA = {
    'variety': ('hindko', 'urdu', 'other'),
    'tree': ('numbered_issue', 'dated_folder'),
    'verse': ('verse', 'non_verse'),
    'book_form': ('prose', 'verse', 'lexicon', 'list', 'mixed'),   # reported only
    'register': ('spontaneous_speech', 'read_speech', 'web_text'),
}


REPORT_ONLY_ATTRS = {'book': ['book_form']}   # reported, not optimised


def record_strata(r):
    """Category of a record under every composition attribute (None = n/a)."""
    v = r['language_variety']
    out = {'variety': v if v in ('hindko', 'urdu') else 'other', 'tree': None, 'verse': None, 'book_form': None,
           'register': None}
    if r['source'] == 'newspaper':
        out['tree'] = 'numbered_issue' if r.get('issue') else 'dated_folder'
    elif r['source'] == 'book':
        out['verse'] = 'verse' if r.get('form') == 'verse' else 'non_verse'
        out['book_form'] = r.get('form') if r.get('form') in STRATA['book_form'] else 'prose'
    elif r['source'] == 'web':
        out['register'] = {'omnilingual_asr_hno': 'spontaneous_speech',
                           'common_voice_hno': 'read_speech'}.get(r['site'], 'web_text')
    return out


def select_source(src_i, src, names, ua, us, comps, W, Ws, cfg):
    """comps: list of (M, ref) - M is (n_groups, K) usable words per category
    of one composition attribute, ref the (K,) word distribution of the
    whole source over those categories."""
    fv = cfg['fractions']['validation']
    ft = cfg['fractions']['test']
    fr = {1: fv, 2: ft}
    sc = cfg['cost_scales']
    lim = cfg['tolerance_points'] / 100.0
    n = len(names)

    # per-group contribution vector: words, strict words, words^2, then the
    # per-category words of every composition attribute
    cols = [ua[:, None], us[:, None], (ua * ua)[:, None]]
    spans = []
    off = 3
    for M, ref in comps:
        cols.append(M)
        spans.append((off, off + M.shape[1], np.asarray(ref, float)))
        off += M.shape[1]
    V = np.concatenate(cols, axis=1)

    def split_cost(T, f):
        """T[..., D] = summed contribution vector of one evaluation split."""
        A, S, Q = T[..., 0], T[..., 1], T[..., 2]
        e1 = A / W - f
        c = (e1 / sc['share']) ** 2 + 1e4 * np.maximum(0, np.abs(e1) - lim) ** 2 / lim ** 2
        if Ws > 0:
            e2 = S / Ws - f
            c = c + (e2 / sc['strict_share']) ** 2 + 1e4 * np.maximum(0, np.abs(e2) - lim) ** 2 / lim ** 2
        Asafe = np.maximum(A, 1e-9)[..., None]
        for a, b, ref in spans:
            dist = np.where(Asafe > 1e-9, T[..., a:b] / Asafe, 0.0)
            l2 = np.sqrt(((dist - ref) ** 2).sum(axis=-1))
            c = c + (l2 / sc['composition']) ** 2
        hhi = np.where(A > 0, Q / Asafe[..., 0] ** 2, 1.0)     # Herfindahl index of group sizes
        return c + (np.maximum(0.0, hhi - sc['concentration_free']) / sc['concentration']) ** 2

    def sums_of(assign):
        return {x: V[assign == x].sum(axis=0) for x in (1, 2)}

    def total(sums):
        return float(split_cost(sums[1], fr[1]) + split_cost(sums[2], fr[2]))

    def local_search(assign):
        sums = sums_of(assign)
        cur = total(sums)
        for _ in range(10000):
            best = (cur - 1e-12, None)
            # single moves: group i from its split to dest
            for dest in (0, 1, 2):
                cand = np.nonzero(assign != dest)[0]
                if not len(cand):
                    continue
                src_split = assign[cand]
                tot = np.zeros(len(cand))
                for x in (1, 2):
                    d = np.where((src_split == x)[:, None], -V[cand], 0.0)
                    if dest == x:
                        d = d + V[cand]
                    tot = tot + split_cost(sums[x][None, :] + d, fr[x])
                k = int(np.argmin(tot))
                if tot[k] < best[0]:
                    best = (float(tot[k]), ('move', int(cand[k]), dest))
            # swaps: i in eval split x  <->  j in split y (train or the other eval split)
            for x in (1, 2):
                I = np.nonzero(assign == x)[0]
                if not len(I):
                    continue
                for y in (0, 3 - x):
                    Jx = np.nonzero(assign == y)[0]
                    if not len(Jx):
                        continue
                    d = V[Jx][None, :, :] - V[I][:, None, :]          # (|I|, |J|, 4)
                    tot = split_cost(sums[x][None, None, :] + d, fr[x])
                    if y == 0:
                        tot = tot + split_cost(sums[3 - x], fr[3 - x])
                    else:
                        tot = tot + split_cost(sums[y][None, None, :] - d, fr[y])
                    k = int(np.argmin(tot))
                    a, b = divmod(k, len(Jx))
                    if tot.flat[k] < best[0]:
                        best = (float(tot.flat[k]), ('swap', int(I[a]), int(Jx[b])))
            if best[1] is None:
                break
            kind, p, q = best[1]
            if kind == 'move':
                assign[p] = q
            else:
                assign[p], assign[q] = assign[q], assign[p]
            sums = sums_of(assign)
            cur = total(sums)
        return cur

    target = {1: fv * W, 2: ft * W}
    best = None
    for trial in range(cfg['restarts']):
        rng = np.random.default_rng([cfg['seed'], 7, src_i, trial])
        order = rng.permutation(n)
        assign = np.zeros(n, np.int8)
        filled = {1: 0.0, 2: 0.0}
        for i in order.tolist():
            for x in (1, 2):
                if filled[x] + ua[i] <= target[x]:
                    assign[i] = x
                    filled[x] += ua[i]
                    break
        c = local_search(assign)
        if best is None or c < best[0] - 1e-12:
            best = (c, assign.copy(), trial)
    c, assign, trial = best
    chosen = {names[i]: SPLITS[int(assign[i])] for i in range(n) if assign[i] != 0}
    return chosen, dict(cost=round(c, 6), best_trial=trial, n_eligible_groups=n)


# ---------------------------------------------------------------------------
# 6. statistics helpers
# ---------------------------------------------------------------------------
def quantiles(x):
    x = np.asarray(x, float)
    if not len(x):
        return {}
    qs = [0.5, 0.9, 0.95, 0.99, 1.0]
    return {('max' if q == 1.0 else 'p%d' % int(q * 100)): round(float(np.quantile(x, q)), 4)
            for q in qs} | {'mean': round(float(x.mean()), 4), 'n': int(len(x))}


def histogram(x, edges):
    x = np.asarray(x, float)
    out = {'== 0': int((x == 0).sum())}
    lo = 0.0
    for e in edges:
        out['(%g, %g]' % (lo, e)] = int(((x > lo) & (x <= e)).sum())
        lo = e
    out['> %g' % lo] = int((x > lo).sum())
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    cfg = CONFIG
    t0 = time.time()
    timing = {}
    log('loading')
    recs, n_strict = load_records()
    N = len(recs)
    uids = [r['uid'] for r in recs]
    src = np.array([SOURCES.index(r['source']) for r in recs], np.int8)
    words = np.array([r['n_words'] for r in recs], np.float64)
    strict = np.array([r['quality_tier'] == 'strict' for r in recs])
    variety = [r['language_variety'] for r in recs]
    timing['load_s'] = round(time.time() - t0, 1)

    log('grouping (rules)')
    keys, group_audit = assign_groups(recs, cfg)

    log('features (leak-normalised tokens, 8-grams, 5-char shingles)')
    t1 = time.time()
    tok_ids, grams, shingles, short, vocab_size = build_features(recs, cfg)
    H, R, OWN, extra_occ = gram_occurrences(tok_ids, grams, short)
    timing['features_s'] = round(time.time() - t1, 1)
    log('  %d gram occurrences, %d records shorter than %d tokens, vocab %d'
        % (len(H), sum(len(v) for v in short.values()), cfg['ngram'], vocab_size))

    log('grouping (merge near-duplicate groups)')
    n_rule_groups = len(set(keys))
    keys, merge_audit = merge_overlapping_groups(keys, recs, H, R, OWN, cfg)
    group_audit['rule_groups'] = n_rule_groups
    group_audit['near_duplicate_group_merges'] = merge_audit
    gnames = sorted(set(keys))
    gindex = {g: i for i, g in enumerate(gnames)}
    gid = np.array([gindex[k] for k in keys], np.int32)
    group_source = {}
    for k, r in zip(keys, recs):
        assert group_source.setdefault(k, r['source']) == r['source'], k
    log('  %d rule groups -> %d groups' % (n_rule_groups, len(gnames)))

    log('minhash')
    t1 = time.time()
    sigs, empty = minhash_signatures(shingles, cfg)
    timing['minhash_s'] = round(time.time() - t1, 1)
    log('all-pairs minhash candidates')
    t1 = time.time()
    PI, PJ, PEST = minhash_candidates(sigs, empty, cfg)
    timing['allpairs_s'] = round(time.time() - t1, 1)
    log('  %d candidate pairs (estimate >= %.2f); exact Jaccard' % (len(PI), cfg['minhash_candidate_floor']))
    t1 = time.time()
    PJAC = exact_jaccard(shingles, PI, PJ)
    timing['exact_jaccard_s'] = round(time.time() - t1, 1)

    idx = LeakIndex(H, R, OWN, gid, N, PI, PJ, PJAC)

    # ---- per-group totals
    n_groups = len(gnames)
    g_words = np.bincount(gid, weights=words, minlength=n_groups)
    src_words = {s: float(words[src == i].sum()) for i, s in enumerate(SOURCES)}
    src_strict_words = {s: float(words[(src == i) & strict].sum()) for i, s in enumerate(SOURCES)}
    # composition attributes: per-record category index, per-source reference mix
    strata = [record_strata(r) for r in recs]
    cat_idx = {a: np.array([STRATA[a].index(st[a]) if st[a] is not None else -1 for st in strata])
               for a in STRATA}
    comp_ref = {}
    for i, s in enumerate(SOURCES):
        for a in cfg['composition'][s] + REPORT_ONLY_ATTRS.get(s, []):
            m = src == i
            w = np.array([words[m & (cat_idx[a] == k)].sum() for k in range(len(STRATA[a]))])
            comp_ref[(s, a)] = w / w.sum()

    extra_P = np.zeros(N, bool)
    iterations = []
    for it in range(1, cfg['max_resolution_iterations'] + 1):
        log('resolution iteration %d: leak-prone fixpoint' % it)
        inP, wc_cont, wc_jac, fp_rounds = leak_prone_fixpoint(idx, cfg, extra_P)
        usable = ~inP
        gu = np.bincount(gid, weights=words * usable, minlength=n_groups)
        gus = np.bincount(gid, weights=words * usable * strict, minlength=n_groups)
        gcat = {a: np.stack([np.bincount(gid, weights=words * usable * (cat_idx[a] == k), minlength=n_groups)
                             for k in range(len(STRATA[a]))], axis=1) for a in STRATA}
        leak_share = np.where(g_words > 0, 1 - gu / np.maximum(g_words, 1), 0)
        ginfo = {}
        for g in gnames:
            i = gindex[g]
            s = group_source[g]
            cap = cfg['max_group_share_of_target'] * cfg['fractions']['validation'] * src_words[s]
            reason = None
            if leak_share[i] >= cfg['group_max_leak_share']:
                reason = 'leak_share>=%.2f' % cfg['group_max_leak_share']
            elif gu[i] > cap:
                reason = 'usable_words>cap(%.0f)' % cap
            elif gu[i] <= 0:
                reason = 'no_usable_words'
            ginfo[g] = dict(source=s, words=float(g_words[i]), usable_words=float(gu[i]),
                            usable_strict_words=float(gus[i]),
                            leak_share=float(leak_share[i]), eligible=reason is None,
                            ineligible_reason=reason)
        log('  P = %d records (%d words); selecting groups' % (inP.sum(), words[inP].sum()))
        chosen = {}
        sel_meta = {}
        for si, s in enumerate(SOURCES):
            names = [g for g in gnames if ginfo[g]['source'] == s and ginfo[g]['eligible']]
            ua = np.array([ginfo[g]['usable_words'] for g in names])
            us = np.array([ginfo[g]['usable_strict_words'] for g in names])
            rows = np.array([gindex[g] for g in names], np.int64)
            comps = [(gcat[a][rows], comp_ref[(s, a)]) for a in cfg['composition'][s]]
            ch, meta = select_source(si, s, names, ua, us, comps, src_words[s],
                                     src_strict_words[s], cfg)
            chosen.update(ch)
            sel_meta[s] = meta
        group_split = np.zeros(n_groups, np.int8)
        for g, sp in chosen.items():
            group_split[gindex[g]] = SPLIT_CODE[sp]
        naive = group_split[gid].copy()                  # whole groups, before resolution
        split_arr = naive.copy()
        split_arr[inP & (naive != 0)] = 0                # leak-prone eval records -> train
        log('  verifying against the actual train set')
        cont, jac, partner = idx.actual(split_arr)
        is_eval = split_arr != 0
        leaked = is_eval & ((cont >= cfg['containment_threshold']) | (jac >= cfg['jaccard_threshold']))
        iterations.append(dict(iteration=it, fixpoint_rounds=fp_rounds,
                               leak_prone_records=int(inP.sum()),
                               leak_prone_words=int(words[inP].sum()),
                               eval_records=int(is_eval.sum()),
                               leaks_after_resolution=int(leaked.sum()),
                               selection=sel_meta))
        log('  leaks after resolution: %d' % leaked.sum())
        if not leaked.any():
            break
        extra_P |= leaked
    else:
        raise SystemExit('leak resolution did not converge')

    # ------------------------------------------------------------------
    # measurements for the report
    # ------------------------------------------------------------------
    log('measuring')
    thr_c, thr_j = cfg['containment_threshold'], cfg['jaccard_threshold']
    # BEFORE: whole selected groups, no record moved
    b_cont, b_jac, b_partner = idx.actual(naive)
    b_eval = naive != 0
    b_leak = b_eval & ((b_cont >= thr_c) | (b_jac >= thr_j))

    def leak_summary(mask_eval, c, j, splitv):
        out = {}
        for sp in EVAL:
            m = mask_eval & (splitv == SPLIT_CODE[sp])
            lc = m & (c >= thr_c)
            lj = m & (j >= thr_j)
            la = lc | lj
            out[sp] = dict(eval_records=int(m.sum()), leaked_records=int(la.sum()),
                           leaked_words=int(words[la].sum()),
                           by_containment=int(lc.sum()), by_jaccard=int(lj.sum()),
                           by_both=int((lc & lj).sum()),
                           leaked_by_source={s: int((la & (src == i)).sum()) for i, s in enumerate(SOURCES)})
        return out

    before = leak_summary(b_eval, b_cont, b_jac, naive)
    after = leak_summary(is_eval, cont, jac, split_arr)

    # For comparison only: the LITERAL minimal procedure on the same group
    # selection - start from whole groups, move only records that actually
    # leak into the current train set, re-measure, repeat until zero. The
    # published split uses the stricter leak-prone rule instead (see RULES).
    lit = naive.copy()
    literal_trace = []
    for rnd in range(1, 51):
        lc_, lj_, _ = idx.actual(lit)
        lk = (lit != 0) & ((lc_ >= thr_c) | (lj_ >= thr_j))
        literal_trace.append(dict(round=rnd, leaked_records=int(lk.sum()), leaked_words=int(words[lk].sum())))
        if not lk.any():
            break
        lit[lk] = 0
    literal = dict(
        rounds=literal_trace,
        records_moved=int(((naive != 0) & (lit == 0)).sum()),
        words_moved=int(words[(naive != 0) & (lit == 0)].sum()),
        published_rule_moved_records=int(((naive != 0) & (split_arr == 0)).sum()),
        published_rule_moved_words=int(words[(naive != 0) & (split_arr == 0)].sum()),
        extra_records_moved_by_published_rule=int(((naive != 0) & (split_arr == 0) & (lit != 0)).sum()),
        note='the published rule also removes evaluation records that duplicate text in the OTHER '
             'evaluation split or in other groups of their own split, which the literal procedure keeps')

    # within-eval: validation vs test (both directions)
    vt = {}
    for a_sp, b_sp in (('validation', 'test'), ('test', 'validation')):
        c2, j2, p2 = idx.actual(split_arr, eval_codes=(SPLIT_CODE[a_sp],), ref_code=SPLIT_CODE[b_sp])
        m = split_arr == SPLIT_CODE[a_sp]
        vt['%s_vs_%s' % (a_sp, b_sp)] = dict(
            records=int(m.sum()),
            over_threshold=int((m & ((c2 >= thr_c) | (j2 >= thr_j))).sum()),
            containment=quantiles(c2[m]), containment_hist=histogram(c2[m], [0.05, 0.1, 0.2, 0.3]),
            best_jaccard=quantiles(j2[m]),
            records_with_any_shared_8gram=int((m & (c2 > 0)).sum()))

    # cross-check 1: hp.clean.Deduper's own normalisation + shingles, for every
    # eval record against all its MinHash-candidate partners in train
    sys.path.insert(0, PIPELINE)
    from hp import clean as hpclean   # noqa: E402
    ded_best = np.zeros(N)
    cand = collections.defaultdict(set)
    for a, b in zip(PI.tolist(), PJ.tolist()):
        if split_arr[a] != 0 and split_arr[b] == 0:
            cand[a].add(b)
        if split_arr[b] != 0 and split_arr[a] == 0:
            cand[b].add(a)
    sh_cache = {}

    def dsh(i):
        if i not in sh_cache:
            sh_cache[i] = hpclean._shingles(hpclean._dedup_norm(recs[i]['text']))
        return sh_cache[i]
    for a in sorted(cand):
        sa = dsh(a)
        for b in sorted(cand[a]):
            sb = dsh(b)
            u = len(sa | sb)
            ded_best[a] = max(ded_best[a], len(sa & sb) / u if u else 0.0)

    # cross-check 2: raw whitespace tokens (no normalisation) 8-gram containment
    raw_vocab = {}
    raw_H, raw_R = [], []
    for i, r in enumerate(recs):
        toks = r['text'].split()
        ids = np.fromiter((raw_vocab.setdefault(t, len(raw_vocab) + 1) for t in toks),
                          dtype=np.uint64, count=len(toks))
        g = np.unique(ngram_hashes(ids, cfg['ngram'])) if len(ids) >= cfg['ngram'] else ngram_hashes(ids, len(ids))
        raw_H.append(g)
        raw_R.append(np.full(len(g), i, np.int32))
    raw_H = np.concatenate(raw_H)
    raw_R = np.concatenate(raw_R)
    tr_h = np.unique(raw_H[split_arr[raw_R] == 0])
    occ = (split_arr[raw_R] != 0) & np.isin(raw_H, tr_h)
    raw_num = np.bincount(raw_R, weights=occ.astype(float), minlength=N)
    raw_den = np.bincount(raw_R, minlength=N)
    raw_cont = np.where(raw_den > 0, raw_num / np.maximum(raw_den, 1), 0)

    # cross-check 3: an independent candidate generator for Jaccard - every
    # train record sharing >= 1 word 8-gram with an evaluation record is
    # compared by EXACT shingle Jaccard (no MinHash involved)
    train_mask = split_arr == 0
    ng_pairs = 0
    ng_best = np.zeros(N)
    for r_ in np.nonzero(split_arr != 0)[0].tolist():
        cnt, _ = idx.shared_counts(r_, train_mask)
        for q in sorted(cnt):
            a, b = shingles[r_], shingles[q]
            inter = np.intersect1d(a, b, assume_unique=True).size
            u = len(a) + len(b) - inter
            ng_best[r_] = max(ng_best[r_], inter / u if u else 0.0)
            ng_pairs += 1

    # MinHash estimator quality on the verified candidate pairs
    mh_err = PEST - PJAC
    minhash_quality = dict(
        pairs=int(len(PI)), mean_error=round(float(mh_err.mean()), 4) if len(PI) else None,
        sd_error=round(float(mh_err.std()), 4) if len(PI) else None,
        max_abs_error=round(float(np.abs(mh_err).max()), 4) if len(PI) else None,
        theoretical_sd_at_0_5=round(float(np.sqrt(0.25 / cfg['num_perm'])), 4),
        min_estimate_among_pairs_with_exact_ge_threshold=(
            round(float(PEST[PJAC >= thr_j].min()), 4) if (PJAC >= thr_j).any() else None))

    # moved records (eval-group records sent to train), with their best match
    moved = np.nonzero((naive != 0) & (split_arr == 0))[0]
    moved_list = []
    for r_ in moved.tolist():
        pool = gid != gid[r_]
        pool |= inP & (np.arange(N) != r_)
        q, qc = idx.top_container(r_, pool)
        moved_list.append(dict(uid=uids[r_], group=keys[r_], source=recs[r_]['source'],
                               would_have_been=SPLITS[int(naive[r_])], n_words=int(words[r_]),
                               worst_case_containment=round(float(wc_cont[r_]), 4),
                               worst_case_jaccard=round(float(wc_jac[r_]), 4),
                               top_containing_record=(uids[q] if q is not None else None),
                               top_containing_group=(keys[q] if q is not None else None),
                               top_container_gram_share=round(qc, 4)))

    # ------------------------------------------------------------------
    # shares, variety mix, groups
    # ------------------------------------------------------------------
    def share_table():
        out = {}
        for si, s in enumerate(SOURCES):
            m_s = src == si
            row = {}
            for sp in SPLITS:
                m = m_s & (split_arr == SPLIT_CODE[sp])
                row[sp] = dict(
                    words=int(words[m].sum()),
                    share_pct=round(100 * words[m].sum() / words[m_s].sum(), 3),
                    records=int(m.sum()),
                    strict_words=int(words[m & strict].sum()),
                    strict_share_pct=round(100 * words[m & strict].sum() / max(words[m_s & strict].sum(), 1), 3),
                    strict_records=int((m & strict).sum()),
                    permissive_only_words=int(words[m & ~strict].sum()),
                    permissive_only_share_pct=round(100 * words[m & ~strict].sum() / max(words[m_s & ~strict].sum(), 1), 3),
                )
            out[s] = row
        tot = {}
        for sp in SPLITS:
            m = split_arr == SPLIT_CODE[sp]
            tot[sp] = dict(words=int(words[m].sum()), share_pct=round(100 * words[m].sum() / words.sum(), 3),
                           records=int(m.sum()), strict_words=int(words[m & strict].sum()),
                           strict_share_pct=round(100 * words[m & strict].sum() / words[strict].sum(), 3),
                           strict_records=int((m & strict).sum()))
        out['ALL'] = tot
        return out

    shares = share_table()
    tol_ok = all(abs(shares[s][sp]['share_pct'] - 100 * cfg['fractions'][sp]) <= cfg['tolerance_points']
                 for s in SOURCES for sp in EVAL)

    def variety_mix():
        out = {}
        for sp in SPLITS:
            m = split_arr == SPLIT_CODE[sp]
            per = {}
            for scope, mm in [('ALL', m)] + [(s, m & (src == i)) for i, s in enumerate(SOURCES)]:
                for tier, tm in (('permissive', mm), ('strict', mm & strict)):
                    tw = words[tm].sum()
                    c = collections.Counter()
                    cr = collections.Counter()
                    for k in np.nonzero(tm)[0].tolist():
                        c[variety[k]] += words[k]
                        cr[variety[k]] += 1
                    per.setdefault(scope, {})[tier] = {
                        v: dict(words=int(c[v]), words_pct=round(100 * c[v] / tw, 2) if tw else 0.0,
                                records=int(cr[v])) for v in sorted(c, key=lambda v: -c[v])}
            out[sp] = per
        return out

    groups_by_split = {sp: [] for sp in SPLITS}
    partial = {}
    for g in gnames:
        i = gindex[g]
        m = gid == i
        sp = SPLITS[int(group_split[i])]
        members_split = collections.Counter(SPLITS[int(x)] for x in split_arr[m])
        groups_by_split[sp].append(dict(group=g, source=group_source[g], records=int(m.sum()),
                                        words=int(words[m].sum()),
                                        records_in_split=int(members_split[sp])))
        if sp != 'train' and members_split['train']:
            partial[g] = dict(split=sp, records_moved_to_train=int(members_split['train']),
                              words_moved_to_train=int(words[m & (split_arr == 0)].sum()))

    # omnilingual upstream split cross-tab
    omni = collections.defaultdict(lambda: collections.Counter())
    for i, r in enumerate(recs):
        if r['source'] == 'web' and r['site'] == 'omnilingual_asr_hno':
            omni[(r['web_meta']['speaker_id'], r['web_meta']['split'])][SPLITS[int(split_arr[i])]] += int(words[i])
    omni_tab = [dict(speaker=spk, upstream_split=up, words_by_our_split=dict(v))
                for (spk, up), v in sorted(omni.items())]

    eval_mask = split_arr != 0
    final_dist = dict(
        containment_in_train=quantiles(cont[eval_mask]),
        containment_hist=histogram(cont[eval_mask], [0.05, 0.1, 0.2, 0.3]),
        best_exact_jaccard_in_train=quantiles(jac[eval_mask]),
        jaccard_hist=histogram(jac[eval_mask], [0.1, 0.2, 0.3, 0.4, 0.5]),
        note_jaccard=('best exact Jaccard among train records whose MinHash estimate >= %.2f; '
                      'records with no such candidate are reported as 0 (true best < ~0.25)'
                      % cfg['minhash_candidate_floor']),
        records_with_any_8gram_in_train=int((eval_mask & (cont > 0)).sum()),
        per_split={sp: dict(containment=quantiles(cont[split_arr == SPLIT_CODE[sp]]),
                            best_jaccard=quantiles(jac[split_arr == SPLIT_CODE[sp]]))
                   for sp in EVAL},
        crosscheck_deduper_normalisation=dict(
            method='hp.clean._dedup_norm + hp.clean._shingles (5-char, whitespace-stripped) exact '
                   'Jaccard vs every MinHash-candidate train partner',
            max=round(float(ded_best[eval_mask].max()), 4) if eval_mask.any() else None,
            over_threshold=int((eval_mask & (ded_best >= thr_j)).sum())),
        crosscheck_raw_whitespace_tokens=dict(
            method='8-gram containment on raw whitespace tokens (no normalisation)',
            quantiles=quantiles(raw_cont[eval_mask]),
            over_threshold=int((eval_mask & (raw_cont >= thr_c)).sum())),
        crosscheck_ngram_candidates_exact_jaccard=dict(
            method='exact 5-char-shingle Jaccard of every evaluation record vs EVERY train record that '
                   'shares at least one word 8-gram with it (candidate generation independent of MinHash)',
            pairs_checked=int(ng_pairs),
            max=round(float(ng_best[eval_mask].max()), 4) if eval_mask.any() else None,
            over_threshold=int((eval_mask & (ng_best >= thr_j)).sum())),
        minhash_estimator_quality=minhash_quality,
        worst_case_vs_entire_corpus_outside_own_group=dict(
            note='every evaluation record against ALL other groups (train, the other evaluation split, '
                 'and other groups of its own split) plus the moved records of its own group',
            containment=quantiles(wc_cont[eval_mask]),
            containment_hist=histogram(wc_cont[eval_mask], [0.05, 0.1, 0.2, 0.3]),
            best_jaccard=quantiles(wc_jac[eval_mask]),
            over_threshold=int((eval_mask & ((wc_cont >= thr_c) | (wc_jac >= thr_j))).sum())),
    )

    # how concentrated each evaluation split is (a split dominated by one
    # group is a poor estimate of the source)
    concentration = {}
    for sp in EVAL:
        for si, s in enumerate(SOURCES):
            m = (split_arr == SPLIT_CODE[sp]) & (src == si)
            gw = collections.Counter()
            for k in np.nonzero(m)[0].tolist():
                gw[keys[k]] += words[k]
            tot_w = sum(gw.values())
            top_g, top_w = max(gw.items(), key=lambda x: (x[1], x[0])) if gw else (None, 0)
            concentration.setdefault(sp, {})[s] = dict(
                n_groups=len(gw), largest_group=top_g,
                largest_group_share_pct=round(100 * top_w / tot_w, 2) if tot_w else 0.0,
                herfindahl=round(sum((w / tot_w) ** 2 for w in gw.values()), 4) if tot_w else 0.0)

    # composition of each split vs the whole source (word shares, %)
    composition = {}
    for si, s in enumerate(SOURCES):
        for a in cfg['composition'][s] + REPORT_ONLY_ATTRS.get(s, []):
            row = {'source_reference': {c: round(100 * float(x), 2) for c, x in zip(STRATA[a], comp_ref[(s, a)])}}
            for sp in SPLITS:
                m = (split_arr == SPLIT_CODE[sp]) & (src == si)
                tw = words[m].sum()
                row[sp] = {c: round(100 * words[m & (cat_idx[a] == k)].sum() / tw, 2) if tw else 0.0
                           for k, c in enumerate(STRATA[a])}
            composition.setdefault(s, {})[a] = row

    # corpus-wide worst-case (independent of the split): how much of the corpus
    # duplicates text outside its own group
    wc_by_source = {s: dict(records=int((inP & (src == i)).sum()),
                            words=int(words[inP & (src == i)].sum()),
                            share_of_source_words_pct=round(100 * words[inP & (src == i)].sum() / src_words[s], 2))
                    for i, s in enumerate(SOURCES)}

    # ------------------------------------------------------------------
    # write manifest
    # ------------------------------------------------------------------
    log('writing manifest')
    with open(MANIFEST_PATH, 'w', encoding='utf-8', newline='\n') as f:
        for i, r in enumerate(recs):
            up = None
            if r['source'] == 'web' and r['site'] == 'omnilingual_asr_hno':
                up = r['web_meta'].get('split')
            row = dict(uid=r['uid'], split=SPLITS[int(split_arr[i])], group=keys[i],
                       source=r['source'], site_or_folder=site_or_folder(r),
                       quality_tier=r['quality_tier'], language_variety=r['language_variety'],
                       n_words=int(r['n_words']),
                       assignment=('group' if split_arr[i] == naive[i] else 'leak_moved_to_train'),
                       upstream_split=up)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    timing['total_s'] = round(time.time() - t0, 1)
    timing['peak_working_set_mb'] = peak_memory_mb()
    report = dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        seed=cfg['seed'],
        config=cfg,
        inputs={os.path.basename(PERMISSIVE_PATH): dict(sha256=sha256_file(PERMISSIVE_PATH), records=N),
                os.path.basename(STRICT_PATH): dict(sha256=sha256_file(STRICT_PATH), records=n_strict,
                                                    verified='identical subset of permissive (uid, text, tier)')},
        code=dict(make_splits_sha256=sha256_file(os.path.abspath(__file__)),
                  python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform()),
        outputs=dict(manifest=os.path.basename(MANIFEST_PATH), manifest_sha256=sha256_file(MANIFEST_PATH)),
        rules=RULES,
        grouping=group_audit | dict(n_groups=n_groups,
                                    groups_by_source={s: sum(1 for g in gnames if group_source[g] == s) for s in SOURCES}),
        leak_normalisation=LEAK_NORM_DOC,
        features=dict(vocab_size=vocab_size, gram_occurrences=int(len(H)),
                      synthetic_short_record_occurrences=int(extra_occ),
                      short_records_by_token_count={int(k): len(v) for k, v in sorted(short.items())},
                      empty_shingle_records=int(empty.sum()),
                      minhash_candidate_pairs=int(len(PI)),
                      candidate_pairs_exact_jaccard_ge_threshold=int((PJAC >= thr_j).sum()),
                      candidate_pairs_cross_group_ge_threshold=int(((PJAC >= thr_j) & (gid[PI] != gid[PJ])).sum())),
        corpus_leak_prone=dict(
            definition='records whose containment >= %.2f or Jaccard >= %.2f against the corpus outside '
                       'their own group (plus leak-prone records of their own group); independent of the split'
                       % (thr_c, thr_j),
            by_source=wc_by_source),
        iterations=iterations,
        targets_pct={sp: 100 * cfg['fractions'][sp] for sp in EVAL},
        shares=shares,
        tolerance_met=tol_ok,
        concentration=concentration,
        composition=composition,
        language_variety_mix=variety_mix(),
        leakage=dict(
            before_resolution=before,
            after_resolution=after,
            records_moved_to_train=len(moved_list),
            words_moved_to_train=int(words[moved].sum()),
            literal_iterative_procedure_for_comparison=literal,
            moved_records=moved_list,
            final_distribution=final_dist,
            validation_vs_test=vt,
        ),
        groups=dict(
            by_split={sp: sorted(v, key=lambda x: x['group']) for sp, v in groups_by_split.items()},
            counts={sp: {s: sum(1 for x in groups_by_split[sp] if x['source'] == s) for s in SOURCES}
                    for sp in SPLITS},
            eval_groups_with_records_moved_to_train=partial,
            ineligible_for_eval={g: v['ineligible_reason'] for g, v in sorted(ginfo.items())
                                 if not v['eligible'] and not g.startswith('web:common_voice_hno:record:')
                                 and ':record:' not in g},
            ineligible_counts=collections.Counter(v['ineligible_reason'].split('(')[0]
                                                  for v in ginfo.values() if not v['eligible']),
        ),
        omnilingual_upstream_split=dict(
            note=('Omnilingual ASR ships its own speaker-disjoint train/dev/test. Its dev+test speakers '
                  'hold %d words (%.1f%% of web words), more than the whole 5%%+5%% web evaluation budget, '
                  'so the upstream split cannot be honoured within the +/-1.5 point target. Speakers are kept '
                  'whole (group = speaker) and assigned like any other group. If you will evaluate on the '
                  'Omnilingual ASR benchmark, drop upstream dev/test speakers from training with '
                  'load_split.iter_split(..., drop_upstream_heldout=True).'
                  % (sum(sum(v.values()) for (spk, up), v in omni.items() if up in ('dev', 'test')),
                     100 * sum(sum(v.values()) for (spk, up), v in omni.items() if up in ('dev', 'test')) / src_words['web'])),
            table=omni_tab),
        timing=timing,
    )
    with open(REPORT_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=_json_default)
    log('done in %.1f s; tolerance met: %s; leaks after: %s'
        % (time.time() - t0, tol_ok, {sp: after[sp]['leaked_records'] for sp in EVAL}))
    for s in SOURCES + ('ALL',):
        log('  %-9s' % s, ' '.join('%s %.2f%% (strict %.2f%%)' % (sp, shares[s][sp]['share_pct'],
                                                               shares[s][sp]['strict_share_pct']) for sp in SPLITS))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


LEAK_NORM_DOC = dict(
    purpose='matching only (never written out); aggressive so that the split is leak-free under any later '
            'canonical form (digits, presentation forms, letter variants are still undecided)',
    steps=['Unicode NFKD (decomposes presentation forms, ligatures such as U+FDF2, hamza/madda carriers)',
           'delete every combining mark (Mn/Me: harakat, superscript alef, hamza/madda above) and every '
           'format character (Cf: ZWNJ, ZWJ, bidi marks)',
           'delete tatweel U+0640',
           'fold ARABIC YEH U+064A and ALEF MAKSURA U+0649 -> U+06CC; ARABIC KAF U+0643 -> U+06A9; '
           'ARABIC HEH U+0647, TEH MARBUTA U+0629, U+06C3, AE U+06D5 -> HEH GOAL U+06C1',
           'Arabic-Indic U+0660-9 and Urdu U+06F0-9 digits -> ASCII 0-9',
           'casefold (Latin)',
           'tokens = maximal runs of letters/digits (regex [^\\W_]+); punctuation dropped'],
    word_ngrams='distinct 8-grams of tokens; a record with k < 8 tokens is one k-gram (whole record) and '
                'counts as contained if that exact token sequence occurs inside another record',
    char_shingles='distinct 5-character shingles of the concatenated tokens (no spaces/punctuation), '
                  'the same shingle width as hp.clean.Deduper',
    hashing='64-bit polynomial hash of token ids / code points + splitmix64 finaliser (numpy, deterministic)',
    minhash='128 permutations h(x) = (a*x + b mod 2^64) >> 32, a odd, seeded; ALL pairs compared '
            '(no LSH banding, so no recall loss); pairs with estimate >= 0.25 verified with exact Jaccard '
            '(an estimate below 0.25 is > 5 standard deviations under 0.50 at 128 permutations)',
)

RULES = dict(
    unit='whole groups are assigned to a split; the strict tier inherits each record\'s split',
    groups={
        'newspaper': 'edition: issue number, else the source_path folder; editions that share a '
                     'day-precision date are merged (union-find), because the numbered tree and the dated '
                     'tree contain the same editions (evidence: issue 321 and folder 8.5.2024 both carry '
                     '2024-05-08). Coarser groups only; a mis-parsed D.M/M.D date can merge two editions, '
                     'which is conservative',
        'book': 'book_folder',
        'web/omnilingual_asr_hno': 'web_meta.speaker_id',
        'web/common_voice_hno': 'the record itself',
        'web/other sites': 'the whole site; a site larger than half of one evaluation split\'s web target '
                           '(0.5 x 5% of web words) is split per record (uid). Justification: a group larger '
                           'than that cap is never eligible for evaluation (it would dominate the split and '
                           'make the +/-1.5 point target unreachable), so without splitting, the biggest web '
                           'sites (noukeqalam alone is ~34% of web words) could never be evaluated; site '
                           'pages are independent posts, and any cross-page copying is caught by the same '
                           'record-level leak check that protects every other boundary (and pages that '
                           'share a quarter of their text are re-merged by the near-duplicate rule below)',
        'near-duplicate groups (all sources)': 'after the rules above, two groups of the same source are '
                           'merged when they share >= 25% of the smaller group\'s distinct word 8-grams '
                           '(8-grams occurring in more than 10 groups are formulae and not counted). This '
                           'catches the same edition filed in both newspaper trees without a shared date '
                           '(issue 307 / folder 17.1.24 share 35%), a dictionary copied into another, '
                           'anthologies reprinting each other and reposted web pages. Merging only makes '
                           'groups coarser. A merged group is named after its first member plus "[+k]"; '
                           'the report lists all members. Cross-source copies (newspaper serials in books) '
                           'are NOT merged, to keep per-source stratification; the leak rule handles them',
    },
    targets='per source (newspaper, book, web): 5% of the source\'s words to validation and 5% to test, '
            'tolerance +/-1.5 points; the search also targets 5% of the source\'s STRICT words and keeps the '
            'category mix of each evaluation split close to the whole source for: language variety '
            '(hindko/urdu/other; all sources), newspaper tree (numbered issue vs dated folder), book verse '
            'share (verse vs everything else; lexicon-heavy groups are mostly too big to hold out), web register (spontaneous speech = Omnilingual, read speech = Common '
            'Voice, web text)',
    eligibility='a group can enter validation/test only if (a) its leak-prone words are < 50% of its words, '
                '(b) its remaining words are <= half of one evaluation split\'s per-source target',
    selection='per source: 200 seeded restarts (numpy Generator seeded with [seed, 7, source_index, trial]) '
              'of random greedy fill + best-improvement local search (single moves and swaps between '
              'train/validation/test); lowest cost wins, ties -> earliest trial. Cost per evaluation split: '
              'squared errors of the word share (scale 0.1 point) and strict-word share (0.2 point) against '
              '5%; for each composition attribute the squared L2 distance between the split\'s category mix '
              'and the source\'s (scale 0.05); and a hinge on the Herfindahl index of group sizes above 0.12 '
              '(scale 0.03), so no single book/edition/site/speaker dominates a split without biasing the '
              'choice toward small groups; plus a steep penalty beyond +/-1.5 points',
    leak_definition='an evaluation record is leaked if the share of its distinct word 8-grams found in ANY '
                    'train record is >= 0.30, or its best exact 5-char-shingle Jaccard with a train record is '
                    '>= 0.50 (both on the leak-normalised text)',
    resolution='leak-prone records (computed against the whole corpus outside the record\'s group, closed '
               'under same-group moves) of evaluation groups are moved to train; a group whose leak-prone '
               'share is >= 50% stays wholly in train. The actual assignment is then verified against the '
               'actual train set; any residual leak is added to the leak-prone set and selection + '
               'verification repeat until zero evaluation records exceed a threshold',
    consequence='because leak-proneness is measured against everything outside the group, evaluation '
                'records are also free of >= threshold overlap with the OTHER evaluation split and with '
                'other groups of their own split',
)

if __name__ == '__main__':
    main()
