"""Step 2: induce stem|suffix segmentations from paradigm evidence.

Input : data/wordfreq_strict.json (build_counts.py), data/urdu_vs_hindko_lines.json,
        review/review_decisions.tsv (manual review; optional), the strict corpus
        (second pass for next-token context only).
Output: data/candidates_all.tsv   every analysed word with its parses and decision
        morph_silver_high.tsv / morph_silver_low.tsv (selected gold)
        data/induce_params.json   every parameter + counts

Method (all rules are explicit; parameters in PARAMS; see MORPHOLOGY.md):
 1. Candidate words: strict-corpus types with freq >= MIN_WORD_FREQ, spread over
    >= MIN_UNITS issues/books/sites, Arabic letters only, >= 3 letters, not a
    closed-class word (morph_inventory.FUNCTION_WORDS).
 2. Every way of writing the word as STEM + inventory suffix(es) is a parse,
    provided the stem has >= MIN_STEM_LEN letters. Each parse family has its own
    paradigm test, using corpus counts of the other members of the paradigm:
      VERB   stem + verb suffixes; 'strong' = >=2 imperfective cells (-دا family)
             AND >=3 distinct slots among IPFV/PFV/FUT/INF/SBJV/CONJ AND the
             stem's -دا forms are followed by an auxiliary (وے اے ایا پیا ...) in
             >= AUX_MIN_STRONG of their occurrences (separates کر|دے 'doing'
             from پردے 'curtains'). A consonant stem ending in ن whose shorter
             vowel stem explains the same -ندا forms is spurious (چاہن).
      CAUS   base VERB strong AND causative stem (base+ا / base+وا) has >=2
             attested verb forms of its own.
      AGR    X+{ا ی ے یاں}: the M.SG cell X+ا must be attested; strong = >=3
             cells and X+ہ rare (adjective-like); 2-cell paradigms must be
             balanced (WEAK_BALANCE). Stems that are infinitives (X+نڑ) or a
             verb stem + glide ی (ہوی|ا) are skipped.
      NOUN_A X+{ہ|ا, ے, یاں}: strong = all three cells attested.
      NOUN_I Xی+اں: Xی frequent, masculine X+ا/ہ/ے negligible, bare X rare.
      NOUN_C X+اں: X attested >= NOUN_C_MIN_STEM_FREQ and >= 20% of Xاں.
      PRON_GEN pronoun + joined genitive (اسدا = اس|دا).
    A member counts when freq >= MIN_MEMBER_FREQ and >= REL_CELL_FLOOR x the
    largest cell of the same paradigm (AGR: AGR_REL_CELL_FLOOR).
 3. The winner is the strong parse of highest family priority. Any other parse
    (weak or strong) that REQUIRES a boundary the winner neither requires nor
    lists as optional is a conflict. For -یاں nouns the masculine (X|یاں) and
    feminine (Xی|اں) readings are weighed by frequency; the larger is primary.
 4. Confidence: high = strong winner, no conflict, Urdu ratio <= 1, not a
    compound verb stem written jointly (ہوجا|سی), passes the per-word context
    check (step 6), not demoted by the manual review (review/review_decisions.tsv).
    Low = weak or conflicting evidence, Urdu-leaning shared forms (1 < ratio
    <= 3), compound stems, or a failed context check. Dropped = Urdu-only words
    (ratio > 3), Hindko-only suffix on an Urdu-leaning word (a look-alike such as
    جہاں), a word that a ONE-letter verb stem plus an inventory suffix explains
    (آنڑیں = آ|نڑیں: out of scope, stems have >= 2 letters), or review 'drop'.
 5. Selection: per-category quotas, most frequent first (deterministic).
 6. Per-word context check (verbs only), from the word's OWN tokens
    (data/context_profiles.json, scripts/context_profiles.py): a finite form or
    participle should rarely stand in a nominal context (followed by a case
    postposition or ولا, preceded by a quantifier / intensifier such as ہر،
    دونوں، بوت; finite forms also: followed by a genitive); an oblique infinitive
    should usually be followed by a postposition, ولا, or لگ- / جوگا. The pooled-stem
    auxiliary gate of step 2 cannot see a homograph whose stem is a real verb
    (جلدی 'quickly', پاسو 'side', سونڑے 'beautiful', ملاں 'mullah / mills').
 7. Alternatives: a conflicting parse is scored as an acceptable alternative
    only if (a) its own lexeme is attested (the -یاں masculine reading needs
    X+ا/ہ and X+ے; the feminine reading needs Xی; each >= ALT_MIN_FREQ and >=
    ALT_MIN_SHARE of the other), or it is a strong parse of another family, AND
    (b) the manual review confirmed by KWIC that the reading occurs (action
    'alt_ok'). Other conflicts stay in the notes only.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, 'F:/Hindko/_pipeline')
sys.path.insert(0, HERE)
from hp.lang import TOKEN_SPLIT_RE, strip_marks  # noqa: E402
from morph_inventory import (VERB_SUFFIXES, CAUS_FORMATIVES, NOUN_SUFFIXES,  # noqa: E402
                             AGR_SUFFIXES, SUPPLETIVE_PFV, FUNCTION_WORDS)

PARAMS = dict(
    MIN_WORD_FREQ=10,          # candidate word frequency in the strict corpus
    MIN_UNITS=3,               # distinct issues / books / web sites
    MIN_STEM_LEN=2,            # letters (code points) in the stem
    MIN_MEMBER_FREQ=3,         # a paradigm member must occur this often
    REL_CELL_FLOOR=0.02,       # ... and >= 2% of the paradigm's largest cell
    VERB_STRONG_IPFV_CELLS=2,
    VERB_STRONG_SLOTS=3,
    VERB_WEAK_MEMBERS=3,
    VERB_WEAK_SLOTS=2,
    AUX_MIN_STRONG=0.12,       # pooled share of the stem's -دا forms followed by an auxiliary
    AUX_MIN_WEAK=0.08,
    CAUS_MIN_OWN_FORMS=2,
    AGR_STRONG_CELLS=3,
    AGR_REL_CELL_FLOOR=0.03,   # AGR cells: >= 3% of the largest cell (گڈا 12 vs گڈی 569 is not a cell)
    AGR_MAX_HEH_SHARE=0.10,    # X+ہ share of (X+ا + X+ہ) above which X is noun-like
    NOUN_C_MIN_STEM_FREQ=10,
    NOUN_C_MIN_STEM_RATIO=0.2,   # freq(X) >= 20% of freq(Xاں)
    NOUN_I_MASC_MAX=0.05,      # masc cells <= 5% of Xی -> NOUN_I reading only
    URDU_RATIO_MAX_HIGH=1.0,   # rel.freq in Urdu lines / rel.freq in Hindko lines
    URDU_RATIO_DROP=3.0,       # above this a word with a suffix shared with Urdu is 'Urdu-only' -> dropped
    WEAK_BALANCE=0.10,         # 2-cell paradigms: smaller cell >= 10% of the larger
    NOUN_RELABEL_PP=0.4,       # -اں word followed by a postposition this often -> labelled N.PL
    # per-word context check (data/context_profiles.json; added after the second review round)
    CTX_NOM_MAX=0.10,          # verb form (not INF): share of its OWN tokens in a nominal context above this -> suspected homograph
    CTX_INF_OBL_MIN=0.30,      # oblique infinitive (-نڑیں -نڑے -نے): share followed by a postposition / ولا / لگ- / جوگا below this ...
    CTX_INF_OBL_AUX_EXEMPT=0.30,   # ... unless followed by an auxiliary this often (Peshawari present: سکنے ہن)
    # alternatives: a competing parse is offered as an acceptable alternative only if its own lexeme is attested
    ALT_MIN_FREQ=10,           # the competing reading's base form (Xی, or X+ا/ہ and X+ے) occurs this often ...
    ALT_MIN_SHARE=0.10,        # ... and is >= 10% of the primary reading's evidence; and review 'alt_ok' confirms it by KWIC
    QUOTA_HIGH=None,           # filled below
    QUOTA_LOW=None,
    TOTAL_MAX=800,
)
# per-category quotas (most frequent first); categories not listed get 0
PARAMS['QUOTA_HIGH'] = {
    'V.IPFV': 70, 'V.PFV': 55, 'V.FUT': 45, 'V.INF': 45, 'V.SBJV': 40, 'V.CAUS': 25, 'V.CONJ': 10,
    'N.PL': 75, 'N.OBL': 30, 'N.OBL.PL': 35, 'N.F.PL': 25, 'ADJ.AGR': 40, 'PRON.POSS': 16, 'PRON.GEN': 8,
}
PARAMS['QUOTA_LOW'] = {
    'V.IPFV': 25, 'V.PFV': 30, 'V.FUT': 25, 'V.INF': 25, 'V.SBJV': 25, 'V.CAUS': 20, 'V.CONJ': 8,
    'N.PL': 40, 'N.OBL': 15, 'N.OBL.PL': 30, 'N.F.PL': 15, 'ADJ.AGR': 25, 'PRON.POSS': 5, 'PRON.GEN': 5,
}

LIGHT_VERBS = ('جا', 'جل', 'سک', 'دی', 'لی', 'دے', 'لے', 'پا', 'بیٹھ', 'چک', 'رہ', 'پے')
BARE_VERBS = {'آ', 'ہو', 'کر', 'دے', 'لے', 'دی', 'لی', 'جا', 'پا'}
# suffixes that do not exist in Urdu: a word carrying one of them that is
# nevertheless Urdu-leaning (ratio > URDU_RATIO_MAX_HIGH) is a look-alike (جہاں، رواں، کیسی، ہنسی) -> dropped
HINDKO_ONLY_SUFFIXES = {'اں', 'یاں', 'نڑاں', 'نڑا', 'نڑیں', 'نڑے', 'نڑی', 'نڑیاں', 'ناں', 'سی', 'ساں', 'سیں', 'سن',
                        'سو', 'دا', 'دی', 'دے', 'دیاں', 'ندا', 'ندی', 'ندے', 'ندیاں', 'وے', 'ون', 'واں', 'ویں'}
SLOT_OF = {s['suffix']: s['slot'] for s in VERB_SUFFIXES}
INF_OBL_SUFFIXES = ('نڑیں', 'نڑے', 'نے')
TRUE_VOWELS = set('اآوےی')     # vowel letters proper (ہ-final stems such as رہ / کہ take -ی / -ے like consonant stems)
FAMILY_PRIORITY = {'PRON_GEN': 0, 'CAUS': 1, 'VERB': 2, 'AGR': 3, 'NOUN_A': 4, 'NOUN_I': 5, 'NOUN_C': 6}
VOWEL_FINAL = set('اآوےیہ')
LETTER_RANGES = [(0x0621, 0x063A), (0x0641, 0x064A), (0x066E, 0x06D3)]
POSTP = set('دا دی دے دیاں نوں کو آں نال اچ بچ وچ تے توں سی کولوں وسے واسطے آسطے تک نے'.split())
# copulas / auxiliaries that follow participles (present, past Peshawari and Hazara, progressive)
AUX = set('وے اے ہے ہیں ہن ون ان ایا ائی ائے ایاں ایہا ایہی ایہے ایہیاں آسا آسی آسے آسیاں اسا سا '
          'پیا پئی پئے پئیاں واں آں ایں ویں ہاں رہیا رئے رئی'.split())
PRONOUN_GEN_STEMS = {'اس': '3SG', 'جس': 'REL.SG', 'کس': 'INTERR.SG', 'ایس': 'DEM.PROX.SG'}
GEN_SUFFIXES = [('دا', 'GEN.M.SG'), ('دی', 'GEN.F.SG'), ('دے', 'GEN.M.PL/OBL'), ('دیاں', 'GEN.F.PL')]
POSSESSIVE_STEMS = {'میر': '1SG.POSS', 'تیر': '2SG.POSS', 'ساڈ': '1PL.POSS', 'سواڈ': '2PL/HON.POSS (Peshawari)',
                    'تہاڈ': '2PL.POSS', 'آپڑ': 'REFL.POSS (Peshawari spelling)', 'اپڑ': 'REFL.POSS',
                    'اپن': 'REFL.POSS (Urdu-like)', 'تواڈ': '2PL.POSS (twaaDaa)', 'اپنڑ': 'REFL.POSS (apNaa spelling)',
                    'مڑ': '1SG.POSS (Peshawari مڑا; homograph of مڑ "turn")'}


def is_letters(w):
    return all(any(lo <= ord(ch) <= hi for lo, hi in LETTER_RANGES) for ch in w)


class Corpus:
    def __init__(self):
        d = json.load(open(os.path.join(ROOT, 'data', 'wordfreq_strict.json'), encoding='utf-8'))
        self.meta = d['meta']
        self.rows = {r['form']: r for r in d['rows']}
        u = json.load(open(os.path.join(ROOT, 'data', 'urdu_vs_hindko_lines.json'), encoding='utf-8'))
        self.ur, self.hi = u['urdu'], u['hindko']
        self.ur_tok, self.hi_tok = u['meta']['urdu_tokens'], u['meta']['hindko_tokens']
        self.tot, self.aux, self.postp = context_stats()
        cp = json.load(open(os.path.join(ROOT, 'data', 'context_profiles.json'), encoding='utf-8'))
        self.ctx, self.ctx_meta = cp['words'], cp['meta']

    def aux_share(self, words):
        n = sum(self.tot[w] for w in words)
        return (sum(self.aux[w] for w in words) / n) if n else 0.0

    def f(self, w):
        if w in FUNCTION_WORDS:
            return 0
        r = self.rows.get(w)
        return r['freq'] if r else 0

    def urdu_ratio(self, w):
        u = self.ur.get(w, 0) / self.ur_tok
        h = self.hi.get(w, 0) / self.hi_tok
        if u == 0 and h == 0:
            return None
        return round(u / h, 3) if h > 0 else float('inf')


def cells_ok(counts, rel=None):
    """counts: {form: freq}. Returns the attested subset (absolute + relative floor)."""
    if not counts:
        return {}
    mx = max(counts.values())
    floor = max(PARAMS['MIN_MEMBER_FREQ'], (PARAMS['REL_CELL_FLOOR'] if rel is None else rel) * mx)
    return {w: c for w, c in counts.items() if c >= floor}


def mk(family, stem, sufs, word, members, strength, slots=(), gloss='', category='', extra=None):
    req, opt, pos = [], [], len(stem)
    req.append(pos)
    for s in sufs:
        for o in s.get('opt', []):
            opt.append(pos + o)
        pos += len(s['suffix'])
        if pos < len(word):
            req.append(pos)
    assert pos == len(word), (word, stem, sufs)
    p = dict(family=family, stem=stem, suffixes=[s['suffix'] for s in sufs], word=word, req=sorted(set(req)),
             opt=sorted(set(opt) - set(req)), members=members, strength=strength, slots=sorted(set(slots)),
             gloss=gloss, category=category)
    if extra:
        p.update(extra)
    return p


class Engine:
    def __init__(self, C):
        self.C = C
        self._verb = {}

    # ---------------- VERB ----------------
    def verb_ev(self, S):
        if S in self._verb:
            return self._verb[S]
        cls = 'V' if S[-1] in VOWEL_FINAL else 'C'
        raw = {}
        slot_of = {}
        for s in VERB_SUFFIXES:
            if s['stem'] not in (cls, 'any'):
                continue
            w = S + s['suffix']
            c = self.C.f(w)
            if c:
                raw[w] = c
                slot_of[w] = s['slot']
        att = cells_ok(raw)
        slots = {slot_of[w] for w in att}
        ipfv_forms = [w for w in att if slot_of[w] == 'IPFV']
        ipfv = len(ipfv_forms)
        aux = self.C.aux_share(ipfv_forms)
        # a consonant stem ending in ن whose shorter vowel stem explains the same
        # -ندا forms is spurious (چاہندا = چاہ|ندا, not چاہن|دا)
        spurious = False
        if cls == 'C' and S.endswith('ن') and len(S) - 1 >= PARAMS['MIN_STEM_LEN'] and S[-2] in VOWEL_FINAL:
            short = self.verb_ev(S[:-1])
            if short['strength'] and any(w in short['members'] for w in ipfv_forms):
                spurious = True
        if spurious:
            st = None
        elif (ipfv >= PARAMS['VERB_STRONG_IPFV_CELLS'] and len(slots) >= PARAMS['VERB_STRONG_SLOTS']
              and aux >= PARAMS['AUX_MIN_STRONG']):
            st = 'strong'
        elif (len(att) >= PARAMS['VERB_WEAK_MEMBERS'] and len(slots) >= PARAMS['VERB_WEAK_SLOTS'] and ipfv >= 1
              and aux >= PARAMS['AUX_MIN_WEAK']):
            st = 'weak'
        else:
            st = None
        res = dict(cls=cls, members=att, slots=slots, strength=st, slot_of=slot_of, aux=round(aux, 3))
        self._verb[S] = res
        return res

    def parses(self, w):
        out = []
        L = PARAMS['MIN_STEM_LEN']
        # PRON_GEN: pronoun + joined genitive
        for suf, gl in GEN_SUFFIXES:
            if w.endswith(suf) and w[:-len(suf)] in PRONOUN_GEN_STEMS:
                X = w[:-len(suf)]
                mem = cells_ok({X + s: self.C.rows.get(X + s, {}).get('freq', 0) for s, _ in GEN_SUFFIXES})
                st = 'strong' if len(mem) >= 2 else ('weak' if mem else None)
                if st:
                    out.append(mk('PRON_GEN', X, [dict(suffix=suf, opt=[1])], w, mem, st,
                                  gloss=PRONOUN_GEN_STEMS[X] + '-' + gl, category='PRON.GEN'))
        # VERB
        for s in VERB_SUFFIXES:
            if not s['target'] or not w.endswith(s['suffix']):
                continue
            S = w[:-len(s['suffix'])]
            if len(S) < L:
                continue
            ev = self.verb_ev(S)
            if s['stem'] not in (ev['cls'], 'any') or not ev['strength']:
                continue
            # the target word itself need not pass the relative cell floor (کرنیاں 103 vs کرنے 6286);
            # the evidence is the rest of the paradigm
            out.append(mk('VERB', S, [s], w, ev['members'], ev['strength'], ev['slots'],
                          gloss=s['gloss'], category='V.' + s['slot']))
        # CAUS: base + (ا|وا) + vowel-stem verb suffix
        for s in VERB_SUFFIXES:
            if not s['target'] or s['stem'] not in ('V', 'any') or not w.endswith(s['suffix']):
                continue
            CS = w[:-len(s['suffix'])]
            for c in CAUS_FORMATIVES:
                if not CS.endswith(c['suffix']):
                    continue
                base = CS[:-len(c['suffix'])]
                if len(base) < L or base[-1] in VOWEL_FINAL:
                    continue
                bev = self.verb_ev(base)
                cev = self.verb_ev(CS)
                if bev['strength'] != 'strong' or self.C.f(w) < PARAMS['MIN_MEMBER_FREQ']:
                    continue
                if len(cev['members']) < PARAMS['CAUS_MIN_OWN_FORMS']:
                    continue
                mem = dict(bev['members'])
                mem.update(cev['members'])
                out.append(mk('CAUS', base, [dict(suffix=c['suffix'], opt=[]), s], w, mem, 'strong',
                              sorted(bev['slots'] | cev['slots']),
                              gloss=c['gloss'] + '-' + s['gloss'], category='V.CAUS',
                              extra=dict(caus_stem=CS, caus_forms=cev['members'])))
        # AGR / NOUN_A / NOUN_I (endings ا ی ے یاں)
        for s in AGR_SUFFIXES:
            if not w.endswith(s['suffix']):
                continue
            X = w[:-len(s['suffix'])]
            if len(X) < L or X.endswith('ئ'):
                continue
            # X+نڑ where X is a verb stem (or too short to test) is an infinitive, not an adjective
            if X.endswith('نڑ') and (len(X) - 2 < L or self.verb_ev(X[:-2])['strength']):
                continue
            # X = verb stem + glide ی (ہوی|ا, رہی|ا are spellings of ہو|یا, رہ|یا)
            if X.endswith('ی') and len(X) - 1 >= L and self.verb_ev(X[:-1])['strength'] == 'strong':
                continue
            # X = a verb stem ending in a vowel letter: its PFV / SBJV forms are X+یا / X+ئی / X+ئے / X+وے,
            # so X+ا / X+ی / X+ے / X+یاں belong to another lexeme (روے is رہوے 'may remain', not رو 'cry';
            # ہوا is 'air'). Stems in ہ (رہ، کہ) take -ی / -ے like consonant stems and are not affected.
            if X[-1] in TRUE_VOWELS and self.verb_ev(X)['strength'] == 'strong':
                continue
            raw = {X + a['suffix']: self.C.f(X + a['suffix']) for a in AGR_SUFFIXES}
            raw = {k: v for k, v in raw.items() if v}
            att = cells_ok(raw, PARAMS['AGR_REL_CELL_FLOOR'])
            heh = self.C.f(X + 'ہ')
            alef = self.C.f(X + 'ا')
            heh_share = heh / (heh + alef) if (heh + alef) else 0.0
            has_msg = (X + 'ا') in att   # the M.SG citation form must be attested
            balanced = len(att) >= 3 or (len(att) == 2 and
                                         min(att.values()) >= PARAMS['WEAK_BALANCE'] * max(att.values()))
            if w in att and len(att) >= 2 and has_msg and balanced:
                if X in POSSESSIVE_STEMS:
                    cat, gl = 'PRON.POSS', POSSESSIVE_STEMS[X] + '.' + s['gloss']
                elif X in SUPPLETIVE_PFV:
                    cat, gl = 'V.PFV', 'PFV.PTCP.' + s['gloss'] + ' (suppletive stem of ' + SUPPLETIVE_PFV[X] + ')'
                elif self.verb_ev(X)['strength']:
                    cat, gl = 'V.PFV', 'PFV.PTCP.' + s['gloss'] + ' (-aa/-ii/-e/-iaa~ agreement)'
                else:
                    cat, gl = 'ADJ.AGR', s['gloss']
                adj_like = heh_share <= PARAMS['AGR_MAX_HEH_SHARE'] or X in SUPPLETIVE_PFV
                st = 'strong' if (len(att) >= PARAMS['AGR_STRONG_CELLS'] and adj_like) else 'weak'
                out.append(mk('AGR', X, [s], w, att, st, sorted(a for a in att), gloss=gl, category=cat,
                              extra=dict(heh_share=round(heh_share, 3))))
            # NOUN_A only for ے / یاں
            if s['suffix'] in ('ے', 'یاں'):
                raw = {X + 'ہ': heh, X + 'ا': alef, X + 'ے': self.C.f(X + 'ے'), X + 'یاں': self.C.f(X + 'یاں')}
                raw = {k: v for k, v in raw.items() if v}
                att = cells_ok(raw)
                dir_ok = (X + 'ہ') in att or (X + 'ا') in att
                n_cells = int(dir_ok) + int((X + 'ے') in att) + int((X + 'یاں') in att)
                ok_bal = n_cells == 3 or bool(att and min(att.values()) >= PARAMS['WEAK_BALANCE'] * max(att.values()))
                if w in att and n_cells >= 2 and ok_bal:
                    st = 'strong' if n_cells == 3 else 'weak'
                    ns = [n for n in NOUN_SUFFIXES if n['suffix'] == s['suffix']][0]
                    out.append(mk('NOUN_A', X, [ns], w, att, st, gloss=ns['gloss'],
                                  category='N.OBL' if s['suffix'] == 'ے' else 'N.OBL.PL',
                                  extra=dict(heh_share=round(heh_share, 3))))
        # NOUN_C / NOUN_I : X + اں
        if w.endswith('اں'):
            X = w[:-2]
            fx, fw = self.C.f(X), self.C.f(w)
            if len(X) >= L and not X.endswith('ئ') and not X.endswith('نڑ') and fx >= PARAMS['NOUN_C_MIN_STEM_FREQ'] and fx >= PARAMS['NOUN_C_MIN_STEM_RATIO'] * fw:
                ns = NOUN_SUFFIXES[0]
                if X.endswith('ی'):
                    X0 = X[:-1]
                    if len(X) < 3 or (X.endswith('ئی') and len(X) <= 3):
                        pass      # گیاں، پیاں، نیاں، آئیاں: stems too short / verb forms
                    else:
                        masc = self.C.f(X0 + 'ا') + self.C.f(X0 + 'ہ') + self.C.f(X0 + 'ے')
                        bare = self.C.f(X0)   # consonant-final noun + -iaa~ (اکھ -> اکھیاں)
                        st = 'strong' if (masc <= PARAMS['NOUN_I_MASC_MAX'] * fx
                                          and bare < PARAMS['WEAK_BALANCE'] * fx) else 'weak'
                        out.append(mk('NOUN_I', X, [ns], w, {X: fx, w: fw}, st, gloss='F.PL (-ii noun)',
                                      category='N.F.PL', extra=dict(masc_cells=masc, bare_stem=bare)))
                else:
                    out.append(mk('NOUN_C', X, [ns], w, {X: fx, w: fw}, 'strong', gloss=ns['gloss'],
                                  category='N.PL'))
        return out

    def analyse(self, w):
        ps = self.parses(w)
        if not ps:
            return None
        key = lambda p: (0 if p['strength'] == 'strong' else 1, FAMILY_PRIORITY[p['family']], p['req'][0])
        ps.sort(key=key)
        W = ps[0]
        cover = set(W['req']) | set(W['opt'])
        conflicts = [p for p in ps[1:] if not set(p['req']) <= cover]
        # -یاں on nouns: masculine X|یاں (بچہ -> بچیاں) vs feminine Xی|اں (بچی -> بچیاں) vs
        # consonant-final X + -iaa~ (اکھ -> اکھیاں). The larger reading is primary, the other an alternative.
        if W['family'] in ('NOUN_A', 'AGR', 'NOUN_I') and w.endswith('یاں'):
            X0 = w[:-3]
            fem = self.C.f(X0 + 'ی')
            masc = self.C.f(X0 + 'ا') + self.C.f(X0 + 'ہ') + self.C.f(X0 + 'ے')
            bare = self.C.f(X0)
            adj = W['family'] == 'AGR' and W.get('heh_share', 1.0) <= PARAMS['AGR_MAX_HEH_SHARE']
            floor = PARAMS['NOUN_I_MASC_MAX']
            if not adj and len(X0) >= PARAMS['MIN_STEM_LEN']:
                ni = mk('NOUN_I', X0 + 'ی', [NOUN_SUFFIXES[0]], w, {X0 + 'ی': fem}, 'weak',
                        gloss='F.PL (-ii noun)', category='N.F.PL')
                na = mk('NOUN_A', X0, [NOUN_SUFFIXES[2]], w, {X0 + 'ا/ہ/ے': masc}, 'weak',
                        gloss='M.OBL.PL | F.PL of consonant-final noun', category='N.OBL.PL')
                if W['family'] != 'NOUN_I' and fem >= masc and fem >= PARAMS['MIN_MEMBER_FREQ']:
                    # the feminine reading is the larger one: it becomes primary
                    old = W
                    W = dict(ni, members={X0 + 'ی': fem, w: self.C.f(w)},
                             strength='strong' if masc <= floor * fem else 'weak')
                    conflicts = [c for c in conflicts if c['req'] != W['req']]
                    if masc > floor * fem:
                        conflicts.append(old)
                elif W['family'] != 'NOUN_I' and fem > max(PARAMS['MIN_MEMBER_FREQ'], floor * masc):
                    if not any(c['req'] == ni['req'] for c in conflicts):
                        conflicts.append(ni)
                if W['family'] == 'NOUN_I' and (masc > floor * max(fem, 1) or bare >= PARAMS['WEAK_BALANCE'] * max(fem, 1)):
                    if not any(c['req'] == na['req'] for c in conflicts):
                        conflicts.append(na)
        # -اں: subjunctive 1SG and noun plural share the boundary; label by syntax
        if W['family'] == 'VERB' and W['suffixes'] == ['اں'] and self.C.tot[w]:
            pp = self.C.postp[w] / self.C.tot[w]
            if pp >= PARAMS['NOUN_RELABEL_PP'] and any(p['family'] == 'NOUN_C' and p['req'] == W['req'] for p in ps):
                W = dict(W, category='N.PL', gloss='PL (noun reading; also SBJV.1SG of the verb)')
        compound = self.compound_stem(W['stem']) if W['family'] in ('VERB', 'CAUS') else None
        return dict(word=w, winner=W, parses=ps, conflicts=conflicts, compound=compound,
                    submin=self.submin_stem(w), alt_candidates=self.alt_candidates(w, W, conflicts))

    def submin_stem(self, w):
        """A verb stem SHORTER than MIN_STEM_LEN (the one-letter آ 'come') + an inventory suffix explains
        the word (آنڑیں = آ|نڑیں, آنڑے = آ|نڑے). The longer-stem parse (آنڑ|یں) is then not trusted, and the
        word is out of scope for this set. Returns the short analysis or None."""
        for s in VERB_SUFFIXES:
            if not s['target'] or not w.endswith(s['suffix']):
                continue
            S = w[:-len(s['suffix'])]
            if 0 < len(S) < PARAMS['MIN_STEM_LEN']:
                ev = self.verb_ev(S)
                if ev['strength'] == 'strong' and s['stem'] in (ev['cls'], 'any'):
                    return S + '|' + s['suffix']
        return None

    def alt_candidates(self, w, W, conflicts):
        """Conflicting parses that may be offered as acceptable alternatives (step 7a). Returns
        [(req, reason)]. Review 'alt_ok' must still confirm each one (step 7b)."""
        out = []
        f = self.C.f
        for c in conflicts:
            if c['req'] == W['req'] or any(c['req'] == r for r, _ in out):
                continue
            if w.endswith('یاں') and c['family'] in ('NOUN_A', 'NOUN_I', 'AGR') and \
                    W['family'] in ('NOUN_A', 'NOUN_I', 'AGR'):
                X0 = w[:-3]
                dir_ = max(f(X0 + 'ا'), f(X0 + 'ہ'))
                obl = f(X0 + 'ے')
                masc = dir_ + obl
                fem = f(X0 + 'ی')
                if c['req'] == [len(X0)]:          # masculine / adjectival reading X|یاں
                    ok = (dir_ >= PARAMS['ALT_MIN_FREQ'] and obl >= PARAMS['ALT_MIN_FREQ']
                          and masc >= PARAMS['ALT_MIN_SHARE'] * fem)
                    why = f'masc. reading: {X0}ا/ہ {dir_}, {X0}ے {obl} vs {X0}ی {fem}'
                elif c['req'] == [len(X0) + 1]:    # feminine reading Xی|اں
                    ok = fem >= PARAMS['ALT_MIN_FREQ'] and fem >= PARAMS['ALT_MIN_SHARE'] * masc
                    why = f'fem. reading: {X0}ی {fem} vs masc. {masc}'
                else:
                    ok, why = False, 'other -یاں split'
            else:
                ok = c['strength'] == 'strong'
                why = f"{c['family']} parse is {c['strength']}"
            if ok:
                out.append((c['req'], why))
        return out

    def compound_stem(self, S):
        """S = A + light verb written jointly (ہوجا = ہو+جا, ہوسک = ہو+سک)? Returns the split or None."""
        for B in LIGHT_VERBS:
            if S.endswith(B) and len(S) > len(B):
                A = S[:-len(B)]
                if A in BARE_VERBS or (len(A) >= PARAMS['MIN_STEM_LEN'] and self.verb_ev(A)['strength'] == 'strong'):
                    return A + '+' + B
        return None


def context_stats():
    """next-token statistics for every word of the strict corpus: total,
    followed by an auxiliary (AUX), followed by a postposition (POSTP)"""
    tot, aux, postp = collections.Counter(), collections.Counter(), collections.Counter()
    for line in open('F:/Hindko/hindko_dataset.jsonl', encoding='utf-8'):
        r = json.loads(line)
        for ln in r['text'].splitlines():
            toks = [strip_marks(t) for t in TOKEN_SPLIT_RE.split(ln) if t]
            toks = [t for t in toks if t]
            for i, t in enumerate(toks):
                tot[t] += 1
                if i + 1 < len(toks):
                    nx = toks[i + 1]
                    if nx in AUX:
                        aux[t] += 1
                    if nx in POSTP:
                        postp[t] += 1
    return tot, aux, postp


def context_check(C, w, W):
    """Step 6. Returns (metric string, flagged). Only verb categories are tested."""
    if not W['category'].startswith('V.'):
        return '', False
    p = C.ctx.get(w)
    if not p:
        return 'no profile', False
    last = W['suffixes'][-1]
    slot = SLOT_OF.get(last, 'PFV') if W['family'] in ('VERB', 'CAUS') else 'PFV'

    def share(side, *classes):
        d = p[side]
        n = sum(d.values())
        return sum(d.get(k, 0) for k in classes) / n if n else 0.0
    if slot == 'INF':
        if last in INF_OBL_SUFFIXES:
            obl, aux = share('next', 'GENP', 'CASE', 'NMLZ', 'INFGOV'), share('next', 'AUX')
            return (f'obl={obl:.2f} aux={aux:.2f}',
                    obl < PARAMS['CTX_INF_OBL_MIN'] and aux < PARAMS['CTX_INF_OBL_AUX_EXEMPT'])
        d = share('prev', 'DETQ')
        return f'detq={d:.2f}', d > PARAMS['CTX_NOM_MAX']
    if slot in ('IPFV', 'PFV'):      # participles may take an attributive genitive (بنڑی دی مسیت): not counted
        nom = share('next', 'CASE', 'NMLZ') + share('prev', 'DETQ')
    else:                            # SBJV / FUT / CONJ: finite or adverbial, a following genitive counts too
        nom = share('next', 'GENP', 'CASE', 'NMLZ') + share('prev', 'DETQ')
    return f'nom={nom:.2f}', nom > PARAMS['CTX_NOM_MAX']


def dialect_label(r, meta):
    gt = meta['group_tokens']
    P, B, H = r['PESH'], r['BOOK'], r['HAZ']
    rp, rh = P / gt['PESH'], H / gt['HAZ']
    exp_h = r['freq'] * gt['HAZ'] / meta['tokens']
    counts = f"P:{P} B:{B} H:{H}"
    if H >= 5 and rh >= 3 * rp:
        return 'Hazara-enriched (' + counts + ')'
    if exp_h >= 5 and H <= exp_h / 5 and P >= 10:
        return f'absent/rare in Hazara sources ({counts}; H expected {exp_h:.0f})'
    if H >= 3 and P >= 3:
        return 'shared (' + counts + ')'
    return 'insufficient (' + counts + ')'


def load_review():
    """review/review_decisions.tsv: word, action, reason, alt. Actions: drop, low (change confidence);
    alt_ok (KWIC confirms the alternative boundary set in 'alt'); noalt (the competing parse in 'alt' is
    not an acceptable segmentation). A word may have several rows."""
    path = os.path.join(ROOT, 'review', 'review_decisions.tsv')
    dec = collections.defaultdict(list)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row['word'].startswith('#'):
                    continue
                assert row['action'] in ('drop', 'low', 'alt_ok', 'noalt'), row
                row['alt'] = (row.get('alt') or '').strip()
                dec[row['word']].append(row)
    return dec


def fmt_members(m, limit=14):
    items = sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return ' '.join(f'{k}:{v}' for k, v in items)


def seg_string(word, req):
    out, last = [], 0
    for b in req:
        out.append(word[last:b]); last = b
    out.append(word[last:])
    return '|'.join(out)


def main():
    C = Corpus()
    E = Engine(C)
    review = load_review()
    cands = [w for w, r in C.rows.items()
             if r['freq'] >= PARAMS['MIN_WORD_FREQ'] and r['n_units'] >= PARAMS['MIN_UNITS']
             and w not in FUNCTION_WORDS and is_letters(w) and len(w) >= 3]
    cands.sort(key=lambda w: (-C.rows[w]['freq'], w))
    analysed = []
    for w in cands:
        a = E.analyse(w)
        if a:
            analysed.append(a)
    ctx = {a['word']: (round(C.postp[a['word']] / C.tot[a['word']], 3) if C.tot[a['word']] else None)
           for a in analysed}
    rows = []
    for a in analysed:
        w, W = a['word'], a['winner']
        r = C.rows[w]
        ur = C.urdu_ratio(w)
        reasons = []
        conf = 'high'
        if W['strength'] != 'strong':
            conf = 'low'; reasons.append('weak paradigm evidence')
        if a['conflicts']:
            conf = 'low'
            reasons.append('competing segmentation: ' + '; '.join(
                f"{c['family']} {seg_string(w, c['req'])}" for c in a['conflicts']))
        if a.get('compound'):
            conf = 'low'; reasons.append('compound verb stem written jointly (' + a['compound'] + ')')
        if ur is not None and ur > PARAMS['URDU_RATIO_MAX_HIGH']:
            last = W['suffixes'][-1]
            if last in HINDKO_ONLY_SUFFIXES or W['family'] in ('NOUN_C', 'NOUN_I'):
                conf = 'drop'; reasons.append(f'Urdu-leaning (ratio {ur}) although the suffix is Hindko-only: look-alike')
            elif ur > PARAMS['URDU_RATIO_DROP']:
                conf = 'drop'; reasons.append(f'Urdu-only word (urdu/hindko line ratio {ur})')
            else:
                conf = 'low'; reasons.append(f'Urdu-leaning shared form (urdu/hindko line ratio {ur})')
        if a['submin'] and conf != 'drop':
            conf = 'drop'
            reasons.append(f"a one-letter verb stem explains the word ({a['submin']}); stems of < "
                           f"{PARAMS['MIN_STEM_LEN']} letters are out of scope")
        ctx_metric, ctx_flag = context_check(C, w, W)
        if ctx_flag:
            if conf == 'high':
                conf = 'low'
            reasons.append(f'context check failed ({ctx_metric}): the word is often used as another lexeme')
        rvs = review.get(w, [])
        for rv in rvs:
            if rv['action'] in ('drop', 'low') and conf != 'drop':
                if rv['action'] == 'drop':
                    conf = 'drop'
                elif conf == 'high':
                    conf = 'low'
                reasons.append('review: ' + rv['reason'])
        ok_alt = {rv['alt'] for rv in rvs if rv['action'] == 'alt_ok'}
        no_alt = {rv['alt']: rv['reason'] for rv in rvs if rv['action'] == 'noalt'}
        alts = []
        for req, why in a['alt_candidates']:
            key = ','.join(map(str, req))
            if key in ok_alt and key not in no_alt:
                alts.append(req)
        for key in ok_alt:
            assert [int(x) for x in key.split(',')] in alts, ('alt_ok for a parse the method did not offer', w, key)
        offered = {','.join(map(str, req)): why for req, why in a['alt_candidates']}
        rejected, seen = [], set()
        for c in a['conflicts']:
            key = ','.join(map(str, c['req']))
            if c['req'] == W['req'] or c['req'] in alts or key in seen:
                continue
            seen.add(key)
            if key in no_alt:
                why = 'review: ' + no_alt[key]
            elif key in offered:
                why = offered[key] + '; not confirmed by review'
            else:
                why = 'its own lexeme is not attested (step 7a)'
            rejected.append(f'{seg_string(w, c["req"])} ({why})')
        if rejected:
            reasons.append('not accepted as alternative: ' + '; '.join(rejected))
        alt_offered = ';'.join(f'{seg_string(w, req)}' for req, _ in a['alt_candidates'])
        rows.append(dict(
            word=w, segmentation=seg_string(w, W['req']), boundaries=','.join(map(str, W['req'])),
            optional_boundaries=','.join(map(str, W['opt'])),
            alternatives=';'.join(','.join(map(str, x)) for x in alts),
            stem=W['stem'], suffixes='+'.join(W['suffixes']), gloss=W['gloss'], category=W['category'],
            family=W['family'], strength=W['strength'], freq=r['freq'], PESH=r['PESH'], BOOK=r['BOOK'],
            HAZ=r['HAZ'], n_units=r['n_units'], dialect=dialect_label(r, C.meta),
            urdu_ratio='' if ur is None else ur, next_is_postposition=ctx.get(w), context=ctx_metric,
            slots=','.join(W['slots']), evidence=fmt_members(W['members']),
            caus_forms=fmt_members(W.get('caus_forms', {})) if W['family'] == 'CAUS' else '',
            n_parses=len(a['parses']), alt_offered=alt_offered, confidence=conf, notes=' | '.join(reasons),
            top_surface=r['top_surface'],
        ))
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    cols = list(rows[0].keys())
    with open(os.path.join(ROOT, 'data', 'candidates_all.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(cols) + '\n')
        for x in rows:
            f.write('\t'.join(str(x[c]) for c in cols) + '\n')

    # ---------------- selection ----------------
    def pick(conf, quota):
        by = collections.defaultdict(list)
        for x in rows:
            if x['confidence'] == conf:
                by[x['category']].append(x)
        sel = []
        for cat in sorted(quota):
            lst = sorted(by.get(cat, []), key=lambda x: (-x['freq'], x['word']))
            sel.extend(lst[:quota[cat]])
        return sel, {c: len(v) for c, v in sorted(by.items())}

    high, pool_h = pick('high', PARAMS['QUOTA_HIGH'])
    low, pool_l = pick('low', PARAMS['QUOTA_LOW'])
    total = len(high) + len(low)
    if total > PARAMS['TOTAL_MAX']:
        low = sorted(low, key=lambda x: (-x['freq'], x['word']))[:PARAMS['TOTAL_MAX'] - len(high)]
    out_cols = ['id', 'word', 'segmentation', 'boundaries', 'optional_boundaries', 'alternatives', 'stem',
                'suffixes', 'gloss', 'category', 'family', 'freq', 'PESH', 'BOOK', 'HAZ', 'n_units', 'dialect',
                'urdu_ratio', 'next_is_postposition', 'context', 'slots', 'evidence', 'caus_forms', 'confidence',
                'notes',
                'top_surface']
    for name, sel, prefix in (('morph_silver_high.tsv', high, 'H'), ('morph_silver_low.tsv', low, 'L')):
        sel = sorted(sel, key=lambda x: (x['category'], -x['freq'], x['word']))
        with open(os.path.join(ROOT, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\t'.join(out_cols) + '\n')
            for i, x in enumerate(sel, 1):
                x = dict(x, id=f'{prefix}{i:04d}')
                f.write('\t'.join(str(x[c]) for c in out_cols) + '\n')
    stats = dict(params=PARAMS, corpus=C.meta, n_candidates=len(cands), n_analysed=len(analysed),
                 decisions=dict(collections.Counter(x['confidence'] for x in rows)),
                 pool_high=pool_h, pool_low=pool_l,
                 selected_high=dict(collections.Counter(x['category'] for x in high)),
                 selected_low=dict(collections.Counter(x['category'] for x in low)),
                 n_high=len(high), n_low=len(low), n_review_words=len(review),
                 n_review_decisions=sum(len(v) for v in review.values()),
                 n_review_by_action=dict(sorted(collections.Counter(rv['action'] for v in review.values()
                                                                     for rv in v).items())),
                 n_context_flagged=sum(1 for x in rows if 'context check failed' in x['notes']),
                 n_submin_dropped=sum(1 for x in rows if 'one-letter verb stem explains' in x['notes']),
                 context_meta={k: v for k, v in C.ctx_meta.items() if k != 'source'})
    with open(os.path.join(ROOT, 'data', 'induce_params.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ('params', 'corpus', 'context_meta')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
