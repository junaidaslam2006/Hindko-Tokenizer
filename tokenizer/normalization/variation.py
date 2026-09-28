"""Orthographic variation that a tokenizer will fragment but normalization
must NOT fold. Measured on the canonical data form (hp.normalize) of the
permissive corpus. Writes variation/variation.json and variation_out.txt.
Deterministic (file order; ties broken by the word string).
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r'F:\Hindko\_pipeline')
from hp import normalize as N  # noqa: E402

PERM = r'F:\Hindko\hindko_dataset_permissive.jsonl'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'variation')
os.makedirs(OUT, exist_ok=True)
SOURCES = ('newspaper', 'book', 'web')


def c(*cps):
    return ''.join(chr(x) for x in cps)


NOON, RREH, NOON_TAH, SUKUN = c(0x0646), c(0x0691), c(0x0768), c(0x0652)
HEH_GOAL, HEH_DO = c(0x06C1), c(0x06BE)
YEH_HAMZA, FA_YEH, YEH_BARREE, YEH_BARREE_HAMZA, HAMZA = c(0x0626), c(0x06CC), c(0x06D2), c(0x06D3), c(0x0621)
HEH_GOAL_HAMZA, HAMZA_ABOVE, KASRA = c(0x06C2), c(0x0654), c(0x0650)
MARKS_RE = re.compile('[' + c(*range(0x0610, 0x061B)) + c(*range(0x064B, 0x0660)) + c(0x0670) + ']')
TONE = {c(0x08BE): c(0x0628, 0x06BE), c(0x08BF): c(0x062F, 0x06BE), c(0x08C0): c(0x0688, 0x06BE),
        c(0x08C1): c(0x062C, 0x06BE), c(0x08C2): c(0x06AF, 0x06BE)}   # tone letter -> voiced aspirate
TONE_PLAIN = {k: v for k, v in zip(TONE, (c(0x067E), c(0x062A), c(0x0679), c(0x0686), c(0x06A9)))}


def main():
    words = Counter()
    words_src = defaultdict(Counter)
    n_tokens = 0
    n_tokens_marked = 0
    izafat = Counter()
    doubled = Counter()
    detached = Counter()
    ascii_stop = Counter()
    with open(PERM, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            s = r['source']
            t = N.normalize(r['text'])
            for m in N.WORD_RE.finditer(t):
                w = m.group()
                words[w] += 1
                words_src[w][s] += 1
                n_tokens += 1
                if MARKS_RE.search(w):
                    n_tokens_marked += 1
            # izafat encodings at word end (followed by space / punctuation / end)
            izafat['HEH_GOAL_WITH_HAMZA (U+06C2) word-final'] += len(re.findall(HEH_GOAL_HAMZA + r'(?=\s|$)', t))
            izafat['HEH_GOAL + HAMZA (U+06C1 U+0621) word-final'] += len(re.findall(HEH_GOAL + HAMZA + r'(?=\s|$)', t))
            izafat['YEH_WITH_HAMZA (U+0626) word-final'] += len(re.findall(YEH_HAMZA + r'(?=\s|$)', t))
            izafat['FARSI_YEH + HAMZA (U+06CC U+0621) word-final'] += len(re.findall(FA_YEH + HAMZA + r'(?=\s|$)', t))
            izafat['FARSI_YEH + HAMZA ABOVE (U+06CC U+0654)'] += t.count(FA_YEH + HAMZA_ABOVE)
            izafat['izafat kasra written after a space'] += len(re.findall(' ' + KASRA + '(?= )', t))
            doubled[s] += len(re.findall('([' + c(*range(0x064B, 0x0653)) + '])\\1', t))
            detached[s] += len(re.findall('(?:^|(?<=[ \\n]))[' + c(*range(0x064B, 0x0660)) + c(0x0670)
                                          + c(*range(0x0610, 0x061B)) + ']', t))
            ascii_stop[s] += len(re.findall('(?<=[' + c(*range(0x0621, 0x06D4)) + '])\\.(?=[ \\n]|$)', t))
            ascii_stop[s + '_urdu_full_stop'] += t.count(c(0x06D4))

    def pairs(fn, min_each=1, top=25):
        """word pairs (w, fn(w)) both attested; fn returns list of variants."""
        out = []
        for w, k in words.items():
            for v in fn(w):
                if v != w and v in words and words[v] >= min_each and k >= min_each:
                    out.append((min(k, words[v]), w, k, v, words[v]))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    res = {'arabic_word_tokens': n_tokens, 'arabic_word_types': len(words),
           'tokens_with_marks': n_tokens_marked}

    # 1 retroflex nasal
    retro_types = {'NOON WITH SMALL TAH': sum(1 for w in words if NOON_TAH in w),
                   'NOON+RREH': sum(1 for w in words if NOON + RREH in w)}
    retro_tok = {'NOON WITH SMALL TAH': sum(k for w, k in words.items() if NOON_TAH in w),
                 'NOON+RREH': sum(k for w, k in words.items() if NOON + RREH in w),
                 'NOON+SUKUN': sum(k for w, k in words.items() if NOON + SUKUN in w)}
    rp = pairs(lambda w: [w.replace(NOON_TAH, NOON + RREH)] if NOON_TAH in w else [])
    res['retroflex'] = {'types': retro_types, 'tokens': retro_tok,
                        'by_source_tokens_noon_small_tah': dict(sum((words_src[w] for w in words if NOON_TAH in w),
                                                                    Counter())),
                        'pairs_both_attested': len(rp), 'pairs_top': rp[:25]}
    # 2 tone letters vs voiced-aspirate respelling and vs the plain letter
    tp = pairs(lambda w: [w.replace(k, v) for k, v in TONE.items() if k in w])
    tq = pairs(lambda w: [w.replace(k, v) for k, v in TONE_PLAIN.items() if k in w])
    res['tone_letters'] = {'tokens_with_tone_letter': sum(k for w, k in words.items() if any(x in w for x in TONE)),
                           'types_with_tone_letter': sum(1 for w in words if any(x in w for x in TONE)),
                           'pairs_vs_voiced_aspirate': len(tp), 'pairs_vs_voiced_aspirate_top': tp[:20],
                           'pairs_vs_plain_letter': len(tq), 'pairs_vs_plain_letter_top': tq[:20]}
    # 3 HEH GOAL vs HEH DOACHASHMEE (single-position swap)
    def swap_h(w):
        out = []
        for i, ch in enumerate(w):
            if ch == HEH_GOAL:
                out.append(w[:i] + HEH_DO + w[i + 1:])
        return out
    hp_ = pairs(swap_h, min_each=5)
    res['heh_goal_vs_doachashmee'] = {'pairs_both_attested_min5': len(hp_), 'pairs_top': hp_[:30]}
    # 4 harakat: same letters with and without marks
    bare = defaultdict(Counter)
    for w, k in words.items():
        bare[MARKS_RE.sub('', w)][w] += k
    multi = [(b, v) for b, v in bare.items() if len(v) > 1]
    multi.sort(key=lambda x: (-sum(x[1].values()), x[0]))
    res['harakat'] = {'bare_types_with_several_mark_spellings': len(multi),
                      'tokens_in_those_types': sum(sum(v.values()) for _, v in multi),
                      'top': [(b, v.most_common(6)) for b, v in multi[:20]]}
    # 5 izafat / hamza encodings
    res['izafat_hamza'] = dict(izafat)
    yb = pairs(lambda w: [w.replace(YEH_BARREE_HAMZA, YEH_HAMZA + YEH_BARREE)] if YEH_BARREE_HAMZA in w else [])
    res['yeh_barree_hamza_vs_hamza_seat'] = {
        'tokens_U+06D3': sum(k for w, k in words.items() if YEH_BARREE_HAMZA in w),
        'tokens_U+0626_U+06D2': sum(k for w, k in words.items() if YEH_HAMZA + YEH_BARREE in w),
        'pairs_both_attested': len(yb), 'pairs_top': yb[:15]}
    # 6 curated Hindko spelling variants (same word, several spellings)
    curated = {
        'not': ['نہیں', 'نئیں', 'نیں', 'نہ', 'ناہیں'],
        'in': ['وچ', 'اچ', 'وِچ', 'اِچ', 'بچ'],
        'is (3sg)': ['ہے', 'اے', 'وے'],
        'then/again': ['فیر', 'پھر', 'پِھر'],
        'we (Hindko)': ['اساں', 'اسی', 'اَسی', 'اسیں'],
        'Peshawar': ['پشاور', 'پشور', 'پیشور'],
        'to be (inf.)': ['ہونڑ', 'ہوݨ', 'ہون', 'ہونا'],
        'water': ['پانڑی', 'پاݨی', 'پانی'],
        'own': ['اپنڑا', 'اپݨا', 'اپنا', 'اپنڑے', 'اپݨے', 'اپنے'],
    }
    res['curated_variants'] = {k: [(w, words.get(w, 0)) for w in v] for k, v in curated.items()}
    res['doubled_identical_harakat'] = dict(doubled)
    res['marks_after_space_or_line_start'] = dict(detached)
    res['ascii_full_stop_after_arabic_letter'] = dict(ascii_stop)

    with open(os.path.join(OUT, 'variation.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, 'variation_out.txt'), 'w', encoding='utf-8') as f:
        for k, v in res.items():
            f.write('== %s\n%s\n' % (k, json.dumps(v, ensure_ascii=False)))
    for k, v in res.items():
        print('==', k)
        print(json.dumps(v, ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main()
