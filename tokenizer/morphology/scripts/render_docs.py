"""Render MORPHOLOGY.md from MORPHOLOGY_template.md, filling every number and
table from the generated files so the documentation cannot drift from the data."""
from __future__ import annotations

import collections
import csv
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def rd(name):
    return list(csv.DictReader(open(os.path.join(ROOT, name), encoding='utf-8'), delimiter='\t'))


def fmt(x, nd=3):
    return 'n/a' if x is None else f'{x:.{nd}f}'


def metric_table(results, only_prefix=None, rename=None):
    keys = [('boundary_precision', 'P'), ('boundary_recall', 'R'), ('boundary_f1', 'F1'),
            ('morphscore', 'MorphScore'), ('morphscore_all', 'MorphScore (all)'), ('stem_intact', 'stem intact'),
            ('stem_boundary_respected', '**respected**'), ('exact_match', 'exact'),
            ('tokens_per_word', 'tok/word')]
    out = ['| set :: tokenizer | ' + ' | '.join(k[1] for k in keys) + ' |',
           '|---|' + '---|' * len(keys)]
    for name, r in results.items():
        if only_prefix and not name.startswith(only_prefix):
            continue
        label = name.replace('.tsv', '').replace('morph_silver_', '')
        if rename:
            label = rename(label)
        out.append(f'| {label} | ' + ' | '.join(fmt(r.get(k)) for k, _ in keys) + ' |')
    return '\n'.join(out)


def main():
    high, low = rd('morph_silver_high.tsv'), rd('morph_silver_low.tsv')
    review_all = rd('review/review_decisions.tsv')
    review = [r for r in review_all if not r['word'].startswith('#')]
    r2_start = next(i for i, r in enumerate(review_all) if r['word'] == '#round2')
    review_r2 = [r for r in review_all[r2_start + 1:] if not r['word'].startswith('#')]
    n_tests = sum(1 for ln in open(os.path.join(ROOT, 'tests', 'test_morph_eval.py'), encoding='utf-8')
                  if ln.startswith('def test_'))
    cands = rd('data/candidates_all.tsv')
    params = json.load(open(os.path.join(ROOT, 'data', 'induce_params.json'), encoding='utf-8'))
    uh = json.load(open(os.path.join(ROOT, 'data', 'urdu_vs_hindko_lines.json'), encoding='utf-8'))['meta']
    inv = json.load(open(os.path.join(ROOT, 'inventory', 'hindko_inflection_inventory.json'), encoding='utf-8'))
    table = open(os.path.join(ROOT, 'inventory', 'inventory_table.md'), encoding='utf-8').read().strip()
    bound, free = table.split('\n\n', 1)
    gt = params['corpus']['group_tokens']
    tot = params['corpus']['tokens']
    retro = inv['meta']['retroflex_n_counts']

    # statistics block
    cats = sorted({r['category'] for r in high + low})
    ch = collections.Counter(r['category'] for r in high)
    cl = collections.Counter(r['category'] for r in low)
    ex = {}
    for r in high:
        ex.setdefault(r['category'], [])
        if len(ex[r['category']]) < 3:
            ex[r['category']].append(r['segmentation'].replace('|', '\\|'))
    lines = ['| category | high | low | high examples |', '|---|---|---|---|']
    for c in cats:
        lines.append(f'| {c} | {ch[c]} | {cl[c]} | {" ".join(ex.get(c, []))} |')
    lines.append(f'| **total** | **{len(high)}** | **{len(low)}** | |')
    stats = ['\n'.join(lines), '']

    def dist(rows):
        return ', '.join(f'{k}: {v}' for k, v in sorted(collections.Counter(
            r['dialect'].split(' (')[0] for r in rows).items()))

    def stemlen(rows):
        return ', '.join(f'{k}: {v}' for k, v in sorted(collections.Counter(
            int(r['boundaries'].split(',')[0]) for r in rows).items()))

    def freqs(rows):
        f = sorted(int(r['freq']) for r in rows)
        return f'min {f[0]}, median {f[len(f) // 2]}, max {f[-1]}, sum {sum(f):,} tokens'
    reasons = collections.Counter()
    for r in low:
        for n in r['notes'].split(' | '):
            if n:
                reasons[n.split(':')[0].split(' (')[0]] += 1
    stats += [
        f'* Frequency in the strict corpus: high {freqs(high)}; low {freqs(low)}.',
        f'* Stem length in letters: high {stemlen(high)}; low {stemlen(low)}.',
        f'* Words with two required boundaries (causatives): high {sum(1 for r in high if "," in r["boundaries"])}, '
        f'low {sum(1 for r in low if "," in r["boundaries"])}. Words with optional boundaries: high '
        f'{sum(1 for r in high if r["optional_boundaries"])}, low {sum(1 for r in low if r["optional_boundaries"])}.',
        f'* Per-word dialect label: high — {dist(high)}; low — {dist(low)}.',
        f'* Why low entries are low (a word can have several reasons): '
        + ', '.join(f'{k} {v}' for k, v in reasons.most_common()) + '.',
        f'* Candidate pool before quotas: {params["decisions"].get("high", 0)} high-eligible, '
        f'{params["decisions"].get("low", 0)} low-eligible, {params["decisions"].get("drop", 0)} dropped '
        f'(of {params["n_analysed"]} analysed, {params["n_candidates"]} candidates). '
        f'Most high-eligible words were not selected only because of the quotas; `data/candidates_all.tsv` '
        f'lists them, but only the selected ones (and every word that entered the selection during the '
        f'review) were reviewed.',
    ]

    trivial = json.load(open(os.path.join(ROOT, 'eval_results', 'trivial_tokenizers.json'), encoding='utf-8'))
    existing_path = os.path.join(ROOT, 'eval_results', 'existing_tokenizers_illustration.json')
    existing = json.load(open(existing_path, encoding='utf-8')) if os.path.exists(existing_path) else {}
    rename = lambda s: s.replace('tokenizers:', '').replace('/tokenizer.json', '').replace(' +prefix-space', '')

    subs = {
        'DATE': datetime.date.today().isoformat(),
        'N_HIGH': len(high), 'N_LOW': len(low), 'N_LOW_ALT': sum(1 for r in low if r['alternatives']),
        'N_INV': len(inv['rows']), 'N_REVIEW': len(review),
        'N_REVIEW_DROP': sum(1 for r in review if r['action'] == 'drop'),
        'N_REVIEW_LOW': sum(1 for r in review if r['action'] == 'low'),
        'N_REVIEW_ALTOK': sum(1 for r in review if r['action'] == 'alt_ok'),
        'N_REVIEW_NOALT': sum(1 for r in review if r['action'] == 'noalt'),
        'N_REVIEW_WORDS': len({r['word'] for r in review}),
        'N_R2': len(review_r2), 'N_R2_DROP': sum(1 for r in review_r2 if r['action'] == 'drop'),
        'N_R1': len(review) - len(review_r2),
        'N_R2_UPDATED': sum(1 for r in review if '|| round 2:' in r['reason']),
        'N_CTX_FLAGGED': params['n_context_flagged'], 'N_SUBMIN': params['n_submin_dropped'],
        'N_TESTS': n_tests,
        'N_CTX_HIGH_VERBS': sum(1 for r in high if r['category'].startswith('V.')),
        'N_LOW_CTX': sum(1 for r in low if 'context check failed' in r['notes']),
        'N_ANALYSED': params['n_analysed'], 'N_CAND': params['n_candidates'],
        'RETRO_NR': f"{retro['ن+ڑ']:,}", 'RETRO_0768': retro['U+0768'], 'RETRO_NT': retro['ن+ط'],
        'TOK_PESH': f"{gt['PESH']:,}", 'TOK_BOOK': f"{gt['BOOK']:,}", 'TOK_HAZ': f"{gt['HAZ']:,}",
        'TOK_OTHER': f"{gt['OTHER']:,}",
        'HI_LINES': f"{uh['hindko_lines']:,}", 'HI_TOKS': f"{uh['hindko_tokens']:,}",
        'UR_LINES': f"{uh['urdu_lines']:,}", 'UR_TOKS': f"{uh['urdu_tokens']:,}",
        'SHARE_PB': f"{100 * (gt['PESH'] + gt['BOOK']) / tot:.1f}", 'SHARE_H': f"{100 * gt['HAZ'] / tot:.1f}",
        'N_INSUFF_HIGH': sum(1 for r in high if r['dialect'].startswith('insufficient')),
        'INVENTORY_BOUND': bound, 'INVENTORY_FREE': free, 'STATS': '\n'.join(stats),
        'TRIVIAL': metric_table(trivial, rename=lambda s: s.replace(' :: ', ' :: ')),
        'EXISTING': metric_table(existing, only_prefix='morph_silver_high', rename=rename) if existing else
        '(not run)',
    }
    text = open(os.path.join(HERE, 'MORPHOLOGY_template.md'), encoding='utf-8').read()
    for k, v in subs.items():
        text = text.replace('{{' + k + '}}', str(v))
    assert '{{' not in text, text[text.index('{{'):text.index('{{') + 40]
    open(os.path.join(ROOT, 'MORPHOLOGY.md'), 'w', encoding='utf-8', newline='\n').write(text)
    print('MORPHOLOGY.md', len(text), 'chars')


if __name__ == '__main__':
    main()
