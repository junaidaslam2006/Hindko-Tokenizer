"""Review sheet: every gold entry with its own context (step 6 metric and most frequent
neighbours), so each word can be checked for homography by reading one line.

    python scripts/context_sheet.py        -> review/context_sheet_high.tsv, review/context_sheet_low.tsv

Columns: id, word, segmentation, category, freq, context metric, share of the word's tokens
that are sentence-initial / sentence-final, top 6 previous tokens, top 6 next tokens (with
counts, from data/context_profiles.json). Deterministic.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    prof = json.load(open(os.path.join(ROOT, 'data', 'context_profiles.json'), encoding='utf-8'))['words']
    for name in ('high', 'low'):
        rows = list(csv.DictReader(open(os.path.join(ROOT, f'morph_silver_{name}.tsv'), encoding='utf-8'),
                                   delimiter='\t'))
        out = os.path.join(ROOT, 'review', f'context_sheet_{name}.tsv')
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write('id\tword\tsegmentation\tcategory\tfreq\tcontext\tstart\tend\ttop_prev\ttop_next\n')
            for r in rows:
                p = prof.get(r['word'])
                if not p:
                    f.write(f"{r['id']}\t{r['word']}\t{r['segmentation']}\t{r['category']}\t{r['freq']}\t"
                            f"no profile\t\t\t\t\n")
                    continue
                n_prev = sum(p['prev'].values()) or 1
                n_next = sum(p['next'].values()) or 1
                f.write('\t'.join([
                    r['id'], r['word'], r['segmentation'], r['category'], r['freq'], r.get('context', ''),
                    f"{p['prev'].get('START', 0) / n_prev:.2f}", f"{p['next'].get('END', 0) / n_next:.2f}",
                    ' '.join(f'{t}:{c}' for t, c in p['top_prev'][:6]),
                    ' '.join(f'{t}:{c}' for t, c in p['top_next'][:6])]) + '\n')
        print(out, len(rows))


if __name__ == '__main__':
    main()
