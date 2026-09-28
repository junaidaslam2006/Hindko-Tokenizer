import json, re, sys
d = json.load(open('F:/Hindko/_tokenizer/morphology/data/wordfreq_strict.json', encoding='utf-8'))
F = {r['form']: r for r in d['rows']}
def show(words):
    for w in words:
        r = F.get(w)
        if r: print(f"{w}\t{r['freq']}\tP{r['PESH']}\tB{r['BOOK']}\tH{r['HAZ']}\tO{r['OTHER']}\tu{r['n_units']}")
        else: print(f"{w}\t0")
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'w':
        show(sys.argv[2:])
    elif mode == 're':
        pat = re.compile(sys.argv[2]); lim = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        n = 0
        for r in d['rows']:
            if pat.fullmatch(r['form']):
                show([r['form']]); n += 1
                if n >= lim: break
