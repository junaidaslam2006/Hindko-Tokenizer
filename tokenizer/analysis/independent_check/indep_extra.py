"""Extra report-only checks, reusing indep_verify.py's own (independent) machinery via runpy."""
import io
import contextlib
import runpy
import itertools
import numpy as np

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(r"F:\Hindko\_tokenizer\analysis\independent_check\indep_verify.py")
main, confirm, screen, bytes0, Boot, BASE = g["main"], g["confirm"], g["screen"], g["bytes0"], g["Boot"], g["BASE"]
all_c = g["all_c"]
bt = Boot(all_c, g["seeds_all"], True)
deq = main["delta_eq"]
best = main["best"]

print(f"TOST (90 pct CI of delta vs best, margin +/-{deq:.6f}):")
for c in main["order"][1:5]:
    d = bt.delta(c, best)
    print(f"  {c:32} ci90 [{d['ci90'][0]:+.6f},{d['ci90'][1]:+.6f}] inside: {d['ci90'][0] > -deq and d['ci90'][1] < deq}")

pairs = [("A1-P1r3-D2-16k", "A1-P1r3-D2-8k", "8k->16k"), ("A1-P1r3-D2-32k", "A1-P1r3-D2-16k", "16k->32k"),
         ("R2-A1-P1r3-D2-48k", "A1-P1r3-D2-32k", "32k->48k"), ("R2-A4-SPnat-D2-48k", "R2-A4-SPnat-D2-32k", "SP-Uni 32k->48k"),
         ("A1-P1r3-D2-16k", "A1-P1r3-D1-16k", "D2 vs D1 (A1 16k)"), ("A1-P1r3-D1-16k", "A1-P1-D1-16k", "P1r3 vs P1"),
         ("A10-MinGram-P1-D1-16k", "A1-P1-D1-16k", "MinGram vs BPE 16k"), ("R2-A10-MinGram-P1r3-D2-32k", "A1-P1r3-D2-32k", "MinGram vs BPE 32k"),
         ("R2-A10-MinGram-P1r3-D2-48k", "R2-A1-P1r3-D2-48k", "MinGram vs BPE 48k"),
         ("R2-A6-SBPE-P1r3-D2-32k-t080", "A1-P1r3-D2-32k", "SuperBPE vs base 32k R2"),
         ("A6-SBPE-P1-D1-16k-t090", "A1-P1-D1-16k", "SuperBPE t090 vs base 16k"),
         ("A6-SBPE-P1-D1-16k-t080", "A1-P1-D1-16k", "SuperBPE t080 vs base 16k")]
print("\nReport-only contrasts (a - b, % of b):")
for a, b, lab in pairs:
    d = bt.delta(a, b)
    print(f"  {lab:28} {d['delta_pct']:+.3f}% [{d['ci95_pct'][0]:+.3f},{d['ci95_pct'][1]:+.3f}] p={d['p']:.4f}")

# Stage 3 vs Stage 4 concordance
s3 = {c: np.mean([screen[c][s]["bits"].sum() / bytes0.sum() for s in screen[c]]) for c in all_c}
s4 = main["boot"].point if False else {c: np.mean([confirm[c][s]["bits"].sum() / bytes0.sum() for s in confirm[c]]) for c in all_c}


def tau_b(cs):
    conc = disc = 0
    flips = []
    for a, b in itertools.combinations(cs, 2):
        x = np.sign(s3[a] - s3[b]); y = np.sign(s4[a] - s4[b])
        if x * y > 0:
            conc += 1
        elif x * y < 0:
            disc += 1; flips.append((a, b))
    n = len(list(itertools.combinations(cs, 2)))
    return (conc - disc) / n, disc, n, flips


t, dsc, n, flips = tau_b(all_c)
t2, _, _, _ = tau_b([c for c in all_c if "SBPE" not in c])
print(f"\nKendall tau_b stage3 vs stage4: {t:.3f} (flips {dsc}/{n}); without SuperBPE {t2:.3f}")
r3 = sorted(all_c, key=lambda c: s3[c]); r4 = sorted(all_c, key=lambda c: s4[c])
for c in ("R2-A6-SBPE-P1r3-D2-32k-t080", "A6-SBPE-P1-D1-32k-t080"):
    print(f"  {c}: stage3 rank {r3.index(c)+1} -> stage4 rank {r4.index(c)+1}")

# power figures
def sdpct(c, seeds):
    v = [confirm[c][s]["bits"].sum() / bytes0.sum() for s in seeds]
    return np.std(v, ddof=1) / np.mean(v) * 100
r1c = [c for c in all_c if not c.startswith("R2-")]; r2c = [c for c in all_c if c.startswith("R2-")]
pool = lambda cs, seeds: float(np.sqrt(np.mean([sdpct(c, seeds) ** 2 for c in cs])))
print(f"\npooled seed s.d. %, stage 4 seeds 1-3: round1 {pool(r1c,[1,2,3]):.3f} round2 {pool(r2c,[1,2,3]):.3f}; all 5 seeds all 18: {pool(all_c,[1,2,3,4,5]):.3f}")
