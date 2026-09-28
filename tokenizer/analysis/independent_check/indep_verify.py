"""Independent re-computation of the PLAN section 6-7 decision from the raw per-document LM result files.

Written from PLAN.md only; imports nothing from analysis/decide.py or analysis/report.py.
Reads: lm/colab_results/<bundle>/results/*.json (dev per-document bits/bytes/ntok),
       splits/split_manifest.jsonl (dev uids only; any other row is skipped on its uid before JSON parsing).
Never reads the test split.
"""
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = r"F:\Hindko\_tokenizer"
RES = os.path.join(ROOT, "lm", "colab_results")
MAN = os.path.join(ROOT, "splits", "split_manifest.jsonl")
OUT = os.path.join(ROOT, "analysis", "independent_check", "indep_result.json")
BASE = "A1-P1r3-D2-16k"
RANK8 = "A6-SBPE-P1-D1-32k-t080"
R1_BUNDLE = "1d24425d2d64"
R2_BUNDLE = "77e1368773fc"
NREP = 10000
SEED = 12345


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- load runs
runs = []  # dicts: bundle, stage, cand, seed, lr, bits(np), bytes(np), ntok(np), uids, bpb_stored
for bundle in (R1_BUNDLE, R2_BUNDLE):
    for fp in sorted(glob.glob(os.path.join(RES, bundle, "results", "*.json"))):
        d = json.load(open(fp, encoding="utf-8"))
        if d.get("status") != "ok":
            print("NOT OK:", fp, d.get("status"))
            continue
        dv = d["dev"]
        runs.append(dict(bundle=bundle, stage=d["stage"], cand=d["candidate"], seed=d["seed"], lr=d["lr"],
                         bits=np.asarray(dv["bits"], dtype=np.float64),
                         bytes=np.asarray(dv["bytes"], dtype=np.int64),
                         ntok=np.asarray(dv["ntok"], dtype=np.int64),
                         uids=tuple(dv["uids"]), bpb_stored=d["bpb"], doc_set=dv.get("doc_set"),
                         n_docs=dv["n_docs"], file=os.path.basename(fp)))
print("runs loaded:", len(runs))

uids0 = runs[0]["uids"]
bytes0 = runs[0]["bytes"]
for r in runs:
    assert r["uids"] == uids0, ("uid order differs", r["file"])
    assert np.array_equal(r["bytes"], bytes0), ("bytes differ", r["file"])
    assert r["n_docs"] == 836 and r["doc_set"] == "all"
    rec = r["bits"].sum() / r["bytes"].sum()
    assert abs(rec - r["bpb_stored"]) < 1e-12, (r["file"], rec, r["bpb_stored"])
print("all runs: same 836 uids in same order, identical per-doc bytes; stored bpb == sum(bits)/sum(bytes)")
print("dev bytes total:", int(bytes0.sum()))

# ---------------------------------------------------------------- duplicates across bundles
by_key = defaultdict(list)
for r in runs:
    by_key[(r["stage"], r["cand"], r["seed"], r["lr"])].append(r)
dup_report = []
for k, lst in sorted(by_key.items()):
    if len(lst) > 1:
        a, b = lst
        same = np.array_equal(a["bits"], b["bits"]) and np.array_equal(a["ntok"], b["ntok"])
        dup_report.append((k, same))
n_dup = len(dup_report)
n_dup_same = sum(1 for _, s in dup_report if s)
print(f"cross-bundle duplicate runs: {n_dup}, bitwise identical bits+ntok: {n_dup_same}")
for k, s in dup_report:
    if not s:
        print("  NOT identical:", k)


def pick(stage, lr=0.001):
    """cand -> {seed: run}; round-1 bundle wins for duplicated reference arms."""
    out = defaultdict(dict)
    for r in runs:
        if r["stage"] != stage or abs(r["lr"] - lr) > 1e-12:
            continue
        if r["seed"] in out[r["cand"]] and r["bundle"] == R2_BUNDLE:
            continue
        out[r["cand"]][r["seed"]] = r
    return out


confirm = pick("confirm")
screen = pick("screen")
print("confirm candidates:", len(confirm), {c: sorted(s) for c, s in confirm.items()})

# ---------------------------------------------------------------- clusters from manifest (dev uids only)
man_sha = sha256(MAN)
dev_uids = set(uids0)
man = {}
skipped = 0
with open(MAN, encoding="utf-8") as f:
    for line in f:
        # uid is the first field: {"uid": "xxxxxxxxxxxxxxxx", ...  -> check before parsing
        u = line[9:25]
        if u not in dev_uids:
            skipped += 1
            continue
        m = json.loads(line)
        assert m["uid"] == u
        man[u] = m
assert len(man) == 836, len(man)
for u, m in man.items():
    assert m["split"] == "validation", m["split"]
    assert m["quality_tier"] == "strict", m["quality_tier"]
print(f"manifest sha256 {man_sha[:12]}...; dev rows matched {len(man)}; other rows skipped unparsed: {skipped}")


def boot_cluster(g):
    # PLAN 6: per-record web groups collapsed to their site
    if g.startswith("web:") and ":record:" in g:
        return g.split(":record:")[0]
    return g


def cluster_index(collapse=True):
    src = [man[u]["source"] for u in uids0]
    cl = [boot_cluster(man[u]["group"]) if collapse else man[u]["group"] for u in uids0]
    keys = sorted(set(zip(src, cl)))
    kidx = {k: i for i, k in enumerate(keys)}
    doc2c = np.array([kidx[(s, c)] for s, c in zip(src, cl)])
    csrc = [k[0] for k in keys]
    return doc2c, csrc, keys


# ---------------------------------------------------------------- bootstrap machinery
class Boot:
    def __init__(self, cands, seeds_of, collapse=True, nrep=NREP, seed=SEED):
        self.doc2c, self.csrc, self.keys = cluster_index(collapse)
        C = len(self.keys)
        self.C = C
        self.cbytes = np.bincount(self.doc2c, weights=bytes0.astype(np.float64), minlength=C)
        rng = np.random.default_rng(seed)
        # stratified cluster resampling: counts per cluster, per replicate
        R = np.zeros((nrep, C), dtype=np.float64)
        for s in sorted(set(self.csrc)):
            idx = np.array([i for i, x in enumerate(self.csrc) if x == s])
            draws = rng.integers(0, len(idx), size=(nrep, len(idx)))
            for j in range(len(idx)):
                R[:, idx[j]] += (draws == j).sum(axis=1)
        self.R = R
        denom = R @ self.cbytes  # (nrep,)
        self.rep = {}
        self.point = {}
        for c in cands:
            seeds = seeds_of(c)
            S = len(seeds)
            cb = np.stack([np.bincount(self.doc2c, weights=confirm[c][s]["bits"], minlength=C) for s in seeds])  # S x C
            per_seed = (R @ cb.T) / denom[:, None]  # nrep x S
            sd = rng.integers(0, S, size=(nrep, S))
            w = np.zeros((nrep, S))
            for j in range(S):
                w[:, j] = (sd == j).sum(axis=1)
            self.rep[c] = (per_seed * w).sum(axis=1) / S
            self.point[c] = float(np.mean([confirm[c][s]["bits"].sum() / bytes0.sum() for s in seeds]))

    def delta(self, a, b):
        """a - b: point, CI (bpb), pct point, pct CI (replicate-wise ratio), p two-sided, 90% CI."""
        d = self.rep[a] - self.rep[b]
        dp = 100.0 * d / self.rep[b]
        pt = self.point[a] - self.point[b]
        n = len(d)
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        p1 = 2 * min(((d <= 0).sum() + 1) / (n + 1), ((d >= 0).sum() + 1) / (n + 1))
        return dict(delta=pt, delta_pct=100 * pt / self.point[b],
                    ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    ci95_pct=[float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))],
                    ci95_pct_pointdenom=[float(100 * np.percentile(d, 2.5) / self.point[b]),
                                         float(100 * np.percentile(d, 97.5) / self.point[b])],
                    ci90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                    p=float(min(1.0, p)), p_plus1=float(min(1.0, p1)))


def holm(pvals):
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    adj = {}
    run = 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def decide(cands, seeds_of, collapse=True, label=""):
    bt = Boot(cands, seeds_of, collapse)
    means = bt.point
    order = sorted(cands, key=lambda c: means[c])
    best = order[0]
    delta_eq = 0.003 * means[BASE]
    raw = {}
    cmp = {}
    for c in order[1:]:
        cmp[c] = bt.delta(c, best)
        raw[c] = cmp[c]["p"]
    adj = holm(raw)
    top = [best]
    tost = {}
    for c in order[1:]:
        lo, hi = cmp[c]["ci90"]
        tost[c] = (lo > -delta_eq) and (hi < delta_eq)
        if adj[c] >= 0.05 or tost[c]:
            top.append(c)
    vs_base = {c: bt.delta(c, BASE) for c in order if c != BASE}
    return dict(label=label, n_clusters=bt.C, order=order, means=means, best=best, cmp=cmp, holm=adj,
                tost=tost, top=top, vs_base=vs_base, delta_eq=delta_eq, boot=bt)


def seeds_all(c):
    return sorted(confirm[c])


def seeds_123(c):
    return [s for s in sorted(confirm[c]) if s <= 3]


# ---------------------------------------------------------------- rank-8 condition (Stage 3 screen)
def smean(c):
    return float(np.mean([screen[c][s]["bits"].sum() / bytes0.sum() for s in sorted(screen[c])])), sorted(screen[c])


s_base, sb = smean(BASE)
s_r2, s2 = smean("A6-SBPE-P1-D1-16k-t090")
s_r5, s5 = smean("A6-SBPE-P1-D1-16k-t080")
rank8_ok = (s_r2 < s_base) or (s_r5 < s_base)
print(f"\nStage 3 screen means: baseline {s_base:.5f} seeds{sb}; rank2 t090 {s_r2:.5f}; rank5 t080 {s_r5:.5f} -> rank-8 condition {rank8_ok}")

all_c = sorted(confirm)
family = [c for c in all_c if not (c == RANK8 and not rank8_ok)]
print("family size k =", len(family))

main = decide(family, seeds_all, True, "main")
bt = main["boot"]
print(f"\nclusters: {bt.C}; per source:", {s: bt.csrc.count(s) for s in sorted(set(bt.csrc))})
print(f"delta (TOST margin) = {main['delta_eq']:.6f} bpb")
print(f"\n{'rank':>4} {'candidate':34} {'mean':>8} {'sd%':>6} {'d%base':>7} {'CI95%':>18} {'p_raw':>7} {'p_holm':>7} TOST")
# ranking includes rank-8 for display
disp = decide(all_c, seeds_all, True, "display_all")  # only for vs-baseline numbers of rank 8
for i, c in enumerate(sorted(all_c, key=lambda c: disp["means"][c])):
    m = disp["means"][c]
    sds = np.std([confirm[c][s]["bits"].sum() / bytes0.sum() for s in sorted(confirm[c])], ddof=1)
    vb = disp["vs_base"].get(c)
    dpc = vb["delta_pct"] if vb else 0.0
    ci = vb["ci95_pct"] if vb else [0, 0]
    pr = main["cmp"].get(c, {}).get("p", float("nan"))
    ph = main["holm"].get(c, float("nan"))
    ts = main["tost"].get(c, "")
    print(f"{i+1:>4} {c:34} {m:8.5f} {100*sds/m:6.3f} {dpc:+7.3f} [{ci[0]:+7.3f},{ci[1]:+7.3f}] {pr:7.4f} {ph:7.4f} {ts}")
print("top set:", main["top"])
print("best:", main["best"])

# ---------------------------------------------------------------- tie-breakers (b): dev bytes/token from LM ntok (minus EOT per doc)
print("\nTie-breaker (b) dev bytes/token (sum bytes / (sum ntok - n_docs)), seed-1 confirm run:")
bpt = {}
for c in main["top"] + [BASE]:
    r = confirm[c][1]
    bpt[c] = float(bytes0.sum() / (r["ntok"].sum() - len(r["ntok"])))
    consistent = all(np.array_equal(confirm[c][s]["ntok"], r["ntok"]) for s in confirm[c])
    print(f"  {c:34} {bpt[c]:.4f}  ntok identical across seeds: {consistent}")

# ---------------------------------------------------------------- improvement claim
print("\nImprovement claims (vs baseline", BASE, "):")
for c in main["top"]:
    v = main["vs_base"][c]
    print(f"  {c:34} d={v['delta_pct']:+.3f}% CI95 [{v['ci95_pct'][0]:+.3f},{v['ci95_pct'][1]:+.3f}] (pointdenom [{v['ci95_pct_pointdenom'][0]:+.3f},{v['ci95_pct_pointdenom'][1]:+.3f}]) p={v['p']:.5f} p+1={v['p_plus1']:.5f}")
chosen = "R2-A10-MinGram-P1r3-D2-48k"
v = main["vs_base"][chosen]
excl0 = v["ci95"][1] < 0 or v["ci95"][0] > 0
print("chosen CI excludes 0:", excl0)
# vs the PLAN-text standard recipe (A1-P1-D1-16k), sensitivity
b2 = main["boot"].delta(chosen, "A1-P1-D1-16k")
print(f"  sensitivity: chosen vs A1-P1-D1-16k (PLAN-text standard recipe): d={b2['delta_pct']:+.3f}% CI95 [{b2['ci95_pct'][0]:+.3f},{b2['ci95_pct'][1]:+.3f}] p={b2['p']:.5f}")

# ---------------------------------------------------------------- sensitivities
def short(res):
    return dict(best=res["best"], top=res["top"],
                holm={c: round(res["holm"][c], 4) for c in res["order"][1:5]},
                chosen_vs_base=None if chosen not in res["vs_base"] else
                [round(res["vs_base"][chosen]["delta_pct"], 3), [round(x, 3) for x in res["vs_base"][chosen]["ci95_pct"]]])


sens = {}
sens["raw_manifest_groups"] = decide(family, seeds_all, False, "raw_groups")
sens["rank8_included"] = decide(all_c, seeds_all, True, "rank8")
sens["seeds_1_3"] = decide(family, seeds_123, True, "seeds123")
r1 = [c for c in family if not c.startswith("R2-")]
sens["round1_only"] = decide(r1, seeds_all, True, "round1")
for k, res in sens.items():
    print(f"\nSensitivity {k}: clusters={res['n_clusters']} k={len(res['order'])}")
    print("  best:", res["best"], f"{res['means'][res['best']]:.5f}")
    print("  top set:", res["top"])
    for c in res["order"][1:6]:
        print(f"    {c:34} p_raw={res['cmp'][c]['p']:.4f} p_holm={res['holm'][c]:.4f} tost={res['tost'][c]}")
    b = res["best"]
    if b != BASE:
        vb = res["vs_base"][b]
        print(f"  best vs base: {vb['delta_pct']:+.3f}% [{vb['ci95_pct'][0]:+.3f},{vb['ci95_pct'][1]:+.3f}]")
    if chosen in res["vs_base"]:
        vb = res["vs_base"][chosen]
        print(f"  chosen vs base: {vb['delta_pct']:+.3f}% [{vb['ci95_pct'][0]:+.3f},{vb['ci95_pct'][1]:+.3f}]")

# ---------------------------------------------------------------- RNG-seed robustness of the main numbers
print("\nMonte-Carlo stability (5 other RNG seeds): chosen-vs-base CI and Holm p of the top set")
for sd in (1, 2, 3, 4, 5):
    b = Boot(family, seeds_all, True, seed=sd)
    d = b.delta(chosen, BASE)
    ps = {c: b.delta(c, main["best"])["p"] for c in main["order"][1:]}
    h = holm(ps)
    print(f"  seed {sd}: CI [{d['ci95_pct'][0]:+.3f},{d['ci95_pct'][1]:+.3f}]  holm: " +
          ", ".join(f"{c.split('-',1)[1][:22]}={h[c]:.3f}" for c in main["order"][1:4]))

# ---------------------------------------------------------------- LR sweep / resweep
print("\nLR sweep (screen, baseline, seed 1):")
for r in runs:
    if r["stage"] == "lrsweep":
        print(f"  {r['cand']} lr={r['lr']:g} bpb={r['bits'].sum()/bytes0.sum():.5f}")
print("Finalist re-sweep (confirm, seed 1):")
for r in sorted(runs, key=lambda r: (r["cand"], r["lr"])):
    if r["stage"] == "confirm_resweep":
        b1 = confirm[r["cand"]][1]["bits"].sum() / bytes0.sum()
        print(f"  {r['cand']:34} lr={r['lr']:g} bpb={r['bits'].sum()/bytes0.sum():.5f} (lr1e-3 s1 {b1:.5f}) 1e-3 better: {b1 < r['bits'].sum()/bytes0.sum()}")

# ---------------------------------------------------------------- write
out = dict(
    manifest_sha256=man_sha, n_runs=len(runs), cross_bundle_duplicates=n_dup, duplicates_bitwise_identical=n_dup_same,
    rank8_condition=dict(screen_base=s_base, screen_rank2=s_r2, screen_rank5=s_r5, holds=rank8_ok),
    family=family, n_clusters=main["n_clusters"], delta_eq=main["delta_eq"],
    ranking=[dict(id=c, mean=disp["means"][c],
                  delta_vs_base_pct=(disp["vs_base"][c]["delta_pct"] if c != BASE else 0.0),
                  ci95_pct=(disp["vs_base"][c]["ci95_pct"] if c != BASE else [0, 0]),
                  p_raw_vs_best=main["cmp"].get(c, {}).get("p"), p_holm_vs_best=main["holm"].get(c),
                  tost=main["tost"].get(c))
             for c in sorted(all_c, key=lambda c: disp["means"][c])],
    top_set=main["top"], bytes_per_token=bpt,
    chosen_vs_base=main["vs_base"][chosen],
    sensitivities={k: short(v) for k, v in sens.items()},
)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nwritten", OUT)
