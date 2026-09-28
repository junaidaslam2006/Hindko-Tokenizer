#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py - reassemble + verify the bundle, then run the tokenizer LM arbiter on a GPU (or CPU).

    python run_all.py --parts /content/upload                 # everything, resumable
    python run_all.py --parts /content/upload --stages parity # only the CPU<->GPU sanity check
    python run_all.py --parts DIR --estimate                  # FLOP-based runtime estimate, no training

Stages (in this order). Resuming: a run is skipped only if its result file exists AND the stored
record matches the requested run exactly (bundle, hk_lm.py sha256, stage, recipe, candidate, seed,
exact peak LR, model, budget bytes, steps, ctx, overrides, device + AMP dtype); any mismatch stops
the script instead of silently reusing the record. Result keys carry the LR losslessly (hk_lm.lr_tag).
  parity   the fixed tiny config of PARITY.json (100 steps, seed 1, baseline) on this device, with the
           production AMP dtype and in fp32; compared with the CPU reference (tolerance 1 % bpb).
  lr       PLAN 5 LR sweep on the baseline: {1e-3, 3e-3, 6e-3} x seed 1 x screening budget.
           The LR with the lowest full-dev bpb is written to lr_choice.json and used everywhere.
  screen   Stage 3: every candidate x seeds {1,2,3}, d=128, 10 MB.
  confirm  Stage 4 (d=192, full train split, 1 epoch) for EVERY candidate (PLAN: top 2 + baseline;
           the GPU makes all of them cheap) x seeds {1,2,3}; + seeds {4,5} for every candidate if
           the PLAN 6 power check says so (pooled seed s.d. > 0.2 % of bpb); then the PLAN 5
           finalist LR re-sweep (report-only) if the top two differ by < 1 %.
  large    ADDITION beyond the pre-registration (PLAN 8(ii)): d=384 L=6, full train x 2 epochs,
           own LR sweep {5e-4, 1e-3, 2e-3} on the baseline (seed 1), then every candidate x
           seeds {1,2}. Report-only.
Results: <out>/<bundle sha12>/results/*.json (one per stage, candidate, seed, lr), summary.json,
summary.txt, progress.log and a results zip. <out> defaults to Google Drive if mounted
(/content/drive/MyDrive/hindko_lm_out) else /content/out (or ./out off Colab).

--force-extra-seeds: confirm always runs seeds 4 and 5 for every candidate (the power check is still
written, report-only). FINAL-TEST bundles (manifest kind 'final_test', build_bundle.py --final-test;
their 'dev' set is the strict TEST split) run only --stages parity,confirm, need --lr and
--force-extra-seeds, and never run the finalist LR re-sweep: see colab/FINAL_TEST.md.
REPORT-ONLY test bundles (kind 'final_test_report', build_bundle.py --final-test --report-only; also the
strict TEST split) run only --stages parity,confirm and need --lr; confirm runs seeds 1-3 (the power check
is written but never adds seeds; --force-extra-seeds adds 4 and 5); no re-sweep, no large stage, no
--smoke: see colab/FRONTIER_LM.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import time

STAGE_ORDER = ("parity", "lr", "screen", "confirm", "large")
FINAL_TEST_STAGES = ("parity", "confirm")     # the only stages a kind='final_test' bundle may run
FINAL_KINDS = ("final_test", "final_test_report")   # bundles whose 'dev' set is the strict TEST split
SEEDS_SCREEN = (1, 2, 3)
SEEDS_CONFIRM = (1, 2, 3)
SEEDS_CONFIRM_EXTRA = (4, 5)
SEEDS_LARGE = (1, 2)
LARGE_LR_GRID = (5e-4, 1e-3, 2e-3)
POWER_REL_SD = 0.002          # PLAN 6: sigma <= 0.2 % of bpb -> 3 seeds suffice
DELTA_MIN = 0.005             # PLAN 6: detect 0.5 % at alpha 0.05, power 0.8 -> n = ceil(15.7 s^2 / d^2)
RESWEEP_GAP = 0.01            # PLAN 5: re-sweep the finalists if their gap is < 1 %
PARITY_TOL = 0.01
HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = ("/content/upload", "/content", "/content/drive/MyDrive/hindko_lm_upload",
               "/content/drive/MyDrive", HERE, os.getcwd())

# assumed *effective* training throughput for these tiny models (ESTIMATE, not measured)
EFFECTIVE_TFLOPS = {"T4": 4.0, "L4": 8.0, "A100": 20.0, "CPU": 0.02}


class Deadline(Exception):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%H:%M:%S")


class Logger:
    def __init__(self):
        self.path = None

    def __call__(self, msg):
        line = "%s %s" % (now(), msg)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


LOG = Logger()


# ============================================================ bundle reassembly
def find_parts_dir(explicit=None):
    dirs = [explicit] if explicit else list(SEARCH_DIRS)
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, "PARTS.json")):
            return os.path.abspath(d)
    raise SystemExit("PARTS.json not found in %s - upload the parts, PARTS.json and run_all.py into one folder"
                     % (dirs,))


def reassemble(parts_dir, work):
    with open(os.path.join(parts_dir, "PARTS.json"), encoding="utf-8") as f:
        P = json.load(f)
    msha = P["bundle_manifest_sha256"]
    dest = os.path.join(work, msha[:12])
    bdir = os.path.join(dest, "bundle")
    marker = os.path.join(dest, "VERIFIED")
    if os.path.isfile(marker) and os.path.isfile(os.path.join(bdir, "manifest.json")) \
            and sha256_file(os.path.join(bdir, "manifest.json")) == msha:
        LOG("bundle %s already extracted and verified at %s" % (msha[:12], bdir))
        return P, bdir
    bad = []
    for p in P["parts"]:
        fp = os.path.join(parts_dir, p["name"])
        if not os.path.isfile(fp):
            bad.append("%s missing" % p["name"])
        elif os.path.getsize(fp) != p["bytes"]:
            bad.append("%s size %d != %d" % (p["name"], os.path.getsize(fp), p["bytes"]))
        elif sha256_file(fp) != p["sha256"]:
            bad.append("%s sha256 mismatch" % p["name"])
    if bad:
        raise SystemExit("PART VERIFICATION FAILED (re-upload these): " + "; ".join(bad))
    LOG("all %d parts verified (sha256)" % len(P["parts"]))
    os.makedirs(dest, exist_ok=True)
    arch = os.path.join(dest, P["archive"])
    h = hashlib.sha256()
    with open(arch, "wb") as fo:
        for p in P["parts"]:
            with open(os.path.join(parts_dir, p["name"]), "rb") as fi:
                blk = fi.read()
            h.update(blk)
            fo.write(blk)
    if h.hexdigest() != P["archive_sha256"]:
        raise SystemExit("reassembled archive sha256 mismatch")
    if os.path.isdir(bdir):
        shutil.rmtree(bdir)
    with tarfile.open(arch, "r:xz") as tf:
        members = tf.getmembers()
        for m in members:
            n = m.name
            if n.startswith("/") or ".." in n.split("/") or not n.startswith("bundle/") or not (m.isfile() or m.isdir()):
                raise SystemExit("unsafe tar member %r" % n)
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:   # Python < 3.12
            tf.extractall(dest, members=members)
    os.remove(arch)
    got = sha256_file(os.path.join(bdir, "manifest.json"))
    if got != msha:
        raise SystemExit("manifest sha256 %s != PARTS.json %s" % (got, msha))
    hk_lm = import_bundled_hk_lm(bdir)
    n = hk_lm.Bundle(bdir).verify()
    with open(marker, "w") as f:
        f.write(msha + "\n")
    LOG("archive reassembled, extracted and %d bundle files verified (sha256) -> %s" % (n, bdir))
    return P, bdir


# ============================================================ helpers
def slug(s):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def mean_sd(xs):
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def import_bundled_hk_lm(bdir):
    """Import hk_lm.py from the extracted bundle (never a stray copy elsewhere on sys.path)."""
    bdir = os.path.abspath(bdir)
    old = sys.modules.get("hk_lm")
    if old is not None and os.path.normcase(os.path.dirname(os.path.abspath(old.__file__))) != os.path.normcase(bdir):
        del sys.modules["hk_lm"]
    if bdir in sys.path:
        sys.path.remove(bdir)
    sys.path.insert(0, bdir)
    import hk_lm
    if os.path.normcase(os.path.dirname(os.path.abspath(hk_lm.__file__))) != os.path.normcase(bdir):
        raise SystemExit("hk_lm imported from %s, not from the bundle %s" % (hk_lm.__file__, bdir))
    return hk_lm


def default_out():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/hindko_lm_out"
    if os.path.isdir("/content"):
        return "/content/out"
    return os.path.join(os.getcwd(), "out")


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ============================================================ runner
class Runner:
    def __init__(self, a, bundle_dir, P):
        hk_lm = import_bundled_hk_lm(bundle_dir)
        self.hk = hk_lm
        self.a = a
        self.bdir = bundle_dir
        self.b = hk_lm.Bundle(bundle_dir)
        self.msha = self.b.manifest_sha256
        self.root = os.path.join(a.out or default_out(), self.msha[:12] + ("_smoke" if a.smoke else ""))
        self.res = os.path.join(self.root, "results")
        os.makedirs(self.res, exist_ok=True)
        LOG.path = os.path.join(self.root, "progress.log")
        # Never mix training precisions inside one study: the first session pins device type + AMP dtype
        # (e.g. a T4 -> fp16); a later session on another GPU reuses the pinned dtype.
        pin_p = os.path.join(self.root, "runtime_pin.json")
        self.pin_p = pin_p
        pin = load_json(pin_p) if os.path.isfile(pin_p) else None
        amp = a.amp
        if pin and a.amp == "auto":
            amp = pin["amp_dtype"]
        try:
            self.rt = hk_lm.Runtime(a.device, threads=a.threads, amp=amp, log=LOG)
        except ValueError:      # e.g. pinned fp16 but no GPU now -> the mismatch check below explains
            self.rt = hk_lm.Runtime(a.device, threads=a.threads, amp=a.amp, log=LOG)
        if pin:
            if pin["device"] != self.rt.device.type or pin["amp_dtype"] != self.rt.amp:
                if not a.allow_mixed:
                    raise SystemExit("results in %s were trained on %s/%s; this session is %s/%s. Mixing precisions "
                                     "across candidates is not allowed (pass --allow-mixed to override, recorded per run)"
                                     % (self.root, pin["device"], pin["amp_dtype"], self.rt.device.type, self.rt.amp))
                LOG("WARNING: --allow-mixed: pinned %s/%s, now %s/%s" % (pin["device"], pin["amp_dtype"],
                                                                        self.rt.device.type, self.rt.amp))
            if self.rt.amp == "bf16" and self.rt.capability and self.rt.capability[0] < 8:
                LOG("WARNING: pinned bf16 on a GPU without native bf16 (emulated, slow)")
            if pin.get("gpu_name") != self.rt.gpu_name:
                LOG("note: GPU changed since the pin (%s -> %s); same dtype, recorded per run" % (pin.get("gpu_name"), self.rt.gpu_name))
        # (the pin itself is written by the first real training run, in self.run(); parity runs never pin)
        self.deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
        cands = self.b.candidate_ids
        if a.only:
            keep = set(a.only.split(","))
            cands = [c for c in cands if c in keep or c == self.b.baseline_id]
        self.cands = cands
        slugs = {}
        for c in self.b.candidate_ids:
            slugs.setdefault(slug(c), []).append(c)
        clash = [v for v in slugs.values() if len(v) > 1]
        if clash:
            raise SystemExit("candidate ids %s map to the same result-file name; rename them in WAVES.json" % clash)
        self.base = self.b.baseline_id
        self.final_test = self.b.m.get("kind") in FINAL_KINDS
        self.report_only = self.b.m.get("kind") == "final_test_report"
        self.n_new = 0
        self._checked = set()     # result files whose identity was verified in this session
        me = sha256_file(os.path.abspath(__file__))
        bundled = self.b.m["builder"]["run_all_sha256"]
        if me != bundled:
            LOG("WARNING: this run_all.py (sha %s) differs from the bundled copy (%s); the bundled hk_lm.py is used "
                "for training either way" % (me[:12], bundled[:12]))
        LOG("bundle %s | %d candidates, baseline %s | out %s" % (self.msha[:12], len(cands), self.base, self.root))
        if self.final_test:
            LOG("FINAL-TEST bundle: its 'dev' set is the strict TEST split; every 'dev' number below is a TEST number")
        if self.report_only:
            LOG("REPORT-ONLY test bundle (%s): no decision depends on these numbers"
                % self.b.m.get("final_test", {}).get("purpose", "")[:160])
        LOG("device %s" % json.dumps(self.rt.info()))

    # -- one run, resumable
    def overrides(self, recipe):
        if not self.a.smoke:
            return None
        steps = {"screen": 12, "confirm": 12, "large": 4}.get(recipe)
        if steps is None:
            return None
        return {"max_steps": steps, "eval": ("first", 10), "curve_eval": ("first", 5)}

    def key(self, label, cand, lr, seed):
        # lr_tag is lossless (shortest round-trip form), so different LRs never share a key
        return "%s__%s__lr%s__s%d" % (label, slug(cand), self.hk.lr_tag(lr), seed)

    def identity(self, label, recipe, cand, seed, lr, amp=None, overrides="auto"):
        return self.hk.run_identity(self.b, cand, recipe, seed, lr, self.rt.device.type, amp or self.rt.amp,
                                    label=label, overrides=self.overrides(recipe) if overrides == "auto" else overrides)

    def check_existing(self, path, rec, want):
        """A stored record may stand in for the requested run only if every identity field matches
        (bundle, hk_lm.py, stage, recipe, candidate, seed, exact LR, model, budget, steps, ctx,
        overrides; device/AMP dtype unless --allow-mixed). Anything else is refused, never skipped."""
        ck = (path, json.dumps(want, sort_keys=True))
        if ck in self._checked:
            return
        if rec.get("bundle_manifest_sha256") != self.msha:
            raise SystemExit("%s belongs to another bundle (%s); use another --out"
                             % (path, str(rec.get("bundle_manifest_sha256"))[:12]))
        bad = self.hk.identity_mismatches(rec, want)
        hard = [x for x in bad if x[0] not in self.hk.IDENTITY_SOFT]
        soft = [x for x in bad if x[0] in self.hk.IDENTITY_SOFT]
        if hard or (soft and not self.a.allow_mixed):
            raise SystemExit(
                "REFUSING TO RESUME: %s exists but was not produced by the run requested now:\n%s\n"
                "Nothing was skipped or overwritten. Move that file away, or use another --out%s."
                % (path, "\n".join("  %-22s stored %r, requested %r" % x for x in bad),
                   "" if hard else " (or pass --allow-mixed to reuse a run trained on another device/dtype)"))
        if soft:
            LOG("WARNING: --allow-mixed: reusing %s trained with %s" %
                (os.path.basename(path), ", ".join("%s=%s (now %s)" % x for x in soft)))
        self._checked.add(ck)

    def run(self, label, recipe, cand, seed, lr):
        lr = float(lr)
        key = self.key(label, cand, lr, seed)
        path = os.path.join(self.res, key + ".json")
        want = self.identity(label, recipe, cand, seed, lr)
        if os.path.isfile(path):
            rec = load_json(path)
            self.check_existing(path, rec, want)
            LOG("skip %s (exists, bpb %s)" % (key, "%.5f" % rec["bpb"] if rec.get("bpb") is not None else rec.get("status")))
            return rec
        if self.deadline and time.time() > self.deadline:
            raise Deadline()
        if not os.path.isfile(self.pin_p):
            self.hk.write_json_atomic(self.pin_p, {"device": self.rt.device.type, "amp_dtype": self.rt.amp,
                                                   "gpu_name": self.rt.gpu_name,
                                                   "pinned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            LOG("precision pinned for this study: %s / %s" % (self.rt.device.type, self.rt.amp))
        rec = self.hk.run_one(self.b, cand, recipe, seed, lr, self.rt, label=label, out_path=path,
                              overrides=self.overrides(recipe), log=LOG)
        bad = self.hk.identity_mismatches(rec, want)
        if bad:     # run_one and run_identity share _recipe/_plan_budget; this guards against drift
            raise SystemExit("internal error: the new record %s does not match its own identity: %s" % (path, bad))
        self._checked.add((path, json.dumps(want, sort_keys=True)))
        self.n_new += 1
        LOG("DONE %s bpb %s status %s %.0fs" % (key, "%.5f" % rec["bpb"] if rec["bpb"] is not None else "-",
                                               rec["status"], rec["wall_time_s"]["total"]))
        self.write_summary(quiet=True)
        return rec

    # -- stages
    def stage_parity(self):
        pp = os.path.join(self.bdir, "PARITY.json")
        if not os.path.isfile(pp):
            LOG("parity: bundle has no PARITY.json - skipped")
            return None
        P = load_json(pp)
        if P["bundle_manifest_sha256"] != self.msha:
            raise SystemExit("PARITY.json belongs to another bundle")
        cpu = P["cpu"]["bpb"]
        modes = [self.rt.amp] + (["fp32"] if self.rt.amp != "fp32" else [])
        rep = {"cpu_reference": {"bpb": cpu, "runtime": P["cpu"]["runtime"]}, "tolerance_rel": P.get("tolerance_rel_bpb", PARITY_TOL),
               "runs": {}}
        for amp in modes:
            path = os.path.join(self.root, "parity", "parity_%s_%s.json" % (self.rt.device.type, amp))
            if os.path.isfile(path):
                rec = load_json(path)
                self.check_existing(path, rec, self.identity("parity_%s" % amp, "parity", P["candidate"], 1,
                                                             P["config"]["lr"], amp=amp, overrides=None))
                LOG("skip parity %s (exists)" % amp)
            else:
                rt = self.rt if amp == self.rt.amp else self.hk.Runtime(self.a.device, threads=self.a.threads, amp=amp, log=LOG)
                rec = self.hk.run_one(self.b, P["candidate"], "parity", 1, P["config"]["lr"], rt,
                                      label="parity_%s" % amp, out_path=path, log=LOG)
            if rec.get("bpb") is None:
                raise SystemExit("PARITY run %s produced no bpb (status %s) - stop and investigate" % (amp, rec.get("status")))
            rel = (rec["bpb"] - cpu) / cpu
            rep["runs"][amp] = {"bpb": rec["bpb"], "rel_diff_vs_cpu": rel, "final_loss": rec["train_loss"][-1],
                                "cpu_final_loss": P["cpu"]["final_loss"], "wall_time_s": rec["wall_time_s"],
                                "bitwise_identical_to_cpu": rec["dev"]["bits"] == P["cpu"]["bits"]}
            LOG("PARITY %-4s on %s: bpb %.6f vs CPU %.6f -> %+.3f %% (tolerance %.1f %%)%s" %
                (amp, self.rt.device.type, rec["bpb"], cpu, 100 * rel, 100 * PARITY_TOL,
                 "" if abs(rel) <= PARITY_TOL else "  <-- OUTSIDE TOLERANCE"))
        rep["pass_fp32"] = abs(rep["runs"].get("fp32", rep["runs"][modes[0]])["rel_diff_vs_cpu"]) <= PARITY_TOL
        rep["pass_all"] = all(abs(r["rel_diff_vs_cpu"]) <= PARITY_TOL for r in rep["runs"].values())
        self.hk.write_json_atomic(os.path.join(self.root, "parity_report.json"), rep)
        if not rep["pass_fp32"] and not self.a.ignore_parity:
            raise SystemExit("PARITY FAILED in fp32: GPU and CPU disagree by more than 1 % bpb - stop and investigate "
                             "(or pass --ignore-parity)")
        if not rep["pass_all"]:
            LOG("WARNING: the AMP run is outside the 1 % parity tolerance (fp32 passed); continuing")
        return rep

    def current_lr(self):
        """The LR the pre-registered stages use in this session: --lr if given, else lr_choice.json, else None."""
        if self.a.lr:
            return float(self.a.lr)
        p = os.path.join(self.root, "lr_choice.json")
        if os.path.isfile(p):
            return float(load_json(p)["chosen_lr"])
        return None

    def chosen_lr(self, sweep=True):
        p = os.path.join(self.root, "lr_choice.json")
        if self.a.lr:
            lr = float(self.a.lr)
            if os.path.isfile(p) and float(load_json(p)["chosen_lr"]) != lr:
                LOG("note: --lr %s overrides lr_choice.json (%s); results are keyed by the exact LR"
                    % (self.hk.lr_tag(lr), self.hk.lr_tag(load_json(p)["chosen_lr"])))
            return lr
        if os.path.isfile(p):
            return float(load_json(p)["chosen_lr"])
        if not sweep:
            raise SystemExit("no lr_choice.json yet - run the 'lr' stage first (or pass --lr)")
        return self.stage_lr()

    def stage_lr(self):
        recs = [self.run("lrsweep", "screen", self.base, 1, lr) for lr in self.hk.LR_GRID]
        ok = [(r["bpb"], r["lr"]) for r in recs if r["status"] == "ok" and r["bpb"] is not None]
        if not ok:
            raise SystemExit("every LR diverged in the sweep")
        best = min(ok)
        lr = best[1]
        info = {"chosen_lr": lr, "grid": list(self.hk.LR_GRID), "candidate": self.base, "seed": 1, "budget": "screen",
                "rule": "PLAN 5: lowest full-dev bpb of the 3-point sweep on the baseline (1 seed, screening budget); "
                        "used for every candidate and every pre-registered stage",
                "table": [{"lr": r["lr"], "bpb": r["bpb"], "status": r["status"]} for r in recs],
                "at_grid_edge": lr in (min(self.hk.LR_GRID), max(self.hk.LR_GRID))}
        self.hk.write_json_atomic(os.path.join(self.root, "lr_choice.json"), info)
        LOG("LR CHOICE %s (bpb %.5f)%s" % (self.hk.lr_tag(lr), best[0], " - at the edge of the pre-registered grid" if info["at_grid_edge"] else ""))
        return lr

    def stage_screen(self):
        lr = self.chosen_lr()
        for c in self.cands:
            for s in SEEDS_SCREEN:
                self.run("screen", "screen", c, s, lr)
        # PLAN 6 literal power check "after the baseline's first wave" (report-only; the seeds 4/5
        # decision is taken on the Stage 4 confirm runs, see stage_confirm)
        m = self._stage_means("screen", lr).get(self.base)
        if m and m[1] is not None:
            rel = m[1] / m[0]
            n_req = math.ceil(15.7 * rel ** 2 / DELTA_MIN ** 2)
            self.hk.write_json_atomic(os.path.join(self.root, "power_check_screen.json"),
                                      {"candidate": self.base, "n": m[2], "mean_bpb": m[0], "sd_bpb": m[1], "rel_sd": rel,
                                       "seeds_needed_for_delta_0.5pct": n_req,
                                       "rule": "PLAN 6: sigma <= 0.2 % -> 3 seeds suffice; n = ceil(15.7 s^2 / 0.005^2)"})
            LOG("baseline Stage 3 seed s.d. %.3f %% of bpb -> seeds needed for 0.5 %%: %d" % (100 * rel, n_req))

    def _records(self, label, recipe, lr):
        """{candidate: [records]} of this stage label at exactly this LR, selected by the records'
        own fields (not by file names), each identity-checked like a resumed run."""
        lr = float(lr)
        out = {}
        cands = set(self.cands)
        for fn in sorted(os.listdir(self.res)):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(self.res, fn)
            try:
                r = load_json(p)
            except (ValueError, OSError):
                continue
            if r.get("stage") != label or r.get("candidate") not in cands or r.get("lr") is None \
                    or float(r["lr"]) != lr:
                continue
            self.check_existing(p, r, self.identity(label, recipe, r["candidate"], r.get("seed", -1), lr))
            out.setdefault(r["candidate"], []).append(r)
        for c in out:
            out[c].sort(key=lambda r: r["seed"])
        return out

    def _pooled_rel_sd(self, label, lr):
        per = {}
        recs = self._records(label, "confirm", lr)
        for c in self.cands:
            xs = [r["bpb"] for r in recs.get(c, []) if r["seed"] in SEEDS_CONFIRM and r.get("bpb") is not None]
            if len(xs) >= 2:
                m, sd = mean_sd(xs)
                per[c] = {"n": len(xs), "mean": m, "sd": sd, "rel_sd": sd / m}
        if not per:
            return None, per
        pooled = math.sqrt(sum(v["rel_sd"] ** 2 for v in per.values()) / len(per))
        return pooled, per

    def stage_confirm(self):
        lr = self.chosen_lr()
        for c in self.cands:
            for s in SEEDS_CONFIRM:
                self.run("confirm", "confirm", c, s, lr)
        pooled, per = self._pooled_rel_sd("confirm", lr)
        n_req = math.ceil(15.7 * pooled ** 2 / DELTA_MIN ** 2) if pooled is not None else None
        by_power = pooled is not None and pooled > POWER_REL_SD
        forced = bool(getattr(self.a, "force_extra_seeds", False))
        extra = forced or (by_power and not self.report_only)
        info = {"rule": "PLAN 6 power check: pooled within-candidate seed s.d. of confirm bpb (relative); > 0.2 % -> "
                        "seeds 4 and 5 for every candidate", "pooled_rel_sd": pooled, "per_candidate": per,
                "seeds_needed_for_delta_0.5pct": n_req, "extra_seeds_run": list(SEEDS_CONFIRM_EXTRA) if extra else []}
        if forced:
            info["extra_seeds_forced"] = "--force-extra-seeds: seeds 4 and 5 run for every candidate, whatever the power check says"
            info["power_check_alone_adds_seeds"] = by_power
        if self.report_only:
            info["report_only_bundle"] = ("seeds 1-3 were fixed before the run; the power check is reported and never "
                                          "adds seeds on this bundle")
            info["power_check_would_add_seeds"] = by_power
        self.hk.write_json_atomic(os.path.join(self.root, "power_check.json"), info)
        LOG("POWER CHECK: pooled seed s.d. %s of bpb -> %s (n needed for 0.5 %%: %s)%s" %
            ("%.3f %%" % (100 * pooled) if pooled is not None else "n/a",
             ("would add seeds 4, 5 (report-only bundle: not added)" if self.report_only else "adding seeds 4, 5")
             if by_power else "3 seeds suffice", n_req,
             " | --force-extra-seeds: seeds 4, 5 run in any case" if forced else ""))
        if n_req and n_req > 5:
            LOG("NOTE: the power formula asks for %d seeds; this script stops at 5 (PLAN 6: add more for finalists only)" % n_req)
        if extra:
            for c in self.cands:
                for s in SEEDS_CONFIRM_EXTRA:
                    self.run("confirm", "confirm", c, s, lr)
        if self.final_test:
            LOG("FINAL-TEST bundle: no finalist LR re-sweep (the test split is scored once, at the fixed LR)")
            return
        # PLAN 5: finalist LR re-sweep when the top two are within 1 % (report-only)
        means = self._stage_means("confirm", lr)
        ranked = sorted((m, c) for c, (m, _sd, _n) in means.items())
        if len(ranked) >= 2:
            gap = (ranked[1][0] - ranked[0][0]) / ranked[0][0]
            LOG("confirm top two: %s %.5f, %s %.5f (gap %.3f %%)" % (ranked[0][1], ranked[0][0], ranked[1][1], ranked[1][0], 100 * gap))
            if gap < RESWEEP_GAP:
                LOG("gap < 1 % -> finalist LR re-sweep (seed 1, report-only)")
                for _, c in ranked[:2]:
                    for lr2 in self.hk.LR_GRID:
                        if float(lr2) != float(lr):
                            self.run("confirm_resweep", "confirm", c, 1, lr2)

    def _stage_means(self, label, lr, recipe=None):
        out = {}
        recs = self._records(label, recipe or label, lr)
        for c in self.cands:
            xs = [r["bpb"] for r in recs.get(c, []) if r.get("bpb") is not None]
            if xs:
                m, sd = mean_sd(xs)
                out[c] = (m, sd, len(xs))
        return out

    def stage_large(self):
        p = os.path.join(self.root, "large_lr_choice.json")
        if self.a.large_lr:
            lr = float(self.a.large_lr)
        elif os.path.isfile(p):
            lr = float(load_json(p)["chosen_lr"])
        else:
            recs = [self.run("large_lrsweep", "large", self.base, 1, x) for x in LARGE_LR_GRID]
            ok = [(r["bpb"], r["lr"]) for r in recs if r["status"] == "ok" and r["bpb"] is not None]
            if not ok:
                raise SystemExit("every LR diverged in the large sweep")
            bpb, lr = min(ok)
            self.hk.write_json_atomic(p, {"chosen_lr": lr, "grid": list(LARGE_LR_GRID), "candidate": self.base, "seed": 1,
                                          "rule": "ADDITION (not pre-registered): lowest full-dev bpb on the baseline, "
                                                  "large model, full large budget, seed 1",
                                          "table": [{"lr": r["lr"], "bpb": r["bpb"], "status": r["status"]} for r in recs]})
            LOG("LARGE LR CHOICE %s (bpb %.5f)" % (self.hk.lr_tag(lr), bpb))
        for c in self.cands:
            for s in SEEDS_LARGE:
                self.run("large", "large", c, s, lr)

    # -- summary
    def write_summary(self, quiet=False):
        recs = []
        for fn in sorted(os.listdir(self.res)):
            if fn.endswith(".json"):
                try:
                    recs.append(load_json(os.path.join(self.res, fn)))
                except (ValueError, OSError):
                    pass
        cmeta = {c["id"]: c for c in self.b.m["candidates"]}
        groups = {}
        for r in recs:
            groups.setdefault((r["stage"], r["lr"]), {}).setdefault(r["candidate"], []).append(r)
        lines, out = [], {"bundle_manifest_sha256": self.msha, "baseline": self.base, "stages": {}}
        order = ["lrsweep", "screen", "confirm", "confirm_resweep", "large_lrsweep", "large"]
        for (stage, lr) in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1])):
            g = groups[(stage, lr)]
            rows = []
            for c, rs in g.items():
                ok = [r["bpb"] for r in rs if r.get("bpb") is not None]
                m, sd = mean_sd(ok)
                rows.append({"candidate": c, "lr": lr, "n_ok": len(ok), "n_runs": len(rs),
                             "seeds": sorted(r["seed"] for r in rs), "mean_bpb": m, "sd_bpb": sd,
                             "rel_sd_pct": 100 * sd / m if (sd is not None and m) else None,
                             "n_vocab": cmeta[c]["n_vocab"], "ctx": cmeta[c]["ctx_tokens"],
                             "train_bpt": cmeta[c]["train_bytes_per_token"],
                             "steps": rs[0]["data"]["steps"], "status": sorted({r["status"] for r in rs}),
                             "preregistered": rs[0].get("preregistered")})
            base = next((x["mean_bpb"] for x in rows if x["candidate"] == self.base and x["mean_bpb"]), None)
            for x in rows:
                x["delta_vs_baseline_pct"] = (100 * (x["mean_bpb"] - base) / base) if (base and x["mean_bpb"]) else None
            rows.sort(key=lambda x: (x["mean_bpb"] is None, x["mean_bpb"] or 0))
            key = "%s@lr%s" % (stage, self.hk.lr_tag(lr))
            out["stages"][key] = rows
            lines.append("")
            lines.append("== %s  (lr %s)%s" % (stage, self.hk.lr_tag(lr),
                                               "" if rows and rows[0].get("preregistered") else "  [not pre-registered / report-only]"))
            lines.append("%-4s %-30s %6s %4s %6s %6s %3s %9s %8s %7s %9s" %
                         ("rank", "candidate", "vocab", "ctx", "b/tok", "steps", "n", "mean bpb", "sd", "sd%", "d vs base"))
            for i, x in enumerate(rows, 1):
                lines.append("%-4d %-30s %6d %4d %6.3f %6d %3d %9s %8s %7s %9s" % (
                    i, x["candidate"][:30], x["n_vocab"], x["ctx"], x["train_bpt"], x["steps"], x["n_ok"],
                    "%.5f" % x["mean_bpb"] if x["mean_bpb"] is not None else "-",
                    "%.5f" % x["sd_bpb"] if x["sd_bpb"] is not None else "-",
                    "%.3f" % x["rel_sd_pct"] if x["rel_sd_pct"] is not None else "-",
                    "%+.3f%%" % x["delta_vs_baseline_pct"] if x["delta_vs_baseline_pct"] is not None else "-"))
        # PLAN 5 crossing check on the Stage 3 subset curve (75 % vs 100 %)
        cross = self._crossing_check(recs)
        if cross:
            out["screen_curve_crossing_check"] = cross
            lines.append("")
            lines.append("screen curve check (0.5 MB subset, top two at 100 %%): %s" % cross["verdict"])
        # GPU determinism: screen baseline seed 1 is an exact re-run of the chosen-LR sweep run
        det = self._determinism_check(recs)
        if det:
            out["determinism_check"] = det
            lines.append("determinism (lrsweep vs screen, baseline s1, same config): %s" % det["verdict"])
        for fn in ("lr_choice.json", "power_check_screen.json", "power_check.json", "large_lr_choice.json",
                   "parity_report.json", "runtime_pin.json"):
            p = os.path.join(self.root, fn)
            if os.path.isfile(p):
                out[fn[:-5]] = load_json(p)
        text = "\n".join(lines)
        out["table_text"] = text
        self.hk.write_json_atomic(os.path.join(self.root, "summary.json"), out)
        with open(os.path.join(self.root, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        if not quiet:
            print(text, flush=True)
        return out

    def _crossing_check(self, recs):
        lr = self.current_lr()
        if lr is None:
            return None
        pts = {}
        for r in recs:
            if r["stage"] != "screen" or float(r["lr"]) != lr or r.get("bpb") is None:
                continue
            for c in r.get("curve", []):
                pts.setdefault(r["candidate"], {}).setdefault(c["frac"], []).append(c["bpb"])
        full = {c: sum(v[1.0]) / len(v[1.0]) for c, v in pts.items() if 1.0 in v and 0.75 in v}
        if len(full) < 2:
            return None
        top = sorted(full, key=full.get)[:2]
        at75 = {c: sum(pts[c][0.75]) / len(pts[c][0.75]) for c in top}
        crossed = (at75[top[0]] > at75[top[1]])
        return {"top_two_at_100pct": top, "subset_bpb_100": {c: full[c] for c in top}, "subset_bpb_75": at75,
                "crossed": crossed,
                "verdict": ("CROSSED between 75 % and 100 % -> screening budget too small for these two (PLAN 5); "
                            "Stage 4 decides") if crossed else "no crossing"}

    def _determinism_check(self, recs):
        lr = self.current_lr()
        if lr is None:
            return None
        a = [r for r in recs if r["stage"] == "lrsweep" and r["candidate"] == self.base and r["seed"] == 1 and float(r["lr"]) == lr]
        b = [r for r in recs if r["stage"] == "screen" and r["candidate"] == self.base and r["seed"] == 1 and float(r["lr"]) == lr]
        if not a or not b or a[0].get("bpb") is None or b[0].get("bpb") is None:
            return None
        same = a[0]["dev"]["bits"] == b[0]["dev"]["bits"]
        return {"lrsweep_bpb": a[0]["bpb"], "screen_bpb": b[0]["bpb"], "bitwise_identical": same,
                "verdict": "bitwise identical" if same else "DIFFERENT (rel %.2e)" % ((b[0]["bpb"] - a[0]["bpb"]) / a[0]["bpb"])}

    def zip_results(self):
        base = self.root.rstrip("/\\") + "_results"
        z = shutil.make_archive(base, "zip", self.root)
        LOG("results zip: %s (%.1f MB)" % (z, os.path.getsize(z) / 1e6))
        if os.path.isdir("/content") and not z.startswith("/content/hindko_lm_results"):
            dst = "/content/hindko_lm_results_%s.zip" % self.msha[:12]
            try:
                shutil.copy2(z, dst)
                LOG("copy for download: %s" % dst)
            except OSError:
                pass
        return z

    # -- estimate
    def estimate(self):
        hk = self.hk
        gpu = (self.rt.gpu_name or "CPU").upper()
        key = next((k for k in EFFECTIVE_TFLOPS if k in gpu), "T4" if self.rt.device.type == "cuda" else "CPU")
        tf = EFFECTIVE_TFLOPS[key] * 1e12
        train_bytes = float(self.b.train_bytes().sum())
        dev_bytes = float(self.b.dev_bytes().sum())
        sub_bytes = float(self.b.dev_bytes()[self.b.eval_doc_index("subset")].sum())
        plan = [("lr", "screen", 1, len(hk.LR_GRID)), ("screen", "screen", len(self.cands), len(SEEDS_SCREEN)),
                ("confirm", "confirm", len(self.cands), len(SEEDS_CONFIRM)),
                ("confirm +seeds 4,5 (if power check)", "confirm", len(self.cands), 2),
                ("large lr sweep", "large", 1, len(LARGE_LR_GRID)), ("large", "large", len(self.cands), len(SEEDS_LARGE))]
        if self.final_test:     # test bundles: only confirm runs (seeds 4, 5 only if forced / not report-only)
            plan = [("confirm", "confirm", len(self.cands), len(SEEDS_CONFIRM))]
            if self.a.force_extra_seeds or not self.report_only:
                plan.append(("confirm seeds 4,5", "confirm", len(self.cands), 2))
        LOG("ESTIMATE (FLOP-based, assumed effective %.3g TFLOP/s for %s; not measured):" % (tf / 1e12, key))
        total = 0.0
        for name, recipe, nc, ns in plan:
            st = hk.STAGES[recipe]
            budget = st["budget_bytes"] if st["budget_bytes"] else st["epochs"] * train_bytes
            sec = 0.0
            for c in (self.cands[:1] if nc == 1 else self.cands):
                cm = self.b.cand(c)
                ctx = cm["ctx_tokens"]
                toks = hk.steps_for_budget(budget) * hk.SEQS_PER_STEP * ctx
                ev_toks = (2 * dev_bytes + 3 * 2 * sub_bytes) / cm["dev_bytes_per_token"]
                sec += ns * (toks * hk.train_flops_per_token(st["model"], cm["n_vocab"], ctx)
                             + ev_toks * hk.eval_flops_per_token(st["model"], cm["n_vocab"], ctx)) / tf
            total += sec
            LOG("  %-38s %3d runs  ~%6.1f min" % (name, nc * ns, sec / 60))
        LOG("  total ~%.1f h (+ per-step overhead; tiny models are launch-bound, so treat as a lower bound)" % (total / 3600))


# ============================================================ main
def build_parser():
    ap = argparse.ArgumentParser(description="Hindko tokenizer LM arbiter on Colab (PLAN 5).")
    ap.add_argument("--parts", default=None, help="folder with PARTS.json and the parts (auto-searched)")
    ap.add_argument("--work", default="/content/work" if os.path.isdir("/content") else os.path.join(os.getcwd(), "work"))
    ap.add_argument("--out", default=None, help="results root (default: Drive if mounted, else /content/out)")
    ap.add_argument("--bundle-dir", default=None, help="use an already extracted bundle directory (skips the parts)")
    ap.add_argument("--stages", default=",".join(STAGE_ORDER))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated candidate ids (the baseline is always kept)")
    ap.add_argument("--lr", type=float, default=None, help="force the peak LR (skips the sweep; recorded)")
    ap.add_argument("--large-lr", type=float, default=None)
    ap.add_argument("--max-hours", type=float, default=None, help="stop launching new runs after this many hours")
    ap.add_argument("--ignore-parity", action="store_true")
    ap.add_argument("--allow-mixed", action="store_true", help="allow a device/AMP dtype different from runtime_pin.json")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets for an end-to-end code check (separate out dir)")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--final-test-large", action="store_true",
                    help="FINAL-TEST bundles only: also allow the report-only 'large' stage (needs --large-lr; "
                         "analysis/AMENDMENT_1.md item 4)")
    ap.add_argument("--force-extra-seeds", action="store_true",
                    help="confirm: always run seeds 4 and 5 for every candidate (required for final-test bundles)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    t0 = time.time()
    if a.bundle_dir:
        bdir, P = os.path.abspath(a.bundle_dir), None
    else:
        pdir = find_parts_dir(a.parts)
        LOG("parts folder: %s" % pdir)
        P, bdir = reassemble(pdir, a.work)
    if a.verify_only:
        return 0
    r = Runner(a, bdir, P)
    if a.estimate:
        r.estimate()
        return 0
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGE_ORDER]
    if bad:
        raise SystemExit("unknown stages %s (choose from %s)" % (bad, STAGE_ORDER))
    if a.device == "auto" and r.rt.device.type == "cpu" and not a.smoke and any(s != "parity" for s in stages):
        raise SystemExit("NO GPU FOUND: refusing to run the arbiter stages on the CPU by accident (days of compute). "
                         "Colab: Runtime > Change runtime type > GPU. To really use the CPU pass --device cpu.")
    if r.report_only:
        # kind 'final_test_report' (build_bundle.py --final-test --report-only): a report-only RE-USE of the test
        # split. parity + confirm only, fixed --lr, seeds 1-3 (--force-extra-seeds optional); never large/smoke.
        bad = [s for s in stages if s not in FINAL_TEST_STAGES]
        if bad or a.smoke or a.final_test_large:
            raise SystemExit("REPORT-ONLY test bundle (its 'dev' set is the TEST split): only --stages parity,confirm may "
                             "run, without --smoke or --final-test-large (refused: %s). Nothing was run. See "
                             "colab/FRONTIER_LM.md" % (bad + (["--smoke"] if a.smoke else [])
                                                       + (["--final-test-large"] if a.final_test_large else [])))
        if "confirm" in stages and not a.lr:
            raise SystemExit("REPORT-ONLY test bundle: the confirm stage needs --lr 1e-3 (the dev choice; no LR sweep "
                             "ever runs on test). Nothing was run. See colab/FRONTIER_LM.md")
    elif r.final_test:
        # analysis/AMENDMENT_1.md item 4: a REPORT-ONLY large-arbiter test run is allowed, with the LR fixed from
        # dev (no LR sweep ever runs on test); it cannot change the released tokenizer (fixed by the amendment).
        allowed = FINAL_TEST_STAGES + (("large",) if a.final_test_large else ())
        if "large" in stages and a.final_test_large and not a.large_lr:
            raise SystemExit("FINAL-TEST bundle: the report-only large stage needs --large-lr (the dev choice, 5e-4); "
                             "no LR sweep runs on test. Nothing was run.")
        bad = [s for s in stages if s not in allowed]
        if bad or a.smoke:
            raise SystemExit("FINAL-TEST bundle (its 'dev' set is the TEST split): only --stages parity,confirm may run, "
                             "without --smoke (refused: %s). Nothing was run. See colab/FINAL_TEST.md"
                             % (bad + (["--smoke"] if a.smoke else [])))
        if "confirm" in stages and (not a.lr or not a.force_extra_seeds):
            raise SystemExit("FINAL-TEST bundle: the confirm stage needs --lr 1e-3 (the dev choice; no LR sweep ever runs "
                             "on test) and --force-extra-seeds (seeds 1-5, fixed before the test). Nothing was run. "
                             "See colab/FINAL_TEST.md")
    rc = 0
    try:
        for s in STAGE_ORDER:
            if s not in stages:
                continue
            LOG("===== stage %s =====" % s)
            getattr(r, "stage_" + s)()
    except Deadline:
        LOG("--max-hours reached: stopping cleanly; re-run the same command to resume")
        rc = 3
    r.write_summary()
    if not a.no_zip:
        r.zip_results()
    LOG("finished in %.1f min (%d new runs)" % ((time.time() - t0) / 60, r.n_new))
    return rc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
