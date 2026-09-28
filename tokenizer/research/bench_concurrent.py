# -*- coding: utf-8 -*-
"""Concurrent-throughput benchmark for the tiny-GPT arbiter (added 2026-09-26 after review).

The earlier statement "three single-thread jobs in parallel give ~2x the aggregate
throughput of one 3-thread job" was arithmetic (3 x solo 1-thread tok/s), not a
measurement. This script measures what a Stage-3/4 "wave" actually does: three
single-thread processes (one per seed) training the SAME configuration at the same time.

For every configuration it runs
  1. one solo 1-thread process (reference), then
  2. three simultaneous 1-thread processes,
using bench_tiny_gpt2.bench (random tokens, FTZ on, chunked CE, fp32).
Writes bench_concurrent.json next to this file.

Usage: python bench_concurrent.py [steps]
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CFGS = [(8192, 128, 4, 4, 256, 8), (16384, 128, 4, 4, 256, 8), (32768, 128, 4, 4, 256, 8),
        (8192, 192, 4, 4, 256, 8), (16384, 192, 4, 4, 256, 8), (32768, 192, 4, 4, 256, 8)]

CHILD = (
    "import sys, json; sys.argv=[sys.argv[0], '1']; sys.path.insert(0, %r);"
    "import bench_tiny_gpt2 as b; print(json.dumps(b.bench(%s, steps=%d, warm=2)))"
)


def run(cfg, n):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    code = CHILD % (HERE, ", ".join(str(x) for x in cfg), STEPS)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=env, text=True) for _ in range(n)]
    outs = []
    for p in procs:
        o, e = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(e)
        outs.append(json.loads(o.strip().splitlines()[-1]))
    return outs, round(time.time() - t0, 1)


def main():
    res = []
    for cfg in CFGS:
        solo, ws = run(cfg, 1)
        conc, wc = run(cfg, 3)
        row = {"V": cfg[0], "d": cfg[1], "L": cfg[2], "ctx": cfg[4], "batch": cfg[5],
               "solo_train_tok_s": solo[0]["train_tok_per_s"], "solo_eval_tok_s": solo[0]["eval_tok_per_s"],
               "conc3_train_tok_s_each": [o["train_tok_per_s"] for o in conc],
               "conc3_eval_tok_s_each": [o["eval_tok_per_s"] for o in conc],
               "wall_s_solo": ws, "wall_s_conc3": wc}
        row["conc3_train_tok_s_mean"] = round(sum(row["conc3_train_tok_s_each"]) / 3)
        row["conc3_train_tok_s_aggregate"] = sum(row["conc3_train_tok_s_each"])
        row["conc3_eval_tok_s_mean"] = round(sum(row["conc3_eval_tok_s_each"]) / 3)
        row["per_process_slowdown_vs_solo"] = round(row["conc3_train_tok_s_mean"] / row["solo_train_tok_s"], 3)
        print(json.dumps(row), flush=True)
        res.append(row)
    with open(os.path.join(HERE, "bench_concurrent.json"), "w") as f:
        json.dump({"what": "tiny-GPT train/eval throughput: solo 1-thread vs 3 simultaneous 1-thread processes",
                   "cpu": "Intel i7-10610U (4C/8T laptop), shared with other agents during measurement",
                   "settings": "fp32, torch.set_flush_denormal(True), OMP_NUM_THREADS=1, chunked CE 512 rows, "
                               "random tokens, %d timed steps after 2 warm-up steps, 2048 tokens/step" % STEPS,
                   "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   "results": res}, f, indent=1)


if __name__ == "__main__":
    main()
