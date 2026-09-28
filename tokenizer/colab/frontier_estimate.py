# -*- coding: utf-8 -*-
"""frontier_estimate.py - T4 runtime and GPU-memory estimate for the frontier LM bundle (colab/FRONTIER_LM.md).

    set PYTHONIOENCODING=utf-8 & python colab\\frontier_estimate.py [--bundle colab/build_frontier_test/staging]

Time model, fitted by least squares on every measured Tesla T4 'confirm' run in lm/colab_results (same hk_lm.py,
d=192 L=4, 3,663 steps, fp16 training / fp32 evaluation; V = 8k-49k):
    train seconds = steps * (a + b * rows*V + c * rows),   rows = 8 * ctx (tokens per optimiser step)
    eval  seconds = e0 + e1 * n_windows * ctx * V          (full test evaluation incl. the 3 curve evaluations)
The large tokenizers (V = 128k-262k) are 3-5x beyond the calibrated V range, so the linear model is an
extrapolation: the output layer (logits GEMMs + fp32 cross-entropy) is bandwidth-bound and linear in rows*V,
but a pessimistic x1.5 is printed next to it.

GPU memory model (hk_lm.train_step on CUDA: one chunk of up to 8,192 rows, i.e. the whole step):
    logits fp16 (kept by the Python variable) 2 B + log_softmax output 4 B + nll grad 4 B + input grad 4 B
    = 14 B per (row, vocab) element at the peak of the backward, + 16 B per embedding parameter (fp32 weight,
    grad, AdamW m and v) + the body; evaluation: 4,096-row chunks, fp32 logits + log_softmax = 8 B per element.
Writes colab/build_frontier_test/FRONTIER_ESTIMATE.json and prints a markdown table.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
T4_BYTES = 15.0e9          # 15,360 MiB card; ~14.6 GiB usable by PyTorch after the CUDA context
SEEDS = 3


def fit():
    X, Y, E, EY, n = [], [], [], [], 0
    for f in sorted(glob.glob(os.path.join(TOK, "lm", "colab_results", "*", "results", "confirm__*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        if r["runtime"]["gpu_name"] != "Tesla T4" or r["data"]["steps"] != 3663 or r.get("bpb") is None:
            continue
        V, ctx = r["model"]["n_vocab"], r["data"]["ctx_tokens"]
        rows = 8 * ctx
        X.append([1.0, rows * V / 1e6, rows / 1e3])
        Y.append(r["wall_time_s"]["train"] / r["data"]["steps"] * 1e3)
        E.append([1.0, r["dev"]["n_windows"] * ctx * V / 1e9])
        EY.append(r["wall_time_s"]["eval"])
        n += 1
    X, Y, E, EY = map(np.asarray, (X, Y, E, EY))
    c = np.linalg.lstsq(X, Y, rcond=None)[0]
    ce = np.linalg.lstsq(E, EY, rcond=None)[0]
    return {"n_runs": n, "train_ms_per_step": {"const": c[0], "per_M_rowsV": c[1], "per_k_rows": c[2]},
            "train_max_abs_resid_ms": float(np.abs(X @ c - Y).max()),
            "eval_s": {"const": ce[0], "per_G_win_ctx_V": ce[1]}, "eval_max_abs_resid_s": float(np.abs(E @ ce - EY).max()),
            "calibrated_V": sorted({int(round(x[1] * 1e3 / x[2])) for x in X})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(HERE, "build_frontier_test", "staging"))
    ap.add_argument("--out", default=os.path.join(HERE, "build_frontier_test", "FRONTIER_ESTIMATE.json"))
    a = ap.parse_args()
    import hk_lm
    F = fit()
    b = hk_lm.Bundle(a.bundle)
    steps = hk_lm.steps_for_budget(int(b.train_bytes().astype(np.int64).sum()))
    t, e = F["train_ms_per_step"], F["eval_s"]
    rows_out, tot, tot_p = [], 0.0, 0.0
    for c in b.m["candidates"]:
        V, ctx = int(c["n_vocab"]), int(c["ctx_tokens"])
        rows = 8 * ctx
        _, offs = b.tokens(c["id"], "dev")
        lens = np.diff(offs).astype(np.int64) + 1          # + EOT target
        n_win = int(sum(len(hk_lm.eval_windows(int(m), ctx)) for m in lens))
        train_s = steps * (t["const"] + t["per_M_rowsV"] * rows * V / 1e6 + t["per_k_rows"] * rows / 1e3) / 1e3
        eval_s = e["const"] + e["per_G_win_ctx_V"] * n_win * ctx * V / 1e9
        run_s = train_s + eval_s + 3.0                      # + model build / stream / R3 (~2-3 s measured)
        d = hk_lm.MODELS["confirm"]["d"]
        mem_train = rows * V * 14 + V * d * 16 + 60e6 + 0.5e9          # + body + CUDA context/allocator slack
        mem_eval = 4096 * V * 8 + V * d * 16 + 0.5e9
        emb = V * d
        rows_out.append({"id": c["id"], "n_vocab": V, "ctx": ctx, "train_bytes_per_token": c["train_bytes_per_token"],
                         "test_bytes_per_token": c["dev_bytes_per_token"], "rows_per_step": rows,
                         "test_eval_windows": n_win, "est_train_s": round(train_s, 1), "est_eval_s": round(eval_s, 1),
                         "est_run_min": round(run_s / 60, 2), "est_3_seeds_min": round(SEEDS * run_s / 60, 1),
                         "est_3_seeds_min_pessimistic_x1.5": round(1.5 * SEEDS * run_s / 60, 1),
                         "peak_gpu_train_GB": round(mem_train / 1e9, 2), "peak_gpu_eval_GB": round(mem_eval / 1e9, 2),
                         "fits_T4": max(mem_train, mem_eval) < T4_BYTES,
                         "params": {"token_embedding_tied": emb, "position_embedding": ctx * d,
                                    "non_embedding": 12 * d * d * 4 + 2 * d * 4 * 2 + 2 * d,
                                    "total_approx": emb + ctx * d + 12 * d * d * 4 + 2 * d * 9}})
        tot += SEEDS * run_s
        tot_p += 1.5 * SEEDS * run_s
    parity_s = 2 * 30.0
    unpack_s = 120.0
    out = {"what": "T4 estimate for run_all.py --stages parity,confirm --lr 1e-3 on the frontier bundle (seeds 1-3)",
           "bundle_manifest_sha256": b.manifest_sha256, "steps_per_run": steps, "fit": F, "candidates": rows_out,
           "total_hours": round((tot + parity_s + unpack_s) / 3600, 2),
           "total_hours_pessimistic_x1.5": round((tot_p + parity_s + unpack_s) / 3600, 2),
           "t4_memory_bytes_assumed": T4_BYTES}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("fit on %d T4 confirm runs (V %s): train ms/step = %.3f + %.4f*rows*V/1e6 + %.3f*rows/1e3 (max resid %.2f ms); "
          "eval s = %.3f + %.4f*win*ctx*V/1e9 (max resid %.2f s)" % (
              F["n_runs"], F["calibrated_V"], t["const"], t["per_M_rowsV"], t["per_k_rows"], F["train_max_abs_resid_ms"],
              e["const"], e["per_G_win_ctx_V"], F["eval_max_abs_resid_s"]))
    print("| # | candidate | V | ctx | train b/tok | test b/tok | min/run | 3 seeds (min) | x1.5 | peak GPU train / eval GB | tied emb. params |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows_out, 1):
        print("| %d | `%s` | %s | %d | %.3f | %.3f | %.1f | %.0f | %.0f | %.1f / %.1f%s | %.1fM |" % (
            i, r["id"], format(r["n_vocab"], ","), r["ctx"], r["train_bytes_per_token"], r["test_bytes_per_token"],
            r["est_run_min"], r["est_3_seeds_min"], r["est_3_seeds_min_pessimistic_x1.5"], r["peak_gpu_train_GB"],
            r["peak_gpu_eval_GB"], "" if r["fits_T4"] else " (!)", r["params"]["token_embedding_tied"] / 1e6))
    print("total ~%.1f h (pessimistic %.1f h) incl. parity and unpacking" % (out["total_hours"],
                                                                         out["total_hours_pessimistic_x1.5"]))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
