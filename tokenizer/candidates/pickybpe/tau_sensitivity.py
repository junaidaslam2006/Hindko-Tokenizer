# -*- coding: utf-8 -*-
"""REPORT-ONLY sensitivity of PickyBPE-16k to tau (not a selection step: PLAN fixes tau = 0.9, with 0.8 only if
0.9 removes < 1% of the learned tokens; 0.9 removed 1.57%, so 0.9 stands whatever this table shows).

For tau in {plain (None), 0.9, 0.8, 0.7}: trains (or reuses) the model, then reports the train support profile
from the trainer's final state (equal to the encoder's train counts, V1 of verify_pickybpe.py) and dev_strict
bytes/token from the event-ordered encoder. Models are written to models/sensitivity/ with their sha256.

    python tau_sensitivity.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickybpe_common as C  # noqa: E402
import pickybpe as PB  # noqa: E402

RUNS = [("plain", None, os.path.join(HERE, "models", "plainbpe_P1_D1_16k_samecode.json")),
        ("0.9", 0.9, os.path.join(HERE, "models", "pickybpe_P1_D1_16k_tau0.9.json")),
        ("0.8", 0.8, os.path.join(HERE, "models", "sensitivity", "pickybpe_P1_D1_16k_tau0.8.json")),
        ("0.7", 0.7, os.path.join(HERE, "models", "sensitivity", "pickybpe_P1_D1_16k_tau0.7.json"))]


def main():
    dev, dinfo = C.load_view("dev_strict", "validation")
    dev_bytes = sum(len(d["text"].encode("utf-8")) for d in dev)
    rows = []
    for label, tau, path in RUNS:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            subprocess.run([sys.executable, os.path.join(HERE, "train_pickybpe.py"), "--tau", str(tau),
                            "--vocab", "16384", "--out", path], check=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        tk = PB.PickyBPETokenizer(path)
        st = pickle.load(open(path[:-5] + ".trainstate.pkl", "rb"))
        tf = st["tfreq"]
        # V1 for every tau: the event-ordered encoder reproduces the trainer's final state of every train type
        mism = sum(1 for wb, seq in zip(st["word_bytes"], st["words"]) if list(tk.encode_word_internal(wb)) != seq)
        emitted_removed = sum(1 for d in dev for i in tk.encode_internal(d["text"]) if i >= 0 and i in tk.removed_at_end)
        rt_fail = sum(1 for d in dev if tk.decode(tk.encode(d["text"])) != d["text"])
        learned = range(PB.N_BASE, tk.vocab_size)
        freq = np.asarray([tf[tk.final_ids[i]] for i in learned])
        comps = {x for a, b, c in tk.merges for x in (a, b)}
        inter = np.asarray([i in comps for i in learned])
        ntok = sum(len(tk.encode(d["text"])) for d in dev)
        s = tk.m["summary"]
        rows.append({"tau": label, "model": path, "sha256": tk.sha256,
                     "removed_at_end": s["removed_at_end"], "removal_events": s["stats"]["removals"],
                     "readded": s["readded"], "merges": s["stats"]["merges"],
                     "removed_pct_of_learned": round(100 * s["removed_at_end"] / len(freq), 2),
                     "train_freq_eq0": int((freq == 0).sum()), "train_freq_lt20": int((freq < 20).sum()),
                     "lt20_intermediate": int(((freq < 20) & inter).sum()),
                     "train_freq_lt100": int((freq < 100).sum()),
                     "dev_strict_tokens": ntok, "dev_strict_bytes_per_token": round(dev_bytes / ntok, 4),
                     "v1_train_types_mismatching": mism, "dev_strict_removed_token_emissions": emitted_removed,
                     "dev_strict_roundtrip_failures": rt_fail,
                     "merges_existing_present": s["stats"]["merges_existing_present"],
                     "split_len_hist": s["stats"]["split_len_hist"]})
        C.log(json.dumps(rows[-1], ensure_ascii=False))
    out = os.path.join(HERE, "verify", "tau_sensitivity.json")
    json.dump({"note": "report-only; not used for selection (PLAN fixes tau = 0.9)", "dev": dinfo, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    C.log("wrote", out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
