# -*- coding: utf-8 -*-
"""Unit test of init_embeddings.py on tiny RANDOM-WEIGHT models with the real vocabulary sizes (no download, no
real checkpoint). For every delivered folder: build a tiny model of the base's architecture family, save it, run
init_embeddings.apply(), reload, and check
  - rows >= len(tokenizer), padded to a multiple of 64 when grown;
  - every base row (id < first_new) is bit-identical to before;
  - new input rows == norm-calibrated uniform mean of the decomposition rows (recomputed here independently);
  - new output rows (untied) == byte-length-weighted mean; tied models: input row == output row;
  - a forward pass on a Hindko sentence that uses new ids runs and gives finite logits of the new width.

    python test_init_embeddings.py      -> results/test_init_embeddings.json
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402
import init_embeddings as IE  # noqa: E402

torch.set_flush_denormal(True)
torch.manual_seed(0)
TMP = os.path.join(HERE, "_test_init")
# embedding rows of real checkpoints where known (Qwen3: 151,936; Llama-3.1: 128,256; Gemma-3 1B / Gemma-4: 262,144)
ROWS = {"qwen-3": 151936, "llama-3": 128256, "gemma-3": 262144, "gemma-4": 262144, "qwen-3.5": None}


def tiny(base, rows, tied):
    kw = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
              head_dim=8, max_position_embeddings=512, tie_word_embeddings=tied)
    if base.startswith("qwen"):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        return Qwen3ForCausalLM(Qwen3Config(vocab_size=rows, **kw)), "Qwen3ForCausalLM"
    if base.startswith("llama"):
        from transformers import LlamaConfig, LlamaForCausalLM
        kw.pop("head_dim")
        return LlamaForCausalLM(LlamaConfig(vocab_size=rows, **kw)), "LlamaForCausalLM"
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    return Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=rows, sliding_window=64, **kw)), "Gemma3ForCausalLM"


def run_one(folder, base, tied):
    info = C.load_json(os.path.join(folder, "EXTENSION.json"))
    first, n = info["first_new_id"], info["len_tokenizer"]
    rows = ROWS[base] or ((first + 63) // 64) * 64
    model, cls = tiny(base, rows, tied)
    mdir = os.path.join(TMP, "%s_%s_model" % (base, "tied" if tied else "untied"))
    odir = mdir.replace("_model", "_out")
    for d in (mdir, odir):
        shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(mdir)
    E0 = model.get_input_embeddings().weight.detach().clone().numpy()
    O0 = model.get_output_embeddings().weight.detach().clone().numpy()
    stats = IE.apply(mdir, folder, odir, dtype="float32")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m2 = AutoModelForCausalLM.from_pretrained(odir, dtype=torch.float32)
    E1 = m2.get_input_embeddings().weight.detach().numpy()
    O1 = m2.get_output_embeddings().weight.detach().numpy()
    toks = IE.load_new_tokens(folder)
    res = {"folder": os.path.basename(folder), "arch": cls, "tied": tied, "rows_before": rows,
           "rows_after": int(E1.shape[0]), "len_tokenizer": n, "stats": stats}
    res["rows_ok"] = E1.shape[0] >= n and (E1.shape[0] == rows or E1.shape[0] % 64 == 0)
    # rows that existed before (Gemma-3 1B has 262,144 rows, so the tokenizer's <image_soft_token> id 262,144 is
    # created by the resize and is neither a base row nor a new-token row)
    nb = min(first, rows)
    res["base_rows_compared"] = nb
    res["base_rows_unchanged"] = bool(np.array_equal(E1[:nb], E0[:nb]) and np.array_equal(O1[:nb], O0[:nb]))
    ref_ids = sorted({b for t in toks for b in t["base_ids"]})
    nu = np.linalg.norm(E0[ref_ids], axis=1).mean()
    exp_in = np.stack([E0[t["base_ids"]].mean(0) for t in toks])
    exp_in = exp_in * (nu / np.linalg.norm(exp_in, axis=1, keepdims=True))
    got_in = E1[first:first + len(toks)]
    res["input_rows_max_abs_err"] = float(np.abs(got_in - exp_in).max())
    if tied:
        res["tied_rows_equal"] = bool(np.array_equal(E1, O1))
        res["output_rows_max_abs_err"] = None
    else:
        exp_out = np.stack([(O0[t["base_ids"]] * np.asarray(t["base_nbytes"], np.float32)[:, None]).sum(0)
                            / sum(t["base_nbytes"]) for t in toks])
        res["output_rows_max_abs_err"] = float(np.abs(O1[first:first + len(toks)] - exp_out).max())
    tok = AutoTokenizer.from_pretrained(odir)
    ids = tok("ہندکو زبان ہزارہ تے پشور دے علاقیاں وچ بولی جاندی اے۔", add_special_tokens=False,
              return_tensors="pt")["input_ids"]
    res["sentence_new_ids"] = int((ids >= first).sum())
    with torch.no_grad():
        lo = m2(input_ids=ids).logits
    res["forward_ok"] = bool(torch.isfinite(lo).all()) and lo.shape[-1] == E1.shape[0]
    res["pass"] = bool(res["rows_ok"] and res["base_rows_unchanged"] and res["input_rows_max_abs_err"] < 1e-5
                       and (res.get("tied_rows_equal", True)) and (tied or res["output_rows_max_abs_err"] < 1e-5)
                       and res["forward_ok"] and res["sentence_new_ids"] > 0)
    shutil.rmtree(mdir, ignore_errors=True)
    shutil.rmtree(odir, ignore_errors=True)
    return res


def main():
    man = C.load_json(os.path.join(C.DELIVER, "deliver_manifest.json"))
    out = []
    for name, m in man.items():
        base = m["base"]
        folder = os.path.join(C.DELIVER, name)
        for tied in ((True,) if base.startswith("gemma") else (True, False)):
            r = run_one(folder, base, tied)
            print(json.dumps({x: r[x] for x in r if x != "stats"}, ensure_ascii=False))
            out.append(r)
    C.dump_json({"what": "init_embeddings.py on tiny random-weight models (not real checkpoints)",
                 "all_pass": all(r["pass"] for r in out), "runs": out},
                os.path.join(C.RESULTS, "test_init_embeddings.json"))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
