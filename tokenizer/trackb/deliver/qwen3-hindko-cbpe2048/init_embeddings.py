# -*- coding: utf-8 -*-
"""Initialise the embedding rows of the new tokens of a Track B extended tokenizer (PLAN.md 9.3), before
continued pretraining (CPT) on a GPU.

STATUS: reference implementation. It was NOT run on a real checkpoint on this project's CPU machine; it is
unit-tested on tiny random-weight Qwen3 / Llama / Gemma3 models that have the real vocabulary sizes
(test_init_embeddings.py). Whoever runs CPT should re-check shapes and tied/untied handling for their checkpoint.

Recipe ("asymmetric subword mean", Joshi et al. 2026, "Beyond Initialization Loss", arXiv:2608.03494, as read
in research/SOTA_TOKENIZATION.md 3.5; the norm target below is this project's operationalisation):
  decomposition  D(t) = the base-token ids whose concatenation formed new token t during continued-BPE training
                 (`base_ids` in new_tokens.jsonl). All ids in D(t) are base ids, so their rows are pretrained.
  input row      e_in(t) = mean_{b in D(t)} E_in[b]   (uniform subword mean), then rescaled to the norm target
                 nu = mean_{b in B} ||E_in[b]||, B = every base id that occurs in some D(t) (the Arabic-script
                 base tokens Hindko text is made of; "script-norm calibration"). --no-norm-calibration disables it.
  output row     e_out(t) = sum_b w_b E_out[b] / sum_b w_b, w_b = UTF-8 byte length of base token b
                 (character-length-weighted subword mean; Arabic-script characters are 2 bytes, so the weight is
                 proportional to character length, and partial-UTF-8 byte tokens get their byte share).
                 Only for UNTIED models. For tied models (Gemma; Qwen3 <= 4B; Llama-3.2 1B/3B) there is one
                 matrix and only the input rule is applied (an asymmetric init would require untying it).
  resizing       the embedding matrix is grown to at least len(tokenizer) rows, padded to a multiple of
                 --pad-to (default 64). Existing padding rows that new ids now occupy (Qwen3 checkpoints have
                 151,936 rows for 151,669 ids) are overwritten like any other new row.
Compare against alternatives (mean-of-all, FOCUS, Token Distillation) with 50-step CPT probes on held-out Hindko
(dev_strict, never test), not by initialisation loss; train embeddings + top-2/bottom-2 layers first
(Yamaguchi et al. 2026), then full CPT. See TRACKB.md.

    python init_embeddings.py --model BASE_CHECKPOINT --tokenizer DELIVERED_FOLDER --out OUT_DIR [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch


def load_new_tokens(tokenizer_dir: str) -> List[dict]:
    path = os.path.join(tokenizer_dir, "new_tokens.jsonl")
    with open(path, encoding="utf-8") as f:
        toks = [json.loads(l) for l in f]
    ids = [t["id"] for t in toks]
    assert ids == list(range(ids[0], ids[0] + len(ids))), "new ids must be contiguous"
    return toks


@torch.no_grad()
def init_rows(E_in: torch.Tensor, E_out: Optional[torch.Tensor], new_tokens: List[dict],
              norm_calibration: bool = True) -> dict:
    """In-place initialisation of the new rows of E_in (and of E_out unless it is None, i.e. tied)."""
    first = new_tokens[0]["id"]
    assert all(b < first for t in new_tokens for b in t["base_ids"]), "decompositions must use base ids only"
    assert E_in.shape[0] > new_tokens[-1]["id"], "resize the embedding matrix first"
    ids = torch.tensor([t["id"] for t in new_tokens], dtype=torch.long)
    rows = torch.stack([E_in[torch.tensor(t["base_ids"], dtype=torch.long)].float().mean(0) for t in new_tokens])
    stats = {"new_rows": len(new_tokens), "tied": E_out is None}
    if norm_calibration:
        ref = torch.tensor(sorted({b for t in new_tokens for b in t["base_ids"]}), dtype=torch.long)
        nu = E_in[ref].float().norm(dim=1).mean()
        before = rows.norm(dim=1).mean().item()
        rows = rows * (nu / rows.norm(dim=1, keepdim=True).clamp_min(1e-12))
        stats.update({"norm_target": nu.item(), "norm_ref_ids": int(ref.numel()), "mean_norm_before": before})
    E_in[ids] = rows.to(E_in.dtype)
    if E_out is not None:
        assert E_out.shape[0] > new_tokens[-1]["id"]
        outs = []
        for t in new_tokens:
            b = torch.tensor(t["base_ids"], dtype=torch.long)
            w = torch.tensor(t["base_nbytes"], dtype=torch.float32)
            outs.append((E_out[b].float() * w[:, None]).sum(0) / w.sum())
        E_out[ids] = torch.stack(outs).to(E_out.dtype)
    return stats


def apply(model_dir: str, tokenizer_dir: str, out_dir: str, dtype: str = "bfloat16", pad_to: int = 64,
          norm_calibration: bool = True) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype))
    new_tokens = load_new_tokens(tokenizer_dir)
    n = len(tok)
    assert n == new_tokens[-1]["id"] + 1, (n, new_tokens[-1]["id"])
    emb = model.get_input_embeddings()
    rows0 = emb.weight.shape[0]
    if rows0 < n:
        model.resize_token_embeddings(n, pad_to_multiple_of=pad_to, mean_resizing=False)
    E_in = model.get_input_embeddings().weight
    out = model.get_output_embeddings()
    tied = out is None or out.weight.data_ptr() == E_in.data_ptr()
    stats = init_rows(E_in.data, None if tied else out.weight.data, new_tokens, norm_calibration=norm_calibration)
    stats.update({"rows_before": rows0, "rows_after": model.get_input_embeddings().weight.shape[0],
                  "len_tokenizer": n, "config_vocab_size": model.config.get_text_config().vocab_size
                  if hasattr(model.config, "get_text_config") else model.config.vocab_size})
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "init_embeddings_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, help="a Track B delivered folder (has new_tokens.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pad-to", type=int, default=64)
    ap.add_argument("--no-norm-calibration", action="store_true")
    a = ap.parse_args()
    print(json.dumps(apply(a.model, a.tokenizer, a.out, a.dtype, a.pad_to, not a.no_norm_calibration), indent=1))


if __name__ == "__main__":
    main()
