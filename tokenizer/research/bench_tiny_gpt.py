# -*- coding: utf-8 -*-
"""Measure training/eval throughput of tiny GPTs on this CPU (3 threads).

Purpose: calibrate the LM-based bits-per-byte protocol in PLAN.md.
Random token data (throughput only; no learning is measured here).
Deterministic seeds. Writes bench_tiny_gpt.json next to this file.
"""
import json, os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(3)
torch.set_num_interop_threads(1)
torch.manual_seed(0)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.proj = nn.Linear(d, d, bias=False)
        s.fc = nn.Linear(d, 4 * d, bias=False); s.out = nn.Linear(4 * d, d, bias=False)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q = q.view(B, T, s.h, C // s.h).transpose(1, 2)
        k = k.view(B, T, s.h, C // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, C // s.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + s.out(F.gelu(s.fc(s.ln2(x))))
        return x

class GPT(nn.Module):
    def __init__(s, V, d, L, h, T):
        super().__init__()
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(T, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf = nn.LayerNorm(d)
    def forward(s, idx, tgt):
        B, T = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(T))
        for b in s.blocks: x = b(x)
        x = s.lnf(x)
        logits = x @ s.tok.weight.t()          # tied embeddings
        return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

def bench(V, d, L, h, T, B, steps=6, warm=2):
    torch.manual_seed(0)
    m = GPT(V, d, L, h, T)
    n_emb = V * d
    n_body = sum(p.numel() for p in m.parameters()) - n_emb - T * d
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, V, (B, T)); tgt = torch.randint(0, V, (B, T))
    for i in range(warm + steps):
        if i == warm: t0 = time.perf_counter()
        loss = m(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    dt = (time.perf_counter() - t0) / steps
    m.eval()
    with torch.no_grad():
        m(idx, tgt)
        t1 = time.perf_counter()
        for _ in range(3): m(idx, tgt)
        de = (time.perf_counter() - t1) / 3
    return {"V": V, "d": d, "L": L, "heads": h, "ctx": T, "batch": B,
            "tokens_per_step": B * T, "non_emb_params_M": round(n_body / 1e6, 2),
            "emb_params_M": round(n_emb / 1e6, 2),
            "train_s_per_step": round(dt, 3), "train_tok_per_s": round(B * T / dt),
            "eval_tok_per_s": round(B * T / de)}

if __name__ == "__main__":
    cfgs = [
        (8192, 256, 4, 4, 512, 8),
        (16384, 256, 4, 4, 512, 8),
        (32768, 256, 4, 4, 512, 8),
        (16384, 384, 6, 6, 512, 8),
        (32768, 384, 6, 6, 512, 8),
    ]
    out = []
    for c in cfgs:
        r = bench(*c)
        print(json.dumps(r), flush=True)
        out.append(r)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_tiny_gpt.json"), "w") as f:
        json.dump({"cpu": "Intel i7-10610U (4C/8T), torch %s, 3 threads, fp32" % torch.__version__,
                   "note": "machine shared with other agents during measurement; numbers are indicative",
                   "results": out}, f, indent=1)
