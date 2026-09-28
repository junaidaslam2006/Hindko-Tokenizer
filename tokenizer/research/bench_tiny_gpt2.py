# -*- coding: utf-8 -*-
"""Throughput of tiny GPTs on this CPU with a chunked output layer (v2).

v1 (bench_tiny_gpt.py) materialised the full [tokens x V] logits and was
dominated by memory traffic. Here the loss is computed in row chunks with a
detached hidden state, so only [chunk x V] logits exist at once.
Random tokens: measures speed only. Writes bench_tiny_gpt2.json.
Usage: python bench_tiny_gpt2.py <threads>
"""
import json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
torch.set_num_threads(THREADS)
# Measured 2026-09-26: without flush-to-zero, denormal floats in the backward pass made
# training steps 2-11x slower and highly variable on this CPU (denormal_test.py).
FTZ = "--no-ftz" not in sys.argv
torch.set_flush_denormal(FTZ)
torch.manual_seed(0)
CHUNK = 512

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
        return x + s.out(F.gelu(s.fc(s.ln2(x))))

class GPT(nn.Module):
    def __init__(s, V, d, L, h, T):
        super().__init__()
        s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(T, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf = nn.LayerNorm(d)
    def hidden(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        return s.lnf(x)

def train_step(m, opt, idx, tgt):
    h = m.hidden(idx).view(-1, m.tok.weight.shape[1])
    hd = h.detach().requires_grad_(True)
    y = tgt.view(-1); n = y.numel(); tot = 0.0
    for i in range(0, n, CHUNK):
        lg = hd[i:i + CHUNK] @ m.tok.weight.t()
        loss = F.cross_entropy(lg, y[i:i + CHUNK], reduction="sum") / n
        loss.backward(); tot += loss.item()
    h.backward(hd.grad)
    opt.step(); opt.zero_grad(set_to_none=True)
    return tot

@torch.no_grad()
def eval_step(m, idx, tgt):
    h = m.hidden(idx).view(-1, m.tok.weight.shape[1]); y = tgt.view(-1); tot = 0.0
    for i in range(0, y.numel(), CHUNK):
        tot += F.cross_entropy(h[i:i + CHUNK] @ m.tok.weight.t(), y[i:i + CHUNK], reduction="sum").item()
    return tot

def bench(V, d, L, h, T, B, steps=5, warm=2):
    torch.manual_seed(0)
    m = GPT(V, d, L, h, T)
    n_emb = V * d
    n_body = sum(p.numel() for p in m.parameters()) - n_emb - T * d
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, V, (B, T)); tgt = torch.randint(0, V, (B, T))
    for i in range(warm + steps):
        if i == warm: t0 = time.perf_counter()
        train_step(m, opt, idx, tgt)
    dt = (time.perf_counter() - t0) / steps
    eval_step(m, idx, tgt); t1 = time.perf_counter()
    for _ in range(3): eval_step(m, idx, tgt)
    de = (time.perf_counter() - t1) / 3
    return {"threads": THREADS, "flush_denormal": FTZ, "V": V, "d": d, "L": L, "heads": h, "ctx": T, "batch": B,
            "tokens_per_step": B * T, "non_emb_params_M": round(n_body / 1e6, 2),
            "emb_params_M": round(n_emb / 1e6, 2), "train_s_per_step": round(dt, 3),
            "train_tok_per_s": round(B * T / dt), "eval_tok_per_s": round(B * T / de)}

if __name__ == "__main__":
    cfgs = [(8192, 192, 4, 4, 256, 8), (16384, 192, 4, 4, 256, 8), (32768, 192, 4, 4, 256, 8),
            (16384, 256, 4, 4, 256, 8), (32768, 256, 4, 4, 256, 8), (16384, 256, 6, 4, 512, 4)]
    res = []
    for c in cfgs:
        r = bench(*c); print(json.dumps(r), flush=True); res.append(r)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_tiny_gpt2_t%d.json" % THREADS)
    with open(path, "w") as f:
        json.dump({"cpu": "Intel i7-10610U (4C/8T, laptop), torch %s, fp32, chunked CE (%d rows)" % (torch.__version__, CHUNK),
                   "note": "machine shared with other agents during measurement; indicative only",
                   "results": res}, f, indent=1)
