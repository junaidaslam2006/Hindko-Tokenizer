# -*- coding: utf-8 -*-
"""Is the slow backward caused by denormal floats? Time train steps with/without FTZ."""
import sys, time, torch
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from bench_tiny_gpt2 import GPT, train_step
torch.set_num_threads(3)
flush = "--flush" in sys.argv
print("set_flush_denormal ->", torch.set_flush_denormal(flush), "flush=", flush)
V, d, L, h, T, B = 8192, 192, 4, 4, 256, 8
torch.manual_seed(0)
m = GPT(V, d, L, h, T)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
idx = torch.randint(0, V, (B, T)); tgt = torch.randint(0, V, (B, T))
for i in range(6):
    t0 = time.perf_counter(); train_step(m, opt, idx, tgt); print("step", i, "%.2fs" % (time.perf_counter() - t0), flush=True)
