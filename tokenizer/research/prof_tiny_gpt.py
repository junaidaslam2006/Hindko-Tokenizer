# -*- coding: utf-8 -*-
"""Profile one training step of the tiny GPT to find the CPU bottleneck."""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from bench_tiny_gpt2 import GPT, train_step

torch.set_num_threads(3)
V, d, L, h, T, B = 8192, 192, 4, 4, 256, 8
m = GPT(V, d, L, h, T)
fused = "--fused" in sys.argv
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, foreach=not fused, fused=fused) if fused else torch.optim.AdamW(m.parameters(), lr=1e-3)
idx = torch.randint(0, V, (B, T)); tgt = torch.randint(0, V, (B, T))
for _ in range(2): train_step(m, opt, idx, tgt)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(2): train_step(m, opt, idx, tgt)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
# timing with manual attention instead of SDPA
