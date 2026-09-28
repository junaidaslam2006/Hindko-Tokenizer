# -*- coding: utf-8 -*-
"""Raw fp32 matmul speed for the shapes a tiny GPT uses, incl. transposed operands."""
import time, torch, sys
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
def t(a, b, n=5):
    torch.mm(a, b)
    t0 = time.perf_counter()
    for _ in range(n): torch.mm(a, b)
    dt = (time.perf_counter() - t0) / n
    return 2 * a.shape[0] * a.shape[1] * b.shape[1] / dt / 1e9
N, d, V = 2048, 192, 8192
x = torch.randn(N, d); W = torch.randn(4 * d, d); g = torch.randn(N, 4 * d)
E = torch.randn(V, d); gl = torch.randn(512, V); h = torch.randn(512, d)
print("threads", torch.get_num_threads())
print("fwd  x@W.T      [2048x192]@[192x768]  GF/s %.1f" % t(x, W.t()))
print("bwd  g@W        [2048x768]@[768x192]  GF/s %.1f" % t(g, W))
print("bwd  g.T@x      [768x2048]@[2048x192] GF/s %.1f" % t(g.t(), x))
print("bwd  g.T@x cont                        GF/s %.1f" % t(g.t().contiguous(), x))
print("head h@E.T      [512x192]@[192x8192]  GF/s %.1f" % t(h, E.t()))
print("head gl@E       [512x8192]@[8192x192] GF/s %.1f" % t(gl, E))
print("head gl.T@h     [8192x512]@[512x192]  GF/s %.1f" % t(gl.t(), h))
print("head gl.T@h cont                       GF/s %.1f" % t(gl.t().contiguous(), h))
