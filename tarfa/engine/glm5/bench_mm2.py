import os, sys, glob, time, numpy as np, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
from q8 import load_q8

@triton.jit
def _q8_mm2(Q, D, X, Y, rows, cols, BR: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < rows
    kb = tl.arange(0, BK)
    jb = tl.arange(0, BK // 32)
    nb = cols // 32
    a0 = tl.zeros([BR], dtype=tl.float32); a1 = tl.zeros([BR], dtype=tl.float32)
    qrow = Q + r.to(tl.int64)[:, None] * cols
    drow = D + r.to(tl.int64)[:, None] * nb
    for k in range(0, cols, BK):
        q = tl.load(qrow + (k + kb)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        d = tl.load(drow + (k // 32 + jb)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        x0 = tl.load(X + k + kb).to(tl.float32)
        x1 = tl.load(X + cols + k + kb).to(tl.float32)
        a0 += tl.sum(tl.sum(tl.reshape(q * x0[None, :], (BR, BK // 32, 32)), axis=2) * d, axis=1)
        a1 += tl.sum(tl.sum(tl.reshape(q * x1[None, :], (BR, BK // 32, 32)), axis=2) * d, axis=1)
    tl.store(Y + r, a0, mask=rm)
    tl.store(Y + rows + r, a1, mask=rm)

idx = {}
for sh in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
    for t in GGUFReader(sh).tensors: idx[t.name] = t
dev = sys.argv[2]; torch.cuda.set_device(dev)
def bench(fn, n=30):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return 1000 * (time.time() - t0) / n
for name, shp in (("output.weight", (154880, 4096)), ("blk.4.attn_q.weight", (8192, 4096)), ("blk.4.attn_output.weight", (4096, 8192)), ("blk.4.ffn_gate_shexp.weight", (2048, 4096))):
    W = load_q8(idx[name], *shp, dev)
    X = (torch.randn(2, shp[1], device=dev) * 0.05).half()
    ref = torch.stack([W @ X[0], W @ X[1]]).float()
    t1 = bench(lambda: W @ X[0])
    res = []
    for BR, BK, NW in ((2, 512, 4), (4, 256, 4), (4, 512, 4), (4, 512, 8), (8, 256, 4), (8, 256, 8), (2, 1024, 4), (1, 1024, 4)):
        Y = torch.empty(2, shp[0], device=dev)
        f = lambda: _q8_mm2[(triton.cdiv(shp[0], BR),)](W.q, W.d, X, Y, shp[0], shp[1], BR=BR, BK=BK, num_warps=NW)
        f(); err = ((Y - ref).norm() / ref.norm()).item()
        res.append((bench(f), BR, BK, NW, err))
    res.sort()
    print(f"{name}: 1x mv {t1:.3f} ms; best mm2 " + ", ".join(f"{t:.3f}ms BR{a} BK{b} NW{c} err{e:.0e}" for t, a, b, c, e in res[:3]), flush=True)
