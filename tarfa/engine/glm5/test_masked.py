import os, sys, glob, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
from experts_gpu import RawExperts, KINDS
from qmoe import moe_masked, moe_fused
d = sys.argv[1]; idx = {}
rs = [GGUFReader(s) for s in sorted(glob.glob(d + "/*.gguf"))]
for r in rs:
    for t in r.tensors: idx[t.name] = t
DEV = "cuda:1"; torch.cuda.set_device(DEV)
raw = RawExperts(idx, DEV)
L = 20; ts = [idx[f"blk.{L}.{k}"] for k in KINDS]; q = [QT(t.tensor_type) for t in ts]; rb = [t.data.shape[2] for t in ts]
print("shapes", [t.data.shape for t in ts], q)
sel = [5, 17, 33, 100, 140, 200, 250, 287]
trips = {e: tuple(raw._load(ts[i], e, DEV) for i in range(3)) for e in sel[:5]}
x = torch.randn(4096, device=DEV) * 0.05; w = torch.rand(8, device=DEV)
ref = torch.zeros(4096, device=DEV)
raw._gpu([(j, (trips[e], DEV)) for j, e in enumerate(sel) if e in trips], x, w, q, ref)
fused = moe_fused([trips[e][0] for e in sel[:5]], [trips[e][1] for e in sel[:5]], [trips[e][2] for e in sel[:5]], *q, x, w[:5])
tb = torch.zeros(3, 288, dtype=torch.long)
for e, tr in trips.items(): tb[:, e] = torch.tensor([t.data_ptr() for t in tr])
tb = tb.to(DEV); selt = torch.tensor(sel, device=DEV)
P = tb[:, selt]
print("P", P[0].tolist()[:6], "contig", P[0].is_contiguous(), P[0].stride(), P.stride())
Hm = torch.zeros(8, 2048, device=DEV); y = torch.zeros(4096, device=DEV)
moe_masked(P, *q, *rb, x, w, Hm, y)
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"fused vs ref {rel(fused, ref):.2e}   masked vs ref {rel(y, ref):.2e}")
P2 = P.contiguous().clone()
moe_masked(P2, *q, *rb, x, w, Hm, y)
print(f"masked(P cloned rows) vs ref {rel(y, ref):.2e}")
# graph-captured
def body():
    P = tb[:, selt]
    moe_masked(P, *q, *rb, x, w, Hm, y)
s = torch.cuda.Stream(DEV); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): body()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): body()
y.zero_(); g.replay(); torch.cuda.synchronize()
print(f"graph replay vs ref {rel(y, ref):.2e}")
x.copy_(torch.randn(4096, device=DEV) * 0.05)
ref2 = torch.zeros(4096, device=DEV)
raw._gpu([(j, (trips[e], DEV)) for j, e in enumerate(sel) if e in trips], x, w, q, ref2)
g.replay(); torch.cuda.synchronize()
print(f"graph replay new x vs ref {rel(y, ref2):.2e}")
