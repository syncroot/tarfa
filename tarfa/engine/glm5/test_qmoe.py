"""Validate qmoe.moe_fused vs the per-expert qmv path; time both."""
import sys, os, glob, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
from qmoe import moe_fused
from qmv import qmv

d = sys.argv[1]; dev = os.environ.get("GLM5_DEV", "cuda:1"); torch.cuda.set_device(dev)
idx = {}; rs = []
for sh in sorted(glob.glob(d + "/*.gguf")):
    r = GGUFReader(sh); rs.append(r)
    for t in r.tensors: idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def ref(gs, us, ds, q, x, w):
    y = torch.zeros(x.shape[0], device=dev)
    for j in range(len(gs)):
        g = qmv(gs[j], q[0], x, x.shape[0]); u = qmv(us[j], q[1], x, x.shape[0])
        h = torch.nn.functional.silu(torch.clamp(g, max=10.0)) * torch.clamp(u, -10.0, 10.0)
        y += w[j] * qmv(ds[j], q[2], h.contiguous(), h.shape[0])
    return y


def bench(fn, n=30):
    fn(); torch.cuda.synchronize(dev); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(dev); return 1000 * (time.time() - t0) / n


random.seed(0)
for L in (10, 11):
    ts = [idx[f"blk.{L}.{k}"] for k in K]
    q = [QT(t.tensor_type) for t in ts]
    es = random.sample(range(288), 8)
    gs, us, ds = ([torch.from_numpy(np.array(ts[i].data[e])).to(dev) for e in es] for i in range(3))
    x = torch.randn(4096, device=dev) * 0.05
    w = torch.rand(8, device=dev); w = w / w.sum() * 2.5
    a = ref(gs, us, ds, q, x, w); b = moe_fused(gs, us, ds, *q, x, w)
    rel = ((a - b).norm() / a.norm()).item()
    print(f"L{L} types {[t.name for t in q]}: rel.err {rel:.2e}   per-expert {bench(lambda: ref(gs, us, ds, q, x, w)):.3f} ms"
          f"  ->  fused {bench(lambda: moe_fused(gs, us, ds, *q, x, w)):.3f} ms")
