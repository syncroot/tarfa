"""Validate CPU MoE vs the GPU qmv path; measure CPU time per expert (cold + warm pages)."""
import sys, os, glob, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
from cpu_moe import moe_cpu, KERNEL
print("CPU kernel:", KERNEL)
from qmv import qmv

d = sys.argv[1]; dev = "cuda:1"; torch.cuda.set_device(dev)
idx = {}; rs = []
for sh in sorted(glob.glob(d + "/*.gguf")):
    r = GGUFReader(sh); rs.append(r)
    for t in r.tensors: idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
random.seed(3)
for L in (10, 11, 20):
    ts = [idx[f"blk.{L}.{k}"] for k in K]; q = [QT(t.tensor_type) for t in ts]
    for E in (1, 3, 8):
        es = random.sample(range(288), E)
        gs, us, ds = ([ts[i].data[e] for e in es] for i in range(3))
        x = (np.random.randn(4096) * 0.05).astype(np.float32); w = np.random.rand(E).astype(np.float32)
        t0 = time.time(); y = moe_cpu(gs, us, ds, *q, x, w); cold = time.time() - t0
        t0 = time.time(); y = moe_cpu(gs, us, ds, *q, x, w); warm = time.time() - t0
        # GPU reference
        xg = torch.from_numpy(x).to(dev); yr = torch.zeros(4096, device=dev)
        for j, e in enumerate(es):
            gr, ur, dr = (torch.from_numpy(np.array(a)).to(dev) for a in (gs[j], us[j], ds[j]))
            g = qmv(gr, q[0], xg, 4096); u = qmv(ur, q[1], xg, 4096)
            hh = torch.nn.functional.silu(torch.clamp(g, max=10.0)) * torch.clamp(u, -10.0, 10.0)
            yr += float(w[j]) * qmv(dr, q[2], hh.contiguous(), hh.shape[0])
        yr = yr.cpu().numpy()
        rel = np.linalg.norm(y - yr) / np.linalg.norm(yr)
        mb = sum(a.nbytes for a in gs + us + ds) / 1e6
        print(f"L{L} {[t.name for t in q]} E={E}: rel {rel:.1e}  cold {1000*cold:6.1f} ms  warm {1000*warm:6.1f} ms "
              f"({mb/warm/1e3:.1f} GB/s warm, {1000*warm/E:.2f} ms/expert)")
