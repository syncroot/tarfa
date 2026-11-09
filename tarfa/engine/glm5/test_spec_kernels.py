"""q8mm (N columns) vs q8mv; moe_cpu_numa_t (2 tokens, union) vs moe_cpu per token. CPU test runs anywhere;
GPU test needs ~1 GB free on GLM5_DEV."""
import os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
idx = {}
for sh in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
    for t in GGUFReader(sh).tensors:
        idx[t.name] = t
import cpu_moe
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
ts = [idx[f"blk.20.{k}"] for k in K]
rng = np.random.default_rng(0)
x = (rng.standard_normal((2, 4096)) * 0.05).astype(np.float32)
sel = [[1, 5, 9, 40, 77, 100, 150, 287], [5, 9, 12, 41, 77, 101, 150, 200]]
wts = rng.random((2, 8)).astype(np.float32)
ref = np.stack([cpu_moe.moe_cpu([ts[0].data[e] for e in sel[t]], [ts[1].data[e] for e in sel[t]], [ts[2].data[e] for e in sel[t]],
                                ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x[t], wts[t]) for t in range(2)])
uni = {}
for t in range(2):
    for j, e in enumerate(sel[t]):
        uni.setdefault(e, np.zeros(2, np.float32))[t] = wts[t, j]
es = list(uni)
got = cpu_moe.moe_cpu_numa_t([ts[0].data[e] for e in es], [ts[1].data[e] for e in es], [ts[2].data[e] for e in es],
                             ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x, np.stack([uni[e] for e in es]))
print("cpu union kernel: max rel err", float(np.abs(got - ref).max() / np.abs(ref).max()), f"({len(es)} experts for 16 routes)")
if len(sys.argv) > 2:
    import torch, time
    from q8 import load_q8
    dev = sys.argv[2]; torch.cuda.set_device(dev)
    for name, shp in (("blk.4.attn_q.weight", (8192, 4096)), ("output.weight", (154880, 4096)), ("blk.4.attn_output.weight", (4096, 8192))):
        W = load_q8(idx[name], *shp, dev)
        X = (torch.randn(2, shp[1], device=dev) * 0.05).half()
        ref = torch.stack([W @ X[0], W @ X[1]]).float(); got = W.mm(X).float()
        def bench(fn, n=30):
            fn(); torch.cuda.synchronize(); t0 = time.time()
            for _ in range(n): fn()
            torch.cuda.synchronize(); return 1000 * (time.time() - t0) / n
        print(f"{name}: q8mm rel err {((got-ref).norm()/ref.norm()).item():.1e}; 2x q8mv {bench(lambda: (W@X[0], W@X[1])):.3f} ms, "
              f"q8mm {bench(lambda: W.mm(X)):.3f} ms")
