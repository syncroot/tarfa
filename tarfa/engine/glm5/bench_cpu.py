"""CPU routed-expert kernel throughput on realistic calls (E experts of one layer, random layers)."""
import os, sys, glob, time, ctypes, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
if os.environ.get('BENCH_TORCH') == '1':
    import torch; torch.zeros(1).cuda()
import cpu_moe
d = sys.argv[1]; E = int(sys.argv[2]) if len(sys.argv) > 2 else 5; N = 60
idx = {}
rs = [GGUFReader(s) for s in sorted(glob.glob(d + "/*.gguf"))]
for r in rs:
    for t in r.tensors: idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
random.seed(1)
calls = [(L, random.sample(range(288), E)) for L in [random.randrange(3, 45) for _ in range(N)]]
x = np.random.randn(4096).astype(np.float32) * 0.05; w = np.ones(E, np.float32) / E
# numa node of pages
libc = ctypes.CDLL("libc.so.6", use_errno=True)
def nodes_of(arr, n=64):
    base = arr.ctypes.data; ln = arr.nbytes; pg = 4096
    addrs = (ctypes.c_void_p * n)(*[(base + i * ln // n) & ~(pg - 1) for i in range(n)])
    st = (ctypes.c_int * n)()
    r = libc.syscall(279, 0, n, addrs, None, st, 0)   # move_pages query
    return list(st)
s = nodes_of(idx["blk.10.ffn_gate_exps.weight"].data, 256)
print("blk.10 gate pages by node:", {k: s.count(k) for k in set(s)})
for pss in range(3):
 tot_b = 0; t_all = 0
 for i, (L, es) in enumerate(calls):
     ts = [idx[f"blk.{L}.{k}"] for k in K]
     g = [ts[0].data[e] for e in es]; u = [ts[1].data[e] for e in es]; dd = [ts[2].data[e] for e in es]
     t0 = time.perf_counter()
     cpu_moe.moe_cpu(g, u, dd, ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x, w)
     dt = time.perf_counter() - t0
     if i >= 5:
         t_all += dt; tot_b += sum(a.nbytes for a in g + u + dd)
     time.sleep(0.002)                                          # gap like the GPU trunk between layers
 n = N - 5
 print(f'pass {pss}: ', end='')
 print(f"E={E} threads={os.environ.get('OMP_NUM_THREADS')}: {1000*t_all/n:.2f} ms/call  "
       f"{1000*t_all/n/E:.3f} ms/expert  {tot_b/t_all/1e9:.1f} GB/s")
