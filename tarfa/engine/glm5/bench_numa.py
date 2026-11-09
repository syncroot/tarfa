"""CPU experts: interleaved/unknown page placement vs NUMA row-split placement + split kernel."""
import os, sys, glob, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
import cpu_moe
d = sys.argv[1]; E = 5; N = 60
idx = {}
for s in sorted(glob.glob(d + "/*.gguf")):
    for t in GGUFReader(s).tensors: idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
random.seed(int(os.environ.get("SEED", "7")))
calls = [(L, random.sample(range(288), E)) for L in [random.randrange(3, 45) for _ in range(N)]]
x = np.random.randn(4096).astype(np.float32) * 0.05; w = np.ones(E, np.float32) / E
def arrs(L, es):
    ts = [idx[f"blk.{L}.{k}"] for k in K]
    return [ts[0].data[e] for e in es], [ts[1].data[e] for e in es], [ts[2].data[e] for e in es], ts
def run(fn):
    tt = 0; ys = []
    for i, (L, es) in enumerate(calls):
        g, u, dd, ts = arrs(L, es)
        t0 = time.perf_counter()
        ys.append(fn(g, u, dd, ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x, w).copy())
        if i >= 5: tt += time.perf_counter() - t0
        time.sleep(0.0015)
    return 1000 * tt / (N - 5) / E, ys
cpu_moe.pin(cpu_moe.node_cpus(0) + cpu_moe.node_cpus(1))
run(cpu_moe._lib and cpu_moe.moe_cpu)                     # map pages
fr = np.mean([cpu_moe.where(arrs(L, es)[0][0]) for L, es in calls], 0)
print(f"before placement: gate pages on node0/1 = {fr.round(2)}")
ms, ref = run(cpu_moe.moe_cpu); print(f"plain kernel, 8 threads:        {ms:.3f} ms/expert")
nn = cpu_moe.numa_setup()
ms, y1 = run(cpu_moe.moe_cpu); print(f"split kernel, unplaced pages:   {ms:.3f} ms/expert  maxdiff {max(np.abs(a-b).max() for a, b in zip(ref, y1)):.1e}")
t0 = time.time(); bad = 0
for L, es in calls:
    for a in sum(arrs(L, es)[:3], []):
        bad += cpu_moe.place_split(a, nn)
print(f"placed {len(calls)*E*3} matrices in {time.time()-t0:.1f}s, unmovable pages {bad}")
g0 = arrs(*calls[0])[0][0]
print(f"after: first half on nodes {np.round(cpu_moe.where(g0[:1024]), 2)}, second half {np.round(cpu_moe.where(g0[1024:]), 2)}")
ms, y2 = run(cpu_moe.moe_cpu); print(f"split kernel, placed pages:     {ms:.3f} ms/expert  maxdiff {max(np.abs(a-b).max() for a, b in zip(ref, y2)):.1e}")
cpu_moe.pin([c for n in (0, 1) for c in sorted(int(v) for v in open(f"/sys/devices/system/node/node{n}/cpulist").read().replace("-", ",").split(",")[:1])] and
            sum(([c for c in range(16) if os.path.exists(f"/sys/devices/system/node/node{n}/cpu{c}")] for n in (0, 1)), []))
cpu_moe._lib.cpumoe_set_nodes(2)
ms, y3 = run(cpu_moe.moe_cpu); print(f"split kernel, placed, 16 thr:   {ms:.3f} ms/expert")
