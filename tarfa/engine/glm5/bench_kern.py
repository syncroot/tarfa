"""CPU experts, NUMA-split kernel on placed pages: ggml AVX2 vs own AVX-512 dots, physical cores vs
all hyperthreads, interleaved in one process (3 rounds) so page-cache/thermal drift hits all alike."""
import os, sys, glob, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
import cpu_moe
d = sys.argv[1]; E = 6; N = 120
idx = {}
for s in sorted(glob.glob(d + "/*.gguf")):
    for t in GGUFReader(s).tensors: idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
random.seed(5)
calls = [(L, random.sample(range(288), E)) for L in [random.randrange(3, 45) for _ in range(N)]]
x = (np.random.randn(2, 4096) * 0.05).astype(np.float32); w = np.ones(E, np.float32) / E
w2 = np.ones((E, 2), np.float32) / E
def arrs(L, es):
    ts = [idx[f"blk.{L}.{k}"] for k in K]
    return [ts[0].data[e] for e in es], [ts[1].data[e] for e in es], [ts[2].data[e] for e in es], ts
nn = cpu_moe.numa_setup()
for L, es in calls:
    for a in sum(arrs(L, es)[:3], []):
        np.asarray(a).sum(); cpu_moe.place_split(a, nn)
phys = [c for n in cpu_moe.numa_nodes() for c in cpu_moe.node_cpus(n)]
allc = [sorted(int(q.rsplit("cpu", 1)[1]) for q in glob.glob(f"/sys/devices/system/node/node{n}/cpu[0-9]*")) for n in cpu_moe.numa_nodes()]
allc = [c for g in allc for c in g]
def run(two):
    tt = 0
    for i, (L, es) in enumerate(calls):
        g, u, dd, ts = arrs(L, es)
        t0 = time.perf_counter()
        if two:
            cpu_moe.moe_cpu_numa_t(g, u, dd, ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x, w2)
        else:
            cpu_moe.moe_cpu(g, u, dd, ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x[0], w)
        if i >= 5: tt += time.perf_counter() - t0
        time.sleep(0.0015)
    return 1000 * tt / (N - 5) / E
res = {}
CFG = [(ht, av, pf) for ht in ("cores", "ht") for av, pf in ((0, 0), (1, 0), (1, 512), (1, 1024), (1, 2048), (1, 4096))]
run(False)
for rnd in range(3):
    for ht, av, pf in CFG:
        cpu_moe.pin(phys if ht == "cores" else allc)
        cpu_moe._lib.cpumoe_set_avx512(av); cpu_moe._lib.cpumoe_set_pf_bytes(pf)
        res.setdefault((ht, av, pf), []).append((run(False), run(True)))
for (ht, av, pf), v in res.items():
    a = np.array(v)[1:]
    print(f"{ht:5s} {'avx512' if av else 'ggml  '} pf {pf:4d}: 1 token {a[:,0].mean():.3f} ms/expert (rounds {np.round(a[:,0],3)}), "
          f"2 tokens {a[:,1].mean():.3f}", flush=True)
