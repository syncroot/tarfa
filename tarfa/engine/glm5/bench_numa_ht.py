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
random.seed(11)
calls = [(L, random.sample(range(288), E)) for L in [random.randrange(3, 45) for _ in range(N)]]
x = np.random.randn(4096).astype(np.float32) * 0.05; w = np.ones(E, np.float32) / E
def arrs(L, es):
    ts = [idx[f"blk.{L}.{k}"] for k in K]
    return [ts[0].data[e] for e in es], [ts[1].data[e] for e in es], [ts[2].data[e] for e in es], ts
def run():
    tt = 0
    for i, (L, es) in enumerate(calls):
        g, u, dd, ts = arrs(L, es)
        t0 = time.perf_counter(); cpu_moe.moe_cpu(g, u, dd, ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x, w)
        if i >= 5: tt += time.perf_counter() - t0
        time.sleep(0.0015)
    return 1000 * tt / (N - 5) / E
nn = cpu_moe.numa_setup(); run()
for L, es in calls:
    for a in sum(arrs(L, es)[:3], []): cpu_moe.place_split(a, nn)
cpus = lambda n: sorted(int(p.rsplit("cpu", 1)[1]) for p in glob.glob(f"/sys/devices/system/node/node{n}/cpu[0-9]*"))
print("node cpus", cpus(0), cpus(1))
for name, groups in (("4+4 cores", [cpu_moe.node_cpus(0), cpu_moe.node_cpus(1)]),
                     ("8+8 with HT", [cpus(0), cpus(1)]),
                     ("6+6 (4 cores + 2 HT)", [cpus(0)[:6] if False else cpu_moe.node_cpus(0) + [c for c in cpus(0) if c not in cpu_moe.node_cpus(0)][:2],
                                               cpu_moe.node_cpus(1) + [c for c in cpus(1) if c not in cpu_moe.node_cpus(1)][:2]])):
    cpu_moe.pin(groups[0] + groups[1]); cpu_moe._lib.cpumoe_set_nodes(2)
    run(); print(f"{name:22s} {run():.3f} ms/expert  pins {groups}")
