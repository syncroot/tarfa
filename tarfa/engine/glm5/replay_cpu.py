"""Replay the CPU expert work of a recorded decode (routing trace: sel + resident set) outside
the engine: the same expert bytes in the same order, with a gap between calls that stands in for
the GPU phase. No GPU needed. Prints where each call's time goes (cpumoe_prof).
Usage: python replay_cpu.py <gguf_dir> <trace.npz> [tokens] [gap_ms] [gap: spin|sleep|none]"""
import os, sys, glob, time, ctypes
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
import cpu_moe

d, trace = sys.argv[1], sys.argv[2]
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
GAP = float(sys.argv[4]) / 1000 if len(sys.argv) > 4 else 0.0007
MODE = sys.argv[5] if len(sys.argv) > 5 else "spin"
idx = {}
for s in sorted(glob.glob(d + "/*.gguf")):
    for t in GGUFReader(s).tensors:
        if "_exps" in t.name:
            idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
z = np.load(trace); sel, res = z["sel"][:NTOK], z["resident"]
calls = []
for t in range(sel.shape[0]):
    for l in range(sel.shape[1]):
        L = l + 3
        miss = [int(e) for e in sel[t, l] if not res[L, e]]
        if miss:
            calls.append((L, miss))
nn = cpu_moe.numa_setup()
ts = {L: [idx[f"blk.{L}.{k}"] for k in K] for L in range(3, 45)}
uniq = sorted({(L, e) for L, es in calls for e in es})
nbytes = sum(sum(ts[L][i].data[e].nbytes for i in range(3)) for L, es in calls for e in es)
t0 = time.time(); bad = 0
for L, e in uniq:                       # cache + place every expert the replay touches
    for i in range(3):
        a = ts[L][i].data[e]; np.asarray(a).sum(); bad += cpu_moe.place_split(a, nn)
print(f"{len(calls)} calls, {sum(len(es) for _, es in calls)} expert jobs, {len(uniq)} distinct experts, "
      f"{nbytes/1e9:.1f} GB read per pass; placed in {time.time()-t0:.0f}s, unmovable pages {bad}", flush=True)
x = (np.random.randn(4096) * 0.05).astype(np.float32)
y = np.empty(4096, np.float32)
lib = cpu_moe._lib
lib.cpumoe_prof_get.argtypes = [ctypes.POINTER(ctypes.c_double)]
NAMES = ["calls", "experts", "total", "wake", "p1 wall", "p1 busy", "quant", "p3 wall", "p3 busy", "join"]


for L in ts:
    cpu_moe.bind_layer(L, *ts[L])
sel8 = np.zeros(8, np.int64); w8 = np.zeros(8, np.float32)
A = [a.ctypes.data for a in (sel8, w8, x, y)]


def one_pass(kernel):                                  # numa: moe_cpu as the engine called it; layer: moe_cpu_layer
    lib.cpumoe_prof(1)
    tk = 0.0; pc = time.perf_counter
    for L, es in calls:
        if kernel == "layer":
            n = len(es); sel8[:n] = es; w8[:n] = 1.0 / n
            t0 = pc()
            cpu_moe.moe_cpu_layer(L, (1 << n) - 1, *A)
        else:
            g = [ts[L][0].data[e] for e in es]; u = [ts[L][1].data[e] for e in es]; dn = [ts[L][2].data[e] for e in es]
            w = np.full(len(es), 1.0 / len(es), np.float32)
            q = [t.tensor_type for t in ts[L]]
            t0 = pc()
            cpu_moe.moe_cpu(g, u, dn, *q, x, w, out=y)
        t1 = pc(); tk += t1 - t0
        if MODE == "spin":
            while pc() - t1 < GAP: pass
        elif MODE == "sleep":
            time.sleep(GAP)
    p = (ctypes.c_double * 10)(); lib.cpumoe_prof_get(p); p = list(p)
    th = (ctypes.c_double * 16)(); lib.cpumoe_prof_threads(th, 8)
    return tk, p, [round(1e6 * v / p[0]) for v in th]


def same():                                             # old vs new entry: same bits
    import random
    random.seed(3); bad = 0
    for L, es in random.sample(calls, 40):
        n = len(es); sel8[:n] = es; w8[:n] = np.random.rand(n).astype(np.float32)
        xx = (np.random.randn(4096) * 0.3).astype(np.float32)
        ref = cpu_moe.moe_cpu([ts[L][0].data[e] for e in es], [ts[L][1].data[e] for e in es], [ts[L][2].data[e] for e in es],
                              *[t.tensor_type for t in ts[L]], xx, w8[:n].copy()).copy()
        out = np.empty(4096, np.float32)
        cpu_moe.moe_cpu_layer(L, (1 << n) - 1, sel8.ctypes.data, w8.ctypes.data, xx.ctypes.data, out.ctypes.data)
        bad += not np.array_equal(ref.view(np.uint32), out.view(np.uint32))
    print(f"bit-identity of moe_cpu_layer vs moe_cpu_numa over 40 calls: {'OK' if not bad else f'{bad} MISMATCHES'}", flush=True)


same()
one_pass("numa")                                        # warm
for rnd in range(int(os.environ.get("ROUNDS", "2"))):
    for k in os.environ.get("KERNELS", "numa,layer").split(","):
        tk, p, th = one_pass(k)
        n = p[0]
        print(f"{k:5s}: {1000*tk/NTOK:.1f} ms/token {nbytes/1e9/tk:.1f} GB/s | us/call: "
              + "  ".join(f"{NAMES[i]} {1e6*p[i]/n:.0f}" for i in range(2, 10))
              + f"  py {1e6*(tk-p[2])/n:.0f} | p1/thread {th[:8]} p3/thread {th[8:]}", flush=True)
