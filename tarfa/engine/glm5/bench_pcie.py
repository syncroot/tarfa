"""Can the GPUs help with CPU-side experts over PCIe? Page-lock a run of experts' page-cache pages
(cudaHostRegister, read-only + mapped), then measure (a) DMA copy per expert to each GPU, alone,
both GPUs at once, and while the CPU expert kernel runs; (b) zero-copy: the masked MoE kernel
reading the weights straight from host memory over PCIe."""
import os, sys, glob, time, threading, mmap
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
import cpu_moe
from qmoe import moe_masked
idx = {}
for s in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
    for t in GGUFReader(s).tensors: idx[t.name] = t
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
L, NE = 20, 16
ts = [idx[f"blk.{L}.{k}"] for k in K]
PG = mmap.PAGESIZE
cr = torch.cuda.cudart()
t0 = time.time()
for t in ts:                                       # one registration per kind over experts 0..NE-1
    a = t.data[:NE]; p = a.ctypes.data; lo = p // PG * PG; hi = -(-(p + a.nbytes) // PG) * PG
    np.asarray(a).sum()
    rc = cr.cudaHostRegister(lo, hi - lo, 1 | 2 | 8)    # portable | mapped | read-only
    assert int(rc) == 0, rc
eb = sum(t.data[0].nbytes for t in ts)
print(f"registered {NE} experts ({NE*eb/1e6:.0f} MB) in {time.time()-t0:.2f}s; expert = {eb/1e6:.1f} MB", flush=True)
nn = cpu_moe.numa_setup()
devs = ["cuda:1", "cuda:0"]
dst = {d: [torch.empty(t.data[0].nbytes, dtype=torch.uint8, device=d) for t in ts] for d in devs}
src = [[torch.from_numpy(np.asarray(t.data[e]).reshape(-1)) for t in ts] for e in range(NE)]
def copy(d, es):
    s = torch.cuda.Stream(d)
    with torch.cuda.device(d), torch.cuda.stream(s):
        for e in es:
            for k in range(3): dst[d][k].copy_(src[e][k], non_blocking=True)
    return s
def timed(fn, n=3):
    fn(); torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1")
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1")
    return (time.perf_counter() - t0) / n
for d in devs:
    dt = timed(lambda: copy(d, range(4)))
    print(f"DMA {d}: {1000*dt/4:.2f} ms/expert = {4*eb/dt/1e9:.1f} GB/s", flush=True)
dt = timed(lambda: (copy("cuda:1", range(4)), copy("cuda:0", range(4, 8))))
print(f"DMA both GPUs at once: {8*eb/dt/1e9:.1f} GB/s total", flush=True)
# CPU kernel on other experts (not registered region) in a thread, DMA meanwhile
cl = [(L2, list(range(100, 106))) for L2 in range(5, 45)]
def cpu_loop(stop, out):
    n = 0; t0 = time.perf_counter()
    x = (np.random.randn(4096) * 0.05).astype(np.float32); w = np.ones(6, np.float32) / 6
    while not stop.is_set():
        L2, es = cl[n % len(cl)]; tt = [idx[f"blk.{L2}.{k}"] for k in K]
        cpu_moe.moe_cpu([tt[0].data[e] for e in es], [tt[1].data[e] for e in es], [tt[2].data[e] for e in es],
                        tt[0].tensor_type, tt[1].tensor_type, tt[2].tensor_type, x, w); n += 1
    out.append(1000 * (time.perf_counter() - t0) / n / 6)
for L2, es in cl:                                   # map + place those pages first
    for k in K:
        for e in es: a = idx[f"blk.{L2}.{k}"].data[e]; np.asarray(a).sum(); cpu_moe.place_split(a, nn)
for label, fn in (("cpu alone", None), ("cpu + DMA both GPUs", lambda: (copy("cuda:1", range(4)), copy("cuda:0", range(4, 8))))):
    stop, out = threading.Event(), []
    th = threading.Thread(target=cpu_loop, args=(stop, out)); th.start()
    if fn is None:
        time.sleep(3)
    else:
        dt = timed(fn, n=10)
    stop.set(); th.join()
    print(f"{label}: CPU {out[0]:.3f} ms/expert" + ("" if fn is None else f", DMA {8*eb/dt/1e9:.1f} GB/s total"), flush=True)
# zero-copy: masked kernel reads weights from host memory
q = [QT(t.tensor_type) for t in ts]; rb = [t.data.shape[2] for t in ts]
for d in devs:
    with torch.cuda.device(d):
        x = torch.randn(4096, device=d) * 0.05
        for E in (1, 2, 4):
            P = torch.zeros(3, 8, dtype=torch.long)
            for j in range(E):
                for k in range(3): P[k, j] = ts[k].data[j].ctypes.data
            P = P.to(d); w = torch.ones(8, device=d) / E; Hm = torch.zeros(8, 2048, device=d); y = torch.zeros(4096, device=d)
            dt = timed(lambda: moe_masked(P, *q, *rb, x, w, Hm, y), n=5)
            print(f"zero-copy kernel {d}, {E} experts: {1000*dt/E:.2f} ms/expert = {E*eb/dt/1e9:.1f} GB/s", flush=True)
