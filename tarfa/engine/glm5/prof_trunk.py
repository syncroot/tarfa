"""Kernel-level profile of one trunk segment body (KDA+MoE layer and MLA+MoE layer), eager on cuda:1."""
import os, sys, torch
os.environ.setdefault("GLM5_Q8", "1"); os.environ.setdefault("GLM5_ROUTER_FP32", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_gdecode as GD
from torch.profiler import profile, ProfilerActivity
GD.N_LAYER = 45
eng = GD.Engine.__new__(GD.Engine)
import glm5_next as G
torch.cuda.set_device(GD.DEV)
m = G.GLM5(sys.argv[1]); eng.m = m
need = {3, 4, 7}            # MLA layer 3? check flags
eng.layers = [m.layer(L) if L in (4, 7) else None for L in range(45)]
# minimal state/buffers (copied from Engine.__init__)
z = lambda *s, dt=GD.DT: torch.zeros(*s, device=GD.DEV, dtype=dt)
eng.kda_s = {L: z(64, 128, 128, dt=torch.float32) for L in (4, 7) if not eng.layers[L]["mla"]}
eng.kda_cs = {L: {c: z(3, 8192) for c in "qkv"} for L in eng.kda_s}
eng.mla_kv = {L: z(2048, 512) for L in (4, 7) if eng.layers[L]["mla"]}
eng.pos_t = torch.zeros(1, device=GD.DEV, dtype=torch.long); eng.ar = torch.arange(2048, device=GD.DEV)
eng.res = z(4, 4096); eng.pf = z(4); eng.cf = z(4, 4); eng.x32 = z(4096, dt=torch.float32); eng.w32 = z(8, dt=torch.float32)
eng.sel = torch.zeros(8, device=GD.DEV, dtype=torch.long); eng.sh = z(4096, dt=torch.float32); eng.yr = z(4096, dt=torch.float32)
eng.y0c = z(4096, dt=torch.float32); eng.ycpu = z(4096, dt=torch.float32)
pin = lambda n, dt: torch.zeros(n, dtype=dt).pin_memory()
eng.h_x, eng.h_w, eng.h_sel = pin(4096, torch.float32), pin(8, torch.float32), pin(8, torch.long)
for L in (4, 7):
    print("layer", L, "MLA" if eng.layers[L]["mla"] else "KDA")
    for _ in range(3): eng._body(L)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): eng._body(L)
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_type.name == "CUDA"] if hasattr(p.key_averages()[0], "device_type") else p.key_averages()
    tot = sum(e.self_device_time_total for e in p.key_averages()) / 10
    nk = sum(e.count for e in p.key_averages()) / 10
    print(f"  GPU time per body {tot/1000:.3f} ms, kernels per body {nk:.0f}")
    for e in sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)[:14]:
        print(f"   {e.self_device_time_total/10/1000:7.3f} ms  x{e.count//10:3d}  {e.key[:90]}")

# ---- torch.compile of the segment body ----
import time
torch._dynamo.config.cache_size_limit = 64
for L in (4, 7):
    f = torch.compile(lambda L=L: eng._body(L), dynamic=False, fullgraph=True)
    t0 = time.time()
    for _ in range(3): f()
    torch.cuda.synchronize(); print(f"compiled layer {L} in {time.time()-t0:.0f}s")
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): f()
        torch.cuda.synchronize()
    ka = p.key_averages()
    print(f"  compiled GPU time per body {sum(e.self_device_time_total for e in ka)/10/1000:.3f} ms, "
          f"kernels {sum(e.count for e in ka if e.self_device_time_total > 0)/10:.0f}")
    f0 = f; f = lambda: (f0(), eng._d2h())
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): f()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g): f()
    for name, fn in (("eager graph", None), ("compiled graph", g)):
        if fn is None:
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2): eng._body(L); eng._d2h()
            fn = g2
        fn.replay(); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50): fn.replay()
        torch.cuda.synchronize(); print(f"  {name}: {1000*(time.time()-t0)/50:.3f} ms per replay")
