"""Phase timeline of real decode steps: wraps glm5_decode's step pieces with syncs."""
import os, sys, time, collections
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_next as G, experts_gpu as XG
acc = collections.defaultdict(float)
def wrap(mod, name, key):
    f = getattr(mod, name)
    def g(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = f(*a, **k)
        torch.cuda.synchronize(); acc[key] += time.perf_counter() - t0; return r
    setattr(mod, name, g)
import glm5_decode as D
for n, k in (("hc_pre", "hc_pre"), ("hc_post", "hc_post"), ("kda_step", "kda"), ("dense_ffn", "dense"),
             ("moe_t", "moe_total"), ("mla_decode_step", "mla")):
    wrap(D, n, k)
wrap(XG.RawExperts, "lookup", "moe_lookup")
# inside moe: cpu future wait and gpu hits
orig_gpu = XG.RawExperts._gpu
def gpu(self, *a):
    torch.cuda.synchronize(); t0 = time.perf_counter(); orig_gpu(self, *a); torch.cuda.synchronize(); acc["moe_gpu_hits"] += time.perf_counter() - t0
XG.RawExperts._gpu = gpu
import cpu_moe
orig_cpu = cpu_moe.moe_cpu
def cpu(*a, **k):
    t0 = time.perf_counter(); r = orig_cpu(*a, **k); acc["moe_cpu_kernel"] += time.perf_counter() - t0; acc["cpu_experts"] += len(a[0]); return r
XG.cpu_moe_fn = cpu
import builtins
sys.modules["cpu_moe"].moe_cpu = cpu
orig_main_decode = D.main
_t = {}
real_sync = torch.cuda.synchronize
sys.argv = [sys.argv[0]] + sys.argv[1:]
try:
    D.main()
finally:
    n = int(sys.argv[3])
    print("=== per-token phase ms (synced; prefill+decode tokens mixed, normalized by total steps) ===")
    steps = n + 5
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {1000*v/steps:8.2f}" if k != "cpu_experts" else f"  {k:16s} {v/steps:8.1f} per token")
