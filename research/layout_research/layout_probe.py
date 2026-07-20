#!/usr/bin/env python3
"""Read-pattern diagnosis for the LUKS/LVM/ext4 NVMe stack.

Measures, with page cache evicted per arm and identical byte volumes where
possible, how throughput depends on request size, contiguity, and queue depth
- the exact quantities RQ1 (interleaving), RQ2 (ordering) and RQ3 (bundles)
would change. Reads come from the real BF16 shards at authoritative offsets.
"""
import concurrent.futures, json, os, pathlib, random, sys, time

sys.path.insert(0, "/home/user/tarfa/runtime")
import exact_bf16

meta = exact_bf16.meta()
LAYERS = sorted(meta)
random.seed(20260711)

def fdcache():
    fds = {}
    def fd(p):
        if p not in fds: fds[p] = os.open(p, os.O_RDONLY)
        return fds[p]
    return fd
fd = fdcache()

def evict():
    seen = set()
    for L in LAYERS:
        for name in ("gate_up_proj", "down_proj"):
            p = meta[L][name][0]
            if p not in seen:
                os.posix_fadvise(fd(p), 0, 0, os.POSIX_FADV_DONTNEED); seen.add(p)

def cpu_ticks():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + vals[4]
    return sum(vals), idle

def run(reqs, workers):
    """reqs: list of (path, offset, nbytes). Returns dict of measurements."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    total = sum(r[2] for r in reqs)
    t_all0, t_idle0 = cpu_ticks(); t0 = time.perf_counter()
    futs = [pool.submit(os.pread, fd(p), n, o) for p, o, n in reqs]
    got = sum(len(f.result()) for f in futs)
    dt = time.perf_counter() - t0
    t_all1, t_idle1 = cpu_ticks()
    busy = 1 - (t_idle1 - t_idle0) / max(t_all1 - t_all0, 1)
    pool.shutdown()
    assert got == total, (got, total)
    return {"bytes": total, "seconds": round(dt, 4),
            "GBps": round(total / 1e9 / dt, 3), "cpu_busy": round(busy, 3)}

def expert_spans(L, experts):
    m = meta[L]; out = []
    for name in ("gate_up_proj", "down_proj"):
        p, off, shape, _ = m[name]; stride = shape[1] * shape[2] * 2
        out.extend((p, off + e * stride, stride) for e in experts)
    return out

# --- arms ---------------------------------------------------------------
N_LAYERS = 12                     # 12 layers x 8 experts x 18.87MB = 1.81GB/arm
G = 12_582_912; D = 6_291_456; REC = G + D
layers = random.sample(LAYERS, N_LAYERS)
routes = {L: random.sample(range(128), 8) for L in layers}
results = {"config": {"layers": N_LAYERS, "experts_per_layer": 8,
                      "record_bytes": REC, "gate_bytes": G, "down_bytes": D}}

# A. production pattern: 16 separate gate/down preads per layer, pool 48
reqs = [r for L in layers for r in expert_spans(L, routes[L])]
evict(); results["A_production_split16"] = run(reqs, 48)

# B. same bytes, one contiguous 18.87MB read per expert (simulated interleaved
#    record): random offsets inside the same shard files, record-aligned
def rand_span(L, nbytes):
    p, off, shape, _ = meta[L]["gate_up_proj"]
    limit = os.path.getsize(p) - nbytes - (1 << 20)
    start = random.randrange(1 << 20, limit, 4096)
    return (p, start, nbytes)
reqs = [rand_span(L, REC) for L in layers for _ in range(8)]
evict(); results["B_interleaved_1x18MB"] = run(reqs, 48)

# C. bundle scale: one contiguous 151MB read per layer (8 records)
reqs = [rand_span(L, REC * 8) for L in layers]
evict(); results["C_bundle_1x151MB"] = run(reqs, 48)

# D. pure sequential scan: 1.81GB in 16MB chunks from one shard, 1 thread
p = meta[layers[0]]["gate_up_proj"][0]
base = 1 << 20; chunk = 16 << 20
reqs = [(p, base + i * chunk, chunk) for i in range(N_LAYERS * 8 * REC // chunk)]
evict(); results["D_sequential_1thread"] = run(reqs, 1)
evict(); results["D_sequential_8threads"] = run(reqs, 8)

# E. queue-depth sweep at record size (18.87MB random)
for depth in (1, 2, 4, 8, 16, 32, 48):
    reqs = [rand_span(L, REC) for L in layers for _ in range(8)]
    evict(); results["E_depth_%02d" % depth] = run(reqs, depth)

# F. queue-depth sweep at production split sizes
for depth in (8, 16, 48):
    reqs = [r for L in layers for r in expert_spans(L, routes[L])]
    evict(); results["F_split_depth_%02d" % depth] = run(reqs, depth)

out = pathlib.Path("/home/user/tarfa/layout_research/read_pattern_results.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
for k, v in results.items():
    if k == "config": continue
    print("%-24s %7.3f GB/s  cpu %4.1f%%  (%.2fs, %.2fGB)"
          % (k, v["GBps"], v["cpu_busy"] * 100, v["seconds"], v["bytes"] / 1e9))
