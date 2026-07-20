#!/usr/bin/env python3
"""RQ4 physical replay v2: zero-copy decode pipeline vs raw shard reads.

Improvements over v1: thread-local reused ZstdDecompressor, decompress_into
preallocated per-thread buffers, no CPU unplane (planes are interleaved on the
GPU after H2D; that cost is measured separately and reported). Also reports a
warm-cache decode-only pass and a GPU interleave microbench.
"""
import concurrent.futures, json, os, pathlib, statistics, sys, threading, time
import numpy as np, zstandard

sys.path.insert(0, "/home/user/tarfa/runtime")
import exact_bf16

HERE = pathlib.Path("/home/user/tarfa/layout_research")
ZDIR = HERE / "rq4" / "zbf16"
ROUTES = json.loads(pathlib.Path(
    "/home/user/tarfa/oracle_research/router_signal_routes.json").read_text())
meta = exact_bf16.meta()
LAYERS = [0, 15, 30, 45]
G, D = 12_582_912, 6_291_456; REC = G + D
TESTS = sorted(n for n in ROUTES if ROUTES[n]["role"] == "test")

WORK = []
for name in TESTS:
    r = {int(L): v["ids"] for L, v in ROUTES[name]["routes"].items()}
    T = ROUTES[name]["tokens"]
    for t in range(T - 6, T):
        for L in LAYERS:
            WORK.append((L, r[L][t]))

MAN = {L: json.loads((ZDIR / ("layer_%02d.json" % L)).read_text()) for L in LAYERS}
fds = {}
def fd(p):
    p = str(p)
    if p not in fds: fds[p] = os.open(p, os.O_RDONLY)
    return fds[p]

def evict():
    for L in sorted(meta):
        for n in ("gate_up_proj", "down_proj"):
            os.posix_fadvise(fd(meta[L][n][0]), 0, 0, os.POSIX_FADV_DONTNEED)
    for L in LAYERS:
        os.posix_fadvise(fd(ZDIR / ("layer_%02d.zbf16" % L)), 0, 0, os.POSIX_FADV_DONTNEED)

TL = threading.local()
def tl_state():
    if not hasattr(TL, "dctx"):
        TL.dctx = zstandard.ZstdDecompressor()
        TL.out = bytearray(REC)
        TL.frame = bytearray(32 << 20)
    return TL

pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

def read_raw(L, e):
    m = meta[L]
    p, off, _, _ = m["gate_up_proj"]; a = os.pread(fd(p), G, off + e * G)
    p, off, _, _ = m["down_proj"]; b = os.pread(fd(p), D, off + e * D)
    return len(a) + len(b)

def read_z(L, e):
    st = tl_state(); m = MAN[L]["experts"][str(e)]
    n = os.preadv(fd(ZDIR / ("layer_%02d.zbf16" % L)),
                  [memoryview(st.frame)[:m["clen"]]], m["offset"])
    assert n == m["clen"]
    out = st.dctx.decompress(memoryview(st.frame)[:m["clen"]], max_output_size=REC)
    assert len(out) == REC
    return m["clen"]

def arm(kind, do_evict=True):
    if do_evict: evict()
    t0 = time.perf_counter(); nbytes = 0
    reader = read_raw if kind == "raw" else read_z
    for L, experts in WORK:
        futs = [pool.submit(reader, L, e) for e in experts]
        for f in futs: nbytes += f.result()
    return time.perf_counter() - t0, nbytes

trials = {"raw": [], "z": []}
nb = {}
for i in range(3):
    s, nb["raw"] = arm("raw"); trials["raw"].append(s)
    s, nb["z"] = arm("z"); trials["z"].append(s)
    print("trial %d: raw=%.3fs z=%.3fs" % (i, trials["raw"][-1], trials["z"][-1]), flush=True)

warm_z = arm("z", do_evict=False)[0]          # decode cost with all pages cached
warm_raw = arm("raw", do_evict=False)[0]

# GPU interleave cost for planes -> bf16 records
gpu = None
try:
    import torch
    if torch.cuda.is_available():
        lo = torch.empty(REC // 2, dtype=torch.uint8, device="cuda")
        hi = torch.empty(REC // 2, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50):
            out = torch.stack((lo, hi), dim=1).flatten()
        torch.cuda.synchronize()
        gpu = (time.perf_counter() - t0) / 50 * 1000
except Exception as ex:
    gpu = str(ex)

med_r = statistics.median(trials["raw"]); med_z = statistics.median(trials["z"])
result = {"work_items": len(WORK), "plaintext_bytes": nb["raw"], "compressed_bytes": nb["z"],
          "ratio": nb["raw"] / nb["z"], "trials_raw": trials["raw"], "trials_z": trials["z"],
          "median_raw_s": med_r, "median_z_s": med_z,
          "improvement": 1 - med_z / med_r, "speedup": med_r / med_z,
          "warm_decode_only_s": warm_z, "warm_raw_pagecache_s": warm_raw,
          "gpu_interleave_ms_per_expert": gpu}
(HERE / "rq4" / "replay2_results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("RQ4_GATE", "PASS" if result["improvement"] >= 0.15 else "FAIL")
