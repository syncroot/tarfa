#!/usr/bin/env python3
"""RQ4 replay v3: process-pool decode as a GIL-free proxy for a C decode path.

Eight worker processes each run read+zstd-decode for their share of the route
work (results stay in-worker, mirroring decompress-into-pinned-buffer in a
native thread pool). Raw arm uses the same process pool for symmetry.
"""
import concurrent.futures, json, os, pathlib, statistics, sys, time

HERE = pathlib.Path("/home/user/tarfa/layout_research")
ZDIR = HERE / "rq4" / "zbf16"
LAYERS = [0, 15, 30, 45]
G, D = 12_582_912, 6_291_456; REC = G + D

def build_work():
    routes = json.loads(pathlib.Path(
        "/home/user/tarfa/oracle_research/router_signal_routes.json").read_text())
    work = []
    for name in sorted(n for n in routes if routes[n]["role"] == "test"):
        r = {int(L): v["ids"] for L, v in routes[name]["routes"].items()}
        T = routes[name]["tokens"]
        for t in range(T - 6, T):
            for L in LAYERS:
                for e in r[L][t]:
                    work.append((L, e))
    return work

_STATE = {}
def _init():
    import zstandard
    sys.path.insert(0, "/home/user/tarfa/runtime")
    import exact_bf16
    _STATE["meta"] = exact_bf16.meta()
    _STATE["man"] = {L: json.loads((ZDIR / ("layer_%02d.json" % L)).read_text())
                     for L in LAYERS}
    _STATE["dctx"] = zstandard.ZstdDecompressor()
    _STATE["fds"] = {}

def _fd(p):
    p = str(p)
    if p not in _STATE["fds"]: _STATE["fds"][p] = os.open(p, os.O_RDONLY)
    return _STATE["fds"][p]

def work_raw(chunk):
    n = 0; meta = _STATE["meta"]
    for L, e in chunk:
        m = meta[L]
        p, off, _, _ = m["gate_up_proj"]; n += len(os.pread(_fd(p), G, off + e * G))
        p, off, _, _ = m["down_proj"]; n += len(os.pread(_fd(p), D, off + e * D))
    return n

def work_z(chunk):
    n = 0
    for L, e in chunk:
        m = _STATE["man"][L]["experts"][str(e)]
        frame = os.pread(_fd(ZDIR / ("layer_%02d.zbf16" % L)), m["clen"], m["offset"])
        out = _STATE["dctx"].decompress(frame, max_output_size=REC)
        assert len(out) == REC
        n += m["clen"]
    return n

def evict():
    sys.path.insert(0, "/home/user/tarfa/runtime")
    import exact_bf16
    meta = exact_bf16.meta(); fds = {}
    def fd(p):
        if p not in fds: fds[p] = os.open(p, os.O_RDONLY)
        return fds[p]
    for L in sorted(meta):
        for nm in ("gate_up_proj", "down_proj"):
            os.posix_fadvise(fd(meta[L][nm][0]), 0, 0, os.POSIX_FADV_DONTNEED)
    for L in LAYERS:
        p = str(ZDIR / ("layer_%02d.zbf16" % L))
        os.posix_fadvise(fd(p), 0, 0, os.POSIX_FADV_DONTNEED)
    for f in fds.values(): os.close(f)

if __name__ == "__main__":
    work = build_work()
    # interleave-chunk the work to spread layers across workers (route order kept
    # inside each chunk; disk sees the same aggregate demand as production)
    NPROC = 8
    chunks = [work[i::NPROC] for i in range(NPROC)]
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=NPROC, initializer=_init)
    # warm the pool
    list(pool.map(work_raw, [[] for _ in range(NPROC)]))
    trials = {"raw": [], "z": []}; nb = {}
    for i in range(3):
        evict(); t0 = time.perf_counter()
        nb["raw"] = sum(f.result() for f in [pool.submit(work_raw, c) for c in chunks])
        trials["raw"].append(time.perf_counter() - t0)
        evict(); t0 = time.perf_counter()
        nb["z"] = sum(f.result() for f in [pool.submit(work_z, c) for c in chunks])
        trials["z"].append(time.perf_counter() - t0)
        print("trial %d: raw=%.3fs z=%.3fs" % (i, trials["raw"][-1], trials["z"][-1]), flush=True)
    med_r = statistics.median(trials["raw"]); med_z = statistics.median(trials["z"])
    result = {"nproc": NPROC, "experts": len(work), "plaintext_bytes": nb["raw"],
              "compressed_bytes": nb["z"], "ratio": nb["raw"] / nb["z"],
              "trials_raw": trials["raw"], "trials_z": trials["z"],
              "median_raw_s": med_r, "median_z_s": med_z,
              "improvement": 1 - med_z / med_r, "speedup": med_r / med_z}
    (HERE / "rq4" / "replay3_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("RQ4_GATE", "PASS" if result["improvement"] >= 0.15 else "FAIL")
