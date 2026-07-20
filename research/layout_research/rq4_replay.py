#!/usr/bin/env python3
"""RQ4 physical replay: raw shard reads vs compressed read+decode pipeline.

Held-out test routes (router_signal_routes.json, role=test) restricted to the
four converted layers. Both arms deliver identical plaintext BF16 bytes into
user-space buffers. Byte equality between the two paths is verified in a
separate untimed pass; timed arms do only read(+decode). Page cache evicted
before every trial; trials interleaved A/B. Gate: >=15% median improvement.
"""
import concurrent.futures, hashlib, json, os, pathlib, statistics, sys, time
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
TOKENS_PER_PROMPT = 6

WORK = []
for name in TESTS:
    r = {int(L): v["ids"] for L, v in ROUTES[name]["routes"].items()}
    T = ROUTES[name]["tokens"]
    for t in range(T - TOKENS_PER_PROMPT, T):
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

def unplane(buf):
    n = len(buf) // 2; out = np.empty(len(buf), dtype=np.uint8)
    out[0::2] = np.frombuffer(buf[:n], dtype=np.uint8)
    out[1::2] = np.frombuffer(buf[n:], dtype=np.uint8)
    return out.tobytes()

pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

def read_raw(L, e):
    m = meta[L]; parts = []
    for name, stride in (("gate_up_proj", G), ("down_proj", D)):
        p, off, _, _ = m[name]
        parts.append(os.pread(fd(p), stride, off + e * stride))
    return b"".join(parts)

def read_z(L, e):
    m = MAN[L]["experts"][str(e)]
    frame = os.pread(fd(ZDIR / ("layer_%02d.zbf16" % L)), m["clen"], m["offset"])
    return unplane(zstandard.ZstdDecompressor().decompress(frame, max_output_size=m["raw"])), m["clen"]

def arm(kind):
    t0 = time.perf_counter(); nbytes = 0
    for L, experts in WORK:
        if kind == "raw":
            futs = [pool.submit(read_raw, L, e) for e in experts]
            for f in futs: nbytes += len(f.result())
        else:
            futs = [pool.submit(read_z, L, e) for e in experts]
            for f in futs: nbytes += f.result()[1]
    return time.perf_counter() - t0, nbytes

# untimed byte-equality verification across all distinct (layer, expert) pairs used
pairs = sorted({(L, e) for L, ex in WORK for e in ex})
for L, e in pairs:
    if read_z(L, e)[0] != read_raw(L, e):
        raise SystemExit("BYTE MISMATCH layer %d expert %d" % (L, e))
print("byte equality verified for %d distinct (layer,expert) pairs" % len(pairs), flush=True)

trials = {"raw": [], "z": []}; nb_raw = nb_z = 0
for i in range(3):
    evict(); s, nb_raw = arm("raw"); trials["raw"].append(s)
    evict(); s, nb_z = arm("z"); trials["z"].append(s)
    print("trial %d: raw=%.3fs z=%.3fs" % (i, trials["raw"][-1], trials["z"][-1]), flush=True)

med_r = statistics.median(trials["raw"]); med_z = statistics.median(trials["z"])
result = {"work_items": len(WORK), "plaintext_bytes": nb_raw, "compressed_bytes": nb_z,
          "ratio": nb_raw / nb_z, "trials_raw": trials["raw"], "trials_z": trials["z"],
          "median_raw_s": med_r, "median_z_s": med_z,
          "improvement": 1 - med_z / med_r, "speedup": med_r / med_z,
          "distinct_pairs_byte_verified": len(pairs)}
(HERE / "rq4" / "replay_results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("RQ4_GATE", "PASS" if result["improvement"] >= 0.15 else "FAIL")
