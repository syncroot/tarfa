#!/usr/bin/env python3
"""RQ4 converter: per-expert byte-planed zstd-1 records for representative layers.

Record = zstd frame of [low-byte plane || high-byte plane] of the exact
concatenated gate_up+down BF16 bytes of one expert. One file per layer plus a
manifest with offsets, sizes and SHA-256 of the RAW bytes. Atomic (tmp+rename),
resumable (validated layers are skipped), originals untouched.
"""
import concurrent.futures, hashlib, json, os, pathlib, sys
import numpy as np, zstandard

sys.path.insert(0, "/home/user/tarfa/runtime")
import exact_bf16

LAYERS = [0, 15, 30, 45]
OUT = pathlib.Path("/home/user/tarfa/layout_research/rq4/zbf16")
OUT.mkdir(parents=True, exist_ok=True)
meta = exact_bf16.meta()
G, D = 12_582_912, 6_291_456
NEXP = meta[0]["gate_up_proj"][2][0]          # 256 experts per layer

def read_expert(L, e):
    m = meta[L]; parts = []
    for name, stride in (("gate_up_proj", G), ("down_proj", D)):
        p, off, shape, _ = m[name]
        with open(p, "rb") as f:
            f.seek(off + e * stride); parts.append(f.read(stride))
    raw = b"".join(parts); assert len(raw) == G + D
    return raw

def plane(raw):
    a = np.frombuffer(raw, dtype=np.uint8)
    return a[0::2].tobytes() + a[1::2].tobytes()

def unplane(buf):
    n = len(buf) // 2; out = np.empty(len(buf), dtype=np.uint8)
    out[0::2] = np.frombuffer(buf[:n], dtype=np.uint8)
    out[1::2] = np.frombuffer(buf[n:], dtype=np.uint8)
    return out.tobytes()

def convert_layer(L):
    final = OUT / ("layer_%02d.zbf16" % L); man_p = OUT / ("layer_%02d.json" % L)
    if final.exists() and man_p.exists():
        return "layer %02d: exists, skipped" % L
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=12)
    def job(e):
        raw = read_expert(L, e)
        cctx = zstandard.ZstdCompressor(level=1)   # contexts are not thread-safe
        return e, cctx.compress(plane(raw)), hashlib.sha256(raw).hexdigest(), len(raw)
    frames = {}
    for e, frame, sha, rawlen in pool.map(job, range(NEXP)):
        frames[e] = (frame, sha, rawlen)
    tmp = final.with_suffix(".tmp"); manifest = {"layer": L, "level": 1,
        "transform": "byteplanes", "raw_bytes": G + D, "experts": {}}
    with open(tmp, "wb") as f:
        off = 0
        for e in range(NEXP):
            frame, sha, rawlen = frames[e]
            f.write(frame)
            manifest["experts"][str(e)] = {"offset": off, "clen": len(frame),
                                           "raw": rawlen, "sha256": sha}
            off += len(frame)
    os.replace(tmp, final)
    man_p.write_text(json.dumps(manifest))
    total_c = sum(v["clen"] for v in json.loads(man_p.read_text())["experts"].values())
    return "layer %02d: %.0f MB -> %.0f MB (x%.3f)" % (
        L, NEXP * (G + D) / 1e6, total_c / 1e6, NEXP * (G + D) / total_c)

def validate_layer(L):
    man = json.loads((OUT / ("layer_%02d.json" % L)).read_text())
    dctx = zstandard.ZstdDecompressor()
    with open(OUT / ("layer_%02d.zbf16" % L), "rb") as f:
        for e in range(NEXP):
            m = man["experts"][str(e)]
            f.seek(m["offset"]); frame = f.read(m["clen"])
            raw = unplane(dctx.decompress(frame, max_output_size=m["raw"]))
            if hashlib.sha256(raw).hexdigest() != m["sha256"]:
                return "layer %02d expert %d: SHA MISMATCH" % (L, e)
            if e in (0, NEXP // 2, NEXP - 1) and raw != read_expert(L, e):
                return "layer %02d expert %d: BYTE MISMATCH vs original" % (L, e)
    return "layer %02d: %d/%d sha ok, spot byte-equality ok" % (L, NEXP, NEXP)

for L in LAYERS:
    print(convert_layer(L), flush=True)
for L in LAYERS:
    print(validate_layer(L), flush=True)
print("CONVERT_DONE")
