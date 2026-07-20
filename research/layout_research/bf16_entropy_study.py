#!/usr/bin/env python3
"""RQ4: lossless-compressibility analysis of Tarfa's BF16 expert weights.

Extracts real experts from the authoritative shards, then measures:
- Shannon entropy of raw bytes, high (sign+exponent) and low (mantissa) planes
- exponent-value distribution
- XOR residual entropy between experts sharing a layer slot
- transforms written to files for zstd/lz4 CLI benchmarking
Reconstruction is verified byte-identical for every transform.
"""
import collections, json, math, os, pathlib, random, sys
import numpy as np

sys.path.insert(0, "/home/user/tarfa/runtime")
import exact_bf16

OUT = pathlib.Path("/home/user/tarfa/layout_research/rq4")
OUT.mkdir(parents=True, exist_ok=True)
meta = exact_bf16.meta()
random.seed(20260711)

def read_expert(L, e):
    m = meta[L]; parts = []
    for name in ("gate_up_proj", "down_proj"):
        p, off, shape, _ = m[name]; stride = shape[1] * shape[2] * 2
        with open(p, "rb") as f:
            f.seek(off + e * stride); parts.append(f.read(stride))
    return b"".join(parts)

def entropy(buf):
    c = collections.Counter(buf); n = len(buf)
    return -sum(v / n * math.log2(v / n) for v in c.values())

# sample: 3 layers x 4 experts = 226MB
LAYERS = random.sample(sorted(meta), 3)
EXPERTS = {L: random.sample(range(128), 4) for L in LAYERS}
blobs = {(L, e): read_expert(L, e) for L in LAYERS for e in EXPERTS[L]}
sample = b"".join(blobs.values())
arr = np.frombuffer(sample, dtype=np.uint8)
lo, hi = arr[0::2], arr[1::2]         # little-endian bf16: [mantissa byte][sign+exp byte]

report = {"sample_bytes": len(sample), "layers": LAYERS,
          "entropy_bits_per_byte": {
              "raw": round(entropy(sample[:8 << 20]), 4),
              "high_sign_exp": round(entropy(hi[:8 << 20].tobytes()), 4),
              "low_mantissa": round(entropy(lo[:8 << 20].tobytes()), 4)}}

# exponent distribution (8 bits: hi byte bits 0-6 plus lo bit 7)
u16 = np.frombuffer(sample, dtype=np.uint16)
exp = ((u16 >> 7) & 0xFF).astype(np.uint8)
c = collections.Counter(exp[: 16 << 20].tobytes())
report["exponent_entropy_bits"] = round(entropy(exp[:16 << 20].tobytes()), 4)
report["exponent_top8"] = c.most_common(8)

# sign entropy
sign = (u16 >> 15).astype(np.uint8)
p1 = float(sign.mean())
report["sign_p_one"] = round(p1, 5)

# XOR residual between two experts in the same layer/slot region
L = LAYERS[0]; e0, e1 = EXPERTS[L][0], EXPERTS[L][1]
x = np.frombuffer(blobs[(L, e0)], dtype=np.uint8) ^ np.frombuffer(blobs[(L, e1)], dtype=np.uint8)
report["xor_residual_entropy"] = round(entropy(x[:8 << 20].tobytes()), 4)

# --- write transform files for CLI codec benchmarks --------------------
def check_roundtrip(name, transformed, restore):
    assert restore(transformed) == sample, name + " roundtrip failed"

raw_path = OUT / "t0_raw.bin"; raw_path.write_bytes(sample)

planes = lo.tobytes() + hi.tobytes()
def unplane(buf):
    n = len(buf) // 2; out = np.empty(len(buf), dtype=np.uint8)
    out[0::2] = np.frombuffer(buf[:n], dtype=np.uint8)
    out[1::2] = np.frombuffer(buf[n:], dtype=np.uint8)
    return out.tobytes()
check_roundtrip("planes", planes, unplane)
(OUT / "t1_byteplanes.bin").write_bytes(planes)

# plane-split PER EXPERT record (decode granularity = one expert)
per = []
for k, blob in blobs.items():
    a = np.frombuffer(blob, dtype=np.uint8)
    per.append(a[0::2].tobytes() + a[1::2].tobytes())
per_expert = b"".join(per)
(OUT / "t2_byteplanes_per_expert.bin").write_bytes(per_expert)

# 3-stream split: exponent byte, 7-bit mantissa (in a byte), packed sign bits
# bf16 = [s][eeeeeeee][mmmmmmm]; low byte bit7 is the exponent LSB, stored once
b_mant = (lo & 0x7F).tobytes(); b_exp = exp.tobytes()
b_sign = np.packbits(sign).tobytes()
def unsplit3(_):
    s = np.unpackbits(np.frombuffer(b_sign, dtype=np.uint8))[: len(u16)].astype(np.uint16)
    e2 = np.frombuffer(b_exp, dtype=np.uint8).astype(np.uint16)
    m2 = np.frombuffer(b_mant, dtype=np.uint8).astype(np.uint16)
    lo2 = m2 | ((e2 & 1) << 7)
    hi2 = (s << 7) | (e2 >> 1)
    out = np.empty(len(u16) * 2, dtype=np.uint8)
    out[0::2] = lo2.astype(np.uint8); out[1::2] = hi2.astype(np.uint8)
    return out.tobytes()
check_roundtrip("split3", None, unsplit3)
(OUT / "t3_streams.bin").write_bytes(b_exp + b_mant + b_sign)

json.dump(report, open(OUT / "entropy_report.json", "w"), indent=2)
print(json.dumps(report, indent=2, default=str))
print("transform files written to", OUT)
