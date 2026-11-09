"""
Fused k-quant matvec (Triton): y = W @ x reading raw ggml Q4_K / Q5_K / Q6_K blocks and
dequantizing in registers -- no materialized fp16/fp32 matrix (llama.cpp MMVQ analogue).
Per block the dot product factors as  d*sum_j S_j*(q.x)_j - dmin*sum_j M_j*(sum x)_j,
accumulated in fp32. Summation order differs from dequant-then-matmul (fp32 rounding only).
Self-test: python qmv.py <gguf_dir>  -> rel. error vs gpu_dequant + matmul, and timings.
"""
import torch, triton, triton.language as tl
from gguf import GGMLQuantizationType as QT


@triton.jit
def _f16(p, m):
    lo = tl.load(p, mask=m, other=0).to(tl.uint16)
    hi = tl.load(p + 1, mask=m, other=0).to(tl.uint16)
    return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _u8(p, m):
    return tl.load(p, mask=m, other=0).to(tl.int32)


@triton.jit
def _sm(bp, rm, j: tl.constexpr):
    # 6-bit scale/min for sub-block j (ggml get_scale_min_k4); scales start at byte 4
    if j < 4:
        s = _u8(bp + 4 + j, rm) & 63
        m = _u8(bp + 4 + j + 4, rm) & 63
    else:
        s = (_u8(bp + 4 + j + 4, rm) & 15) | ((_u8(bp + 4 + j - 4, rm) >> 6) << 4)
        m = (_u8(bp + 4 + j + 4, rm) >> 4) | ((_u8(bp + 4 + j, rm) >> 6) << 4)
    return s.to(tl.float32), m.to(tl.float32)


@triton.jit
def _q45k_mv(W, X, Y, rows, row_bytes, nblk, BB: tl.constexpr, IS5: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rm = r < rows
    l32 = tl.arange(0, 32)
    acc = tl.zeros([BR], dtype=tl.float32)
    rowp = W + r.to(tl.int64) * row_bytes
    QS: tl.constexpr = 48 if IS5 else 16
    for b in range(nblk):
        bp = rowp + b * BB
        d = _f16(bp, rm)
        dmin = _f16(bp + 2, rm)
        if IS5:
            qh = tl.load(bp[:, None] + 16 + l32[None, :], mask=rm[:, None], other=0).to(tl.int32)
        xb = X + b * 256
        for c in tl.static_range(4):
            q = tl.load(bp[:, None] + QS + 32 * c + l32[None, :], mask=rm[:, None], other=0).to(tl.int32)
            lo = q & 15
            hi = q >> 4
            if IS5:
                lo += ((qh >> (2 * c)) & 1) << 4
                hi += ((qh >> (2 * c + 1)) & 1) << 4
            xl = tl.load(xb + 64 * c + l32)
            xh = tl.load(xb + 64 * c + 32 + l32)
            qxl = tl.sum(lo.to(tl.float32) * xl[None, :], axis=1)
            qxh = tl.sum(hi.to(tl.float32) * xh[None, :], axis=1)
            sxl = tl.sum(xl, axis=0)
            sxh = tl.sum(xh, axis=0)
            s0, m0 = _sm(bp, rm, 2 * c)
            s1, m1 = _sm(bp, rm, 2 * c + 1)
            acc += d * (s0 * qxl + s1 * qxh) - dmin * (m0 * sxl + m1 * sxh)
    tl.store(Y + r, acc, mask=rm)


@triton.jit
def _q6k_mv(W, X, Y, rows, row_bytes, nblk, BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rm = r < rows
    l32 = tl.arange(0, 32)
    lo16 = l32 < 16
    acc = tl.zeros([BR], dtype=tl.float32)
    rowp = W + r.to(tl.int64) * row_bytes
    for b in range(nblk):
        bp = rowp + b * 210
        d = _f16(bp + 208, rm)
        for n in tl.static_range(2):
            A = tl.load(bp[:, None] + 64 * n + l32[None, :], mask=rm[:, None], other=0).to(tl.int32)
            B = tl.load(bp[:, None] + 64 * n + 32 + l32[None, :], mask=rm[:, None], other=0).to(tl.int32)
            h = tl.load(bp[:, None] + 128 + 32 * n + l32[None, :], mask=rm[:, None], other=0).to(tl.int32)
            xb = X + b * 256 + 128 * n
            for g in tl.static_range(4):
                if g == 0:
                    q = (A & 15) | ((h & 3) << 4)
                elif g == 1:
                    q = (B & 15) | (((h >> 2) & 3) << 4)
                elif g == 2:
                    q = (A >> 4) | (((h >> 4) & 3) << 4)
                else:
                    q = (B >> 4) | (((h >> 6) & 3) << 4)
                p = (q - 32).to(tl.float32) * tl.load(xb + 32 * g + l32)[None, :]
                u0 = tl.sum(tl.where(lo16[None, :], p, 0.0), axis=1)
                u1 = tl.sum(tl.where(lo16[None, :], 0.0, p), axis=1)
                sc0 = tl.load(bp + 192 + 8 * n + 2 * g, mask=rm, other=0).to(tl.int8, bitcast=True).to(tl.float32)
                sc1 = tl.load(bp + 192 + 8 * n + 2 * g + 1, mask=rm, other=0).to(tl.int8, bitcast=True).to(tl.float32)
                acc += d * (sc0 * u0 + sc1 * u1)
    tl.store(Y + r, acc, mask=rm)


BR_DEFAULT = 32


def qmv(raw, qtype, x, cols, out=None, BR=BR_DEFAULT):
    """raw: uint8 cuda [rows, row_bytes] (contiguous); x: fp32 cuda [cols] -> y fp32 [rows]."""
    qtype = QT(qtype)
    rows, row_bytes = raw.shape
    nblk = cols // 256
    y = out if out is not None else torch.empty(rows, device=raw.device, dtype=torch.float32)
    grid = (triton.cdiv(rows, BR),)
    with torch.cuda.device(raw.device):          # compile for / launch on the tensor's GPU
        if qtype == QT.Q4_K:
            _q45k_mv[grid](raw, x, y, rows, row_bytes, nblk, BB=144, IS5=False, BR=BR)
        elif qtype == QT.Q5_K:
            _q45k_mv[grid](raw, x, y, rows, row_bytes, nblk, BB=176, IS5=True, BR=BR)
        elif qtype == QT.Q6_K:
            _q6k_mv[grid](raw, x, y, rows, row_bytes, nblk, BR=BR)
        else:
            raise NotImplementedError(qtype.name)
    return y


if __name__ == "__main__":
    import sys, glob, os, time, numpy as np
    from gguf import GGUFReader
    from gpu_dequant import dequant_rows, BLOCK_BYTES
    d = sys.argv[1]
    idx = {}; rs = []
    for sh in sorted(glob.glob(d + "/*.gguf")):
        r = GGUFReader(sh); rs.append(r)
        for t in r.tensors: idx[t.name] = t
    picks = {}
    for name, t in idx.items():
        qt = QT(t.tensor_type)
        if "_exps" in name and qt in BLOCK_BYTES and qt not in picks:
            picks[qt] = t
    for dev in ("cuda:1", "cuda:0"):
        torch.cuda.set_device(dev)
        print(f"== {dev} {torch.cuda.get_device_name(dev)}")
        for qt, t in picks.items():
            rawn = np.array(t.data[7])
            rows = rawn.shape[0]; cols = rawn.shape[1] // BLOCK_BYTES[qt] * 256
            raw = torch.from_numpy(rawn).to(dev)
            x = torch.randn(cols, device=dev, dtype=torch.float32)
            ref = dequant_rows(rawn, qt, rows, cols, dev, torch.float32) @ x
            for BR in (16, 32, 64):
                y = qmv(raw, qt, x, cols, BR=BR)
                rel = (torch.norm(y - ref) / torch.norm(ref)).item()
                torch.cuda.synchronize(dev)
                for _ in range(3): qmv(raw, qt, x, cols, BR=BR)
                torch.cuda.synchronize(dev); t0 = time.time()
                for _ in range(50): qmv(raw, qt, x, cols, BR=BR)
                torch.cuda.synchronize(dev); ms = 1000 * (time.time() - t0) / 50
                gbs = raw.numel() / (ms / 1000) / 1e9
                print(f"  {qt.name} [{rows}x{cols}] BR={BR:2d}: rel.err {rel:.2e}  {ms:.3f} ms  ({gbs:.0f} GB/s raw)")
