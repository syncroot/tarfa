"""
GPU dequantization of ggml k-quant blocks (Q4_K, Q5_K, Q6_K) in torch, vectorized over
all blocks. Replaces the numpy gguf.quants.dequantize on the expert hot path (73 ms ->
GPU). Block layouts (QK_K = 256):
  Q4_K 144 B: d f16 | dmin f16 | scales u8[12] | qs u8[128]
  Q5_K 176 B: d f16 | dmin f16 | scales u8[12] | qh u8[32] | qs u8[128]
  Q6_K 210 B: ql u8[128] | qh u8[64] | scales i8[16] | d f16
Self-test: python gpu_dequant.py <gguf_dir>  -> bit-exact vs gguf.quants.dequantize.
"""
import torch
from gguf import GGMLQuantizationType as QT

BLOCK_BYTES = {QT.Q4_K: 144, QT.Q5_K: 176, QT.Q6_K: 210}


def _f16(raw, a, b):
    return raw[:, a:b].contiguous().view(torch.float16).float()       # [nb, 1]


def _scale_min_k4(sc):
    # sc int [nb,12] -> S, M int [nb,8] (6-bit scales/mins, ggml get_scale_min_k4)
    lo = torch.stack([sc[:, j] & 63 for j in range(4)], 1)
    lo_m = torch.stack([sc[:, j + 4] & 63 for j in range(4)], 1)
    hi = torch.stack([(sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4) for j in range(4, 8)], 1)
    hi_m = torch.stack([(sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4) for j in range(4, 8)], 1)
    return torch.cat([lo, hi], 1).float(), torch.cat([lo_m, hi_m], 1).float()


def deq_q4k(raw):                                   # raw uint8 [nb,144] -> [nb*256] fp32
    nb = raw.shape[0]
    d, dmin = _f16(raw, 0, 2), _f16(raw, 2, 4)
    S, M = _scale_min_k4(raw[:, 4:16].int())
    qs = raw[:, 16:144].int().reshape(nb, 4, 32)
    q = torch.stack([qs & 0xF, qs >> 4], 2).reshape(nb, 8, 32).float()
    y = d[:, :, None] * S[:, :, None] * q - dmin[:, :, None] * M[:, :, None]
    return y.reshape(-1)


def deq_q5k(raw):                                   # raw uint8 [nb,176]
    nb = raw.shape[0]
    d, dmin = _f16(raw, 0, 2), _f16(raw, 2, 4)
    S, M = _scale_min_k4(raw[:, 4:16].int())
    qh = raw[:, 16:48].int()                        # [nb,32]
    qs = raw[:, 48:176].int().reshape(nb, 4, 32)
    c = torch.arange(4, device=raw.device)
    hlo = ((qh[:, None, :] >> (2 * c)[None, :, None]) & 1) << 4
    hhi = ((qh[:, None, :] >> (2 * c + 1)[None, :, None]) & 1) << 4
    q = torch.stack([(qs & 0xF) + hlo, (qs >> 4) + hhi], 2).reshape(nb, 8, 32).float()
    y = d[:, :, None] * S[:, :, None] * q - dmin[:, :, None] * M[:, :, None]
    return y.reshape(-1)


def deq_q6k(raw):                                   # raw uint8 [nb,210]
    nb = raw.shape[0]
    ql = raw[:, 0:128].int(); qh = raw[:, 128:192].int()
    sc = raw[:, 192:208].contiguous().view(torch.int8).int().float()
    d = _f16(raw, 208, 210)
    idx = torch.arange(32, device=raw.device) // 16
    halves = []
    for n in range(2):
        A = ql[:, 64 * n:64 * n + 32]; B = ql[:, 64 * n + 32:64 * n + 64]
        h = qh[:, 32 * n:32 * n + 32]; s = sc[:, 8 * n:8 * n + 8]
        q1 = ((A & 0xF) | ((h & 3) << 4)) - 32
        q2 = ((B & 0xF) | (((h >> 2) & 3) << 4)) - 32
        q3 = ((A >> 4) | (((h >> 4) & 3) << 4)) - 32
        q4 = ((B >> 4) | (((h >> 6) & 3) << 4)) - 32
        halves.append(torch.cat([q1 * s[:, idx], q2 * s[:, idx + 2],
                                 q3 * s[:, idx + 4], q4 * s[:, idx + 6]], 1))
    return (torch.cat(halves, 1) * d).reshape(-1)


_DEQ = {QT.Q4_K: deq_q4k, QT.Q5_K: deq_q5k, QT.Q6_K: deq_q6k}


def dequant_rows(raw_np, qtype, rows, cols, device, out_dtype=torch.float16):
    """raw_np: numpy uint8 [rows, bytes_per_row] (one expert slice) -> torch [rows, cols]."""
    qtype = QT(qtype)
    bb = BLOCK_BYTES[qtype]
    raw = torch.from_numpy(raw_np).to(device, non_blocking=True).reshape(-1, bb)
    return _DEQ[qtype](raw).reshape(rows, cols).to(out_dtype)


if __name__ == "__main__":
    import sys, glob, os, time, numpy as np, gguf
    from gguf import GGUFReader
    d = sys.argv[1]; dev = os.environ.get("GLM5_DEV", "cuda:1")
    idx = {}; rs = []
    for sh in sorted(glob.glob(d + "/*.gguf")):
        r = GGUFReader(sh); rs.append(r)
        for t in r.tensors: idx[t.name] = t
    seen = set()
    for name, t in idx.items():
        qt = QT(t.tensor_type)
        if "_exps" not in name or qt not in BLOCK_BYTES or qt in seen: continue
        seen.add(qt)
        raw = np.array(t.data[7])
        ref = gguf.quants.dequantize(raw, qt).astype(np.float32)
        rows, cols = ref.shape
        mine = dequant_rows(raw, qt, rows, cols, dev, torch.float32).cpu().numpy()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20): dequant_rows(raw, qt, rows, cols, dev)
        torch.cuda.synchronize(); ms = 1000 * (time.time() - t0) / 20
        print(f"{qt.name:5s} {name:30s} exact={np.array_equal(mine, ref)} "
              f"maxdiff={np.max(np.abs(mine-ref)):.2e}  gpu {ms:.2f} ms/matrix")
