"""
Batched routed-MoE for one layer in two Triton launches (vs ~10 launches per expert):
  A) _moe_gu   grid (E, rows/BR):  h[e] = silu(min(gate_e x, L)) * clamp(up_e x, -L, L)
  B) _moe_down grid (rows/BR):     y    = sum_e w[e] * (down_e h[e])
Experts are addressed through an int64 device-pointer table (raw k-quant bytes stay where
the cache put them). Dot products reuse the qmv block math (fp32 accumulate).
"""
import torch, triton, triton.language as tl
from gguf import GGMLQuantizationType as QT
from qmv import _f16, _sm

LIMIT = 10.0
BB = {QT.Q4_K: 144, QT.Q5_K: 176, QT.Q6_K: 210}


@triton.jit
def _q45k_rowdot(rowp, X, nblk, rm, BBY: tl.constexpr, IS5: tl.constexpr, BR: tl.constexpr):
    l32 = tl.arange(0, 32)
    acc = tl.zeros([BR], dtype=tl.float32)
    QS: tl.constexpr = 48 if IS5 else 16
    for b in range(nblk):
        bp = rowp + b * BBY
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
            s0, m0 = _sm(bp, rm, 2 * c)
            s1, m1 = _sm(bp, rm, 2 * c + 1)
            acc += d * (s0 * tl.sum(lo.to(tl.float32) * xl[None, :], axis=1)
                        + s1 * tl.sum(hi.to(tl.float32) * xh[None, :], axis=1)) \
                - dmin * (m0 * tl.sum(xl, axis=0) + m1 * tl.sum(xh, axis=0))
    return acc


@triton.jit
def _q6k_rowdot(rowp, X, nblk, rm, BR: tl.constexpr):
    l32 = tl.arange(0, 32)
    lo16 = l32 < 16
    acc = tl.zeros([BR], dtype=tl.float32)
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
                sc0 = tl.load(bp + 192 + 8 * n + 2 * g, mask=rm, other=0).to(tl.int8, bitcast=True).to(tl.float32)
                sc1 = tl.load(bp + 192 + 8 * n + 2 * g + 1, mask=rm, other=0).to(tl.int8, bitcast=True).to(tl.float32)
                acc += d * (sc0 * tl.sum(tl.where(lo16[None, :], p, 0.0), axis=1)
                            + sc1 * tl.sum(tl.where(lo16[None, :], 0.0, p), axis=1))
    return acc


@triton.jit
def _moe_gu(GP, UP, X, H, rows, g_rb, u_rb, nblk,
            BBG: tl.constexpr, IS5G: tl.constexpr, BBU: tl.constexpr, IS5U: tl.constexpr,
            BR: tl.constexpr, LIM: tl.constexpr):
    e = tl.program_id(0)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    rm = r < rows
    gb = tl.load(GP + e).to(tl.pointer_type(tl.uint8))
    ub = tl.load(UP + e).to(tl.pointer_type(tl.uint8))
    g = _q45k_rowdot(gb + r.to(tl.int64) * g_rb, X, nblk, rm, BBG, IS5G, BR)
    u = _q45k_rowdot(ub + r.to(tl.int64) * u_rb, X, nblk, rm, BBU, IS5U, BR)
    g = tl.minimum(g, LIM)
    u = tl.minimum(tl.maximum(u, -LIM), LIM)
    tl.store(H + e * rows + r, g / (1.0 + tl.exp(-g)) * u, mask=rm)


@triton.jit
def _moe_down(DP, Hm, W, Y, rows, d_rb, nblk, hcols, E,
              BBD: tl.constexpr, IS5D: tl.constexpr, ISQ6: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < rows
    acc = tl.zeros([BR], dtype=tl.float32)
    for e in range(E):
        db = tl.load(DP + e).to(tl.pointer_type(tl.uint8))
        rowp = db + r.to(tl.int64) * d_rb
        if ISQ6:
            v = _q6k_rowdot(rowp, Hm + e * hcols, nblk, rm, BR)
        else:
            v = _q45k_rowdot(rowp, Hm + e * hcols, nblk, rm, BBD, IS5D, BR)
        acc += tl.load(W + e) * v
    tl.store(Y + r, acc, mask=rm)


def moe_fused(gs, us, ds, qg, qu, qd, x32, w32, BR=32):
    """gs/us/ds: lists of raw uint8 cuda tensors (same device, same shapes) for E experts;
    x32 fp32 [Ne]; w32 fp32 [E] -> fp32 [Ne] = sum_e w_e * down_e(swiglu(gate_e x, up_e x))."""
    dev = x32.device
    E = len(gs)
    nrows_gu, g_rb = gs[0].shape                     # [n_ff, bytes]
    ncols = x32.shape[0]
    rows_d, d_rb = ds[0].shape                       # [Ne, bytes]
    ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    Hm = torch.empty(E, nrows_gu, device=dev, dtype=torch.float32)
    y = torch.empty(rows_d, device=dev, dtype=torch.float32)
    qg, qu, qd = QT(qg), QT(qu), QT(qd)
    with torch.cuda.device(dev):
        _moe_gu[(E, triton.cdiv(nrows_gu, BR))](
            ptr(gs), ptr(us), x32, Hm, nrows_gu, g_rb, us[0].shape[1], ncols // 256,
            BBG=BB[qg], IS5G=(qg == QT.Q5_K), BBU=BB[qu], IS5U=(qu == QT.Q5_K), BR=BR, LIM=LIMIT)
        _moe_down[(triton.cdiv(rows_d, BR),)](
            ptr(ds), Hm, w32.contiguous(), y, rows_d, d_rb, nrows_gu // 256, nrows_gu, E,
            BBD=BB[qd], IS5D=(qd == QT.Q5_K), ISQ6=(qd == QT.Q6_K), BR=BR)
    return y


# ---- masked variants for CUDA graphs: a 0 pointer means "expert not resident here" ----
@triton.jit
def _moe_gu_m(GP, UP, X, H, rows, g_rb, u_rb, nblk,
              BBG: tl.constexpr, IS5G: tl.constexpr, BBU: tl.constexpr, IS5U: tl.constexpr,
              BR: tl.constexpr, LIM: tl.constexpr):
    e = tl.program_id(0)
    gi = tl.load(GP + e)
    if gi != 0:
        r = tl.program_id(1) * BR + tl.arange(0, BR)
        rm = r < rows
        gb = gi.to(tl.pointer_type(tl.uint8))
        ub = tl.load(UP + e).to(tl.pointer_type(tl.uint8))
        g = _q45k_rowdot(gb + r.to(tl.int64) * g_rb, X, nblk, rm, BBG, IS5G, BR)
        u = _q45k_rowdot(ub + r.to(tl.int64) * u_rb, X, nblk, rm, BBU, IS5U, BR)
        g = tl.minimum(g, LIM)
        u = tl.minimum(tl.maximum(u, -LIM), LIM)
        tl.store(H + e * rows + r, g / (1.0 + tl.exp(-g)) * u, mask=rm)


@triton.jit
def _moe_down_m(DP, Hm, W, Y, rows, d_rb, nblk, hcols, E,
                BBD: tl.constexpr, IS5D: tl.constexpr, ISQ6: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < rows
    acc = tl.zeros([BR], dtype=tl.float32)
    for e in range(E):
        di = tl.load(DP + e)
        if di != 0:
            rowp = di.to(tl.pointer_type(tl.uint8)) + r.to(tl.int64) * d_rb
            if ISQ6:
                v = _q6k_rowdot(rowp, Hm + e * hcols, nblk, rm, BR)
            else:
                v = _q45k_rowdot(rowp, Hm + e * hcols, nblk, rm, BBD, IS5D, BR)
            acc += tl.load(W + e) * v
    tl.store(Y + r, acc, mask=rm)


def moe_masked(P, qg, qu, qd, g_rb, u_rb, d_rb, x32, w32, Hm, y, BR=32):
    """P int64 [3, E] device pointers (gate, up, down; 0 = skip) -> y = sum of resident experts.
    Static shapes and no host reads: safe to capture in a CUDA graph."""
    E, nff = Hm.shape
    ne = x32.shape[0]
    qg, qu, qd = QT(qg), QT(qu), QT(qd)
    with torch.cuda.device(x32.device):
        _moe_gu_m[(E, triton.cdiv(nff, BR))](
            P[0], P[1], x32, Hm, nff, g_rb, u_rb, ne // 256,
            BBG=BB[qg], IS5G=(qg == QT.Q5_K), BBU=BB[qu], IS5U=(qu == QT.Q5_K), BR=BR, LIM=LIMIT)
        _moe_down_m[(triton.cdiv(y.shape[0], BR),)](
            P[2], Hm, w32, y, y.shape[0], d_rb, nff // 256, nff, E,
            BBD=BB[qd], IS5D=(qd == QT.Q5_K), ISQ6=(qd == QT.Q6_K), BR=BR)
    return y
