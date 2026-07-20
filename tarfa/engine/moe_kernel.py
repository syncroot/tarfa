# -*- coding: utf-8 -*-
"""
Fused int4-dequant + GEMV MoE decode kernel (Triton).  ISOLATED - self-contained, no external engine imports.

One kernel does: unpack int4 in registers/SRAM -> matrix-vector product -> per-row scale ->
(optional) routing-weight scale -> (optional) accumulate into a shared output vector.
The dequantized fp16 weights are NEVER materialized in VRAM - only (q-8) lives in registers,
and the per-row scale is folded onto the final partial sum.

Format (int4 nibble-packed shards, symmetric int4):
  W_packed : uint8  [E, N, KB]   two 4-bit weights per byte (low nibble = even col, high = odd col)
  Scale    : fp16   [E, N]       one scale per OUTPUT row n  (group size = full input row)
  X        : fp16/bf16 [E, IN]   per-expert input vector (IN == 2*KB)
  value    : w = (nibble - 8) * scale[n]      (no explicit zero-point; zero is the implicit 8)

Two projections compose the SwiGLU FFN:
  up   : gate_up  N=2*intermediate, IN=hidden     -> [E, 2*inter]  (no routing, no accumulate)
  swiglu (cheap elementwise, plain torch)          -> [E, inter]
  down : N=hidden, IN=intermediate                 -> accumulate (sum over E) -> Y[hidden], routing-scaled
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_moe_gemv(
    Wp_ptr, S_ptr, X_ptr, R_ptr, Y_ptr,
    N, KB,
    stride_we, stride_wn,                 # W_packed [E,N,KB]  (kb stride = 1)
    stride_se,                            # Scale    [E,N]     (n  stride = 1)
    stride_xe,                            # X        [E,IN]    (col stride = 1)
    BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr, SPLIT_K: tl.constexpr,
    ACCUMULATE: tl.constexpr, HAS_ROUTING: tl.constexpr,
):
    pid = tl.program_id(0)                 # flattened (expert, output-row-block)
    pid_k = tl.program_id(1)               # split-K slice over the reduction (KB) dimension
    num_n = tl.cdiv(N, BLOCK_N)
    e = pid // num_n
    nb = pid % num_n
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)             # output rows handled by this block
    n_mask = n < N

    kb_per = tl.cdiv(KB, SPLIT_K)                        # this program owns bytes [kb_start, kb_end)
    kb_start = pid_k * kb_per
    kb_end = tl.minimum(kb_start + kb_per, KB)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(0, kb_end - kb_start, BLOCK_KB):
        bs = kb_start + i + tl.arange(0, BLOCK_KB)       # byte indices (each byte = 2 input cols)
        b_mask = bs < kb_end
        wptr = Wp_ptr + e * stride_we + n[:, None] * stride_wn + bs[None, :]
        packed = tl.load(wptr, mask=n_mask[:, None] & b_mask[None, :], other=0).to(tl.int32)
        lo = (packed & 0xF) - 8                          # in-SRAM unpack: even cols
        hi = ((packed >> 4) & 0xF) - 8                   #                 odd cols
        # split x into even/odd lanes so we never have to physically interleave the nibbles
        xe = tl.load(X_ptr + e * stride_xe + 2 * bs,     mask=b_mask, other=0.0).to(tl.float32)
        xo = tl.load(X_ptr + e * stride_xe + 2 * bs + 1, mask=b_mask, other=0.0).to(tl.float32)
        acc += tl.sum(lo.to(tl.float32) * xe[None, :] + hi.to(tl.float32) * xo[None, :], axis=1)

    scale = tl.load(S_ptr + e * stride_se + n, mask=n_mask, other=0.0).to(tl.float32)
    y = acc * scale                                      # fold per-row scale onto the dot
    if HAS_ROUTING:
        y = y * tl.load(R_ptr + e).to(tl.float32)

    if ACCUMULATE:                                       # down-proj: sum over experts (and split-K) -> Y[N]
        tl.atomic_add(Y_ptr + n, y, mask=n_mask)
    elif SPLIT_K > 1:                                    # up-proj with split-K: sum partials over K
        tl.atomic_add(Y_ptr + e * N + n, y, mask=n_mask)
    else:                                                # up-proj, single-K: plain store -> Y[E,N]
        tl.store(Y_ptr + e * N + n, y, mask=n_mask)


def _launch(Wp, S, X, R, Y, N, KB, accumulate, split_k=1, BLOCK_N=64, BLOCK_KB=256, num_warps=4):
    E = Wp.shape[0]
    grid = (E * triton.cdiv(N, BLOCK_N), split_k)
    _fused_moe_gemv[grid](
        Wp, S, X, (R if R is not None else Wp), Y,
        N, KB,
        Wp.stride(0), Wp.stride(1),
        S.stride(0),
        X.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_KB=BLOCK_KB, SPLIT_K=split_k,
        ACCUMULATE=accumulate, HAS_ROUTING=(R is not None),
        num_warps=num_warps,
    )


def moe_ffn(x, gup_packed, gup_scale, down_packed, down_scale, routing_w, split_k_up=1, split_k_down=1):
    """Full SwiGLU MoE FFN for ONE token. x:[hidden]. Returns y:[hidden] (fp16)."""
    E = gup_packed.shape[0]
    dev, dt = x.device, x.dtype
    gN, gKB = gup_packed.shape[1], gup_packed.shape[2]          # up:  N=2*inter, KB=hidden/2
    dN, dKB = down_packed.shape[1], down_packed.shape[2]        # down: N=hidden, KB=inter/2
    xb = x.unsqueeze(0).expand(E, -1).contiguous()             # broadcast token to all experts
    gu = torch.empty((E, gN), device=dev, dtype=torch.float32)
    _launch(gup_packed, gup_scale, xb, None, gu, gN, gKB, accumulate=False, split_k=split_k_up)
    g, u = gu.chunk(2, dim=-1)
    act = (F.silu(g) * u).to(dt).contiguous()                  # [E, inter]
    down_out = torch.empty((E, dN), device=dev, dtype=torch.float32)   # per-expert store (no atomic_add)
    _launch(down_packed, down_scale, act, None, down_out, dN, dKB, accumulate=False, split_k=split_k_down)
    y = (down_out * routing_w.to(torch.float32)[:, None]).sum(0)       # deterministic fixed-order weighted sum
    return y.to(dt)


# ----------------------------------------------------------------------------------------
#  reference + self-test  (run:  python moe_kernel.py)
# ----------------------------------------------------------------------------------------
def _dequant_ref(packed, scale):                               # reference dequant for validation
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = ((packed >> 4) & 0xF).to(torch.int16) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)   # interleave: even=lo, odd=hi
    return q.to(torch.float16) * scale[:, None]


def _ref_ffn(x, gup, gus, dp, ds, rw):
    E = gup.shape[0]; out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(E):
        gw = _dequant_ref(gup[e], gus[e]).float()              # [2*inter, hidden]
        dw = _dequant_ref(dp[e], ds[e]).float()                # [hidden, inter]
        gu = gw @ x.float()
        g, u = gu.chunk(2); a = F.silu(g) * u
        out += rw[e].float() * (dw @ a)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"; dt = torch.float16
    E, HID, INTER = 8, 3072, 1024
    GN, GKB = 2 * INTER, HID // 2          # up:   [E, 2048, 1536]
    DN, DKB = HID, INTER // 2              # down: [E, 3072, 512]
    gup = torch.randint(0, 256, (E, GN, GKB), dtype=torch.uint8, device=dev)
    dp = torch.randint(0, 256, (E, DN, DKB), dtype=torch.uint8, device=dev)
    gus = (torch.randn(E, GN, device=dev) * 0.02).to(dt)
    ds = (torch.randn(E, DN, device=dev) * 0.02).to(dt)
    x = torch.randn(HID, device=dev, dtype=dt) * 0.5
    rw = torch.softmax(torch.randn(E, device=dev), 0).to(dt)

    y = moe_ffn(x, gup, gus, dp, ds, rw)
    ref = _ref_ffn(x, gup, gus, dp, ds, rw)
    err = (y.float() - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    print(f"max|err| {err:.4e}  rel {rel:.2e}   {'PASS' if rel < 2e-2 else 'FAIL'}")

    import time
    for _ in range(5): moe_ffn(x, gup, gus, dp, ds, rw)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(200): moe_ffn(x, gup, gus, dp, ds, rw)
    torch.cuda.synchronize()
    print(f"{(time.time()-t)/200*1e3:.3f} ms / token-layer  (1 MoE FFN, {E} experts)")
