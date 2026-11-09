"""
Q8_0 trunk weights kept quantized on the GPU (the GGUF stores every non-expert matrix as Q8_0).
Blocks of 32: fp16 d + 32 int8. Repacked once at load into int8 qs [rows, cols] + fp16 d
[rows, cols/32] (aligned loads), and multiplied by a Triton matvec: y = sum_b d_b * (q_b . x),
fp32 accumulate. d*q is exact in fp32, so this is the exact dequantized-weight product; the
fp16-dequantized resident weights it replaces rounded d*q to 11 bits. Half the bytes of fp16.
Self-test: python q8.py <gguf_dir>
"""
import torch, triton, triton.language as tl
from gguf import GGMLQuantizationType as QT


@triton.jit
def _q8_mv(Q, D, X, Y, rows, cols, BR: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < rows
    kb = tl.arange(0, BK)
    jb = tl.arange(0, BK // 32)
    nb = cols // 32
    acc = tl.zeros([BR], dtype=tl.float32)
    qrow = Q + r.to(tl.int64)[:, None] * cols
    drow = D + r.to(tl.int64)[:, None] * nb
    for k in range(0, cols, BK):
        q = tl.load(qrow + (k + kb)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        x = tl.load(X + k + kb).to(tl.float32)
        s = tl.sum(tl.reshape(q * x[None, :], (BR, BK // 32, 32)), axis=2)     # per-block q.x
        d = tl.load(drow + (k // 32 + jb)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        acc += tl.sum(s * d, axis=1)
    tl.store(Y + r, acc, mask=rm)


@torch.library.custom_op("glm5::q8mv", mutates_args=())
def q8mv(q: torch.Tensor, d: torch.Tensor, x: torch.Tensor, BR: int, BK: int, NW: int) -> torch.Tensor:
    """fp32 y = dequant(q, d) @ x. A custom op so torch.compile treats it as one opaque call."""
    rows, cols = q.shape
    y = torch.empty(rows, device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        _q8_mv[(triton.cdiv(rows, BR),)](q, d, x.contiguous(), y, rows, cols, BR=BR, BK=BK, num_warps=NW)
    return y


@q8mv.register_fake
def _(q, d, x, BR, BK, NW):
    return q.new_empty(q.shape[0], dtype=torch.float32)


@triton.jit
def _q8_mm2(Q, D, X, Y, rows, cols, BR: tl.constexpr, BK: tl.constexpr):
    """Y[n, r] = sum_k W[r, k] X[n, k] for two columns n: each weight tile is read once."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < rows
    kb = tl.arange(0, BK)
    jb = tl.arange(0, BK // 32)
    nb = cols // 32
    a0 = tl.zeros([BR], dtype=tl.float32)
    a1 = tl.zeros([BR], dtype=tl.float32)
    qrow = Q + r.to(tl.int64)[:, None] * cols
    drow = D + r.to(tl.int64)[:, None] * nb
    for k in range(0, cols, BK):
        q = tl.load(qrow + (k + kb)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        d = tl.load(drow + (k // 32 + jb)[None, :], mask=rm[:, None], other=0).to(tl.float32)
        x0 = tl.load(X + k + kb).to(tl.float32)
        x1 = tl.load(X + cols + k + kb).to(tl.float32)
        a0 += tl.sum(tl.sum(tl.reshape(q * x0[None, :], (BR, BK // 32, 32)), axis=2) * d, axis=1)
        a1 += tl.sum(tl.sum(tl.reshape(q * x1[None, :], (BR, BK // 32, 32)), axis=2) * d, axis=1)
    tl.store(Y + r, a0, mask=rm)
    tl.store(Y + rows + r, a1, mask=rm)


@torch.library.custom_op("glm5::q8mm", mutates_args=())
def q8mm(q: torch.Tensor, d: torch.Tensor, x: torch.Tensor, BR: int, BK: int, NW: int) -> torch.Tensor:
    """fp32 Y [N, rows] = X [N, cols] @ dequant(q, d).T for N = 1 or 2 (BR=8, BK=256 measured best
    for N = 2 on the RTX 6000: the 155k-row head in 1.29 ms vs 1.13 ms for one column)."""
    rows, cols = q.shape
    if x.shape[0] == 1:
        return q8mv(q, d, x[0], BR, BK, NW)[None]
    assert x.shape[0] == 2
    y = torch.empty(2, rows, device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        _q8_mm2[(triton.cdiv(rows, 8),)](q, d, x.contiguous(), y, rows, cols, BR=8, BK=256, num_warps=4)
    return y


@q8mm.register_fake
def _(q, d, x, BR, BK, NW):
    return q.new_empty(x.shape[0], q.shape[0], dtype=torch.float32)


def mv(W, x):
    """W @ x for a 1-D x, or x @ W.T for x [N, cols] (N <= 2 tokens), whether W is a Q8W or a dense
    tensor (torch.compile-friendly)."""
    if x.dim() == 2:
        return W.mm(x) if isinstance(W, Q8W) else x @ W.T
    return W.mv(x) if isinstance(W, Q8W) else W @ x


class Q8W:
    """Quantized [rows, cols] matrix; `W @ x` for a 1-D x returns y in `out_dtype`."""
    def __init__(self, raw_u8, rows, cols, dev, out_dtype=torch.float16, BR=4, BK=512, NW=4):
        b = torch.from_numpy(raw_u8).to(dev).view(rows, cols // 32, 34)
        self.d = b[:, :, 0:2].contiguous().view(torch.float16).view(rows, cols // 32)
        self.q = b[:, :, 2:34].contiguous().view(torch.int8).view(rows, cols)
        self.shape = (rows, cols)
        self.dtype = out_dtype
        self.device = torch.device(dev)
        self.BR, self.BK, self.NW = BR, BK, NW

    def __matmul__(self, x):
        return self.mv(x)

    def mv(self, x):
        assert x.dim() == 1 and x.shape[0] == self.shape[1]
        return torch.ops.glm5.q8mv(self.q, self.d, x, self.BR, self.BK, self.NW).to(self.dtype)

    def mm(self, x):                    # x [N, cols] -> [N, rows]
        assert x.dim() == 2 and x.shape[1] == self.shape[1]
        return torch.ops.glm5.q8mm(self.q, self.d, x, self.BR, self.BK, self.NW).to(self.dtype)

    def nbytes(self):
        return self.q.numel() + 2 * self.d.numel()


def load_q8(t, rows, cols, dev, out_dtype=torch.float16):
    import numpy as np
    assert QT(t.tensor_type) == QT.Q8_0, t.name
    return Q8W(np.ascontiguousarray(t.data).reshape(rows, cols // 32 * 34), rows, cols, dev, out_dtype)


if __name__ == "__main__":
    import sys, glob, time, numpy as np, gguf
    from gguf import GGUFReader
    idx = {}
    for sh in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
        for t in GGUFReader(sh).tensors:
            idx[t.name] = t
    dev = "cuda:1"; torch.cuda.set_device(dev)
    for name, shp in (("blk.4.attn_q.weight", (8192, 4096)), ("blk.4.attn_output.weight", (4096, 8192)),
                      ("blk.4.ffn_gate_shexp.weight", (2048, 4096)), ("blk.4.ssm_f_a.weight", (128, 4096)),
                      ("output.weight", (154880, 4096))):
        t = idx[name]
        ref = torch.from_numpy(gguf.quants.dequantize(np.array(t.data), t.tensor_type).reshape(shp).astype(np.float32)).to(dev)
        x = (torch.randn(shp[1], device=dev) * 0.05).half()
        r32 = ref @ x.float()
        f16 = ref.half()
        W = load_q8(t, *shp, dev)
        for BR, BK, NW in ((1, 1024, 4), (2, 512, 4), (2, 1024, 4), (4, 512, 4), (4, 1024, 8), (8, 512, 8), (8, 256, 4)):
            W.BR, W.BK, W.NW = BR, BK, NW
            y = (W @ x).float()
            def bench(fn, n=50):
                fn(); torch.cuda.synchronize(); t0 = time.time()
                for _ in range(n): fn()
                torch.cuda.synchronize(); return 1000 * (time.time() - t0) / n
            tq = bench(lambda: W @ x); th = bench(lambda: f16 @ x)
            print(f"{name:28s} BR={BR} BK={BK} NW={NW}: rel.err vs fp32 dequant {((y-r32).norm()/r32.norm()).item():.2e} "
                  f"(fp16 weights {(((f16@x).float()-r32).norm()/r32.norm()).item():.2e})  "
                  f"q8 {tq:.3f} ms ({W.nbytes()/tq/1e6:.0f} GB/s)  fp16 {th:.3f} ms ({f16.numel()*2/th/1e6:.0f} GB/s)")
