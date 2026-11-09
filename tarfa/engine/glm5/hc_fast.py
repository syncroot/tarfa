"""
Fast mHC hc_pre: ~5 launches instead of ~130. The 4-stream mix gates and the 20-iteration
Sinkhorn on the 4x4 combine matrix run in ONE single-program Triton kernel (registers).
Math identical to glm5_next.hc_pre / hc_sinkhorn (validated vs the llama.cpp oracle).
"""
import torch, triton, triton.language as tl


@triton.jit
def _hc_gates(MIX, SCALE, BASE, PRE, POST, COMB, eps, ITERS: tl.constexpr):
    i4 = tl.arange(0, 4)
    s0 = tl.load(SCALE + 0).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    pre = tl.load(MIX + i4) * s0 + tl.load(BASE + i4).to(tl.float32)
    pre = 1.0 / (1.0 + tl.exp(-pre)) + eps
    post = tl.load(MIX + 4 + i4) * s1 + tl.load(BASE + 4 + i4).to(tl.float32)
    post = 2.0 / (1.0 + tl.exp(-post))
    # comb[dst=i, src=j] = comb16[j*4 + i]
    off = 8 + i4[None, :] * 4 + i4[:, None]
    c = tl.load(MIX + off) * s2 + tl.load(BASE + off).to(tl.float32)
    m = tl.max(c, axis=0)
    e = tl.exp(c - m[None, :])
    c = e / tl.sum(e, axis=0)[None, :] + eps                    # softmax over dst, + eps
    c = c / (tl.sum(c, axis=1)[:, None] + eps)                  # col-norm (over src)
    for _ in range(ITERS - 1):        # runtime loop: a 19x static unroll stalls the compiler
        c = c / (tl.sum(c, axis=0)[None, :] + eps)              # row-norm (over dst)
        c = c / (tl.sum(c, axis=1)[:, None] + eps)              # col-norm (over src)
    tl.store(PRE + i4, pre)
    tl.store(POST + i4, post)
    tl.store(COMB + i4[:, None] * 4 + i4[None, :], c)


@torch.library.custom_op("glm5::hc_gates", mutates_args=())
def hc_gates(mix: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, eps: float, iters: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.empty(4, device=mix.device, dtype=torch.float32)
    post = torch.empty(4, device=mix.device, dtype=torch.float32)
    comb = torch.empty(4, 4, device=mix.device, dtype=torch.float32)
    with torch.cuda.device(mix.device):
        _hc_gates[(1,)](mix, scale, base, pre, post, comb, eps, ITERS=iters)
    return pre, post, comb


@hc_gates.register_fake
def _(mix, scale, base, eps, iters):
    return mix.new_empty(4), mix.new_empty(4), mix.new_empty(4, 4)


def hc_pre_fast(x, fn, scale, base, eps=1e-5, norm_eps=1e-5, iters=20):
    """x [4, Ne] -> (collapsed [Ne], post [4], comb [4,4]) in x.dtype."""
    flat = x.reshape(-1).float()
    rms = torch.rsqrt(flat.pow(2).mean() + norm_eps)
    mix = (fn.float() @ flat) * rms if fn.dtype == torch.float32 else (fn @ x.reshape(-1)).float() * rms
    pre, post, comb = torch.ops.glm5.hc_gates(mix.contiguous(), scale, base, eps, iters)
    collapsed = (pre.to(x.dtype) @ x)                           # sum_s pre[s] * x[s]
    return collapsed, post.to(x.dtype), comb.to(x.dtype)


if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import glm5_next as G
    dev = os.environ.get("GLM5_DEV", "cuda:1"); torch.cuda.set_device(dev)
    for dt in (torch.float32, torch.float16):
        G.DEV, G.DT = dev, dt
        m = G.GLM5(sys.argv[1]); w = m.layer(4)
        Hs = (torch.randn(4, 4096, device=dev) * 0.05).to(dt)
        r = G.hc_pre(Hs, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
        f = hc_pre_fast(Hs, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
        for name, a, b in zip(("collapsed", "post", "comb"), r, f):
            a, b = a.float(), b.float()
            print(f"{str(dt):14s} {name:9s} maxdiff {(a-b).abs().max().item():.2e}  rel {((a-b).norm()/a.norm()).item():.2e}")
        def bench(fn):
            fn(); torch.cuda.synchronize(dev); t0 = time.time()
            for _ in range(50): fn()
            torch.cuda.synchronize(dev); return 1000 * (time.time() - t0) / 50
        print(f"{str(dt):14s} hc_pre torch {bench(lambda: G.hc_pre(Hs, w['hc_attn_fn'], w['hc_attn_scale'], w['hc_attn_base'])):.3f} ms"
              f"  ->  fast {bench(lambda: hc_pre_fast(Hs, w['hc_attn_fn'], w['hc_attn_scale'], w['hc_attn_base'])):.3f} ms")
