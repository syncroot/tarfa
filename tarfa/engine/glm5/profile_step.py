"""Per-op compute profile of one decode step (experts pre-cached, fetch excluded)."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_next as G
DEV = os.environ.get("GLM5_DEV", "cuda:1"); G.DEV = DEV; G.DT = torch.float16
torch.cuda.set_device(DEV)
from glm5_next import GLM5, hc_pre, hc_post, rmsnorm, kda_step, dense_ffn, moe_t, Ne, H, Dh, Di, HC
from glm5_decode import mla_decode_step
from experts_gpu import RawExperts

m = GLM5(sys.argv[1]); m.raw = RawExperts(m.idx, DEV)
DT = G.DT


def T(fn, n=30):
    fn(); torch.cuda.synchronize(DEV); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(DEV); return 1000 * (time.time() - t0) / n


rows = []
for L, label, count in ((0, "KDA+dense", 3), (4, "KDA+MoE", 31), (3, "MLA+MoE", 11)):
    w = m.layer(L)
    Hs = torch.randn(HC, Ne, device=DEV, dtype=DT) * 0.05
    x = torch.randn(Ne, device=DEV, dtype=DT) * 0.05
    t_hc = T(lambda: hc_pre(Hs, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"]))
    c, pa, ca = hc_pre(Hs, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
    t_post = T(lambda: hc_post(x, Hs, pa, ca))
    t_norm = T(lambda: rmsnorm(x, w["attn_norm"]))
    if w["mla"]:
        kv = [torch.randn(512, device=DEV, dtype=DT) for _ in range(16)]
        t_attn = T(lambda: (mla_decode_step(x, w, kv), kv.pop()))
    else:
        cs = {k: torch.zeros(3, Di, device=DEV, dtype=DT) for k in "qkv"}
        s = torch.zeros(H, Dh, Dh, device=DEV, dtype=DT)
        t_attn = T(lambda: kda_step(x, w, cs, s))
    if w["moe"]:
        moe_t(x, w, m)                                  # warm: pull this token's experts in
        t_ffn = T(lambda: moe_t(x, w, m))
    else:
        t_ffn = T(lambda: dense_ffn(x, w))
    per = 2 * t_hc + 2 * t_post + 2 * t_norm + t_attn + t_ffn
    rows.append((label, count, t_hc, t_post, t_norm, t_attn, t_ffn, per))
    del w; torch.cuda.empty_cache()

print(f"{'layer':10s} {'n':>3s} {'hc_pre':>7s} {'hc_post':>7s} {'norm':>6s} {'attn':>7s} {'ffn':>7s} {'=layer':>7s} {'x n':>8s}  (ms)")
tot = 0
for lab, n, a, b, c_, d, e, p in rows:
    tot += p * n
    print(f"{lab:10s} {n:3d} {a:7.3f} {b:7.3f} {c_:6.3f} {d:7.3f} {e:7.3f} {p:7.3f} {p*n:8.1f}")
print(f"estimated compute per token: {tot:.0f} ms  (hc_pre share: "
      f"{sum(2*r[2]*r[1] for r in rows)/tot:.0%})")
