"""
Tarfa exact engine for GLM-5.3-Flash (glm5-next) — torch/CUDA, Milestone 1.
The numpy reference kernels (kda_ref.py, mla_ref.py, moe_ref.py) are the exact-math spec.
M1 scope: GGUF->torch loader + a torch forward on the 4060 Ti, verified layer-by-layer
against the llama.cpp oracle taps. Experts dequantized on-demand (the fast path is
glm5_gdecode.py: CPU expert kernel, GPU expert residency, CUDA graphs).

Self-test (python glm5_next.py <gguf_dir> <dumps2_dir>) runs layer 0 on GPU and compares
l_out-0 to the oracle.
"""
import os, sys, glob
import numpy as np
import torch
import gguf
from gguf import GGUFReader, GGMLQuantizationType

DEV = os.environ.get("GLM5_DEV", "cuda:0")
DT = torch.float32
Ne, H, Dh, Di = 4096, 64, 128, 8192
HC, HC_DIM, HC_MIX, NFF = 4, 4 * 4096, 24, 12288
NORM_EPS, GATE_LB, L2_EPS, SWIGLU_LIMIT = 1e-5, -5.0, 1e-6, 10.0


def _deq_np(t):
    if t.tensor_type == GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32).reshape(-1)
    return gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float32).reshape(-1)


def _t(np_flat, shape):
    return torch.from_numpy(np_flat.reshape(shape).copy()).to(DEV, DT)


# ---- torch kernels (translations of the validated numpy references) ----
def rmsnorm(x, w, eps=NORM_EPS):          # x [...,D], w [D]
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def rms_noweight(x, eps=NORM_EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def silu(x):
    return torch.nn.functional.silu(x)

def l2n(x, eps=L2_EPS):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def hc_sinkhorn(comb, eps, iters=20):     # comb [4,4] dst,src
    comb = torch.softmax(comb, dim=0) + eps
    comb = comb / (comb.sum(1, keepdim=True) + eps)
    for _ in range(1, iters):
        comb = comb / (comb.sum(0, keepdim=True) + eps)
        comb = comb / (comb.sum(1, keepdim=True) + eps)
    return comb

_HC_FAST = os.environ.get("GLM5_HC_FAST", "1") == "1"


def hc_pre(x, fn, scale, base, eps=NORM_EPS):   # x [4, Ne] -> (collapsed [Ne], post [4], comb [4,4])
    if _HC_FAST:                                  # fused gates + Sinkhorn (hc_fast.py), ~5x faster
        from hc_fast import hc_pre_fast
        return hc_pre_fast(x, fn, scale, base, eps=eps, norm_eps=eps)
    flat = rms_noweight(x.reshape(-1), eps)
    mix = fn @ flat
    pre = torch.sigmoid(mix[0:4] * scale[0] + base[0:4]) + eps
    post = torch.sigmoid(mix[4:8] * scale[1] + base[4:8]) * 2.0
    comb = (mix[8:24] * scale[2] + base[8:24]).reshape(4, 4).T
    comb = hc_sinkhorn(comb, eps)
    collapsed = (pre[:, None] * x).sum(0)
    return collapsed, post, comb

def hc_post(block_out, residual, post, comb):   # residual [4,Ne] -> [4,Ne]
    return post[:, None] * block_out[None, :] + comb @ residual

def swiglu_clamp(g, u, limit=SWIGLU_LIMIT):
    return silu(torch.clamp(g, max=limit)) * torch.clamp(u, -limit, limit)


def kda_step(x, L, conv_state, s):   # x [Ne]; returns out [Ne], new conv_state, new s
    def conv(proj, cw, cs):
        xp = proj @ x
        win = torch.cat([cs, xp[None, :]], 0)      # [d_conv, Di]
        y = (win * cw).sum(0)
        return silu(y).reshape(H, Dh), win[1:, :].clone()
    q, csq = conv(L["wq"], L["q_conv"], conv_state["q"])
    k, csk = conv(L["wk"], L["k_conv"], conv_state["k"])
    v, csv = conv(L["wv"], L["v_conv"], conv_state["v"])
    sd = s.dtype                     # recurrence runs in the state's dtype (fp32 state = llama.cpp-like)
    g1 = (L["ssm_f_b"] @ (L["ssm_f_a"] @ x) + L["ssm_dt_b"]).reshape(H, Dh).to(sd)
    g1 = torch.sigmoid(-(g1 * L["ssm_a"].reshape(H, 1).to(sd))) * GATE_LB
    beta = torch.sigmoid((L["ssm_beta"] @ x).to(sd)).reshape(H, 1)
    q, k, v = q.to(sd), k.to(sd), v.to(sd)
    qn = l2n(q) / (Dh ** 0.5); kn = l2n(k)
    g = torch.exp(g1)
    s = s * g[:, :, None]            # KDA decay on the KEY axis (see kda_ref.py; validated all positions)
    sk = torch.einsum("hab,ha->hb", s, kn)
    d = (v - sk) * beta
    s = s + kn[:, :, None] * d[:, None, :]
    o = torch.einsum("hab,ha->hb", s, qn)
    g2 = (L["ssm_g_b"] @ (L["ssm_g_a"] @ x)).reshape(H, Dh).to(sd)
    on = rmsnorm(o, L["ssm_o_norm"].to(sd), NORM_EPS)
    out = L["wo"] @ (on * torch.sigmoid(g2)).reshape(Di).to(L["wo"].dtype)
    return out, {"q": csq, "k": csk, "v": csv}, s


def dense_ffn(x, L):
    return L["ffn_down"] @ swiglu_clamp(L["ffn_gate"] @ x, L["ffn_up"] @ x)


Q_LORA, KV_LORA, HEAD_K = 1536, 512, 256
KQ_SCALE = 1.0 / (HEAD_K ** 0.5)


def mla_forward_t(x, L):   # x [Ne, T] -> attn_out [Ne, T]  (nope-only causal MLA)
    T = x.shape[1]
    qr = rmsnorm((L["wq_a"] @ x).T, L["q_a_norm"])                 # [T, 1536]
    kv = rmsnorm((L["wkv_a_mqa"] @ x).T, L["kv_a_norm"])           # [T, 512]
    q = (L["wq_b"] @ qr.T).T.reshape(T, H, HEAD_K)                 # [T, H, 256]
    qa = torch.einsum("thd,hkd->thk", q, L["wk_b"])               # [T, H, 512]
    out = torch.empty(Ne, T, device=DEV, dtype=DT)
    for t in range(T):
        sc = torch.einsum("hk,sk->hs", qa[t], kv[:t + 1]) * KQ_SCALE   # [H, t+1]
        w = torch.softmax(sc, dim=1)
        ctx = w @ kv[:t + 1]                                       # [H, 512]
        o = torch.einsum("hvk,hk->hv", L["wv_b"], ctx)            # [H, 256]
        out[:, t] = L["wo"] @ o.reshape(H * HEAD_K)
    return out


def moe_t(x, L, model):    # x [Ne] -> [Ne]
    raw = getattr(model, "raw", None)
    if raw is not None:                                   # fused raw-k-quant path (fp32)
        x32 = x.float().contiguous()
        probs = torch.sigmoid(L["router"].float() @ x32)
        sel = torch.argsort(probs + L["exp_bias"].float())[-8:]
        w = probs[sel]; w = w / w.sum() * 2.5
        y = raw.moe(L["L"], x32, sel.tolist(), w)
        gs = L["sh_gate"] @ x; us = L["sh_up"] @ x
        return (y + (L["sh_down"] @ swiglu_clamp(gs, us)).float()).to(x.dtype)
    probs = torch.sigmoid(L["router"] @ x)
    sel = torch.argsort(probs + L["exp_bias"])[-8:]
    w = probs[sel]; w = w / w.sum() * 2.5
    out = torch.zeros(Ne, device=DEV, dtype=DT)
    for e, we in zip(sel.tolist(), w):
        g = model.expert(L, "ffn_gate_exps.weight", e) @ x
        u = model.expert(L, "ffn_up_exps.weight", e) @ x
        d = model.expert(L, "ffn_down_exps.weight", e) @ swiglu_clamp(g, u)
        out = out + we * d
    gs = L["sh_gate"] @ x; us = L["sh_up"] @ x
    return out + L["sh_down"] @ swiglu_clamp(gs, us)


# GLM5_Q8=1: the large trunk matrices stay Q8_0 on the GPU (q8.Q8W, `W @ x` for 1-D x only, so
# decode paths only); GLM5_ROUTER_FP32=1: the router keeps its GGUF F32 weights.
Q8_NAMES = set() if os.environ.get("GLM5_Q8", "0") != "1" else {
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_a.weight",
    "attn_q_b.weight", "attn_kv_a_mqa.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}
FP32_NAMES = {"ffn_gate_inp.weight"} if os.environ.get("GLM5_ROUTER_FP32", "0") == "1" else set()


class GLM5:
    def __init__(self, gguf_dir):
        self.idx = {}; self._r = []
        for sh in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
            r = GGUFReader(sh); self._r.append(r)
            for t in r.tensors:
                self.idx[t.name] = t

    def is_mla(self, L): return f"blk.{L}.attn_kv_a_mqa.weight" in self.idx

    def layer(self, L):
        b = f"blk.{L}."
        def g(n, s):
            t = self.idx[b + n]
            if n in Q8_NAMES and len(s) == 2 and t.tensor_type == GGMLQuantizationType.Q8_0:
                from q8 import load_q8                       # stays quantized (q8.py)
                return load_q8(t, *s, DEV, DT)
            if n in FP32_NAMES:
                return torch.from_numpy(_deq_np(t).reshape(s).copy()).to(DEV, torch.float32)
            return _t(_deq_np(t), s)
        d = {"attn_norm": g("attn_norm.weight", (Ne,)), "ffn_norm": g("ffn_norm.weight", (Ne,)),
             "hc_attn_fn": g("hc_attn_fn.weight", (24, HC_DIM)), "hc_attn_scale": g("hc_attn_scale.weight", (3,)),
             "hc_attn_base": g("hc_attn_base.weight", (24,)), "hc_ffn_fn": g("hc_ffn_fn.weight", (24, HC_DIM)),
             "hc_ffn_scale": g("hc_ffn_scale.weight", (3,)), "hc_ffn_base": g("hc_ffn_base.weight", (24,)),
             "L": L}
        if not self.is_mla(L):
            d["mla"] = False
            for n, s in (("attn_q", (Di, Ne)), ("attn_k", (Di, Ne)), ("attn_v", (Di, Ne)),
                         ("ssm_f_a", (Dh, Ne)), ("ssm_f_b", (Di, Dh)), ("ssm_g_a", (Dh, Ne)),
                         ("ssm_g_b", (Di, Dh)), ("attn_output", (Ne, Di)), ("ssm_beta", (H, Ne))):
                d[n.replace("attn_q", "wq").replace("attn_k", "wk").replace("attn_v", "wv").replace("attn_output", "wo")] = g(n + ".weight", s)
            for n in ("q", "k", "v"):
                d[n + "_conv"] = _t(_deq_np(self.idx[b + f"ssm_conv1d_{n}.weight"]).reshape(Di, -1).T.copy().reshape(-1), (4, Di))
            d["ssm_dt_b"] = g("ssm_dt.bias", (Di,)); d["ssm_a"] = g("ssm_a", (H,)); d["ssm_o_norm"] = g("ssm_norm.weight", (Dh,))
        else:
            d["mla"] = True
            d["wq_a"] = g("attn_q_a.weight", (Q_LORA, Ne)); d["q_a_norm"] = g("attn_q_a_norm.weight", (Q_LORA,))
            d["wq_b"] = g("attn_q_b.weight", (H * HEAD_K, Q_LORA))
            d["wkv_a_mqa"] = g("attn_kv_a_mqa.weight", (KV_LORA, Ne)); d["kv_a_norm"] = g("attn_kv_a_norm.weight", (KV_LORA,))
            d["wk_b"] = g("attn_k_b.weight", (H, KV_LORA, HEAD_K)); d["wv_b"] = g("attn_v_b.weight", (H, HEAD_K, KV_LORA))
            d["wo"] = g("attn_output.weight", (Ne, H * HEAD_K))
        if L < 3:
            d["moe"] = False
            d["ffn_gate"] = g("ffn_gate.weight", (NFF, Ne)); d["ffn_up"] = g("ffn_up.weight", (NFF, Ne)); d["ffn_down"] = g("ffn_down.weight", (Ne, NFF))
        else:
            d["moe"] = True
            d["router"] = g("ffn_gate_inp.weight", (288, Ne)); d["exp_bias"] = g("exp_probs_b.bias", (288,))
            d["sh_gate"] = g("ffn_gate_shexp.weight", (NFF // 6, Ne)); d["sh_up"] = g("ffn_up_shexp.weight", (NFF // 6, Ne))
            d["sh_down"] = g("ffn_down_shexp.weight", (Ne, NFF // 6))
            for h in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
                d[h] = self.idx[b + h]        # ReaderTensor handle, dequant per selected expert
        return d

    def expert(self, L, key, e):
        """Routed expert matrix on DEV. GPU k-quant dequant (bit-exact) + LRU cache of
        dequantized matrices (GLM5_EXP_CACHE = capacity in matrices; 0 disables)."""
        from collections import OrderedDict
        from gpu_dequant import dequant_rows, BLOCK_BYTES
        if not hasattr(self, "cache"):
            self.cache = OrderedDict(); self.hits = 0; self.misses = 0
            self.cap = int(os.environ.get("GLM5_EXP_CACHE", "300"))
        t = L[key]
        k = (t.name, e)
        if k in self.cache:
            self.cache.move_to_end(k); self.hits += 1
            return self.cache[k]
        self.misses += 1
        qt = gguf.GGMLQuantizationType(t.tensor_type)
        raw = np.array(t.data[e])                                   # [rows, bytes_per_row]
        if qt in BLOCK_BYTES:
            w = dequant_rows(raw, qt, raw.shape[0], raw.shape[1] // BLOCK_BYTES[qt] * 256, DEV, DT)
        else:                                                       # fallback (rare types)
            w = torch.from_numpy(gguf.quants.dequantize(raw, qt).astype(np.float32)).to(DEV, DT)
        if self.cap > 0:
            self.cache[k] = w
            if len(self.cache) > self.cap:
                self.cache.popitem(last=False)
        return w


def selftest(gguf_dir, sprint):
    from probe_layer0 import read_dump, cos as npcos
    m = GLM5(gguf_dir)
    L0 = m.layer(0)
    raw, _ = read_dump(os.path.join(sprint, "dumps2"), "hc_init")
    T = raw.size // (HC * Ne)
    hc_init = torch.from_numpy(raw[:T * HC * Ne].reshape(T, HC, Ne).copy()).to(DEV, DT)
    t = 0
    Hs = hc_init[t]                               # [4, Ne]
    residual = Hs
    c, pa, ca = hc_pre(Hs, L0["hc_attn_fn"], L0["hc_attn_scale"], L0["hc_attn_base"])
    an = rmsnorm(c, L0["attn_norm"])
    cs = {k: torch.zeros(3, Di, device=DEV, dtype=DT) for k in "qkv"}
    s = torch.zeros(H, Dh, Dh, device=DEV, dtype=DT)
    kda, cs, s = kda_step(an, L0, cs, s)
    Hs = hc_post(kda, residual, pa, ca)
    residual = Hs
    c, pf, cf = hc_pre(Hs, L0["hc_ffn_fn"], L0["hc_ffn_scale"], L0["hc_ffn_base"])
    fn = rmsnorm(c, L0["ffn_norm"])
    ff = dense_ffn(fn, L0)
    Hs = hc_post(ff, residual, pf, cf)            # [4, Ne] = l_out token 0
    ref, _ = read_dump(os.path.join(sprint, "dumps2"), "l_out-0")
    mine = Hs.reshape(-1).float().cpu().numpy(); r = ref[:HC * Ne]
    print(f"[glm5_next torch, {DEV}] layer-0  l_out   cos={npcos(mine, r):.6f}  maxdiff={np.max(np.abs(mine-r)):.3e}")

    # layer-3 MLA (dumps3): feed attn_norm-3 (all tokens) -> attn_out-3
    L3 = m.layer(3)
    an3, _ = read_dump(os.path.join(sprint, "dumps3"), "attn_norm-3"); T3 = an3.size // Ne
    x3 = torch.from_numpy(an3[:T3 * Ne].reshape(T3, Ne).T.copy()).to(DEV, DT)
    ao = mla_forward_t(x3, L3).T.reshape(-1).float().cpu().numpy()   # [Ne,T].T -> oracle column-major layout
    aor, _ = read_dump(os.path.join(sprint, "dumps3"), "attn_out-3")
    print(f"[glm5_next torch, {DEV}] layer-3  attn_out cos={npcos(ao, aor[:ao.size]):.6f}")

    # layer-3 MoE (dumps4): feed ffn_norm-3 token 0 -> ffn_out-3
    fn3, _ = read_dump(os.path.join(sprint, "dumps4"), "ffn_norm-3")
    xf = torch.from_numpy(fn3[:Ne].copy()).to(DEV, DT)
    mo = moe_t(xf, L3, m).float().cpu().numpy()
    mor, _ = read_dump(os.path.join(sprint, "dumps4"), "ffn_out-3")
    print(f"[glm5_next torch, {DEV}] layer-3  ffn_out  cos={npcos(mo, mor[:Ne]):.6f}")


if __name__ == "__main__":
    selftest(sys.argv[1], sys.argv[2])
