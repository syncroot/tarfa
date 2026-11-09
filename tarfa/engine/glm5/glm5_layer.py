"""
glm5-next full layer-0 forward (dense FFN + mHC 4-stream residual + validated KDA),
numpy reference. Transcribed from glm5-next.cpp (build_hc_pre/sinkhorn/post, build_ffn).
Per single token (nt=1); streams stored as [hc=4, n_embd] (row = stream).
"""
import os, sys, glob
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kda_ref import kda_decode_step, rmsnorm, silu, sigmoid

H, Dh, Ne = 64, 128, 4096
Di = H * Dh
HC = 4
HC_DIM = HC * Ne          # 16384
HC_MIX = (2 + HC) * HC    # 24
NFF = 12288
NORM_EPS = 1e-5

# ggml name -> (attr, logical numpy shape).  conv handled specially.
WMAP = {
    "blk.0.attn_q.weight": ("wq", (Di, Ne)), "blk.0.attn_k.weight": ("wk", (Di, Ne)),
    "blk.0.attn_v.weight": ("wv", (Di, Ne)),
    "blk.0.ssm_conv1d_q.weight": ("q_conv", None), "blk.0.ssm_conv1d_k.weight": ("k_conv", None),
    "blk.0.ssm_conv1d_v.weight": ("v_conv", None),
    "blk.0.ssm_f_a.weight": ("ssm_f_a", (Dh, Ne)), "blk.0.ssm_f_b.weight": ("ssm_f_b", (Di, Dh)),
    "blk.0.ssm_dt.bias": ("ssm_dt_b", (Di,)), "blk.0.ssm_a": ("ssm_a", (H,)),
    "blk.0.ssm_beta.weight": ("ssm_beta", (H, Ne)),
    "blk.0.ssm_g_a.weight": ("ssm_g_a", (Dh, Ne)), "blk.0.ssm_g_b.weight": ("ssm_g_b", (Di, Dh)),
    "blk.0.ssm_norm.weight": ("ssm_o_norm", (Dh,)), "blk.0.attn_output.weight": ("wo", (Ne, Di)),
    # mHC
    "blk.0.hc_attn_fn.weight": ("hc_attn_fn", (HC_MIX, HC_DIM)),
    "blk.0.hc_attn_scale.weight": ("hc_attn_scale", (3,)),
    "blk.0.hc_attn_base.weight": ("hc_attn_base", (HC_MIX,)),
    "blk.0.hc_ffn_fn.weight": ("hc_ffn_fn", (HC_MIX, HC_DIM)),
    "blk.0.hc_ffn_scale.weight": ("hc_ffn_scale", (3,)),
    "blk.0.hc_ffn_base.weight": ("hc_ffn_base", (HC_MIX,)),
    # norms + dense FFN
    "blk.0.attn_norm.weight": ("attn_norm", (Ne,)), "blk.0.ffn_norm.weight": ("ffn_norm", (Ne,)),
    "blk.0.ffn_up.weight": ("ffn_up", (NFF, Ne)), "blk.0.ffn_gate.weight": ("ffn_gate", (NFF, Ne)),
    "blk.0.ffn_down.weight": ("ffn_down", (Ne, NFF)),
}


class W:  # weight bag
    pass


def deq_flat(t):
    if t.tensor_type == GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32).reshape(-1)
    return gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float32).reshape(-1)


def load(gguf_dir):
    w = W()
    need = set(WMAP)
    for sh in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
        r = GGUFReader(sh)
        for t in r.tensors:
            if t.name not in need:
                continue
            attr, shp = WMAP[t.name]
            flat = deq_flat(t)
            if shp is None:                      # conv: (d_inner, d_conv) -> (d_conv, d_inner)
                a = flat.reshape(Di, -1).T
            else:
                a = flat.reshape(shp)
            setattr(w, attr, a)
            need.discard(t.name)
    assert not need, f"missing: {need}"
    return w


def rms_noweight(x, eps=NORM_EPS):
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)


def hc_sinkhorn(comb, eps, iters=20):
    # comb[dst, src]; softmax over dst, then column/row Sinkhorn normalization.
    m = comb - comb.max(axis=0, keepdims=True)
    comb = np.exp(m); comb /= comb.sum(axis=0, keepdims=True)
    comb = comb + eps

    def norm_cols():  # normalize over src (axis 1)
        return comb / (comb.sum(axis=1, keepdims=True) + eps)

    def norm_rows():  # normalize over dst (axis 0)
        return comb / (comb.sum(axis=0, keepdims=True) + eps)

    comb = norm_cols()
    for _ in range(1, iters):
        comb = norm_rows()
        comb = norm_cols()
    return comb


def hc_pre(x, hc_fn, hc_scale, hc_base, eps):
    # x: [hc=4, n_embd] streams. Returns (collapsed[n_embd], post[4], comb[4,4]).
    flat = x.reshape(-1)                          # [hc_dim], stream-major (matches ggml ne0=n_embd fastest)
    flat_n = rms_noweight(flat, eps).astype(np.float32)
    mixes = hc_fn @ flat_n                         # [24]
    pre = mixes[0:HC] * hc_scale[0] + hc_base[0:HC]
    pre = sigmoid(pre) + eps                        # scale_bias(1.0, eps)
    post = mixes[HC:2 * HC] * hc_scale[1] + hc_base[HC:2 * HC]
    post = sigmoid(post) * 2.0
    comb16 = mixes[2 * HC:2 * HC + HC * HC] * hc_scale[2] + hc_base[2 * HC:2 * HC + HC * HC]
    comb = comb16.reshape(HC, HC).T                 # -> comb[dst, src]
    comb = hc_sinkhorn(comb, eps)
    collapsed = (pre[:, None] * x).sum(axis=0)      # Σ_s pre[s]*x[s]  -> [n_embd]
    return collapsed.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def hc_post(block_out, residual, post, comb):
    # residual: [hc, n_embd]; returns [hc, n_embd]
    return (post[:, None] * block_out[None, :] + comb @ residual).astype(np.float32)


def dense_ffn(x, w, limit=10.0):
    # clamped SwiGLU: gate<=limit, up in [-limit,limit] (swiglu_clamp_shexp)
    g = np.minimum(w.ffn_gate @ x, limit)
    u = np.clip(w.ffn_up @ x, -limit, limit)
    return (w.ffn_down @ (silu(g) * u)).astype(np.float32)


def layer0_forward(hc_init, w):
    """hc_init: [hc=4, n_embd] (embedding streams). Returns dict of intermediates + l_out [hc,n_embd]."""
    out = {}
    # attention sub-block
    residual = hc_init
    collapsed, post_a, comb_a = hc_pre(hc_init, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, NORM_EPS)
    an = rmsnorm(collapsed, w.attn_norm, NORM_EPS)
    out["attn_norm"] = an
    cs = {c: np.zeros((3, Di), np.float32) for c in "qkv"}
    kda, _, _ = kda_decode_step(an.astype(np.float32), w, cs, np.zeros((H, Dh, Dh), np.float32))
    out["kda_out"] = kda
    inpL = hc_post(kda, residual, post_a, comb_a)
    out["hc_attn_post"] = inpL
    # ffn sub-block
    residual = inpL
    collapsed, post_f, comb_f = hc_pre(inpL, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, NORM_EPS)
    fn = rmsnorm(collapsed, w.ffn_norm, NORM_EPS)
    out["ffn_norm"] = fn
    ffn = dense_ffn(fn.astype(np.float32), w)
    out["ffn_out"] = ffn
    inpL = hc_post(ffn, residual, post_f, comb_f)
    out["l_out"] = inpL
    return out
