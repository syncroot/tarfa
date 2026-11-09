"""
glm5-next KDA (Kimi gated-delta) block — numpy reference, decode path.

Day-1 Tarfa port. Transcribed from llama.cpp Oct-3 GLM build:
  src/models/glm5-next.cpp  : build_kda_layer, glm5_conv1d
  src/models/delta-net-base.cpp : build_delta_net_autoregressive

STATUS: UNVALIDATED until probed against the llama.cpp activation oracle
(cb tap `kda_out` / `l_out` at layer 0). The axis/broadcast of the gate in the
delta recurrence is the most error-prone spot; the oracle probe is what settles it.

Shapes (GLM-5.3-Flash): n_embd=4096, n_head H=64, head_dim Dh=128,
d_inner=H*Dh=8192, d_conv=4, kda_gate_lower_bound=-5.0.
All matmuls use the ggml convention weight[out,in] @ x[in] -> [out], i.e. in numpy
with W stored row-major as [out,in], compute W @ x.
"""
import numpy as np

GATE_LB = -5.0
L2_EPS = 1e-6
NORM_EPS = 1e-5  # glm5-next.attention.layer_norm_rms_epsilon (ssm output RMSNorm)


def rmsnorm(x, w, eps):
    # x: [..., D] ; w: [D]
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(x.dtype) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def l2norm_lastdim(x, eps=L2_EPS):
    n = np.sqrt(np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    return (x / n).astype(x.dtype)


class KDAWeights:
    """Per-layer KDA tensors, already dequantized to f32 and in ggml [out,in] order."""
    # projections: wq/wk/wv  [d_inner, n_embd]
    # conv: q_conv/k_conv/v_conv  [d_conv, d_inner]   (depthwise taps per channel)
    # ssm_f_a [head_dim, n_embd], ssm_f_b [d_inner, head_dim], ssm_dt_b [d_inner]
    # ssm_a [n_head]  (== -exp(A_log))
    # ssm_beta [n_head, n_embd]
    # ssm_g_a [head_dim, n_embd], ssm_g_b [d_inner, head_dim]
    # ssm_o_norm [head_dim]
    # wo [n_embd, d_inner]
    pass


def conv_branch(x, proj_w, conv_w, conv_state, H, Dh):
    """proj -> causal depthwise conv1d(d_conv) -> SiLU -> [H, Dh]. Returns (y, new_state)."""
    d_conv = conv_w.shape[0]
    xp = proj_w @ x                                   # [d_inner]
    win = np.concatenate([conv_state, xp[None, :]], 0)  # [d_conv, d_inner]
    y = np.sum(win * conv_w, axis=0)                   # depthwise: sum taps per channel
    new_state = win[1:, :].copy()                      # keep last d_conv-1 inputs
    y = silu(y)
    return y.reshape(H, Dh), new_state                 # ggml fastest dim = head_dim


def kda_decode_step(x, W, conv_state, s, H=64, Dh=128):
    """
    One decode token. x: [n_embd] post attn_norm.
    conv_state: dict q/k/v each [d_conv-1, d_inner]. s: recurrent state [H, Dh, Dh].
    Returns (out[n_embd], new_conv_state, new_s).
    """
    q, csq = conv_branch(x, W.wq, W.q_conv, conv_state["q"], H, Dh)  # [H,Dh]
    k, csk = conv_branch(x, W.wk, W.k_conv, conv_state["k"], H, Dh)
    v, csv = conv_branch(x, W.wv, W.v_conv, conv_state["v"], H, Dh)

    # decay gate g1 -> log-decay per (head, head_dim)
    f_a = W.ssm_f_a @ x                       # [head_dim]
    g1 = W.ssm_f_b @ f_a + W.ssm_dt_b         # [d_inner]
    g1 = g1.reshape(H, Dh)
    A = W.ssm_a.reshape(H, 1)                 # -exp(A_log)
    g1 = sigmoid(-(g1 * A)) * GATE_LB         # [H, Dh]  (gate_lower_bound branch)

    beta = sigmoid(W.ssm_beta @ x).reshape(H, 1)  # [H,1]

    # FLA l2 norm on q,k over head_dim, then scale q
    q = l2norm_lastdim(q) / np.sqrt(Dh)
    k = l2norm_lastdim(k)

    # gated delta rule (decode), per head. state s[h, a, b]: a = ne0 (key dim), b = ne1 (value dim).
    # Key and query contract the a axis; the output reads out the b axis. The KDA per-channel
    # decay must act on the KEY axis (a), as in build_delta_net_chunking (g_last_exp_t is
    # [S_k,1]). llama.cpp's non-fused build_delta_net_autoregressive applies it along ne1 (value
    # axis): harmless for scalar GDN gates, wrong for KDA; invisible at position 0 (state = 0).
    # Probe6b vs the prefill oracle: value-axis pos2 out cos 0.902; key-axis >= 0.99996 at all pos.
    g = np.exp(g1)                                 # [H, Dh]
    s = s * g[:, :, None]                          # s[h,a,b] *= g[h,a]   (decay the key axis)
    sk = np.einsum("hab,ha->hb", s, k)             # sk[h,b] = sum_a s[h,a,b]*k[h,a]
    d = (v - sk) * beta                            # [H, Dh]  (v,sk indexed alike)
    s = s + k[:, :, None] * d[:, None, :]          # s[h,a,b] += k[h,a]*d[h,b]
    o = np.einsum("hab,ha->hb", s, q)              # o[h,b] = sum_a s[h,a,b]*q[h,a]

    # output gate + per-head RMSNorm
    g_a = W.ssm_g_a @ x                        # [head_dim]
    g2 = (W.ssm_g_b @ g_a).reshape(H, Dh)      # [H, Dh]
    o = rmsnorm(o, W.ssm_o_norm, NORM_EPS)     # RMSNorm over head_dim (eps 1e-5)
    gated = o * sigmoid(g2)                     # [H, Dh]

    out = W.wo @ gated.reshape(H * Dh)          # [n_embd]
    return out, {"q": csq, "k": csk, "v": csv}, s


if __name__ == "__main__":
    # smoke test with random weights: checks shapes + finiteness only (not correctness).
    rng = np.random.default_rng(0)
    H, Dh, Ne = 64, 128, 4096
    Di = H * Dh
    W = KDAWeights()
    W.wq = rng.standard_normal((Di, Ne), np.float32) * 0.02
    W.wk = rng.standard_normal((Di, Ne), np.float32) * 0.02
    W.wv = rng.standard_normal((Di, Ne), np.float32) * 0.02
    W.q_conv = rng.standard_normal((4, Di), np.float32) * 0.1
    W.k_conv = rng.standard_normal((4, Di), np.float32) * 0.1
    W.v_conv = rng.standard_normal((4, Di), np.float32) * 0.1
    W.ssm_f_a = rng.standard_normal((Dh, Ne), np.float32) * 0.02
    W.ssm_f_b = rng.standard_normal((Di, Dh), np.float32) * 0.02
    W.ssm_dt_b = rng.standard_normal((Di,), np.float32) * 0.1
    W.ssm_a = -np.exp(rng.standard_normal((H,), np.float32) * 0.5)
    W.ssm_beta = rng.standard_normal((H, Ne), np.float32) * 0.02
    W.ssm_g_a = rng.standard_normal((Dh, Ne), np.float32) * 0.02
    W.ssm_g_b = rng.standard_normal((Di, Dh), np.float32) * 0.02
    W.ssm_o_norm = np.ones((Dh,), np.float32)
    W.wo = rng.standard_normal((Ne, Di), np.float32) * 0.02
    cs = {k: np.zeros((3, Di), np.float32) for k in "qkv"}
    s = np.zeros((H, Dh, Dh), np.float32)
    x = rng.standard_normal((Ne,), np.float32)
    out, cs, s = kda_decode_step(x, W, cs, s)
    print("out", out.shape, "finite", np.isfinite(out).all(), "state", s.shape)
