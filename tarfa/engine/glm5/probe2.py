"""Stepwise KDA diagnostic: compare every intermediate to its oracle tap (token 0)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import load_weights, read_dump, cos, H, Dh, Ne, Di
from kda_ref import conv_branch, rmsnorm, sigmoid, l2norm_lastdim, GATE_LB


def cmp(name, mine, dd, oname, n):
    ref, _ = read_dump(dd, oname)
    ref = ref[:n]
    m = mine.ravel()[:n]
    print(f"{name:10s} cos={cos(m, ref):.6f}  maxdiff={np.max(np.abs(m-ref)):.3e}  "
          f"rms m={np.sqrt(np.mean(m**2)):.3e} r={np.sqrt(np.mean(ref**2)):.3e}")
    return ref


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    W = load_weights(gguf_dir)
    x, _ = read_dump(dd, "attn_norm-0"); x = x[:Ne]
    cs = {c: np.zeros((3, Di), np.float32) for c in "qkv"}

    q, _ = conv_branch(x, W.wq, W.q_conv, cs["q"], H, Dh)
    k, _ = conv_branch(x, W.wk, W.k_conv, cs["k"], H, Dh)
    v, _ = conv_branch(x, W.wv, W.v_conv, cs["v"], H, Dh)
    cmp("q_conv", q, dd, "kda_q_conv-0", Di)
    cmp("k_conv", k, dd, "kda_k_conv-0", Di)
    cmp("v_conv", v, dd, "kda_v_conv-0", Di)

    f_a = W.ssm_f_a @ x
    g1 = (W.ssm_f_b @ f_a + W.ssm_dt_b).reshape(H, Dh)
    A = W.ssm_a.reshape(H, 1)
    g1 = sigmoid(-(g1 * A)) * GATE_LB
    cmp("g1", g1, dd, "kda_g1-0", Di)

    beta = sigmoid(W.ssm_beta @ x).reshape(H, 1)
    cmp("beta", beta, dd, "kda_beta-0", H)

    qn = l2norm_lastdim(q) / np.sqrt(Dh)
    kn = l2norm_lastdim(k)
    g = np.exp(g1)
    s = np.zeros((H, Dh, Dh), np.float32)
    s = s * g[:, None, :]
    sk = np.einsum("hab,ha->hb", s, kn)
    d = (v - sk) * beta
    s = s + kn[:, :, None] * d[:, None, :]
    o = np.einsum("hab,ha->hb", s, qn)
    cmp("scan_out", o, dd, "kda_scan_out-0", Di)

    g_a = W.ssm_g_a @ x
    g2 = (W.ssm_g_b @ g_a).reshape(H, Dh)
    cmp("g2", g2, dd, "kda_g2-0", Di)

    on = rmsnorm(o, W.ssm_o_norm, 1e-5)
    cmp("normed", on, dd, "kda_normed-0", Di)

    gated = on * sigmoid(g2)
    out = W.wo @ gated.reshape(H * Dh)
    cmp("out", out, dd, "kda_out-0", Ne)


if __name__ == "__main__":
    main()
