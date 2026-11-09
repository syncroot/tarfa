"""Localize the KDA pos>0 drift: compare the SCAN output (pre-output-gate) per position.
Inlines the KDA step so I can capture scan (o) before the output gate; compares to
kda_scan_out-0 [head_dim, n_head, T]. If scan matches but kda_out doesn't -> output gate;
if scan diverges -> the delta recurrence / state carry."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import load_weights, read_dump, cos, Ne, Di, H, Dh
from kda_ref import conv_branch, rmsnorm, silu, sigmoid, l2norm_lastdim, GATE_LB, NORM_EPS


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    w = load_weights(gguf_dir)
    an, _ = read_dump(dd, "attn_norm-0"); T = an.size // Ne
    x = an[:T * Ne].reshape(T, Ne).T
    scan_ref, _ = read_dump(dd, "kda_scan_out-0")        # [head_dim, n_head, T] -> dh+128h+8192t
    out_ref, _ = read_dump(dd, "kda_out-0")              # [n_embd, T]

    cs = {c: np.zeros((3, Di), np.float32) for c in "qkv"}
    s = np.zeros((H, Dh, Dh), np.float32)
    print(f"T={T}; per-position scan (pre-gate) and full-out cosines:")
    for t in range(T):
        xt = x[:, t].astype(np.float32)
        q, cs["q"] = conv_branch(xt, w.wq, w.q_conv, cs["q"], H, Dh)
        k, cs["k"] = conv_branch(xt, w.wk, w.k_conv, cs["k"], H, Dh)
        v, cs["v"] = conv_branch(xt, w.wv, w.v_conv, cs["v"], H, Dh)
        g1 = (w.ssm_f_b @ (w.ssm_f_a @ xt) + w.ssm_dt_b).reshape(H, Dh)
        g1 = sigmoid(-(g1 * w.ssm_a.reshape(H, 1))) * GATE_LB
        beta = sigmoid(w.ssm_beta @ xt).reshape(H, 1)
        qn = l2norm_lastdim(q) / np.sqrt(Dh); kn = l2norm_lastdim(k)
        g = np.exp(g1)
        s = s * g[:, None, :]
        sk = np.einsum("hab,ha->hb", s, kn)
        d = (v - sk) * beta
        s = s + kn[:, :, None] * d[:, None, :]
        o = np.einsum("hab,ha->hb", s, qn)               # scan output [H, Dh]
        sref = scan_ref[t * Di:(t + 1) * Di]
        scos = cos(o.ravel(), sref)
        # full block out
        g2 = (w.ssm_g_b @ (w.ssm_g_a @ xt)).reshape(H, Dh)
        on = rmsnorm(o, w.ssm_o_norm, NORM_EPS)
        out = w.wo @ (on * sigmoid(g2)).reshape(H * Dh)
        ocos = cos(out, out_ref[t * Ne:(t + 1) * Ne])
        print(f"  pos {t}: scan cos={scos:.6f}   out cos={ocos:.6f}")


if __name__ == "__main__":
    main()
