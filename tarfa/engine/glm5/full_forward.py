"""
glm5-next full 46-layer forward in numpy, assembled from the validated kernels.
Feeds the oracle hc_init (embedding streams) and compares final hidden/logits to llama.cpp.
KDA layers run sequentially (autoregressive recurrence, carried conv+state) which equals
the chunked prefill result; MLA layers run causal; MoE per token; mHC streams across layers.
Usage: python full_forward.py <gguf_dir> <dump5_dir>
"""
import os, sys, glob, time
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kda_ref import kda_decode_step, rmsnorm as rms_w
from glm5_layer import hc_pre, hc_post, dense_ffn, H, Dh, Ne, Di, HC, NORM_EPS
from mla_ref import mla_forward
from moe_ref import swiglu_clamp, sigmoid, silu, N_USED, SCALE_W

N_LAYER = 45   # trunk decode layers; block 45 is the NextN/MTP head (nextn_predict_layers=1)
N_DENSE = 3
Q_LORA, KV_LORA, HEAD_K = 1536, 512, 256


class Bag:
    pass


def deq(t):
    if t.tensor_type == GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32)
    return gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)


def deq_flat(t):
    return deq(t).reshape(-1)


class Model:
    def __init__(self, gguf_dir):
        self.idx = {}
        self._r = []
        for sh in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
            r = GGUFReader(sh)
            for t in r.tensors:
                self.idx[t.name] = t
            self._r.append(r)

    def f(self, name, shape=None):
        a = deq_flat(self.idx[name])
        return a.reshape(shape) if shape else a

    def is_mla(self, L):
        return f"blk.{L}.attn_kv_a_mqa.weight" in self.idx

    def load_layer(self, L):
        b = f"blk.{L}."
        w = Bag()
        w.attn_norm = self.f(b + "attn_norm.weight")
        w.ffn_norm = self.f(b + "ffn_norm.weight")
        w.hc_attn_fn = self.f(b + "hc_attn_fn.weight", (24, HC * Ne))
        w.hc_attn_scale = self.f(b + "hc_attn_scale.weight")
        w.hc_attn_base = self.f(b + "hc_attn_base.weight")
        w.hc_ffn_fn = self.f(b + "hc_ffn_fn.weight", (24, HC * Ne))
        w.hc_ffn_scale = self.f(b + "hc_ffn_scale.weight")
        w.hc_ffn_base = self.f(b + "hc_ffn_base.weight")
        if self.is_mla(L):
            w.mla = True
            w.wq_a = self.f(b + "attn_q_a.weight", (Q_LORA, Ne))
            w.q_a_norm = self.f(b + "attn_q_a_norm.weight")
            w.wq_b = self.f(b + "attn_q_b.weight", (H * HEAD_K, Q_LORA))
            w.wkv_a_mqa = self.f(b + "attn_kv_a_mqa.weight", (KV_LORA, Ne))
            w.kv_a_norm = self.f(b + "attn_kv_a_norm.weight")
            w.wk_b = self.f(b + "attn_k_b.weight", (H, KV_LORA, HEAD_K))
            w.wv_b = self.f(b + "attn_v_b.weight", (H, HEAD_K, KV_LORA))
            w.wo = self.f(b + "attn_output.weight", (Ne, H * HEAD_K))
        else:
            w.mla = False
            w.wq = self.f(b + "attn_q.weight", (Di, Ne))
            w.wk = self.f(b + "attn_k.weight", (Di, Ne))
            w.wv = self.f(b + "attn_v.weight", (Di, Ne))
            w.q_conv = self.f(b + "ssm_conv1d_q.weight").reshape(Di, -1).T
            w.k_conv = self.f(b + "ssm_conv1d_k.weight").reshape(Di, -1).T
            w.v_conv = self.f(b + "ssm_conv1d_v.weight").reshape(Di, -1).T
            w.ssm_f_a = self.f(b + "ssm_f_a.weight", (Dh, Ne))
            w.ssm_f_b = self.f(b + "ssm_f_b.weight", (Di, Dh))
            w.ssm_dt_b = self.f(b + "ssm_dt.bias")
            w.ssm_a = self.f(b + "ssm_a")
            w.ssm_beta = self.f(b + "ssm_beta.weight", (H, Ne))
            w.ssm_g_a = self.f(b + "ssm_g_a.weight", (Dh, Ne))
            w.ssm_g_b = self.f(b + "ssm_g_b.weight", (Di, Dh))
            w.ssm_o_norm = self.f(b + "ssm_norm.weight")
            w.wo = self.f(b + "attn_output.weight", (Ne, Di))
        if L < N_DENSE:
            w.moe = False
            w.ffn_gate = self.f(b + "ffn_gate.weight", (12288, Ne))
            w.ffn_up = self.f(b + "ffn_up.weight", (12288, Ne))
            w.ffn_down = self.f(b + "ffn_down.weight", (Ne, 12288))
        else:
            w.moe = True
            w.router = self.f(b + "ffn_gate_inp.weight", (288, Ne))
            w.exp_bias = self.f(b + "exp_probs_b.bias")
            w.gate_exps = self.idx[b + "ffn_gate_exps.weight"]
            w.up_exps = self.idx[b + "ffn_up_exps.weight"]
            w.down_exps = self.idx[b + "ffn_down_exps.weight"]
            w.sh_gate = self.f(b + "ffn_gate_shexp.weight", (2048, Ne))
            w.sh_up = self.f(b + "ffn_up_shexp.weight", (2048, Ne))
            w.sh_down = self.f(b + "ffn_down_shexp.weight", (Ne, 2048))
        return w


def moe_forward(model, x, w):
    logits = w.router @ x
    probs = sigmoid(logits)
    sel = np.argsort(probs + w.exp_bias)[-N_USED:]
    ww = probs[sel]; ww = ww / ww.sum() * SCALE_W
    out = np.zeros(Ne, np.float32)
    for e, we in zip(sel, ww):
        g = deq_slice(w.gate_exps, e) @ x
        u = deq_slice(w.up_exps, e) @ x
        d = deq_slice(w.down_exps, e) @ swiglu_clamp(g, u)
        out += we * d
    gs = w.sh_gate @ x
    us = w.sh_up @ x
    out = out + w.sh_down @ swiglu_clamp(gs, us)
    return out.astype(np.float32)


def deq_slice(t, e):
    return gguf.quants.dequantize(np.array(t.data[e]), t.tensor_type).astype(np.float32)


def forward(model, hc_init):
    # hc_init: [T, hc, n_embd]
    T = hc_init.shape[0]
    Hs = hc_init.copy()
    for L in range(N_LAYER):
        w = model.load_layer(L)
        # attention sub-block
        residual = Hs.copy()
        collapsed = np.zeros((Ne, T), np.float32); pa = [None] * T; ca = [None] * T
        for t in range(T):
            c, pa[t], ca[t] = hc_pre(Hs[t], w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, NORM_EPS)
            collapsed[:, t] = rms_w(c, w.attn_norm, NORM_EPS)
        if w.mla:
            attn, _ = mla_forward(collapsed, w)
        else:
            cs = {c: np.zeros((3, Di), np.float32) for c in "qkv"}
            s = np.zeros((H, Dh, Dh), np.float32)
            attn = np.zeros((Ne, T), np.float32)
            for t in range(T):
                o, cs, s = kda_decode_step(collapsed[:, t], w, cs, s)
                attn[:, t] = o
        for t in range(T):
            Hs[t] = hc_post(attn[:, t], residual[t], pa[t], ca[t])
        # ffn sub-block
        residual = Hs.copy()
        for t in range(T):
            c, pf, cf = hc_pre(Hs[t], w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, NORM_EPS)
            fn = rms_w(c, w.ffn_norm, NORM_EPS)
            ff = (moe_forward(model, fn.astype(np.float32), w) if w.moe
                  else dense_ffn(fn.astype(np.float32), w))
            Hs[t] = hc_post(ff, residual[t], pf, cf)
        print(f"  layer {L:2d} ({'MLA' if w.mla else 'KDA'},{'MoE' if w.moe else 'dense'}) done", flush=True)
    # head: mean streams, output_norm (with weight), output projection
    out_norm = model.f("output_norm.weight")
    H_mean = Hs.mean(axis=1)                       # [T, n_embd]
    hidden = np.stack([rms_w(H_mean[t], out_norm, NORM_EPS) for t in range(T)], 1)  # [n_embd, T]
    return hidden


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    from probe_layer0 import read_dump
    model = Model(gguf_dir)
    raw, _ = read_dump(dd, "hc_init")
    T = raw.size // (HC * Ne)
    hc_init = raw[:T * HC * Ne].reshape(T, HC, Ne).astype(np.float32)
    print(f"T={T} tokens; running {N_LAYER} layers...", flush=True)
    t0 = time.time()
    hidden = forward(model, hc_init)               # [n_embd, T]
    print(f"forward done in {time.time()-t0:.0f}s")

    rn, ne = read_dump(dd, "result_norm")
    nout = rn.size // Ne
    rn = rn.reshape(nout, Ne).T                     # [n_embd, nout]
    # result_norm may hold only the last token(s); compare the overlap from the end
    k = min(nout, T)
    mine = hidden[:, T - k:]
    ref = rn[:, nout - k:]
    print(f"result_norm: oracle nout={nout}, mine T={T}; comparing last {k}")
    for j in range(k):
        print(f"  tok -{k-j}: cos={cos(mine[:, j], ref[:, j]):.6f}  maxdiff={np.max(np.abs(mine[:,j]-ref[:,j])):.3e}")

    # logits + argmax (the real gate) on the last token
    ow = model.idx["output.weight"]
    out_w = deq(ow).reshape(-1, Ne)                 # [vocab, n_embd]
    vocab = out_w.shape[0]
    my_logits = out_w @ hidden[:, -1]               # last token
    ro, rne = read_dump(dd, "result_output")
    r_nout = ro.size // vocab
    ref_logits = ro.reshape(r_nout, vocab)[-1]      # last token
    print(f"logits: vocab={vocab}  cos={cos(my_logits, ref_logits):.6f}")
    def top5p(lg):
        p = np.exp(lg - lg.max()); p /= p.sum(); ix = np.argsort(p)[-5:][::-1]
        return "  ".join(f"{int(i)}:{p[i]:.3f}" for i in ix)
    print(f"  my  top5 probs: {top5p(my_logits)}")
    print(f"  ref top5 probs: {top5p(ref_logits)}")
    print(f"  my argmax   = {int(np.argmax(my_logits))}  (top5 {np.argsort(my_logits)[-5:][::-1].tolist()})")
    print(f"  ref argmax  = {int(np.argmax(ref_logits))}  (top5 {np.argsort(ref_logits)[-5:][::-1].tolist()})")
    print(f"  ARGMAX MATCH: {int(np.argmax(my_logits)) == int(np.argmax(ref_logits))}")


if __name__ == "__main__":
    main()
