"""
glm5-next MLA / DSA attention — numpy reference (multi-token causal, nope-only).
Transcribed from glm5-next.cpp build_dsa_layer. The k-pool sparse indexer is a NO-OP
when the number of keys <= indexer_top_k (2048), so a short-prompt probe validates the
full MLA path (latent q/kv, wk_b absorb, causal softmax, wv_b, wo) with the indexer
inactive. Layer index configurable; defaults to 3 (first MLA layer).
"""
import os, sys, glob
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType

Ne = 4096
Q_LORA = 1536
KV_LORA = 512
H = 64
HEAD_K = 256          # n_embd_head_k_mla (nope-only, n_rot=0)
HEAD_V = 256          # n_embd_head_v_mla
NORM_EPS = 1e-5
KQ_SCALE = 1.0 / np.sqrt(HEAD_K)


def rmsnorm(x, w, eps=NORM_EPS):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32) * w


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def deq_flat(t):
    if t.tensor_type == GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32).reshape(-1)
    return gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float32).reshape(-1)


class W:
    pass


def load(gguf_dir, L=3):
    b = f"blk.{L}."
    SH = {
        b + "attn_q_a.weight": ("wq_a", (Q_LORA, Ne)),
        b + "attn_q_a_norm.weight": ("q_a_norm", (Q_LORA,)),
        b + "attn_q_b.weight": ("wq_b", (H * HEAD_K, Q_LORA)),
        b + "attn_kv_a_mqa.weight": ("wkv_a_mqa", (KV_LORA, Ne)),
        b + "attn_kv_a_norm.weight": ("kv_a_norm", (KV_LORA,)),
        b + "attn_k_b.weight": ("wk_b", (H, KV_LORA, HEAD_K)),      # [h, kv, nope]
        b + "attn_v_b.weight": ("wv_b", (H, HEAD_V, KV_LORA)),      # [h, v, kv]
        b + "attn_output.weight": ("wo", (Ne, H * HEAD_V)),
    }
    w = W()
    need = set(SH)
    for sh in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
        r = GGUFReader(sh)
        for t in r.tensors:
            if t.name in need:
                attr, shp = SH[t.name]
                setattr(w, attr, deq_flat(t).reshape(shp))
                need.discard(t.name)
    assert not need, f"missing: {need}"
    return w


def mla_forward(x, w):
    """x: [n_embd, T] (columns = tokens). Returns (attn_out [n_embd, T], intermediates)."""
    T = x.shape[1]
    qr = np.stack([rmsnorm(w.wq_a @ x[:, t], w.q_a_norm) for t in range(T)], 1)   # [1536, T]
    kv = np.stack([rmsnorm(w.wkv_a_mqa @ x[:, t], w.kv_a_norm) for t in range(T)], 1)  # [512, T]
    # per-token per-head absorbed query: qa[t][h] = wk_b[h] @ q[t][h]
    qa = np.zeros((T, H, KV_LORA), np.float32)
    for t in range(T):
        q = (w.wq_b @ qr[:, t]).reshape(H, HEAD_K)       # [H, 256]
        for h in range(H):
            qa[t, h] = w.wk_b[h] @ q[h]                   # [512]
    out = np.zeros((Ne, T), np.float32)
    kqv_all = np.zeros((H * HEAD_V, T), np.float32)
    for t in range(T):
        o = np.zeros((H, HEAD_V), np.float32)
        for h in range(H):
            sc = (qa[t, h] @ kv[:, :t + 1]) * KQ_SCALE     # scores over s<=t
            ww = softmax(sc)
            ctx = kv[:, :t + 1] @ ww                        # [512]
            o[h] = w.wv_b[h] @ ctx                          # [256]
        flat = o.reshape(-1)                                # [v + 256*h]
        kqv_all[:, t] = flat
        out[:, t] = w.wo @ flat
    return out, {"q_resid": qr, "kv_cmpr": kv, "kqv_out": kqv_all}
