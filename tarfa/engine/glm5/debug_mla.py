"""Localize the torch-MLA bug: compare torch vs numpy MLA intermediates on layer 3."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import read_dump, cos
import mla_ref, glm5_next as G

gguf, dd = sys.argv[1], sys.argv[2]
an, _ = read_dump(os.path.join(dd, "dumps3"), "attn_norm-3"); T = an.size // 4096
x = an[:T * 4096].reshape(T, 4096).T.astype(np.float32)        # [Ne, T]

# numpy
wn = mla_ref.load(gguf, L=3)
# torch
G.DEV = "cuda:0"
mt = G.GLM5(gguf); Lt = mt.layer(3)

def cmp(name, a_np, a_t):
    a = a_np.ravel(); b = a_t.detach().float().cpu().numpy().ravel()
    print(f"{name:10s} cos={cos(a, b):.6f}  shapes np{a_np.shape} t{tuple(a_t.shape)}")

# weights match?
cmp("wq_a", wn.wq_a, Lt["wq_a"]); cmp("wq_b", wn.wq_b, Lt["wq_b"])
cmp("wk_b", wn.wk_b, Lt["wk_b"]); cmp("wv_b", wn.wv_b, Lt["wv_b"])
cmp("wkv_a", wn.wkv_a_mqa, Lt["wkv_a_mqa"]); cmp("wo", wn.wo, Lt["wo"])

# numpy intermediates
qr_np = mla_ref.rmsnorm(wn.wq_a @ x[:, 0], wn.q_a_norm)
q_np = (wn.wq_b @ qr_np).reshape(64, 256)
qa_np = np.stack([wn.wk_b[h] @ q_np[h] for h in range(64)])    # [H,512]
out_np, _ = mla_ref.mla_forward(x, wn)

# torch intermediates
xt = torch.from_numpy(x.copy()).to("cuda:0", torch.float32)
qr_t = G.rmsnorm((Lt["wq_a"] @ xt).T, Lt["q_a_norm"])[0]        # token 0
q_t = (Lt["wq_b"] @ qr_t).reshape(64, 256)
qa_t = torch.einsum("hd,hkd->hk", q_t, Lt["wk_b"])
out_t = G.mla_forward_t(xt, Lt)

cmp("qr(t0)", qr_np, qr_t)
cmp("q(t0)", q_np, q_t)
cmp("qa(t0)", qa_np, qa_t)
cmp("attn_out", out_np, out_t)
