"""
GLM-5.3 NextN (MTP) draft head, block 45: predicts token i+2 from the trunk's final hidden at
position i and the embedding of token i+1 (DeepSeek-V3-style multi-token prediction):
  x = eh_proj([enorm(emb(t_{i+1})); hnorm(h_i)])
  x = x + MLA(attn_norm(x))          (own latent cache; nope-only, indexer selects all < 2048)
  x = x + MoE(ffn_norm(x))           (288 routed + 1 shared, no hyper-connections in block 45)
  logits = output(shared_head_norm(x))
llama.cpp has no MTP graph for glm5-next yet, so the layout choices (concat order, which hidden)
are selected by acceptance rate: GLM5_MTP_ORDER=eh|he, GLM5_MTP_H=norm|raw.
"""
import os, time, numpy as np, torch
import glm5_next as G
from glm5_next import Ne, H, Q_LORA, KV_LORA, HEAD_K, rmsnorm, swiglu_clamp, _deq_np
from q8 import mv, load_q8

KINDS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
NL = 45


class MTP:
    def __init__(self, eng):
        self.eng = eng
        m, dev = eng.m, eng.ar.device
        b = f"blk.{NL}."
        t = lambda n, s: G._t(_deq_np(m.idx[b + n]), s)
        q = lambda n, r, c: load_q8(m.idx[b + n], r, c, dev, G.DT)
        self.w = w = {
            "enorm": t("nextn.enorm.weight", (Ne,)), "hnorm": t("nextn.hnorm.weight", (Ne,)),
            "head_norm": t("nextn.shared_head_norm.weight", (Ne,)),
            "eh": q("nextn.eh_proj.weight", Ne, 2 * Ne),
            "attn_norm": t("attn_norm.weight", (Ne,)), "ffn_norm": t("ffn_norm.weight", (Ne,)),
            "wq_a": q("attn_q_a.weight", Q_LORA, Ne), "q_a_norm": t("attn_q_a_norm.weight", (Q_LORA,)),
            "wq_b": q("attn_q_b.weight", H * HEAD_K, Q_LORA),
            "wkv_a_mqa": q("attn_kv_a_mqa.weight", KV_LORA, Ne), "kv_a_norm": t("attn_kv_a_norm.weight", (KV_LORA,)),
            "wk_b": t("attn_k_b.weight", (H, KV_LORA, HEAD_K)), "wv_b": t("attn_v_b.weight", (H, HEAD_K, KV_LORA)),
            "wo": q("attn_output.weight", Ne, H * HEAD_K),
            "router": torch.from_numpy(_deq_np(m.idx[b + "ffn_gate_inp.weight"]).reshape(288, Ne).copy()).to(dev),
            "exp_bias": t("exp_probs_b.bias", (288,)).float(),
            "sh_gate": q("ffn_gate_shexp.weight", 2048, Ne), "sh_up": q("ffn_up_shexp.weight", 2048, Ne),
            "sh_down": q("ffn_down_shexp.weight", Ne, 2048),
        }
        self.ts = [m.idx[b + k] for k in KINDS]
        self.kv = torch.zeros(eng.ar.shape[0], KV_LORA, device=dev, dtype=G.DT)
        self.order = os.environ.get("GLM5_MTP_ORDER", "eh")
        self.hsrc = os.environ.get("GLM5_MTP_H", "norm")

    def reset(self):
        self.kv.zero_()

    @torch.no_grad()
    def draft(self, h, emb, pos):
        """h [n, Ne]: trunk final hiddens at positions `pos` (consecutive); emb [n, Ne]: embeddings
        of the tokens at pos+1 -> logits [n, vocab] for pos+2."""
        from glm5_spec import mla_t
        from cpu_moe import moe_cpu_numa_t
        w, n = self.w, h.shape[0]
        e_, h_ = rmsnorm(emb, w["enorm"]), rmsnorm(h, w["hnorm"])
        x = mv(w["eh"], torch.cat([e_, h_] if self.order == "eh" else [h_, e_], -1).to(G.DT)).float()
        pt = torch.tensor(pos, device=x.device)
        x = x + mla_t(rmsnorm(x, w["attn_norm"].float()).to(G.DT), w, self.kv, pt, self.eng.ar).float()
        fn = rmsnorm(x, w["ffn_norm"].float())
        probs = torch.sigmoid(fn @ w["router"].T)
        sel = torch.topk(probs + w["exp_bias"], 8, dim=-1).indices
        ww = probs.gather(1, sel); ww = ww / ww.sum(-1, keepdim=True) * 2.5
        f16 = fn.to(G.DT)
        sh = mv(w["sh_down"], swiglu_clamp(mv(w["sh_gate"], f16), mv(w["sh_up"], f16))).float()
        uni = {}
        for t, (st, wt) in enumerate(zip(sel.tolist(), ww.tolist())):
            for e, we in zip(st, wt):
                uni.setdefault(e, np.zeros(n, np.float32))[t] = we
        es = list(uni); g, u, d = self.ts
        y = moe_cpu_numa_t([g.data[e] for e in es], [u.data[e] for e in es], [d.data[e] for e in es],
                           g.tensor_type, u.tensor_type, d.tensor_type, fn.cpu().numpy(),
                           np.stack([uni[e] for e in es]))
        x = x + sh + torch.from_numpy(y).to(x.device)
        return mv(self.eng.out_w, rmsnorm(x, w["head_norm"].float()).to(G.DT))

    # ---- graphed draft for decode: [pre-routing graph] -> host CPU experts -> [residual + head + argmax graph]
    def _pre(self, n):
        from glm5_spec import mla_t
        w, b = self.w, self.buf[n]
        e_, h_ = rmsnorm(b["e"], w["enorm"]), rmsnorm(b["h"], w["hnorm"])
        x = mv(w["eh"], torch.cat([e_, h_] if self.order == "eh" else [h_, e_], -1).to(G.DT)).float()
        x = x + mla_t(rmsnorm(x, w["attn_norm"].float()).to(G.DT), w, self.kv, b["pos"], self.eng.ar).float()
        fn = rmsnorm(x, w["ffn_norm"].float())
        probs = torch.sigmoid(fn @ w["router"].T)
        sel = torch.topk(probs + w["exp_bias"], 8, dim=-1).indices
        ww = probs.gather(1, sel)
        f16 = fn.to(G.DT)
        b["x"].copy_(x + mv(w["sh_down"], swiglu_clamp(mv(w["sh_gate"], f16), mv(w["sh_up"], f16))).float())
        b["fn"].copy_(fn); b["sel"].copy_(sel); b["w"].copy_(ww / ww.sum(-1, keepdim=True) * 2.5)
        b["h_fn"].copy_(b["fn"], non_blocking=True); b["h_sel"].copy_(b["sel"], non_blocking=True)
        b["h_w"].copy_(b["w"], non_blocking=True)

    def _post(self, n):
        w, b = self.w, self.buf[n]
        x = (b["x"] + b["y"])[n - 1:]                         # only the last position's draft is used
        lg = mv(self.eng.out_w, rmsnorm(x, w["head_norm"].float()).to(G.DT))
        b["tok"].copy_(lg.argmax(-1))
        b["h_tok"].copy_(b["tok"], non_blocking=True)

    def capture(self):
        dev = self.kv.device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        pin = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt).pin_memory()
        self.buf, self.gA, self.gB = {}, {}, {}
        kv0 = self.kv.clone()
        for n in (1, 2):
            self.buf[n] = dict(h=z(n, Ne), e=z(n, Ne), pos=torch.arange(n, device=dev), x=z(n, Ne), fn=z(n, Ne),
                               sel=torch.zeros(n, 8, device=dev, dtype=torch.long), w=z(n, 8), y=z(n, Ne),
                               tok=torch.zeros(1, device=dev, dtype=torch.long), h_fn=pin(n, Ne), h_w=pin(n, 8),
                               h_sel=pin(n, 8, dt=torch.long), h_y=pin(n, Ne), h_tok=pin(1, dt=torch.long))
            s = torch.cuda.Stream(dev)
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._pre(n); self._post(n)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            for gd, f in ((self.gA, self._pre), (self.gB, self._post)):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    f(n)
                gd[n] = g
        self.kv.copy_(kv0)
        self.ev = torch.cuda.Event()
        self.cpu_args = None
        if getattr(self.eng, "cpu_layer", None) is not None:     # block 45 through the bound per-layer call too
            import cpu_moe
            cpu_moe.bind_layer(45, *self.ts)
            self.cpu_args = {n: tuple(b[k].data_ptr() for k in ("h_sel", "h_w", "h_fn", "h_y")) for n, b in self.buf.items()}

    def draft_g(self, h, emb, pos):
        """Graphed draft: h, emb [n, Ne] (n = 1 or 2), consecutive positions from `pos` -> draft token."""
        from cpu_moe import moe_cpu_numa_t
        n = h.shape[0]; b = self.buf[n]
        b["h"].copy_(h); b["e"].copy_(emb); b["pos"].copy_(torch.arange(pos, pos + n))
        pc = time.perf_counter; t0 = pc()
        self.gA[n].replay()
        self.ev.record(); self.ev.synchronize()
        t1 = pc()
        if self.cpu_args is not None:
            import cpu_moe
            cpu_moe.moe_cpu_layer_t(45, n, (1 << 8 * n) - 1, *self.cpu_args[n])
        else:
            uni = {}
            for t in range(n):
                for e, we in zip(b["h_sel"][t].tolist(), b["h_w"][t].tolist()):
                    uni.setdefault(e, np.zeros(n, np.float32))[t] = we
            es = list(uni); g, u, d = self.ts
            moe_cpu_numa_t([g.data[e] for e in es], [u.data[e] for e in es], [d.data[e] for e in es],
                           g.tensor_type, u.tensor_type, d.tensor_type, b["h_fn"].numpy(),
                           np.stack([uni[e] for e in es]), out=b["h_y"].numpy())
        t2 = pc()
        b["y"].copy_(b["h_y"], non_blocking=True)
        self.gB[n].replay()
        self.ev.record(); self.ev.synchronize()
        st = self.eng.stats; st["ms mtp gpu pre"] += 1000 * (t1 - t0); st["ms mtp cpu"] += 1000 * (t2 - t1)
        st["ms mtp gpu post"] += 1000 * (pc() - t2)
        return int(b["h_tok"][0])
