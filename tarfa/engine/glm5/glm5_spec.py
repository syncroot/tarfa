"""
Speculative decode with GLM-5.3's own NextN (MTP) draft head (mtp.py), on top of glm5_gdecode.

Each step verifies two tokens in one trunk pass: [a, d] = (last emitted token, MTP draft) at
positions p, p+1. Weight reads are shared: the Q8 trunk matrices are read once for both
tokens (q8.q8mm), and the CPU experts run over the union of both tokens' misses
(moe_cpu_numa_t), each row read once. Greedy acceptance: if argmax(logits[p]) == d, both d and
argmax(logits[p+1]) are emitted; otherwise only argmax(logits[p]), and the KDA recurrent/conv
states roll back to their snapshot after token a (the MLA caches need no rollback: the stale
row p+1 is rewritten before the next query reads it). Greedy-equivalent only up to rounding:
the batched verify rounds differently from single-token decode and can flip a near-tie, so this
is a separate, labeled mode (run_spec.sh), not the default.

Prefill and the first token use the single-token graphs of glm5_gdecode (shared state).
Usage: GLM5_SPEC=1 ... python glm5_spec.py <gguf_dir> <out_dir> [n_gen]   (GLM5_PROMPT set)
"""
import os, sys, time, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_gdecode as GD
from glm5_gdecode import Engine, N_LAYER, DEV, DT, tokenize, check_len
from glm5_next import Ne, H, Dh, Di, HC, rmsnorm, hc_pre, hc_post, swiglu_clamp, l2n, silu, GATE_LB, NORM_EPS
from q8 import mv

T = 2


def kda_t(x, L, cs, s, snap_cs, snap_s):     # x [T, Ne]; snapshots hold the state after token 0
    n = x.shape[0]
    def conv(proj, cw, c, sc):
        win = torch.cat([c, mv(proj, x)], 0)                               # [3 + n, Di]
        y = torch.stack([(win[t:t + 4] * cw).sum(0) for t in range(n)])
        sc.copy_(win[1:4]); c.copy_(win[n:n + 3])
        return silu(y).reshape(n, H, Dh)
    q = conv(L["wq"], L["q_conv"], cs["q"], snap_cs["q"])
    k = conv(L["wk"], L["k_conv"], cs["k"], snap_cs["k"])
    v = conv(L["wv"], L["v_conv"], cs["v"], snap_cs["v"])
    sd = s.dtype
    g1 = (mv(L["ssm_f_b"], mv(L["ssm_f_a"], x)) + L["ssm_dt_b"]).reshape(n, H, Dh).to(sd)
    eg = torch.exp(torch.sigmoid(-(g1 * L["ssm_a"].reshape(H, 1).to(sd))) * GATE_LB)
    beta = torch.sigmoid(mv(L["ssm_beta"], x).to(sd)).reshape(n, H, 1)
    qn = l2n(q.to(sd)) / (Dh ** 0.5); kn = l2n(k.to(sd)); v = v.to(sd)
    st, os_ = s, []
    for t in range(n):
        st = st * eg[t][:, :, None]                                        # decay on the KEY axis
        d = (v[t] - torch.einsum("hab,ha->hb", st, kn[t])) * beta[t]
        st = st + kn[t][:, :, None] * d[:, None, :]
        if t == 0:
            snap_s.copy_(st)
        os_.append(torch.einsum("hab,ha->hb", st, qn[t]))
    s.copy_(st)
    g2 = mv(L["ssm_g_b"], mv(L["ssm_g_a"], x)).reshape(n, H, Dh).to(sd)
    on = rmsnorm(torch.stack(os_), L["ssm_o_norm"].to(sd), NORM_EPS)
    return mv(L["wo"], (on * torch.sigmoid(g2)).reshape(n, Di).to(DT))


def mla_t(xt, L, kvb, pos_t, ar):            # xt [n, Ne], pos_t [n] (consecutive positions)
    n = xt.shape[0]
    qr = rmsnorm(mv(L["wq_a"], xt), L["q_a_norm"])
    q = mv(L["wq_b"], qr).reshape(n, H, 256)
    kvb.index_copy_(0, pos_t, rmsnorm(mv(L["wkv_a_mqa"], xt), L["kv_a_norm"]))
    qa = torch.einsum("thd,hkd->thk", q, L["wk_b"])
    sc = torch.einsum("thk,sk->ths", qa, kvb) * (1.0 / (256 ** 0.5))
    w = torch.softmax(sc.float().masked_fill(ar[None, None, :] > pos_t[:, None, None], float("-inf")), -1).to(DT)
    o = torch.einsum("hvk,thk->thv", L["wv_b"], torch.einsum("ths,sk->thk", w, kvb))
    return mv(L["wo"], o.reshape(n, H * 256))


class SpecEngine(Engine):
    def __init__(self, gguf_dir):
        super().__init__(gguf_dir)
        z = lambda *s, dt=DT: torch.zeros(*s, device=DEV, dtype=dt)
        self.snap_s = {L: torch.zeros_like(s) for L, s in self.kda_s.items()}
        self.snap_cs = {L: {c: torch.zeros_like(t) for c, t in d.items()} for L, d in self.kda_cs.items()}
        self.pos2 = torch.zeros(T, device=DEV, dtype=torch.long)
        self.emb2 = z(T, Ne); self.res2 = z(T, HC, Ne); self.pf2 = z(T, HC); self.cf2 = z(T, HC, HC)
        self.x2 = z(T, Ne, dt=torch.float32); self.w2 = z(T, 8, dt=torch.float32)
        self.sel2 = torch.zeros(T, 8, device=DEV, dtype=torch.long)
        f32 = lambda: z(T, Ne, dt=torch.float32)
        self.sh2, self.yr2, self.y0c2, self.ycpu2 = f32(), f32(), f32(), f32()
        self.logits2 = z(T, self.out_w.shape[0]); self.hn2 = z(T, Ne)
        pin = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt).pin_memory()
        self.h_x2, self.h_w2, self.h_sel2, self.h_y2 = pin(T, Ne), pin(T, 8), pin(T, 8, dt=torch.long), pin(T, Ne)
        self.graphs2 = {}
        self.stats = collections.Counter()
        self.trace2 = None

    # ---- two-token graph bodies ----
    def _attn_t(self, L, Hs):                # Hs [T, HC, Ne]
        w = self.layers[L]
        g = [hc_pre(Hs[t], w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"]) for t in range(T)]
        an = rmsnorm(torch.stack([c for c, _, _ in g]), w["attn_norm"])
        blk = (mla_t(an, w, self.mla_kv[L], self.pos2, self.ar) if w["mla"]
               else kda_t(an, w, self.kda_cs[L], self.kda_s[L], self.snap_cs[L], self.snap_s[L]))
        return torch.stack([hc_post(blk[t], Hs[t], g[t][1], g[t][2]) for t in range(T)])

    def _ffn_pre_t(self, L, Hs):
        w = self.layers[L]
        g = [hc_pre(Hs[t], w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"]) for t in range(T)]
        return (rmsnorm(torch.stack([c for c, _, _ in g]), w["ffn_norm"]),
                torch.stack([p for _, p, _ in g]), torch.stack([c for _, _, c in g]))

    def _front_t(self, L, Hs):
        Hs = self._attn_t(L, Hs)
        fn, pf, cf = self._ffn_pre_t(L, Hs)
        w = self.layers[L]
        self.res2.copy_(Hs); self.pf2.copy_(pf); self.cf2.copy_(cf)
        x32 = fn.float()
        probs = torch.sigmoid(x32 @ w["router"].float().T)
        sel = torch.topk(probs + w["exp_bias"].float(), 8, dim=-1).indices.flip(-1)
        ww = probs.gather(1, sel)
        self.x2.copy_(x32); self.sel2.copy_(sel); self.w2.copy_(ww / ww.sum(-1, keepdim=True) * 2.5)
        self.sh2.copy_(mv(w["sh_down"], swiglu_clamp(mv(w["sh_gate"], fn), mv(w["sh_up"], fn))).float())

    def _finish_t(self):
        ff = (self.yr2 + self.y0c2 + self.ycpu2 + self.sh2).to(DT)
        return torch.stack([hc_post(ff[t], self.res2[t], self.pf2[t], self.cf2[t]) for t in range(T)])

    def _body_t(self, k):
        if k == 3:
            Hs = self.emb2[:, None, :].repeat(1, HC, 1)
            for L in range(3):
                Hs = self._attn_t(L, Hs)
                fn, pf, cf = self._ffn_pre_t(L, Hs)
                wl = self.layers[L]
                ff = mv(wl["ffn_down"], swiglu_clamp(mv(wl["ffn_gate"], fn), mv(wl["ffn_up"], fn)))
                Hs = torch.stack([hc_post(ff[t], Hs[t], pf[t], cf[t]) for t in range(T)])
            self._front_t(3, Hs)
        elif k < N_LAYER:
            self._front_t(k, self._finish_t())
        else:
            hn = rmsnorm(self._finish_t().mean(1), self.out_norm)
            self.hn2.copy_(hn)
            self.logits2.copy_(mv(self.out_w, hn))

    def _d2h_t(self):
        self.h_x2.copy_(self.x2, non_blocking=True); self.h_w2.copy_(self.w2, non_blocking=True)
        self.h_sel2.copy_(self.sel2, non_blocking=True)

    def _rollback(self):
        for L in self.kda_s:
            self.kda_s[L].copy_(self.snap_s[L])
            for c in "qkv":
                self.kda_cs[L][c].copy_(self.snap_cs[L][c])

    def capture_t(self):
        t0 = time.time()
        run = self._body_t
        if os.environ.get("GLM5_COMPILE", "0") == "1":
            comp = {k: torch.compile(lambda k=k: self._body_t(k), fullgraph=True, dynamic=False)
                    for k in range(3, N_LAYER + 1)}
            run = lambda k: comp[k]()
        def body(k):
            run(k)
            if k < N_LAYER:
                self._d2h_t()
        s = torch.cuda.Stream(DEV)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for k in range(3, N_LAYER + 1):
                body(k)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for k in range(3, N_LAYER + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body(k)
            self.graphs2[k] = g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._rollback()
        self.g_rollback = g
        self.reset()
        print(f"  captured {len(self.graphs2)} two-token graphs in {time.time()-t0:.1f}s", flush=True)

    def freeze_hits_t(self):
        """Two-token hit graphs per (layer, device), sharing freeze_hits' pointer tables."""
        from qmoe import moe_masked
        H2 = {}
        for d, b1 in self.hits.items():
            z = lambda *s, dt=torch.float32: torch.zeros(*s, device=d, dtype=dt)
            b = H2[d] = {"tbl": b1["tbl"], "Hm": z(8, 2048), "g": {}}
            if d == DEV:
                b.update(x=self.x2, w=self.w2, sel=self.sel2, y=self.yr2)
            else:
                b.update(x=z(T, Ne), w=z(T, 8), sel=torch.zeros(T, 8, device=d, dtype=torch.long), y=z(T, Ne))
        def body(d, L):
            b = H2[d]; q, rb = self.qmeta[L]
            for t in range(T):
                moe_masked(b["tbl"][L][:, b["sel"][t]], *q, *rb, b["x"][t], b["w"][t], b["Hm"], b["y"][t])
        for d in H2:
            with torch.cuda.device(d):
                s = torch.cuda.Stream(d)
                s.wait_stream(torch.cuda.current_stream(d))
                with torch.cuda.stream(s):
                    for L in range(3, N_LAYER):
                        body(d, L)
                torch.cuda.current_stream(d).wait_stream(s)
                torch.cuda.synchronize(d)
                for L in range(3, N_LAYER):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=s):
                        body(d, L)
                    H2[d]["g"][L] = g
        self.hits2 = H2

    def verify(self, a, d, p):
        """Run tokens [a, d] at positions [p, p+1]; returns greedy argmax at both positions."""
        from cpu_moe import moe_cpu_numa_t
        from experts_gpu import KINDS
        self.emb2.copy_(torch.stack([self.embed(a), self.embed(d)]))
        self.pos2.copy_(torch.arange(p, p + T))
        cur = torch.cuda.current_stream()
        H2, d0 = self.hits2, self.dev0
        hx, hw, hsel, hy = self.h_x2.numpy(), self.h_w2.numpy(), self.h_sel2.numpy(), self.h_y2.numpy()
        raw = self.raw
        tl, pc = self.stats, time.perf_counter
        ta = pc()
        self.graphs2[3].replay()
        for L in range(3, N_LAYER):
            self.ev.record(cur)
            H2[DEV]["g"][L].replay()
            t0 = pc()
            self.ev.synchronize()
            t1 = pc(); tl["ms wait trunk"] += 1000 * (t1 - t0); tl["ms launch"] += 1000 * (t0 - ta)
            if d0:
                b0 = H2[d0]
                with torch.cuda.device(d0):
                    b0["x"].copy_(self.h_x2, non_blocking=True); b0["w"].copy_(self.h_w2, non_blocking=True)
                    b0["sel"].copy_(self.h_sel2, non_blocking=True)
                    b0["g"][L].replay()
                    self.ev0.record(torch.cuda.current_stream(d0))
            if self.trace2 is not None:
                self.trace2.append(hsel.copy())
            res = self.resident[L]
            uni = {}
            for t in range(T):
                for j, e in enumerate(hsel[t].tolist()):
                    if e in res:
                        raw.hits += 1
                    else:
                        raw.misses += 1
                        uni.setdefault(e, np.zeros(T, np.float32))[t] = hw[t, j]
            self.stats["cpu experts"] += len(uni)
            t2 = pc(); tl["ms host pre"] += 1000 * (t2 - t1)
            es_s = self.streamer.split(uni) if uni and self.streamer is not None else []
            if es_s:                                           # some misses to the GPU over PCIe
                ev_s = self.streamer.launch(L, es_s, uni, self.x2, self.ev)
                uni = {e: v for e, v in uni.items() if e not in es_s}
            if uni:
                es = list(uni)
                ts = [self.m.idx[f"blk.{L}.{k}"] for k in KINDS]
                q, _ = self.qmeta[L]
                t0 = time.perf_counter()
                moe_cpu_numa_t([ts[0].data[e] for e in es], [ts[1].data[e] for e in es],
                               [ts[2].data[e] for e in es], *q, hx, np.stack([uni[e] for e in es]), out=hy)
                raw.t_cpu = getattr(raw, "t_cpu", 0.0) + time.perf_counter() - t0
                self.ycpu2.copy_(self.h_y2, non_blocking=True)
            else:
                self.ycpu2.zero_()
            if es_s:
                cur.wait_event(ev_s)
                self.ycpu2.add_(self.streamer.y)
            if d0:
                cur.wait_event(self.ev0)
                self.y0c2.copy_(H2[d0]["y"], non_blocking=True)
            t3 = pc(); tl["ms cpu experts"] += 1000 * (t3 - t2)
            self.graphs2[L + 1].replay()
            ta = pc(); tl["ms post"] += 1000 * (ta - t3)
        r = self.logits2.float().argmax(-1).tolist()
        tl["ms head+argmax"] += 1000 * (pc() - ta)
        return r


def check(eng, first, n):
    """Teacher-forced equivalence (GLM5_SPEC_CHECK=n): n greedy single-token steps give the
    reference logits; then, from the same saved state, the two-token path covers the same
    positions alternating an accepted pair and a rejected step (wrong draft + rollback)."""
    st = ({L: s.clone() for L, s in eng.kda_s.items()},
          {L: {c: t.clone() for c, t in d.items()} for L, d in eng.kda_cs.items()},
          {L: t.clone() for L, t in eng.mla_kv.items()}, eng.pos)
    ref, seq, tok = [], [first], first
    eng.trace = []
    for i in range(n):
        lg = eng.step(eng.embed(tok)).float().clone()
        ref.append(lg); tok = int(lg.argmax()); seq.append(tok)
    def restore():
        for L, v in st[0].items(): eng.kda_s[L].copy_(v)
        for L, d in st[1].items():
            for c, v in d.items(): eng.kda_cs[L][c].copy_(v)
        for L, v in st[2].items(): eng.mla_kv[L].copy_(v)
    p0 = eng.pos = st[3]
    # causality / determinism: row 0 must not depend on the token in slot 1
    rows = []
    for d in (seq[1], seq[1], (seq[1] + 1) % ref[0].shape[0], 7):
        restore()
        eng.verify(seq[0], d, p0)
        rows.append(eng.logits2[0].float().clone())
    rd = lambda a, b: float((a - b).abs().max() / b.abs().max())
    print(f"  slot-1 independence: same draft {rd(rows[1], rows[0]):.1e}, other drafts {rd(rows[2], rows[0]):.1e} "
          f"{rd(rows[3], rows[0]):.1e}; vs single-token ref {rd(rows[0], ref[0]):.1e}", flush=True)
    restore()
    rep = eng.step(eng.embed(seq[0])).float().clone()
    print(f"  single-token repeat vs ref {rd(rep, ref[0]):.1e}", flush=True)
    restore(); eng.pos = p0
    nl = N_LAYER - 3
    r1 = [eng.trace[j * nl:(j + 1) * nl] for j in range(n)]          # [pos][layer] -> sel
    eng.trace, eng.trace2, r2 = None, [], {}
    got, kind, i, k = {}, {}, 0, 0
    while i < n - 1:
        if k % 2 == 0:                                        # accepted pair
            eng.verify(seq[i], seq[i + 1], p0 + i)
            got[i], got[i + 1] = eng.logits2[0].float().clone(), eng.logits2[1].float().clone()
            tr = eng.trace2[-nl:]
            r2[i], r2[i + 1] = [x[0].tolist() for x in tr], [x[1].tolist() for x in tr]
            kind[i], kind[i + 1] = "pair a", "pair d"
            i += 2
        else:                                                 # rejected: wrong draft, roll back
            eng.verify(seq[i], (seq[i + 1] + 1) % ref[0].shape[0], p0 + i)
            eng.g_rollback.replay()
            got[i] = eng.logits2[0].float().clone()
            r2[i] = [x[0].tolist() for x in eng.trace2[-nl:]]
            kind[i] = "reject"
            i += 1
        k += 1
    worst, flips = 0.0, []
    for j, g in got.items():
        r = ref[j]
        dev = float((g - r).abs().max() / r.abs().max())
        worst = max(worst, dev)
        first_diff = next((L for L in range(nl) if sorted(r1[j][L]) != sorted(r2[j][L])), None)
        nd = sum(len(set(r1[j][L]) ^ set(r2[j][L])) // 2 for L in range(nl))
        print(f"  pos {j:2d} ({kind[j]}): rel dev {dev:.1e}, "
              f"routing differs in {nd} expert slots (first at layer {None if first_diff is None else first_diff + 3})")
        if int(g.argmax()) != int(r.argmax()):
            top = torch.topk(r, 2).values
            flips.append((j, float(top[0] - top[1])))
    print(f"spec check over {len(got)} positions: max |dlogit|/max|logit| {worst:.2e}, "
          f"argmax differs at {len(flips)} (ref top-2 margins {flips})", flush=True)


def main():
    gguf_dir, sprint = sys.argv[1], sys.argv[2]
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    ids = tokenize(gguf_dir, os.environ["GLM5_PROMPT"])
    check_len(len(ids), n_gen, extra=2)                  # a verify step writes the draft's position too
    from mtp import MTP
    eng = SpecEngine(gguf_dir)
    eng.capture()
    eng.capture_t()
    mtp = MTP(eng)
    from collections import OrderedDict
    eng.raw.cache.setdefault(45, OrderedDict())             # populate + NUMA-place block 45's experts too
    print(f"prefill {len(ids)} tokens ...", flush=True)
    for j, t in enumerate(ids):
        logits = eng.step(eng.embed(t))
        nxt = ids[j + 1] if j + 1 < len(ids) else int(torch.argmax(logits.float()))
        lg = mtp.draft(eng.hn[None].float(), eng.embed(nxt)[None].float(), [eng.pos - 1])
    first = last = nxt
    draft = int(lg[0].float().argmax())
    print(f"first-gen argmax = {last}", flush=True)
    n = eng.raw.end_prefill()
    print(f"  GPU expert cache after prefill: {n} experts resident", flush=True)
    eng.freeze_hits()
    eng.freeze_hits_t()
    if os.environ.get("GLM5_STREAM", "0") == "1":
        from stream import Streamer
        eng.streamer = Streamer(eng, DEV, T)
    mtp.capture()
    if int(os.environ.get("GLM5_SPEC_CHECK", "0")) > 0:
        check(eng, first, int(os.environ["GLM5_SPEC_CHECK"]))
        return
    src = eng.raw
    h0, m0, c0 = src.hits, src.misses, getattr(src, "t_cpu", 0.0)
    out, steps, acc, t_mtp = [], 0, 0, 0.0
    pos = eng.pos                                           # position of `last` (not yet computed)
    torch.cuda.synchronize(); t0 = time.time()
    while len(out) < n_gen:
        g0, g1 = eng.verify(last, draft, pos)
        steps += 1
        tm = time.perf_counter()
        if g0 == draft:                                      # draft accepted: emit it and the bonus
            acc += 1
            out += [draft, g1]
            draft = mtp.draft_g(eng.hn2, torch.stack([eng.embed(draft), eng.embed(g1)]), pos)
            last, pos = g1, pos + 2
        else:
            eng.g_rollback.replay()
            out.append(g0)
            draft = mtp.draft_g(eng.hn2[:1], eng.embed(g0)[None], pos)
            last, pos = g0, pos + 1
        t_mtp += time.perf_counter() - tm
    torch.cuda.synchronize(); dt = time.time() - t0
    out = out[:n_gen]
    hh, mm = src.hits - h0, src.misses - m0
    print(f"decode: {len(out)} tokens in {dt:.1f}s = {len(out)/dt:.2f} tok/s  ({steps} verify steps, "
          f"draft accepted {acc}/{steps} = {acc/steps:.1%}, {len(out)/steps:.2f} tokens/step)")
    print(f"per step: {1000*dt/steps:.1f} ms (MTP draft {1000*t_mtp/steps:.1f}, cpu kernel "
          f"{1000*(getattr(src, 't_cpu', 0.0)-c0)/steps:.1f}); CPU experts {eng.stats['cpu experts']/steps:.0f}/step; "
          f"hit-rate {hh/max(1,hh+mm):.1%}" + (f"; streamed {eng.streamer.n_streamed/steps:.1f}/step" if eng.streamer else ""))
    print("timeline ms/step: " + "  ".join(f"{k[3:]} {v/steps:.1f}" for k, v in eng.stats.items() if k.startswith("ms ")))
    print(f"generated ids: {out}")
    vocab = eng.m._r[0].fields["tokenizer.ggml.tokens"].contents()
    print("text: " + repr("".join(vocab[i] for i in [first] + out).replace("Ġ", " ").replace("Ċ", "\n")))


if __name__ == "__main__":
    main()
