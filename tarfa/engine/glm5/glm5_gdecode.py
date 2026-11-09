"""
glm5-next decode with the trunk replayed as CUDA graphs (Milestone 5).

The only host decision per token is expert routing, so the trunk splits at each MoE router:
graph[L] = finish the previous MoE layer (routed + shared + hc_post) -> attention of layer L
-> ffn hc_pre/norm -> router (sel, weights) + shared expert. Graph[3] also holds the embedding
and the three dense layers; graph[45] holds the last finish and the output head. Between
graphs the host reads sel and runs RawExperts.moe (CPU misses + GPU hits, not captured).
State is static: KDA conv/recurrent buffers and a fixed-size MLA latent cache masked by the
position (unused rows are zero, so masked softmax equals the truncated one).
Math is glm5_next / glm5_decode's (validated); glm5_decode.py stays the eager reference.
Usage: python glm5_gdecode.py <gguf_dir> <out_dir> [n_gen]
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_next as G
from q8 import mv

DEV = os.environ.get("GLM5_DEV", "cuda:1")
G.DEV = DEV
G.DT = torch.float16
DT = G.DT
from glm5_next import (GLM5, Ne, H, Dh, Di, HC, rmsnorm, hc_pre, hc_post, swiglu_clamp, l2n, silu,
                       dense_ffn, _deq_np, GATE_LB, NORM_EPS)

N_LAYER = 45
MAXLEN = int(os.environ.get("GLM5_MAXLEN", "2048"))


def kda_static(x, L, cs, s):          # kda_step with in-place state (conv windows, recurrent s)
    def conv(proj, cw, c):
        win = torch.cat([c, mv(proj, x)[None, :]], 0)
        y = (win * cw).sum(0)
        c.copy_(win[1:])
        return silu(y).reshape(H, Dh)
    q = conv(L["wq"], L["q_conv"], cs["q"])
    k = conv(L["wk"], L["k_conv"], cs["k"])
    v = conv(L["wv"], L["v_conv"], cs["v"])
    sd = s.dtype
    g1 = (L["ssm_f_b"] @ (L["ssm_f_a"] @ x) + L["ssm_dt_b"]).reshape(H, Dh).to(sd)
    g1 = torch.sigmoid(-(g1 * L["ssm_a"].reshape(H, 1).to(sd))) * GATE_LB
    beta = torch.sigmoid((L["ssm_beta"] @ x).to(sd)).reshape(H, 1)
    q, k, v = q.to(sd), k.to(sd), v.to(sd)
    qn = l2n(q) / (Dh ** 0.5); kn = l2n(k)
    s2 = s * torch.exp(g1)[:, :, None]                 # decay on the KEY axis
    d = (v - torch.einsum("hab,ha->hb", s2, kn)) * beta
    s2 = s2 + kn[:, :, None] * d[:, None, :]
    s.copy_(s2)
    o = torch.einsum("hab,ha->hb", s2, qn)
    g2 = (L["ssm_g_b"] @ (L["ssm_g_a"] @ x)).reshape(H, Dh).to(sd)
    on = rmsnorm(o, L["ssm_o_norm"].to(sd), NORM_EPS)
    return mv(L["wo"], (on * torch.sigmoid(g2)).reshape(Di).to(DT))


def mla_static(xt, L, kvb, pos_t, ar):  # mla_decode_step over a fixed [MAXLEN,512] cache
    qr = rmsnorm(mv(L["wq_a"], xt), L["q_a_norm"])
    q = mv(L["wq_b"], qr).reshape(H, 256)
    kvb.index_copy_(0, pos_t, rmsnorm(mv(L["wkv_a_mqa"], xt), L["kv_a_norm"])[None, :])
    qa = torch.einsum("hd,hkd->hk", q, L["wk_b"])
    sc = (qa @ kvb.T) * (1.0 / (256 ** 0.5))
    w = torch.softmax(sc.float().masked_fill(ar > pos_t, float("-inf")), dim=1).to(DT)
    o = torch.einsum("hvk,hk->hv", L["wv_b"], w @ kvb)
    return mv(L["wo"], o.reshape(H * 256))


class Engine:
    def __init__(self, gguf_dir):
        torch.cuda.set_device(DEV)
        self.m = m = GLM5(gguf_dir)
        print(f"loading {N_LAYER} layers resident (fp16) on {DEV} ...", flush=True)
        t0 = time.time()
        self.layers = [m.layer(L) for L in range(N_LAYER)]
        self.out_norm = G._t(_deq_np(m.idx["output_norm.weight"]), (Ne,))
        if G.Q8_NAMES:                                     # head stays Q8_0; embeddings by row
            from q8 import load_q8
            ow = m.idx["output.weight"]
            self.out_w = load_q8(ow, ow.data.shape[0], Ne, DEV, DT)
            self.tok_embd = None
        else:
            self.out_w = G._t(_deq_np(m.idx["output.weight"]), (-1, Ne))
            self.tok_embd = G._t(_deq_np(m.idx["token_embd.weight"]), (-1, Ne))
        print(f"  resident in {time.time()-t0:.0f}s", flush=True)
        from experts_gpu import RawExperts
        m.raw = self.raw = RawExperts(m.idx, DEV)
        z = lambda *s, dt=DT: torch.zeros(*s, device=DEV, dtype=dt)
        SDT = torch.float32 if os.environ.get("GLM5_STATE_FP32", "1") == "1" else DT
        self.kda_s = {L: z(H, Dh, Dh, dt=SDT) for L in range(N_LAYER) if not self.layers[L]["mla"]}
        self.kda_cs = {L: {c: z(3, Di) for c in "qkv"} for L in self.kda_s}
        self.mla_kv = {L: z(MAXLEN, 512) for L in range(N_LAYER) if self.layers[L]["mla"]}
        self.pos_t = torch.zeros(1, device=DEV, dtype=torch.long)
        self.ar = torch.arange(MAXLEN, device=DEV)
        # cross-graph buffers
        self.emb = z(Ne); self.res = z(HC, Ne); self.pf = z(HC); self.cf = z(HC, HC)
        self.x32 = z(Ne, dt=torch.float32); self.w32 = z(8, dt=torch.float32)
        self.sel = torch.zeros(8, device=DEV, dtype=torch.long)
        self.sh = z(Ne, dt=torch.float32); self.yr = z(Ne, dt=torch.float32)
        self.y0c = z(Ne, dt=torch.float32); self.ycpu = z(Ne, dt=torch.float32)
        self.logits = z(self.out_w.shape[0])
        self.hn = z(Ne); self.hraw = z(Ne)                 # final hidden (post/pre output_norm) for MTP
        # pinned host mirrors: the trunk graph writes x/sel/w here for the CPU-miss path
        pin = lambda n, dt: torch.zeros(n, dtype=dt).pin_memory()
        self.h_x, self.h_w, self.h_sel = pin(Ne, torch.float32), pin(8, torch.float32), pin(8, torch.long)
        self.h_y = pin(Ne, torch.float32)
        self.graphs = {}
        self.hits = None
        self.adapt = None
        self.streamer = None                               # stream.Streamer (GLM5_STREAM=1)
        self.trace = [] if os.environ.get('GLM5_SEL_DUMP') else None
        import collections
        self.tl = collections.defaultdict(float) if os.environ.get('GLM5_TL') == '1' else None

    # ---- graph bodies ----
    def _attn(self, L, Hs):
        w = self.layers[L]
        c, pa, ca = hc_pre(Hs, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
        an = rmsnorm(c, w["attn_norm"])
        blk = (mla_static(an, w, self.mla_kv[L], self.pos_t, self.ar) if w["mla"]
               else kda_static(an, w, self.kda_cs[L], self.kda_s[L]))
        return hc_post(blk, Hs, pa, ca)

    def _ffn_pre(self, L, Hs):
        w = self.layers[L]
        c, pf, cf = hc_pre(Hs, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
        return rmsnorm(c, w["ffn_norm"]), pf, cf

    def _front(self, L, Hs):            # attention of MoE layer L up to its routing decision
        Hs = self._attn(L, Hs)
        fn, pf, cf = self._ffn_pre(L, Hs)
        w = self.layers[L]
        self.res.copy_(Hs); self.pf.copy_(pf); self.cf.copy_(cf)
        x32 = fn.float()
        probs = torch.sigmoid(w["router"].float() @ x32)
        sel = torch.topk(probs + w["exp_bias"].float(), 8).indices.flip(0)   # = argsort[-8:], no full sort
        ww = probs[sel]
        self.x32.copy_(x32); self.sel.copy_(sel); self.w32.copy_(ww / ww.sum() * 2.5)
        self.sh.copy_(mv(w["sh_down"], swiglu_clamp(mv(w["sh_gate"], fn), mv(w["sh_up"], fn))).float())

    def _d2h(self):                     # routed x/sel/w -> pinned host (kept out of torch.compile)
        self.h_x.copy_(self.x32, non_blocking=True); self.h_w.copy_(self.w32, non_blocking=True)
        self.h_sel.copy_(self.sel, non_blocking=True)

    def _finish(self):                  # close the previous MoE layer
        ff = (self.yr + self.y0c + self.ycpu + self.sh).to(DT)
        return hc_post(ff, self.res, self.pf, self.cf)

    def _body(self, k):
        if k == 3:
            Hs = self.emb[None, :].repeat(HC, 1)
            for L in range(3):
                Hs = self._attn(L, Hs)
                fn, pf, cf = self._ffn_pre(L, Hs)
                wl = self.layers[L]
                Hs = hc_post(mv(wl["ffn_down"], swiglu_clamp(mv(wl["ffn_gate"], fn), mv(wl["ffn_up"], fn))), Hs, pf, cf)
            self._front(3, Hs)
        elif k < N_LAYER:
            self._front(k, self._finish())
        else:
            Hs = self._finish()
            hm = Hs.mean(0)
            hn = rmsnorm(hm, self.out_norm)
            self.hraw.copy_(hm); self.hn.copy_(hn)
            self.logits.copy_(mv(self.out_w, hn))

    def capture(self):
        t0 = time.time()
        run = self._body
        if os.environ.get("GLM5_COMPILE", "0") == "1":      # fuse the glue ops (inductor), then capture
            torch._dynamo.config.cache_size_limit = 128
            comp = {k: torch.compile(lambda k=k: self._body(k), fullgraph=True, dynamic=False)
                    for k in range(3, N_LAYER + 1)}
            run = lambda k: comp[k]()
        def body(k):
            run(k)
            if k < N_LAYER:
                self._d2h()
        s = torch.cuda.Stream(DEV)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for k in range(3, N_LAYER + 1):
                body(k)                                        # warm-up (compiles Triton kernels)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for k in range(3, N_LAYER + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body(k)
            self.graphs[k] = g
        self.reset()
        print(f"  captured {len(self.graphs)} graphs in {time.time()-t0:.1f}s", flush=True)

    def embed(self, tok):               # token embedding [Ne] (Q8 mode: dequantize one row)
        if self.tok_embd is not None:
            return self.tok_embd[tok]
        t = self.m.idx["token_embd.weight"]
        b = np.asarray(t.data[tok]).reshape(-1, 34)               # Q8_0 blocks: fp16 d + 32 int8
        row = b[:, 2:].view(np.int8).astype(np.float32) * b[:, :2].copy().view(np.float16).astype(np.float32)
        return torch.from_numpy(row.reshape(-1)).to(DEV, DT)

    def reset(self):
        for t in self.kda_s.values(): t.zero_()
        for d in self.kda_cs.values():
            for t in d.values(): t.zero_()
        for t in self.mla_kv.values(): t.zero_()
        self.pos = 0

    def step(self, emb):                # emb [Ne] -> logits [vocab] (static buffer)
        if self.hits is not None:
            return self._step_static(emb)
        self.emb.copy_(emb)
        self.pos_t.fill_(self.pos)
        self.graphs[3].replay()
        for L in range(3, N_LAYER):
            sel = self.sel.tolist()                            # the per-layer host sync
            self.yr.copy_(self.raw.moe(L, self.x32, sel, self.w32))
            self.graphs[L + 1].replay()
        self.pos += 1
        return self.logits

    # ---- decode with a static GPU expert set: hits run inside graphs on both GPUs ----
    def freeze_hits(self):
        """After prefill: per (layer, device) pointer tables of the resident experts, and one
        graph per (layer, device) computing that device's hits from the routed sel on-device."""
        from qmoe import moe_masked
        from experts_gpu import KINDS
        from gguf import GGMLQuantizationType as QT
        raw = self.raw
        devs = list(raw.cap)
        self.dev0 = next((d for d in devs if d != DEV), None)
        self.resident = {L: set(raw.cache.get(L, {})) for L in range(3, N_LAYER)}
        t0 = time.time()
        H = {}
        for d in devs:
            z = lambda *s, dt=torch.float32: torch.zeros(*s, device=d, dtype=dt)
            b = H[d] = {"tbl": {}, "Hm": z(8, 2048), "g": {}}
            if d == DEV:
                b.update(x=self.x32, w=self.w32, sel=self.sel, y=self.yr)
            else:
                b.update(x=z(Ne), w=z(8), sel=torch.zeros(8, device=d, dtype=torch.long), y=z(Ne))
            for L in range(3, N_LAYER):
                tb = torch.zeros(3, 288, dtype=torch.long)
                for e, (trip, dd) in raw.cache.get(L, {}).items():
                    if dd == d:
                        tb[:, e] = torch.tensor([t.data_ptr() for t in trip])
                b["tbl"][L] = tb.to(d)
        self.qmeta = {}
        for L in range(3, N_LAYER):
            ts = [self.m.idx[f"blk.{L}.{k}"] for k in KINDS]
            self.qmeta[L] = ([QT(t.tensor_type) for t in ts], [t.data.shape[2] for t in ts])
        def body(d, L):
            b = H[d]; q, rb = self.qmeta[L]
            P = b["tbl"][L][:, b["sel"]]
            moe_masked(P, *q, *rb, b["x"], b["w"], b["Hm"], b["y"])
        for d in devs:
            with torch.cuda.device(d):
                s = torch.cuda.Stream(d)
                s.wait_stream(torch.cuda.current_stream(d))
                with torch.cuda.stream(s):
                    for L in range(3, N_LAYER):
                        body(d, L)                             # compile + warm
                torch.cuda.current_stream(d).wait_stream(s)
                torch.cuda.synchronize(d)
                for L in range(3, N_LAYER):
                    g = torch.cuda.CUDAGraph()
                    # explicit stream: torch.cuda.graph's default capture stream is a class-level
                    # stream created on whichever device captured first (would capture nothing)
                    with torch.cuda.graph(g, stream=s):
                        body(d, L)
                    H[d]["g"][L] = g
        self.hits = H
        if os.environ.get("GLM5_CHECK", "0") == "1":
            self._check_hits()
        self.ev = torch.cuda.Event()
        self.ev0 = torch.cuda.Event() if self.dev0 else None
        self.y0c.zero_(); self.ycpu.zero_()
        n = sum(len(v) for v in self.resident.values())
        print(f"  froze {n} resident experts; captured {len(devs)*(N_LAYER-3)} hit graphs "
              f"in {time.time()-t0:.1f}s", flush=True)

    def _check_hits(self):
        """Hit graphs vs the eager RawExperts path on the same resident experts."""
        raw, H = self.raw, self.hits
        from experts_gpu import KINDS
        for L in (3, 20, 44):
            res = sorted(raw.cache[L].items())
            sel = [e for e, _ in res[:8]] + [e for e in range(288) if e not in raw.cache[L]][:max(0, 8 - len(res))]
            self.x32.copy_(torch.randn(Ne, device=DEV) * 0.05)
            self.w32.copy_(torch.rand(8, device=DEV))
            self.sel.copy_(torch.tensor(sel, device=DEV))
            q, _ = self.qmeta[L]
            msg = []
            for d, b in H.items():
                ref = torch.zeros(Ne, device=DEV)
                raw._gpu([(j, raw.cache[L][e]) for j, e in enumerate(sel)
                          if e in raw.cache[L] and raw.cache[L][e][1] == d], self.x32, self.w32, q, ref)
                if d != DEV:
                    b["x"].copy_(self.x32); b["w"].copy_(self.w32); b["sel"].copy_(self.sel)
                b["g"][L].replay(); torch.cuda.synchronize(d)
                got = b["y"].to(DEV)
                msg.append(f"{d}: rel.err {((got - ref).norm() / (ref.norm() + 1e-30)).item():.2e} |ref| {ref.norm().item():.3f} |got| {got.norm().item():.3f}")
            print(f"  check L{L}: resident in sel {sum(e in raw.cache[L] for e in sel)}/8  " + "  ".join(msg), flush=True)

    def _step_static(self, emb):
        from cpu_moe import moe_cpu
        from experts_gpu import KINDS
        TL = self.tl is not None
        pc = time.perf_counter
        self.emb.copy_(emb)
        self.pos_t.fill_(self.pos)
        cur = torch.cuda.current_stream()
        H, d0 = self.hits, self.dev0
        hx, hw, hsel, hy = self.h_x.numpy(), self.h_w.numpy(), self.h_sel.numpy(), self.h_y.numpy()
        raw = self.raw
        evs = []
        if self.adapt is not None:
            self.adapt.tick()
        self.graphs[3].replay()
        for L in range(3, N_LAYER):
            self.ev.record(cur)
            if TL:
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]; e[0].record(cur)
            H[DEV]["g"][L].replay()                            # hits on the trunk GPU, queued
            if TL:
                e[1].record(cur); t0 = pc()
            self.ev.synchronize()                              # trunk + router done; x/sel/w pinned
            if TL:
                t1 = pc(); self.tl["wait"] += t1 - t0
            if d0:
                b0 = H[d0]
                with torch.cuda.device(d0):
                    s0 = torch.cuda.current_stream(d0)
                    b0["x"].copy_(self.h_x, non_blocking=True); b0["w"].copy_(self.h_w, non_blocking=True)
                    b0["sel"].copy_(self.h_sel, non_blocking=True)
                    b0["g"][L].replay()
                    self.ev0.record(s0)
            sel = hsel.tolist()
            if self.trace is not None:
                self.trace.append(sel)
            if self.adapt is not None:
                self.adapt.observe(L, sel)
            res = self.resident[L]
            miss = [j for j, e_ in enumerate(sel) if e_ not in res]
            raw.hits += 8 - len(miss); raw.misses += len(miss)
            if TL:
                t2 = pc(); self.tl["pre"] += t2 - t1
            es_s = []
            if miss and self.streamer is not None:              # some misses to the GPU over PCIe
                uni = {sel[j]: np.array([hw[j]], np.float32) for j in miss}
                es_s = self.streamer.split(uni)
                if es_s:
                    ev_s = self.streamer.launch(L, es_s, uni, self.x32[None], self.ev)
                    miss = [j for j in miss if sel[j] not in es_s]
            if miss:
                ts = [self.m.idx[f"blk.{L}.{k}"] for k in KINDS]
                q, _ = self.qmeta[L]
                t0 = time.perf_counter()
                moe_cpu([ts[0].data[sel[j]] for j in miss], [ts[1].data[sel[j]] for j in miss],
                        [ts[2].data[sel[j]] for j in miss], *q, hx, hw[miss], out=hy)
                raw.t_cpu = getattr(raw, "t_cpu", 0.0) + time.perf_counter() - t0
                self.ycpu.copy_(self.h_y, non_blocking=True)
            else:
                self.ycpu.zero_()
            if es_s:
                cur.wait_event(ev_s)
                self.ycpu.add_(self.streamer.y[0])
            if TL:
                t3 = pc(); self.tl["cpu+call"] += t3 - t2
            if d0:
                cur.wait_event(self.ev0)
                self.y0c.copy_(H[d0]["y"], non_blocking=True)
            if TL:
                e[2].record(cur); evs.append(e)
            self.graphs[L + 1].replay()
            if TL:
                self.tl["post"] += pc() - t3
        self.pos += 1
        if TL:
            torch.cuda.synchronize()
            for i, e in enumerate(evs):
                self.tl["gpu hits1"] += e[0].elapsed_time(e[1]) / 1000
                self.tl["gpu waitcpu+y0"] += e[1].elapsed_time(e[2]) / 1000
                nxt = evs[i + 1][0] if i + 1 < len(evs) else None
                if nxt is not None:
                    self.tl["gpu trunk"] += e[2].elapsed_time(nxt) / 1000
            self.tl["steps"] += 1
        return self.logits


def bench_cpu_inproc(eng):
    """CPU expert calls (5 random experts of a random layer) inside the engine process, for
    several OpenMP pinnings; per-thread CPU ticks show whether the team actually runs."""
    import random, glob as _g
    import cpu_moe
    from experts_gpu import KINDS
    random.seed(1)
    calls = [(L, random.sample(range(288), 5)) for L in [random.randrange(3, 45) for _ in range(60)]]
    x = np.random.randn(Ne).astype(np.float32) * 0.05; w = np.ones(5, np.float32) / 5
    def ticks():
        out = {}
        for st in _g.glob("/proc/self/task/*/stat"):
            f = open(st).read().rsplit(")", 1)[1].split()
            out[st.split("/")[4]] = (int(f[11]) + int(f[12]), int(f[36]))
        return out
    def run():
        tt = 0.0
        for i, (L, es) in enumerate(calls):
            ts = [eng.m.idx[f"blk.{L}.{k}"] for k in KINDS]
            t0 = time.perf_counter()
            cpu_moe.moe_cpu([ts[0].data[e] for e in es], [ts[1].data[e] for e in es], [ts[2].data[e] for e in es],
                            ts[0].tensor_type, ts[1].tensor_type, ts[2].tensor_type, x, w)
            if i >= 5: tt += time.perf_counter() - t0
        return 1000 * tt / 55 / 5
    run()                                                    # map pages once
    for pins in (list(range(8)), list(range(16)), list(range(12)), [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]):
        team, bad = cpu_moe.pin(pins)
        t0 = ticks(); w0 = time.time(); ms = run(); el = time.time() - w0; t1 = ticks()
        busy = sorted(((t1[k][0] - t0[k][0], t1[k][1]) for k in t1 if k in t0), reverse=True)[:18]
        print(f"pin {len(pins)} threads (team {team}, bad {bad}): {ms:.3f} ms/expert; busiest threads "
              f"(ticks/{el:.1f}s @cpu): {busy}", flush=True)


def tokenize(gguf_dir, text):
    import subprocess
    shards = sorted(f for f in os.listdir(gguf_dir) if f.endswith(".gguf"))
    if not shards:
        sys.exit(f"no .gguf files in {gguf_dir}")
    tok_bin = os.environ.get("LLAMA_TOKENIZE", "llama-tokenize")   # llama.cpp build with glm5-next support
    try:
        r = subprocess.run([tok_bin, "-m", os.path.join(gguf_dir, shards[0]), "-p", text, "--ids", "--log-disable"],
                           capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    except FileNotFoundError:
        sys.exit(f"tokenizer not found: {tok_bin} (set LLAMA_TOKENIZE)")
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        sys.exit(f"{tok_bin} failed (exit {r.returncode}): {r.stderr.strip()[-500:]}")
    ids = [int(v) for v in lines[-1].strip("[]").split(",") if v.strip()]
    if not ids:
        sys.exit(f"{tok_bin} returned no token ids")
    return ids


def check_len(n_prompt, n_gen, extra=1):
    """The MLA caches hold MAXLEN positions and the engine attends to every key (no sparse indexer):
    exact for up to 2048 keys, where llama.cpp's indexer (top_k 2048) also keeps them all."""
    need = n_prompt + n_gen + extra
    if need > MAXLEN:
        sys.exit(f"prompt {n_prompt} + {n_gen} generated tokens needs {need} positions > GLM5_MAXLEN {MAXLEN}")
    if MAXLEN > 2048:
        print("warning: beyond 2048 keys llama.cpp's sparse indexer prunes; this engine attends to all keys",
              file=sys.stderr)


def main():
    gguf_dir, sprint = sys.argv[1], sys.argv[2]
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    from probe_layer0 import read_dump
    if os.environ.get("GLM5_PROMPT"):                   # any text, tokenized by llama.cpp's tokenizer
        ids = tokenize(gguf_dir, os.environ["GLM5_PROMPT"])
        check_len(len(ids), n_gen)
    eng = Engine(gguf_dir)
    eng.capture()
    if os.environ.get("GLM5_PROMPT"):
        print(f"prefill {len(ids)} tokens (GLM5_PROMPT) ...", flush=True)
        mtp = None
        if os.environ.get("GLM5_MTP") == "measure":
            from mtp import MTP
            mtp = MTP(eng); mtp_stats = [0, 0]; draft = None
        for j, t in enumerate(ids):
            logits = eng.step(eng.embed(t))
            if mtp is not None:
                nxt = ids[j + 1] if j + 1 < len(ids) else int(torch.argmax(logits.float()))
                if draft is not None:
                    mtp_stats[0] += draft == nxt; mtp_stats[1] += 1
                h = eng.hn if mtp.hsrc == "norm" else eng.hraw
                draft = int(torch.argmax(mtp.draft(h[None].float(), eng.embed(nxt)[None].float(), [eng.pos - 1])[0].float()))
            if mtp is not None and j == len(ids) - 1:
                print(f"  MTP on the prompt: accepted {mtp_stats[0]}/{mtp_stats[1]}", flush=True)
                mtp_stats = [0, 0]
    else:
        raw, _ = read_dump(os.path.join(sprint, "dumps2"), "hc_init")
        T = raw.size // (HC * Ne)
        hc = torch.from_numpy(raw[:T * HC * Ne].reshape(T, HC, Ne).copy()).to(DEV, DT)
        print(f"prefill {T} tokens ...", flush=True)
        for t in range(T):
            logits = eng.step(hc[t, 0])
    tok = first = int(torch.argmax(logits.float()))
    if os.environ.get("GLM5_TOPK_LOG") == "1":
        pv, pi = torch.topk(torch.softmax(logits.float(), 0), 5)
        print("  step 1: " + "  ".join(f"{int(a)}:{float(b):.3f}" for a, b in zip(pi, pv)), flush=True)
    if os.environ.get("GLM5_PROMPT"):
        print(f"first-gen argmax = {tok}", flush=True)
    else:                                               # the oracle dump's prompt, "The capital of France is"
        print(f"first-gen argmax = {tok}  (oracle llama.cpp = 12089; match={tok == 12089})", flush=True)
    n = eng.raw.end_prefill()
    print(f"  GPU expert cache after prefill: {n} experts resident", flush=True)
    if os.environ.get("GLM5_HIT_GRAPHS", "1") == "1":
        eng.freeze_hits()
        if os.environ.get("GLM5_STREAM", "0") == "1":
            from stream import Streamer
            eng.streamer = Streamer(eng, DEV, 1)
        if int(os.environ.get("GLM5_ADAPT", "0")) > 0:
            from adapt import Adapter
            eng.adapt = Adapter(eng, int(os.environ["GLM5_ADAPT"]))
    src = eng.raw
    h0, m0, c0 = src.hits, src.misses, getattr(src, "t_cpu", 0.0)
    toks = []
    import resource
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    torch.cuda.synchronize(); t0 = time.time()
    mtp = locals().get("mtp")
    for i in range(n_gen):
        lg = eng.step(eng.embed(tok)).float()
        if mtp is not None:
            t2 = int(torch.argmax(lg))
            mtp_stats[0] += draft == t2; mtp_stats[1] += 1
            h = eng.hn if mtp.hsrc == "norm" else eng.hraw
            draft = int(torch.argmax(mtp.draft(h[None].float(), eng.embed(t2)[None].float(), [eng.pos - 1])[0].float()))
        if os.environ.get("GLM5_TOPK_LOG") == "1" and i < 6:
            pv, pi = torch.topk(torch.softmax(lg, 0), 5)
            print(f"  step {i+2}: " + "  ".join(f"{int(a)}:{float(b):.3f}" for a, b in zip(pi, pv)), flush=True)
        tok = int(torch.argmax(lg))
        toks.append(tok)
    torch.cuda.synchronize(); dt = time.time() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    print(f"faults during decode: major {ru1.ru_majflt-ru0.ru_majflt}  minor {ru1.ru_minflt-ru0.ru_minflt}")
    hh, mm = src.hits - h0, src.misses - m0
    print(f"decode: {n_gen} tokens in {dt:.1f}s = {n_gen/dt:.2f} tok/s")
    print(f"experts during decode: hits {hh}  misses {mm}  hit-rate {hh/max(1,hh+mm):.1%}  "
          f"cpu kernel {getattr(src, 't_cpu', 0.0)-c0:.1f}s of {dt:.1f}s"
          + (f"  streamed over PCIe {eng.streamer.n_streamed}" if eng.streamer else ""))
    if eng.tl:
        n = eng.tl.pop("steps")
        print("timeline ms/token: " + "  ".join(f"{k} {1000*v/n:.1f}" for k, v in eng.tl.items()))
    if eng.trace is not None:
        res = np.zeros((N_LAYER, 288), np.int8)
        for L, s in eng.resident.items():
            res[L, list(s)] = 1
        np.savez(os.environ["GLM5_SEL_DUMP"], sel=np.array(eng.trace, np.int16).reshape(n_gen, N_LAYER - 3, 8),
                 resident=res)
        print(f"routing trace -> {os.environ['GLM5_SEL_DUMP']}")
    if eng.adapt is not None and os.environ.get("GLM5_ADAPT_DEBUG"):
        import sys as _s, traceback, threading as _th
        for th in _th.enumerate():
            if th.name == "adapt-worker":
                print("adapt worker stack:\n" + "".join(traceback.format_stack(_s._current_frames()[th.ident])), flush=True)
        print("swap stages:", [(s["L"], s["stage"]) for s in eng.adapt.swaps])
    if eng.adapt is not None:
        print(f"adapt: {eng.adapt.done} swaps completed, {len(eng.adapt.swaps)} in flight, "
              f"worker busy {eng.adapt.t_worker:.1f}s")
    if mtp is not None:
        print(f"MTP acceptance during decode: {mtp_stats[0]}/{mtp_stats[1]} = {mtp_stats[0]/max(1,mtp_stats[1]):.1%} "
              f"(order {mtp.order}, h {mtp.hsrc})")
    print(f"generated ids: {toks}")
    if os.environ.get("GLM5_BENCH_CPU") == "1":
        bench_cpu_inproc(eng)
    vocab = eng.m._r[0].fields["tokenizer.ggml.tokens"].contents()
    print("text: " + repr("".join(vocab[i] for i in [first] + toks).replace("Ġ", " ").replace("Ċ", "\n")))


if __name__ == "__main__":
    main()
