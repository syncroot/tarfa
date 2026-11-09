"""
glm5-next M1 decode loop + first tok/s baseline. UNOPTIMIZED: resident non-expert weights
(fp16) on one card, experts dequantized on-demand per routed expert each token. No CUDA
graphs, no expert residency yet (milestones 3-5). Purpose: an end-to-end generating engine
and an honest first tok/s number vs llama.cpp's ~5.

State per layer: KDA {conv [3,Di], s [H,Dh,Dh]}; MLA {kv cache [n,512]}. mHC is per-token.
Prefill uses the AR KDA path (validated as the faithful decode recurrence); the argmax of
the prompt's last token should match llama.cpp (12089 for "The capital of France is").
"""
import os, sys, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_next as G
from glm5_next import (GLM5, Ne, H, Dh, Di, HC, rmsnorm, hc_pre, hc_post, kda_step,
                       dense_ffn, mla_forward_t, moe_t, _deq_np, GATE_LB, NORM_EPS, L2_EPS)
import gguf
from gguf import GGMLQuantizationType

DEV = os.environ.get("GLM5_DEV", "cuda:1")   # Quadro (24 GB) by default
G.DEV = DEV
G.DT = torch.float16                           # resident weights fp16 to fit
DT = G.DT
N_LAYER = 45


def mla_decode_step(xt, L, kv):   # xt [Ne]; kv [n,512] grown in place (list). returns out [Ne], appends to kv
    qr = rmsnorm((L["wq_a"] @ xt), L["q_a_norm"])
    q = (L["wq_b"] @ qr).reshape(H, 256)
    kvn = rmsnorm((L["wkv_a_mqa"] @ xt), L["kv_a_norm"])           # [512]
    kv.append(kvn)
    Kc = torch.stack(kv)                                           # [n, 512]
    qa = torch.einsum("hd,hkd->hk", q, L["wk_b"])                 # [H,512]
    sc = torch.einsum("hk,nk->hn", qa, Kc) * (1.0 / (256 ** 0.5))
    w = torch.softmax(sc.float(), dim=1).to(DT)
    ctx = w @ Kc                                                   # [H,512]
    o = torch.einsum("hvk,hk->hv", L["wv_b"], ctx)               # [H,256]
    return L["wo"] @ o.reshape(H * 256)


def main():
    gguf_dir, sprint = sys.argv[1], sys.argv[2]
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    from probe_layer0 import read_dump
    torch.cuda.set_device(DEV)
    m = GLM5(gguf_dir)

    print(f"loading {N_LAYER} layers resident (fp16) on {DEV} ...", flush=True)
    t0 = time.time()
    layers = [m.layer(L) for L in range(N_LAYER)]
    out_norm = G._t(_deq_np(m.idx["output_norm.weight"]), (Ne,))
    out_w = G._t(_deq_np(m.idx["output.weight"]), (-1, Ne))        # [vocab, Ne]
    tok_embd = G._t(_deq_np(m.idx["token_embd.weight"]), (-1, Ne)) # [vocab, Ne]
    vocab = out_w.shape[0]
    print(f"  resident in {time.time()-t0:.0f}s  (vocab {vocab})", flush=True)
    if os.environ.get("GLM5_FUSED", "1") == "1":
        from experts_gpu import RawExperts
        m.raw = RawExperts(m.idx, DEV)
        print(f"  fused raw-k-quant experts: per-layer cache {m.raw.per_layer}, "
              f"threads {m.raw.pool._max_workers}", flush=True)

    # state
    SDT = torch.float32 if os.environ.get("GLM5_STATE_FP32", "1") == "1" else DT
    kda_s = [(None, torch.zeros(H, Dh, Dh, device=DEV, dtype=SDT)) for _ in range(N_LAYER)]
    kda_cs = [{c: torch.zeros(3, Di, device=DEV, dtype=DT) for c in "qkv"} for _ in range(N_LAYER)]
    mla_kv = [[] for _ in range(N_LAYER)]

    def step(emb):                       # emb [Ne] single token embedding -> logits [vocab]
        Hs = emb[None, :].repeat(HC, 1)  # [4, Ne]
        for L in range(N_LAYER):
            w = layers[L]
            res = Hs
            c, pa, ca = hc_pre(Hs, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
            an = rmsnorm(c, w["attn_norm"])
            if w["mla"]:
                blk = mla_decode_step(an, w, mla_kv[L])
            else:
                cs = kda_cs[L]; s = kda_s[L][1]
                blk, newcs, s = kda_step(an, w, cs, s)
                kda_cs[L] = newcs; kda_s[L] = (None, s)
            Hs = hc_post(blk, res, pa, ca)
            res = Hs
            c, pf, cf = hc_pre(Hs, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
            fn = rmsnorm(c, w["ffn_norm"])
            ff = moe_t(fn, w, m) if w["moe"] else dense_ffn(fn, w)
            Hs = hc_post(ff, res, pf, cf)
        h = rmsnorm(Hs.mean(0), out_norm)
        return out_w @ h

    # prefill from oracle hc_init embeddings (stream 0 = the raw embedding)
    raw, _ = read_dump(os.path.join(sprint, "dumps2"), "hc_init")
    T = raw.size // (HC * Ne)
    hc = torch.from_numpy(raw[:T * HC * Ne].reshape(T, HC, Ne).copy()).to(DEV, DT)
    print(f"prefill {T} tokens ...", flush=True)
    logits = None
    for t in range(T):
        logits = step(hc[t, 0])          # stream 0 == the token embedding
    tok = int(torch.argmax(logits.float()))
    first = tok
    print(f"first-gen argmax = {tok}  (oracle llama.cpp = 12089; match={tok == 12089})", flush=True)
    if getattr(m, "raw", None) is not None:
        n = m.raw.end_prefill()
        print(f"  GPU expert cache after prefill: {n} experts resident (warm={m.raw.warm})", flush=True)

    # decode
    print(f"decoding {n_gen} tokens ...", flush=True)
    src = getattr(m, "raw", None) or m
    src.hits = getattr(src, "hits", 0); src.misses = getattr(src, "misses", 0)
    h0, m0 = src.hits, src.misses
    f0 = getattr(src, "t_fetch", 0.0)
    toks = []
    temp = float(os.environ.get("GLM5_TEMP", "0"))           # 0 = greedy (loops on this prompt)
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    def pick(lg):
        if temp <= 0:
            return int(torch.argmax(lg.float()))
        v, ix = torch.topk(lg.float() / temp, 40)
        return int(ix[torch.multinomial(torch.softmax(v, 0), 1, generator=gen)])
    torch.cuda.synchronize(); t0 = time.time()
    if hasattr(src, "t_cpu"): src.t_cpu = 0.0
    for i in range(n_gen):
        emb = tok_embd[tok]
        logits = step(emb)
        if os.environ.get("GLM5_TOPK_LOG") == "1" and i < 6:
            pr = torch.softmax(logits.float(), 0); pv, pi = torch.topk(pr, 5)
            print(f"  step {i+2}: " + "  ".join(f"{int(a)}:{float(b):.3f}" for a, b in zip(pi, pv)), flush=True)
        tok = pick(logits); toks.append(tok)
        if i == 0:
            torch.cuda.synchronize(); print(f"  first decode token {time.time()-t0:.1f}s", flush=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    hh, mm = src.hits - h0, src.misses - m0
    tf = getattr(src, "t_fetch", 0.0) - f0
    print(f"cpu kernel time (all steps incl. prefill): {getattr(src, 't_cpu', 0.0):.1f}s")
    print(f"decode: {n_gen} tokens in {dt:.1f}s = {n_gen/dt:.2f} tok/s  (llama.cpp ~5 tok/s)")
    print(f"experts during decode: hits {hh}  misses {mm}  hit-rate {hh/max(1,hh+mm):.1%}  "
          f"fetch {tf:.1f}s of {dt:.1f}s ({tf/dt:.0%})")
    print(f"generated ids: {toks}")
    try:
        vocab = m._r[0].fields["tokenizer.ggml.tokens"].contents()
        txt = "".join(vocab[i] for i in [first] + toks).replace("Ġ", " ").replace("Ċ", "\n")
        print(f"text: {txt!r}")
    except Exception as ex:
        print(f"(detok failed: {ex})")
    dump = os.environ.get("GLM5_ROUTE_DUMP")
    if dump and getattr(m, "raw", None) is not None:
        np.save(dump, np.stack([m.raw.route.get(L, np.zeros(288, np.int64)) for L in range(N_LAYER)]))
        print(f"routing histogram -> {dump}")


if __name__ == "__main__":
    main()
