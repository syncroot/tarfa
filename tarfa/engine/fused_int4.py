# -*- coding: utf-8 -*-
"""Tarfa fast path - fused int4 kernel + PIPELINED expert prefetch.
While layer L's kernel/attention runs, a side CUDA stream prefetches layer L+1's PREDICTED
experts (= what L+1 used last token). Hits are already on-GPU at L+1 (their H2D overlapped
L's compute); misses stream in. Deterministic: the experts/values are identical, only the
*timing* of the transfer changes. """
import os, re, types, time, threading, torch
from collections import OrderedDict
from collections import deque
import torch.nn.functional as F
import channel_q4 as cq
from channel_q4 import (ChannelModelQ4, dequant_int4, _read_experts, _q4fd, _q4meta,
                        _POOL, _NAMES, TOPK_EXPERTS)
from moe_kernel import moe_ffn

PIPE = os.environ.get("PIPE", "0") != "0"   # prefetch is a net-loss on a bandwidth-bound bus (proven: 2.14 vs 2.98); off by default
JTOPK = int(os.environ.get("JTOPK", "8"))   # QUALITY: native routing is top-8; 8 = no truncation (was 5 for speed)
OVERLAP = os.environ.get("OVERLAP", "0") == "1"   # Lever A: net-LOSS (chunking kills NVMe queue depth: 2.00->1.44->1.01). Off; dormant.
OCHUNK = int(os.environ.get("OCHUNK", "2"))       # experts per pipeline chunk (finer = more overlap, lower NVMe queue depth)
BF16_PREFILL_BATCH = max(1, int(os.environ.get("BF16_PREFILL_BATCH", "16")))
BF16_CACHE_PER_LAYER = max(0, int(os.environ.get("BF16_CACHE_PER_LAYER", "0")))
BF16_BATCHED_DECODE = os.environ.get("BF16_BATCHED_DECODE", "0") == "1"

_OPIN = [None, None]; _OEV = [None, None]         # double-buffered pinned + completion events for the overlap pipeline
def _opin(meta, idx, chunk):
    if _OPIN[idx] is None:
        def mk(name):
            dt = torch.uint8 if name.endswith("_q") else torch.float16
            t = torch.empty((chunk,) + meta[name][2], dtype=dt, pin_memory=True)
            return t, memoryview(t.numpy()).cast("B")
        _OPIN[idx] = {nm: mk(nm) for nm in _NAMES}
    return _OPIN[idx]

def _load_overlap(path, experts, device, dtype, stream, chunk=None):
    """pread chunk k+1 (NVMe, CPU) concurrently with the async H2D of chunk k (PCIe, GPU). Same data, deterministic."""
    if not experts: return {}
    chunk = chunk or OCHUNK; fd = _q4fd(path); meta = _q4meta(path)
    if _OEV[0] is None:
        _OEV[0] = torch.cuda.Event(); _OEV[1] = torch.cuda.Event(); _OEV[0].record(); _OEV[1].record()
    batches = [experts[i:i+chunk] for i in range(0, len(experts), chunk)]; res = {}
    def pread_into(bi, b):
        pins = _opin(meta, bi, chunk); reqs = []
        for j, e in enumerate(b):
            for nm in _NAMES:
                st, rb, shp, dt = meta[nm]; reqs.append((fd, pins[nm][1][j*rb:(j+1)*rb], st + e*rb))
        for _ in _POOL.map(lambda r: os.preadv(r[0], [r[1]], r[2]), reqs): pass
    def h2d_from(bi, b):
        pins = _opin(meta, bi, chunk); E = len(b)
        with torch.cuda.stream(stream):
            gq = pins["gate_up_q"][0][:E].to(device, non_blocking=True)
            gs = pins["gate_up_s"][0][:E].to(device, non_blocking=True).to(dtype)
            dq = pins["down_q"][0][:E].to(device, non_blocking=True)
            ds = pins["down_s"][0][:E].to(device, non_blocking=True).to(dtype)
            for j, e in enumerate(b): res[e] = (gq[j].clone(), gs[j].clone(), dq[j].clone(), ds[j].clone())
            _OEV[bi].record(stream)
    _OEV[0].synchronize(); pread_into(0, batches[0])           # first chunk exposed; rest pipelined
    for k in range(len(batches)):
        h2d_from(k % 2, batches[k])                            # async PCIe transfer of chunk k
        if k + 1 < len(batches):
            nbi = (k + 1) % 2; _OEV[nbi].synchronize()         # ensure that pinned half drained
            pread_into(nbi, batches[k + 1])                    # CPU NVMe read of chunk k+1 -- overlaps chunk k's H2D
    return res
BF16 = os.environ.get("BF16", "0") == "1"   # QUALITY: stream full-precision bf16 experts from original shards (zero quant loss, ~8x bytes)
if BF16:
    from exact_bf16 import read_bf16
_TLOCK = threading.Lock()
_TSTAT = {"moe_calls": 0, "moe_seconds": 0.0, "prefill_moe_calls": 0,
          "prefill_moe_seconds": 0.0, "decode_moe_calls": 0,
          "decode_moe_seconds": 0.0, "bf16_cache_hits": 0,
          "bf16_cache_misses": 0}
_ROUTE_HISTORY = {i: deque(maxlen=16) for i in range(48)}
_ROUTE = {"decode_steps": 0, "expert_uses": 0, "unique_layer_experts": set(),
          "reuse_1": 0, "reuse_2": 0, "reuse_4": 0, "reuse_8": 0, "reuse_16": 0}

def _record_route(layer, experts):
    hist = _ROUTE_HISTORY[layer]; current = set(experts)
    with _TLOCK:
        _ROUTE["decode_steps"] += 1; _ROUTE["expert_uses"] += len(experts)
        _ROUTE["unique_layer_experts"].update((layer, e) for e in current)
        previous = list(hist)
        for window in (1, 2, 4, 8, 16):
            seen = set().union(*previous[-window:]) if previous else set()
            _ROUTE[f"reuse_{window}"] += len(current & seen)
        hist.append(current)

def routing_telemetry(reset=False):
    with _TLOCK:
        uses = _ROUTE["expert_uses"]
        out = {k: (len(v) if k == "unique_layer_experts" else v) for k, v in _ROUTE.items()}
        out["reuse_rates"] = {str(w): round(_ROUTE[f"reuse_{w}"] / max(uses, 1), 4) for w in (1,2,4,8,16)}
        if reset:
            for h in _ROUTE_HISTORY.values(): h.clear()
            _ROUTE.update(decode_steps=0, expert_uses=0, unique_layer_experts=set(),
                          reuse_1=0, reuse_2=0, reuse_4=0, reuse_8=0, reuse_16=0)
    return out

def telemetry(reset=False):
    with _TLOCK:
        out = dict(_TSTAT)
        if reset:
            for key in _TSTAT:
                _TSTAT[key] = 0 if key.endswith(("calls", "hits", "misses")) else 0.0
    return out

def _record_moe(kind, elapsed):
    with _TLOCK:
        _TSTAT["moe_calls"] += 1; _TSTAT["moe_seconds"] += elapsed
        _TSTAT[f"{kind}_moe_calls"] += 1
        _TSTAT[f"{kind}_moe_seconds"] += elapsed
_PSTREAM = None
def _pstream():
    global _PSTREAM
    if _PSTREAM is None: _PSTREAM = torch.cuda.Stream()
    return _PSTREAM

_PINS = [None, None]   # [0]=miss path, [1]=prefetch path  (separate pinned buffers)
def _pins(meta, which, maxE=8):
    if _PINS[which] is None:
        def mk(name):
            dt = torch.uint8 if name.endswith("_q") else torch.float16
            t = torch.empty((maxE,) + meta[name][2], dtype=dt, pin_memory=True)
            return t, memoryview(t.numpy()).cast("B")
        _PINS[which] = {nm: mk(nm) for nm in _NAMES}
    return _PINS[which]

def _load(path, experts, device, dtype, which, stream):
    """pread `experts` into pinned[which], DMA on `stream`, return {e:(gq,gs,dq,ds)} independent GPU tensors."""
    if not experts: return {}
    fd = _q4fd(path); meta = _q4meta(path); pins = _pins(meta, which); E = len(experts)
    reqs = []
    for j, e in enumerate(experts):
        for nm in _NAMES:
            st, rb, shp, dt = meta[nm]
            reqs.append((fd, pins[nm][1][j*rb:(j+1)*rb], st + e*rb))
    for _ in _POOL.map(lambda r: os.preadv(r[0], [r[1]], r[2]), reqs): pass
    res = {}
    with torch.cuda.stream(stream):
        gq = pins["gate_up_q"][0][:E].to(device, non_blocking=True)
        gs = pins["gate_up_s"][0][:E].to(device, non_blocking=True).to(dtype)
        dq = pins["down_q"][0][:E].to(device, non_blocking=True)
        ds = pins["down_s"][0][:E].to(device, non_blocking=True).to(dtype)
        for j, e in enumerate(experts):
            res[e] = (gq[j].clone(), gs[j].clone(), dq[j].clone(), ds[j].clone())   # independent of pinned reuse
    return res


class TarfaFused(ChannelModelQ4):
    def _install_streamed_experts(self, experts, pre):
        i = int(re.search(r"layers\.(\d+)\.", pre).group(1))
        q4f = f"{cq.Q4DIR}/layer_{i}.safetensors"
        q4f_next = f"{cq.Q4DIR}/layer_{i+1}.safetensors"
        eng = self; layer_i = i
        if not hasattr(eng, "_predict"): eng._predict = {}; eng._pf = {}; eng._pf_stat = [0, 0]

        def forward(self, hidden_states, top_k_index, top_k_weights):
            n = self.num_experts
            if top_k_weights.shape[-1] > JTOPK:
                tv, ti = top_k_weights.topk(JTOPK, dim=-1)
                top_k_index = torch.gather(top_k_index, -1, ti)
                top_k_weights = tv / tv.sum(dim=-1, keepdim=True)
            out = torch.zeros_like(hidden_states)

            if hidden_states.shape[0] == 1:                       # DECODE - pipelined
                _mt0 = time.perf_counter()
                el = []; rwl = []; we = top_k_weights[0]
                for j, e in enumerate(top_k_index[0].tolist()):
                    if int(e) != n: el.append(int(e)); rwl.append(we[j])
                if not el: return out
                _record_route(layer_i, el)
                if BF16:                                          # full-precision experts, no dequant/kernel
                    if not hasattr(eng, "_bf16_cache"): eng._bf16_cache = {}
                    cache = eng._bf16_cache.setdefault(layer_i, OrderedDict())
                    misses = [e for e in el if e not in cache]
                    loaded = read_bf16(layer_i, misses, eng.device, eng.dtype, use_ram_cache=True) if misses else {}
                    W = {e: cache[e] if e in cache else loaded[e] for e in el}
                    with _TLOCK:
                        _TSTAT["bf16_cache_hits"] += len(el) - len(misses)
                        _TSTAT["bf16_cache_misses"] += len(misses)
                    x = hidden_states[0]
                    acc = torch.zeros_like(x)
                    if BF16_BATCHED_DECODE:
                        gu = torch.stack([W[e][0] for e in el])
                        dn = torch.stack([W[e][1] for e in el])
                        xb = x.view(1, -1, 1).expand(len(el), -1, -1)
                        gate_up = torch.bmm(gu, xb).squeeze(-1)
                        gate, up = gate_up.chunk(2, dim=-1)
                        act = (self.act_fn(gate) * up).unsqueeze(-1)
                        down = torch.bmm(dn, act).squeeze(-1)
                        for j in range(len(el)): acc = acc + rwl[j] * down[j]
                        del gu, dn, xb, gate_up, gate, up, act, down
                    else:
                        for j, e in enumerate(el):
                            gu, dn = W[e]; g, u = F.linear(x, gu).chunk(2, dim=-1)
                            acc = acc + rwl[j] * F.linear(self.act_fn(g) * u, dn)
                    if BF16_CACHE_PER_LAYER:
                        # Router top-k is strongest-first; insert in reverse so the
                        # strongest routes are the newest survivors without GPU scalar syncs.
                        for j in reversed(range(len(el))):
                            e = el[j]; cache[e] = W[e]; cache.move_to_end(e)
                        while len(cache) > BF16_CACHE_PER_LAYER: cache.popitem(last=False)
                    del W, loaded
                    out[0] = acc; eng._predict[layer_i] = el
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    _record_moe("decode", time.perf_counter() - _mt0)
                    return out
                pf = eng._pf.pop(layer_i, None) if PIPE else None
                if pf is not None:
                    torch.cuda.current_stream().wait_stream(_pstream())   # prefetch H2D for this layer done
                else: pf = {}
                hits = {e: pf[e] for e in el if e in pf}
                misses = [e for e in el if e not in pf]
                eng._pf_stat[0] += len(hits); eng._pf_stat[1] += len(el)
                if OVERLAP and not pf:                            # Lever A: pipelined NVMe||PCIe read
                    got = _load_overlap(q4f, el, eng.device, eng.dtype, torch.cuda.current_stream())
                else:
                    got = {**hits, **(_load(q4f, misses, eng.device, eng.dtype, 0, torch.cuda.current_stream()) if misses else {})}
                gq = torch.stack([got[e][0] for e in el]); gs = torch.stack([got[e][1] for e in el])
                dq = torch.stack([got[e][2] for e in el]); ds = torch.stack([got[e][3] for e in el])
                out[0] = moe_ffn(hidden_states[0], gq, gs, dq, ds, torch.stack(rwl))
                eng._predict[layer_i] = el                        # remember for next token
                if PIPE:
                    nxt = eng._predict.get(layer_i + 1)
                    if nxt and (layer_i + 1) < eng.nL:
                        eng._pf[layer_i + 1] = _load(q4f_next, nxt, eng.device, eng.dtype, 1, _pstream())
                return out

            # PREFILL - per-expert loop (one-time, no pipeline)
            mask = F.one_hot(top_k_index, num_classes=n).permute(2, 1, 0)
            hit = [int(e) for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist() if int(e) != n]
            if not hit: return out
            if BF16:                                              # full-precision prefill (one expert at a time → bounded VRAM)
                _mt0 = time.perf_counter()
                # Read several exact experts together to raise NVMe queue depth. Compute and
                # accumulate in the original sorted expert order, so numerical semantics do
                # not change; the batch only bounds the temporary BF16 staging footprint.
                for bi in range(0, len(hit), BF16_PREFILL_BATCH):
                    batch = hit[bi:bi + BF16_PREFILL_BATCH]
                    weights = read_bf16(layer_i, batch, eng.device, eng.dtype)
                    for ei in batch:
                        gu, dn = weights[ei]
                        pos, tok = torch.where(mask[ei]); cur = hidden_states[tok]
                        g, u = F.linear(cur, gu).chunk(2, dim=-1)
                        ch = F.linear(self.act_fn(g) * u, dn) * top_k_weights[tok, pos, None]
                        out.index_add_(0, tok, ch.to(out.dtype)); del gu, dn
                    del weights
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _record_moe("prefill", time.perf_counter() - _mt0)
                return out
            parts, meta = _read_experts(q4f, hit); W = {}
            for e in hit:
                p = parts[e]; vt = {nm: torch.frombuffer(p[nm], dtype=meta[nm][3]).reshape(meta[nm][2]) for nm in _NAMES}
                W[e] = (vt["gate_up_q"].to(eng.device), vt["gate_up_s"].to(eng.device, eng.dtype),
                        vt["down_q"].to(eng.device), vt["down_s"].to(eng.device, eng.dtype))
            for ei in hit:
                pos, tok = torch.where(mask[ei]); cur = hidden_states[tok]
                gqe, gse, dqe, dse = W[ei]
                g, u = F.linear(cur, dequant_int4(gqe, gse)).chunk(2, dim=-1)
                ch = F.linear(self.act_fn(g) * u, dequant_int4(dqe, dse))
                ch = ch * top_k_weights[tok, pos, None]
                out.index_add_(0, tok, ch.to(out.dtype))
            return out
        experts.forward = types.MethodType(forward, experts)
