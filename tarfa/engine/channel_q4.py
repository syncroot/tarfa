import os, re, types, torch
import torch.nn.functional as F
from collections import OrderedDict
from safetensors import safe_open
from channel import ChannelModel
Q4DIR = os.path.expanduser("~/airllm-test/q4_122b")
TOPK_EXPERTS = 5          # cut top-8 -> top-K: measured within int4's own noise floor (KL 0.022 < 0.026); set 8 to disable
CACHE_PER_LAYER = 8       # hot-expert LRU: keep this many int4 experts/layer resident on GPU (skip disk+H2D); 0 disables. VRAM-bound on 16GB card.

def dequant_int4(packed, scale):              # packed [N,in//2] uint8 (GPU), scale [N] fp16 -> [N,in] fp16
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = ((packed >> 4) & 0xF).to(torch.int16) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return q.to(torch.float16) * scale[:, None]

import struct, json
from concurrent.futures import ThreadPoolExecutor
_NAMES = ("gate_up_q", "gate_up_s", "down_q", "down_s")
_DT = {"U8": torch.uint8, "F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16}
_POOL = ThreadPoolExecutor(max_workers=32)    # raise NVMe queue depth: batched parallel pread = 3.2x vs serial mmap (measured)
import time as _time, math
_STAT = {"read_s": 0.0, "read_b": 0, "calls": 0, "experts": 0}    # in-engine read profiling
_FD = {}; _META = {}
def _q4fd(path):
    fd = _FD.get(path)
    if fd is None: fd = _FD[path] = os.open(path, os.O_RDONLY)
    return fd
def _q4meta(path):
    m = _META.get(path)
    if m is None:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        m = {}
        for k in _NAMES:
            t = h[k]; o0, o1 = t["data_offsets"]; ne = t["shape"][0]; rb = (o1 - o0) // ne
            m[k] = (base + o0, rb, tuple(t["shape"][1:]), _DT[t["dtype"]])
        _META[path] = m
    return m
def _read_experts(path, experts):             # parallel pread the missing experts' byte-ranges (high queue depth)
    fd = _q4fd(path); meta = _q4meta(path)
    reqs = [(e, name) for e in experts for name in _NAMES]
    def rd(req):
        e, name = req; st, rb, _, _ = meta[name]
        buf = bytearray(rb); os.preadv(fd, [buf], st + e * rb); return e, name, buf
    parts = {e: {} for e in experts}
    _t = _time.perf_counter(); _b = 0
    for e, name, buf in _POOL.map(rd, reqs): parts[e][name] = buf; _b += len(buf)
    _STAT["read_s"] += _time.perf_counter() - _t; _STAT["read_b"] += _b
    _STAT["calls"] += 1; _STAT["experts"] += len(experts)
    return parts, meta

# #2: static pinned L2 buffer — pread directly into page-locked host memory, one DMA per layer
_PIN = None; _PIN_NP = None; _PIN_MV = None
def _pin(nbytes):
    global _PIN, _PIN_NP, _PIN_MV
    if _PIN is None or _PIN.numel() < nbytes:
        _PIN = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        _PIN_NP = _PIN.numpy(); _PIN_MV = memoryview(_PIN_NP)
    return _PIN
def _read_pinned(path, experts, device, dtype):
    fd = _q4fd(path); meta = _q4meta(path)
    per = sum(meta[k][1] for k in _NAMES)
    _pin(len(experts) * per)
    layout = []; off = 0; reqs = []
    for e in experts:
        ent = []
        for name in _NAMES:
            st, rb, shp, dt = meta[name]
            reqs.append((fd, _PIN_MV[off:off+rb], st + e*rb)); ent.append((name, off, rb, shp, dt)); off += rb
        layout.append((e, ent))
    _t = _time.perf_counter()
    for _ in _POOL.map(lambda r: os.preadv(r[0], [r[1]], r[2]), reqs): pass   # parallel pread into pinned
    gpu = _PIN[:off].to(device)                                               # ONE DMA, pinned -> full PCIe bw
    _STAT["read_s"] += _time.perf_counter() - _t; _STAT["read_b"] += off; _STAT["calls"] += 1; _STAT["experts"] += len(experts)
    res = {}
    for (e, ent) in layout:
        t = {}
        for (name, o, rb, shp, dt) in ent:
            seg = gpu[o:o+rb]
            t[name] = seg.reshape(shp).clone() if dt == torch.uint8 else seg.view(dt).reshape(shp).to(dtype).clone()
        res[e] = (t["gate_up_q"], t["gate_up_s"], t["down_q"], t["down_s"])
    return res

class ChannelModelQ4(ChannelModel):
    def _install_streamed_experts(self, experts, pre):
        i = int(re.search(r"layers\.(\d+)\.", pre).group(1))
        q4f = f"{Q4DIR}/layer_{i}.safetensors"
        eng = self; layer_i = i
        if not hasattr(eng, "_expert_cache"): eng._expert_cache = {}; eng._cache_stat = [0, 0]
        def forward(self, hidden_states, top_k_index, top_k_weights):
            n = self.num_experts
            if top_k_weights.shape[-1] > TOPK_EXPERTS:                    # keep the TOPK highest-weight experts, renormalize
                tv, ti = top_k_weights.topk(TOPK_EXPERTS, dim=-1)
                top_k_index = torch.gather(top_k_index, -1, ti)
                top_k_weights = tv / tv.sum(dim=-1, keepdim=True)
            mask = F.one_hot(top_k_index, num_classes=n).permute(2, 1, 0)
            hit = [int(e) for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist() if int(e) != n]
            out = torch.zeros_like(hidden_states)
            if not hit: return out
            lru = eng._expert_cache.setdefault(layer_i, OrderedDict())    # hot-expert LRU: GPU-resident int4 experts
            miss = [e for e in hit if e not in lru]
            eng._cache_stat[0] += len(hit) - len(miss); eng._cache_stat[1] += len(hit)
            if miss:                                                      # parallel pread + per-expert build (clean baseline)
                parts, meta = _read_experts(q4f, miss)
                for e in miss:
                    p = parts[e]; vt = {}
                    for name in _NAMES:
                        st, rb, shp, dt = meta[name]
                        vt[name] = torch.frombuffer(p[name], dtype=dt).reshape(shp)
                    lru[e] = (vt["gate_up_q"].to(eng.device), vt["gate_up_s"].to(eng.device, eng.dtype),
                              vt["down_q"].to(eng.device), vt["down_s"].to(eng.device, eng.dtype))
            for ei in hit:                                               # compute first — every hit is guaranteed in lru
                pos, tok = torch.where(mask[ei])
                cur = hidden_states[tok]
                gqe, gse, dqe, dse = lru[ei]
                gw = dequant_int4(gqe, gse)
                dw = dequant_int4(dqe, dse)
                g, u = F.linear(cur, gw).chunk(2, dim=-1)
                ch = F.linear(self.act_fn(g) * u, dw)
                ch = ch * top_k_weights[tok, pos, None]
                out.index_add_(0, tok, ch.to(out.dtype))
            for e in hit: lru.move_to_end(e)                              # THEN mark recent + evict coldest (never mid-use)
            pinned = getattr(eng, "_pinned", None)
            keep = pinned.get(layer_i) if pinned else None
            if keep:                                                      # protect pinned experts: never evict, don't count vs budget
                ev = [e for e in lru if e not in keep]
                while len(ev) > CACHE_PER_LAYER: del lru[ev.pop(0)]
            else:
                while len(lru) > CACHE_PER_LAYER: lru.popitem(last=False)
            return out
        experts.forward = types.MethodType(forward, experts)
