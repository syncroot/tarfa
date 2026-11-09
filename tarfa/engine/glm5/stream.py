"""
Offload part of a layer's CPU expert misses to the trunk GPU over PCIe (GLM5_STREAM=1).

bench_pcie.py on the reference machine: DMA from page-locked page cache reaches 12.1 GB/s to the Quadro
(1.25 ms per 15 MB expert) and 6.2 to the 4060 Ti, but only 12.4 GB/s with both at once, so only
the trunk GPU streams. The CPU kernel needs ~0.27 ms per expert in-engine, so streaming pays
only when a layer has many misses: with m CPU experts, g go to the GPU when
    max(c_cpu * (m - g), c_dma * g + c_fix) < c_cpu * m,
preferring experts that several verified tokens share (each costs the CPU one dot per token).
All CPU-side expert pages are page-locked once (cudaHostRegister, read-only) so the copies are
true async DMA while the CPU kernel runs; the copied experts then run through the masked k-quant
kernel from a few fixed VRAM slots.
"""
import os, time, mmap
import numpy as np, torch
from qmoe import moe_masked

KINDS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
PG = mmap.PAGESIZE


class Streamer:
    def __init__(self, eng, dev, T, slots=None):
        self.eng, self.dev, self.T = eng, dev, T
        self.slots = slots or int(os.environ.get("GLM5_STREAM_SLOTS", "2"))
        self.c_cpu = float(os.environ.get("GLM5_STREAM_CCPU", "0.27"))
        self.c_dma = float(os.environ.get("GLM5_STREAM_CDMA", "1.25"))
        self.c_fix = float(os.environ.get("GLM5_STREAM_CFIX", "0.15"))
        m = eng.m
        # page-lock every CPU-side expert matrix: merge page-rounded intervals, register each once
        t0 = time.time()
        iv = []
        for L in range(3, 45):
            res = eng.resident[L]
            for k in KINDS:
                t = m.idx[f"blk.{L}.{k}"]
                for e in range(t.data.shape[0]):
                    if e not in res:
                        a = t.data[e]; p = a.ctypes.data
                        iv.append((p // PG * PG, -(-(p + a.nbytes) // PG) * PG))
        iv.sort()
        merged = []
        for lo, hi in iv:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        cr = torch.cuda.cudart()
        self.regs, bad = [], 0
        for lo, hi in merged:
            if int(cr.cudaHostRegister(lo, hi - lo, 1 | 8)) == 0:     # portable | read-only
                self.regs.append(lo)
            else:
                bad += 1
        nb = sum(hi - lo for lo, hi in merged)
        print(f"  stream: page-locked {nb/1e9:.1f} GB of CPU experts in {len(merged)} ranges "
              f"({bad} failed) in {time.time()-t0:.0f}s", flush=True)
        # VRAM slots sized for the largest matrix of each kind
        mx = [max(m.idx[f"blk.{L}.{k}"].data[0].nbytes for L in range(3, 45)) for k in KINDS]
        with torch.cuda.device(dev):
            self.buf = [[torch.empty(n, dtype=torch.uint8, device=dev) for n in mx] for _ in range(self.slots)]
            self.P = []                                              # pointer table for g slots
            for g in range(self.slots + 1):
                P = torch.zeros(3, 8, dtype=torch.long)
                for j in range(g):
                    for k in range(3):
                        P[k, j] = self.buf[j][k].data_ptr()
                self.P.append(P.to(dev))
            self.w = torch.zeros(T, 8, device=dev)
            self.Hm = torch.zeros(8, 2048, device=dev)
            self.y = torch.zeros(T, eng.x32.shape[0], device=dev)
            self.cs = torch.cuda.Stream(dev)
        self.h_w = torch.zeros(T, 8).pin_memory()
        self.ev = torch.cuda.Event()
        self.n_streamed = 0

    def split(self, uni):
        """uni: {expert: weights [T]} of the CPU-side misses -> list of experts to stream."""
        m = len(uni)
        best, g_best = self.c_cpu * m, 0
        for g in range(1, min(self.slots, m) + 1):
            t = max(self.c_cpu * (m - g), self.c_dma * g + self.c_fix)
            if t < best:
                best, g_best = t, g
        if g_best == 0:
            return []
        return sorted(uni, key=lambda e: -int((uni[e] != 0).sum()))[:g_best]

    def launch(self, L, es, uni, x, ev_x):
        """Copy experts `es` of layer L into the slots and run them for the n <= T tokens of x
        [n, Ne] (ready at event ev_x) on the side stream; result in self.y[:n] at the returned event."""
        ts = [self.eng.m.idx[f"blk.{L}.{k}"] for k in KINDS]
        q, rb = self.eng.qmeta[L]
        hw = self.h_w.numpy(); hw[:] = 0
        for j, e in enumerate(es):
            hw[:, j] = uni[e]
        with torch.cuda.device(self.dev), torch.cuda.stream(self.cs):
            for j, e in enumerate(es):
                for k in range(3):
                    src = torch.from_numpy(np.asarray(ts[k].data[e]).reshape(-1))
                    self.buf[j][k][:src.numel()].copy_(src, non_blocking=True)
            self.w.copy_(self.h_w, non_blocking=True)
            self.cs.wait_event(ev_x)
            for t in range(x.shape[0]):
                moe_masked(self.P[len(es)], *q, *rb, x[t], self.w[t], self.Hm, self.y[t])
            self.ev.record(self.cs)
        self.n_streamed += len(es)
        return self.ev
