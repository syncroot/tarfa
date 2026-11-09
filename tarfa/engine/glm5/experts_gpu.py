"""
Routed-expert execution for glm5-next from RAW k-quant bytes, hybrid GPU/CPU.
- per-layer LRU of raw (gate, up, down) bytes spread over one or two GPUs; a miss is placed
  by weighted round-robin over the GPUs' PCIe links (GLM5_LINK_W, default 2,1) and evicts
  the oldest entry on that card (never an expert used by the current token)
- cached experts run on their GPU via the fused Triton k-quant matvec (qmv)
- GLM5_CPU_MISSES=1 (default): missed experts are computed on the CPU (cpu_moe, from the
  page-cached mmap, overlapping the GPU's hit compute) and fetched into the GPU cache in
  the background; =0: misses are fetched synchronously and computed on the GPU
Env: GLM5_RAW_PER_LAYER (8), GLM5_RAW2_DEV / GLM5_RAW2_PER_LAYER, GLM5_FETCH_THREADS (12).
"""
import os, time, queue, threading
import numpy as np, torch
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from gguf import GGMLQuantizationType as QT
from qmv import qmv

KINDS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
LIMIT = 10.0


class RawExperts:
    def __init__(self, idx, dev):
        self.idx, self.dev = idx, dev
        self.per_layer = int(os.environ.get("GLM5_RAW_PER_LAYER", "8"))
        self.dev2 = os.environ.get("GLM5_RAW2_DEV")
        self.per_layer2 = int(os.environ.get("GLM5_RAW2_PER_LAYER", "0")) if self.dev2 else 0
        self.cap = {self.dev: self.per_layer}
        if self.per_layer2:
            self.cap[self.dev2] = self.per_layer2
        self.cpu_misses = os.environ.get("GLM5_CPU_MISSES", "1") == "1"
        # background-fetch CPU misses into the GPU cache: "1" always, "prefill" only until
        # the decode loop calls end_prefill() (static hot set, no PCIe traffic during decode)
        self.warm_mode = os.environ.get("GLM5_WARM", "prefill")
        self.warm = self.warm_mode in ("1", "prefill")
        self.warm_futs = []
        self.route = {}                                            # L -> np counts[288]
        # selective warming: admit an expert only after GLM5_WARM_MIN misses, at most
        # GLM5_WARM_BUDGET new warms per layer per token (traffic dies off as the set settles)
        self.warm_min = int(os.environ.get("GLM5_WARM_MIN", "1"))
        self.warm_budget = int(os.environ.get("GLM5_WARM_BUDGET", "99"))
        self.miss_n = {}
        # worker threads must not inherit the main thread's single-CPU pin (cpu_moe pins the
        # main thread as its OpenMP master); an OpenMP team started from them would share it
        unpin = lambda: os.sched_setaffinity(0, range(os.cpu_count()))
        self.cpu_exec = ThreadPoolExecutor(1, initializer=unpin)    # CPU experts beside GPU launches
        if self.cpu_misses:
            from cpu_moe import moe_cpu
            self._moe_cpu = moe_cpu
        self.pool = ThreadPoolExecutor(int(os.environ.get("GLM5_FETCH_THREADS", "12")), initializer=unpin)
        self.cache, self.cnt, self.inflight = {}, {}, set()
        self.lock = threading.Lock()
        self.hits = self.misses = 0
        self.t_fetch = 0.0          # sync fetch time (GPU-miss mode) or CPU miss compute time
        nb, maxb = 48, 8 << 20
        self.pins = [torch.empty(maxb, dtype=torch.uint8).pin_memory() for _ in range(nb)]
        self.free = queue.Queue()
        for i in range(nb):
            self.free.put(i)
        self.tls = threading.local()

    # ---------- host -> GPU copy of one raw matrix (pinned staging, per-thread stream) ----------
    def _load(self, t, e, dev):
        src = t.data[e]
        streams = getattr(self.tls, "streams", None)
        if streams is None:
            streams = self.tls.streams = {}
        s = streams.get(dev)
        if s is None:
            s = streams[dev] = torch.cuda.Stream(device=dev)
        i = self.free.get()
        try:
            buf = self.pins[i][:src.nbytes]
            np.copyto(buf.numpy().reshape(src.shape), src)
            g = torch.empty(src.shape, dtype=torch.uint8, device=dev)
            with torch.cuda.stream(s):
                g.copy_(buf.view(src.shape), non_blocking=True)
            s.synchronize()
            return g
        finally:
            self.free.put(i)

    # ---------- placement ----------
    def _next_dev(self):
        if not hasattr(self, "rr"):
            w = [int(v) for v in os.environ.get("GLM5_LINK_W", "2,1").split(",")]
            devs = list(self.cap)
            self.rr = [d for d, k in zip(devs, w + [1] * len(devs)) for _ in range(k)]
            self.rr_i = 0
        d = self.rr[self.rr_i % len(self.rr)]
        self.rr_i += 1
        return d

    def _place(self, L, c, keep):
        cnt = self.cnt.setdefault(L, {d: 0 for d in self.cap})
        d = self._next_dev()
        if cnt[d] < self.cap[d]:
            cnt[d] += 1
            return d
        for e_old, (_, dd) in c.items():
            if dd == d and e_old not in keep:
                c.pop(e_old)
                return d
        for d2 in self.cap:
            if cnt[d2] < self.cap[d2]:
                cnt[d2] += 1
                return d2
        for e_old in c:
            if e_old not in keep:
                return c.pop(e_old)[1]
        return None

    def _warm(self, L, e, d, ts):
        """background: fetch expert e of layer L into the cache slot reserved on device d."""
        try:
            trip = tuple(self._load(ts[i], e, d) for i in range(3))
            with self.lock:
                self.cache[L][e] = (trip, d)
        finally:
            with self.lock:
                self.inflight.discard((L, e))

    # ---------- lookup ----------
    def lookup(self, L, sel):
        ts = [self.idx[f"blk.{L}.{k}"] for k in KINDS]
        hits, misses, keep = {}, [], set(sel)
        with self.lock:
            c = self.cache.setdefault(L, OrderedDict())
            for e in sel:
                if e in c:
                    c.move_to_end(e); hits[e] = c[e]; self.hits += 1
                else:
                    misses.append(e); self.misses += 1
            todo = []
            if self.cpu_misses and self.warm and sum(self.cap.values()):
                for e in misses:
                    k = (L, e)
                    self.miss_n[k] = self.miss_n.get(k, 0) + 1
                    if len(todo) >= self.warm_budget or k in self.inflight or self.miss_n[k] < self.warm_min:
                        continue
                    d = self._place(L, c, keep)
                    if d is not None:
                        self.inflight.add(k); todo.append((e, d))
        for e, d in todo:
            self.warm_futs.append(self.pool.submit(self._warm, L, e, d, ts))
        rc = self.route.get(L)
        if rc is None:
            rc = self.route[L] = np.zeros(288, np.int64)
        rc[sel] += 1
        return hits, misses, ts, [QT(t.tensor_type) for t in ts]

    def end_prefill(self):
        """Wait for in-flight warm fetches; stop warming if GLM5_WARM=prefill. With a static hot
        set, GLM5_POPULATE=1 (default) makes RAM hold exactly the CPU-side experts: GPU-resident
        experts are dropped from the page cache and every other expert is read in and mapped
        now (MADV_POPULATE_READ), so decode takes no page faults (llama.cpp mmaps with
        MAP_POPULATE). The 186 GB GGUF does not fit the box's page cache whole; this set does."""
        for f in self.warm_futs:
            f.result()
        self.warm_futs = []
        if self.warm_mode == "prefill":
            self.warm = False
            if os.environ.get("GLM5_PRIOR"):
                self.fill_prior(np.load(os.environ["GLM5_PRIOR"]))
            if os.environ.get("GLM5_POPULATE", "1") == "1":
                self.populate()
        return sum(len(c) for c in self.cache.values())

    def fill_prior(self, prior):
        """Fill GPU slots the prefill left free with the most-routed experts of a calibration
        histogram (glm5_calib.py, generic text) - a static profiled hot set. Synchronous."""
        t0 = time.time(); futs = []
        for L in range(prior.shape[0]):
            if not prior[L].any():
                continue
            c = self.cache.setdefault(L, OrderedDict())
            cnt = self.cnt.setdefault(L, {d: 0 for d in self.cap})
            free = sum(self.cap[d] - cnt[d] for d in self.cap)
            ts = [self.idx[f"blk.{L}.{k}"] for k in KINDS]
            for e in np.argsort(-prior[L], kind="stable"):
                e = int(e)
                if free <= 0 or prior[L][e] == 0:
                    break
                if e in c:
                    continue
                d = self._next_dev()
                while cnt[d] >= self.cap[d]:
                    d = self._next_dev()
                cnt[d] += 1; free -= 1
                self.inflight.add((L, e))
                futs.append(self.pool.submit(self._warm, L, e, d, ts))
        for f in futs:
            f.result()
        print(f"  prior: +{len(futs)} experts into free GPU slots in {time.time()-t0:.0f}s", flush=True)

    def numa_place(self):
        """Split every CPU-side expert matrix by rows across the NUMA nodes (part k's pages ->
        node k) and switch the CPU kernel to the node-split variant (cpu_moe.numa_setup)."""
        import cpu_moe
        nn = cpu_moe.numa_setup()
        t0 = time.time()
        def one(L):
            gpu, bad = set(self.cache.get(L, {})), 0
            for k in KINDS:
                t = self.idx[f"blk.{L}.{k}"]
                for e in range(t.data.shape[0]):
                    if e not in gpu:
                        bad += cpu_moe.place_split(t.data[e], nn)
            return bad
        with ThreadPoolExecutor(8, initializer=lambda: os.sched_setaffinity(0, range(os.cpu_count()))) as ex:
            bad = sum(ex.map(one, sorted(self.cache)))
        print(f"  numa: experts split across {nn} nodes in {time.time()-t0:.0f}s, unmovable pages {bad}", flush=True)

    def populate(self):
        import ctypes, mmap as _mm
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        PG, DONTNEED, POPULATE_READ = _mm.PAGESIZE, 4, 22
        files = {}                                               # memmap base address -> (fd, base)
        def where(a):
            p = a.ctypes.data
            for base, (fd, size) in files.items():
                if base <= p < base + size:
                    return fd, p - base
            raise KeyError
        for t in self.idx.values():
            mm = t.data.base
            while mm is not None and not isinstance(mm, np.memmap):
                mm = mm.base
            if mm is not None and mm.ctypes.data not in files:
                files[mm.ctypes.data] = (os.open(mm.filename, os.O_RDONLY), mm.nbytes)
        t0 = time.time(); dropped = kept = 0
        for L in sorted(self.cache):
            gpu = set(self.cache[L])
            for k in KINDS:
                t = self.idx[f"blk.{L}.{k}"]
                for e in range(t.data.shape[0]):
                    a = t.data[e]; p, n = a.ctypes.data, a.nbytes
                    if e in gpu:                                  # unmap + drop (pages fully inside)
                        lo, hi = -(-p // PG) * PG, (p + n) // PG * PG
                        if hi > lo:
                            libc.madvise(lo, hi - lo, DONTNEED)
                            fd, off = where(a)
                            os.posix_fadvise(fd, off + (lo - p), hi - lo, os.POSIX_FADV_DONTNEED)
                        dropped += n
        for L in sorted(self.cache):
            gpu = set(self.cache[L])
            for k in KINDS:
                t = self.idx[f"blk.{L}.{k}"]
                for e in range(t.data.shape[0]):
                    if e not in gpu:
                        a = t.data[e]; p, n = a.ctypes.data, a.nbytes
                        lo, hi = p // PG * PG, -(-(p + n) // PG) * PG
                        libc.madvise(lo, hi - lo, POPULATE_READ); kept += n
        for fd, _ in files.values():
            os.close(fd)
        if os.environ.get("GLM5_NUMA", "0") == "1":
            self.numa_place()
        print(f"  populate: dropped {dropped/1e9:.1f} GB GPU-resident experts from page cache, "
              f"mapped {kept/1e9:.1f} GB CPU experts in {time.time()-t0:.0f}s", flush=True)

    def _fetch_sync(self, L, misses, ts):
        """GPU-miss mode: fetch now (placed + cached), return {e: (trip, dev)}."""
        out, keep = {}, set(misses)
        with self.lock:
            c = self.cache[L]
            place = {e: (self._place(L, c, keep) if sum(self.cap.values()) else None) for e in misses}
        futs = {(e, i): self.pool.submit(self._load, ts[i], e, place[e] or self.dev)
                for e in misses for i in range(3)}
        for e in misses:
            trip = tuple(futs[(e, i)].result() for i in range(3))
            d = place[e] or self.dev
            out[e] = (trip, d)
            if place[e] is not None:
                with self.lock:
                    self.cache[L][e] = (trip, d)
        return out

    # ---------- compute ----------
    def _expert(self, trip, w, x32, q):
        g_raw, u_raw, d_raw = trip
        g = qmv(g_raw, q[0], x32, x32.shape[0])
        u = qmv(u_raw, q[1], x32, x32.shape[0])
        h = torch.nn.functional.silu(torch.clamp(g, max=LIMIT)) * torch.clamp(u, -LIMIT, LIMIT)
        return w * qmv(d_raw, q[2], h.contiguous(), h.shape[0])

    def _gpu(self, items, x32, w32, q, y):
        other = {}
        for j, (trip, d) in items:
            if d == self.dev:
                y += self._expert(trip, w32[j], x32, q)
            else:
                other.setdefault(d, []).append((trip, j))
        for d, its in other.items():
            xd = x32.to(d); wd = w32.to(d); yd = torch.zeros_like(xd)
            for trip, j in its:
                yd += self._expert(trip, wd[j], xd, q)
            y += yd.to(self.dev)

    def _cpu_timed(self, *a):
        t0 = time.perf_counter(); r = self._moe_cpu(*a)
        self.t_cpu = getattr(self, "t_cpu", 0.0) + time.perf_counter() - t0
        self.t_cpu_wait0 = t0
        return r

    def moe(self, L, x32, sel, w32):
        """x32 fp32 [Ne] on dev; sel list[int]; w32 fp32 [E] -> fp32 [Ne] (routed part)."""
        hits, misses, ts, q = self.lookup(L, sel)
        y = torch.zeros(x32.shape[0], device=self.dev, dtype=torch.float32)
        fut = None
        if misses and self.cpu_misses:
            xw = torch.cat([x32, w32]).cpu().numpy()            # before launching GPU work
            x_np, w_np = xw[:x32.shape[0]], xw[x32.shape[0]:]
            ix = [sel.index(e) for e in misses]
            t0 = time.time()                                     # CPU starts first (GIL released in C)
            fut = self.cpu_exec.submit(self._cpu_timed, [ts[0].data[e] for e in misses],
                                       [ts[1].data[e] for e in misses], [ts[2].data[e] for e in misses],
                                       *q, x_np, w_np[ix])
        self._gpu([(sel.index(e), hits[e]) for e in hits], x32, w32, q, y)   # async GPU launches
        if misses:
            if fut is not None:
                y += torch.from_numpy(fut.result()).to(self.dev, non_blocking=True)
            else:
                t0 = time.time()
                got = self._fetch_sync(L, misses, ts)
                self._gpu([(sel.index(e), got[e]) for e in misses], x32, w32, q, y)
            self.t_fetch += time.time() - t0
        return y
