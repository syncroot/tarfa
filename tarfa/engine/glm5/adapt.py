"""
Adaptive GPU expert set during decode (GLM5_ADAPT=<swaps started per token>).

Per layer, decayed routing counts pick the hottest CPU-side expert and the coldest GPU-resident
one; when the gap is large enough the GPU slot is handed over. A swap is a staged pipeline so
the graphs and the host never disagree about which experts are resident in a token:
  1. worker:  read the outgoing expert back into the page cache (it was dropped at startup),
              NUMA-place it for the CPU kernel, and page-lock the incoming expert's mmap pages
              (cudaHostRegister, read-only) so the copy is a DMA from the page cache, no staging
  2. main, at a token boundary: drop the outgoing expert (device table entry -> 0, host set)
              and record an event on that device's stream after the table write
  3. worker:  copy stream waits on that event, DMAs the incoming expert into the freed slot
  4. main, at a later token boundary, once the copy event completed: publish the incoming
              expert (device table entry -> slot pointers, host set); worker unregisters and drops
              its page-cache pages
Simulated on a 256-token trace: hit rate 38% static -> 53% at 4 swaps/token.

With GLM5_PIN_PER_LAYER (experts_gpu.choose_pinned) RAM already holds every expert that is not
pinned GPU-only, NUMA-placed. Swaps then touch neither the disk nor the page cache: the outgoing
expert is ready for the CPU kernel as it is, the incoming one stays in RAM after the copy, and
pinned experts are never evicted (they have no RAM copy to fall back to).

GLM5_ADAPT_SRC=stage copies the incoming expert into a page-locked staging buffer (allocated
once) and DMAs from there instead of page-locking the mmap pages per swap (cudaHostRegister,
about 2.3 ms). Measured slower on the reference machine: the extra memcpy competes with the CPU
kernel for memory bandwidth.

Measured 2026-11-10 (story prompt, 128 tokens, warm): none of this beats the static set there.
Hit rate rises from 43% to 49-51%, but each swap costs more decode time than its hits save.
"""
import os, time, threading, queue, ctypes, mmap
import numpy as np, torch

KINDS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
PG = mmap.PAGESIZE
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def _span(a, outward=True):
    p, n = a.ctypes.data, a.nbytes
    if outward:
        return p // PG * PG, -(-(p + n) // PG) * PG
    return -(-p // PG) * PG, (p + n) // PG * PG


class Adapter:
    def __init__(self, eng, budget, decay=0.97, margin=1.0, max_inflight=None):
        self.eng, self.budget, self.decay, self.margin = eng, budget, decay, margin
        self.max_inflight = max_inflight or 2 * budget
        self.nl = 45
        self.cnt = np.zeros((self.nl, 288))
        self.res = np.zeros((self.nl, 288), bool)
        for L, s in eng.resident.items():
            self.res[L, list(s)] = True
        self.busy = np.zeros((self.nl, 288), bool)              # in some swap pipeline
        self.keep_ram = eng.raw.pinned is not None               # see module docstring
        self.pin = np.zeros((self.nl, 288), bool)
        for L, s in (eng.raw.pinned or {}).items():
            self.pin[L, list(s)] = True
        self.swaps = []                                          # active swap dicts
        self.q = queue.Queue()
        self.cr = torch.cuda.cudart()
        self.cs = {d: torch.cuda.Stream(d) for d in eng.hits}    # copy streams
        self.numa = os.environ.get("GLM5_NUMA", "0") == "1"
        self.done = 0
        self.debug = int(os.environ.get("GLM5_ADAPT_DEBUG", "0"))
        self.lazy = os.environ.get("GLM5_ADAPT_LAZY", "1") == "1"
        self.t_worker = self.t_tick = self.t_tbl = 0.0
        # table updates: "async" queues a pinned-host copy on the device's stream (ordered with the
        # hit graphs, no wait); "sync" is the blocking write
        self.tbl_async = os.environ.get("GLM5_ADAPT_TBL", "async") == "async"
        self.tbl_pin = torch.empty(512, 3, dtype=torch.long).pin_memory()   # ring: a row is reused
        self.tbl_np, self.tbl_i = self.tbl_pin.numpy(), 0                    # long after its copy ran
        self.stage = os.environ.get("GLM5_ADAPT_SRC", "register") == "stage"
        self.pool = queue.SimpleQueue()                          # free staging triples
        if self.stage:
            big = [max(eng.m.idx[f"blk.{L}.{k}"].data[0].nbytes for L in range(3, self.nl)) for k in KINDS]
            for _ in range(self.max_inflight):
                self.pool.put([torch.empty(n, dtype=torch.uint8).pin_memory() for n in big])
        self.files = {}                                          # memmap base -> (fd, size)
        for t in eng.m.idx.values():
            mm = t.data.base
            while mm is not None and not isinstance(mm, np.memmap):
                mm = mm.base
            if mm is not None and mm.ctypes.data not in self.files:
                self.files[mm.ctypes.data] = (os.open(mm.filename, os.O_RDONLY), mm.nbytes)
        threading.Thread(target=self._worker, daemon=True, name='adapt-worker').start()

    def _drop(self, lo, hi):
        for base, (fd, size) in self.files.items():
            if base <= lo < base + size:
                os.posix_fadvise(fd, lo - base, hi - lo, os.POSIX_FADV_DONTNEED)
                return

    # ---------- main thread ----------
    def observe(self, L, sel):
        self.cnt[L, sel] += 1

    def tick(self):
        """Token boundary: advance pipelines, then start new swaps."""
        t0 = time.perf_counter()
        self._tick()
        self.t_tick += time.perf_counter() - t0

    def _tick(self):
        eng = self.eng
        for sw in list(self.swaps):
            if sw["stage"] == "prepared":                       # 2: drop the outgoing expert
                L, r, d = sw["L"], sw["out"], sw["dev"]
                self._table(d, L, r, None)
                eng.resident[L].discard(r); self.res[L, r] = False
                del eng.raw.cache[L][r]
                ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(d))
                sw["ev_rm"] = ev; sw["stage"] = "copying"
                self.q.put(sw)
            elif sw["stage"] == "copied" and sw["ev_cp"].query():    # 4: publish the incoming one
                L, e, d = sw["L"], sw["in"], sw["dev"]
                self._table(d, L, e, sw["trip"])
                eng.resident[L].add(e); self.res[L, e] = True
                eng.raw.cache[L][e] = (sw["trip"], d)
                sw["stage"] = "cleanup"; self.q.put(sw)
                self.swaps.remove(sw)
                self.busy[L, e] = self.busy[L, sw["out"]] = False
                self.done += 1
            elif sw["stage"] == "failed":
                self.swaps.remove(sw)
                self.busy[sw["L"], sw["in"]] = self.busy[sw["L"], sw["out"]] = False
        self.cnt *= self.decay
        room = self.max_inflight - len(self.swaps)
        n = min(self.budget, room)
        if n <= 0:
            return
        free = ~self.busy
        outside = np.where(~self.res & free, self.cnt, -np.inf)
        inside = np.where(self.res & free & ~self.pin, self.cnt, np.inf)
        eo = outside.argmax(1); ei = inside.argmin(1)
        L_ = np.arange(self.nl)
        gain = outside[L_, eo] - inside[L_, ei]
        for L in np.argsort(-gain)[:n]:
            if not gain[L] > self.margin:
                break
            L, e_in, r = int(L), int(eo[L]), int(ei[L])
            trip, d = eng.raw.cache[L][r]
            sw = dict(L=L, **{"in": e_in, "out": r}, dev=d, trip=trip, stage="new")
            self.busy[L, e_in] = self.busy[L, r] = True
            self.swaps.append(sw)
            self.q.put(sw)

    def _table(self, d, L, e, trip):
        t0 = time.perf_counter()
        tb = self.eng.hits[d]["tbl"][L]
        vals = [0, 0, 0] if trip is None else [t.data_ptr() for t in trip]
        with torch.cuda.device(d):
            if self.tbl_async:
                k = self.tbl_i % 512; self.tbl_i += 1
                self.tbl_np[k] = vals
                tb[:, e].copy_(self.tbl_pin[k], non_blocking=True)
            else:
                tb[:, e] = torch.tensor(vals, dtype=torch.long).to(d, non_blocking=False)
        self.t_tbl += time.perf_counter() - t0

    def verify(self):
        """After a run (GLM5_ADAPT_VERIFY=1): every GPU slot holds the GGUF bytes of the expert the
        device table points at, and the table has no entry for an expert that is not resident."""
        eng = self.eng
        torch.cuda.synchronize()
        bad_bytes = bad_tbl = n = 0
        for L, c in eng.raw.cache.items():
            ts = [eng.m.idx[f"blk.{L}.{k}"] for k in KINDS]
            for d in eng.hits:
                tb = eng.hits[d]["tbl"][L].cpu().numpy()
                for e in range(288):
                    here = e in c and c[e][1] == d
                    want = [t.data_ptr() for t in c[e][0]] if here else [0, 0, 0]
                    bad_tbl += [int(v) for v in tb[:, e]] != want
                    if here:
                        n += 1
                        bad_bytes += not all(
                            np.array_equal(g.cpu().numpy().reshape(-1).view(np.uint8), np.asarray(t.data[e]).reshape(-1).view(np.uint8))
                            for g, t in zip(c[e][0], ts))
        print(f"adapt verify: {n} GPU experts compared with the GGUF bytes: {bad_bytes} differ; "
              f"table entries wrong: {bad_tbl}", flush=True)

    # ---------- worker thread ----------
    def _worker(self):
        os.sched_setaffinity(0, range(os.cpu_count()))
        idx = self.eng.m.idx
        while True:
            sw = self.q.get()
            t0 = time.perf_counter()
            try:
                L = sw["L"]; ts = [idx[f"blk.{L}.{k}"] for k in KINDS]
                if sw["stage"] == "new":                      # 1: CPU readiness of outgoing, pin incoming
                    tt = sw["tt"] = {}; c = time.perf_counter()
                    for t in ts if not (self.lazy or self.keep_ram) else ():   # lazy: the cold outgoing expert
                        a = t.data[sw["out"]]                    # faults in on demand instead
                        lo, hi = _span(a)
                        _libc.madvise(lo, hi - lo, 22)        # MADV_POPULATE_READ
                        if self.numa:
                            import cpu_moe
                            cpu_moe.place_split(a, 2)
                    tt["out-populate+place"] = time.perf_counter() - c; c = time.perf_counter()
                    srcs, regs = [], []
                    if self.stage:
                        sw["bufs"] = self.pool.get_nowait()
                    for t, buf in zip(ts, sw.get("bufs", ())):
                        a = np.asarray(t.data[sw["in"]])
                        b = buf.numpy()[:a.nbytes]
                        _libc.memcpy(b.ctypes.data, a.ctypes.data, a.nbytes)
                        srcs.append(torch.from_numpy(b.view(a.dtype).reshape(a.shape)))
                    for t in ts if not self.stage else ():
                        a = t.data[sw["in"]]
                        lo, hi = _span(a)                     # (registering faults the pages in)
                        rc = self.cr.cudaHostRegister(lo, hi - lo, 9)   # ReadOnly | Portable
                        if int(rc) != 0:
                            raise RuntimeError(f"cudaHostRegister {rc}")
                        regs.append(lo)
                        srcs.append(torch.from_numpy(np.asarray(a)))
                    tt["in-stage" if self.stage else "in-populate+register"] = time.perf_counter() - c
                    sw["srcs"], sw["regs"] = srcs, regs
                    sw["stage"] = "prepared"
                elif sw["stage"] == "copying":                # 3: DMA into the freed slot
                    d = sw["dev"]; s = self.cs[d]
                    with torch.cuda.device(d), torch.cuda.stream(s):
                        s.wait_event(sw["ev_rm"])
                        for dst, src in zip(sw["trip"], sw["srcs"]):
                            dst.copy_(src, non_blocking=True)
                        ev = torch.cuda.Event(); ev.record(s)
                    sw["ev_cp"] = ev
                    sw["stage"] = "copied"
                elif sw["stage"] == "cleanup":                # after publish: unpin, drop page cache
                    sw["ev_cp"].synchronize()
                    for lo in sw["regs"]:
                        self.cr.cudaHostUnregister(lo)
                    sw["srcs"] = None
                    if sw.get("bufs"):
                        self.pool.put(sw.pop("bufs"))
                    for t in ts if not self.keep_ram else ():
                        a = t.data[sw["in"]]
                        lo, hi = _span(a, outward=False)
                        if hi > lo:
                            _libc.madvise(lo, hi - lo, 4)     # unmap, then drop from the page cache
                            self._drop(lo, hi)
            except Exception as ex:                          # keep decoding; report once
                print(f"  adapt: swap L{sw['L']} {sw['out']}->{sw['in']} failed at {sw['stage']}: {ex}", flush=True)
                if sw["stage"] in ("new",):
                    sw["stage"] = "failed"
                    if sw.get("bufs"):
                        self.pool.put(sw.pop("bufs"))
            dt = time.perf_counter() - t0
            self.t_worker += dt
            if self.debug > 0:
                self.debug -= 1
                print(f"  adapt[dbg] L{sw['L']} {sw['out']}->{sw['in']} stage-> {sw['stage']} in {1000*dt:.1f} ms "
                      f"(steps: {', '.join(f'{k} {1000*v:.1f}' for k, v in sw.get('tt', {}).items())})", flush=True)
