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
"""
import os, time, threading, queue, ctypes, mmap
import numpy as np, torch

KINDS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
PG = mmap.PAGESIZE
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


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
        self.swaps = []                                          # active swap dicts
        self.q = queue.Queue()
        self.cr = torch.cuda.cudart()
        self.cs = {d: torch.cuda.Stream(d) for d in eng.hits}    # copy streams
        self.numa = os.environ.get("GLM5_NUMA", "0") == "1"
        self.done = 0
        self.debug = int(os.environ.get("GLM5_ADAPT_DEBUG", "0"))
        self.lazy = os.environ.get("GLM5_ADAPT_LAZY", "1") == "1"
        self.t_worker = 0.0
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
        inside = np.where(self.res & free, self.cnt, np.inf)
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
        tb = self.eng.hits[d]["tbl"][L]
        vals = [0, 0, 0] if trip is None else [t.data_ptr() for t in trip]
        with torch.cuda.device(d):
            tb[:, e] = torch.tensor(vals, dtype=torch.long).to(d, non_blocking=False)

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
                    for t in ts if not self.lazy else ():      # lazy: the cold outgoing expert
                        a = t.data[sw["out"]]                    # faults in on demand instead
                        lo, hi = _span(a)
                        _libc.madvise(lo, hi - lo, 22)        # MADV_POPULATE_READ
                        if self.numa:
                            import cpu_moe
                            cpu_moe.place_split(a, 2)
                    tt["out-populate+place"] = time.perf_counter() - c; c = time.perf_counter()
                    srcs, regs = [], []
                    for t in ts:
                        a = t.data[sw["in"]]
                        lo, hi = _span(a)                     # (registering faults the pages in)
                        rc = self.cr.cudaHostRegister(lo, hi - lo, 9)   # ReadOnly | Portable
                        if int(rc) != 0:
                            raise RuntimeError(f"cudaHostRegister {rc}")
                        regs.append(lo)
                        srcs.append(torch.from_numpy(np.asarray(a)))
                    tt["in-populate+register"] = time.perf_counter() - c
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
                    for t in ts:
                        a = t.data[sw["in"]]
                        lo, hi = _span(a, outward=False)
                        if hi > lo:
                            _libc.madvise(lo, hi - lo, 4)     # unmap, then drop from the page cache
                            self._drop(lo, hi)
            except Exception as ex:                          # keep decoding; report once
                print(f"  adapt: swap L{sw['L']} {sw['out']}->{sw['in']} failed at {sw['stage']}: {ex}", flush=True)
                if sw["stage"] in ("new",):
                    sw["stage"] = "failed"
            dt = time.perf_counter() - t0
            self.t_worker += dt
            if self.debug > 0:
                self.debug -= 1
                print(f"  adapt[dbg] L{sw['L']} {sw['out']}->{sw['in']} stage-> {sw['stage']} in {1000*dt:.1f} ms "
                      f"(steps: {', '.join(f'{k} {1000*v:.1f}' for k, v in sw.get('tt', {}).items())})", flush=True)
