# Stream BF16 experts directly from the original 39-shard safetensors (zero quant loss).
import os, glob, struct, json, re, threading, time, mmap, torch
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
# Source checkpoint. TARFA_SNAP wins; otherwise resolve the newest snapshot of TARFA_MODEL
# from any known HF cache root. Falls back to the original hardcoded 122B path so existing
# behaviour is unchanged when nothing is set.
TARFA_MODEL = os.environ.get("TARFA_MODEL", "Qwen/Qwen3.5-122B-A10B")

def _resolve_snap():
    env = os.environ.get("TARFA_SNAP", "").strip()
    if env:
        return os.path.expanduser(env)
    slug = "models--" + TARFA_MODEL.replace("/", "--")
    roots = [os.environ.get("HF_HOME", ""), "/mnt/cold/hf",
             os.path.expanduser("~/.cache/huggingface/hub")]
    for r in roots:
        if not r:
            continue
        for base in (os.path.join(r, slug, "snapshots"), os.path.join(r, "hub", slug, "snapshots")):
            if os.path.isdir(base):
                snaps = [os.path.join(base, d) for d in os.listdir(base)]
                snaps = [d for d in snaps if os.path.isdir(d)]
                if snaps:
                    return max(snaps, key=os.path.getmtime)
    return os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/dc4d348443bc740c68e2d77492492c11606384d5")

SNAP = _resolve_snap()
# JNF v1 execution format. When TARFA_JNF points at a converted+verified tree, experts are read
# from it instead of the raw HF shards. Same bytes (conversion is byte-verified against the
# checkpoint), but each expert projection is 4096-aligned and independently addressable, which is
# what the format exists for. Unset => original safetensors path, unchanged.
JNF_DIR = os.environ.get("TARFA_JNF", "")
_META = None; _FDS = {}; _POOL = ThreadPoolExecutor(max_workers=48)
_LOCK = threading.Lock()
PINNED = os.environ.get("BF16_PINNED", "1") == "1"
_PIN_LOCK = threading.Lock()
_PIN = {"g": None, "d": None, "g_mv": None, "d_mv": None,
        "g_cap": 0, "d_cap": 0}
RAM_CACHE_GB = float(os.environ.get("BF16_RAM_CACHE_GB", "0"))
_RAM = None

class ExpertArena:
    """Fixed-slot raw-BF16 arena with independent segmented queues per layer.

    `layers` was hardcoded to 48 (the 122B's depth). On any model with a different depth
    the budget was divided by the wrong count, so the arena could not hold a full layer's
    expert set no matter how large the budget: a 40-layer model at a 65 GB budget got
    per_layer=215 against 256 experts. It is now derived from the checkpoint. Caching only
    -- no weight, router or arithmetic change."""
    def __init__(self, budget_bytes, slot_bytes, layers=None, experts_per_layer=None):
        if layers is None:
            try: layers = len(meta())
            except Exception: layers = 48
        total = int(budget_bytes // slot_bytes)
        self.per_layer = max(1, total // layers); self.slots = self.per_layer * layers
        # Full residency is only possible when a layer's whole expert set has slots.
        self.full_resident = (experts_per_layer is not None
                              and self.per_layer >= experts_per_layer)
        self.slot_bytes = slot_bytes; self.mem = mmap.mmap(-1, self.slots * slot_bytes)
        self.free = list(range(self.slots - 1, -1, -1)); self.where = {}
        self.probation = [OrderedDict() for _ in range(layers)]
        self.protected = [OrderedDict() for _ in range(layers)]
        self.protected_cap = max(1, self.per_layer * 3 // 4)
        self.hits = self.misses = self.evictions = self.avoided_bytes = 0

    def get(self, layer, expert, gdst, ddst):
        key = (layer, expert); item = self.where.get(key)
        if item is None: self.misses += 1; return False
        slot, segment = item; start = slot * self.slot_bytes
        gdst[:] = self.mem[start:start + len(gdst)]
        ddst[:] = self.mem[start + len(gdst):start + len(gdst) + len(ddst)]
        if segment == "probation":
            self.probation[layer].pop(expert, None); self.protected[layer][expert] = slot
            self.where[key] = (slot, "protected"); self._trim_protected(layer)
        else: self.protected[layer].move_to_end(expert)
        self.hits += 1; self.avoided_bytes += self.slot_bytes; return True

    def put(self, layer, expert, gsrc, dsrc):
        key = (layer, expert)
        if key in self.where: return
        while len(self.probation[layer]) + len(self.protected[layer]) >= self.per_layer:
            queue = self.probation[layer] if self.probation[layer] else self.protected[layer]
            old, slot = queue.popitem(last=False); self.where.pop((layer, old)); self.evictions += 1
            self.free.append(slot)
        slot = self.free.pop(); start = slot * self.slot_bytes
        self.mem[start:start + len(gsrc)] = gsrc
        self.mem[start + len(gsrc):start + len(gsrc) + len(dsrc)] = dsrc
        self.probation[layer][expert] = slot; self.where[key] = (slot, "probation")

    def _trim_protected(self, layer):
        while len(self.protected[layer]) > self.protected_cap:
            expert, slot = self.protected[layer].popitem(last=False)
            self.probation[layer][expert] = slot; self.where[(layer, expert)] = (slot, "probation")

    def snapshot(self):
        return {"enabled": True, "budget_gb": RAM_CACHE_GB, "slots": self.slots,
                "per_layer": self.per_layer, "full_resident": self.full_resident,
                "per_layer": self.per_layer, "resident": len(self.where), "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions,
                "hit_rate": round(self.hits / max(self.hits + self.misses, 1), 4),
                "avoided_bytes": self.avoided_bytes}

def ram_cache_stats():
    return _RAM.snapshot() if _RAM is not None else {"enabled": False}
_STAT = {"calls": 0, "experts": 0, "bytes": 0, "read_seconds": 0.0,
         "h2d_seconds": 0.0, "prefetch_calls": 0, "prefetch_bytes": 0}

def stats(reset=False):
    with _LOCK:
        out = dict(_STAT)
        if reset:
            _STAT.update(calls=0, experts=0, bytes=0, read_seconds=0.0,
                         h2d_seconds=0.0, prefetch_calls=0, prefetch_bytes=0)
    out["read_GBps"] = round(out["bytes"] / 1e9 / out["read_seconds"], 3) if out["read_seconds"] else 0.0
    return out

def _build_jnf():
    """Build the expert map from a JNF v1 manifest. Offsets are 0 because each layer file holds
    exactly its 256 experts back to back; read_bf16 already indexes with offset + e*expert_bytes."""
    global _META; _META = {}
    root = os.path.expanduser(JNF_DIR)
    man = json.load(open(os.path.join(root, "manifest.json")))
    if man.get("format") != "JNF" or int(man.get("version", 0)) != 1:
        raise RuntimeError("unsupported JNF manifest: %r v%r" % (man.get("format"), man.get("version")))
    if man.get("dtype") != "BF16":
        raise RuntimeError("JNF dtype is %r, expected BF16" % man.get("dtype"))
    for lid, rec in man["layers"].items():
        entry = {}
        for name in ("gate_up_proj", "down_proj"):
            r = rec[name]
            path = os.path.join(root, r["file"])
            shape = tuple(r["shape"])
            if shape[1] * shape[2] * 2 != r["expert_bytes"]:
                raise RuntimeError("layer %s %s: shape/expert_bytes disagree" % (lid, name))
            entry[name] = (path, 0, shape, "BF16")
        _META[int(lid)] = entry
    return _META

def _build():
    if JNF_DIR:
        return _build_jnf()
    global _META; _META = {}
    _src = {}          # (layer, proj) -> checkpoint key, so a double binding is caught
    shards = [s for s in glob.glob(os.path.join(SNAP, "*.safetensors*")) if "-of-" in os.path.basename(s)]
    for sh in shards:
        with open(sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100_000_000: continue                  # not a real safetensors header
            hdr = json.loads(f.read(n)); base = 8 + n
        for k, v in hdr.items():
            # The checkpoint also ships an MTP (multi-token-prediction) draft block whose
            # tensors are named mtp.layers.N.mlp.experts.*, which this unanchored search
            # matched just as happily as the real stack. Whichever key the shard scan saw
            # last won, so layer 0 could silently bind its experts to the draft block.
            if k.startswith("mtp.") or ".mtp." in k:
                continue
            m = re.search(r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)", k)
            if m:
                lid, proj = int(m.group(1)), m.group(2)
                prev = _src.get((lid, proj))
                if prev is not None and prev != k:
                    raise RuntimeError(
                        "two checkpoint tensors claim layer %d %s: %r and %r. Refusing to guess "
                        "which is the real expert stack." % (lid, proj, prev, k))
                _src[(lid, proj)] = k
                _META.setdefault(lid, {})[proj] = (sh, base + v["data_offsets"][0], tuple(v["shape"]), v["dtype"])
    return _META

def meta():
    global _META
    if _META is None: _build()
    return _META

def _fd(p):
    if p not in _FDS: _FDS[p] = os.open(p, os.O_RDONLY)
    return _FDS[p]

def prefetch_bf16(predictions, per_layer=2):
    """Best-effort async page-cache hints; never changes authoritative weights."""
    if not hasattr(os, "posix_fadvise") or not predictions: return
    reqs = []
    for layer, experts in predictions.items():
        if layer not in meta(): continue
        m = meta()[layer]
        shg, offg, shpg, _ = m["gate_up_proj"]; shd, offd, shpd, _ = m["down_proj"]
        eg = shpg[1] * shpg[2] * 2; ed = shpd[1] * shpd[2] * 2
        for expert in experts[:per_layer]:
            reqs.append((_fd(shg), offg + expert * eg, eg))
            reqs.append((_fd(shd), offd + expert * ed, ed))
    def hint(r): os.posix_fadvise(r[0], r[1], r[2], os.POSIX_FADV_WILLNEED)
    for _ in _POOL.map(hint, reqs): pass
    with _LOCK:
        _STAT["prefetch_calls"] += 1
        _STAT["prefetch_bytes"] += sum(r[2] for r in reqs)

def _pinned(tag, needed):
    cap_key, mv_key = f"{tag}_cap", f"{tag}_mv"
    if _PIN[tag] is None or _PIN[cap_key] < needed:
        buf = torch.empty(needed, dtype=torch.uint8, pin_memory=True)
        _PIN[tag] = buf; _PIN[cap_key] = needed
        _PIN[mv_key] = memoryview(buf.numpy()).cast("B")
    return _PIN[tag], _PIN[mv_key]

def prewarm_tree(root=None, block=8<<20):
    """Make the JNF tree page-cache resident by reading it once, sequentially.

    Rationale, measured: the ExpertArena reaches an 84.67% hit rate and avoids 133.3 GiB
    of reads, yet loses 36% end-to-end warm (0.857 vs 1.333 tok/s) because it inserts a
    copy: file -> page cache -> arena -> pinned buffer. Once the pages are cached the
    extra memcpy costs more than the disk read it saves. Page-cache residency gets the
    same bytes into RAM with NO second copy -- the reader mmaps the file and the kernel
    serves it from cache.

    Exact by construction: this only reads bytes and issues a readahead hint. It changes
    no weight, no routing decision, and no arithmetic. Enable with TARFA_PREWARM=1.

    Cost is a one-off startup read of the tree (60 GiB from a spinning disk ~7 min).
    """
    root = root or os.environ.get("TARFA_JNF", "")
    if not root:
        return {"enabled": False, "reason": "TARFA_JNF unset"}
    root = os.path.expanduser(root)
    man_path = os.path.join(root, "manifest.json")
    if not os.path.exists(man_path):
        return {"enabled": False, "reason": "no manifest"}
    man = json.load(open(man_path))
    files, total = [], 0
    for lid, rec in man["layers"].items():
        for name in ("gate_up_proj", "down_proj"):
            fp = os.path.join(root, rec[name]["file"])
            if os.path.exists(fp):
                files.append(fp); total += os.path.getsize(fp)
    t0 = time.time(); read = 0
    buf = bytearray(block)
    mv = memoryview(buf)
    for fp in sorted(files):
        fd = os.open(fp, os.O_RDONLY)
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            while True:
                n = os.preadv(fd, [mv], read_off := os.lseek(fd, 0, os.SEEK_CUR))
                if n <= 0: break
                os.lseek(fd, n, os.SEEK_CUR); read += n
        finally:
            os.close(fd)
    el = time.time() - t0
    return {"enabled": True, "files": len(files), "bytes": read,
            "gib": round(read/2**30, 2), "seconds": round(el, 1),
            "mb_s": round(read/max(el, 1e-9)/1e6, 1)}


def read_bf16(layer, experts, device, dtype, use_ram_cache=False):
    """Return {e: (gate_up [2048,3072], down [3072,1024])} on device."""
    M = meta()[layer]
    shg, offg, shpg, dtg = M["gate_up_proj"]; shd, offd, shpd, dtd = M["down_proj"]
    eg = shpg[1] * shpg[2] * 2; ed = shpd[1] * shpd[2] * 2          # bytes/expert (bf16=2B)
    global _RAM
    if use_ram_cache and RAM_CACHE_GB > 0 and _RAM is None:
        _RAM = ExpertArena(RAM_CACHE_GB * 1e9, eg + ed,
                           layers=len(meta()), experts_per_layer=shpg[0])
    if not experts: return {}
    if PINNED:
        with _PIN_LOCK:
            return _read_bf16_pinned(layer, experts, device, dtype, shg, offg, shpg,
                                     shd, offd, shpd, eg, ed, use_ram_cache)
    reqs = []
    for e in experts:
        reqs.append(("g", e, shg, offg + e * eg, eg, shpg[1:]))
        reqs.append(("d", e, shd, offd + e * ed, ed, shpd[1:]))
    def rd(r):
        tag, e, sh, off, nb, shp = r; ba = bytearray(nb); os.preadv(_fd(sh), [ba], off)
        return tag, e, torch.frombuffer(ba, dtype=torch.bfloat16).reshape(shp)
    out = {}; t0 = time.perf_counter()
    for tag, e, t in _POOL.map(rd, reqs):
        out.setdefault(e, {})[tag] = t
    read_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    for e in experts:
        out[e]["g"] = out[e]["g"].to(device, dtype)
        out[e]["d"] = out[e]["d"].to(device, dtype)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    h2d_s = time.perf_counter() - t1
    with _LOCK:
        _STAT["calls"] += 1; _STAT["experts"] += len(experts)
        _STAT["bytes"] += len(experts) * (eg + ed)
        _STAT["read_seconds"] += read_s; _STAT["h2d_seconds"] += h2d_s
    return {e: (out[e]["g"], out[e]["d"]) for e in experts}

def _read_bf16_pinned(layer, experts, device, dtype, shg, offg, shpg,
                      shd, offd, shpd, eg, ed, use_ram_cache):
    """pread directly into reusable page-locked buffers, then async H2D."""
    count = len(experts)
    gb, gmv = _pinned("g", count * eg); db, dmv = _pinned("d", count * ed)
    reqs = []
    for j, expert in enumerate(experts):
        gs, ds = gmv[j*eg:(j+1)*eg], dmv[j*ed:(j+1)*ed]
        if not (use_ram_cache and _RAM is not None and _RAM.get(layer, expert, gs, ds)):
            reqs.append((_fd(shg), gs, offg + expert * eg))
            reqs.append((_fd(shd), ds, offd + expert * ed))
    t0 = time.perf_counter()
    for _ in _POOL.map(lambda r: os.preadv(r[0], [r[1]], r[2]), reqs): pass
    if use_ram_cache and _RAM is not None:
        for j, expert in enumerate(experts):
            if (layer, expert) not in _RAM.where:
                _RAM.put(layer, expert, gmv[j*eg:(j+1)*eg], dmv[j*ed:(j+1)*ed])
    read_s = time.perf_counter() - t0
    g_cpu = gb[:count*eg].view(torch.bfloat16).reshape((count,) + tuple(shpg[1:]))
    d_cpu = db[:count*ed].view(torch.bfloat16).reshape((count,) + tuple(shpd[1:]))
    t1 = time.perf_counter()
    g_gpu = g_cpu.to(device, dtype=dtype, non_blocking=True)
    d_gpu = d_cpu.to(device, dtype=dtype, non_blocking=True)
    if str(device).startswith("cuda"): torch.cuda.synchronize()
    h2d_s = time.perf_counter() - t1
    out = {expert: (g_gpu[j], d_gpu[j]) for j, expert in enumerate(experts)}
    with _LOCK:
        _STAT["calls"] += 1; _STAT["experts"] += count
        _STAT["bytes"] += count * (eg + ed)
        _STAT["read_seconds"] += read_s; _STAT["h2d_seconds"] += h2d_s
    return out
