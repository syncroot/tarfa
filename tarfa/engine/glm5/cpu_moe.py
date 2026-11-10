"""ctypes wrapper for libcpumoe.so (CPU routed experts from raw k-quant bytes in the mmap).
ctypes releases the GIL during the call, so the GPU side keeps launching meanwhile."""
import os, ctypes
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_USE_GGML = os.environ.get("GLM5_CPU_KERNEL", "ggml") == "ggml" and os.path.exists(os.path.join(_HERE, "libcpumoe_ggml.so"))
# RTLD_DEEPBIND: resolve OpenMP against this library's own libgomp. Without it, in a process that
# imported torch, GOMP_* bind to torch's bundled libgomp (loaded RTLD_GLOBAL) - a different runtime.
_MODE = os.RTLD_DEEPBIND | ctypes.RTLD_LOCAL if os.environ.get("GLM5_CPU_DEEPBIND", "1") == "1" else ctypes.DEFAULT_MODE
_lib = ctypes.CDLL(os.path.join(_HERE, "libcpumoe_ggml.so" if _USE_GGML else "libcpumoe.so"), mode=_MODE)
if _USE_GGML:
    _lib.cpumoe_init()                       # ggml_cpu_init: tables used by the vec_dot kernels
    # GLM5_CPU_PIN: comma list of CPUs, one per OpenMP thread ("" = leave unpinned)
    # default: one thread per physical core (hyperthread siblings measured slower in-engine)
    def _cores():
        import glob
        firsts = set()
        for f in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"):
            firsts.add(int(open(f).read().replace("-", ",").split(",")[0]))
        return sorted(firsts) or list(range(os.cpu_count()))
    _pin = os.environ.get("GLM5_CPU_PIN", ",".join(map(str, _cores())))
    if _pin:
        _c = [int(v) for v in _pin.split(",")]
        _r = _lib.cpumoe_pin(len(_c), (ctypes.c_int * len(_c))(*_c))
        if _r % 1000 or _r // 1000 != len(_c):
            print(f"cpu_moe: pinned team {_r // 1000}/{len(_c)} threads, {_r % 1000} pin failures", flush=True)
_P = ctypes.POINTER(ctypes.c_uint8)
_F = ctypes.POINTER(ctypes.c_float)
_lib.moe_cpu.argtypes = [ctypes.c_int, ctypes.POINTER(_P), ctypes.POINTER(_P), ctypes.POINTER(_P),
                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.c_long, ctypes.c_long, ctypes.c_long, _F, _F, ctypes.c_float, _F, _F]
if not _USE_GGML:
    _lib.qmv_cpu.argtypes = [_P, ctypes.c_int, ctypes.c_int, ctypes.c_long, _F, ctypes.c_int, _F]
KERNEL = "ggml" if _USE_GGML else "own"
if _USE_GGML:
    # own AVX-512BW Q4_K/Q5_K dots with in-row software prefetch (bench_kern.py: 0.22 vs ggml's 0.28 ms/expert)
    _lib.cpumoe_set_avx512(int(os.environ.get("GLM5_CPU_AVX512", "1")))
    _lib.cpumoe_set_pf_bytes(int(os.environ.get("GLM5_CPU_PF_BYTES", "4096")))
    _lib.cpumoe_dot_test.restype = ctypes.c_float
    _lib.cpumoe_dot_test.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def _ptrs(arrs):
    a = (_P * len(arrs))()
    for i, m in enumerate(arrs):
        a[i] = ctypes.cast(m.ctypes.data, _P)
    return a


def moe_cpu(gs, us, ds, qg, qu, qd, x, w, lim=10.0, out=None):
    """gs/us/ds: numpy uint8 [rows, row_bytes] slices (mmap views ok); x fp32 [Ne]; w fp32 [E].
    out: optional fp32 [Ne] destination (e.g. a pinned buffer)."""
    E = len(gs); nff = gs[0].shape[0]; ne = x.shape[0]
    x = np.ascontiguousarray(x, np.float32); w = np.ascontiguousarray(w, np.float32)
    h = np.empty(E * nff, np.float32)
    y = out if out is not None else np.empty(ds[0].shape[0], np.float32)
    _lib.moe_cpu(E, _ptrs(gs), _ptrs(us), _ptrs(ds), int(qg), int(qu), int(qd), nff, ne,
                 gs[0].shape[1], us[0].shape[1], ds[0].shape[1],
                 x.ctypes.data_as(_F), w.ctypes.data_as(_F), lim, h.ctypes.data_as(_F), y.ctypes.data_as(_F))
    return y


def qmv_cpu(W, qt, x):
    x = np.ascontiguousarray(x, np.float32); y = np.empty(W.shape[0], np.float32)
    from gguf import GGMLQuantizationType as QT
    bb = {12: 144, 13: 176, 14: 210}[int(qt)]
    _lib.qmv_cpu(ctypes.cast(W.ctypes.data, _P), int(qt), W.shape[0], W.shape[1],
                 x.ctypes.data_as(_F), W.shape[1] // bb * 256, y.ctypes.data_as(_F))
    return y


def pin(cpus):
    """(Re)pin the OpenMP team: thread t -> cpus[t]; moe_cpu then uses len(cpus) threads."""
    r = _lib.cpumoe_pin(len(cpus), (ctypes.c_int * len(cpus))(*cpus))
    return r // 1000, r % 1000


# ---- NUMA split (GLM5_NUMA=1): rows of each expert split across nodes, see cpu_moe_ggml.c ----
_lib.moe_cpu_numa.argtypes = _lib.moe_cpu.argtypes
_lib.cpumoe_place.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.cpumoe_place.restype = ctypes.c_long
_lib.cpumoe_where.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]


def numa_nodes():
    import glob
    return sorted(int(p.rsplit("node", 1)[1]) for p in glob.glob("/sys/devices/system/node/node[0-9]*"))


def node_cpus(node):                          # one CPU per physical core on that node
    import glob
    out = set()
    for f in glob.glob(f"/sys/devices/system/node/node{node}/cpu[0-9]*/topology/thread_siblings_list"):
        out.add(int(open(f).read().replace("-", ",").split(",")[0]))
    return sorted(out)


def numa_setup():
    """Pin the team node by node and switch moe_cpu to the split kernel. Returns #nodes."""
    nodes = numa_nodes()
    if os.environ.get("GLM5_NUMA_HT", "0") == "1":     # all HW threads per node: faster standalone, slower in-engine
        import glob
        groups = [sorted(int(q.rsplit("cpu", 1)[1]) for q in glob.glob(f"/sys/devices/system/node/node{n}/cpu[0-9]*"))
                  for n in nodes]
    else:
        groups = [node_cpus(n) for n in nodes]
    k = min(len(g) for g in groups)
    pin([c for g in groups for c in g[:k]])
    _lib.cpumoe_set_nodes(len(nodes))
    global moe_cpu
    moe_cpu = moe_cpu_numa
    return len(nodes)


def place_split(arr, nn):
    """Pages of row-part k of a [rows, row_bytes] array -> node k. Returns unmovable pages."""
    rows, rb = arr.shape
    base = arr.ctypes.data; bad = 0
    for k in range(nn):
        a, b = rows * k // nn, rows * (k + 1) // nn
        bad += _lib.cpumoe_place(base + a * rb, (b - a) * rb, k)
    return bad


def where(arr, nn=2):
    f = (ctypes.c_double * nn)()
    _lib.cpumoe_where(arr.ctypes.data, arr.nbytes, nn, f)
    return list(f)


def moe_cpu_numa(gs, us, ds, qg, qu, qd, x, w, lim=10.0, out=None):
    E = len(gs); nff = gs[0].shape[0]
    x = np.ascontiguousarray(x, np.float32); w = np.ascontiguousarray(w, np.float32)
    h = np.empty(E * nff, np.float32)
    y = out if out is not None else np.empty(ds[0].shape[0], np.float32)
    _lib.moe_cpu_numa(E, _ptrs(gs), _ptrs(us), _ptrs(ds), int(qg), int(qu), int(qd), nff, x.shape[0],
                      gs[0].shape[1], us[0].shape[1], ds[0].shape[1],
                      x.ctypes.data_as(_F), w.ctypes.data_as(_F), lim, h.ctypes.data_as(_F), y.ctypes.data_as(_F))
    return y


def moe_cpu_numa_t(gs, us, ds, qg, qu, qd, x, w, lim=10.0, out=None):
    """T tokens over the union of their experts. x fp32 [T, Ne]; w fp32 [E, T], 0 where token t
    does not route to expert e. out: optional fp32 [T, Ne]."""
    E, T = w.shape; nff = gs[0].shape[0]
    x = np.ascontiguousarray(x, np.float32); w = np.ascontiguousarray(w, np.float32)
    h = np.empty(E * T * nff, np.float32)
    y = out if out is not None else np.empty((T, ds[0].shape[0]), np.float32)
    _lib.moe_cpu_numa_t(E, T, _ptrs(gs), _ptrs(us), _ptrs(ds), int(qg), int(qu), int(qd), nff, x.shape[1],
                        gs[0].shape[1], us[0].shape[1], ds[0].shape[1],
                        x.ctypes.data_as(_F), w.ctypes.data_as(_F), lim, h.ctypes.data_as(_F), y.ctypes.data_as(_F))
    return y


if _USE_GGML:
    _lib.moe_cpu_numa_t.argtypes = [ctypes.c_int, ctypes.c_int] + _lib.moe_cpu.argtypes[1:]


# ---- one call per layer, no per-call marshaling (cpu_moe_ggml.c: moe_cpu_layer) ----
if _USE_GGML:
    _lib.cpumoe_bind_layer.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_long] * 3
    _lib.moe_cpu_layer.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_float, ctypes.c_void_p]
    _lib.moe_cpu_layer_t.argtypes = [ctypes.c_int, ctypes.c_int] + _lib.moe_cpu_layer.argtypes[1:]


def bind_layer(L, tg, tu, td):
    """tg/tu/td: the layer's gate/up/down expert tensors (GGUF reader tensors, data [288, rows, row_bytes])."""
    g, u, d = tg.data, tu.data, td.data
    _lib.cpumoe_bind_layer(L, g.ctypes.data, u.ctypes.data, d.ctypes.data,
                           int(tg.tensor_type), int(tu.tensor_type), int(td.tensor_type),
                           g.shape[1], d.shape[1], g.shape[2], u.shape[2], d.shape[2])


def moe_cpu_layer(L, mask, sel_addr, w_addr, x_addr, y_addr, lim=10.0):
    """Routed slots j with bit j of mask set: experts sel[j] (int64) with weights w[j], x fp32 [Ne]
    -> y fp32 [Ne]. Arguments are addresses of buffers that outlive the call. Needs numa_setup()."""
    return _lib.moe_cpu_layer(L, mask, sel_addr, w_addr, x_addr, lim, y_addr)


def moe_cpu_layer_t(L, T, mask, sel_addr, w_addr, x_addr, y_addr, lim=10.0):
    """T tokens over the union of their experts (moe_cpu_numa_t on a bound layer): bit 8*t + j of
    mask = compute token t's routed slot j. sel int64 [T, 8], w fp32 [T, 8], x, y fp32 [T, Ne],
    all passed as addresses. Returns the number of distinct experts computed."""
    return _lib.moe_cpu_layer_t(L, T, mask, sel_addr, w_addr, x_addr, lim, y_addr)
