"""Bit-identity of the bound per-layer CPU calls (moe_cpu_layer, moe_cpu_layer_t) against the
wrappers they replace (moe_cpu_numa, moe_cpu_numa_t), on real expert bytes. No GPU needed.
Usage: python check_cpu_layer.py <gguf_dir> [layers, default 3,20,44,45] [experts per layer, default 24]"""
import os, sys, glob, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
import cpu_moe

d = sys.argv[1]
LAYERS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "3,20,44,45").split(",")]
NPOOL = int(sys.argv[3]) if len(sys.argv) > 3 else 24
K = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
idx = {}
for s in sorted(glob.glob(d + "/*.gguf")):
    for t in GGUFReader(s).tensors:
        if "_exps" in t.name:
            idx[t.name] = t
cpu_moe.numa_setup()
ts = {L: [idx[f"blk.{L}.{k}"] for k in K] for L in LAYERS}
random.seed(5); rs = np.random.RandomState(5)
pool = {L: sorted(random.sample(range(288), NPOOL)) for L in LAYERS}
for L in LAYERS:
    cpu_moe.bind_layer(L, *ts[L])
bits = lambda a: a.view(np.uint32)
bad1 = n1 = bad2 = n2 = 0
for it in range(40):                                    # one token: mask over the 8 routed slots
    L = random.choice(LAYERS)
    sel = np.array(random.sample(pool[L], 8), np.int64); w = rs.rand(8).astype(np.float32)
    x = (rs.randn(4096) * 0.3).astype(np.float32)
    mask = 255 if it % 4 == 3 else random.getrandbits(8)
    js = [j for j in range(8) if mask >> j & 1]
    out = np.full(4096, 7.0, np.float32)
    n = cpu_moe.moe_cpu_layer(L, mask, sel.ctypes.data, w.ctypes.data, x.ctypes.data, out.ctypes.data)
    if not js:
        bad1 += n != 0; continue
    es = [int(sel[j]) for j in js]
    ref = cpu_moe.moe_cpu([ts[L][0].data[e] for e in es], [ts[L][1].data[e] for e in es], [ts[L][2].data[e] for e in es],
                          *[t.tensor_type for t in ts[L]], x, w[js].copy())
    n1 += 1; bad1 += n != len(es) or not np.array_equal(bits(ref), bits(out))
for it in range(80):                                    # T tokens over the union of their experts
    L = random.choice(LAYERS); T = 1 if it % 5 == 4 else 2
    sel = np.array([random.sample(pool[L], 8) for _ in range(T)], np.int64)
    if T == 2 and it % 3 == 0:                          # experts shared by both tokens, in other slots
        sel[1, :4] = sel[0, [5, 1, 7, 2]]
        sel[1, 4:] = random.sample([e for e in pool[L] if e not in sel[1, :4]], 4)
    w = rs.rand(T, 8).astype(np.float32)
    x = (rs.randn(T, 4096) * 0.3).astype(np.float32)
    mask = (1 << 8 * T) - 1 if it % 4 == 3 else random.getrandbits(8 * T)
    uni = {}                                            # as glm5_spec.verify / mtp.draft_g build it
    for t in range(T):
        for j, e in enumerate(sel[t].tolist()):
            if mask >> (8 * t + j) & 1:
                uni.setdefault(e, np.zeros(T, np.float32))[t] = w[t, j]
    out = np.full((T, 4096), 7.0, np.float32)
    n = cpu_moe.moe_cpu_layer_t(L, T, mask, sel.ctypes.data, w.ctypes.data, x.ctypes.data, out.ctypes.data)
    if not uni:
        bad2 += n != 0; continue
    es = list(uni)
    ref = cpu_moe.moe_cpu_numa_t([ts[L][0].data[e] for e in es], [ts[L][1].data[e] for e in es],
                                 [ts[L][2].data[e] for e in es], *[t.tensor_type for t in ts[L]], x,
                                 np.stack([uni[e] for e in es]))
    n2 += 1; bad2 += n != len(es) or not np.array_equal(bits(ref), bits(out))
print(f"moe_cpu_layer   vs moe_cpu_numa:   {n1} calls, {'bit-identical' if not bad1 else f'{bad1} MISMATCHES'}")
print(f"moe_cpu_layer_t vs moe_cpu_numa_t: {n2} calls (1 and 2 tokens, shared experts, partial masks), "
      f"{'bit-identical' if not bad2 else f'{bad2} MISMATCHES'}")
sys.exit(1 if bad1 or bad2 else 0)
