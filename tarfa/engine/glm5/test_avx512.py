"""AVX-512 Q4_K/Q5_K dots vs ggml's, per row; then the expert kernel with each."""
import os, sys, glob, time, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader
import cpu_moe
idx = {}
for s in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
    for t in GGUFReader(s).tensors: idx[t.name] = t
rng = np.random.default_rng(3)
for name in ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_down_exps.weight", "blk.45.ffn_down_exps.weight"):
    t = idx[name]; worst = 0.0
    for _ in range(200):
        e = int(rng.integers(288)); r = int(rng.integers(t.data.shape[1]))
        row = np.ascontiguousarray(t.data[e][r]); n = row.nbytes // (144 if t.tensor_type == 12 else 176) * 256
        x = (rng.standard_normal(n) * 0.05).astype(np.float32)
        a = cpu_moe._lib.cpumoe_dot_test(int(t.tensor_type), n, row.ctypes.data, x.ctypes.data, 0)
        b = cpu_moe._lib.cpumoe_dot_test(int(t.tensor_type), n, row.ctypes.data, x.ctypes.data, 1)
        worst = max(worst, abs(a - b) / (abs(a) + 1e-6))
    print(f"{name} ({t.tensor_type.name}): max rel diff ggml vs avx512 over 200 rows {worst:.1e}", flush=True)
