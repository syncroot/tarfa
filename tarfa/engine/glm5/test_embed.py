import sys, glob, time, numpy as np, gguf
from gguf import GGUFReader
for sh in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
    for t in GGUFReader(sh).tensors:
        if t.name == "token_embd.weight": T = t
for tok in (0, 12089, 154879):
    t0 = time.perf_counter(); ref = gguf.quants.dequantize(np.array(T.data[tok]), T.tensor_type).astype(np.float32).reshape(-1); t1 = time.perf_counter()
    b = np.asarray(T.data[tok]).reshape(-1, 34)
    row = (b[:, 2:].view(np.int8).astype(np.float32) * b[:, :2].copy().view(np.float16).astype(np.float32)).reshape(-1); t2 = time.perf_counter()
    print(tok, "equal" if np.array_equal(ref, row) else f"DIFF {np.abs(ref-row).max()}", f"gguf {1000*(t1-t0):.2f} ms, direct {1000*(t2-t1):.3f} ms")
