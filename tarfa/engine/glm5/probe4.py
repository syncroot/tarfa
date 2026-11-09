"""Layer-3 MLA probe: feed oracle attn_norm-3 (all tokens) through mla_forward, match taps."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import read_dump, cos
from mla_ref import load, mla_forward, Ne, Q_LORA, KV_LORA, H, HEAD_V


def mat(dd, name, rows):
    raw, _ = read_dump(dd, name)
    T = raw.size // rows
    return raw[:rows * T].reshape(T, rows).T  # [rows, T]


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    w = load(gguf_dir, L=3)
    x = mat(dd, "attn_norm-3", Ne)              # [n_embd, T]
    T = x.shape[1]
    print(f"T={T} tokens; indexer inactive (keys {T} << top_k 2048)")
    out, im = mla_forward(x, w)

    for key, oname, rows in [
        ("q_resid", "q_resid-3", Q_LORA),
        ("kv_cmpr", "kv_cmpr-3", KV_LORA),
        ("kqv_out", "kqv_out-3", H * HEAD_V),
        ("attn_out", "attn_out-3", Ne),
    ]:
        if not os.path.exists(os.path.join(dd, oname + ".bin")):
            print(f"{key:10s} (no tap {oname})"); continue
        ref = mat(dd, oname, rows)
        mine = (im[key] if key in im else out)
        m = mine.ravel(); r = ref.ravel()[:m.size]
        print(f"{key:10s} cos={cos(m, r):.6f}  maxdiff={np.max(np.abs(m-r)):.3e}  "
              f"rms m={np.sqrt(np.mean(m**2)):.3e} r={np.sqrt(np.mean(r**2)):.3e}")


if __name__ == "__main__":
    main()
