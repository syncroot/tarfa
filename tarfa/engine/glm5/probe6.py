"""Multi-token KDA probe: run the KDA recurrence sequentially over all tokens (carrying
conv + recurrent state) and compare each position's scan output to the oracle. Validates
the gated-delta recurrence at positions >0 (probe2 only covered position 0 / fresh state)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import load_weights, read_dump, cos, Ne, Di, H, Dh
from kda_ref import kda_decode_step


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    w = load_weights(gguf_dir)
    an, _ = read_dump(dd, "attn_norm-0")
    T = an.size // Ne
    x = an[:T * Ne].reshape(T, Ne).T                    # [n_embd, T]
    ko, _ = read_dump(dd, "kda_out-0")                  # [n_embd, T] full KDA block output
    cs = {c: np.zeros((3, Di), np.float32) for c in "qkv"}
    s = np.zeros((H, Dh, Dh), np.float32)
    print(f"T={T} tokens; KDA recurrence sequential, comparing full block output per position:")
    for t in range(T):
        o, cs, s = kda_decode_step(x[:, t].astype(np.float32), w, cs, s)
        ref = ko[t * Ne:(t + 1) * Ne]
        print(f"  pos {t}: cos={cos(o.ravel(), ref):.6f}  maxdiff={np.max(np.abs(o.ravel()-ref)):.3e}  "
              f"rms m={np.sqrt(np.mean(o**2)):.3e} r={np.sqrt(np.mean(ref**2)):.3e}")


if __name__ == "__main__":
    main()
