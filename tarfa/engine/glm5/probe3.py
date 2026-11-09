"""Full layer-0 probe: feed oracle hc_init through the mHC+KDA+FFN layer, match every tap."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import read_dump, cos
from glm5_layer import load, layer0_forward, Ne, HC

TAPS = {  # my-key -> (oracle name, per-token element count)
    "attn_norm": ("attn_norm-0", Ne),
    "kda_out": ("kda_out-0", Ne),
    "hc_attn_post": ("hc_attn_post-0", HC * Ne),
    "ffn_norm": ("ffn_norm-0", Ne),
    "ffn_out": ("ffn_out-0", Ne),
    "l_out": ("l_out-0", HC * Ne),
}


def tok0(dd, name, k):
    raw, _ = read_dump(dd, name)
    return raw[:k]


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    w = load(gguf_dir)

    # hc_init: [n_embd, hc, nt] -> token 0 -> [hc, n_embd]
    hc_name = None
    for cand in ("hc_init", "hc_init--1", "hc_init-0"):
        if os.path.exists(os.path.join(dd, cand + ".bin")):
            hc_name = cand; break
    assert hc_name, "no hc_init dump found"
    raw, _ = read_dump(dd, hc_name)
    hc_init = raw[:HC * Ne].reshape(HC, Ne).astype(np.float32)
    print(f"hc_init tap = {hc_name}  stream0==stream1: {np.allclose(hc_init[0], hc_init[1])}")

    out = layer0_forward(hc_init, w)
    for key, (oname, k) in TAPS.items():
        if not os.path.exists(os.path.join(dd, oname + ".bin")):
            print(f"{key:14s} (no oracle tap {oname})"); continue
        ref = tok0(dd, oname, k)
        mine = out[key].ravel()[:k]
        print(f"{key:14s} cos={cos(mine, ref):.6f}  maxdiff={np.max(np.abs(mine-ref)):.3e}  "
              f"rms m={np.sqrt(np.mean(mine**2)):.3e} r={np.sqrt(np.mean(ref**2)):.3e}")


if __name__ == "__main__":
    main()
