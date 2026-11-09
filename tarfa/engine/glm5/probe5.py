"""MoE FFN probe (layer 3, token 0): feed oracle ffn_norm-3 through moe_ref, match taps."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import read_dump, cos
from moe_ref import MoE, Ne


def tok0(dd, name):
    raw, _ = read_dump(dd, name)
    return raw[:Ne]


def main():
    gguf_dir, dd = sys.argv[1], sys.argv[2]
    m = MoE().load(gguf_dir, L=3)
    x = tok0(dd, "ffn_norm-3").astype(np.float32)   # MoE input (token 0)
    r = m.forward(x)
    print("selected experts:", r["sel"].tolist())
    for key, oname in [("moe", "ffn_moe_out-3"), ("shexp", "ffn_shexp-3"), ("out", "ffn_out-3")]:
        if not os.path.exists(os.path.join(dd, oname + ".bin")):
            print(f"{key:6s} (no tap {oname})"); continue
        ref = tok0(dd, oname)
        mine = r[key]
        print(f"{key:6s} cos={cos(mine, ref):.6f}  maxdiff={np.max(np.abs(mine-ref)):.3e}  "
              f"rms m={np.sqrt(np.mean(mine**2)):.3e} r={np.sqrt(np.mean(ref**2)):.3e}")


if __name__ == "__main__":
    main()
