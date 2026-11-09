"""Long-context indexer probe: validate the k-pool score + top-k selection (the one
untested path). Uses oracle indexer intermediates (iq, pooled_k, weights) to recompute
score[pool] per the lightning-indexer formula, and checks the top-512 pool selection.
score[p,t] = sum_h relu(iq[:,h,t] . pooled_k[:,p]) * weights[h,t], + causal pool mask."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_layer0 import read_dump, cos

KPOOL = 4
N_TOPPOOL = 512


def rd(dd, name, shape):
    raw, _ = read_dump(dd, name)
    n = int(np.prod(shape))
    return raw[:n].reshape(shape)


def main():
    dd = sys.argv[1]
    # shapes are numpy = reversed ne
    # iq ne[128,32,T] ; pooled ne[128,1,P] ; weights ne[32,T] ; score ne[P,T] ; topk ne[512,T]
    sc_raw, sc_ne = read_dump(dd, "indexer_score-3")
    import re
    # read T and P from the score meta
    meta = open(os.path.join(dd, "indexer_score-3.meta")).read()
    P, T = [int(x) for x in [p for p in meta.split() if p.startswith("ne=")][0][3:].split(",")[:2]]
    print(f"P(pools)={P}  T(tokens)={T}  n_top_pool={N_TOPPOOL}  -> pruning {'ACTIVE' if P>N_TOPPOOL else 'inactive'}")

    iq = rd(dd, "indexer_q-3", (T, 32, 128))
    pooled = rd(dd, "indexer_pool_k-3", (P, 128))
    weights = rd(dd, "indexer_weights-3", (T, 32))
    score_ref = rd(dd, "indexer_score-3", (T, P))
    topk_ref = np.fromfile(os.path.join(dd, "indexer_top_k-3.bin"), dtype=np.int32)[:T * N_TOPPOOL].reshape(T, N_TOPPOOL).astype(np.int64)

    for t in (T - 1, T // 2):
        K = pooled @ iq[t].T                 # [P, 32]  pooled_k . iq per head
        s = np.maximum(K, 0.0) * weights[t][None, :]
        s = s.sum(axis=1)                    # [P]
        mask = (np.arange(P) * KPOOL) > t    # causal: pool's first token > query -> hidden
        s = np.where(mask, -np.inf, s)
        ref = score_ref[t].copy()
        fin = np.isfinite(ref) & np.isfinite(s)
        c = cos(s[fin], ref[fin])
        mine_top = set(np.argsort(np.where(np.isfinite(s), s, -np.inf))[-N_TOPPOOL:].tolist())
        ref_top = set(int(x) for x in topk_ref[t].tolist() if 0 <= x < P)
        overlap = len(mine_top & ref_top) / max(1, len(ref_top))
        print(f"  tok {t}: score cos={c:.6f} (over {fin.sum()} visible pools)  "
              f"top-{N_TOPPOOL} overlap={overlap:.4f}  (ref kept {len(ref_top)})")


if __name__ == "__main__":
    main()
