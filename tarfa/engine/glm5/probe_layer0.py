"""
Layer-0 KDA probe: compare kda_ref against the llama.cpp activation oracle.

Oracle: patched llama-eval-callback dumps full tensors to $GLM5_DUMP_DIR as
<name>.bin (+ <name>.meta). I feed the oracle's `attn_norm-0` tap into
kda_decode_step and compare to the oracle's `kda_out-0` tap. Fresh conv+recurrent
state (single-token prompt), so this is an isolated unit test of build_kda_layer.

Usage: python probe_layer0.py <gguf_dir> <dump_dir>
"""
import sys, glob, os
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType

H, Dh, Ne = 64, 128, 4096
Di = H * Dh

# ggml tensor name -> KDAWeights attribute
NAME_MAP = {
    "blk.0.attn_q.weight": "wq", "blk.0.attn_k.weight": "wk", "blk.0.attn_v.weight": "wv",
    "blk.0.ssm_conv1d_q.weight": "q_conv", "blk.0.ssm_conv1d_k.weight": "k_conv",
    "blk.0.ssm_conv1d_v.weight": "v_conv",
    "blk.0.ssm_f_a.weight": "ssm_f_a", "blk.0.ssm_f_b.weight": "ssm_f_b",
    "blk.0.ssm_dt.bias": "ssm_dt_b", "blk.0.ssm_a": "ssm_a",
    "blk.0.ssm_beta.weight": "ssm_beta",
    "blk.0.ssm_g_a.weight": "ssm_g_a", "blk.0.ssm_g_b.weight": "ssm_g_b",
    "blk.0.ssm_norm.weight": "ssm_o_norm", "blk.0.attn_output.weight": "wo",
}


# logical numpy (out, in) for each matmul weight; conv/vectors handled separately.
SHAPES = {
    "wq": (Di, Ne), "wk": (Di, Ne), "wv": (Di, Ne),
    "ssm_f_a": (Dh, Ne), "ssm_f_b": (Di, Dh),
    "ssm_beta": (H, Ne), "ssm_g_a": (Dh, Ne), "ssm_g_b": (Di, Dh),
    "wo": (Ne, Di),
}
CONV = {"q_conv", "k_conv", "v_conv"}
VEC = {"ssm_dt_b": (Di,), "ssm_a": (H,), "ssm_o_norm": (Dh,)}


def deq_flat(t):
    # ggml memory is ne0-fastest; dequantize preserves that order. Return a flat f32 array.
    if t.tensor_type == GGMLQuantizationType.F32:
        a = np.array(t.data, dtype=np.float32)
    else:
        a = gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
    return a.reshape(-1)


def load_weights(gguf_dir):
    from kda_ref import KDAWeights
    W = KDAWeights()
    need = set(NAME_MAP)
    for sh in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
        r = GGUFReader(sh)
        for t in r.tensors:
            if t.name not in need:
                continue
            attr = NAME_MAP[t.name]
            flat = deq_flat(t)
            if attr in SHAPES:
                a = flat.reshape(SHAPES[attr])          # (out, in), ne0=in fastest -> row-major (out,in)
            elif attr in CONV:
                a = flat.reshape(Di, -1).T              # (d_inner, d_conv) -> (d_conv, d_inner)
            else:
                a = flat.reshape(VEC[attr])
            setattr(W, attr, a)
            need.discard(t.name)
    assert not need, f"missing tensors: {need}"
    return W


def read_dump(dump_dir, name):
    meta = open(os.path.join(dump_dir, name + ".meta")).read()
    kv = dict(p.split("=", 1) for p in meta.split()[:1] + [x for x in meta.split() if "=" in x])
    ty = int(kv["type"]); ne = [int(x) for x in kv["ne"].split(",")]
    dt = {0: np.float32, 1: np.float16}.get(ty)
    assert dt is not None, f"{name}: unhandled dump dtype {ty}"
    raw = np.fromfile(os.path.join(dump_dir, name + ".bin"), dtype=dt).astype(np.float32)
    n = 1
    for d in ne:
        n *= d if d else 1
    return raw[:n], ne


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    gguf_dir, dump_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kda_ref import kda_decode_step

    W = load_weights(gguf_dir)
    x, _ = read_dump(dump_dir, "attn_norm-0")          # [n_embd, n_tokens] flat, n_embd fastest
    x = x[:Ne]                                          # token 0 (fresh conv+recurrent state)
    cs = {k: np.zeros((3, Di), np.float32) for k in "qkv"}
    s = np.zeros((H, Dh, Dh), np.float32)
    out, _, _ = kda_decode_step(x.astype(np.float32), W, cs, s)

    ref, ne = read_dump(dump_dir, "kda_out-0")          # [n_embd, n_tokens]
    ref = ref[:Ne]                                       # token 0
    print(f"kda_out: oracle ne={ne}  mine={out.shape}")
    print(f"  cosine     = {cos(out, ref):.6f}")
    print(f"  max|diff|  = {np.max(np.abs(out - ref)):.4e}")
    print(f"  rms(mine)  = {np.sqrt(np.mean(out**2)):.4e}  rms(oracle) = {np.sqrt(np.mean(ref**2)):.4e}")

    # optional: isolate the conv by checking the q-after-conv tap if present
    qc = os.path.join(dump_dir, "kda_q_conv-0.bin")
    if os.path.exists(qc):
        from kda_ref import conv_branch
        q_mine, _ = conv_branch(x.astype(np.float32), W.wq, W.q_conv, cs["q"], H, Dh)
        q_ref, _ = read_dump(dump_dir, "kda_q_conv-0")
        q_ref = q_ref[:Di]                               # token 0
        print(f"kda_q_conv: cosine = {cos(q_mine.ravel(), q_ref):.6f}  "
              f"max|diff| = {np.max(np.abs(q_mine.ravel() - q_ref)):.4e}  "
              f"rms mine={np.sqrt(np.mean(q_mine**2)):.4e} ref={np.sqrt(np.mean(q_ref**2)):.4e}")


if __name__ == "__main__":
    main()
