"""
glm5-next MoE FFN — numpy reference (DeepSeek-style sigmoid router + shared expert).
From build_moe_ffn + glm5-next.cpp: sigmoid gating, exp_probs_b bias used ONLY for
selection, weights from the pre-bias probs, weights_norm, scale 2.5, top-8 / 288, plus a
dense shared expert added to the MoE output. Experts dequantized per-selected-expert so
only the routed 8 (not all 288) are materialized.
"""
import os, glob
import numpy as np
import gguf
from gguf import GGUFReader, GGMLQuantizationType

Ne = 4096
N_EXPERT = 288
N_USED = 8
N_FF_EXP = 2048
SCALE_W = 2.5
SWIGLU_LIMIT = 10.0   # swiglu_clamp_exp / swiglu_clamp_shexp


def swiglu_clamp(g, u, limit=SWIGLU_LIMIT):
    # gate clamped to [-inf, limit]; up clamped to [-limit, limit]; then silu(gate)*up
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    return silu(g) * u


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x / (1.0 + np.exp(-x))


def _deq(arr, qt):
    if qt == GGMLQuantizationType.F32:
        return np.array(arr, dtype=np.float32)
    return gguf.quants.dequantize(np.array(arr), qt).astype(np.float32)


class MoE:
    def __init__(self):
        self.t = {}   # name suffix -> ReaderTensor (kept alive for lazy mmap slicing)
        self._readers = []

    def load(self, gguf_dir, L=3):
        b = f"blk.{L}."
        want = {b + n for n in (
            "ffn_gate_inp.weight", "exp_probs_b.bias",
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")}
        for sh in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
            r = GGUFReader(sh)
            keep = False
            for t in r.tensors:
                if t.name in want:
                    self.t[t.name[len(b):]] = t
                    keep = True
            if keep:
                self._readers.append(r)
        self.b = b
        self.router = _deq(self.t["ffn_gate_inp.weight"].data, GGMLQuantizationType.F32)  # [288,4096]
        self.bias = _deq(self.t["exp_probs_b.bias"].data, GGMLQuantizationType.F32)    # [288]
        return self

    def _expert(self, key, e):
        t = self.t[key]
        return _deq(t.data[e], t.tensor_type)    # [rows, n_cols]

    def _shared(self, key):
        t = self.t[key]
        return _deq(t.data, t.tensor_type)

    def forward(self, x):
        logits = self.router @ x                 # [288]
        probs = sigmoid(logits)
        sel = np.argsort(probs + self.bias)[-N_USED:]        # selection uses biased probs
        w = probs[sel]                                        # weights from pre-bias probs
        w = w / w.sum() * SCALE_W
        moe = np.zeros(Ne, np.float32)
        for e, we in zip(sel, w):
            g = self._expert("ffn_gate_exps.weight", e) @ x   # [n_ff]
            u = self._expert("ffn_up_exps.weight", e) @ x
            d = self._expert("ffn_down_exps.weight", e) @ swiglu_clamp(g, u)   # [n_embd]
            moe += we * d
        # shared expert (dense)
        gs = self._shared("ffn_gate_shexp.weight") @ x
        us = self._shared("ffn_up_shexp.weight") @ x
        shexp = self._shared("ffn_down_shexp.weight") @ swiglu_clamp(gs, us)
        return {"moe": moe.astype(np.float32), "shexp": shexp.astype(np.float32),
                "out": (moe + shexp).astype(np.float32), "sel": np.sort(sel)}
