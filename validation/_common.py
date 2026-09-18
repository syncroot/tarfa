"""Shared helpers for the phase validators.

These scripts were written against the 122B on the reference machine and hardcoded its
geometry: 48 layers, 256 experts per layer, 18874368 bytes per expert, and port 8089.
On any other checkpoint every derived check fails for the wrong reason. Everything here
is read from the running server or the checkpoint instead.
"""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "tarfa", "engine")
if ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)


def base_url():
    return "http://127.0.0.1:%s" % os.environ.get("TARFA_PORT", "8089")


def profile(timeout=10):
    with urllib.request.urlopen(base_url() + "/profile", timeout=timeout) as r:
        return json.load(r)


def model_shape():
    """(n_layers, experts_per_layer, bytes_per_expert) from the checkpoint."""
    import exact_bf16
    m = exact_bf16.meta()
    lid = sorted(m)[0]
    shg = m[lid]["gate_up_proj"][2]
    shd = m[lid]["down_proj"][2]
    return len(m), shg[0], shg[1] * shg[2] * 2 + shd[1] * shd[2] * 2


def topk():
    return int(os.environ.get("JTOPK", "8"))
