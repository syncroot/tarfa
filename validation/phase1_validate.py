#!/usr/bin/env python3
"""Phase-1 acceptance test for Tarfa exact BF16 expert input path."""
import json
import os
import random
import sys
import time

import torch
from safetensors import safe_open

RUNTIME = os.environ.get("TARFA_RUNTIME", os.path.join(os.path.dirname(__file__), "runtime"))
sys.path.insert(0, RUNTIME)
import exact_bf16


def fail(message):
    raise AssertionError(message)


def checkpoint_tensor(layer, projection, expert):
    shard, _offset, shape, dtype = exact_bf16.meta()[layer][projection]
    if dtype != "BF16":
        fail(f"layer {layer} {projection}: expected BF16, found {dtype}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = [k for k in handle.keys() if f"layers.{layer}.mlp.experts.{projection}" in k]
        if len(keys) != 1:
            fail(f"layer {layer} {projection}: expected one checkpoint key, found {keys}")
        value = handle.get_slice(keys[0])[expert]
    if tuple(value.shape) != tuple(shape[1:]):
        fail(f"layer {layer} {projection}: shape mismatch {value.shape} vs {shape[1:]}")
    return value


def main():
    random.seed(122)
    metadata = exact_bf16.meta()
    if sorted(metadata) != list(range(48)):
        fail(f"expected layers 0..47, found {sorted(metadata)}")

    samples = [(0, 0), (47, 255)]
    samples += [(random.randrange(48), random.randrange(256)) for _ in range(4)]
    results = []
    started = time.perf_counter()
    for layer, expert in samples:
        streamed = exact_bf16.read_bf16(layer, [expert], "cpu", torch.bfloat16)[expert]
        for projection, actual in zip(("gate_up_proj", "down_proj"), streamed):
            expected = checkpoint_tensor(layer, projection, expert)
            equal = torch.equal(actual, expected)
            results.append({
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "byte_exact": equal,
            })
            if not equal:
                fail(f"layer {layer} expert {expert} {projection}: streamed bytes differ")

    report = {
        "phase": 1,
        "status": "PASS",
        "mode": {"BF16": 1, "JTOPK": 8, "PIPE": 0, "OVERLAP": 0},
        "layers_indexed": len(metadata),
        "experts_checked": len(samples),
        "tensors_checked": len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checks": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
