#!/usr/bin/env python3
"""Validate and reconcile telemetry from a completed exact-BF16 request."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8089/profile"
BYTES_PER_EXPERT = (2048 * 3072 + 3072 * 1024) * 2

with urllib.request.urlopen(URL, timeout=10) as response:
    profile = json.load(response)

runtime, moe, io = profile["runtime"], profile["moe"], profile["bf16_io"]
checks = {
    "exact_mode_reported": runtime["mode"] == "bf16",
    "request_was_profiled": runtime["forward_calls"] > 0,
    "one_moe_per_layer_forward": moe["moe_calls"] == runtime["forward_calls"] * 48,
    "prefill_decode_partition": moe["moe_calls"] == moe["prefill_moe_calls"] + moe["decode_moe_calls"],
    "expert_bytes_reconcile": io["bytes"] == io["experts"] * BYTES_PER_EXPERT,
    "timings_are_positive": min(runtime["layer_seconds"], moe["moe_seconds"], io["read_seconds"], io["h2d_seconds"]) > 0,
    "moe_within_layer_time": moe["moe_seconds"] <= runtime["layer_seconds"] * 1.01,
}
report = {"phase": 2, "status": "PASS" if all(checks.values()) else "FAIL",
          "checks": checks, "profile": profile}
print(json.dumps(report, indent=2))
sys.exit(0 if report["status"] == "PASS" else 1)
