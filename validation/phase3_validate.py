#!/usr/bin/env python3
"""Phase-3 gate: validate batched exact-BF16 prefill telemetry."""
import json
import sys
import urllib.request

import _common
p = _common.profile()
_NL, _NE, _BPE = _common.model_shape()
r, m, io = p["runtime"], p["moe"], p["bf16_io"]
batch = r["bf16_prefill_batch"]
checks = {
    "exact_bf16": r["mode"] == "bf16",
    "batching_enabled": batch > 1,
    "prefill_measured": r["prefill_calls"] > 0 and m["prefill_moe_calls"] == r["prefill_calls"] * _NL,
    "batched_reads_observed": io["calls"] < io["experts"],
    "byte_accounting": io["bytes"] == io["experts"] * _BPE,
}
report = {"phase": 3, "status": "PASS" if all(checks.values()) else "FAIL",
          "checks": checks, "profile": p}
print(json.dumps(report, indent=2))
sys.exit(0 if report["status"] == "PASS" else 1)
