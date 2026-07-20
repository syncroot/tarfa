#!/usr/bin/env python3
"""Phase-3 gate: validate batched exact-BF16 prefill telemetry."""
import json
import sys
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8089/profile", timeout=10) as response:
    p = json.load(response)
r, m, io = p["runtime"], p["moe"], p["bf16_io"]
batch = r["bf16_prefill_batch"]
checks = {
    "exact_bf16": r["mode"] == "bf16",
    "batching_enabled": batch > 1,
    "prefill_measured": r["prefill_calls"] > 0 and m["prefill_moe_calls"] == r["prefill_calls"] * 48,
    "batched_reads_observed": io["calls"] < io["experts"],
    "byte_accounting": io["bytes"] == io["experts"] * 18874368,
}
report = {"phase": 3, "status": "PASS" if all(checks.values()) else "FAIL",
          "checks": checks, "profile": p}
print(json.dumps(report, indent=2))
sys.exit(0 if report["status"] == "PASS" else 1)
