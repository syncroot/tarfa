#!/usr/bin/env python3
import json, sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8089/profile", timeout=10) as response:
    p = json.load(response)
r, io = p["runtime"], p["bf16_io"]
checks = {
    "exact_bf16": r["mode"] == "bf16",
    "pinned_enabled": r["bf16_pinned"] is True,
    "request_profiled": r["forward_calls"] > 0,
    "byte_accounting": io["bytes"] == io["experts"] * 18874368,
    "transfer_measured": io["h2d_seconds"] > 0,
}
report = {"phase": 5, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "profile": p}
print(json.dumps(report, indent=2)); sys.exit(0 if report["status"] == "PASS" else 1)
