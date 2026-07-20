#!/usr/bin/env python3
import json, sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8089/profile", timeout=10) as response:
    p = json.load(response)
r, route, system = p["runtime"], p["routing"], p["system"]
rates = route["reuse_rates"]
checks = {
    "exact_bf16": r["mode"] == "bf16",
    "long_decode_observed": r["decode_calls"] >= 8,
    "routing_accounted": route["expert_uses"] == r["decode_calls"] * 48 * 8,
    "reuse_monotonic": rates["1"] <= rates["2"] <= rates["4"] <= rates["8"] <= rates["16"],
    "memory_observed": system["mem_available_gb"] > 0,
    "io_observed": system["process_io"]["rchar"] > 0,
}
report = {"phase": 6, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "profile": p}
print(json.dumps(report, indent=2)); sys.exit(0 if report["status"] == "PASS" else 1)
