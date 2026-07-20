#!/usr/bin/env python3
import json, sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8089/profile", timeout=10) as response:
    p = json.load(response)
r, m = p["runtime"], p["moe"]
total = m["bf16_cache_hits"] + m["bf16_cache_misses"]
checks = {
    "exact_bf16": r["mode"] == "bf16",
    "bounded_cache_enabled": r["bf16_cache_per_layer"] > 0,
    "bounded_residency": r["bf16_cached_experts"] <= 48 * r["bf16_cache_per_layer"],
    "decode_cache_accounting": total == r["decode_calls"] * 48 * 8,
    "cache_hits_observed": m["bf16_cache_hits"] > 0,
}
report = {"phase": 4, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "profile": p}
print(json.dumps(report, indent=2)); sys.exit(0 if report["status"] == "PASS" else 1)
