#!/usr/bin/env python3
import json, sys, urllib.request
import _common
p = _common.profile()
_NL, _NE, _BPE = _common.model_shape()
_TOPK = _common.topk()
r, m = p["runtime"], p["moe"]
total = m["bf16_cache_hits"] + m["bf16_cache_misses"]
checks = {
    "exact_bf16": r["mode"] == "bf16",
    "bounded_cache_enabled": r["bf16_cache_per_layer"] > 0,
    "bounded_residency": r["bf16_cached_experts"] <= _NL * r["bf16_cache_per_layer"],
    "decode_cache_accounting": total == r["decode_calls"] * _NL * _TOPK,
    "cache_hits_observed": m["bf16_cache_hits"] > 0,
}
report = {"phase": 4, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "profile": p}
print(json.dumps(report, indent=2)); sys.exit(0 if report["status"] == "PASS" else 1)
