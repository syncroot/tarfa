#!/usr/bin/env python3
import json, sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8089/profile", timeout=10) as response: p=json.load(response)
r, c, s = p["runtime"], p["ram_cache"], p["system"]
checks={"exact_bf16":r["mode"]=="bf16", "arena_enabled":c["enabled"],
        "bounded":c["resident"]<=c["slots"], "hits":c["hits"]>0,
        "bytes_avoided":c["avoided_bytes"]>0, "memory_headroom":s["mem_available_gb"]>=4,
        "no_gpu_cache":r["bf16_cache_per_layer"]==0}
report={"phase":7,"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"profile":p}
print(json.dumps(report,indent=2));sys.exit(0 if report["status"]=="PASS" else 1)
