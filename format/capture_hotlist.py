#!/usr/bin/env python3
"""Export measured routing hotlists and theoretical memory/coverage tradeoffs."""
import hashlib, json, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8089"
reports = {}
for k in (1, 2, 4, 8):
    with urllib.request.urlopen(f"{BASE}/routing-hotlist?k={k}", timeout=20) as response:
        reports[str(k)] = json.load(response)
expert_bytes = 18874368
summary = {k: {"coverage": value["coverage"], "uses": value["uses"],
               "gpu_gb": round(48 * int(k) * expert_bytes / 1e9, 3)} for k, value in reports.items()}
out = {"format": "tarfa-routing-hotlist-v1", "summary": summary,
       "corpus_fingerprint": hashlib.sha256(json.dumps(reports["8"], sort_keys=True).encode()).hexdigest(),
       "hotlist_k8": reports["8"]}
target = Path(__file__).with_name("hotlist_capture.json"); target.write_text(json.dumps(out, indent=2))
print(json.dumps({"status": "PASS", "path": str(target), "summary": summary}, indent=2))
