#!/usr/bin/env python3
"""Print SHA-256 hashes for the authoritative Tarfa runtime."""
import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent / "runtime"
files = sorted(p for p in root.iterdir() if p.is_file())
print(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
