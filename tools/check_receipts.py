#!/usr/bin/env python3
"""Fail if any performance number in the public docs drifts from the committed receipts.

Every tok/s, s/tok, joule, watt, utilization and phase-12 figure in the docs must be derivable
from validation/*.json. Run: python tools/check_receipts.py  (CI runs it on every push/PR).
"""
import json, pathlib, re, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = ["README.md", "EXACTNESS.md", "ARCHITECTURE.md", "BENCHMARKS.md", "docs/STORY.md"]
V = REPO / "validation"
exact = json.loads((V / "bench_exact_bf16.json").read_text())
fast = json.loads((V / "bench_fast_int4.json").read_text())
power = json.loads((V / "power_metrics.json").read_text())
p12 = json.loads((V / "phase12/completion_results.json").read_text())

# Documented spread of fast-mode runs (BENCHMARKS.md section 1). Not a receipt; declared here so a
# change to it is a reviewed diff, never a silent edit.
FAST_RUN_RANGE = (1.4, 1.8)
HARDWARE = {"W": {160.0}}  # card power limit, a spec not a measurement

def r(x, n): return round(float(x), n)

allowed = {
    "tok/s": {r(exact["tok_per_s"], 2), r(fast["tok_per_s"], 2), *FAST_RUN_RANGE},
    "s/tok": {r(exact["s_per_tok_mean"], 2), r(fast["s_per_tok_mean"], 2)},
    "J": {r(power["exact"]["j_per_tok"], 1), r(power["fast"]["j_per_tok"], 1)},
    "W": {r(power["exact"]["watts"], 1), r(power["fast"]["watts"], 1),
          r(power["exact"]["watts"], 0), r(power["fast"]["watts"], 0), *HARDWARE["W"]},
    "%util": {r(power["exact"]["util"], 0), r(power["fast"]["util"], 0)},
}
p12_vals = {r(p12["median_serial_s"], 2), r(p12["median_completion_s"], 2)}
p12_pct = r(p12["improvement"] * 100, 1)

UNIT = {"tok/s": "tok/s", "tokens/sec": "tok/s", "tokens per second": "tok/s",
        "s/tok": "s/tok", "s/token": "s/tok", "seconds per token": "s/tok",
        "J": "J", "joules": "J", "W": "W"}
NUM = r"(\d+(?:\.\d+)?)"
CLAIM = re.compile(NUM + r"(?:\s?(?:–|-|to)\s?" + NUM + r")?\s?(tok/s|tokens/sec|tokens per second|"
                   r"s/tok|s/token|seconds per token|joules|J\b|W\b)")
UTIL = re.compile(r"~?" + NUM + r"% (?:GPU )?util")
P12 = re.compile(r"2\.\d\d\s?(?:→|to)\s?2\.\d\d")
# Strings that once contradicted the receipts; they must never come back.
BANNED = ["2–3 tok/s", "2-3 tok/s", "~0.34", "2.63 s/tok", "(−11%)", "Private preview"]

errors = []
def bad(f, n, msg): errors.append(f"{f}:{n}: {msg}")

for f in DOCS:
    for n, line in enumerate((REPO / f).read_text().splitlines(), 1):
        for s in BANNED:
            if s in line: bad(f, n, f"banned stale claim {s!r}")
        for m in CLAIM.finditer(line):
            unit = UNIT[m.group(3)]
            for v in filter(None, m.group(1, 2)):
                if r(v, 2) not in allowed[unit] and r(v, 1) not in allowed[unit]:
                    bad(f, n, f"{v} {unit} not in receipts {sorted(allowed[unit])}")
        for m in UTIL.finditer(line):
            if r(m.group(1), 0) not in allowed["%util"]:
                bad(f, n, f"{m.group(1)}% util not in receipts {sorted(allowed['%util'])}")
        for m in P12.finditer(line):
            if {r(x, 2) for x in re.findall(r"2\.\d\d", m.group(0))} != p12_vals:
                bad(f, n, f"phase-12 figure {m.group(0)!r} != receipt {sorted(p12_vals, reverse=True)}")
        if "Phase 12" in line or "phase 12" in line.lower() or "| 12 |" in line:
            for pct in re.findall(r"−?(\d+(?:\.\d+)?)%", line):
                if r(pct, 1) != p12_pct: bad(f, n, f"phase-12 {pct}% != receipt {p12_pct}%")

# Table cells must carry the receipt values, not just avoid wrong ones.
def must(f, needle):
    if needle not in (REPO / f).read_text(): bad(f, 0, f"missing receipt value {needle!r}")
for f in ("README.md", "BENCHMARKS.md"):
    must(f, f"{r(exact['tok_per_s'], 2)} tok/s"); must(f, f"{r(fast['tok_per_s'], 2)} tok/s")

if errors:
    print("Docs drifted from validation/ receipts:\n  " + "\n  ".join(errors)); sys.exit(1)
print(f"OK: {len(DOCS)} docs consistent with receipts "
      f"(exact {exact['tok_per_s']}, fast {fast['tok_per_s']} tok/s; p12 {p12_pct}%)")
