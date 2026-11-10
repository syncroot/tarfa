#!/usr/bin/env python3
"""Fail if any performance number in the public docs drifts from the committed receipts.

Every tok/s, s/tok, joule, watt, utilization and phase-12 figure in the docs must be derivable
from validation/*.json. The experimental GLM-5.3-Flash engine has its own, stricter checks below:
in tarfa/engine/glm5/README.md and README.md's GLM section every tok/s figure, run count, range,
speedup ratio and the draft acceptance is tied to its engine's runs in validation/glm5/bench_story.json,
and the results tables may hold only the known rows. Other numbers there (hardware, timings, cosines)
and the dated lab notes next to that engine (PERFORMANCE.md, KERNELS.md, VERDICT.md) are not checked. Run: python tools/check_receipts.py  (CI runs it on every push/PR).
"""
import json, pathlib, re, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = ["README.md", "EXACTNESS.md", "ARCHITECTURE.md", "BENCHMARKS.md", "docs/STORY.md"]
V = REPO / "validation"
exact = json.loads((V / "bench_exact_bf16.json").read_text())
exact_jul = json.loads((V / "history/2026-07-20/bench_exact_bf16.json").read_text())  # dated history
fast = json.loads((V / "bench_fast_int4.json").read_text())
power = json.loads((V / "power_metrics.json").read_text())
p12 = json.loads((V / "phase12/completion_results.json").read_text())
glm5 = json.loads((V / "glm5/bench_story.json").read_text())

# Documented spread of fast-mode runs (BENCHMARKS.md section 1). Not a receipt; declared here so a
# change to it is a reviewed diff, never a silent edit.
FAST_RUN_RANGE = (1.4, 1.8)
HARDWARE = {"W": {160.0}}  # card power limit, a spec not a measurement

def r(x, n): return round(float(x), n)

allowed = {
    "tok/s": {r(exact["tok_per_s"], 2), r(exact_jul["tok_per_s"], 2), r(fast["tok_per_s"], 2), *FAST_RUN_RANGE},
    "s/tok": {r(exact["s_per_tok_mean"], 2), r(exact_jul["s_per_tok_mean"], 2), r(fast["s_per_tok_mean"], 2)},
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

GLM5_DOC = "tarfa/engine/glm5/README.md"
def glm5_section(f, lines):
    """Line numbers of README.md's GLM-5.3-Flash section; the GLM checks below cover them."""
    if f != "README.md": return set()
    out, inside = set(), False
    for n, line in enumerate(lines, 1):
        if line.startswith("## "): inside = line.startswith("## Experimental: GLM-5.3-Flash")
        if inside: out.add(n)
    return out

for f in DOCS:
    lines = (REPO / f).read_text().splitlines()
    skip = glm5_section(f, lines)
    for n, line in enumerate(lines, 1):
        if n in skip: continue
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

# ---- GLM-5.3-Flash engine (experimental) ----
GJ = "validation/glm5/bench_story.json"
ROWS = {"llamacpp": "| llama.cpp, production flags (", "tarfa_plain": "| Tarfa, plain decode (",
        "tarfa_mtp": "| Tarfa, MTP speculative decode ("}
old = glm5["session_2026_11_07"]                     # superseded first comparison, its own table
OLD_ROWS = {k: v.rstrip(" (") + ", 2026-11-07" for k, v in ROWS.items()}
prev = glm5["session_2026_11_09"]                    # previous comparison (older engine), its own table
PREV_ROWS = {k: v.rstrip(" (") + ", 2026-11-09" for k, v in ROWS.items()}
best = {}
for k in ROWS:
    runs = glm5[k]["runs_tok_per_s"]
    best[k] = max(runs)
    if glm5[k]["best_tok_per_s"] != best[k]: bad(GJ, 0, f"{k}: best_tok_per_s is not the max of its runs")
chk = glm5["checks_2026_11_09"]
CHECK_ROWS = {"tarfa_plain": "| Tarfa, plain decode, check", "tarfa_mtp": "| Tarfa, MTP speculative decode, check"}
mtp_sets = [glm5["tarfa_mtp"], chk["tarfa_mtp"], prev["tarfa_mtp"], old["tarfa_mtp"]]
for m in mtp_sets:
    if not (len(m["runs_tok_per_s"]) == len(m["verify_steps"]) == len(m["drafts_accepted"])):
        bad(GJ, 0, "tarfa_mtp: runs, verify_steps and drafts_accepted differ in length")
fmt = lambda v: f"{v:.2f}"
runs_2dp = ({fmt(v) for k in ROWS for v in glm5[k]["runs_tok_per_s"]}
            | {fmt(v) for k in CHECK_ROWS for v in chk[k]["runs_tok_per_s"]}
            | {fmt(v) for k in PREV_ROWS for v in prev[k]["runs_tok_per_s"]}
            | {fmt(v) for k in OLD_ROWS for v in old[k]["runs_tok_per_s"]})
# Tarfa's run 1 is its first start on an evicted cache; the ratios span the starts after it (slowest, best).
def spread(k, base): return tuple(f"{f(glm5[k]['runs_tok_per_s'][1:]) / base:.2f}" for f in (min, max))
ratio = spread("tarfa_plain", best["llamacpp"]) + spread("tarfa_mtp", best["llamacpp"])
prev_best = max(prev["llamacpp"]["runs_tok_per_s"])  # the same starts against the previous session's llama.cpp best
cross = spread("tarfa_plain", prev_best) + spread("tarfa_mtp", prev_best)
prev_ratio = {k: f"{max(prev[k]['runs_tok_per_s']) / prev_best:.1f}" for k in ("tarfa_plain", "tarfa_mtp")}
old_ratio = {k: f"{max(old[k]['runs_tok_per_s']) / max(old['llamacpp']['runs_tok_per_s']):.1f}" for k in ("tarfa_plain", "tarfa_mtp")}
accept = round(100 * sum(sum(m["drafts_accepted"]) for m in mtp_sets) / sum(sum(m["verify_steps"]) for m in mtp_sets))

g = (REPO / GLM5_DOC).read_text().splitlines()
for k, label in ROWS.items():                        # each table row carries its own engine's figures
    runs = glm5[k]["runs_tok_per_s"]
    want = f"| {fmt(max(runs))} tok/s | {fmt(min(runs))}–{fmt(max(runs))} tok/s, {len(runs)} runs |"
    hits = [n for n, line in enumerate(g, 1) if line.startswith(label)]
    if len(hits) != 1 or not g[hits[0] - 1].endswith(want):
        bad(GLM5_DOC, hits[0] if hits else 0, f"{label[2:]} row must end with {want!r}")
EARLIER = ([(prev[k]["runs_tok_per_s"], label) for k, label in PREV_ROWS.items()]
           + [(old[k]["runs_tok_per_s"], label) for k, label in OLD_ROWS.items()])
for runs, label in EARLIER:                          # the earlier sessions, same row format
    want = f"| {fmt(max(runs))} tok/s | {fmt(min(runs))}–{fmt(max(runs))} tok/s, {len(runs)} runs |"
    hits = [n for n, line in enumerate(g, 1) if line.startswith(label)]
    if len(hits) != 1 or not g[hits[0] - 1].endswith(want):
        bad(GLM5_DOC, hits[0] if hits else 0, f"{label[2:]} row must end with {want!r}")
def span(runs):
    rng = fmt(runs[0]) if len(runs) == 1 else f"{fmt(min(runs))}–{fmt(max(runs))}"
    return f"{rng} tok/s, {len(runs)} run{'s' if len(runs) > 1 else ''}"
for k, label in CHECK_ROWS.items():                  # check runs: their own table, their own runs
    want = f"| {span(chk[k]['runs_tok_per_s'])} |"
    hits = [n for n, line in enumerate(g, 1) if line.startswith(label)]
    if len(hits) != 1 or not g[hits[0] - 1].endswith(want):
        bad(GLM5_DOC, hits[0] if hits else 0, f"{label[2:]} row must end with {want!r}")
known = tuple(ROWS.values()) + tuple(PREV_ROWS.values()) + tuple(OLD_ROWS.values()) + tuple(CHECK_ROWS.values())
for n, line in enumerate(g, 1):                     # results tables hold only the known rows
    if line.startswith("|") and CLAIM.search(line) and not line.startswith(known):
        bad(GLM5_DOC, n, "unknown results row")
for n, line in enumerate(g, 1):                     # tok/s figures appear only in the table rows
    if not line.startswith("|") and CLAIM.search(line):
        bad(GLM5_DOC, n, "tok/s figure outside the results table")
    for c in CLAIM.finditer(line):
        if UNIT[c.group(3)] != "tok/s": bad(GLM5_DOC, n, f"unexpected unit {c.group(3)!r}"); continue
        for v in filter(None, c.group(1, 2)):
            if v not in runs_2dp: bad(GLM5_DOC, n, f"{v} tok/s is not a run in {GJ} (two decimals)")
    if not line.startswith("|") and re.search(r"\d+ runs?\b", line):
        bad(GLM5_DOC, n, "run counts belong in the results tables")
gtext = " ".join(g)
RATIO_SENTENCE = re.compile(r"([\d.]+)–([\d.]+)× llama\.cpp with plain decode, ([\d.]+)–([\d.]+)× with MTP")
CROSS_SENTENCE = re.compile(r"against it, the 2026-11-10 Tarfa starts above are ([\d.]+)–([\d.]+)× and ([\d.]+)–([\d.]+)×")
PREV_RATIO = re.compile(r"\(2026-11-09\) gave ([\d.]+)× and ([\d.]+)×")
OTHER_X = ("2× Xeon", "~1.7× the CPU experts")       # not speedups: hardware count, expert-union size
OLD_RATIO = re.compile(r"\(2026-11-07\) gave ([\d.]+)× and ([\d.]+)×")
c = OLD_RATIO.search(gtext)
if not c or c.groups() != (old_ratio["tarfa_plain"], old_ratio["tarfa_mtp"]):
    bad(GLM5_DOC, 0, f"earlier-session ratios must read {old_ratio['tarfa_plain']}× and {old_ratio['tarfa_mtp']}×")
c = PREV_RATIO.search(gtext)
if not c or c.groups() != (prev_ratio["tarfa_plain"], prev_ratio["tarfa_mtp"]):
    bad(GLM5_DOC, 0, f"previous-session ratios must read {prev_ratio['tarfa_plain']}× and {prev_ratio['tarfa_mtp']}×")
c = CROSS_SENTENCE.search(gtext)
if not c or c.groups() != cross:
    bad(GLM5_DOC, 0, "ratios against the previous llama.cpp best must read %s–%s× and %s–%s×" % cross)
rest = gtext
for pat in (RATIO_SENTENCE, CROSS_SENTENCE, PREV_RATIO, OLD_RATIO): rest = pat.sub("", rest)
for x in OTHER_X: rest = rest.replace(x, "")
if re.search(r"\d+(?:\.\d+)?\s?[×x](?=\W)", rest):
    bad(GLM5_DOC, 0, "speedup figures are allowed only in the ratio sentences")
acc = re.findall(r"(\d+)%\s+of drafts", gtext)
if not acc or any(int(x) != accept for x in acc):
    bad(GLM5_DOC, 0, f"draft acceptance {acc} != receipt {accept}%")
c = RATIO_SENTENCE.search(gtext)
if not c or c.groups() != ratio:
    bad(GLM5_DOC, 0, "ratios must read %s–%s× plain, %s–%s× with MTP" % ratio)

lines = (REPO / "README.md").read_text().splitlines()
sec = " ".join(lines[n - 1] for n in sorted(glm5_section("README.md", lines)))
c = re.search(r"best ([\d.]+) tok/s plain and ([\d.]+) tok/s with MTP.*?best of ([\d.]+) tok/s", sec)
want = (fmt(best["tarfa_plain"]), fmt(best["tarfa_mtp"]), fmt(best["llamacpp"]))
if not c or c.groups() != want:
    bad("README.md", 0, f"GLM section must read best {want[0]} tok/s plain and {want[1]} tok/s with MTP, "
                        f"against llama.cpp's best of {want[2]} tok/s")
if len(CLAIM.findall(sec)) != 3: bad("README.md", 0, "GLM section has speed figures the check does not cover")
if re.search(r"\d+ runs?\b", sec): bad("README.md", 0, "GLM section: run counts belong in the engine README")
if re.findall(r"\d+(?:\.\d+)?×(?! Xeon)", sec): bad("README.md", 0, "GLM section: speedup figures belong in the engine README")

if errors:
    print("Docs drifted from validation/ receipts:\n  " + "\n  ".join(errors)); sys.exit(1)
print(f"OK: {len(DOCS) + 1} docs consistent with receipts "
      f"(exact {exact['tok_per_s']}, fast {fast['tok_per_s']} tok/s; p12 {p12_pct}%)")
