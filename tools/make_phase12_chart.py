"""Regenerate docs/media/phase12_overlap.{svg,png} from the committed Phase 12 receipt.
Usage: uv run --with matplotlib python tools/make_phase12_chart.py
"""
import json, pathlib, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
r = json.loads((REPO / "validation/phase12/completion_results.json").read_text())
BG, FG, MUTED, GRID, ACCENT = "#0c0e14", "#e7ecf5", "#8b94a8", "#2a3145", "#3ad29f"
plt.rcParams.update({"svg.hashsalt": "tarfa", "svg.fonttype": "path"})

fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=BG)
ax.set_facecolor(BG)
runs = range(1, len(r["serial"]) + 1)
ms, mc = statistics.median(r["serial"]), statistics.median(r["completion"])
ax.plot(runs, r["serial"], "o-", color=MUTED, label=f"serial (median {ms:.2f} s)")
ax.plot(runs, r["completion"], "o-", color=ACCENT, label=f"completion-driven (median {mc:.2f} s)")
ax.set_title(f"Phase 12: completion-driven expert reads, exact, {r['improvement']*100:.1f}% faster (median)", color=FG)
ax.set_xlabel("trial (page cache evicted each arm)", color=MUTED)
ax.set_ylabel(f"seconds per replay ({r['batches']} held-out expert reads)", color=MUTED)
ax.set_xticks(list(runs))
ax.tick_params(colors=MUTED)
for s in ax.spines.values(): s.set_color(GRID)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
fig.text(0.5, 0.01, "receipt: validation/phase12/completion_results.json · I/O read-path replay, not end-to-end decode",
         color=MUTED, ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = REPO / "docs/media"
fig.savefig(out / "phase12_overlap.svg", facecolor=BG, metadata={"Date": None})
fig.savefig(out / "png/phase12_overlap.png", facecolor=BG, dpi=100)
