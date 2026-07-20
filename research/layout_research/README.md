# Phase 10: route-optimized physical BF16 layout

Question: can exact BF16 inference be accelerated by reorganizing how expert
weights sit on disk, interleaving, co-occurrence ordering, route bundles, or
model-specific lossless compression? Answer: **no, on this storage stack.**
All four research questions were rejected at their first predeclared gates,
each with a measured physical cause. No production code was touched.

## Baseline (2026-07-11, commit 5902e50/b77edce, ordinary exact mode)

- Repo clean; Phase 9 audit confirmed; `tarfa serve`, dreamer off during all
  measurements; correctness reference = ordinary single-token greedy decoding.
- VRAM 12.3 GB; RAM 31 GB (≈21 available); swap 19 GB (zram + cryptswap).
- Expert = 18,874,368 B exact BF16 (gate_up 12,582,912 + down 6,291,456);
  2 preads/expert in production; 256 experts/layer; 48 layers; top-8.
- Decode forward 2.13–2.15 s (one service instance) to 2.77–2.85 s (another,
  same day): cross-instance system-state variance is large, so every
  comparison here is paired A/B on the same process with cache eviction.
- Lifetime telemetry: expert reads ≈69% of layer time, H2D ≈27%.
- Storage stack: single NVMe → LUKS (dm-crypt) → LVM → ext4. No passwordless
  root, so raw-device throughput could not be isolated from LUKS.

## Read-pattern diagnosis (`layout_probe.py`, `read_pattern_results.json`)

1.81 GB per arm from the real shards, page cache evicted, three repeat trials:

| pattern | GB/s |
|---|---|
| production: 16 split gate/down preads/layer, QD48 | 3.54 (3.08–3.10 repeats) |
| one 18.87 MB contiguous read per expert (RQ1 shape) | 3.63 (3.10–3.13, **+1.1±0.3%**) |
| one 151 MB contiguous read per layer (RQ3 shape) | 3.45 (3.00–3.04, **−2.3%**) |
| pure sequential, 1 thread | 2.28 |
| queue-depth sweep, 18.87 MB random | saturates ≈3.6 at QD≥8 |

The channel is bandwidth-capped at ≈3.5–3.6 GB/s for every pattern once queue
depth ≥8; CPU is 20–27% busy during reads (consistent with dm-crypt cost,
which is per byte, not per seek). Contiguity and request count are worth ≈1–2%.

## Verdicts

- **RQ1 interleaved records: REJECTED.** Upper bound measured directly at
  +1.1±0.3% vs a 15% gate, before any conversion. (This is also why JNF v1
  and native io_uring previously failed end-to-end: they reduce request
  overhead on a channel that charges per byte.)
- **RQ2 co-occurrence ordering: REJECTED.** Its only mechanism is turning
  scattered reads into contiguous ranges; contiguity is worth ≤2% here, and
  gap reads add amplification on top.
- **RQ3 route bundles: REJECTED.** A perfect bundle hit converts 8 scattered
  reads into one 151 MB sequential read, measured *slower* (−2.3%); partial
  hits add read amplification and 1.25–2× storage cost buys nothing.
- **RQ4 lossless compression: REJECTED at the replay gate (6.7% < 15%).**
  Details below, this was the only lever that reduces bytes, and it earned a
  full physical test.

## RQ4 detail (`bf16_entropy_study.py`, `rq4_convert.py`, `rq4_replay*.py`)

Entropy of 226 MB of real expert weight bytes: sign+exponent byte 2.65
bits/byte (exponents concentrated in 114–121), mantissa byte 7.97 bits/byte
(incompressible); order-0 ceiling ≈1.51×. XOR between experts: 6.46 bits/byte
worse than raw, killing base+residual and cross-expert dictionary schemes.

Byte-plane split + zstd level 1 (level 1 beat levels 2–19 on ratio *and*
speed): 1.451× file-level; 1.37–1.45× as independent per-expert frames
(layers 0/15/30/45 converted: 19.3 GB → 13.6 GB, 256/256 SHA-256 verified per
layer, spot byte-equality vs originals, atomic + resumable converter).

Physical replay on held-out test routes (4 domains × 6 tokens × 4 layers ×
top-8; 14.5 GB plaintext, byte equality verified for all 396 distinct pairs):

| pipeline | median | vs raw |
|---|---|---|
| raw shard reads (16 Python threads) | 3.04 s |, |
| v1: naive decode (alloc per call, CPU unplane) | 10.8 s | −217% |
| v2: reused dctx, no unplane, GPU interleave 0.32 ms/expert | 3.26 s | −7% |
| v3: 8 decode **processes** (GIL-free, C-path proxy) raw=2.15 s | 2.01 s | **+6.7%** |

Why 6.7% and not the naive 30%: consecutive tokens re-select ≈34% of experts,
so the page cache serves a large share of re-reads at RAM speed. Compression
saves bytes only on cache misses but pays decompression on every access.
Charging everything honestly, the gain converges to ≈7%, under the 15% gate,
and it would shrink further after H2D and runtime integration. A dedicated C
decode path cannot rescue it; v3 already models that.

## Consequences for future phases

The exact-BF16 read path is within ~2% of what this hardware+stack can
deliver for any file layout, any request pattern, and any practical lossless
compression. Software has exhausted the storage channel. The remaining
levers are hardware/system-level:

1. Determine whether the ≈3.5 GB/s cap is the NVMe or dm-crypt: read the raw
   partition with root, and test `cryptsetup refresh` with
   `--perf-no_read_workqueue` / `--perf-submit_from_crypt_cpus` (documented
   to lift LUKS NVMe read throughput substantially on some systems). Zero
   hardware cost; needs root and care.
2. A second NVMe with experts striped across both drives: reads are
   embarrassingly parallel across experts, so ≈2× channel is ≈1.45× end-to-end
   (read is 69% of layer time).
3. RAM: 31 GB cannot hold the 232 GB working set; a RAM upgrade only helps
   the page-cache reuse fraction, already exploited.

## Files

- `layout_probe.py`, `read_pattern_results.json`, read-pattern diagnosis
- `bf16_entropy_study.py`, `rq4/entropy_report.json`, compressibility
- `rq4_convert.py`, `rq4/zbf16/` (13.6 GB, gitignored), validated converter
- `rq4_replay.py`, `rq4_replay2.py`, `rq4_replay3.py`, `rq4/replay*_results.json`
