<p align="center">
  <img src="assets/tarfa.png" width="200" alt="Tarfa">
</p>

# Tarfa

**Run a 122-billion-parameter MoE with bit-identical BF16 weights on a 16 GB consumer GPU.**

Tarfa is an exact-first Mixture-of-Experts runtime. It streams experts from NVMe through
pinned RAM into VRAM as one managed memory hierarchy — and it makes a promise most engines don't:
**the model you run is the checkpoint, provably.** No quantization, no router truncation, no silent
precision changes. A clearly-labeled fast mode exists for when you want speed over exactness — but
exact is the default, and every performance change in Tarfa's history had to prove exactness or be
rejected (the rejections are documented, with numbers, in [EXACTNESS.md](EXACTNESS.md)).

```
$ tarfa serve
  Tarfa · Qwen3.5-122B-A10B · exact BF16 · streaming
  ✓ ready · resident 12.3 GB VRAM · experts on NVMe
```

## The numbers, at a glance

<p align="center">
  <img src="docs/media/price_of_exactness.svg" width="70%" alt="exact vs fast decode speed">
</p>

Same GPU, same 122B model, same prompt: exact BF16 costs ~5× the speed of the labeled int4 fast
mode — because each exact token moves **~19 GB of verified expert weights from NVMe**.

<p align="center">
  <img src="docs/media/power_utilization.svg" width="90%" alt="power and utilization traces">
</p>

The measured surprise: while running a 122B model, the GPU averages **26% utilization at ~32 W of
its 160 W budget** — the workload is storage-bound, not compute-bound. The ceiling on consumer
hardware is engineering, not silicon. Full receipts and methodology: [BENCHMARKS.md](BENCHMARKS.md).

## Why Tarfa exists

Every practical way to run a 100B+ model on consumer hardware quietly changes the model:
int4/int8 quantization, truncated router top-k, speculative decoding whose greedy output diverges.
Fine for chat — fatal for reproducible research, model auditing, interpretability, and any
correctness-sensitive work. Tarfa's contract:

- expert weights are read **bit-identical** from the original BF16 checkpoint;
- native router semantics (top-8) are never reduced;
- anything that trades exactness for speed is a **separate, explicitly-labeled mode** — never a default.

## Measured (RTX 4060 Ti 16 GB, PCIe4 NVMe, Qwen3.5-122B-A10B)

| mode | decode speed | weights | greedy output |
|---|---|---|---|
| **exact BF16** (default) | ~0.34 tok/s (2.9–3.1 s/tok; ~2.6 with completion overlap) | bit-identical | reference |
| **fast** (fused int4 Triton) | ~2–3 tok/s | int4 symmetric | may differ — labeled |

Exact mode moves ~19 GB of expert weights from NVMe per decoded token, verified against the
checkpoint. That it works at all on a $400 GPU is the point; that it's *provably exact* is the
contribution.

## Architecture (short version)

```
NVMe (JNF v1: aligned, SHA-256-verified BF16 experts)
  └─ parallel pread → pinned RAM ring
       └─ non-blocking H2D → VRAM
            ├─ expert-heat ledger (learned residency)
            ├─ static RAM pin + zero-copy VRAM hot tier
            └─ per-layer top-8 routed compute (exact) / fused int4 kernel (fast)
```

Benchmarks + charts: [BENCHMARKS.md](BENCHMARKS.md) · Details: [ARCHITECTURE.md](ARCHITECTURE.md) · on-disk format: [format/JNF_V1.md](format/JNF_V1.md)
· full validate-or-reject engineering log: [docs/PHASE_LOG.md](docs/PHASE_LOG.md)

## Install

```bash
git clone <repo> && cd tarfa && pip install -e .   # PyPI planned
tarfa convert --layers 0-47      # build SHA-256-verified JNF from the HF checkpoint
tarfa convert --verify           # re-verify every emitted byte against the source
tarfa serve                  # OpenAI-compatible API on :8089
tarfa chat                   # terminal chat
tarfa start-fast             # the labeled non-exact fast mode
```

Requirements: NVIDIA GPU ≥ 12 GB, NVMe with the checkpoint (~234 GB BF16), Python 3.11, PyTorch ≥ 2.x.

## Status

Private preview. Exact mode is the authoritative path (validation suite in `validation/`);
the fast fused-int4 mode is functional and labeled. Currently validated on Qwen3.5-122B-A10B;
the design generalizes to other expert-parallel MoEs.

## License

MIT.
