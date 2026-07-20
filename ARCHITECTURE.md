# Architecture

Tarfa treats **NVMe → pinned RAM → VRAM** as one managed memory hierarchy for expert-parallel
MoE inference. Only what a token actually routes to ever moves.

## The problem shape

Qwen3.5-122B-A10B: 48 layers × 256 experts, top-8 routed → ~10B active params/token out of 122B.
Dense residency needs ~234 GB (BF16); the GPU has 16 GB. But per token per layer only 8/256
experts fire — so the working set per token is ~19 GB of expert weights, streamed.

## Components

**Layered materialization** (`engine/layered.py`, `engine/channel.py`) — the transformer skeleton
is materialized layer-by-layer; attention/norm weights stay resident (~12.3 GB), expert tensors are
deleted from the module tree and replaced by a streaming `experts.forward`.

**JNF v1** (`format/JNF_V1.md`) — the on-disk execution format: per-layer BF16 expert files,
4096-byte aligned per projection, independently addressable gate_up/down, SHA-256 per projection
and per file, atomic conversion from the authoritative HF checkpoint.

**Exact BF16 transport** (`engine/exact_bf16.py`) — parallel `preadv` of exactly the routed experts
into reusable pinned host buffers (Phase 5), non-blocking H2D, read/compute overlap where proven
exact (Phase 8), completion-overlap serving (−11%, Phase 12).

**Heat tiering** (the heat ledger) — a learned expert-heat ledger ranks experts by observed routing
frequency; the hottest get a static RAM pin, the hottest-of-hot a zero-copy VRAM tier. Hits skip
NVMe entirely; the device/dtype contract is validated on every tier hit. Exactness is unaffected —
tiers change *where* bytes come from, never *which* bytes.

**Fast mode** (`engine/fused_int4.py` + `engine/moe_kernel.py`) — int4 nibble-packed experts
(per-row fp16 scale) with a fused Triton MoE kernel; ~2–3 tok/s. Non-exact, labeled, opt-in.

**Serving** (`engine/serve.py`) — OpenAI-compatible HTTP API (:8089), single-GPU aware,
usage ledger, terminal + web UI.

## Data path (exact decode, one token)

```
router top-8 per layer
  → JNF offsets (index.json)
  → parallel preadv (NVMe, high queue depth)   ┐ overlapped
  → pinned RAM ring → non-blocking H2D          ┘ (Phase 8)
  → BF16 GEMV per expert, sorted accumulation order
  → residual; fp32 logits at the last position
tier short-circuit: VRAM hot tier → RAM pin → NVMe
```

## Design rules discovered the hard way

- NVMe queue depth is the whole game: batched parallel `preadv` ≈ 3.2× serial mmap.
- On a bandwidth-bound PCIe bus, predictive prefetch is a net loss (measured, rejected).
- Caches must earn VRAM: a 1.8 GB cache with a 3.3% hit rate is worse than nothing.
- Every "obvious win" gets a validation script before it gets merged. Most obvious wins lose.
