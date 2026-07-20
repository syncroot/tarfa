# The Exactness Contract

Tarfa's defining rule: **a performance change is accepted only if the output remains exactly the
model's output.** Not "close", not "within quantization noise" — the same checkpoint weights, the
same router semantics, the same greedy tokens. Anything else is a separate labeled mode.

## Invariants (exact mode)

1. Expert weights are read from the original BF16 checkpoint (or its SHA-256-verified JNF
   conversion) — bit-identical, never re-quantized.
2. Router top-k is the model's native 8. No truncation (`JTOPK=8`).
3. Final logits are computed in fp32 (bf16 over a 152k vocab flips argmax).
4. Predictive prefetch / overlap tricks that can change numerics are off in exact mode.
5. Every optimization ships with a validation script (`validation/phase*_validate.py`) proving
   invariants 1–4 before it is accepted.

## The ledger: accepted and rejected

What makes this credible is what was **rejected**. Performance engineering that only reports wins
is advertising; this table is the audit trail.

| phase | idea | result | verdict |
|---|---|---|---|
| 3 | batch 16 selected experts per prefill read | I/O-scheduling only, exactness proven | ✅ accepted |
| 4 | 2-expert-per-layer GPU cache | 3.3% hit rate for ~1.8 GB VRAM | ❌ rejected |
| 5 | pinned host buffers + non-blocking H2D | exact, faster | ✅ accepted (default) |
| 6 | exact adaptive GPU-lossless BF16 transport | exact | ✅ accepted |
| 7 | 4 GB raw-BF16 RAM arena | avoided 24.7 GB NVMe traffic but −9.9% end-to-end, 12.6% hit | ❌ rejected |
| 8 | overlap exact expert reads with GPU reconstruction | exact | ✅ accepted |
| 9 | speculative decoding audit | batched-verification numerics diverge from single-token decode → greedy output can differ | ⚠ demoted to labeled fast mode |
| 10 | physical-layout & lossless-compression acceleration | no exact win | ❌ rejected |
| 11 | request-local router-signal prefetch | gated | ⚠ gated |
| 12 | completion-overlap serving | 2.95 → 2.63 s/tok (−11%), exact | ✅ accepted |
| heat | learned expert-heat ledger, RAM pin, zero-copy VRAM hot tier | exact (device/dtype contract honored) | ✅ accepted |

Full narrative: [docs/PHASE_LOG.md](docs/PHASE_LOG.md).

## The fast mode is not exact — and says so

`tarfa start-fast` runs the fused int4 Triton path (~2–3 tok/s vs ~0.34). Its output may differ
from exact greedy decoding. It exists because sometimes you want speed; it is never the default,
and it must never be used for equivalence testing, reproducible research, exact benchmarks, or
correctness-sensitive evaluation. If a future optimization makes fast mode exact, it graduates;
until then the label stays.

## Why this matters

Model auditing, interpretability, and safety evaluation all assume you are studying *the model* —
not a quantized cousin with a truncated router. On consumer hardware that assumption is almost
always false. Tarfa makes it true, and makes the cost of making it true visible: ~19 GB of verified
NVMe reads per token. Exactness is expensive. Hiding that expense is how everyone else got fast.
