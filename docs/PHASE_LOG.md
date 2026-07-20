# Tarfa exact-BF16 track

Phase 1 preserves the working Tarfa runtime and proves that its BF16 expert
reader returns the original checkpoint weights without quantization or expert
truncation.

Exact-mode invariants:

- expert weights come directly from the original BF16 safetensors checkpoint;
- native router top-k is 8 (`JTOPK=8`);
- predictive prefetch and overlap are disabled (`PIPE=0`, `OVERLAP=0`);
- no INT4 expert file is read by the BF16 path;
- validation must pass before performance changes are accepted.

Phase 3 batches 16 already-selected BF16 experts per prefill read. This changes
I/O scheduling only: expert computation and accumulation retain their original
sorted order. Set `BF16_PREFILL_BATCH=1` for the Phase-2 baseline.

Phase 4 tested a two-expert-per-layer GPU cache. It produced only a 3.3% hit
rate while reserving about 1.8 GB, so it is disabled by default. The code is
retained for experiments through `BF16_CACHE_PER_LAYER`; exact mode uses `0`.

Phase 5 reads BF16 experts directly into reusable pinned host buffers before
non-blocking H2D transfer. It is enabled by default with `BF16_PINNED=1`; use
`0` to reproduce the pageable-memory baseline.

Phase 7 tested a fixed raw-BF16 RAM arena with per-layer segmented queues.
The 4 GB configuration avoided 24.67 GB of NVMe traffic but slowed the exact
benchmark by 9.9%, produced only a 12.6% hit rate, and reduced swap headroom.
It is rejected and remains disabled with `BF16_RAM_CACHE_GB=0`.

Run on the inference host:

```bash
python phase1_validate.py
```

Phase 9 audited the optional speculative runtime and demoted it: batched
verification numerics diverge from single-token decode, so its greedy output
can differ from exact decoding (`spec_audit/`). `tarfa serve` (ordinary exact
BF16) is the authoritative production mode. `tarfa start-fast` (alias
`start-spec`) starts the clearly labeled fast speculative mode: output may
differ from exact greedy decoding; never use it for equivalence testing,
reproducible research, exact benchmarks, or correctness-sensitive work.
