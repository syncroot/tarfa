# Tarfa native execution format (JNF) v1

JNF v1 is an execution format for exact Qwen3.5-122B-A10B BF16 inference. The
original Hugging Face checkpoint remains authoritative; conversion must verify
every emitted region against its source tensor.

## Directory

```text
jnf-v1/
  manifest.json
  hotlist.json
  layers/00/gate_up.bf16
  layers/00/down.bf16
  layers/00/index.json
  ...
```

## Invariants

- native router top-k remains eight;
- weights remain bit-identical BF16;
- each expert projection begins at a 4096-byte boundary;
- gate/up and down are independently addressable;
- expert order is 0 through 255;
- offsets and lengths are explicit 64-bit integers;
- SHA-256 exists for every expert projection and complete layer file;
- format version, model revision, shapes, dtype and tokenizer revision are
  mandatory manifest fields;
- conversion is atomic per layer (`.partial` then rename);
- runtime refuses unknown versions, shapes or failed checksums.

## Fixed runtime slots

Decode owns eight logical expert slots. Transport owns two pinned buffer sets;
each set holds a configurable group of exact gate/up and down projections.
CUDA kernels consume stable GPU addresses. Python does not construct per-expert
tensor dictionaries in the final native path.

## Hotlist

`hotlist.json` is generated from a named routing corpus. It records per-layer
expert counts, corpus hash, token count and predicted coverage for budgets of
1, 2, 4 and 8 experts per layer. Hotlist placement is an optimization only and
never changes routing or numerical values.

## Acceptance

1. Source-to-JNF projection hashes match.
2. Random layer outputs match the current exact-BF16 reference tolerance.
3. Greedy continuation vectors match.
4. Four-layer transport beats the current reader in cold and warm trials.
5. Full conversion starts only after the four-layer gate passes.
