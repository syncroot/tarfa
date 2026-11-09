# GLM-5.3-Flash on Tarfa (experimental)

A from-scratch decode engine for GLM-5.3-Flash (`glm5-next`, 320B parameters, ~18B active)
that runs the 186 GiB `UD-Q4_K_XL` GGUF on a machine with 183 GiB of RAM and two older GPUs,
faster than llama.cpp on the same machine and file.

**This is not Tarfa's exact-BF16 mode.** I had no BF16 copy of this model, so the engine
computes the quantized GGUF (Q4_K/Q5_K/Q6_K experts, Q8_0 trunk). It is checked against a numpy
reference and against llama.cpp, not against a BF16 checkpoint. Treat it as an experiment,
tuned for one machine.

## Result

Story prompt (17 tokens), 128 greedy tokens, same machine and GGUF, one session (2026-11-09).
Before each engine started, the model file was dropped from the page cache, so each engine
loaded it its own way. Every run is in the receipt
[`validation/glm5/bench_story.json`](../../../validation/glm5/bench_story.json).

| engine | best | all runs |
|---|---|---|
| llama.cpp, production flags (`-ncmoe 40 -ts 41,4 -t 16`) | 9.09 tok/s | 1.13–9.09 tok/s, 10 runs |
| Tarfa, plain decode (`run_best.sh`) | 12.94 tok/s | 9.24–12.94 tok/s, 2 runs |
| Tarfa, MTP speculative decode (`run_spec.sh`) | 13.68 tok/s | 12.80–13.68 tok/s, 2 runs |

Best against best: 1.4× llama.cpp with plain decode, 1.5× with MTP. The slow ends of the ranges
are first starts: llama.cpp's first request after a load runs while it is still reading expert
pages, and Tarfa's first start after the cache is emptied can still take page faults during
decode; its second start reuses the pages the first one placed. Both engines report a rate per
decode step after the prompt (the receipt's method note has the exact counting).

I also tried to tune llama.cpp: automatic GPU placement (`--fit on`), a smaller context, fewer
threads and loading without mmap (which does not fit this machine's RAM). None beat the
production flags; every run is in
[`validation/glm5/llamacpp_sweep.json`](../../../validation/glm5/llamacpp_sweep.json).

An earlier session (2026-11-07) gave 1.6× and 1.7×, but it was unfair to llama.cpp: it ran on a
page cache Tarfa had filled and split across the NUMA nodes for its own kernel.

| earlier session | best | all runs |
|---|---|---|
| llama.cpp, production flags, 2026-11-07 | 7.88 tok/s | 2.18–7.88 tok/s, 19 runs |
| Tarfa, plain decode, 2026-11-07 | 12.56 tok/s | 11.87–12.56 tok/s, 3 runs |
| Tarfa, MTP speculative decode, 2026-11-07 | 13.57 tok/s | 13.03–13.57 tok/s, 3 runs |

Check runs of the published launchers (2026-11-09, outside both sessions):

| check | runs |
|---|---|
| Tarfa, plain decode, check | 12.38–12.73 tok/s, 3 runs |
| Tarfa, MTP speculative decode, check | 12.36–12.38 tok/s, 2 runs |

Machine: 2× Xeon Gold 5122 (8 cores, 2 NUMA nodes), 183 GiB DDR4-2666, Quadro RTX 6000 24 GB
(PCIe Gen3 x16, sm_75) and RTX 4060 Ti 16 GB (PCIe Gen3 x8, sm_89).

What the receipts cover: the speeds, the France top-5 check and the model files. The hardware
details above, the component cosines and the microbenchmark figures below come from my own
notes and are not backed by files in this repo.

## How it works

- **Trunk on the GPU in CUDA graphs**: 34 KDA linear-attention layers and 11 MLA layers,
  4-stream hyper-connections with Sinkhorn mixing, Q8_0 weights kept quantized (`q8.py`).
- **Experts split three ways**: a profiled hot set lives in VRAM on both cards (`experts_gpu.py`,
  `qmoe.py`, `glm5_calib.py`); the rest run on the CPU from the memory-mapped GGUF, split by
  rows across NUMA nodes (`cpu_moe_ggml.c`). The CPU expert kernel takes about 60% of decode
  time (6.0–6.4 s of 10.2–10.8 s in the 2026-11-07 plain runs).
- **CPU kernel**: llama.cpp's ggml k-quant dot products, plus my own AVX-512BW Q4_K/Q5_K dots
  with 2–4 KB in-row prefetch (0.28 → 0.22 ms per expert in my `bench_kern.py` runs, noted in
  PERFORMANCE.md). Same integer math as ggml (`test_avx512.py`).
- **MTP speculative decode** (`mtp.py`, `glm5_spec.py`): the GGUF's NextN head (block 45)
  drafts one token, a two-token pass verifies it, KDA state rolls back on a reject. About 81%
  of drafts are accepted (all recorded runs), but two tokens touch ~1.8× the CPU experts, so the gain is small.

The step-by-step history, with what helped and what didn't (adaptive expert swapping, PCIe
streaming of CPU misses), is in [PERFORMANCE.md](PERFORMANCE.md). [KERNELS.md](KERNELS.md) and
[VERDICT.md](VERDICT.md) are the dated notes from validating each component; later sections
there correct earlier ones.

## Fidelity

- From my dated validation notes (KERNELS.md; the dumps and logs are not in the repo): each
  component (KDA, hyper-connections, dense FFN, MLA, MoE) was checked once against
  llama.cpp full-tensor dumps at one layer and the first positions: component-output cosine
  0.99994–0.99998, the MoE's inner routed-expert sum 0.9998 (`probe2.py`–`probe6b.py`,
  KERNELS.md). On a 2801-token prompt the numpy reference of the sparse indexer kept the same
  pools as llama.cpp at the last position, apart from one tie at the cutoff; that check starts
  from llama.cpp's own indexer queries, pooled keys and weights (`probe7.py`). A 45-layer numpy forward
  built from these pieces predicts llama.cpp's next token.
- The engine itself, on the prompt "The capital of France is": same argmax as llama.cpp and the
  same top-5 token set, in a different order below the top token
  ([`validation/glm5/fidelity_france.json`](../../../validation/glm5/fidelity_france.json)).
  Greedy text can split from llama.cpp's at near-ties. I have not isolated why; likely
  contributors are activation quantization (llama.cpp quantizes activations to 8 bits in its
  matrix products, this engine only in the CPU expert kernel) and different accumulation order.
- MTP decode is greedy-equivalent only up to rounding: the two-token verify accumulates in a
  different order, which can flip a near-tie (`GLM5_SPEC_CHECK=n` measures it).

## Run it

Requirements: Linux, x86-64 (AVX-512BW dots when the compiler targets it, ggml's otherwise), two
NVIDIA GPUs with 40 GB VRAM between them, ~180 GiB RAM, Python with PyTorch 2.6, Triton 3.2,
numpy and `gguf` (tested with 0.19), and llama.cpp built with shared libraries. I used llama.cpp
commit `bed0a85` (2026-10-02), which supports `glm5-next`.

The model: Unsloth's `UD-Q4_K_XL` GGUF of GLM-5.3-Flash,
[`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) at revision
`621d456e93e926e4b52f85cff5f634358c1828f9` (186 GiB; file sizes and SHA-256 in
[`validation/glm5/model_files.json`](../../../validation/glm5/model_files.json)). Shards 2–6 come from `UD-Q4_K_XL/`;
shard 1 is the repo's rewritten `Shard_Rewrite/…-00001-of-00006.gguf_file` (9.4 MB). All six
must sit in one directory under their `.gguf` names.

```bash
# model (the GGUF reader needs all six shards side by side)
pip install torch==2.6.* triton==3.2.* numpy gguf "huggingface_hub>=2.4"   # hf CLI with repeatable --include
hf download unsloth/GLM-5.3-Flash-GGUF --revision 621d456e93e926e4b52f85cff5f634358c1828f9 \
    --include "UD-Q4_K_XL/*-0000[2-6]-of-00006.gguf" --include "Shard_Rewrite/*Q4_K_XL-00001-of-00006*" \
    --local-dir glm53
mkdir -p glm53/model && (cd glm53/model && ln -s ../UD-Q4_K_XL/*.gguf . && \
    ln -s ../Shard_Rewrite/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf_file \
          GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf)

# llama.cpp: shared libggml-cpu / libggml-base and llama-tokenize
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout bed0a85
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && cd ..

cd tarfa/engine/glm5
B=/path/to/llama.cpp/build/bin
gcc -O3 -march=native -fopenmp -shared -fPIC cpu_moe_ggml.c -o libcpumoe_ggml.so \
    -L$B -lggml-cpu -lggml-base -Wl,-rpath,$B
export GLM5_GGUF=/path/to/glm53/model LLAMA_TOKENIZE=$B/llama-tokenize
./run_calib.sh                       # once: profiled GPU hot set -> out/route_prior.npy
GLM5_PROMPT="Write a short story about a lighthouse keeper who finds a message in a bottle." \
  ./run_spec.sh 128                  # or ./run_best.sh 128
```

The launchers' shared defaults live in [`glm5_env.sh`](glm5_env.sh); each can be overridden from
the environment. The main inputs:

| variable | default | meaning |
|---|---|---|
| `GLM5_GGUF` | (required) | directory with the GGUF shards |
| `GLM5_PROMPT` | (required for run_best/run_spec) | prompt text |
| `GLM5_OUT` | `./out` | output directory; the prior is looked up here |
| `GLM5_PRIOR` | `$GLM5_OUT/route_prior.npy` | routing prior from `run_calib.sh`; runs without it are slower |
| `LLAMA_TOKENIZE` | `llama-tokenize` on PATH | llama.cpp tokenizer binary |
| `PYTHON` | `python` | interpreter with the packages above |
| `GLM5_DEV`, `GLM5_RAW_PER_LAYER` | `cuda:1`, `19` | trunk GPU (24 GB card) and its expert slots per layer |
| `GLM5_RAW2_DEV`, `GLM5_RAW2_PER_LAYER` | `cuda:0`, `23` | second GPU (16 GB card) and its slots |
| `GLM5_MAXLEN` | `2048` | positions in the attention caches (see Limits) |

Other GPUs need other device and slot values. CPU placement: with the default `GLM5_NUMA=1` the
CPU kernel uses one thread per physical core on every NUMA node, split by node; with
`GLM5_NUMA=0` it uses one shared team, which `GLM5_CPU_PIN` (comma list of CPUs, one per thread)
can place. `GLM5_CPU_PIN` has no effect in NUMA mode. The launchers stop with a message if the
prompt plus the requested tokens do not fit in `GLM5_MAXLEN`.

## Limits

- Single sequence, greedy decode, no server. Startup takes about 4 minutes (6 with MTP).
- Tuned and measured on one machine only; the build steps above have not been tried elsewhere.
- Context is capped at 2048 positions. The engine attends to every key and has no sparse
  indexer; that is exact while there are at most 2048 keys, because llama.cpp's indexer
  (top_k 2048) then keeps them all. Raising `GLM5_MAXLEN` lifts the cap but not the
  difference: beyond 2048 keys llama.cpp prunes and this engine does not.
- Prefill runs token by token through the decode path.
- `bench_*.py`, `prof_*.py` and `profile_step.py` assume the reference machine's GPU order and
  NUMA layout (2 nodes, 16 CPUs); they are measurement scripts, not part of the engine.
- `probe*.py`, `full_forward.py` and `glm5_next.py`'s self-test need llama.cpp tensor dumps,
  which are not in the repo. They come from `llama-eval-callback` with
  [`oracle/eval-callback-dump.patch`](oracle/eval-callback-dump.patch) applied to llama.cpp
  `bed0a85`, run CPU-only (`CUDA_VISIBLE_DEVICES=""`) with `GLM5_DUMP=<tensor names>` and
  `GLM5_DUMP_DIR=<dir>`.

## Credits

GLM-5.3-Flash by Z.ai (MIT license); the GGUF quantization by Unsloth; llama.cpp / ggml for the
reference implementation and the CPU k-quant kernels this engine links against.

Parts of this directory derive from llama.cpp (MIT, Copyright (c) 2023-2026 The ggml authors; full
text in [LICENSE-llama.cpp](LICENSE-llama.cpp)): the numpy reference kernels transcribed from its
`glm5-next` model code (`kda_ref.py`, `mla_ref.py`, `glm5_layer.py`), the k-quant block layouts and
dot products in `cpu_moe.c` and `cpu_moe_ggml.c`, and `oracle/eval-callback-dump.patch`, a patch to
its `eval-callback` example.
