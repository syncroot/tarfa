# glm5-next decode performance (2026-11-06/07)

Lab notebook of the optimization steps, in order; the step figures are single runs from that
day. The headline comparison with all runs is in README.md and `validation/glm5/bench_story.json`.

Machine: 2x Xeon Gold 5122 (2x4 cores, HT, 2 NUMA nodes, 6x DDR4-2666), 183 GiB RAM,
Quadro RTX 6000 24 GB (cuda:1, sm_75) + RTX 4060 Ti 16 GB (cuda:0, sm_89). Model GGUF
GLM-5.3-Flash-UD-Q4_K_XL, 186 GiB (experts Q4_K/Q5_K/Q6_K, trunk Q8_0) - does not fit RAM.
Baseline: llama.cpp production flags (-ncmoe 40, -ts 41,4, -t 16), story prompt @128: 2.2-7.9 tok/s over 19 runs, best 7.88 (-t 8: best 6.87 over 8 runs).
Correction (2026-11-09): those llama.cpp runs used a page cache Tarfa had filled and NUMA-split for
its own kernel. Loaded by itself (cache evicted first), llama.cpp gives 8.9-9.1 tok/s with the same
flags, so the fair ratio is about 1.4x plain and 1.5x with MTP (README.md, bench_story.json).

| step | tok/s | where |
|---|---|---|
| first torch engine, experts dequantized in numpy (~90 s/token) | 0.01 | glm5_decode.py (M1) |
| eager torch, GPU dequant, fused k-quant matvec, 2-GPU expert cache | 2.18 | glm5_decode.py |
| CPU experts on llama.cpp's ggml kernels + static prefill hot set | 4.12 | glm5_decode.py (after KDA fix) |
| CUDA graph per MoE layer (sync only at routing) | 5.2 | glm5_gdecode.py |
| GPU hits inside per-(layer,device) graphs (device pointer tables) | 5.4 | qmoe.moe_masked |
| CPU OpenMP fix (team size + pinning; torch's libgomp capture) | 6.8 | cpu_moe |
| Q8_0 trunk kept quantized, F32 router, 19+23 slots/layer | 8.1 | q8.py |
| calibration prior fills free slots | 8.7 @48 / 8.1 @128 | glm5_calib.py |
| NUMA row-split of CPU experts | 10.1 @128 | moe_cpu_numa |
| torch.compile'd trunk bodies + router topk | 11.1 @128, 10.3 @256 (story prompt) | GLM5_COMPILE |
| own AVX-512BW Q4_K/Q5_K dots + in-row prefetch (CPU 0.28 -> 0.22 ms/expert) | 12.5 @128 (story; was 10.95) | cpu_moe_ggml.c |
| MTP speculative decode (NextN draft, 2-token verify) | **13.0 @128** (story) | glm5_spec.py, run_spec.sh |
| plain decode: CPU experts called once per layer on bound tensors (2026-11-10) | 13.2-13.6 @128 (story; 12.5-12.9 before, same day) | cpu_moe.moe_cpu_layer |
| MTP speculative decode: the same per-layer call for the 2-token union (2026-11-10) | 14.3-14.9 @128 (story; 13.2-13.9 before, same session) | cpu_moe.moe_cpu_layer_t |

Per token at 11.1 tok/s (GLM5_TL=1): CPU experts ~57 ms (~185 misses x 0.29 ms), trunk graphs
~21 ms, host glue ~9 ms; GPU hits (~16 ms) overlap the CPU work.

Speculative step at 13.0 tok/s (1.80 tokens/step, 82% drafts accepted): 138 ms = CPU experts 88
(330 union experts), trunk graphs 27, host glue 10, MTP draft 10 (CPU experts 3.5).

## Per-layer CPU call (2026-11-10)
The plain decode step built three Python lists of memmap slices per MoE layer and went through the
general ctypes wrapper. `cpumoe_bind_layer` now registers each layer's three expert tensors once
and `moe_cpu_layer` takes the layer, a miss mask and the addresses of the pinned host buffers; the
C side calls the same `moe_cpu_numa`. Output is bit-identical (replay_cpu.py, 40 random calls).
Replay of a recorded decode outside the engine: 41.2-41.4 -> 38.7-39.1 ms/token, Python cost per call
58 -> 5 us. In-engine, warm, story prompt @128, interleaved runs: published checkout (9887cdf)
12.71 / 12.72 / 12.75 tok/s against 13.40 / 13.50 / 13.60; with the old call selected in the new
code (GLM5_CPU_LAYER=0) 12.50-12.87 against 13.17-13.60 over four rounds. About +6%. llama.cpp
was not re-measured that day.
A per-call dynamic thread schedule in C was tried in the same replay and was slower (40.1-40.6
ms/token) than the static NUMA split; it was removed.

The speculative path got the same treatment the same day: `moe_cpu_layer_t` takes one mask with
8 bits per token, builds the union of both tokens' misses in C in the order the Python wrapper
used, and calls the same `moe_cpu_numa_t`; the draft head's block 45 goes through it too.
Bit-identical to the old wrapper (check_cpu_layer.py: 80 calls with 1 and 2 tokens, shared
experts and partial masks; 40 calls for the one-token entry). run_spec.sh, warm, story prompt
@128, four interleaved rounds after one discarded warm-up:

| | tok/s | ms per verify step | tokens/step |
|---|---|---|---|
| published checkout (9887cdf) | 13.92 / 13.23 / 13.61 / 13.81 | 131.3 / 138.2 / 130.6 / 130.6 | 1.83 / 1.83 / 1.78 / 1.80 |
| moe_cpu_layer_t | 14.34 / 14.44 / 14.55 / 14.88 | 122.3 / 124.8 / 123.9 / 122.9 | 1.75 / 1.80 / 1.80 / 1.83 |

Means 13.64 -> 14.55 tok/s (+7%), 132.7 -> 123.5 ms per step with the same ~330 union experts per
step: the CPU phase went from 85-88 to 78-80 ms and host pre from 4.1 to 3.5 ms. Draft acceptance
varies run to run (77-84%), so the per-step time is the like-for-like figure. GLM5_CPU_LAYER=0
selects the old wrapper; the PCIe streaming mode (GLM5_STREAM=1) keeps it.

## MTP speculative decoding (glm5_spec.py, mtp.py)
- Draft: block 45 NextN, x = eh_proj([enorm(emb(t+1)); hnorm(h_t)]) with h_t the trunk's
  post-output_norm hidden, plain-residual MLA + MoE (no hyper-connections), shared_head_norm,
  output head. llama.cpp has no MTP graph for glm5-next; the layout was picked by acceptance:
  [emb; h] post-norm 78%, pre-norm h 73%, [h; emb] 0%.
- Verify [last, draft] in one pass: Q8 trunk matrices read once for both tokens (q8.q8mm, the
  155k-row head 1.29 ms vs 1.13 for one token), CPU experts over the union of both tokens'
  misses (moe_cpu_numa_t), GPU hits per token. Reject -> KDA recurrent+conv state rolls back to
  the snapshot taken after token 0 (one captured graph); MLA rows need no rollback.
- Why the gain is small: the CPU expert work is per expert, and two consecutive tokens share
  few experts (union 1.74-1.78x the misses of one token, routing trace) - about the 1.8 tokens a
  step yields - so only the trunk, host glue and head amortize, minus the 10 ms draft.
- Exactness (GLM5_SPEC_CHECK=n): row 0 of a verify is bit-identical whatever the draft token
  (causal), both paths are deterministic, and the 2-token path differs from the 1-token path by
  ~1e-2 rel. at the first position (accumulation order: q8mm vs q8mv, GEMM vs GEMV); top-8 routing
  flips at near-ties then amplify drift over later positions, like run-to-run expert-set noise.

## Findings worth keeping
- torch loads its bundled libgomp RTLD_GLOBAL: OpenMP in a ctypes library binds to it (team =
  torch's thread count). Load with RTLD_DEEPBIND, set num_threads explicitly, pin in C.
- In-engine, one CPU thread per physical core beats using hyperthreads (both kernels), although
  standalone benchmarks favour 16 threads.
- The CPU expert kernel is memory-latency bound for one token (ggml AVX2 and own AVX-512 dots
  equal, 0.29 ms/expert) but compute bound for the second token on an L1-hot row (+56%).
  Prefetching 2-4 KB ahead inside the dot (across row ends; the L2 streamer stops at 4 KB pages)
  gives 0.22 ms/expert; prefetching whole rows ahead from the row loop was 3x slower
  (bench_kern.py). Own dots match ggml's integer math (test_avx512.py).
- Token embedding: gguf.quants.dequantize of one Q8_0 row took up to 9 ms; direct numpy 0.06 ms.
- NUMA: splitting each expert's rows across nodes (move_pages once; persists in page cache) and
  computing node-locally: 0.36 -> 0.28 ms/expert, bit-identical.
- torch.cuda.graph's default capture stream is created on the first capturing device: pass
  stream= when capturing on another GPU (otherwise the graph is empty).
- Compiled bodies capture only if pinned D2H copies stay outside the compiled region.
- Adaptive GPU expert set (adapt.py, GLM5_ADAPT): hit rate 38% -> 49-53% as simulated, but no
  speedup - evicted experts must come back from NVMe (RAM < model) and the page-cache churn /
  on-demand faults slow the CPU kernel by as much as the extra hits save. Off by default.
- Adaptive set without disk traffic (2026-11-10, GLM5_PIN_PER_LAYER=21 with GLM5_ADAPT): 21
  GPU experts per layer are GPU-only, every other expert keeps a NUMA-placed RAM copy (172.0 GB
  mapped), so a swap is one 15 MB DMA and two table writes. Warm, story @128, static 13.2-13.6
  the same day: 4 swaps/token 12.4-12.7 tok/s (hit rate 47-49.5%, CPU kernel 4.8-5.0 s instead of 5.3),
  8 swaps/token 11.6-11.7 (51%). The swaps cost more than the hits save: with GLM5_TL=1 the host
  time around the graph launches grows from 8.2 to 13.6 ms/token while the CPU phase shrinks by
  2.5. Page-locked staging buffers instead of a cudaHostRegister per swap (GLM5_ADAPT_SRC=stage)
  were slower still (12.0-12.1): the extra memcpy runs against the CPU kernel. After a run,
  GLM5_ADAPT_VERIFY=1 compares every GPU slot with the GGUF bytes (1760 experts, 0 differ).
- More expert slots (2026-11-10): 21+25 per layer instead of 19+23 fills the cards to a peak of
  23,462 of 24,576 MiB and 15,640 of 16,380 MiB. Hit rate 42-43% -> 43.5-46%, 13.36-13.68 tok/s against 13.17-13.60 over
  four interleaved rounds: no difference outside run-to-run noise. Default unchanged.

## Fidelity
Step-1 top-5 token set equals llama.cpp's and argmax matches; hit-graph self-test (GLM5_CHECK=1)
rel.err ~5e-7 vs the eager path. Divergence at near-ties (e.g. step 3 of the Paris prompt) is not isolated; likely contributors
are llama.cpp's 8-bit activation quantization and different accumulation order (fp16 and fp32
paths). Gate against the fp32 reference, not token-for-token llama.cpp.

- PCIe streaming of CPU misses (stream.py, GLM5_STREAM=1, off by default): all 158 GB of CPU-side
  experts page-locked in 16 s (cudaHostRegister read-only, 4596 merged ranges), and per layer a
  cost model sends some misses to the Quadro by DMA into VRAM slots. bench_pcie.py: 12.1 GB/s to
  the Quadro, 6.2 to the 4060 Ti, 12.4 for both together (shared limit). In-engine it loses:
  spec 13.0 -> 12.0 (2 slots) / 12.3 (1 slot, conservative), plain 12.5 -> 11.8. The DMA slows
  the memory-latency-bound CPU kernel (0.255 -> 0.287 ms/expert) and half of each NUMA-split
  expert crosses the socket link; the CPU phase got longer, not shorter.

- Cold start (2026-11-09): with both NUMA nodes nearly full after a cold read, move_pages cannot
  migrate the expert pages (no free page on the target node): 4.4M pages stayed put and the first
  start after an evicted cache decoded at 8.2 tok/s. The populate step now reads each expert's
  row-part k on node k with readahead off (pread, then MADV_POPULATE_READ only maps), so the pages
  are allocated where the kernel uses them: 0.2-0.7M stragglers, first starts at 9.2-12.8 tok/s and
  12.9-13.7 from the second start. Some first starts still take page faults during decode.

## Open levers
MTP draft in fewer syncs; more fusion of the trunk glue. More RAM alone would not make adaptive swapping pay (see the 2026-11-10 test above).
