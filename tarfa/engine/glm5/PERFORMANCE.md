# glm5-next decode performance (2026-11-06/07)

Lab notebook of the optimization steps, in order; the step figures are single runs from that
day. The headline comparison with all runs is in README.md and `validation/glm5/bench_story.json`.

Machine: 2x Xeon Gold 5122 (2x4 cores, HT, 2 NUMA nodes, 6x DDR4-2666), 183 GiB RAM,
Quadro RTX 6000 24 GB (cuda:1, sm_75) + RTX 4060 Ti 16 GB (cuda:0, sm_89). Model GGUF
GLM-5.3-Flash-UD-Q4_K_XL, 186 GiB (experts Q4_K/Q5_K/Q6_K, trunk Q8_0) - does not fit RAM.
Baseline: llama.cpp production flags (-ncmoe 40, -ts 41,4, -t 16), story prompt @128: 2.2-7.9 tok/s over 19 runs, best 7.88 (-t 8: best 6.87 over 8 runs).

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

Per token at 11.1 tok/s (GLM5_TL=1): CPU experts ~57 ms (~185 misses x 0.29 ms), trunk graphs
~21 ms, host glue ~9 ms; GPU hits (~16 ms) overlap the CPU work.

Speculative step at 13.0 tok/s (1.80 tokens/step, 82% drafts accepted): 138 ms = CPU experts 88
(330 union experts), trunk graphs 27, host glue 10, MTP draft 10 (CPU experts 3.5).

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

## Open levers
MTP draft in fewer syncs; more RAM (whole model cached would make adaptive swapping pay); more
fusion of the trunk glue.
