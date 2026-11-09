# Tarfa × GLM-5.3-Flash — day-1 feasibility verdict

_2026-11-06. Historical note from the first day of the port; the engine and its numbers came later._

## Verdict

**An exact Tarfa port of GLM-5.3-Flash (`glm5-next`, 320B-A18B) is feasible.** Every
component of the architecture has been reproduced in a numpy reference and validated
against llama.cpp (the Oct-3 GLM build, CPU-only deterministic, via a patched
`llama-eval-callback` full-tensor dump). The full 45-layer forward, assembled only from
those validated kernels, predicts the same next token as llama.cpp.

This was reached in one day on a model Tarfa had never handled, starting from the
discovery that the previous Flash-Next source weights had been deleted.

| component | probe | cosine vs llama.cpp |
|---|---|---|
| KDA conv + projection | probe2 | 0.999999 |
| KDA gated-delta block (decode) | probe2 | **0.999959** |
| mHC 4-stream residual (sinkhorn) + dense FFN → full layer-0 `l_out` | probe3 | **0.999971** |
| MLA/DSA latent attention | probe4 | **0.999977** |
| MoE FFN (router + 8/288 + shared, clamped SwiGLU) | probe5 | **0.999952** |
| full 45-layer forward | full_forward | logits 0.958 (0.989 after the KDA fix), **next-token argmax == llama.cpp** |

Residual per-kernel error (~1e-4) is Q8_0/Q4_K dequant + fp32 order — consistent with the
model's quantization and accumulation order, not an algorithmic gap.

### What `glm5-next` is (the three things that made it look hard)
- **KDA** — Kimi gated-delta linear attention: causal depthwise conv1d(k=4)+SiLU on q/k/v,
  a per-(head,head_dim) decay gate `sigmoid(-(f_b·f_a(x)+dt)·A)·(-5)`, β=sigmoid, L2-norm
  q/k, then the gated delta rule with a rank-1 state update; output gate RMSNorm·sigmoid.
- **mHC** — multi-hyper-connection: 4 residual streams, mixed each sub-block by a
  Sinkhorn-normalized (20 iters) 4×4 combine matrix + per-stream pre/post gates.
- **MLA** — DeepSeek-style latent attention, nope-only (no rope), q-lora 1536 / kv-lora 512,
  `wk_b` absorbing the query into the 512 latent, `wv_b` projecting out; plus a k-pool
  sparse indexer (top_k 2048) that selects which keys to attend.
- MoE: sigmoid router, `exp_probs_b` bias for top-8/288 SELECTION only, weights from the
  pre-bias probs (L1-norm ×2.5), clamped-SwiGLU experts + one shared expert.

### Gotchas (cost real debugging time; documented in KERNELS.md)
1. ssm output RMSNorm eps is **1e-5** (not 1e-6); `scan_out` is tiny so +eps dominates — a
   wrong eps gave a 3× magnitude error with every op otherwise correct.
2. The FFN is a **clamped SwiGLU** (limit 10): gate ≤ 10, up ∈ [-10,10], then silu(gate)·up.
3. GGUFReader returns shapes in ne order; reshape flat dequant to logical `(out, in)`.
4. The MoE bias tensor is `exp_probs_b.bias`, not `ffn_exp_probs_b`.
5. `n_layer = block_count - nextn_predict_layers = 45`; block 45 is the NextN/MTP head.

### Open items (honest)
- **KDA at positions >0** was not isolated-validated (only position 0 / fresh state was).
  The full-forward argmax match is indirect evidence it is correct; the final-hidden cosine
  of 0.883 is 45-layer accumulation in numpy fp64, but could partly be a pos>0 KDA detail.
  (It was: after the key-axis fix the final-hidden cosine is 0.972 and the logits 0.989.)
  → Add a multi-token KDA probe (compare `kda_scan_out-0` across all 5 token columns).
- **Sparse indexer pruning** is untested: with ≤2048 keys it selects everything, so the
  short-prompt probes never exercise the top-k. → Needs a long-context (>2048 token) probe.
- This is a numpy correctness reference, not an engine. No performance work was done.

## What came next

The day-2 plan (port the validated kernels into a torch/CUDA engine, then CUDA graphs, GPU
expert residency and MTP speculative decoding) became `glm5_gdecode.py` and `glm5_spec.py`;
results and findings are in [PERFORMANCE.md](PERFORMANCE.md) and [README.md](README.md). The
KDA position>0 open item was a real bug: I had transcribed llama.cpp's non-fused autoregressive
path, which decays the value axis, while KDA's state decays along the key axis (KERNELS.md,
"CORRECTION"; probe6/probe6b).

## Reproduce
```
cd tarfa/engine/glm5
python probe2.py <GGUF_DIR> <DUMP_DIR>/dumps    # KDA
python probe3.py <GGUF_DIR> <DUMP_DIR>/dumps2   # full layer-0
python probe4.py <GGUF_DIR> <DUMP_DIR>/dumps3   # MLA
python probe5.py <GGUF_DIR> <DUMP_DIR>/dumps4   # MoE
python full_forward.py <GGUF_DIR> <DUMP_DIR>/dumps5   # end-to-end (~450s)
```
Oracle = llama.cpp's `llama-eval-callback` (GLM build of 2026-10-03), patched to dump full
tensors for `GLM5_DUMP` names. The dumps are not in the repo. GGUF_DIR holds the
`GLM-5.3-Flash-UD-Q4_K_XL` shards.
