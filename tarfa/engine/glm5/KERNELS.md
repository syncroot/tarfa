# GLM-5.3-Flash (`glm5-next`) — reverse-engineered kernel spec

Day-1 Tarfa port notes, kept as dated lab notes: later sections correct earlier ones.
 Source of truth: llama.cpp's GLM build of 2026-10-03
(`src/models/{glm5-next,glm-dsa,delta-net-base}.cpp`).
GGUF: `GLM-5.3-Flash-UD-Q4_K_XL` (320B total, ~18B active; 288 routed experts per MoE layer, 8 selected).

## Shapes
- `n_embd` 4096, `vocab` 154880, `block_count` 46 = 45 trunk layers + block 45, the NextN/MTP head
- `n_head` 64, `head_dim (kda)` 128, `d_inner` = 64*128 = 8192, `d_conv` 4
- MLA: `q_lora_rank` 1536, `kv_lora_rank` 512, `key_len_mla` 256, `val_len_mla` 256
- Indexer (DSA): `n_head` 32, `key_len` 128, `top_k` 2048, `kpool` 4
- MoE: 288 experts, 8 used + 1 shared, `n_ff_exp` 2048, gating = **sigmoid**,
  `expert_weights_norm` true, `expert_weights_scale` 2.5; dense `n_ff` 12288
- mHC hyper-connections: `hc_mult` 4, sinkhorn iters 20, eps 1e-6
- `kda_gate_lower_bound` -5.0

## Layer pattern
- Layers 0-2: leading **dense** FFN. Attention = KDA (recurrent, `n_head_kv==0`).
- Then repeating **3x KDA + 1x MLA**. MLA layers are `is_recr(il)==false`
  (`n_head_kv==1`). 34 KDA and 11 MLA over the 45 trunk layers.
- Every layer = two mHC-wrapped sub-blocks: (attn_norm -> KDA|MLA) then
  (ffn_norm -> FFN|MoE+shared). `build_hc_pre/post` mix 4 residual streams;
  `inpL` starts repeated 4x (hc_mult 4).

## KDA block  (`build_kda_layer`, decode path = `build_delta_net_autoregressive`)
Input `x` = RMSNorm(attn_norm, hidden) : [4096].
1. For each of Q,K,V: `proj_w @ x` (4096->8192) -> causal depthwise conv1d over
   `d_conv=4` taps (per-channel weight `[d_conv, d_inner]`, carries a conv state
   of the last 3 inputs) -> **SiLU** -> reshape `[head_dim=128, n_head=64]`.
2. Decay gate g1: `f_a = ssm_f_a @ x` (4096->128); `g1 = ssm_f_b @ f_a` (128->8192);
   `g1 += ssm_dt_b`. Reshape `[128,64]`. `A = ssm_a` ([64], holds **-exp(A_log)**).
   Since gate_lower_bound(-5.0) > -inf:  `g1 = sigmoid(-(g1 * A)) * (-5.0)`.
   (A broadcasts over head_dim.) g1 is the per-(head,head_dim) **log-decay**.
3. beta: `sigmoid(ssm_beta @ x)` -> `[1, n_head]`.
4. **L2-normalize** Q,K over head_dim (eps 1e-6, "FLA l2 norm"). Scale Q by 1/sqrt(128).
5. Gated delta rule, per head, state `s=[head_dim, head_dim]` (=[S_v,S_v]):
   decode step (n_tokens==1), KDA => gate g has the head_dim axis:
       g = exp(g1)                      # [head_dim]  (per head)
       s = s * g                        # decay along the S_k axis
       sk = sum_rows(s * k)             # [S_v]
       d  = (v - sk) * beta             # delta
       s  = s + outer(k, d)             # rank-1 update  (k_repeat * d^T)
       o  = sum_rows(s * q)             # [S_v]
   (chunked prefill path is `build_delta_net_chunking`: triangular solve +
   cumulative-decay mask; not needed for the decode probe.)
6. Output gate: `g_a = ssm_g_a @ x` (4096->128); `g2 = ssm_g_b @ g_a` (128->8192);
   `o = RMSNorm(o_scan, ssm_o_norm[head_dim])`; `gated = o * sigmoid(g2)`.
7. `cur = wo @ gated` (8192->4096).

## MLA block  (`build_dsa_layer` / `glm-dsa.cpp`) — summary, for the later probe
- Q: `wq_a` (4096->1536) -> RMSNorm(q_a_norm) -> `wq_b` -> per-head [n_embd_head_k].
- KV: `wkv_a_mqa` (4096 -> kv_lora_rank+rope) -> split; RMSNorm(kv_a_norm) on the
  512 latent; `wk_b`/`wv_b` expand per head. RoPE on the rope part only ("nope MLA"
  = partial rope). Attention over the compressed latent cache.
- DSA indexer (k-pool): `indexer_attn_q_b`, `indexer_attn_k` (+k_norm),
  `indexer_kpool_gate/ape`, selects top_k=2048 keys pooled in groups of kpool=4.
  Sparse attention — the hard part of the MLA layers.

## FFN
- Dense (layers 0-2): SiLU-gated `ffn_up/ffn_gate/ffn_down`, n_ff=12288.
- MoE: `build_moe_ffn` sigmoid gating, weights-norm, scale 2.5, 8/288 +
  `exp_probs_b` bias; plus a shared expert (`*_shexp`) added to the MoE output.

## Oracle / gating
- No `transformers` support for `glm5-next` (5.17). Reference oracle = llama.cpp
  (`llama-eval-callback`, or patched full-tensor dump) on the Oct-3 GLM build.
- cb() tap names to diff at layer 0: `attn_norm`, `kda_q_conv`, `kda_g1`,
  `kda_beta`, `kda_scan_out`, `kda_normed`, `kda_out`, `ffn_out`, `l_out`.
- Gate target: per-token hidden after layer 0 (`l_out`), cosine >= ~0.999 vs oracle.

## VALIDATION (2026-11-06) — KDA block confirmed
Oracle = patched `llama-eval-callback` (full-tensor dump, CPU-only / `CUDA_VISIBLE_DEVICES=""`,
deterministic) on the Oct-3 GLM build, prompt "The capital of France is", token 0
(fresh conv+recurrent state). `probe2.py` vs oracle taps, cosine:
  q/k/v_conv 0.999999 | g1 0.999997 | beta 1.000000 | scan_out 0.999999 |
  g2 0.999995 | normed 0.999998 | **kda_out 0.999959** (max|diff| 1.7e-4, rms exact)
Residual gap = Q8_0 dequant + fp32 order (matches the conv's own 2e-4) => consistent with
quantization and accumulation order, not an algorithmic gap. The KDA gated-delta kernel is CORRECT.

**Gotcha that cost the first run:** the ssm output RMSNorm uses eps = 1e-5
(`attention.layer_norm_rms_epsilon`), NOT 1e-6. `scan_out` rms is ~2.2e-4, so the +eps
term dominates the denominator; eps=1e-6 gave a ~3x magnitude error (cos 0.987) even
though every op was structurally right. Also: GGUFReader returns shapes in ne order
(in, out) not reversed — reshape the flat dequant to logical (out, in) explicitly.

Oracle dumps live in `<dump_dir>/dumps/` (and `dumps2/` for the full layer); they are not in the repo.

## VALIDATION 2 (2026-11-06) — full layer-0 (mHC + KDA + dense FFN) confirmed
`probe3.py` feeds the oracle `hc_init` (embedding streams) through the whole layer and
matches every tap, cosine:
  attn_norm 1.000000 | kda_out 0.999959 | hc_attn_post 0.999972 |
  ffn_norm 0.999969 | ffn_out 0.999941 | **l_out 0.999971**
So the mHC 4-stream residual (hc_pre RMS-norm->mix->sinkhorn comb->collapse; hc_post
recombine) and the dense FFN are correct. This completes the KDA-type decoder layer,
covering all 34 KDA layers (the `is_recr` layers; layers 0-2 are the dense-FFN variant).
Implemented in `glm5_layer.py`.

mHC notes: streams carried as [hc=4, n_embd]; hc_pre flat-RMS-norms the 4x4096 concat
(NO weight, eps 1e-5), projects via hc_*_fn to 24 = [pre(4) | post(4) | comb(16)];
pre=sigmoid(.)+eps, post=2*sigmoid(.), comb reshaped [dst,src] then Sinkhorn
(softmax over dst, then col-norm, then 19x (row-norm,col-norm)). collapse = Σ_s pre[s]*x[s].
hc_post[dst] = post[dst]*block_out + Σ_src comb[dst,src]*residual[src].

## VALIDATION 3 (2026-11-06) — MLA/DSA attention (layer 3) confirmed
`probe4.py` feeds oracle `attn_norm-3` (all 5 tokens) through `mla_ref.mla_forward`,
cosine: q_resid 0.999994 | kv_cmpr 0.999996 | kqv_out 0.999975 | **attn_out 0.999977**.
MLA is nope-only (n_rot=0): qr=RMSNorm(wq_a x); q=wq_b qr -> [256,H]; kv=RMSNorm(wkv_a_mqa x)
-> 512 latent; q absorbed by wk_b into the 512 space; causal softmax over the latent cache
(scale 1/sqrt(256)); value = latent, projected by wv_b; wo. The k-pool indexer is a NO-OP
here (5 keys << top_k 2048) so its PRUNING is not exercised — only long-context gating would.

### ALL THREE NOVEL KERNELS VALIDATED: KDA (gated-delta), mHC (sinkhorn 4-stream), MLA.

## VALIDATION 4 (2026-11-06) — MoE FFN (layer 3) confirmed
`probe5.py` feeds oracle `ffn_norm-3` (token 0) through `moe_ref.MoE`: selected experts
[10,20,97,149,202,205,252,285]; cosine moe 0.999815 | shexp 0.999976 | **ffn_out 0.999952**.
MoE: sigmoid(router x) + exp_probs_b bias used ONLY for top-8/288 SELECTION; weights are the
PRE-bias probs, L1-normalized, x2.5; each routed expert and the shared expert use a clamped
SwiGLU; sum. Experts dequantized per-selected-expert (Q4_K gate/up, Q5_K down) so only 8 of
288 are materialized. **Gotcha:** the FFN is a CLAMPED SwiGLU (swiglu_clamp_exp/shexp = 10.0):
gate clamped to (-inf, 10], up to [-10, 10], then silu(gate)*up. Without it shexp was 0.9977
(a few >10 activations); with it 0.999976. Also the bias tensor is `exp_probs_b.bias`
(not `ffn_exp_probs_b`).

## VALIDATION 5 (2026-11-06) — full 45-layer forward: ARGMAX MATCH
`full_forward.py` runs all 45 trunk layers (block 45 = NextN/MTP head, skipped) from the
oracle hc_init on "The capital of France is" (5 tokens), then output_norm + output.
Result: **next-token argmax 12089 == llama.cpp**, top-5 the same set (reordered); logits
cos 0.958, final-hidden cos 0.883. (Pre-fix figures: after the KDA key-axis fix in the CORRECTION
section below, the same forward gives logits cos 0.989 and final-hidden cos 0.972, argmax and
top-5 set unchanged.) The lower cos is expected accumulation over 45 layers
in numpy fp64 + the KDA recurrence at positions >0 (only pos 0 was isolated-validated);
the decisive metric (the model's actual prediction) matches. n_layer()=block_count-nextn=45.

## PHASE A finding (2026-11-06) — KDA decode-AR vs prefill-chunked
`probe6.py`/`probe6b.py` run the KDA recurrence sequentially over all 5 prefill tokens and
compare per position. Result: pos0 scan 0.999999 / out 0.999959, but pos>0 drifts
(pos2 scan 0.965 / out 0.902). Isolated to the SCAN (delta recurrence), not the output gate.
Cause: `kda_decode_step` is a faithful transcription of llama.cpp's AUTOREGRESSIVE path
(`build_delta_net_autoregressive`), but the oracle ran a 5-token PREFILL, which uses the
CHUNKED path (`build_delta_net_chunking`: triangular solve + cumulative-decay masks). The
two are mathematically equivalent but differ numerically across a chunk (identical for 1
token -> pos0 matches). So this is NOT a decode bug; it is AR-vs-chunked, and it also
explains the full-forward's 0.88 hidden cos (the forward used sequential AR over a prefill
the oracle did chunked; argmax still matched). TODO to fully close prefill correctness:
implement the chunked path and validate IT against this oracle. The AR path is correct for
DECODE (the tok/s-critical path).

## PHASE A finding 2 (2026-11-06) — k-pool indexer validated at long context
`probe7.py` on a 2801-token prefill (704 pools; top_k=2048/kpool=4 -> keep 512 pools, so
**pruning is ACTIVE**: 192 pools / 768 keys dropped per query). Using oracle indexer
intermediates (iq, pooled_k, weights), recompute the lightning-indexer score
`score[p,t] = sum_h relu(iq[:,h,t]·pooled_k[:,p])·weights[h,t]` + causal pool mask:
  token 2800 (700 pools visible, real pruning): **score cos 1.000000, top-512 overlap 0.998**
  (the 1 differing pool is a score tie at the cutoff; selection is otherwise exact).
(A mid-token with <512 visible pools is a don't-care: nothing is pruned and the oracle
pads the selection with arbitrary masked slots.) The sparse indexer scoring + top-k pool
selection are confirmed. Open: the chunked KDA prefill path (above) is the last gap.

### EVERY glm5-next COMPONENT NOW VALIDATED vs llama.cpp (cos >= 0.9998):
_(Written before the two sections above it were added; the full forward and the indexer probe
it lists as remaining are done there.)_
KDA, mHC (sinkhorn 4-stream), dense FFN, MLA attention, MoE FFN. A full 45-layer forward is
assemblable from validated pieces. Remaining: wire the full forward + end-to-end logit gate;
a long-context probe for the sparse indexer PRUNING (the only path not exercised); then the
performance work (CUDA graphs for KDA/trunk, dual-GPU hot experts) to beat llama.cpp's ~5 tok/s.

## CORRECTION (2026-11-06, later) — the Phase A "AR-vs-chunked" conclusion was WRONG
The pos>0 KDA drift was a real bug, not numerics. llama.cpp's chunked prefill decays the
KDA state along the KEY axis (`g_last_exp_t` is `[S_k,1]` -> multiplies ne0), but its
non-fused `build_delta_net_autoregressive` reshapes g to `[1,S_k]` and multiplies ne1 (the
VALUE axis). For scalar-gate GDN the axis is irrelevant; for KDA's per-channel gate it is
wrong (llama.cpp's real decode uses the fused `ggml_gated_delta_net` op instead). I had
transcribed the non-fused path. Position 0 cannot see it (state = 0), which is why every
pos-0 probe passed.
Fix: `s = s * g[:, :, None]` (decay the key axis). probe6/probe6b vs the prefill oracle:
block-output cos at positions 0..4 = 0.99996 / 0.99996 / 0.99998 / 0.99996 / 0.99998
(was 0.99996 / 0.9968 / 0.902 / 0.9963 / 0.963). End to end, greedy decode went from a
degenerate loop ("Paris. Paris is the capital of France." repeated) to fluent text. Still
diverges from llama.cpp's greedy text at token 3 (near-tie / precision; under study). Later, with the final engine, the top-5 set matches
llama.cpp's (validation/glm5/fidelity_france.json); the cause of near-tie splits is still not isolated.
