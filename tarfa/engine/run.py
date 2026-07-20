import os
# -*- coding: utf-8 -*-
"""Tarfa driver — build the fused-kernel engine, prefill, then a timed greedy decode loop.
Measures real streaming tok/s and prints the continuation to confirm correctness vs the exact path."""
import sys, time, torch
sys.path.insert(0, "/tmp"); sys.path.insert(0, os.path.expanduser("~/airllm-test/runner"))
from fused_int4 import TarfaFused
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

M = TarfaFused("Qwen/Qwen3.5-122B-A10B")
for i in range(M.nL): M._make_layer(i)
M._mat(M.lm.embed_tokens, M.prefix + "embed_tokens")           # input embed resident
if not M.tied: M._mat(M.model.lm_head, "lm_head")              # output proj resident (non-tied)
print(f"residency built (embed+lm_head resident, tied={M.tied})", flush=True)

@torch.no_grad()
def big_fwd(token_list, cache):
    cur = torch.tensor([token_list], device=M.device); past = cache.get_seq_length(); S = cur.shape[1]
    pos = torch.arange(past, past + S, device=M.device)
    h = M.lm.embed_tokens(cur)                                  # embed resident: free input lookup
    if M.mrope:
        p4 = pos.view(1,1,-1).expand(4,1,-1); text_pos, rope_pos = p4[0], p4[1:]
    else: text_pos = rope_pos = pos.unsqueeze(0)
    pe = M.lm.rotary_emb(h, rope_pos)
    causal = create_causal_mask(config=M.cfg, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=text_pos)
    for i in range(M.nL):
        lt = M.layer_types[i] if M.layer_types else "full_attention"
        mask = None if lt == "linear_attention" else causal
        o = M._make_layer(i)(h, position_embeddings=pe, attention_mask=mask, position_ids=text_pos, past_key_values=cache, use_cache=True)
        h = o[0] if isinstance(o, tuple) else o
    M._mat(M.lm.norm, M.prefix + "norm"); h = M.lm.norm(h); M._free(M.lm.norm)
    if M.tied:
        logits = torch.matmul(h.float(), M.lm.embed_tokens.weight.float().t())   # resident, no reload
    else:
        logits = M.model.lm_head(h).float()                                     # resident lm_head, no reload
    return logits[0]

prompt = "Write a detailed paragraph about how rivers shape the landscape over geological time."
ids = M.tok(prompt, return_tensors="pt").input_ids[0].tolist()
cache = DynamicCache(config=M.cfg); big_fwd(ids[:-1], cache); pending = ids[-1]
for _ in range(6): pending = int(big_fwd([pending], cache).argmax())          # warm cache + kernel
torch.cuda.synchronize(); t0 = time.time(); out = []
for _ in range(80):
    pending = int(big_fwd([pending], cache).argmax()); out.append(pending)
torch.cuda.synchronize(); dt = time.time() - t0
print(f"TARFA tok/s = {80/dt:.2f}   (80 tok / {dt:.0f}s)", flush=True)
print("OUTPUT:", M.tok.decode(out).replace(chr(10), " ")[:220], flush=True)
print("TARFA_DONE", flush=True)
