"""Tarfa repo smoke-test + benchmark. Imports the engine ONLY from the repo copy
(proves the repo is self-contained), then runs prefill + timed greedy decode.
Mode from env: BF16=1 -> exact BF16 track; default -> fast fused-int4 track.
Writes JSON to validation/bench_<mode>.json.
"""
import os, sys, json, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(REPO, "tarfa", "engine")
sys.path.insert(0, ENGINE)                       # repo engine FIRST — no /tmp, no runner
for p in list(sys.path):
    if p.startswith("/tmp") or "airllm-test/runner" in p:
        sys.path.remove(p)

import torch
MODE = "exact_bf16" if os.environ.get("BF16") == "1" else "fast_int4"
from fused_int4 import TarfaFused       # resolves inside repo engine/
import fused_int4
assert fused_int4.__file__.startswith(ENGINE), f"LEAK: engine imported from {fused_int4.__file__}"
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

t0 = time.time()
M = TarfaFused("Qwen/Qwen3.5-122B-A10B")
for i in range(M.nL): M._make_layer(i)
M._mat(M.lm.embed_tokens, M.prefix + "embed_tokens")
if not M.tied: M._mat(M.model.lm_head, "lm_head")
load_s = time.time() - t0
print(f"[{MODE}] engine={fused_int4.__file__}", flush=True)
print(f"[{MODE}] load {load_s:.1f}s vram {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

@torch.no_grad()
def fwd(tl, cache):
    cur = torch.tensor([tl], device=M.device); past = cache.get_seq_length(); S = cur.shape[1]
    pos = torch.arange(past, past + S, device=M.device); h = M.lm.embed_tokens(cur)
    if M.mrope:
        p4 = pos.view(1, 1, -1).expand(4, 1, -1); tp, rp = p4[0], p4[1:]
    else: tp = rp = pos.unsqueeze(0)
    pe = M.lm.rotary_emb(h, rp)
    cz = create_causal_mask(config=M.cfg, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=tp)
    for i in range(M.nL):
        m = None if M.layer_types[i] == "linear_attention" else cz
        o = M._make_layer(i)(h, position_embeddings=pe, attention_mask=m, position_ids=tp, past_key_values=cache, use_cache=True)
        h = o[0] if isinstance(o, tuple) else o
    h = M.lm.norm(h)[:, -1:, :] if hasattr(M.lm.norm, "weight") and M.lm.norm.weight.device.type == "cuda" else h[:, -1:, :]
    return M.model.lm_head(h).float()[0] if not M.tied else torch.matmul(h.float(), M.lm.embed_tokens.weight.float().t())[0]

M._mat(M.lm.norm, M.prefix + "norm")
PROMPT = "The three most important considerations when designing a database index are"
ids = M.tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
cache = DynamicCache(config=M.cfg)
tp0 = time.time(); fwd(ids[:-1], cache); prefill_s = time.time() - tp0
pending = ids[-1]; toks = []
N = int(os.environ.get("BENCH_N", "24"))
td0 = time.time(); times = []
for k in range(N):
    t1 = time.time()
    pending = int(fwd([pending], cache).argmax())
    times.append(time.time() - t1); toks.append(pending)
decode_s = time.time() - td0
text = M.tok.decode(toks)
import statistics as st
res = {"mode": MODE, "load_s": round(load_s, 1), "prefill_s": round(prefill_s, 1),
       "prefill_tokens": len(ids) - 1, "decode_tokens": N,
       "s_per_tok_mean": round(decode_s / N, 3), "s_per_tok_median": round(st.median(times), 3),
       "tok_per_s": round(N / decode_s, 3), "continuation": text,
       "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(os.path.join(REPO, "validation", f"bench_{MODE}.json"), "w"), indent=2)
print(f"BENCH_DONE_{MODE}", flush=True)
