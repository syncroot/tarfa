import sys, json, threading, copy, os, subprocess, time, torch, requests
sys.path.insert(0, os.path.expanduser("~/airllm-test/runner")); sys.path.insert(0, "/tmp")
from safetensors import safe_open
from fused_int4 import TarfaFused as ChannelModel      # fused Triton MoE kernel
import dreamer
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
import uvicorn

LMSTUDIO = "http://localhost:1234"                 # gateway: route other models here
OUR_ID = "qwen3.5-122b-tarfa"                        # the fused-kernel 122B
import quality                                  # decode-time quality strategies (N x compute -> better reasoning)
import fused_int4, exact_bf16
QMODES = {"tarfa-vote-5": ("vote", 5), "tarfa-judge-3": ("judge", 3), "tarfa-refine-2": ("refine", 2)}
QLABEL = {"tarfa-vote-5": "Tarfa 122B — Self-consistency ×5 (vote)",
          "tarfa-judge-3": "Tarfa 122B — Best-of-3 + judge",
          "tarfa-refine-2": "Tarfa 122B — Self-refine ×2"}
DRAFT_ID = "qwen3.5-0.8b"                           # the small model, talkable on its own
LMS_CLI = os.path.expanduser("~/.lmstudio/bin/lms")
BACKEND = {"cur": "122b", "model": None}
LAST_REQ = {"ts": 0.0, "model": None}
PROFILE = {"mode": "bf16" if os.environ.get("BF16") == "1" else "int4",
           "bf16_prefill_batch": int(os.environ.get("BF16_PREFILL_BATCH", "16")),
           "bf16_cache_per_layer": int(os.environ.get("BF16_CACHE_PER_LAYER", "0")),
           "bf16_pinned": os.environ.get("BF16_PINNED", "1") == "1",
           "bf16_ram_cache_gb": float(os.environ.get("BF16_RAM_CACHE_GB", "0")),
           "bf16_batched_decode": os.environ.get("BF16_BATCHED_DECODE", "0") == "1",
           "bf16_fadvise_top": int(os.environ.get("BF16_FADVISE_TOP", "0")),
           "forward_calls": 0, "prefill_calls": 0, "decode_calls": 0,
           "prefill_tokens": 0, "decode_tokens": 0,
           "prefill_seconds": 0.0, "decode_seconds": 0.0,
           "layer_seconds": 0.0, "last_request_seconds": 0.0}

def system_stats():
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1); mem[key] = int(value.strip().split()[0])
    except Exception: pass
    vm = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                key, value = line.split();
                if key in ("pgmajfault", "pswpin", "pswpout"): vm[key] = int(value)
    except Exception: pass
    io = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                key, value = line.split(":", 1)
                if key in ("read_bytes", "rchar"): io[key] = int(value)
    except Exception: pass
    return {"mem_available_gb": round(mem.get("MemAvailable", 0) / 1048576, 3),
            "cached_gb": round((mem.get("Cached", 0) + mem.get("SReclaimable", 0)) / 1048576, 3),
            "swap_free_gb": round(mem.get("SwapFree", 0) / 1048576, 3),
            "vm": vm, "process_io": io}

def gpu_stats():                                   # full nvidia-smi telemetry for the browser monitor
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        p = [x.strip() for x in o.split(",")]
        return {"util": int(float(p[0])), "mem_used": int(float(p[1])), "mem_total": int(float(p[2])),
                "temp": int(float(p[3])), "power": round(float(p[4])), "power_max": round(float(p[5]))}
    except Exception:
        return {}

def free_lmstudio():
    try: subprocess.run([LMS_CLI, "unload", "--all"], timeout=40, capture_output=True)
    except Exception: pass
def free_122b():
    if getattr(M, "_layer_cache", None): M._layer_cache.clear()
    if getattr(M, "_expert_cache", None): M._expert_cache.clear()
    torch.cuda.empty_cache()
def load_122b():
    if not getattr(M, "_layer_cache", None):
        for i in range(M.nL): M._make_layer(i)

print("loading Tarfa (fused-kernel 122B)...", flush=True)
M = ChannelModel("Qwen/Qwen3.5-122B-A10B")
for i in range(M.nL): M._make_layer(i)
M._mat(M.lm.embed_tokens, M.prefix + "embed_tokens")
if not M.tied: M._mat(M.model.lm_head, "lm_head")              # resident lm_head (no per-token reload)
LEAN = os.environ.get("LEAN", "1") != "0"                     # lean (default ON): drop drafter+dreamer -> ~1.6GB freed for context
DRAFT = None
if not LEAN:
    print("loading 0.8B...", flush=True)
    DRAFT = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float16).to("cuda").eval()
LOCK = threading.Lock()
EOS = M.tok.eos_token_id
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
print("READY", flush=True)

@torch.no_grad()
def big_fwd(token_list, cache):
    _f0 = time.perf_counter()
    cur = torch.tensor([token_list], device=M.device); past = cache.get_seq_length(); S = cur.shape[1]
    if S == 1 and PROFILE["mode"] == "bf16" and PROFILE["bf16_fadvise_top"]:
        exact_bf16.prefetch_bf16(getattr(M, "_predict", {}), PROFILE["bf16_fadvise_top"])
    pos = torch.arange(past, past + S, device=M.device)
    if M.tied:
        h = M.lm.embed_tokens(cur)
    else:
        ekey = M.prefix + "embed_tokens.weight"
        with safe_open(M._shard(ekey), framework="pt", device="cpu") as _f:
            _rows = torch.stack([_f.get_slice(ekey)[t] for t in token_list])
        h = _rows.to(M.device, M.dtype).unsqueeze(0)
    if M.mrope:
        p4 = pos.view(1,1,-1).expand(4,1,-1); text_pos, rope_pos = p4[0], p4[1:]
    else: text_pos = rope_pos = pos.unsqueeze(0)
    pe = M.lm.rotary_emb(h, rope_pos)
    causal = create_causal_mask(config=M.cfg, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=text_pos)
    _l0 = time.perf_counter()
    for i in range(M.nL):
        lt = M.layer_types[i] if M.layer_types else "full_attention"
        mask = None if lt == "linear_attention" else causal
        o = M._make_layer(i)(h, position_embeddings=pe, attention_mask=mask, position_ids=text_pos, past_key_values=cache, use_cache=True)
        h = o[0] if isinstance(o, tuple) else o
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _layer_s = time.perf_counter() - _l0
    M._mat(M.lm.norm, M.prefix + "norm"); h = M.lm.norm(h); M._free(M.lm.norm)
    h = h[:, -1:, :]                                            # only the last position's logits are ever used (decode S=1;
    if M.tied:                                                  # prefill discards them) -> avoids a ~Nx152k fp32 OOM on long prompts
        logits = torch.matmul(h.float(), M.lm.embed_tokens.weight.float().t())
    else:
        logits = M.model.lm_head(h).float()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _elapsed = time.perf_counter() - _f0; _kind = "decode" if S == 1 else "prefill"
    PROFILE["forward_calls"] += 1; PROFILE[f"{_kind}_calls"] += 1
    PROFILE[f"{_kind}_tokens"] += S; PROFILE[f"{_kind}_seconds"] += _elapsed
    PROFILE["layer_seconds"] += _layer_s
    return logits[0]

if not LEAN and not os.environ.get("DREAMER_OFF"):
    dreamer.start(M, DRAFT, big_fwd, LOCK, BACKEND, LAST_REQ, EOS)
    print("dreamer started", flush=True)

CANCEL = threading.Event()

@torch.no_grad()
def decode_stream(ids, max_new):
    _r0 = time.perf_counter()
    try:
        pids = ids[0].tolist(); cache = DynamicCache(config=M.cfg)
        if len(pids) > 1: big_fwd(pids[:-1], cache)
        pending = pids[-1]; produced = 0
        while produced < max_new:
            if CANCEL.is_set(): return
            nxt = int(big_fwd([pending], cache)[-1].argmax())
            if nxt == EOS: return
            yield M.tok.decode([nxt]); pending = nxt; produced += 1
    finally:
        PROFILE["last_request_seconds"] = time.perf_counter() - _r0

def draft_gen(prompt, max_new):
    ids = M.tok(prompt, return_tensors="pt").input_ids.to(M.device)
    streamer = TextIteratorStreamer(M.tok, skip_prompt=True, skip_special_tokens=True)
    threading.Thread(target=DRAFT.generate, daemon=True,
                     kwargs=dict(input_ids=ids, max_new_tokens=max_new, do_sample=False, streamer=streamer, pad_token_id=EOS)).start()
    for text in streamer:
        if text: yield text

def build_prompt(msgs, think=False):
    try: return M.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    except TypeError: return M.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home(): return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")).read()

@app.get("/v1/models")
def models():
    data = [{"id": OUR_ID, "object": "model", "owned_by": "tarfa", "label": "Tarfa 122B (fused kernel)"}]
    for qid in QMODES: data.append({"id": qid, "object": "model", "owned_by": "tarfa", "label": QLABEL[qid] + "  ·  slow, high-power"})
    if DRAFT is not None: data.append({"id": DRAFT_ID, "object": "model", "owned_by": "tarfa", "label": "Qwen3.5 0.8B (small, fast)"})
    try:
        for m in requests.get(LMSTUDIO + "/v1/models", timeout=4).json().get("data", []):
            if "embed" not in m.get("id", "").lower(): data.append({"id": m["id"], "object": "model", "owned_by": "lmstudio", "label": m["id"] + " (LM Studio)"})
    except Exception: pass
    return {"object": "list", "data": data}

@app.post("/stop")
def stop(): CANCEL.set(); return {"stopped": True}

@app.get("/status")
def status():
    h, t = getattr(M, "_cache_stat", [0, 0])
    on122 = BACKEND["cur"] == "122b"
    return {"engine": "Tarfa", "backend": BACKEND["cur"],
            "model": OUR_ID if on122 else BACKEND.get("model"),
            "big_model": "Qwen3.5-122B · fused MoE kernel" if on122 else BACKEND.get("model"),
            "accel": ("BF16 exact top-8 streaming" if PROFILE["mode"] == "bf16" else "int4 experts + fused Triton kernel") if on122 else "LM Studio",
            "serving": LOCK.locked(),
            "idle_seconds": round(time.time() - LAST_REQ["ts"], 1) if LAST_REQ["ts"] else None,
            "last_model": LAST_REQ.get("model"),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "cache_hit_rate": round(h / max(t, 1), 3),
            "cached_experts": sum(len(v) for v in getattr(M, "_expert_cache", {}).values()),
            "gpu": gpu_stats(),
            "dreamer": dreamer.snapshot() if (not LEAN and not os.environ.get("DREAMER_OFF")) else None}

@app.get("/dreamer")
def dreamer_atlas(): return dreamer.full_atlas()

import channel_q4
@app.get("/iostat")
def iostat():
    s = dict(channel_q4._STAT)
    s["read_GBps"] = round(s["read_b"]/1e9/s["read_s"], 2) if s["read_s"] else 0
    channel_q4._STAT.update(read_s=0.0, read_b=0, calls=0, experts=0)
    return s

@app.get("/profile")
def profile(reset: bool = False):
    cache = getattr(M, "_bf16_cache", {})
    out = {"runtime": {**PROFILE, "bf16_cached_experts": sum(len(v) for v in cache.values())},
           "moe": fused_int4.telemetry(reset=reset),
           "routing": fused_int4.routing_telemetry(reset=reset),
           "ram_cache": exact_bf16.ram_cache_stats(),
           "system": system_stats(),
           "bf16_io": exact_bf16.stats(reset=reset) if PROFILE["mode"] == "bf16" else None}
    if reset:
        mode = PROFILE["mode"]
        batch = PROFILE["bf16_prefill_batch"]
        cache_n = PROFILE["bf16_cache_per_layer"]
        pinned = PROFILE["bf16_pinned"]
        ram_gb = PROFILE["bf16_ram_cache_gb"]
        batched_decode = PROFILE["bf16_batched_decode"]
        fadvise_top = PROFILE["bf16_fadvise_top"]
        for key in PROFILE:
            PROFILE[key] = mode if key == "mode" else (batch if key == "bf16_prefill_batch" else (cache_n if key == "bf16_cache_per_layer" else (pinned if key == "bf16_pinned" else (ram_gb if key == "bf16_ram_cache_gb" else (batched_decode if key == "bf16_batched_decode" else (fadvise_top if key == "bf16_fadvise_top" else (0 if key.endswith(("calls", "tokens")) else 0.0)))))))
    return out

@app.post("/v1/chat/completions")
async def chat(req: Request):
    CANCEL.clear(); b = await req.json()
    model = b.get("model", OUR_ID)
    LAST_REQ["ts"] = time.time(); LAST_REQ["model"] = model
    stream = bool(b.get("stream"))
    if model == DRAFT_ID and DRAFT is not None:                 # === the 0.8B directly ===
        prompt = build_prompt(b.get("messages", [])); mx = min(int(b.get("max_tokens") or 512), 32768)
        def mkd(d, f): return "data: " + json.dumps({"id": "d", "object": "chat.completion.chunk", "model": DRAFT_ID, "choices": [{"index": 0, "delta": d, "finish_reason": f}]}) + "\n\n"
        if stream:
            def gd():
                with LOCK:
                    first = True
                    for p in draft_gen(prompt, mx):
                        yield mkd({"role": "assistant", "content": p} if first else {"content": p}, None); first = False
                yield mkd({}, "stop"); yield "data: [DONE]\n\n"
            return StreamingResponse(gd(), media_type="text/event-stream")
        def gend():
            with LOCK: return "".join(draft_gen(prompt, mx))
        return {"id": "d", "object": "chat.completion", "model": DRAFT_ID, "choices": [{"index": 0, "message": {"role": "assistant", "content": await run_in_threadpool(gend)}, "finish_reason": "stop"}]}
    if model in QMODES:                                        # === QUALITY MODE: N x compute on the 122B ===
        strat, nn = QMODES[model]; msgs = b.get("messages", []); mx = min(int(b.get("max_tokens") or 1024), 32768)
        def mkq(d, f): return "data: " + json.dumps({"id": "q", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": d, "finish_reason": f}]}) + "\n\n"
        if stream:
            def gq():
                with LOCK:
                    to_122b(); first = True
                    for kind, txt in quality.run(strat, nn, M, big_fwd, EOS, build_prompt, msgs, mx, CANCEL):
                        c = (f"\n> ⚙ {txt}…\n\n" if kind == "status" else txt)
                        yield mkq({"role": "assistant", "content": c} if first else {"content": c}, None); first = False
                yield mkq({}, "stop"); yield "data: [DONE]\n\n"
            return StreamingResponse(gq(), media_type="text/event-stream")
        def genq():
            with LOCK:
                to_122b(); out = []
                for kind, txt in quality.run(strat, nn, M, big_fwd, EOS, build_prompt, msgs, mx, CANCEL):
                    if kind == "token": out.append(txt)
                return "".join(out)
        return {"id": "q", "object": "chat.completion", "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": await run_in_threadpool(genq)}, "finish_reason": "stop"}]}
    if model != OUR_ID:                                        # === gateway -> LM Studio ===
        def to_lms():
            if BACKEND["cur"] != "lms" or BACKEND.get("model") != model:
                with LOCK:
                    free_122b()
                    try: subprocess.run([LMS_CLI, "load", model, "-c", "16384", "-y"], timeout=300, capture_output=True)
                    except Exception: pass
                    BACKEND["cur"] = "lms"; BACKEND["model"] = model
        if stream:
            def pg():
                to_lms()
                with requests.post(LMSTUDIO + "/v1/chat/completions", json=b, stream=True, timeout=1800) as r:
                    for ch in r.iter_content(chunk_size=None):
                        if ch: yield ch
            return StreamingResponse(pg(), media_type="text/event-stream")
        def do(): to_lms(); return requests.post(LMSTUDIO + "/v1/chat/completions", json=b, timeout=1800)
        r = await run_in_threadpool(do); return JSONResponse(r.json(), status_code=r.status_code)
    # === Tarfa fused 122B ===
    msgs = b.get("messages", []); mx = min(int(b.get("max_tokens") or 2048), 32768)
    ids = M.tok(build_prompt(msgs, bool(b.get("thinking", False))), return_tensors="pt").input_ids
    def mkc(d, f): return "data: " + json.dumps({"id": "j", "object": "chat.completion.chunk", "model": OUR_ID, "choices": [{"index": 0, "delta": d, "finish_reason": f}]}) + "\n\n"
    def to_122b():
        if BACKEND["cur"] != "122b": free_lmstudio(); load_122b(); BACKEND["cur"] = "122b"
    if stream:
        def g():
            with LOCK:
                to_122b(); first = True
                for p in decode_stream(ids, mx):
                    yield mkc({"role": "assistant", "content": p} if first else {"content": p}, None); first = False
            yield mkc({}, "stop"); yield "data: [DONE]\n\n"
        return StreamingResponse(g(), media_type="text/event-stream")
    def gen():
        with LOCK: to_122b(); return "".join(decode_stream(ids, mx))
    return {"id": "j", "object": "chat.completion", "model": OUR_ID, "choices": [{"index": 0, "message": {"role": "assistant", "content": await run_in_threadpool(gen)}, "finish_reason": "stop"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089, log_level="warning")
