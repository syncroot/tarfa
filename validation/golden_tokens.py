#!/usr/bin/env python3
"""Golden-token check: Tarfa exact mode vs an external reference (Hugging Face transformers).

Greedy-decodes N fixed prompts with each side and reports the token match rate. At every step it
also records the reference's top-1/top-2 logit margin, so a divergence can be classified:
a near-tie (margin below --tie-eps) is a precision flip at the argmax; a large-margin divergence
is a real bug. "Bit-identical weights + N/N golden match" is the full exactness contract.

  # 1. reference (slow: HF with accelerate disk offload, reads the original safetensors)
  python validation/golden_tokens.py reference --out validation/golden/reference_hf.json
  # 2. Tarfa exact BF16, same prompts, same token budget
  BF16=1 python validation/golden_tokens.py tarfa --out validation/golden/tarfa_exact.json
  # 3. compare -> validation/golden/report.json (commit all three as receipts)
  python validation/golden_tokens.py compare

Status: harness written 2026-10-20, not yet run on the target machine. No match rate is claimed
until report.json is committed.
"""
import argparse, json, os, pathlib, sys, time

REPO = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = REPO / "validation" / "golden"
MODEL = os.environ.get("TARFA_MODEL", "Qwen/Qwen3.5-122B-A10B")
PROMPTS = [
    "The three most important considerations when designing a database index are",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: What is 17 * 23?\nA:",
    "The capital of Saudi Arabia is",
    "عاصمة المملكة العربية السعودية هي",
    "In one sentence, explain why the sky is blue:",
    "Translate to French: 'The weather is nice today.'\n",
    "List the first five prime numbers:",
]


def record(prompt, ids, toks, margins, logits_dtype):
    return {"prompt": prompt, "prompt_ids": ids, "tokens": toks, "top2_margin": margins,
            "logits_dtype": logits_dtype}


def run_reference(n_new):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        offload_folder=os.environ.get("HF_OFFLOAD", str(GOLDEN / "offload")))
    model.eval()
    out = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        cur, past, toks, margins = ids.to(model.device), None, [], []
        with torch.no_grad():
            for _ in range(n_new):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[0, -1].float()
                top = torch.topk(logits, 2)
                nxt = int(top.indices[0])
                toks.append(nxt); margins.append(float(top.values[0] - top.values[1]))
                cur = torch.tensor([[nxt]], device=model.device)
        print(f"[reference] {p[:40]!r} -> {tok.decode(toks)!r}", flush=True)
        out.append(record(p, ids[0].tolist(), toks, margins, str(o.logits.dtype)))
    return out


def run_tarfa(n_new):
    if os.environ.get("BF16") != "1":
        sys.exit("refusing: set BF16=1 so the engine imports in exact mode")
    import torch
    engine = REPO / "tarfa" / "engine"
    sys.path.insert(0, str(engine))
    import fused_int4
    from fused_int4 import TarfaFused
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    assert pathlib.Path(fused_int4.__file__).resolve().is_relative_to(engine.resolve())
    M = TarfaFused(MODEL)
    for i in range(M.nL): M._make_layer(i)
    M._mat(M.lm.embed_tokens, M.prefix + "embed_tokens")
    if not M.tied: M._mat(M.model.lm_head, "lm_head")
    M._mat(M.lm.norm, M.prefix + "norm")

    @torch.no_grad()
    def fwd(tl, cache):  # identical to validation/bench_repo.py: the path the receipts measure
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
        h = M.lm.norm(h)[:, -1:, :]
        lg = M.logits(h)
        return lg[0, -1], str(lg.dtype)

    out = []
    for p in PROMPTS:
        ids = M.tok(p, return_tensors="pt").input_ids[0].tolist()
        cache = DynamicCache(config=M.cfg)
        logits, dt = fwd(ids, cache)
        toks, margins = [], []
        for _ in range(n_new):
            top = torch.topk(logits, 2); nxt = int(top.indices[0])
            toks.append(nxt); margins.append(float(top.values[0] - top.values[1]))
            logits, dt = fwd([nxt], cache)
        print(f"[tarfa] {p[:40]!r} -> {M.tok.decode(toks)!r}", flush=True)
        out.append(record(p, ids, toks, margins, dt))
    return out


def compare(ref_path, tarfa_path, tie_eps):
    ref, tar = json.loads(ref_path.read_text()), json.loads(tarfa_path.read_text())
    rows, matched = [], 0
    for a, b in zip(ref["results"], tar["results"]):
        assert a["prompt_ids"] == b["prompt_ids"], f"tokenization differs for {a['prompt']!r}"
        div = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
        ok = div is None and len(a["tokens"]) == len(b["tokens"])
        matched += ok
        rows.append({"prompt": a["prompt"], "match": ok, "first_divergence": div,
                     "ref_margin_at_divergence": None if div is None else a["top2_margin"][div],
                     "near_tie": None if div is None else a["top2_margin"][div] < tie_eps})
    report = {"model": MODEL, "reference": ref["engine"], "candidate": tar["engine"],
              "tokens_per_prompt": ref["n_new"], "prompts": len(rows), "matched": matched,
              "match_rate": f"{matched}/{len(rows)}", "tie_eps": tie_eps,
              "logits_dtype": {"reference": ref["results"][0]["logits_dtype"],
                               "tarfa": tar["results"][0]["logits_dtype"]},
              "rows": rows}
    (GOLDEN / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if matched == len(rows) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reference", "tarfa", "compare"])
    ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--tie-eps", type=float, default=0.125, help="logit margin treated as a near-tie")
    a = ap.parse_args()
    GOLDEN.mkdir(parents=True, exist_ok=True)
    if a.cmd == "compare":
        sys.exit(compare(GOLDEN / "reference_hf.json", GOLDEN / "tarfa_exact.json", a.tie_eps))
    t0 = time.time()
    res = run_reference(a.n_new) if a.cmd == "reference" else run_tarfa(a.n_new)
    engine = "transformers (HF, bf16, accelerate offload)" if a.cmd == "reference" else "tarfa exact BF16"
    out = a.out or GOLDEN / ("reference_hf.json" if a.cmd == "reference" else "tarfa_exact.json")
    out.write_text(json.dumps({"engine": engine, "model": MODEL, "n_new": a.n_new,
                               "elapsed_s": round(time.time() - t0, 1), "results": res},
                              indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
