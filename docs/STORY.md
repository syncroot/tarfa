# The Tarfa story

*(This is the narrative behind the project — the launch post draws from here. Every date comes
from file timestamps and commit history in this repo's lineage. Nothing is rounded up.)*

## Twenty-three days

**June 27, 2026.** A 234 GB checkpoint — Qwen3.5-122B-A10B, a 122-billion-parameter
Mixture-of-Experts — starts downloading onto a desktop with one RTX 4060 Ti. Sixteen gigabytes
of VRAM. The kind of GPU you buy for 1080p gaming.

The obvious question was never "can it run fast." It was: **what does it actually take to run a
model fifteen times bigger than your GPU — without changing the model?**

- **Day 1** (Jun 27): first layer-streaming experiments; the experts quantized to int4 the same
  night, because that's the obvious first move — make it smaller.
- **Day 3** (Jun 29): a fused Triton kernel and an active-expert streaming engine are serving the
  122B at usable speed. The obvious move worked. But it changed the model.
- **Days 4–8** (Jun 30–Jul 4): the running engine gets used as a scientific instrument — probing
  the model's 12,288 experts, mapping which fire for what, training small models to predict the
  big one. Doing research on the model made the problem with int4 concrete: *if you're studying a
  model, a quantized copy is not the model.*
- **Day 15** (Jul 11): the exact track starts. New rule, written down before any code: **a
  performance change is accepted only if the output is provably the checkpoint's own output** —
  bit-identical BF16 weights, native top-8 routing, validation script or it doesn't merge.
- **Days 15–24** (Jul 11–20): eleven optimization phases. Five accepted. Four **rejected with
  their numbers published** — including a cache that "obviously" should have helped (3.3% hit
  rate) and a RAM arena that saved 24.7 GB of disk traffic per request and still lost 9.9%
  end-to-end. Speculative decoding got demoted to a labeled fast mode when its greedy output
  diverged. The rejections stayed in the repo on purpose.
- **Day 24** (Jul 20): a learned expert-heat ledger and a zero-copy VRAM hot tier land — the
  last accepted phase. The engine is benchmarked from a clean clone and packaged.

Result: **a 122B MoE, running with bit-identical BF16 weights, on a $400 GPU — 0.35 tokens/sec,
every byte verified, every optimization decision auditable.** And a labeled fast mode at 1.7
tok/s for when you want speed instead of proof.

## Why "exact" is the story

Everyone who runs big models on small hardware makes the same silent trade: quantize, truncate
the router, speculate — the model that answers is a close cousin of the model you meant to run.
For chat, nobody cares. For research, auditing, evals, and interpretability, the cousin is a
methodological bug: you're publishing findings about a model nobody is actually studying.

Tarfa's position: **the trade is fine — hiding it is not.** Exact is the default. Fast is a
label, not a footnote. And the engineering log keeps its failures, because performance work that
only publishes wins is advertising.

## The name

Tarfa — after the star Tarf: a small point of light, an enormous thing behind it.

## Why this is released

I'm not a company, and this isn't a product looking for users. There's no support contract, no
roadmap promises, no Discord to answer at 3am. I built this to see if it could be done, and it
could.

I'm releasing it for the people who understand what they're reading — the ones who can look at a
phase log, a rejected-optimization table, and an expert-streaming transport and see what's worth
keeping. If there's anything here to learn, take it and go further than I did. That's the whole
license, morally speaking; MIT, legally speaking.

And there's a quieter hope underneath. Most of the industry answers "the model doesn't fit" with
"buy more hardware" — and the price of that answer is set accordingly. Every proof that a $400
GPU can do honest work on a 122-billion-parameter model is a small argument that the ceiling is
engineering, not silicon. If enough of those arguments land, maybe the hardware market calms down
a little — and gamers can just buy RAM again.

## One-liners (pick per audience)

- **The claim:** Run a 122B model on a 16 GB GPU — *without changing a single weight.*
- **The hook:** My GPU has 16 GB. The model has 122 billion parameters. Nothing was quantized.
- **The stance:** Slow, exact, and honest about which one you're getting.
- **The methodology:** An inference engine where failed optimizations are part of the docs.
- **The build story:** From checkpoint download to a validated exact-BF16 runtime in 23 days,
  on hardware most people already own.
