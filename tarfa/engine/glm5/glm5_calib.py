"""
Routing calibration: run generic text (NOT the evaluation prompt) through the engine and save
the per-layer expert-selection histogram [45, 288]. glm5_gdecode loads it (GLM5_PRIOR) to fill
GPU expert slots that the request's own prefill leaves empty: a static, profiled hot set.
Usage: python glm5_calib.py <gguf_dir> <out.npy> [text_file]   (or run_calib.sh, same settings as the launchers)
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GLM5_WARM", "0")                 # CPU experts only; no cache traffic
import glm5_gdecode as GD

TEXT = """The following is a mixed sample of everyday text used to profile which experts a mixture-of-experts language model uses.

def merge_sorted(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + a[i:] + b[j:]

User: Can you help me plan a three-day trip to Kyoto in autumn? I like temples, food markets and quiet walks.
Assistant: Of course. On the first day, start early at Fushimi Inari before the crowds, then walk down to the Tofuku-ji gardens, which are famous for their maple leaves. In the afternoon, explore Nishiki Market for pickles, grilled skewers and matcha sweets.

The derivative of x squared times sine of x is 2x sin x plus x squared cos x, by the product rule. If a train leaves at 3:15 and travels 180 kilometres at 72 kilometres per hour, it arrives at 5:45.

Photosynthesis converts light energy into chemical energy: carbon dioxide and water become glucose and oxygen inside the chloroplasts of plant cells. Climate scientists measure atmospheric carbon dioxide in parts per million.

SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2025-01-01' GROUP BY customer_id ORDER BY total DESC LIMIT 10;

Dear team, the quarterly report is attached. Revenue grew eight percent, mostly from the new subscription tier, while support costs stayed flat. Please send comments by Friday.
"""


def main():
    gguf_dir, out = sys.argv[1], sys.argv[2]
    text = open(sys.argv[3]).read() if len(sys.argv) > 3 else TEXT
    ids = GD.tokenize(gguf_dir, text)
    GD.check_len(len(ids), 0, extra=0)               # one position per calibration token
    print(f"calibration text: {len(ids)} tokens", flush=True)
    eng = GD.Engine(gguf_dir)
    eng.capture()
    t0 = time.time()
    for i, t in enumerate(ids):
        eng.step(eng.embed(t))
        if i % 50 == 0:
            print(f"  {i}/{len(ids)} tokens  {time.time()-t0:.0f}s", flush=True)
    hist = np.stack([eng.raw.route.get(L, np.zeros(288, np.int64)) for L in range(GD.N_LAYER)])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, hist)
    top = np.sort(hist[3:], axis=1)[:, ::-1]
    for k in (16, 28, 42, 64):
        print(f"  top-{k}/layer covers {top[:, :k].sum() / top.sum():.1%} of calibration routing")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
