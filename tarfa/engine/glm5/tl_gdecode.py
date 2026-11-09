"""Host-side timeline of graph decode: where each MoE layer's critical path goes."""
import os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, glm5_gdecode as GD, experts_gpu as XG
acc = collections.defaultdict(float)
pc = time.perf_counter
orig_moe = XG.RawExperts.moe
def moe(self, L, x32, sel, w32):
    t0 = pc(); hits, misses, ts, q = self.lookup(L, sel); acc["lookup"] += pc() - t0
    y = torch.zeros(x32.shape[0], device=self.dev, dtype=torch.float32)
    t0 = pc()
    xw = torch.cat([x32, w32]).cpu().numpy(); acc["d2h_x"] += pc() - t0
    x_np, w_np = xw[:x32.shape[0]], xw[x32.shape[0]:]
    ix = [sel.index(e) for e in misses]
    t0 = pc()
    fut = self.cpu_exec.submit(self._cpu_timed, [ts[0].data[e] for e in misses], [ts[1].data[e] for e in misses],
                               [ts[2].data[e] for e in misses], *q, x_np, w_np[ix]) if misses else None
    t1 = pc(); acc["submit"] += t1 - t0
    self._gpu([(sel.index(e), hits[e]) for e in hits], x32, w32, q, y)
    t2 = pc(); acc["gpu_launch"] += t2 - t1
    if fut is not None:
        r = fut.result(); t3 = pc(); acc["cpu_wait"] += t3 - t2
        y += torch.from_numpy(r).to(self.dev, non_blocking=True); acc["h2d"] += pc() - t3
    return y
XG.RawExperts.moe = moe
orig_step = GD.Engine.step
def step(self, emb):
    self.emb.copy_(emb); self.pos_t.fill_(self.pos)
    self.graphs[3].replay()
    for L in range(3, GD.N_LAYER):
        t0 = pc(); sel = self.sel.tolist(); acc["graph_wait"] += pc() - t0
        y = self.raw.moe(L, self.x32, sel, self.w32)
        t0 = pc(); self.yr.copy_(y); self.graphs[L + 1].replay(); acc["replay"] += pc() - t0
    self.pos += 1
    acc["steps"] += 1
    return self.logits
GD.Engine.step = step
try:
    GD.main()
finally:
    n = acc.pop("steps")
    print(f"=== host timeline, ms per token (avg over {n:.0f} steps incl prefill) ===")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {1000*v/n:7.2f}")
