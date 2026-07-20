import os, types, torch
import torch.nn.functional as F
from safetensors import safe_open
from layered import LayeredModel

class ChannelModel(LayeredModel):
    def _shard(self, key):
        return self.single if self.wmap is None else os.path.join(self.path, self.wmap[key])

    def _gather_skip(self, pre, skip):
        out, byfile = {}, {}
        for k, fn in self.wmap.items():
            if k.startswith(pre) and k not in skip: byfile.setdefault(fn, []).append(k)
        for fn, keys in byfile.items():
            with safe_open(os.path.join(self.path, fn), framework="pt", device="cpu") as f:
                for k in keys: out[k] = f.get_tensor(k)
        return out

    def _make_layer(self, i):
        cache = self.__dict__.setdefault("_layer_cache", {})
        if i in cache: return cache[i]
        with torch.device("cpu"):
            layer = type(self.lm.layers[i])(self.cfg, i)
        pre = f"{self.prefix}layers.{i}."
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        skip = set()
        if experts is not None:
            skip = {pre + "mlp.experts.gate_up_proj", pre + "mlp.experts.down_proj"}
            for pn in ("gate_up_proj", "down_proj"):
                experts._parameters.pop(pn, None)
        layer = layer.to(self.dtype)
        sd = self._gather_skip(pre, skip)
        layer.load_state_dict({k[len(pre):]: v.to(self.dtype) for k, v in sd.items()}, strict=False)
        layer = layer.to(self.device).eval()
        if experts is not None:
            self._install_streamed_experts(layer.mlp.experts, pre)
        cache[i] = layer
        return layer

    def _install_streamed_experts(self, experts, pre):
        eng = self
        gu_key, dn_key = pre + "mlp.experts.gate_up_proj", pre + "mlp.experts.down_proj"
        gu_shard, dn_shard = eng._shard(gu_key), eng._shard(dn_key)
        def forward(self, hidden_states, top_k_index, top_k_weights):
            n = self.num_experts
            mask = F.one_hot(top_k_index, num_classes=n).permute(2, 1, 0)
            hit = [int(e) for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist() if int(e) != n]
            out = torch.zeros_like(hidden_states)
            if not hit: return out
            with safe_open(gu_shard, framework="pt", device="cpu") as fgu, safe_open(dn_shard, framework="pt", device="cpu") as fdn:
                gws = torch.stack([fgu.get_slice(gu_key)[e] for e in hit]).to(eng.device, eng.dtype)
                dws = torch.stack([fdn.get_slice(dn_key)[e] for e in hit]).to(eng.device, eng.dtype)
            for j, ei in enumerate(hit):
                pos, tok = torch.where(mask[ei])
                cur = hidden_states[tok]
                g, u = F.linear(cur, gws[j]).chunk(2, dim=-1)
                ch = F.linear(self.act_fn(g) * u, dws[j])
                ch = ch * top_k_weights[tok, pos, None]
                out.index_add_(0, tok, ch.to(out.dtype))
            return out
        experts.forward = types.MethodType(forward, experts)
