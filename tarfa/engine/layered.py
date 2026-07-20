"""Layered inference on transformers 5.x. Materializes one module at a time.
Handles standard archs (Qwen2.5/Llama) AND Qwen3.5 (multimodal hybrid
linear-attention + MoE): language_model nesting, fresh-init layers for the
linear-attention computed buffers, mrope position_ids, and the linear-attn mask."""
import os, json, torch, time
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

class LayeredModel:
    def __init__(self, repo_id, dtype=torch.float16, device="cuda"):
        self.device, self.dtype = device, dtype
        self.path = snapshot_download(repo_id, allow_patterns=["*.json","*.safetensors","*.model","tokenizer*","*.txt","merges*","vocab*"])
        self.config = AutoConfig.from_pretrained(self.path)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, dtype=dtype)
        self.model.eval()
        self.lm = self.model.model            # text model (Qwen2/Qwen3_5TextModel)
        self.cfg = self.lm.config
        self.nL = self.cfg.num_hidden_layers
        self.LayerCls = type(self.lm.layers[0])
        self.lm.rotary_emb = type(self.lm.rotary_emb)(config=self.cfg).to(device)
        idxf = os.path.join(self.path, "model.safetensors.index.json")
        self.wmap = json.load(open(idxf))["weight_map"] if os.path.exists(idxf) else None
        self.single = os.path.join(self.path, "model.safetensors")
        if self.wmap:
            lk = next((k for k in self.wmap if ".layers.0." in k and not k.startswith("mtp")), None)
            self.prefix = lk.split("layers.0.")[0] if lk else "model."   # e.g. "model.language_model."
        else:
            self.prefix = "model."
        self.layer_types = getattr(self.cfg, "layer_types", None)
        self.tied = getattr(self.model.config, "tie_word_embeddings", False)
        self.mrope = "qwen3_5" in getattr(self.config, "model_type", "")
        self.lm_head_prefix = "lm_head" if (self.wmap and any(k.startswith("lm_head") for k in self.wmap)) else None

    def _gather(self, pre):
        out = {}
        if self.wmap is None:
            with safe_open(self.single, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(pre): out[k] = f.get_tensor(k)
            return out
        byfile = {}
        for k, fn in self.wmap.items():
            if k.startswith(pre): byfile.setdefault(fn, []).append(k)
        for fn, keys in byfile.items():
            with safe_open(os.path.join(self.path, fn), framework="pt", device="cpu") as f:
                for k in keys: out[k] = f.get_tensor(k)
        return out

    def _mat(self, module, prefix):           # for embed / norm (no computed buffers)
        pre = prefix if prefix.endswith(".") else prefix + "."
        sd = self._gather(pre)
        module.load_state_dict({k[len(pre):]: v.to(self.device, self.dtype) for k, v in sd.items()}, strict=False, assign=True)

    def _free(self, module): module.to_empty(device="meta")

    def _make_layer(self, i):                 # fresh init => linear-attn buffers computed
        with torch.device("cpu"):             # force real allocation (init_empty_weights leaves default=meta)
            layer = type(self.lm.layers[i])(self.cfg, i)
        layer = layer.to(self.dtype)
        pre = f"{self.prefix}layers.{i}."
        sd = self._gather(pre)
        layer.load_state_dict({k[len(pre):]: v.to(self.dtype) for k, v in sd.items()}, strict=False)
        return layer.to(self.device).eval()

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=12, log=True):
        ids = ids.to(self.device); gen = ids; past = 0
        cache = DynamicCache(config=self.cfg); tt = []
        for step in range(max_new_tokens):
            t0 = time.time()
            cur = gen if step == 0 else gen[:, -1:]; S = cur.shape[1]
            pos = torch.arange(past, past + S, device=self.device)
            self._mat(self.lm.embed_tokens, self.prefix + "embed_tokens")
            h = self.lm.embed_tokens(cur)
            if not self.tied: self._free(self.lm.embed_tokens)
            if self.mrope:
                p4 = pos.view(1, 1, -1).expand(4, ids.shape[0], -1)
                text_pos, rope_pos = p4[0], p4[1:]
            else:
                text_pos = rope_pos = pos.unsqueeze(0)
            pe = self.lm.rotary_emb(h, rope_pos)
            causal = create_causal_mask(config=self.cfg, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=text_pos)
            for i in range(self.nL):
                lt = self.layer_types[i] if self.layer_types else "full_attention"
                mask = None if lt == "linear_attention" else causal
                layer = self._make_layer(i)
                out = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=text_pos, past_key_values=cache, use_cache=True)
                h = out[0] if isinstance(out, tuple) else out
                del layer; torch.cuda.empty_cache()
            self._mat(self.lm.norm, self.prefix + "norm"); h = self.lm.norm(h); self._free(self.lm.norm)
            if self.tied:
                W = self.lm.embed_tokens.weight
                logits = torch.matmul(h[:, -1:, :].float(), W.float().t()).to(self.dtype); self._free(self.lm.embed_tokens)
            else:
                self._mat(self.model.lm_head, "lm_head"); logits = self.model.lm_head(h[:, -1:, :]); self._free(self.model.lm_head)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1); past += S
            dt = time.time() - t0; tt.append(dt)
            if log: print(f"  tok {step+1}: {dt:.1f}s {self.tok.decode(nxt[0])!r}", flush=True)
        return gen, tt
