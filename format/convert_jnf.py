#!/usr/bin/env python3
"""tarfa convert — build a verified JNF v1 tree from the HF BF16 checkpoint.

Per format/JNF_V1.md: per-layer gate_up.bf16 / down.bf16 with every expert projection
4096-aligned and independently addressable; SHA-256 per expert projection and per file;
atomic per-layer conversion (.partial then rename); mandatory manifest fields; and a
--verify mode that re-reads every emitted region against the source tensors.

Usage:
  python convert_jnf.py [--out DIR] [--layers 0-47] [--verify]
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tarfa" / "engine"))
import exact_bf16

BLOCK = 4096

def parse_layers(spec, n=48):
    if "-" in spec:
        a, b = spec.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]

def read_exact(fd, offset, size):
    data = bytearray(size); done = 0
    while done < size:
        n = os.preadv(fd, [memoryview(data)[done:]], offset + done)
        if n <= 0: raise IOError("short read")
        done += n
    return data

def projection_info(layer, name):
    shard, offset, shape, dtype = exact_bf16.meta()[layer][name]
    size = shape[1] * shape[2] * 2
    if dtype != "BF16": raise SystemExit(f"layer {layer} {name}: dtype {dtype} != BF16")
    if size % BLOCK: raise SystemExit(f"layer {layer} {name}: expert bytes {size} not {BLOCK}-aligned")
    return shard, offset, shape, size

def model_meta():
    m = {"model_id": "Qwen/Qwen3.5-122B-A10B", "revision": None, "tokenizer_revision": None}
    try:
        import glob
        snaps = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/*"))
        if snaps:
            m["revision"] = m["tokenizer_revision"] = os.path.basename(snaps[0])
    except Exception:
        pass
    return m

def convert(layers, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "JNF", "version": 1, "dtype": "BF16", "alignment": BLOCK,
                "router_top_k": 8, **model_meta(), "layers": {}}
    t0 = time.perf_counter(); total = 0
    for layer in layers:
        ld = out / "layers" / f"{layer:02d}"; ld.mkdir(parents=True, exist_ok=True)
        record = {}
        for name in ("gate_up_proj", "down_proj"):
            shard, offset, shape, size = projection_info(layer, name)
            final = ld / ("gate_up.bf16" if name == "gate_up_proj" else "down.bf16")
            partial = final.with_suffix(".partial")
            fd = os.open(shard, os.O_RDONLY)
            hashes = []; whole = hashlib.sha256()
            with open(partial, "wb", buffering=0) as f:
                for e in range(shape[0]):
                    d = read_exact(fd, offset + e * size, size)
                    f.write(d); whole.update(d); hashes.append(hashlib.sha256(d).hexdigest())
                    total += size
            os.close(fd); os.replace(partial, final)
            record[name] = {"file": str(final.relative_to(out)), "shape": list(shape),
                            "expert_bytes": size, "file_sha256": whole.hexdigest(),
                            "expert_sha256": hashes}
        idx = {"layer": layer,
               "gate_up": {"expert_bytes": record["gate_up_proj"]["expert_bytes"],
                            "offsets": [e * record["gate_up_proj"]["expert_bytes"] for e in range(shape[0])]},
               "down": {"expert_bytes": record["down_proj"]["expert_bytes"],
                        "offsets": [e * record["down_proj"]["expert_bytes"] for e in range(shape[0])]}}
        (ld / "index.json").write_text(json.dumps(idx))
        manifest["layers"][str(layer)] = record
        el = time.perf_counter() - t0
        print(f"[convert] layer {layer:02d} done  {total/1e9:.1f} GB  {el:.0f}s", flush=True)
    tmp = out / "manifest.json.partial"
    tmp.write_text(json.dumps(manifest, indent=1)); os.replace(tmp, out / "manifest.json")
    print(f"[convert] COMPLETE: {len(layers)} layers, {total/1e9:.1f} GB, {time.perf_counter()-t0:.0f}s", flush=True)

def verify(layers, out):
    man = json.loads((out / "manifest.json").read_text())
    bad = 0
    for layer in layers:
        rec = man["layers"][str(layer)]
        for name in ("gate_up_proj", "down_proj"):
            r = rec[name]
            shard, offset, shape, size = projection_info(layer, name)
            fd_src = os.open(shard, os.O_RDONLY)
            with open(out / r["file"], "rb") as f:
                for e in range(shape[0]):
                    emitted = f.read(size)
                    src = read_exact(fd_src, offset + e * size, size)
                    if hashlib.sha256(emitted).hexdigest() != r["expert_sha256"][e] or emitted != bytes(src):
                        print(f"  MISMATCH layer {layer} {name} expert {e}"); bad += 1
            os.close(fd_src)
        print(f"[verify] layer {layer:02d} ok", flush=True)
    print("VERIFY_FAILED" if bad else "VERIFY_OK", flush=True)
    return 1 if bad else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "jnf-v1"))
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    if a.verify:
        sys.exit(verify(layers, Path(a.out)))
    convert(layers, Path(a.out))
