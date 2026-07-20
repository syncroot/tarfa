#!/usr/bin/env python3
"""Convert and benchmark the four-layer JNF v1 acceptance sample."""
import argparse, concurrent.futures, hashlib, json, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tarfa" / "engine"))
import exact_bf16

LAYERS = (0, 15, 31, 47); OUT = ROOT / "jnf-v1-sample"; BLOCK = 4096

def projection_info(layer, name):
    shard, offset, shape, dtype = exact_bf16.meta()[layer][name]
    size = shape[1] * shape[2] * 2
    assert dtype == "BF16" and size % BLOCK == 0
    return shard, offset, shape, size

def read_exact(fd, offset, size):
    data=bytearray(size); done=0
    while done<size:
        n=os.preadv(fd,[memoryview(data)[done:]],offset+done)
        if n<=0: raise IOError("short read")
        done+=n
    return data

def convert():
    OUT.mkdir(exist_ok=True); manifest={"format":"JNF","version":1,"dtype":"BF16","alignment":BLOCK,"layers":{}}
    started=time.perf_counter()
    for layer in LAYERS:
        layer_dir=OUT / "layers" / f"{layer:02d}"; layer_dir.mkdir(parents=True,exist_ok=True)
        record={}
        for name in ("gate_up_proj","down_proj"):
            shard,offset,shape,size=projection_info(layer,name); fd=os.open(shard,os.O_RDONLY)
            final=layer_dir / ("gate_up.bf16" if name=="gate_up_proj" else "down.bf16"); partial=final.with_suffix(".partial")
            hashes=[]; whole=hashlib.sha256()
            with open(partial,"wb",buffering=0) as output:
                for expert in range(shape[0]):
                    data=read_exact(fd,offset+expert*size,size); output.write(data); whole.update(data); hashes.append(hashlib.sha256(data).hexdigest())
            os.close(fd); os.replace(partial,final)
            record[name]={"file":str(final.relative_to(OUT)),"shape":shape,"expert_bytes":size,
                          "file_sha256":whole.hexdigest(),"expert_sha256":hashes}
        manifest["layers"][str(layer)]=record
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"status":"PASS","layers":LAYERS,"bytes":sum(p.stat().st_size for p in OUT.rglob("*.bf16")),
                      "seconds":round(time.perf_counter()-started,3)},indent=2))

def read_hash(fd,offset,size): return hashlib.sha256(read_exact(fd,offset,size)).hexdigest()

def bench_mode(layer, selected, source, cold, repeats):
    reqs=[]
    for name in ("gate_up_proj","down_proj"):
        shard,offset,shape,size=projection_info(layer,name)
        if source=="jnf":
            shard=str(OUT / "layers" / f"{layer:02d}" / ("gate_up.bf16" if name=="gate_up_proj" else "down.bf16")); offset=0
        fd=os.open(shard,os.O_RDONLY)
        if cold and hasattr(os,"posix_fadvise"): os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        reqs += [(fd,offset+e*size,size) for e in selected]
    times=[]; valid=True; manifest=json.loads((OUT/"manifest.json").read_text())
    expected=[]
    for name in ("gate_up_proj","down_proj"): expected += [manifest["layers"][str(layer)][name]["expert_sha256"][e] for e in selected]
    for _ in range(repeats):
        start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool: got=list(pool.map(lambda r:read_hash(*r),reqs))
        times.append(time.perf_counter()-start); valid &= got==expected
    for fd in {r[0] for r in reqs}: os.close(fd)
    total=sum(r[2] for r in reqs)
    return {"layer":layer,"source":source,"cold":cold,"valid":valid,"mean_ms":sum(times)/len(times)*1000,
            "GBps":total*repeats/sum(times)/1e9}

def bench(repeats):
    random.seed(122); results=[]
    for layer in LAYERS:
        selected=random.sample(range(256),8)
        for cold in (True,False):
            for source in ("hf","jnf"): results.append(bench_mode(layer,selected,source,cold,repeats))
    print(json.dumps({"status":"PASS" if all(r["valid"] for r in results) else "FAIL","results":results},indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=("convert","bench"));ap.add_argument("--repeats",type=int,default=2)
    ap.add_argument("--full",action="store_true",help="convert all 48 layers into jnf-v1")
    args=ap.parse_args()
    if args.full:
        LAYERS=tuple(range(48)); OUT=ROOT / "jnf-v1"
    convert() if args.action=="convert" else bench(args.repeats)
