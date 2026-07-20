#!/usr/bin/env python3
"""Held-out Phase 11 serial codec versus completion-driven scheduling."""
import glob,json,os,pathlib,statistics,sys,time,torch
R=pathlib.Path(os.path.expanduser("~/tarfa"));sys.path.insert(0,str(R/"runtime"));import tarfa_nibble
root=str(R/"phase11/adaptive_full");routes=json.loads((R/"oracle_research/router_signal_routes.json").read_text());LS=[0,15,30,45]
work=[]
for name in sorted(n for n,x in routes.items() if x["role"]=="test"):
 x=routes[name]
 for t in range(x["tokens"]-6,x["tokens"]):
  for L in LS:work.append((L,x["routes"][str(L)]["ids"][t]))
def evict():
 for p in glob.glob(root+"/*.nib"):
  fd=os.open(p,os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
def arm(completion):
 evict();t=time.perf_counter();nb=0
 for L,E in work:
  _,s=tarfa_nibble.read(root,L,E,"cuda",torch.bfloat16,completion=completion);nb+=s["bytes"]
 return time.perf_counter()-t,nb
# Compile both paths before cold trials.
tarfa_nibble.read(root,0,work[0][1],"cuda",torch.bfloat16,completion=False)
tarfa_nibble.read(root,0,work[0][1],"cuda",torch.bfloat16,completion=True)
res={"serial":[],"completion":[]}
for i in range(5):
 a,nb=arm(False);b,_=arm(True);res["serial"].append(a);res["completion"].append(b)
 print(i,round(a,3),round(b,3),round((1-b/a)*100,2),flush=True)
sa=statistics.median(res["serial"]);sb=statistics.median(res["completion"])
res.update({"batches":len(work),"bytes":nb,"median_serial_s":sa,"median_completion_s":sb,
 "improvement":1-sb/sa,"speedup":sa/sb})
(R/"phase12/completion_results.json").write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
