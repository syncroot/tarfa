import os, sys, glob, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader, GGMLQuantizationType as QT
import glm5_gdecode as GD
from experts_gpu import RawExperts, KINDS
d = sys.argv[1]; idx = {}
for s in sorted(glob.glob(d + "/*.gguf")):
    for t in GGUFReader(s).tensors: idx[t.name] = t
DEV = "cuda:1"; torch.cuda.set_device(DEV)
os.environ["GLM5_RAW2_DEV"] = "cuda:0"; os.environ["GLM5_RAW2_PER_LAYER"] = "20"
raw = RawExperts(idx, DEV)
class M: pass
eng = GD.Engine.__new__(GD.Engine)
eng.m = M(); eng.m.idx = idx; eng.raw = raw
Ne = 4096
z = lambda *s, dt=torch.float32: torch.zeros(*s, device=DEV, dtype=dt)
eng.x32 = z(Ne); eng.w32 = z(8); eng.sel = torch.zeros(8, device=DEV, dtype=torch.long); eng.yr = z(Ne)
eng.y0c = z(Ne); eng.ycpu = z(Ne)
from collections import OrderedDict
for L in range(3, 45):
    raw.cache[L] = OrderedDict()
for L in (3, 20, 44):
    ts = [idx[f"blk.{L}.{k}"] for k in KINDS]
    for i, e in enumerate((5, 17, 33, 100, 140, 200)):
        dv = DEV if i < 3 else "cuda:0"
        raw.cache[L][e] = (tuple(raw._load(ts[k], e, dv) for k in range(3)), dv)
os.environ["GLM5_CHECK"] = "1"
GD.N_LAYER = 45
eng.freeze_hits()
