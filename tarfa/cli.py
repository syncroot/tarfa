"""tarfa CLI — exact-first control layer.

  tarfa serve        exact BF16 serving (default; bit-identical weights)
  tarfa start-fast   labeled fast mode (fused int4; output may differ from exact greedy)
  tarfa chat         terminal chat against a running server
  tarfa run          one-shot timed generation (engine driver)
  tarfa convert      build verified JNF v1 from the HF checkpoint
  tarfa version
"""
import os, sys, subprocess, argparse

PKG = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(PKG, "engine")
FMT = os.path.join(os.path.dirname(PKG), "format")

def _run(script, extra=None, env=None, cwd=ENGINE):
    e = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **(env or {}))
    e["PYTHONPATH"] = ENGINE + os.pathsep + e.get("PYTHONPATH", "")
    return subprocess.call([sys.executable, script] + (extra or []), env=e, cwd=cwd)

def main():
    ap = argparse.ArgumentParser(prog="tarfa", description="Exact-first MoE runtime")
    ap.add_argument("cmd", choices=["serve", "start-fast", "chat", "run", "convert", "version"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "version":
        print("tarfa 0.1.0 — exact BF16 by default; fast mode is labeled non-exact"); return 0
    if a.cmd == "serve":
        print("[tarfa] EXACT mode: bit-identical BF16 weights, native top-8 routing.")
        return _run(os.path.join(ENGINE, "serve.py"), env={"BF16": "1", "JTOPK": "8"})
    if a.cmd == "start-fast":
        print("[tarfa] FAST mode: fused int4 — output may differ from exact greedy decoding.")
        print("[tarfa] Never use fast mode for equivalence tests, reproducible research, or exact benchmarks.")
        return _run(os.path.join(ENGINE, "serve.py"), env={"BF16": "0", "JTOPK": "8"})
    if a.cmd == "chat":
        return _run(os.path.join(ENGINE, "chat.py"), a.rest)
    if a.cmd == "run":
        return _run(os.path.join(ENGINE, "run.py"), a.rest)
    if a.cmd == "convert":
        return _run(os.path.join(FMT, "convert_jnf.py"), a.rest, cwd=FMT)

if __name__ == "__main__":
    sys.exit(main())
