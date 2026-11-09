# Shared settings for run_best.sh, run_spec.sh and run_calib.sh (sourced, not run).
# Every value is a default: set any of these in the environment to override it.
# Inputs: GLM5_GGUF (directory with the GGUF shards, required), GLM5_OUT (output dir, default ./out),
# GLM5_PRIOR (routing prior, default $GLM5_OUT/route_prior.npy, made by run_calib.sh),
# LLAMA_TOKENIZE (llama.cpp's llama-tokenize, default: on PATH), PYTHON (default: python).
: "${GLM5_GGUF:?set GLM5_GGUF to the GLM-5.3-Flash-UD-Q4_K_XL directory}"
GLM5_OUT=${GLM5_OUT:-$PWD/out}; mkdir -p "$GLM5_OUT"
# Device layout of the reference machine: trunk + 19 expert slots/layer on the 24 GB card (cuda:1),
# 23 slots/layer on the 16 GB card (cuda:0). Other GPUs need other values.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
  GLM5_Q8=${GLM5_Q8:-1} GLM5_ROUTER_FP32=${GLM5_ROUTER_FP32:-1} GLM5_COMPILE=${GLM5_COMPILE:-1} \
  GLM5_NUMA=${GLM5_NUMA:-1} GLM5_DEV=${GLM5_DEV:-cuda:1} GLM5_RAW_PER_LAYER=${GLM5_RAW_PER_LAYER:-19} \
  GLM5_RAW2_DEV=${GLM5_RAW2_DEV:-cuda:0} GLM5_RAW2_PER_LAYER=${GLM5_RAW2_PER_LAYER:-23} \
  GLM5_CPU_MISSES=${GLM5_CPU_MISSES:-1} GLM5_FETCH_THREADS=${GLM5_FETCH_THREADS:-4}
PYTHON=${PYTHON:-python}
