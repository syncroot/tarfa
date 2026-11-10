#!/bin/bash
# Plain glm5-next decode, the configuration measured in README.md (story prompt @128 on the
# reference machine). Needs both GPUs free. Settings: glm5_env.sh.
# Usage: GLM5_GGUF=<dir with the GGUF shards> GLM5_PROMPT="..." ./run_best.sh [n_tokens]
cd "$(dirname "$0")"
: "${GLM5_PROMPT:?set GLM5_PROMPT}"
. ./glm5_env.sh
export GLM5_WARM=${GLM5_WARM:-prefill}
PRIOR=${GLM5_PRIOR:-$GLM5_OUT/route_prior.npy}
if [ -f "$PRIOR" ]; then export GLM5_PRIOR=$PRIOR; else unset GLM5_PRIOR; echo "no $PRIOR: run run_calib.sh first for the best hit rate" >&2; fi
exec "$PYTHON" -u glm5_gdecode.py "$GLM5_GGUF" "$GLM5_OUT" "${1:-48}"
