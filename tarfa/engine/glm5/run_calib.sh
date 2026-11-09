#!/bin/bash
# Routing calibration (glm5_calib.py): runs generic text, not the evaluation prompt, through the
# engine with the same settings as the launchers and writes $GLM5_OUT/route_prior.npy (or GLM5_PRIOR).
# Usage: GLM5_GGUF=<dir with the GGUF shards> ./run_calib.sh [text_file]
cd "$(dirname "$0")"
. ./glm5_env.sh
OUT=${GLM5_PRIOR:-$GLM5_OUT/route_prior.npy}
unset GLM5_PRIOR
exec "$PYTHON" -u glm5_calib.py "$GLM5_GGUF" "$OUT" "$@"
