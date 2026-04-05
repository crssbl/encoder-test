#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-routing_encoder_first/configs/cvrp100_fixedval_pilot_nohup.yaml}"
CUDA_SET="${2:-0,1}"
EXP_NAME="$(python - <<PY
import yaml
with open("$CONFIG_PATH", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["experiment_name"])
PY
)"

OUTPUT_DIR="routing_encoder_first/outputs/${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"
LOG_PATH="${OUTPUT_DIR}/nohup.log"

setsid nohup bash -lc "
source /public/home/chenrs/anaconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=${CUDA_SET}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -u routing_encoder_first/src/run_main.py --config ${CONFIG_PATH}
" >"$LOG_PATH" 2>&1 < /dev/null &

PID=$!
echo "PID=${PID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_SET}"
echo "LOG=${ROOT_DIR}/${LOG_PATH}"
