#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source /public/home/chenrs/anaconda3/etc/profile.d/conda.sh
conda activate easynco

export CUDA_VISIBLE_DEVICES=0,1

python routing_encoder_first/src/run_main.py \
  --config routing_encoder_first/configs/cvrp200_eval.yaml \
  --eval-only
