#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/scratch/users/nus/e1538612/mimic_a2
EXP=${EXP:?EXP is required}
BOOT=${BOOT:-1000}

cd "$ROOT_DIR"
module load python/3.10.9
source .venv/bin/activate

python -m src.train \
  --exp "$EXP" \
  --config "$ROOT_DIR/configs/${EXP}.yaml" \
  --data_dir "$ROOT_DIR/data/processed" \
  --out_metrics_dir "$ROOT_DIR/outputs/metrics" \
  --out_models_dir "$ROOT_DIR/outputs/models" \
  --seed 42 \
  --n_bootstrap "$BOOT"
