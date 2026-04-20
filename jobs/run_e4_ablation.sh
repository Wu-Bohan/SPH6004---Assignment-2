#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/scratch/users/nus/e1538612/mimic_a2
EXP_NAME=${EXP_NAME:?EXP_NAME is required}
MODE=${MODE:?MODE is required}
BOOT=${BOOT:-1000}

cd "$ROOT_DIR"
module load python/3.10.9
source .venv/bin/activate

python -m src.e4_ablation \
  --exp_name "$EXP_NAME" \
  --mode "$MODE" \
  --config "$ROOT_DIR/configs/E4.yaml" \
  --data_dir "$ROOT_DIR/data/processed" \
  --out_metrics_dir "$ROOT_DIR/outputs/metrics" \
  --out_models_dir "$ROOT_DIR/outputs/models" \
  --seed 42 \
  --n_bootstrap "$BOOT"
