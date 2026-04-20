#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/users/nus/e1538612/mimic_a2}"
MET_DIR="$ROOT_DIR/outputs/metrics"
FIG_DIR="$ROOT_DIR/outputs/figures"
MODEL_DIR="$ROOT_DIR/outputs/models"
REP_DIR="$ROOT_DIR/reports"
PROC_DIR="$ROOT_DIR/data/processed"

module load python/3.10.9
source "$ROOT_DIR/.venv/bin/activate"
cd "$ROOT_DIR"

run_exp () {
  local exp="$1"
  local json="$MET_DIR/${exp}.json"
  if [ -f "$json" ]; then
    echo "[$(date)] ${exp} already exists, skip"
    return
  fi
  echo "[$(date)] start ${exp}"
  python -m src.train \
    --exp "$exp" \
    --config "$ROOT_DIR/configs/${exp}.yaml" \
    --data_dir "$PROC_DIR" \
    --out_metrics_dir "$MET_DIR" \
    --out_models_dir "$MODEL_DIR" \
    --seed 42 \
    --n_bootstrap 1000
  echo "[$(date)] done ${exp}"
}

run_exp E3
run_exp E4
run_exp E5
run_exp E6

for i in $(seq 1 720); do
  if [ -f "$MET_DIR/E2.json" ]; then
    echo "[$(date)] E2.json detected"
    break
  fi
  echo "[$(date)] waiting E2.json ..."
  sleep 60
done

python -m src.evaluate \
  --metrics_dir "$MET_DIR" \
  --figures_dir "$FIG_DIR" \
  --data_dir "$PROC_DIR" \
  --models_dir "$MODEL_DIR" \
  --out_csv "$MET_DIR/summary.csv"

python -m src.build_report \
  --data_dir "$PROC_DIR" \
  --metrics_dir "$MET_DIR" \
  --figures_dir "$FIG_DIR" \
  --summary_csv "$MET_DIR/summary.csv" \
  --out_experiment_docx "$REP_DIR/Assignment2_Experiment_Log.docx" \
  --out_final_docx "$REP_DIR/Assignment2_Final_Report.docx" \
  --out_final_md "$REP_DIR/Assignment2_Final_Report.md"

echo "[$(date)] E3-E6+report pipeline completed"
