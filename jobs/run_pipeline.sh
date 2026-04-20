#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/users/nus/e1538612/mimic_a2}"
RAW_DIR="$ROOT_DIR/data/raw"
PROC_DIR="$ROOT_DIR/data/processed"
MET_DIR="$ROOT_DIR/outputs/metrics"
FIG_DIR="$ROOT_DIR/outputs/figures"
MODEL_DIR="$ROOT_DIR/outputs/models"
REP_DIR="$ROOT_DIR/reports"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$RAW_DIR" "$PROC_DIR" "$MET_DIR" "$FIG_DIR" "$MODEL_DIR" "$REP_DIR" "$LOG_DIR"

module load python/3.10.9

if [ ! -d "$ROOT_DIR/.venv" ]; then
  python -m venv "$ROOT_DIR/.venv"
fi
source "$ROOT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [ -f "$RAW_DIR/Assignment2_mimic dataset.zip" ]; then
  unzip -o "$RAW_DIR/Assignment2_mimic dataset.zip" -d "$RAW_DIR" >/dev/null
fi

cd "$ROOT_DIR"

python -m src.preprocess \
  --raw_dir "$RAW_DIR" \
  --out_dir "$PROC_DIR" \
  --horizon_hours 24 \
  --seed 42

for EXP in E0 E1 E2 E3 E4 E5; do
  echo "Running ${EXP}..."
  python -m src.train \
    --exp "$EXP" \
    --config "$ROOT_DIR/configs/${EXP}.yaml" \
    --data_dir "$PROC_DIR" \
    --out_metrics_dir "$MET_DIR" \
    --out_models_dir "$MODEL_DIR" \
    --seed 42 \
    --n_bootstrap 1000
  echo "${EXP} done"
done

python -m src.train \
  --exp E6 \
  --config "$ROOT_DIR/configs/E6.yaml" \
  --data_dir "$PROC_DIR" \
  --out_metrics_dir "$MET_DIR" \
  --out_models_dir "$MODEL_DIR" \
  --seed 42 \
  --n_bootstrap 1000

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

echo "Pipeline completed."
