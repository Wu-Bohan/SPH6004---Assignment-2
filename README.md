# MIMIC-IV Assignment2 Pipeline

This project implements end-to-end multi-modal ICU time-to-discharge modeling on server.

## Structure
- `src/preprocess.py`: data preprocessing and feature extraction
- `src/train.py`: E0-E6 experiments
- `src/evaluate.py`: metrics aggregation + figures
- `src/build_report.py`: experiment log + final report generation
- `jobs/run_pipeline.sh`: one-shot full pipeline
- `jobs/pipeline.pbs`: PBS batch job script

## Server run
```bash
bash jobs/run_pipeline.sh /scratch/users/nus/e1538612/mimic_a2
# or
qsub jobs/pipeline.pbs
```

## Outputs
- metrics: `outputs/metrics/*.json`, `outputs/metrics/summary.csv`
- figures: `outputs/figures/*.png`
- reports: `reports/Assignment2_Experiment_Log.docx`, `reports/Assignment2_Final_Report.docx`, `reports/Assignment2_Final_Report.md`
