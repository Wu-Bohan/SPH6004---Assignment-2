#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-/scratch/users/nus/e1538612/mimic_a2}"
cd "$ROOT_DIR"
qsub jobs/pipeline.pbs
