#!/bin/bash
set -euo pipefail
umask 000
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/rf-perception-papers/compass/.venv/bin/python "$1/driver.py" --campaign "$1" --task "$SLURM_ARRAY_TASK_ID"
