#!/bin/bash
#SBATCH --job-name=cl-har-cache
#SBATCH --output=logs/har_cache_%j.out
#SBATCH --error=logs/har_cache_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=research
#SBATCH --mem=128G

set -e

RUN_DIR="${CUBELEARN_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
PYTHON="${CUBELEARN_PYTHON:-$RUN_DIR/.venv/bin/python}"
ARCHIVE="${CUBELEARN_ARCHIVE:-$RUN_DIR/data/HAR_data.zip}"
CACHE="${CUBELEARN_CACHE:-$RUN_DIR/data/har_cache}"

cd "$RUN_DIR"
mkdir -p logs

# 44.8 GB zip -> 30.2 GB int16 cache (2,20,128,8,256) per sample.
# Drops antennas 8-11 (unused by every released model) and narrows the integer
# ADC counts from float64 to int16, which is lossless and asserted per file.
"$PYTHON" -u prepare_har_cache.py \
  --zip "$ARCHIVE" \
  --out "$CACHE" \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
