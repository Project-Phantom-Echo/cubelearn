#!/bin/bash
#SBATCH --job-name=cl-har-dat
#SBATCH --output=logs/har_dat_%j.out
#SBATCH --error=logs/har_dat_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G

set -e

RUN_DIR="${CUBELEARN_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
PYTHON="${CUBELEARN_PYTHON:-$RUN_DIR/.venv/bin/python}"
CACHE="${CUBELEARN_CACHE:-$RUN_DIR/data/har_cache}"
RESULTS="${CUBELEARN_RESULTS:-$RUN_DIR/results}"
MODEL="dat_2dcnn_lstm"

# Paper says 30 epochs, but at 30 the DFT arm was still climbing steeply
# (val 0.64 -> 0.72 -> 0.83 over epochs 20-30, best at 27/30) and landed at
# 78.3 in-set vs the paper's 98.8. HAR has only 540 training samples = 17 steps
# per epoch, so 30 epochs is ~510 updates. Override to train longer:
#   sbatch --export=ALL,EPOCHS=200 submit_har_dat.sh
# Longer runs are extended-budget experiments, not the stated paper protocol.
EPOCHS="${EPOCHS:-30}"

cd "$RUN_DIR"
mkdir -p logs

nvidia-smi --query-gpu=name,memory.total --format=csv

# Both arms of the paper's central claim. The only difference is --lpp-lr:
#   0     transform layers frozen at DFT init -> "DFT"       target 98.8 / 86.9
#   1e-4  transform layers trained            -> "CubeLearn" target 99.4 / 91.1
# (targets are in-set / out-of-set, read off Figure 10)
for ARM in dft cubelearn; do
  if [ "$ARM" = "dft" ]; then LPP_LR=0; else LPP_LR=1e-4; fi
  OUTPUT="$RESULTS/$MODEL/$ARM/ep${EPOCHS}_job${SLURM_JOB_ID}"
  if [ -e "$OUTPUT" ]; then
    echo "ERROR: refusing to overwrite existing run: $OUTPUT" >&2
    exit 1
  fi
  mkdir -p "$OUTPUT"
  echo "=============== $ARM (lpp_lr=$LPP_LR, epochs=$EPOCHS) ==============="
  "$PYTHON" -u train_har.py \
    --model "$MODEL" \
    --lpp-lr "$LPP_LR" \
    --lr 3e-4 \
    --epochs "$EPOCHS" \
    --batch-size 32 \
    --seed 0 \
    --workers 8 \
    --cache-dir "$CACHE" \
    --split-dir dataset_split \
    --output-dir "$OUTPUT"
done
