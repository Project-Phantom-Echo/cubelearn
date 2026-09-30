# HAR partial reproduction

This fork preserves the local D-A-T CNN-LSTM experiment. It did **not** reproduce
the paper's HAR accuracy under the stated 30-epoch budget. See [notes.md](notes.md)
and the [retained results](results/har_reproduction.md). Training below is optional;
the existing results can be inspected without rerunning it.

## Environment

Use Python 3.11. The historical runs used an H100, PyTorch 2.1.1+cu121 and
cplxmodule 2022.6. The NumPy pin matches the available validated environment;
this small dependency file is not a complete historical environment freeze.
Create an independent environment; do not modify the shared COMPASS environment.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-har.txt
.venv/bin/python report_har.py results/
.venv/bin/python verify_har.py --smoke
```

## Data and optional replay

Obtain `HAR_data.zip` from the dataset link in the upstream [README](README.md).
The required cache has 1,440 arrays shaped `(2,20,128,8,256)`, stored as int16.
It retains the eight horizontal antennas. Cache creation checks that conversion
from float64 ADC counts to int16 is exact; it does not preserve the four unused
elevation antennas. Allow about 45 GB for the archive and 31 GB for the cache.

```bash
export CUBELEARN_ARCHIVE=/path/to/HAR_data.zip
export CUBELEARN_CACHE=/path/to/har_cache
.venv/bin/python prepare_har_cache.py --zip "$CUBELEARN_ARCHIVE" --out "$CUBELEARN_CACHE" --workers 8

# Original 30-epoch, seed-0 comparison; use new output directories.
.venv/bin/python train_har.py --lpp-lr 0 --cache-dir "$CUBELEARN_CACHE" --output-dir results/new-dft
.venv/bin/python train_har.py --lpp-lr 1e-4 --cache-dir "$CUBELEARN_CACHE" --output-dir results/new-cubelearn
```

The defaults remain batch 32, classifier LR 3e-4, 30 epochs and seed 0.
The extended 200-epoch runs are **not** reproductions of the paper's training budget.
The train/validation/seen-test repetition IDs are released CSVs; the assignment
of users 0–5 to seen and 6–7 to held-out remains inferred from the gesture loader.
No subject, seed or budget was selected to match test targets.

## Slurm launchers

Submit from this repository root, creating `logs/` before submission so Slurm
can open its output files:

```bash
mkdir -p logs
export CUBELEARN_REPO="$PWD"
export CUBELEARN_PYTHON="$PWD/.venv/bin/python"
export CUBELEARN_RESULTS=/path/to/new-results
sbatch submit_har_cache.sh
sbatch submit_har_dat.sh
```

`CUBELEARN_REPO`, `CUBELEARN_PYTHON`, `CUBELEARN_ARCHIVE`, `CUBELEARN_CACHE`
and `CUBELEARN_RESULTS` are configurable. Defaults are the submission directory,
its `.venv/bin/python`, `data/HAR_data.zip`, `data/har_cache`, and `results/`.
GPU type, partition, memory and time directives describe the original cluster;
override them for your scheduler. `EPOCHS=200` requests the extended-budget run.

## Retained evidence

`verify_har.py` checks the original source hashes, all ten result records,
validation-based checkpoint selection, and confusion-matrix accuracies.
`--smoke` additionally checks split separation, independent FFT equivalence and
a CPU forward pass. Neither command trains a model or rewrites result records.

`results/` includes all ten result records, six diagnostic commands/logs, and
the hashed historical source/split snapshot. Checkpoints and raw data remain
external. The historical diagnostic runner refuses to overwrite its existing
campaign. `--report-only` rewrites only its summary, if explicitly requested.

The implementation and setup fixes do not explain the accuracy gap. The recorded
seed and batch experiments show sensitivity; exact author HAR hyperparameters
and participant identities remain unverified.
Author correspondence and the proposed separate windowing experiment are
consolidated in [notes.md](notes.md). The commands above replay the historical
20-frame recipe; they do not implement the proposed windowing experiments.
