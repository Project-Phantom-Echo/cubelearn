# CubeLearn HAR reproduction and author discussion

## Hyperparameters: author configuration versus our experiments

All rows below use full 20-frame samples. Classifier LR is the constant rate or
WSD peak rate; complex LR applies to CubeLearn and is always **0 for DFT**.
Each accuracy is held-out test mean ± sample SD (%), using every seed in the row.
A single-seed result has no SD. Checkpoints use best validation accuracy, then
validation loss. Epochs specify the selection budget, not necessarily the selected epoch.

| Experiment | Batch | Epochs | Classifier LR | Complex LR | Schedule / dropout | Seeds | DFT held-out | CubeLearn held-out |
|---|---:|---:|---:|---:|---|---|---:|---:|
| Historical baseline | 32 | 30 | 0.0003 | 0.0001 | Constant / 0 | 0,1,2 | 71.85 ± 9.63 | 77.31 ± 9.82 |
| Historical baseline | 8 | 30 | 0.0003 | 0.0001 | Constant / 0 | 0 | 74.44 | 86.94 |
| Both LRs increased | 32 | 30 | 0.0015 | 0.0005 | Constant / 0 | 0,1,2 | 79.72 ± 9.14 | 82.69 ± 10.04 |
| Both LRs increased | 32 | 30 | 0.0030 | 0.0010 | Constant / 0 | 0,1,2 | 83.15 ± 3.69 | 83.06 ± 4.59 |
| Classifier-only LR | 32 | 100 | 0.0015 | 0.0001 | Constant / 0 | 0,1,2 | 85.93 ± 6.30 | 85.28 ± 8.01 |
| Classifier-only LR | 32 | 100 | 0.0030 | 0.0001 | Constant / 0 | 0,1,2 | 87.04 ± 1.60 | 89.44 ± 5.14 |
| Classifier-only LR | 32 | 100 | 0.0015 | 0.0001 | WSD / 0 | 0,1,2 | 85.65 ± 2.63 | 85.00 ± 6.50 |
| Classifier-only LR | 32 | 100 | 0.0030 | 0.0001 | WSD / 0 | 0,1,2 | 90.37 ± 3.95 | 89.54 ± 1.05 |
| Fresh 200-epoch run, evaluated through 100 | 32 | 100 | 0.0030 | 0.0001 | Constant / 0 | 0,1,2 | 85.65 ± 6.25 | 91.57 ± 2.92 |
| Longer training | 32 | 200 | 0.0030 | 0.0001 | Constant / 0 | 0,1,2 | 85.93 ± 6.35 | 91.39 ± 3.38 |
| Dropout follow-up | 32 | 100 | 0.0030 | 0.0001 | WSD / 0.2 | 0,1,2 | 88.43 ± 0.16 | 88.98 ± 5.14 |
| Dropout follow-up | 32 | 100 | 0.0030 | 0.0001 | WSD / 0.5 | 0,1,2 | 92.78 ± 1.21 | 91.11 ± 2.89 |
| **Our author-config run** | 8 | 60 | 0.0003 | 0.0010 | Constant / 0 | 0,1,2,3,4 | 86.11 ± 3.74 | 90.22 ± 5.40 |
| **Author's rerun** | **8** | **60** | **0.0003** | **0.0010** | Not specified | 0,1,2,3,4 | **86.78 ± 5.25** | **90.94 ± 5.47** |

**What worked:** the author-config run closely matches both his DFT and CubeLearn
means and their gap, using five seeds. Unlike our earlier classifier-only grid,
it keeps classifier LR at 0.0003 and raises complex LR to 0.001, with batch 8
and 60 epochs. Those factors changed together, so we cannot attribute success
to one factor alone.

**Higher accuracy versus matching the comparison:** our fresh constant-10×
100-epoch checkpoint reached CubeLearn 91.57%, but the earlier nominally equivalent
100-epoch runs reached 89.44%; repeatability remains unresolved. Dropout 0.5
reached CubeLearn 91.11%, but DFT reached 92.78%, reversing the paper ranking.
Extending the fresh constant-rate runs from 100 to 200 epochs did not improve
CubeLearn's mean. Thus the largest CubeLearn number alone is not the criterion
for matching the reported DFT/CubeLearn comparison.

WSD here means classifier warmup for epochs 1–5, constant through epoch 80,
and linear decay to 10% through epoch 100; complex LR stays fixed. Dropout is
applied after the activated 128-dimensional classifier hidden layer.
The author specified neither a scheduler nor dropout in his successful-config
email; our matching run uses neither. Earlier 50-epoch checkpoint comparisons
and baseline 180-epoch trials are retained in their campaign records and the
conversation summary above. These were exploratory tests, with different seed
counts and budgets, not controlled evidence that one hyperparameter alone is better.

Updated 2026-10-08. This consolidates the author correspondence supplied by Filya
and our completed experiments. Author recollections, his new rerun, and our own
measurements are distinguished below. Scope: D-A-T 2D CNN–LSTM HAR only.

## Current result

The author's new configuration closely reproduces his five-seed results locally.
It recovers CubeLearn's held-out advantage with **all five seeds included**.
Accuracy (%); our uncertainty is sample standard deviation across seeds 0–4.
The author's mean ± SD values are transcribed from his email.

| Source | DFT seen | CubeLearn seen | DFT held-out | CubeLearn held-out |
|---|---:|---:|---:|---:|
| Author's rerun | 99.28 ± 0.50 | 99.83 ± 0.15 | 86.78 ± 5.25 | 90.94 ± 5.47 |
| Our author-config run | 99.28 ± 0.42 | 99.89 ± 0.15 | 86.11 ± 3.74 | 90.22 ± 5.40 |
| Paper values supplied by author | 98.8 | 99.5 | 87.1 | 91.3 |

**Our individual seed results (%)**, with all five seeds included in the main
summary. Values are rounded to four decimals from the retained test records.

| Seed | DFT seen | CubeLearn seen | DFT held-out | CubeLearn held-out |
|---|---:|---:|---:|---:|
| 0 | 98.6111 | 100.0000 | 81.6667 | 92.7778 |
| 1 | 99.1667 | 100.0000 | 82.7778 | 85.5556 |
| 2 | 99.7222 | 99.7222 | 89.1667 | 83.8889 |
| 3 | 99.4444 | 100.0000 | 86.9444 | 96.9444 |
| 4 | 99.4444 | 99.7222 | 90.0000 | 91.9444 |

The author did not supply his complete per-seed results. His main mean ± SD
includes all five seeds; we use that comparison, not his supplementary trimming.

Our held-out mean is 0.67 points below his rerun for DFT and 0.72 below for
CubeLearn. CubeLearn's advantage is +4.11 points locally versus +4.16 in his rerun.
This is close agreement with the author's rerun, not proof of identical original
paper protocol: the HAR subject assignment and his checkpoint rule remain unconfirmed.
Our earlier approximate chart readings (98.8/99.4 seen, 86.9/91.1 held-out) are
superseded for comparisons by the author-supplied paper values above. We have not
independently verified those values against the final IEEE version.

## What the author suggested and what we tried

1. **Initial gap and slow convergence.** We described adapting the released
   HGR D-A-T CNN–LSTM to HAR and using the released data and repetition CSVs.
   The initial 30-epoch, batch-32, seed-0 results were 79.72% DFT / 84.44%
   CubeLearn held-out. Classifier LR was 0.0003 and CubeLearn complex-layer LR
   0.0001. The author remembered averaging multiple seeds, but not the original
   held-out subjects. He suggested higher LR, more updates, and checking whether
   we had fewer samples per epoch. We tried batch 8 at 30 epochs with seed 0:
   74.44% DFT / 86.94% CubeLearn held-out, still using complex LR 0.0001.

2. **Tentative sliding-window suggestion, subsequently withdrawn.** The author
   initially recalled two-second samples and suggested 75%, 80% or 90% overlap.
   We tried windows within each released 20-frame sample: 12/3 and 16/4
   frames/stride for 75%, 10/2 and 15/3 for 80%, and 10/1 for 90%.
   We tested individual-window scoring and averaging window probabilities into
   one prediction per original sample. Several settings improved accuracy, but
   did not consistently recover CubeLearn's advantage. He later clarified that
   training, validation and testing should all use full 20-frame samples, with
   sample-level validation selection. He noted the paper says one minute was
   cut into 30 samples and said he probably did not use sliding windows if the
   released data was already cut. It is already cut. Windowing is excluded from
   the reported reproduction; its implementation and launchers were removed
   from this working repository on 2026-10-08. Only this experiment-history
   summary is retained here; archived exploratory evidence is outside the repo.

3. **Initialization and larger learning rates.** He asked us to verify complex
   linear-layer initialization against DFT. Our numerical checks matched the
   three corresponding transforms within floating-point precision. He suggested
   a 10× LR increase and longer training. At batch 32 and 30 epochs, we scaled
   both parameter groups by 5× or 10× (DFT complex layers remained frozen).
   Held-out means over seeds 0–2 were 79.72 ± 9.14 / 82.69 ± 10.04 at 5×,
   and 83.15 ± 3.69 / 83.06 ± 4.59 at 10× (DFT / CubeLearn).
   Baseline-rate full-sample training for 180 epochs had previously reached
   82.13 ± 6.52 / 81.67 ± 10.82.

4. **Classifier-only LR, schedules and dropout.** He could not recover the
   original code and said released parameters appeared to be for HGR rather
   than HAR. He suggested 50 or 100 epochs, higher LR provided training
   converges, lowering LR during training, possibly dropout, and increasing
   only classifier LR while keeping complex LR unchanged. We tested classifier
   LR 0.0015/0.003 with complex LR 0.0001, batch 32, seeds 0–2, constant LR
   or WSD. WSD used five warmup epochs, a steady classifier LR through epoch 80,
   and linear decay to 10% by epoch 100; complex LR stayed fixed. We saved best
   validation checkpoints through 50 and 100 epochs and tested them afterward.
   We then tried constant 10× LR for 200 epochs, and 100-epoch WSD with dropout
   0.2/0.5 after the activated 128-dimensional classifier hidden layer. These
   were exploratory settings, not recovered paper settings. Most 10× runs
   already reached 100% training accuracy. See selected outcomes below.

5. **Author's successful new rerun.** He subsequently supplied batch **8**,
   **60 epochs**, classifier LR **0.0003**, complex LR **0.001** for CubeLearn
   (0 for DFT), and seeds **0,1,2,3,4**. We had not tried this exact combination:
   our batch-8 test was only 30 epochs with complex LR 0.0001; our earlier
   complex-LR-0.001 runs also raised classifier LR to 0.003 and used batch 32.
   This time we kept classifier LR at baseline and raised only the complex-layer
   rate. Batch 8 gives 68 updates per epoch for our 540 training samples,
   compared with 17 at batch 32. All ten author-config jobs completed, producing
   the agreement in the first table. The experiment identifies a successful
   combination, not the isolated causal contribution of any one parameter.

6. **Publication access and randomness.** He recommended the final IEEE version
   over arXiv, but later said he also lacked access and could not supply it.
   He subsequently supplied the paper values in his email. He noted that
   CuDNN and hardware can change outcomes even with identical seeds. We also
   observed differing same-seed results across fresh nominally equivalent runs;
   the cause has not been isolated. It must not be treated as resolved merely
   by attributing it to hardware.

Final five-seed records and frozen sources are also retained in this fork under
[`results/author_config_20261007`](results/author_config_20261007/). Commands and
submission scripts there are historical cluster records; use [setup below](#setup-and-replay)
for a fresh replay. Checkpoints and data remain external.

## Final local protocol and provenance

- Released D-A-T CNN–LSTM adapted from 10 frames/12 HGR classes to 20 frames/6
  HAR classes; two upstream `super()` constructor typos corrected. Local test
  loops added. We used the released implementation, not an original released
  HAR training script. See [implementation issues](bugs.md).
- Full 20-frame, two-second samples; no sliding windows, added dropout or LR
  scheduler in the final author-config run. Adam, cross-entropy, batch 8,
  60 epochs, classifier LR 0.0003, complex LR 0.001 (CubeLearn) or 0 (DFT).
- Best sample-level validation accuracy, breaking ties with validation loss.
  Both test splits evaluated using that checkpoint. His latest email did not
  explicitly confirm this checkpoint rule or whether a scheduler was used.
- Released repetition CSVs; inferred HAR users 0–5 seen and 6–7 held out.
  Counts: 540 train, 180 validation, 360 seen test, 360 held-out test.
- Original data: 1,440 samples, 8 users × 6 actions × 30 repetitions. Cache
  retains eight horizontal channels; model uses first 64 chirps/128 ADC samples.
  Cache: `/mnt/weka/fgeikyan/cubelearn_data/har_cache`.
- Latest frozen campaign: `../cubelearn-author-20261007-a`, Slurm array 316374.
  Manifest, source/split hashes, commands, environments, logs, checkpoint and
  per-seed `results.json` are retained there. Source hashes, completion,
  validation selection and all-seed summary were checked.
- Earlier campaigns: `../cubelearn-lr-grid-20261001-a` (306668),
  `../cubelearn-wsd-20261005-a` (313492), and
  `../cubelearn-followup-20261006-a` (313688). See their manifests and summaries.
- [Historical full-sample records](results/har_reproduction.md) remain unchanged.
  Retired windowing code/evidence and pre-consolidation documentation are archived
  at `../old/cubelearn-windowing-20261008`; other frozen historical campaigns
  are preserved. They are not part of the reported reproduction.
- [Setup and explicit author-config replay](#setup-and-replay). This fork tracks
  `Project-Phantom-Echo/cubelearn`, with `zhaoymn/cubelearn` as upstream.

## Setup and replay

### Environment

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

### Data and optional replay

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
All five seeds are retained in the latest primary comparison. Earlier grids were
exploratory and informed by previously observed test results.

### Slurm launchers

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

### Retained evidence

`verify_har.py` checks the original source hashes, all ten result records,
validation-based checkpoint selection, and confusion-matrix accuracies.
`--smoke` additionally checks split separation, independent FFT equivalence and
a CPU forward pass. Neither command trains a model or rewrites result records.

`results/` includes all ten result records, six diagnostic commands/logs, and
the hashed historical source/split snapshot. Checkpoints and raw data remain
external. The historical diagnostic runner refuses to overwrite its existing
campaign. `--report-only` rewrites only its summary, if explicitly requested.

### Latest author-config replay

Use full 20-frame samples, batch 8, 60 epochs, classifier LR 0.0003,
and complex-layer LR 0 for DFT or 0.001 for CubeLearn. Run seeds 0–4.
There is no added dropout or scheduler. Checkpoints use sample-level validation
accuracy, then validation loss; the latest author email did not specify his
checkpoint rule. Use a fresh output directory for each run:

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python train_har.py --batch-size 8 --epochs 60 --lr 0.0003 --lpp-lr 0 --seed "$seed" --cache-dir "$CUBELEARN_CACHE" --output-dir "results/author-replay-dft-seed$seed"
  .venv/bin/python train_har.py --batch-size 8 --epochs 60 --lr 0.0003 --lpp-lr 0.001 --seed "$seed" --cache-dir "$CUBELEARN_CACHE" --output-dir "results/author-replay-cubelearn-seed$seed"
done
```

The five-seed JSON results, frozen source/splits and run configuration are retained
in `results/author_config_20261007/`. Its driver and submission record preserve
historical absolute paths; use the explicit replay commands above on another system.

The completed cluster campaign is `../cubelearn-author-20261007-a/` (job 316374).
Do not resubmit it: its lock protects completed results. The default training
arguments still describe the historical baseline; the explicit arguments above
select the author configuration. See [discussion above](#what-the-author-suggested-and-what-we-tried) for the conversation,
experiments, all-seed results and limitations. Sliding-window launchers and
implementation have been retired from this repository.
