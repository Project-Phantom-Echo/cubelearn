# CubeLearn HAR reproduction

Scope: D-A-T 2D CNN-LSTM, comparing fixed DFT preprocessing
with learned CubeLearn preprocessing. The other nine released configurations have
not been reproduced here.

## Status and results

Partial reproduction. Our original 30-epoch run falls well below the paper's
accuracy. At 200 epochs, seen-user accuracy matches the approximate paper targets,
but held-out accuracy remains about 3 percentage points lower. The longer run is
an extended-budget experiment, not a reproduction of the stated training protocol.

| Setting | Model | Seen-user test (%) | Held-out test (%) | Selected epoch |
|---|---|---:|---:|---:|
| Paper, approximate | DFT | 98.8 | 86.9 | — |
| Paper, approximate | CubeLearn | 99.4 | 91.1 | — |
| 30 epochs, seed 0 | DFT | 78.33 | 79.72 | 27 |
| 30 epochs, seed 0 | CubeLearn | 81.11 | 84.44 | 26 |
| 200 epochs, seed 0 | DFT | 98.89 | 83.33 | 119 |
| 200 epochs, seed 0 | CubeLearn | 99.44 | 88.06 | 200 |

Targets were estimated from the HAR chart in the [preprint](https://arxiv.org/abs/2111.03976)
(Figure 10); the corresponding chart is Figure 11 in the
[2023 accepted paper](https://www.pure.ed.ac.uk/ws/portalfiles/portal/330691128/CubeLearn_ZHAO_DOA_17012023_AFV_CC_BY.pdf).
Treat these as approximate (about ±0.5 points), not exact published decimals.
The selected CNN-LSTM is not the best held-out model in that chart: D-A-T 3D CNN
is higher. Across the three batch-32 seeds, CubeLearn improved held-out accuracy
each time, while its seen-user benefit was inconsistent.

[All run results and diagnostic status](results/har_reproduction.md) update
automatically after each new run. Historical jobs: `217297` (30 epochs) and
`217299` (200 epochs). Diagnostic job `252689` completed all six runs.

### Diagnostic results

All runs below used 30 epochs. Batch-32 rows show ranges across seeds 0, 1 and 2;
batch-8 rows are seed 0 only.

| Batch | Model | Seen-user test (%) | Held-out test (%) |
|---:|---|---:|---:|
| 32 | DFT | 78.33–89.17 | 61.11–79.72 |
| 32 | CubeLearn | 81.11–90.83 | 66.11–84.44 |
| 8 | DFT | 91.11 | 74.44 |
| 8 | CubeLearn | 92.50 | 86.94 |

Neither setting reached the paper targets. Batch 8 improved seen-user accuracy
for both arms compared with seed 0/batch 32; held-out accuracy improved for
CubeLearn but worsened for DFT. Smaller batches therefore do not resolve the gap.
The large variation across seeds shows that random initialization/sample ordering
matters even with the same held-out users. The cause of the remaining gap is unresolved;
the author's HAR configuration or checkpoint remains the most useful next evidence.

## Data and split

- Archive: `/mnt/weka/fgeikyan/cubelearn_data/HAR_data.zip` (44.83 GB).
- Training cache: `/mnt/weka/fgeikyan/cubelearn_data/har_cache/` (30.20 GB).
  Extracted int16 ADC, channels 0–7; only this cache is used for training.
- 1,440 two-second samples: 8 users × 6 actions × 30 repetitions.
  Filename: `{user_id}_{action_id}_{repetition_id}.npy`.
- Archive shape: `float64 (2,20,128,12,256)` = real/imaginary × frames × chirps ×
  virtual channels × ADC samples. Cache shape: `int16 (2,20,128,8,256)`.
- IWR6843, 3 Tx × 4 Rx: eight horizontal channels and four elevation channels.
  This reproduction uses horizontal channels only, then the first 64 chirps and
  128 ADC samples, as specified for D-A-T in the paper.

| Split | Users | Repetitions per action/person | Samples |
|---|---|---:|---:|
| Train | 0–5 | 15 | 540 |
| Validation | 0–5 | 5 | 180 |
| Seen-user test | 0–5 | 10 | 360 |
| Held-out-user test | 6–7 | 30 | 360 |

Repetition IDs follow the [released CSVs](dataset_split/). User IDs follow the
[gesture loader](https://github.com/zhaoymn/cubelearn/blob/bd787c8/dataset.py);
the paper specifies six seen/two held-out users but does not identify them.
This assignment remains inferred for HAR. Action names in our loader follow the
paper's listed order; the release labels are numeric.

## Implementation and settings

[Upstream code](https://github.com/zhaoymn/cubelearn/tree/bd787c8), local base
`bd787c8`. [network_har.py](network_har.py) adapts the released D-A-T model from
10 frames/12 classes to 20 frames/6 classes. [train_har.py](train_har.py) adds
both test loops. Two incorrect `super()` calls in upstream `network.py` are
fixed locally; see [implementation issues](bugs.md). These changes and the historical diagnostic source snapshot are preserved in this fork.

D-A-T applies range, Doppler and angle transforms, sums magnitude over range,
and feeds each Doppler-angle frame through a CNN followed by an LSTM over time.
DFT weights remain fixed with `--lpp-lr 0`; `--lpp-lr 1e-4` trains them.
Both arms use identical classifier initialization and sample ordering for a given seed.

Adam, classifier learning rate `3e-4`, batch 32, cross-entropy, 30 epochs.
Select checkpoints by validation accuracy, then lower validation loss; test only
the selected checkpoint. No input normalization or augmentation is added.
Batch 32 and classifier learning rate follow the reference code. The paper's
`5e-5`–`1e-4` transform learning-rate result concerns a D-T gesture experiment;
it does not confirm the authors' exact D-A-T HAR learning rate.

Environment: historical runs used the sibling COMPASS environment, PyTorch
`2.1.1+cu121`, `cplxmodule` `2022.6`, and an H100 GPU. See
[SETUP-HAR.md](SETUP-HAR.md) for an independent environment, configurable paths,
and optional replay commands. The retained results do not require retraining.

## Investigation and remaining uncertainty

The code/data audit found no training defect: all 1,440 cache headers and split
memberships passed; 48 archive/cache comparisons covering every user/action pair
matched exactly. Frozen preprocessing matched an independent FFT calculation;
the wrapper matched upstream at the upstream frame/class sizes. Saved weights,
checkpoint selection and confusion matrices were consistent for all four historical
runs. Full test inference was not repeated during that audit.

The accepted paper confirms the cropping, split counts, and 30-epoch budget.
The [public issues](https://github.com/zhaoymn/cubelearn/issues) provide no author
clarification of the accuracy gap as of this review. There is no dedicated released
HAR training recipe or checkpoint with which to establish exact equivalence.

The bounded diagnostic plan is fixed in [run_har_diagnostics.py](run_har_diagnostics.py):
both arms at 30 epochs for `(seed, batch)` = `(1,32)`, `(2,32)`, `(0,8)`.
The extra seeds test variability; batch 8 tests sensitivity to more optimizer updates
(2,040 versus 510), while also changing batch statistics. It is not a confirmed
paper setting. Each run records its command and uses a saved source/split snapshot.

The remaining gap is unexplained. Seed sensitivity is observed; unknown author
settings and a different subject pair remain unconfirmed explanations. Do not select subjects,
seeds, or training budgets by agreement with test targets. Validation selection alone
does not eliminate overfitting, and a better score at a changed setting does not
resolve missing protocol information.
