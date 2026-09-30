# CubeLearn HAR reproduction

Consolidated checkpoint: 2026-09-30, before implementing windowing experiments.
This is the authoritative summary of completed runs, author correspondence,
research findings, and next steps. Correspondence is paraphrased from replies
provided by Filya; recollections are distinguished from recovered records.

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
the author's original HAR configuration or checkpoint has not been recovered.

Since the author confirmed averaging multiple runs, the existing batch-32,
30-epoch results across seeds 0, 1, and 2 are also summarized below. Values are
mean ± sample standard deviation, in percentage points; these are not confidence
intervals. The author's seed count and split policy remain unknown.

| Model | Seen-user test (%) | Held-out test (%) |
|---|---:|---:|
| DFT | 83.80 ± 5.42 | 71.85 ± 9.63 |
| CubeLearn | 85.19 ± 5.05 | 77.31 ± 9.82 |

The original seed-0 runs ended at training accuracies of 81.67% (DFT) and
77.96% (CubeLearn). In the separate 200-epoch experiments, validation accuracy
first reached 95% at epochs 55 and 53, respectively. These observations support
incomplete optimization at 30 epochs under our recipe, but do not establish its
cause. The long-run histories are not exact continuations of the short runs.

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

The earlier code/data audit established no defect in the checked paths: all 1,440 cache headers and split
memberships passed; 48 archive/cache comparisons covering every user/action pair
matched exactly. Frozen preprocessing matched an independent FFT calculation;
the wrapper matched upstream at the upstream frame/class sizes. Saved weights,
checkpoint selection and confusion matrices were consistent for all four historical
runs. Full test inference was not repeated during that audit.

The accepted paper specifies the cropping, original clip split counts, and
30-epoch budget. Clip counts do not establish the number of windows processed
per epoch. There is no recovered dedicated HAR training recipe or author
checkpoint with which to establish exact equivalence.

The bounded diagnostic plan is fixed in [run_har_diagnostics.py](run_har_diagnostics.py):
both arms at 30 epochs for `(seed, batch)` = `(1,32)`, `(2,32)`, `(0,8)`.
The extra seeds test variability; batch 8 tests sensitivity to more optimizer updates
(2,040 versus 510), while also changing batch statistics. It is not a confirmed
paper setting. Each run records its command and uses a saved source/split snapshot.

## Author discussion

The initial email to Chris Xiaoxuan Lu described the local HAR adaptation,
30-epoch accuracy gap, input cropping, inferred subject IDs, and hyperparameters.
A follow-up to the author team added the longer-budget and seed/batch diagnostics,
requesting the original HAR recipe, checkpoint, exact split, and averaging policy.

In the response recorded on 2026-09-28, Peijun Zhao:

- Confirmed that the reported results averaged multiple random-seed runs.
- Did not remember the exact held-out user/activity selection.
- Offered to look for original records, while doubting he could recover them.
  The records' loss has not been confirmed.
- Saw no major hyperparameter discrepancy in the description and suggested
  checking learning rate and the number of optimizer updates per epoch.
- Recalled sliding-window sampling when building the training set.

We clarified that the released script reads individual two-second/20-frame files
and asked whether these were already windowed. In a subsequent reply provided
before this checkpoint, he thought the release was probably not windowed and
suggested trying 75%, 80%, or 90% overlap. He explicitly did not remember the
precise setting. These percentages are proposed experiments, not recovered
author settings. Window length, stride, sample count, split-before-windowing
policy, seed count, and whether splits varied across runs remain unconfirmed.

Further implementation decisions will therefore be explicit reconstruction
assumptions rather than claims of an exact author recipe.

## Sliding-window research

Section 6.2 of the [preprint](https://arxiv.org/pdf/2111.03976), also reflected
in the accepted paper's collection description, says each participant performed
each activity for one minute, subsequently cut into 30 samples. The data are
radar recordings, not video inputs. The downloaded archive contains 1,440 NumPy
arrays and no separate continuous recording or timing metadata.

The current loader and retained diagnostic snapshot each enumerate one item per
file; neither adds temporal windows. A 20-frame window within a 20-frame file
produces one sample regardless of overlap. Batch 32 gives 17 updates per epoch
(510 over 30 epochs); batch 8 gives 68 (2,040 over 30 epochs). This also changes
batch statistics, so the batch-8 result does not isolate update count.

Read-only signal checks covered all 48 user/activity groups and all 28,800 frames.
For each frame, the check sampled antenna 0, chirp 0, and the first 128 ADC
samples, retaining both real and imaginary values. No sampled frame fingerprints
repeated. This supports the release lacking exact shared frames between clips;
it does not exclude transformed duplicates or establish recording order.

For temporal continuity, the check compared RMS distances between `log1p(abs(FFT))`
range profiles of those same samples. Median distances were 0.4746 within clips,
0.4695 across numerically adjacent clip boundaries, and 0.4847 across other
distinct-clip boundaries within the same user/activity. Adjacent boundaries had
lower median distance in 42/48 groups. However, the next numeric clip was the
closest candidate only 6.97% of the time (chance 3.45%); its median rank was 12
among 29 candidates. This is weak supporting evidence, not proof of chronological,
gap-free order. No signal-based reordering was performed.

The released split interleaves clip IDs. For example, the 15 training IDs form
nine contiguous numeric blocks. Sliding over an entire reconstructed minute
without respecting those boundaries would mix training and evaluation frames.
If chronological order were assumed, allowing 20-frame windows only within
consecutive training-only blocks would give the following counts (no padding):

| Overlap | Stride (frames) | Training windows/epoch | Updates/epoch, batch 32 |
|---|---:|---:|---:|
| Existing baseline | 20 | 540 | 17 |
| 75% | 5 | 1,188 | 38 |
| 80% | 4 | 1,404 | 44 |
| 90% | 2 | 2,484 | 78 |

These are hypothetical counts, not implemented or run. Do not concatenate
nonadjacent training clips, or cross user/activity/split boundaries.

## Next experiment: windows within each released clip

Planned, not implemented or trained at this checkpoint. Start with two settings
that do not depend on reconstructing chronology:

| Window length | Stride | Overlap | Windows/clip | Training windows/epoch | Updates/epoch, batch 32 |
|---|---:|---:|---:|---:|---:|
| 16 frames / 1.6 s | 4 | 75% | 2 | 1,080 | 34 |
| 10 frames / 1 s | 2 | 80% | 6 | 3,240 | 102 |

- Use both fixed DFT and CubeLearn, seeds 0, 1, and 2, 30 epochs, batch 32,
  and the existing learning rates. Keep preprocessing and subject/repetition
  assignments fixed. Shorter temporal input is an explicit protocol deviation.
- Retain every window from a clip in that clip's original split. Do not use
  test accuracy to select settings. Keep one validation prediction per original
  clip by averaging its window class probabilities; select by clip accuracy,
  then clip negative log likelihood. Use the same aggregation for final testing
  only after fixing the experiment choice.
- Include 20-frame controls with equal optimizer-update budgets: 60 epochs
  (1,020 updates) for the 16-frame experiment and 180 epochs (3,060 updates)
  for the 10-frame experiment. This comparison still differs in temporal context
  and frames processed; it does not fully isolate a causal mechanism. Existing
  seed-0, 200-epoch train/validation histories can be inspected first, but are
  not complete multi-seed controls or saved checkpoints for every epoch.
- Preserve historical source, launchers, environments, results, and checkpoints.
  Implement through a separate entry point with frozen source/split manifests
  and new output directories. Validate window boundaries, counts, aggregation,
  and baseline equivalence before providing launch commands.
- Defer 10-frame windows with stride 1 (90% overlap; 5,940 training windows)
  and speculative cross-file windows until the initial bounded experiment is
  understood. Do not add a simultaneous learning-rate search.

Status remains partial reproduction. Seed sensitivity is observed; the exact
sampling recipe and subject pair remain unresolved. Better scores under a new
setting would not establish faithful reproduction of the published protocol.
