# HAR reproduction results

Accuracies are percentages. All runs use D-A-T 2D CNN-LSTM and the same user/repetition split.

| Run | Model | Seed | Batch | Epochs | Best epoch | Validation | Seen test | Held-out test |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [diag20260909_seed0_bs8](dat_2dcnn_lstm/cubelearn/diag20260909_seed0_bs8/results.json) | CubeLearn | 0 | 8 | 30 | 27 | 95.56 | 92.50 | 86.94 |
| [diag20260909_seed1_bs32](dat_2dcnn_lstm/cubelearn/diag20260909_seed1_bs32/results.json) | CubeLearn | 1 | 32 | 30 | 30 | 85.56 | 83.61 | 66.11 |
| [diag20260909_seed2_bs32](dat_2dcnn_lstm/cubelearn/diag20260909_seed2_bs32/results.json) | CubeLearn | 2 | 32 | 30 | 30 | 91.67 | 90.83 | 81.39 |
| [ep200_job217299](dat_2dcnn_lstm/cubelearn/ep200_job217299/results.json) | CubeLearn | 0 | 32 | 200 | 200 | 100.00 | 99.44 | 88.06 |
| [job217297](dat_2dcnn_lstm/cubelearn/job217297/results.json) | CubeLearn | 0 | 32 | 30 | 26 | 81.11 | 81.11 | 84.44 |
| [diag20260909_seed0_bs8](dat_2dcnn_lstm/dft/diag20260909_seed0_bs8/results.json) | DFT | 0 | 8 | 30 | 26 | 91.11 | 91.11 | 74.44 |
| [diag20260909_seed1_bs32](dat_2dcnn_lstm/dft/diag20260909_seed1_bs32/results.json) | DFT | 1 | 32 | 30 | 30 | 87.22 | 83.89 | 61.11 |
| [diag20260909_seed2_bs32](dat_2dcnn_lstm/dft/diag20260909_seed2_bs32/results.json) | DFT | 2 | 32 | 30 | 29 | 91.67 | 89.17 | 74.72 |
| [ep200_job217299](dat_2dcnn_lstm/dft/ep200_job217299/results.json) | DFT | 0 | 32 | 200 | 119 | 99.44 | 98.89 | 83.33 |
| [job217297](dat_2dcnn_lstm/dft/job217297/results.json) | DFT | 0 | 32 | 30 | 27 | 84.44 | 78.33 | 79.72 |

## Fixed diagnostic plan

30 epochs for both DFT and CubeLearn at (seed, batch size): (1, 32), (2, 32), (0, 8).
The first two pairs extend the original seed-0 run; batch 8 is a sensitivity experiment, not a claimed author setting.
No user-split search or selection by test accuracy. Failures remain visible below.

| Seed | Batch | Model | Status |
|---:|---:|---|---|
| 1 | 32 | dft | complete |
| 1 | 32 | cubelearn | complete |
| 2 | 32 | dft | complete |
| 2 | 32 | cubelearn | complete |
| 0 | 8 | dft | complete |
| 0 | 8 | cubelearn | complete |

## Interpretation

- DFT, 30 epochs/batch 32, 3 seeds: seen test 78.33–89.17%; held-out test 61.11–79.72%.
- CubeLearn, 30 epochs/batch 32, 3 seeds: seen test 81.11–90.83%; held-out test 66.11–84.44%.

The historical 200-epoch runs are extended-budget experiments. Numerical agreement at a changed budget does not establish a faithful reproduction.
See [protocol and limitations](../notes.md). Statuses and results are refreshed after every run.
