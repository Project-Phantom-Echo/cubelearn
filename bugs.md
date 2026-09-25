# CubeLearn implementation issues

Reproduction status and commands: [notes.md](notes.md).

| Issue | Effect | Local handling |
|---|---|---|
| Incorrect `super()` in `Range_Fourier_Net_Small` and `Doppler_Fourier_Net_Small` | D-A-T and R-D-A-T constructors fail | Fixed in `network.py`; preserved in this fork |
| Released model definitions use 10 frames and 12 classes | Incompatible with HAR's 20 frames and 6 classes | `network_har.py` adapts D-A-T 2D CNN-LSTM |
| Released training script has no test loop | Cannot produce seen/held-out test metrics directly | `train_har.py` evaluates both using the best validation checkpoint |
| CSVs contain repetition IDs, not user IDs | They cannot establish the paper's participant assignment | `har_dataset.py` uses 0–5 seen, 6–7 held out, inferred from the gesture loader |

The two constructor fixes were already present in the historical runs. The
2026-09-09 audit found no additional training defect. Previous notes overstated
reproduction success, claimed that validation selection prevents overfitting, and
attributed the held-out gap to user choice without evidence; those statements have
been corrected.
