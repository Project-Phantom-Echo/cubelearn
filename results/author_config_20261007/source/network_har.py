"""HAR variants of the CubeLearn classifiers.

Upstream network.py is hardcoded for hand-gesture recognition: T=10 frames and
12 classes. HAR is T=20 and 6 classes, so every reshape over the frame axis and
the final head have to change. For D-A-T 2DCNN-LSTM that is exactly two edits
(`view(-1, T, 784)` and the head), on top of the `super()` fix applied to
Range_Fourier_Net_Small / Doppler_Fourier_Net_Small in network.py.

The learnable-preprocessing (LPP) submodules are named in LPP_MODULES so the
training script can put them in their own optimizer group: lr=0 reproduces
conventional DFT preprocessing, lr>0 is CubeLearn.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cplxmodule.nn import CplxModulus, CplxToCplx

from network import (
    AOA_Fourier_Net,
    Doppler_Fourier_Net_Small,
    Range_Fourier_Net_Small,
)

LPP_MODULES = ("range_net", "doppler_net", "aoa_net")


class DAT_2DCNNLSTM_HAR(nn.Module):
    """Doppler-Angle-Time, 2D CNN per frame + LSTM over frames.

    Approximate paper-chart targets: ~99.4% seen and ~91.1% held-out for
    CubeLearn, ~98.8 / ~86.9 for fixed DFT. These are not our reproduced scores.
    Range magnitudes are computed and summed before the CNN-LSTM.
    """

    def __init__(self, n_frames=20, n_classes=6):
        super().__init__()
        self.n_frames = n_frames
        self.range_net = Range_Fourier_Net_Small()
        self.doppler_net = Doppler_Fourier_Net_Small()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(784, 512, 1, batch_first=True)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, T, 64 chirps, 8 antennas, 128 samples) complex
        x = x[:, :, :, 0:8, :].contiguous()
        x = self.range_net(x)                       # range FFT over samples
        x = x.view(-1, 64, 8, 128)
        x = self.cplx_transpose(1, 3)(x)
        x = self.doppler_net(x)                     # Doppler FFT over chirps
        x = self.cplx_transpose(2, 3)(x)
        x = self.aoa_net(x)                         # angle FFT over antennas
        x = CplxModulus()(x)
        x = torch.sum(x, dim=1)                     # marginalise range
        x = x.view(-1, 1, 64, 64)                   # Doppler x angle per frame
        for conv, bn in ((self.conv1, self.bn1),
                         (self.conv2, self.bn2),
                         (self.conv3, self.bn3)):
            x = self.maxpool(F.relu(bn(conv(x))))
        x = x.view(-1, self.n_frames, 784)          # HGR used 10 frames
        output, _ = self.lstm(x)
        x = F.relu(self.fc_2(output[:, -1, :]))
        return self.fc_3(x)


MODELS = {
    "dat_2dcnn_lstm": DAT_2DCNNLSTM_HAR,
}

# Models that consume the halved chirp/sample axes (see har_dataset.CROPPED_MODELS)
CROPPED = {"dat_2dcnn_lstm"}
