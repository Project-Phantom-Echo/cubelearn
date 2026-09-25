"""CubeLearn HAR dataset with the paper's actual split.

The split CSVs in dataset_split/ hold **sample ids (0-29), not user ids** -- the
paper splits repetitions, not subjects, for its "in-set" numbers:

    train        in-set users x 6 activities x 15 sample ids  =  540
    validation   in-set users x 6 activities x  5 sample ids  =  180
    test_in      in-set users x 6 activities x 10 sample ids  =  360
    test_out     held-out users x 6 activities x all 30       =  360
                                                       total = 1440

The paper states 6 in-set + 2 out-of-set participants but never names which;
upstream dataset.py uses users=[0..5], so in-set = 0-5 and out-of-set = 6-7 is
inferred from that convention, not verified against the authors.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

N_ACTIVITIES = 6
IN_SET_USERS = (0, 1, 2, 3, 4, 5)
OUT_OF_SET_USERS = (6, 7)
ACTIVITY_NAMES = (
    "marching on the spot",
    "jogging on the spot",
    "clapping",
    "waving right hand",
    "sweeping the floor with right hand",
    "rubbing left arm with right hand",
)

# D-A-T and R-D-A-T use the first half of the chirp and sample axes (paper: GPU
# memory), matching the commented line in upstream dataset.py.
CROPPED_MODELS = ("dat", "rdat")


def _read_ids(path):
    return [int(v) for v in np.atleast_1d(np.genfromtxt(path))]


class HARDataset(Dataset):
    def __init__(self, cache_dir, split_dir, split, crop=False):
        super().__init__()
        self.cache_dir = cache_dir
        self.crop = crop
        self.split = split

        train_ids = _read_ids(os.path.join(split_dir, "train_samples.csv"))
        val_ids = _read_ids(os.path.join(split_dir, "validation_samples.csv"))
        test_ids = _read_ids(os.path.join(split_dir, "test_samples.csv"))

        if split == "train":
            users, ids = IN_SET_USERS, train_ids
        elif split == "val":
            users, ids = IN_SET_USERS, val_ids
        elif split == "test_in":
            users, ids = IN_SET_USERS, test_ids
        elif split == "test_out":
            # out-of-set subjects contribute every repetition
            users, ids = OUT_OF_SET_USERS, sorted(set(train_ids + val_ids + test_ids))
        else:
            raise ValueError(f"unknown split {split!r}")

        self.items = [(u, a, s) for u in users for a in range(N_ACTIVITIES) for s in ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        user, activity, sample = self.items[idx]
        path = os.path.join(self.cache_dir, f"{user}_{activity}_{sample}.npy")
        a = np.load(path)                       # (2, 20, 128, 8, 256) int16
        if self.crop:
            a = a[:, :, :64, :, :128]           # -> (2, 20, 64, 8, 128)
        data = a[0].astype(np.float32) + 1j * a[1].astype(np.float32)
        return torch.from_numpy(data.astype(np.complex64)), activity
