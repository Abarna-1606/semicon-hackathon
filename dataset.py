import os
import numpy as np
import torch
from torch.utils.data import Dataset


class DegradedDataset(Dataset):
    """
    Expects folder structure:
        data_root/
            GT/           -> ground truth .npy files (512x512 or 256x256)
            NoisyLR/      -> degraded .npy files (256x256 or 128x128)

    Filenames must match between GT and NoisyLR (e.g. 000058.npy in both).
    Update GT_SUBDIR / LR_SUBDIR below if your dataset uses different folder names.
    """

    GT_SUBDIR = "GT"
    LR_SUBDIR = "NoisyLR"

    def __init__(self, data_root, augment=True):
        self.gt_dir = os.path.join(data_root, self.GT_SUBDIR)
        self.lr_dir = os.path.join(data_root, self.LR_SUBDIR)
        self.files = sorted(os.listdir(self.lr_dir))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _load(path):
        return np.load(path).astype(np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]
        lr = self._load(os.path.join(self.lr_dir, fname))
        gt = self._load(os.path.join(self.gt_dir, fname))

        # Degraded image can exceed [0,1] due to speckle noise (expected).
        # Only clip negatives; keep the overshoot info for the model to learn from.
        lr = np.clip(lr, 0.0, None)
        gt = np.clip(gt, 0.0, 1.0)

        if self.augment:
            if np.random.rand() < 0.5:
                lr = np.fliplr(lr).copy()
                gt = np.fliplr(gt).copy()
            if np.random.rand() < 0.5:
                lr = np.flipud(lr).copy()
                gt = np.flipud(gt).copy()
            k = np.random.randint(0, 4)
            if k:
                lr = np.rot90(lr, k).copy()
                gt = np.rot90(gt, k).copy()

        lr_t = torch.from_numpy(lr).unsqueeze(0)  # (1, H, W)
        gt_t = torch.from_numpy(gt).unsqueeze(0)  # (1, 2H, 2W)
        return lr_t, gt_t


class TestDataset(Dataset):
    """For inference only — no ground truth, just degraded .npy files."""

    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        arr = np.load(os.path.join(self.test_dir, fname)).astype(np.float32)
        arr = np.clip(arr, 0.0, None)
        return torch.from_numpy(arr).unsqueeze(0), fname
