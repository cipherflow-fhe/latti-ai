import numpy as np
from pathlib import Path
"""
ECG Dataset Module for Loading NumPy Arrays.

This module provides a custom dataset class that loads ECG data from NumPy (.npy) files,
supporting optional data transformations. It implements the standard __len__ and __getitem__
interface for compatibility with PyTorch DataLoader or similar iteration patterns.
"""
class ECGNpyDataset:
    def __init__(self, x_path, y_path, transform=None):
        self.x = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y