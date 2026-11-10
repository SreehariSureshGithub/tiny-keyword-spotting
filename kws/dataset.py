"""PyTorch Dataset over the prepared Dataset/<split>/<label>/*.wav layout."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import LogMel, fix_length, load_wav
from .config import SAMPLE_RATE


class KeywordDataset(Dataset):
    """Loads clips from `root/0` (not the keyword) and `root/1` (the keyword).

    With augment=True each clip independently gets, with 50% probability each:
    a random time shift of up to +/-100 ms, a random gain of 0.7-1.3x, and
    light Gaussian noise.
    """

    def __init__(self, root, augment: bool = False):
        self.files: list[Path] = []
        self.labels: list[int] = []
        self.augment = augment

        for label in ("0", "1"):
            for f in sorted((Path(root) / label).glob("*.wav")):
                self.files.append(f)
                self.labels.append(int(label))

        if not self.files:
            raise FileNotFoundError(
                f"No .wav files under {root}/0 or {root}/1. Run prepare_dataset.py first."
            )

        self.features = LogMel()

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _time_shift(wav):
        shift = int(random.uniform(-0.1, 0.1) * SAMPLE_RATE)
        return torch.roll(wav, shifts=shift, dims=1)

    @staticmethod
    def _gain(wav):
        return wav * random.uniform(0.7, 1.3)

    @staticmethod
    def _add_noise(wav):
        return wav + torch.randn_like(wav) * 0.005

    def __getitem__(self, idx):
        wav = fix_length(load_wav(self.files[idx]))

        if self.augment:
            if random.random() < 0.5:
                wav = self._time_shift(wav)
            if random.random() < 0.5:
                wav = self._gain(wav)
            if random.random() < 0.5:
                wav = self._add_noise(wav)

        return self.features(wav), torch.tensor(self.labels[idx], dtype=torch.float32)
