"""Shared constants. These must match the values the model was trained with."""

from pathlib import Path

import torch

# Audio
SAMPLE_RATE = 16_000          # Speech Commands is recorded at 16 kHz
CLIP_LEN = SAMPLE_RATE        # every clip is padded / trimmed to exactly 1 second

# Log-mel features (25 ms window, 10 ms hop -> 40 x 101 spectrogram per clip)
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160

# Training
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 5                  # early-stopping patience (epochs without val improvement)
LEARNING_RATE = 1e-3
THRESHOLD = 0.5               # sigmoid probability above which a clip counts as the keyword

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"
MODEL_PATH = ROOT / "models" / "best_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
