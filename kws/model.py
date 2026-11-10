"""TinyKWS: a ~23k-parameter CNN for binary keyword spotting."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyKWS(nn.Module):
    """Three conv blocks + global average pooling + one output logit.

    Input:  log-mel spectrogram, shape [batch, 1, 40, 101]
    Output: raw logit, shape [batch, 1] (apply sigmoid for a probability)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # [B, 16, 20, 50]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 32, 10, 25]
        x = F.relu(self.conv3(x))              # [B, 64, 10, 25]
        x = self.gap(x).flatten(1)             # [B, 64]
        return self.fc(x)                      # [B, 1]


def load_model(path, device="cpu") -> TinyKWS:
    """Build TinyKWS and load trained weights from a state_dict file."""
    model = TinyKWS().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
