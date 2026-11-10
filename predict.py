"""Detect the keyword in one or more audio files.

    python predict.py samples/test-stop.wav samples/test-step.wav
    python predict.py my_clip.wav --threshold 0.8

Any format soundfile can read works (WAV, FLAC, OGG); any sample rate or channel
count is converted to 16 kHz mono. Only the first second of each file is used.
"""

import argparse
from pathlib import Path

import torch

from kws.audio import LogMel, fix_length, load_wav
from kws.config import DEVICE, MODEL_PATH, THRESHOLD
from kws.model import load_model


class KeywordDetector:
    def __init__(self, model_path=MODEL_PATH, threshold: float = THRESHOLD):
        self.model = load_model(model_path, DEVICE)
        self.features = LogMel()
        self.threshold = threshold

    @torch.no_grad()
    def probability(self, path) -> float:
        """Probability (0-1) that the clip contains the keyword."""
        mel = self.features(fix_length(load_wav(path)))
        return torch.sigmoid(self.model(mel.unsqueeze(0).to(DEVICE))).item()

    def detect(self, path) -> bool:
        return self.probability(path) > self.threshold


def main():
    parser = argparse.ArgumentParser(description="Detect the keyword 'stop' in audio files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--model", default=MODEL_PATH, type=Path)
    parser.add_argument("--threshold", default=THRESHOLD, type=float)
    args = parser.parse_args()

    detector = KeywordDetector(args.model, args.threshold)
    for f in args.files:
        p = detector.probability(f)
        verdict = "STOP detected" if p > args.threshold else "no keyword"
        print(f"{f}: {verdict} (p={p:.3f})")


if __name__ == "__main__":
    main()
