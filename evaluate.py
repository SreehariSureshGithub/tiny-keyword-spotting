"""Evaluate a trained model on a prepared split (default: Dataset/test).

    python evaluate.py
    python evaluate.py --split val --threshold 0.7
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kws.config import BATCH_SIZE, DATASET_DIR, DEVICE, MODEL_PATH, THRESHOLD
from kws.dataset import KeywordDataset
from kws.model import load_model


@torch.no_grad()
def evaluate(model, data_dir, threshold: float = THRESHOLD) -> dict:
    loader = DataLoader(KeywordDataset(data_dir), batch_size=BATCH_SIZE)
    tp = fp = tn = fn = 0
    for x, y in loader:
        preds = torch.sigmoid(model(x.to(DEVICE))).squeeze(1).cpu() > threshold
        y = y.bool()
        tp += (preds & y).sum().item()
        fp += (preds & ~y).sum().item()
        tn += (~preds & ~y).sum().item()
        fn += (~preds & y).sum().item()

    total = tp + fp + tn + fn
    m = {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
    }
    m["f1"] = (2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
               if m["precision"] + m["recall"] else 0.0)

    print(f"clips: {total} ({tp + fn} keyword, {tn + fp} other)  threshold: {threshold}")
    print(f"accuracy  {m['accuracy']:.4f}   precision {m['precision']:.4f}   "
          f"recall {m['recall']:.4f}   f1 {m['f1']:.4f}   false-alarm rate {m['false_alarm_rate']:.4f}")
    print("confusion matrix (rows = true, cols = predicted):")
    print("              other  keyword")
    print(f"  other     {tn:7d}  {fp:7d}")
    print(f"  keyword   {fn:7d}  {tp:7d}")
    return m


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyKWS on a dataset split.")
    parser.add_argument("--data", default=DATASET_DIR, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", default=MODEL_PATH, type=Path)
    parser.add_argument("--threshold", default=THRESHOLD, type=float)
    args = parser.parse_args()

    evaluate(load_model(args.model, DEVICE), args.data / args.split, args.threshold)


if __name__ == "__main__":
    main()
