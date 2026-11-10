"""Train TinyKWS on Dataset/train, validate on Dataset/val, keep the best checkpoint.

    python train.py
    python train.py --epochs 60 --out models/my_model.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate import evaluate
from kws.config import (BATCH_SIZE, DATASET_DIR, DEVICE, EPOCHS, LEARNING_RATE,
                        MODEL_PATH, PATIENCE)
from kws.dataset import KeywordDataset
from kws.model import TinyKWS, load_model


def main():
    parser = argparse.ArgumentParser(description="Train the TinyKWS keyword detector.")
    parser.add_argument("--data", default=DATASET_DIR, type=Path)
    parser.add_argument("--out", default=MODEL_PATH, type=Path)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--patience", default=PATIENCE, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--workers", default=0, type=int, help="DataLoader worker processes")
    args = parser.parse_args()

    train_ds = KeywordDataset(args.data / "train", augment=True)
    val_ds = KeywordDataset(args.data / "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    # "stop" is the minority class, so weight positive examples by neg/pos.
    pos = sum(train_ds.labels)
    neg = len(train_ds.labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=DEVICE)
    print(f"train: {len(train_ds)} clips ({pos} keyword) | val: {len(val_ds)} | pos_weight={pos_weight.item():.2f} | device={DEVICE}")

    model = TinyKWS().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val, bad_epochs = float("inf"), 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                val_loss += loss_fn(model(x), y).item() * len(x)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        msg = f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}"
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), args.out)
            msg += "  -> saved"
        else:
            bad_epochs += 1
        print(msg)
        if bad_epochs >= args.patience:
            print("early stopping")
            break

    test_dir = args.data / "test"
    if test_dir.exists():
        print("\nTest-set results for the best checkpoint:")
        evaluate(load_model(args.out, DEVICE), test_dir)


if __name__ == "__main__":
    main()
